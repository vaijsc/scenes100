#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import gzip
import math
import random
import tqdm
import glob
import psutil
import argparse
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_homography_mixup')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


class HomographyTransform(torch.nn.Module):
    def __init__(self, sx, sy, lx, ly):
        super(HomographyTransform, self).__init__()
        # fix sx/sy in range (0.5, 1.5), sx = sigmoid(sx_logit) + 0.5
        assert min(sx, sy) > 0.51 and max(sx, sy) < 1.49
        self.sxsy_logit = torch.nn.Parameter(torch.FloatTensor([np.log((sx - 0.5) / (1.5 - sx)), np.log((sy - 0.5) / (1.5 - sy))]))

        # fix lx/ly in range (-0.75, 0.75), lx = 1.5 * sigmoid(lx_logit) - 0.75
        assert min(lx, ly) > -0.74 and max(lx, ly) < 0.74
        self.lxly_logit = torch.nn.Parameter(torch.FloatTensor([np.log((lx / 1.5 + 0.5) / (0.5 - lx / 1.5)), np.log((ly / 1.5 + 0.5) / (0.5 - ly / 1.5))]))

        # homography matrix should only have 4 DoF
        self.zero6 = torch.nn.Parameter(torch.FloatTensor([0, 0, 0, 0, 0, 0]))
        self.zero1 = torch.nn.Parameter(torch.FloatTensor([0]))
        self.one1 = torch.nn.Parameter(torch.FloatTensor([1]))

    def forward(self):
        raise NotImplementedError

    def get_sxsylxly(self):
        return tuple(list(map(float, torch.sigmoid(self.sxsy_logit) + 0.5)) + list(map(float, torch.sigmoid(self.lxly_logit) * 1.5 - 0.75)))

    def _get_matrix(self):
        return torch.cat([self.zero6.detach(), torch.sigmoid(self.lxly_logit) * 1.5 - 0.75, self.one1.detach()], dim=0).view(3, 3) + \
            torch.diag(torch.cat([torch.sigmoid(self.sxsy_logit) + 0.5, self.zero1.detach()]))

    # https://github.com/vidit09/geoshift/blob/main/mnist/stn_arch.py
    def _apply_matrix(self, x: torch.Tensor, mat: torch.Tensor, eps=1e-8):
        _device = self.sxsy_logit.device
        B, C, H, W = x.size()
        xs, ys = torch.meshgrid(
            torch.arange(-W // 2, W // 2, device=_device),
            torch.arange(-H // 2, H // 2, device=_device)
        )
        homogenous = torch.stack([xs.float(), ys.float(), torch.ones((W, H), device=_device)]).reshape(1, 3, -1)
        homogenous = (homogenous / torch.tensor([(max(W, H) - 1) / 2, (max(W, H) - 1) / 2, 1], device=_device).reshape(1,3,1))
        homogenous, mat = homogenous.expand(B, -1, -1), mat.expand(B, -1, -1)

        map_indices = mat.bmm(homogenous)
        mask = torch.abs(map_indices[:, -1, :]) > eps
        scale = torch.where(mask, 1.0 / (map_indices[:, -1, :] + eps), torch.ones_like(map_indices[:, -1, :]))
        map_indices = (map_indices[:, :-1, :] * scale.unsqueeze(1)).view(B, 2, W, H)
        if W > H:
            map_indices = (map_indices / torch.tensor([1, (H / W)], device=_device).reshape(1, 2, 1, 1))
        elif W < H:
            map_indices = (map_indices / torch.tensor([(W / H), 1], device=_device).reshape(1, 2, 1, 1))
        map_indices = (map_indices * torch.tensor([1, 1], device=_device).reshape(1, 2, 1, 1))
        grid = map_indices.permute(0, 3, 2, 1)

        x_tf = torch.nn.functional.grid_sample(x, grid, mode='bilinear', align_corners=True, padding_mode='zeros')
        x_tf[torch.isnan(x_tf)] = 0
        return x_tf

    def warp(self, x: torch.Tensor):
        mat = self._get_matrix()
        # print(mat)
        return self._apply_matrix(x, mat)

    def unwarp(self, x: torch.Tensor):
        mat = torch.inverse(self._get_matrix())
        return self._apply_matrix(x, mat)


class WeightedFeatureMap(torch.nn.Module):
    def __init__(self, n_channels, n_maps):
        super(WeightedFeatureMap, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 16, 3, padding=1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(16), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 8, 3, padding=1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(8), torch.nn.ReLU(inplace=True),
        )
        self.weighting = torch.nn.Sequential(
            torch.nn.Conv2d(8 * n_maps, n_maps, 3, padding=1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(n_maps), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_maps, n_maps, 3, padding=1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(n_maps), torch.nn.Softmax(dim=1),
        )

    def forward(self, maps):
        B, N, C, H, W = maps.size() # batch x maps x channels x H x W
        maps_reduce = self.conv(maps.view(B * N, C, H, W)).view(B, N, -1, H, W).view(B, -1, H, W)
        weights = self.weighting(maps_reduce).unsqueeze(2)
        return (maps * weights).sum(dim=1)


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFinetuneHomography(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def aggregate_warped_features(self, images_tensor):
        features_tf_list = [self.backbone(images_tensor)] # always keep unwarped image feature
        assert set(features_tf_list[0].keys()) == set(self.aggregators.keys())
        for tf in self.homographies:
            images_warp = tf.warp(images_tensor.detach())
            # do not train backbone from warped images, reduce VRAM usage
            with torch.no_grad():
                features_warp = self.backbone(images_warp)
            features_tf_list.append({k: tf.unwarp(features_warp[k]) for k in features_warp})

            # with torch.no_grad():
            #     images_restore = tf.unwarp(images_warp.detach())
            #     _diff = images_tensor - images_restore
            # plt.figure()
            # plt.subplot(2, 3, 1)
            # _im = images_tensor[0].detach().cpu().numpy().transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            # plt.imshow(_im)
            # plt.subplot(2, 3, 4)
            # _im = images_warp[0].detach().cpu().numpy().transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            # plt.imshow(_im); plt.title('%.3f %.3f %.3f %.3f' % tf.get_sxsylxly())
            # plt.subplot(2, 3, 2)
            # _im = images_restore[0].detach().cpu().numpy().transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            # plt.imshow(_im); plt.title('restored')
            # plt.subplot(2, 3, 5)
            # _im = _diff[0].detach().cpu().numpy().transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            # plt.imshow(_im); plt.title('diff')
            # plt.subplot(2, 3, 6)
            # _im = features_warp['p2'][0][:3].detach().cpu().numpy().transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            # plt.imshow(_im)
            # plt.subplot(2, 3, 3)
            # _im = features_tf_list[-1]['p2'][0][:3].detach().cpu().numpy().transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            # plt.imshow(_im)
            # plt.show()
        features_tf_dict = {k: torch.stack([f[k] for f in features_tf_list], dim=1) for k in features_tf_list[0]}
        for k in features_tf_dict:
            assert features_tf_dict[k].size(1) == len(self.homographies) + 1
            assert features_tf_dict[k].size(2) == 256
            features_tf_dict[k] = self.aggregators[k](features_tf_dict[k])
        return features_tf_dict

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        # print(self.backbone.bottom_up.stem.conv1.weight.data[0, 0, 0, 0].item(), self.roi_heads.box_head.fc1.weight.data[0, 0].item(), self.homographies[0].sxsy[0].item())
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features = self.aggregate_warped_features(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        images = self.preprocess_image(batched_inputs)
        # features = self.backbone(images.tensor)
        features = self.aggregate_warped_features(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.homographies = torch.nn.ModuleList([
            # HomographyTransform(0.8, 0.8, 0.3, 0).to(net.device),
            # HomographyTransform(0.8, 0.8, -0.3, 0).to(net.device),
            # HomographyTransform(0.8, 0.8, 0, 0.3).to(net.device),
            HomographyTransform(0.8, 0.8, 0, -0.3).to(net.device),
            HomographyTransform(0.6, 0.6, 0, -0.5).to(net.device),
        ])
        net.aggregators = torch.nn.ModuleDict({
            'p2': WeightedFeatureMap(256, len(net.homographies) + 1).to(net.device),
            'p3': WeightedFeatureMap(256, len(net.homographies) + 1).to(net.device),
            'p4': WeightedFeatureMap(256, len(net.homographies) + 1).to(net.device),
            'p5': WeightedFeatureMap(256, len(net.homographies) + 1).to(net.device),
            'p6': WeightedFeatureMap(256, len(net.homographies) + 1).to(net.device),
        })
        net.__class__ = GeneralizedRCNNFinetuneHomography
        return net


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneHomographyTrainer(DefaultTrainer):
    def __init__(self, cfg, is_base_model):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        model = GeneralizedRCNNFinetuneHomography.create_from_sup(model)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        assert (not isinstance(model, torch.nn.DataParallel)) and (not isinstance(model, torch.nn.parallel.DistributedDataParallel))
        self._trainer.faster_rcnn_homography_model = model
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []
        self._trainer.homography_history = []
        if is_base_model:
            self._trainer.iters_aggr_base_pretrain = cfg.SOLVER.MAX_ITER // 10 # when train the base model, only train aggregators in the beginning

    def build_hooks(self):
        ret = super().build_hooks()
        self.eval_results_all = {}
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = copy.deepcopy(self._last_eval_results)
            return self._last_eval_results
        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)


# DefaultTrainer._trainer is instance of SimpleTrainer
# DefaultTrainer & SimpleTrainer are subclass of TrainerBase
def homography_base_trainer_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    if self.iter < 2:
        print('freeze Faster RCNN parameters, only train aggregators')
        for m in [self.faster_rcnn_homography_model.backbone, self.faster_rcnn_homography_model.proposal_generator, self.faster_rcnn_homography_model.roi_heads, self.faster_rcnn_homography_model.homographies]:
            for p in m.parameters():
                p.requires_grad = False
    if self.iter == self.iters_aggr_base_pretrain:
        print('train whole network')
        for p in self.faster_rcnn_homography_model.parameters():
            p.requires_grad = True

    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start
    loss_dict = self.model(data)
    loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {'total_loss': loss_dict}
    else:
        losses = sum(loss_dict.values())
    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    # If you need gradient clipping/scaling or other processing, you can wrap the optimizer with your custom `step()` method. But it is suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})
    self.homography_history.append({'iter': self.iter, 'sxsylxly': [_aggr.get_sxsylxly() for _aggr in self.faster_rcnn_homography_model.homographies]})


def homography_finetune_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start
    loss_dict = self.model(data)
    loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {'total_loss': loss_dict}
    else:
        losses = sum(loss_dict.values())
    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    # If you need gradient clipping/scaling or other processing, you can wrap the optimizer with your custom `step()` method. But it is suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})
    self.homography_history.append({'iter': self.iter, 'sxsylxly': [_aggr.get_sxsylxly() for _aggr in self.faster_rcnn_homography_model.homographies]})


def train_base(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(args, 'valid')
    desc_cocotrain, dst_cocotrain = 'mscoco2017_train_remap', get_coco_dicts(args, 'train')
    del _tensor
    gc.collect()
    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model(args.model)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 5
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_cocotrain,)
    cfg.DATASETS.TEST = (desc_cocovalid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = FinetuneHomographyTrainer(cfg, is_base_model=True)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(homography_base_trainer_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperMixup.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, 0, None, None, None)
    trainer.resume_or_load(resume=False)

    prefix = 'mscoco2017_remap_homography_%s' % args.model
    results_0 = None
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    del data_loader, evaluator
    trainer.eval_results_all[0] = results_0
    trainer.train()
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return

    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'homography_history': trainer._trainer.homography_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    aps, lr_history, loss_history, homography_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history, trainer._trainer.homography_history
    iter_list = sorted(list(aps.keys()))
    dst_list = {'mAP': [], 'AP50': []}
    for i in iter_list:
        dst_list['mAP'].append(aps[i]['bbox']['AP'])
        dst_list['AP50'].append(aps[i]['bbox']['AP50'])
    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]
    homography_history_iter_list = np.array([_h['iter'] for _h in homography_history])
    homography_history = np.array([_h['sxsylxly'] for _h in homography_history])

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')
    plt.subplot(2, 2, 2)
    colors, color_i = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000'], 0
    legends = []
    for loss_key in loss_history_dict:
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')

    plt.subplot(4, 2, 5)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 0], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, :2].min() - 0.01, homography_history[:, :, :2].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('sx')
    plt.subplot(4, 2, 6)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 1], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, :2].min() - 0.01, homography_history[:, :, :2].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('sy')
    plt.subplot(4, 2, 7)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 2], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, 2:].min() - 0.01, homography_history[:, :, 2:].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('lx')
    plt.subplot(4, 2, 8)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 3], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, 2:].min() - 0.01, homography_history[:, :, 2:].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('ly')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))


def adapt(args):
    assert args.fn_max_samples <= 0
    assert not args.mixup_random_position
    assert 0 <= args.mixup_p <= 1 and 0 <= args.mixup_r <= 1, '%s %s' % (args.mixup_p, args.mixup_r)
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(args, 'valid')
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]
    desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, get_annotation_dict(args)
    desc_pseudo_anno = 'refine_' + '_'.join(args.anno_models)
    dst_pseudo_anno = refine_annotations(args)[0]
    # include sample mixup sources, this increases RAM usage
    dst_pseudo_anno_copy = copy.deepcopy(dst_pseudo_anno)
    for im in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources'):
        for _ in range(0, 3):
            im['mixup_src_images'] = [dst_pseudo_anno_copy[random.randrange(0, len(dst_pseudo_anno_copy))]]
    del dst_pseudo_anno_copy
    if args.train_on_coco:
        random.seed(42)
        dst_cocotrain = get_coco_dicts(args, 'train')
        random.shuffle(dst_cocotrain)
        dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
        desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
        print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
    for i in range(0, len(dst_pseudo_anno)):
        dst_pseudo_anno[i]['image_id'] = i + 1
    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    print('loading checkpoint:', args.ckpt)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output

    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid, desc_cocovalid)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneHomographyTrainer(cfg, is_base_model=False)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(homography_finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperMixup.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, args.mixup_p, args.mixup_r, args.mixup_overlap_thres, args.mixup_random_position)
    trainer.resume_or_load(resume=False)

    prefix = 'adapt%s_%s_anno_%s_homography%s' % (args.id, args.model, desc_pseudo_anno, '_mixup' if args.mixup_p > 0 else '')
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    del data_loader, evaluator
    trainer.eval_results_all[0] = results_0
    trainer.train()
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return

    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'homography_history': trainer._trainer.homography_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    aps, lr_history, loss_history, homography_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history, trainer._trainer.homography_history
    iter_list = sorted(list(aps.keys()))
    dst_list = [desc_cocovalid, desc_manual_valid]
    assert len(dst_list) == 2
    dst_list = {k: {'mAP': [], 'AP50': []} for k in dst_list}
    for i in iter_list:
        for k in dst_list:
            dst_list[k]['mAP'].append(aps[i][k]['bbox']['AP'])
            dst_list[k]['AP50'].append(aps[i][k]['bbox']['AP50'])
    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]
    homography_history_iter_list = np.array([_h['iter'] for _h in homography_history])
    homography_history = np.array([_h['sxsylxly'] for _h in homography_history])

    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid]['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid]['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP', 'Manual Valid AP50', 'Manual Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')
    plt.subplot(2, 2, 2)
    colors, color_i = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000'], 0
    legends = []
    for loss_key in loss_history_dict:
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')

    plt.subplot(4, 2, 5)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 0], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, :2].min() - 0.01, homography_history[:, :, :2].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('sx')
    plt.subplot(4, 2, 6)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 1], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, :2].min() - 0.01, homography_history[:, :, :2].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('sy')
    plt.subplot(4, 2, 7)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 2], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, 2:].min() - 0.01, homography_history[:, :, 2:].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('lx')
    plt.subplot(4, 2, 8)
    for h_i in range(0, homography_history.shape[1]):
        plt.plot(homography_history_iter_list, homography_history[:, h_i, 3], linestyle='-', color=colors[h_i])
    plt.legend([('H%d' % h_i) for h_i in range(0, homography_history.shape[1])])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(homography_history[:, :, 2:].min() - 0.01, homography_history[:, :, 2:].max() + 0.01)
    plt.xlabel('Training Iterations')
    plt.title('ly')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorHomography(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNFinetuneHomography.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format


def evaluate(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorHomography(cfg)

    results = {}
    detections = []
    loader = torchdata.DataLoader(
        EvaluationDataset(
            copy.deepcopy(images),
            [os.path.join(inputdir, 'unmasked', im['file_name']) for im in images]
        ),
        batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=2
    )
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting %s validation frames' % args.id):
        det = copy.deepcopy(im)
        det['annotations'] = []
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results['manual_%s' % args.id] = evaluate_masked(args.id, detections, outputfile=args.eval_outputfile)

    if not args.eval_skip_coco:
        detections = get_coco_dicts(args, 'valid')
        for im in tqdm.tqdm(detections, ascii=True, desc='detecting MSCOCO2017 valid'):
            im_arr = skimage.io.imread(im['file_name'])
            if len(im_arr.shape) == 2:
                im_arr = np.stack([im_arr] * 3, axis=2)
            instances = detector(im_arr[:, :, ::-1])['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            im['annotations'] = []
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results['mscoco2017_valid'] = evaluate_cocovalid(args.cocodir, detections)
    print(vars(args))
    for dst in results:
        print('\n            %s\n' % dst)
        print(   '             %s' % '/'.join(results[dst]['metrics']))
        for c in sorted(results[dst]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[dst]['results'][c])))
    return vars(args), results


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorHomography(cfg)
    images_tensor = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        tf = detector.aug.get_transform(im_arr)
        images_tensor.append(torch.as_tensor(tf.apply_image(im_arr).astype('float32').transpose(2, 0, 1)))
    N1, N2 = 100, 900
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            detector.model.inference([{'image': images_tensor[i % len(images)], 'height': images[i % len(images)]['height'], 'width': images[i % len(images)]['width']}])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')

    parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--smallscale', default=False, type=bool)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    parser.add_argument('--fn_min_score', type=float, default=0.99, help='minimum objectiveness score of false negatives')
    parser.add_argument('--fn_max_samples', type=int, default=-1, help='maximum number of false negatives per frame')
    parser.add_argument('--fn_max_samples_det_p', type=float, default=0.5, help='maximum number of false negatives per frame as percentage of number of detections')
    parser.add_argument('--fn_min_area', type=float, default=50, help='minimum area of false negative boxes')
    parser.add_argument('--fn_max_width_p', type=float, default=0.3333, help='maximum percentage width of false negative boxes')
    parser.add_argument('--fn_max_height_p', type=float, default=0.3333, help='maximum percentage height of false negative boxes')

    parser.add_argument('--mixup_p', type=float, default=0.3, help='probability of applying mixup to an image')
    parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')
    parser.add_argument('--mixup_random_position', type=bool, default=False, help='randomly position patch')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--refine_visualize_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    parser.add_argument('--eval_outputfile', default=None, type=str)
    parser.add_argument('--hold', default=0.005, type=float)

    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    args.anno_models = sorted(list(set(args.anno_models)))
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'adapt':
        adapt(args)
    elif args.opt == 'base':
        train_base(args)
    elif args.opt == 'eval':
        evaluate(args)
    elif args.opt == 'tp':
        inference_throughput(args)
    else:
        pass


'''
conda deactivate && conda activate detectron2
cd /nfs/detection/zekun/Intersections/scripts/baseline

python finetune_homography_mixup.py --opt base --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 0 --iters 300 --eval_interval 101 --image_batch_size 2 --lr 0.01 --smallscale 1
python finetune_homography_mixup.py --opt base --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 40000 --eval_interval 3000 --image_batch_size 4

python finetune_homography_mixup.py --id 001 --opt adapt --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt mscoco2017_remap_homography_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 0 --iters 300 --eval_interval 101 --train_on_coco 1 --image_batch_size 2 --not_eval_coco 1 --lr 0.01 --mixup_p 0

python finetune_homography_mixup.py --opt eval --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --id 001 --ckpt F:\\intersections_results\\homography_r101\\adapt001_r101-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_homography.pth

python finetune_homography_mixup.py --opt eval --id 001 --cocodir ../../../MSCOCO2017 --model r50-fpn-3x --ckpt adapt_r50-fpn-3x_anno_refine_r50-fpn-3x.pth
python finetune_homography_mixup.py --opt tp --id 001 --model r101-fpn-3x --ckpt mscoco2017_remap_homography_r101-fpn-3x.pth

python compare_baselines.py --opt compare --model r101-fpn-3x --arch homography --compare_ckpt_dir F:\\intersections_results\\cvpr24\\homography_r101
'''
