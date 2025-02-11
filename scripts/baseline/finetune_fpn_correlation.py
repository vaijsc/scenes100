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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_fpn_correlation')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFinetuneFPCorr(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], is_tgt: bool = False):
        # print(self.backbone.bottom_up.stem.conv1.weight.data[0,0,0,0].item(), self.roi_heads.box_head.fc1.weight.data[0,0].item(), self.fp_corr.convs['p2'][0].weight.data[0,0,0,0].item())
        if not self.training:
            return self.inference(batched_inputs, is_tgt=is_tgt)
        images = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features = self.backbone(images.tensor)
        if is_tgt:
            features = self.fp_corr(features, images.tensor)
            proposals, proposal_losses = self.proposal_generator_merge(images, features, gt_instances)
            _, detector_losses = self.roi_heads_merge(images, features, proposals, gt_instances)
        else:
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

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, is_tgt: bool = False):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if is_tgt:
            features = self.fp_corr(features, images.tensor)
            proposals, _ = self.proposal_generator_merge(images, features, None)
            results, _ = self.roi_heads_merge(images, features, proposals, None)
        else:
            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def compute_fp(self, batched_inputs):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert len(batched_inputs) == 1
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        return images.tensor[0], {k: features[k][0] for k in features}

    @staticmethod
    def create_from_sup(net, fp_keys=['p2', 'p3', 'p4', 'p5', 'p6'], image_h=2, image_w=2, fp_dims={'p2': [1,1,1,1], 'p3': [1,1,1,1], 'p4': [1,1,1,1], 'p5': [1,1,1,1], 'p6': [1,1,1,1]}):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.fp_corr = FeaturePyramidCorrelation(256, fp_keys, image_h, image_w, fp_dims).to(net.device)
        net.proposal_generator_merge, net.roi_heads_merge = copy.deepcopy(net.proposal_generator), copy.deepcopy(net.roi_heads)
        net.__class__ = GeneralizedRCNNFinetuneFPCorr
        return net


class FeaturePyramidCorrelation(torch.nn.Module):
    def __init__(self, n_channels, fp_keys, h, w, fp_dims):
        super(FeaturePyramidCorrelation, self).__init__()
        self.correlate = torch.nn.ModuleDict({
            k: torch.nn.Sequential(
                torch.nn.Conv2d(2 * n_channels, 64, 1, padding=0),
                torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 2, 1, padding=0),
                torch.nn.BatchNorm2d(2), torch.nn.Softmax(dim=1)
            ) for k in fp_keys
        })
        # image size
        self.H = torch.nn.Parameter(torch.zeros((h,)).float())
        self.W = torch.nn.Parameter(torch.zeros((w,)).float())
        self.mean_fp = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.zeros(fp_dims[k]).float()) for k in fp_keys
        })

    def update_fp(self, fp_list, size_list):
        assert len(fp_list) == len(size_list)
        size_list = np.array(list(map(list, size_list)), dtype=np.int32)
        assert (size_list[:, 0] != 3).sum() == 0
        assert size_list[:, 1].min() == size_list[:, 1].max()
        assert size_list[:, 2].min() == size_list[:, 2].max()
        print('image dimensions for FP correlation computation: %d x %d x %d, %d feature pyramids' % (size_list[0, 0], size_list[0, 1], size_list[0, 2], len(fp_list)))
        self.H = torch.nn.Parameter(torch.zeros((size_list[0, 1],)).float())
        self.W = torch.nn.Parameter(torch.zeros((size_list[0, 2],)).float())
        mean_fp = {k: [] for k in self.mean_fp}
        for _fp in fp_list:
            for k in _fp:
                mean_fp[k].append(_fp[k])
        for k in mean_fp:
            mean_fp[k] = torch.stack(mean_fp[k], dim=0).mean(dim=0, keepdims=True)
            self.mean_fp[k] = torch.nn.Parameter(mean_fp[k].detach())
            print(k, mean_fp[k].size(), mean_fp[k].min(), mean_fp[k].max())

    def forward(self, fp, images_tensor):
        assert images_tensor.size(1) == 3 and images_tensor.size(2) == self.H.size(0) and images_tensor.size(3) == self.W.size(0), 'wrong image dimensions: %s' % str(images_tensor.size())
        fp_merge = {}
        for k in fp:
            _mean_fp_expand = self.mean_fp[k].expand(fp[k].size(0), -1, -1, -1).detach()          # B x C x     H x W
            _fp_stack = torch.stack([fp[k], _mean_fp_expand], dim=2)                              # B x C x 2 x H x W
            _weights = self.correlate[k](torch.cat([fp[k], _mean_fp_expand], dim=1)).unsqueeze(1) # B x 1 x 2 x H x W
            fp_merge[k] = (_fp_stack * _weights).sum(dim=2)                                       # B x C x     H x W
        # print(fp_merge['p3'].size(), fp_merge['p3'].min(), fp_merge['p3'].max(), fp_merge['p3'].mean())
        # plt.figure()
        # plt.subplot(2, 2, 1);im=images_tensor[0].detach().cpu().numpy().transpose(1,2,0);im-=im.min();im/=im.max();plt.imshow(im);plt.title(images_tensor.size())
        # plt.subplot(2, 2, 2);im=fp_weights['p3'][0, 0, 0].detach().cpu().numpy();im-=im.min();im/=im.max();plt.imshow(im);plt.title(fp_weights['p3'].size())
        # plt.subplot(2, 2, 3);im=fp['p3'][0, :3].detach().cpu().numpy().transpose(1,2,0);im-=im.min();im/=im.max();plt.imshow(im);plt.title(fp['p3'].size())
        # plt.subplot(2, 2, 4);im=fp_merge['p3'][0, :3].detach().cpu().numpy().transpose(1,2,0);im-=im.min();im/=im.max();plt.imshow(im);plt.title(fp_merge['p3'].size())
        # plt.show()
        return fp_merge


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneFPNTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        model = GeneralizedRCNNFinetuneFPCorr.create_from_sup(model)
        optimizer = self.build_optimizer(cfg, model)

        cfg_src = cfg.clone()
        cfg_src.DATASETS.TRAIN = cfg_src.DATASETS.TRAIN_SOURCE
        data_loader_src = self.build_train_loader(cfg_src)
        cfg_tgt = cfg.clone()
        cfg_tgt.DATASETS.TRAIN = cfg_tgt.DATASETS.TRAIN_TARGET
        data_loader_tgt = self.build_train_loader(cfg_tgt)
        model = create_ddp_model(model, broadcast_buffers=False)

        assert not cfg.SOLVER.AMP.ENABLED
        self._trainer = SimpleTrainer(model, data_loader_src, optimizer)
        del self._trainer.data_loader, self._trainer._data_loader_iter
        self._trainer.data_loader_src = data_loader_src
        self._trainer._data_loader_iter_src = iter(self._trainer.data_loader_src)
        self._trainer.data_loader_tgt = data_loader_tgt
        self._trainer._data_loader_iter_tgt = iter(self._trainer.data_loader_tgt)
        self.data_loader_src, self.data_loader_tgt = self._trainer.data_loader_src, self._trainer.data_loader_tgt

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        hooks = self.build_hooks()
        hooks = [FeaturePyramidUpdater(self.cfg.SOLVER.UPDATE_FP_PERIOD, cfg.clone(), model)] + hooks
        self.register_hooks(hooks)
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        assert (not isinstance(model, torch.nn.DataParallel)) and (not isinstance(model, torch.nn.parallel.DistributedDataParallel))
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []

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


class FeaturePyramidUpdater(detectron2.engine.hooks.HookBase):
    def __init__(self, update_period, cfg, model):
        self.update_period = update_period
        self.cfg = cfg
        assert len(cfg.DATASETS.TRAIN_TARGET) == 1
        self.dataset_name = cfg.DATASETS.TRAIN_TARGET[0]
        self.model = model
        self.logger = logging.getLogger(__name__)

    def _get_fp_list(self):
        print('update mean feature pyramid at iteration', self.trainer.iter)
        dataset_sample = detectron2.data.get_detection_dataset_dicts(self.dataset_name, filter_empty=False, proposal_files=None)
        dataset_sample = [dataset_sample[random.randint(0, len(dataset_sample))] for i in range(0, self.cfg.SOLVER.UPDATE_FP_EXEMPLARS)]
        print('randomly sampled %d images' % len(dataset_sample))
        mapper = detectron2.data.DatasetMapper(self.cfg, False)
        dataset_sample = detectron2.data.MapDataset(dataset_sample, mapper)
        assert not isinstance(dataset_sample, torchdata.IterableDataset)
        sampler = detectron2.data.samplers.InferenceSampler(len(dataset_sample))
        loader = torchdata.DataLoader(dataset_sample, batch_size=None, sampler=sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS)

        is_training = self.model.training
        self.model.eval()
        size_list, fp_list = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, ascii=True):
                _im_tf, _fp = self.model.compute_fp([batch])
                size_list.append(_im_tf.size())
                fp_list.append(_fp)
        if is_training:
            self.model.train()
        self.model.fp_corr.update_fp(fp_list, size_list)

    def after_step(self):
        if self.trainer.iter == 0 or (self.trainer.iter % self.update_period) == 0:
            self._get_fp_list()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._get_fp_list()


def fpn_corr_finetune_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    # alternate between source domain & target domain
    if 0 == (self.iter % 2):
        data = next(self._data_loader_iter_src)
        is_tgt = False
    else:
        data = next(self._data_loader_iter_tgt)
        is_tgt = True
    data_time = time.perf_counter() - start
    loss_dict = self.model(data, is_tgt=is_tgt)
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

    assert args.train_on_coco
    random.seed(42)
    dst_cocotrain = get_coco_dicts(args, 'train')
    random.shuffle(dst_cocotrain)
    dst_cocotrain = dst_cocotrain[:len(dst_pseudo_anno)]
    desc_cocotrain = 'cocotrain%d' % len(dst_cocotrain)
    print('include MSCOCO2017 training images, totally %d images' % (len(dst_pseudo_anno) + len(dst_cocotrain)))
    for i in range(0, len(dst_pseudo_anno)):
        dst_pseudo_anno[i]['image_id'] = i + 1
    for i in range(0, len(dst_cocotrain)):
        dst_cocotrain[i]['image_id'] = i + 1
    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes
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

    cfg.DATASETS.TRAIN_SOURCE = (desc_cocotrain,)
    cfg.DATASETS.TRAIN_TARGET = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid, desc_cocovalid)
    cfg.SOLVER.UPDATE_FP_PERIOD = args.iters // 20
    cfg.SOLVER.UPDATE_FP_EXEMPLARS = args.fp_exemplars

    # https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.TEST.AUG.ENABLED = False
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneFPNTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(fpn_corr_finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader_tgt.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader_tgt.dataset.dataset.dataset._map_func._obj = DatasetMapperMixup.create_from_sup(trainer.data_loader_tgt.dataset.dataset.dataset._map_func._obj, args.mixup_p, args.mixup_r, args.mixup_overlap_thres, args.mixup_random_position)
    trainer.resume_or_load(resume=False)
    trainer.model.proposal_generator_merge.load_state_dict(trainer.model.proposal_generator.state_dict())
    trainer.model.roi_heads_merge.load_state_dict(trainer.model.roi_heads.state_dict())

    prefix = 'adapt%s_%s_anno_%s_%s_fpncorr%s' % (args.id, args.model, desc_pseudo_anno, desc_cocotrain, '_mixup' if args.mixup_p > 0 else '')
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
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
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

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
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
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorFPCorr(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        _sd = torch.load(cfg.MODEL.WEIGHTS)
        fp_dims = {
            k: list(map(int, _sd['fp_corr.mean_fp.' + k].size()))
            for k in ['p2', 'p3', 'p4', 'p5', 'p6']
        }
        self.model = GeneralizedRCNNFinetuneFPCorr.create_from_sup(self.model, fp_keys=list(fp_dims.keys()), image_h=_sd['fp_corr.H'].size(0), image_w=_sd['fp_corr.W'].size(0), fp_dims=fp_dims)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, original_image, is_tgt):
        with torch.no_grad():
            if self.input_format == 'RGB':
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            inputs = {'image': image, 'height': height, 'width': width}
            predictions = self.model([inputs], is_tgt=is_tgt)[0]
            return predictions


def evaluate(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.TEST.AUG.ENABLED = False
    detector = PredictorFPCorr(cfg)

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
        instances = detector(im_arr, is_tgt=True)['instances'].to('cpu')
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
            instances = detector(im_arr[:, :, ::-1], is_tgt=False)['instances'].to('cpu')
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
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.TEST.AUG.ENABLED = False
    detector = PredictorFPCorr(cfg)
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
            detector.model.inference([{'image': images_tensor[i % len(images)], 'height': images[i % len(images)]['height'], 'width': images[i % len(images)]['width']}], is_tgt=True)
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
    parser.add_argument('--fp_exemplars', type=int, default=64, help='number of samples for average feature pyramid computing')

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
python finetune_fpn_correlation.py --id 001 --opt adapt --model r101-fpn-3x --anno_models r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 0 --iters 300 --eval_interval 101 --train_on_coco 1 --image_batch_size 2 --not_eval_coco 1 --lr 0.01 --mixup_p 0

python finetune_fpn_correlation.py --opt eval --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --id 001 --ckpt adapt001_r101-fpn-3x_anno_refine_r101-fpn-3x_cocotrain27000_fpncorr.pth

python finetune_fpn_correlation.py --opt tp --id 001 --model r101-fpn-3x --ckpt F:\\intersections_results\\cvpr24\\fpcorr_r101\\adapt001_r101-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain27000_fpncorr.pth

python compare_baselines.py --opt compare --model r101-fpn-3x --arch fpncorr --compare_ckpt_dir F:\\intersections_results\\cvpr24\\fpcorr_r101
'''
