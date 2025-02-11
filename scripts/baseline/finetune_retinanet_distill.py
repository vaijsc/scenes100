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
from detectron2.structures import BoxMode, Boxes
from detectron2.structures import ImageList, Instances
from fvcore.nn import sigmoid_focal_loss_jit

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_detector_train import get_coco_dicts
from base_detector_retinanet import get_cfg_base_model_retinanet


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_distill_retinanet')


# wrap detectron2/engine/defaults.py:DefaultTrainer
class DistillationTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        model = RetinaNetDistillation.create_from_sup(model)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
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


def distillation_trainer_run_step(self):
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
    self.optimizer.step()
    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


def get_unlabeled_dicts(args):
    lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', args.id))
    with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
        meta = json.load(fp)
    ifilelist = meta['ifilelist']
    dict_json = []
    for i in range(0, len(ifilelist)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': [{'bbox': [10, 10, 20, 20], 'iscrowd': 0, 'bbox_mode': BoxMode.XYWH_ABS, 'segmentation': [], 'area': 400, 'category_id': 0}]})
    print('unlabeled frames of video %s at %s: %d images' % (args.id, lmdb_path, len(dict_json)))
    return dict_json

def all_unlabeled_dicts(args, total_images):
    random.seed(42)
    images_per_video_cap = int(total_images / len(video_id_list))
    dict_json_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        dict_json_v = get_unlabeled_dicts(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all = dict_json_all + dict_json_v
    args.id = id_back
    for i in range(0, len(dict_json_all)):
        dict_json_all[i]['image_id'] = i + 1
    print('all videos %d images' % len(dict_json_all))
    return dict_json_all


def train_distillation(args):
    from finetune import get_annotation_dict
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    if args.id == 'coco':
        desc_valid, dst_valid = 'mscoco2017_valid_remap', get_coco_dicts(args, 'valid')
        desc_train, dst_train = 'mscoco2017_train_remap', get_coco_dicts(args, 'train')
    elif args.id in video_id_list:
        desc_valid, dst_valid = '%s_manual' % args.id, get_annotation_dict(args)
        desc_train, dst_train = '%s_unlabeled_coco' % args.id, get_unlabeled_dicts(args)
        dst_cocotrain = get_coco_dicts(args, 'train')
        random.shuffle(dst_cocotrain)
        dst_train.extend(dst_cocotrain[: len(dst_train) // 3])
    elif args.id == 'compound':
        desc_valid, dst_valid, id_back = 'compound_manual', [], args.id
        for v in video_id_list:
            args.id = v
            dst_valid.extend(get_annotation_dict(args))
        for i in range(0, len(dst_valid)):
            dst_valid[i]['image_id'] = i + 1
        args.id = id_back
        desc_train, dst_train = 'compound_unlabeled_coco', all_unlabeled_dicts(args, args.image_batch_size * args.iters * 1.25)
        dst_cocotrain = get_coco_dicts(args, 'train')
        random.shuffle(dst_cocotrain)
        dst_train.extend(dst_cocotrain[: len(dst_train) // 3])
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    else:
        raise NotImplementedError
    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_valid, lambda: dst_valid)
    MetadataCatalog.get(desc_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_train, lambda: dst_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes

    assert (args.ckpt is not None) and os.access(args.ckpt, os.R_OK)
    print('loading checkpoint:', args.ckpt)
    cfg = get_cfg_base_model_retinanet('101', ckpt=args.ckpt)
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
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_valid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    prefix = 'distill_retinanet_%s' % args.id
    trainer = DistillationTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(distillation_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)

    results_0 = {}
    assert len(trainer.cfg.DATASETS.TEST) == 1
    print('Evaluate on %s' % trainer.cfg.DATASETS.TEST[0])
    data_loader = trainer.build_test_loader(trainer.cfg, trainer.cfg.DATASETS.TEST[0])
    evaluator = trainer.build_evaluator(trainer.cfg, trainer.cfg.DATASETS.TEST[0])
    results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
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

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Valid AP50', 'Valid mAP'])
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
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


class RetinaNetDistillation(detectron2.modeling.meta_arch.RetinaNet):
    def _add_p2_to_list(self, features: Dict[str, torch.Tensor], feature_res2: torch.Tensor):
        fused_features = torch.nn.functional.interpolate(features['p3'], scale_factor=2.0, mode='nearest') + self.fpn_lateral2(feature_res2)
        if self.backbone._fuse_type == 'avg':
            fused_features /= 2
        features_p2 = self.fpn_output2(fused_features)
        features = [features[f] for f in self.head_in_features]
        features = [features_p2] + features[:-1] # shift one level downward p3~p7 -> p2~p6
        return features

    def forward_distillation_loss(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert self.training
        assert not torch.jit.is_scripting(), 'Not supported'

        images = self.preprocess_image(batched_inputs)
        images.tensor = torch.nn.functional.interpolate(images.tensor, scale_factor=2, mode='bilinear', align_corners=True)
        B = images.tensor.size(0)
        features, feature_res2 = self.backbone(images.tensor)
        features = self._add_p2_to_list(features, feature_res2)
        HW_levels = [(f.size(2), f.size(3)) for f in features]
        predictions = self.head(features)
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(predictions, [self.num_classes, 4])
        with torch.no_grad():
            features_x2 = self.backbone_teacher(images.tensor)
            features_x2 = [features_x2[f] for f in self.head_in_features]
            HW_x2_levels = [(f.size(2), f.size(3)) for f in features_x2]
            predictions_x2 = self.head_teacher(features_x2)
            pred_logits_x2, pred_anchor_deltas_x2 = self._transpose_dense_predictions(predictions_x2, [self.num_classes, 4])
        # each level: B x (H x W x A) x num_classes/4

        losses_scores, losses_deltas = [], []
        for p_logits, p_deltas, (H, W), p_logits_x2, p_deltas_x2, (H2, W2) in zip(pred_logits, pred_anchor_deltas, HW_levels, pred_logits_x2, pred_anchor_deltas_x2, HW_x2_levels):
            assert H == H2 and W == W2
            scores = p_logits.view(B, H, W, -1, self.num_classes).sigmoid_()
            p_deltas = p_deltas.view(B, H, W, -1, 4)

            # cut gradient for teacher
            scores_x2 = p_logits_x2.view(B, H2, W2, -1, self.num_classes).sigmoid_().detach()
            p_deltas_x2 = p_deltas_x2.view(B, H2, W2, -1, 4).detach()

            # focus on activated regions, note that box regression is class-agnostic
            weights = (torch.cat([scores.detach(), scores_x2], dim=4)).max(dim=4, keepdims=True).values.detach()

            # plt.figure()
            # _im1 = scores[0, :, :, :3, 1].detach().cpu().numpy(); plt.subplot(2, 2, 1); plt.imshow(_im1); plt.title('x1')
            # _im2 = scores_x2[0, :, :, :3, 1].detach().cpu().numpy(); plt.subplot(2, 2, 2); plt.imshow(_im2); plt.title('x2')
            # _im = images.tensor[0].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]; _im-=_im.min(); _im/=_im.max(); plt.subplot(2, 2, 3); plt.imshow(_im)
            # _imw = weights[0, :, :, :3, 0].detach().cpu().numpy(); plt.subplot(2, 2, 4); plt.imshow(_imw); plt.title('weights')
            # plt.show()

            # losses
            L_s = torch.nn.functional.kl_div(scores.log(), scores_x2, log_target=False, reduction='none') + torch.nn.functional.kl_div((1.0 - scores).log(), 1.0 - scores_x2, log_target=False, reduction='none')
            L_d = torch.nn.functional.mse_loss(p_deltas, p_deltas_x2, reduction='none')
            losses_scores.append((L_s * weights).sum() / weights.sum())
            losses_deltas.append((L_d * weights).sum() / weights.sum())
        return {'distill_cls': sum(losses_scores) / len(losses_scores), 'distill_box': sum(losses_deltas) / len(losses_deltas)}

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if self.training:
            return self.forward_distillation_loss(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features, feature_res2 = self.backbone(torch.nn.functional.interpolate(images.tensor, scale_factor=2, mode='bilinear'))
        features = self._add_p2_to_list(features, feature_res2)
        predictions = self.head(features)

        results = self.forward_inference(images, features, predictions)
        if torch.jit.is_scripting():
            return results
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get('height', image_size[0])
            width = input_per_image.get('width', image_size[1])
            r = detectron2.modeling.postprocessing.detector_postprocess(results_per_image, height, width)
            processed_results.append({'instances': r})
        return processed_results

    def forward_inference(self, images: ImageList, features: List[torch.Tensor], predictions: List[List[torch.Tensor]]):
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(predictions, [self.num_classes, 4])

        # "pretend" image is upscaled x2 for anchors to match
        images.image_sizes = [(2 * h, 2 * w) for (h, w) in images.image_sizes]
        anchors = self.anchor_generator(features)

        # plt.figure()
        # plt.subplot(2,2,1); _im=images.tensor[0].permute(1,2,0).detach().cpu().numpy(); _im-=_im.min(); _im/=_im.max(); plt.imshow(_im)
        # xyxys = anchors[1].tensor.view(features[1].size(2),features[1].size(3),-1,4)[np.arange(0,features[1].size(2),10)][:,np.arange(0,features[1].size(3),10)][:,:,[0,7]].detach().cpu().numpy().reshape(-1,4); print(xyxys)
        # for x1,y1,x2,y2 in xyxys:
        #     plt.plot([x1,x1,x2,x2],[y1,y2,y2,y1],'r-',lw=1)
        #     plt.plot([x1,x2],[y1,y1],'r-',lw=1)
        # plt.subplot(2,2,3); _im=features[0][0][:3].permute(1,2,0).detach().cpu().numpy(); _im-=_im.min(); _im/=_im.max(); plt.imshow(_im)
        # plt.subplot(2,2,4); _im=pred_logits[0].view(features[0].size(0),features[0].size(2),features[0].size(3),-1).sigmoid_().max(dim=3).values[0].detach().cpu().numpy(); plt.imshow(_im)
        # plt.show()

        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.RetinaNet), 'network is not detectron2.modeling.meta_arch.RetinaNet'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.backbone_teacher = copy.deepcopy(net.backbone)
        net.head_teacher = copy.deepcopy(net.head) # teacher model uses x2 input, 2x2 stride
        net.backbone = FPNStudent.create_from_sup(net.backbone)
        net.backbone.bottom_up.stem.conv1.stride = (4, 4) # student model uses x2 input, 4x4 stride
        # for generating p2 feature
        net.fpn_lateral2 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0).to(net.backbone.bottom_up.stem.conv1.weight.device)
        net.fpn_output2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1).to(net.backbone.bottom_up.stem.conv1.weight.device)
        net.__class__ = RetinaNetDistillation
        return net


class FPNStudent(detectron2.modeling.backbone.FPN):
    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        feature_res2 = bottom_up_features['res2']
        del bottom_up_features['res2']
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = torch.nn.functional.interpolate(prev_features, scale_factor=2.0, mode='nearest')
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == 'avg':
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        feature_pyramid = {f: res for f, res in zip(self._out_features, results)}
        return feature_pyramid, feature_res2

    @staticmethod
    def create_from_sup(net):
        net.__class__ = FPNStudent
        assert net.bottom_up._out_features == ['res3', 'res4', 'res5']
        net.bottom_up._out_features.insert(0, 'res2')
        return net


def duplicate_teacher():
    state_dict = torch.load('mscoco2017_remap_retinanet_r101.pth')
    keys = list(state_dict.keys())
    for k in keys:
        if k.startswith('backbone.'):
            state_dict[k.replace('backbone.', 'backbone_teacher.')] = copy.deepcopy(state_dict[k])
        if k.startswith('head.'):
            state_dict[k.replace('head.', 'head_teacher.')] = copy.deepcopy(state_dict[k])
    torch.save(state_dict, 'mscoco2017_remap_retinanet_r101_student_teacher.pth')


class PredictorDistillation(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.RetinaNet), 'model class mismatch'
        self.model = RetinaNetDistillation.create_from_sup(self.model)
        self.model.backbone_teacher, self.model.head_teacher = None, None # teacher is not used in inference
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        self.checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format


def evaluate_all_videos(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_retinanet.json'), 'r') as fp:
        base_AP = json.load(fp)
    results = {}

    if not args.ckpt is None:
        assert args.ckpts_dir is None
        results_file = '%s_results_AP%s' % (args.ckpt, args.tag)
        print(results_file)
        cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
        detector = PredictorDistillation(cfg)
        video_ckpts = [(v, None) for v in video_id_list]
    else:
        assert not args.ckpts_dir is None
        results_file = os.path.join(args.ckpts_dir, 'results_AP%s' % args.tag)
        print(results_file)
        video_ckpts = sorted(glob.glob(os.path.join(args.ckpts_dir, 'distill_retinanet_*.pth')))
        cfg = get_cfg_base_model_retinanet(ckpt=video_ckpts[0])
        detector = PredictorDistillation(cfg)
        video_ckpts = [(os.path.basename(f)[-7 : -4], f) for f in video_ckpts]
        print('%d presented video checkpoints:' % len(video_ckpts))
        print(' '.join([v for (v, _) in video_ckpts]))
        print('missing:')
        print(' '.join(sorted(list(set(video_id_list) - set([v for (v, _) in video_ckpts])))))

    t_total, N_total = 0, 0
    for video_i, (video_id, f) in enumerate(video_ckpts):
        if not f is None:
            detector.checkpointer.load(f)
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        N_total += len(images)
        detections = []
        loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name'])) for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
        torch.cuda.empty_cache()
        t0 = time.time()
        for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting %s validation frames' % video_id):
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

            # f = Image.fromarray(im_arr); draw = ImageDraw.Draw(f)
            # for ann in det['annotations']:
            #     if ann['score'] < 0.5: continue
            #     x1, y1, x2, y2 = ann['bbox']; draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='#000000', width=3)
            # plt.figure(); plt.imshow(np.array(f)); plt.show()
        t_total += time.time() - t0
        print('[%d/%d finished in %.1f minutes]\n' % (video_i + 1, len(video_ckpts), t_total / 60))
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        results[video_id]['detections'] = detections
        print(   '             %s' % '/'.join(results[video_id]['metrics']))
        for c in sorted(results[video_id]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[video_id]['results'][c])))
    if not args.not_save_results_json:
        with open(results_file + '.json', 'w') as fp:
            json.dump(results, fp, indent=2)
    print('processed %d images in %.2f seconds, %.3f ms/image' % (N_total, t_total, t_total * 1000 / N_total))

    videos = sorted(list(results.keys()))
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    for video_id in videos:
        AP1 = base_AP[video_id]['results']
        AP2 = results[video_id]['results']
        for cat in categories:
            improvements[cat].append([AP2[cat][0] - AP1[cat][0], AP2[cat][1] - AP1[cat][1]])
    for cat in categories:
        improvements[cat] = np.array(improvements[cat]) * 100.0
    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        axes[i].plot([-1, xs.max() + 1], [0, 0], 'k-')
        axes[i].plot(xs, improvements[categories[i]][:, 0], 'r.-')
        axes[i].plot(xs, improvements[categories[i]][:, 1], 'b.-')
        axes[i].legend(['0', 'mAP %.4f' % improvements[categories[i]][:, 0].mean(), 'AP50 %.4f' % improvements[categories[i]][:, 1].mean()])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        # axes[i].set_ylim(-3, 3)
        axes[i].set_ylabel('AP improvement (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    # plt.tight_layout()
    plt.suptitle('%s [%.3f ms/image]' % (results_file, t_total * 1000 / N_total))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(args.outputdir, results_file + '.pdf'))
    plt.close()
    print('saved to:', results_file)


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    detector = PredictorDistillation(cfg)

    inputs_list = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        im_tensor = detector.aug.get_transform(im_arr).apply_image(im_arr)
        im_tensor = torch.as_tensor(im_tensor.astype('float32').transpose(2, 0, 1))
        inputs_list.append([{'image': im_tensor, 'height': im['height'], 'width': im['width']}])
    stats_all = {}
    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            detector.model(inputs_list[i % len(images)])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


def correlation():
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i1.00_a1.00.json', 'r') as fp:
        results_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i2.00_a1.00.json', 'r') as fp:
        results_x2 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\distill_retinanet\\results_AP.json', 'r') as fp:
        results_distill = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\distill_retinanet_compound_coco.pth_results_AP.json', 'r') as fp:
        results_distill_compound = json.load(fp)

    APs_x1 = np.array([results_x1[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_x2 = np.array([results_x2[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_distill = np.array([results_distill[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_distill_compound = np.array([results_distill_compound[v]['results']['weighted'] for v in video_id_list]) * 100
    print('x1:', APs_x1.mean(), 'x2:', APs_x2.mean(), 'distilled individual:', APs_distill.mean(), 'distill compound:', APs_distill_compound.mean())
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(APs_x2[:, 1] - APs_x1[:, 1], APs_distill[:, 1] - APs_x1[:, 1], marker='+', c='b', alpha=0.8)
    plt.scatter(APs_x2[:, 0] - APs_x1[:, 0], APs_distill[:, 0] - APs_x1[:, 0], marker='x', c='r', alpha=0.8)
    plt.xlabel('x2 input improvement'); plt.ylabel('distilled individually improvement'); plt.legend(['$AP50 \\uparrow$', '$mAP \\uparrow$'])
    plt.xlim(-10, 35); plt.ylim(-10, 35); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(APs_x2[:, 1] - APs_x1[:, 1], APs_distill_compound[:, 1] - APs_x1[:, 1], marker='+', c='b', alpha=0.8)
    plt.scatter(APs_x2[:, 0] - APs_x1[:, 0], APs_distill_compound[:, 0] - APs_x1[:, 0], marker='x', c='r', alpha=0.8)
    plt.xlabel('x2 input improvement'); plt.ylabel('distilled compound improvement'); plt.legend(['$AP50 \\uparrow$', '$mAP \\uparrow$'])
    plt.xlim(-10, 35); plt.ylim(-10, 35); plt.grid(True)
    plt.tight_layout()
    plt.savefig('AP_retinanet_distill_correlation.pdf')


def compare_features(args):
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i1.00_a1.00.json', 'r') as fp:
        mAP_x1 = json.load(fp)
        mAP_x1 = {v: mAP_x1[v]['results']['weighted'][0] * 100 for v in video_id_list}
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i2.00_a1.00.json', 'r') as fp:
        mAP_x2 = json.load(fp)
        mAP_x2 = {v: mAP_x2[v]['results']['weighted'][0] * 100 for v in video_id_list}
    with open('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\untrained.pth_results_AP.json', 'r') as fp:
        mAP_rand = json.load(fp)
        mAP_rand = {v: mAP_rand[v]['results']['weighted'][0] * 100 for v in video_id_list}
    with open('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\distill_retinanet_coco.pth_results_AP.json', 'r') as fp:
        mAP_coco = json.load(fp)
        mAP_coco = {v: mAP_coco[v]['results']['weighted'][0] * 100 for v in video_id_list}
    with open('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\distill_retinanet_compound_coco.pth_results_AP.json', 'r') as fp:
        mAP_comp = json.load(fp)
        mAP_comp = {v: mAP_comp[v]['results']['weighted'][0] * 100 for v in video_id_list}

    ckpt_base = 'mscoco2017_remap_retinanet_r101.pth'
    detector = DefaultPredictor(get_cfg_base_model_retinanet(ckpt=ckpt_base))
    model_base = detector.model
    model_rand = RetinaNetDistillation.create_from_sup(copy.deepcopy(model_base))
    model_coco = RetinaNetDistillation.create_from_sup(copy.deepcopy(model_base))
    model_comp = RetinaNetDistillation.create_from_sup(copy.deepcopy(model_base))
    model_rand.load_state_dict(torch.load('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\untrained.pth'))
    model_coco.load_state_dict(torch.load('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\distill_retinanet_coco.pth'))
    model_comp.load_state_dict(torch.load('F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\distill_retinanet_compound.pth'))
    model_base.eval(); model_rand.eval(); model_coco.eval(); model_comp.eval()

    for v in ['001', '006', '040', '049', '090', '135', '141', '152']:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', v)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        for im in images:
            im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
            im_tensor = detector.aug.get_transform(im_arr).apply_image(im_arr)
            im_tensor = torch.as_tensor(im_tensor.astype('float32').transpose(2, 0, 1))
            im_tensor = model_base.preprocess_image([{'image': im_tensor, 'height': im['height'], 'width': im['width']}])
            im_tensor = torch.nn.functional.interpolate(im_tensor.tensor, scale_factor=2, mode='bilinear', align_corners=True)
            print(im['file_name'], im_tensor.size())
            with torch.no_grad():
                features_base = model_base.backbone(im_tensor)
                features_base = [features_base[f] for f in model_base.head_in_features]
                logits_base, deltas_base = model_base._transpose_dense_predictions(model_base.head(features_base), [model_base.num_classes, 4])

                features_rand, _res2 = model_rand.backbone(im_tensor)
                features_rand = model_rand._add_p2_to_list(features_rand, _res2)
                logits_rand, deltas_rand = model_rand._transpose_dense_predictions(model_rand.head(features_rand), [model_rand.num_classes, 4])

                features_coco, _res2 = model_coco.backbone(im_tensor)
                features_coco = model_coco._add_p2_to_list(features_coco, _res2)
                logits_coco, deltas_coco = model_coco._transpose_dense_predictions(model_coco.head(features_coco), [model_coco.num_classes, 4])

                features_comp, _res2 = model_comp.backbone(im_tensor)
                features_comp = model_comp._add_p2_to_list(features_comp, _res2)
                logits_comp, deltas_comp = model_comp._transpose_dense_predictions(model_comp.head(features_comp), [model_comp.num_classes, 4])

            plt.figure(figsize=(20, 45))
            _im = im_tensor[0].cpu().numpy()[::-1].transpose(1, 2, 0); _im -= _im.min(); _im /= _im.max()
            plt.subplot(16, 4, 1); plt.imshow(_im); plt.axis('off'); plt.title('teacher $mAP$  $%+.2f$' % (mAP_x2[v] - mAP_x1[v]))
            plt.subplot(16, 4, 2); plt.imshow(_im); plt.axis('off'); plt.title('un-trained $mAP$  $%+.2f$' % (mAP_rand[v] - mAP_x1[v]))
            plt.subplot(16, 4, 3); plt.imshow(_im); plt.axis('off'); plt.title('MSCOCO distilled $mAP$  $%+.2f$' % (mAP_coco[v] - mAP_x1[v]))
            plt.subplot(16, 4, 4); plt.imshow(_im); plt.axis('off'); plt.title('Scenes100+MSCOCO distilled $mAP$  $%+.2f$' % (mAP_comp[v] - mAP_x1[v]))
            for i in range(0, 5):
                B, _, H, W = features_base[i].size()
                f1, f2, f3, f4 = map(lambda t: t[0].cpu().numpy().transpose(1, 2, 0)[:, :, :3], [features_base[i], features_rand[i], features_coco[i], features_comp[i]])
                f_min = min(f1.min(), f2.min(), f3.min(), f4.min()); f1, f2, f3, f4 = f1 - f_min, f2 - f_min, f3 - f_min, f4 - f_min
                f_max = max(f1.max(), f2.max(), f3.max(), f4.max()); f1, f2, f3, f4 = f1 / f_max, f2 / f_max, f3 / f_max, f4 / f_max
                plt.subplot(16, 4, 4 + i * 12 + 1); plt.imshow(f1); plt.axis('off'); plt.title('%s $%d\\times %d$' % (model_base.head_in_features[i], H, W))
                plt.subplot(16, 4, 4 + i * 12 + 2); plt.imshow(f2); plt.axis('off')
                plt.subplot(16, 4, 4 + i * 12 + 3); plt.imshow(f3); plt.axis('off')
                plt.subplot(16, 4, 4 + i * 12 + 4); plt.imshow(f4); plt.axis('off')

                p1, p2, p3, p4 = map(lambda t: t.view(B, H, W, -1, 2)[0].sigmoid_().cpu().numpy().max(axis=2), [logits_base[i], logits_rand[i], logits_coco[i], logits_comp[i]])
                p1, p2, p3, p4 = map(lambda t: np.stack([t[:, :, 0], np.zeros_like(t[:, :, 0]), t[:, :, 1]], axis=2), [p1, p2, p3, p4])
                plt.subplot(16, 4, 4 + i * 12 + 5); plt.imshow(p1); plt.axis('off'); plt.title('class score')
                plt.subplot(16, 4, 4 + i * 12 + 6); plt.imshow(p2); plt.axis('off')
                plt.subplot(16, 4, 4 + i * 12 + 7); plt.imshow(p3); plt.axis('off')
                plt.subplot(16, 4, 4 + i * 12 + 8); plt.imshow(p4); plt.axis('off')

                mse2, mse3, mse4 = map(lambda t: t.view(B, H, W, -1).square()[0].cpu().numpy().sum(axis=2), [deltas_rand[i] - deltas_base[i], deltas_coco[i] - deltas_base[i], deltas_comp[i] - deltas_base[i]])
                mse_max = max(mse2.max(), mse3.max(), mse4.max())
                mse2, mse3, mse4 = mse2 / mse_max, mse3 / mse_max, mse4 / mse_max
                plt.subplot(16, 4, 4 + i * 12 + 10); plt.imshow(mse2); plt.axis('off'); plt.title('box deltas MSE to teacher')
                plt.subplot(16, 4, 4 + i * 12 + 11); plt.imshow(mse3); plt.axis('off')
                plt.subplot(16, 4, 4 + i * 12 + 12); plt.imshow(mse4); plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.outputdir, '%s_%s.pdf' % (v, im['file_name'][:-4])))
            plt.close()
            break


if __name__ == '__main__':
    # duplicate_teacher(); exit()
    # correlation(); exit()
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound', 'coco'], help='video ID')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--smallscale', default=False, type=bool)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    parser.add_argument('--eval_outputfile', default=None, type=str)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'distill':
        if args.ddp_num_gpus <= 1:
            train_distillation(args)
        else:
            from detectron2.engine import launch
            launch(train_distillation, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    if args.opt == 'tp':
        inference_throughput(args)
    if args.opt == 'eval':
        evaluate_all_videos(args)
    if args.opt == 'show':
        compare_features(args)

'''
python finetune_retinanet_distill.py --opt distill --ckpt mscoco2017_remap_retinanet_r101_student_teacher.pth --id coco --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 101 --image_batch_size 2 --num_workers 2 --smallscale 1
python finetune_retinanet_distill.py --opt distill --ckpt mscoco2017_remap_retinanet_r101_student_teacher.pth --id coco --cocodir ../../../MSCOCO2017 --iters 60000 --eval_interval 5001 --image_batch_size 8 --num_workers 8
python finetune_retinanet_distill.py --opt distill --ckpt mscoco2017_remap_retinanet_r101_student_teacher.pth --id compound --iters 60000 --eval_interval 5001 --image_batch_size 8 --num_workers 8

python finetune_retinanet_distill.py --opt distill --ckpt mscoco2017_remap_retinanet_r101_student_teacher.pth --id 001 --iters 20000 --eval_interval 1500 --image_batch_size 4 --num_workers 4

python finetune_retinanet_distill.py --opt eval --ckpt distill_retinanet_001.pth
python finetune_retinanet_distill.py --opt eval --ckpts_dir cvpr24/distill_retinanet
python finetune_retinanet_distill.py --opt tp --ckpt distill_retinanet_001.pth --id 001

python finetune_retinanet_distill.py --opt show --outputdir F:\\intersections_results\\cvpr24\\distill_retinanet_compound\\compare
'''
