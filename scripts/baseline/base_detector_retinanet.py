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


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_retinanet')



def get_cfg_base_model_retinanet(levels='101', ckpt=None):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    assert levels in ['50', '101']
    cfg = get_cfg()
    config_path = 'COCO-Detection/retinanet_R_%s_FPN_3x.yaml' % levels
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.RETINANET.NUM_CLASSES = len(thing_classes)
    if not ckpt is None:
        assert os.access(ckpt, os.R_OK), '%s not readable' % ckpt
        cfg.MODEL.WEIGHTS = os.path.normpath(ckpt)
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)

    print('detectron2 model:', config_path)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.RETINANET.NUM_CLASSES)
    return cfg


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(FinetuneTrainer, self).__init__(cfg)
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


'''
envs\detectron2\Lib\site-packages\fvcore\nn\focal_loss.py:
sigmoid_focal_loss_jit = sigmoid_focal_loss
'''
def simple_trainer_run_step(self):
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


def train_base(args):
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(args, 'valid')
    desc_cocotrain, dst_cocotrain = 'mscoco2017_train_remap', get_coco_dicts(args, 'train')

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model_retinanet('101', ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model_retinanet('101')
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
    cfg.DATASETS.TRAIN = (desc_cocotrain,)
    cfg.DATASETS.TEST = (desc_cocovalid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    prefix = 'mscoco2017_remap_retinanet_r101'
    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)
    if args.stride_alternate:
        assert isinstance(trainer.model.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        assert trainer.model.backbone.bottom_up.stem.conv1.stride == (2, 2)
        trainer.model.backbone.__class__ = FPNStrideAlternate
        prefix = prefix + '_stride_alternate'

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
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP'])
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


class FPNStrideAlternate(detectron2.modeling.backbone.FPN):
    def forward(self, x):
        if self.training:
            ret = []
            for i in range(0, x.size(0)):
                r = random.uniform(0, 1)
                if r < 1.0 / 3:
                    ret.append(super(FPNStrideAlternate, self).forward(x[i : i + 1]))
                elif r < 2.0 / 3:
                    self.bottom_up.stem.conv1.stride = (3, 3)
                    ret.append(super(FPNStrideAlternate, self).forward(torch.nn.functional.interpolate(x[i : i + 1], scale_factor=1.5, mode='bilinear')))
                    self.bottom_up.stem.conv1.stride = (2, 2)
                else:
                    self.bottom_up.stem.conv1.stride = (4, 4)
                    ret.append(super(FPNStrideAlternate, self).forward(torch.nn.functional.interpolate(x[i : i + 1], scale_factor=2, mode='bilinear')))
                    self.bottom_up.stem.conv1.stride = (2, 2)
            ret = {k: torch.cat([r[k] for r in ret], dim=0) for k in ret[0]}
            return ret
        else:
            return super(FPNStrideAlternate, self).forward(x)


class RetinaNetFeatureScaling(detectron2.modeling.meta_arch.RetinaNet):
    def _scale_input(self, batched_inputs, images):
        images.tensor = torch.nn.functional.interpolate(images.tensor, scale_factor=self.input_scale, mode='bilinear')
        image_sizes_scale = []
        for (h, w) in images.image_sizes:
            image_sizes_scale.append((int(h * self.input_scale), int(w * self.input_scale)))
        images.image_sizes = image_sizes_scale
        for inp in batched_inputs:
            if not 'instances' in inp:
                continue
            inst = inp['instances']
            inst._image_size = (int(inst._image_size[0] * self.input_scale), int(inst._image_size[1] * self.input_scale))
            inst.gt_boxes.tensor *= self.input_scale
        return batched_inputs, images

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        if abs(self.input_scale - 1.0) > 0.01:
            # print(images.tensor.size(), images.image_sizes)
            batched_inputs, images = self._scale_input(batched_inputs, images)
            # print(images.tensor.size(), images.image_sizes)
        # images.tensor = torch.nn.functional.interpolate(images.tensor, scale_factor=2, mode='bilinear')
        # images.tensor = torch.nn.functional.interpolate(images.tensor, scale_factor=1.5, mode='bilinear')
        features = self.backbone(images.tensor)
        # for k in features: print(k, features[k].size())
        # exit()
        # plt.figure(); legends = []
        # for k in features:
        #     hist, bin_edges = np.histogram(features[k].flatten().detach().cpu().numpy(), bins=10); hist = hist / hist.sum(); plt.plot(bin_edges[1:], hist); legends.append(k)
        # plt.legend(legends); plt.show()
        if self.add_p2:
            features['p2'] = torch.nn.functional.interpolate(features['p3'], scale_factor=2, mode='nearest')
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), 'Not supported'
            assert 'instances' in batched_inputs[0], 'Instance annotations are missing in training!'
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
            return self.forward_training(images, features, predictions, gt_instances)
        else:
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

    # def forward_training(self, images, features, predictions, gt_instances):
    #     # Transpose the Hi*Wi*A dimension to the middle:
    #     pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(predictions, [self.num_classes, 4])
    #     anchors = self.anchor_generator(features)
    #     gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
    #     return self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

    # def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
    #     num_images = len(gt_labels)
    #     gt_labels = torch.stack(gt_labels)  # (N, R)

    #     valid_mask = gt_labels >= 0
    #     pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
    #     num_pos_anchors = pos_mask.sum().item()
    #     detectron2.utils.events.get_event_storage().put_scalar('num_pos_anchors', num_pos_anchors / num_images)
    #     normalizer = self._ema_update('loss_normalizer', max(num_pos_anchors, 1), 100)

    #     # classification and regression loss
    #     gt_labels_target = torch.nn.functional.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1] # no loss for the last (background) class
    #     loss_cls = sigmoid_focal_loss_jit(
    #         detectron2.layers.cat(pred_logits, dim=1)[valid_mask],
    #         gt_labels_target.to(pred_logits[0].dtype),
    #         alpha=self.focal_loss_alpha,
    #         gamma=self.focal_loss_gamma,
    #         reduction='sum',
    #     )
    #     loss_box_reg = detectron2.modeling.box_regression._dense_box_regression_loss(
    #         anchors,
    #         self.box2box_transform,
    #         pred_anchor_deltas,
    #         gt_boxes,
    #         pos_mask,
    #         box_reg_loss_type=self.box_reg_loss_type,
    #         smooth_l1_beta=self.smooth_l1_beta,
    #     )
    #     return {'loss_cls': loss_cls / normalizer, 'loss_box_reg': loss_box_reg / normalizer}

    # def forward_inference(self, images: ImageList, features: List[torch.Tensor], predictions: List[List[torch.Tensor]]):
    #     pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(predictions, [self.num_classes, 4])
    #     anchors = self.anchor_generator(features)

    #     results: List[Instances] = []
    #     for img_idx, image_size in enumerate(images.image_sizes):
    #         scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
    #         deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
    #         results_per_image = self.inference_single_image(anchors, scores_per_image, deltas_per_image, image_size)
    #         results.append(results_per_image)
    #     return results

    def inference_single_image(self, anchors: List[Boxes], box_cls: List[torch.Tensor], box_delta: List[torch.Tensor], image_size: Tuple[int, int]):
        if self.add_p2:
            keep_idxs = np.arange(0, box_cls[0].size(0) - 0.1, 8).astype(np.int32)
            box_cls[0], box_delta[0], anchors[0].tensor = box_cls[0][keep_idxs], box_delta[0][keep_idxs], anchors[0].tensor[keep_idxs]
        if abs(1 - self.anchors_ratio) > 0.01:
            for i in range(0, len(anchors)): # each level: [H x W x A]
                keep_idxs = np.arange(0, box_cls[i].size(0) - 0.1, 1 / self.anchors_ratio).astype(np.int32)
                box_cls[i], box_delta[i], anchors[i].tensor = box_cls[i][keep_idxs], box_delta[i][keep_idxs], anchors[i].tensor[keep_idxs]
        pred = self._decode_multi_level_predictions(anchors, box_cls, box_delta, self.test_score_thresh, self.test_topk_candidates, image_size)
        keep = detectron2.layers.batched_nms(pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh) # per-class NMS
        return pred[keep[: self.max_detections_per_image]]

    @staticmethod
    def create_from_sup(net, input_scale, anchors_ratio, add_p2=False):
        assert isinstance(net, detectron2.modeling.meta_arch.RetinaNet), 'network is not detectron2.modeling.meta_arch.RetinaNet'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.__class__ = RetinaNetFeatureScaling
        assert anchors_ratio <= 1.0
        net.input_scale, net.anchors_ratio = input_scale, anchors_ratio
        net.add_p2 = add_p2
        if add_p2:
            assert net.head_in_features == ['p3', 'p4', 'p5', 'p6', 'p7']
            net.head_in_features = ['p2'] + net.head_in_features
        return net


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorScaling(DefaultPredictor):
    def __init__(self, cfg, input_scale, anchors_ratio, add_p2=False):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.RetinaNet), 'model class mismatch'
        self.model = RetinaNetFeatureScaling.create_from_sup(self.model, input_scale, anchors_ratio, add_p2)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
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
    results_file = 'results_AP_retinanet_i%.2f_a%.2f%s%s' % (args.input_scale, args.anchors_ratio, '_addp2' if args.add_p2 else '', args.tag)
    print(results_file)
    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)

    detector = PredictorScaling(cfg, args.input_scale, args.anchors_ratio, args.add_p2)
    print(detector.model.backbone.bottom_up.stem)
    print(detector.model.head)
    print(detector.model.anchor_generator)
    if args.add_p2:
        print('add p2 level at inference time')
        cfg.MODEL.RETINANET.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7']
        sizes_p3 = cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[s / 2 for s in sizes_p3]] + cfg.MODEL.ANCHOR_GENERATOR.SIZES
        backbone_shape = detector.model.backbone.output_shape()
        backbone_shape['p2'] = detectron2.layers.ShapeSpec(channels=256, stride=4)
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = detectron2.modeling.meta_arch.retinanet.RetinaNetHead(cfg, feature_shapes)
        head.load_state_dict(detector.model.head.state_dict())
        anchor_generator = detectron2.modeling.anchor_generator.build_anchor_generator(cfg, feature_shapes)
        anchor_generator.load_state_dict(detector.model.anchor_generator.state_dict())
        print(head)
        print(anchor_generator)
        detector.model.head, detector.model.anchor_generator = head.to(detector.model.backbone.bottom_up.stem.conv1.weight.device), anchor_generator.to(detector.model.backbone.bottom_up.stem.conv1.weight.device)

    # detector.model.backbone.bottom_up.stem.conv1 = torch.nn.Sequential(detector.model.backbone.bottom_up.stem.conv1, torch.nn.MaxPool2d(2, stride=2))
    # detector.model.backbone.bottom_up.stem.conv1.stride = (4, 4)
    # detector.model.backbone.bottom_up.stem.conv1.stride = (3, 3)
    # detector.model.backbone.bottom_up.res2[1].shortcut = torch.nn.MaxPool2d(1, stride=2)
    # detector.model.backbone.bottom_up.res2[1].conv2.stride = (2, 2)
    # detector.model.backbone.bottom_up.res4[1].shortcut = torch.nn.MaxPool2d(1, stride=2)
    # detector.model.backbone.bottom_up.res4[1].conv2.stride = (2, 2)

    if not args.not_compute_loss:
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)
        cfg.INPUT.RANDOM_FLIP = 'none'
        mapper = detectron2.data.DatasetMapper(**detectron2.data.DatasetMapper.from_config(cfg, is_train=True))
        images = []
        for video_id in video_id_list:
            inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _images_v = json.load(fp)
                for im in _images_v:
                    im['file_name'] = os.path.normpath(os.path.join(inputdir, 'masked', im['file_name']))
                images = images + _images_v
        detector.model.train()
        losses = {}
        with detectron2.utils.events.EventStorage() as storage:
            for i, im in tqdm.tqdm(enumerate(images), total=len(images), ascii=True, desc='computing losses'):
                if (i % 100) == 1:
                    torch.cuda.empty_cache() # for some reasons upscaling features causes VRAM leak
                with torch.no_grad():
                    L = detector.model([mapper(im)])
                for k in L:
                    if not k in losses:
                        losses[k] = []
                    losses[k].append(float(L[k].item()))
        losses = {k: np.array(losses[k]) for k in losses}
        losses = ' '.join(['%s %.4f(%.4f)' % (k, losses[k].mean(), losses[k].std()) for k in losses])
        print(losses)
        detector.model.eval()
    else:
        losses = 'loss computation skipped'

    t_total, N_total = 0, 0
    for video_id in video_id_list:
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
        print('[%d/%d finished in %.1f minutes]\n' % (video_id_list.index(video_id) + 1, len(video_id_list), t_total / 60))
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
    plt.suptitle('%s [%.3f ms/image] %s' % (results_file, t_total * 1000 / N_total, losses))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(args.outputdir, results_file + '.pdf'))
    plt.close()
    print('saved to:', results_file)


def evaluate_bdd100k(args):
    import contextlib
    from evaluation import eval_AP
    from bdd100k import get_bdd100k_dicts
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    images = get_bdd100k_dicts(args.bdddir, 'val')
    loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [im['file_name'] for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
    images_HW, detections = [], []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting BDD100K val'):
        im['height'], im['width'] = im_arr.shape[0], im_arr.shape[1]
        images_HW.append(im)
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        det = copy.deepcopy(im)
        det['annotations'] = []
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = eval_AP(images_HW, detections)
    del results['raw']
    print(   '             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))


def evaluate_kitti(args):
    import contextlib
    from evaluation import eval_AP
    from kitti import get_kitti_dicts
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    images = get_kitti_dicts(args.kittidir)
    loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [im['file_name'] for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
    images_HW, detections = [], []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting KITTI'):
        im['height'], im['width'] = im_arr.shape[0], im_arr.shape[1]
        images_HW.append(im)
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        det = copy.deepcopy(im)
        det['annotations'] = []
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = eval_AP(images_HW, detections)
    del results['raw']
    print(   '             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))


def evaluate_cityscapes(args):
    import contextlib
    from evaluation import eval_AP
    from cityscapes import get_cityscapes_dicts
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    images_train = get_cityscapes_dicts(args.cityscapesdir, 'train')
    images_val = get_cityscapes_dicts(args.cityscapesdir, 'val')
    images = images_train + images_val
    del images_train, images_val
    for i in range(0, len(images)):
        images[i]['image_id'] = i + 1
    loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [im['file_name'] for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
    detections = []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting CityScapes trainval'):
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        det = copy.deepcopy(im)
        det['annotations'] = []
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = eval_AP(images, detections)
    del results['raw']
    print(   '             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))


def evaluate_coco(args):
    import contextlib
    from evaluation import eval_AP
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    args.smallscale = False
    images = get_coco_dicts(args, 'valid')
    for im in images:
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
    loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [im['file_name'] for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
    detections = []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting MSCOCO2017 valid'):
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        det = copy.deepcopy(im)
        det['annotations'] = []
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = eval_AP(images, detections)
    del results['raw']
    print(   '             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))


def evaluate_scenes100_whole(args):
    import contextlib
    from evaluation import eval_AP
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    with open(os.path.join(args.scenes100dir, 'images.json'), 'r') as fp:
        images = json.load(fp)
        for im in images:
            im['file_name'] = os.path.join(args.scenes100dir, im['file_name'])
    loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [im['file_name'] for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
    detections = []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting scenes100 valid'):
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        det = copy.deepcopy(im)
        det['annotations'] = []
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = eval_AP(images, detections)
    del results['raw']
    print(   '             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))


def evaluate_base_all_videos():
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset
    categories = ['person', 'vehicle', 'overall', 'weighted']
    cfg = get_cfg_base_model_retinanet(ckpt=os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_retinanet_r101.pth'))
    detector = DefaultPredictor(cfg)
    results_AP, output_json = {}, os.path.join(os.path.dirname(__file__), 'results_AP_base_retinanet')
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        detections = []
        loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name'])) for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
        torch.cuda.empty_cache()
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
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_AP[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        del results_AP[video_id]['raw']
        print(   '             %s' % '/'.join(results_AP[video_id]['metrics']))
        for c in sorted(results_AP[video_id]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results_AP[video_id]['results'][c])))
    with open(output_json + '.json', 'w') as fp:
        json.dump(results_AP, fp, indent=2)

    videos = sorted(list(results_AP.keys()))
    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(4, 1, figsize=(28, 28))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        mAP_AP50 = np.array([results_AP[v]['results'][categories[i]] for v in videos]) * 100
        valid_mask = mAP_AP50[:, 0] >= 0
        axes[i].plot(xs[valid_mask], mAP_AP50[valid_mask, 0], 'rx-')
        axes[i].plot(xs[valid_mask], mAP_AP50[valid_mask, 1], 'bx-')
        axes[i].legend([
            'mAP valid mean: %.4f' % mAP_AP50[valid_mask, 0].mean(),
            'AP50 valid mean: %.4f' % mAP_AP50[valid_mask, 1].mean(),
        ])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        axes[i].set_ylim(0, 105)
        axes[i].set_ylabel('AP (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    # plt.tight_layout()
    plt.suptitle(output_json)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(output_json + '.pdf')
    plt.close()


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    detector = PredictorScaling(cfg, args.input_scale, args.anchors_ratio)

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


if __name__ == '__main__':
    # evaluate_base_all_videos(); exit()
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--input_scale', type=float, default=1.0)
    parser.add_argument('--anchors_ratio', type=float, default=1.0)
    parser.add_argument('--add_p2', type=bool, default=False)
    parser.add_argument('--not_compute_loss', type=bool, default=False)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--smallscale', default=False, type=bool)
    parser.add_argument('--bdddir', type=str, help='BDD100K directory')
    parser.add_argument('--cityscapesdir', type=str, help='CityScapes directory')
    parser.add_argument('--kittidir', type=str, help='KITTI directory')
    parser.add_argument('--scenes100dir', type=str, help='scenes100 directory')

    parser.add_argument('--stride_alternate', type=bool, default=False)
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

    if args.opt == 'base':
        if args.ddp_num_gpus <= 1:
            train_base(args)
        else:
            from detectron2.engine import launch
            launch(train_base, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    elif args.opt == 'tp':
        inference_throughput(args)
    elif args.opt == 'eval':
        evaluate_all_videos(args)
    elif args.opt == 'coco':
        evaluate_coco(args)
    elif args.opt == 'scenes100':
        evaluate_scenes100_whole(args)
    elif args.opt == 'bdd':
        evaluate_bdd100k(args)
    elif args.opt == 'cityscapes':
        evaluate_cityscapes(args)
    elif args.opt == 'kitti':
        evaluate_kitti(args)

'''
python base_detector_retinanet.py --opt base --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 101 --image_batch_size 2 --num_workers 2 --smallscale 1
python base_detector_retinanet.py --opt base --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1500 --image_batch_size 4 --num_workers 4
python base_detector_retinanet.py --opt base --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1500 --image_batch_size 4 --num_workers 4 --stride_alternate 1

python base_detector_retinanet.py --opt tp --ckpt mscoco2017_remap_retinanet_r101.pth --id 001 --input_scale 1
python base_detector_retinanet.py --opt eval --ckpt mscoco2017_remap_retinanet_r101.pth --input_scale 1
python base_detector_retinanet.py --opt eval --ckpt mscoco2017_remap_retinanet_r101_stride_alternate.pth --tag _stride_alternate --input_scale 1

python base_detector_retinanet.py --opt coco --ckpt mscoco2017_remap_retinanet_r101.pth --cocodir ../../../MSCOCO2017 --input_scale 1
python base_detector_retinanet.py --opt scenes100 --ckpt mscoco2017_remap_retinanet_r101.pth --scenes100dir F:\\self_drive_datasets\\Scenes100 --input_scale 1
python base_detector_retinanet.py --opt bdd --ckpt mscoco2017_remap_retinanet_r101.pth --bdddir F:\\self_drive_datasets\\BDD100K\\images_100k --input_scale 1
python base_detector_retinanet.py --opt cityscapes --ckpt mscoco2017_remap_retinanet_r101.pth --cityscapesdir F:\\self_drive_datasets\\CityScapes --input_scale 1
python base_detector_retinanet.py --opt kitti --ckpt mscoco2017_remap_retinanet_r101.pth --kittidir F:\\self_drive_datasets\\KITTI --input_scale 1

'''
