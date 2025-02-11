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
import skimage.transform
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
from detectron2.structures import ImageList, Instances, Boxes

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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_train_fpn_skip')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


def mosaic(cocodir, split, cocodir_output, scale):
    assert scale in [2, 3]
    def _stitch(images_batch):
        if len(images_batch) < scale * scale:
            images_batch = images_batch + [images_batch[-1]] * (scale * scale - len(images_batch))
        assert len(images_batch) == scale * scale, len(images_batch)
        images_np = [skimage.io.imread(im['file_name']) for im in images_batch]
        annotations_batch = []
        H, W = images_np[0].shape[:2]
        for im_i in range(0, len(images_np)):
            if len(images_np[im_i].shape) < 3:
                images_np[im_i] = np.stack([images_np[im_i], images_np[im_i], images_np[im_i]], axis=2)
            if im_i > 0:
                _s_h, _s_w = H / images_np[im_i].shape[0], W / images_np[im_i].shape[1]
                images_np[im_i] = (skimage.transform.resize(images_np[im_i], (H, W)) * 255).astype(np.uint8)
                for ann in images_batch[im_i]['annotations']:
                    x, y, w, h = ann['bbox']
                    ann['bbox'] = [x * _s_w + (im_i % scale) * W, y * _s_h + (im_i // scale) * H, w * _s_w, h * _s_h]
                    ann['area'] = ann['area'] * _s_h * _s_w
            for ann in images_batch[im_i]['annotations']:
                annotations_batch.append(ann)
        images_stitch = np.zeros(shape=(H * scale, W * scale, 3), dtype=np.uint8)
        for im_i in range(0, len(images_np)):
            images_stitch[(im_i // scale) * H : (im_i // scale + 1) * H, (im_i % scale) * W : (im_i % scale + 1) * W] = images_np[im_i]
        file_name_list = [os.path.basename(im['file_name'])[:-4] for im in images_batch]
        file_name_list = '_'.join(file_name_list)
        # f = Image.fromarray(images_stitch); draw = ImageDraw.Draw(f)
        # for ann in annotations_batch:
        #     x, y, w, h = ann['bbox']
        #     draw.line(((x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)), fill='#000000', width=2)
        # plt.figure(); plt.imshow(np.array(f)); plt.title(file_name_list); plt.show()
        return images_stitch, annotations_batch, file_name_list

    if split == 'valid':
        dataset_json = os.path.join(cocodir, 'annotations', 'instances_val2017.json')
    elif split == 'train':
        dataset_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
    else:
        return None

    with open(dataset_json, 'r') as fp:
        dataset = json.load(fp)
    images, annotations = dataset['images'], dataset['annotations']
    print('source: %d images, %d objects' % (len(images), len(annotations)))
    for im in images:
        im['file_name'] = os.path.join(cocodir, 'images', 'val2017' if split == 'valid' else 'train2017', im['file_name'])
        im['annotations'] = []
    images = {im['id']: im for im in images}
    for ann in annotations:
        del ann['segmentation']
        images[ann['image_id']]['annotations'].append(ann)

    output_images, output_dataset = [], {'categories': dataset['categories'], 'licenses': dataset['licenses'], 'info': dataset['info']}
    outputdir = os.path.join(cocodir_output, 'images', 'val2017' if split == 'valid' else 'train2017')
    aspect_buckets = {}
    for i in images:
        _r = '%.2f' % (images[i]['height'] / images[i]['width'])
        if not _r in aspect_buckets:
            aspect_buckets[_r] = []
        aspect_buckets[_r].append(i)
    for _r in aspect_buckets:
        aspect_buckets[_r].sort(key=lambda i: images[i]['height'], reverse=True)
        images_r = [images[i] for i in aspect_buckets[_r]]
        for b in tqdm.tqdm(range(0, len(images_r) // (scale * scale) + 1), ascii=True, desc='%s %d' % (_r, len(aspect_buckets[_r]))):
            images_batch = images_r[b * scale * scale : (b + 1) * scale * scale]
            if len(images_batch) > 0:
                images_stitch, annotations_batch, file_name_list = _stitch(images_batch)
                skimage.io.imsave(os.path.join(outputdir, file_name_list + '.jpg'), images_stitch, quality=90)
                output_images.append({'file_name': file_name_list + '.jpg', 'height': images_stitch.shape[0], 'width': images_stitch.shape[1], 'id': None, 'annotations': annotations_batch})
    output_annotations = []
    for i, im in enumerate(output_images):
        im['id'] = i + 1
        for ann in im['annotations']:
            ann['image_id'] = im['id']
            ann['id'] = len(output_annotations) + 1
            output_annotations.append(ann)
        del im['annotations']
    print('target: %d images, %d objects' % (len(output_images), len(output_annotations)))
    output_dataset['images'], output_dataset['annotations'] = output_images, output_annotations
    with open(os.path.join(cocodir_output, 'annotations', 'instances_val2017.json' if split == 'valid' else 'instances_train2017.json'), 'w') as fp:
        json.dump(output_dataset, fp)


class GeneralizedRCNNFPNSkip(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training: return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # at training time, train all levels, with separate losses
        features_levels = self.backbone(images.tensor, levels=['res2', 'res3', 'res4', 'res5'])

        losses = {}
        for L in features_levels:
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features_levels[L], gt_instances)
            else:
                assert 'proposals' in batched_inputs[0]
                proposals = [x['proposals'].to(self.device) for x in batched_inputs]
                proposal_losses = {}
            _, detector_losses = self.roi_heads(images, features_levels[L], proposals, gt_instances)

            losses.update({L + '_' + k : detector_losses[k] for k in detector_losses})
            losses.update({L + '_' + k : proposal_losses[k] for k in proposal_losses})

        if self.vis_period > 0:
            storage = detectron2.utils.events.get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, L: str = 'res5'):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor, levels=[L])[L]
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            return results
        else:
            return results

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers), 'roi is not detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'
        net.__class__ = GeneralizedRCNNFPNSkip
        net.backbone = FPNSkip.create_from_sup(net.backbone)
        return net


class FPNSkip(detectron2.modeling.backbone.FPN):
    def get_pyramid(self, bottom_up_features, L):
        results = []
        prev_features = self.lateral_convs_levels[L][0](bottom_up_features[self.in_features_levels[L][-1]])
        results.append(self.output_convs_levels[L][0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs_levels[L], self.output_convs_levels[L])):
            if idx > 0:
                features = self.in_features_levels[L][-idx - 1]
                features = bottom_up_features[features]
                top_down_features = torch.nn.functional.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        while len(results) < len(self._out_features):
            if len(results) == len(self._out_features) - 1:
                results.extend(self.top_block(results[-1]))
            else:
                B, C, H, W = results[-1].size()
                results.append(self.pool_2x2_levels[len(results) - 1](results[-1].view(B * C, H, W).unsqueeze(1)).squeeze(1).view(B, C, H // 2, W // 2))
        feature_pyramid = {f: res for f, res in zip(self._out_features, results)}
        return feature_pyramid

    def forward(self, x, levels=['res2', 'res3', 'res4', 'res5']):
        bottom_up_features = self.resnet_forward(x, top_level=sorted(levels)[-1])
        feature_pyramid_levels = {L: self.get_pyramid(bottom_up_features, L) for L in levels}
        # for L in feature_pyramid_levels:
        #     print(L)
        #     for k in feature_pyramid_levels[L]:
        #         print(k, feature_pyramid_levels[L][k].size())
        return feature_pyramid_levels

    def resnet_forward(self, x, top_level='res5'):
        assert top_level in ['res2', 'res3', 'res4', 'res5'], 'top_level not supported'
        assert x.dim() == 4, f'ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!'
        outputs = {}
        x = self.bottom_up.stem(x)
        assert not 'stem' in self.bottom_up._out_features
        assert self.bottom_up.num_classes is None
        for name, stage in zip(self.bottom_up.stage_names, self.bottom_up.stages):
            x = stage(x)
            if name in self.bottom_up._out_features:
                outputs[name] = x
            if name == top_level:
                break
        return outputs

    def load_lateral_output_weights(self):
        for k in self.lateral_convs_levels:
            for i in range(0, len(self.lateral_convs_levels[k])):
                self.lateral_convs_levels[k][i].load_state_dict(self.lateral_convs[i + len(self.lateral_convs) - len(self.lateral_convs_levels[k])].state_dict())
            for i in range(0, len(self.output_convs_levels[k])):
                self.output_convs_levels[k][i].load_state_dict(self.output_convs[i + len(self.output_convs) - len(self.output_convs_levels[k])].state_dict())

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.backbone.FPN), 'network is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.bottom_up, detectron2.modeling.backbone.resnet.ResNet), 'bottom_up is not detectron2.modeling.backbone.resnet.ResNet'
        assert list(net.in_features) == ['res2', 'res3', 'res4', 'res5'], 'feature list not supported'
        assert list(net.bottom_up._out_features) == ['res2', 'res3', 'res4', 'res5'], 'feature list not supported'
        net.lateral_convs_levels = torch.nn.ModuleDict({
            'res5': torch.nn.ModuleList([
                copy.deepcopy(net.lateral_convs[0]),
                copy.deepcopy(net.lateral_convs[1]),
                copy.deepcopy(net.lateral_convs[2]),
                copy.deepcopy(net.lateral_convs[3]),
            ]),
            'res4': torch.nn.ModuleList([
                copy.deepcopy(net.lateral_convs[1]),
                copy.deepcopy(net.lateral_convs[2]),
                copy.deepcopy(net.lateral_convs[3]),
            ]),
            'res3': torch.nn.ModuleList([
                copy.deepcopy(net.lateral_convs[2]),
                copy.deepcopy(net.lateral_convs[3]),
            ]),
            'res2': torch.nn.ModuleList([
                copy.deepcopy(net.lateral_convs[3]),
            ]),
        }).to(net.bottom_up.stem.conv1.weight.device)
        net.output_convs_levels = torch.nn.ModuleDict({
            'res5': torch.nn.ModuleList([
                copy.deepcopy(net.output_convs[0]),
                copy.deepcopy(net.output_convs[1]),
                copy.deepcopy(net.output_convs[2]),
                copy.deepcopy(net.output_convs[3]),
            ]),
            'res4': torch.nn.ModuleList([
                copy.deepcopy(net.output_convs[1]),
                copy.deepcopy(net.output_convs[2]),
                copy.deepcopy(net.output_convs[3]),
            ]),
            'res3': torch.nn.ModuleList([
                copy.deepcopy(net.output_convs[2]),
                copy.deepcopy(net.output_convs[3]),
            ]),
            'res2': torch.nn.ModuleList([
                copy.deepcopy(net.output_convs[3]),
            ]),
        }).to(net.bottom_up.stem.conv1.weight.device)
        net.pool_2x2_levels = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 1, 2, stride=2, padding=0),
            torch.nn.Conv2d(1, 1, 2, stride=2, padding=0),
            torch.nn.Conv2d(1, 1, 2, stride=2, padding=0),
        ]).to(net.bottom_up.stem.conv1.weight.device)
        net.in_features_levels = {
            'res5': ('res2', 'res3', 'res4', 'res5'),
            'res4': ('res2', 'res3', 'res4'),
            'res3': ('res2', 'res3'),
            'res2': ('res2',),
        }
        net.__class__ = FPNSkip
        return net


class TrainerFPNSkip(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        model = GeneralizedRCNNFPNSkip.create_from_sup(model)
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
        self._trainer.lr_history, self._trainer.loss_history = [], []

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


def finetune_simple_trainer_run_step(self):
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


def train_base(args):
    _args = copy.deepcopy(args)
    print('read from:', _args.cocodir)
    desc_cocotrain_x1, dst_cocotrain_x1 = 'mscoco2017_train_remap_x1', get_coco_dicts(_args, 'train')
    desc_cocovalid_x1, dst_cocovalid_x1 = 'mscoco2017_valid_remap_x1', get_coco_dicts(_args, 'valid')
    _args.cocodir = args.cocodir_x2
    print('read from:', _args.cocodir)
    desc_cocotrain_x2, dst_cocotrain_x2 = 'mscoco2017_train_remap_x2', get_coco_dicts(_args, 'train')
    desc_cocovalid_x2, dst_cocovalid_x2 = 'mscoco2017_valid_remap_x2', get_coco_dicts(_args, 'valid')
    _args.cocodir = args.cocodir_x3
    print('read from:', _args.cocodir)
    desc_cocotrain_x3, dst_cocotrain_x3 = 'mscoco2017_train_remap_x3', get_coco_dicts(_args, 'train')
    desc_cocovalid_x3, dst_cocovalid_x3 = 'mscoco2017_valid_remap_x3', get_coco_dicts(_args, 'valid')
    del _args
    # for _dst in [dst_cocotrain_x1, dst_cocovalid_x1, dst_cocotrain_x2, dst_cocovalid_x2, dst_cocotrain_x3, dst_cocovalid_x3]:
    #     im = _dst[len(_dst) // 3]
    #     f = Image.fromarray(skimage.io.imread(im['file_name'])); draw = ImageDraw.Draw(f)
    #     for ann in im['annotations']:
    #         x, y, w, h = ann['bbox']
    #         draw.line(((x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)), fill=bbox_rgbs[ann['category_id']], width=2)
    #     plt.figure(); plt.imshow(np.array(f)); plt.title(im['file_name']); plt.show()
    # exit(0)

    if args.train_on_mosaic:
        desc_cocotrain_x1x2x3, dst_cocotrain_x1x2x3 = 'mscoco2017_train_remap_x1x2x3', dst_cocotrain_x1 + dst_cocotrain_x2 + dst_cocotrain_x3
        del dst_cocotrain_x1, dst_cocotrain_x2, dst_cocotrain_x3, desc_cocotrain_x1, desc_cocotrain_x2, desc_cocotrain_x3
        prefix = 'mscoco2017_remap_x1x2x3_fpnskip_%s' % args.model
    else:
        desc_cocotrain_x1x2x3, dst_cocotrain_x1x2x3 = 'mscoco2017_train_remap_x1', dst_cocotrain_x1
        del dst_cocotrain_x1, dst_cocotrain_x2, dst_cocotrain_x3, desc_cocotrain_x1, desc_cocotrain_x2, desc_cocotrain_x3
        prefix = 'mscoco2017_remap_x1_fpnskip_%s' % args.model
    for i in range(0, len(dst_cocotrain_x1x2x3)):
        dst_cocotrain_x1x2x3[i]['image_id'] = i + 1
    print('merged training images:', len(dst_cocotrain_x1x2x3))
    DatasetCatalog.register(desc_cocotrain_x1x2x3, lambda: dst_cocotrain_x1x2x3)
    MetadataCatalog.get(desc_cocotrain_x1x2x3).thing_classes = thing_classes

    DatasetCatalog.register(desc_cocovalid_x1, lambda: dst_cocovalid_x1)
    MetadataCatalog.get(desc_cocovalid_x1).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocovalid_x2, lambda: dst_cocovalid_x2)
    MetadataCatalog.get(desc_cocovalid_x2).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocovalid_x3, lambda: dst_cocovalid_x3)
    MetadataCatalog.get(desc_cocovalid_x3).thing_classes = thing_classes

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model(args.model)
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
    cfg.DATASETS.TRAIN = (desc_cocotrain_x1x2x3,)
    cfg.DATASETS.TEST = (desc_cocovalid_x1, desc_cocovalid_x2, desc_cocovalid_x3)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = TrainerFPNSkip(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)
    trainer.model.backbone.load_lateral_output_weights()

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
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
    dst_list = {}
    for _desc in [desc_cocovalid_x1, desc_cocovalid_x2, desc_cocovalid_x3]:
        dst_list[_desc] = {'mAP': [], 'AP50': []}
    for i in iter_list:
        for _desc in [desc_cocovalid_x1, desc_cocovalid_x2, desc_cocovalid_x3]:
            dst_list[_desc]['mAP'].append(aps[i][_desc]['bbox']['AP'])
            dst_list[_desc]['AP50'].append(aps[i][_desc]['bbox']['AP50'])

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    for i in range(0, len(loss_history)):
        _losses = {'res2': 0, 'res3': 0, 'res4': 0, 'res5': 0}
        for loss_key in loss_history[i]['loss']:
            for _prefix in _losses:
                if loss_key.startswith(_prefix):
                    _losses[_prefix] += loss_history[i]['loss'][loss_key]
        loss_history[i]['loss'] = _losses
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid_x1]['AP50']) / 100, linestyle='-', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid_x1]['mAP']) / 100, linestyle='-', marker='x', color='#0000FF')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid_x2]['AP50']) / 100, linestyle='--', marker='+', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid_x2]['mAP']) / 100, linestyle='--', marker='+', color='#0000FF')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid_x3]['AP50']) / 100, linestyle='-.', marker='d', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid_x3]['mAP']) / 100, linestyle='-.', marker='d', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP', 'MSCOCO x2 Valid AP50', 'MSCOCO x2 Valid mAP', 'MSCOCO x3 Valid AP50', 'MSCOCO x3 Valid mAP'])
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
        legends.append(loss_key + ' branch total loss')
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorFPNSkip(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNFPNSkip.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, original_image, L):
        with torch.no_grad():
            if self.input_format == 'RGB':
                original_image = original_image[:, :, ::-1]
            assert original_image.dtype == np.uint8
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            predictions = self.model.inference([{'image': image, 'height': height, 'width': width}], L=L)
            return predictions[0]


def evaluate_all_videos(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}

    results_file = 'results_AP_fpnskip_i%.2f_%s_%s' % (args.input_scale, args.inference_level, args.model)
    print(results_file)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = PredictorFPNSkip(cfg)

    t_total, N_total = 0, 0
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        N_total += len(images)
        detections = []
        loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [os.path.join(inputdir, 'unmasked', im['file_name']) for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
        t0 = time.time()
        for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting %s validation frames' % video_id):
            det = copy.deepcopy(im)
            det['annotations'] = []
            instances = detector(im_arr, L=args.inference_level)['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            detections.append(det)
            # f = Image.fromarray(im_arr); draw = ImageDraw.Draw(f)
            # for ann in det['annotations']:
            #     if ann['score'] < 0.75: continue
            #     x1, y1, x2, y2 = ann['bbox']
            #     draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='#000000', width=4)
            # plt.figure(); plt.imshow(np.array(f)); plt.show()
        t_total += time.time() - t0
        print('[%d/%d finished in %.1f minutes]\n' % (video_id_list.index(video_id) + 1, len(video_id_list), t_total / 60))
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        del results[video_id]['raw']
        print(   '             %s' % '/'.join(results[video_id]['metrics']))
        for c in sorted(results[video_id]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[video_id]['results'][c])))
    with open(results_file + '.json', 'w') as fp:
        json.dump(results, fp, indent=2)
    print('processed %d images in %.2f seconds, %.3f ms/image' % (N_total, t_total, t_total * 1000 / N_total))

    videos = sorted(list(results.keys()))
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    for video_id in videos:
        AP1 = base_AP['manual_' + video_id]['results']
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

    if args.eval_skip_coco:
        return
    args.smallscale = False
    images = get_coco_dicts(args, 'valid')
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
        im['annotations'] = []
        for i in range(0, len(label)):
            im['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(im)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_mscoco2017_valid = evaluate_cocovalid(args.cocodir, detections)
    print(   '             %s' % '/'.join(results_mscoco2017_valid['metrics']))
    for c in sorted(results_mscoco2017_valid['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results_mscoco2017_valid['results'][c])))


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = PredictorFPNSkip(cfg)

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
            detector.model.inference(inputs_list[i % len(images)], L=args.inference_level)
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    # mosaic('../../../MSCOCO2017', 'valid', '../../../MSCOCO2017_x2', 2); mosaic('../../../MSCOCO2017', 'valid', '../../../MSCOCO2017_x3', 3); exit(0)
    # mosaic('../../../MSCOCO2017', 'train', '../../../MSCOCO2017_x2', 2); mosaic('../../../MSCOCO2017', 'train', '../../../MSCOCO2017_x3', 3); exit(0)
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--inference_level', type=str)
    parser.add_argument('--input_scale', type=float, default=1.0)

    # parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--cocodir_x2', type=str, help='MSCOCO2017 directory, 2x2 mosaic')
    parser.add_argument('--cocodir_x3', type=str, help='MSCOCO2017 directory, 3x3 mosaic')
    parser.add_argument('--train_on_mosaic', type=bool, default=False, help='include mosaic images in training set')
    # parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    # parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--smallscale', default=False, type=bool)
    # parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    # parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    # parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    # parser.add_argument('--fn_min_score', type=float, default=0.99, help='minimum objectiveness score of false negatives')
    # parser.add_argument('--fn_max_samples', type=int, default=-1, help='maximum number of false negatives per frame')
    # parser.add_argument('--fn_max_samples_det_p', type=float, default=0.5, help='maximum number of false negatives per frame as percentage of number of detections')
    # parser.add_argument('--fn_min_area', type=float, default=50, help='minimum area of false negative boxes')
    # parser.add_argument('--fn_max_width_p', type=float, default=0.3333, help='maximum percentage width of false negative boxes')
    # parser.add_argument('--fn_max_height_p', type=float, default=0.3333, help='maximum percentage height of false negative boxes')

    # parser.add_argument('--mixup_p', type=float, default=0.3, help='probability of applying mixup to an image')
    # parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    # parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')
    # parser.add_argument('--mixup_random_position', type=bool, default=False, help='randomly position patch')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    # parser.add_argument('--refine_visualize_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    # parser.add_argument('--eval_outputfile', default=None, type=str)
    # parser.add_argument('--hold', default=0.005, type=float)

    # parser.add_argument('--ddp_num_gpus', type=int, default=1)
    # parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    # args.anno_models = sorted(list(set(args.anno_models)))
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'base':
        train_base(args)
    elif args.opt == 'eval':
        evaluate_all_videos(args)
    elif args.opt == 'tp':
        inference_throughput(args)
    else:
        pass


'''
python finetune_fpn_skip.py --opt base --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --cocodir_x2 ../../../MSCOCO2017_x2 --cocodir_x3 ../../../MSCOCO2017_x3 --smallscale 1 --iters 300 --eval_interval 101 --image_batch_size 2 --train_on_mosaic 1

python finetune_fpn_skip.py --opt base --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --cocodir_x2 ../../../MSCOCO2017_x2 --cocodir_x3 ../../../MSCOCO2017_x3 --iters 40000 --eval_interval 2001 --image_batch_size 4 --num_workers 4 --train_on_mosaic 1
python finetune_fpn_skip.py --opt base --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --cocodir_x2 ../../../MSCOCO2017_x2 --cocodir_x3 ../../../MSCOCO2017_x3 --iters 40000 --eval_interval 2001 --image_batch_size 4 --num_workers 4

python finetune_fpn_skip.py --opt eval --model r101-fpn-3x --ckpt mscoco2017_remap_x1x2x3_fpnskip_r101-fpn-3x.pth --eval_skip_coco 1 --inference_level res5 --input_scale 1
python finetune_fpn_skip.py --opt tp --model r101-fpn-3x --ckpt mscoco2017_remap_x1x2x3_fpnskip_r101-fpn-3x.pth --id 001 --inference_level res5 --input_scale 1
'''
