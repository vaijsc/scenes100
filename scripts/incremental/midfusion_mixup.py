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
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import bbox_inside, intersect_ratios


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def construct_image_w_background(image, image_background):
    image_diff = (image.astype(np.float16) - image_background) # float16, [-255, 255]
    image_diff = ((image_diff + 255) * 0.5).astype(np.uint8)
    return image, image_background, image_diff


# wrap detectron2/detectron2/data/dataset_mapper.py:DatasetMapper
class DatasetMapperBackgroundMixup(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        if 'mixup_src_images' in dataset_dict and random.uniform(0.0, 1.0) < self.mixup_p:
            mixup_src_dict = dataset_dict['mixup_src_images'][random.randrange(0, len(dataset_dict['mixup_src_images']))]
            src_image = detectron2.data.detection_utils.read_image(mixup_src_dict['file_name'], format=self.image_format)
            assert src_image.shape == image.shape
            src_annotations = mixup_src_dict['annotations']
            random.shuffle(src_annotations)
            src_annotations = src_annotations[: max(1, int(self.mixup_r * len(src_annotations)))]
            for ann in src_annotations:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = map(int, ann['bbox'])
                x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
                image[y1 : y2, x1 : x2] = src_image[y1 : y2, x1 : x2]
            annotations_trimmed = []
            for ann in dataset_dict['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                _trim = False
                for ann2 in src_annotations:
                    if intersect_ratios(ann['bbox'], ann2['bbox'])[0] >= self.mixup_overlap_thres or bbox_inside(ann['bbox'], ann2['bbox']):
                        _trim = True
                        break
                if not _trim:
                    annotations_trimmed.append(ann)
            for ann in src_annotations:
                annotations_trimmed.append(ann)
            dataset_dict['annotations'] = annotations_trimmed

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)

        # additional channels
        image_background = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
        assert image_background.shape == image.shape
        image, _, image_diff = construct_image_w_background(image, image_background)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_diff = transforms.apply_image(image_diff)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = np.concatenate([image, image_diff], axis=2)
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    @staticmethod
    def create_from_sup(mapper, mixup_p, mixup_r, mixup_overlap_thres):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperBackgroundMixup
        mapper.mixup_p, mapper.mixup_r, mapper.mixup_overlap_thres = mixup_p, mixup_r, mixup_overlap_thres
        return mapper


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFinetuneBackground(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        # print(self.backbone.bottom_up.stem.conv1.weight.data[0, 0, 0, 0].item(), self.roi_heads.box_head.fc1.weight.data[0, 0].item(), self.roi_heads_merge.box_head.fc1.weight.data[0, 0].item())
        if not self.training: return self.inference(batched_inputs)
        images_orig, images_diff = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features_orig, features_diff = self.backbone(images_orig.tensor), self.backbone(images_diff.tensor)
        proposals_orig, proposal_losses_orig = self.proposal_generator(images_orig, features_orig, gt_instances)
        _, detector_losses_orig = self.roi_heads(images_orig, features_orig, proposals_orig, gt_instances)
        if self.vis_period > 0:
            raise Exception('visualization of multi-task training not supported')
            storage = detectron2.utils.events.get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals_orig)

        if self.fusion is None:
            features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
        else:
            features_merge = self.fusion(features_orig, features_diff)
        proposals_merge, proposal_losses_merge = self.proposal_generator_merge(images_orig, features_merge, gt_instances)
        _, detector_losses_merge = self.roi_heads_merge(images_orig, features_merge, proposals_merge, gt_instances)

        losses = {}
        losses.update({k: detector_losses_orig[k] * (1 - self.loss_alpha) + detector_losses_merge[k] * self.loss_alpha for k in detector_losses_orig})
        losses.update({k: proposal_losses_orig[k] * (1 - self.loss_alpha) + proposal_losses_merge[k] * self.loss_alpha for k in proposal_losses_orig})
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, return_both: bool = False):
        # default: only compute & return results from merged features
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        images_orig, images_diff = self.preprocess_image(batched_inputs)
        features_orig, features_diff = self.backbone(images_orig.tensor), self.backbone(images_diff.tensor)
        assert detected_instances is None, 'pre-computed instances not supported'
        if return_both:
            proposals_orig, _ = self.proposal_generator(images_orig, features_orig, None)
            results_orig, _ = self.roi_heads(images_orig, features_orig, proposals_orig, None)

        if self.fusion is None:
            features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
        else:
            features_merge = self.fusion(features_orig, features_diff)
        proposals_merge, _ = self.proposal_generator_merge(images_orig, features_merge, None)
        results_merge, _ = self.roi_heads_merge(images_orig, features_merge, proposals_merge, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
            if return_both:
                results_orig = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results_orig, batched_inputs, images_orig.image_sizes)
            results_merge = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results_merge, batched_inputs, images_orig.image_sizes)
        if return_both:
            return results_orig, results_merge
        else:
            return results_merge

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x['image'].to(self.device) for x in batched_inputs]
        images_orig = ImageList.from_tensors([(x[0:3] - self.pixel_mean) / self.pixel_std for x in images], self.backbone.size_divisibility)
        images_diff = ImageList.from_tensors([(x[3:6] - self.pixel_mean) / self.pixel_std for x in images], self.backbone.size_divisibility)
        return images_orig, images_diff

    @staticmethod
    def create_from_sup(net, fusion_type, loss_alpha):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.proposal_generator_merge, net.roi_heads_merge = copy.deepcopy(net.proposal_generator), copy.deepcopy(net.roi_heads)
        if fusion_type == 'average':
            net.fusion, net.fusion_desc = None, ''
        elif fusion_type == 'conv':
            net.fusion, net.fusion_desc = FeaturePyramidFusionConv().to(net.backbone.bottom_up.stem.conv1.weight.device), 'conv'
        elif fusion_type == 'attn':
            net.fusion, net.fusion_desc = FeaturePyramidFusionAttn().to(net.backbone.bottom_up.stem.conv1.weight.device), 'attn'
        else:
            raise Exception('unsupported fusion type: %s' % fusion_type)
        net.loss_alpha = loss_alpha
        net.__class__ = GeneralizedRCNNFinetuneBackground
        return net


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


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneBackgroundTrainer(DefaultTrainer):
    def __init__(self, cfg, fusion_type, multitask_loss_alpha):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        model = GeneralizedRCNNFinetuneBackground.create_from_sup(model, fusion_type, multitask_loss_alpha)
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
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = detectron2.data.build_detection_test_loader(cfg, dataset_name)
        assert isinstance(loader.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
        loader.dataset._map_func._obj = DatasetMapperBackgroundMixup.create_from_sup(loader.dataset._map_func._obj, 0.3, 0.5, 0.65)
        return loader


def get_midfusion_avg_trainer(model, ckpt, num_workers, output_dir, lr, im_batch, roi_batch, iters_dict, datasets_dict):
    assert os.access(ckpt, os.R_OK)
    cfg = get_cfg()
    if model == 'r50-fpn-3x':
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    elif model == 'r101-fpn-3x':
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    else:
        raise NotImplementedError
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = len(thing_classes)
    cfg.MODEL.WEIGHTS                        = ckpt
    cfg.OUTPUT_DIR                           = output_dir
    cfg.DATALOADER.NUM_WORKERS               = num_workers
    cfg.SOLVER.BASE_LR                       = lr
    cfg.SOLVER.IMS_PER_BATCH                 = im_batch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch
    cfg.SOLVER.WARMUP_ITERS                  = iters_dict['warmup']
    cfg.SOLVER.GAMMA                         = iters_dict['gamma']
    cfg.SOLVER.STEPS                         = iters_dict['steps']
    cfg.SOLVER.MAX_ITER                      = iters_dict['total']
    cfg.TEST.EVAL_PERIOD                     = iters_dict['eval_interval']
    cfg.DATASETS.TRAIN                       = datasets_dict['train']
    cfg.DATASETS.TEST                        = datasets_dict['eval']
    trainer = FinetuneBackgroundTrainer(cfg, 'average', 0.5)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackgroundMixup.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, 0.3, 0.5, 0.65)
    trainer.resume_or_load(resume=False)
    return cfg, trainer


def get_midfusion_avg_detector(model, ckpt):
    class PredictorBackground(DefaultPredictor):
        def __init__(self, cfg):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = detectron2.modeling.build_model(self.cfg)
            assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
            self.model = GeneralizedRCNNFinetuneBackground.create_from_sup(self.model, 'average', None)
            self.model.eval()
            if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
            self.input_format = cfg.INPUT.FORMAT
            assert self.input_format in ['RGB', 'BGR'], self.input_format
        def __call__(self, original_image, image_diff):
            with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
                # Apply pre-processing to image.
                assert self.input_format == 'BGR'
                height, width = original_image.shape[:2]
                tf = self.aug.get_transform(original_image)
                image = torch.as_tensor(tf.apply_image(original_image).astype('float32').transpose(2, 0, 1))
                image_diff = torch.as_tensor(tf.apply_image(image_diff).astype('float32').transpose(2, 0, 1))
                inputs = {'image': torch.cat([image, image_diff], dim=0), 'height': height, 'width': width}
                predictions_orig, predictions_merge = self.model.inference([inputs], return_both=True)
                return {'orig': predictions_orig[0], 'merge': predictions_merge[0]}
    assert os.access(ckpt, os.R_OK)
    cfg = get_cfg()
    if model == 'r50-fpn-3x':
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    elif model == 'r101-fpn-3x':
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    else:
        raise NotImplementedError
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.WEIGHTS = ckpt
    detector = PredictorBackground(cfg)
    return cfg, detector



if __name__ == '__main__':
    pass
