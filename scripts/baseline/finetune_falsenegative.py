#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import enum
import gzip
import math
import random
import tqdm
import glob
import psutil
import argparse
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

import sklearn.utils
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, ImageList, Instances, pairwise_iou

from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
import detectron2.modeling
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

import logging
import weakref

from finetune import refine_annotations, get_annotation_dict, finetune_simple_trainer_run_step
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from utils import IoU, bbox_inside, intersect_ratios


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_falsenegative')


class GeneralizedRCNNFalseNegative(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.proposal_generator, detectron2.modeling.proposal_generator.rpn.RPN), 'rpn is not detectron2.modeling.proposal_generator.rpn.RPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        net.__class__ = GeneralizedRCNNFalseNegative
        net.proposal_generator.__class__ = RPNFalseNegative
        net.proposal_generator.fn_discard_stats = {'0': [0], '1': [0], '-1': [0], 'fn-discard': [0]}
        net.roi_heads.__class__ = StandardROIHeadsFalseNegative
        net.roi_heads.fn_discard_stats = {'0': [0], '1': [0], 'fn-discard': [0]}
        return net


class RPNFalseNegative(detectron2.modeling.proposal_generator.rpn.RPN):
    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        anchors = Boxes.cat(anchors)
        gt_boxes    = [x.gt_boxes[x.fn_mask == 0] for x in gt_instances]
        gt_boxes_fn = [x.gt_boxes[x.fn_mask == 1] for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i, gt_boxes_fn_i in zip(image_sizes, gt_boxes, gt_boxes_fn):
            match_quality_matrix = detectron2.utils.memory.retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = detectron2.utils.memory.retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            # discard anchors that are not matched with GT but matched with FN
            if gt_boxes_fn_i.tensor.size(0) > 0:
                match_quality_matrix_fn = detectron2.utils.memory.retry_if_cuda_oom(pairwise_iou)(gt_boxes_fn_i, anchors)
                _, gt_labels_fn_i = detectron2.utils.memory.retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix_fn)
                gt_labels_fn_i = gt_labels_fn_i.to(device=gt_boxes_fn_i.device)
                del match_quality_matrix_fn
                fn_discard_mask = torch.logical_and(gt_labels_i == 0, gt_labels_fn_i == 1)
                # print('total %d, + %d / - %d / ignore %d, fn-discard %d' % (fn_discard_mask.size(0), (gt_labels_i == 1).sum(), (gt_labels_i == 0).sum(), (gt_labels_i == -1).sum(),  fn_discard_mask.sum()))
                gt_labels_i[fn_discard_mask] = -1
                self.fn_discard_stats['0'].append(float((gt_labels_i == 0).sum().item()))
                self.fn_discard_stats['1'].append(float((gt_labels_i == 1).sum().item()))
                self.fn_discard_stats['-1'].append(float((gt_labels_i == -1).sum().item()))
                self.fn_discard_stats['fn-discard'].append(float(fn_discard_mask.sum().item()))

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)
            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes


class StandardROIHeadsFalseNegative(detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        targets_pl = [t[t.fn_mask == 0] for t in targets]
        targets_fn = [t[t.fn_mask == 1] for t in targets]
        targets = targets_pl
        if self.proposal_append_gt:
            proposals = detectron2.modeling.proposal_generator.proposal_utils.add_ground_truth_to_proposals(targets, proposals)
        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image, targets_fn_per_image in zip(proposals, targets, targets_fn):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            # discard anchors that are not matched with GT but matched with FN
            if len(targets_fn_per_image) > 0:
                match_quality_matrix_fn = pairwise_iou(targets_fn_per_image.gt_boxes, proposals_per_image.proposal_boxes)
                matched_idxs_fn, matched_labels_fn = self.proposal_matcher(match_quality_matrix_fn)
                fn_discard_mask = torch.logical_and(matched_labels == 0, matched_labels_fn == 1)
                # print('total %d, + %d / - %d, fn-discard %d' % (fn_discard_mask.size(0), (matched_labels == 1).sum(), (matched_labels == 0).sum(),  fn_discard_mask.sum()))
                proposals_per_image, matched_idxs, matched_labels = proposals_per_image[~fn_discard_mask], matched_idxs[~fn_discard_mask], matched_labels[~fn_discard_mask]
                self.fn_discard_stats['0'].append(float((matched_labels == 0).sum().item()))
                self.fn_discard_stats['1'].append(float((matched_labels == 1).sum().item()))
                self.fn_discard_stats['fn-discard'].append(float(fn_discard_mask.sum().item()))

            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith('gt_') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = detectron2.utils.events.get_event_storage()
        storage.put_scalar('roi_head/num_fg_samples', np.mean(num_fg_samples))
        storage.put_scalar('roi_head/num_bg_samples', np.mean(num_bg_samples))
        return proposals_with_gt


class GeneralizedRCNNMidFusionAvgFalseNegative(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
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
        features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
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
        features_merge = {k: (features_orig[k] + features_diff[k]) / 2.0 for k in features_orig} # merge features
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
    def create_from_sup(net, loss_alpha):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        net.__class__ = GeneralizedRCNNMidFusionAvgFalseNegative
        net.proposal_generator.__class__ = RPNFalseNegative
        net.proposal_generator.fn_discard_stats = {'0': [0], '1': [0], '-1': [0], 'fn-discard': [0]}
        net.roi_heads.__class__ = StandardROIHeadsFalseNegative
        net.roi_heads.fn_discard_stats = {'0': [0], '1': [0], 'fn-discard': [0]}
        net.proposal_generator_merge, net.roi_heads_merge = copy.deepcopy(net.proposal_generator), copy.deepcopy(net.roi_heads)
        net.loss_alpha = loss_alpha
        return net


class DatasetMapperMixupFalseNegative(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
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
                if not self.mixup_random_position:
                    x1, y1, x2, y2 = map(int, ann['bbox'])
                    x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
                    image[y1 : y2, x1 : x2] = src_image[y1 : y2, x1 : x2]
                else:
                    x1, y1, x2, y2 = map(int, ann['bbox'])
                    x1, y1, x2, y2 = map(lambda x: 1 if x < 1 else x, [x1, y1, x2, y2])
                    x2, y2 = min(image.shape[1], max(x2, x1 + 1)), min(image.shape[0], max(y2, y1 + 1))
                    x_shift, y_shift = np.random.randint(-1 * x1, image.shape[1] - x2), np.random.randint(-1 * y1, image.shape[0] - y2)
                    image[y1 + y_shift : y2 + y_shift, x1 + x_shift : x2 + x_shift] = src_image[y1 : y2, x1 : x2]
                    ann['bbox'] = [x1 + x_shift, y1 + y_shift, x2 + x_shift, y2 + y_shift]
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
        fn_mask = [1 if ann['src'] == 'fn' else 0 for ann in dataset_dict['annotations']]

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        if 'sem_seg_file_name' in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop('sem_seg_file_name'), 'L').squeeze(2)
        else:
            sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict['sem_seg'] = torch.as_tensor(sem_seg_gt.astype('long'))
        if self.proposal_topk is not None:
            detectron2.data.detection_utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if len(fn_mask) == len(dataset_dict['instances']):
            dataset_dict['instances'].fn_mask = torch.tensor(fn_mask, dtype=torch.int8)
        else:
            print('dataset_dict & instances mis-match:', dataset_dict['file_name'], len(fn_mask), len(dataset_dict['instances']))
            dataset_dict['instances'].fn_mask = torch.tensor([0 for _ in range(0, len(dataset_dict['instances']))], dtype=torch.int8)
        return dataset_dict
    @staticmethod
    def create_from_sup(mapper, mixup_p, mixup_r, mixup_overlap_thres, mixup_random_position):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperMixupFalseNegative
        mapper.mixup_p, mapper.mixup_r, mapper.mixup_overlap_thres, mapper.mixup_random_position = mixup_p, mixup_r, mixup_overlap_thres, mixup_random_position
        return mapper


from finetune_wdiff_earlyfusion import construct_image_w_background
class DatasetMapperBackgroundMixupFalseNegative(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
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
        fn_mask = [1 if ann['src'] == 'fn' else 0 for ann in dataset_dict['annotations']]
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)

        # additional channels
        image_background = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
        assert image_background.shape == image.shape
        image, _, image_diff = construct_image_w_background(image, image_background)

        sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_diff = transforms.apply_image(image_diff)
        image_shape = image.shape[:2]  # h, w
        image = np.concatenate([image, image_diff], axis=2)
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if len(fn_mask) == len(dataset_dict['instances']):
            dataset_dict['instances'].fn_mask = torch.tensor(fn_mask, dtype=torch.int8)
        else:
            print('dataset_dict & instances mis-match:', dataset_dict['file_name'], len(fn_mask), len(dataset_dict['instances']))
            dataset_dict['instances'].fn_mask = torch.tensor([0 for _ in range(0, len(dataset_dict['instances']))], dtype=torch.int8)
        return dataset_dict
    @staticmethod
    def create_from_sup(mapper, mixup_p, mixup_r, mixup_overlap_thres):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperBackgroundMixupFalseNegative
        mapper.mixup_p, mapper.mixup_r, mapper.mixup_overlap_thres = mixup_p, mixup_r, mixup_overlap_thres
        return mapper


class FinetuneTrainer(DefaultTrainer):
    def __init__(self, cfg, args):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        assert args.opt in ['crossteach', 'mixup']
        model = GeneralizedRCNNFalseNegative.create_from_sup(model)
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


from finetune_wdiff_earlyfusion import DatasetMapperBackground
class FinetuneTrainerMidFusion(DefaultTrainer):
    def __init__(self, cfg, args):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        assert args.opt in ['midfusion', 'midfusionmixup']
        model = GeneralizedRCNNMidFusionAvgFalseNegative.create_from_sup(model, args.multitask_loss_alpha)
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
        return COCOEvaluator(dataset_name, output_dir=finetune_output)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = detectron2.data.build_detection_test_loader(cfg, dataset_name)
        assert isinstance(loader.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
        loader.dataset._map_func._obj = DatasetMapperBackground.create_from_sup(loader.dataset._map_func._obj)
        return loader


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(args, 'valid')
    for im in dst_cocovalid:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'val2017', os.path.basename(im['file_name'])))
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]

    background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
    background_frame_idx = list(map(lambda x: os.path.basename(x), background_files))
    background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), background_frame_idx)))
    desc_manual_valid, dst_manual_valid = '%s_manual_wdiff_earlyfusion' % args.id, get_annotation_dict(args)
    for im in dst_manual_valid:
        im['file_name_background'] = background_files[-1] # choice of background images here does not affect training

    # do not mix false negatives with refined pseudo labels, process false negatives separately
    fn_max_samples = args.fn_max_samples
    args.fn_max_samples = -1
    desc_pseudo_anno = 'refine_' + '_'.join(args.anno_models)
    dst_pseudo_anno = refine_annotations(args)[0]
    for im in dst_pseudo_anno:
        i = os.path.basename(im['file_name'])
        i = int(i[:i.find('.')])
        im['file_name_background'] = background_files[np.absolute(background_frame_idx - i).argmin()]

    # do not use false negatives as mixup source
    if args.opt.find('mixup') >= 0:
        dst_pseudo_anno_copy = copy.deepcopy(dst_pseudo_anno)
        for im in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources'):
            im['mixup_src_images'] = [dst_pseudo_anno_copy[random.randrange(0, len(dst_pseudo_anno_copy))]]
        del dst_pseudo_anno_copy

    # include false negatives
    args.fn_max_samples = fn_max_samples
    fn_file = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label', '%s_false_negative_mining_objthres0.9900.json.gz' % args.id))
    print('%s [%.2fMB]' % (fn_file, os.path.getsize(fn_file) / (1024 ** 2)))
    with gzip.open(fn_file, 'rt') as fp:
        fn_dets = json.loads(fp.read())['dets']
    for m in args.anno_models:
        assert len(fn_dets[m]) == len(dst_pseudo_anno)
    for i in range(0, len(fn_dets[args.anno_models[0]])):
        fn_annotations = []
        for m in args.anno_models:
            for j in range(0, len(fn_dets[m][i]['label'])):
                if fn_dets[m][i]['obj_score'][j] < args.fn_min_score: continue
                x1, y1, x2, y2 = fn_dets[m][i]['bbox'][j]
                if x2 - x1 > args.fn_max_width_p * dst_pseudo_anno[i]['width']: continue
                if y2 - y1 > args.fn_max_height_p * dst_pseudo_anno[i]['height']: continue
                if (x2 - x1) * (y2 - y1) < args.fn_min_area: continue
                fn_annotations.append({'bbox': fn_dets[m][i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': fn_dets[m][i]['label'][j], 'src': 'fn', 'obj_score': fn_dets[m][i]['obj_score'][j]})
        if len(fn_annotations) > min(args.fn_max_samples, args.fn_max_samples_det_p * len(dst_pseudo_anno[i]['annotations'])):
            random.shuffle(fn_annotations)
            fn_annotations = fn_annotations[:args.fn_max_samples]
        dst_pseudo_anno[i]['annotations'] = dst_pseudo_anno[i]['annotations'] + fn_annotations
        dst_pseudo_anno[i]['fn_count'] = len(fn_annotations)
    print('finish reading from hard negative mining results')

    if args.train_on_coco:
        random.seed(42)
        dst_cocotrain = get_coco_dicts(args, 'train')
        for im in dst_cocotrain:
            for ann in im['annotations']:
                ann['src'] = 'gt'
        for im in dst_cocotrain:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'train2017', os.path.basename(im['file_name'])))
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
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid, desc_cocovalid)
    # print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    if args.opt == 'crossteach':
        trainer = FinetuneTrainer(cfg, args)
        trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
        trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperMixupFalseNegative.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, None, None, None, False) # if mixup is performed it will fail to type-incompatible
        trainer.resume_or_load(resume=args.resume)
        prefix = 'adapt%s_%s_anno_%s_FN_discard' % (args.id, args.model, desc_pseudo_anno)
    elif args.opt == 'mixup':
        trainer = FinetuneTrainer(cfg, args)
        trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
        trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperMixupFalseNegative.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, args.mixup_p, args.mixup_r, args.mixup_overlap_thres, False)
        trainer.resume_or_load(resume=args.resume)
        prefix = 'adapt%s_%s_anno_%s_FN_discard_mixup' % (args.id, args.model, desc_pseudo_anno)
    elif args.opt == 'midfusion':
        trainer = FinetuneTrainerMidFusion(cfg, args)
        trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
        trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackgroundMixupFalseNegative.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, None, None, None) # if mixup is performed it will fail to type-incompatible
        trainer.resume_or_load(resume=args.resume)
        prefix = 'adapt%s_%s_anno_%s_FN_discard_midfusion' % (args.id, args.model, desc_pseudo_anno)
    elif args.opt == 'midfusionmixup':
        trainer = FinetuneTrainerMidFusion(cfg, args)
        trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
        trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackgroundMixupFalseNegative.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, args.mixup_p, args.mixup_r, args.mixup_overlap_thres)
        trainer.resume_or_load(resume=args.resume)
        prefix = 'adapt%s_%s_anno_%s_FN_discard_midfusionmixup' % (args.id, args.model, desc_pseudo_anno)
    else:
        raise NotImplementedError

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'rpn_fn_discard_stats': m.proposal_generator.fn_discard_stats, 'roi_fn_discard_stats': m.roi_heads.fn_discard_stats}, fp)
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    aps, lr_history, loss_history, rpn_fn_discard_stats, roi_fn_discard_stats = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history, m.proposal_generator.fn_discard_stats, m.roi_heads.fn_discard_stats
    for d in [rpn_fn_discard_stats, roi_fn_discard_stats]:
        for k in d:
            d[k] = int(np.array(d[k]).mean())
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
    plt.title('losses RPN + %d/- %d/drop %d ROI + %d/- %d/drop %d' % (rpn_fn_discard_stats['1'], rpn_fn_discard_stats['0'], rpn_fn_discard_stats['fn-discard'], roi_fn_discard_stats['1'], roi_fn_discard_stats['0'], roi_fn_discard_stats['fn-discard']))
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))


def stats():
    rpn_fn_discard_stats, roi_fn_discard_stats, apg_improve = {}, {}, {}
    # with open('F:\\intersections_results\\fn_discard\\crossteach_fn_discard\\results_AP.json', 'r') as fp:
    with open('F:\\intersections_results\\fn_discard\\midfusion_fn_discard\\results_AP_merge_dynamic.json', 'r') as fp:
        results_AP_fn = json.load(fp)
    # with open('F:\\intersections_results\\baseline_crossteach_r101\\results_AP.json', 'r') as fp:
    with open('F:\\intersections_results\\fusion_coco_mask_inpaint\\object_diff_midfusion_r101\\results_AP_merge_dynamic.json', 'r') as fp:
        results_AP_crossteach = json.load(fp)
    for v in results_AP_fn:
        apg_improve[v] = results_AP_fn[v]['results']['weighted'][0] - results_AP_crossteach[v]['results']['weighted'][0]
    # for f in tqdm.tqdm(glob.glob('F:\\intersections_results\\fn_discard\\crossteach_fn_discard\\adapt*.json'), ascii=True):
    for f in tqdm.tqdm(glob.glob('F:\\intersections_results\\fn_discard\\midfusion_fn_discard\\adapt*.json'), ascii=True):
        with open(f, 'r') as fp:
            data = json.load(fp)
        v = os.path.basename(f)[5 : 8]
        rpn_fn_discard_stats[v] = sum(data['rpn_fn_discard_stats']['fn-discard']) / sum(data['rpn_fn_discard_stats']['0'])
        roi_fn_discard_stats[v] = sum(data['roi_fn_discard_stats']['fn-discard']) / sum(data['roi_fn_discard_stats']['0'])
    plt.figure(figsize=(8, 8))
    # xs = np.array([rpn_fn_discard_stats[v] for v in video_id_list]) * 100
    xs = np.array([roi_fn_discard_stats[v] for v in video_id_list]) * 100
    ys = np.array([apg_improve[v] for v in video_id_list])
    plt.scatter(xs, ys, marker='x', s=32, c='blue')
    plt.xlabel('false negative discard percentage $\\mu=%.3f$ $\\sigma=%.3f$' % (xs.mean(), xs.std()))
    plt.ylabel('APG difference')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # stats(); exit()
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option', choices=['crossteach', 'mixup', 'midfusion', 'midfusionmixup'])
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')

    parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--smallscale', type=bool, default=False)

    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    parser.add_argument('--fn_min_score', type=float, default=0.99, help='minimum objectiveness score of false negatives')
    parser.add_argument('--fn_max_samples', type=int, default=10000, help='maximum number of false negatives per frame')
    parser.add_argument('--fn_max_samples_det_p', type=float, default=100.0, help='maximum number of false negatives per frame as percentage of number of detections')
    parser.add_argument('--fn_min_area', type=float, default=50, help='minimum area of false negative boxes')
    parser.add_argument('--fn_max_width_p', type=float, default=0.3333, help='maximum percentage width of false negative boxes')
    parser.add_argument('--fn_max_height_p', type=float, default=0.3333, help='maximum percentage height of false negative boxes')

    parser.add_argument('--mixup_p', type=float, default=0.3, help='probability of applying mixup to an image')
    parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')

    parser.add_argument('--multitask_loss_alpha', type=float, default=0.5, help='relative weight of multitasking losses')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    args.anno_models = sorted(list(set(args.anno_models)))
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)
    assert args.fn_max_samples > 0

    adapt(args)


'''
python finetune_falsenegative.py --opt crossteach --id 023 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --anno_models r50-fpn-3x r101-fpn-3x --train_on_coco 1 --cocodir ../../../MSCOCO2017 --smallscale 1 --iters 200 --eval_interval 101 --image_batch_size 2 --num_workers 1
python finetune_falsenegative.py --opt mixup --id 023 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --anno_models r50-fpn-3x r101-fpn-3x --train_on_coco 1 --cocodir ../../../MSCOCO2017 --smallscale 1 --iters 200 --eval_interval 101 --image_batch_size 2 --num_workers 1
python finetune_falsenegative.py --opt midfusion --id 023 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --anno_models r50-fpn-3x r101-fpn-3x --train_on_coco 1 --cocodir ../../../MSCOCO2017 --smallscale 1 --iters 200 --eval_interval 101 --image_batch_size 2 --num_workers 1
python finetune_falsenegative.py --opt midfusionmixup --id 023 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --anno_models r50-fpn-3x r101-fpn-3x --train_on_coco 1 --cocodir ../../../MSCOCO2017 --smallscale 1 --iters 200 --eval_interval 101 --image_batch_size 2 --num_workers 1
'''
