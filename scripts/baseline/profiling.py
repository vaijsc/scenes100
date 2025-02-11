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
from torch.profiler import profile, record_function, ProfilerActivity

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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_multiscales')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNProfiling(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        with record_function('RCNN:preprocess_image'):
            images = self.preprocess_image(batched_inputs)
        with record_function('RCNN:backbone'):
            features = self.backbone(images.tensor)
        with record_function('RCNN:proposal_generator'):
            proposals, _ = self.proposal_generator(images, features, None)
        with record_function('RCNN:roi_heads'):
            results, _ = self.roi_heads(images, features, proposals, None)

        assert do_postprocess
        assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        with record_function('RCNN:_postprocess'):
            results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results


# wrap detectron2/modeling/proposal_generator/rpn.py:RPN
class RPNProfiling(detectron2.modeling.proposal_generator.rpn.RPN):
    def forward(self, images: ImageList, features: Dict[str, torch.Tensor], gt_instances: Optional[List[Instances]] = None):
        features = [features[f] for f in self.in_features]
        with record_function('RPN:anchor_generator'):
            anchors = self.anchor_generator(features)

        with record_function('RPN:rpn_head'):
            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        with record_function('RPN:permute'):
            # Transpose the Hi*Wi*A dimension to the middle:
            pred_objectness_logits = [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in pred_objectness_logits
            ]
            pred_anchor_deltas = [
                # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
                x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in pred_anchor_deltas
            ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        # https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/proposal_generator/proposal_utils.py#L22
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    def predict_proposals(self, anchors: List[torch.Tensor], pred_objectness_logits: List[torch.Tensor], pred_anchor_deltas: List[torch.Tensor], image_sizes: List[Tuple[int, int]]):
        with torch.no_grad():
            with record_function('RPN:_decode_proposals'):
                pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            proposals = find_top_rpn_proposals_profiling(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )
            return proposals


def find_top_rpn_proposals_profiling(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    num_images = len(image_sizes)
    device = proposals[0].device

    with record_function('RPN:perlevel_topk'):
        # 1. Select top-k anchor for every level and every image
        topk_scores = []  # #lvl Tensor, each of shape N x topk
        topk_proposals = []
        level_ids = []  # #lvl Tensor, each of shape (topk,)
        batch_idx = torch.arange(num_images, device=device)
        for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
            Hi_Wi_A = logits_i.shape[1]
            if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
                num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
            else:
                num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

            # sort is faster than topk: https://github.com/pytorch/pytorch/issues/22812
            # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
            logits_i, idx = logits_i.sort(descending=True, dim=1)
            topk_scores_i = logits_i.narrow(1, 0, num_proposals_i)
            topk_idx = idx.narrow(1, 0, num_proposals_i)

            # each is N x topk
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    with record_function('RPN:concat'):
        # 2. Concat all levels together
        topk_scores = detectron2.layers.cat(topk_scores, dim=1)
        topk_proposals = detectron2.layers.cat(topk_proposals, dim=1)
        level_ids = detectron2.layers.cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        with record_function('RPN:filter_nan'):
            valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
            if not valid_mask.all():
                if training:
                    raise FloatingPointError(
                        "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                    )
                boxes = boxes[valid_mask]
                scores_per_img = scores_per_img[valid_mask]
                lvl = lvl[valid_mask]
            boxes.clip(image_size)

        with record_function('RPN:filter_empty'):
            # filter empty boxes
            keep = boxes.nonempty(threshold=min_box_size)
            if detectron2.modeling.proposal_generator.proposal_utils._is_tracing() or keep.sum().item() != len(boxes):
                boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        with record_function('RPN:NMS'):
            keep = detectron2.layers.batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
            keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


class FPNProfiling(detectron2.modeling.backbone.FPN):
    def forward(self, x):
        with record_function('FPN:bottom_up'):
            bottom_up_features = self.bottom_up(x)
        results = []
        with record_function('FPN:top_down'):
            prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
            results.append(self.output_convs[0](prev_features))

            # Reverse feature maps into top-down order (from low to high resolution)
            for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
                # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
                # Therefore we loop over all modules but skip the first one
                if idx > 0:
                    features = self.in_features[-idx - 1]
                    features = bottom_up_features[features]
                    top_down_features = torch.nn.functional.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                    lateral_features = lateral_conv(features)
                    prev_features = lateral_features + top_down_features
                    if self._fuse_type == "avg":
                        prev_features /= 2
                    results.insert(0, output_conv(prev_features))

            if self.top_block is not None:
                if self.top_block.in_feature in bottom_up_features:
                    top_block_in_feature = bottom_up_features[self.top_block.in_feature]
                else:
                    top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
                results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


def profile_rcnn(args):
    class Predictor(DefaultPredictor):
        def __init__(self, cfg, scale):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = detectron2.modeling.build_model(self.cfg)
            assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
            self.model.__class__ = GeneralizedRCNNProfiling
            self.model.eval()
            if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = detectron2.data.transforms.ResizeShortestEdge([int(scale * cfg.INPUT.MIN_SIZE_TEST), int(scale * cfg.INPUT.MIN_SIZE_TEST)], int(scale * cfg.INPUT.MAX_SIZE_TEST))
            self.input_format = cfg.INPUT.FORMAT
            assert self.input_format in ['RGB', 'BGR'], self.input_format

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    print('- cfg.INPUT')
    print(cfg.INPUT)
    print('- cfg.MODEL.RPN')
    print(cfg.MODEL.RPN)
    print('- cfg.MODEL.ANCHOR_GENERATOR')
    print(cfg.MODEL.ANCHOR_GENERATOR)
    detector = Predictor(cfg, args.scale)
    inputs_list = preload_images(detector.aug)

    N1, N2 = 10, 190
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as prof:
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
                detector.model.inference(inputs_list[i % len(inputs_list)])
    prof_avg = prof.key_averages()
    print(prof_avg)
    prof_avg_rcnn = {}
    for p in prof_avg:
        if not p.key.startswith('RCNN:'):
            continue
        prof_avg_rcnn[p.key[5:]] = {'cpu': p.cpu_time_total, 'cuda': p.cuda_time_total, 'count': p.count}
    print(prof_avg_rcnn)


def profile_rpn(args):
    class Predictor(DefaultPredictor):
        def __init__(self, cfg, scale):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = detectron2.modeling.build_model(self.cfg)
            assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
            self.model.proposal_generator.__class__ = RPNProfiling
            self.model.eval()
            if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = detectron2.data.transforms.ResizeShortestEdge([int(scale * cfg.INPUT.MIN_SIZE_TEST), int(scale * cfg.INPUT.MIN_SIZE_TEST)], int(scale * cfg.INPUT.MAX_SIZE_TEST))
            self.input_format = cfg.INPUT.FORMAT
            assert self.input_format in ['RGB', 'BGR'], self.input_format

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    print('- cfg.INPUT')
    print(cfg.INPUT)
    print('- cfg.MODEL.RPN')
    print(cfg.MODEL.RPN)
    print('- cfg.MODEL.ANCHOR_GENERATOR')
    print(cfg.MODEL.ANCHOR_GENERATOR)
    detector = Predictor(cfg, args.scale)
    inputs_list = preload_images(detector.aug)

    N1, N2 = 10, 190
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as prof:
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
                detector.model.inference(inputs_list[i % len(inputs_list)])
    prof_avg = prof.key_averages()
    print(prof_avg)
    prof_avg_rpn = {}
    for p in prof_avg:
        if not p.key.startswith('RPN:'):
            continue
        prof_avg_rpn[p.key[4:]] = {'cpu': p.cpu_time_total, 'cuda': p.cuda_time_total, 'count': p.count}
    print(prof_avg_rpn)


def profile_fpn(args):
    class Predictor(DefaultPredictor):
        def __init__(self, cfg, scale):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = detectron2.modeling.build_model(self.cfg)
            assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
            self.model.backbone.__class__ = FPNProfiling
            self.model.eval()
            if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.aug = detectron2.data.transforms.ResizeShortestEdge([int(scale * cfg.INPUT.MIN_SIZE_TEST), int(scale * cfg.INPUT.MIN_SIZE_TEST)], int(scale * cfg.INPUT.MAX_SIZE_TEST))
            self.input_format = cfg.INPUT.FORMAT
            assert self.input_format in ['RGB', 'BGR'], self.input_format

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    print('- cfg.INPUT')
    print(cfg.INPUT)
    print('- cfg.MODEL.RPN')
    print(cfg.MODEL.RPN)
    print('- cfg.MODEL.ANCHOR_GENERATOR')
    print(cfg.MODEL.ANCHOR_GENERATOR)
    detector = Predictor(cfg, args.scale)
    inputs_list = preload_images(detector.aug)

    N1, N2 = 10, 190
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as prof:
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
                detector.model.inference(inputs_list[i % len(inputs_list)])
    prof_avg = prof.key_averages()
    print(prof_avg)
    prof_avg_fpn = {}
    for p in prof_avg:
        if not p.key.startswith('FPN:'):
            continue
        prof_avg_fpn[p.key[4:]] = {'cpu': p.cpu_time_total, 'cuda': p.cuda_time_total, 'count': p.count}
    print(prof_avg_fpn)


def preload_images(aug):
    id_list_sample = np.array(video_id_list).reshape(25, 4)[:, 0].tolist()
    print('use frames from videos:', id_list_sample)
    images = []
    for video_id in id_list_sample:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', '001')
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            _im = json.load(fp)[1]
            _im['file_name'] = os.path.join(inputdir, 'unmasked', _im['file_name'])
            images.append(_im)
    print('pre-load %d images' % len(images))
    inputs_list = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(im['file_name'], format='BGR')
        im_tensor = aug.get_transform(im_arr).apply_image(im_arr)
        im_tensor = torch.as_tensor(im_tensor.astype('float32').transpose(2, 0, 1))
        inputs_list.append([{'image': im_tensor, 'height': im['height'], 'width': im['width']}])
    return inputs_list


def show_results():
    scales = [1, 1.2, 1.5, 1.7, 2, 2.25]
    linestyles = ['ko-', 'b^-', 'r+-', 'cx-', 'm*-']
    # stats = [
    #     {'preprocess_image': {'cpu': 510899, 'cuda': 508389, 'count': 200}, 'backbone': {'cpu': 13520925, 'cuda': 13833881, 'count': 200}, 'proposal_generator': {'cpu': 9253901, 'cuda': 8942192, 'count': 200}, 'roi_heads': {'cpu': 3610302, 'cuda': 3608139, 'count': 200}, '_postprocess': {'cpu': 636209, 'cuda': 634086, 'count': 200}},
    #     {'preprocess_image': {'cpu': 611802, 'cuda': 609278, 'count': 200}, 'backbone': {'cpu': 13658720, 'cuda': 14191678, 'count': 200}, 'proposal_generator': {'cpu': 9563742, 'cuda': 9031479, 'count': 200}, 'roi_heads': {'cpu': 3657297, 'cuda': 3655485, 'count': 200}, '_postprocess': {'cpu': 635238, 'cuda': 633110, 'count': 200}},
    #     {'preprocess_image': {'cpu': 802905, 'cuda': 801423, 'count': 200}, 'backbone': {'cpu': 13117213, 'cuda': 14063646, 'count': 200}, 'proposal_generator': {'cpu': 9802224, 'cuda': 8857290, 'count': 200}, 'roi_heads': {'cpu': 3674523, 'cuda': 3672253, 'count': 200}, '_postprocess': {'cpu': 640435, 'cuda': 638963, 'count': 200}},
    #     {'preprocess_image': {'cpu': 919810, 'cuda': 918418, 'count': 200}, 'backbone': {'cpu': 12856225, 'cuda': 14686537, 'count': 200}, 'proposal_generator': {'cpu': 11064209, 'cuda': 9236666, 'count': 200}, 'roi_heads': {'cpu': 3651105, 'cuda': 3648983, 'count': 200}, '_postprocess': {'cpu': 626413, 'cuda': 624281, 'count': 200}},
    #     {'preprocess_image': {'cpu': 1206962, 'cuda': 1234976, 'count': 200}, 'backbone': {'cpu': 15532253, 'cuda': 18534760, 'count': 200}, 'proposal_generator': {'cpu': 12975408, 'cuda': 9948517, 'count': 200}, 'roi_heads': {'cpu': 3582417, 'cuda': 3580327, 'count': 200}, '_postprocess': {'cpu': 652933, 'cuda': 650847, 'count': 200}},
    #     {'preprocess_image': {'cpu': 1448398, 'cuda': 1494630, 'count': 200}, 'backbone': {'cpu': 18969366, 'cuda': 23149930, 'count': 200}, 'proposal_generator': {'cpu': 14586538, 'cuda': 10361240, 'count': 200}, 'roi_heads': {'cpu': 3725897, 'cuda': 3723690, 'count': 200}, '_postprocess': {'cpu': 637983, 'cuda': 635881, 'count': 200}}
    # ]
    stats = [
        {'preprocess_image': {'cpu': 1325653, 'cuda': 1183893, 'count': 200}, 'backbone': {'cpu': 6891552, 'cuda': 8810373, 'count': 200}, 'proposal_generator': {'cpu': 5912516, 'cuda': 1501099, 'count': 200}, 'roi_heads': {'cpu': 1690191, 'cuda': 933039, 'count': 200}, '_postprocess': {'cpu': 253622, 'cuda': 11732, 'count': 200}},
        {'preprocess_image': {'cpu': 1329049, 'cuda': 1071573, 'count': 200}, 'backbone': {'cpu': 6899647, 'cuda': 12843562, 'count': 200}, 'proposal_generator': {'cpu': 10599392, 'cuda': 2131237, 'count': 200}, 'roi_heads': {'cpu': 1664736, 'cuda': 949973, 'count': 200}, '_postprocess': {'cpu': 246434, 'cuda': 12727, 'count': 200}},
        {'preprocess_image': {'cpu': 1593436, 'cuda': 1500580, 'count': 200}, 'backbone': {'cpu': 6663646, 'cuda': 19861609, 'count': 200}, 'proposal_generator': {'cpu': 18620783, 'cuda': 3177288, 'count': 200}, 'roi_heads': {'cpu': 1732538, 'cuda': 1031302, 'count': 200}, '_postprocess': {'cpu': 245079, 'cuda': 13311, 'count': 200}},
        {'preprocess_image': {'cpu': 1574504, 'cuda': 1507257, 'count': 200}, 'backbone': {'cpu': 6766465, 'cuda': 25431165, 'count': 200}, 'proposal_generator': {'cpu': 25948733, 'cuda': 5034836, 'count': 200}, 'roi_heads': {'cpu': 1717168, 'cuda': 1034981, 'count': 200}, '_postprocess': {'cpu': 245530, 'cuda': 13408, 'count': 200}},
        {'preprocess_image': {'cpu': 1758863, 'cuda': 1661198, 'count': 200}, 'backbone': {'cpu': 6580109, 'cuda': 34290614, 'count': 200}, 'proposal_generator': {'cpu': 35616715, 'cuda': 5534163, 'count': 200}, 'roi_heads': {'cpu': 1700786, 'cuda': 1079225, 'count': 200}, '_postprocess': {'cpu': 220021, 'cuda': 13586, 'count': 200}},
        {'preprocess_image': {'cpu': 2202150, 'cuda': 2133969, 'count': 200}, 'backbone': {'cpu': 6550629, 'cuda': 42930670, 'count': 200}, 'proposal_generator': {'cpu': 45608388, 'cuda': 6986245, 'count': 200}, 'roi_heads': {'cpu': 1703062, 'cuda': 1075045, 'count': 200}, '_postprocess': {'cpu': 220625, 'cuda': 13410, 'count': 200}}
    ]
    plt.figure(figsize=(6, 6))
    keys = ['preprocess_image', 'backbone', 'proposal_generator', 'roi_heads']
    descs = {'preprocess_image': 'preprocessing', 'backbone': 'feature pyramid extraction', 'proposal_generator': 'propsal generation', 'roi_heads': 'classifier', '_postprocess': 'postprocessing'}
    legends = []
    for i, k in enumerate(keys):
        plt.plot(scales, [_s[k]['cpu'] / _s[k]['count'] / 1000 for _s in stats], linestyles[i] + '-')
        legends.append(descs[k] + ' [CPU]')
        plt.plot(scales, [_s[k]['cuda'] / _s[k]['count'] / 1000 for _s in stats], linestyles[i])
        legends.append(descs[k] + ' [CUDA]')
    plt.legend(legends)
    # plt.ylim(0, 120)
    plt.ylim(0, 250)
    plt.xlim(0.95, 2.3)
    plt.xlabel('input scale')
    plt.ylabel('inference time (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_results_rpn():
    scales = [1, 1.2, 1.5, 1.7, 2, 2.25]
    linestyles = ['ko-', 'b^-', 'r+-', 'cx-', 'm*-', 'yd-', 'g1-']
    # stats = [
    #     {'anchor_generator': {'cpu': 1279019, 'cuda': 968387, 'count': 200}, 'rpn_head': {'cpu': 1038302, 'cuda': 1036552, 'count': 200}, 'permute': {'cpu': 392554, 'cuda': 390844, 'count': 200}, '_decode_proposals': {'cpu': 2969040, 'cuda': 2966139, 'count': 200}, 'perlevel_topk': {'cpu': 1190019, 'cuda': 1188232, 'count': 200}, 'concat': {'cpu': 69597, 'cuda': 67643, 'count': 200}, 'filter_nan': {'cpu': 367052, 'cuda': 364874, 'count': 200}, 'filter_empty': {'cpu': 159005, 'cuda': 156966, 'count': 200}, 'NMS': {'cpu': 2020570, 'cuda': 2018554, 'count': 200}},
    #     {'anchor_generator': {'cpu': 1424292, 'cuda': 901669, 'count': 200}, 'rpn_head': {'cpu': 1116948, 'cuda': 1110019, 'count': 200}, 'permute': {'cpu': 404900, 'cuda': 401541, 'count': 200}, '_decode_proposals': {'cpu': 2950618, 'cuda': 2948308, 'count': 200}, 'perlevel_topk': {'cpu': 1191442, 'cuda': 1189649, 'count': 200}, 'concat': {'cpu': 70257, 'cuda': 68110, 'count': 200}, 'filter_nan': {'cpu': 364367, 'cuda': 362596, 'count': 200}, 'filter_empty': {'cpu': 159186, 'cuda': 157293, 'count': 200}, 'NMS': {'cpu': 1983691, 'cuda': 1981638, 'count': 200}},
    #     {'anchor_generator': {'cpu': 1114458, 'cuda': 773375, 'count': 200}, 'rpn_head': {'cpu': 1342892, 'cuda': 1284222, 'count': 200}, 'permute': {'cpu': 489369, 'cuda': 292795, 'count': 200}, '_decode_proposals': {'cpu': 3251172, 'cuda': 2931301, 'count': 200}, 'perlevel_topk': {'cpu': 1182292, 'cuda': 1180306, 'count': 200}, 'concat': {'cpu': 69591, 'cuda': 67806, 'count': 200}, 'filter_nan': {'cpu': 363126, 'cuda': 361164, 'count': 200}, 'filter_empty': {'cpu': 161564, 'cuda': 159431, 'count': 200}, 'NMS': {'cpu': 2038095, 'cuda': 2036137, 'count': 200}},
    #     {'anchor_generator': {'cpu': 1034871, 'cuda': 721934, 'count': 200}, 'rpn_head': {'cpu': 1159083, 'cuda': 1518249, 'count': 200}, 'permute': {'cpu': 1224187, 'cuda': 315555, 'count': 200}, '_decode_proposals': {'cpu': 3784138, 'cuda': 2769130, 'count': 200}, 'perlevel_topk': {'cpu': 1322907, 'cuda': 1320749, 'count': 200}, 'concat': {'cpu': 71644, 'cuda': 69791, 'count': 200}, 'filter_nan': {'cpu': 370275, 'cuda': 368137, 'count': 200}, 'filter_empty': {'cpu': 164396, 'cuda': 162304, 'count': 200}, 'NMS': {'cpu': 1975172, 'cuda': 1973638, 'count': 200}},
    #     {'anchor_generator': {'cpu': 1196584, 'cuda': 731891, 'count': 200}, 'rpn_head': {'cpu': 1004132, 'cuda': 1955503, 'count': 200}, 'permute': {'cpu': 2587327, 'cuda': 330767, 'count': 200}, '_decode_proposals': {'cpu': 4121621, 'cuda': 2894521, 'count': 200}, 'perlevel_topk': {'cpu': 1400078, 'cuda': 1398045, 'count': 200}, 'concat': {'cpu': 75311, 'cuda': 73065, 'count': 200}, 'filter_nan': {'cpu': 382685, 'cuda': 380327, 'count': 200}, 'filter_empty': {'cpu': 168511, 'cuda': 166501, 'count': 200}, 'NMS': {'cpu': 2073435, 'cuda': 2071840, 'count': 200}},
    #     {'anchor_generator': {'cpu': 1396887, 'cuda': 715846, 'count': 200}, 'rpn_head': {'cpu': 1401966, 'cuda': 2388492, 'count': 200}, 'permute': {'cpu': 3245687, 'cuda': 295929, 'count': 200}, '_decode_proposals': {'cpu': 4397623, 'cuda': 2941417, 'count': 200}, 'perlevel_topk': {'cpu': 1404411, 'cuda': 1402167, 'count': 200}, 'concat': {'cpu': 76279, 'cuda': 74017, 'count': 200}, 'filter_nan': {'cpu': 401134, 'cuda': 397045, 'count': 200}, 'filter_empty': {'cpu': 173563, 'cuda': 171344, 'count': 200}, 'NMS': {'cpu': 2127339, 'cuda': 2125988, 'count': 200}}
    # ]
    stats = [
        {'anchor_generator': {'cpu': 397648, 'cuda': 14829, 'count': 200}, 'rpn_head': {'cpu': 480925, 'cuda': 1222906, 'count': 200}, 'permute': {'cpu': 110408, 'cuda': 7962, 'count': 200}, '_decode_proposals': {'cpu': 955516, 'cuda': 70169, 'count': 200}, 'perlevel_topk': {'cpu': 430593, 'cuda': 73615, 'count': 200}, 'concat': {'cpu': 27991, 'cuda': 2032, 'count': 200}, 'filter_nan': {'cpu': 2735458, 'cuda': 8799, 'count': 200}, 'filter_empty': {'cpu': 58281, 'cuda': 3341, 'count': 200}, 'NMS': {'cpu': 863113, 'cuda': 101282, 'count': 200}},
        {'anchor_generator': {'cpu': 419363, 'cuda': 16715, 'count': 200}, 'rpn_head': {'cpu': 533531, 'cuda': 1792119, 'count': 200}, 'permute': {'cpu': 117162, 'cuda': 9718, 'count': 200}, '_decode_proposals': {'cpu': 1004263, 'cuda': 95111, 'count': 200}, 'perlevel_topk': {'cpu': 461050, 'cuda': 89032, 'count': 200}, 'concat': {'cpu': 29703, 'cuda': 2298, 'count': 200}, 'filter_nan': {'cpu': 6740571, 'cuda': 10080, 'count': 200}, 'filter_empty': {'cpu': 62734, 'cuda': 3695, 'count': 200}, 'NMS': {'cpu': 931537, 'cuda': 107893, 'count': 200}},
        {'anchor_generator': {'cpu': 423126, 'cuda': 19440, 'count': 200}, 'rpn_head': {'cpu': 540270, 'cuda': 2768332, 'count': 200}, 'permute': {'cpu': 114667, 'cuda': 13661, 'count': 200}, '_decode_proposals': {'cpu': 1014410, 'cuda': 142757, 'count': 200}, 'perlevel_topk': {'cpu': 467162, 'cuda': 112611, 'count': 200}, 'concat': {'cpu': 29684, 'cuda': 2358, 'count': 200}, 'filter_nan': {'cpu': 14970333, 'cuda': 10582, 'count': 200}, 'filter_empty': {'cpu': 71448, 'cuda': 3868, 'count': 200}, 'NMS': {'cpu': 968696, 'cuda': 111867, 'count': 200}},
        {'anchor_generator': {'cpu': 313750, 'cuda': 22525, 'count': 200}, 'rpn_head': {'cpu': 527374, 'cuda': 4575109, 'count': 200}, 'permute': {'cpu': 112428, 'cuda': 15832, 'count': 200}, '_decode_proposals': {'cpu': 1186407, 'cuda': 173259, 'count': 200}, 'perlevel_topk': {'cpu': 471020, 'cuda': 121493, 'count': 200}, 'concat': {'cpu': 29832, 'cuda': 2395, 'count': 200}, 'filter_nan': {'cpu': 22500773, 'cuda': 10597, 'count': 200}, 'filter_empty': {'cpu': 63289, 'cuda': 4077, 'count': 200}, 'NMS': {'cpu': 926666, 'cuda': 114036, 'count': 200}},
        {'anchor_generator': {'cpu': 324931, 'cuda': 29612, 'count': 200}, 'rpn_head': {'cpu': 529602, 'cuda': 4952086, 'count': 200}, 'permute': {'cpu': 116073, 'cuda': 20251, 'count': 200}, '_decode_proposals': {'cpu': 1271490, 'cuda': 224837, 'count': 200}, 'perlevel_topk': {'cpu': 503322, 'cuda': 152984, 'count': 200}, 'concat': {'cpu': 29677, 'cuda': 2396, 'count': 200}, 'filter_nan': {'cpu': 31297452, 'cuda': 11078, 'count': 200}, 'filter_empty': {'cpu': 61375, 'cuda': 4056, 'count': 200}, 'NMS': {'cpu': 876784, 'cuda': 117767, 'count': 200}},
        {'anchor_generator': {'cpu': 325332, 'cuda': 42915, 'count': 200}, 'rpn_head': {'cpu': 539120, 'cuda': 6336453, 'count': 200}, 'permute': {'cpu': 112343, 'cuda': 24303, 'count': 200}, '_decode_proposals': {'cpu': 1259762, 'cuda': 281280, 'count': 200}, 'perlevel_topk': {'cpu': 497397, 'cuda': 179159, 'count': 200}, 'concat': {'cpu': 29152, 'cuda': 2402, 'count': 200}, 'filter_nan': {'cpu': 41599499, 'cuda': 11281, 'count': 200}, 'filter_empty': {'cpu': 60992, 'cuda': 4176, 'count': 200}, 'NMS': {'cpu': 885844, 'cuda': 117457, 'count': 200}}
    ]
    for _s in stats:
        _s['filter'] = {k: _s['filter_nan'][k] + _s['filter_empty'][k] for k in ['cpu', 'cuda']}
        _s['filter']['count'] = _s['filter_nan']['count']
    plt.figure(figsize=(6, 6))
    keys = ['anchor_generator', 'rpn_head', 'permute', '_decode_proposals', 'perlevel_topk', 'filter', 'NMS']
    descs = {'anchor_generator': 'anchor generation', 'rpn_head': 'scores & deltas generation', 'permute': 'tensor permutation', '_decode_proposals': 'applying deltas', 'perlevel_topk': 'per-level top-K', 'concat': 'proposal concat', 'filter': 'remove Inf/NaN & empty boxes', 'NMS': 'per-level NMS & global top-K'}
    legends = []
    for i, k in enumerate(keys):
        plt.plot(scales, [_s[k]['cpu'] / _s[k]['count'] / 1000 for _s in stats], linestyles[i] + '-')
        legends.append(descs[k] + ' [CPU]')
        plt.plot(scales, [_s[k]['cuda'] / _s[k]['count'] / 1000 for _s in stats], linestyles[i])
        legends.append(descs[k] + ' [CUDA]')
    plt.legend(legends)
    # plt.ylim(0, 25)
    plt.ylim(0, 225)
    plt.xlim(0.95, 2.3)
    plt.xlabel('input scale')
    plt.ylabel('inference time (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_results_fpn():
    scales = [1, 1.2, 1.5, 1.7, 2, 2.25]
    linestyles = ['ko-', 'b^-', 'r+-']
    # stats = [
    #     {'bottom_up': {'cpu': 12737962, 'cuda': 12735868, 'count': 200}, 'top_down': {'cpu': 593384, 'cuda': 909144, 'count': 200}},
    #     {'bottom_up': {'cpu': 12883369, 'cuda': 12881050, 'count': 200}, 'top_down': {'cpu': 599247, 'cuda': 1133718, 'count': 200}},
    #     {'bottom_up': {'cpu': 12471654, 'cuda': 12469812, 'count': 200}, 'top_down': {'cpu': 584339, 'cuda': 1530670, 'count': 200}},
    #     {'bottom_up': {'cpu': 12398654, 'cuda': 13057794, 'count': 200}, 'top_down': {'cpu': 640801, 'cuda': 1760904, 'count': 200}},
    #     {'bottom_up': {'cpu': 14899296, 'cuda': 16313433, 'count': 200}, 'top_down': {'cpu': 664665, 'cuda': 2325166, 'count': 200}},
    #     {'bottom_up': {'cpu': 18458075, 'cuda': 20394513, 'count': 200}, 'top_down': {'cpu': 678863, 'cuda': 2968224, 'count': 200}}
    # ]
    stats = [
        {'bottom_up': {'cpu': 5993412, 'cuda': 7343844, 'count': 200}, 'top_down': {'cpu': 376179, 'cuda': 1504089, 'count': 200}},
        {'bottom_up': {'cpu': 6163412, 'cuda': 10622966, 'count': 200}, 'top_down': {'cpu': 384864, 'cuda': 2194176, 'count': 200}},
        {'bottom_up': {'cpu': 6026996, 'cuda': 16470601, 'count': 200}, 'top_down': {'cpu': 446044, 'cuda': 3394570, 'count': 200}},
        {'bottom_up': {'cpu': 6236719, 'cuda': 20079133, 'count': 200}, 'top_down': {'cpu': 526458, 'cuda': 5308586, 'count': 200}},
        {'bottom_up': {'cpu': 6133392, 'cuda': 28219315, 'count': 200}, 'top_down': {'cpu': 557959, 'cuda': 6023590, 'count': 200}},
        {'bottom_up': {'cpu': 6588434, 'cuda': 35231336, 'count': 200}, 'top_down': {'cpu': 599478, 'cuda': 7623790, 'count': 200}}
    ]
    plt.figure(figsize=(6, 6))
    keys = ['bottom_up', 'top_down']
    descs = {'bottom_up': 'resnet bottom up extraction', 'top_down': 'feature pyramid generation'}
    legends = []
    for i, k in enumerate(keys):
        plt.plot(scales, [_s[k]['cpu'] / _s[k]['count'] / 1000 for _s in stats], linestyles[i] + '-')
        legends.append(descs[k] + ' [CPU]')
        plt.plot(scales, [_s[k]['cuda'] / _s[k]['count'] / 1000 for _s in stats], linestyles[i])
        legends.append(descs[k] + ' [CUDA]')
    plt.legend(legends)
    # plt.ylim(0, 105)
    plt.ylim(0, 180)
    plt.xlim(0.95, 2.3)
    plt.xlabel('input scale')
    plt.ylabel('inference time (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    # parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--scale', type=float, default=1.0)

    # parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    # parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    # parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    # parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    # parser.add_argument('--smallscale', default=False, type=bool)
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

    # parser.add_argument('--iters', type=int, help='total training iterations')
    # parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    # parser.add_argument('--image_batch_size', default=4, type=int)
    # parser.add_argument('--roi_batch_size', default=128, type=int)
    # parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--num_workers', default=0, type=int)
    # parser.add_argument('--refine_visualize_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    # parser.add_argument('--eval_outputfile', default=None, type=str)
    # parser.add_argument('--hold', default=0.005, type=float)

    # parser.add_argument('--ddp_num_gpus', type=int, default=1)
    # parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    # args.anno_models = sorted(list(set(args.anno_models)))
    print(args)

    # if not os.access(finetune_output, os.W_OK):
    #     os.mkdir(finetune_output)
    # assert os.path.isdir(finetune_output)
    # assert os.path.isdir(args.outputdir)
    # assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'rcnn':
        profile_rcnn(args)
    if args.opt == 'rpn':
        profile_rpn(args)
    if args.opt == 'fpn':
        profile_fpn(args)
    if args.opt == 'show':
        # show_results()
        # show_results_rpn()
        show_results_fpn()
    else:
        pass


'''
python profiling.py --opt rcnn --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --scale 1
python profiling.py --opt show
'''
