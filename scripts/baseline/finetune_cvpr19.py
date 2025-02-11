#!python3

'''
our implementation of
https://openaccess.thecvf.com/content_CVPR_2019/papers/RoyChowdhury_Automatic_Adaptation_of_Object_Detectors_to_New_Domains_Using_Self-Training_CVPR_2019_paper.pdf

official repo
https://github.com/AruniRC/detectron-self-train

set s_i per Eq(3), lambda=0.7
only apply forward tracking, no bbox refining, only self-training

This file should be more self-contained
'''

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
from detectron2.structures import BoxMode

from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
import detectron2.modeling
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_cvpr19')


class AnnotationType(enum.IntEnum):
    RELIABLE = 0
    PSEUDO   = 1


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
# apply per-bbox weights on the losses
class GeneralizedRCNNFinetune(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        assert 'proposals' not in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features = self.backbone(images.tensor)

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

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.proposal_generator, detectron2.modeling.proposal_generator.rpn.RPN), 'rpn is not detectron2.modeling.proposal_generator.rpn.RPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers), 'roi is not detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'

        net.__class__ = GeneralizedRCNNFinetune
        net.proposal_generator.__class__ = RPNFinetune
        net.roi_heads.__class__ = StandardROIHeadsFinetune
        net.roi_heads.box_predictor.__class__ = FastRCNNOutputLayersFinetune
        return net


# wrap detectron2/modeling/roi_heads/fast_rcnn.py:FastRCNNOutputLayers
class FastRCNNOutputLayersFinetune(detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers):
    def losses(self, predictions, proposals, targets=None, matched_gt_indices=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
        if not len(proposals):
            pseudo_scores = torch.empty(0)
        else:
            pseudo_scores = cat([targets[i].pseudo_scores[matched_gt_indices[i]] for i in range(0, len(targets))])
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, 'Proposals should not require gradients!'
            # If "gt_boxes" does not exist, the proposals must be all negative and should not be included in regression loss computation. Here we just use proposal_boxes as an arbitrary placeholder because its value won't be used in self.box_reg_loss().
            gt_boxes = cat([(p.gt_boxes if p.has('gt_boxes') else p.proposal_boxes).tensor for p in proposals], dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        loss_cls = cross_entropy(scores, gt_classes, reduction='none')
        loss_cls = (loss_cls * pseudo_scores).mean()
        losses = {
            # 'loss_cls': cross_entropy(scores, gt_classes, reduction='mean'),
            'loss_cls': loss_cls,
            'loss_box_reg': self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


# wrap detectron2/modeling/roi_heads/roi_heads.py:StandardROIHeads
class StandardROIHeadsFinetune(detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, '\'targets\' argument is required during training'
            proposals, matched_gt_indices = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self.training:
            losses = self._forward_box(features, proposals, targets, matched_gt_indices)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None, matched_gt_indices=None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals, targets, matched_gt_indices)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes. In the case of learned proposals (e.g., RPN), when training starts the proposals will be low quality due to random initialization. It's possible that none of these initial proposals have high enough overlap with the gt objects to be used as positive examples for the second stage components (box head, cls head, mask head). Adding the gt boxes to the set of proposals ensures that the second stage components will have some positive examples from the start of training. For RPN, this augmentation improves convergence and empirically improves box AP on COCO by about 0.5 points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []
        matched_gt_indices = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            matched_gt_indices.append(matched_idxs[sampled_idxs])

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_" and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads like masks, keypoints, etc, will filter the proposals again, (by foreground/background, or number of keypoints in the image, etc) so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith('gt_') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be. Therefore the returned proposals won't have any gt_* fields, except for a gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar('roi_head/num_fg_samples', np.mean(num_fg_samples))
        storage.put_scalar('roi_head/num_bg_samples', np.mean(num_bg_samples))

        return proposals_with_gt, matched_gt_indices


# wrap detectron2/modeling/proposal_generator/rpn.py:RPN
class RPNFinetune(detectron2.modeling.proposal_generator.rpn.RPN):
    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.
        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, 'RPN requires gt_instances in training!'
            gt_labels, gt_boxes, matched_gt_indices = self.label_and_sample_anchors(anchors, gt_instances)
            pseudo_scores = []
            for i in range(0, len(matched_gt_indices)):
                pseudo_scores.append(gt_instances[i].pseudo_scores[matched_gt_indices[i]])
            losses = self.losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, pseudo_scores)
        else:
            losses = {}

        proposals = self.predict_proposals(anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes)
        return proposals, losses

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.
        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_indices = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

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
            matched_gt_indices.append(matched_idxs)
        return gt_labels, matched_gt_boxes, matched_gt_indices

    @torch.jit.unused
    def losses(self, anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, pseudo_scores):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.
        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        pseudo_scores = torch.stack(pseudo_scores)

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar('rpn/num_pos_anchors', num_pos_anchors / num_images)
        storage.put_scalar('rpn/num_neg_anchors', num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(anchors, self.box2box_transform, pred_anchor_deltas, gt_boxes, pos_mask, box_reg_loss_type=self.box_reg_loss_type, smooth_l1_beta=self.smooth_l1_beta)

        valid_mask = gt_labels >= 0
        # objectness_loss = F.binary_cross_entropy_with_logits(cat(pred_objectness_logits, dim=1)[valid_mask], gt_labels[valid_mask].to(torch.float32), reduction='sum')

        # apply bbox weights to objectness loss
        objectness_loss = F.binary_cross_entropy_with_logits(cat(pred_objectness_logits, dim=1)[valid_mask], gt_labels[valid_mask].to(torch.float32), reduction='none')
        pseudo_scores_valid = pseudo_scores[valid_mask]
        # print(objectness_loss.sum(), objectness_loss.size(), pseudo_scores_valid.size(), pseudo_scores_valid.min(), pseudo_scores_valid.max())
        objectness_loss = (objectness_loss * pseudo_scores_valid).sum()
        normalizer = self.batch_size_per_image * num_images
        # print(objectness_loss, normalizer)
        losses = {
            'loss_rpn_cls': objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer for loc loss. But it doesn't matter in practice
            'loss_rpn_loc': localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses


def gather_annotations(args):
    dst = TrainingFrames(args.id)
    imagedir = os.path.join(dst.lmdb_path, 'jpegs')
    det_filelist, sot_filelist = [], []
    for m in args.anno_models:
        det_filelist.append(os.path.join(dst.lmdb_path, 'detect_%s.json.gz' % m))
        sot_filelist.append(os.path.join(dst.lmdb_path, 'detect_%s_DiMP.json.gz' % m))
    for f in det_filelist + sot_filelist:
        assert os.access(f, os.R_OK), '%s not readable' % f

    # collate bboxes from tracking & detection
    dict_json, count_det, count_sot = [], 0, 0
    for i in range(0, len(dst)):
        dict_json.append({'file_name': os.path.join(imagedir, dst.ifilelist[i]), 'image_id': i, 'height': dst.meta['meta']['video']['H'], 'width': dst.meta['meta']['video']['W'], 'annotations': [], 'det_count': 0, 'sot_count': 0})

    for f in det_filelist:
        with gzip.open(f, 'rt') as fp:
            dets = json.loads(fp.read())['dets']
        assert len(dets) == len(dict_json), 'detection & dataset mismatch'
        for i in range(0, len(dets)):
            for j in range(0, len(dets[i]['score'])):
                if dets[i]['score'][j] < args.det_score_thres:
                    continue
                dict_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'src': 'det', 'score': dets[i]['score'][j]})
                dict_json[i]['det_count'] += 1
                count_det += 1

    for f in sot_filelist:
        with gzip.open(f, 'rt') as fp:
            _t = json.loads(fp.read())
            _forward = _t['forward']
        assert len(_forward) == len(dict_json), 'tracking & dataset mismatch'
        for i in range(0, len(_forward)):
            for tr in _forward[i]:
                dict_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
                dict_json[i]['sot_count'] += 1
                count_sot += 1
    print('finish reading from detection & tracking results')
    print('%d images, detections %d, tracks %d' % (len(dict_json), count_det, count_sot))
    return dict_json, count_det + count_sot


def all_pseudo_annotations(args):
    random.seed(42)
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    dict_json_all, count_bboxes_all, id_back = [], 0, args.id
    for v in video_id_list:
        args.id = v
        dict_json_v, count_bboxes_v = gather_annotations(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            count_bboxes_v *= images_per_video_cap / len(dict_json_v)
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all.append(dict_json_v)
        count_bboxes_all += count_bboxes_v
    args.id = id_back
    print('all videos %d images, %d bboxes' % (sum(map(len, dict_json_all)), count_bboxes_all))
    return dict_json_all, count_bboxes_all


def get_annotation_dict(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    return annotations


def all_annotation_dict(args):
    annotations_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        annotations_all = annotations_all + get_annotation_dict(args)
    args.id = id_back
    for i in range(0, len(annotations_all)):
        annotations_all[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(annotations_all), sum(list(map(lambda x: len(x['annotations']), annotations_all)))))
    return annotations_all


######################################################
#####   many RCNN library methods are modified   #####
##### modded RCNN only tested on detectron2 v0.6 #####
#####  with models: R50-FPN, R101-FPN, X101-FPN  #####
######################################################

# wrap detectron2/detectron2/data/dataset_mapper.py:DatasetMapper
# include score information in the mapped dataset, later be used for adjusting per-bbox weights
class DatasetMapperFinetune(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        ret = super(DatasetMapperFinetune, self).__call__(dataset_dict)
        score_theta, score_lambda = 0.5, 0.7
        if 'annotations' in dataset_dict:
            pseudo_scores = [1.0 for _ in range(0, ret['instances'].gt_classes.size(0))]
            # not sure why this happens, but it seems very rare, just use all 1.0 for pseudo_scores
            if len(dataset_dict['annotations']) != len(pseudo_scores):
                print('gt_classes & annotations mismatch %s %s' % (len(ret['instances'].gt_classes), len(dataset_dict['annotations'])))
                # print(ret['instances'], dataset_dict)
                ret['instances'].annotation_type = torch.tensor([AnnotationType.RELIABLE for _ in range(0, len(pseudo_scores))], dtype=torch.int8)
                ret['instances'].pseudo_scores = torch.tensor(pseudo_scores)
            else:
                annotation_type = [AnnotationType.PSEUDO if 'src' in ann else AnnotationType.RELIABLE for ann in dataset_dict['annotations']]
                annotation_type = torch.tensor(annotation_type, dtype=torch.int8)
                assert annotation_type.sum() == 0 or annotation_type.sum() == annotation_type.size(0), str(dataset_dict['annotations']) # should not have mixture of sources
                # all fields in an instance must have the same length
                if annotation_type.sum() != 0:
                    for i in range(0, len(pseudo_scores)):
                        ann = dataset_dict['annotations'][i]
                        if ann['src'] == 'det':
                            pseudo_scores[i] = ann['score'] * score_lambda + (1 - score_lambda) * 1.0
                        elif ann['src'] == 'sot':
                            pseudo_scores[i] = score_theta * score_lambda + (1 - score_lambda) * 1.0
                        else:
                            raise NotImplementedError # this should not happen
                ret['instances'].annotation_type = annotation_type
                ret['instances'].pseudo_scores = torch.tensor(pseudo_scores)
        return ret
    @staticmethod
    def create_from_sup(mapper):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperFinetune
        return mapper


# DefaultTrainer._trainer is instance of SimpleTrainer
# DefaultTrainer & SimpleTrainer are subclass of TrainerBase
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
class FinetuneTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
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


def train_eval(trainer, prefix):
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)

    trainer.eval_results_all[0] = results_0
    trainer.train()

    with open(os.path.join(trainer.cfg.OUTPUT_DIR, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(trainer.cfg.OUTPUT_DIR, prefix + '.pth'))


def adapt():
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    _args.smallscale = False
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(_args, 'valid')
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]

    if args.id in video_id_list:
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, get_annotation_dict(args)
        desc_pseudo_anno = '_'.join(args.anno_models)
        dst_pseudo_anno = gather_annotations(args)[0]
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(_args, 'train')
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
    elif args.id == 'compound':
        import functools
        args.id = '_compound'
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, all_annotation_dict(args)
        desc_pseudo_anno = '_'.join(args.anno_models)
        dst_pseudo_anno = all_pseudo_annotations(args)[0]
        dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(_args, 'train')
            dst_cocotrain = dst_cocotrain * (len(dst_pseudo_anno) // len(dst_cocotrain) + 1)
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
    else:
        raise NotImplementedError

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
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperFinetune.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj)
    assert isinstance(trainer.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
    trainer.model = GeneralizedRCNNFinetune.create_from_sup(trainer.model)
    trainer.resume_or_load(resume=False)

    prefix = 'adapt%s_%s_anno_%s_CVPR19' % (args.id, args.model, desc_pseudo_anno)
    train_eval(trainer, prefix)

    os.replace(os.path.join(cfg.OUTPUT_DIR, prefix + '.json'), os.path.join(os.path.dirname(__file__), prefix + '.json'))
    os.replace(os.path.join(cfg.OUTPUT_DIR, prefix + '.pth'), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'r') as fp:
        data = json.load(fp)
    aps, lr_history, loss_history = data['results'], data['lr_history'], data['loss_history']
    iter_list = sorted(list(map(int, aps.keys())))
    dst_list = [desc_cocovalid, desc_manual_valid]
    assert len(dst_list) == 2
    dst_list = {k: {'mAP': [], 'AP50': []} for k in dst_list}
    for i in iter_list:
        for k in dst_list:
            dst_list[k]['mAP'].append(aps[str(i)][k]['bbox']['AP'])
            dst_list[k]['AP50'].append(aps[str(i)][k]['bbox']['AP50'])

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
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')

    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')

    parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    args.anno_models = sorted(list(set(args.anno_models)))
    print(args)

    assert len(args.anno_models) == 1 and args.anno_models[0] == args.model, 'CVPR19 paper only uses self-training'

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'adapt':
        adapt()
    else:
        pass
    exit(0)


'''
conda deactivate && conda activate detectron2
cd /nfs/detection/zekun/Intersections/scripts/baseline

python finetune_cvpr19.py --id 001 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --train_on_coco 1 --image_batch_size 4

python finetune_cvpr19.py --id 050 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 0 --iters 300 --eval_interval 30 --train_on_coco 1 --image_batch_size 2 --not_eval_coco 1 --lr 0.001
'''
