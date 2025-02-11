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
from detectron2.structures import BoxMode, Boxes
from detectron2.structures import ImageList, Instances
from detectron2.layers import batched_nms

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import bbox_inside, intersect_ratios


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF', '#FFFF33', '#00FF00', '#CC00CC']


class GeneralizedRCNNFinetune(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert len(batched_inputs) == 1
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert self.proposal_generator is not None and detected_instances is None and do_postprocess
        proposals, _ = self.proposal_generator(images, features, None)
        results, _, unfiltered_boxes, unfiltered_scores = self.roi_heads(images, features, proposals, None) # (Ri, K * B), (Ri, K + 1)
        assert len(proposals[0]) == unfiltered_boxes[0].size(0) == unfiltered_scores[0].size(0)
        assert not torch.jit.is_scripting(), 'Scripting is not supported for postprocess.'
        return detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), [im.detach().cpu() for im in images], [p.to('cpu') for p in proposals], [b.detach().cpu() for b in unfiltered_boxes], [s.cpu().detach() for s in unfiltered_scores]

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN)
        assert isinstance(net.proposal_generator, detectron2.modeling.proposal_generator.rpn.RPN)
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads)
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers)
        net.__class__ = GeneralizedRCNNFinetune
        net.proposal_generator.__class__ = RPNFinetune
        net.roi_heads.__class__ = StandardROIHeadsFinetune
        net.roi_heads.box_predictor.__class__ = FastRCNNOutputLayersFinetune
        return net


class FastRCNNOutputLayersFinetune(detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers):
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference(boxes, scores, image_shapes, self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image), boxes, scores


class StandardROIHeadsFinetune(detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            assert targets, '\'targets\' argument is required during training'
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, unfiltered_boxes, unfiltered_scores = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, unfiltered_boxes, unfiltered_scores

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
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            (pred_instances, _), unfiltered_boxes, unfiltered_scores = self.box_predictor.inference(predictions, proposals)
            return pred_instances, unfiltered_boxes, unfiltered_scores


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
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, 'RPN requires gt_instances in training!'
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
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
        return gt_labels, matched_gt_boxes


class Predictor(DefaultPredictor):
    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == 'RGB': original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {'image': image, 'height': height, 'width': width}
            return self.model([inputs])


def draw_bbox(im, annotations, thres=-1):
    fontsize, linewidth = int(min(im.shape[0], im.shape[1]) * 0.02), int(min(im.shape[0], im.shape[1]) / 300)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=fontsize)
    im = Image.fromarray(im, 'RGB')
    draw = ImageDraw.Draw(im)
    for ann in annotations:
        if 'score' in ann:
            if ann['score'] <= thres:
                continue
        assert ann['bbox_mode'] == BoxMode.XYXY_ABS
        x1, y1, x2, y2 = ann['bbox']
        cat = ann['category_id']
        draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[cat], width=linewidth)
        if 'score' in ann:
            draw.text((x1 + 1, y1 + 1), ('%.3f' % ann['score']).lstrip('0'), fill=bbox_rgbs[cat], stroke_width=1, font=font)
    im = np.array(im)
    return im


def show():
    from annotate import get_annotation_dict

    basedir = os.path.normpath(os.path.dirname(__file__))
    images_dense, images_sparse = get_annotation_dict()
    images = images_dense + images_sparse
    im1 = list(filter(lambda x: x['file_name'].find('SantaClausVillage_20221206_073403.00081893.jpg') >= 0, images))[0]
    images = [im1, copy.deepcopy(im1)]
    images[0]['file_name'] = 'SantaClausVillage_20221206_073403.00081893.jpg'
    images[1]['file_name'] = 'SantaClausVillage_20221206_073403.00081893.mixup.jpg'

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    if args.categories == 'mscoco':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
        for im in images:
            for ann in im['annotations']:
                if ann['category_id'] == 1: ann['category_id'] = 2
        _map = {_cat: 4 for _cat in range(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES)}
        _map[0], _map[2], _map[5], _map[7] = 0, 1, 1, 1
    elif args.categories == 'remapped':
        cfg.MODEL.WEIGHTS = os.path.join(os.path.normpath(os.path.dirname(__file__)), 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    else: return
    detector = Predictor(cfg)
    detector.model = GeneralizedRCNNFinetune.create_from_sup(detector.model)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    for im in tqdm.tqdm(images, ascii=True):
        im_arr = detectron2.data.detection_utils.read_image(im['file_name'], format='BGR')
        results, im_pp, proposals, class_boxes, class_scores = detector(im_arr)
        results = results[0]['instances'].to('cpu')
        results = {
            'bbox': results.pred_boxes.tensor.numpy().tolist(),
            'score': results.scores.numpy().tolist(),
            'label': results.pred_classes.numpy().tolist()
        }
        im_pp = im_pp[0].numpy().transpose(1, 2, 0)[:, :, ::-1]
        im_pp -= im_pp.min()
        im_pp /= im_pp.max()
        im_pp = (im_pp * 255).astype(np.uint8)
        proposals, class_boxes, class_scores = proposals[0], class_boxes[0].numpy(), class_scores[0].numpy()
        class_boxes = class_boxes.reshape(class_boxes.shape[0], class_scores.shape[1] - 1, -1)
        proposal_boxes = proposals.proposal_boxes.tensor
        proposal_scores = torch.sigmoid(proposals.objectness_logits)

        annotations_gt = im['annotations']
        thres_pred = 0.5
        annotations_pred = [{'bbox_mode': BoxMode.XYXY_ABS, 'bbox': results['bbox'][i], 'category_id': results['label'][i], 'score': results['score'][i]} for i in range(0, len(results['score']))]
        if args.categories == 'mscoco':
            for ann in annotations_pred: ann['category_id'] = _map[ann['category_id']]
        thres_proposal = 0.995
        annotations_proposal = [{'bbox_mode': BoxMode.XYXY_ABS, 'bbox': proposal_boxes[i].numpy().tolist(), 'category_id': 3, 'score': proposal_scores[i].item()} for i in range(0, len(proposals))]
        # thres_class = 0.75
        annotations_classes_bg = []
        if args.categories == 'remapped':
            for i in range(0, class_scores.shape[0]):
                if proposal_scores[i] <= thres_proposal:
                    continue
                k, s = class_scores[i].argmax(), class_scores[i].max()
                if k < cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                    bbox = class_boxes[i, k].tolist()
                else:
                    k = 2
                    bbox = proposal_boxes[i].numpy().tolist()
                annotations_classes_bg.append({'bbox_mode': BoxMode.XYXY_ABS, 'bbox': bbox, 'category_id': k, 'score': s})
        else:
            for i in range(0, class_scores.shape[0]):
                if proposal_scores[i] <= thres_proposal:
                    continue
                k, s = class_scores[i].argmax(), class_scores[i].max()
                if k < cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                    bbox = class_boxes[i, k].tolist()
                    k = _map[k]
                else:
                    k = 2
                    bbox = proposal_boxes[i].numpy().tolist()
                annotations_classes_bg.append({'bbox_mode': BoxMode.XYXY_ABS, 'bbox': bbox, 'category_id': k, 'score': s})

        plt.figure(figsize=(15.5, 9))
        plt.subplot(2, 2, 1)
        plt.imshow(draw_bbox(im_arr[:, :, ::-1], annotations_gt))
        plt.axis('off')
        plt.title('ground truth')
        plt.subplot(2, 2, 2)
        plt.imshow(draw_bbox(im_arr[:, :, ::-1], annotations_pred, thres=thres_pred))
        plt.axis('off')
        plt.title('Faster-RCNN output (%d-classes score > %.3f)' % (cfg.MODEL.ROI_HEADS.NUM_CLASSES, thres_pred))
        plt.subplot(2, 2, 3)
        plt.imshow(draw_bbox(im_pp, annotations_proposal, thres=thres_proposal))
        plt.axis('off')
        plt.title('RPN proposals (objectiveness score > %.3f)' % thres_proposal)
        plt.subplot(2, 2, 4)
        plt.imshow(draw_bbox(im_pp, annotations_classes_bg))
        plt.axis('off')
        plt.title('ROI scores of filtered proposals (%d+1-classes score)' % cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        plt.suptitle(im['file_name'])
        plt.tight_layout()
        plt.show()


def correlation(args):
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    if args.dataset == 'mscoco':
        from base_detector_train import get_coco_dicts
        images = get_coco_dicts(args, 'valid')
    elif args.dataset == 'scenes100':
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
        from finetune import all_annotation_dict
        images = all_annotation_dict(args)
    else: return
    # images = images[:50]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    if args.categories == 'mscoco':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    elif args.categories == 'remapped':
        cfg.MODEL.WEIGHTS = os.path.join(os.path.normpath(os.path.dirname(__file__)), 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    else: return
    detector = Predictor(cfg)
    detector.model = GeneralizedRCNNFinetune.create_from_sup(detector.model)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    count_gt, count_pred, obj_scores, roi_bg_scores, obj_score_thres = 0, 0, [], [], 0.995
    for im in tqdm.tqdm(images, ascii=True):
        results, _, proposals, _, class_scores = detector(detectron2.data.detection_utils.read_image(im['file_name'], format='BGR'))
        count_gt += len(im['annotations'])
        count_pred += results[0]['instances'].to('cpu').scores.size(0)
        obj_scores.append(torch.sigmoid(proposals[0].objectness_logits).numpy())
        roi_bg_scores.append(class_scores[0].numpy()[:, -1])
    obj_scores, roi_bg_scores = map(np.concatenate, [obj_scores, roi_bg_scores])
    print(count_gt, count_pred, obj_scores.shape, roi_bg_scores.shape)

    plt.figure(figsize=(16, 8.5))
    for i, (thres, desc) in enumerate(zip([0, obj_score_thres], ['all proposals', 'objectiveness > %.4f' % obj_score_thres])):
        thres_idx = obj_scores > thres
        xs, ys = obj_scores[thres_idx], roi_bg_scores[thres_idx]
        x_hist, x_bins = np.histogram(xs, bins=np.arange(thres - 0.0001, 1.0001, (1 - thres) / 30))
        x_hist = x_hist / x_hist.max() * 0.3
        x_bins = (x_bins[:-1] + x_bins[1:]) / 2
        y_hist, y_bins = np.histogram(ys, bins=np.arange(0.0001, 1.0001, 0.01))
        y_bins = (y_bins[:-1] + y_bins[1:]) / 2
        y_hist = (y_hist / y_hist.max() * 0.3 * (1 - thres)) + thres

        linear = LinearRegression().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
        k, b = linear.coef_[0, 0], linear.intercept_[0]
        r, _ = pearsonr(xs, ys)
        print(k, b, r)
        sample_idx = np.random.randint(0, xs.shape[0], size=(2000,))

        plt.subplot(1, 2, i + 1)
        # plt.grid(True)
        plt.scatter(xs[sample_idx], ys[sample_idx], marker='+', s=100, c='blue', alpha=0.5)
        plt.plot([-1, 2], [-1 * k + b, 2 * k + b], 'r-', lw=2, alpha=0.75)
        plt.plot(x_bins, x_hist, 'k--', lw=2, alpha=0.75)
        plt.plot(y_hist, y_bins, 'k:', lw=2, alpha=0.75)
        plt.xlim([thres - (1 - thres) * 0.02, 1 + (1 - thres) * 0.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('RPN objectiveness score')
        plt.ylabel('ROI background score')
        plt.legend(['Pearson $r=%.4f$' % r, '$y = %+.4f x %+.4f $' % (k, b), 'RPN histogram', 'ROI histogram'])
        plt.title(desc)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
    plt.suptitle('categories %s, %d classes, dataset %s, %d images, %d GT boxes, %d detections' % (args.categories.upper(), cfg.MODEL.ROI_HEADS.NUM_CLASSES, args.dataset.upper(), len(images), count_gt, count_pred))
    # plt.show()
    plt.savefig('rpn_roi_correlation_categories_%s_dataset_%s.pdf' % (args.categories.upper(), args.dataset.upper()))


def mine(args):
    from pseudo_label import DetectDataset
    basedir = os.path.normpath(os.path.dirname(__file__))
    cfg_r50 = get_cfg()
    cfg_r50.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    cfg_r50.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg_r50.MODEL.WEIGHTS = os.path.join(basedir, 'models', 'mscoco2017_remap_r50-fpn-3x.pth')
    cfg_r101 = get_cfg()
    cfg_r101.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    cfg_r101.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg_r101.MODEL.WEIGHTS = os.path.join(basedir, 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
    detectors = {'r50': Predictor(cfg_r50), 'r101': Predictor(cfg_r101)}
    for model_k in detectors:
        detectors[model_k].model = GeneralizedRCNNFinetune.create_from_sup(detectors[model_k].model)

    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        vfilelist = json.load(fp)['days'][args.mine_day]
    for vfilename in vfilelist:
        print('detect in', vfilename)
        frame_objs, ifilelist = {'r50': [], 'r101': []}, {'r50': [], 'r101': []}
        with open(os.path.join(basedir, 'frames', vfilename + '.json'), 'r') as fp:
            chunks = json.load(fp)['chunks']
        for idx_list in chunks:
            image_dicts = [{'file_name': os.path.join(basedir, 'frames', vfilename, '%08d.jpg' % x)} for x in idx_list]
            loader = torchdata.DataLoader(DetectDataset(image_dicts), batch_size=None, collate_fn=DetectDataset.collate, shuffle=False, num_workers=2)
            for fn, im in tqdm.tqdm(loader, total=len(image_dicts), ascii=True, desc='%d ~ %d' % (min(idx_list), max(idx_list))):
                for model_k in ['r50', 'r101']:
                    ifilelist[model_k].append(os.path.basename(fn))
                    frame_objs[model_k].append([])
                    _, im_pp, proposals, class_boxes, class_scores = detectors[model_k](im)
                    im_pp = im_pp[0].numpy().transpose(1, 2, 0)
                    proposals, class_boxes, class_scores = proposals[0], class_boxes[0].numpy() * (im.shape[0] / im_pp.shape[0]), class_scores[0].numpy()
                    proposal_scores = torch.sigmoid(proposals.objectness_logits).detach().cpu().numpy()
                    class_boxes = class_boxes.reshape(class_boxes.shape[0], class_scores.shape[1] - 1, -1)
                    assert proposal_scores.shape[0] == class_boxes.shape[0] == class_scores.shape[0]
                    for i in range(0, class_scores.shape[0]):
                        if proposal_scores[i] <= args.mine_obj_thres: continue
                        if len(thing_classes) != class_scores[i].argmax(): continue
                        k = int(class_scores[i, :len(thing_classes)].argmax())
                        frame_objs[model_k][-1].append({
                            'bbox': class_boxes[i, k].tolist(),
                            'label': k,
                            'obj_score': float(proposal_scores[i]),
                            'class_score': class_scores[i].tolist()
                        })

        result_json_zip = os.path.join(basedir, 'self_supervision', 'fn_mining', 'mining_%s_objthres%.4f.json.gz' % (vfilename, args.mine_obj_thres))
        with gzip.open(result_json_zip, 'wt') as fp:
            fp.write(json.dumps({
                'classes': thing_classes,
                'frames': ifilelist,
                'dets': frame_objs
            }))


def sample_false_negatives(day_idx, obj_score_thres=0.99, max_sample_per_image=10, min_box_area=100, max_box_area=250000, max_w_ratio=0.5, max_h_ratio=0.5):
    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        clips = json.load(fp)
    meta, vfilelist = clips['meta'], clips['days'][day_idx]
    print('hard negatives from day %d: %d videos' % (day_idx, len(vfilelist)))
    images = []
    for vfilename in tqdm.tqdm(vfilelist, ascii=True):
        with gzip.open(os.path.join(basedir, 'self_supervision', 'fn_mining', 'mining_%s_objthres0.9900.json.gz' % vfilename), 'rt') as fp:
            data = json.loads(fp.read())
        dets, ifilelist = data['dets'], data['frames']
        assert set(dets.keys()) == set(ifilelist.keys()) == set(['r50', 'r101'])
        for i in range(0, len(ifilelist['r50'])):
            assert ifilelist['r50'][i] == ifilelist['r101'][i]
            annotations = []
            for model_k in ['r50', 'r101']:
                for ann in dets[model_k][i]:
                    if ann['obj_score'] <= obj_score_thres: continue
                    w, h = ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]
                    if w * h < min_box_area or w * h > max_box_area: continue
                    if w > meta['W'] * max_w_ratio or h > meta['H'] * max_h_ratio: continue
                    annotations.append({'bbox': ann['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': ann['label'], 'score': ann['obj_score'], 'model': model_k})
            if len(annotations) > max_sample_per_image:
                random.shuffle(annotations)
                annotations = annotations[:max_sample_per_image]
            images.append({'file_name': os.path.join(basedir, 'frames', vfilename, ifilelist['r50'][i]), 'image_id': 0, 'height': meta['H'], 'width': meta['W'], 'annotations': annotations})
    print('%d images, %d bboxes' % (len(images), sum(map(lambda ann: len(ann['annotations']), images))))
    return images


from annotate import draw_bbox
def _draw_bbox_wrapper(im):
    im_arr = skimage.io.imread(im['file_name'])
    im_arr = draw_bbox(im_arr, im['annotations'], im['file_name'][-46:])
    return im_arr

def visualize_fn_mining():
    basedir = os.path.normpath(os.path.dirname(__file__))
    dst_fn_mine = []
    for d in [0, 1, 2, 3]:
        dst_fn_mine = dst_fn_mine + sample_false_negatives(d, obj_score_thres=0.01, max_sample_per_image=10000)
    writer = skvideo.io.FFmpegWriter(os.path.join(basedir, 'self_supervision', 'fn_mining.mp4'), inputdict={'-r': '5'}, outputdict={'-vcodec': 'libx265', '-r': '5', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '27'})
    im_Q = []
    for im in tqdm.tqdm(dst_fn_mine, ascii=True):
        im_Q.append(im)
        if len(im_Q) > 300:
            pool = ProcessPool(processes=6)
            im_arr_list = pool.map_async(_draw_bbox_wrapper, im_Q).get()
            pool.close()
            pool.join()
            for im_arr in im_arr_list:
                writer.writeFrame(im_arr)
            im_Q, im_arr_list = [], None
    if len(im_Q) > 0:
        for im in im_Q:
            writer.writeFrame(_draw_bbox_wrapper(im))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['show', 'correlation', 'mine', 'visualize'])
    parser.add_argument('--dataset', type=str, choices=['mscoco', 'scenes100'])
    parser.add_argument('--categories', type=str, choices=['mscoco', 'remapped'])
    parser.add_argument('--mine_day', type=int)
    parser.add_argument('--mine_obj_thres', type=float, default=0.99)
    args = parser.parse_args()

    if args.opt == 'show':
        show()
    if args.opt == 'correlation':
        args.id, args.cocodir, args.smallscale = '', '../../../MSCOCO2017', False
        correlation(args)
    if args.opt == 'mine':
        args.dataset = args.categories = None
        mine(args)
    if args.opt == 'visualize':
        visualize_fn_mining()

# for X in mscoco scenes100 ; do for Y in mscoco remapped ; do python false_negative_mining.py --opt correlation --dataset ${X} --categories ${Y} ; done ; done
# python false_negative_mining.py --opt mine --mine_day 0