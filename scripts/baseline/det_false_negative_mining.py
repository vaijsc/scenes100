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
from models import get_cfg_base_model
from decode_training import TrainingFrames


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF', '#FFFF33', '#00FF00', '#CC00CC']


def move_det_sot():
    filenames = ['detect_r50-fpn-3x.json.gz', 'detect_r50-fpn-3x_DiMP.json.gz', 'detect_r101-fpn-3x.json.gz', 'detect_r101-fpn-3x_DiMP.json.gz', 'refine_r101-fpn-3x_r50-fpn-3x.mp4']
    for video_id in tqdm.tqdm(video_id_list, ascii=True):
        inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', video_id))
        outputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label'))
        for f in filenames:
            assert os.access(os.path.join(inputdir, f), os.R_OK)
            os.rename(os.path.join(inputdir, f), os.path.join(outputdir, video_id + '_' + f))


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
            image = torch.as_tensor(image.astype(np.float32).transpose(2, 0, 1))
            inputs = {'image': image, 'height': height, 'width': width}
            return self.model([inputs])


def mine(args):
    def round_floats(o, n):
        if isinstance(o, float):         return round(o, n)
        if isinstance(o, dict):          return {k: round_floats(v, n) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [round_floats(x, n) for x in o]
        return o

    result_json_zip = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label', '%s_false_negative_mining_objthres%.4f.json.gz' % (args.id, args.mine_obj_thres)))
    if os.access(result_json_zip, os.R_OK) and not args.override:
        print(result_json_zip, 'exists, skipped')
        return

    model_list = ['r50-fpn-3x', 'r101-fpn-3x']
    detectors = {}
    for model_k in model_list:
        p = Predictor(get_cfg_base_model(model_k))
        p.model = GeneralizedRCNNFinetune.create_from_sup(p.model)
        detectors[model_k] = p
    dst = TrainingFrames(args.id)
    print('mine hard negatives in %s' % dst, flush=True)
    frame_objs, ifilelist = {model_k: [] for model_k in model_list}, {model_k: [] for model_k in model_list}

    for im, _, fn, _ in tqdm.tqdm(dst, ascii=True, desc='mining hard negatives in %s' % args.id):
        im = im[:, :, ::-1]
        for model_k in model_list:
            ifilelist[model_k].append(os.path.basename(fn))
            _, im_pp, proposals, class_boxes, class_scores = detectors[model_k](im)
            im_pp = im_pp[0].numpy().transpose(1, 2, 0)
            proposals, class_boxes, class_scores = proposals[0], class_boxes[0].numpy() * (im.shape[0] / im_pp.shape[0]), class_scores[0].numpy()
            proposal_scores = torch.sigmoid(proposals.objectness_logits).detach().cpu().numpy()
            class_boxes = class_boxes.reshape(class_boxes.shape[0], class_scores.shape[1] - 1, -1)
            assert proposal_scores.shape[0] == class_boxes.shape[0] == class_scores.shape[0]
            frame_objs[model_k].append({'bbox': [], 'label': [], 'obj_score': [], 'class_score': []})
            for i in range(0, class_scores.shape[0]):
                if proposal_scores[i] <= args.mine_obj_thres: continue
                if len(thing_classes) != class_scores[i].argmax(): continue
                k = int(class_scores[i, :len(thing_classes)].argmax())
                frame_objs[model_k][-1]['bbox'].append(round_floats(class_boxes[i, k].tolist(), 2))
                frame_objs[model_k][-1]['label'].append(k)
                frame_objs[model_k][-1]['obj_score'].append(round_floats(float(proposal_scores[i]), 5))
                frame_objs[model_k][-1]['class_score'].append(round_floats(class_scores[i].tolist(), 4))

    with gzip.open(result_json_zip, 'wt') as fp:
        fp.write(json.dumps({
            'classes': thing_classes,
            'frames': ifilelist,
            'dets': frame_objs
        }))


def fn_val_set():
    detector = Predictor(get_cfg_base_model('r101-fpn-3x'))
    detector.model = GeneralizedRCNNFinetune.create_from_sup(detector.model)

    results_json = {}
    result_json_zip = os.path.normpath(os.path.join(os.path.dirname(__file__), 'scenes100_val_false_negative_mining_objthres0.99.json.gz'))
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        for i in range(0, len(annotations)):
            annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
            annotations[i]['image_id'] = i + 1
        detections = copy.deepcopy(annotations)
        for im in detections:
            im['annotations'] = []
        for im in tqdm.tqdm(detections, ascii=True, desc=video_id):
            im_arr = detectron2.data.detection_utils.read_image(im['file_name'], format=detector.input_format)
            results, im_pp, proposals, class_boxes, class_scores = detector(im_arr)
            results = results[0]['instances'].to('cpu')
            for bbox, score, label in zip(results.pred_boxes.tensor.numpy().tolist(), results.scores.numpy().tolist(), results.pred_classes.numpy().tolist()):
                im['annotations'].append({'bbox': bbox, 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label, 'src': 'det', 'score': score})

            im_pp = im_pp[0].numpy().transpose(1, 2, 0)
            proposals, class_boxes, class_scores = proposals[0], class_boxes[0].numpy() * (im_arr.shape[0] / im_pp.shape[0]), class_scores[0].numpy()
            proposal_scores = torch.sigmoid(proposals.objectness_logits).detach().cpu().numpy()
            class_boxes = class_boxes.reshape(class_boxes.shape[0], class_scores.shape[1] - 1, -1)
            assert proposal_scores.shape[0] == class_boxes.shape[0] == class_scores.shape[0]
            for i in range(0, class_scores.shape[0]):
                if proposal_scores[i] <= 0.99: continue
                if len(thing_classes) != class_scores[i].argmax(): continue
                k = int(class_scores[i, :len(thing_classes)].argmax())
                im['annotations'].append({'bbox': class_boxes[i, k].tolist(), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': k, 'src': 'fn', 'obj_score': float(proposal_scores[i]), 'class_score': class_scores[i].tolist()})

        results_json[video_id] = {'annotations': annotations, 'detections': detections}
    with gzip.open(result_json_zip, 'wt') as fp:
        fp.write(json.dumps(results_json))


def draw_bbox(im):
    im_arr, annotations = skimage.io.imread(im['file_name']), im['annotations']
    fontsize, linewidth = int(min(im_arr.shape[0], im_arr.shape[1]) * 0.04), int(min(im_arr.shape[0], im_arr.shape[1]) / 300)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=fontsize)
    im_arr = Image.fromarray(im_arr, 'RGB')
    draw = ImageDraw.Draw(im_arr)
    for ann in annotations:
        assert ann['bbox_mode'] == BoxMode.XYXY_ABS
        x1, y1, x2, y2 = ann['bbox']
        cat = ann['category_id']
        draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[cat], width=linewidth)
    draw.text((2, 2), os.path.basename(im['file_name']), fill='#FFFFFF', stroke_width=1, font=font)
    im_arr = np.array(im_arr)
    return im_arr

def visualize_fn_mining(args):
    dst = TrainingFrames(args.id)
    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label'))
    with gzip.open(os.path.join(basedir, '%s_false_negative_mining_objthres0.9900.json.gz' % args.id), 'rt') as fp:
        data = json.loads(fp.read())
    ifilelist, dets = data['frames']['r101-fpn-3x'], data['dets']
    images = []
    for i in range(0, len(ifilelist), 25):
        annotations = []
        for model_k in ['r50-fpn-3x', 'r101-fpn-3x']:
            for j in range(0, len(dets[model_k][i]['label'])):
                annotations.append({'bbox': dets[model_k][i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'category_id': dets[model_k][i]['label'][j]})
        images.append({'file_name': os.path.join(dst.lmdb_path, 'jpegs', ifilelist[i]), 'annotations': annotations})
    writer = skvideo.io.FFmpegWriter(os.path.join(basedir, '%s_false_negative_mining_objthres0.9900.mp4' % args.id), inputdict={'-r': '5'}, outputdict={'-vcodec': 'libx265', '-r': '5', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '27'})
    im_Q = []
    for im in tqdm.tqdm(images, ascii=True):
        im_Q.append(im)
        if len(im_Q) > 200:
            pool = ProcessPool(processes=6)
            im_arr_list = pool.map_async(draw_bbox, im_Q).get()
            pool.close()
            pool.join()
            for im_arr in im_arr_list:
                writer.writeFrame(im_arr)
            im_Q, im_arr_list = [], None
    if len(im_Q) > 0:
        for im in im_Q:
            writer.writeFrame(draw_bbox(im))
    writer.close()


if __name__ == '__main__':
    # move_det_sot(); exit()
    fn_val_set(); exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['mine', 'visualize'])
    parser.add_argument('--ids', nargs='+', default=[], choices=video_id_list, help='video IDs')
    parser.add_argument('--mine_obj_thres', type=float, default=0.99)
    parser.add_argument('--override', type=bool, default=False)
    args = parser.parse_args()

    if args.opt == 'mine':
        for video_id in args.ids:
            args.id = video_id
            mine(args)
    if args.opt == 'visualize':
        for video_id in args.ids:
            args.id = video_id
            visualize_fn_mining(args)

'''
python det_false_negative_mining.py --opt mine --ids 001
python det_false_negative_mining.py --opt visualize --ids 001

python finetune.py --id 003 --opt adapt --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 500 --eval_interval 251 --train_on_coco 1 --image_batch_size 4 --not_eval_coco 1 --fn_max_samples 10

python finetune_mixup.py --id 003 --opt adapt --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 500 --eval_interval 251 --train_on_coco 1 --image_batch_size 4 --not_eval_coco 1 --fn_max_samples 10

python finetune_wdiff_midfusion_mixup.py --opt adapt --id 003 --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --fusion_type average --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 160 --image_batch_size 4 --num_workers 3 --not_eval_coco 1 --fn_max_samples 10
'''
