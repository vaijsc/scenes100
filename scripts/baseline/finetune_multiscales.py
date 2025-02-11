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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_multiscales')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFinetuneMultiScales(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
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

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def inference_multiscales(self, batched_inputs_scales: List[List[Dict[str, torch.Tensor]]], scales: List[float], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert detected_instances is None, 'pre-computed instances not supported'
        assert do_postprocess, 'must do postprocess to match different scales'
        assert len(batched_inputs_scales) == len(scales)

        raw_boxes_scales, raw_scores_scales, raw_shapes_scales, image_sizes_scales = [], [], [], []
        for batched_inputs in batched_inputs_scales:
            assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            proposals, _ = self.proposal_generator(images, features, None)
            _boxes, _scores, _shapes = self.roi_heads._forward_box_raw_boxes(features, proposals)
            # print(_scores[0].size(), _boxes[0].size(), _shapes[0])
            raw_boxes_scales.append(_boxes)
            raw_scores_scales.append(_scores)
            raw_shapes_scales.append(_shapes)
            image_sizes_scales.append(images.image_sizes)

        # convert list of # scales to tuple of # images
        raw_boxes_combine, raw_scores_combine, raw_shapes_combine, image_sizes_combine = [[] for _ in range(len(raw_boxes_scales[0]))], [[] for _ in range(len(raw_scores_scales[0]))], [[] for _ in range(0, len(raw_shapes_scales[0]))], [[] for _ in range(0, len(image_sizes_scales[0]))]
        for _scale, _boxes, _scores, _shapes, _sizes in zip(scales, raw_boxes_scales, raw_scores_scales, raw_shapes_scales, image_sizes_scales):
            for i, (_boxes_i, _scores_i, _shapes_i, _sizes_i) in enumerate(zip(_boxes, _scores, _shapes, _sizes)):
                raw_boxes_combine[i].append(_boxes_i / _scale)
                raw_scores_combine[i].append(_scores_i)
                raw_shapes_combine[i].append((int(_shapes_i[0] / _scale), int(_shapes_i[1] / _scale)))
                image_sizes_combine[i].append((int(_sizes_i[0] / _scale), int(_sizes_i[1] / _scale)))
        for i in range(0, len(raw_boxes_combine)):
            raw_boxes_combine[i] = torch.cat(raw_boxes_combine[i], dim=0)
            raw_scores_combine[i] = torch.cat(raw_scores_combine[i], dim=0)
            raw_shapes_combine[i] = raw_shapes_combine[i][0]
            image_sizes_combine[i] = image_sizes_combine[i][0]
            # print(i, raw_scores_combine[i].size(), raw_boxes_combine[i].size(), raw_shapes_combine[i], image_sizes_combine[i])
        raw_boxes_combine, raw_scores_combine, raw_shapes_combine = map(tuple, [raw_boxes_combine, raw_scores_combine, raw_shapes_combine])

        results, _ = detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference(
            raw_boxes_combine,
            raw_scores_combine,
            raw_shapes_combine,
            self.roi_heads.box_predictor.test_score_thresh,
            self.roi_heads.box_predictor.test_nms_thresh,
            self.roi_heads.box_predictor.test_topk_per_image,
        )
        assert not self.roi_heads.mask_on, 'mask forward not supported'
        assert not self.roi_heads.keypoint_on, 'keypoint forward not supported'
        # print(batched_inputs_scales[0][0]['height'], batched_inputs_scales[0][0]['width'], len(results[0]), results[0].pred_boxes.tensor.max(dim=0).values)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            # for postprocessing, only 'height' & 'width' of input is used, they should be the same for all scales because they are the original image dimensions
            results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs_scales[0], image_sizes_combine)
        # print(len(results[0]['instances']), results[0]['instances'].pred_boxes.tensor.max(dim=0).values)
        return results

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers), 'roi is not detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'
        net.__class__ = GeneralizedRCNNFinetuneMultiScales
        net.roi_heads.__class__ = StandardROIHeadsFinetuneMultiScales
        net.roi_heads.box_predictor.__class__ = FastRCNNOutputLayersFinetuneMultiScales
        return net


class StandardROIHeadsFinetuneMultiScales(detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
    def _forward_box_raw_boxes(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        assert not self.training, 'multiscales only implemented for inference'
        return self.box_predictor.inference_raw_boxes(predictions, proposals)


class FastRCNNOutputLayersFinetuneMultiScales(detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers):
    def inference_raw_boxes(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return boxes, scores, image_shapes


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorMultiScales(DefaultPredictor):
    def __init__(self, cfg, scales):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNFinetuneMultiScales.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.scales = scales
        self.aug_list = [detectron2.data.transforms.ResizeShortestEdge([int(_s * cfg.INPUT.MIN_SIZE_TEST), int(_s * cfg.INPUT.MIN_SIZE_TEST)], int(_s * cfg.INPUT.MAX_SIZE_TEST)) for _s in self.scales]
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == 'RGB':
                original_image = original_image[:, :, ::-1]
            assert original_image.dtype == np.uint8
            height, width = original_image.shape[:2]
            image_scaleup = original_image
            if max(self.scales) > 1.001:
                image_scaleup = (skimage.transform.rescale(original_image, (max(self.scales), max(self.scales), 1.0)) * 255).astype(np.uint8)
            inputs = []
            for aug in self.aug_list:
                image = aug.get_transform(image_scaleup).apply_image(image_scaleup)
                image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
                inputs.append([{'image': image, 'height': height, 'width': width}])
            predictions = self.model.inference_multiscales(inputs, self.scales)[0]
            return predictions


def evaluate_all_videos(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}

    print('inference scales:', args.scales)
    results_file = 'results_AP_multiscales_%s_%s' % ('_'.join(list(map(lambda x: '%.2f' % x, args.scales))), args.model)
    print(results_file)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorMultiScales(cfg, args.scales)

    t0 = time.time()
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        detections = []
        loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [os.path.join(inputdir, 'unmasked', im['file_name']) for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
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
            results[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        del results[video_id]['raw']
        print(   '             %s' % '/'.join(results[video_id]['metrics']))
        for c in sorted(results[video_id]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[video_id]['results'][c])))
        print('[%d/%d finished in %.1f minutes]\n' % (video_id_list.index(video_id) + 1, len(video_id_list), (time.time() - t0) / 60.0))
    with open(results_file + '.json', 'w') as fp:
        json.dump(results, fp, indent=2)

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
    plt.suptitle(results_file)
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
    detector = PredictorMultiScales(cfg, args.scales)

    inputs_list = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        if max(args.scales) > 1.001:
            im_arr = (skimage.transform.rescale(im_arr, (max(args.scales), max(args.scales), 1.0)) * 255).astype(np.uint8)
        inputs = []
        for aug in detector.aug_list:
            im_tensor = aug.get_transform(im_arr).apply_image(im_arr)
            im_tensor = torch.as_tensor(im_tensor.astype('float32').transpose(2, 0, 1))
            inputs.append([{'image': im_tensor, 'height': im['height'], 'width': im['width']}])
        inputs_list.append(inputs)
    N1, N2 = 100, 900
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            detector.model.inference_multiscales(inputs_list[i % len(images)], args.scales)
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


def stats(args):
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    def get_bbox_stats(video_id):
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        bbox_area_stats = []
        for im in annotations:
            for ann in im['annotations']:
                if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                    _r = ann['bbox'][2] * ann['bbox'][3] / (im['width'] * im['height'])
                    bbox_area_stats.append(_r)
                elif ann['bbox_mode'] == BoxMode.XYXY_ABS:
                    _r = (ann['bbox'][2] - ann['bbox'][0]) * (ann['bbox'][3] - ann['bbox'][1]) / (im['width'] * im['height'])
                    bbox_area_stats.append(_r)
        return np.array(bbox_area_stats)

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    del base_AP['all_videos'], base_AP['mscoco2017_valid']
    for k in list(base_AP.keys()):
        del base_AP[k]['raw']
        base_AP[k[7:]] = base_AP[k]
    AP_scales = {}
    for _s in args.scales:
        with open(os.path.join(args.outputdir, 'results_AP_multiscales_%.2f_%s.json' % (_s, args.model)), 'r') as fp:
            AP_scales[_s] = json.load(fp)
    box_area_ratios, optimal_scales = [], []
    for video_id in video_id_list:
        box_area_ratios.append(get_bbox_stats(video_id).mean() * 100)
        _aps = [(_s, AP_scales[_s][video_id]['results']['weighted'][0]) for _s in args.scales]
        optimal_scales.append(max(_aps, key=lambda x: x[1])[0])

    optimal_scales, box_area_ratios = map(np.array, [optimal_scales, box_area_ratios])
    linear = LinearRegression().fit(optimal_scales.reshape(-1, 1), box_area_ratios.reshape(-1, 1))
    k, b = linear.coef_[0, 0], linear.intercept_[0]
    r, _ = pearsonr(optimal_scales, box_area_ratios)
    print(k, b, r)

    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    plt.scatter(optimal_scales, box_area_ratios, marker='x')
    plt.plot([0, 10], [b, 10 * k + b], 'r-', lw=2, alpha=0.75)
    plt.legend(['videos', 'Pearson $r=%.4f$' % r])
    plt.grid(True)
    plt.xlim(0.25, 4.25)
    plt.ylim(-0.2, 2)
    plt.xlabel('optimal scale in term of APG$^m_w$')
    plt.ylabel('average relative bounding box area')

    mscoco_scales = [0.5,   0.8,   1.0,   1.2,   1.5,   2.0,   2.5,   3.0,   3.5,   4.0]
    mscoco_mAPs   = [42.61, 49.49, 51.29, 51.62, 50.76, 47.08, 41.95, 36.69, 32.32, 28.65]
    plt.subplot(2, 1, 2)
    plt.plot(mscoco_scales, mscoco_mAPs,  'b+-', lw=2)
    plt.legend(['MSCOCO-2017'])
    plt.grid(True)
    plt.xlim(0.25, 4.25)
    plt.ylim(27.5, 52.5)
    plt.xlabel('scale')
    plt.ylabel('$AP^m$')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--scales', nargs='+', default=[1.0])

    # parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
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
    args.scales = sorted(list(map(float, args.scales)))
    print(args)

    # if not os.access(finetune_output, os.W_OK):
    #     os.mkdir(finetune_output)
    # assert os.path.isdir(finetune_output)
    # assert os.path.isdir(args.outputdir)
    # assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'eval':
        evaluate_all_videos(args)
    elif args.opt == 'tp':
        inference_throughput(args)
    elif args.opt == 'stats':
        stats(args)
    else:
        pass


'''
python finetune_multiscales.py --opt eval --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --scales 1
python finetune_multiscales.py --opt tp --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --id 001 --scales 1
python finetune_multiscales.py --opt stats --model r101-fpn-3x --scales 0.5 0.8 1 1.2 1.5 2 2.5 3 3.5 4 --outputdir F:\\intersections_results\\cvpr24\\multiscales
'''
