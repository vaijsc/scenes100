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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_multiscales')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFeatureScaling(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
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

    def _scale_feature(self, batched_inputs, images):
        image_sizes_scale = []
        for (h, w) in images.image_sizes:
            image_sizes_scale.append((int(h * self.feature_scale), int(w * self.feature_scale)))
        images.image_sizes = image_sizes_scale
        for inp in batched_inputs:
            if not 'instances' in inp:
                continue
            inst = inp['instances']
            inst._image_size = (int(inst._image_size[0] * self.feature_scale), int(inst._image_size[1] * self.feature_scale))
            inst.gt_boxes.tensor *= self.feature_scale
        return batched_inputs, images

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        images = self.preprocess_image(batched_inputs)
        if abs(self.input_scale - 1.0) > 0.01:
            # print(images.tensor.size(), images.image_sizes)
            batched_inputs, images = self._scale_input(batched_inputs, images)
            # print(images.tensor.size(), images.image_sizes)
        features = self.backbone(images.tensor)
        if abs(self.feature_scale - 1.0) > 0.01:
            batched_inputs, images = self._scale_feature(batched_inputs, images)
            # print(images.tensor.size(), images.image_sizes)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            return results
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        assert 'instances' in batched_inputs[0]
        assert self.proposal_generator is not None

        images = self.preprocess_image(batched_inputs)
        if abs(self.input_scale - 1.0) > 0.01:
            batched_inputs, images = self._scale_input(batched_inputs, images)
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        features = self.backbone(images.tensor)
        if abs(self.feature_scale - 1.0) > 0.01:
            batched_inputs, images = self._scale_feature(batched_inputs, images)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    @staticmethod
    def create_from_sup(net, input_scale, feature_scale):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers), 'roi is not detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'
        net.__class__ = GeneralizedRCNNFeatureScaling
        net.backbone = FPNScaling.create_from_sup(net.backbone, feature_scale)
        net.input_scale = input_scale
        net.feature_scale = feature_scale
        return net


class FPNScaling(detectron2.modeling.backbone.FPN):
    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        # x = torch.nn.functional.interpolate(x, scale_factor=1.5, mode='bilinear')
        # _, _, H, W = x.size()
        # if (H % 32) != 0 or (W % 32) != 0:
        #     print(x.size())
        #     H = math.ceil(H / 32) * 32
        #     W = math.ceil(W / 32) * 32
        bottom_up_features = self.bottom_up(x)
        # for k in bottom_up_features:
        #     print(k, bottom_up_features[k].size())
        # exit()
        # bottom_up_features['res2'] = torch.nn.functional.interpolate(bottom_up_features['res2'], scale_factor=0.5, mode='bilinear')
        # bottom_up_features['res3'] = torch.nn.functional.interpolate(bottom_up_features['res3'], scale_factor=0.5, mode='bilinear')
        if abs(self.feature_scale - 1.0) > 0.01:
            for k in bottom_up_features:
                bottom_up_features[k] = torch.nn.functional.interpolate(bottom_up_features[k], scale_factor=self.feature_scale, mode='bilinear')
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
        feature_pyramid = {f: res for f, res in zip(self._out_features, results)}
        return feature_pyramid

    @staticmethod
    def create_from_sup(net, feature_scale):
        net.feature_scale = feature_scale
        net.__class__ = FPNScaling
        return net


def compare_feature(args):
    def _to_im(t):
        assert len(t.size()) == 3
        if t.size(0) > 3:
            # t = torch.stack([t.mean(dim=0), t.abs().mean(dim=0), t.std(dim=0)], dim=0)
            t = t[:3]
        t = t.cpu().numpy().transpose(1, 2, 0)
        t -= t.min()
        t /= t.max()
        return t

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = DefaultPredictor(cfg)
    model = detector.model
    for im in images:
        im_1 = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        with torch.no_grad():
            im_1 = torch.as_tensor(detector.aug.get_transform(im_1).apply_image(im_1).astype('float32').transpose(2, 0, 1))
            im_1_pp = model.preprocess_image([{'image': im_1, 'height': im['height'], 'width': im['width']}]).tensor
            im_2_pp = torch.nn.functional.interpolate(im_1_pp, scale_factor=2, mode='bilinear')
            tensors = [{'im': im_1_pp}, {'im': im_2_pp}]
            for _d in tensors:
                _d['bottom_up'] = model.backbone.bottom_up(_d['im'])

        key = 'res4'
        plt.figure(figsize=(20, 8))
        plt.subplot(2, 4, 1); _im=tensors[0]['im'][0]; plt.imshow(_to_im(_im)); plt.title('image: ' + str(list(_im.size())))
        plt.subplot(2, 4, 2); _im=tensors[0]['bottom_up'][key][0]; plt.imshow(_to_im(_im)); plt.title(key + ': ' + str(list(_im.size())))
        _f_2x = torch.nn.functional.interpolate(tensors[0]['bottom_up'][key], scale_factor=2, mode='bicubic')
        plt.subplot(2, 4, 3); _im=_f_2x[0]; plt.imshow(_to_im(_im)); plt.title(key + ' 2x: ' + str(list(_im.size())))
        plt.subplot(2, 4, 5); _im=tensors[1]['im'][0]; plt.imshow(_to_im(_im)); plt.title('image: ' + str(list(_im.size())))
        plt.subplot(2, 4, 7); _im=tensors[1]['bottom_up'][key][0]; plt.imshow(_to_im(_im)); plt.title(key + ': ' + str(list(_im.size())))
        _pixels = tensors[1]['bottom_up'][key][0].flatten().cpu().numpy()
        _pixels_2 = _f_2x[0].flatten().cpu().numpy()
        _max = max(np.absolute(_pixels).max(), np.absolute(_pixels_2).max())
        _bins = np.arange(-1.02 * _max, 1.02 * _max, _max / 34 * 2)
        _centers = (_bins[:-1] + _bins[1:]) / 2
        plt.subplot(2, 4, 4); plt.plot(_centers, np.histogram(_pixels, bins=_bins)[0]); plt.plot(_centers, np.histogram(_pixels_2, bins=_bins)[0]); plt.xlim(_max / -9, _max / 3); plt.legend(['upscaled image', 'upscaled feature map'])
        _diff = (_f_2x - tensors[1]['bottom_up'][key]).abs()
        plt.subplot(2, 4, 8); _im=_diff[0]; plt.imshow(_to_im(_im)); plt.title('diff: ' + str(list(_im.size())))
        plt.tight_layout(); plt.show()


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorScaling(DefaultPredictor):
    def __init__(self, cfg, input_scale, feature_scale):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNFeatureScaling.create_from_sup(self.model, input_scale, feature_scale)
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

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}

    results_file = 'results_AP_resscale_i%.2f_f%.2f_%s%s' % (args.input_scale, args.feature_scale, args.model, args.tag)
    print(results_file)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorScaling(cfg, args.input_scale, args.feature_scale)

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
    plt.suptitle('%s [%.3f ms/image] %s' % (results_file, t_total * 1000 / N_total, losses))
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


def get_diag_ratios(images):
    ratios = []
    for im in images:
        diag = (im['height'] ** 2 + im['width'] ** 2) ** 0.5
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                ratios.append(((ann['bbox'][2] - ann['bbox'][0]) ** 2 + (ann['bbox'][3] - ann['bbox'][1]) ** 2) ** 0.5 / diag)
            elif ann['bbox_mode'] == BoxMode.XYWH_ABS:
                ratios.append((ann['bbox'][2] ** 2 + ann['bbox'][3] ** 2) ** 0.5 / diag)
            else:
                raise NotImplementedError
    return np.array(ratios)


def evaluate_bdd100k(args):
    import contextlib
    from evaluation import eval_AP
    from bdd100k import get_bdd100k_dicts
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
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
    ratios = get_diag_ratios(images_HW)
    print('%d objects, mean diagonal percentage %.4f (%.4f)' % (ratios.shape[0], ratios.mean(), ratios.std()))
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

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
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
    ratios = get_diag_ratios(images_HW)
    print('%d objects, mean diagonal percentage %.4f (%.4f)' % (ratios.shape[0], ratios.mean(), ratios.std()))
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

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    images_train = get_cityscapes_dicts(args.cityscapesdir, 'train')
    images_val = get_cityscapes_dicts(args.cityscapesdir, 'val')
    images = images_train + images_val
    del images_train, images_val
    for i in range(0, len(images)):
        images[i]['image_id'] = i + 1
    ratios = get_diag_ratios(images)
    print('%d objects, mean diagonal percentage %.4f (%.4f)' % (ratios.shape[0], ratios.mean(), ratios.std()))
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

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
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
    ratios = get_diag_ratios(images)
    print('%d objects, mean diagonal percentage %.4f (%.4f)' % (ratios.shape[0], ratios.mean(), ratios.std()))
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

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    with open(os.path.join(args.scenes100dir, 'images.json'), 'r') as fp:
        images = json.load(fp)
        for im in images:
            im['file_name'] = os.path.join(args.scenes100dir, im['file_name'])
    ratios = get_diag_ratios(images)
    print('%d objects, mean diagonal percentage %.4f (%.4f)' % (ratios.shape[0], ratios.mean(), ratios.std()))
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


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorScaling(cfg, args.input_scale, args.feature_scale)

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
            detector.model.inference(inputs_list[i % len(images)])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


def occlusion_level(args):
    images_scenes100 = []
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            _images_v = json.load(fp)
        images_scenes100 = images_scenes100 + _images_v
    args.smallscale = False
    images_coco = get_coco_dicts(args, 'valid')
    for im in images_coco:
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
    from cityscapes import get_cityscapes_dicts
    images_train = get_cityscapes_dicts(args.cityscapesdir, 'train')
    images_val = get_cityscapes_dicts(args.cityscapesdir, 'val')
    images_cityscapes = images_train + images_val
    from kitti import get_kitti_dicts
    images_kitti = get_kitti_dicts(args.kittidir)
    from bdd100k import get_bdd100k_dicts
    images_bdd = get_bdd100k_dicts(args.bdddir, 'val')
    images_datasets = {'Scenes100': images_scenes100, 'COCO': images_coco, 'CityScapes': images_cityscapes, 'KITTI': images_kitti, 'BDD100K': images_bdd}

    def _sum_over_union(images):
        area_sum, area_union = 0, 0
        for im in tqdm.tqdm(images, ascii=True):
            if len(im['annotations']) < 1:
                continue
            for ann in im['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            xyxy = np.array([ann['bbox'] for ann in im['annotations']]).astype(np.int32)
            xyxy[np.where(xyxy < 0)] = 0
            blank = np.zeros(shape=(xyxy[:, 2].max(), xyxy[:, 3].max()), dtype=np.float32)
            for x1, y1, x2, y2 in xyxy:
                area_sum += (x2 - x1) * (y2 - y1)
                blank[x1 : x2, y1 : y2] = 1
            area_union += blank.sum()
        print(area_sum, area_union, area_sum / area_union)
    for k in images_datasets:
        print(k)
        _sum_over_union(images_datasets[k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--input_scale', type=float, default=1.0)
    parser.add_argument('--feature_scale', type=float, default=1.0)
    parser.add_argument('--not_compute_loss', type=bool, default=False)
    parser.add_argument('--not_save_results_json', type=bool, default=False)
    parser.add_argument('--tag', type=str, default='')

    # parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    # parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    # parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    # parser.add_argument('--smallscale', default=False, type=bool)
    parser.add_argument('--bdddir', type=str, help='BDD100K directory')
    parser.add_argument('--cityscapesdir', type=str, help='CityScapes directory')
    parser.add_argument('--kittidir', type=str, help='KITTI directory')
    parser.add_argument('--scenes100dir', type=str, help='scenes100 directory')
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
    # occlusion_level(args); exit(0)

    if args.opt == 'eval':
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
    elif args.opt == 'tp':
        inference_throughput(args)
    elif args.opt == 'compare':
        compare_feature(args)
    else:
        pass


'''
python finetune_feature_scaling.py --opt eval --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --eval_skip_coco 1 --input_scale 1 --feature_scale 1
python finetune_feature_scaling.py --opt tp --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --id 001 --input_scale 1 --feature_scale 1
python finetune_feature_scaling.py --opt compare --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --id 001 --input_scale 1 --feature_scale 1

python finetune_feature_scaling.py --opt coco --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --input_scale 1
python finetune_feature_scaling.py --opt scenes100 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --scenes100dir F:\\self_drive_datasets\\Scenes100 --input_scale 1
python finetune_feature_scaling.py --opt bdd --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --bdddir F:\\self_drive_datasets\\BDD100K\\images_100k --input_scale 1
python finetune_feature_scaling.py --opt cityscapes --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cityscapesdir F:\\self_drive_datasets\\CityScapes --input_scale 1
python finetune_feature_scaling.py --opt kitti --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --kittidir F:\\self_drive_datasets\\KITTI --input_scale 1

python finetune_feature_scaling.py --cocodir ../../../MSCOCO2017 --bdddir F:\\self_drive_datasets\\BDD100K\\images_100k --cityscapesdir F:\\self_drive_datasets\\CityScapes --kittidir F:\\self_drive_datasets\\KITTI
'''
