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

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref
import contextlib
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from fusion_models import GeneralizedRCNNEarlyFusion, GeneralizedRCNNMidFusion, GeneralizedRCNNLateFusion
from base_detector_fusion_mixup import construct_image_w_background
from evaluation import evaluate_masked, evaluate_cocovalid
from base_detector_midfusion_scaling import AdaptationPredictor, EvaluationDataset


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def _compare(base_AP, APs, prefix):
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    videos = sorted(list(APs.keys()))
    for vid in videos:
        AP1 = base_AP['manual_' + vid]['results']
        AP2 = APs[vid]['results']
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
        axes[i].set_ylabel('AP improvement (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    plt.suptitle(prefix)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(prefix + '.pdf')
    plt.close()


def scale_uniform(args):
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    for scale in [0.5, 0.75, 0.9, 1, 2, 3, 4]:
        print('comparing scale %.2f' % scale)
        with open(os.path.join('results', 'midfusion_base_scale%.2f.json' % scale), 'r') as fp:
            results_AP_scale = json.load(fp)
        results_AP_scale = {k: results_AP_scale[k]['results'] for k in results_AP_scale}
        _compare(base_AP, results_AP_scale, 'uniform_scale%.2f' % scale)


def scale_oracle(args):
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    results_AP_scale_list = {}
    for scale in [0.5, 0.75, 0.9, 1, 2, 3, 4]:
        with open(os.path.join('results', 'midfusion_base_scale%.2f.json' % scale), 'r') as fp:
            results_AP_scale_list[scale] = json.load(fp)
    results_AP_best = {}
    for video_id in video_id_list:
        _aps = {s: results_AP_scale_list[s][video_id]['results'] for s in results_AP_scale_list}
        _best_s = max(_aps, key=lambda x: _aps[x]['results']['weighted'][0])
        results_AP_best[video_id] = _aps[_best_s]
        print('%s,%s' % (video_id, _best_s))
    _compare(base_AP, results_AP_best, 'oracle_scale')


def scale_nms(args):
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    results_AP_scale_list = {}
    for scale in [0.9, 1, 2, 3]:
        with open(os.path.join('results', 'midfusion_base_scale%.2f.json' % scale), 'r') as fp:
            results_AP_scale_list[scale] = json.load(fp)
    results_AP_nms = {}
    for video_id in video_id_list:
        _detections = [results_AP_scale_list[s][video_id]['detections'] for s in results_AP_scale_list]
        for _d in _detections:
            assert len(_d) == len(_detections[0])
            for i in range(0, len(_d)):
                assert _d[i]['file_name'] == _detections[0][i]['file_name']
        _detections_nms = copy.deepcopy(_detections[0])
        count_before, count_after = 0, 0
        for i in range(0, len(_detections_nms)):
            _detections_nms[i]['annotations'] = []
            for j in range(0, len(_detections)):
                _detections_nms[i]['annotations'] = _detections_nms[i]['annotations'] + _detections[j][i]['annotations']
            for ann in _detections_nms[i]['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            count_before += len(_detections_nms[i]['annotations'])
            _boxes = torch.tensor([ann['bbox'] for ann in _detections_nms[i]['annotations']]).float()
            _scores = torch.tensor([ann['score'] for ann in _detections_nms[i]['annotations']]).float()
            _cats = torch.tensor([ann['category_id'] for ann in _detections_nms[i]['annotations']]).long()
            _idxs_nms = torchvision.ops.batched_nms(_boxes, _scores, _cats, 0.4)
            count_after += _idxs_nms.size(0)
            _detections_nms[i]['annotations'] = []
            for j in _idxs_nms:
                _detections_nms[i]['annotations'].append({'bbox': list(map(float, _boxes[j])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(_cats[j]), 'score': float(_scores[j])})
        print(video_id, count_before, '->', count_after, round(count_after / count_before, 4))
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_AP_nms[video_id] = evaluate_masked(video_id, _detections_nms)
    _compare(base_AP, results_AP_nms, 'nms_scales')


def scale_imagewise(args):
    assert args.a3 <= args.a2
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    with open(os.path.join('results', 'midfusion_base_scale1.00.json'), 'r') as fp:
        results_AP_scale1 = json.load(fp)
    with open(os.path.join('results', 'midfusion_base_scale2.00.json'), 'r') as fp:
        results_AP_scale2 = json.load(fp)
    with open(os.path.join('results', 'midfusion_base_scale3.00.json'), 'r') as fp:
        results_AP_scale3 = json.load(fp)

    results_AP_adaptive = {}
    for video_id in tqdm.tqdm(video_id_list, ascii=True):
        bbox_areas = []
        for im in results_AP_scale1[video_id]['detections']:
            for ann in im['annotations']:
                if ann['score'] > 0.75:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    x1, y1, x2, y2 = ann['bbox']
                    bbox_areas.append((x2 - x1) * (y2 - y1))
        if len(bbox_areas) < 10:
            results_AP_adaptive[video_id] = results_AP_scale1[video_id]['results']
        else:
            bbox_areas = np.array(bbox_areas).mean()
            if bbox_areas < args.a3:
                results_AP_adaptive[video_id] = results_AP_scale3[video_id]['results']
            elif bbox_areas < args.a2:
                results_AP_adaptive[video_id] = results_AP_scale2[video_id]['results']
            else:
                results_AP_adaptive[video_id] = results_AP_scale1[video_id]['results']
    _compare(base_AP, results_AP_adaptive, 'imagewise_adaptive_scale_%d_%d' % (args.a3, args.a2))


def scale_patchwise(args):
    args.patch_scale = int(args.patch_scale)
    assert args.patch_scale in [2, 3]
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    with open(os.path.join('results', 'midfusion_base_scale1.00.json'), 'r') as fp:
        results_AP_scale1 = json.load(fp)
    with open(os.path.join('results', 'midfusion_base_scale%.2f.json' % args.patch_scale), 'r') as fp:
        results_AP_scale_up = json.load(fp)

    results_AP_adaptive = {}
    t0 = time.time()
    for video_id in video_id_list:
        for dets in [results_AP_scale1[video_id]['detections'], results_AP_scale_up[video_id]['detections']]:
            for im in dets:
                im['annotations_patch'] = []
                for patch_i in range(0, args.patch_scale):
                    im['annotations_patch'].append([])
                    for patch_j in range(0, args.patch_scale):
                        im['annotations_patch'][patch_i].append([])
                for ann in im['annotations']:
                    x1, y1, x2, y2 = ann['bbox']
                    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                    im['annotations_patch'][math.floor(xc / im['width'] * args.patch_scale)][math.floor(yc / im['height'] * args.patch_scale)].append(ann)
        bbox_areas = []
        for patch_i in range(0, args.patch_scale):
            bbox_areas.append([])
            for patch_j in range(0, args.patch_scale):
                bbox_areas[patch_i].append([])
        for im in results_AP_scale1[video_id]['detections']:
            for patch_i in range(0, args.patch_scale):
                for patch_j in range(0, args.patch_scale):
                    for ann in im['annotations_patch'][patch_i][patch_j]:
                        if ann['score'] > 0.75:
                            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                            x1, y1, x2, y2 = ann['bbox']
                            bbox_areas[patch_i][patch_j].append((x2 - x1) * (y2 - y1))
        upscale_ij = []
        for patch_i in range(0, args.patch_scale):
            for patch_j in range(0, args.patch_scale):
                if len(bbox_areas[patch_i][patch_j]) > 5:
                    if np.array(bbox_areas[patch_i][patch_j]).mean() < args.patch_a:
                        upscale_ij.append([patch_i, patch_j])
        print(video_id, upscale_ij)

        detections_adaptive = copy.deepcopy(results_AP_scale1[video_id]['detections'])
        for im, im_scale_up in zip(detections_adaptive, results_AP_scale_up[video_id]['detections']):
            for patch_i, patch_j in upscale_ij:
                im['annotations_patch'][patch_i][patch_j] = im_scale_up['annotations_patch'][patch_i][patch_j]
            im['annotations'] = []
            for patch_i in range(0, args.patch_scale):
                for patch_j in range(0, args.patch_scale):
                    im['annotations'] = im['annotations'] + im['annotations_patch'][patch_i][patch_j]
            del im['annotations_patch']
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_AP_adaptive[video_id] = evaluate_masked(video_id, detections_adaptive)
    _compare(base_AP, results_AP_adaptive, 'patchwise_adaptive_scale_%d_a%d' % (args.patch_scale, args.patch_a))


def scale_estimated_quality(args):
    from bbox_quality_estimator import IoUEstimator
    estimator = IoUEstimator()
    estimator.load_state_dict(torch.load(args.ckpt_estimator))
    estimator = estimator.cuda()
    estimator.eval()
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((160, 160), antialias=True),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    results_AP_scale_list = {}
    for scale in [0.5, 0.75, 0.9, 1, 2, 3, 4]:
        with open(os.path.join('results', 'midfusion_base_scale%.2f.json' % scale), 'r') as fp:
            results_AP_scale_list[scale] = json.load(fp)

    results_AP_adaptive, scale_adaptive, upscale_for_crop = {}, {}, 1
    for video_id in video_id_list:
        _detections_scale_list = {s: results_AP_scale_list[s][video_id]['detections'] for s in results_AP_scale_list}
        for i in range(0, len(_detections_scale_list[0.5])):
            assert len(set([_detections_scale_list[s][i]['file_name'] for s in _detections_scale_list])) == 1 # check everything matches
        # FasterRCNN scores & quality estimator scores
        _confidence_scale_list, _quality_scale_list = {s: [] for s in _detections_scale_list}, {s: [] for s in _detections_scale_list}
        for i in tqdm.tqdm(range(0, len(_detections_scale_list[0.5])), ascii=True, desc=video_id):
            file_name = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id, 'unmasked', _detections_scale_list[0.5][i]['file_name'])
            im_arr = skimage.io.imread(file_name).astype(np.float32) / 255.0
            im_arr = skimage.transform.resize(im_arr, (im_arr.shape[0] * upscale_for_crop, im_arr.shape[1] * upscale_for_crop))
            for s in _detections_scale_list:
                patches_tensor, patches_scores = [], []
                for ann in _detections_scale_list[s][i]['annotations']:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    x1, y1, x2, y2 = ann['bbox']
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    x1, y1, x2, y2 = map(lambda x: int(x * upscale_for_crop), [x1, y1, x2, y2])
                    patches_tensor.append(tf(torch.from_numpy(im_arr[y1 : y2, x1 : x2, :].transpose(2, 0, 1)).float()))
                    patches_scores.append(ann['score'])
                patches_tensor = torch.stack(patches_tensor, dim=0).cuda()
                _confidence_scale_list[s] = _confidence_scale_list[s] + patches_scores
                with torch.no_grad():
                    _quality_scale_list[s] = _quality_scale_list[s] + estimator(patches_tensor).flatten().cpu().numpy().tolist()
                del patches_scores
        _best_s = {}
        for s in _detections_scale_list:
            _quality_scale_list[s] = np.array(_quality_scale_list[s])
            _confidence_scale_list[s] = np.array(_confidence_scale_list[s])
            _best_s[s] = (_quality_scale_list[s] * _confidence_scale_list[s]).sum() / _confidence_scale_list[s].sum()
            print('%s: %.4f' % (s, _best_s[s]), end=' ')
        _best_s = max(_best_s, key=lambda x: _best_s[x])
        scale_adaptive[video_id] = _best_s
        results_AP_adaptive[video_id] = results_AP_scale_list[_best_s][video_id]['results']
        print('best:', _best_s)
    _compare(base_AP, results_AP_adaptive, 'estimated_quality_adaptive_scale')
    for video_id in scale_adaptive:
        print('%s,%s' % (video_id, scale_adaptive[video_id]))


def scale_estimated_scale(args):
    from optimal_scale_estimator import ScaleEstimator
    estimator = ScaleEstimator()
    estimator.load_state_dict(torch.load(args.ckpt_estimator))
    estimator = estimator.cuda()
    estimator.eval()
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), antialias=True),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scale_targets = np.array(ScaleEstimator.scale_targets).astype(np.float32)
    scales = np.array(ScaleEstimator.scales).astype(np.float32)

    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    results_AP_scale_list = {}
    for s in scales:
        with open(os.path.join('results', 'midfusion_base_scale%.2f.json' % s), 'r') as fp:
            results_AP_scale_list[s] = json.load(fp)

    results_AP_adaptive, scale_adaptive = {}, {}
    for video_id in video_id_list:
        _detections_scale_list = {s: results_AP_scale_list[s][video_id]['detections'] for s in results_AP_scale_list}
        for i in range(0, len(_detections_scale_list[1])):
            assert len(set([_detections_scale_list[s][i]['file_name'] for s in _detections_scale_list])) == 1 # check everything matches
        estimated_scale_targets = []
        for i in tqdm.tqdm(range(0, len(_detections_scale_list[0.5])), ascii=True, desc=video_id):
            file_name = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id, 'unmasked', _detections_scale_list[0.5][i]['file_name'])
            im_arr = skimage.io.imread(file_name).transpose(2, 0, 1).astype(np.float32) / 255.0
            im_tensor = torch.stack([tf(torch.from_numpy(im_arr))], dim=0)
            with torch.no_grad():
                estimated_scale_targets.append(estimator(im_tensor.cuda()).detach().cpu().numpy()[0, 0])
        estimated_scale_targets = np.array(estimated_scale_targets).astype(np.float32)
        estimated_scale_i = np.absolute(estimated_scale_targets.mean() - scale_targets).argmin()
        print('estimated scale targets: min = %.4f, max = %.4f, mean = %.4f, std = %.4f' % (estimated_scale_targets.min(), estimated_scale_targets.max(), estimated_scale_targets.mean(), estimated_scale_targets.std()))
        print('select scale x %.2f' % scales[estimated_scale_i])
        scale_adaptive[video_id] = scales[estimated_scale_i]
        results_AP_adaptive[video_id] = results_AP_scale_list[scales[estimated_scale_i]][video_id]['results']
    _compare(base_AP, results_AP_adaptive, 'estimated_scale_adaptive_scale')
    for video_id in scale_adaptive:
        print('%s,%s' % (video_id, scale_adaptive[video_id]))


if __name__ == '__main__':
    # import csv
    # with open('optimal_scale.csv', 'r') as fp:
    #     lines = list(csv.reader(fp, delimiter='\t'))[1:]
    # print(lines)
    # xs, xlabel = np.array([float(l[1]) for l in lines]), 'optimal scale'
    # ys, ylabel = np.array([float(l[3]) for l in lines]), 'estimated scale'
    # xs, ys = xs + np.random.rand(xs.shape[0]) * 0.01, ys + np.random.rand(ys.shape[0]) * 0.01
    # plt.figure(figsize=(6, 6))
    # plt.scatter(xs, ys, marker='+', alpha=0.5)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.tight_layout()
    # plt.savefig('compare.pdf')
    # exit()

    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, choices=['uniform', 'oracle', 'imagewise', 'patchwise', 'nms', 'estimated_quality', 'estimated_scale'])
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--a2', default=5000, type=int)
    parser.add_argument('--a3', default=2000, type=int)
    parser.add_argument('--patch_scale', default=2, type=float)
    parser.add_argument('--patch_a', default=5000, type=int)
    parser.add_argument('--ckpt_estimator', type=str)
    args = parser.parse_args()

    print(args)
    if args.opt == 'uniform':
        scale_uniform(args)
    elif args.opt == 'oracle':
        scale_oracle(args)
    elif args.opt == 'imagewise':
        scale_imagewise(args)
    elif args.opt == 'patchwise':
        scale_patchwise(args)
    elif args.opt == 'nms':
        scale_nms(args)
    elif args.opt == 'estimated_quality':
        scale_estimated_quality(args)
    elif args.opt == 'estimated_scale':
        scale_estimated_scale(args)
