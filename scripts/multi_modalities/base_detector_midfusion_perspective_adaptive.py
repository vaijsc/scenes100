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
import cv2

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
import contextlib
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from evaluation import evaluate_masked, evaluate_cocovalid

from fusion_models import GeneralizedRCNNEarlyFusion, GeneralizedRCNNMidFusion, GeneralizedRCNNLateFusion
from base_detector_fusion_mixup import construct_image_w_background
from base_detector_midfusion_scaling import AdaptationPredictor


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


class EvaluationDatasetPerspective(torch.utils.data.Dataset):
    def __init__(self, image_dicts, image_format, M_warp, w_warp, h_warp, scale_split):
        super(EvaluationDatasetPerspective, self).__init__()
        self.image_dicts = image_dicts
        self.image_format = image_format
        self.M_warp, self.w_warp, self.h_warp, self.scale_split = M_warp, w_warp, h_warp, scale_split
        print('warp image to %d x %d, upscale by %d' % (self.w_warp, self.h_warp, self.scale_split))
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name'], format=self.image_format)
        assert 'file_name_background' in self.image_dicts[i]
        image_bg = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name_background'], format=self.image_format)
        image, _, image_diff = construct_image_w_background(image, image_bg)
        image_warped = cv2.warpPerspective(image, self.M_warp, (self.w_warp, self.h_warp))
        image_bg_warped = cv2.warpPerspective(image_bg, self.M_warp, (self.w_warp, self.h_warp))
        image_warped, _, image_diff_warped = construct_image_w_background(image_warped, image_bg_warped)
        if self.scale_split == 1:
            image_warped_scale, image_diff_warped_scale = None, None
        if self.scale_split > 1:
            assert image_warped.dtype == np.uint8
            W0, W1, _ = image_warped.shape
            image_warped_scale, image_diff_warped_scale = map(lambda _im: (skimage.transform.resize(_im, (W0 * self.scale_split, W1 * self.scale_split)) * 255.0).astype(np.uint8), [image_warped, image_diff_warped])
        return {'dict': copy.deepcopy(self.image_dicts[i]), 'image': image, 'image_diff': image_diff, 'image_warped': image_warped, 'image_diff_warped': image_diff_warped, 'image_warped_scale': image_warped_scale, 'image_diff_warped_scale': image_diff_warped_scale}
    @staticmethod
    def collate(batch):
        return batch


def evaluate(args):
    assert args.scale_w > 0.99 and args.scale_h > 0.99
    inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id))
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    background_file = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))[-1]
    for im in images:
        im['file_name'] = os.path.join(inputdir, 'unmasked', im['file_name'])
        im['file_name_background'] = background_file

    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    cfg.MODEL.WEIGHTS = args.ckpt
    detector = AdaptationPredictor(cfg, args)

    w, h = images[0]['width'], images[0]['height']
    for im in images:
        assert w == images[0]['width'] and h == images[0]['height']
    pt_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pt_warped = np.float32([[0, 0], [w * args.scale_w, 0], [w * (args.scale_w + 1) / 2, h * args.scale_h], [w * (args.scale_w - 1) / 2, h * args.scale_h]])
    M_warp = cv2.getPerspectiveTransform(pt_corners, pt_warped)
    M_unwarp = cv2.getPerspectiveTransform(pt_warped, pt_corners)
    scale_split = round(max(args.scale_w, args.scale_h))

    loader = torch.utils.data.DataLoader(
        EvaluationDatasetPerspective(images, cfg.INPUT.FORMAT, M_warp, int(w * args.scale_w), int(h * args.scale_h), scale_split),
        batch_size=None, collate_fn=EvaluationDatasetPerspective.collate, shuffle=False, num_workers=args.num_workers
    )
    detections = []

    for batch in tqdm.tqdm(loader, total=len(images), ascii=True):
        im = batch['dict']
        im['file_name'] = os.path.basename(im['file_name'])
        im['annotations'], im['annotations_warped'] = [], []
        if scale_split == 1:
            instances = detector(batch['image_warped'], batch['image_diff_warped'])[0]['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor
            score = instances.scores
            label = instances.pred_classes
            for i in range(0, len(label)):
                im['annotations_warped'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
        else:
            W0, W1, _ = batch['image_warped'].shape
            for patch_i in range(0, scale_split):
                for patch_j in range(0, scale_split):
                    image_warped_patch = batch['image_warped_scale'][patch_i * W0 : (patch_i + 1) * W0, patch_j * W1 : (patch_j + 1) * W1, :]
                    image_diff_warped_patch = batch['image_diff_warped_scale'][patch_i * W0 : (patch_i + 1) * W0, patch_j * W1 : (patch_j + 1) * W1, :]
                    instances = detector(image_warped_patch, image_diff_warped_patch)[0]['instances'].to('cpu')
                    # bbox has format [x1, y1, x2, y2]
                    bbox = instances.pred_boxes.tensor
                    bbox = bbox / scale_split
                    bbox[:, 0] += patch_j * W1 / scale_split
                    bbox[:, 2] += patch_j * W1 / scale_split
                    bbox[:, 1] += patch_i * W0 / scale_split
                    bbox[:, 3] += patch_i * W0 / scale_split
                    score = instances.scores
                    label = instances.pred_classes
                    for i in range(0, len(label)):
                        im['annotations_warped'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
        for ann in im['annotations_warped']:
            x1, y1, x2, y2 = ann['bbox']
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            pts = np.float32([[[x1, yc], [xc, y1], [x2, yc], [xc, y2]]])
            pts_unwarped = cv2.perspectiveTransform(pts, M_unwarp)[0]
            im['annotations'].append(copy.deepcopy(ann))
            im['annotations'][-1]['bbox'] = list(map(float, [max(0, pts_unwarped[0, 0]), max(0, pts_unwarped[1, 1]), min(pts_unwarped[2, 0], im['width']), min(pts_unwarped[3, 1], im['height'])]))
        detections.append(im)
        # import matplotlib.patches as patches
        # _, ax = plt.subplots(1, 2); ax = ax.reshape(-1)
        # ax[0].imshow(batch['image_warped'])
        # ax[0].set_axis_off()
        # for ann in detections[-1]['annotations_warped']:
        #     (x1, y1, x2, y2), k = ann['bbox'], ann['category_id']
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
        #     ax[0].add_patch(rect)
        # ax[1].imshow(batch['image'])
        # ax[1].set_axis_off()
        # for ann in detections[-1]['annotations']:
        #     (x1, y1, x2, y2), k = ann['bbox'], ann['category_id']
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
        #     ax[1].add_patch(rect)
        # plt.tight_layout()
        # plt.show()
        # raise NotImplementedError

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = evaluate_masked(args.id, detections, check_empty=True)
    print('             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))
    return vars(args), results, detections


def eval_all(args):
    for scale_w, scale_h in [(1, 1), (2, 2), (2, 1.5), (2, 1), (3, 3), (3, 2)]:
        results_AP_perspective = {}
        t0 = time.time()
        for i, video_id in enumerate(video_id_list):
            print('\nvideo', video_id)
            args.id, args.scale_w, args.scale_h = video_id, scale_w, scale_h
            results_AP_perspective[video_id] = {}
            _, results_AP_perspective[video_id]['results'], results_AP_perspective[video_id]['detections'] = evaluate(args)
            del results_AP_perspective[video_id]['results']['raw']
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
        with open('midfusion_base_perspective_%.1f_%.1f.json' % (scale_w, scale_h), 'w') as fp:
            json.dump(results_AP_perspective, fp)


def oracle_perspective(args):
    from base_detector_midfusion_scaling_adaptive import _compare
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    scales = [(1, 1), (2, 2), (2, 1.5), (2, 1), (3, 3), (3, 2)]
    results_AP_perspective_scales, detections_perspective_scales = [], []
    for scale_w, scale_h in scales:
        print('reading scale %.1f %.1f' % (scale_w, scale_h))
        with open(os.path.join('results', 'midfusion_base_perspective_%.1f_%.1f.json' % (scale_w, scale_h)), 'r') as fp:
            data = json.load(fp)
            results_AP_perspective_scales.append({video_id: data[video_id]['results'] for video_id in data})
            detections_perspective_scales.append({video_id: data[video_id]['detections'] for video_id in data})
            _compare(base_AP, results_AP_perspective_scales[-1], 'uniform_perspective_%.1f_%.1f' % (scale_w, scale_h))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, choices=['single', 'eval_all', 'oracle', 'adaptive'])
    parser.add_argument('--id', type=str, choices=video_id_list)
    parser.add_argument('--fusion', type=str, choices=['early', 'mid', 'late'])
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--scale_w', default=2, type=float)
    parser.add_argument('--scale_h', default=1.5, type=float)
    args = parser.parse_args()

    print(args)
    if args.opt == 'single':
        evaluate(args)
    elif args.opt == 'eval_all':
        eval_all(args)
    elif args.opt == 'oracle':
        oracle_perspective(args)


'''
python base_detector_midfusion_perspective_adaptive.py --model r101-fpn-3x --fusion mid --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --id 001 --opt single
python base_detector_midfusion_perspective_adaptive.py --model r101-fpn-3x --fusion mid --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --num_workers 3 --opt eval_all
'''
