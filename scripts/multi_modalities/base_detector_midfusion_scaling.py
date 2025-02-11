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


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


class AdaptationPredictor(DefaultPredictor):
    def __init__(self, cfg, args):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        if args.fusion == 'early':
            self.model = GeneralizedRCNNEarlyFusion.create_from_sup(self.model)
        elif args.fusion == 'mid':
            self.model = GeneralizedRCNNMidFusion.create_from_sup(self.model, None)
        elif args.fusion == 'late':
            self.model = GeneralizedRCNNLateFusion.create_from_sup(self.model, None)
        else:
            raise NotImplementedError
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, image, image_diff):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            assert self.input_format == 'BGR'
            height, width = image.shape[:2]
            tf = self.aug.get_transform(image)
            image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
            image_diff = torch.as_tensor(tf.apply_image(image_diff).astype('float32').transpose(2, 0, 1))
            inputs = {'image': torch.cat([image, image_diff], dim=0), 'height': height, 'width': width}
            return self.model.inference([inputs])


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dicts, image_format):
        super(EvaluationDataset, self).__init__()
        self.image_dicts = image_dicts
        self.image_format = image_format
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name'], format=self.image_format)
        if 'file_name_background' in self.image_dicts[i]:
            image_bg = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name_background'], format=self.image_format)
            image, _, image_diff = construct_image_w_background(image, image_bg)
        else:
            image_diff = None
        return {'dict': copy.deepcopy(self.image_dicts[i]), 'image': image, 'image_diff': image_diff}
    @staticmethod
    def collate(batch):
        return batch


def evaluate(args):
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

    loader = torch.utils.data.DataLoader(
        EvaluationDataset(images, cfg.INPUT.FORMAT),
        batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=args.num_workers
    )
    if args.scale > 0.99:
        args.scale = round(args.scale)
    detections = []

    if args.scale == 1:
        for batch in tqdm.tqdm(loader, total=len(images), ascii=True):
            im = batch['dict']
            im['file_name'] = os.path.basename(im['file_name'])
            im['annotations'] = []
            instances = detector(batch['image'], batch['image_diff'])[0]['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor
            score = instances.scores
            label = instances.pred_classes
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
            detections.append(im)
    elif args.scale > 1:
        print('split image by %d x %d' % (args.scale, args.scale))
        for batch in tqdm.tqdm(loader, total=len(images), ascii=True):
            im = batch['dict']
            im['file_name'] = os.path.basename(im['file_name'])
            im['annotations'] = []
            assert batch['image'].dtype == np.uint8
            W0, W1, _ = batch['image'].shape
            image_scale, image_diff_scale = map(lambda _im: (skimage.transform.resize(_im, (W0 * args.scale, W1 * args.scale)) * 255.0).astype(np.uint8), [batch['image'], batch['image_diff']])
            for patch_i in range(0, args.scale):
                for patch_j in range(0, args.scale):
                    image_patch = image_scale[patch_i * W0 : (patch_i + 1) * W0, patch_j * W1 : (patch_j + 1) * W1, :]
                    image_diff_patch = image_diff_scale[patch_i * W0 : (patch_i + 1) * W0, patch_j * W1 : (patch_j + 1) * W1, :]
                    instances = detector(image_patch, image_diff_patch)[0]['instances'].to('cpu')
                    # bbox has format [x1, y1, x2, y2]
                    bbox = instances.pred_boxes.tensor
                    bbox = bbox / args.scale
                    bbox[:, 0] += patch_j * W1 / args.scale
                    bbox[:, 2] += patch_j * W1 / args.scale
                    bbox[:, 1] += patch_i * W0 / args.scale
                    bbox[:, 3] += patch_i * W0 / args.scale
                    score = instances.scores
                    label = instances.pred_classes
                    for i in range(0, len(label)):
                        im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
            detections.append(im)
            # import matplotlib.patches as patches
            # _, ax = plt.subplots()
            # ax.imshow(batch['image'])
            # for ann in detections[-1]['annotations']:
            #     (x1, y1, x2, y2), k = ann['bbox'], ann['category_id']
            #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
            #     ax.add_patch(rect)
            # plt.tight_layout()
            # plt.show()
    else:
        print('downsample image by %.2f x %.2f' % (args.scale, args.scale))
        for batch in tqdm.tqdm(loader, total=len(images), ascii=True):
            im = batch['dict']
            im['file_name'] = os.path.basename(im['file_name'])
            im['annotations'] = []
            assert batch['image'].dtype == np.uint8
            W0, W1, _ = batch['image'].shape
            image_scale, image_diff_scale = map(lambda _im: (skimage.transform.resize(_im, (int(W0 * args.scale), int(W1 * args.scale))) * 255.0).astype(np.uint8), [batch['image'], batch['image_diff']])
            image_pad, image_diff_pad = np.zeros_like(batch['image']), np.zeros_like(batch['image_diff'])
            image_pad[: image_scale.shape[0], : image_scale.shape[1], :] = image_scale
            image_diff_pad[: image_diff_scale.shape[0], : image_diff_scale.shape[1], :] = image_diff_scale
            instances = detector(image_pad, image_diff_pad)[0]['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor / args.scale
            score = instances.scores
            label = instances.pred_classes
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
            detections.append(im)

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results = evaluate_masked(args.id, detections)
    print('             %s' % '/'.join(results['metrics']))
    for c in sorted(results['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))
    return vars(args), results, detections


def eval_all(args):
    for scale in [0.5, 0.75, 0.9, 1, 2, 3, 4]:
        results_AP_scale = {}
        t0 = time.time()
        for i, video_id in enumerate(video_id_list):
            print('\nvideo', video_id)
            args.id, args.scale = video_id, scale
            results_AP_scale[video_id] = {}
            _, results_AP_scale[video_id]['results'], results_AP_scale[video_id]['detections'] = evaluate(args)
            del results_AP_scale[video_id]['results']['raw']
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
        with open('midfusion_base_scale%.2f.json' % scale, 'w') as fp:
            json.dump(results_AP_scale, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, choices=['single', 'eval_all'])
    parser.add_argument('--id', type=str, choices=video_id_list)
    parser.add_argument('--fusion', type=str, choices=['early', 'mid', 'late'])
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--scale', default=1, type=float)
    args = parser.parse_args()

    print(args)
    if args.opt == 'single':
        evaluate(args)
    elif args.opt == 'eval_all':
        eval_all(args)


'''
python base_detector_midfusion_scaling.py --model r101-fpn-3x --fusion mid --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --id 001 --scale 2 --opt single
python base_detector_midfusion_scaling.py --model r101-fpn-3x --fusion mid --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --num_workers 1 --opt eval_all
'''
