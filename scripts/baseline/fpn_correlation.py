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
import cv2
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
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios, count_parameters
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
# finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_wdiff_midfusion')

from finetune import refine_annotations, get_annotation_dict, finetune_simple_trainer_run_step
# from finetune_wdiff_earlyfusion import construct_image_w_background, DatasetMapperBackground, all_pseudo_manual_annotations_with_background, EvaluationDatasetWithBackground
# from fusion_modules import FeaturePyramidFusionConv, FeaturePyramidFusionAttn


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFPN(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def inference_fp(self, batched_inputs):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert len(batched_inputs) == 1
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        return images.tensor[0], {k: features[k][0] for k in features}

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.__class__ = GeneralizedRCNNFPN
        return net


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorFPN(DefaultPredictor):
    def __init__(self, cfg, args):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNFPN.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            assert self.input_format == 'BGR'
            height, width = image.shape[:2]
            tf = self.aug.get_transform(image)
            image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
            inputs = {'image': image, 'height': height, 'width': width}
            return self.model.inference_fp([inputs])


def show_correlation(args):
    assert args.id in video_id_list
    lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', args.id))
    with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
        meta = json.load(fp)
    ifilelist = meta['ifilelist']
    images = []
    for i in range(0, len(ifilelist)):
        images.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': []})

    indices = np.arange(0, len(images), len(images) // 51)[:50]
    images_exemplar = [images[i] for i in indices]
    print('exemplar frames of video %s at %s: %d images' % (args.id, lmdb_path, len(images_exemplar)))
    images = [images[i] for i in sorted(np.concatenate([indices + s for s in range(0, 10)], axis=0).tolist())]
    print('unlabeled frames of video %s at %s: %d images' % (args.id, lmdb_path, len(images)))

    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorFPN(cfg, args)

    fps_norm = {'p2': [], 'p3': [], 'p4': [], 'p5': [], 'p6': []}
    for _d in tqdm.tqdm(images_exemplar, ascii=True):
        im = detectron2.data.detection_utils.read_image(_d['file_name'], format='BGR')
        with torch.no_grad():
            _, _fp = detector(im)
        for k in _fp:
            fps_norm[k].append(_fp[k])
    for k in fps_norm:
        fps_norm[k] = torch.stack(fps_norm[k], dim=0).mean(dim=0)
        fps_norm[k] /= torch.sqrt(torch.square(fps_norm[k]).sum(dim=0, keepdims=True))
        print(k, fps_norm[k].size(), fps_norm[k].dtype, fps_norm[k].min(), fps_norm[k].max())

    H, W = 720, 1280
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=36)
    writer = skvideo.io.FFmpegWriter('FPN_correlation_%s.mp4' % args.id, inputdict={'-r': '5'}, outputdict={'-vcodec': 'hevc_nvenc', '-r': '5', '-preset': 'medium'})

    for _d in tqdm.tqdm(images, ascii=True):
        im = detectron2.data.detection_utils.read_image(_d['file_name'], format='BGR')
        with torch.no_grad():
            _im_tf, _fp = detector(im)
        _im_tf = _im_tf.detach().cpu().numpy().transpose(1, 2, 0)
        _im_tf -= _im_tf.min()
        _im_tf /= _im_tf.max()
        _im_tf = (_im_tf * 255).astype(np.uint8)[:, :, ::-1]

        _cossin = {}
        with torch.no_grad():
            for k in _fp:
                _fp[k] /= torch.sqrt(torch.square(_fp[k]).sum(dim=0, keepdims=True))
                _cossin[k] = (_fp[k] * fps_norm[k]).sum(dim=0).detach().cpu()

        f1 = np.concatenate([_im_tf, overlay_correlation(_im_tf, _cossin['p2']), overlay_correlation(_im_tf, _cossin['p3'])], axis=1)
        f2 = np.concatenate([overlay_correlation(_im_tf, _cossin['p4']), overlay_correlation(_im_tf, _cossin['p5']), overlay_correlation(_im_tf, _cossin['p6'])], axis=1)
        f = np.concatenate([f1, f2], axis=0)
        f = (skimage.transform.resize(f, (H * 2, W * 3)) * 255.0).astype(np.uint8)

        f = Image.fromarray(f)
        draw = ImageDraw.Draw(f)
        draw.text((2, 2), os.path.basename(os.path.basename(_d['file_name'])), fill='#000000', stroke_width=3, font=font)
        draw.text((2, 2), os.path.basename(os.path.basename(_d['file_name'])), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((W + 2, 2), os.path.basename('P2'), fill='#000000', stroke_width=3, font=font)
        draw.text((W + 2, 2), os.path.basename('P2'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((W * 2 + 2, 2), os.path.basename('P3'), fill='#000000', stroke_width=3, font=font)
        draw.text((W * 2 + 2, 2), os.path.basename('P3'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((2, H + 2), os.path.basename('P4'), fill='#000000', stroke_width=3, font=font)
        draw.text((2, H + 2), os.path.basename('P4'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((W + 2, H + 2), os.path.basename('P5'), fill='#000000', stroke_width=3, font=font)
        draw.text((W + 2, H + 2), os.path.basename('P5'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((W * 2 + 2, H + 2), os.path.basename('P6'), fill='#000000', stroke_width=3, font=font)
        draw.text((W * 2 + 2, H + 2), os.path.basename('P6'), fill='#FFFFFF', stroke_width=1, font=font)
        writer.writeFrame(np.array(f))
    writer.close()


def overlay_correlation(im, corr):
    cm = plt.get_cmap('viridis')
    # corr = torch.nn.functional.interpolate(corr.unsqueeze(0).unsqueeze(0), size=im.shape[:2], mode='bilinear', align_corners=True)[0, 0] # [-1, +1]
    corr = torch.nn.functional.interpolate(corr.unsqueeze(0).unsqueeze(0), size=im.shape[:2], mode='nearest')[0, 0] # [-1, +1]
    corr = ((corr + 1.0) * 0.5).numpy().astype(np.float32) # [0, 1.0]
    heatmap = (cm(corr)[:, :, :3] * 255.0).astype(np.float32)
    im = im.astype(np.float32)
    im_overlay = (0.5 * im + 0.5 * heatmap).astype(np.uint8)
    return im_overlay


def show_correlation_mscoco(args):
    args.cocodir = '../../../MSCOCO2017'
    args.smallscale = False
    images = get_coco_dicts(args, 'train')
    images = [images[i] for i in np.arange(0, len(images), len(images) // 360)]
    for im in images:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'train2017', os.path.basename(im['file_name'])))
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorFPN(cfg, args)

    HW = 720
    def pad_square(im):
        H, W, C = im.shape
        if H > W:
            im = np.concatenate([im, np.zeros(shape=(H, H - W, C), dtype=im.dtype)], axis=1)
        elif H < W:
            im = np.concatenate([im, np.zeros(shape=(W - H, W, C), dtype=im.dtype)], axis=0)
        return (skimage.transform.resize(im, (HW, HW)) * 255.0).astype(np.uint8)

    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=36)
    writer = skvideo.io.FFmpegWriter('FPN_correlation_%s.mp4' % args.id, inputdict={'-r': '2'}, outputdict={'-vcodec': 'hevc_nvenc', '-r': '2', '-preset': 'medium'})
    for _d in tqdm.tqdm(images, ascii=True):
        _cossin = {}
        with torch.no_grad():
            _im_tf, _fp = detector(detectron2.data.detection_utils.read_image(_d['file_name'], format='BGR'))
            _, _fp_bg = detector(detectron2.data.detection_utils.read_image(_d['file_name_background'], format='BGR'))
            for k in _fp:
                _fp[k] /= torch.sqrt(torch.square(_fp[k]).sum(dim=0, keepdims=True))
                _fp_bg[k] /= torch.sqrt(torch.square(_fp_bg[k]).sum(dim=0, keepdims=True))
                _cossin[k] = (_fp[k] * _fp_bg[k]).sum(dim=0).detach().cpu()
        _im_tf = _im_tf.detach().cpu().numpy().transpose(1, 2, 0)
        _im_tf -= _im_tf.min()
        _im_tf /= _im_tf.max()
        _im_tf = (_im_tf * 255).astype(np.uint8)[:, :, ::-1]

        frame = {k: pad_square(overlay_correlation(_im_tf, 1.0 - _cossin[k])) for k in _cossin}
        frame['im'] = pad_square(_im_tf)
        f1 = np.concatenate([frame['im'], frame['p2'], frame['p3']], axis=1)
        f2 = np.concatenate([frame['p4'], frame['p5'], frame['p6']], axis=1)
        f = np.concatenate([f1, f2], axis=0)

        f = Image.fromarray(f)
        draw = ImageDraw.Draw(f)
        draw.text((2, 2), os.path.basename(os.path.basename(_d['file_name'])), fill='#000000', stroke_width=3, font=font)
        draw.text((2, 2), os.path.basename(os.path.basename(_d['file_name'])), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((HW + 2, 2), os.path.basename('P2'), fill='#000000', stroke_width=3, font=font)
        draw.text((HW + 2, 2), os.path.basename('P2'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((HW * 2 + 2, 2), os.path.basename('P3'), fill='#000000', stroke_width=3, font=font)
        draw.text((HW * 2 + 2, 2), os.path.basename('P3'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((2, HW + 2), os.path.basename('P4'), fill='#000000', stroke_width=3, font=font)
        draw.text((2, HW + 2), os.path.basename('P4'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((HW + 2, HW + 2), os.path.basename('P5'), fill='#000000', stroke_width=3, font=font)
        draw.text((HW + 2, HW + 2), os.path.basename('P5'), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((HW * 2 + 2, HW + 2), os.path.basename('P6'), fill='#000000', stroke_width=3, font=font)
        draw.text((HW * 2 + 2, HW + 2), os.path.basename('P6'), fill='#FFFFFF', stroke_width=1, font=font)
        writer.writeFrame(np.array(f))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'coco'], help='video ID')
    parser.add_argument('--model', type=str, default='r101-fpn-3x', help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--eval_background', type=str, default='', choices=['', 'dynamic', 'last', 'average'], help='use inference time dynamic background or last training time background')
    args = parser.parse_args()
    print(args)

    if args.id == 'coco':
        show_correlation_mscoco(args)
    else:
        show_correlation(args)

'''
python fpn_correlation.py --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --id 001
'''
