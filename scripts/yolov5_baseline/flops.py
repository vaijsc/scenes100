#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import math
import random
import tqdm
import glob
import psutil
import contextlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple

import torch
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model

from calflops import calculate_flops


def read_image(cfg, video_id='001', read_background=False):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        f = json.load(fp)[0]['file_name']
    image_np = detectron2.data.detection_utils.read_image(os.path.normpath(os.path.join(inputdir, 'unmasked', f)), format=cfg.INPUT.FORMAT)
    height, width = image_np.shape[:2]
    aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    tf = aug.get_transform(image_np)
    image = torch.as_tensor(tf.apply_image(image_np).astype('float32').transpose(2, 0, 1))
    if read_background:
        from finetune_wdiff_midfusion import construct_image_w_background
        background_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', video_id, 'inpaint', '*inpaint.jpg')))
        image_bg = detectron2.data.detection_utils.read_image(background_files[-1], format=cfg.INPUT.FORMAT)
        _, _, image_diff = construct_image_w_background(image_np, image_bg)
        image_diff = torch.as_tensor(tf.apply_image(image_diff).astype('float32').transpose(2, 0, 1))
        return image, image_diff, height, width, video_id
    else:
        return image, height, width, video_id


def measure_faster_rcnn(model, scale=1):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models import get_cfg_base_model
    from inference_server_simulate import GeneralizedRCNNServer

    class FasterRCNNInferenceWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            net.eval()
            self.net = net
        def forward(self, image, width, height, video_id):
            batched_inputs = [{'image': image, 'height': int(height.item()), 'width': int(width.item()), 'video_id': '%03d' % video_id.item()}]
            return self.net.inference(batched_inputs, do_postprocess=True)

    cfg = get_cfg_base_model(model)
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = int(scale * cfg.INPUT.MIN_SIZE_TEST), int(scale * cfg.INPUT.MAX_SIZE_TEST)
    net = DefaultPredictor(cfg).model
    net.load_state_dict(torch.load('../../models/mscoco2017_remap_%s.pth' % model))
    image, height, width, video_id = read_image(cfg)
    print(image.size(), scale, height, width, video_id)

    wrapper = FasterRCNNInferenceWrapper(net)
    args = {'image': image, 'width': torch.tensor(width), 'height': torch.tensor(height), 'video_id': torch.tensor(int(video_id))}
    flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
    print('vanilla', flops, macs, params)
    del wrapper

    for B in [10, 100]:
        net_moe = GeneralizedRCNNServer.create_from_sup(copy.deepcopy(net), B)
        wrapper = FasterRCNNInferenceWrapper(net_moe)
        flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
        print('MoE B =', B, flops, macs, params)
        del wrapper


def measure_yolov5(scale=1):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models import get_cfg_base_model
    from yolov5 import load_yolov5
    from inference_server_simulate_yolov5_all import YOLOServer

    class YOLOInferenceWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            net.eval()
            self.net = net
        def forward(self, image, width, height, video_id):
            batched_inputs = [{'image': image, 'height': int(height.item()), 'width': int(width.item()), 'video_id': '%03d' % video_id.item()}]
            return self.net.inference(batched_inputs)

    cfg = get_cfg_base_model('r101-fpn-3x')
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = int(scale * cfg.INPUT.MIN_SIZE_TEST), int(scale * cfg.INPUT.MAX_SIZE_TEST)
    net = load_yolov5('../../configs/yolov5s.yaml')
    # net.load_state_dict(torch.load('../../models/mscoco2017_remap_%s.pth' % model))
    image, height, width, video_id = read_image(cfg)
    print(image.size(), scale, height, width, video_id)

    wrapper = YOLOInferenceWrapper(net)
    args = {'image': image, 'width': torch.tensor(width), 'height': torch.tensor(height), 'video_id': torch.tensor(int(video_id))}
    flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
    print('vanilla', flops, macs, params)
    del wrapper

    for B in [10, 100]:
        net_moe = YOLOServer.create_from_sup(copy.deepcopy(net), B, [0, 1, 2, 3, 4, -1])
        wrapper = YOLOInferenceWrapper(net_moe)
        flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
        print('MoE B =', B, flops, macs, params)
        del wrapper



def measure_midfusion(model):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models import get_cfg_base_model
    from finetune_wdiff_midfusion import GeneralizedRCNNFinetuneBackground

    class MidfusionInferenceWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            net.eval()
            self.net = net
        def forward(self, image, width, height):
            batched_inputs = [{'image': image, 'height': int(height.item()), 'width': int(width.item())}]
            return self.net.inference(batched_inputs, do_postprocess=True)

    cfg = get_cfg_base_model(model)
    net = detectron2.modeling.build_model(cfg)
    net = GeneralizedRCNNFinetuneBackground.create_from_sup(net, 'average', None)
    net.load_state_dict(torch.load('../../models/mscoco2017_remap_wdiff_midfusion_%s.pth' % model))
    image, image_diff, height, width, video_id = read_image(cfg, read_background=True)
    image = torch.cat([image, image_diff], dim=0)
    print(image.size(), height, width, video_id)

    wrapper = MidfusionInferenceWrapper(net)
    args = {'image': image, 'width': torch.tensor(width), 'height': torch.tensor(height)}
    flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
    print('Mid-fusion', flops, macs, params)


def measure_goeshift_lzu():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models import get_cfg_base_model
    from finetune_homography_mixup import GeneralizedRCNNFinetuneHomography
    from finetune_lzu_mixup import GeneralizedRCNNFinetuneLZU

    class FasterRCNNInferenceWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            net.eval()
            self.net = net
        def forward(self, image, width, height):
            batched_inputs = [{'image': image, 'height': int(height.item()), 'width': int(width.item())}]
            return self.net.inference(batched_inputs, do_postprocess=True)

    cfg = get_cfg_base_model('r101-fpn-3x')
    net = detectron2.modeling.build_model(cfg)
    net = GeneralizedRCNNFinetuneHomography.create_from_sup(net)
    net.load_state_dict(torch.load('/mnt/f/intersections_results/cvpr24/mscoco2017_remap_homography_r101-fpn-3x.pth'))
    image, height, width, video_id = read_image(cfg)
    print(image.size(), height, width, video_id)

    wrapper = FasterRCNNInferenceWrapper(net)
    args = {'image': image, 'width': torch.tensor(width), 'height': torch.tensor(height)}
    flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
    print('GeoShift', flops, macs, params)
    del wrapper

    cfg = get_cfg_base_model('r101-fpn-3x')
    net = detectron2.modeling.build_model(cfg)
    net = GeneralizedRCNNFinetuneLZU.create_from_sup(net)
    net.load_state_dict(torch.load('/mnt/f/intersections_results/cvpr24/paper_models/lzu_r101_kde/adapt001_r101-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_lzu.pth'))
    print(image.size(), height, width, video_id)

    wrapper = FasterRCNNInferenceWrapper(net)
    args = {'image': image, 'width': torch.tensor(width), 'height': torch.tensor(height)}
    flops, macs, params = calculate_flops(model=wrapper, kwargs=args, print_results=False)
    print('LZU', flops, macs, params)


if __name__ == '__main__':
    for s in [1, 2]:
        measure_yolov5(scale=s)

    # measure_midfusion('r101-fpn-3x')
    # measure_goeshift_lzu()
