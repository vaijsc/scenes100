#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
from copy import deepcopy
import gzip
import math
import random
import tqdm
import glob
import psutil
import hashlib
import argparse
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
from multiprocessing import Pool as ProcessPool
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import contextlib

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances

from yolov5 import *
from inference_server_simulate_yolov5 import YOLOServer
import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from evaluation import evaluate_masked, evaluate_cocovalid
import warnings
warnings.filterwarnings("ignore")

# video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
video_id_list = ['001']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_inference_server')


class SemiRandomClient(torchdata.Dataset):
    def __init__(self, cfg, scale=1):
        super(SemiRandomClient, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        self.scale = scale
        assert self.input_format == 'BGR'

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'scenes100', 'annotation', video_id)
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _dicts = json.load(fp)
            for im in _dicts:
                im['md5'] = '%s_%s' % (video_id, im['file_name']) # for pseudo-random shuffling
                im['md5'] = hashlib.md5(im['md5'].encode('utf-8')).hexdigest()
                im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['video_id'] = video_id
            self.images.extend(_dicts)
        self.images.sort(key=lambda x: x['md5'])
        self.preloaded_images = None

    def preload(self):
        if self.preloaded_images is not None:
            return
        self.preloaded_images = []
        for i in tqdm.tqdm(range(0, len(self.images)), ascii=True, desc='preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        image = torchvision.transforms.functional.resize(image, size=[int(self.scale*image.shape[1]), int(self.scale*image.shape[2])])
        return {'image': image, 'height': height, 'width': width, 'video_id': self.images[i]['video_id']}

    def __getitem__(self, i):
        if self.preloaded_images is None:
            return self.read(i), self.images[i]
        else:
            return self.preloaded_images[i], self.images[i]

    @staticmethod
    def collate(batch):
        return batch

class COCOEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, images, cfg):
        super(COCOEvaluationDataset, self).__init__()
        self.images = images
        self.input_format = cfg.INPUT.FORMAT
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        image = torchvision.transforms.functional.resize(image, size=[image.shape[1], image.shape[2]])
        return {'image': image, 'height': height, 'width': width, 'video_id': 'coco'}, self.images[i]
    @staticmethod
    def collate(batch):
        return batch
    
    
def evaluate_coco(args):
    cfg = get_cfg_base_model("r101-fpn-3x")
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK), 'checkpoint not readable: ' + args.ckpt
    try:
        model = load_yolov5(args.config, args.ckpt)
    except:
        model = load_yolov5(args.config)
        model = YOLOServer.create_from_sup(model, args.budget)
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
        if args.budget > 1:
            mapper = torch.load(args.mapper)
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']
            # print(mapper)
            # breakpoint()
    model.eval()

    images = get_coco_dicts(args, 'valid')

    for i in range(len(images)):
        for j, ann in enumerate(images[i]['annotations']):
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x, y, w, h = ann['bbox']
                x1 = x 
                y1 = y
                x2 = x + w
                y2 = y + h
                images[i]['annotations'][j]['bbox'] = [x1, y1, x2, y2]
                images[i]['annotations'][j]['bbox_mode'] = BoxMode.XYXY_ABS

    loader = torch.utils.data.DataLoader(
        COCOEvaluationDataset(images, cfg),
        batch_size=None, collate_fn=COCOEvaluationDataset.collate, shuffle=False, num_workers=2
    )

    detections = []
   
    for input, im in tqdm.tqdm(loader, total=len(images), ascii=True):
        im = deepcopy(im) 
        # breakpoint()       
        im['annotations'] = []
        with torch.no_grad():
            instances = model([input])[0]['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor
            score = instances.scores
            label = instances.pred_classes
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
        detections.append(im)
    # breakpoint()
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_AP = evaluate_cocovalid(args.cocodir, detections)
    print(results_AP['results']['weighted'])
    return results_AP['results']['weighted']  


def evaluate_scenes100(args):
    cfg = get_cfg_base_model("r101-fpn-3x")
    try:
        model = load_yolov5(args.config, args.ckpt)
    except:
        model = load_yolov5(args.config)
        model = YOLOServer.create_from_sup(model, args.budget, args.split_list)
        model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
        if args.budget > 1:
            mapper = torch.load(args.mapper)
            model.video_id_to_index = mapper['video_id_to_index']
            model.used_indices = mapper['used_indices']
            model.un_used_indices = mapper['un_used_indices']
            # print(mapper)
            # breakpoint()
    model.eval()
    dataset = SemiRandomClient(cfg, args.scale)
    if args.preload:
        dataset.preload()
    loader = torchdata.DataLoader(dataset, batch_size=None, collate_fn=SemiRandomClient.collate, shuffle=False, num_workers=1)
    gc.collect()
    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()
    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='detecting'):
        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model.inference([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
        detections[im['video_id']].append(det)
    t_total = time.time() - t_total
    print('%d finished in %.1f seconds, throughput %.3f images/sec' % (len(dataset), t_total, len(dataset) / t_total))

    results = {}
    for video_id in tqdm.tqdm(detections, ascii=True, desc='evaluating'):
        for det in detections[video_id]:
            # bbox has format [x1, y1, x2, y2]
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)
        # results[video_id]['detections'] = detections
    categories = ['person', 'vehicle', 'overall', 'weighted']
    avg_result = {c: [] for c in categories}
    for video_id in results:
        AP = results[video_id]['results']
        for cat in categories:
            avg_result[cat].append([AP[cat][0], AP[cat][1]])
    for cat in categories:
        avg_result[cat] = np.array(avg_result[cat]) * 100.0
        print('%s: mAP %.4f, AP50 %.4f' % (cat, avg_result[cat][:, 0].mean(), avg_result[cat][:, 1].mean()))
    return [avg_result['weighted'][:, 0].mean(), avg_result['weighted'][:, 1].mean()]

def inference_throughput(args):
    cfg = get_cfg_base_model("r101-fpn-3x")
    model = load_yolov5(args.config, args.ckpt)
    model.eval()
    dataset = SemiRandomClient(cfg)
    dataset.images = list(filter(lambda x: x['video_id'] == args.id, dataset.images))
    dataset.images = sorted(dataset.images, key=lambda x: x['file_name'])[:10]
    dataset.preload()
    gc.collect()
    torch.cuda.empty_cache()
    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            model.inference([dataset[i % len(dataset)][0]])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


def evaluate_individual_scenes100(args):
    global video_id_list
    print(video_id_list)
    video_id_list_copy = deepcopy(video_id_list)
    weighted_APs = {idx: [] for idx in video_id_list}
    for idx in video_id_list_copy:
        video_id_list = [idx]
        args.ckpt = os.path.join(os.path.dirname(__file__), "finetune_bs28_lr0.0001_teacherx1.5_conf0.4", idx, "adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.3999.pth")
        weighted_APs[idx] = evaluate_scenes100(args)
        print(f"Video {idx}: mAP {weighted_APs[idx][0]}, AP50 {weighted_APs[idx][1]}")
    mAPs = np.array([weighted_APs[idx][0] for idx in video_id_list_copy])
    AP50s = np.array([weighted_APs[idx][1] for idx in video_id_list_copy])
    print(f"Avg mAP: {np.mean(mAPs)}, Avg AP50: {np.mean(AP50s)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov5s_remap.pth", help='weights checkpoint of model')
    parser.add_argument('--mapper', type=str, default="", help='mapper checkpoint of model')
    parser.add_argument('--config', type=str, default="../../configs/yolov5s.yaml", help='config of model')
    parser.add_argument('--scale', type=float, default=1)

    parser.add_argument('--cocodir', type=str, default='.')

    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--instances', type=int, default=1)
    parser.add_argument('--val_set', type=str, choices=['scenes100', 'coco'], default='scenes100')
    parser.add_argument('--split_list', type=int, nargs='+')
    
    args = parser.parse_args()
    print(args)

    if args.opt == 'server':
        assert args.instances > 0
        if args.instances == 1:
            if args.val_set == 'scenes100':
                evaluate_scenes100(args)
            else:
                evaluate_coco(args)
        # else:
        #     simulate_concurrent(args)
    if args.opt == 'tp':
        inference_throughput(args)


'''
python eval_yolo.py --model r101-fpn-3x --opt server --ckpt ../../models/yolov5s_remap.pth --config ../../configs/yolov5s.yaml
'''
