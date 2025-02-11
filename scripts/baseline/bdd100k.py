#!python3

import os
import sys
import types
import time
import datetime
import json
import copy
import math
import random
import tqdm
import glob
import gzip
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

import skimage.io
import skvideo.io
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
import networkx

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.engine import launch as detectron2_launcher
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import BoxMode


# BDD100K detection categories: {'train', 'motor', 'truck', 'traffic sign', 'bus', 'bike', 'lane', 'traffic light', 'car', 'rider', 'person', 'drivable area'}
thing_classes_coco = [['person', 'rider'], ['car', 'bus', 'truck']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)


def get_bdd100k_dicts(bdddir, split):
    images_dir = os.path.join(bdddir, split)
    annotations_json = os.path.join(bdddir, 'bdd100k_labels_images_%s.json' % split)
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    for im in annotations:
        im['labels'] = list(filter(lambda x: x['category'] in ['person', 'rider', 'car', 'bus', 'truck'], im['labels']))
        for ann in im['labels']:
            assert 'box2d' in ann
    annotations = list(filter(lambda x: len(x['labels']) > 0, annotations))

    category_id_remap = {}
    for i in range(0, len(thing_classes_coco)):
        for cat in thing_classes_coco[i]:
            category_id_remap[cat] = i

    bdd100k_dicts = []
    for im in annotations:
        im_d2 = {'file_name': os.path.join(images_dir, im['name']), 'image_id': len(bdd100k_dicts) + 1, 'annotations': []}
        for ann in im['labels']:
            x1, y1, x2, y2 = ann['box2d']['x1'], ann['box2d']['y1'], ann['box2d']['x2'], ann['box2d']['y2']
            im_d2['annotations'].append({'bbox': [x1, y1, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'area': (x2 - x1) * (y2 - y1), 'category_id': category_id_remap[ann['category']]})
        bdd100k_dicts.append(im_d2)
    count_images, count_bboxes = len(bdd100k_dicts), sum(map(lambda ann: len(ann['annotations']), bdd100k_dicts))
    print('BDD100K detection %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return bdd100k_dicts


if __name__ == '__main__':
    get_bdd100k_dicts('F:\\BDD100K\\images_100k', 'train')
    get_bdd100k_dicts('F:\\BDD100K\\images_100k', 'val')
