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


# KITTI detection categories: ['Car', 'Cyclist', 'DontCare', 'Misc', 'Pedestrian', 'Person_sitting', 'Tram', 'Truck', 'Van']
thing_classes_coco = [['Cyclist', 'Pedestrian', 'Person_sitting'], ['Car', 'Truck', 'Van']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)


def get_kitti_dicts(kittidir):
    images_dir = os.path.join(kittidir, 'training', 'image_2')
    ifilelist = glob.glob(os.path.join(images_dir, '*.png'))
    annotations_dir = os.path.join(kittidir, 'training', 'label_2')

    category_id_remap = {}
    for i in range(0, len(thing_classes_coco)):
        for cat in thing_classes_coco[i]:
            category_id_remap[cat] = i

    kitti_dicts = []
    for f in tqdm.tqdm(ifilelist, ascii=True):
        im_d2 = {'file_name': f, 'image_id': len(kitti_dicts) + 1, 'annotations': []}
        with open(os.path.join(annotations_dir, os.path.basename(f)[:-3] + 'txt'), 'r') as fp:
            annotations = fp.readlines()
        annotations = [ann.strip().split(' ') for ann in annotations]

        for ann in annotations:
            lb, _, _, _, x1, y1, x2, y2 = ann[:8]
            if not lb in category_id_remap:
                continue
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
            im_d2['annotations'].append({'bbox': [x1, y1, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'area': (x2 - x1) * (y2 - y1), 'category_id': category_id_remap[lb]})
        kitti_dicts.append(im_d2)

        # font = ImageFont.truetype('../DejaVuSansCondensed.ttf', size=18)
        # f = Image.fromarray(skimage.io.imread(f)); draw = ImageDraw.Draw(f)
        # for ann in im_d2['annotations']:
        #     x1, y1, x2, y2 = ann['bbox']
        #     draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='#0000FF', width=2)
        #     draw.text((x1 + 2, y1 + 2), thing_classes[ann['category_id']], fill='#0000FF', font=font)
        # plt.figure(); plt.imshow(np.array(f)); plt.show()
    count_images, count_bboxes = len(kitti_dicts), sum(map(lambda ann: len(ann['annotations']), kitti_dicts))
    print('KITTI detection: %d images, %d bboxes' % (count_images, count_bboxes))
    return kitti_dicts


if __name__ == '__main__':
    get_kitti_dicts('F:\\KITTI')
