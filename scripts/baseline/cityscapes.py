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


# CityScapes segmentation categories: ['bicycle', 'bicyclegroup', 'bridge', 'building', 'bus', 'car', 'caravan', 'cargroup', 'dynamic', 'ego vehicle', 'fence', 'ground', 'guard rail', 'license plate', 'motorcycle', 'motorcyclegroup', 'out of roi', 'parking', 'person', 'persongroup', 'pole', 'polegroup', 'rail track', 'rectification border', 'rider', 'ridergroup', 'road', 'sidewalk', 'sky', 'static', 'terrain', 'traffic light', 'traffic sign', 'trailer', 'train', 'truck', 'truckgroup', 'tunnel', 'vegetation', 'wall']
thing_classes_coco = [['person', 'rider'], ['bus', 'car', 'caravan', 'truck', 'trailer']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)


def get_cityscapes_dicts(cityscapedir, split):
    images_dir = os.path.join(cityscapedir, 'leftImg8bit', split)
    cities_list = [os.path.basename(c) for c in glob.glob(os.path.join(images_dir, '*'))]
    annotations_dir = os.path.join(cityscapedir, 'gtFine', split)

    category_id_remap = {}
    for i in range(0, len(thing_classes_coco)):
        for cat in thing_classes_coco[i]:
            category_id_remap[cat] = i

    cityscapes_dicts = []
    for city in cities_list:
        ifilelist = sorted(glob.glob(os.path.join(images_dir, city, '*.png')))
        for f in tqdm.tqdm(ifilelist, ascii=True, desc=city):
            with open(os.path.join(annotations_dir, city, os.path.basename(f)[:-15] + 'gtFine_polygons.json'), 'r') as fp:
                annotations = json.load(fp)
            im_d2 = {'file_name': f, 'height': annotations['imgHeight'], 'width': annotations['imgWidth'], 'image_id': len(cityscapes_dicts) + 1, 'annotations': []}
            annotations['objects'] = list(filter(lambda x: not x['label'] in ['bridge', 'building', 'fence', 'ground', 'guard rail', 'license plate', 'out of roi', 'pole', 'polegroup', 'rail track', 'rectification border', 'road', 'sidewalk', 'sky', 'terrain', 'traffic light', 'traffic sign', 'tunnel', 'vegetation', 'wall', 'static', 'dynamic', 'ego vehicle', 'cargroup', 'persongroup', 'ridergroup', 'bicycle', 'bicyclegroup', 'motorcycle', 'motorcyclegroup', 'parking', 'train', 'truckgroup'], annotations['objects']))
            for obj in annotations['objects']:
                polygon = np.array(obj['polygon'])
                x1, y1, x2, y2 = map(float, [polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()])
                im_d2['annotations'].append({'bbox': [x1, y1, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'area': (x2 - x1) * (y2 - y1), 'category_id': category_id_remap[obj['label']]})
            cityscapes_dicts.append(im_d2)

            # font = ImageFont.truetype('../DejaVuSansCondensed.ttf', size=18)
            # f = Image.fromarray(skimage.io.imread(f)); draw = ImageDraw.Draw(f)
            # for obj in annotations['objects']:
            #     draw.line(tuple(map(tuple, obj['polygon'])), fill='#FF0000', width=2)
            # for ann in im_d2['annotations']:
            #     x1, y1, x2, y2 = ann['bbox']
            #     draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='#0000FF', width=2)
            #     draw.text((x1 + 2, y1 + 2), thing_classes[ann['category_id']], fill='#0000FF', font=font)
            # plt.figure(); plt.imshow(np.array(f)); plt.show()
    count_images, count_bboxes = len(cityscapes_dicts), sum(map(lambda ann: len(ann['annotations']), cityscapes_dicts))
    print('CityScapes detection %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return cityscapes_dicts


if __name__ == '__main__':
    get_cityscapes_dicts('F:\\CityScapes', 'train')
