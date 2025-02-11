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
import skvideo.io
import networkx

import detectron2
from detectron2.structures import BoxMode

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def get_annotation_dict(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    return annotations


def stats():
    def _diag(w, h):
        return (w ** 2 + h ** 2) ** 0.5

    bbox_diag_stats = {'coco_train': {0: [], 1: [], 'c': []}, 'coco_valid': {0: [], 1: [], 'c': []}, 'scenes100': {}}
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cocodir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017')
    args.smallscale = False
    dst_cocovalid = get_coco_dicts(args, 'valid')
    for im in dst_cocovalid:
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                _r = _diag(ann['bbox'][2], ann['bbox'][3]) / _diag(im['width'], im['height'])
                bbox_diag_stats['coco_valid'][ann['category_id']].append(_r)
                bbox_diag_stats['coco_valid']['c'].append(_r)
            elif ann['bbox_mode'] == BoxMode.XYXY_ABS:
                _r = _diag(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) / _diag(im['width'], im['height'])
                bbox_diag_stats['coco_valid'][ann['category_id']].append(_r)
                bbox_diag_stats['coco_valid']['c'].append(_r)
    bbox_diag_stats['coco_valid'][0] = np.array(bbox_diag_stats['coco_valid'][0])
    bbox_diag_stats['coco_valid'][1] = np.array(bbox_diag_stats['coco_valid'][1])
    bbox_diag_stats['coco_valid']['c'] = np.array(bbox_diag_stats['coco_valid']['c'])

    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                _r = _diag(ann['bbox'][2], ann['bbox'][3]) / _diag(im['width'], im['height'])
                bbox_diag_stats['coco_train'][ann['category_id']].append(_r)
                bbox_diag_stats['coco_train']['c'].append(_r)
            elif ann['bbox_mode'] == BoxMode.XYXY_ABS:
                _r = _diag(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) / _diag(im['width'], im['height'])
                bbox_diag_stats['coco_train'][ann['category_id']].append(_r)
                bbox_diag_stats['coco_train']['c'].append(_r)
    bbox_diag_stats['coco_train'][0] = np.array(bbox_diag_stats['coco_train'][0])
    bbox_diag_stats['coco_train'][1] = np.array(bbox_diag_stats['coco_train'][1])
    bbox_diag_stats['coco_train']['c'] = np.array(bbox_diag_stats['coco_train']['c'])

    for video_id in video_id_list:
        args.id = video_id
        manual_v = get_annotation_dict(args)
        bbox_diag_stats['scenes100'][video_id] = {0: [], 1: [], 'c': []}
        for im in manual_v:
            for ann in im['annotations']:
                if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                    _r = _diag(ann['bbox'][2], ann['bbox'][3]) / _diag(im['width'], im['height'])
                    bbox_diag_stats['scenes100'][video_id][ann['category_id']].append(_r)
                    bbox_diag_stats['scenes100'][video_id]['c'].append(_r)
                elif ann['bbox_mode'] == BoxMode.XYXY_ABS:
                    _r = _diag(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) / _diag(im['width'], im['height'])
                    bbox_diag_stats['scenes100'][video_id][ann['category_id']].append(_r)
                    bbox_diag_stats['scenes100'][video_id]['c'].append(_r)
        bbox_diag_stats['scenes100'][video_id][0] = np.array(bbox_diag_stats['scenes100'][video_id][0])
        bbox_diag_stats['scenes100'][video_id][1] = np.array(bbox_diag_stats['scenes100'][video_id][1])
        bbox_diag_stats['scenes100'][video_id]['c'] = np.array(bbox_diag_stats['scenes100'][video_id]['c'])

    categories = {0: 'person', 1: 'vehicle', 'c': 'combined'}
    fig, axes = plt.subplots(3, 1, figsize=(20, 20))
    axes = axes.reshape(-1)
    for i, cat in enumerate([0, 1, 'c']):
        area_means = [bbox_diag_stats['coco_train'][cat].mean(), bbox_diag_stats['coco_valid'][cat].mean()]
        area_stds = [bbox_diag_stats['coco_train'][cat].std(), bbox_diag_stats['coco_valid'][cat].std()]
        xticks = ['COCO train', 'COCO valid']
        area_means_stds_scenes = []
        for video_id in video_id_list:
            if len(bbox_diag_stats['scenes100'][video_id][cat]) > 0:
                area_means_stds_scenes.append([bbox_diag_stats['scenes100'][video_id][cat].mean(), bbox_diag_stats['scenes100'][video_id][cat].std(), video_id])
            else:
                area_means_stds_scenes.append([-1, -1, video_id])
        area_means_stds_scenes.sort(key=lambda x: x[0])
        area_means = np.array(area_means + [x[0] for x in area_means_stds_scenes])
        area_stds = np.array(area_stds + [x[1] for x in area_means_stds_scenes])
        xticks = xticks + [x[2] for x in area_means_stds_scenes]
        xs = np.arange(0, len(xticks))

        mask = area_means > 0
        axes[i].errorbar(xs[mask], area_means[mask], yerr=area_stds[mask], capsize=2, marker='.', linestyle='-', color={0: 'r', 1: 'b', 'c': 'k'}[cat])
        axes[i].set_xticks(range(0, len(xticks)))
        axes[i].set_xticklabels(xticks, rotation='vertical', fontsize=10)
        axes[i].set_xlim(-1, len(xticks))
        ymin, ymax = (area_means - area_stds)[mask].min() * 1.05, (area_means + area_stds)[mask].max() * 1.05
        axes[i].set_ylim(ymin, ymax)
        axes[i].set_ylabel('bbox diagonal in percentage of image bbox diagonal (mean & std)')
        # axes[i].set_yticks(np.arange(int(ymin) // 10000 * 10000, int(ymax) // 10000 * 10000, 5000))
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[cat]))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.96)
    plt.savefig('bbox_stats.pdf')
    plt.close()


if __name__ == '__main__':
    stats()
