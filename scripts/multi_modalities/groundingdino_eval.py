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
from evaluation import evaluate_masked, evaluate_cocovalid


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def evaluate(args):
    with open(os.path.join(os.path.dirname(__file__), '..', 'baseline', 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]

    def _compare(APs, prefix):
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

    with open(args.json, 'r') as fp:
        dino_detections = json.load(fp)
    for k in dino_detections:
        for im in dino_detections[k]:
            for ann in im['annotations']:
                ann['bbox'] = ann['xyxy']
                ann['bbox_mode'] = BoxMode.XYXY_ABS
    dino_AP = {}
    for video_id in video_id_list:
        print('\nvideo', video_id)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = evaluate_masked(video_id, dino_detections[video_id])
        print('             %s' % '/'.join(results['metrics']))
        for c in sorted(results['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results['results'][c])))
        del results['raw']
        dino_AP[video_id] = results
    _compare(dino_AP, os.path.basename(args.json)[:-5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    print(args)
    evaluate(args)


'''
python groundingdino_eval.py --model r101-fpn-3x --json ../../../Grounded-Segment-Anything/GroundingDINO/scenes100_swint_detections.json
'''
