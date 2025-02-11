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

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref

from midfusion_mixup import get_midfusion_avg_detector, construct_image_w_background
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


class DetectDataset(torchdata.Dataset):
    def __init__(self, image_dicts):
        super(DetectDataset, self).__init__()
        self.image_dicts = image_dicts
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name'], format='BGR')
        if 'file_name_background' in self.image_dicts[i]:
            image_background = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name_background'], format='BGR')
            image, _, image_diff = construct_image_w_background(image, image_background)
            return self.image_dicts[i]['file_name'], self.image_dicts[i]['file_name_background'], image, image_diff
        else:
            return self.image_dicts[i]['file_name'], image
    @staticmethod
    def collate(batch):
        return batch

def detect_pseudo_label(model, ckpt, model_iteration, day_idx):
    basedir = os.path.normpath(os.path.dirname(__file__))
    assert model_iteration >= 0
    if model_iteration == 0: # base models
        cfg = get_cfg()
        if model == 'r50-fpn-3x':
            cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(basedir, 'models', 'mscoco2017_remap_r50-fpn-3x.pth')
        elif model == 'r101-fpn-3x':
            cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(basedir, 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
        else:
            raise NotImplementedError
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        detector = DefaultPredictor(cfg)
    else: # adapted models
        assert model == 'r101-fpn-3x-midfusion-mixup'
        cfg, detector = get_midfusion_avg_detector('r101-fpn-3x', ckpt)
    print('detectron2 model: %s iteration %d' % (model, model_iteration))
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        vfilelist = json.load(fp)['days'][day_idx]
    for vfilename in vfilelist:
        print('detect in', vfilename)
        frame_objs, ifilelist = [], []
        with open(os.path.join(basedir, 'frames', vfilename + '.json'), 'r') as fp:
            chunks = json.load(fp)['chunks']
        for idx_list in chunks:
            if model_iteration == 0: # base models only use frame image
                image_dicts = [{'file_name': os.path.join(basedir, 'frames', vfilename, '%08d.jpg' % x)} for x in idx_list]
            else: # adapted models also use background image
                image_dicts = [{
                    'file_name': os.path.join(basedir, 'frames', vfilename, '%08d.jpg' % x),
                    'file_name_background': os.path.join(basedir, 'frames', 'background', '%s.%08d.inpaint.jpg' % (vfilename, idx_list[-1]))
                    } for x in idx_list]

            ifilelist.append([])
            frame_objs.append([])
            loader = torchdata.DataLoader(DetectDataset(image_dicts), batch_size=None, collate_fn=DetectDataset.collate, shuffle=False, num_workers=2)
            if model_iteration == 0:
                for fn, im in tqdm.tqdm(loader, total=len(image_dicts), ascii=True, desc='%d ~ %d' % (min(idx_list), max(idx_list))):
                    ifilelist[-1].append(os.path.basename(fn))
                    instances = detector(im)['instances'].to('cpu')
                    frame_objs[-1].append({
                        # bbox has format [x1, y1, x2, y2]
                        'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                        'score': instances.scores.numpy().tolist(),
                        'label': instances.pred_classes.numpy().tolist()
                    })
            else:
                for fn, _, im, im_diff in tqdm.tqdm(loader, total=len(image_dicts), ascii=True, desc='%d ~ %d' % (min(idx_list), max(idx_list))):
                    ifilelist[-1].append(os.path.basename(fn))
                    instances = detector(im, im_diff)['merge']['instances'].to('cpu')
                    frame_objs[-1].append({
                        # bbox has format [x1, y1, x2, y2]
                        'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                        'score': instances.scores.numpy().tolist(),
                        'label': instances.pred_classes.numpy().tolist()
                    })

        result_json_zip = os.path.join(basedir, 'self_supervision', 'iteration%d' % model_iteration, 'detect_%s_%s_iteration%d.json.gz' % (vfilename, model, model_iteration))
        with gzip.open(result_json_zip, 'wt') as fp:
            fp.write(json.dumps({
                'model': model,
                'iteration': model_iteration,
                'ckpt': os.path.basename(cfg.MODEL.WEIGHTS),
                'classes': thing_classes,
                'frames': ifilelist,
                'dets': frame_objs
            }))


def _graph_refine(params):
    _dict_json, _iou_thres, desc = params['dict'], params['iou_thres'], params['desc']
    count_bboxes = 0
    for annotations in tqdm.tqdm(_dict_json, ascii=True, desc='refining chunk ' + desc):
        G = networkx.Graph()
        [G.add_node(i) for i in range(0, len(annotations['annotations']))]
        for i in range(0, len(annotations['annotations'])):
            for j in range(i, len(annotations['annotations'])):
                iou_value = IoU(annotations['annotations'][i]['bbox'], annotations['annotations'][j]['bbox'])
                if annotations['annotations'][i]['category_id'] == annotations['annotations'][j]['category_id'] and iou_value > _iou_thres:
                    G.add_edge(i, j, iou=iou_value)
        subs = list(networkx.algorithms.components.connected_components(G))

        anns_refine = []
        for sub_nodes in subs:
            max_degree, max_degree_n = -1, -1
            for n in sub_nodes:
                D = sum(map(lambda t: t[2], list(G.edges(n, data='iou'))))
                if D > max_degree:
                    max_degree, max_degree_n = D, n
            anns_refine.append(annotations['annotations'][max_degree_n])
        annotations['annotations'] = anns_refine
        count_bboxes += len(annotations['annotations'])
    return _dict_json, count_bboxes

def refine(day_idx, interation_list, refine_det_score_thres=0.5, refine_iou_thres=0.85):
    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        clips = json.load(fp)
    meta = clips['meta']
    vfilelist = clips['days'][day_idx]

    refined_pseudo_list = []
    for vfilename in vfilelist:
        dicts_v = []
        with open(os.path.join(basedir, 'frames', vfilename + '.json'), 'r') as fp:
            chunks = json.load(fp)['chunks']
        for idx_list in chunks:
            dicts_v.append([{'file_name': os.path.join(basedir, 'frames', vfilename, '%08d.jpg' % x), 'image_id': 0, 'height': meta['H'], 'width': meta['W'], 'annotations': []} for x in idx_list])
        det_filelist = []
        for i in interation_list:
            if i == 0:
                det_filelist.append(os.path.join(basedir, 'self_supervision', 'iteration0', 'detect_%s_r50-fpn-3x_iteration0.json.gz' % vfilename))
                det_filelist.append(os.path.join(basedir, 'self_supervision', 'iteration0', 'detect_%s_r101-fpn-3x_iteration0.json.gz' % vfilename))
            else:
                det_filelist.append(os.path.join(basedir, 'self_supervision', 'iteration%d' % i, 'detect_%s_r101-fpn-3x-midfusion-mixup_iteration%d.json.gz' % (vfilename, i)))
        det_filelist = list(filter(lambda f: os.access(f, os.R_OK), det_filelist))

        count_box_raw = 0
        print('refining', vfilename)
        for f in det_filelist:
            print(' - ...' + f[-110 :])
            with gzip.open(f, 'rt') as fp:
                dets = json.loads(fp.read())['dets']
            assert len(dets) == len(dicts_v), 'detection & dataset mismatch'
            for i in range(0, len(dets)):
                assert len(dets[i]) == len(dicts_v[i]), 'detection & dataset mismatch'
                for j in range(0, len(dets[i])):
                    for k in range(0, len(dets[i][j]['score'])):
                        if dets[i][j]['score'][k] < refine_det_score_thres:
                            continue
                        dicts_v[i][j]['annotations'].append({'bbox': dets[i][j]['bbox'][k], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i][j]['label'][k], 'score': dets[i][j]['score'][k]})
                        count_box_raw += 1

        pool = ProcessPool(processes=len(dicts_v))
        params_list = []
        for i, dicts_chunk in enumerate(dicts_v):
            params_list.append({'dict': dicts_chunk, 'iou_thres': refine_iou_thres, 'desc': '%02d/%02d' % (i + 1, len(dicts_v))})
        dicts_v = pool.map_async(_graph_refine, params_list).get()
        pool.close()
        pool.join()
        count_box_refine = sum([r[1] for r in dicts_v])
        dicts_v = [r[0] for r in dicts_v]
        print('bounding boxes: %d after score thresholding => %d after refinement' % (count_box_raw, count_box_refine))
        refined_pseudo_list.append(dicts_v)
    return refined_pseudo_list, vfilelist


def dynamic_background(day_idx, Q_size=256):
    import cv2
    from collections import deque

    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        clips = json.load(fp)
    meta = clips['meta']
    vfilelist = clips['days'][day_idx]
    refined_pseudo_list, _ = refine(day_idx, [0])

    for v in range(0, len(vfilelist)):
        for i in range(0, len(refined_pseudo_list[v])):
            dicts_chunk = refined_pseudo_list[v][i]
            ifilename_last = os.path.basename(dicts_chunk[-1]['file_name'])[: -4]
            print('extracting background: %s %i %s' % (vfilelist[v], i, ifilename_last))
            dicts_chunk = sklearn.utils.shuffle(dicts_chunk)[: Q_size]
            im_arr_list = np.stack([skimage.io.imread(d['file_name']).astype(np.float16) for d in dicts_chunk], axis=0)
            M_arr_list = np.ones_like(im_arr_list[:, :, :, 0 : 1])
            for j, d in enumerate(dicts_chunk):
                for ann in d['annotations']:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    x1, y1, x2, y2 = map(int, map(lambda x: max(x, 0), ann['bbox']))
                    if x2 - x1 > 500 or y2 - y1 > 500: continue
                    M_arr_list[j, y1 : y2, x1 : x2] = 0.0
            averaged, weighted, M_averaged = im_arr_list.mean(axis=0), (im_arr_list * M_arr_list).mean(axis=0), M_arr_list.mean(axis=0)
            M_master = np.zeros(shape=M_averaged.shape, dtype=np.uint8) + 255
            for x in range(0, weighted.shape[0]):
                for y in range(0, weighted.shape[1]):
                    if M_averaged[x, y, 0] < 0.5 / Q_size:
                        M_averaged[x, y], weighted[x, y], M_master[x, y] = 1, averaged[x, y], 0
            weighted = weighted / M_averaged
            prefix = os.path.join(basedir, 'frames', 'background', '%s.%s' % (vfilelist[v], ifilename_last))
            skimage.io.imsave(prefix + '.mask.png', M_master, check_contrast=False)
            skimage.io.imsave(prefix + '.average.jpg', averaged.astype(np.uint8), quality=95)
            skimage.io.imsave(prefix + '.background.jpg', weighted.astype(np.uint8), quality=95)
            if M_master.min() == 255:
                pass
            else:
                M_master = 255 - M_master
                pixels = M_master.sum() / 255.0
                if pixels < 10:
                    pass
                else:
                    R = min(M_master.shape) // 10
                    weighted = cv2.inpaint(weighted.astype(np.uint8), M_master, R, cv2.INPAINT_TELEA)
                    # background = cv2.inpaint(background.astype(np.uint8), M_master, R, cv2.INPAINT_NS)
            skimage.io.imsave(prefix + '.inpaint.jpg', weighted.astype(np.uint8), quality=95)


from annotate import draw_bbox
def _draw_bbox_wrapper(im):
    im_arr = skimage.io.imread(im['file_name'])
    im_arr = draw_bbox(im_arr, im['annotations'], im['file_name'][-46:])
    return im_arr

def visualize_pseudo_labels_iter0():
    import functools
    basedir = os.path.normpath(os.path.dirname(__file__))
    dst_pseudo_anno = []
    for d in [0, 1, 2, 3]:
        dst_pseudo_anno = dst_pseudo_anno + refine(d, [0])[0]
    dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
    dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
    writer = skvideo.io.FFmpegWriter(os.path.join(basedir, 'self_supervision', 'pseudo_label_iteration0.mp4'), inputdict={'-r': '5'}, outputdict={'-vcodec': 'libx265', '-r': '5', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '27'})
    im_Q = []
    for im in tqdm.tqdm(dst_pseudo_anno, ascii=True):
        im_Q.append(im)
        if len(im_Q) > 300:
            pool = ProcessPool(processes=6)
            im_arr_list = pool.map_async(_draw_bbox_wrapper, im_Q).get()
            pool.close()
            pool.join()
            for im_arr in im_arr_list:
                writer.writeFrame(im_arr)
            im_Q, im_arr_list = [], None
    if len(im_Q) > 0:
        for im in im_Q:
            writer.writeFrame(_draw_bbox_wrapper(im))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['label', 'background', 'visualize'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--day', nargs='+', default=[])
    parser.add_argument('--iteration', nargs='+', default=[])
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    args.day, args.iteration = list(map(int, set(args.day))), list(map(int, set(args.iteration)))

    if args.opt == 'label':
        for d in args.day:
            for i in args.iteration:
                detect_pseudo_label(args.model, args.ckpt, i, d)

    if args.opt == 'background':
        for d in args.day:
            dynamic_background(d)

    if args.opt == 'visualize':
        visualize_pseudo_labels_iter0()


'''
python pseudo_label.py --opt label --model r50-fpn-3x --iteration 0 --day 0
python pseudo_label.py --opt label --model r101-fpn-3x --iteration 0 --day 2
python pseudo_label.py --opt background --day 0 1

python pseudo_label.py --opt label --model r101-fpn-3x-midfusion-mixup --iteration 1 --day 0 --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth
'''
