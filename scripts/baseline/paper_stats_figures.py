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
import networkx

import sklearn.utils
from sklearn.mixture import GaussianMixture

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes_coco = [['person'], ['car', 'bus', 'truck']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)
bbox_rgbs = ['#FF0000', '#0000FF']


class Dummy(object):
    pass


def draw_bbox(im, annotations, linewidth=-1):
    if linewidth < 0:
        linewidth = int(min(im.shape[0], im.shape[1]) / 300)
    im = Image.fromarray(im, 'RGB')
    draw = ImageDraw.Draw(im)
    for ann in annotations:
        # bbox has format [x1, y1, x2, y2]
        x1, y1, x2, y2 = ann['bbox']
        cat = ann['category_id']
        draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[cat], width=linewidth)
    im = np.array(im)
    return im


def stats_coco(cocodir, split):
    if split == 'valid':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_val2017.json')
    elif split == 'train':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
    else:
        return None
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap = {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i
    coco_dicts = {}
    images_dir = os.path.join(cocodir, 'images', 'val2017' if split == 'valid' else 'train2017')
    print('MSCOCO-2017 %s in %s' % (split, cocodir))
    count_orig_image, count_orig_box = len(annotations['images']), len(annotations['annotations'])
    count_remap_image, count_remap_box = 0, 0
    for ann in annotations['annotations']:
        if not ann['category_id'] in category_id_remap:
            continue
        count_remap_box += 1
    print('%d images, %d original boxes, %d remapped boxes' % (count_orig_image, count_orig_box, count_remap_box))


def coco_mosaic(cocodir):
    annotations_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
    images_dir = os.path.join(cocodir, 'images', 'train2017')
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap, coco_dicts = {}, {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i
    for im in annotations['images']:
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        if ann['category_id'] in category_id_remap:
            x, y, w, h = ann['bbox']
            coco_dicts[ann['image_id']]['annotations'].append({'bbox': [x, y, x + w, y + h], 'bbox_mode': BoxMode.XYXY_ABS, 'area': ann['area'], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    coco_dicts = list(filter(lambda x: x['width'] == x['height'], coco_dicts))
    for im in coco_dicts:
        im['counts'] = [0, 0]
        for ann in im['annotations']:
            im['counts'][ann['category_id']] += 1
    coco_dicts_0 = list(filter(lambda x: x['counts'][0] > x['counts'][1] * 2, coco_dicts))
    coco_dicts_1 = list(filter(lambda x: x['counts'][1] > x['counts'][0] * 2, coco_dicts))
    coco_dicts_01 = list(filter(lambda x: x['counts'][1] == x['counts'][0], coco_dicts))
    coco_dicts = coco_dicts_0[:100] + coco_dicts_1[:100] + coco_dicts_01[:100]
    random.shuffle(coco_dicts)
    coco_dicts = coco_dicts[:64]
    im_mosaic = np.zeros(shape=(4000, 4000, 3), dtype=np.uint8) + 255
    for i, im in tqdm.tqdm(enumerate(coco_dicts)):
        im_arr = skimage.io.imread(im['file_name'])
        W = im_arr.shape[0]
        im_arr = (skimage.transform.resize(im_arr, (480, 480), anti_aliasing=True) * 255.0).astype(np.uint8)
        for ann in im['annotations']:
            ann['bbox'] = list(map(lambda x: x * (im_arr.shape[0] / W), ann['bbox']))
        im_arr = draw_bbox(im_arr, im['annotations'], linewidth=15)
        im_mosaic[(i // 8) * 500 : (i // 8) * 500 + 480, (i % 8) * 500 : (i % 8) * 500 + 480] = im_arr
    skimage.io.imsave('/mnt/c/Users/zhang/Desktop/TMP/mscoco_mosaic.png', im_mosaic)


class EvaluationDataset(torchdata.Dataset):
    def __init__(self, image_dicts, image_list):
        super(EvaluationDataset, self).__init__()
        assert len(image_dicts) == len(image_list)
        self.image_dicts = image_dicts
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, i):
        im_arr = skimage.io.imread(self.image_list[i])
        if len(im_arr.shape) < 3:
            im_arr = np.stack([im_arr] * 3, axis=2)
        return self.image_dicts[i], im_arr[:, :, ::-1]
    @staticmethod
    def collate(batch):
        return batch

def base_perform_coco(model):
    from evaluation import evaluate_cocovalid
    args = Dummy()
    args.smallscale = False
    args.cocodir = '../../../MSCOCO2017'
    images = get_coco_dicts(args, 'valid')
    loader = torchdata.DataLoader(
        EvaluationDataset(
            copy.deepcopy(images),
            [im['file_name'] for im in images]
        ),
        batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=4
    )
    cfg = get_cfg_base_model(model)
    detector = DefaultPredictor(cfg)
    detections = []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True):
        det = copy.deepcopy(im)
        det['annotations'] = []
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    results = evaluate_cocovalid(args.cocodir, detections)
    results['raw'] = None
    print(results)


def refine_annotations(args):
    def _graph_refine(params):
        _dict_json, _args, desc = params['dict'], params['args'], params['desc']
        count_bboxes = 0
        for annotations in tqdm.tqdm(_dict_json, ascii=True, desc='refining chunk ' + desc):
            G = networkx.Graph()
            [G.add_node(i) for i in range(0, len(annotations['annotations']))]
            for i in range(0, len(annotations['annotations'])):
                for j in range(i, len(annotations['annotations'])):
                    iou_value = IoU(annotations['annotations'][i]['bbox'], annotations['annotations'][j]['bbox'])
                    if annotations['annotations'][i]['category_id'] == annotations['annotations'][j]['category_id'] and iou_value > _args.refine_iou_thres:
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
            if 'det_count' in annotations: del annotations['det_count']
            if 'sot_count' in annotations: del annotations['sot_count']
            annotations['bbox_count'] = len(annotations['annotations'])
            count_bboxes += annotations['bbox_count']
        return _dict_json, count_bboxes
    dst = TrainingFrames(args.id)
    imagedir = os.path.join(dst.lmdb_path, 'jpegs')
    det_filelist, sot_filelist = [], []
    for m in args.anno_models:
        det_filelist.append(os.path.normpath(os.path.join(dst.lmdb_path, '..', '..', 'train_pseudo_label', '%s_detect_%s.json.gz' % (args.id, m))))
        sot_filelist.append(os.path.normpath(os.path.join(dst.lmdb_path, '..', '..', 'train_pseudo_label', '%s_detect_%s_DiMP.json.gz' % (args.id, m))))
    for f in det_filelist + sot_filelist:
        assert os.access(f, os.R_OK), '%s not readable' % f
    dict_json = []
    for i in range(0, len(dst)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(imagedir, dst.ifilelist[i])), 'image_id': i, 'height': dst.meta['meta']['video']['H'], 'width': dst.meta['meta']['video']['W'], 'annotations': [], 'det_count': 0, 'sot_count': 0})
    for f in det_filelist:
        print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
        with gzip.open(f, 'rt') as fp:
            dets = json.loads(fp.read())['dets']
        assert len(dets) == len(dict_json), 'detection & dataset mismatch'
        for i in range(0, len(dets)):
            for j in range(0, len(dets[i]['score'])):
                if dets[i]['score'][j] < args.refine_det_score_thres:
                    continue
                dict_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'src': 'det', 'score': dets[i]['score'][j]})
                dict_json[i]['det_count'] += 1
    for f in sot_filelist:
        print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
        with gzip.open(f, 'rt') as fp:
            _t = json.loads(fp.read())
            _forward, _backward = _t['forward'], _t['backward']
        assert len(_forward) == len(dict_json) and len(_backward) == len(dict_json), 'tracking & dataset mismatch'
        for i in range(0, len(_forward)):
            for tr in _forward[i]:
                dict_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
                dict_json[i]['sot_count'] += 1
        for i in range(0, len(_backward)):
            for tr in _backward[i]:
                dict_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
                dict_json[i]['sot_count'] += 1
    print('finish reading from detection & tracking results')
    dict_json_before_refine = []
    for im in dict_json:
        if im['det_count'] < im['sot_count']:
            dict_json_before_refine.append(im)
    # dict_json_before_refine.sort(key=lambda x: len(x['annotations']) * -1)
    # dict_json_before_refine = dict_json_before_refine[:50]
    dict_json_before_refine = [dict_json_before_refine[i] for i in [100, 600, 1100, 1600]]
    dict_json = copy.deepcopy(dict_json_before_refine)
    dict_json, _ = _graph_refine({'dict': dict_json, 'args': args, 'desc': ''})
    for i, im in enumerate(dict_json):
        print(dict_json[i]['file_name'])
        im = skimage.io.imread(dict_json[i]['file_name'])
        im_det = draw_bbox(im, list(filter(lambda x: x['src'] == 'det', dict_json_before_refine[i]['annotations'])))
        im_sot = draw_bbox(im, list(filter(lambda x: x['src'] == 'sot', dict_json_before_refine[i]['annotations'])))
        im_refine = draw_bbox(im, dict_json[i]['annotations'])
        skimage.io.imsave(os.path.join('E:\\banner', 'original_' + os.path.basename(dict_json[i]['file_name'])), im, quality=80)
        skimage.io.imsave(os.path.join('E:\\banner', 'pseudo_boxes_detect_' + os.path.basename(dict_json[i]['file_name'])), im_det, quality=80)
        skimage.io.imsave(os.path.join('E:\\banner', 'pseudo_boxes_track_' + os.path.basename(dict_json[i]['file_name'])), im_sot, quality=80)
        skimage.io.imsave(os.path.join('E:\\banner', 'pseudo_boxes_refine_' + os.path.basename(dict_json[i]['file_name'])), im_refine, quality=80)

    for i, im in enumerate(dict_json):
        while True:
            j = random.randint(0, len(dict_json) - 1)
            if j != i: break
        im = skimage.io.imread(dict_json[i]['file_name'])
        im_src = skimage.io.imread(dict_json[j]['file_name'])
        src_annotations = dict_json[j]['annotations'][: max(1, int(args.mixup_r * len(dict_json[j]['annotations'])))]
        src_annotations_rand = copy.deepcopy(src_annotations)
        im_mixup, im_mixup_rand = im.copy(), im.copy()
        for ann in src_annotations:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            x1, y1, x2, y2 = map(int, ann['bbox'])
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
            im_mixup[y1 : y2, x1 : x2] = im_src[y1 : y2, x1 : x2]
        for ann in src_annotations_rand:
            x1, y1, x2, y2 = map(int, ann['bbox'])
            x1, y1, x2, y2 = map(lambda x: 1 if x < 1 else x, [x1, y1, x2, y2])
            x2, y2 = min(im.shape[1], max(x2, x1 + 1)), min(im.shape[0], max(y2, y1 + 1))
            x_shift, y_shift = np.random.randint(-1 * x1, im.shape[1] - x2), np.random.randint(-1 * y1, im.shape[0] - y2)
            im_mixup_rand[y1 + y_shift : y2 + y_shift, x1 + x_shift : x2 + x_shift] = im_src[y1 : y2, x1 : x2]
            ann['bbox'] = [x1 + x_shift, y1 + y_shift, x2 + x_shift, y2 + y_shift]
        annotations_trimmed, annotations_trimmed_rand = [], []
        for ann in dict_json[i]['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            _trim = False
            for ann2 in src_annotations:
                if intersect_ratios(ann['bbox'], ann2['bbox'])[0] >= args.mixup_overlap_thres or bbox_inside(ann['bbox'], ann2['bbox']):
                    _trim = True
                    break
            if not _trim:
                annotations_trimmed.append(ann)
        for ann in src_annotations:
            annotations_trimmed.append(ann)
        for ann in dict_json[i]['annotations']:
            _trim = False
            for ann2 in src_annotations_rand:
                if intersect_ratios(ann['bbox'], ann2['bbox'])[0] >= args.mixup_overlap_thres or bbox_inside(ann['bbox'], ann2['bbox']):
                    _trim = True
                    break
            if not _trim:
                annotations_trimmed_rand.append(ann)
        for ann in src_annotations_rand:
            annotations_trimmed_rand.append(ann)
        skimage.io.imsave(os.path.join('E:\\banner', 'mixup_no_anno_' + os.path.basename(dict_json[i]['file_name'])), im_mixup, quality=80)
        skimage.io.imsave(os.path.join('E:\\banner', 'mixup_' + os.path.basename(dict_json[i]['file_name'])), draw_bbox(im_mixup, annotations_trimmed), quality=80)
        skimage.io.imsave(os.path.join('E:\\banner', 'mixup_rand_' + os.path.basename(dict_json[i]['file_name'])), draw_bbox(im_mixup_rand, annotations_trimmed_rand), quality=80)


def inpaint_dynamic():
    from collections import deque
    from finetune import BoxMode, refine_annotations
    args = Dummy()
    args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '001', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    outputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'example'))
    dict_json, _ = refine_annotations(args, visualize=False)
    dst = TrainingFrames(args.id)
    assert dst.meta['sample_fps'] == 5 and len(dst) == len(dict_json)
    N = len(dst)
    background_interval, sample_interval = 90, 2 # seconds
    background_idx = set(np.arange(0, N, background_interval * dst.meta['sample_fps']).astype(int).tolist()[1:])
    sample_idx = set(np.arange(0, N, sample_interval * dst.meta['sample_fps']).astype(int).tolist())
    buffer_size = 100
    Q = deque([], maxlen=buffer_size)
    for i in tqdm.tqdm(range(0, N), ascii=True, desc=args.id):
        if i in sample_idx:
            im_arr, _, fn, _ = dst[i]
            anns = dict_json[i]['annotations']
            M_arr = np.ones_like(im_arr[:, :, 0 : 1])
            for ann in anns:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = map(int, map(lambda x: max(x, 0), ann['bbox']))
                M_arr[y1 : y2, x1 : x2] = 0.0
            Q.append({'im_arr': im_arr, 'M_arr': M_arr, 'fn': fn, 'annotations': anns})
        if i in background_idx:
            break
    for pair in tqdm.tqdm(Q, ascii=True):
        fn = os.path.basename(pair['fn'])
        im_arr, M_arr = pair['im_arr'], pair['M_arr']
        for x in range(im_arr.shape[0]):
            for y in range(im_arr.shape[1]):
                if M_arr[x, y, 0] < 0.5:
                    im_arr[x, y] = np.uint8(im_arr[x, y].mean() * 0.3)
                    im_arr[x, y, 0] *= 3
        skimage.io.imsave(os.path.join(outputdir, fn + '_mask.jpg'), im_arr, quality=80)
    im_arr = skimage.io.imread(os.path.join(outputdir, '..', 'pngs', '00002700.jpg_background.jpg'))
    M_arr = skimage.io.imread(os.path.join(outputdir, '..', 'pngs', '00002700.jpg_mask.png'))
    for x in range(im_arr.shape[0]):
        for y in range(im_arr.shape[1]):
            if M_arr[x, y] < 0.5:
                im_arr[x, y] = np.uint8(im_arr[x, y].mean() * 0.3)
                im_arr[x, y, 0] *= 3
    skimage.io.imsave(os.path.join(outputdir, 'background_mask.jpg'), im_arr, quality=80)


def show_mask():
    outputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', '001', 'example'))
    im_back = skimage.io.imread(os.path.join(outputdir, 'background_inpaint.jpg'))
    im = skimage.io.imread(os.path.join(outputdir, 'objectmask_origin.jpg'))
    im_diff = (im.astype(np.float16) - im_back)
    im_diff = ((im_diff + 255) * 0.5).astype(np.uint8)
    skimage.io.imsave(os.path.join(outputdir, 'objectmask_mask.jpg'), im_diff, quality=80)


def _read_ap_gain():
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        results_base = json.load(fp)['r101-fpn-3x']
    methods = {
        'ST': ['paper_baselines', 'baseline_cvpr19_r101', 'results_AP.json'],
        'STAC': ['paper_baselines', 'arxiv2020_stac_r101', 'results_AP.json'],
        'AT': ['paper_baselines', 'cvpr22_adaptive_teacher_r101', 'results_AP.json'],
        'H$^2$FA': ['paper_baselines', 'cvpr22_h2fa_rcnn_r101', 'results_AP.json'],
        'TIA': ['paper_baselines', 'cvpr22_tia_iters250_r101', 'results_AP.json'],
        'LODS': ['paper_baselines', 'cvpr22_LODS_iter250_r101', 'results_AP.json'],

        'PL+MX+MF': ['fusion_coco_mask_inpaint', 'object_diff_midfusion_mixup_r101', 'results_AP_merge_last.json'],

        'PL': ['baseline_crossteach_r101', 'results_AP.json'],
        'PL+MX': ['mixup_ablation', 'mixup_r101_p0.3_r0.5_overlap0.65', 'results_AP.json'],
        'PL+EF': ['fusion_coco_mask_inpaint', 'object_diff_earlyfusion_r101', 'results_AP_last.json'],
        'PL+MF': ['fusion_coco_mask_inpaint', 'object_diff_midfusion_r101', 'results_AP_merge_last.json'],
        'PL+LF': ['fusion_coco_mask_inpaint', 'object_diff_latefusion_r101', 'results_AP_merge_last.json'],
    }
    mAP_improvements, min_impr, max_impr = {}, 100, -100
    for m in methods:
        print('reading results of', m)
        with open(os.path.join(*(['/mnt', 'f', 'intersections_results'] + methods[m])), 'r') as fp:
            results = json.load(fp)
        assert len(results) == 100, results.keys()
        mAP_improvements[m] = 100 * np.array([results[v]['results']['weighted'][0] - results_base['manual_' + v]['results']['weighted'][0] for v in results])
        min_impr = min(min_impr, mAP_improvements[m].min())
        max_impr = max(max_impr, mAP_improvements[m].max())
    return mAP_improvements, min_impr, max_impr


def ap_gain_distribution():
    mAP_improvements, _, _ = _read_ap_gain()
    bins = np.arange(-35, 20.1, 5)
    plt.figure(figsize=(8, 4.5))
    legends = []
    for m in ['ST', 'STAC', 'AT', 'H$^2$FA', 'TIA', 'PL+MX+MF']:
        hist, xs = np.histogram(mAP_improvements[m], bins=bins)
        assert hist.sum() == 100, '%s %s %s' % (hist.sum(), mAP_improvements[m].min(), mAP_improvements[m].max())
        cdf = np.cumsum(hist)
        plt.plot(xs[1:], cdf)
        legends.append(m)
    plt.legend(legends)
    plt.tight_layout()
    plt.show()


def ap_gain_correlation():
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    mAP_improvements, _, _ = _read_ap_gain()
    for m in ['ST', 'STAC', 'AT', 'H$^2$FA', 'TIA', 'LODS', 'PL', 'PL+MX', 'PL+EF', 'PL+MF', 'PL+LF']:
        apg_x = mAP_improvements[m]
        apg_y = mAP_improvements['PL+MX+MF']
        xy_range = max(apg_x.max() - apg_x.min(), apg_y.max() - apg_y.min())

        linear = LinearRegression().fit(apg_x.reshape(-1, 1), apg_y.reshape(-1, 1))
        k, b = linear.coef_[0, 0], linear.intercept_[0]
        r, _ = pearsonr(apg_x, apg_y)
        print(m, k, b, r)

        x_range, y_range = [apg_x.min() - xy_range * 0.05, apg_x.min() + xy_range * 1.05], [apg_y.min() - xy_range * 0.05, apg_y.min() + xy_range * 1.05]
        plt.figure(figsize=(4, 4))
        plt.plot(x_range, [0, 0], 'k-', lw=0.5, alpha=0.75)
        plt.plot([0, 0], y_range, 'k-', lw=0.5, alpha=0.75)
        plt.scatter(apg_x, apg_y, marker='+', s=100, c='blue')
        plt.plot([-100, 100], [-100 * k + b, 100 * k + b], 'r-', lw=2, alpha=0.75)
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.xlabel('$APG^m_w$ of %s' % m)
        plt.ylabel('$APG^m_w$ of PL+MX+MF')
        plt.title('Pearson $r=%.4f$' % r)
        # plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig('apg_correlation_%s.pdf' % m.replace('$', '').replace('^', '').replace('+', '_'))


def _check_overlap(mask, bbox):
    if mask is None:
        return False
    x1, y1, x2, y2 = map(int, bbox)
    H, W = mask.shape
    x1 = min(max(x1, 0), W - 1)
    x2 = min(max(x2, 0), W - 1)
    y1 = min(max(y1, 0), H - 1)
    y2 = min(max(y2, 0), H - 1)
    return (mask[y1, x1] > 1e-3 or mask[y1, x2] > 1e-3 or mask[y2, x1] > 1e-3 or mask[y2, x2] > 1e-3)

def pseudo_box_quality():
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    import imantics
    from finetune import get_annotation_dict, refine_annotations

    # with open(os.path.join(os.path.dirname(__file__), '..', '..', 'masks.json'), 'r') as fp:
    #     masks = json.load(fp)
    # masks = {m['video']: m['polygons'] for m in masks}

    # args = Dummy()
    # args.anno_models = ['r101-fpn-3x', 'r50-fpn-3x']
    # args.refine_det_score_thres = 0.5
    # args.refine_iou_thres = 0.85
    # args.refine_remove_no_sot = True

    # box_per_image = {}
    # for vid in video_id_list:
    #     args.id = vid
    #     images_manual = get_annotation_dict(args)
    #     images_pseudo, _ = refine_annotations(args)
    #     for im in images_manual + images_pseudo:
    #         for ann in im['annotations']:
    #             assert ann['bbox_mode'] == BoxMode.XYXY_ABS
    #     box_count_manual = sum(list(map(lambda x: len(x['annotations']), images_manual)))
    #     box_count_pseudo = sum(list(map(lambda x: len(x['annotations']), images_pseudo)))
    #     print('video %s:         manual %d images %d bboxes, pseudo %d images %d bboxes' % (vid, len(images_manual), box_count_manual, len(images_pseudo), box_count_pseudo))

    #     if len(masks[vid]) > 0:
    #         m_arr = imantics.Annotation.from_polygons(masks[vid], image=imantics.Image(skimage.io.imread(images_manual[0]['file_name']))).array.astype(np.float16)
    #         for im in images_manual + images_pseudo:
    #             im['annotations'] = list(filter(lambda x: not _check_overlap(m_arr, x['bbox']), im['annotations']))
    #     box_count_manual = sum(list(map(lambda x: len(x['annotations']), images_manual)))
    #     box_count_pseudo = sum(list(map(lambda x: len(x['annotations']), images_pseudo)))
    #     print('video %s trimmed: manual %d images %d bboxes, pseudo %d images %d bboxes' % (vid, len(images_manual), box_count_manual, len(images_pseudo), box_count_pseudo))
    #     box_per_image[vid] = {'manual': box_count_manual / len(images_manual), 'pseudo': box_count_pseudo / len(images_pseudo)}

    box_per_image = {'001': {'manual': 25.357142857142858, 'pseudo': 37.34154275092937}, '003': {'manual': 8.95774647887324, 'pseudo': 12.709166915497164}, '005': {'manual': 16.270833333333332, 'pseudo': 31.55661218424963}, '006': {'manual': 57.3, 'pseudo': 46.58150074294205}, '007': {'manual': 39.57142857142857, 'pseudo': 47.224665676077265}, '008': {'manual': 105.83333333333333, 'pseudo': 66.96679049034175}, '009': {'manual': 54.78947368421053, 'pseudo': 43.842644873699854}, '011': {'manual': 11.0, 'pseudo': 18.954521654574016}, '012': {'manual': 26.96551724137931, 'pseudo': 36.86745913818722}, '013': {'manual': 3.2566844919786098, 'pseudo': 6.718338399189463}, '014': {'manual': 5.135593220338983, 'pseudo': 13.983416375399717}, '015': {'manual': 30.61904761904762, 'pseudo': 44.30794947994057}, '016': {'manual': 35.27272727272727, 'pseudo': 61.94450222882615}, '017': {'manual': 11.438356164383562, 'pseudo': 31.682763744427934}, '019': {'manual': 202.75, 'pseudo': 65.05034623217922}, '020': {'manual': 29.344827586206897, 'pseudo': 24.357206537890043}, '023': {'manual': 1.309090909090909, 'pseudo': 2.2148389726865063}, '025': {'manual': 14.754716981132075, 'pseudo': 36.099108469539374}, '027': {'manual': 36.41379310344828, 'pseudo': 34.65037147102526}, '034': {'manual': 11.512820512820513, 'pseudo': 17.534646790021725}, '036': {'manual': 27.866666666666667, 'pseudo': 30.786552748885587}, '039': {'manual': 39.206896551724135, 'pseudo': 29.72364312267658}, '040': {'manual': 30.06451612903226, 'pseudo': 40.37050520059435}, '043': {'manual': 12.758620689655173, 'pseudo': 17.504642803654335}, '044': {'manual': 35.470588235294116, 'pseudo': 39.35291493158834}, '046': {'manual': 23.483870967741936, 'pseudo': 32.28997028231798}, '048': {'manual': 24.441176470588236, 'pseudo': 35.526864893062076}, '049': {'manual': 87.46666666666667, 'pseudo': 53.54260938115885}, '050': {'manual': 50.53333333333333, 'pseudo': 58.22644873699851}, '051': {'manual': 37.0, 'pseudo': 53.23759286775631}, '053': {'manual': 26.424242424242426, 'pseudo': 39.784323922734025}, '054': {'manual': 10.525423728813559, 'pseudo': 9.76988015376498}, '055': {'manual': 12.526315789473685, 'pseudo': 24.25715134854001}, '056': {'manual': 23.4, 'pseudo': 44.43116160096299}, '058': {'manual': 14.279069767441861, 'pseudo': 21.946136701337295}, '059': {'manual': 29.192307692307693, 'pseudo': 49.61426448736999}, '060': {'manual': 23.714285714285715, 'pseudo': 29.61820208023774}, '066': {'manual': 20.026315789473685, 'pseudo': 32.054680534918276}, '067': {'manual': 19.53846153846154, 'pseudo': 26.038187221396733}, '068': {'manual': 21.34375, 'pseudo': 29.243507462686566}, '069': {'manual': 63.733333333333334, 'pseudo': 65.65594951923077}, '070': {'manual': 20.56756756756757, 'pseudo': 35.608320950965826}, '071': {'manual': 32.73913043478261, 'pseudo': 40.05891530460624}, '073': {'manual': 12.114583333333334, 'pseudo': 10.958686283251598}, '074': {'manual': 18.60377358490566, 'pseudo': 32.14561664190193}, '075': {'manual': 5.877358490566038, 'pseudo': 14.652897473997028}, '076': {'manual': 28.333333333333332, 'pseudo': 41.690638930163445}, '077': {'manual': 10.031578947368422, 'pseudo': 8.527813527813528}, '080': {'manual': 22.78787878787879, 'pseudo': 28.960326894502227}, '085': {'manual': 14.82857142857143, 'pseudo': 16.746508172362557}, '086': {'manual': 20.38888888888889, 'pseudo': 23.30364041604755}, '087': {'manual': 62.142857142857146, 'pseudo': 74.60178306092125}, '088': {'manual': 13.232323232323232, 'pseudo': 13.799954257833345}, '090': {'manual': 12.572916666666666, 'pseudo': 10.490809968847351}, '091': {'manual': 10.699248120300751, 'pseudo': 7.981314672820665}, '092': {'manual': 13.8125, 'pseudo': 11.146943038483169}, '093': {'manual': 9.229166666666666, 'pseudo': 8.57260228034876}, '094': {'manual': 5.522222222222222, 'pseudo': 6.0861683437150615}, '095': {'manual': 30.64, 'pseudo': 42.74665676077266}, '098': {'manual': 6.947058823529412, 'pseudo': 10.919457978075517}, '099': {'manual': 57.88235294117647, 'pseudo': 58.19895988112927}, '105': {'manual': 6.602150537634409, 'pseudo': 16.93150074294205}, '108': {'manual': 4.688622754491018, 'pseudo': 9.691603660962992}, '110': {'manual': 8.6875, 'pseudo': 9.254616212635588}, '112': {'manual': 21.736842105263158, 'pseudo': 30.53038632986627}, '114': {'manual': 14.902439024390244, 'pseudo': 25.85648113559955}, '115': {'manual': 10.25, 'pseudo': 18.849973705957478}, '116': {'manual': 62.13333333333333, 'pseudo': 62.355794947994056}, '117': {'manual': 19.02777777777778, 'pseudo': 39.08997028231798}, '118': {'manual': 24.71875, 'pseudo': 10.47423573222482}, '125': {'manual': 16.04255319148936, 'pseudo': 34.318796433878155}, '127': {'manual': 15.307692307692308, 'pseudo': 29.287964338781574}, '128': {'manual': 12.392156862745098, 'pseudo': 32.30401188707281}, '129': {'manual': 8.64516129032258, 'pseudo': 20.767292750999474}, '130': {'manual': 17.205128205128204, 'pseudo': 25.25676077265973}, '131': {'manual': 64.28571428571429, 'pseudo': 62.602080237741454}, '132': {'manual': 27.37037037037037, 'pseudo': 22.160846953937593}, '135': {'manual': 16.14, 'pseudo': 30.985884101040117}, '136': {'manual': 19.952380952380953, 'pseudo': 26.517533432392273}, '141': {'manual': 23.214285714285715, 'pseudo': 32.61894502228826}, '146': {'manual': 43.111111111111114, 'pseudo': 47.67808320950966}, '148': {'manual': 48.705882352941174, 'pseudo': 38.8005515390922}, '149': {'manual': 44.375, 'pseudo': 41.961478396668404}, '150': {'manual': 12.155172413793103, 'pseudo': 28.350668647845467}, '152': {'manual': 12.582089552238806, 'pseudo': 14.486313868613138}, '154': {'manual': 27.75862068965517, 'pseudo': 36.526617100371745}, '156': {'manual': 8.075471698113208, 'pseudo': 11.54955414012739}, '158': {'manual': 41.21052631578947, 'pseudo': 45.63781575037147}, '159': {'manual': 5.9921875, 'pseudo': 9.079103436246294}, '160': {'manual': 27.03846153846154, 'pseudo': 20.994650817236256}, '161': {'manual': 22.536585365853657, 'pseudo': 17.11292719167905}, '164': {'manual': 9.338842975206612, 'pseudo': 27.21607658410576}, '167': {'manual': 2.949748743718593, 'pseudo': 4.892535197001925}, '169': {'manual': 12.30188679245283, 'pseudo': 20.835066864784547}, '170': {'manual': 14.595744680851064, 'pseudo': 18.57176820208024}, '171': {'manual': 10.956521739130435, 'pseudo': 23.22511144130758}, '172': {'manual': 8.0, 'pseudo': 19.837494427106552}, '175': {'manual': 63.06666666666667, 'pseudo': 33.274961721331294}, '178': {'manual': 14.436363636363636, 'pseudo': 17.41032689450223}, '179': {'manual': 24.097560975609756, 'pseudo': 20.213075780089152}}
    print(box_per_image)
    box_per_image_manual = np.array(list(map(lambda x: x['manual'], box_per_image.values())))
    box_per_image_pseudo = np.array(list(map(lambda x: x['pseudo'], box_per_image.values())))
    linear = LinearRegression().fit(box_per_image_manual.reshape(-1, 1), box_per_image_pseudo.reshape(-1, 1))
    k, b = linear.coef_[0, 0], linear.intercept_[0]
    r, _ = pearsonr(box_per_image_manual, box_per_image_pseudo)
    print(k, b, r)

    xy_max = max(box_per_image_manual.max(), box_per_image_pseudo.max())
    plt.figure(figsize=(8, 8))
    plt.plot([-100, 1000], [-100, 1000], 'k-', lw=0.5, alpha=0.75)
    plt.scatter(box_per_image_manual, box_per_image_pseudo, marker='+', s=100, c='blue')
    plt.plot([-100, 1000], [-100 * k + b, 1000 * k + b], 'r-', lw=2, alpha=0.75)
    plt.xlim([-0.02 * xy_max, 1.02 * xy_max])
    plt.ylim([-0.02 * xy_max, 1.02 * xy_max])
    plt.xlabel('manual annotation boxes per image')
    plt.ylabel('pseudo-labeling boxes per image')
    plt.title('Pearson $r=%.4f$' % r)
    # plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('pseudo_manual_boxes.pdf')


def compare_backbones():
    backbones = [
        {'name': 'R-18',  'params': 28.29, 'cpu': 0.630, 'titan': 0.080, '4090': 0.056, 'mAP_COCO': 45.64, 'mAP_scenes100': 35.68},
        {'name': 'R-34',  'params': 38.41, 'cpu': 0.654, 'titan': 0.085, '4090': 0.058, 'mAP_COCO': 49.26, 'mAP_scenes100': 38.78},
        {'name': 'R-50',  'params': 41.41, 'cpu': 0.888, 'titan': 0.101, '4090': 0.061, 'mAP_COCO': 50.05, 'mAP_scenes100': 41.28},
        {'name': 'R-101', 'params': 60.45, 'cpu': 0.978, 'titan': 0.125, '4090': 0.073, 'mAP_COCO': 51.29, 'mAP_scenes100': 41.96},
        {'name': 'R-152', 'params': 76.14, 'cpu': 1.176, 'titan': 0.113, '4090': 0.079, 'mAP_COCO': 51.72, 'mAP_scenes100': 40.92},
        {'name': 'X-50',  'params': 40.89, 'cpu': 0.930, 'titan': 0.088, '4090': 0.061, 'mAP_COCO': 49.51, 'mAP_scenes100': 39.59},
        {'name': 'X-101', 'params':104.79, 'cpu': 1.460, 'titan': 0.504, '4090': 0.084, 'mAP_COCO': 52.35, 'mAP_scenes100': 44.48},
    ]
    names = [_d['name'] for _d in backbones]
    params = [_d['params'] for _d in backbones]
    times_cpu = [_d['cpu'] for _d in backbones]
    times_titan = [_d['titan'] for _d in backbones]
    times_4090 = [_d['4090'] for _d in backbones]
    mAP_coco = [_d['mAP_COCO'] for _d in backbones]
    mAP_w = [_d['mAP_scenes100'] for _d in backbones]

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    xy = list(zip(names, params, mAP_coco, mAP_w))
    xy.sort(key=lambda t: t[1])
    plt.plot([t[2] for t in xy], [t[1] for t in xy], 'ko-', ms=10)
    plt.plot([t[3] for t in xy], [t[1] for t in xy], 'bo-', ms=10)
    plt.legend(['$mAP$ on remapped MSCOCO-2017', '$mAP$ on Scenes100 weighted'])
    for i in range(0, len(xy)):
        plt.text(xy[i][2] + 0.1, xy[i][1] - 2, xy[i][0], rotation=-30, horizontalalignment='left', verticalalignment='center', size=16, color='red')
        plt.text(xy[i][3] + 0.1, xy[i][1] - 2, xy[i][0], rotation=-30, horizontalalignment='left', verticalalignment='center', size=16, color='red')
    plt.xlim(35, 55)
    plt.ylim(25, 110)
    plt.xlabel('mean average precision')
    plt.ylabel('$\\times 10^6$ parameters')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    xy = list(zip(names, params, times_4090, times_titan, times_cpu))
    xy.sort(key=lambda t: t[1])
    plt.plot([1/30, 1/30], [0, 120], 'k--', alpha=0.5)
    plt.plot([1/10, 1/10], [0, 120], 'k:', alpha=0.5)
    plt.plot([t[2] for t in xy], [t[1] for t in xy], 'kx-', ms=10)
    plt.plot([t[3] for t in xy], [t[1] for t in xy], 'bx-', ms=10)
    plt.plot([t[4] for t in xy], [t[1] for t in xy], 'mx-', ms=10)
    plt.legend(['30 FPS', '10 FPS', 'fast GPU', 'slow GPU', 'CPU'])
    for i in range(0, len(xy)):
        plt.text(xy[i][3] + 0.03, xy[i][1], xy[i][0], rotation=0, horizontalalignment='left', verticalalignment='center', size=16, color='red')
    plt.xlim(0, 1.5)
    plt.ylim(25, 110)
    plt.xlabel('inference seconds/frame')
    plt.ylabel('$\\times 10^6$ parameters')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def compare_fusion_throughput():
    class ARGS(object): pass
    images = []
    for video_id in ['001', '003', '019', '048', '090']:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            im = json.load(fp)[-1]
        im['file_name'] = os.path.join(inputdir, 'unmasked', im['file_name'])
        im['image'] = detectron2.data.detection_utils.read_image(im['file_name'], format='BGR')
        im['H'], im['W'] = im['image'].shape[:2]
        file_name_background = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', video_id, 'inpaint', '*inpaint.jpg')))[-1]
        image_background = detectron2.data.detection_utils.read_image(file_name_background, format='BGR')
        im['image'] = torch.as_tensor(im['image'].astype('float32').transpose(2, 0, 1))
        im['image_wdiff'] = torch.cat([im['image'], torch.as_tensor(image_background.astype('float32').transpose(2, 0, 1))], dim=0)
        images.append(im)
    print(len(images), 'images')
    N1, N2 = 200, 1500
    cfg = get_cfg_base_model('r101-fpn-3x', ckpt='../../models/mscoco2017_remap_r101-fpn-3x.pth')

    model = detectron2.modeling.build_model(cfg)
    model.eval()
    checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    delta_t = None
    for i in tqdm.tqdm(range(0, N2 + 10), ascii=True, desc='none-fusion'):
        if i == N1: delta_t = time.perf_counter()
        if i == N2: delta_t = time.perf_counter() - delta_t
        instances = model.inference([{'image': images[i % len(images)]['image'], 'height': images[i % len(images)]['H'], 'width': images[i % len(images)]['W']}])[0]
    print('none-fusion:', (N2 - N1) / delta_t, '\n')
    del model

    from finetune_wdiff_earlyfusion import GeneralizedRCNNFinetuneBackground
    cfg.MODEL.WEIGHTS = '../../models/mscoco2017_remap_wdiff_earlyfusion_r101-fpn-3x.pth'
    model = detectron2.modeling.build_model(cfg)
    model = GeneralizedRCNNFinetuneBackground.create_from_sup(model)
    model.eval()
    checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    delta_t = None
    for i in tqdm.tqdm(range(0, N2 + 10), ascii=True, desc='early-fusion'):
        if i == N1: delta_t = time.perf_counter()
        if i == N2: delta_t = time.perf_counter() - delta_t
        instances = model.inference([{'image': images[i % len(images)]['image_wdiff'], 'height': images[i % len(images)]['H'], 'width': images[i % len(images)]['W']}])[0]
    print('early-fusion:', (N2 - N1) / delta_t, '\n')
    del model

    from finetune_wdiff_midfusion import GeneralizedRCNNFinetuneBackground
    cfg.MODEL.WEIGHTS = '../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth'
    model = detectron2.modeling.build_model(cfg)
    model = GeneralizedRCNNFinetuneBackground.create_from_sup(model, 'average', None)
    model.eval()
    checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.proposal_generator = model.roi_heads = torch.nn.Identity()
    delta_t = None
    for i in tqdm.tqdm(range(0, N2 + 10), ascii=True, desc='mid-fusion'):
        if i == N1: delta_t = time.perf_counter()
        if i == N2: delta_t = time.perf_counter() - delta_t
        instances = model.inference([{'image': images[i % len(images)]['image_wdiff'], 'height': images[i % len(images)]['H'], 'width': images[i % len(images)]['W']}], single_image=True)[0]
    print('mid-fusion:', (N2 - N1) / delta_t, '\n')
    del model

    from finetune_wdiff_latefusion import GeneralizedRCNNFinetuneBackground
    cfg.MODEL.WEIGHTS = '../../models/mscoco2017_remap_wdiff_latefusion_r101-fpn-3x.pth'
    model = detectron2.modeling.build_model(cfg)
    model = GeneralizedRCNNFinetuneBackground.create_from_sup(model, None)
    model.eval()
    checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.roi_heads = torch.nn.Identity()
    delta_t = None
    for i in tqdm.tqdm(range(0, N2 + 10), ascii=True, desc='late-fusion'):
        if i == N1: delta_t = time.perf_counter()
        if i == N2: delta_t = time.perf_counter() - delta_t
        instances = model.inference([{'image': images[i % len(images)]['image_wdiff'], 'height': images[i % len(images)]['H'], 'width': images[i % len(images)]['W']}], single_image=True)[0]
    print('late-fusion:', (N2 - N1) / delta_t, '\n')
    del model


def case_study():
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        results_base = json.load(fp)['r101-fpn-3x']
    with open(os.path.join('F:\\', 'intersections_results', 'fusion_coco_mask_inpaint', 'object_diff_midfusion_mixup_r101', 'results_AP_merge_dynamic.json'), 'r') as fp:
        results_best = json.load(fp)
    assert len(results_best) == 100, results_best.keys()
    mAP_improvements = [[v, results_base['manual_' + v]['results']['weighted'][0] * 100, results_best[v]['results']['weighted'][0] * 100] for v in results_best]
    for apg in mAP_improvements:
        apg.append(apg[2] - apg[1])
    del apg
    mAP_improvements.sort(key=lambda x: -1 * x[3])

    from finetune_wdiff_midfusion import PredictorBackground, construct_image_w_background
    detector_base = DefaultPredictor(get_cfg_base_model('r101-fpn-3x'))
    score_thres = 0.5
    for i in [0, 1, 2, 3, 95, 96, 97, 98, 99]:
        apg = mAP_improvements[i]
        outputdir = os.path.join('F:\\', 'intersections_results', 'case_study', 'case%02d_%.2f_to_%.2f' % (i, apg[1], apg[2]))
        if not os.access(outputdir, os.W_OK):
            os.mkdir(outputdir)
        pth_v = list(glob.glob(os.path.join('F:\\', 'intersections_results', 'fusion_coco_mask_inpaint', 'object_diff_midfusion_mixup_r101', 'adapt' + apg[0] + '*.pth')))
        assert len(pth_v) == 1
        args = Dummy()
        args.fusion_type = 'average'
        detector_v = PredictorBackground(get_cfg_base_model('r101-fpn-3x', ckpt=pth_v[0]), args)

        print(i, apg)
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', apg[0])
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        for j, im in tqdm.tqdm(enumerate(annotations), ascii=True):
            dets_base, dets_best = [], []
            image = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
            image_background = detectron2.data.detection_utils.read_image(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_background_lmdb', apg[0], 'inpaint', im['file_name'] + '_inpaint.jpg')), format='BGR')
            image, image_background, image_diff = construct_image_w_background(image, image_background)

            instances = detector_base(image)['instances'].to('cpu')
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            for k in range(0, len(label)):
                if score[k] < score_thres: continue
                dets_base.append({'bbox': bbox[k], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[k], 'score': score[k]})

            instances = detector_v(image, image_diff)['merge']['instances'].to('cpu')
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            for k in range(0, len(label)):
                if score[k] < score_thres: continue
                dets_best.append({'bbox': bbox[k], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[k], 'score': score[k]})

            skimage.io.imsave(os.path.join(outputdir, '%s_%s_diff.jpg' % (apg[0], im['file_name'][:-4])), image_diff[:, :, ::-1], quality=80)
            skimage.io.imsave(os.path.join(outputdir, '%s_%s_base.jpg' % (apg[0], im['file_name'][:-4])), draw_bbox(image[:, :, ::-1], dets_base), quality=80)
            skimage.io.imsave(os.path.join(outputdir, '%s_%s_best.jpg' % (apg[0], im['file_name'][:-4])), draw_bbox(image[:, :, ::-1], dets_best), quality=80)


def dataset_mosaic():
    all_annotations = []
    for vid in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', vid)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        annotations = list(filter(lambda x: len(x['annotations']) > 0, annotations))
        for i in range(0, len(annotations)):
            annotations[i]['file_name'] = os.path.join(inputdir, 'unmasked', annotations[i]['file_name'])
        all_annotations.append(annotations)
    print(len(all_annotations))
    T = 24
    for t in list(range(0, T))[::-1]:
        print('\n', t)
        im_mosaic = np.zeros(shape=(3600, 6400, 3), dtype=np.float64)
        for i in range(0, 100):
            t_i = int((len(all_annotations[i]) - 1) * t / (T - 1))
            im = skimage.io.imread(all_annotations[i][t_i]['file_name'])
            im = draw_bbox(im, all_annotations[i][t_i]['annotations'])
            if abs(im.shape[1] / im.shape[0] - 16 / 9) > 0.01:
                H, W = im.shape[:2]
                if W / H > 16 / 9:
                    W2 = int(H / 9 * 16)
                    im = im[:, (W - W2) // 2 : (W - W2) // 2 + W2, :]
                else:
                    H2 = int(W / 16 * 9)
                    im = im[(H - H2) // 2 : (H - H2) // 2 + H2, :, :]
            im = skimage.transform.resize(im, (360, 640), anti_aliasing=True)
            x, y = i // 10, i % 10
            print(i, x, y, end=' ')
            im_mosaic[x * 360 : (x + 1) * 360, y * 640 : (y + 1) * 640] = im
        im_mosaic = (skimage.transform.resize(im_mosaic, (2160, 3840), anti_aliasing=True) * 255).astype(np.uint8)
        skimage.io.imsave('C:\\Users\\zhang\\Desktop\\TMP\\mosaic_%02d.png' % t, im_mosaic)
        # ffmpeg -framerate 4 -f image2 -i mosaic_%02d.png -c:v libvpx-vp9 -crf 24 -pix_fmt yuva420p mosaic.webm

def dataset_mosaic_gif():
    import imageio
    with imageio.get_writer('/mnt/c/Users/zhang/Desktop/TMP/scenes100.gif', duration=0.25, mode='I') as writer:
        reader = skvideo.io.vreader(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scenes100', 'media', 'mosaic.webm'))
        for frame in reader:
            frame = (skimage.transform.resize(frame, (1080, 1920), anti_aliasing=True) * 255).astype(np.uint8)
            writer.append_data(frame)


def dataset_mosaic_png():
    W_target, H_target = 1920, 1080
    all_annotations = []
    for vid in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', vid)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        annotations = list(filter(lambda x: len(x['annotations']) > 0, annotations))
        for i in range(0, len(annotations)):
            annotations[i]['file_name'] = os.path.join(inputdir, 'unmasked', annotations[i]['file_name'])
        all_annotations.append(annotations)
    print(len(all_annotations))
    im_mosaic = np.zeros(shape=(10 * H_target, 10 * W_target, 3), dtype=np.uint8)
    for i in tqdm.tqdm(range(0, 100), ascii=True):
        im = skimage.io.imread(all_annotations[i][0]['file_name'])
        im = draw_bbox(im, all_annotations[i][0]['annotations'])
        if abs(im.shape[1] / im.shape[0] - 16 / 9) > 0.01:
            H, W = im.shape[:2]
            if W / H > 16 / 9:
                W2 = int(H / 9 * 16)
                im = im[:, (W - W2) // 2 : (W - W2) // 2 + W2, :]
            else:
                H2 = int(W / 16 * 9)
                im = im[(H - H2) // 2 : (H - H2) // 2 + H2, :, :]
        im = (skimage.transform.resize(im, (H_target, W_target), anti_aliasing=True) * 255).astype(np.uint8)
        x, y = i // 10, i % 10
        im_mosaic[x * H_target : (x + 1) * H_target, y * W_target : (y + 1) * W_target] = im
    skimage.io.imsave('C:\\Users\\zhang\\Desktop\\TMP\\mosaic.jpg', im_mosaic, quality=85)


def dataset_slides_gif():
    import imageio
    for video_id in ['001', '017', '056', '141']:
        images = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', video_id, 'jpegs', '*.jpg')))[:60]
        skimage.io.imsave('/mnt/c/Users/zhang/Desktop/TMP/%s.jpg' % video_id, (skimage.transform.resize(skimage.io.imread(images[0]), (900, 1600), anti_aliasing=True) * 255).astype(np.uint8), quality=85)
        with imageio.get_writer('/mnt/c/Users/zhang/Desktop/TMP/%s.gif' % video_id, duration=0.2, mode='I') as writer:
            for f in tqdm.tqdm(images, ascii=True, desc=video_id):
                im = skimage.io.imread(f)
                im = (skimage.transform.resize(im, (450, 800), anti_aliasing=True) * 255).astype(np.uint8)
                writer.append_data(im)


def ap_gain_cdf():
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        results_base = json.load(fp)['r101-fpn-3x']
    # mAP_improvements, min_impr, max_impr = {}, 100, -100
    with open(os.path.join('E:\\intersections_results', 'object_diff_midfusion_mixup_r101', 'results_AP_merge_last.json'), 'r') as fp:
        results = json.load(fp)
    assert len(results) == 100, results.keys()
    mAP_improvements = 100 * np.array([results[v]['results']['weighted'][0] - results_base['manual_' + v]['results']['weighted'][0] for v in results])
    bins = np.arange(-5, 11.1, 1)
    print(mAP_improvements, mAP_improvements.max(), mAP_improvements.min(), bins)
    hist, xs = np.histogram(mAP_improvements, bins=bins)
    assert hist.sum() == 100, '%s %s %s' % (hist.sum(), mAP_improvements[m].min(), mAP_improvements[m].max())
    cdf = np.cumsum(hist)
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.bar(xs[1:], hist, width=0.98, color='b')
    plt.plot(xs[1:], cdf, lw=3, color='r')
    plt.xlim(-5, 12)
    plt.ylim(0, 102)
    plt.ylabel('percentage')
    plt.xlabel('$mAP$-gain on individual scene')
    plt.legend(['distribution histogram', 'cumulative distribution'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # stats_coco('../../../MSCOCO2017', 'train')
    # stats_coco('../../../MSCOCO2017', 'valid')
    # base_perform_coco('r50-fpn-3x')
    # base_perform_coco('r101-fpn-3x')
    # args = Dummy()
    # args.id = '001'
    # args.anno_models = ['r101-fpn-3x', 'r50-fpn-3x']
    # args.refine_det_score_thres = 0.5
    # args.refine_iou_thres = 0.85
    # refine_annotations(args)
    # args = Dummy()
    # args.id = '001'
    # args.anno_models = ['r101-fpn-3x', 'r50-fpn-3x']
    # args.refine_det_score_thres = 0.5
    # args.refine_iou_thres = 0.85
    # args.refine_remove_no_sot = False
    # args.mixup_r = 0.5
    # args.mixup_overlap_thres = 0.65
    # args.mixup_random_position = True
    # mixup(args)
    # inpaint_dynamic()
    # show_mask()
    # ap_gain_distribution()
    # ap_gain_correlation()
    # pseudo_box_quality()
    # compare_backbones()
    # compare_fusion_throughput()
    # case_study()
    # dataset_mosaic()
    # dataset_mosaic_gif()
    # dataset_mosaic_png()
    # coco_mosaic('../../../MSCOCO2017')
    # dataset_slides_gif()
    ap_gain_cdf()
