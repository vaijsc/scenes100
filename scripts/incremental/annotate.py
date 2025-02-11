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
import shutil
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import imantics
import networkx

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def copy_frames(training):
    basedir = os.path.normpath(os.path.dirname(__file__))
    if not training:
        with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
            vfilelist = json.load(fp)['days'][4]
        for vfilename in tqdm.tqdm(vfilelist, ascii=True):
            with open(os.path.join(basedir, 'frames', vfilename + '.json'), 'r') as fp:
                chunks = json.load(fp)['chunks']
            for idx_list in chunks:
                shutil.copyfile(os.path.join(basedir, 'frames', vfilename, '%08d.jpg' % idx_list[-1]), os.path.join(basedir, 'annotation', 'evaluation', '%s.%08d.jpg' % (vfilename, idx_list[-1])))
                shutil.copyfile(os.path.join(basedir, 'frames', 'background', '%s.%08d.inpaint.jpg' % (vfilename, idx_list[-1])), os.path.join(basedir, 'annotation', 'evaluation', '%s.%08d.inpaint.jpg' % (vfilename, idx_list[-1])))
    else:
        with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
            data = json.load(fp)['days']
        vfilelist = data[0] + data[1] + data[2] + data[3]
        for vfilename in tqdm.tqdm(vfilelist, ascii=True):
            with open(os.path.join(basedir, 'frames', vfilename + '.json'), 'r') as fp:
                idx_list = json.load(fp)['chunks'][0]
            shutil.copyfile(os.path.join(basedir, 'frames', vfilename, '%08d.jpg' % idx_list[-1]), os.path.join(basedir, 'annotation', 'training', '%s.%08d.jpg' % (vfilename, idx_list[-1])))
            shutil.copyfile(os.path.join(basedir, 'frames', 'background', '%s.%08d.inpaint.jpg' % (vfilename, idx_list[-1])), os.path.join(basedir, 'annotation', 'training', '%s.%08d.inpaint.jpg' % (vfilename, idx_list[-1])))


def mask_overlay(training):
    basedir = os.path.normpath(os.path.dirname(__file__))
    if not training:
        ifilelist = glob.glob(os.path.join(basedir, 'annotation', 'evaluation', '*.jpg'))
    else:
        ifilelist = glob.glob(os.path.join(basedir, 'annotation', 'training', '*.jpg'))
    ifilelist = list(filter(lambda f: f.find('inpaint') < 0, ifilelist))
    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        polygons = json.load(fp)['mask']
    ann = imantics.Annotation.from_polygons(polygons, image=imantics.Image.from_path(ifilelist[0]))
    M_arr = np.expand_dims(ann.array.astype(np.float16), 2)
    M_arr_overlay, M_arr_blank = M_arr * 0.3, 1.0 - (M_arr / M_arr.max())
    M_rgb = [0, 1, 0]
    for ifilename in tqdm.tqdm(ifilelist, ascii=True):
        im_arr = skimage.io.imread(ifilename)
        im_arr_overlay = ((1 - M_arr_overlay) * im_arr + M_arr_overlay * (np.array(M_rgb) * 255).astype(np.uint8).reshape(1, 1, 3)).astype(np.uint8)
        skimage.io.imsave(os.path.join(os.path.dirname(ifilename), os.path.basename(ifilename)[:-4] + '.mask.jpg'), im_arr_overlay, quality=80)
        im_arr_blank = (im_arr * M_arr_blank).astype(np.uint8)
        skimage.io.imsave(os.path.join(os.path.dirname(ifilename), os.path.basename(ifilename)[:-4] + '.blank.jpg'), im_arr_blank, quality=80)


def _check_overlap(mask, bbox):
    if mask is None:
        return False
    x1, y1, x2, y2 = map(int, bbox)
    H, W = mask.shape[:2]
    x1 = min(max(x1, 0), W - 1)
    x2 = min(max(x2, 0), W - 1)
    y1 = min(max(y1, 0), H - 1)
    y2 = min(max(y2, 0), H - 1)
    return (mask[y1, x1, 0] > 1e-3 or mask[y1, x2, 0] > 1e-3 or mask[y2, x1, 0] > 1e-3 or mask[y2, x2, 0] > 1e-3)


def draw_bbox(im, annotations, desc):
    fontsize, linewidth = int(min(im.shape[0], im.shape[1]) * 0.04), int(min(im.shape[0], im.shape[1]) / 300)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=fontsize)
    counts = {x : 0 for x in range(0, len(thing_classes))}
    im = Image.fromarray(im, 'RGB')
    draw = ImageDraw.Draw(im)
    for ann in annotations:
        assert ann['bbox_mode'] == BoxMode.XYXY_ABS
        x1, y1, x2, y2 = ann['bbox']
        cat = ann['category_id']
        draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[cat], width=linewidth)
        counts[cat] += 1
    desc = desc + ' ' + ' '.join(['%d %s(s)' % (counts[cat], thing_classes[cat]) for cat in range(0, len(thing_classes))])
    draw.text((6, 2), desc, fill='#FFFFFF', stroke_width=3, font=font)
    draw.text((6, 2), desc, fill='#000000', stroke_width=2, font=font)
    draw.text((6, 2), desc, fill='#FFFFFF', stroke_width=1, font=font)
    im = np.array(im)
    return im


def convert_splits(training):
    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        clips = json.load(fp)
    meta = clips['meta']
    polygons = clips['mask']
    with open(os.path.join(basedir, 'annotation', 'training' if training else 'evaluation', 'cvat.json'), 'r') as fp:
        anns_coco = json.load(fp)
    images, annotations, categories = anns_coco['images'], anns_coco['annotations'], anns_coco['categories']
    categories = [[cat['id'], cat['name']] for cat in categories]
    assert categories[0] == [1, 'person'] and categories[1] == [2, 'vehicle']
    print('categories:', categories)
    images_detectron2 = {im['id']: {'file_name': im['file_name'], 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []} for im in images}
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x2, y2 = x + w, y + h
        x = min(meta['W'] - 1, max(0, x))
        x2 = min(meta['W'] - 1, max(0, x2))
        y = min(meta['H'] - 1, max(0, y))
        y2 = min(meta['H'] - 1, max(0, y2))
        images_detectron2[ann['image_id']]['annotations'].append({'bbox': [x, y, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': ann['category_id'] - 1})
    images_detectron2 = list(images_detectron2.values())
    M_arr = np.expand_dims(imantics.Annotation.from_polygons(polygons, image=imantics.Image.from_path(os.path.join(basedir, 'annotation', 'training' if training else 'evaluation', images_detectron2[0]['file_name']))).array.astype(np.float16), 2)
    for im in images_detectron2:
        im['annotations'] = list(filter(lambda x: not _check_overlap(M_arr, x['bbox']), im['annotations']))
    print('%d images' % len(images_detectron2))
    images_detectron2 = sorted(images_detectron2, key=lambda x: x['file_name'])
    images_detectron2 = list(filter(lambda x: len(x['annotations']) > 0, images_detectron2))
    print('%d images after filtering unannotated' % len(images_detectron2))
    num_bboxes = [len(im['annotations']) for im in images_detectron2]
    print(num_bboxes)
    dense_thres = 30

    writer = skvideo.io.FFmpegWriter(os.path.join(basedir, 'annotation', 'training' if training else 'evaluation', 'annotation.mp4'), inputdict={'-r': '1'}, outputdict={'-vcodec': 'libx265', '-r': '1', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
    for im in tqdm.tqdm(images_detectron2, ascii=True, desc='writing video'):
        im['density'] = 'dense' if len(im['annotations']) > dense_thres else 'sparse'
        im_arr = skimage.io.imread(os.path.join(basedir, 'annotation', 'training' if training else 'evaluation', im['file_name']))
        im_arr = draw_bbox(im_arr, im['annotations'], '%s [%s]' % (im['file_name'], im['density']))
        writer.writeFrame(im_arr)
        im['file_name'] = im['file_name'][: -9] # only keep prefix
    writer.close()
    with open(os.path.join(basedir, 'annotation', 'training' if training else 'evaluation', 'detectron2.json'), 'w') as fp:
        json.dump(images_detectron2, fp)


def get_annotation_dict(mask=''):
    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'annotation', 'evaluation', 'detectron2.json'), 'r') as fp:
        images = json.load(fp)
    indicator = {'': '', 'overlay': '.mask', 'blank': '.blank'}[mask]
    images_dense, images_sparse = [], []
    for im in images:
        prefix = im['file_name']
        im['file_name'] = os.path.join(basedir, 'annotation', 'evaluation', prefix + indicator + '.jpg')
        im['file_name_background'] = os.path.join(basedir, 'annotation', 'evaluation', prefix + '.inpaint.jpg')
        if im['density'] == 'dense':
            images_dense.append(im)
        elif im['density'] == 'sparse':
            images_sparse.append(im)
        else:
            raise NotImplementedError
    print('SantaClausVillage manual evaluation dense : %d images, %d bboxes' % (len(images_dense), sum(map(lambda ann: len(ann['annotations']), images_dense))))
    print('SantaClausVillage manual evaluation sparse: %d images, %d bboxes' % (len(images_sparse), sum(map(lambda ann: len(ann['annotations']), images_sparse))))
    return images_dense, images_sparse


def get_annotation_dict_training(mask=''):
    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'annotation', 'training', 'detectron2.json'), 'r') as fp:
        images = json.load(fp)
    indicator = {'': '', 'overlay': '.mask', 'blank': '.blank'}[mask]
    headers = ['SantaClausVillage_20221202', 'SantaClausVillage_20221203', 'SantaClausVillage_20221204', 'SantaClausVillage_20221205']
    images_days = [[] for _ in headers]
    for im in images:
        prefix = im['file_name']
        im['file_name'] = os.path.join(basedir, 'annotation', 'training', prefix + indicator + '.jpg')
        im['file_name_background'] = os.path.join(basedir, 'annotation', 'training', prefix + '.inpaint.jpg')
        images_days[headers.index(prefix[: 26])].append(im)
    print('SantaClausVillage manual training')
    for i, images_i in enumerate(images_days):
        print(' - day %d: %d images, %d bboxes' % (i, len(images_i), sum(map(lambda ann: len(ann['annotations']), images_i))))
    return images_days


def calculate_AP(detections, split, annotations=None):
    import tempfile
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    basedir = os.path.normpath(os.path.dirname(__file__))
    assert split in ['dense', 'sparse']
    if annotations is None:
        annotations = get_annotation_dict()[0 if split == 'dense' else 1]
    assert len(annotations) == len(detections)
    annotations = sorted(annotations, key=lambda x: x['file_name'])
    detections = sorted(detections, key=lambda x: x['file_name'])
    for im1, im2 in zip(annotations, detections):
        assert os.path.basename(im1['file_name']) == os.path.basename(im2['file_name'])

    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        polygons = json.load(fp)['mask']
    M_arr = np.expand_dims(imantics.Annotation.from_polygons(polygons, image=imantics.Image.from_path(detections[0]['file_name'])).array.astype(np.float16), 2)
    for im in detections:
        for ann in im['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
        im['annotations'] = list(filter(lambda x: not _check_overlap(M_arr, x['bbox']), im['annotations']))

    coco_dict = {'info': {'year': 0, 'version': '', 'description': '', 'contributor': '', 'url': '', 'date_created': ''}, 'licenses': [{'id': 0, 'name': '', 'url': ''}], 'categories': None, 'images': None, 'annotations': None}
    coco_dict['categories'] = [{'id': i, 'name': thing_classes[i], 'supercategory': ''} for i in range(0, len(thing_classes))]
    coco_images, coco_annotations, coco_dets = [], [], []
    weight_per_class = [0 for _ in range(0, len(thing_classes))]
    for i in range(0, len(annotations)):
        coco_images.append({'id': i, 'width': 0, 'height': 0, 'file_name': '', 'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': ''})
        for ann in annotations[i]['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            x1, y1, x2, y2 = ann['bbox']
            coco_annotations.append({'id': len(coco_annotations), 'image_id': i, 'category_id': ann['category_id'], 'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'segmentation': [], 'iscrowd': 0})
            weight_per_class[ann['category_id']] += 1
        for ann in detections[i]['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            x1, y1, x2, y2 = ann['bbox']
            coco_dets.append({'id': len(coco_dets), 'image_id': i, 'category_id': ann['category_id'], 'score': ann['score'], 'bbox': [x1, y1, x2 - x1, y2 - y1]})
    coco_dict['images'], coco_dict['annotations'] = coco_images, coco_annotations
    weight_sum = sum(weight_per_class)
    for i in range(0, len(weight_per_class)):
        weight_per_class[i] /= weight_sum
    fd, coco_dict_f = tempfile.mkstemp(suffix='.json', text=True)
    with os.fdopen(fd, 'w') as fp:
        json.dump(coco_dict, fp)
    coco_gt = COCO(coco_dict_f)
    fd, coco_dets_f = tempfile.mkstemp(suffix='.json', text=True)
    with os.fdopen(fd, 'w') as fp:
        json.dump(coco_dets, fp)
    coco_det = coco_gt.loadRes(coco_dets_f)
    os.unlink(coco_dict_f)
    os.unlink(coco_dets_f)
    coco_eval = COCOeval(coco_gt, coco_det, 'bbox')
    print('evaluate for', coco_eval.params.catIds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results = {
        'metrics': ['mAP', 'AP50'],
        'thing_classes': thing_classes,
        'results': {
            'overall': coco_eval.stats[:2].tolist()
        },
        'weights': {
            'total': weight_sum,
            'classes': weight_per_class
        }
    }
    for i in range(0, len(thing_classes)):
        coco_eval.params.catIds = [i]
        print('evaluate for', coco_eval.params.catIds)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        results['results'][thing_classes[i]] = coco_eval.stats[:2].tolist()
    results['results']['weighted'] = [0.0, 0.0]
    for i in range(0, len(thing_classes)):
        results['results']['weighted'][0] += weight_per_class[i] * results['results'][thing_classes[i]][0]
        results['results']['weighted'][1] += weight_per_class[i] * results['results'][thing_classes[i]][1]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['copy', 'mask', 'convert'])
    parser.add_argument('--training', type=bool, default=False)
    args = parser.parse_args()

    if args.opt == 'copy':
        copy_frames(args.training)

    if args.opt == 'mask':
        mask_overlay(args.training)

    if args.opt == 'convert':
        convert_splits(args.training)


'''
python annotate.py --opt copy
python annotate.py --opt copy --training 1
python annotate.py --opt mask
python annotate.py --opt mask --training 1
python annotate.py --opt convert
python annotate.py --opt convert --training 1
'''
