#!python3

import os
import sys
import time
import datetime
import json
import gzip
import copy
import math
import random
import tqdm
import glob
import lmdb
import argparse
import contextlib
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import skvideo.io
import skimage.io
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from decode_training import TrainingFrames
from finetune import BoxMode, refine_annotations

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
class Dummy(object): pass


def dynamic_background(args):
    lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id))
    if not os.access(lmdb_path, os.W_OK):
        os.mkdir(lmdb_path)
    outputdir = os.path.join(lmdb_path, 'pngs')
    if not os.access(outputdir, os.W_OK):
        os.mkdir(outputdir)

    refined_json_zip = os.path.join(lmdb_path, 'cache_refine_%s.json.gz' % '_'.join(args.anno_models))
    if os.access(refined_json_zip, os.R_OK):
        with gzip.open(refined_json_zip, 'rt') as fp:
            dict_json = json.loads(fp.read())['refined']
    else:
        dict_json, _ = refine_annotations(args, visualize=False)
        with gzip.open(refined_json_zip, 'wt') as fp:
            fp.write(json.dumps({'refined': dict_json, 'args': vars(args)}))

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
                # if x2 - x1 > 500 or y2 - y1 > 500: continue
                M_arr[y1 : y2, x1 : x2] = 0.0
            Q.append({'im_arr': im_arr, 'M_arr': M_arr, 'fn': fn, 'annotations': anns})

        if i in background_idx:
            images_arr = np.stack(list(map(lambda x: x['im_arr'], Q))).astype(np.float16)
            masks_arr = np.stack(list(map(lambda x: x['M_arr'], Q))).astype(np.float16)
            average, images_arr, masks_arr = images_arr.mean(axis=0), (images_arr * masks_arr).mean(axis=0), masks_arr.mean(axis=0)
            M = np.zeros(shape=masks_arr.shape, dtype=np.uint8) + 255
            for x in range(0, images_arr.shape[0]):
                for y in range(0, images_arr.shape[1]):
                    if masks_arr[x, y, 0] < 0.5 / len(Q):
                        masks_arr[x, y], images_arr[x, y], M[x, y] = 1, average[x, y], 0
            skimage.io.imsave(os.path.join(outputdir, '%s_mask.png' % Q[-1]['fn']), M, check_contrast=False)
            skimage.io.imsave(os.path.join(outputdir, '%s_background.jpg' % Q[-1]['fn']), (images_arr / masks_arr).astype(np.uint8), quality=80)


def _inpaint_background(param):
    background, mask, fn, imsave_params = param
    assert background.dtype == mask.dtype and background.shape[:2] == mask.shape
    mask = 255 - mask
    pixels = mask.sum() / 255.0
    if pixels < 10:
        pass
    else:
        # R = int(pixels ** 0.5 * 0.7)
        R = min(mask.shape) // 10
        background = cv2.inpaint(background, mask, R, cv2.INPAINT_TELEA)
        # background = cv2.inpaint(background, mask, R, cv2.INPAINT_NS)
    skimage.io.imsave(fn, background, **imsave_params)


def inpaint_dynamic(args):
    from multiprocessing import Pool as ProcessPool
    inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'pngs'))
    outputdir = os.path.normpath(os.path.join(inputdir, '..', 'inpaint'))
    if not os.access(outputdir, os.W_OK):
        os.mkdir(outputdir)
    filenames = map(os.path.basename, glob.glob(os.path.join(inputdir, '*.png')))
    filenames = sorted(list(set(map(lambda x: x[:x.find('_')], filenames))))
    params_list = []
    for fn in filenames:
        background = skimage.io.imread(os.path.join(inputdir, fn + '_background.jpg'))
        mask = skimage.io.imread(os.path.join(inputdir, fn + '_mask.png'))
        params_list.append([background, mask, os.path.join(outputdir, fn + '_inpaint.jpg'), {'quality': 80}])
    pool = ProcessPool(processes=args.procs)
    inpainted_list = pool.map_async(_inpaint_background, params_list).get()
    pool.close()
    pool.join()


def inpaint_dynamic_valid(args):
    from multiprocessing import Pool as ProcessPool
    inputnpz = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_lmdb', args.id, 'dynamic_background.npz'))
    outputdir1 = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_background_lmdb', args.id, 'pngs'))
    outputdir2 = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_background_lmdb', args.id, 'inpaint'))
    if not os.access(outputdir1, os.W_OK):
        os.makedirs(outputdir1)
    if not os.access(outputdir2, os.W_OK):
        os.makedirs(outputdir2)
    fp = np.load(inputnpz)
    masks, backgrounds, filenames = fp['masks'][:, :, :, 0], fp['backgrounds'], list(map(str, fp['filenames'].tolist()))
    fp.close()

    params_list = []
    for i in range(0, len(filenames)):
        skimage.io.imsave(os.path.join(outputdir1, filenames[i] + '_background.jpg'), backgrounds[i], quality=80)
        skimage.io.imsave(os.path.join(outputdir1, filenames[i] + '_mask.png'), masks[i], check_contrast=False)
        params_list.append([backgrounds[i], masks[i], os.path.join(outputdir2, filenames[i] + '_inpaint.jpg'), {'quality': 80}])
    pool = ProcessPool(processes=args.procs)
    inpainted_list = pool.map_async(_inpaint_background, params_list).get()
    pool.close()
    pool.join()


def inpaint_coco(split, procs):
    import imantics
    from multiprocessing import Pool as ProcessPool
    from base_detector_train import get_coco_dicts

    args = Dummy()
    args.cocodir, args.smallscale = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017')), False
    coco_dicts = get_coco_dicts(args, split, segment=True)
    coco_dicts.sort(key=lambda x: x['file_name'])
    outputdir = os.path.join(args.cocodir, 'inpaint', 'val2017' if split == 'valid' else 'train2017')
    if not os.access(outputdir, os.W_OK):
        os.makedirs(outputdir)

    batchsize, params_list = 100, []
    for im in tqdm.tqdm(coco_dicts, ascii=True, desc='MSCOCO-2017 ' + split):
        im_arr = skimage.io.imread(im['file_name'])
        if len(im_arr.shape) < 3: im_arr = np.stack([im_arr] * 3, axis=2)
        polygons = []
        for ann in im['annotations']:
            if ann['iscrowd'] == 0:
                polygons = polygons + ann['segmentation']
        count = sum(list(map(len, polygons)))
        if count > 0:
            m_arr = imantics.Annotation.from_polygons(polygons, image=imantics.Image(im_arr)).array.astype(np.uint8) * 255
            m_arr = 255 - m_arr
        else:
            m_arr = np.ones_like(im_arr[:, :, 0])
        params_list.append([im_arr, m_arr, os.path.join(outputdir, os.path.basename(im['file_name'])), {'quality': 80}])
        if len(params_list) >= batchsize:
            pool = ProcessPool(processes=procs)
            pool.map_async(_inpaint_background, params_list).get()
            pool.close()
            pool.join()
            params_list = []
    if len(params_list) > 0:
        pool = ProcessPool(processes=procs)
        pool.map_async(_inpaint_background, params_list).get()
        pool.close()
        pool.join()


def inpaint_coco_box(split, procs):
    from multiprocessing import Pool as ProcessPool
    from base_detector_train import get_coco_dicts

    args = Dummy()
    args.cocodir, args.smallscale = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017')), False
    coco_dicts = get_coco_dicts(args, split, segment=False)
    coco_dicts.sort(key=lambda x: x['file_name'])
    outputdir = os.path.join(args.cocodir, 'inpaint_box', 'val2017' if split == 'valid' else 'train2017')
    if not os.access(outputdir, os.W_OK):
        os.makedirs(outputdir)

    batchsize, params_list = 100, []
    for im in tqdm.tqdm(coco_dicts, ascii=True, desc='MSCOCO-2017 ' + split):
        im_arr = skimage.io.imread(im['file_name'])
        if len(im_arr.shape) < 3: im_arr = np.stack([im_arr] * 3, axis=2)
        for ann in im['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYWH_ABS
        m_arr = np.zeros_like(im_arr[:, :, 0]) + 255
        for ann in im['annotations']:
            x, y, w, h = map(int, ann['bbox'])
            m_arr[y : y + h, x : x + w] = 0
        params_list.append([im_arr, m_arr, os.path.join(outputdir, os.path.basename(im['file_name'])), {'quality': 80}])
        if len(params_list) >= batchsize:
            pool = ProcessPool(processes=procs)
            pool.map_async(_inpaint_background, params_list).get()
            pool.close()
            pool.join()
            params_list = []
    if len(params_list) > 0:
        pool = ProcessPool(processes=procs)
        pool.map_async(_inpaint_background, params_list).get()
        pool.close()
        pool.join()


if __name__ == '__main__':
    # vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']

    # from multiprocessing import Pool as ProcessPool
    # params_list = []
    # for vid in vid_list:
    #     args = Dummy()
    #     args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = vid, sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    #     params_list.append(args)
    # procs = 3
    # pool = ProcessPool(processes=procs)
    # _ = pool.map_async(dynamic_background, params_list).get()
    # pool.close()
    # pool.join()

    # args = Dummy()
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '008', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '027', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '049', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '056', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '087', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '128', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '146', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '175', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    # dynamic_background(args)

    # for vid in tqdm.tqdm(vid_list, ascii=True):
    #     args = Dummy()
    #     args.id, args.procs = vid, 8
    #     # inpaint_dynamic(args)
    #     inpaint_dynamic_valid(args)
    #     print('\n', vid)

    # inpaint_coco('valid', 8)
    # inpaint_coco('train', 8)

    # inpaint_coco_box('valid', 8)
    # inpaint_coco_box('train', 8)

    # for vid in ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']:
    #     imagedir = '../../images/train_background_lmdb/' + vid
    #     files1 = glob.glob(os.path.join(imagedir, 'pngs', '*background.jpg'))
    #     files2 = glob.glob(os.path.join(imagedir, 'pngs', '*mask.png'))
    #     files3 = glob.glob(os.path.join(imagedir, 'inpaint', '*inpaint.jpg'))
    #     if not len(files1) == len(files2) == len(files3) == 59:
    #         print(vid, len(files1), len(files2), len(files3))

    # for vid in ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']:
    #     imagedir = '../../images/annotated/' + vid + '/unmasked'
    #     files1 = glob.glob(os.path.join(imagedir, '*.jpg'))
    #     imagedir = '../../images/valid_background_lmdb/' + vid
    #     files2 = glob.glob(os.path.join(imagedir, 'pngs', '*mask.png'))
    #     files3 = glob.glob(os.path.join(imagedir, 'pngs', '*background.jpg'))
    #     files4 = glob.glob(os.path.join(imagedir, 'inpaint', '*inpaint.jpg'))
    #     if not len(files1) == len(files2) == len(files3) == len(files4):
    #         print(vid, len(files1), len(files2), len(files3), len(files4))

    pass
