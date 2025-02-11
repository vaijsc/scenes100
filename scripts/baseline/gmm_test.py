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
import pickle
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
from sklearn.mixture import GaussianMixture

import torch
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from finetune import refine_annotations, get_annotation_dict, bbox_GMM
from evaluation import evaluate_masked, evaluate_cocovalid, eval_AP


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def evaluate_all_videos():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.cocodir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017')
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)

    results_AP, masked_bboxes_per_video, detections_per_video = {}, {}, {}
    for model in ['r50-fpn-3x', 'r101-fpn-3x']:
        args.model, args.ckpt = model, None
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
        detector = DefaultPredictor(cfg)
        results_AP[args.model] = {}

        args.smallscale = False
        detections_coco = get_coco_dicts(args, 'valid')
        for im in tqdm.tqdm(detections_coco, ascii=True, desc='%s detecting MSCOCO2017 valid' % args.model):
            im_arr = skimage.io.imread(im['file_name'])
            if len(im_arr.shape) == 2:
                im_arr = np.stack([im_arr] * 3, axis=2)
            instances = detector(im_arr[:, :, ::-1])['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            im['annotations'] = []
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})

        detections_per_video[args.model] = {}
        for f in files:
            args.id = f['id']
            inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                images = json.load(fp)
            detections = []
            for im in tqdm.tqdm(images, ascii=True, desc='%s detecting %s validation frames' % (args.model, args.id)):
                det = im
                det['annotations'] = []
                instances = detector(skimage.io.imread(os.path.join(inputdir, 'unmasked', im['file_name']))[:, :, ::-1])['instances'].to('cpu')
                # bbox has format [x1, y1, x2, y2]
                bbox = instances.pred_boxes.tensor.numpy().tolist()
                score = instances.scores.numpy().tolist()
                label = instances.pred_classes.numpy().tolist()
                for i in range(0, len(label)):
                    det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
                detections.append(det)
            detections_per_video[args.model][args.id] = detections

        # AP evaluation
        masked_bboxes_per_video[args.model] = {}
        for f in files:
            args.id = f['id']
            results_id, images_id, detections_id = evaluate_masked(args.id, copy.deepcopy(detections_per_video[args.model][args.id]), return_bboxes=True)
            results_AP[args.model]['manual_%s' % args.id] = results_id
            masked_bboxes_per_video[args.model][args.id] = {'annotations': images_id, 'detections': detections_id}

        annotations_all_video, detections_all_video = [], []
        for f in files:
            annotations_all_video = annotations_all_video + masked_bboxes_per_video[args.model][f['id']]['annotations']
            detections_all_video = detections_all_video + masked_bboxes_per_video[args.model][f['id']]['detections']
        results_AP[args.model]['all_videos'] = eval_AP(annotations_all_video, detections_all_video)

        results_AP[args.model]['mscoco2017_valid'] = evaluate_cocovalid(args.cocodir, detections_coco)
    with gzip.open(os.path.join(os.path.dirname(__file__), 'gmm_test_results.json.gz'), 'wt') as fp:
        fp.write(json.dumps({'AP': results_AP, 'masked_bboxes': masked_bboxes_per_video, 'raw_bboxes': detections_per_video}))


def show():
    def _non_negative_mean(nums):
        return np.array(list(filter(lambda x: x >= 0, nums))).mean()

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    with gzip.open(os.path.join(os.path.dirname(__file__), 'gmm_test_results.json.gz'), 'rt') as fp:
        data = json.loads(fp.read())
    results_AP, masked_bboxes_per_video = data['AP'], data['masked_bboxes']

    models = ['r50-fpn-3x', 'r101-fpn-3x']
    count_person_all_video, count_vehicle_all_video = 0, 0
    for f in files:
        with open(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', f['id'], 'annotations.json'), 'r') as fp:
            images_i = json.load(fp)
        count_person = sum([sum(list(map(lambda x: 1 if x['category_id'] == 0 else 0, im['annotations']))) for im in images_i])
        count_vehicle = sum([sum(list(map(lambda x: 1 if x['category_id'] == 1 else 0, im['annotations']))) for im in images_i])
        count_person_all_video += count_person
        count_vehicle_all_video += count_vehicle
        weight_person = count_person / (count_person + count_vehicle)
        for m in models:
            _r = results_AP[m]['manual_%s' % f['id']]['results']
            _r['overall_weighted'] = [None, None]
            _r['overall_weighted'][0] = _r['person'][0] * weight_person + _r['vehicle'][0] * (1 - weight_person)
            _r['overall_weighted'][1] = _r['person'][1] * weight_person + _r['vehicle'][1] * (1 - weight_person)
    weight_person_all_video = count_person_all_video / (count_person_all_video + count_vehicle_all_video)
    for m in models:
        _r = results_AP[m]['all_videos']['results']
        _r['overall_weighted'] = [None, None]
        _r['overall_weighted'][0] = _r['person'][0] * weight_person_all_video + _r['vehicle'][0] * (1 - weight_person_all_video)
        _r['overall_weighted'][1] = _r['person'][1] * weight_person_all_video + _r['vehicle'][1] * (1 - weight_person_all_video)

    categories = ['overall', 'overall_weighted', 'person', 'vehicle']
    plt.figure(figsize=(16, 9))
    for i in range(0, len(categories)):
        print('inspect for category', categories[i])
        plt.subplot(2, 2, i + 1)
        APs = [{**{m: results_AP[m]['manual_%s' % f['id']]['results'][categories[i]] for m in models}, **{'id': f['id']}} for f in files]
        APs.sort(key=lambda x: x[models[1]][0]) # sort by mAP of r101-fpn-3x
        APs_non_neg = list(filter(lambda x: x[models[1]][0] >= 0, APs))
        print('worst 10:')
        for x in APs_non_neg[:10]:
            print(x)
        print('best 10:')
        for x in APs_non_neg[-10:]:
            print(x)
        xs = np.arange(1, len(APs) + 1)
        r50_mAP = list(map(lambda x: x['r50-fpn-3x'][0], APs))
        plt.plot(xs, r50_mAP, 'ro-')
        r50_AP50 = list(map(lambda x: x['r50-fpn-3x'][1], APs))
        plt.plot(xs, r50_AP50, 'rx-')
        r101_mAP = list(map(lambda x: x['r101-fpn-3x'][0], APs))
        plt.plot(xs, r101_mAP, 'bo-')
        r101_AP50 = list(map(lambda x: x['r101-fpn-3x'][1], APs))
        plt.plot(xs, r101_AP50, 'bx-')
        plt.xlim(0, 101)
        plt.ylim(-0.05, 1.05)
        plt.xticks([])
        plt.grid(True)
        plt.legend(['r50-fpn-3x mAP %.4f' % _non_negative_mean(r50_mAP), 'r50-fpn-3x AP50 %.4f' % _non_negative_mean(r50_AP50), 'r101-fpn-3x mAP %.4f' % _non_negative_mean(r101_mAP), 'r101-fpn-3x AP50 %.4f' % _non_negative_mean(r101_AP50)])
        plt.title('object category %s\nall frames %s %.4f/%.4f %s %.4f/%.4f' % (categories[i],
            models[0], results_AP[models[0]]['all_videos']['results'][categories[i]][0], results_AP[models[0]]['all_videos']['results'][categories[i]][1],
            models[1], results_AP[models[1]]['all_videos']['results'][categories[i]][0], results_AP[models[1]]['all_videos']['results'][categories[i]][1]))
    plt.tight_layout()
    plt.show()


def gmm_test():
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_inv(x):
        assert x > 0 and x < 1, x
        return np.log(x / (1 - x))

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.anno_models = sorted(['r101-fpn-3x', 'r50-fpn-3x'])
    args.refine_det_score_thres = 0.5
    args.refine_iou_thres = 0.85
    args.refine_remove_no_sot = False
    args.gmm_max_samples = 100000
    args.gmm_n_components = 25

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    with gzip.open(os.path.join(os.path.dirname(__file__), 'gmm_test_results.json.gz'), 'rt') as fp:
        data = json.loads(fp.read())
    results_AP, masked_bboxes_per_video, raw_bboxes_per_video = data['AP'], data['masked_bboxes'], data['raw_bboxes']

    models = ['r101-fpn-3x', 'r50-fpn-3x']
    # categories = ['overall', 'person', 'vehicle']
    categories = ['overall', 'weighted']
    vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '034', '050', '169']
    # vid_list = list(map(lambda x: x['id'], files))[:30]
    # vid_list = list(map(lambda x: x['id'], files))

    improvements = {}
    for vid in vid_list:
        args.id = vid
        improvements[args.id] = {}
        gmm_models_f = os.path.join(os.path.dirname(__file__), 'gmm_cache', 'gmm_%s_%s.pickle' % (args.id, '_'.join(args.anno_models)))
        if os.access(gmm_models_f, os.R_OK):
            with open(gmm_models_f, 'rb') as fp:
                data = pickle.load(fp)
                assert data['args'] == vars(args), 'arguments mismatch, clear cache file\ncached:\n%s\ncurrent:\n%s' % (data['args'], vars(args))
                gmm_models = data['models']
            print('loaded cached model from', gmm_models_f)
        else:
            images_pseudo_anno = refine_annotations(args)[0]
            images_pseudo_anno, gmm_models = bbox_GMM(images_pseudo_anno, args)
            with open(gmm_models_f, 'wb') as fp:
                pickle.dump({'models': gmm_models, 'args': vars(args)}, fp)
        print(gmm_models)

        for m in models:
            improvements[args.id][m] = {}
            raw_bboxes, masked_bboxes, gt_bboxes = map(copy.deepcopy, [raw_bboxes_per_video[m][args.id], masked_bboxes_per_video[m][args.id]['detections'], masked_bboxes_per_video[m][args.id]['annotations']])
            ap_cached = results_AP[m]['manual_%s' % args.id]['results']

            # compute GMM likelihoods
            for images_det in [raw_bboxes, masked_bboxes]:
                log_likelihoods_all, scores_all = [], []
                for im in images_det:
                    for ann in im['annotations']:
                        assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                        x1, y1, x2, y2 = ann['bbox']
                        ann['bbox_norm'] = [x1 / im['width'], y1 / im['height'], x2 / im['width'], y2 / im['height']]
                        ann['gmm_log_likelihood'] = gmm_models['per_class'][ann['category_id']].score_samples([ann['bbox_norm']])[0]
                        # stats = gmm_models['gmm_log_likelihood_stats']['per_class'][ann['category_id']]
                        # ann['gmm_log_likelihood_norm'] = (ann['gmm_log_likelihood'] - stats['mean']) / stats['std']
                        log_likelihoods_all.append(ann['gmm_log_likelihood'])
                        scores_all.append(ann['score'])
                log_likelihoods_all = np.array(log_likelihoods_all)
                llh_mean, llh_std = log_likelihoods_all.mean(), log_likelihoods_all.std()
                scores_all = np.array(scores_all)
                scores_mean, scores_std = scores_all.mean(), scores_all.std()
                # print(llh_mean, llh_std, scores_mean, scores_std)

                for im in images_det:
                    for i in range(0, len(im['annotations'])):
                        s = im['annotations'][i]['score']
                        llh = im['annotations'][i]['gmm_log_likelihood']
                        if s > scores_mean:
                            continue
                        im['annotations'][i]['score'] = _sigmoid(_sigmoid_inv(s) + (llh - llh_mean) / (llh_std * 1))
                        # print(im['annotations'][i]['score'] - s)

            ap_masked_rectified = evaluate_masked(args.id, masked_bboxes)['results']
            ap_raw_rectified = evaluate_masked(args.id, raw_bboxes)['results']

            improvements[args.id][m]['masked_rectified'], improvements[args.id][m]['raw_rectified'] = {}, {}
            for k in categories:
                improvements[args.id][m]['masked_rectified'][k] = [ap_masked_rectified[k][0] - ap_cached[k][0], ap_masked_rectified[k][1] - ap_cached[k][1]]
                improvements[args.id][m]['raw_rectified'][k] = [ap_raw_rectified[k][0] - ap_cached[k][0], ap_raw_rectified[k][1] - ap_cached[k][1]]
    print(improvements)

    for view_key in ['raw_rectified', 'masked_rectified']:
        plt.figure(figsize=(16, 8))
        for i in range(0, len(categories)):
            for j in range(0, len(models)):
                improvements_cat_m = [{'id': v, 'raw_rectified': improvements[v][models[j]]['raw_rectified'][categories[i]], 'masked_rectified': improvements[v][models[j]]['masked_rectified'][categories[i]]} for v in improvements]
                improvements_cat_m.sort(key=lambda x: x[view_key][0])
                xs = np.arange(1, len(improvements_cat_m) + 1, 1)
                plt.subplot(2, 2, 1 + i + j * 2)
                plt.plot([0, xs.max() + 1], [0, 0], 'k-')
                ys_mAP = np.array(list(map(lambda x: x[view_key][0], improvements_cat_m))) * 100
                plt.plot(xs, ys_mAP, 'ro-')
                ys_AP50 = np.array(list(map(lambda x: x[view_key][1], improvements_cat_m))) * 100
                plt.plot(xs, ys_AP50, 'bx-', alpha=0.75)
                for k in range(0, len(xs)):
                    plt.text(xs[k], ys_mAP[k] + 0.4, improvements_cat_m[k]['id'], rotation=-80, horizontalalignment='right', verticalalignment='center', size=9, color='red')
                plt.legend(['0', 'mAP %.4f' % ys_mAP.mean(), 'AP50 %.4f' % ys_AP50.mean()])
                plt.xticks([])
                plt.xlim(0, xs.max() + 1)
                plt.ylim(-3, 3)
                plt.ylabel('AP (0-100)')
                plt.grid(True)
                plt.title('AP shift <%s>\nmodel <%s> category <%s>' % (view_key, models[j], categories[i]))
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()


if __name__ == '__main__':
    # evaluate_all_videos()
    # show()
    gmm_test()
