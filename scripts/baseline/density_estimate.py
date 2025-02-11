#!python3

import os
import sys
import types
import functools
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import sklearn.utils
import skimage.io
import skvideo.io
import networkx

import sklearn.utils
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
from scipy.stats import gaussian_kde

import torch
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import torch
import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from finetune import refine_annotations, get_annotation_dict
from finetune import bbox_GMM_2D_XYXY, bbox_GMM_2D_XYWH
from evaluation import evaluate_masked, evaluate_cocovalid, eval_AP
from beta_mixture import Beta, BetaMixture1D


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def bbox_KDE(images, args, compute_likelihood=True):
    boxes_all, boxes_per_class = [], [[] for _ in range(0, len(thing_classes))]
    W, H = images[0]['width'], images[0]['height'],
    for im in images:
        for ann in im['annotations']:
            boxes_all.append(ann['bbox_norm'])
            boxes_per_class[ann['category_id']].append(ann['bbox_norm'])
    boxes_all = np.array(boxes_all).astype(np.float32)
    boxes_per_class = [np.array(x).astype(np.float32) for x in boxes_per_class]
    MAX_N = args.kde_max_samples
    kde_models = {'all': None, 'per_class': [None for _ in range(0, len(thing_classes))]}
    print('fitting KDE to all bboxes', boxes_all.shape, boxes_all.dtype)
    features = boxes_all
    if features.shape[0] > MAX_N:
        print('randomly reduce to %s samples' % MAX_N)
        features = sklearn.utils.shuffle(features)[: MAX_N]
    kde = gaussian_kde(features.T, bw_method='silverman')
    s = kde.logpdf(features.T)
    print('BW', kde.factor, 'LogLikelihodd', s.mean())
    kde_models['all'] = kde
    for cat in range(0, len(thing_classes)):
        print('fitting KDE to %s bboxes' % thing_classes[cat], boxes_per_class[cat].shape, boxes_per_class[cat].dtype)
        features = boxes_per_class[cat]
        if features.shape[0] > MAX_N:
            print('randomly reduce to %s samples' % MAX_N)
            features = sklearn.utils.shuffle(features)[: MAX_N]
        kde = gaussian_kde(features.T, bw_method='silverman')
        s = kde.logpdf(features.T)
        print('BW', kde.factor, 'LogLikelihodd', s.mean())
        kde_models['per_class'][cat] = kde
    print(kde_models)

    if compute_likelihood:
        minibatchsize = 10000
        batches = [{'image_i': [], 'anno_j': [], 'bboxes': []} for _ in range(0, len(thing_classes) + 1)]
        for i in tqdm.tqdm(range(0, len(images)), ascii=True, desc='indexing bboxes'):
            for j in range(0, len(images[i]['annotations'])):
                _ann = images[i]['annotations'][j]
                batches[_ann['category_id']]['image_i'].append(i)
                batches[_ann['category_id']]['anno_j'].append(j)
                batches[_ann['category_id']]['bboxes'].append(_ann['bbox_norm'])
                batches[len(thing_classes)]['image_i'].append(i)
                batches[len(thing_classes)]['anno_j'].append(j)
                batches[len(thing_classes)]['bboxes'].append(_ann['bbox_norm'])
        for cat in tqdm.tqdm(range(0, len(thing_classes) + 1), ascii=True, desc='computing bbox log-likelihood'):
            if cat < len(thing_classes):
                scores = []
                features = np.array(batches[cat]['bboxes'], dtype=np.float32)
                for b in tqdm.tqdm(range(0, features.shape[0] // minibatchsize + 1), ascii=True):
                    features_b = features[minibatchsize * b : minibatchsize * (b + 1)]
                    if features_b.shape[0] > 0:
                        scores.append(kde_models['per_class'][cat].logpdf(features_b.T))
                scores = np.concatenate(scores, axis=0)
                for k in range(0, len(batches[cat]['image_i'])):
                    i, j = batches[cat]['image_i'][k], batches[cat]['anno_j'][k]
                    images[i]['annotations'][j]['kde_log_likelihood'] = scores[k]
            else:
                scores = []
                features = np.array(batches[cat]['bboxes'], dtype=np.float32)
                for b in tqdm.tqdm(range(0, features.shape[0] // minibatchsize + 1), ascii=True):
                    features_b = features[minibatchsize * b : minibatchsize * (b + 1)]
                    if features_b.shape[0] > 0:
                        scores.append(kde_models['all'].logpdf(features_b.T))
                scores = np.concatenate(scores, axis=0)
                for k in range(0, len(batches[cat]['image_i'])):
                    i, j = batches[cat]['image_i'][k], batches[cat]['anno_j'][k]
                    images[i]['annotations'][j]['kde_log_likelihood_all'] = scores[k]
    return images, kde_models


class BBoxDepthEstimator(object):
    def __init__(self, vid):
        from depth_net.inference import DepthNetEstimator
        import torchvision.transforms as transforms

        self.net = DepthNetEstimator()
        _ckpt = os.path.join(os.path.dirname(__file__), '..', 'camcal', 'ckpt_%s.pth' % vid)
        assert os.access(_ckpt, os.R_OK)
        print('loading network weights from', _ckpt)
        self.net.load_state_dict(torch.load(_ckpt)['state_dict'])
        self.net.cuda()
        self.net.eval()
        self.tf = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def estimate(self, im):
        im_arr = skimage.io.imread(im['file_name']).astype(np.float32) / 255.0
        im_tensor = torch.from_numpy(im_arr.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            im_tensor = self.tf(im_tensor).cuda()
            d = self.net(im_tensor)[0, 0].detach().cpu().numpy()
        for ann in im['annotations']:
            x1, y1, x2, y2 = map(int, ann['bbox'])
            d_crop = d[y1 : y2, x1 : x2]
            if d_crop.size > 0:
                ann['bbox_depth_mean'], ann['bbox_depth_std'] = d_crop.mean(), d_crop.std()
            else:
                ann['bbox_depth_mean'], ann['bbox_depth_std'] = 0.0, 1.0
        return im_arr, d

    @staticmethod
    def make_3d_bbox(im, d_min, d_max):
        assert d_max > d_min
        W, H = im['width'], im['height']
        W = max(W, H)
        for ann in im['annotations']:
            x1, y1, x2, y2 = ann['bbox']
            d_norm = (ann['bbox_depth_mean'] - d_min) / (d_max - d_min)
            ann['bbox_3D'] = [
                x1 * d_norm / W, y1 * d_norm / W, d_norm,
                x2 * d_norm / W, y2 * d_norm / W, d_norm
            ]


def bbox_GMM_3D(images, args, compute_likelihood=True, show_boxes=False):
    bbox_depth_estimator = BBoxDepthEstimator(args.id)
    depth_min, depth_max = 1e10, -1e10
    for im in tqdm.tqdm(images, ascii=True, desc='estimating depth'):
        im_arr, d = bbox_depth_estimator.estimate(im)
        depth_min, depth_max = min(depth_min, d.min()), max(depth_max, d.max())
        # plt.figure()
        # plt.subplot(1, 2, 1); plt.imshow(im_arr)
        # plt.subplot(1, 2, 2); d=d-d.min(); d/=d.max(); x1,y1,x2,y2=map(int,im['annotations'][0]['bbox']); d[y1:y2,x1:x2]=0; plt.imshow(d)
        # plt.show()
    del bbox_depth_estimator
    print('depth range', depth_min, depth_max)

    boxes_all, boxes_per_class = [], [[] for _ in range(0, len(thing_classes))]
    W, H = images[0]['width'], images[0]['height']
    for im in tqdm.tqdm(images, ascii=True, desc='gathering & making-3D bboxes'):
        assert H == im['height'] and W == im['width']
        BBoxDepthEstimator.make_3d_bbox(im, depth_min, depth_max)
        for ann in im['annotations']:
            boxes_all.append(ann['bbox_3D'])
            boxes_per_class[ann['category_id']].append(ann['bbox_3D'])
        if show_boxes:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1); ax.imshow(skimage.io.imread(im['file_name']))
            for ann in im['annotations']:
                x1, y1, x2, y2 = ann['bbox']; c = bbox_rgbs[ann['category_id']]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=c, facecolor='none')
                ax.add_patch(rect)
            ax = fig.add_subplot(1, 2, 2, projection='3d');
            for ann in im['annotations']:
                x1, y1, z1, x2, y2, z2 = ann['bbox_3D']; c = bbox_rgbs[ann['category_id']]
                ax.plot3D([x1, x1, x1, x1, x1], [y1, y1, y2, y2, y1], [z1, z2, z2, z1, z1], color=c)
                ax.plot3D([x2, x2, x2, x2, x2], [y1, y1, y2, y2, y1], [z1, z2, z2, z1, z1], color=c)
                ax.plot3D([x1, x2], [y1, y1], [z1, z1], color=c); ax.plot3D([x1, x2], [y2, y2], [z1, z1], color=c)
                ax.plot3D([x1, x2], [y2, y2], [z2, z2], color=c); ax.plot3D([x1, x2], [y1, y1], [z2, z2], color=c)
            ax.set_xlim3d(0, 1); ax.set_ylim3d(0, 1); ax.set_zlim3d(0, 1)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            plt.show()
    boxes_all = np.array(boxes_all).astype(np.float32)
    boxes_per_class = [np.array(x).astype(np.float32) for x in boxes_per_class]
    print('%.5f ~ %.5f, 99%%=%.5f' % (boxes_all.min(), boxes_all.max(), np.percentile(boxes_all, 99)))
    for i in range(0, len(boxes_per_class)):
        print('%.5f ~ %.5f, 99%%=%.5f' % (boxes_per_class[i].min(), boxes_per_class[i].max(), np.percentile(boxes_per_class[i], 99)))
    MAX_N = args.gmm_max_samples
    gmm_models_3d = {
        'all': GaussianMixture(n_components=args.gmm_n_components, max_iter=200, n_init=3, init_params='kmeans'),
        'per_class': [GaussianMixture(n_components=args.gmm_n_components, max_iter=200, n_init=3, init_params='kmeans') for _ in range(0, len(thing_classes))],
        'depth_stats': [depth_min, depth_max]
    }

    print('fitting GMM to all 3D bboxes', boxes_all.shape, boxes_all.dtype)
    features = boxes_all
    if features.shape[0] > MAX_N:
        print('randomly reduce to %s samples' % MAX_N)
        features = sklearn.utils.shuffle(features)[: MAX_N]
    gmm_models_3d['all'].fit(features)
    assert gmm_models_3d['all'].converged_, 'not converged'
    for cat in range(0, len(thing_classes)):
        print('fitting GMM to %s 3D bboxes' % thing_classes[cat], boxes_per_class[cat].shape, boxes_per_class[cat].dtype)
        features = boxes_per_class[cat]
        if features.shape[0] > MAX_N:
            print('randomly reduce to %s samples' % MAX_N)
            features = sklearn.utils.shuffle(features)[: MAX_N]
        gmm_models_3d['per_class'][cat].fit(features)
        assert gmm_models_3d['per_class'][cat].converged_, 'not converged'

    if compute_likelihood:
        batches = [{'image_i': [], 'anno_j': [], 'bboxes': []} for _ in range(0, len(thing_classes) + 1)]
        for i in tqdm.tqdm(range(0, len(images)), ascii=True, desc='indexing bboxes'):
            for j in range(0, len(images[i]['annotations'])):
                _ann = images[i]['annotations'][j]
                batches[_ann['category_id']]['image_i'].append(i)
                batches[_ann['category_id']]['anno_j'].append(j)
                batches[_ann['category_id']]['bboxes'].append(_ann['bbox_3D'])
                batches[len(thing_classes)]['image_i'].append(i)
                batches[len(thing_classes)]['anno_j'].append(j)
                batches[len(thing_classes)]['bboxes'].append(_ann['bbox_3D'])
        for cat in tqdm.tqdm(range(0, len(thing_classes) + 1), ascii=True, desc='computing bbox log-likelihood'):
            if cat < len(thing_classes):
                scores = gmm_models_3d['per_class'][cat].score_samples(np.array(batches[cat]['bboxes'], dtype=np.float32))
                for k in range(0, len(batches[cat]['image_i'])):
                    i, j = batches[cat]['image_i'][k], batches[cat]['anno_j'][k]
                    images[i]['annotations'][j]['gmm_3d_log_likelihood'] = scores[k]
            else:
                scores = gmm_models_3d['all'].score_samples(np.array(batches[cat]['bboxes'], dtype=np.float32))
                for k in range(0, len(batches[cat]['image_i'])):
                    i, j = batches[cat]['image_i'][k], batches[cat]['anno_j'][k]
                    images[i]['annotations'][j]['gmm_3d_log_likelihood_all'] = scores[k]
    return images, gmm_models_3d


class MLPRegressor():
    def __init__(self):
        self.lr, self.bs = 1e-3, 16
        self.mlp = torch.nn.Sequential(torch.nn.Linear(5, 200), torch.nn.ReLU(), torch.nn.Linear(200, 200), torch.nn.ReLU(), torch.nn.Linear(200, 1), torch.nn.Sigmoid())
        self.opt = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        self.mlp.eval()

    def feature_eng(self, X):
        X_sq, X_cr = torch.square(X), X[:, 0 : 1] * X[:, 1 : 2]
        return torch.cat([X, X_sq, X_cr], dim=1)

    def fit_Xy(self, X, y):
        self.mlp.train()
        X, y = map(torch.from_numpy, [X.astype(np.float32), y.astype(np.float32)])
        loss_ep = []
        for ep in range(0, 50):
            X_aug = X + torch.randn(*X.size(), dtype=X.dtype) * 0.01
            X_aug = self.feature_eng(X_aug)
            X_aug, y_aug = sklearn.utils.shuffle(X_aug, y)
            loss_ep.append([])
            for i in range(0, math.ceil(X_aug.size(0) / self.bs)):
                self.opt.zero_grad()
                Xi, yi = X_aug[i * self.bs : (i + 1) * self.bs], y_aug[i * self.bs : (i + 1) * self.bs]
                L = torch.square(self.mlp(Xi)[:, 0] - yi).mean()
                L.backward()
                self.opt.step()
                loss_ep[-1].append(L.item())
            loss_ep[-1] = np.array(loss_ep[-1]).mean()
            if np.isnan(loss_ep[-1]): raise Exception('training diverged')
            if len(loss_ep) < 5: continue
            _mean, _std = np.array(loss_ep[-5 :]).mean(), np.array(loss_ep[-5 :]).std()
            if _std < _mean * 0.005: break
        print('converged, losses = ', loss_ep)
        self.mlp.eval()
        return self

    def inference(self, X):
        X = self.feature_eng(torch.from_numpy(X.astype(np.float32)))
        with torch.no_grad(): y_ = self.mlp(X)
        return y_[:, 0].detach().numpy()


def inspect_bbox_scores(raw_bboxes, masked_bboxes, gt_bboxes, AP, desc, rectify=False, save=False):
    def _IoU(ann1, ann2):
        assert ann1['bbox_mode'] == ann2['bbox_mode'] == BoxMode.XYXY_ABS
        x11, y11, x12, y12 = ann1['bbox']
        x21, y21, x22, y22 = ann2['bbox']
        xA, yA = max(x11,x21), max(y11,y21)
        xB, yB = min(x12,x22), min(y12,y22)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        overlap = max(xB - xA, 0) * max(yB - yA, 0)
        return overlap / (area1 + area2 - overlap)

    def annotation_equal(ann1, ann2):
        return ann1['bbox'] == ann2['bbox'] and ann1['category_id'] == ann2['category_id']

    raw_bboxes, masked_bboxes, gt_bboxes, AP = map(copy.deepcopy, [raw_bboxes, masked_bboxes, gt_bboxes, AP])

    # copy GMM info from raw_bboxes to masked_bboxes
    for i in range(0, len(masked_bboxes)):
        for j in range(0, len(masked_bboxes[i]['annotations'])):
            for ann2 in raw_bboxes[i]['annotations']:
                if annotation_equal(masked_bboxes[i]['annotations'][j], ann2):
                    masked_bboxes[i]['annotations'][j] = copy.deepcopy(ann2)
                    break
    print('inpsecting', desc)
    for ims, d in [[raw_bboxes, 'raw_bboxes'], [masked_bboxes, 'masked_bboxes'], [gt_bboxes, 'gt_bboxes']]:
        print('%s: %d images, %d boxes' % (d, len(ims), sum(list(map(lambda x: len(x['annotations']), ims)))))
    print('AP:', AP)

    gt_counts = {k: sum(list(map(lambda x: sum(list(map(lambda a: 1 if a['category_id'] == k else 0, x['annotations']))), gt_bboxes))) for k in range(0, len(thing_classes))}
    print(gt_counts)

    # gmm_key = 'gmm_2d_log_likelihood'
    gmm_key = 'gmm_2d_xywh_log_likelihood'
    # gmm_key = 'gmm_3d_log_likelihood'
    pdf_dir = os.path.join(os.path.dirname(__file__), 'tmp', gmm_key)
    if not os.access(pdf_dir, os.W_OK):
        os.mkdir(pdf_dir)

    for i in range(0, len(gt_bboxes)):
        for n in range(0, len(masked_bboxes[i]['annotations'])):
            ious_vec = []
            for m in range(0, len(gt_bboxes[i]['annotations'])):
                if gt_bboxes[i]['annotations'][m]['category_id'] == masked_bboxes[i]['annotations'][n]['category_id']:
                    ious_vec.append(_IoU(gt_bboxes[i]['annotations'][m], masked_bboxes[i]['annotations'][n]))
            if len(ious_vec) > 0:
                masked_bboxes[i]['annotations'][n]['max_IoU'] = max(ious_vec)
            else:
                masked_bboxes[i]['annotations'][n]['max_IoU'] = 0
    if save:
        with open(os.path.join(pdf_dir, desc + '.json'), 'w') as fp:
            json.dump({'gt_bboxes': gt_bboxes, 'masked_bboxes': masked_bboxes, 'raw_bboxes': raw_bboxes}, fp)

    detections_invalid = {k: [] for k in range(0, len(thing_classes))}
    for i in range(0, len(raw_bboxes)):
        for ann in raw_bboxes[i]['annotations']:
            _masked = False
            for ann2 in masked_bboxes[i]['annotations']:
                if annotation_equal(ann, ann2):
                    _masked = True
                    break
            if not _masked:
                detections_invalid[ann['category_id']].append({'score': ann['score'], 'llh': ann[gmm_key]})

    iou_thres_dict = {x : copy.deepcopy(masked_bboxes) for x in [0.5, 0.7, 0.9]}
    detections = {x : {k: [] for k in range(0, len(thing_classes))} for x in iou_thres_dict}
    for iou_thres in iou_thres_dict:
        for i in range(0, len(gt_bboxes)):
            for m in range(0, len(gt_bboxes[i]['annotations'])):
                scores_vec = np.ones(shape=(len(iou_thres_dict[iou_thres][i]['annotations']), ), dtype=np.float32) * -1
                for n in range(0, scores_vec.shape[0]):
                    if gt_bboxes[i]['annotations'][m]['category_id'] == iou_thres_dict[iou_thres][i]['annotations'][n]['category_id'] and _IoU(gt_bboxes[i]['annotations'][m], iou_thres_dict[iou_thres][i]['annotations'][n]) >= iou_thres:
                        scores_vec[n] = iou_thres_dict[iou_thres][i]['annotations'][n]['score'] # matched
                idx_sorted = scores_vec.argsort()[::-1]
                if scores_vec[idx_sorted[0]] > 0:
                    iou_thres_dict[iou_thres][i]['annotations'][idx_sorted[0]]['TP'] = 1

        for i in range(0, len(iou_thres_dict[iou_thres])):
            for n in range(0, len(iou_thres_dict[iou_thres][i]['annotations'])):
                _ann = iou_thres_dict[iou_thres][i]['annotations'][n]
                detections[iou_thres][_ann['category_id']].append({'score': _ann['score'], 'TP': True if 'TP' in _ann else False, 'llh': _ann[gmm_key], 'max_IoU': _ann['max_IoU'], 'im_idx': i, 'ann_idx': n})
        for k in detections[iou_thres]:
            detections[iou_thres][k] = sorted(detections[iou_thres][k], key=lambda x: x['score'] * -1)
    iou_thres_dict, detections, detections_invalid = map(copy.deepcopy, [iou_thres_dict, detections, detections_invalid])
    iou_thres = None

    oracles = [None for _ in range(0, len(thing_classes))]
    bmm_score_all, beta_lh = [None for _ in range(0, len(thing_classes))], [None for _ in range(0, len(thing_classes))]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for k in range(0, len(thing_classes)):
        axes[k][0].set_title('distribution <%s>' % thing_classes[k])
        if gt_counts[k] <= 10:
            continue
        score_llh_invalid = list(map(lambda x: [x['score'], x['llh']], detections_invalid[k]))
        score_llh_valid = list(map(lambda x: [x['score'], x['llh']], detections[0.5][k]))
        score_llh_all = score_llh_invalid + score_llh_valid
        score_llh_all, score_llh_valid, score_llh_invalid = map(np.array, [score_llh_all, score_llh_valid, score_llh_invalid])
        lh_exp_norm = lambda x: np.exp((x - score_llh_all[:, 1].min()) / (score_llh_all[:, 1].max() - score_llh_all[:, 1].min()))
        score_lh_iou = np.array(list(map(lambda x: [x['score'], lh_exp_norm(x['llh']), x['max_IoU']], detections[0.5][k])))

        _axe = axes[k][0]
        legends = []
        if score_llh_invalid.size > 0:
            _axe.scatter(score_llh_invalid[:, 0], lh_exp_norm(score_llh_invalid[:, 1]), marker='x', s=12, c='k', alpha=0.4)
            legends.append('invalid detections %d' % score_llh_invalid.shape[0])
        if score_llh_valid.size > 0:
            _axe.scatter(score_llh_valid[:, 0], lh_exp_norm(score_llh_valid[:, 1]), marker='o', s=12, c='r', alpha=0.4)
            legends.append('valid detections %d' % score_llh_valid.shape[0])
        _axe.legend(legends)
        _axe.set_xlim(-0.02, 1.02)
        _axe.set_xlabel('score')
        _axe.set_ylim(0.98, 2.74)
        _axe.set_ylabel('likelihood (%s)' % gmm_key)

        _axe = axes[k][1]
        legends = []
        if rectify and score_llh_all.size > 0:
            bins = np.arange(0, 1.01, 1 / 15)
            freqs, _ = np.histogram(score_llh_all[:, 0], bins=bins)
            freqs = freqs / freqs.max() * 0.5 + 1
            _axe.plot((bins[: -1] + bins[1 :]) / 2, freqs, 'b-')
            legends.append('scores histogram')
            bins = np.arange(1, 2.72, 1.72 / 15)
            freqs, _ = np.histogram(lh_exp_norm(score_llh_all[:, 1]), bins=bins)
            freqs = freqs / freqs.max() * 0.5
            _axe.plot(freqs, (bins[: -1] + bins[1 :]) / 2, 'b-')
            legends.append('likelihood histogram')

            bmm_score_all[k] = BetaMixture1D().fit(score_llh_all[:, 0])
            print(bmm_score_all[k])
            bins = np.arange(0.0001, 1.0001, 0.01)
            _density_0 = bmm_score_all[k].likelihood(bins, 0)
            _density_1 = bmm_score_all[k].likelihood(bins, 1)
            _axe.plot(bins, _density_0 / max(_density_0.max(), _density_1.max()) * 0.5 + 1, 'k:')
            _axe.plot(bins, _density_1 / max(_density_0.max(), _density_1.max()) * 0.5 + 1, 'r:')
            legends.append('scores $\\beta$-0')
            legends.append('scores $\\beta$-1')

            beta_lh[k] = Beta().fit(lh_exp_norm(score_llh_all[:, 1]))
            print(beta_lh[k])
            bins = np.arange(1.0001, lh_exp_norm(score_llh_all[:, 1].max()) + 0.0001, 0.01)
            _density = beta_lh[k].pdf(bins)
            _axe.plot(_density / _density.max() * 0.5, bins, 'k:')
            legends.append('likelihood $\\beta$')
        _axe.legend(legends)
        _axe.set_xlim(-0.02, 1.02)
        _axe.set_xlabel('score')
        _axe.set_ylim(0.98, 2.74)
        _axe.set_ylabel('likelihood (%s)' % gmm_key)
        _axe.set_title('$\\beta$ distribution')

        _axe = axes[k][2]
        _axe.scatter(score_lh_iou[:, 0], score_lh_iou[:, 1], c=score_lh_iou[:, 2], cmap='viridis', marker='o', s=12, alpha=0.6)
        _axe.set_xlim(-0.02, 1.02)
        _axe.set_xlabel('score')
        _axe.set_ylim(0.98, 2.74)
        _axe.set_ylabel('likelihood (%s)' % gmm_key)
        _axe.set_title('IoU')

        _axe = axes[k][3]
        if rectify and score_lh_iou.shape[0] > 30:
            X_rgr, y_rgr = score_lh_iou[:, :2], score_lh_iou[:, 2]
            rgr = MLPRegressor().fit_Xy(X_rgr, y_rgr)
            oracles[k] = rgr
            xys = np.mgrid[0 : 1.001 : 0.02, 1 : 2.72 : 0.03].transpose(1, 2, 0).reshape(-1, 2)
            _axe.scatter(xys[:, 0], xys[:, 1], c=rgr.inference(xys), cmap='viridis', marker='o', s=12, alpha=0.6)
        _axe.set_xlim(-0.02, 1.02)
        _axe.set_xlabel('score')
        _axe.set_ylim(0.98, 2.74)
        _axe.set_ylabel('likelihood (%s)' % gmm_key)
        _axe.set_title('IoU regression')

        _axe = axes[k][4]
        legends = []
        for iou_thres in iou_thres_dict:
            score_threses, recalls, precisions = [1.0], [0], [1]
            cumTP, cumFP = 0, 0
            for j in range(0, len(detections[iou_thres][k])):
                if detections[iou_thres][k][j]['TP']: cumTP += 1
                else: cumFP += 1
                score_threses.append(detections[iou_thres][k][j]['score'])
                recalls.append(cumTP / gt_counts[k])
                precisions.append(cumTP / (cumTP + cumFP))
            score_threses.append(0)
            if cumTP + cumFP <= 0:
                recalls.append(0)
                precisions.append(0)
            else:
                recalls.append(cumTP / gt_counts[k])
                precisions.append(cumTP / (cumTP + cumFP))
            ap = 0
            for j in range(0, len(recalls) - 1):
                ap += (recalls[j + 1] - recalls[j]) * precisions[j + 1]
            _axe.plot(recalls, precisions, linestyle='-', color={0.5: 'g', 0.7: 'b', 0.9: 'r'}[iou_thres])
            legends.append('AP@IoU%d=%.4f' % (iou_thres * 100, ap))
        _axe.legend(legends)
        _axe.set_xlim(0, 1.02)
        _axe.set_xlabel('recall')
        _axe.set_ylim(0, 1.02)
        _axe.set_ylabel('precisions')
        iou_thres = None
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(pdf_dir, desc + '.pdf'))
    plt.close()

    if not rectify:
        return None

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_inv(x):
        assert x >= 0 and x <= 1, x
        return np.log(x / (1 - x))

    # log_likelihoods_2d_all, log_likelihoods_3d_all, scores_all = [[], []], [[], []], [[], []]
    # for im in raw_bboxes:
    #     for ann in im['annotations']:
    #         log_likelihoods_2d_all[ann['category_id']].append(ann['gmm_2d_log_likelihood'])
    #         scores_all[ann['category_id']].append(ann['score'])
    #         log_likelihoods_3d_all[ann['category_id']].append(ann['gmm_3d_log_likelihood'])
    #         scores_all[ann['category_id']].append(ann['score'])

    # log_likelihoods_2d_all.append(log_likelihoods_2d_all[0] + log_likelihoods_2d_all[1])
    # log_likelihoods_3d_all.append(log_likelihoods_3d_all[0] + log_likelihoods_3d_all[1])
    # scores_all.append(scores_all[0] + scores_all[1])
    # log_likelihoods_2d_all = list(map(np.array, log_likelihoods_2d_all))
    # llh_2d_mean = list(map(np.mean, log_likelihoods_2d_all))
    # llh_2d_std = list(map(np.std, log_likelihoods_2d_all))
    # log_likelihoods_3d_all = list(map(np.array, log_likelihoods_3d_all))
    # llh_3d_mean = list(map(np.mean, log_likelihoods_3d_all))
    # llh_3d_std = list(map(np.std, log_likelihoods_3d_all))
    # scores_all = list(map(np.array, scores_all))
    # scores_mean = list(map(np.mean, scores_all))
    # scores_std = list(map(np.std, scores_all))

    rectify_idx = []
    rectify_norm = [-1.0 for _ in range(0, len(thing_classes))]
    for i in range(0, len(raw_bboxes)):
        im = raw_bboxes[i]
        for j in range(0, len(im['annotations'])):
            for ann in masked_bboxes[i]['annotations']:
                if annotation_equal(im['annotations'][j], ann):
                    s, cat = im['annotations'][j]['score'], im['annotations'][j]['category_id']

                    # llh_2d = im['annotations'][j]['gmm_2d_log_likelihood']
                    # if s > scores_mean: continue
                    # if s > 0.9: continue
                    # if im['annotations'][j]['category_id'] == thing_classes.index('vehicle'): continue

                    # im['annotations'][j]['score'] = _sigmoid(_sigmoid_inv(s) + (llh_2d - llh_2d_mean[-1]) / (llh_2d_std[-1] * 1))
                    # im['annotations'][j]['score'] = _sigmoid(_sigmoid_inv(s) + (llh_2d - llh_2d_mean[cat]) / (llh_2d_std[cat] * 1))
                    # im['annotations'][j]['score'] = _sigmoid(_sigmoid_inv(s) + (llh_3d - llh_3d_mean) / (llh_3d_std * 1))

                    if not oracles[cat] is None:
                        # im['annotations'][j]['score'] = float(oracles[cat].inference(np.array([[s, lh_exp_norm(im['annotations'][j][gmm_key])]]))[0])
                        s_oracle = float(oracles[cat].inference(np.array([[s, lh_exp_norm(im['annotations'][j][gmm_key])]]))[0])
                        im['annotations'][j]['score'] = _sigmoid(_sigmoid_inv(s) + _sigmoid_inv(s_oracle))

                    # if not bmm_score_all[cat] is None and not beta_lh[k] is None:
                    #     s_beta = bmm_score_all[cat].likelihood(im['annotations'][j]['score'], 1) * beta_lh[cat].pdf(np.array([lh_exp_norm(im['annotations'][j][gmm_key])]))[0]
                    #     if s_beta == np.inf:
                    #         s_beta = 0
                    #     rectify_idx.append([i, j, s, cat, s_beta])
                    #     rectify_norm[cat] = max(rectify_norm[cat], s_beta)
                    break
    # print(rectify_norm)
    # for i, j, s, cat, s_beta in rectify_idx:
    #     # raw_bboxes[i]['annotations'][j]['score'] = s_beta / rectify_norm[cat] * 0.98 + 0.01
    #     raw_bboxes[i]['annotations'][j]['score'] = _sigmoid(_sigmoid_inv(s) + _sigmoid_inv(s_beta / rectify_norm[cat] * 0.98 + 0.01))
    return raw_bboxes


def density_test():
    class DummyGMM(object):
        def __init__(self, n_components):
            self.n_components = n_components
            self.means_ = np.zeros(shape=(self.n_components, 1), dtype=np.float32)
        def predict_proba(self, X):
            return np.ones(shape=(X.shape[0], self.n_components), dtype=np.float32) / self.n_components

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.anno_models = sorted(['r101-fpn-3x', 'r50-fpn-3x'])
    args.refine_det_score_thres = 0.5
    args.refine_iou_thres = 0.85
    args.refine_remove_no_sot = False
    args.gmm_max_samples = 100000
    args.gmm_n_components = 25
    # args.kde_max_samples = 15000

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    with gzip.open(os.path.join(os.path.dirname(__file__), 'gmm_test_results.json.gz'), 'rt') as fp:
        data = json.loads(fp.read())
    results_AP, masked_bboxes_per_video, raw_bboxes_per_video = data['AP'], data['masked_bboxes'], data['raw_bboxes']

    vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    # vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '034', '050', '169']
    # vid_list = ['001', '003', '005', '006', '012', '020', '034', '050', '169']
    # vid_list = ['001', '050']
    # vid_list = list(map(lambda x: x['id'], files))[:30]
    # vid_list = list(map(lambda x: x['id'], files))

    # models = ['r101-fpn-3x', 'r50-fpn-3x']
    models = ['r101-fpn-3x']
    categories = ['person', 'vehicle', 'overall', 'weighted']

    improvements = {}
    for vid in vid_list:
        images_pseudo_anno = None
        args.id = vid

        gmm_models_2d_f = os.path.join(os.path.dirname(__file__), 'gmm_cache', 'gmm_%s_%s.pickle' % (args.id, '_'.join(args.anno_models)))
        if os.access(gmm_models_2d_f, os.R_OK):
            with open(gmm_models_2d_f, 'rb') as fp:
                data = pickle.load(fp)
                assert data['args'] == vars(args), 'arguments mismatch, clear cache file\ncached:\n%s\ncurrent:\n%s' % (data['args'], vars(args))
                gmm_models_2d = data['models']
            print('loaded cached model from', gmm_models_2d_f)
        else:
            if images_pseudo_anno is None:
                images_pseudo_anno = refine_annotations(args)[0]
            images_pseudo_anno, gmm_models_2d = bbox_GMM_2D_XYXY(images_pseudo_anno, args)
            with open(gmm_models_2d_f, 'wb') as fp:
                pickle.dump({'models': gmm_models_2d, 'args': vars(args)}, fp)
        print('2D GMM:', gmm_models_2d)

        gmm_models_2d_xywh_f = os.path.join(os.path.dirname(__file__), 'gmm_cache', 'gmm_2d_xywh_%s_%s.pickle' % (args.id, '_'.join(args.anno_models)))
        if os.access(gmm_models_2d_xywh_f, os.R_OK):
            with open(gmm_models_2d_xywh_f, 'rb') as fp:
                data = pickle.load(fp)
                assert data['args'] == vars(args), 'arguments mismatch, clear cache file\ncached:\n%s\ncurrent:\n%s' % (data['args'], vars(args))
                gmm_models_2d_xywh = data['models']
            print('loaded cached model from', gmm_models_2d_xywh_f)
        else:
            if images_pseudo_anno is None:
                images_pseudo_anno = refine_annotations(args)[0]
            images_pseudo_anno, gmm_models_2d_xywh = bbox_GMM_2D_XYWH(images_pseudo_anno, args)
            with open(gmm_models_2d_xywh_f, 'wb') as fp:
                pickle.dump({'models': gmm_models_2d_xywh, 'args': vars(args)}, fp)
        print('2D XYWH GMM:', gmm_models_2d_xywh)

        # gmm_models_3d_f = os.path.join(os.path.dirname(__file__), 'gmm_cache', 'gmm_3d_%s_%s.pickle' % (args.id, '_'.join(args.anno_models)))
        # if os.access(gmm_models_3d_f, os.R_OK):
        #     with open(gmm_models_3d_f, 'rb') as fp:
        #         data = pickle.load(fp)
        #         assert data['args'] == vars(args), 'arguments mismatch, clear cache file\ncached:\n%s\ncurrent:\n%s' % (data['args'], vars(args))
        #         gmm_models_3d = data['models']
        #     print('loaded cached model from', gmm_models_3d_f)
        # else:
        #     if images_pseudo_anno is None:
        #         images_pseudo_anno = refine_annotations(args)[0]
        #     images_pseudo_anno, gmm_models_3d = bbox_GMM_3D(images_pseudo_anno, args)
        #     with open(gmm_models_3d_f, 'wb') as fp:
        #         pickle.dump({'models': gmm_models_3d, 'args': vars(args)}, fp)
        # print('3D GMM:', gmm_models_3d)

        improvements[args.id] = {}
        # bbox_depth_estimator = BBoxDepthEstimator(args.id)
        for m in models:
            improvements[args.id][m] = {}
            raw_bboxes, masked_bboxes, gt_bboxes = map(copy.deepcopy, [raw_bboxes_per_video[m][args.id], masked_bboxes_per_video[m][args.id]['detections'], masked_bboxes_per_video[m][args.id]['annotations']])
            ap_cached = results_AP[m]['manual_%s' % args.id]['results']

            # compute GMM likelihoods
            for im in tqdm.tqdm(raw_bboxes, ascii=True, desc='%s %s' % (args.id, m)):
                for ann in im['annotations']:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    x1, y1, x2, y2 = ann['bbox']
                    ann['bbox_norm'] = [x1 / im['width'], y1 / im['height'], x2 / im['width'], y2 / im['height']]
                    ann['gmm_2d_log_likelihood'] = gmm_models_2d['per_class'][ann['category_id']].score_samples([ann['bbox_norm']])[0]

                for ann in im['annotations']:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    x1, y1, x2, y2 = ann['bbox']
                    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    ann['bbox_xy_log_wh'] = [xc / gmm_models_2d_xywh['gmm_log_likelihood_stats']['W'], yc / gmm_models_2d_xywh['gmm_log_likelihood_stats']['H'], math.log(w) / gmm_models_2d_xywh['gmm_log_likelihood_stats']['logWH_norm'], math.log(h) / gmm_models_2d_xywh['gmm_log_likelihood_stats']['logWH_norm']]
                    ann['gmm_2d_xywh_log_likelihood'] = gmm_models_2d_xywh['per_class'][ann['category_id']].score_samples([ann['bbox_xy_log_wh']])[0]

                # _im = copy.deepcopy(im)
                # _im['file_name'] = os.path.join(os.path.basename(__file__), '..', '..', '..', 'images', 'annotated', args.id, 'unmasked', _im['file_name'])
                # bbox_depth_estimator.estimate(_im)
                # bbox_depth_estimator.make_3d_bbox(_im, gmm_models_3d['depth_stats'][0], gmm_models_3d['depth_stats'][1])
                # im['annotations'] = _im['annotations']
                # for ann in im['annotations']:
                #     ann['gmm_3d_log_likelihood'] = gmm_models_3d['per_class'][ann['category_id']].score_samples([ann['bbox_3D']])[0]

            if False:
                def filter_50(im):
                    im = copy.deepcopy(im)
                    im['annotations'] = list(filter(lambda x: x['score'] > 0.5, im['annotations']))
                    return im
                images_show = list(map(filter_50, raw_bboxes))
                images_show = list(filter(lambda x: len(x['annotations']) > 0, images_show))
                images_show = sklearn.utils.shuffle(images_show)[:4]
                fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
                axes = list(axes[0]) + list(axes[1])
                for i in range(0, 4):
                    im_arr = skimage.io.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id, 'unmasked', images_show[i]['file_name']))
                    anns = sklearn.utils.shuffle(images_show[i]['annotations'])[:16]
                    axes[i].imshow(im_arr)
                    axes[i].set_title(images_show[i]['file_name'][-30:])
                    for k in range(0, len(anns)):
                        x1, y1, x2, y2 = anns[k]['bbox']
                        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[anns[k]['category_id']], facecolor='none')
                        axes[i].add_patch(rect)
                        axes[i].text((x1 + x2) / 2, (y1 + y2) / 2, '%.2f/%.2f' % (anns[k]['gmm_2d_log_likelihood'], anns[k]['gmm_2d_xywh_log_likelihood']), size=12, color=bbox_rgbs[anns[k]['category_id']])
                plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.05)
                plt.show()

            raw_bboxes = inspect_bbox_scores(raw_bboxes, masked_bboxes, gt_bboxes, ap_cached, '%s_%s' % (args.id, m), rectify=True, save=True)
            inspect_bbox_scores(raw_bboxes, masked_bboxes, gt_bboxes, ap_cached, '%s_%s_rectify' % (args.id, m), rectify=False, save=False)

            ap_raw_rectified = evaluate_masked(args.id, raw_bboxes)['results']
            for k in categories:
                improvements[args.id][m][k] = [ap_raw_rectified[k][0] - ap_cached[k][0], ap_raw_rectified[k][1] - ap_cached[k][1]]
    print(improvements)

    # fig, axes = plt.subplots(len(models), len(categories), figsize=(45, 5))
    fig, axes = plt.subplots(2, 2, figsize=(30, 10))
    axes = list(axes[0]) + list(axes[1])
    for i in range(0, len(categories)):
        # for j in range(0, len(models)):
            j = 0
            _axe = axes[i]
            improvements_cat_m = [{'id': v, 'improvement': improvements[v][models[j]][categories[i]]} for v in improvements]
            improvements_cat_m.sort(key=lambda x: x['improvement'][0])
            xs = np.arange(1, len(improvements_cat_m) + 1, 1)
            _axe.plot([0, xs.max() + 1], [0, 0], 'k-')
            ys_mAP = np.array(list(map(lambda x: x['improvement'][0], improvements_cat_m))) * 100
            _axe.plot(xs, ys_mAP, 'ro-')
            ys_AP50 = np.array(list(map(lambda x: x['improvement'][1], improvements_cat_m))) * 100
            _axe.plot(xs, ys_AP50, 'bx-', alpha=0.75)
            y_max = max(np.absolute(ys_mAP).max(), np.absolute(ys_AP50).max())
            for k in range(0, len(xs)):
                _axe.text(xs[k], ys_mAP[k] + 0.15 * y_max, improvements_cat_m[k]['id'], rotation=-80, horizontalalignment='right', verticalalignment='center', size=9, color='red')
            _axe.legend(['0', 'mAP %.4f' % ys_mAP.mean(), 'AP50 %.4f' % ys_AP50.mean()])
            _axe.set_xticks([])
            _axe.set_xlim(0, xs.max() + 1)
            _axe.set_ylim(-1.1 * y_max, 1.1 * y_max)
            _axe.set_ylabel('AP (0-100)')
            _axe.grid(True)
            _axe.set_title('AP shift <%s> <%s>' % (models[j], categories[i]))
    # plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.94, top=0.94, bottom=0.02)
    # plt.show()
    plt.savefig('improvements.pdf')


class BetaMixtureRegressor(object):
    def __init__(self):
        pass

    def fit(self, likelihoods, scores):
        assert likelihoods.shape[0] == scores.shape[0]
        self.beta_lh = Beta().fit(likelihoods)
        pdfs = self.beta_lh.pdf(likelihoods)
        pdfs[np.isinf(pdfs)] = 0
        self.norm_lh = pdfs.max()
        self.bmm_s = BetaMixture1D().fit(scores)
        pdfs = self.bmm_s.likelihood(scores, 1)
        pdfs[np.isinf(pdfs)] = 0
        self.norm_s = pdfs.max()
        assert self.norm_lh > 0 and self.norm_s > 0 and not np.isinf(self.norm_lh) and not np.isinf(self.norm_s), '%s %s' % (self.norm_lh, self.norm_s)
        # self.p_lh, self.p_s = 1.0, 1.0
        self.p_lh, self.p_s = 0.33, 0.33
        return self

    def fit_IoU(self, likelihoods, scores, ious):
        p_lh_list = np.concatenate([np.arange(0.01, 0.1, 0.05), np.arange(0.1, 1, 0.1), np.arange(1, 10.1, 1)])
        p_s_list = np.concatenate([np.arange(0.01, 0.1, 0.05), np.arange(0.1, 1, 0.1), np.arange(1, 10.1, 1)])
        dists = np.ones(shape=(p_lh_list.shape[0], p_s_list.shape[0]), dtype=np.float32) * 100
        for i in range(0, p_lh_list.shape[0]):
            for j in range(0, p_s_list.shape[0]):
                dists[i, j] = np.absolute(self.pdf_normalized(likelihoods, scores, p_lh_list[i], p_s_list[j]) - ious).mean()
        min_i, min_j = np.unravel_index(dists.argmin(), dists.shape)
        self.p_lh, self.p_s = p_lh_list[min_i], p_s_list[min_j]
        return self

    def pdf_normalized(self, likelihoods, scores, p_lh, p_s):
        assert likelihoods.shape[0] == scores.shape[0]
        pdf_lh = self.beta_lh.pdf(likelihoods) / self.norm_lh
        pdf_lh[pdf_lh > 1] = 1.0
        pdf_s = self.bmm_s.likelihood(scores, 1) / self.norm_s
        pdf_s[pdf_s > 1] = 1.0
        # mu, var = self.beta_lh.stats()
        # pdf_lh = np.exp(-1 * np.square((likelihoods - 1) / (np.exp(1) - 1) - mu) / var)
        # mu, var = self.bmm_s.stats()
        # mu, var = mu[1], var[1]
        # pdf_s = np.exp(-1 * np.square(scores - mu) / var)
        return np.power(pdf_lh, p_lh) * np.power(pdf_s, p_s)

    def inference(self, likelihoods, scores):
        return self.pdf_normalized(likelihoods, scores, self.p_lh, self.p_s)

    def __repr__(self):
        return '%s [^%.2f] %s [^%.2f]' % (self.beta_lh, self.p_lh, self.bmm_s, self.p_s)


def test_oracle():
    import functools
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    def _sigmoid_inv(x):
        assert x >= 0 and x <= 1, x
        if x > 0.9999: x = 0.9999
        if x < 0.0001: x = 0.0001
        return np.log(x / (1 - x))

    vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    # vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012']
    # vid_list = ['006']
    categories = ['person', 'vehicle', 'overall', 'weighted']
    gt_bboxes_list, raw_bboxes_list, masked_bboxes_list = [], [], []

    for v in vid_list:
        with open(os.path.join(os.path.dirname(__file__), 'tmp', 'gmm_2d_xywh_log_likelihood', v + '_r101-fpn-3x.json'), 'r') as fp:
            data = json.load(fp)
        gt_bboxes_list.append(data['gt_bboxes'])
        raw_bboxes_list.append(data['raw_bboxes'])
        masked_bboxes_list.append(data['masked_bboxes'])

    gmm_key = 'gmm_2d_xywh_log_likelihood'
    normalizers = [{k: None for k in range(0, len(thing_classes))} for _ in range(0, len(vid_list))]
    oracles = [{k: None for k in range(0, len(thing_classes))} for _ in range(0, len(vid_list))]
    oracle_allvideos, score_lh_iou_allvideos = [None for _ in range(0, len(thing_classes))], [[] for _ in range(0, len(thing_classes))]
    bmms = [{k: None for k in range(0, len(thing_classes))} for _ in range(0, len(vid_list))]

    for i in tqdm.tqdm(range(0, len(vid_list)), ascii=True, desc='learning rectifiers'):
        _gt, _raw, _masked = gt_bboxes_list[i], raw_bboxes_list[i], masked_bboxes_list[i]
        for k in range(0, len(thing_classes)):
            score_llh_all = functools.reduce(lambda x, y: x + y, map(lambda z: z['annotations'], _raw), [])
            score_llh_all = filter(lambda x: x['category_id'] == k, score_llh_all)
            score_llh_all = np.array(list(map(lambda x: [x['score'], x[gmm_key]], score_llh_all)))
            normalizers[i][k] = {'min': score_llh_all[:, 1].min(), 'max': score_llh_all[:, 1].max()}
            lh_exp_norm = lambda x: np.exp((x - score_llh_all[:, 1].min()) / (score_llh_all[:, 1].max() - score_llh_all[:, 1].min()))
            score_lh_iou = functools.reduce(lambda x, y: x + y, map(lambda z: z['annotations'], _masked), [])
            score_lh_iou = filter(lambda x: x['category_id'] == k, score_lh_iou)
            score_lh_iou = np.array(list(map(lambda x: [x['score'], lh_exp_norm(x[gmm_key]), x['max_IoU']], score_lh_iou)))
            if score_lh_iou.shape[0] > 0:
                score_lh_iou_allvideos[k].append(score_lh_iou)
            # if score_lh_iou.shape[0] > 30:
            #     while True:
            #         try:
            #             oracles[i][k] = MLPRegressor().fit_Xy(score_lh_iou[:, :2], score_lh_iou[:, 2])
            #         except:
            #             print('re-train')
            #             continue
            #         break
            if score_lh_iou.shape[0] > 30:
                bmms[i][k] = BetaMixtureRegressor().fit(lh_exp_norm(score_llh_all[:, 1]), score_llh_all[:, 0])
                # bmms[i][k].fit_IoU(score_lh_iou[:, 1], score_lh_iou[:, 0], score_lh_iou[:, 2])
        del _gt, _raw, _masked, score_llh_all, lh_exp_norm, score_lh_iou

    beta_stats = {k: {'score': {'alpha': [], 'beta': [], 'p': []}, 'likelihood': {'alpha': [], 'beta': [], 'p': []}} for k in range(0, len(thing_classes))}
    for i in range(0, len(vid_list)):
        for k in range(0, len(thing_classes)):
            if not bmms[i][k] is None:
                beta_stats[k]['score']['alpha'].append(bmms[i][k].bmm_s.alphas[1])
                beta_stats[k]['score']['beta'].append(bmms[i][k].bmm_s.betas[1])
                beta_stats[k]['score']['p'].append(bmms[i][k].p_s)
                beta_stats[k]['likelihood']['alpha'].append(bmms[i][k].beta_lh.alpha)
                beta_stats[k]['likelihood']['beta'].append(bmms[i][k].beta_lh.beta)
                beta_stats[k]['likelihood']['p'].append(bmms[i][k].p_lh)
    for k in range(0, len(thing_classes)):
        for k2 in ['score', 'likelihood']:
            for k3 in ['alpha', 'beta', 'p']:
                beta_stats[k][k2][k3] = np.array(beta_stats[k][k2][k3])
            beta_stats[k][k2]['std'] = (beta_stats[k][k2]['alpha'] * beta_stats[k][k2]['beta'] / ((beta_stats[k][k2]['alpha'] + beta_stats[k][k2]['beta'] + 1) * (beta_stats[k][k2]['alpha'] + beta_stats[k][k2]['beta']) ** 2)) ** 0.5
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for k in range(0, len(thing_classes)):
        _axe = axes[k][0]
        _axe.set_title('%s score' % thing_classes[k])
        _axe.scatter(beta_stats[k]['score']['std'], beta_stats[k]['score']['p'])
        _axe.set_yscale('log')
        _axe.set_xlabel('std')
        _axe.set_ylabel('p')
        _axe = axes[k][1]
        _axe.set_title('%s likelihood' % thing_classes[k])
        _axe.scatter(beta_stats[k]['likelihood']['std'], beta_stats[k]['likelihood']['p'])
        _axe.set_yscale('log')
        _axe.set_xlabel('std')
        _axe.set_ylabel('p')
    plt.tight_layout()
    plt.show()
    # raise

    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # for k in range(0, len(thing_classes)):
    #     score_lh_iou_allvideos[k] = sklearn.utils.shuffle(np.concatenate(score_lh_iou_allvideos[k], axis=0))
    #     print('all videos bboxes <%s> %s' % (thing_classes[k], score_lh_iou_allvideos[k].shape))
    #     oracle_allvideos[k] = MLPRegressor().fit_Xy(score_lh_iou_allvideos[k][:, :2], score_lh_iou_allvideos[k][:, 2])
    #     _axe = axes[k][0]
    #     _axe.scatter(score_lh_iou_allvideos[k][:, 0], score_lh_iou_allvideos[k][:, 1], c=score_lh_iou_allvideos[k][:, 2], cmap='viridis', marker='o', s=12, alpha=0.5)
    #     _axe.set_xlim(-0.02, 1.02)
    #     _axe.set_xlabel('score')
    #     _axe.set_ylim(0.98, 2.74)
    #     _axe.set_ylabel('likelihood (%s)' % gmm_key)
    #     _axe.set_title('IoU <%s> [%d]' % (thing_classes[k], score_lh_iou_allvideos[k].shape[0]))
    #     _axe = axes[k][1]
    #     xys = np.mgrid[0 : 1.001 : 0.02, 1 : 2.72 : 0.03].transpose(1, 2, 0).reshape(-1, 2)
    #     _axe.scatter(xys[:, 0], xys[:, 1], c=oracle_allvideos[k].inference(xys), cmap='viridis', marker='o', s=12, alpha=0.5)
    #     _axe.set_xlim(-0.02, 1.02)
    #     _axe.set_xlabel('score')
    #     _axe.set_ylim(0.98, 2.74)
    #     _axe.set_ylabel('likelihood (%s)' % gmm_key)
    #     _axe.set_title('IoU regression')
    # plt.tight_layout()
    # plt.savefig('oracle_allvideos.png')

    AP_list, AP_merge, AP_average = [], None, {}
    for i in range(0, len(vid_list)):
        AP_list.append(eval_AP(gt_bboxes_list[i], raw_bboxes_list[i])['results'])
    for cat in categories:
        AP_average[cat] = []
        for i in range(0, len(AP_list)):
            if AP_list[i][cat][0] >= 0:
                AP_average[cat].append(AP_list[i][cat])
        AP_average[cat] = np.array(AP_average[cat]).mean(axis=0).tolist()
    AP_merge = eval_AP(*map(lambda l: functools.reduce(lambda x, y: x + y, l, []), [gt_bboxes_list, raw_bboxes_list]))['results']
    print('averaged\n', AP_average)
    print('merged\n', AP_merge)

    for alpha in np.arange(0, 1.01, 0.1).tolist():
        assert 0 <= alpha <= 1, alpha
        raw_bboxes_rectified_list = []
        for i in range(0, len(vid_list)):
            _raw_rectified = copy.deepcopy(raw_bboxes_list[i])
            for im in _raw_rectified:
                for ann in im['annotations']:
                    s, cat = ann['score'], ann['category_id']
                    # if not oracles[i][cat] is None:
                    #     s_oracle = float(oracles[i][cat].inference(np.array([[s, np.exp((ann[gmm_key] - normalizers[i][cat]['min']) / (normalizers[i][cat]['max'] - normalizers[i][cat]['min']))]]))[0])
                    #     # s_oracle = float(oracle_allvideos[cat].inference(np.array([[s, np.exp((ann[gmm_key] - normalizers[i][cat]['min']) / (normalizers[i][cat]['max'] - normalizers[i][cat]['min']))]]))[0])
                    #     ann['score'] = _sigmoid(_sigmoid_inv(s) * alpha + _sigmoid_inv(s_oracle) * (1 - alpha))
                    if not bmms[i][cat] is None:
                        s_bmm = float(bmms[i][cat].inference(np.array([np.exp((ann[gmm_key] - normalizers[i][cat]['min']) / (normalizers[i][cat]['max'] - normalizers[i][cat]['min']))]), np.array([s]))[0])
                        ann['score'] = _sigmoid(_sigmoid_inv(s) * alpha + _sigmoid_inv(s_bmm) * (1 - alpha))
            raw_bboxes_rectified_list.append(_raw_rectified)

        AP_rectified_list, AP_rectified_merge, AP_rectified_average = [], None, {}
        for i in range(0, len(vid_list)):
            AP_rectified_list.append(eval_AP(gt_bboxes_list[i], raw_bboxes_rectified_list[i])['results'])
        for cat in categories:
            AP_rectified_average[cat] = []
            for i in range(0, len(AP_rectified_list)):
                if AP_rectified_list[i][cat][0] >= 0:
                    AP_rectified_average[cat].append(AP_rectified_list[i][cat])
            AP_rectified_average[cat] = np.array(AP_rectified_average[cat]).mean(axis=0).tolist()
        AP_rectified_merge = eval_AP(*map(lambda l: functools.reduce(lambda x, y: x + y, l, []), [gt_bboxes_list, raw_bboxes_rectified_list]))['results']
        print('rectified averaged\n', AP_rectified_average)
        print('rectified merged\n', AP_rectified_merge)

        improvements = []
        for i in range(0, len(vid_list)):
            _impr = {}
            for cat in categories:
                _impr[cat] = [AP_rectified_list[i][cat][0] - AP_list[i][cat][0], AP_rectified_list[i][cat][1] - AP_list[i][cat][1]]
            improvements.append(_impr)

        fig, axes = plt.subplots(2, 2, figsize=(30, 10))
        axes = axes.reshape(-1)
        for k in range(0, len(categories)):
            _axe = axes[k]
            improvements_cat = [{'id': vid_list[i], 'improvement': improvements[i][categories[k]]} for i in range(0, len(vid_list))]
            improvements_cat.sort(key=lambda x: x['improvement'][0])
            xs = np.arange(1, len(improvements_cat) + 1, 1)
            _axe.plot([0, xs.max() + 1], [0, 0], 'k-')
            ys_mAP = np.array(list(map(lambda x: x['improvement'][0], improvements_cat))) * 100
            _axe.plot(xs, ys_mAP, 'ro-')
            ys_AP50 = np.array(list(map(lambda x: x['improvement'][1], improvements_cat))) * 100
            _axe.plot(xs, ys_AP50, 'bx-', alpha=0.75)
            y_max = max(np.absolute(ys_mAP).max(), np.absolute(ys_AP50).max())
            for i in range(0, len(xs)):
                _axe.text(xs[i], ys_mAP[i] + 0.15 * y_max, improvements_cat[i]['id'], rotation=-80, horizontalalignment='right', verticalalignment='center', size=9, color='red')
            _axe.legend(['0', 'mAP %.4f' % ys_mAP.mean(), 'AP50 %.4f' % ys_AP50.mean()])
            _axe.set_xticks([])
            _axe.set_xlim(0, xs.max() + 1)
            _axe.set_ylim(-1.1 * y_max, 1.1 * y_max)
            _axe.set_ylabel('AP (0-100)')
            _axe.grid(True)
            _axe.set_title('AP shift <%s>' % categories[k])
        plt.subplots_adjust(left=0.02, right=0.94, top=0.94, bottom=0.02)
        # plt.savefig('improvements_oracle_alpha%.1f.pdf' % alpha)
        plt.savefig('improvements_bmm_alpha%.1f.pdf' % alpha)

        improvements_average, improvements_merge = {}, {}
        for cat in categories:
            improvements_average[cat] = [AP_rectified_average[cat][0] - AP_average[cat][0], AP_rectified_average[cat][1] - AP_average[cat][1]]
            improvements_merge[cat] = [AP_rectified_merge[cat][0] - AP_merge[cat][0], AP_rectified_merge[cat][1] - AP_merge[cat][1]]
        print('\nimprovements averaged\n', improvements_average)
        print('improvements merged\n', improvements_merge)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--id', type=str, help='video ID')
    # args = parser.parse_args()
    # args.anno_models = sorted(['r101-fpn-3x', 'r50-fpn-3x'])
    # args.refine_det_score_thres = 0.5
    # args.refine_iou_thres = 0.85
    # args.refine_remove_no_sot = False
    # args.gmm_max_samples = 100000
    # args.gmm_n_components = 25
    # images_pseudo_anno = refine_annotations(args)[0][:1000]
    # images_pseudo_anno, gmm_models_3d = bbox_GMM_3D(images_pseudo_anno, args, compute_likelihood=True, show_boxes=True)
    # print(gmm_models_3d)

    # X = np.array([[0, 0], [0, 1], [1, 0], [3, 3], [2, 3], [3, 2], [1, 1]])
    # y = np.array([0, 0, 0, 1, 1, 1, 1])
    # svm = SVC(C=1, kernel='linear').fit(X, y)
    # (a, b), c = svm.coef_[0].tolist(), svm.intercept_[0]
    # print(a, b, c)

    # plt.figure(figsize=(8, 8))
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='_')
    # plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+')
    # plt.plot([0, 1], [-1 * c / b, -1 * (a + c) / b], 'k-')
    # plt.xlim(-5, 15)
    # plt.ylim(-5, 15)
    # plt.show()

    # density_test()
    test_oracle()
