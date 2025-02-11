#!python3

'''
our implementation of
https://github.com/MCG-NJU/TIA

This file should be more self-contained
'''

import os
import sys
import types
import enum
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
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

import sklearn.utils

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_tia')


def get_unlabeled_dicts(args):
    lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', args.id))
    with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
        meta = json.load(fp)
    ifilelist = meta['ifilelist']
    dict_json = []
    for i in range(0, len(ifilelist)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': []})
    print('unlabeled frames of video %s at %s: %d images' % (args.id, lmdb_path, len(dict_json)))
    return dict_json

def all_unlabeled_dicts(args, total_images):
    random.seed(42)
    images_per_video_cap = int(total_images / len(video_id_list))
    dict_json_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        dict_json_v = get_unlabeled_dicts(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all = dict_json_all + dict_json_v
    args.id = id_back
    for i in range(0, len(dict_json_all)):
        dict_json_all[i]['image_id'] = i + 1
    print('all videos %d images' % len(dict_json_all))
    return dict_json_all


def get_annotation_dict(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))), flush=True)
    return annotations

def all_annotation_dict(args):
    annotations_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        annotations_all = annotations_all + get_annotation_dict(args)
    args.id = id_back
    for i in range(0, len(annotations_all)):
        annotations_all[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(annotations_all), sum(list(map(lambda x: len(x['annotations']), annotations_all)))), flush=True)
    return annotations_all


######################################################
#####   many RCNN library methods are modified   #####
##### modded RCNN only tested on detectron2 v0.6 #####
#####  with models: R50-FPN, R101-FPN, X101-FPN  #####
######################################################


class TrainingSchema(enum.IntEnum):
    AUX   = 0
    ADAPT = 1


class DomainLabel(enum.IntEnum):
    SOURCE = 0
    TARGET = 1


class GradientReversalFunction(torch.autograd.Function):
    """
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None
# m1 = torch.nn.Linear(2, 2)
# m2 = torch.nn.Linear(2, 1)
# opt = torch.optim.Adam(list(m1.parameters()) + list(m2.parameters()), lr=0.01)
# x = torch.from_numpy(np.array([[1.0, 2.0]])).float()
# y = torch.from_numpy(np.array([0.0])).float()
# for p in m1.parameters(): p.requires_grad = False
# for p in m2.parameters(): p.requires_grad = False
# for _ in range(0, 10):
#     opt.zero_grad()
#     L = torch.square(m2(GradientReversalFunction.apply(m1(x), 1.0)) - y).sum()
#     L.backward()
#     opt.step()
#     print(L)
# exit(0)


class FPNDomainClassifier(torch.nn.Module):
    def __init__(self, fpn_level):
        super(FPNDomainClassifier, self).__init__()
        conv_params = {
            'p2': [(7, 4), (5, 2), (3, 1)],
            'p3': [(7, 4), (5, 1), (3, 1)],
            'p4': [(5, 2), (3, 1), (3, 1)],
            'p5': [(3, 1), (3, 1), (3, 1)],
            'p6': [(3, 1), (3, 1), (3, 1)],
            'box': [(3, 1), (3, 1), (1, 1)]
        }
        assert fpn_level in conv_params
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, conv_params[fpn_level][0][0], stride=conv_params[fpn_level][0][1]),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, conv_params[fpn_level][1][0], stride=conv_params[fpn_level][1][1]),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, conv_params[fpn_level][2][0], stride=conv_params[fpn_level][2][1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(128),
        )
        self.fc = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNTIA(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs] if 'instances' in batched_inputs[0] else None
        features = self.backbone(images.tensor)
        self.roi_heads.box_predictor.current_image_batch = len(batched_inputs)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            raise Exception('visualization of TIA training not supported')

        if self.train_schema == TrainingSchema.AUX:
            losses_da_image = {}
        elif self.train_schema == TrainingSchema.ADAPT:
            # image level domain classifier loss
            domain_gt = torch.tensor(list(map(lambda x: x['domain'].value, batched_inputs))).long().cuda()
            loss_da_p4 = self.DA_loss_fn(self.DA_p4(GradientReversalFunction.apply(features['p4'], 1.0)), domain_gt)
            loss_da_p5 = self.DA_loss_fn(self.DA_p5(GradientReversalFunction.apply(features['p5'], 1.0)), domain_gt)
            loss_da_p6 = self.DA_loss_fn(self.DA_p6(GradientReversalFunction.apply(features['p6'], 1.0)), domain_gt)

            # proposal level domain classifier loss
            da_scores_box = []
            for im_i in range(0, len(proposals)):
                features_pooled = self.roi_heads.box_pooler(
                    [features[f][im_i : im_i + 1] for f in self.roi_heads.box_in_features],
                    [proposals[im_i].proposal_boxes]
                )
                da_scores_box.append(self.DA_box(GradientReversalFunction.apply(features_pooled, 1.0)))
            count_boxes = np.array(list(map(lambda x: x.size(0), da_scores_box)))
            if count_boxes.min() != count_boxes.max():
                print('in-batch proposals number mismatch', count_boxes.tolist())
                da_scores_box = list(map(lambda x: x[:count_boxes.min()], da_scores_box))
            da_scores_box = torch.stack(da_scores_box, dim=0)
            domain_gt_box = domain_gt.view(-1, 1).expand(domain_gt.size(0), da_scores_box.size(1))
            loss_da_box = self.DA_loss_fn(da_scores_box.view(-1, 2), domain_gt_box.reshape(-1))

            losses_da_image = {}
            loss_da_image = (loss_da_p4 + loss_da_p5 + loss_da_p6) / 3.0
            if torch.isnan(loss_da_image):
                print('image-level domain classifier loss is NaN, set to 0')
                losses_da_image['loss_da_image'] = self.dummy_loss
            else:
                losses_da_image['loss_da_image'] = loss_da_image
            if torch.isnan(loss_da_box):
                print('box-level domain classifier loss is NaN, set to 0')
                losses_da_image['loss_da_box'] = self.dummy_loss
            else:
                losses_da_image['loss_da_box'] = loss_da_box
        else:
            raise NotImplementedError

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(losses_da_image)
        return losses

    @staticmethod
    def create_from_sup(net, tia_n_cls, tia_m_loc, tia_l_da, tia_l_da_cls, tia_l_da_loc):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.tia_n_cls, net.tia_m_loc = tia_n_cls, tia_m_loc
        net.tia_l_da, net.tia_l_da_cls, net.tia_l_da_loc = tia_l_da, tia_l_da_cls, tia_l_da_loc
        net.train_schema = TrainingSchema.AUX
        print('creating domain classifiers on FPN')
        net.DA_p4 = FPNDomainClassifier('p4').cuda()
        net.DA_p5 = FPNDomainClassifier('p5').cuda()
        net.DA_p6 = FPNDomainClassifier('p6').cuda()
        net.DA_box = FPNDomainClassifier('box').cuda()
        net.DA_loss_fn = torch.nn.CrossEntropyLoss()
        net.dummy_loss = torch.tensor(0.0).float().cuda()
        net.dummy_loss.requires_grad = False
        print('creating ROI classifiers & localizers')
        net.roi_heads.box_predictor = FastRCNNOutputLayersTIA.create_from_sup(net.roi_heads.box_predictor, tia_n_cls, tia_m_loc, tia_l_da_cls, tia_l_da_loc)
        net.__class__ = GeneralizedRCNNTIA
        return net


# wrap detectron2/modeling/roi_heads/fast_rcnn.py:FastRCNNOutputLayers
class FastRCNNOutputLayersTIA(detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers):
    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        if not self.training:
            return scores, proposal_deltas

        if self.train_schema == TrainingSchema.AUX:
            x_detach = x.detach()
            scores_aux = [net(x_detach) for net in self.aux_cls_score_list]
            proposal_deltas_aux = [net(x_detach) for net in self.aux_bbox_pred_list]
            return scores, proposal_deltas, scores_aux, proposal_deltas_aux

        elif self.train_schema == TrainingSchema.ADAPT:
            scores_aux = [net(GradientReversalFunction.apply(x, 1.0)) for net in self.aux_cls_score_list]
            proposal_deltas_aux = [net(GradientReversalFunction.apply(x, 1.0)) for net in self.aux_bbox_pred_list]
            x_detach = x.detach()
            scores_aux_detach = [net(x_detach) for net in self.aux_cls_score_list]
            proposal_deltas_aux_detach = [net(x_detach) for net in self.aux_bbox_pred_list]
            return scores, proposal_deltas, scores_aux, proposal_deltas_aux, scores_aux_detach, proposal_deltas_aux_detach

        else:
            raise NotImplementedError

    def losses(self, predictions, proposals):
        if self.train_schema == TrainingSchema.AUX:
            scores, proposal_deltas, scores_aux, proposal_deltas_aux = predictions
            gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
            _log_classification_stats(scores, gt_classes)
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, 'Proposals should not require gradients!'
                gt_boxes = cat([(p.gt_boxes if p.has('gt_boxes') else p.proposal_boxes).tensor for p in proposals], dim=0)
            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            losses = {
                'loss_cls': cross_entropy(scores, gt_classes, reduction='mean'),
                'loss_box_reg': self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
            }

            # loss of auxiliary classifiers & localizers
            for i in range(0, self.tia_n_cls):
                losses['loss_cls_aux_%d' % i] = cross_entropy(scores_aux[i], gt_classes, reduction='mean')
            for i in range(0, self.tia_m_loc):
                losses['loss_box_reg_aux_%d' % i] = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas_aux[i], gt_classes)

            return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

        elif self.train_schema == TrainingSchema.ADAPT:
            scores, proposal_deltas, scores_aux, proposal_deltas_aux, scores_aux_detach, proposal_deltas_aux_detach = predictions
            gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
            _log_classification_stats(scores, gt_classes)
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, 'Proposals should not require gradients!'
                gt_boxes = cat([(p.gt_boxes if p.has('gt_boxes') else p.proposal_boxes).tensor for p in proposals], dim=0)
            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            losses = {
                'loss_cls': cross_entropy(scores, gt_classes, reduction='mean'),
                'loss_box_reg': self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes),
            }

            # loss of auxiliary classifiers & localizers
            losses['loss_cls_aux_sum'] = torch.stack([cross_entropy(scores_aux_detach[i], gt_classes, reduction='mean') for i in range(0, self.tia_n_cls)], dim=0).sum()
            losses['loss_box_reg_aux_sum'] = torch.stack([self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas_aux_detach[i], gt_classes) for i in range(0, self.tia_m_loc)], dim=0).sum()

            # inconsistency-aware losses for classifiers, see Sec 3.2.1 & Sec 3.2.2 in paper
            M_cls = torch.stack(scores_aux, dim=2) # (image_batchsize * roi_batchsize) x 3 x tia_n_cls
            M_cls_softmax = torch.nn.functional.softmax(M_cls, dim=2)
            Q = M_cls_softmax.mean(dim=2)
            P = (-1.0 * M_cls_softmax * torch.log(M_cls_softmax)).sum(dim=2)
            loss_ia_cls = (-1.0 * (P * Q).sum(dim=1)).mean()
            if torch.isnan(loss_ia_cls):
                print('inconsistency-aware loss for classifiers is NaN, set to 0')
                losses['loss_ia_cls'] = self.dummy_loss
            else:
                losses['loss_ia_cls'] = loss_ia_cls

            # inconsistency-aware losses for localizers, see Sec 3.2.1 & Sec 3.2.3 in paper
            M_loc = torch.stack(proposal_deltas_aux, dim=2) # (image_batchsize * roi_batchsize) * 8 * tia_m_loc
            M_loc_mean = M_loc.mean(dim=2, keepdims=True)
            M_loc_var = torch.square(M_loc - M_loc_mean).sum(dim=2) / (self.tia_m_loc ** 0.5)
            loss_ia_loc = M_loc_var.mean()
            if torch.isnan(loss_ia_loc):
                print('inconsistency-aware loss for localizers is NaN, set to 0')
                losses['loss_ia_loc'] = self.dummy_loss
            else:
                losses['loss_ia_loc'] = loss_ia_loc

            return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

        else:
            raise NotImplementedError

    @staticmethod
    def create_from_sup(net, tia_n_cls, tia_m_loc, tia_l_da_cls, tia_l_da_loc):
        net.tia_n_cls, net.tia_m_loc = tia_n_cls, tia_m_loc
        net.tia_l_da_cls, net.tia_l_da_loc = tia_l_da_cls, tia_l_da_loc
        net.train_schema = TrainingSchema.AUX
        print('creating %d AUX classifiers' % net.tia_n_cls)
        net.aux_cls_score_list = torch.nn.ModuleList([copy.deepcopy(net.cls_score) for _ in range(0, net.tia_n_cls)])
        print('creating %d AUX localizers' % net.tia_m_loc)
        net.aux_bbox_pred_list = torch.nn.ModuleList([copy.deepcopy(net.bbox_pred) for _ in range(0, net.tia_m_loc)])
        net.dummy_loss = torch.tensor(0.0).float().cuda()
        net.dummy_loss.requires_grad = False
        net.__class__ = FastRCNNOutputLayersTIA
        return net


# DefaultTrainer._trainer is instance of SimpleTrainer
# DefaultTrainer & SimpleTrainer are subclass of TrainerBase
def finetune_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    # print(self.model.proposal_generator.rpn_head.anchor_deltas.weight.data.min().item(), self.model.roi_heads.box_predictor.cls_score.weight.data.min().item(), self.model.roi_heads.box_predictor.bbox_pred.weight.data.min().item())
    # print(self.model.roi_heads.box_predictor.aux_cls_score_list[0].weight.data.min().item(), self.model.roi_heads.box_predictor.aux_bbox_pred_list[0].weight.data.min().item())
    # print(self.model.DA_p6.fc.weight.data.min().item())

    if self.model.train_schema == TrainingSchema.AUX:
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        loss_dict = self.model(data)
        loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
        assert not isinstance(loss_dict, torch.Tensor)
        losses = sum(loss_dict.values())

    elif self.model.train_schema == TrainingSchema.ADAPT:
        start = time.perf_counter()
        data_src = list(next(self._data_loader_iter))
        data_tgt = list(next(self._data_loader_tgt_iter))
        assert len(data_tgt) == len(data_src)
        for _d in data_src:
            assert _d['domain'] == DomainLabel.SOURCE
        for _d in data_tgt:
            assert _d['domain'] == DomainLabel.TARGET
        data_time = time.perf_counter() - start

        loss_dict = {
            # FasterRCNN losses
            'loss_rpn_cls': [],
            'loss_rpn_loc': [],
            'loss_cls': [],
            'loss_box_reg': [],
            # aux losses, detached from FPN & RPN
            'loss_cls_aux_sum': [],
            'loss_box_reg_aux_sum': [],
            # domain classifier losses
            'loss_da_image': [],
            'loss_da_box': [],
            # inconsistency-aware losses
            'loss_ia_cls_src': [],
            'loss_ia_loc_src': [],
            'loss_ia_cls_tgt': [],
            'loss_ia_loc_tgt': []
        }
        loss_dict_src = self.model(data_src)
        # for labeled source domain, use all losses
        for k in ['loss_rpn_cls', 'loss_rpn_loc', 'loss_cls', 'loss_box_reg', 'loss_cls_aux_sum', 'loss_box_reg_aux_sum', 'loss_da_image', 'loss_da_box']:
            loss_dict[k].append(loss_dict_src[k])
        loss_dict['loss_ia_cls_src'].append(loss_dict_src['loss_ia_cls'])
        loss_dict['loss_ia_loc_src'].append(loss_dict_src['loss_ia_loc'])
        # for unlabeled target domain, only use DA loss and IA loss
        loss_dict_tgt = self.model(data_tgt)
        for k in ['loss_da_image', 'loss_da_box']:
            loss_dict[k].append(loss_dict_tgt[k])
        loss_dict['loss_ia_cls_tgt'].append(loss_dict_tgt['loss_ia_cls'])
        loss_dict['loss_ia_loc_tgt'].append(loss_dict_tgt['loss_ia_loc'])
        for k in loss_dict:
            if len(loss_dict[k]) < 1:
                loss_dict[k] = torch.tensor(0.0).float().cuda()
            else:
                loss_dict[k] = torch.stack(loss_dict[k], dim=0).mean()
        loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
        # print(loss_dict_items)

        # inconsistency-aware losses are maximized on target domain, minimized on source domain
        for k in ['loss_ia_cls_src', 'loss_ia_loc_src', 'loss_ia_cls_tgt', 'loss_ia_loc_tgt']:
            if abs(loss_dict_items[k]) > 1.5:
                print('%s = %.4f too large, set to 0' % (k, loss_dict_items[k]))
                loss_dict[k] = self.model.dummy_loss
        losses = loss_dict['loss_rpn_cls'] + loss_dict['loss_rpn_loc'] + loss_dict['loss_cls'] + loss_dict['loss_box_reg'] + \
            (loss_dict['loss_cls_aux_sum'] + loss_dict['loss_box_reg_aux_sum']) + \
            (loss_dict['loss_da_image'] + loss_dict['loss_da_box']) * self.model.tia_l_da + \
            (loss_dict['loss_ia_cls_tgt'] * self.model.tia_l_da_cls + loss_dict['loss_ia_loc_tgt'] * self.model.tia_l_da_loc) * -1.0 + \
            (loss_dict['loss_ia_cls_src'] * self.model.tia_l_da_cls + loss_dict['loss_ia_loc_src'] * self.model.tia_l_da_loc) * +1.0

    else:
        raise NotImplementedError

    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


# wrap detectron2/engine/defaults.py:DefaultTrainer
class AUXTrainer(DefaultTrainer):
    def __init__(self, cfg, args):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        model = GeneralizedRCNNTIA.create_from_sup(model, args.tia_n_cls, args.tia_m_loc, args.tia_l_da, args.tia_l_da_cls, args.tia_l_da_loc)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        model = create_ddp_model(model, broadcast_buffers=False)

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []


class APDropException(Exception):
    pass

# wrap detectron2/engine/defaults.py:DefaultTrainer
class AdaptTrainer(DefaultTrainer):
    def __init__(self, cfg, args):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        model = GeneralizedRCNNTIA.create_from_sup(model, args.tia_n_cls, args.tia_m_loc, args.tia_l_da, args.tia_l_da_cls, args.tia_l_da_loc)
        model.train_schema = TrainingSchema.ADAPT
        model.roi_heads.box_predictor.train_schema = TrainingSchema.ADAPT
        optimizer = self.build_optimizer(cfg, model)
        model = create_ddp_model(model, broadcast_buffers=False)

        # hack to make sure source/target domain are sampled balancely
        assert 0 == (cfg.SOLVER.IMS_PER_BATCH % 2), cfg.SOLVER.IMS_PER_BATCH
        cfg_src = copy.deepcopy(cfg)
        cfg_src.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH // 2
        data_loader_src = self.build_train_loader(cfg_src)
        cfg_tgt = copy.deepcopy(cfg)
        cfg_tgt.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH // 2
        cfg_tgt.DATASETS.TRAIN = cfg.DATASETS.TRAIN_TARGET
        data_loader_tgt = self.build_train_loader(cfg_tgt)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader_src, optimizer)
        self._trainer._data_loader_tgt_iter = iter(data_loader_tgt)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []
        self.best_iter, self.top3_mAP, self.best_params = -1, [-1.0, -1.0, -1.0], None

    def build_hooks(self):
        ret = super().build_hooks()
        self.eval_results_all = {}
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            # mAP_i = self._last_eval_results['bbox']['AP']
            # if mAP_i > max(self.top3_mAP):
            #     self.best_iter, self.top3_mAP = self.iter + 0, sorted(self.top3_mAP + [mAP_i + 0.0], reverse=True)[:3]
            #     self.best_params = self.model.state_dict()
            #     for k in list(self.best_params.keys()):
            #         self.best_params[k] = self.best_params[k].cpu()
            # elif mAP_i < min(self.top3_mAP):
            #     raise APDropException
            # else:
            #     self.top3_mAP = sorted(self.top3_mAP + [mAP_i + 0.0], reverse=True)[:3]
            self.eval_results_all[self.iter] = copy.deepcopy(self._last_eval_results)
            return self._last_eval_results
        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)


def train_aux(args):
    args.smallscale = False
    desc_cocotrain, dst_cocotrain = 'mscoco2017_train_remap_tia_pretrain', get_coco_dicts(args, 'train')
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model(args.model)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_cocotrain,)
    cfg.DATASETS.TEST = ()
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = AUXTrainer(cfg, args)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)

    # for net_i in trainer.model.roi_heads.box_predictor.aux_cls_score_list:
    #     net_i.load_state_dict(trainer.model.roi_heads.box_predictor.cls_score.state_dict())
    # for net_i in trainer.model.roi_heads.box_predictor.aux_bbox_pred_list:
    #     net_i.load_state_dict(trainer.model.roi_heads.box_predictor.bbox_pred.state_dict())
    np.random.seed(100)
    torch.manual_seed(100)
    for net_i in trainer.model.roi_heads.box_predictor.aux_cls_score_list:
        torch.nn.init.kaiming_normal_(net_i.weight)
        net_i.bias.data = sklearn.utils.shuffle(trainer.model.roi_heads.box_predictor.cls_score.bias.data)
    for net_i in trainer.model.roi_heads.box_predictor.aux_bbox_pred_list:
        torch.nn.init.kaiming_normal_(net_i.weight)
        net_i.bias.data = sklearn.utils.shuffle(trainer.model.roi_heads.box_predictor.bbox_pred.bias.data)

    for p in trainer.model.parameters(): # freeze whole model
        p.requires_grad = False
    for p in trainer.model.roi_heads.box_predictor.aux_cls_score_list.parameters(): # only train auxiliary modules
        p.requires_grad = True
    for p in trainer.model.roi_heads.box_predictor.aux_bbox_pred_list.parameters():
        p.requires_grad = True
    print(trainer.model.roi_heads)

    trainer.train()
    for p in trainer.model.parameters():
        p.requires_grad = True
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return

    prefix = 'mscoco2017_remap_%s_tia_aux_trained' % args.model
    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    lr_history, loss_history = trainer._trainer.lr_history, trainer._trainer.loss_history
    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_key_list = sorted(list(filter(lambda k: k.startswith('loss_cls') or k.startswith('loss_box_reg'), loss_history[0]['loss'].keys())))
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_key_list:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(14, 10))
    colors = ['#000000', '#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#FF9944', '#99FF44', '#4499FF']
    legends = []
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='-', color='#00FF00')
    legends.append('lr ($\\times$%.1e)' % lr_history[:, 1].max())
    color_i = 0
    for loss_key in filter(lambda k: k.startswith('loss_cls'), loss_key_list):
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle='--', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    color_i = 0
    for loss_key in filter(lambda k: k.startswith('loss_box_reg'), loss_key_list):
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle=':', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(0, lr_history[:, 0].max() * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    _args.smallscale = False
    args.ckpt = os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_%s_tia_aux_trained.pth' % args.model)
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(_args, 'valid')
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]
    for i in range(0, len(dst_cocovalid)):
        dst_cocovalid[i]['image_id'] = i + 1
    desc_cocotrain, dst_cocotrain = 'mscoco2017_train_remap', get_coco_dicts(_args, 'train')
    for im in dst_cocotrain:
        im['domain'] = DomainLabel.SOURCE
    for i in range(0, len(dst_cocotrain)):
        dst_cocotrain[i]['image_id'] = i + 1

    if args.id in video_id_list:
        desc_manual_valid, dst_manual_valid = 'TIA_manual_%s' % args.id, get_annotation_dict(args)
        desc_train, dst_train = 'TIA_unlabeled_%s' % args.id, get_unlabeled_dicts(args)
        for im in dst_train:
            im['domain'] = DomainLabel.TARGET
            im['annotations'] = [{'bbox': [0, 0, 1, 1], 'iscrowd': 0, 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}] # to pass sanity check
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    elif args.id == 'compound':
        args.id = '_compound'
        import functools
        desc_manual_valid, dst_manual_valid = 'TIA_manual_%s' % args.id, all_annotation_dict(args)
        desc_train, dst_train = 'TIA_unlabeled_%s' % args.id, all_unlabeled_dicts(args, args.image_batch_size * args.iters * 1.5)
        # dst_train = functools.reduce(lambda x, y: x + y, dst_train)
        for im in dst_train:
            im['domain'] = DomainLabel.TARGET
            im['annotations'] = [{'bbox': [0, 0, 1, 1], 'iscrowd': 0, 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}] # to pass sanity check
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    else:
        raise NotImplementedError

    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes
    DatasetCatalog.register(desc_train, lambda: dst_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes

    cfg = get_cfg_base_model(args.model)
    assert os.access(args.ckpt, os.R_OK)
    print('loading checkpoint:', args.ckpt)
    cfg.MODEL.WEIGHTS = args.ckpt
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_cocotrain,)
    cfg.DATASETS.TRAIN_TARGET = (desc_train,)
    cfg.DATASETS.TEST = (desc_manual_valid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = AdaptTrainer(cfg, args)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    # assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    # trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperStrongAugmentation.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj)
    trainer.resume_or_load(resume=False)
    for p in trainer.model.parameters():
        p.requires_grad = True

    prefix = 'adapt%s_%s_TIA' % (args.id, args.model)
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    try:
        trainer.train()
    except APDropException:
        print('AP drop detected, early stop')
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return

    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    # torch.save(trainer.best_params, os.path.join(os.path.dirname(__file__), '%s.iter%d.pth' % (prefix, trainer.best_iter)))
    torch.save(trainer.model.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = aps.keys()
    dst_list = {'mAP': [aps[i]['bbox']['AP'] for i in iter_list], 'AP50': [aps[i]['bbox']['AP50'] for i in iter_list]}

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Manual Valid AP50', 'Manual Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')

    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    linestyles = ['--', ':', '-.']

    plt.subplot(1, 3, 2)
    loss_keys = [
        'loss_rpn_cls', 'loss_rpn_loc', 'loss_cls', 'loss_box_reg', # FasterRCNN losses
        'loss_cls_aux_sum', 'loss_box_reg_aux_sum', # aux losses
    ]
    legends = []
    for i in range(0, len(loss_keys)):
        _k = loss_keys[i]
        plt.plot(loss_history_dict[_k][:, 0], loss_history_dict[_k][:, 1], linestyle=':', color=colors[i])
        legends.append(_k)
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('supervised losses')

    plt.subplot(1, 3, 3)
    loss_keys = [
        'loss_da_image', 'loss_da_box', # domain classifier losses
        'loss_ia_cls_src', 'loss_ia_loc_src', # inconsistency-aware losses
        'loss_ia_cls_tgt', 'loss_ia_loc_tgt' # inconsistency-aware losses
    ]
    legends = []
    for i in range(0, len(loss_keys)):
        _k = loss_keys[i]
        plt.plot(loss_history_dict[_k][:, 0], loss_history_dict[_k][:, 1], linestyle=':', color=colors[i])
        legends.append(_k)
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('adversarial training losses')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')

    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')

    parser.add_argument('--tia_n_cls', type=int, default=8, help='number of auxiliary classifiers')
    parser.add_argument('--tia_m_loc', type=int, default=4, help='number of auxiliary localizers')
    parser.add_argument('--tia_l_da', type=float, default=1.0, help='image & instance level domain classifier loss weight')
    # in the paper tia_l_da_cls=1.0 tia_l_da_loc=0.01, but that can cause training diverge
    parser.add_argument('--tia_l_da_cls', type=float, default=0.2, help='auxiliary classifier loss weight')
    parser.add_argument('--tia_l_da_loc', type=float, default=0.002, help='auxiliary localizers loss weight')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)

    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'adapt':
        if args.ddp_num_gpus <= 1:
            adapt(args)
        else:
            from detectron2.engine import launch
            launch(adapt, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='tcp://127.0.0.1:%d' % args.ddp_port, args=(args,))
    elif args.opt == 'aux':
        if args.ddp_num_gpus <= 1:
            train_aux(args)
        else:
            from detectron2.engine import launch
            launch(train_aux, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='tcp://127.0.0.1:%d' % args.ddp_port, args=(args,))
    else:
        pass
    exit(0)


'''
conda deactivate && conda activate detectron2

python finetune_tia.py --opt aux --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --iters 200 --eval_interval 1 --image_batch_size 2 --lr 0.01 --num_workers 0
python finetune_tia.py --opt aux --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1 --image_batch_size 4 --lr 0.0005 --num_workers 4

python finetune_tia.py --opt adapt --id compound --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --iters 5000 --eval_interval 1001 --image_batch_size 4 --num_workers 4

001 003 005 006 007 008 009 011 012 013 014 015 016 017 019 020 023 025 027 034 036 039 040 043 044 046 048 049 050 051 053 054 055 056 058 059 060 066 067 068 069 070 071 073 074 075 076 077 080 085 086 087 088 090 091 092 093 094 095 098 099 105 108 110 112 114 115 116 117 118 125 127 128 129 130 131 132 135 136 141 146 148 149 150 152 154 156 158 159 160 161 164 167 169 170 171 172 175 178 179

nohup python run_experiments.py --opt tia --model r101-fpn-3x --gpus 5 6 --ids 056 060 066 069 070 071 073 074 075 076 077 080 085 086 087 088 090 091 092 093 094 095 098 099 105 108 110 112 114 115 116 117 118 125 127 128 129 130 131 132 135 136 141 146 148 149 150 152 154 156 158 159 160 161 164 167 169 170 171 172 175 178 179 &> nohup_bigtoken_tia.log &

CUDA_VISIBLE_DEVICES=7 nohup python finetune_mixup.py --id compound --opt adapt --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 300000 --eval_interval 20010 --train_on_coco 1 --image_batch_size 4 &> train_compound_mixup.bigtoken.7.log &

'''
