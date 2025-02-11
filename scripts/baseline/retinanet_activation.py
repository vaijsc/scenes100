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
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes
from detectron2.structures import ImageList, Instances
from fvcore.nn import sigmoid_focal_loss_jit

import logging
import weakref
from base_detector_retinanet import get_cfg_base_model_retinanet
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_retinanet')


class RetinaNetActivation(detectron2.modeling.meta_arch.RetinaNet):
    def forward_raw_scores(self, batched_inputs: List[Dict[str, torch.Tensor]], input_scale: float = 1.0):
        images = self.preprocess_image(batched_inputs)
        if abs(input_scale - 1.0) > 0.01:
            images.tensor = torch.nn.functional.interpolate(images.tensor, scale_factor=input_scale, mode='bilinear').to(self.backbone.bottom_up.stem.conv1.weight.device)
        images.tensor.requires_grad = True
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)
        scores, _ = predictions
        return images.tensor, features, scores

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.RetinaNet), 'network is not detectron2.modeling.meta_arch.RetinaNet'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.__class__ = RetinaNetActivation
        return net


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorScaling(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.RetinaNet), 'model class mismatch'
        self.model = RetinaNetActivation.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format


def inspect_activation(args):
    from finetune import EvaluationDataset
    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    print('cfg.MODEL.ANCHOR_GENERATOR')
    print(cfg.MODEL.ANCHOR_GENERATOR)
    print('\ncfg.MODEL.RETINANET')
    print(cfg.MODEL.RETINANET)
    detector = PredictorScaling(cfg)
    print('')
    print(detector.model.head)
    # aspect ratios (height / width): [0.5, 1.0, 2.0]
    # sizes: [[32, 40.31747359663594, 50.79683366298238], [64, 80.63494719327188, 101.59366732596476], [128, 161.26989438654377, 203.18733465192952], [256, 322.53978877308754, 406.37466930385904], [512, 645.0795775461751, 812.7493386077181]]

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    for im in images:
        im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))

    for im in tqdm.tqdm(images, ascii=True):
        im_arr = skimage.io.imread(im['file_name'])[:, :, ::-1]
        height, width = im_arr.shape[:2]
        im_arr = detector.aug.get_transform(im_arr).apply_image(im_arr)
        im_arr = torch.as_tensor(im_arr.astype('float32').transpose(2, 0, 1))
        results = []
        for input_scale in [1, 2]:
            im_pp, feature_levels, score_levels = detector.model.forward_raw_scores([{'image': im_arr, 'height': height, 'width': width}], input_scale)

            # score_levels = [s.sigmoid_().view(s.size(0), len(cfg.MODEL.ANCHOR_GENERATOR.SIZES[0]), len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]), cfg.MODEL.RETINANET.NUM_CLASSES, s.size(2), s.size(3)) for s in score_levels] # N x (Anchors x K) x H x W -> N x Sizes x Aspects x K x H x W
            # score_p4_person = score_levels[1][:, 0, 2, 0, :, :] # for person, use smallest size, aspect ratio 2.0
            # score_p4_vehicle = score_levels[1][:, 2, 0, 1, :, :] # for vehicle, use largest size, aspect ratio 0.5

            # score_levels = [s.sigmoid_().view(s.size(0), -1, cfg.MODEL.RETINANET.NUM_CLASSES, s.size(2), s.size(3)) for s in score_levels] # N x (Anchors x K) x H x W -> N x Anchors x K x H x W
            # score_p4_person = score_levels[1][:, :, 0, :, :].max(dim=1).values
            # score_p4_vehicle = score_levels[1][:, :, 1, :, :].max(dim=1).values

            feature_levels, score_levels = feature_levels[:3], score_levels[:3] # keep p3 p4 p5
            feature_levels_resize = [torch.nn.functional.interpolate(f, scale_factor=(feature_levels[0].size(2) / f.size(2)), mode='bilinear') for f in feature_levels]
            feature_levels_resize = torch.stack(feature_levels_resize, dim=1) # N x 3 x 256 x H x W
            score_levels_resize = [torch.nn.functional.interpolate(s.sigmoid_(), scale_factor=(score_levels[0].size(2) / s.size(2)), mode='bilinear') for s in score_levels]
            score_levels_resize = [s.view(s.size(0), -1, cfg.MODEL.RETINANET.NUM_CLASSES, s.size(2), s.size(3)) for s in score_levels_resize] # N x (Anchors x K) x H x W -> N x Anchors x K x H x W
            score_levels_resize = torch.stack(score_levels_resize, dim=1) # N x 3 x Anchors x K x H x W
            score_person  = score_levels_resize[:, :, :, 0, :, :].max(dim=2).values
            score_vehicle = score_levels_resize[:, :, :, 1, :, :].max(dim=2).values

            results.append({'image': im_pp, 'feature_levels': feature_levels_resize, 'score_person': score_person, 'score_vehicle': score_vehicle})

        plt.figure(figsize=(18, 6))
        for s in range(0, len(results)):
            plt.subplot(2, 4, 1 + s * 4)
            _im = results[s]['image'][0].detach().cpu().numpy(); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)
            plt.imshow(_im[:, :, ::-1]); plt.title('image')
            plt.subplot(2, 4, 2 + s * 4)
            _im = results[s]['feature_levels'][0].detach().cpu().numpy().max(axis=1); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)
            plt.imshow(_im); plt.title('feature map max value (p3=R p4=G p5=B)')
            plt.subplot(2, 4, 3 + s * 4)
            _im = results[s]['score_person'][0].detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(_im); plt.title('<person> max score (p3=R p4=G p5=B)')
            plt.subplot(2, 4, 4 + s * 4)
            _im = results[s]['score_vehicle'][0].detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(_im); plt.title('<vehicle> max score (p3=R p4=G p5=B)')
        plt.tight_layout()
        plt.show()


def inspect_gradient(args):
    from finetune import EvaluationDataset
    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    print('cfg.MODEL.ANCHOR_GENERATOR')
    print(cfg.MODEL.ANCHOR_GENERATOR)
    print('\ncfg.MODEL.RETINANET')
    print(cfg.MODEL.RETINANET)
    detector = PredictorScaling(cfg)
    print('')
    print(detector.model.head)

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    for im in images:
        im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))

    for im in tqdm.tqdm(images, ascii=True):
        im_arr = skimage.io.imread(im['file_name'])[:, :, ::-1]
        height, width = im_arr.shape[:2]
        im_arr = detector.aug.get_transform(im_arr).apply_image(im_arr)
        im_arr = torch.as_tensor(im_arr.astype('float32').transpose(2, 0, 1))
        results = []
        for input_scale in [1, 2]:
            im_pp, feature_levels, score_levels = detector.model.forward_raw_scores([{'image': im_arr, 'height': height, 'width': width}], input_scale)

            feature_levels = feature_levels[:3]
            feature_levels_resize = [torch.nn.functional.interpolate(f, scale_factor=(feature_levels[0].size(2) / f.size(2)), mode='bilinear') for f in feature_levels]
            feature_levels_resize = torch.stack(feature_levels_resize, dim=1) # N x 3 x 256 x H x W
            score_levels = [s.sigmoid_().view(s.size(0), -1, cfg.MODEL.RETINANET.NUM_CLASSES, s.size(2), s.size(3)) for s in score_levels] # N x (Anchors x K) x H x W -> N x Anchors x K x H x W

            im_pp.grad = None
            sum([s[0, :, 0, :, :].sum() / (s.size(3) * s.size(4)) for s in score_levels]).backward(retain_graph=True)
            gradient_person = im_pp.grad[0].detach().square().sum(dim=0).sqrt().cpu().numpy()
            im_pp.grad = None
            sum([s[0, :, 1, :, :].sum() / (s.size(3) * s.size(4)) for s in score_levels]).backward(retain_graph=True)
            gradient_vehicle = im_pp.grad[0].detach().square().sum(dim=0).sqrt().cpu().numpy()

            results.append({'image': im_pp, 'feature_levels': feature_levels_resize, 'gradient_person': gradient_person, 'gradient_vehicle': gradient_vehicle, 'gradient_max': max(gradient_person.max(), gradient_vehicle.max())})

        plt.figure(figsize=(18, 6))
        for s in range(0, len(results)):
            plt.subplot(2, 4, 1 + s * 4)
            _im = results[s]['image'][0].detach().cpu().numpy(); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)
            plt.imshow(_im[:, :, ::-1]); plt.title('image')
            plt.subplot(2, 4, 2 + s * 4)
            _im = results[s]['feature_levels'][0][:3].detach().cpu().numpy().max(axis=1); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)
            plt.imshow(_im); plt.title('feature map max value (p3=R p4=G p5=B)')
            plt.subplot(2, 4, 3 + s * 4)
            _im = results[s]['gradient_person'] / results[s]['gradient_max']
            plt.imshow(_im ** 0.4, cmap='gray'); plt.title('<person> gradient')
            plt.subplot(2, 4, 4 + s * 4)
            _im = results[s]['gradient_vehicle'] / results[s]['gradient_max']
            plt.imshow(_im ** 0.4, cmap='gray'); plt.title('<vehicle> gradient')
        plt.tight_layout()
        plt.show()


def inspect_receptive_field(args):
    from finetune import EvaluationDataset
    cfg = get_cfg_base_model_retinanet(ckpt=args.ckpt)
    print('cfg.MODEL.ANCHOR_GENERATOR')
    print(cfg.MODEL.ANCHOR_GENERATOR)
    print('\ncfg.MODEL.RETINANET')
    print(cfg.MODEL.RETINANET)
    detector = PredictorScaling(cfg)

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    for im in images:
        im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))


    for im in tqdm.tqdm(images, ascii=True):
        im_arr = skimage.io.imread(im['file_name'])[:, :, ::-1]
        height, width = im_arr.shape[:2]
        im_arr = detector.aug.get_transform(im_arr).apply_image(im_arr)
        im_arr = torch.as_tensor(im_arr.astype('float32').transpose(2, 0, 1))
        results = []
        for input_scale in [1, 2]:
            im_pp, feature_levels, _ = detector.model.forward_raw_scores([{'image': im_arr, 'height': height, 'width': width}], input_scale)
            h_l0, w_l0 = feature_levels[0].size(2) // 2, feature_levels[0].size(3) // 2
            h_l4, w_l4 = feature_levels[4].size(2) // 2, feature_levels[4].size(3) // 2
            im_pp.grad = None
            feature_levels[0][0, :, h_l0, w_l0].square().sum().backward(retain_graph=True)
            gradient_l0 = im_pp.grad[0].detach().abs().sum(dim=0).detach().cpu()
            im_pp.grad = None
            feature_levels[4][0, :, h_l4, w_l4].square().sum().backward(retain_graph=True)
            gradient_l4 = im_pp.grad[0].detach().abs().sum(dim=0).detach().cpu()
            results.append({'image': im_pp, 'feature_levels': feature_levels, 'gradient_l0': gradient_l0, 'c0': (h_l0, w_l0), 'gradient_l4': gradient_l4, 'c4': (h_l4, w_l4)})

        plt.figure(figsize=(18, 6))
        for s in range(0, len(results)):
            plt.subplot(2, 4, 1 + s * 4)
            _im = results[s]['feature_levels'][0][0, :3, :, :].detach().cpu().numpy(); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)
            plt.imshow(_im); plt.scatter([results[s]['c0'][1]], [results[s]['c0'][0]], marker='+', c='r', s=100); plt.title('p3 feature map')
            plt.subplot(2, 4, 2 + s * 4)
            _im = results[s]['image'][0].detach().cpu().numpy(); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)[:, :, ::-1]
            _field = results[s]['gradient_l0']; _field = torch.where(_field > 0, 1.0, 0.0).unsqueeze(2).numpy() * 0.5; _mask = np.array([1.0, 0, 0]).reshape(1, 1, 3)
            _im = _im * (1 - _field) + _field * _mask
            plt.imshow(_im); plt.title('p3 receptive field')
            plt.subplot(2, 4, 3 + s * 4)
            _im = results[s]['feature_levels'][4][0, :3, :, :].detach().cpu().numpy(); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)
            plt.imshow(_im); plt.scatter([results[s]['c4'][1]], [results[s]['c4'][0]], marker='+', c='r', s=100); plt.title('p7 feature map')
            plt.subplot(2, 4, 4 + s * 4)
            _im = results[s]['image'][0].detach().cpu().numpy(); _im -= _im.min(); _im /= _im.max(); _im = _im.transpose(1, 2, 0)[:, :, ::-1]
            _field = results[s]['gradient_l4']; _field = torch.where(_field > 0, 1.0, 0.0).unsqueeze(2).numpy() * 0.5; _mask = np.array([1.0, 0, 0]).reshape(1, 1, 3)
            _im = _im * (1 - _field) + _field * _mask
            plt.imshow(_im); plt.title('p7 receptive field')
        plt.tight_layout()
        plt.show()


def show_qualitative(args):
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i1.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_frcnn_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i2.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_frcnn_x2 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i1.00_a1.00.json', 'r') as fp:
        results_retina_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i2.00_a1.00.json', 'r') as fp:
        results_retina_x2 = json.load(fp)

    APs_frcnn_x1 = np.array([results_frcnn_x1[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_frcnn_x2 = np.array([results_frcnn_x2[v]['results']['weighted'] for v in video_id_list]) * 100
    print('Faster-RCNN x1:', APs_frcnn_x1.mean(), 'Faster-RCNN x2:', APs_frcnn_x2.mean())
    APs_retina_x1 = np.array([results_retina_x1[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_retina_x2 = np.array([results_retina_x2[v]['results']['weighted'] for v in video_id_list]) * 100
    print('RetinaNet x1:', APs_retina_x1.mean(), 'RetinaNet x2:',  APs_retina_x2.mean())
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(APs_frcnn_x1[:, 0], APs_retina_x1[:, 0], marker='x', c='r')
    plt.scatter(APs_frcnn_x1[:, 1], APs_retina_x1[:, 1], marker='+', c='b')
    plt.xlabel('FasterRCNN input x1'); plt.ylabel('RetinaNet input x1'); plt.legend(['$mAP$', '$AP50$'])
    plt.subplot(1, 3, 2)
    plt.scatter(APs_frcnn_x2[:, 0], APs_retina_x2[:, 0], marker='x', c='r')
    plt.scatter(APs_frcnn_x2[:, 1], APs_retina_x2[:, 1], marker='+', c='b')
    plt.xlabel('FasterRCNN input x2'); plt.ylabel('RetinaNet input x2'); plt.legend(['$mAP$', '$AP50$'])
    plt.subplot(1, 3, 3)
    plt.scatter(APs_frcnn_x2[:, 0] - APs_frcnn_x1[:, 0], APs_retina_x2[:, 0] - APs_retina_x1[:, 0], marker='x', c='r')
    plt.scatter(APs_frcnn_x2[:, 1] - APs_frcnn_x1[:, 1], APs_retina_x2[:, 1] - APs_retina_x1[:, 1], marker='+', c='b')
    plt.xlabel('FasterRCNN improvement'); plt.ylabel('RetinaNet improvement'); plt.legend(['$mAP \\uparrow$', '$AP50 \\uparrow$'])
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, 'AP_scaling_correlation.pdf'))

    # 'iouThrs', 'recThrs', 'catIds', 'areaRng', 'maxDets', 'precision', 'recall', 'scores'
    # 'precision': (T,R,K,A,M), 'recall': (T,K,A,M), 'scores': (T,R,K,A,M)
    # T: IoU thres, R: recall thres, K: classes, A: areas, M: max dets
    font_label = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansMono.ttf'), size=20)
    font_title = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansMono.ttf'), size=50)
    for v in ['007', '015', '049', '060', '090', '095', '135', '013', '040', '050', '088', '130', '158', '172']:
        print(v)
        iou, T, A, M = 0.75, 5, 0, 2
        assert abs(results_frcnn_x1[v]['raw']['iouThrs'][T] - iou) < 1e-3 and abs(results_frcnn_x2[v]['raw']['iouThrs'][T] - iou) < 1e-3
        assert abs(results_retina_x1[v]['raw']['iouThrs'][T] - iou) < 1e-3 and abs(results_retina_x2[v]['raw']['iouThrs'][T] - iou) < 1e-3

        pr_f_x1 = np.array(results_frcnn_x1[v]['raw']['precision'])[T, :, :, A, M].T
        s_f_x1 =  np.array(results_frcnn_x1[v]['raw']['scores'])[T, :, :, A, M].T
        rc_f_x1 = np.array(results_frcnn_x1[v]['raw']['recThrs'])
        pr_f_x2 = np.array(results_frcnn_x2[v]['raw']['precision'])[T, :, :, A, M].T
        s_f_x2 =  np.array(results_frcnn_x2[v]['raw']['scores'])[T, :, :, A, M].T
        rc_f_x2 = np.array(results_frcnn_x2[v]['raw']['recThrs'])

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(rc_f_x2, s_f_x1[0], c='gray', ls='-.')
        plt.plot(rc_f_x2, s_f_x2[0], c='gray', ls=':')
        plt.plot(rc_f_x1, pr_f_x1[0], c='r', ls='-')
        plt.plot(rc_f_x2, pr_f_x2[0], c='r', ls='--')
        plt.legend(['x1 score', 'x2 score', 'x1 $AP=%.2f$' % (pr_f_x1[0].mean() * 100), 'x2 $AP=%.2f$' % (pr_f_x2[0].mean() * 100)])
        plt.xlabel('recall'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nFasterRCNN $IoU=%.2f$' % (v, thing_classes[0], results_frcnn_x1[v]['weights']['total'] * results_frcnn_x1[v]['weights']['classes'][0], results_frcnn_x1[v]['raw']['iouThrs'][T]))
        plt.subplot(2, 2, 2)
        plt.plot(rc_f_x1, s_f_x1[1], c='gray', ls='-.')
        plt.plot(rc_f_x2, s_f_x2[1], c='gray', ls=':')
        plt.plot(rc_f_x1, pr_f_x1[1], c='b', ls='-')
        plt.plot(rc_f_x2, pr_f_x2[1], c='b', ls='--')
        plt.legend(['x1 score', 'x2 score', 'x1 $AP=%.2f$' % (pr_f_x1[1].mean() * 100), 'x2 $AP=%.2f$' % (pr_f_x2[1].mean() * 100)])
        plt.xlabel('recall'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nFasterRCNN $IoU=%.2f$' % (v, thing_classes[1], results_frcnn_x1[v]['weights']['total'] * results_frcnn_x1[v]['weights']['classes'][1], results_frcnn_x1[v]['raw']['iouThrs'][T]))

        pr_r_x1 = np.array(results_retina_x1[v]['raw']['precision'])[T, :, :, A, M].T
        s_r_x1 =  np.array(results_retina_x1[v]['raw']['scores'])[T, :, :, A, M].T
        rc_r_x1 = np.array(results_retina_x1[v]['raw']['recThrs'])
        pr_r_x2 = np.array(results_retina_x2[v]['raw']['precision'])[T, :, :, A, M].T
        s_r_x2 =  np.array(results_retina_x2[v]['raw']['scores'])[T, :, :, A, M].T
        rc_r_x2 = np.array(results_retina_x2[v]['raw']['recThrs'])

        plt.subplot(2, 2, 3)
        plt.plot(rc_r_x1, s_r_x1[0], c='gray', ls='-.')
        plt.plot(rc_r_x2, s_r_x2[0], c='gray', ls=':')
        plt.plot(rc_r_x1, pr_r_x1[0], c='r', ls='-')
        plt.plot(rc_r_x2, pr_r_x2[0], c='r', ls='--')
        plt.legend(['x1 score', 'x2 score', 'x1 $AP=%.2f$' % (pr_r_x1[0].mean() * 100), 'x2 $AP=%.2f$' % (pr_r_x2[0].mean() * 100)])
        plt.xlabel('recall'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nRetinaNet $IoU=%.2f$' % (v, thing_classes[0], results_retina_x1[v]['weights']['total'] * results_retina_x1[v]['weights']['classes'][0], results_retina_x1[v]['raw']['iouThrs'][T]))
        plt.subplot(2, 2, 4)
        plt.plot(rc_r_x1, s_r_x1[1], c='gray', ls='-.')
        plt.plot(rc_r_x2, s_r_x2[1], c='gray', ls=':')
        plt.plot(rc_r_x1, pr_r_x1[1], c='b', ls='-')
        plt.plot(rc_r_x2, pr_r_x2[1], c='b', ls='--')
        plt.legend(['x1 score', 'x2 score', 'x1 $AP=%.2f$' % (pr_r_x1[1].mean() * 100), 'x2 $AP=%.2f$' % (pr_r_x2[1].mean() * 100)])
        plt.xlabel('recall'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nRetinaNet $IoU=%.2f$' % (v, thing_classes[1], results_retina_x1[v]['weights']['total'] * results_retina_x1[v]['weights']['classes'][1], results_retina_x1[v]['raw']['iouThrs'][T]))
        plt.tight_layout()
        plt.savefig(os.path.join(args.outputdir, 'AP_%s_scaling_curve.pdf' % v))

        with open(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', v, 'annotations.json'), 'r') as fp:
            images_gt = json.load(fp)

        mAP_f_x1 = results_frcnn_x1[v]['results']['weighted'][0] * 100
        mAP_f_x2 = results_frcnn_x2[v]['results']['weighted'][0] * 100
        mAP_r_x1 = results_retina_x1[v]['results']['weighted'][0] * 100
        mAP_r_x2 = results_retina_x2[v]['results']['weighted'][0] * 100
        t_f_x1 = np.absolute(pr_f_x1 - 0.6).argmin(axis=1)
        t_f_x2 = np.absolute(pr_f_x2 - 0.6).argmin(axis=1)
        t_r_x1 = np.absolute(pr_r_x1 - 0.6).argmin(axis=1)
        t_r_x2 = np.absolute(pr_r_x2 - 0.6).argmin(axis=1)

        writer = skvideo.io.FFmpegWriter(os.path.join(args.outputdir, 'detections_%s.mp4' % v), inputdict={'-r': '1'}, outputdict={'-vcodec': 'libx265', '-r': '1', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '24'})
        for im_gt, det_f_x1, det_f_x2, det_r_x1, det_r_x2 in zip(images_gt, results_frcnn_x1[v]['detections'], results_frcnn_x2[v]['detections'], results_retina_x1[v]['detections'], results_retina_x2[v]['detections']):
            # im = skimage.io.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', v, 'unmasked', det_f_x1['file_name']))
            im = skimage.io.imread(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotations', 'images_mask_overlay', v + '_' + det_f_x1['file_name']))
            im_labeled = []
            for det, desc, score_thres in [
                (im_gt, [('Ground Truth', '#000000')], None),
                (   det_f_x1,
                    [   ('FasterRCNN x1 mAP=%.2f' % mAP_f_x1, '#CC6600'),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[0], pr_f_x1[0][t_f_x1[0]], rc_f_x1[t_f_x1[0]], s_f_x1[0][t_f_x1[0]]), bbox_rgbs[0]),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[1], pr_f_x1[1][t_f_x1[1]], rc_f_x1[t_f_x1[1]], s_f_x1[1][t_f_x1[1]]), bbox_rgbs[1])
                    ],
                    (s_f_x1[0][t_f_x1[0]], s_f_x1[1][t_f_x1[1]])
                ),
                (   det_f_x2,
                    [   ('FasterRCNN x2 mAP=%.2f' % mAP_f_x2, '#CC6600'),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[0], pr_f_x2[0][t_f_x2[0]], rc_f_x2[t_f_x2[0]], s_f_x2[0][t_f_x2[0]]), bbox_rgbs[0]),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[1], pr_f_x2[1][t_f_x2[1]], rc_f_x2[t_f_x2[1]], s_f_x2[1][t_f_x2[1]]), bbox_rgbs[1])
                    ],
                    (s_f_x2[0][t_f_x2[0]], s_f_x2[1][t_f_x2[1]])
                ),
                (None, None, None),
                (   det_r_x1,
                    [   ('RetinaNet x1 mAP=%.2f' % mAP_r_x1, '#009911'),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[0], pr_r_x1[0][t_r_x1[0]], rc_r_x1[t_r_x1[0]], s_r_x1[0][t_r_x1[0]]), bbox_rgbs[0]),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[1], pr_r_x1[1][t_r_x1[1]], rc_r_x1[t_r_x1[1]], s_r_x1[1][t_r_x1[1]]), bbox_rgbs[1])
                    ],
                    (s_r_x1[0][t_r_x1[0]], s_r_x1[1][t_r_x1[1]])
                ),
                (   det_r_x2,
                    [   ('RetinaNet x2 mAP=%.2f' % mAP_r_x2, '#009911'),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[0], pr_r_x2[0][t_r_x2[0]], rc_r_x2[t_r_x2[0]], s_r_x2[0][t_r_x2[0]]), bbox_rgbs[0]),
                        ('IoU=%.2f %08s pr=%.2f rc=%.2f s=%.2f' % (iou, thing_classes[1], pr_r_x2[1][t_r_x2[1]], rc_r_x2[t_r_x2[1]], s_r_x2[1][t_r_x2[1]]), bbox_rgbs[1])
                    ],
                    (s_r_x2[0][t_r_x2[0]], s_r_x2[1][t_r_x2[1]])
                )]:
                if det is None:
                    im_labeled.append(np.zeros_like(im_labeled[-1]))
                    continue
                f = Image.fromarray(copy.deepcopy(im))
                draw = ImageDraw.Draw(f)
                desc.insert(0, ('%s %s' % (v, det['file_name']), '#000000'))
                for shift, (text, c) in enumerate(desc):
                    draw.text((2, 2 + shift * 52), text, fill='#FFFFFF', stroke_width=5, font=font_title)
                    draw.text((2, 2 + shift * 52), text, fill=c, stroke_width=1, font=font_title)
                for ann in det['annotations']:
                    if 'score' in ann and ann['score'] < score_thres[ann['category_id']]:
                        continue
                    c = bbox_rgbs[ann['category_id']]
                    x1, y1, x2, y2 = ann['bbox']
                    draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=c, width=3)
                    if 'score' in ann:
                        draw.text((x1 + 2, y1 - 22), '%.2f' % ann['score'], fill=c, font=font_label)
                im_labeled.append(np.array(f))
            im_labeled = np.concatenate([np.concatenate(im_labeled[:3], axis=0), np.concatenate(im_labeled[3:], axis=0)], axis=1)
            writer.writeFrame(im_labeled)
        writer.close()


def show_scores(args):
    def _to_percentile(scores):
        percentiles = [(scores <= s).sum() / scores.shape[0] for s in scores]
        return np.array(percentiles)

    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i1.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_frcnn_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i2.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_frcnn_x2 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i1.00_a1.00.json', 'r') as fp:
        results_retina_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling_retinanet\\results_AP_retinanet_i2.00_a1.00.json', 'r') as fp:
        results_retina_x2 = json.load(fp)

    # 'iouThrs', 'recThrs', 'catIds', 'areaRng', 'maxDets', 'precision', 'recall', 'scores'
    # 'precision': (T,R,K,A,M), 'recall': (T,K,A,M), 'scores': (T,R,K,A,M)
    # T: IoU thres, R: recall thres, K: classes, A: areas, M: max dets
    for v in ['007', '015', '049', '060', '090', '095', '135', '013', '040', '050', '088', '130', '158', '172']:
        print(v)
        iou, T, A, M = 0.75, 5, 0, 2
        assert abs(results_frcnn_x1[v]['raw']['iouThrs'][T] - iou) < 1e-3 and abs(results_frcnn_x2[v]['raw']['iouThrs'][T] - iou) < 1e-3
        assert abs(results_retina_x1[v]['raw']['iouThrs'][T] - iou) < 1e-3 and abs(results_retina_x2[v]['raw']['iouThrs'][T] - iou) < 1e-3

        pr_f_x1 = np.array(results_frcnn_x1[v]['raw']['precision'])[T, :, :, A, M].T
        s_f_x1 =  np.array(results_frcnn_x1[v]['raw']['scores'])[T, :, :, A, M].T
        rc_f_x1 = np.array(results_frcnn_x1[v]['raw']['recThrs'])
        pr_f_x2 = np.array(results_frcnn_x2[v]['raw']['precision'])[T, :, :, A, M].T
        s_f_x2 =  np.array(results_frcnn_x2[v]['raw']['scores'])[T, :, :, A, M].T
        rc_f_x2 = np.array(results_frcnn_x2[v]['raw']['recThrs'])

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(s_f_x1[0], pr_f_x1[0], c='k', ls='-.')
        plt.plot(s_f_x2[0], pr_f_x2[0], c='k', ls=':')
        plt.plot(_to_percentile(s_f_x1[0]), pr_f_x1[0], c='#550099', ls='-.')
        plt.plot(_to_percentile(s_f_x2[0]), pr_f_x2[0], c='#550099', ls=':')
        plt.legend(['x1 $AP=%.2f$' % (pr_f_x1[0].mean() * 100), 'x2 $AP=%.2f$' % (pr_f_x2[0].mean() * 100), 'x1 percentile', 'x2 percentile'])
        plt.xlabel('score'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nFasterRCNN $IoU=%.2f$' % (v, thing_classes[0], results_frcnn_x1[v]['weights']['total'] * results_frcnn_x1[v]['weights']['classes'][0], results_frcnn_x1[v]['raw']['iouThrs'][T]))
        plt.subplot(2, 2, 2)
        plt.plot(s_f_x1[1], pr_f_x1[1], c='k', ls='-.')
        plt.plot(s_f_x2[1], pr_f_x2[1], c='k', ls=':')
        plt.plot(_to_percentile(s_f_x1[1]), pr_f_x1[1], c='#550099', ls='-.')
        plt.plot(_to_percentile(s_f_x2[1]), pr_f_x2[1], c='#550099', ls=':')
        plt.legend(['x1 $AP=%.2f$' % (pr_f_x1[1].mean() * 100), 'x2 $AP=%.2f$' % (pr_f_x2[1].mean() * 100), 'x1 percentile', 'x2 percentile'])
        plt.xlabel('score'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nFasterRCNN $IoU=%.2f$' % (v, thing_classes[1], results_frcnn_x1[v]['weights']['total'] * results_frcnn_x1[v]['weights']['classes'][1], results_frcnn_x1[v]['raw']['iouThrs'][T]))

        pr_r_x1 = np.array(results_retina_x1[v]['raw']['precision'])[T, :, :, A, M].T
        s_r_x1 =  np.array(results_retina_x1[v]['raw']['scores'])[T, :, :, A, M].T
        rc_r_x1 = np.array(results_retina_x1[v]['raw']['recThrs'])
        pr_r_x2 = np.array(results_retina_x2[v]['raw']['precision'])[T, :, :, A, M].T
        s_r_x2 =  np.array(results_retina_x2[v]['raw']['scores'])[T, :, :, A, M].T
        rc_r_x2 = np.array(results_retina_x2[v]['raw']['recThrs'])

        plt.subplot(2, 2, 3)
        plt.plot(s_r_x1[0], pr_r_x1[0], c='k', ls='-.')
        plt.plot(s_r_x2[0], pr_r_x2[0], c='k', ls=':')
        plt.plot(_to_percentile(s_r_x1[0]), pr_r_x1[0], c='#550099', ls='-.')
        plt.plot(_to_percentile(s_r_x2[0]), pr_r_x2[0], c='#550099', ls=':')
        plt.legend(['x1 $AP=%.2f$' % (pr_r_x1[0].mean() * 100), 'x2 $AP=%.2f$' % (pr_r_x2[0].mean() * 100), 'x1 percentile', 'x2 percentile'])
        plt.xlabel('score'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nRetinaNet $IoU=%.2f$' % (v, thing_classes[0], results_retina_x1[v]['weights']['total'] * results_retina_x1[v]['weights']['classes'][0], results_retina_x1[v]['raw']['iouThrs'][T]))
        plt.subplot(2, 2, 4)
        plt.plot(s_r_x1[1], pr_r_x1[1], c='k', ls='-.')
        plt.plot(s_r_x2[1], pr_r_x2[1], c='k', ls=':')
        plt.plot(_to_percentile(s_r_x1[1]), pr_r_x1[1], c='#550099', ls='-.')
        plt.plot(_to_percentile(s_r_x2[1]), pr_r_x2[1], c='#550099', ls=':')
        plt.legend(['x1 $AP=%.2f$' % (pr_r_x1[1].mean() * 100), 'x2 $AP=%.2f$' % (pr_r_x2[1].mean() * 100), 'x1 percentile', 'x2 percentile'])
        plt.xlabel('score'); plt.ylabel('precision'); plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01); plt.grid(True)
        plt.title('video %s <%s> #%d\nRetinaNet $IoU=%.2f$' % (v, thing_classes[1], results_retina_x1[v]['weights']['total'] * results_retina_x1[v]['weights']['classes'][1], results_retina_x1[v]['raw']['iouThrs'][T]))
        plt.tight_layout()
        plt.savefig(os.path.join(args.outputdir, 'scores_%s_scaling_curve.pdf' % v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    args = parser.parse_args()
    print(args)
    if args.opt == 'activation':
        inspect_activation(args)
    if args.opt == 'gradient':
        inspect_gradient(args)
    if args.opt == 'receptive':
        inspect_receptive_field(args)
    if args.opt == 'qualitative':
        show_qualitative(args)
    if args.opt == 'score':
        show_scores(args)

'''
python retinanet_activation.py --ckpt mscoco2017_remap_retinanet_r101.pth --id 001 --opt activation
python retinanet_activation.py --ckpt mscoco2017_remap_retinanet_r101.pth --id 001 --opt gradient
python retinanet_activation.py --ckpt mscoco2017_remap_retinanet_r101.pth --id 001 --opt receptive
python retinanet_activation.py --opt qualitative --outputdir F:\\intersections_results\\cvpr24\\feature_scaling
python retinanet_activation.py --opt score --outputdir F:\\intersections_results\\cvpr24\\feature_scaling
'''
