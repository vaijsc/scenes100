#!python3

'''
our implementation of
https://github.com/Flashkong/Source-Free-Object-Detection-by-Learning-to-Overlook-Domain-Style

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
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames

from finetune import get_annotation_dict, all_annotation_dict, finetune_simple_trainer_run_step
from finetune_tia import get_unlabeled_dicts, all_unlabeled_dicts
from lods_enhance_vgg16 import enhance_vgg16


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_lods_output')


class gromovWasserstein(torch.nn.Module):
    def __init__(self, beta=0.5, affinity_type='cosine', l_type='KL'):
        super(gromovWasserstein, self).__init__()
        self.affinity_type=affinity_type
        self.l_type=l_type
        self.beta = beta
        self.rate = 0.99
        self.iter_num = 50
        print("gw add rate is :"+str(self.beta))

    def forward(self, feat_stu, feat_tea, t):
        affinity_stu = self.affinity_matrix(feat_stu)
        affinity_tea = self.affinity_matrix(feat_tea)
        T = torch.eye(feat_stu.size(0)).cuda()

        if type(t)!=int:
            T = self.beta*t + T
        T = T/T.sum()
        cost = self.L(affinity_stu,affinity_tea,T)
        loss = (cost * T).sum()
        return loss

    def affinity_matrix_cross(self, feat1, feat2):
        if self.affinity_type=='cosine':
            energy1 = torch.sqrt(torch.sum(feat1 ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            energy2 = torch.sqrt(torch.sum(feat2 ** 2, dim=1, keepdim=True))
            cos_sim = torch.matmul(feat1, torch.t(feat2)) / (torch.matmul(energy1, torch.t(energy2)))
            affinity = cos_sim
        else:
            pass
        return affinity

    def affinity_matrix(self, feat):
        if self.affinity_type=='cosine':
            energy = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cos_sim = torch.matmul(feat, torch.t(feat)) / (torch.matmul(energy, torch.t(energy)) )
            affinity = cos_sim
        else:
            feat = torch.matmul(feat, torch.t(feat))  # (batch_size, batch_size)
            feat_diag = torch.diag(feat).view(-1, 1).repeat(1, feat.size(0))  # (batch_size, batch_size)
            affinity = 1-torch.exp(-(feat_diag + torch.t(feat_diag) - 2 * feat)/feat.size(1))
        return affinity

    def L(self, affinity_stu, affinity_tea, T):
        stu_1 = torch.ones(affinity_stu.size(0),1).cuda()
        tea_1 = torch.ones(affinity_tea.size(0),1).cuda()
        p=T.mm(tea_1)
        q=T.t().mm(stu_1)
        if self.l_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(affinity_stu)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(affinity_tea)^T
            # cost = cost_st - h1(affinity_stu)*T*h2(affinity_tea)^T
            f1_st = (affinity_stu ** 2).mm(p).mm(tea_1.t())
            f2_st = stu_1.mm(q.t()).mm((affinity_tea ** 2).t())
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * affinity_stu.mm(T).mm(affinity_tea.t())
        elif self.l_type=='KL':
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(affinity_stu)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(affinity_tea)^T
            # cost = cost_st - h1(affinity_stu)*T*h2(affinity_tea)^T
            f1_st = torch.matmul(affinity_stu * torch.log(affinity_stu+ 1e-7) - affinity_stu, p).mm(tea_1.t())
            f2_st = stu_1.mm(torch.matmul(torch.t(q), torch.t(affinity_tea)))
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(affinity_stu, T), torch.t(torch.log(affinity_tea+1e-7)))
        return cost


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNLODS(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    @staticmethod
    def get_pooled_features(fpn_feature, proposals, roi_heads):
        box_features = roi_heads.box_pooler([fpn_feature[f] for f in roi_heads.box_in_features], [x.proposal_boxes for x in proposals])
        box_features = roi_heads.box_head(box_features)
        class_probs = torch.softmax(roi_heads.box_predictor(box_features)[0], dim=1)
        box_features = torch.bmm(class_probs.unsqueeze(2), box_features.unsqueeze(1)).view(class_probs.size(0), -1)
        return box_features, class_probs

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training: return self.inference(batched_inputs)
        # print('student fpn %.5f roi %.5f, teacher fpn %.5f roi %.5f, rpn %.5f' % (
        #     self.backbone.fpn_output5.weight.data.min().detach().item(),
        #     self.roi_heads.box_head.fc2.weight.data.min().detach().min().item(),
        #     self.backbone_teacher.fpn_output5.weight.data.min().detach().item(),
        #     self.roi_heads_teacher.box_head.fc2.weight.data.min().detach().min().item(),
        #     self.proposal_generator.rpn_head.anchor_deltas.weight.data.min().detach().min().item()
        # ))
        images = self.preprocess_image(batched_inputs)

        # generate pseudo boxes from teacher model using original image
        self.backbone_teacher.eval()
        self.roi_heads_teacher.eval()
        self.proposal_generator.eval()
        features_teacher = self.backbone_teacher(images.tensor)
        proposals, _ = self.proposal_generator(images, features_teacher, None)
        results_teacher, _ = self.roi_heads_teacher(images, features_teacher, proposals, None)
        for i in range(0, len(results_teacher)):
            mask_thres = results_teacher[i].scores > self.score_thres
            results_teacher[i] = results_teacher[i][mask_thres]
            results_teacher[i].gt_classes = results_teacher[i].pred_classes
            results_teacher[i].gt_boxes = results_teacher[i].pred_boxes
            results_teacher[i].remove('pred_boxes')
            results_teacher[i].remove('pred_classes')
            batched_inputs[i]['instances'] = results_teacher[i]

        # enhancement
        images_enhanced = self.enhancer.add_style(images.tensor, 0).detach()
        # im_show_orig = images.tensor.cpu().numpy(); im_show_orig = im_show_orig / 255 + 0.5; im_show_orig = im_show_orig.transpose(0, 2, 3, 1)
        # im_show_enhance = images_enhanced.cpu().numpy(); im_show_enhance = im_show_enhance / 255 + 0.5; im_show_enhance = im_show_enhance.transpose(0, 2, 3, 1)
        # plt.figure()
        # for i in range(0, im_show_orig.shape[0]):
        #     plt.subplot(im_show_orig.shape[0], 2, i * im_show_orig.shape[0] + 1); plt.imshow(im_show_orig[i][:, :, ::-1])
        #     plt.subplot(im_show_orig.shape[0], 2, i * im_show_orig.shape[0] + 2); plt.imshow(im_show_enhance[i][:, :, ::-1])
        # plt.show()
        images_enhanced = ImageList.from_tensors([images_enhanced[i] for i in range(0, images_enhanced.size(0))], self.backbone.size_divisibility)
        assert images.tensor.size() == images_enhanced.tensor.size()

        # student ROI loss using pseudo boxes from enhanced image
        gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        features = self.backbone(images_enhanced.tensor)
        _, detector_losses = self.roi_heads(images_enhanced, features, proposals, gt_instances)

        # global gw alignment loss
        features_teacher_flatten = torch.cat([features_teacher[k].view(len(batched_inputs), -1) for k in features_teacher], dim=1)
        features_student_flatten = torch.cat([features[k].view(len(batched_inputs), -1) for k in features], dim=1)
        gw_losses = {'loss_gw_global': self.gw(features_student_flatten, features_teacher_flatten, 0) * self.loss_alpha_global}

        # instance gw alignment loss
        features_inst_teacher, class_probs_teacher = self.get_pooled_features(features_teacher, proposals, self.roi_heads_teacher)
        features_inst_student, _ = self.get_pooled_features(features, proposals, self.roi_heads)
        mask = (torch.max(class_probs_teacher[:, :2], dim=1)[0] > self.score_thres).detach()
        features_inst_teacher, class_probs_teacher, features_inst_student = features_inst_teacher[mask], class_probs_teacher[mask], features_inst_student[mask]
        class_probs_teacher_norm = class_probs_teacher / torch.sqrt((class_probs_teacher ** 2).sum(dim=1, keepdim=True))
        t = class_probs_teacher_norm.mm(class_probs_teacher_norm.t()).detach()
        gw_losses.update({'loss_gw_ins': self.gw(features_inst_student, features_inst_teacher, t) * self.loss_alpha_ins})

        # update teacher
        for p_s, p_t in zip(self.backbone.parameters(), self.backbone_teacher.parameters()):
            p_t.data = self.moving_eta * p_t.data.detach() + (1 - self.moving_eta) * p_s.data.detach()
        for p_s, p_t in zip(self.roi_heads.parameters(), self.roi_heads_teacher.parameters()):
            p_t.data = self.moving_eta * p_t.data.detach() + (1 - self.moving_eta) * p_s.data.detach()

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        losses = {}
        losses.update(detector_losses)
        losses.update(gw_losses)
        return losses

    @staticmethod
    def create_from_sup(net, video_id, loss_alpha_ins=0.1, loss_alpha_global=0.1, gw_add_rate=0.5, score_thres=0.8, moving_eta=0.999):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        net.backbone_teacher, net.roi_heads_teacher = copy.deepcopy(net.backbone), copy.deepcopy(net.roi_heads)
        net.gw = gromovWasserstein(gw_add_rate)
        net.enhancer = enhance_vgg16(video_id)
        net.loss_alpha_ins, net.loss_alpha_global, net.gw_add_rate, net.score_thres, net.moving_eta = loss_alpha_ins, loss_alpha_global, gw_add_rate, score_thres, moving_eta
        net.__class__ = GeneralizedRCNNLODS
        return net


class APDropException(Exception):
    pass

# wrap detectron2/engine/defaults.py:DefaultTrainer
class LODSTrainer(DefaultTrainer):
    def __init__(self, cfg, args):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        model = GeneralizedRCNNLODS.create_from_sup(model, args.id, args.loss_alpha_ins, args.loss_alpha_global, args.gw_add_rate, args.score_thres, args.moving_eta)
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
        # teacher & RPN not trained
        for m in [model.backbone_teacher, model.roi_heads_teacher, model.proposal_generator]:
            for p in m.parameters():
                p.requires_grad = False
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
            #         if k.startswith('backbone_teacher') or k.startswith('roi_heads_teacher'):
            #             del self.best_params[k]
            #         else:
            #             self.best_params[k] = self.best_params[k].cpu()
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


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()

    if args.id in video_id_list:
        desc_manual_valid, dst_manual_valid = 'LODS_manual_%s' % args.id, get_annotation_dict(args)
        desc_train, dst_train = 'LODS_unlabeled_%s' % args.id, get_unlabeled_dicts(args)
        for im in dst_train:
            im['annotations'] = [{'bbox': [500, 500, 700, 700], 'iscrowd': 0, 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}] # to pass sanity check
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    elif args.id == 'compound':
        args.id = '_compound'
        import functools
        desc_manual_valid, dst_manual_valid = 'LODS_manual_%s' % args.id, all_annotation_dict(args)
        desc_train, dst_train = 'LODS_unlabeled_%s' % args.id, all_unlabeled_dicts(args, args.batch_size * args.iters)
        dst_train = functools.reduce(lambda x, y: x + y, dst_train)
        for im in dst_train:
            im['annotations'] = [{'bbox': [500, 500, 700, 700], 'iscrowd': 0, 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}] # to pass sanity check
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    else:
        raise NotImplementedError

    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
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
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_manual_valid,)
    # print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 100
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 100

    trainer = LODSTrainer(cfg, args)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)

    prefix = 'adapt%s_%s_LODS' % (args.id, args.model)
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
    trainer.model.backbone_teacher, trainer.model.roi_heads_teacher = None, None
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

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
    loss_keys = ['loss_cls', 'loss_box_reg', 'loss_gw_global', 'loss_gw_ins']
    legends = []
    for i in range(0, len(loss_keys)):
        _k = loss_keys[i]
        xs, ys = loss_history_dict[_k][:, 0], loss_history_dict[_k][:, 1]
        y_max = np.absolute(ys).max()
        ys /= y_max
        plt.plot(xs, ys, linestyle='-', color=colors[i])
        legends.append(_k + ' ($\\times %.4f$)' % y_max)
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


def convert_ckpt():
    ckpt = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
    state_dict = torch.load(ckpt)
    keys = list(state_dict.keys())
    for k in keys:
        if k.startswith('backbone.'):
            k2 = 'backbone_teacher.' + k[9:]
            print(k, '->', k2)
            state_dict[k2] = copy.deepcopy(state_dict[k])
        if k.startswith('roi_heads.'):
            k2 = 'roi_heads_teacher.' + k[10:]
            print(k, '->', k2)
            state_dict[k2] = copy.deepcopy(state_dict[k])
    torch.save(state_dict, ckpt[:-4] + '_LODS.pth')


if __name__ == '__main__':
    # convert_ckpt(); exit(0)
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')

    parser.add_argument('--loss_alpha_ins', type=float, default=0.1)
    parser.add_argument('--loss_alpha_global', type=float, default=0.1)
    parser.add_argument('--gw_add_rate', type=float, default=0.5)
    parser.add_argument('--score_thres', type=float, default=0.8)
    parser.add_argument('--moving_eta', type=float, default=0.999)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    adapt(args)


'''
python finetune_lods.py --id 001 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x_LODS.pth --iters 400 --eval_interval 50 --image_batch_size 2 --num_worker 1
'''
