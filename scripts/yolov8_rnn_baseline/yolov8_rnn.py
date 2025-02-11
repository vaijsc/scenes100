#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
from copy import deepcopy
import gzip
import math
import random
import tqdm
import glob
import psutil
import hashlib
import argparse
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
from multiprocessing import Pool as ProcessPool
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import networkx

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.data as torchdata
import torchvision
from torch import nn
# torch.multiprocessing.set_start_method('spawn')
import multiprocessing
multiprocessing.set_start_method('spawn')
torch.autograd.set_detect_anomaly(True)
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator, inference_context
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances
from detectron2.config import get_cfg
from detectron2.data.samplers import TrainingSampler
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager

from yolov8 import *

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


# Global Variables
video_id_list = [
    '001', '003', '005', '006', '007', '008', '009', '011', '012', '013', 
    '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', 
    '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', 
    '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', 
    '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', 
    '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', 
    '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', 
    '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', 
    '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', 
    '161', '164', '167', '169', '170', '171', '172', '175', '178', '179'
]
# video_id_list = ['005']

bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_inference_server')


def shuffle_batch(data, batch_size):
    assert len(data) % batch_size == 0, "Total num samples is not dividable by batch size!"
    num_batch = len(data) // batch_size
    shuffled_data = []
    shuffled_batch_index = list(range(num_batch))
    random.shuffle(shuffled_batch_index)
    for i in range(num_batch):
        shuffled_data.extend(data[batch_size * shuffled_batch_index[i]: batch_size * (shuffled_batch_index[i] + 1)])
    return shuffled_data


class TemporalEnhancer(torch.nn.Module): #TODO: cross attention?
    def __init__(self, in_channels):
        super(TemporalEnhancer, self).__init__()
        # Linear layer to compute compatibility score (optional, depends on your design)
        self.fc = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        # Convolutional layer to refine the combined feature map (optional)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = torch.nn.Softmax2d()
        self.init(in_channels)

    def forward(self, accumulated_feature_map, current_feature_map):
        # Step 1: Compute Compatibility Scores
        concatenated_maps = torch.cat([current_feature_map, accumulated_feature_map], dim=1) # Shape: (B, 2C, H, W)
        compatibility_score = self.fc(concatenated_maps)  # Shape: (B, 1, H, W)

        # Step 2: Apply Softmax to Get Attention Weights
        # attention_weights = self.softmax(compatibility_score)  # Softmax across spatial dimensions
        # breakpoint()
        # Step 3: Multiply the Attention Weights with Feature Maps
        # weighted_current_map = attention_weights * current_feature_map
        weighted_accumulated_map = compatibility_score * accumulated_feature_map.clone()
        # breakpoint()
        # Step 4: Combine the Weighted Feature Maps (using addition here)
        combined_feature_map = weighted_accumulated_map + (1 - compatibility_score) * current_feature_map

        # Optional: Further refine the combined feature map
        final = self.conv(combined_feature_map)
        return final

    def init(self, in_channels):
        with torch.no_grad():
            self.fc.weight.fill_(0.0)  # Set all weights to zero
            self.fc.bias.fill_(0.0) 

            # Set the weights to the identity matrix
            self.conv.weight.data = torch.eye(in_channels).view(in_channels, in_channels, 1, 1)
            # Set the bias to zero
            self.conv.bias.data.zero_()

class YOLORNN(DetectionModel):
    # def __init__(self, cfg="yolov8s.yaml", ch=3, nc=None, verbose=True):
    #     super().__init__(cfg, ch, nc, verbose)
    #     # Uncomment if hidden state is learnable
    #     # self.h = ...

    #     self.temp_enhancers = [TemporalEnhancer(in_channels) for in_channels in [128, 256, 512]]

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        batched_inputs: have to be consecutive frames
        """
        if not self.training:
                return self.inference(batched_inputs)

        batch = self.preprocess_image(batched_inputs)
        x = batch['img']
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        
        y = []  # outputs
        for i, m in enumerate(self.model):
            if i == len(self.model) - 1:
                break 
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        # Now x is the final features before the detection head.
        # 15, 18, 21
        # self.h = [[torch.zeros((1,) + y[15].shape[1:], requires_grad=True, device='cuda'), torch.zeros((1,) + y[18].shape[1:], requires_grad=True, device='cuda'), torch.zeros((1,) + y[21].shape[1:], requires_grad=True, device='cuda')]]
        self.h = [[y[15][:1].clone().detach().requires_grad_(True), y[18][:1].clone().detach().requires_grad_(True), y[21][:1].clone().detach().requires_grad_(True)]] 
        # [(1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)]
        # TODO: Learnable hidden state
        self.inner_optimizer = torch.optim.Adam(self.h[-1], lr=self.agg_lr)
        hidden_state_copy = list(tensor.detach().clone().requires_grad_(False) for tensor in self.h[-1])
        self.h.insert(-1, hidden_state_copy)

        hidden_state_clone = list(tensor.clone().requires_grad_(True) for tensor in self.h[-1])
        all_feats = [y[15], y[18], y[21]] # (BxC1xH1xW1, BxC2xH2xW2, BxC3xH3xW3)
        all_feats_clone = list(tensor.clone().requires_grad_(True) for tensor in all_feats)
        
        # Get Os infer
        self.model[-1].training = False 
        os = self.model[-1](all_feats_clone)  
        self.model[-1].training = True       
        outputs_os = non_max_suppression(os[0], conf_thres=0.4, iou_thres=0.45)  # conf=0.4 to avoid noise
        results_os = self.reverse_yolo_transform(outputs_os, batched_inputs, output_format='yolo')
        all_instances = []
        
        for i in range(len(batched_inputs)):
            instances = results_os[i].clone() if isinstance(results_os[i], torch.Tensor) else np.copy(results_os[i])
            # XYXY abs to XYWH normalized
            instances[:, 2] = ((results_os[i][..., 0] + results_os[i][..., 2]) / 2) / batch['img'].shape[-1]
            instances[:, 3] = ((results_os[i][..., 1] + results_os[i][..., 3]) / 2) / batch['img'].shape[-2]
            instances[:, 4] = ((results_os[i][..., 2] - results_os[i][..., 0])) / batch['img'].shape[-1]
            instances[:, 5] = ((results_os[i][..., 3] - results_os[i][..., 1])) / batch['img'].shape[-2]
            instances[:, 1] = results_os[i][..., 5]
            instances[:, 0] = i
            
            all_instances.append(instances.cuda())

        targets = torch.cat(all_instances, dim=0).cuda()
        targets_os = {} # O(t) anno
        targets_os['batch_idx'] = targets[:, 0].squeeze().cuda() # N
        targets_os['cls'] = targets[:, 1].view(-1, 1).cuda()     # Nx1
        targets_os['bboxes'] = targets[:, 2:].cuda()             # Nx4

        ds_t = []
        # Traverse through each time step
        # last_hidden = self.h[-1] # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
        # First step (t = 1)
        enhanced_feats =  [self.temp_enhancers[i](h, f[0:1]) for i, (h, f) in enumerate(zip(self.h[-1], all_feats))]
        # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
        d_t_train = self.model[-1](enhanced_feats)
        # return (1xCxH1xW1, 1xCxH2xW2, 1xCxH3xW3) in training mode
        ds_t.append(d_t_train)  
        p_t_prev = d_t_train      

        for t in range(1, batch['img'].shape[0]): # t from 1 to B-1
            last_hidden = self.h[-1] # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
            enhanced_feats =  [self.temp_enhancers[i](h, f[t: t+1]) for i, (h, f) in enumerate(zip(last_hidden, all_feats))]
            # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
            enhanced_feats_clone = list(tensor.clone().requires_grad_(True) for tensor in enhanced_feats)
            d_t_train = self.model[-1](enhanced_feats_clone)
            # return (1xCxH1xW1, 1xCxH2xW2, 1xCxH3xW3) in training mode
            ds_t.append(d_t_train)

            self.model[-1].training = False         
            d_t_infer = self.model[-1](enhanced_feats)
            self.model[-1].training = True
            # Update new hidden state based on L(D(t), P(t-1))
            outputs_d_t = non_max_suppression(d_t_infer[0], conf_thres=0.4, iou_thres=0.45)  # conf=0.4 to avoid noise
            results_d_t = self.reverse_yolo_transform(outputs_d_t, batched_inputs, output_format='yolo')
            
            targets_d_t = {} # D(t) anno

            instances = results_d_t[0].clone() if isinstance(results_d_t[0], torch.Tensor) else np.copy(results_d_t[0])
            
            # XYXY abs to XYWH normalized
            instances[:, 2] = ((results_d_t[0][..., 0] + results_d_t[0][..., 2]) / 2) / batch['img'].shape[-1]
            instances[:, 3] = ((results_d_t[0][..., 1] + results_d_t[0][..., 3]) / 2) / batch['img'].shape[-2]
            instances[:, 4] = ((results_d_t[0][..., 2] - results_d_t[0][..., 0])) / batch['img'].shape[-1]
            instances[:, 5] = ((results_d_t[0][..., 3] - results_d_t[0][..., 1])) / batch['img'].shape[-2]
            
            instances[:, 1] = results_d_t[0][..., 5]
            instances[:, 0] = 0 
            
            targets = instances.cuda()
            targets_d_t['batch_idx'] = targets[:, 0].squeeze().cuda()
            targets_d_t['cls'] = targets[:, 1].view(-1, 1).cuda()
            targets_d_t['bboxes'] = targets[:, 2:].cuda()

            inner_pred_loss, inner_pred_loss_components = self.criterion(p_t_prev, targets_d_t)

            mask = targets_os['batch_idx'] == t
            K = mask.sum().item() 
            targets_o_t = {'batch_idx': torch.zeros(K).cuda(), 'cls': targets_os['cls'][mask].view(-1, 1), 'bboxes': targets_os['bboxes'][mask].view(-1, 4)}
            inner_consistency_loss, inner_consistency_loss_components = self.criterion(d_t_train, targets_o_t)

            inner_loss = inner_consistency_loss + inner_pred_loss # * alpha
            self.inner_optimizer.zero_grad()
            inner_loss.backward(retain_graph=True)
            # self._write_metrics(loss_dict, data_time)
            self.inner_optimizer.step()
            hidden_state_copy = list(tensor.detach().clone().requires_grad_(False) for tensor in self.h[-1])
            self.h.insert(-1, hidden_state_copy)

            # Compute P(t)
            new_enhanced_feats =  [self.temp_enhancers[i](h, f[t: t+1]) for i, (h, f) in enumerate(zip(self.h[-1], all_feats))]
            p_t_prev = self.model[-1](new_enhanced_feats) 

        # Return outer loss
        all_d_t = []
        for i in range(len(ds_t[0])):
            all_d_t.append(torch.cat([d_t[i] for d_t in ds_t], dim=0))
        outer_loss, outer_loss_components = self.criterion(all_d_t, batch)   # loss between d_t and GT/pseudo_label
        print(outer_loss/len(batched_inputs))
        # Consistency loss?
        
        return outer_loss, outer_loss_components

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        batched_inputs: have to be consecutive frames
        """
        batch = self.preprocess_image(batched_inputs)
        x = batch['img']
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        y = []  # outputs
        for i, m in enumerate(self.model):
            if i == len(self.model) - 1:
                break 
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        
        # torch.save([y[15], y[18], y[21]], 'base_feats.pth')
        # Now x is the final features before the detection head.
        # 15, 18, 21
        with torch.enable_grad():
            # self.h = [[torch.zeros((1,) + y[15].shape[1:], requires_grad=True, device='cuda'), torch.zeros((1,) + y[18].shape[1:], requires_grad=True, device='cuda'), torch.zeros((1,) + y[21].shape[1:], requires_grad=True, device='cuda')]]
            self.h = [[y[15][:1].clone().detach().requires_grad_(True), y[18][:1].clone().detach().requires_grad_(True), y[21][:1].clone().detach().requires_grad_(True)]] 
            breakpoint()
            # [(1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)]
            # TODO: Learnable hidden state
            self.inner_optimizer = torch.optim.Adam(self.h[-1], lr=self.agg_lr)
            hidden_state_copy = list(tensor.detach().clone().requires_grad_(False) for tensor in self.h[-1])
            self.h.insert(-1, hidden_state_copy)
            
            all_feats = [y[15], y[18], y[21]] # (BxC1xH1xW1, BxC2xH2xW2, BxC3xH3xW3)
            all_feats_clone = list(tensor.clone().requires_grad_(True) for tensor in all_feats)

            os = self.model[-1](all_feats_clone) # Original prediction
            outputs_os = non_max_suppression(os[0], conf_thres=0.4, iou_thres=0.45)  # conf=0.4 to avoid noise
            results_os = self.reverse_yolo_transform(outputs_os, batched_inputs, output_format='yolo')
            all_instances = []
            
            for i in range(len(batched_inputs)):
                instances = results_os[i].clone() if isinstance(results_os[i], torch.Tensor) else np.copy(results_os[i])
                # XYXY abs to XYWH normalized
                instances[:, 2] = ((results_os[i][..., 0] + results_os[i][..., 2]) / 2) / batch['img'].shape[-1]
                instances[:, 3] = ((results_os[i][..., 1] + results_os[i][..., 3]) / 2) / batch['img'].shape[-2]
                instances[:, 4] = ((results_os[i][..., 2] - results_os[i][..., 0])) / batch['img'].shape[-1]
                instances[:, 5] = ((results_os[i][..., 3] - results_os[i][..., 1])) / batch['img'].shape[-2]
                instances[:, 1] = results_os[i][..., 5]
                instances[:, 0] = i
                
                all_instances.append(instances.cuda())

            targets = torch.cat(all_instances, dim=0).cuda()
            targets_os = {} # O(t) anno
            targets_os['batch_idx'] = targets[:, 0].squeeze().cuda()
            targets_os['cls'] = targets[:, 1].view(-1, 1).cuda()
            targets_os['bboxes'] = targets[:, 2:].cuda()

            ds_t = []
            # Traverse through each time step
            # First step (t = 1) 
            enhanced_feats =  [self.temp_enhancers[i](h, f[0: 1]) for i, (h, f) in enumerate(zip(self.h[-1], all_feats))]
            # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
            enhanced_feats_clone = list(tensor.clone().requires_grad_(True) for tensor in enhanced_feats) # list(tensor.clone().requires_grad_(True) for tensor in enhanced_feats)
            d_t_infer = self.model[-1](enhanced_feats_clone)     
            outputs_d_t = non_max_suppression(d_t_infer[0], conf_thres=0.25, iou_thres=0.45) # conf=0.25 to obtain results
            reversed_outputs_d_t = self.reverse_yolo_transform(outputs_d_t, batched_inputs)
            results_d_t = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(
                reversed_outputs_d_t, batched_inputs, [im['image'].shape[1:] for im in batched_inputs]
                )
            ds_t.extend(results_d_t)    

            self.model[-1].training = True
            p_t_prev = self.model[-1](enhanced_feats)  # P1 = D1 in training mode
            self.model[-1].training = False

            for t in range(1, batch['img'].shape[0]): # t from 1 to B-1
                # Get D(t)
                last_hidden = self.h[-1] # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
                enhanced_feats =  [self.temp_enhancers[i](h, f[t: t+1]) for i, (h, f) in enumerate(zip(last_hidden, all_feats))]
                # (1xC1xH1xW1, 1xC2xH2xW2, 1xC3xH3xW3)
                # need the enhanced feats to be in the same shape as normal feats

                self.model[-1].training = True
                enhanced_feats_clone = list(tensor.clone().requires_grad_(True) for tensor in enhanced_feats) 
                d_t_train = self.model[-1](enhanced_feats_clone)
                self.model[-1].training = False

                d_t_infer = self.model[-1](enhanced_feats) 
        
                outputs_d_t = non_max_suppression(d_t_infer[0], conf_thres=0.25, iou_thres=0.45) # conf=0.25 to obtain results
                reversed_outputs_d_t = self.reverse_yolo_transform(outputs_d_t, batched_inputs)
                results_d_t = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(
                    reversed_outputs_d_t, batched_inputs, [im['image'].shape[1:] for im in batched_inputs]
                    )
                ds_t.extend(results_d_t)
                # Update new hidden state based on L(D(t), P(t-1))
                outputs_d_t = non_max_suppression(d_t_infer[0], conf_thres=0.4, iou_thres=0.45) # conf=0.4 to teach P(t)
                results_d_t = self.reverse_yolo_transform(outputs_d_t, batched_inputs, output_format='yolo')

                targets_d_t = {} # D(t) anno
                instances = results_d_t[0].clone() if isinstance(results_d_t[0], torch.Tensor) else np.copy(results_d_t[0])
                
                # XYXY abs to XYWH normalized
                instances[:, 2] = ((results_d_t[0][..., 0] + results_d_t[0][..., 2]) / 2) / batch['img'].shape[-1]
                instances[:, 3] = ((results_d_t[0][..., 1] + results_d_t[0][..., 3]) / 2) / batch['img'].shape[-2]
                instances[:, 4] = ((results_d_t[0][..., 2] - results_d_t[0][..., 0])) / batch['img'].shape[-1]
                instances[:, 5] = ((results_d_t[0][..., 3] - results_d_t[0][..., 1])) / batch['img'].shape[-2]
                
                instances[:, 1] = results_d_t[0][..., 5]
                instances[:, 0] = 0
                targets = instances.cuda()
                targets_d_t['batch_idx'] = targets[:, 0].squeeze().cuda()
                targets_d_t['cls'] = targets[:, 1].view(-1, 1).cuda()
                targets_d_t['bboxes'] = targets[:, 2:].cuda()

                inner_pred_loss, inner_pred_loss_components = self.criterion(p_t_prev, targets_d_t)

                mask = targets_os['batch_idx'] == t
                K = mask.sum().item() 
                targets_o_t = {'batch_idx': torch.zeros(K).cuda(), 'cls': targets_os['cls'][mask].view(-1, 1), 'bboxes': targets_os['bboxes'][mask].view(-1, 4)}
                inner_consistency_loss, inner_consistency_loss_components = self.criterion(d_t_train, targets_o_t)
                inner_loss = inner_consistency_loss + inner_pred_loss # * alpha
                self.inner_optimizer.zero_grad()
                inner_loss.backward()
                # self._write_metrics(loss_dict, data_time)
                self.inner_optimizer.step()
                
                hidden_state_copy = list(tensor.detach().clone().requires_grad_(False) for tensor in self.h[-1])
                self.h.insert(-1, hidden_state_copy)
                # Compute P(t)
                new_enhanced_feats =  [self.temp_enhancers[i](h, f[t: t+1]) for i, (h, f) in enumerate(zip(self.h[-1], all_feats))]
                self.model[-1].training = True
                p_t_prev = self.model[-1](new_enhanced_feats) 
                self.model[-1].training = False

        # for i in range(batch['img'].shape[0]):
        #     DetectionModel.draw(batch['img'][i], ds_t[i]['instances'], format='frcnn', input=False, ratio=batched_inputs[i]['image'].shape[-2] / batched_inputs[i]['height'], file_name=f"output_model_{i}.png", thing_classes=self.thing_classes)
        # breakpoint()
        return ds_t

    @classmethod
    def create_from_sup(cls, net, agg_lr, thing_classes=None):
        net.temp_enhancers = nn.ModuleList([TemporalEnhancer(in_channels).cuda() for in_channels in [128, 256, 512]])
        # Uncomment if hidden state is learnable
        net.agg_lr = agg_lr
        net.thing_classes = thing_classes
        # self.h = ...
        net.__class__ = cls
        return net


class AdaptativePartialTrainer(DefaultTrainer):
    """Trainer class for adaptive partial training of YOLO models."""

    def __init__(self, cfg, train_whole, incremental_videos, ckpt, full_ckpt):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):
            detectron2.utils.logger.setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        model = load_yolov8(os.path.join(os.path.dirname(__file__), cfg.YOLO_CONFIG_PATH), ckpt)
        
        if cfg.DATASETS.NAME == 'scenes100':
            teacher_path = "../../models/yolov8s_remap.pth"
        elif cfg.DATASETS.NAME == 'vidvrd':
            teacher_path = "../../models/yolov8s_vidvrd.pth"
        elif cfg.DATASETS.NAME == 'imagenet_vid':
            teacher_path = "../../models/yolov8s_imagenet_vid.pth"

        self.model_teacher = load_yolov8(os.path.join(os.path.dirname(__file__), cfg.YOLO_CONFIG_PATH), teacher_path)
        self.model_teacher.output_format = "yolo"
        self.model_teacher.eval()
 
        model = YOLORNN.create_from_sup(model, cfg.SOLVER.AGG_LR, cfg.DATASETS.THING_CLASSES)
        if full_ckpt is not None:
            model.load_state_dict(torch.load(full_ckpt))
    
        if train_whole:
            trainable_modules = [model]
        else:
            trainable_modules = model.temp_enhancers
        if cfg.SOLVER.TRAIN_BASE:
            trainable_modules = model.model

        _count_all, _count_train = 0, 0
        for p in model.parameters():
            _count_all += p.numel()
            p.requires_grad = False
        for m in trainable_modules:
            for p in m.parameters():
                _count_train += p.numel()
                p.requires_grad = True
        print(f"Training {_count_train} parameters in total {_count_all} parameters, which is {_count_train/_count_all*100 : .2f}%")
        optimizer = self.build_optimizer(cfg, torch.nn.ModuleList(trainable_modules))

        if incremental_videos:
            data_loader = detectron2.data.build_detection_train_loader(
                cfg, sampler=detectron2.data.samplers.distributed_sampler.TrainingSampler(cfg.DATASETS.TRAIN_NUM_IMAGES, shuffle=False)
            )
        else:
            sampler = TrainingSampler(cfg.DATASETS.LEN_TRAIN, shuffle=False)
            data_loader = self.build_train_loader(cfg, sampler=sampler)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self._trainer.model_teacher = self.model_teacher
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []
        self._trainer.pseudo_det_min_score = cfg.SOLVER.PSEUDO_DET_MIN_SCORE

    def build_hooks(self):
        """Builds hooks for evaluation and saving model states."""
        ret = super().build_hooks()
        self.eval_results_all = {}
        self.best_AP = -1
        self.best_iter = -1
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = deepcopy(self._last_eval_results)
            # breakpoint()
            if self._last_eval_results['bbox']['AP'] > self.best_AP:
                self.best_iter = self.iter
                self.best_AP = self._last_eval_results['bbox']['AP']
            return self._last_eval_results

        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)

        def save_model_state():
            model = self.model
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            prefix = '%s.iter.%d' % (self.cfg.SOLVER.SAVE_PREFIX, self.iter)
            torch.save(model.state_dict(), prefix + '.pth')
            if self.iter == self.best_iter:
                torch.save(model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, 'best.pth'))

        ret.append(detectron2.engine.hooks.EvalHook(self.cfg.SOLVER.SAVE_INTERVAL, save_model_state))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Builds evaluator for COCO format."""
        return COCOEvaluator(dataset_name, output_dir=finetune_output)
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, sampler=sampler)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, batch_size=cfg.SOLVER.IMS_PER_BATCH)


def finetune_ema_simple_trainer_run_step(self):
    """Defines the step to run during training for fine-tuning."""
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start

    pseudo_idx, pseudo_inputs = [], []
    for _i, _d in enumerate(data):
        if 'image_test' in _d:
            pseudo_idx.append(_i)
            _h, _w = _d['instances'].image_size
            pseudo_inputs.append({'image': _d['image_test'], 'height': _h, 'width': _w})
    if len(pseudo_idx) > 0:
        with torch.no_grad():
            pseudo_labels = self.model_teacher.inference(pseudo_inputs)
            for _i, _pred in zip(pseudo_idx, pseudo_labels):
                data[_i]['instances'] = _pred
                del data[_i]['image_test']

    loss, loss_components = self.model(data)

    loss_dict_items = {
        "Box loss": float(loss_components[0]),
        "Class loss": float(loss_components[1]),
        "DFL loss": float(loss_components[2])
    }
    loss_dict = {
        "Box loss": loss_components[0],
        "Class loss": loss_components[1],
        "DFL loss": loss_components[2]
    }

    self.optimizer.zero_grad()
    loss.backward()
    self._write_metrics(loss_dict, data_time)
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


class DatasetMapperPseudo(detectron2.data.DatasetMapper):
    """DatasetMapper class for generating pseudo-labels during training."""

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = deepcopy(dataset_dict)  # it will be modified by code below
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)

        if 'source' in dataset_dict and dataset_dict['source'] == 'unlabeled':
            image_test = self.apply_test_transform(image)
            if image_test is not None:
                dataset_dict['image_test'] = torch.as_tensor(np.ascontiguousarray(image_test.transpose(2, 0, 1)))

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    def apply_test_transform(self, image):
        """Applies test-time transformations to the input image."""
        if not (image.dtype == np.uint8 and len(image.shape) == 3 and image.shape[2] == 3):
            return None
        h, w = image.shape[:2]
        scale = 2
        min_size, max_size = map(lambda x: int(x * scale), [self.min_size_test, self.max_size_test])
        newh, neww = self.get_output_shape(h, w, min_size, max_size)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
        return np.asarray(pil_image)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
        """Calculates new output dimensions for the image."""
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    @staticmethod
    def create_from_sup(mapper, cfg):
        """Creates a DatasetMapperPseudo instance from a base mapper."""
        assert not cfg.INPUT.CROP.ENABLED
        assert cfg.INPUT.RANDOM_FLIP == 'none'
        mapper.min_size_test, mapper.max_size_test = cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperPseudo
        return mapper

def get_pseudo_label(args, dataset, model_teacher=None, scale=2):
    # if os.path.exists("valid_pseudo_anno.json"):
    #     with open("valid_pseudo_anno.json", "r") as json_file:
    #         return json.load(json_file)
    
    def apply_test_transform(image, scale=2):
        """Applies test-time transformations to the input image."""
        if not (image.dtype == np.uint8 and len(image.shape) == 3 and image.shape[2] == 3):
            return None
        h, w = image.shape[:2]
        newh = scale * h  
        neww = scale * w
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
        return np.asarray(pil_image)
    
    if model_teacher is None:
        model_teacher = load_yolov8(os.path.join(os.path.dirname(__file__), args.config), "../../models/yolov8s_remap.pth")
        model_teacher.output_format = "yolo"
        model_teacher.eval()
    
    for dataset_dict in tqdm.tqdm(dataset):
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format='BGR')
        if 'source' in dataset_dict and dataset_dict['source'] == 'unlabeled':
            image_test = apply_test_transform(image)
            if image_test is not None:
                img_test = torch.as_tensor(np.ascontiguousarray(image_test.transpose(2, 0, 1)))
    
            pseudo_inputs = [{'image': img_test, 'height': dataset_dict['height'], 'width': dataset_dict['width']}]
            with torch.no_grad():
                results = model_teacher.inference(pseudo_inputs)[0]
            # DetectionModel.draw(image.transpose(2, 0, 1)[(2, 1, 0), :, :], results, input=False, ratio=1, file_name='pseudo_label.png')
            annotations = []
            for i in range(results.shape[0]):
                anno = {'bbox': results[i][:4].tolist(), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(results[i][5])}
                annotations.append(anno)
            dataset_dict['annotations'] = annotations
            del dataset_dict['source']
    
    # with open("valid_pseudo_anno.json", "w") as json_file:
    #     json.dump(dataset, json_file, indent=4) 
    
    return dataset

def get_scenes100_images_split_by_frame(args):
    # Load validation images
    dst_manual_valid, desc_manual_valid = [], 'allvideos_manual'

    # Load training images
    dst_pseudo_anno, desc_pseudo_anno = [], 'allvideos_unlabeled_cocotrain'
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    for v in video_id_list:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = meta['ifilelist']
        
        # make the number of training frames for each video divisible by batch size
        k = 1 # number of batches for testing
        num_train = ((len(ifilelist) - k*args.image_batch_size) // args.image_batch_size) * args.image_batch_size
        ifilelist_train = ifilelist[:num_train]
        ifilelist_valid = ifilelist[-k*args.image_batch_size:]
        
        dict_json_train = [{
            'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
            'image_id': i,
            'height': meta['meta']['video']['H'],
            'width': meta['meta']['video']['W'],
            'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}],
            'source': 'unlabeled',
            'video_id': v
        } for i, fname in enumerate(ifilelist_train)]
        
        print(f'unlabeled frames for training of video {v} at {lmdb_path}: {len(dict_json_train)} images')
        if len(dict_json_train) > images_per_video_cap:
            # random.shuffle(dict_json)
            dict_json_train = dict_json_train[:images_per_video_cap]
            # print(f'randomly downsampled to: {len(dict_json)} images')
        dst_pseudo_anno.extend(dict_json_train)

        dict_json_valid = [{
            'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
            'image_id': i,
            'height': meta['meta']['video']['H'],
            'width': meta['meta']['video']['W'],
            'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}],
            'source': 'unlabeled',
            'video_id': v
        } for i, fname in enumerate(ifilelist_valid)]
        
        print(f'unlabeled frames for validation of video {v} at {lmdb_path}: {len(dict_json_valid)} images')
        dst_manual_valid.extend(dict_json_valid)

    return dst_pseudo_anno, desc_pseudo_anno, dst_manual_valid, desc_manual_valid


def get_scenes100_images_split_by_video(args):
    # Load validation images
    dst_manual_valid, desc_manual_valid = [], 'allvideos_manual'

    # Load training images
    dst_pseudo_anno, desc_pseudo_anno = [], 'allvideos_unlabeled_cocotrain'

    video_train = random.sample(video_id_list, 80) # TODO: edit
    video_test = [i for i in video_id_list if i not in video_train]

    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    for v in video_train:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = meta['ifilelist']
        
        # make the number of training frames for each video divisible by batch size
        num_train = (len(ifilelist) // args.image_batch_size) * args.image_batch_size
        ifilelist_train = ifilelist[:num_train]
        # ifilelist_valid = ifilelist[-k*args.image_batch_size:]
        
        dict_json_train = [{
            'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
            'image_id': i,
            'height': meta['meta']['video']['H'],
            'width': meta['meta']['video']['W'],
            'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}],
            'source': 'unlabeled',
            'video_id': v
        } for i, fname in enumerate(ifilelist_train)]
        
        print(f'unlabeled frames for training of video {v} at {lmdb_path}: {len(dict_json_train)} images')
        if len(dict_json_train) > images_per_video_cap:
            # random.shuffle(dict_json)
            dict_json_train = dict_json_train[:images_per_video_cap]
            # print(f'randomly downsampled to: {len(dict_json)} images')
        dst_pseudo_anno.extend(dict_json_train)  

    for v in video_test:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = meta['ifilelist']
        
        # make the number of training frames for each video divisible by batch size
        num_test = (len(ifilelist) // args.image_batch_size) * args.image_batch_size
        ifilelist_test = ifilelist[:num_test]
        # ifilelist_valid = ifilelist[-k*args.image_batch_size:]
        
        dict_json_test = [{
            'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
            'image_id': i,
            'height': meta['meta']['video']['H'],
            'width': meta['meta']['video']['W'],
            'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}],
            'source': 'unlabeled',
            'video_id': v
        } for i, fname in enumerate(ifilelist_train)]
        
        print(f'unlabeled frames for testing of video {v} at {lmdb_path}: {len(dict_json_test)} images')
        if len(dict_json_test) > images_per_video_cap:
            # random.shuffle(dict_json)
            dict_json_test = dict_json_test[:images_per_video_cap]
            # print(f'randomly downsampled to: {len(dict_json)} images')
        dst_manual_valid.extend(dict_json_test)      

    return dst_pseudo_anno, desc_pseudo_anno, dst_manual_valid, desc_manual_valid


def get_scenes100_images(args, split_by='video'):
    thing_classes = ['person', 'vehicle']
    if split_by == 'video':
        return get_scenes100_images_split_by_video(args), thing_classes
    elif split_by == 'frame':
        return get_scenes100_images_split_by_frame(args), thing_classes
    raise TypeError("Split type is not supported.")


def get_vidvrd_images(args, vidvrd_path='../../../VIDVRD'):
    thing_classes = [
        "turtle", "antelope", "bicycle", "lion", "ball", "motorcycle", 
        "cattle", "airplane", "red_panda", "horse", "watercraft", 
        "monkey", "fox", "elephant", "bird", "sheep", "frisbee", 
        "giant_panda", "squirrel", "bus", "bear", "tiger", "train", 
        "snake", "rabbit", "whale", "sofa", "skateboard", "dog", 
        "domestic_cat", "person", "lizard", "hamster", "car", "zebra"
    ]
    class_dict = {i: val for (i, val) in enumerate(thing_classes)}
    class_dict_reverse = {val: i for (i, val) in enumerate(thing_classes)}

    def _load(split='train'):
        dataset = []
        # Load video list
        video_list = sorted(os.listdir(os.path.join(vidvrd_path, 'images', split)))
        for video_id in video_list:
            with open(os.path.join(vidvrd_path, 'annotations', split, f'{video_id}.json'), 'r') as fp:
                meta = json.load(fp)
            frame_list = sorted(os.listdir(os.path.join(vidvrd_path, 'images', split, video_id)))
            # breakpoint()
            if len(meta['trajectories']) != len(frame_list):
                print(f"video {video_id} anno is corrupted!???")
            num_frames = (min(len(frame_list), len(meta['trajectories'])) // args.image_batch_size) * args.image_batch_size
            frame_list = frame_list[:num_frames]
            for i, frame in enumerate(frame_list):
                # if i >= len(meta['trajectories']):
                #     continue
                data_dict = {
                    'file_name': os.path.join(vidvrd_path, 'images', split, video_id, frame),
                    'image_id': i,
                    'height': meta['height'],
                    'width': meta['width'],
                    'annotations': [],
                    'video_id': video_id
                }
                frame_anno = meta['trajectories'][i]
                for obj in frame_anno:
                    x1, y1, x2, y2 = obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']
                    tid = obj['tid']
                    class_id = None
                    for cat in meta['subject/objects']:
                        if cat['tid'] == tid:
                            class_id = class_dict_reverse[cat['category']]
                            break
                    assert class_id is not None, 'class_id not found!'
                    data_dict['annotations'].append({'bbox': [x1, y1, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': class_id})
                
                dataset.append(data_dict)
        return dataset
    
    # Load validation images
    dst_manual_valid, desc_manual_valid = _load('test'), 'allvideos_valid'

    # Load training images
    dst_pseudo_anno, desc_pseudo_anno = _load('train'), 'allvideos_train'

    # for i in range(10):
    #     draw_boxes(dst_manual_valid[i], f'gt_{i}.png')

    return dst_pseudo_anno, desc_pseudo_anno, dst_manual_valid, desc_manual_valid, thing_classes  


def get_imagenet_vid_images_split_by_video(args, imagenet_vid_path='../../../ILSVRC2015'):
    import itertools
    import xml.etree.ElementTree as ET

    class_dict = {
        "n02691156": "airplane",
        "n02419796": "antelope",
        "n02131653": "bear",
        "n02834778": "bicycle",
        "n01503061": "bird",
        "n02924116": "bus",
        "n02958343": "car",
        "n02402425": "cattle",
        "n02084071": "dog",
        "n02121808": "domestic cat",
        "n02503517": "elephant",
        "n02118333": "fox",
        "n02510455": "giant panda",
        "n02342885": "hamster",
        "n02374451": "horse",
        "n02129165": "lion",
        "n01674464": "lizard",
        "n02484322": "monkey",
        "n03790512": "motorcycle",
        "n02324045": "rabbit",
        "n02509815": "red panda",
        "n02411705": "sheep",
        "n01726692": "snake",
        "n02355227": "squirrel",
        "n02129604": "tiger",
        "n04468005": "train",
        "n01662784": "turtle",
        "n04530566": "watercraft",
        "n02062744": "whale",
        "n02391049": "zebra"
    }

    class_ids = {key: i for (i, key) in enumerate(class_dict.keys())}
    thing_classes = ["airplane", "antelope", "bear", "bicycle", "bird", "bus",
                    "car", "cattle", "dog", "domestic cat", "elephant", "fox",
                    "giant panda", "hamster", "horse", "lion", "lizard", "monkey",
                    "motorcycle", "rabbit", "red panda", "sheep", "snake", "squirrel",
                    "tiger", "train", "turtle", "watercraft", "whale", "zebra"]
    class_dict = {i: val for (i, val) in enumerate(thing_classes)}
    class_dict_reverse = {val: i for (i, val) in enumerate(thing_classes)}

    def _load(split='train'):
        dataset = []
        if split == 'train':
            video_list =  sorted(list(itertools.chain(*[os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split, x)) for x in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split))])))
            video_paths = sorted([os.path.join(imagenet_vid_path, "Data", "VID", split, x, y) for x in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split)) for y in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split, x))])
        elif split == 'val':
            video_list =  sorted(os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split)))
            video_paths = sorted([os.path.join(imagenet_vid_path, "Data", "VID", split, x) for x in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split))])
        # video_list = video_list[:1] # TODO: test only
        # video_paths = video_paths[:1]
        for i, video_id in tqdm.tqdm(enumerate(video_list)):
            frame_list = sorted([os.path.join(video_paths[i], x) for x in os.listdir(video_paths[i])])
            num_frames = (len(frame_list) // args.image_batch_size) * args.image_batch_size
            print(f"Video {video_id}: {num_frames} frames extracted.")
            frame_list = frame_list[:num_frames]
            for j, frame_path in enumerate(frame_list):
                anno_path = frame_path.replace("Data", "Annotations").replace("JPEG", "xml")
                tree = ET.parse(anno_path)
                root = tree.getroot()

                # Extract image dimensions
                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)

                data_dict = {
                    'file_name': frame_path,
                    'image_id': j,
                    'height': height,
                    'width': width,
                    'annotations': [],
                    'video_id': video_id
                }

                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    # Get the class ID from the dictionary
                    assert class_name in class_ids, "Class not exist!"
                     
                    class_id = class_ids[class_name]

                    # Get bounding box coordinates
                    xmin = int(obj.find('bndbox/xmin').text)
                    xmax = int(obj.find('bndbox/xmax').text)
                    ymin = int(obj.find('bndbox/ymin').text)
                    ymax = int(obj.find('bndbox/ymax').text)

                    data_dict['annotations'].append({'bbox': [xmin, ymin, xmax, ymax], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': class_id})
                
                dataset.append(data_dict)
        
        return dataset

    # Load validation images
    dst_manual_valid, desc_manual_valid = _load('val'), 'allvideos_valid'

    # Load training images
    dst_pseudo_anno, desc_pseudo_anno = _load('train'), 'allvideos_train'

    for i in range(10):
        draw_boxes(dst_manual_valid[-i], f'gt_{i}.png')

    return shuffle_batch(dst_pseudo_anno, args.image_batch_size), desc_pseudo_anno, dst_manual_valid, desc_manual_valid, thing_classes  

def get_imagenet_vid_images_split_by_frame(args, imagenet_vid_path='../../../ILSVRC2015'):
    import itertools
    import xml.etree.ElementTree as ET

    class_dict = {
        "n02691156": "airplane",
        "n02419796": "antelope",
        "n02131653": "bear",
        "n02834778": "bicycle",
        "n01503061": "bird",
        "n02924116": "bus",
        "n02958343": "car",
        "n02402425": "cattle",
        "n02084071": "dog",
        "n02121808": "domestic cat",
        "n02503517": "elephant",
        "n02118333": "fox",
        "n02510455": "giant panda",
        "n02342885": "hamster",
        "n02374451": "horse",
        "n02129165": "lion",
        "n01674464": "lizard",
        "n02484322": "monkey",
        "n03790512": "motorcycle",
        "n02324045": "rabbit",
        "n02509815": "red panda",
        "n02411705": "sheep",
        "n01726692": "snake",
        "n02355227": "squirrel",
        "n02129604": "tiger",
        "n04468005": "train",
        "n01662784": "turtle",
        "n04530566": "watercraft",
        "n02062744": "whale",
        "n02391049": "zebra"
    }

    class_ids = {key: i for (i, key) in enumerate(class_dict.keys())}
    thing_classes = ["airplane", "antelope", "bear", "bicycle", "bird", "bus",
                    "car", "cattle", "dog", "domestic cat", "elephant", "fox",
                    "giant panda", "hamster", "horse", "lion", "lizard", "monkey",
                    "motorcycle", "rabbit", "red panda", "sheep", "snake", "squirrel",
                    "tiger", "train", "turtle", "watercraft", "whale", "zebra"]
    class_dict = {i: val for (i, val) in enumerate(thing_classes)}
    class_dict_reverse = {val: i for (i, val) in enumerate(thing_classes)}

    def _load(split, train_set, val_set, test_set, args):
        # dataset = []
        if split == 'train':
            video_list =  sorted(list(itertools.chain(*[os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split, x)) for x in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split))])))
            video_paths = sorted([os.path.join(imagenet_vid_path, "Data", "VID", split, x, y) for x in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split)) for y in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split, x))])
        elif split == 'val':
            video_list =  sorted(os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split)))
            video_paths = sorted([os.path.join(imagenet_vid_path, "Data", "VID", split, x) for x in os.listdir(os.path.join(imagenet_vid_path, "Data", "VID", split))])
        video_list = video_list[int(args.id):int(args.id)+1] # TODO: test only
        video_paths = video_paths[int(args.id):int(args.id)+1]
        for i, video_id in tqdm.tqdm(enumerate(video_list)):
            frame_list = sorted([os.path.join(video_paths[i], x) for x in os.listdir(video_paths[i])])
            k = 2 # number of batches for testing & validation
            num_train = ((len(frame_list) - k*args.image_batch_size) // args.image_batch_size) * args.image_batch_size
            # num_frames = (len(frame_list) // args.image_batch_size) * args.image_batch_size
            print(f"Video {video_id}: {len(frame_list)} frames extracted.")
            # frame_list = frame_list[:num_frames]
            for j, frame_path in enumerate(frame_list):
                anno_path = frame_path.replace("Data", "Annotations").replace("JPEG", "xml")
                tree = ET.parse(anno_path)
                root = tree.getroot()

                # Extract image dimensions
                width = int(root.find('size/width').text)
                height = int(root.find('size/height').text)

                data_dict = {
                    'file_name': frame_path,
                    'image_id': j,
                    'height': height,
                    'width': width,
                    'annotations': [],
                    'video_id': video_id
                }

                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    # Get the class ID from the dictionary
                    assert class_name in class_ids, "Class not exist!"
                     
                    class_id = class_ids[class_name]

                    # Get bounding box coordinates
                    xmin = int(obj.find('bndbox/xmin').text)
                    xmax = int(obj.find('bndbox/xmax').text)
                    ymin = int(obj.find('bndbox/ymin').text)
                    ymax = int(obj.find('bndbox/ymax').text)

                    data_dict['annotations'].append({'bbox': [xmin, ymin, xmax, ymax], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': class_id})
                
                if j < num_train:
                    train_set.append(data_dict)
                elif len(frame_list) - (k-1)*args.image_batch_size > j >= len(frame_list) - k*args.image_batch_size and len(frame_list) - k*args.image_batch_size >= 0:
                    val_set.append(data_dict)
                elif j >= len(frame_list) - (k-1)*args.image_batch_size and len(frame_list) - k*args.image_batch_size >= 0:
                    test_set.append(data_dict)
        

    train_set = []
    val_set = []
    test_set = []
    # Load validation images
    _load('val', train_set, val_set, test_set, args)

    # # Load training images
    # _load('train', train_set, test_set, val_set, args)

    for i in range(10):
        draw_boxes(train_set[-i], f'gt_{i}.png')
    # return train_set, 'allvideos_valid', train_set, 'allvideos_train', thing_classes   

    return shuffle_batch(train_set, args.image_batch_size), 'allvideos_train', test_set, 'allvideos_valid', thing_classes   


def get_imagenet_vid_images(args, imagenet_vid_path='../../../ILSVRC2015', split_by='video'):
    if split_by == 'video':
        return get_imagenet_vid_images_split_by_video(args, imagenet_vid_path)
    elif split_by == 'frame':
        return get_imagenet_vid_images_split_by_frame(args, imagenet_vid_path)
    raise TypeError("Split type is not supported.")


def draw_boxes(image_data, output_path):
    """
    Draw bounding boxes on the image and save the result.

    :param image_data: Dictionary containing 'file_name' and 'annotations' (COCO format data).
    :param output_path: Path to save the image with drawn boxes.
    """
    # Load the image from the file path
    import cv2

    thing_classes = ["airplane", "antelope", "bear", "bicycle", "bird", "bus",
                    "car", "cattle", "dog", "domestic cat", "elephant", "fox",
                    "giant panda", "hamster", "horse", "lion", "lizard", "monkey",
                    "motorcycle", "rabbit", "red panda", "sheep", "snake", "squirrel",
                    "tiger", "train", "turtle", "watercraft", "whale", "zebra"]

    image = cv2.imread(image_data['file_name'])
    
    # Loop through each annotation
    for annotation in image_data['annotations']:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = annotation['bbox']
        class_name = thing_classes[annotation['category_id']]
        
        # Define the box color (let's use green for the box) and thickness
        box_color = (0, 255, 0)  # RGB for green
        box_thickness = 2
        
        # Draw the rectangle on the image
        image = cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)

        # Define font, scale, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        text_color = (255, 255, 255)  # White text

        # Get the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, text_thickness)

        # Calculate the text position (slightly above the top-left corner of the box)
        text_position = (x1, y1 - 10 if y1 - 10 > 0 else y1 + text_height + 10)

        # Put the text (class name) on the image
        image = cv2.putText(image, class_name, text_position, font, font_scale, text_color, text_thickness)
    
    # Save the resulting image
    cv2.imwrite(output_path, image)


def adapt(args):
    """Adapts the model and prepares datasets based on provided arguments."""
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = deepcopy(args)
    _args.cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'MSCOCO2017'))
    _args.smallscale = False
    random.seed(42)

    if args.dataset == 'scenes100':
        data, thing_classes = get_scenes100_images(args, split_by='frame')
        dst_pseudo_anno, desc_pseudo_anno, dst_manual_valid, desc_manual_valid = data
        # get pseudo label for dst_manual_valid
        dst_manual_valid = get_pseudo_label(args, dst_manual_valid)
    elif args.dataset == 'vidvrd':
        dst_pseudo_anno, desc_pseudo_anno, dst_manual_valid, desc_manual_valid, thing_classes = get_vidvrd_images(args)
    elif args.dataset == 'imagenet_vid':
        dst_pseudo_anno, desc_pseudo_anno, dst_manual_valid, desc_manual_valid, thing_classes = get_imagenet_vid_images(args, split_by='frame')

    print(f'total images: {len(dst_pseudo_anno) + len(dst_manual_valid)} images')
    
    for i, ann in enumerate(dst_pseudo_anno):
        assert 'video_id' in ann
        ann['image_id'] = i + 1

    for i, ann in enumerate(dst_manual_valid):
        assert 'video_id' in ann
        ann['image_id'] = i + 1

    # print(f'include MSCOCO2017 training images, totally {len(dst_pseudo_anno)} images')
    prefix = f'adaptive_partial_server_yolov8s_rnn_anno_{desc_pseudo_anno}.{args.tag}'

    # Clean up and register datasets
    del _tensor, _args
    gc.collect()
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    # Configure model
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.INPUT.MIN_SIZE_TEST = 640  
    cfg.INPUT.MAX_SIZE_TEST = 640 
    cfg.INPUT.MIN_SIZE_TRAIN = (640,) 
    cfg.INPUT.MAX_SIZE_TRAIN = 640    

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = finetune_output
    
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.AGG_LR = args.agg_lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 200
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.PSEUDO_DET_MIN_SCORE = args.refine_det_score_thres
    cfg.SOLVER.SAVE_INTERVAL = args.save_interval
    cfg.SOLVER.SAVE_PREFIX = os.path.join(args.outputdir, prefix)
    cfg.SOLVER.TRAIN_BASE = args.train_base

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    cfg.MODEL.SPLIT_LIST = args.split_list
    cfg.MODEL.MAPPER = args.mapper

    cfg.TEST.EVAL_PERIOD = args.eval_interval
    
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid,)
    cfg.DATASETS.LEN_TRAIN = len(dst_pseudo_anno)
    cfg.DATASETS.NAME = args.dataset    
    cfg.DATASETS.TRAIN_NUM_IMAGES = len(dst_pseudo_anno)
    cfg.DATASETS.TEST_NUM_IMAGES = len(dst_manual_valid)
    cfg.DATASETS.THING_CLASSES = thing_classes
    
    cfg.YOLO_CONFIG_PATH = args.config

    print(cfg)

    # Adjust training parameters
    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = AdaptativePartialTrainer(cfg, args.train_whole, args.incremental_videos, args.ckpt, args.full_ckpt)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_ema_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperPseudo.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, cfg)

    # Manually load the parameters from checkpoint
    print(f'loading weights from: {args.ckpt}')
    assert trainer.model is trainer._trainer.model
    assert trainer.model_teacher is trainer._trainer.model_teacher

    print(prefix)
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print(f'Evaluate on {dataset_name}')
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    # TODO: test base model
    # with open("result_fgfa_trained.txt", 'a') as file:
    #     file.write(f"Video {args.id}: mAP {results_0['bbox']['AP']}, AP50 {results_0['bbox']['AP50']}\n")
    # sys.exit(0)
    trainer.train()

    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(args.outputdir, f'{prefix}.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)

    # Plotting results
    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(aps.keys())
    dst_list = {'mAP': [], 'AP50': []}
    for i in iter_list:
        dst_list['mAP'].append(aps[i]['bbox']['AP'])
        dst_list['AP50'].append(aps[i]['bbox']['AP50'])

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for x in loss_history:
        for loss_key, loss_val in x['loss'].items():
            if loss_key not in loss_history_dict:
                loss_history_dict[loss_key] = []
            loss_history_dict[loss_key].append([x['iter'], loss_val])
    loss_history_dict = {loss_key: np.array(vals) for loss_key, vals in loss_history_dict.items()}
    for loss_key, loss_vals in loss_history_dict.items():
        for i in range(smooth_L, loss_vals.shape[0]):
            loss_vals[i, 1] = loss_vals[i - smooth_L:i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_vals[smooth_L + 1:, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Valid AP50', 'Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')

    plt.subplot(1, 2, 2)
    colors, color_i = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000'], 0
    legends = []
    for loss_key, loss_vals in loss_history_dict.items():
        plt.plot(loss_vals[:, 0], loss_vals[:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, f'{prefix}.pdf'))


class SemiRandomClient(torchdata.Dataset):
    """Dataset class for semi-random client data loading."""

    def __init__(self, cfg):
        super(SemiRandomClient, self).__init__()
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == 'BGR'

        self.images = []
        for video_id in video_id_list:
            inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
            with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
                _dicts = json.load(fp)
            for im in _dicts:
                im['md5'] = hashlib.md5(f"{video_id}_{im['file_name']}".encode('utf-8')).hexdigest()
                im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['video_id'] = video_id
            self.images.extend(_dicts)
        self.images.sort(key=lambda x: x['md5'])
        self.preloaded_images = None

    def preload(self):
        """Preloads images into memory."""
        if self.preloaded_images is not None:
            return
        self.preloaded_images = []
        for i in tqdm.tqdm(range(len(self.images)), ascii=True, desc='preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
        """Reads an image and applies transformations."""
        image = detectron2.data.detection_utils.read_image(self.images[i]['file_name'], format=self.input_format)
        height, width = image.shape[:2]
        tf = self.aug.get_transform(image)
        image = torch.as_tensor(tf.apply_image(image).astype('float32').transpose(2, 0, 1))
        return {'image': image, 'height': height, 'width': width, 'video_id': self.images[i]['video_id']}

    def __getitem__(self, i):
        if self.preloaded_images is None:
            return self.read(i), self.images[i]
        else:
            return self.preloaded_images[i], self.images[i]

    @staticmethod
    def collate(batch):
        return batch


class _MapperDataset(torchdata.Dataset):
    """Dataset class for mapping images."""

    def __init__(self, cfg, images):
        super(_MapperDataset, self).__init__()
        self.image_mapper = detectron2.data.DatasetMapper(**detectron2.data.DatasetMapper.from_config(cfg, is_train=False))
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.image_mapper(self.images[i])

    @staticmethod
    def collate(batch):
        return batch


def random_cluster(args):
    """Performs random clustering and saves the mapper."""
    for seed in args.gen_seed:
        random.seed(seed)
        mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(args.budget)}}
        index_list = list(range(args.budget))
        for idx in video_id_list:
            choice = random.choice(index_list)
            mapper['video_id_to_index'][idx] = choice
            mapper['used_indices'][choice] = True
            if choice in mapper['un_used_indices']:
                del mapper['un_used_indices'][choice]
        print(mapper)
        torch.save(mapper, os.path.join(args.outputdir, f"mapper_random_{seed}_b{args.budget}.pth"))


def offline_cluster(args):
    """Performs offline clustering using K-Means and saves the mapper."""
    from sklearn.cluster import KMeans

    if args.from_base:
        model = load_yolov8(args.config, args.ckpt)
        model = YOLOServer.create_from_sup(model, 1, args.split_list)
    else:
        # Load trained B=1 model
        mapper = torch.load(os.path.join(args.ckpts_dir, f'{args.ckpts_tag}.mapper.pth'))
        assert mapper['budget'] == 1, 'can only run offline clustering from budget=1 model'
        
        model = load_yolov8(args.config)
        model = YOLOServer.create_from_sup(model, mapper['budget'], args.split_list)
        state_dict = torch.load(os.path.join(args.ckpts_dir, f'{args.ckpts_tag}.pth'))
        model.load_state_dict(state_dict)
        print(f'loaded weights from: {os.path.join(args.ckpts_dir, f"{args.ckpts_tag}.pth")}')
        model.budget = mapper['budget']
        model.video_id_to_index = mapper['video_id_to_index']
        model.used_indices = mapper['used_indices']
        model.un_used_indices = mapper['un_used_indices']
    model.eval()
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'base_solver.yaml'))
    video_images, num_images_per_video = [], 400
    for v in tqdm.tqdm(video_id_list, ascii=True, desc='loading images'):
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = sorted(meta['ifilelist'])
        assert len(ifilelist) / num_images_per_video > 2
        ifilelist = np.array(ifilelist[: (len(ifilelist) // num_images_per_video) * num_images_per_video])
        ifilelist = ifilelist.reshape(-1, num_images_per_video)[0]
        for i, fname in enumerate(ifilelist):
            video_images.append({
                'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', fname)),
                'image_id': i,
                'height': meta['meta']['video']['H'],
                'width': meta['meta']['video']['W'],
                'annotations': [],
                'video_id': v
            })
    print(f'total images: {len(video_images)}')
    dataset = _MapperDataset(cfg, video_images)
    loader = torchdata.DataLoader(dataset, batch_size=args.image_batch_size, collate_fn=_MapperDataset.collate, shuffle=False, num_workers=6)
    
    video_id_features = []
    video_features_p3, video_features_p4, video_features_p5 = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, ascii=True, desc='extracting features'):
            video_id_features.extend([im['video_id'] for im in batch])
            features = model.inference(batch, feature=True)
            features_p4 = torch.nn.functional.adaptive_avg_pool2d(features[1], (9, 16)).view(len(batch), -1)
            video_features_p4.append(features_p4.detach().cpu())
            features_p5 = torch.nn.functional.adaptive_avg_pool2d(features[0], (5, 9)).view(len(batch), -1)
            video_features_p5.append(features_p5.detach().cpu())
    
    video_id_features = np.array(video_id_features)
    video_features_p4 = torch.cat(video_features_p4, dim=0).detach().numpy()
    video_features_p5 = torch.cat(video_features_p5, dim=0).detach().numpy()
    torch.cuda.empty_cache()
    
    for features, desc in [
            (video_features_p4, 'fpn.p4'),
            (video_features_p5, 'fpn.p5'),
            (np.concatenate([video_features_p4, video_features_p5], axis=1), 'fpn.p4.p5')]:
        
        print(f'running {args.budget}-Means for {desc}: {features.shape} {features.dtype}')
        kmeans = KMeans(n_clusters=args.budget, random_state=0).fit(features)
        mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(args.budget)}}
        for v in video_id_list:
            cluster_ids = kmeans.labels_[video_id_features == v]
            i = np.argmax(np.bincount(cluster_ids))
            mapper['video_id_to_index'][v] = i
            mapper['used_indices'][i] = True
            if i in mapper['un_used_indices']:
                del mapper['un_used_indices'][i]
        print(mapper)
        prefix = os.path.join(args.ckpts_dir, f'{args.ckpts_tag}.{args.budget}means.{desc}')
        torch.save(mapper, prefix + f'.new{".frombase" if args.from_base else ""}.mapper.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--dataset', type=str, default='scenes100', help='option')
    parser.add_argument('--config', type=str, default='../../configs/yolov8s.yaml', help='detection model config path')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov8s_remap.pth", help='weights checkpoint of base model')
    parser.add_argument('--full_ckpt', type=str, default=None, help='weights checkpoint of full model (RNN included)')
    parser.add_argument('--mapper', type=str, default=None, help='weights checkpoint of model')

    parser.add_argument('--incremental_videos', type=bool, default=False)
    parser.add_argument('--train_whole', type=bool, default=False)
    parser.add_argument('--train_base', type=bool, default=False)

    # cluster parameters
    parser.add_argument('--split_list', type=int, nargs='+')
    parser.add_argument('--from_base', type=bool, default=False)
    parser.add_argument('--random', type=bool, default=False)
    parser.add_argument('--gen_seed', type=int, nargs='+')

    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--ckpts_tag', type=str, default='')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)
    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--instances', type=int, default=1)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--save_interval', type=int, help='interval for saving model')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--agg_lr', default=0.00001, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)

    # used for random test
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.id != '':
        video_id_list = [args.id]

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    finetune_output = args.outputdir
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'adapt':
        if args.ddp_num_gpus <= 1:
            adapt(args)
        else:
            from detectron2.engine import launch
            launch(adapt, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    if args.opt == 'cluster':
        if args.random:
            random_cluster(args)
        else:
            offline_cluster(args)

