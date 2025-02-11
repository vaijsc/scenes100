#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import enum
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
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Boxes, ImageList, Instances, pairwise_iou

from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
import detectron2.modeling
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

import logging
import weakref

from finetune import refine_annotations, get_annotation_dict, finetune_simple_trainer_run_step
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from utils import IoU, bbox_inside, intersect_ratios


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_falsenegative_control')

from finetune_falsenegative import GeneralizedRCNNFalseNegative, DatasetMapperMixupFalseNegative, FinetuneTrainer


def get_oracle_splits_with_corruption(args):
    random.seed(42)
    images = get_annotation_dict(args)
    for im in images:
        im['bbox_count_original'] = len(im['annotations'])
    assert len(images) >= 4
    n_train = 2
    print('training/validation: %d/%d' % (n_train, len(images) - n_train))
    images_train, images_valid = images[: n_train], images[n_train :]
    for im in images_train:
        for ann in im['annotations']:
            ann['src'] = 'gt'
    print('oracle training set:   %d images, %d bboxes' % (len(images_train), sum(list(map(lambda x: len(x['annotations']), images_train)))))
    print('oracle validation set: %d images, %d bboxes' % (len(images_valid), sum(list(map(lambda x: len(x['annotations']), images_valid)))))
    if args.fn_rate > 0:
        for im in tqdm.tqdm(images_train, ascii=True, desc='dropping %d%% true bboxes' % args.fn_rate):
            fn_idx = list(range(0, im['bbox_count_original']))
            random.shuffle(fn_idx)
            for i in fn_idx[: int(im['bbox_count_original'] * args.fn_rate / 100)]:
                im['annotations'][i]['src'] = 'fn'
    return images_train, images_valid


def adapt(args):
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap_50im_%s' % args.id, get_coco_dicts(args, 'valid')
    print('use dummy MSCOCO2017-validation during training')
    dst_cocovalid = dst_cocovalid[:25] + dst_cocovalid[-25:]
    desc_train, desc_valid = '%s_manual_fn%02d_cocotrain' % (args.id, args.fn_rate), '%s_manual' % args.id
    dst_train, dst_valid = get_oracle_splits_with_corruption(args)
    if args.fn_opt != 'fn_discard':
        for im in dst_train:
            im['annotations'] = list(filter(lambda x: x['src'] != 'fn', im['annotations']))
    dst_train = copy.deepcopy((dst_train * (1 + args.iters * args.image_batch_size // len(dst_train)))[: args.iters * args.image_batch_size])

    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        for ann in im['annotations']:
            ann['src'] = 'gt'
    random.shuffle(dst_cocotrain)
    dst_train = dst_train + dst_cocotrain[: len(dst_train)]
    print('include MSCOCO2017 training images, totally %d images' % len(dst_train))
    for i in range(0, len(dst_train)):
        dst_train[i]['image_id'] = i + 1

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid, lambda: dst_valid)
    MetadataCatalog.get(desc_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_train, lambda: dst_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes

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
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_valid, desc_cocovalid)
    # print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 50
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 50

    args.opt = 'crossteach'
    trainer = FinetuneTrainer(cfg, args)
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperMixupFalseNegative.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, None, None, None, False) # if mixup is performed it will fail to type-incompatible
    trainer.resume_or_load(resume=args.resume)
    prefix = 'adapt%s_%s_manual_FN%02d' % (args.id, args.model, args.fn_rate)
    if args.fn_opt == 'fn_discard':
        prefix = prefix + '_discard'

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'rpn_fn_discard_stats': m.proposal_generator.fn_discard_stats, 'roi_fn_discard_stats': m.roi_heads.fn_discard_stats}, fp)
    torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

    aps, lr_history, loss_history, rpn_fn_discard_stats, roi_fn_discard_stats = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history, m.proposal_generator.fn_discard_stats, m.roi_heads.fn_discard_stats
    for d in [rpn_fn_discard_stats, roi_fn_discard_stats]:
        for k in d:
            d[k] = int(np.array(d[k]).mean())
    iter_list = sorted(list(aps.keys()))
    dst_list = [desc_cocovalid, desc_valid]
    assert len(dst_list) == 2
    dst_list = {k: {'mAP': [], 'AP50': []} for k in dst_list}
    for i in iter_list:
        for k in dst_list:
            dst_list[k]['mAP'].append(aps[i][k]['bbox']['AP'])
            dst_list[k]['AP50'].append(aps[i][k]['bbox']['AP50'])

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
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid]['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_cocovalid]['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.plot(iter_list, np.array(dst_list[desc_valid]['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_valid]['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP', 'Manual Valid AP50', 'Manual Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')
    plt.subplot(1, 2, 2)
    colors, color_i = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000'], 0
    legends = []
    for loss_key in loss_history_dict:
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses RPN + %d/- %d/drop %d ROI + %d/- %d/drop %d' % (rpn_fn_discard_stats['1'], rpn_fn_discard_stats['0'], rpn_fn_discard_stats['fn-discard'], roi_fn_discard_stats['1'], roi_fn_discard_stats['0'], roi_fn_discard_stats['fn-discard']))
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))
    plt.close()


def adapt_all_videos(args):
    for video_id in video_id_list:
        args.id = video_id
        if len(glob.glob(os.path.join(args.outputdir, 'adapt%s*.pth' % args.id))) > 0:
            print('skipped on existing:', args.id)
            continue
        adapt(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, choices=['single', 'batch'])
    parser.add_argument('--fn_opt', type=str, choices=['fn', 'fn_discard'])
    parser.add_argument('--id', type=str, default='', choices=video_id_list+[''])
    parser.add_argument('--model', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--outputdir', type=str, default='.')

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--smallscale', type=bool, default=False)
    parser.add_argument('--fn_rate', type=int, default=0)

    parser.add_argument('--iters', type=int, default=800)
    parser.add_argument('--eval_interval', type=int, default=201)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'batch':
        adapt_all_videos(args)
    else:
        adapt(args)

'''
python finetune_falsenegative_control.py --opt single --id 003 --model r101-fpn-3x --fn_rate 0 --fn_opt fn --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN00
python finetune_falsenegative_control.py --opt single --id 003 --model r101-fpn-3x --fn_rate 15 --fn_opt fn --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN15
python finetune_falsenegative_control.py --opt single --id 003 --model r101-fpn-3x --fn_rate 15 --fn_opt fn_discard --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN15_discard


python finetune_falsenegative_control.py --opt batch --model r101-fpn-3x --fn_rate 0 --fn_opt fn --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN00
python finetune_falsenegative_control.py --opt batch --model r101-fpn-3x --fn_rate 40 --fn_opt fn --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN40
python finetune_falsenegative_control.py --opt batch --model r101-fpn-3x --fn_rate 40 --fn_opt fn_discard --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN40_discard

python finetune_falsenegative_control.py --opt batch --model r101-fpn-3x --fn_rate 80 --fn_opt fn_discard --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN80_discard && rm -r finetune_output_falsenegative finetune_output_falsenegative_control && python finetune_falsenegative_control.py --opt batch --model r101-fpn-3x --fn_rate 80 --fn_opt fn --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN_control/FN80 && rm -r finetune_output_falsenegative finetune_output_falsenegative_control
'''
