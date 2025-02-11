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
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios, count_parameters
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_weaklysupervised')

from finetune import get_annotation_dict, finetune_simple_trainer_run_step
from finetune_wdiff_earlyfusion import all_pseudo_manual_annotations_with_background, construct_image_w_background
from finetune_wdiff_midfusion import GeneralizedRCNNFinetuneBackground, FinetuneBackgroundTrainer
from finetune_wdiff_midfusion_mixup import DatasetMapperBackgroundMixup


def get_weaklysupervised_dicts(args, N, mixup=True, background=None):
    inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id))
    with open(os.path.join(inputdir, 'weaksupervised.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'weaksupervised', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('weakly supervised annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    assert N >= len(annotations)
    if not background is None:
        background_files, background_frame_idx = background
        for im in annotations:
            i = os.path.basename(im['file_name'])
            i = int(i[:i.find('.')])
            im['file_name_background'] = background_files[np.absolute(background_frame_idx - i).argmin()]
    annotations = (annotations * int(N / len(annotations) + 10))[: N]
    if mixup and len(annotations) > 1:
        annotations_copy = copy.deepcopy(annotations)
        for im in tqdm.tqdm(annotations, ascii=True, desc='populating mixup sources'):
            for _ in range(0, 3):
                im['mixup_src_images'] = [annotations_copy[random.randrange(0, len(annotations_copy))]]
        del annotations_copy
    return annotations


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    random.seed(42)

    if args.id in video_id_list:
        background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
        background_frame_idx = list(map(lambda x: os.path.basename(x), background_files))
        background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), background_frame_idx)))
        desc_manual_valid, dst_manual_valid = '%s_manual_wdiff_midfusion_mixup' % args.id, get_annotation_dict(args)
        for im in dst_manual_valid:
            im['file_name_background'] = background_files[-1] # choice of background images here does not affect training
        desc_weaklysupervised, dst_weaklysupervised = 'weaklysupervised_%s' % args.id, get_weaklysupervised_dicts(args, args.image_batch_size * args.iters, mixup=True, background=[background_files, background_frame_idx])

        dst_cocotrain = get_coco_dicts(args, 'train')
        for im in dst_cocotrain:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
        random.shuffle(dst_cocotrain)
        dst_weaklysupervised = dst_weaklysupervised + dst_cocotrain[:len(dst_weaklysupervised) // 2]
        desc_weaklysupervised = desc_weaklysupervised + '_cocotrain'
        print('include MSCOCO2017 training images, totally %d images' % len(dst_weaklysupervised))
        for i in range(0, len(dst_weaklysupervised)):
            dst_weaklysupervised[i]['image_id'] = i + 1
        desc_weaklysupervised = desc_weaklysupervised + '_wdiff_midfusion'

    elif args.id == 'compound':
        desc_manual_valid, dst_manual_valid = 'compound_manual_wdiff_midfusion_mixup', []
        desc_weaklysupervised, dst_weaklysupervised = 'weaklysupervised_compound', []
        for v in video_id_list:
            args.id = v
            background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
            background_frame_idx = list(map(lambda x: os.path.basename(x), background_files))
            background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), background_frame_idx)))
            dst_manual_valid_v = get_annotation_dict(args)
            for im in dst_manual_valid_v:
                im['file_name_background'] = background_files[-1] # choice of background images here does not affect training
            dst_manual_valid = dst_manual_valid + dst_manual_valid_v
            dst_weaklysupervised_v = get_weaklysupervised_dicts(args, int(args.image_batch_size * args.iters / len(video_id_list)), mixup=True, background=[background_files, background_frame_idx])
            dst_weaklysupervised = dst_weaklysupervised + dst_weaklysupervised_v
        args.id = 'compound'
        print('compound validation set: %d images, %d bboxes' % (len(dst_manual_valid), sum(list(map(lambda x: len(x['annotations']), dst_manual_valid)))))

        dst_cocotrain = get_coco_dicts(args, 'train')
        for im in dst_cocotrain:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
        random.shuffle(dst_cocotrain)
        dst_weaklysupervised = dst_weaklysupervised + dst_cocotrain[:len(dst_weaklysupervised) // 2]
        desc_weaklysupervised = desc_weaklysupervised + '_cocotrain'
        print('include MSCOCO2017 training images, totally %d images' % len(dst_weaklysupervised))
        for i in range(0, len(dst_weaklysupervised)):
            dst_weaklysupervised[i]['image_id'] = i + 1
        for i in range(0, len(dst_manual_valid)):
            dst_manual_valid[i]['image_id'] = i + 1
        desc_weaklysupervised = desc_weaklysupervised + '_wdiff_midfusion'
    else:
        raise NotImplementedError
    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_weaklysupervised, lambda: dst_weaklysupervised)
    MetadataCatalog.get(desc_weaklysupervised).thing_classes = thing_classes

    cfg = get_cfg_base_model(args.model)
    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    print('loading checkpoint:', args.ckpt)
    cfg.MODEL.WEIGHTS = args.ckpt
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 4, args.iters // 2, args.iters * 3 // 4)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_weaklysupervised,)
    cfg.DATASETS.TEST = (desc_manual_valid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 100
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 100

    trainer = FinetuneBackgroundTrainer(cfg, args.fusion_type, args.multitask_loss_alpha)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackgroundMixup.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, args.mixup_p, args.mixup_r, args.mixup_overlap_thres)
    trainer.resume_or_load(resume=False)

    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt%s_%s_anno_%s%s_boxinpaint_mixup' % (args.id, args.model, desc_weaklysupervised, trainer.model.fusion_desc)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    dst_list = {desc_manual_valid: {'mAP': [], 'AP50': []}}
    for i in iter_list:
        dst_list[desc_manual_valid]['mAP'].append(aps[i]['bbox']['AP'])
        dst_list[desc_manual_valid]['AP50'].append(aps[i]['bbox']['AP50'])

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
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Manual Valid AP50', 'Manual Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.yticks(np.arange(0, 1.01, 0.1))
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
    plt.title('losses')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


def correlation():
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)['r101-fpn-3x']
    with open('F:\\intersections_results\\fusion_coco_box_inpaint\\weaklysupervised_boxinpaint_midfusion_mixup_r101\\results_AP_merge_dynamic.json', 'r') as fp:
        individual_AP = json.load(fp)
    with open('F:\\intersections_results\\fusion_coco_box_inpaint\\compound_weaklysupervised_boxinpaint_midfusion_mixup_r101\\results_AP_merge_dynamic.json', 'r') as fp:
        compound_AP = json.load(fp)
    with open('F:\\intersections_results\\fusion_coco_box_inpaint\\boxinpaint_midfusion_mixup_r101\\results_AP_merge_dynamic.json', 'r') as fp:
        individual_self_AP = json.load(fp)
    with open('F:\\intersections_results\\fusion_coco_box_inpaint\\compound_boxinpaint_midfusion_mixup_r101\\results_AP_merge_dynamic.json', 'r') as fp:
        compound_self_AP = json.load(fp)

    results = []
    for vid in tqdm.tqdm(video_id_list, ascii=True):
        inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', vid))
        with open(os.path.join(inputdir, 'weaksupervised.json'), 'r') as fp:
            images_weaksupervised = json.load(fp)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images_fullysupervised = json.load(fp)
        results.append({
            'v': vid,
            'weakly': sum(list(map(lambda x: len(x['annotations']), images_weaksupervised))) / len(images_weaksupervised),
            'fully': sum(list(map(lambda x: len(x['annotations']), images_fullysupervised))) / len(images_fullysupervised),
            'ap_base': base_AP['manual_' + vid]['results']['weighted'],
            'ap_individual': individual_AP[vid]['results']['weighted'],
            'ap_compound': compound_AP[vid]['results']['weighted'],
            'ap_individual_self': individual_self_AP[vid]['results']['weighted'],
            'ap_compound_self': compound_self_AP[vid]['results']['weighted'],
        })
    results = [{
        'v': r['v'], 'supervision': r['weakly'] / r['fully'],
        'mapg_individual': 100 * (r['ap_individual'][0] - r['ap_base'][0]),
        'mapg_compound': 100 * (r['ap_compound'][0] - r['ap_base'][0]),
        'mapg_individual_self': 100 * (r['ap_individual_self'][0] - r['ap_base'][0]),
        'mapg_compound_self': 100 * (r['ap_compound_self'][0] - r['ap_base'][0]),
    } for r in results]
    results.sort(key=lambda x: x['supervision'])

    xs = np.arange(0, len(results), 1)
    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    axes = axes.reshape(-1)
    ys = np.array([r['mapg_individual'] for r in results])
    axes[0].plot(xs, ys, 'r.-')
    axes[0].legend(['APG $\\mu=%.2f$' % ys.mean()], loc='upper left')
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([r['v'] for r in results], rotation='vertical', fontsize=10)
    axes[0].set_xlim(0, xs.max())
    axes[0].set_ylim(-25, 35)
    axes[0].set_ylabel('weighted $APG^m$ (0-100)')
    axes[0].grid(True)
    axes[0].set_title('individually adapted')
    axes_y2 = axes[0].twinx()
    axes_y2.plot(xs, [r['supervision'] for r in results], 'k-')
    axes_y2.set_ylim(0, 1.6)
    axes_y2.set_ylabel('supervision rate\n# weaklysupervised boxes / # fullysupervised boxes')
    axes_y2.legend(['supervision rate'], loc='lower right')

    ys = np.array([r['mapg_compound'] for r in results])
    axes[1].plot(xs, ys, 'r.-')
    axes[1].legend(['APG $\\mu=%.2f$' % ys.mean()], loc='upper left')
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([r['v'] for r in results], rotation='vertical', fontsize=10)
    axes[1].set_xlim(0, xs.max())
    axes[1].set_ylim(-25, 35)
    axes[1].set_ylabel('weighted $APG^m$ (0-100)')
    axes[1].grid(True)
    axes[1].set_title('compoundly adapted')
    axes_y2 = axes[1].twinx()
    axes_y2.plot(xs, [r['supervision'] for r in results], 'k-')
    axes_y2.set_ylim(0, 1.6)
    axes_y2.set_ylabel('supervision rate\n# weaklysupervised boxes / # fullysupervised boxes')
    axes_y2.legend(['supervision rate'], loc='lower right')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

    plt.figure(figsize=(26, 8))
    plt.subplot(1, 3, 1)
    plt.scatter([r['supervision'] for r in results], [r['mapg_individual'] for r in results], c='red', s=64, marker='x')
    plt.xlabel('supervision rate')
    plt.ylabel('APG individually adapted')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.scatter([r['supervision'] for r in results], [r['mapg_compound'] for r in results], c='red', s=64, marker='x')
    plt.xlabel('supervision rate')
    plt.ylabel('APG compoundly adapted')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.scatter([r['mapg_individual'] for r in results], [r['mapg_compound'] for r in results], c='red', s=64, marker='x')
    plt.xlabel('APG individually adapted')
    plt.ylabel('APG compoundly adapted')
    plt.grid(True)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.07)
    plt.show()

    plt.figure(figsize=(26, 8))
    plt.subplot(1, 3, 1)
    plt.scatter([r['mapg_individual'] for r in results], [r['mapg_individual_self'] for r in results], c='red', s=64, marker='x')
    plt.xlabel('APG individually adapted')
    plt.ylabel('APG individually adapted (self-supervised)')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.scatter([r['mapg_compound'] for r in results], [r['mapg_compound_self'] for r in results], c='red', s=64, marker='x')
    plt.xlabel('APG compoundly adapted')
    plt.ylabel('APG compoundly adapted (self-supervised)')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.scatter([r['mapg_individual_self'] for r in results], [r['mapg_compound_self'] for r in results], c='red', s=64, marker='x')
    plt.xlabel('APG individually adapted (self-supervised)')
    plt.ylabel('APG compoundly adapted (self-supervised)')
    plt.grid(True)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.07)
    plt.show()


if __name__ == '__main__':
    # correlation(); exit(0)
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--smallscale', default=False, type=bool)

    parser.add_argument('--mixup_p', type=float, default=0.5, help='probability of applying mixup to an image')
    parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')

    parser.add_argument('--fusion_type', type=str, choices=['average', 'conv', 'attn'], default='average', help='feature pyramids fusion method')
    parser.add_argument('--multitask_loss_alpha', type=float, default=0.5, help='relative weight of multitasking losses')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    assert 0 <= args.multitask_loss_alpha <= 1, str(args.multitask_loss_alpha)
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    adapt(args)


'''
python finetune_weaklysupervised.py --id 001 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --iters 1000 --eval_interval 201 --num_workers 4
python finetune_weaklysupervised.py --id compound --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 2001 --num_workers 4

for V in 001 003 005 006 007 008 009 011 012 013 014 015 016 017 019 020 023 025 027 034 036 039 040 043 044 046 048 049 050 051 053 054 055 056 058 059 060 066 067 068 069 070 071 073 074 075 076 077 080 085 086 087 088 090 091 092 093 094 095 098 099 105 108 110 112 114 115 116 117 118 125 127 128 129 130 131 132 135 136 141 146 148 149 150 152 154 156 158 159 160 161 164 167 169 170 171 172 175 178 179 ; do python finetune_weaklysupervised.py --id ${V} --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --iters 1000 --eval_interval 201 --num_workers 4 ; done

python compare_baselines.py --opt compare --model r101-fpn-3x --arch midfusion --fusion_type average --eval_background dynamic --compare_ckpt_dir /mnt/f/intersections_results/fusion_coco_box_inpaint/weaklysupervised_boxinpaint_midfusion_mixup_r101
python compare_baselines.py --opt compare_compound --model r101-fpn-3x --arch midfusion --fusion_type average --eval_background dynamic --compare_ckpt_dir /mnt/f/intersections_results/fusion_coco_box_inpaint/compound_weaklysupervised_boxinpaint_midfusion_mixup_r101
'''
