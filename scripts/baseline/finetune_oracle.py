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
from collections import OrderedDict
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io
import functools

import sklearn.utils
from sklearn.mixture import GaussianMixture

import torch
import torch.utils.data as torchdata

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
from finetune import get_annotation_dict
from finetune_wdiff_midfusion_mixup import FinetuneBackgroundTrainer, DatasetMapperBackgroundMixup, finetune_simple_trainer_run_step


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_oracle_output')


def get_oracle_splits_with_corruption(args):
    images = get_annotation_dict(args)
    for im in images:
        im['bbox_count_original'] = len(im['annotations'])
    assert len(images) >= 4
    n_train = min(max(len(images) // 3, 2), 5)
    print('training/validation: %d/%d' % (n_train, len(images) - n_train))
    images_train, images_valid = images[: n_train], images[n_train :]
    print('oracle training set:   %d images, %d bboxes' % (len(images_train), sum(list(map(lambda x: len(x['annotations']), images_train)))))
    print('oracle validation set: %d images, %d bboxes' % (len(images_valid), sum(list(map(lambda x: len(x['annotations']), images_valid)))))

    if args.fn_rate > 0:
        for im in tqdm.tqdm(images_train, ascii=True, desc='dropping %d%% true bboxes' % args.fn_rate):
            bboxes = im['annotations']
            random.shuffle(bboxes)
            im['annotations'] = bboxes[int(im['bbox_count_original'] * args.fn_rate / 100) :]
    if args.fp_rate > 0:
        for i, im in tqdm.tqdm(enumerate(images_train), total=len(images_train), ascii=True, desc='adding %d%% false bboxes' % args.fp_rate):
            # 2/3 false boxes from other images
            candidate_bboxes = [images_train[j] for j in filter(lambda x: x != i, range(0, len(images_train)))]
            candidate_bboxes = functools.reduce(lambda x, y: x + y, [im['annotations'] for im in candidate_bboxes])
            random.shuffle(candidate_bboxes)
            im['annotations'] = im['annotations'] + candidate_bboxes[: int(im['bbox_count_original'] * args.fp_rate * 0.667 / 100)]
            # 1/3 false boxes totally random
            for _ in range(0, int(im['bbox_count_original'] * args.fp_rate * 0.333 / 100)):
                x2, y2 = random.randint(20, im['width'] - 1), random.randint(20, im['height'] - 1)
                x1, y1 = random.randint(1, x2 - 10), random.randint(1, y2 - 10)
                im['annotations'].append({'bbox': [x1, y1, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': thing_classes.index('person')})
    print('after corruption: %d images, %d bboxes' % (len(images_train), sum(map(lambda ann: len(ann['annotations']), images_train))))
    return images_train, images_valid


def adapt(args):
    random.seed(42)
    desc_train, desc_valid = 'oracle_%strain_cocotrain' % args.id, 'oracle_%svalid' % args.id
    images_train, images_valid = get_oracle_splits_with_corruption(args)
    background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
    background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), list(map(lambda x: os.path.basename(x), background_files)))))
    for im in images_train:
        i = os.path.basename(im['file_name'])
        i = int(i[:i.find('.')])
        im['file_name_background'] = background_files[np.absolute(background_frame_idx - i).argmin()]
    for im in images_valid:
        im['file_name_background'] = background_files[-1] # choice of background images here does not affect training

    images_train = copy.deepcopy((images_train * (1 + 2000 // len(images_train)))[: 2000])
    assert len(images_train) == 2000
    images_train_copy = copy.deepcopy(images_train)
    for im in tqdm.tqdm(images_train, ascii=True, desc='populating mixup sources'):
        im['mixup_src_images'] = [images_train_copy[random.randrange(0, len(images_train))]]
    del images_train_copy

    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'train2017', os.path.basename(im['file_name'])))
    random.shuffle(dst_cocotrain)
    images_train = images_train + dst_cocotrain[: 2000]

    for i in range(0, len(images_train)):
        images_train[i]['image_id'] = i + 1
    for i in range(0, len(images_valid)):
        images_valid[i]['image_id'] = i + 1
    print('include MSCOCO2017 training images, totally %d images' % len(images_train))

    DatasetCatalog.register(desc_train, lambda: images_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid, lambda: images_valid)
    MetadataCatalog.get(desc_valid).thing_classes = thing_classes

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
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_valid,)
    # print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

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

    prefix = 'adapt%s_%s_oracle_FN%02d_FP%02d_cocotrain_midfusionmixup' % (args.id, args.model, args.fn_rate, args.fp_rate)
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    torch.save(trainer.model.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))

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
    plt.savefig(os.path.join(args.outputdir, prefix + '.pdf'))
    plt.close()


def adapt_all_videos(args):
    for video_id in video_id_list:
        args.id = video_id
        pth = os.path.join(args.outputdir, 'adapt%s_%s_oracle_FN%02d_FP%02d_cocotrain_midfusionmixup.pth' % (args.id, args.model, args.fn_rate, args.fp_rate))
        if os.access(pth, os.R_OK):
            print('skipped on existing:', pth)
            continue
        adapt(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, choices=['single', 'batch'], help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--outputdir', type=str, default=None)

    parser.add_argument('--cocodir', type=str)
    parser.add_argument('--smallscale', type=bool, default=False)

    parser.add_argument('--mixup_p', type=float, default=0.3)
    parser.add_argument('--mixup_r', type=float, default=0.5)
    parser.add_argument('--mixup_overlap_thres', type=float, default=0.65)
    parser.add_argument('--fusion_type', type=str, choices=['average', 'conv', 'attn'], default='average')
    parser.add_argument('--multitask_loss_alpha', type=float, default=0.5)

    parser.add_argument('--fp_rate', type=int, default=0)
    parser.add_argument('--fn_rate', type=int, default=0)

    parser.add_argument('--iters', type=int, default=1200)
    parser.add_argument('--eval_interval', type=int, default=150)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.access(args.outputdir, os.W_OK)
    assert args.fp_rate >= 0 and args.fp_rate < 100
    assert args.fn_rate >= 0 and args.fn_rate < 100

    if args.opt == 'single':
        adapt(args)
    elif args.opt == 'batch':
        adapt_all_videos(args)


'''
python finetune_oracle.py --opt single --id 001 --fp_rate 20 --fn_rate 20 --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --iters 200 --eval_interval 101 --image_batch_size 2 --num_workers 2 --outputdir .

python finetune_oracle.py --opt batch --model r101-fpn-3x --fp_rate 0 --fn_rate 0 --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --image_batch_size 4 --num_workers 4 --outputdir /mnt/f/intersections_results/oracle_r101/FN00_FP00

'''
