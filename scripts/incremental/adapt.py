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
import contextlib
import functools
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
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref

from annotate import get_annotation_dict, get_annotation_dict_training, calculate_AP
from pseudo_label import refine
from midfusion_mixup import get_midfusion_avg_detector, construct_image_w_background, DatasetMapperBackgroundMixup, get_midfusion_avg_trainer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_detector_train import get_coco_dicts


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
basedir = os.path.normpath(os.path.dirname(__file__))


def evaluate_model(args):
    assert args.model_iteration >= 0
    if args.model_iteration == 0: # base models
        cfg = get_cfg()
        if args.model == 'r50-fpn-3x':
            cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(basedir, 'models', 'mscoco2017_remap_r50-fpn-3x.pth')
        elif args.model == 'r101-fpn-3x':
            cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(basedir, 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
        else:
            raise NotImplementedError
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        detector = DefaultPredictor(cfg)
    else: # adapted models
        assert args.model == 'r101-fpn-3x-midfusion-mixup'
        cfg, detector = get_midfusion_avg_detector('r101-fpn-3x', args.ckpt)
    print('detectron2 model: %s iteration %d' % (args.model, args.model_iteration))
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    images_dense, images_sparse = get_annotation_dict()
    annotations = {'dense': images_dense, 'sparse': images_sparse}
    detections, results = {}, {}
    for density in ['dense', 'sparse']:
        detections[density] = copy.deepcopy(annotations[density])
        for i, im in tqdm.tqdm(enumerate(annotations[density]), total=len(annotations[density]), ascii=True, desc='detecting in %s' % density):
            if args.model_iteration == 0:
                im_arr = detectron2.data.detection_utils.read_image(im['file_name'], format='BGR')
                instances = detector(im_arr)['instances'].to('cpu')
            else:
                im_arr = detectron2.data.detection_utils.read_image(im['file_name'], format='BGR')
                im_background = detectron2.data.detection_utils.read_image(im['file_name_background'], format='BGR')
                im_arr, _, im_diff = construct_image_w_background(im_arr, im_background)
                instances = detector(im_arr, im_diff)['merge']['instances'].to('cpu')

            detections[density][i]['annotations'] = []
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            for k in range(0, len(label)):
                detections[density][i]['annotations'].append({'bbox': bbox[k], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[k], 'score': score[k]})

    for density in detections:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[density] = calculate_AP(detections[density], density, annotations=annotations[density])
    results_json = os.path.join(basedir, 'results.json')
    if os.access(results_json, os.R_OK):
        with open(results_json, 'r') as fp:
            results_all = json.load(fp)
    else:
        results_all = {}
    results_all[os.path.basename(cfg.MODEL.WEIGHTS)] = results
    assert 'mscoco2017_remap_r101-fpn-3x.pth' in results_all, 'base model needs evaluation first'
    with open(results_json, 'w') as fp:
        json.dump(results_all, fp)
    print(results)
    for density in results:
        print('\n            %s' % density)
        print(   '             %s' % '/'.join(results[density]['metrics']))
        for c in sorted(results[density]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[density]['results'][c])))

    R1, R2 = results_all['mscoco2017_remap_r101-fpn-3x.pth'], results
    gains = (
        (R2['dense']['results']['person'][0] - R1['dense']['results']['person'][0]) * 100,
        (R2['dense']['results']['person'][1] - R1['dense']['results']['person'][1]) * 100,
        (R2['sparse']['results']['person'][0] - R1['sparse']['results']['person'][0]) * 100,
        (R2['sparse']['results']['person'][1] - R1['sparse']['results']['person'][1]) * 100
    )
    print('\nperson AP improvements mAP/AP50:\ndense:  %.2f/%.2f\nsparse: %.2f/%.2f\n' % gains)
    print(list(gains))
    return gains


def show_results(args, prefix, dataset_desc_list, aps, lr_history, loss_history):
    iter_list = sorted(list(aps.keys()))
    assert len(dataset_desc_list) == 2
    desc_valid_dense, desc_valid_sparse = dataset_desc_list
    dataset_desc_list = {k: [] for k in dataset_desc_list}
    for i in iter_list:
        for k in dataset_desc_list:
            dataset_desc_list[k].append(aps[i][k]['bbox']['AP-person'])
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
    plt.plot(iter_list, np.array(dataset_desc_list[desc_valid_sparse]) / 100, linestyle='--', marker='+', color='#0000FF')
    plt.plot(iter_list, np.array(dataset_desc_list[desc_valid_dense]) / 100, linestyle='-', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Manual Valid Sprase mAP (person)', 'Manual Valid Dense mAP (person)'])
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
    plt.savefig(os.path.join(basedir, prefix + args.tag + '.pdf'))


def adapt_oracle_partial(args):
    assert args.model == 'r101-fpn-3x-midfusion-mixup'
    adapt_output = os.path.join(basedir, 'adapt_output')
    random.seed(42)
    desc_valid_dense, desc_valid_sparse = 'manual_dense', 'manual_sparse'
    dst_valid_dense, dst_valid_sparse = get_annotation_dict(mask='blank')
    dst_train_oracle = get_annotation_dict_training()
    if not args.day1:
        dst_train_oracle = functools.reduce(lambda x, y: x + y, dst_train_oracle) # use all dates images for partial training
        print('manual training images of days 0 1 2 3: %d images, %d bboxes' % (len(dst_train_oracle), sum(map(lambda ann: len(ann['annotations']), dst_train_oracle))))
    else:
        dst_train_oracle = dst_train_oracle[1]
        print('manual training images of days 1: %d images, %d bboxes' % (len(dst_train_oracle), sum(map(lambda ann: len(ann['annotations']), dst_train_oracle))))
    for im in dst_train_oracle:
        im['bbox_count_original'] = len(im['annotations'])

    if args.fn_rate > 0:
        for im in tqdm.tqdm(dst_train_oracle, ascii=True, desc='dropping %d%% true bboxes' % args.fn_rate):
            bboxes = im['annotations']
            random.shuffle(bboxes)
            im['annotations'] = bboxes[int(im['bbox_count_original'] * args.fn_rate / 100) :]
    if args.fp_rate > 0:
        for i, im in tqdm.tqdm(enumerate(dst_train_oracle), total=len(dst_train_oracle), ascii=True, desc='adding %d%% false bboxes' % args.fp_rate):
            # 2/3 false boxes from other images
            candidate_bboxes = [dst_train_oracle[j] for j in filter(lambda x: x != i, range(0, len(dst_train_oracle)))]
            candidate_bboxes = functools.reduce(lambda x, y: x + y, [im['annotations'] for im in candidate_bboxes])
            random.shuffle(candidate_bboxes)
            im['annotations'] = im['annotations'] + candidate_bboxes[: int(im['bbox_count_original'] * args.fp_rate * 0.667 / 100)]
            # 1/3 false boxes totally random
            for _ in range(0, int(im['bbox_count_original'] * args.fp_rate * 0.333 / 100)):
                x2, y2 = random.randint(20, im['width'] - 1), random.randint(20, im['height'] - 1)
                x1, y1 = random.randint(1, x2 - 10), random.randint(1, y2 - 10)
                im['annotations'].append({'bbox': [x1, y1, x2, y2], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': thing_classes.index('person')})
    print('after corruption: %d images, %d bboxes' % (len(dst_train_oracle), sum(map(lambda ann: len(ann['annotations']), dst_train_oracle))))
    if not args.day1:
        desc_manual_anno = 'manual_anno_FN%02d_FP%02d_cocotrain' % (args.fn_rate, args.fp_rate)
    else:
        desc_manual_anno = 'manual_anno_day1_FN%02d_FP%02d_cocotrain' % (args.fn_rate, args.fp_rate)

    dst_train_oracle = (dst_train_oracle * int(5000 / len(dst_train_oracle)))[: 4000]
    dst_train_oracle_copy = copy.deepcopy(dst_train_oracle)
    for im in tqdm.tqdm(dst_train_oracle, ascii=True, desc='populating mixup sources'):
        im['mixup_src_images'] = [dst_train_oracle_copy[random.randrange(0, len(dst_train_oracle_copy))]]
    del dst_train_oracle_copy
    dst_cocotrain = get_coco_dicts(args, 'train')
    random.shuffle(dst_cocotrain)
    dst_cocotrain = dst_cocotrain[: 2000]
    for im in dst_cocotrain:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
    dst_train_oracle = dst_train_oracle + dst_cocotrain
    print('include MSCOCO2017 training images, totally %d images' % len(dst_train_oracle))
    del dst_cocotrain
    for dst in (dst_valid_dense, dst_valid_sparse, dst_train_oracle):
        for i in range(0, len(dst)):
            dst[i]['image_id'] = i + 1

    DatasetCatalog.register(desc_valid_dense, lambda: dst_valid_dense)
    MetadataCatalog.get(desc_valid_dense).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid_sparse, lambda: dst_valid_sparse)
    MetadataCatalog.get(desc_valid_sparse).thing_classes = thing_classes
    DatasetCatalog.register(desc_manual_anno, lambda: dst_train_oracle)
    MetadataCatalog.get(desc_manual_anno).thing_classes = thing_classes

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 100
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 100
    _, trainer = get_midfusion_avg_trainer(
        'r101-fpn-3x', args.ckpt,
        args.num_workers, adapt_output, args.lr, args.image_batch_size, args.roi_batch_size,
        {'warmup': 100, 'gamma': 0.5, 'steps': (250, 500, 750), 'total': 1000, 'eval_interval': 201},
        {'train': (desc_manual_anno,), 'eval': (desc_valid_dense, desc_valid_sparse)}
    )
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt_%s_%s' % (args.model, desc_manual_anno)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(basedir, prefix + args.tag + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(basedir, prefix + args.tag + '.pth'))
    show_results(args, prefix, [desc_valid_dense, desc_valid_sparse], trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history)


def adapt_oracle_distort(args):
    def _overlap(bbox1, bbox2):
        x11, y11, x12, y12 = bbox1
        x21, y21, x22, y22 = bbox2
        xA, yA = max(x11,x21), max(y11,y21)
        xB, yB = min(x12,x22), min(y12,y22)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        return max(xB - xA, 0) * max(yB - yA, 0) > 4

    assert args.model == 'r101-fpn-3x-midfusion-mixup'
    adapt_output = os.path.join(basedir, 'adapt_output')
    random.seed(42)
    desc_valid_dense, desc_valid_sparse = 'manual_dense', 'manual_sparse'
    dst_valid_dense, dst_valid_sparse = get_annotation_dict(mask='blank')
    dst_train_oracle = get_annotation_dict_training()
    if not args.day1:
        dst_train_oracle = functools.reduce(lambda x, y: x + y, dst_train_oracle) # use all dates images for partial training
        print('manual training images of days 0 1 2 3: %d images, %d bboxes' % (len(dst_train_oracle), sum(map(lambda ann: len(ann['annotations']), dst_train_oracle))))
    else:
        dst_train_oracle = dst_train_oracle[1]
        print('manual training images of days 1: %d images, %d bboxes' % (len(dst_train_oracle), sum(map(lambda ann: len(ann['annotations']), dst_train_oracle))))
    for im in dst_train_oracle:
        for ann in im['annotations']:
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS

    if args.de_occlusion > 0:
        for im in dst_train_oracle:
            for i, ann_i in enumerate(im['annotations']):
                x1, y1, x2, y2 = ann_i['bbox']
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                x1_occ, y1_occ, x2_occ, y2_occ = copy.deepcopy(ann_i['bbox'])
                for j, ann_j in enumerate(im['annotations']):
                    if j == i: continue
                    if not _overlap(ann_i['bbox'], ann_j['bbox']): continue
                    x1_j, y1_j, x2_j, y2_j = ann_j['bbox']
                    if x2_j > x1 and x2_j < xc:
                        x1_occ = max(x1_occ, x2_j)
                    if y2_j > y1 and y2_j < yc:
                        y1_occ = max(y1_occ, y2_j)
                    if x1_j < x2 and x1_j > xc:
                        x2_occ = min(x2_occ, x1_j)
                    if y1_j < y2 and y1_j > yc:
                        y2_occ = min(y2_occ, y1_j)
                ann_i['bbox_shrink'] = [x1_occ, y1_occ, x2_occ, y2_occ]
            for ann_i in im['annotations']:
                x1, y1, x2, y2 = ann_i['bbox']
                [x1_occ, y1_occ, x2_occ, y2_occ] = ann_i['bbox_shrink']
                r = args.de_occlusion / 100
                ann_i['bbox'] = [x1 + r * (x1_occ - x1), y1 + r * (y1_occ - y1), x2 - r * (x2 - x2_occ), y2 - r * (y2 - y2_occ)]
    if args.perturbation > 0:
        from scipy.stats import truncnorm
        for im in dst_train_oracle:
            for ann in im['annotations']:
                x1, y1, x2, y2 = ann['bbox']
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                x_std, y_std = args.perturbation / 100 * w / 6, args.perturbation / 100 * h / 6
                ann['bbox'] = list(map(float, [
                    truncnorm.rvs((0 - x1) / x_std, (xc - w * 0.01 - x1) / x_std, loc=x1, scale=x_std),
                    truncnorm.rvs((0 - y1) / y_std, (yc - h * 0.01 - y1) / y_std, loc=y1, scale=y_std),
                    truncnorm.rvs((xc + w * 0.01 - x2) / x_std, (im['width'] - x2) / x_std, loc=x2, scale=x_std),
                    truncnorm.rvs((yc + h * 0.01 - y2) / y_std, (im['height'] - y2) / y_std, loc=y2, scale=y_std)
                ]))
    if not args.day1:
        desc_manual_anno = 'manual_anno_DeOcc%02d_Perturb%02d_cocotrain' % (args.de_occlusion, args.perturbation)
    else:
        desc_manual_anno = 'manual_anno_day1_DeOcc%02d_Perturb%02d_cocotrain' % (args.de_occlusion, args.perturbation)
    # from annotate import draw_bbox
    # writer = skvideo.io.FFmpegWriter(os.path.join(basedir, desc_manual_anno + '.mp4'), inputdict={'-r': '1'}, outputdict={'-vcodec': 'libx265', '-r': '1', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
    # for im in tqdm.tqdm(dst_train_oracle, ascii=True, desc='writing video'):
    #     im_arr = skimage.io.imread(im['file_name'])
    #     im_arr = draw_bbox(im_arr, im['annotations'], im['file_name'])
    #     writer.writeFrame(im_arr)
    # writer.close()
    # return

    dst_train_oracle = (dst_train_oracle * int(5000 / len(dst_train_oracle)))[: 4000]
    dst_train_oracle_copy = copy.deepcopy(dst_train_oracle)
    for im in tqdm.tqdm(dst_train_oracle, ascii=True, desc='populating mixup sources'):
        im['mixup_src_images'] = [dst_train_oracle_copy[random.randrange(0, len(dst_train_oracle_copy))]]
    del dst_train_oracle_copy
    dst_cocotrain = get_coco_dicts(args, 'train')
    random.shuffle(dst_cocotrain)
    dst_cocotrain = dst_cocotrain[: 2000]
    for im in dst_cocotrain:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
    dst_train_oracle = dst_train_oracle + dst_cocotrain
    print('include MSCOCO2017 training images, totally %d images' % len(dst_train_oracle))
    del dst_cocotrain
    for dst in (dst_valid_dense, dst_valid_sparse, dst_train_oracle):
        for i in range(0, len(dst)):
            dst[i]['image_id'] = i + 1

    DatasetCatalog.register(desc_valid_dense, lambda: dst_valid_dense)
    MetadataCatalog.get(desc_valid_dense).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid_sparse, lambda: dst_valid_sparse)
    MetadataCatalog.get(desc_valid_sparse).thing_classes = thing_classes
    DatasetCatalog.register(desc_manual_anno, lambda: dst_train_oracle)
    MetadataCatalog.get(desc_manual_anno).thing_classes = thing_classes

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 100
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 100
    _, trainer = get_midfusion_avg_trainer(
        'r101-fpn-3x', args.ckpt,
        args.num_workers, adapt_output, args.lr, args.image_batch_size, args.roi_batch_size,
        {'warmup': 100, 'gamma': 0.5, 'steps': (250, 500, 750), 'total': 1000, 'eval_interval': 201},
        {'train': (desc_manual_anno,), 'eval': (desc_valid_dense, desc_valid_sparse)}
    )
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt_%s_%s' % (args.model, desc_manual_anno)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(basedir, prefix + args.tag + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(basedir, prefix + args.tag + '.pth'))
    show_results(args, prefix, [desc_valid_dense, desc_valid_sparse], trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history)


def adapt(args):
    assert args.model == 'r101-fpn-3x-midfusion-mixup'
    adapt_output = os.path.join(basedir, 'adapt_output')
    random.seed(42)
    desc_valid_dense, desc_valid_sparse = 'manual_dense', 'manual_sparse'
    dst_valid_dense, dst_valid_sparse = get_annotation_dict(mask='blank')

    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        day_lists = json.load(fp)['days']
    assert min(args.anno_day) >= 0 and max(args.anno_day) < len(day_lists) - 1
    desc_pseudo_anno = 'refined_anno_days_%s_anno_iters_%s' % ('_'.join(map(str, args.anno_day)), '_'.join(map(str, args.anno_iteration)))
    dst_pseudo_anno, vfilelist = [], []
    for d in args.anno_day:
        dst_pseudo_anno_d, vfilelist_d = refine(d, args.anno_iteration)
        dst_pseudo_anno, vfilelist = dst_pseudo_anno + dst_pseudo_anno_d, vfilelist + vfilelist_d
    print('pseudo labeling of [%s], %d videos:\n%s' % (desc_pseudo_anno, len(vfilelist), '\n'.join(vfilelist)))
    for v, vfilename in enumerate(vfilelist):
        for chunk in dst_pseudo_anno[v]:
            prefix = os.path.basename(chunk[-1]['file_name'])[: -4]
            for im in chunk:
                im['file_name_background'] = os.path.join(basedir, 'frames', 'background', '%s.%s.inpaint.jpg' % (vfilename, prefix))
    dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
    dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
    if args.fn_mining_num > 0:
        from false_negative_mining import sample_false_negatives
        print('include mined false negatives')
        dst_fn_mine = []
        for d in args.anno_day:
            dst_fn_mine = dst_fn_mine + sample_false_negatives(d, obj_score_thres=args.fn_mining_thres, max_sample_per_image=args.fn_mining_num)
        assert len(dst_pseudo_anno) == len(dst_fn_mine)
        for i in range(0, len(dst_pseudo_anno)):
            assert dst_pseudo_anno[i]['file_name'] == dst_fn_mine[i]['file_name']
            dst_pseudo_anno[i]['annotations'] = dst_pseudo_anno[i]['annotations'] + dst_fn_mine[i]['annotations']
        desc_pseudo_anno = desc_pseudo_anno + '_fn%d_thres%.3f' % (args.fn_mining_num, args.fn_mining_thres)

    num_images_cap = args.iters * args.image_batch_size
    if num_images_cap < len(dst_pseudo_anno):
        print('capping # of images according to training schedule: %d => %d' % (len(dst_pseudo_anno), num_images_cap))
        random.shuffle(dst_pseudo_anno)
        dst_pseudo_anno = dst_pseudo_anno[: num_images_cap]
    dst_pseudo_anno_copy = copy.deepcopy(dst_pseudo_anno)
    for im in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources'):
        im['mixup_src_images'] = [dst_pseudo_anno_copy[random.randrange(0, len(dst_pseudo_anno_copy))]]
    del dst_pseudo_anno_copy

    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
    while len(dst_cocotrain) < len(dst_pseudo_anno):
        dst_cocotrain = dst_cocotrain + dst_cocotrain
    random.shuffle(dst_cocotrain)
    dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: int(0.5 * len(dst_pseudo_anno))]
    desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
    print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))

    for dst in (dst_valid_dense, dst_valid_sparse, dst_pseudo_anno):
        for i in range(0, len(dst)):
            dst[i]['image_id'] = i + 1

    DatasetCatalog.register(desc_valid_dense, lambda: dst_valid_dense)
    MetadataCatalog.get(desc_valid_dense).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid_sparse, lambda: dst_valid_sparse)
    MetadataCatalog.get(desc_valid_sparse).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 100
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 100
    _, trainer = get_midfusion_avg_trainer(
        'r101-fpn-3x', args.ckpt,
        args.num_workers, adapt_output, args.lr, args.image_batch_size, args.roi_batch_size,
        {
            'warmup': min(args.iters // 10, 2000),
            'gamma': 0.5,
            'steps': (args.iters // 4, args.iters // 2, args.iters * 3 // 4),
            'total': args.iters,
            'eval_interval': args.eval_interval
        },
        {
            'train': (desc_pseudo_anno,),
            'eval': (desc_valid_dense, desc_valid_sparse)
        }
    )

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt_%s_%s' % (args.model, desc_pseudo_anno)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(basedir, prefix + args.tag + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(basedir, prefix + args.tag + '.pth'))
    show_results(args, prefix, [desc_valid_dense, desc_valid_sparse], trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history)


def adapt_semi(args):
    assert args.model == 'r101-fpn-3x-midfusion-mixup'
    adapt_output = os.path.join(basedir, 'adapt_output')
    random.seed(42)
    desc_valid_dense, desc_valid_sparse = 'manual_dense', 'manual_sparse'
    dst_valid_dense, dst_valid_sparse = get_annotation_dict(mask='blank')

    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        day_lists = json.load(fp)['days']
    assert min(args.anno_day) >= 0 and max(args.anno_day) < len(day_lists) - 1
    desc_pseudo_anno = 'refined_anno_days_%s_anno_iters_%s' % ('_'.join(map(str, args.anno_day)), '_'.join(map(str, args.anno_iteration)))
    dst_pseudo_anno, vfilelist = [], []
    for d in args.anno_day:
        dst_pseudo_anno_d, vfilelist_d = refine(d, args.anno_iteration)
        dst_pseudo_anno, vfilelist = dst_pseudo_anno + dst_pseudo_anno_d, vfilelist + vfilelist_d
    print('pseudo labeling of [%s], %d videos:\n%s' % (desc_pseudo_anno, len(vfilelist), '\n'.join(vfilelist)))
    for v, vfilename in enumerate(vfilelist):
        for chunk in dst_pseudo_anno[v]:
            prefix = os.path.basename(chunk[-1]['file_name'])[: -4]
            for im in chunk:
                im['file_name_background'] = os.path.join(basedir, 'frames', 'background', '%s.%s.inpaint.jpg' % (vfilename, prefix))
    dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
    dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
    num_images_cap = args.iters * args.image_batch_size
    if num_images_cap < len(dst_pseudo_anno):
        print('capping # of images according to training schedule: %d => %d' % (len(dst_pseudo_anno), num_images_cap))
        random.shuffle(dst_pseudo_anno)
        dst_pseudo_anno = dst_pseudo_anno[: num_images_cap]
    dst_pseudo_anno_copy = copy.deepcopy(dst_pseudo_anno)
    for im in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources pseudo -> pseudo'):
        im['mixup_src_images'] = [dst_pseudo_anno_copy[random.randrange(0, len(dst_pseudo_anno_copy))]]
    del dst_pseudo_anno_copy

    dst_train_oracle = get_annotation_dict_training()
    dst_train_oracle = functools.reduce(lambda x, y: x + y, [dst_train_oracle[i] for i in args.anno_day])
    print('manual training images of days %s: %d images, %d bboxes' % (' '.join(map(str, args.anno_day)), len(dst_train_oracle), sum(map(lambda ann: len(ann['annotations']), dst_train_oracle))))
    for im in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources manual -> pseudo'):
        im['mixup_src_images'].append(copy.deepcopy(dst_train_oracle[random.randrange(0, len(dst_train_oracle))]))
    dst_train_oracle = (dst_train_oracle * int(1.1 * len(dst_pseudo_anno) / len(dst_train_oracle)))[: len(dst_pseudo_anno)]
    dst_train_oracle_copy = copy.deepcopy(dst_train_oracle)
    for im in tqdm.tqdm(dst_train_oracle, ascii=True, desc='populating mixup sources manual -> manual'):
        im['mixup_src_images'] = [dst_train_oracle_copy[random.randrange(0, len(dst_train_oracle_copy))]]
    del dst_train_oracle_copy

    dst_cocotrain = get_coco_dicts(args, 'train')
    for im in dst_cocotrain:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
    while len(dst_cocotrain) < len(dst_pseudo_anno):
        dst_cocotrain = dst_cocotrain + dst_cocotrain
    random.shuffle(dst_cocotrain)
    dst_pseudo_anno = dst_pseudo_anno + dst_train_oracle + dst_cocotrain[: len(dst_pseudo_anno)]
    desc_pseudo_anno = desc_pseudo_anno + '_manual_cocotrain'
    print('include MSCOCO2017 & manual annotated training images, totally %d images' % len(dst_pseudo_anno))

    for dst in (dst_valid_dense, dst_valid_sparse, dst_pseudo_anno):
        for i in range(0, len(dst)):
            dst[i]['image_id'] = i + 1

    DatasetCatalog.register(desc_valid_dense, lambda: dst_valid_dense)
    MetadataCatalog.get(desc_valid_dense).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid_sparse, lambda: dst_valid_sparse)
    MetadataCatalog.get(desc_valid_sparse).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 100
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 100
    _, trainer = get_midfusion_avg_trainer(
        'r101-fpn-3x', args.ckpt,
        args.num_workers, adapt_output, args.lr, args.image_batch_size, args.roi_batch_size,
        {
            'warmup': min(args.iters // 10, 2000),
            'gamma': 0.5,
            'steps': (args.iters // 4, args.iters // 2, args.iters * 3 // 4),
            'total': args.iters,
            'eval_interval': args.eval_interval
        },
        {
            'train': (desc_pseudo_anno,),
            'eval': (desc_valid_dense, desc_valid_sparse)
        }
    )

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt_%s_%s' % (args.model, desc_pseudo_anno)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(basedir, prefix + args.tag + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history, 'args': vars(args)}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(basedir, prefix + args.tag + '.pth'))
    show_results(args, prefix, [desc_valid_dense, desc_valid_sparse], trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history)


def plot_results():
    metrics = ['dense-$AP^m$', 'dense-$AP^{50}$', 'sparse-$AP^m$', 'sparse-$AP^{50}$']
    # ap_stds = [0.106, 0.042, 0.152, 0.122]
    ap_days = {
        'self':   np.array([[2.97, 3.26, 0.92, 0.20], [2.82, 3.21, 0.52, 0.29], [1.80, 1.49, 0.03, 0.06], [2.20, 2.57, 1.38, 1.42]]),
        'oracle': np.array([[3.68, 3.83, 3.28, 2.05], [3.78, 2.65, 4.77, 1.61], [5.62, 5.09, 3.56, 1.50], [4.68, 4.08, 4.71, 1.51]]),
        'semi':   np.array([[3.14, 2.42, 3.11, 0.61], [3.54, 3.15, 2.96, 0.06], [3.62, 1.65, 3.46, 0.24], [2.91, 2.59, 3.25, 1.32]])
    }
    ap_cumul = {
        'self':   np.array([[2.97, 3.26, 0.92, 0.20], [2.84, 3.16, 0.80, 0.52], [1.81, 1.44, 0.18, -0.08], [2.48, 2.50, 1.11, 1.17]]),
        'oracle': np.array([[3.68, 3.83, 3.28, 2.05], [4.80, 4.06, 4.79, 2.14], [5.84, 4.77, 4.18, 2.28], [6.15, 4.92, 5.18, 2.37]]),
        'semi':   np.array([[3.14, 2.42, 3.11, 0.61], [3.77, 3.33, 3.23, 0.32], [4.34, 2.62, 3.35, 1.34], [4.95, 4.47, 4.58, 1.63]])
    }
    plt.figure(figsize=(12, 4))
    for i, (desc, key) in enumerate([('self-supervised', 'self'), ('supervised', 'oracle'), ('combined', 'semi')]):
        # aps = ap_days[key]
        aps = ap_cumul[key]
        plt.subplot(1, 3, i + 1)
        plt.plot(np.arange(0, 4) - 0.02, aps[:, 0], 'b-')
        plt.plot(np.arange(0, 4) + 0.02, aps[:, 1], 'r-')
        plt.plot(np.arange(0, 4) - 0.02, aps[:, 2], 'b--')
        plt.plot(np.arange(0, 4) + 0.02, aps[:, 3], 'r--')
        plt.ylim(-0.1, 6.5)
        plt.xlim(-0.2, 3.2)
        plt.xticks(np.arange(0, 4))
        plt.legend(metrics)
        plt.grid(True)
        plt.title(desc)
    plt.tight_layout()
    plt.show()


def plot_results_2():
    # class ARGS(object): pass
    # results = []
    # for fn in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #     for fp in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #         args = ARGS()
    #         args.model, args.model_iteration = 'r101-fpn-3x-midfusion-mixup', 1
    #         # args.ckpt = '/mnt/f/intersections_results/incremental/corruption_day0123/adapt_r101-fpn-3x-midfusion-mixup_manual_anno_FN%02d_FP%02d_cocotrain.pth' % (fn, fp)
    #         args.ckpt = '/mnt/f/intersections_results/incremental/corruption_day1/adapt_r101-fpn-3x-midfusion-mixup_manual_anno_day1_FN%02d_FP%02d_cocotrain.pth' % (fn, fp)
    #         results.append({'fn': fn, 'fp': fp, 'apg': evaluate_model(args)})
    # with open(os.path.join(basedir, 'corruption_results.json'), 'w') as fp:
    #     json.dump(results, fp)
    # return

    from matplotlib.colors import ListedColormap
    with open('corruption_results_day0123.json', 'r') as fp:
    # with open('corruption_results_day1.json', 'r') as fp:
        results = json.load(fp)
    thres = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # apg_idx, ap_thres = 0, 2.08
    apg_idx, ap_thres = 2, 0.98
    apg_m_dense = np.ones((len(thres), len(thres)), dtype=np.float32) * -100
    apg_m_dense_color = np.zeros_like(apg_m_dense).astype(np.int32)
    for r in results:
        i, j = thres.index(r['fn']), thres.index(r['fp'])
        apg_m_dense[i, j] = r['apg'][apg_idx]
        if r['apg'][apg_idx] <= 0:
            apg_m_dense_color[i, j] = 0
        elif r['apg'][apg_idx] <= ap_thres:
            apg_m_dense_color[i, j] = 1
        else:
            apg_m_dense_color[i, j] = 2
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(apg_m_dense_color, cmap=ListedColormap([(0.5, 0.5, 0.5), (0.2, 0.5, 0.2), (0.5, 0.2, 0.2)]))
    ax.set_xlabel('false positive rate (%)')
    ax.set_xticks(np.arange(0, len(thres)))
    ax.set_xticklabels(thres)
    ax.set_ylabel('false negative rate (%)')
    ax.set_yticks(np.arange(0, len(thres)))
    ax.set_yticklabels(thres)
    for (i, j), z in np.ndenumerate(apg_m_dense):
        ax.text(j, i, '%.2f' % z, ha='center', va='center', color='white')
    plt.tight_layout()
    plt.show()


def plot_results_3():
    # class ARGS(object): pass
    # results = []
    # for dc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #     for pt in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    #         args = ARGS()
    #         args.model, args.model_iteration = 'r101-fpn-3x-midfusion-mixup', 1
    #         args.ckpt = '/mnt/f/intersections_results/incremental/distortion_day0123/adapt_r101-fpn-3x-midfusion-mixup_manual_anno_DeOcc%02d_Perturb%02d_cocotrain.pth' % (dc, pt)
    #         results.append({'de_occlusion': dc, 'perturbation': pt, 'apg': evaluate_model(args)})
    # with open(os.path.join(basedir, 'distortion_results.json'), 'w') as fp:
    #     json.dump(results, fp)
    # return

    from matplotlib.colors import ListedColormap
    with open('distortion_results_day0123.json', 'r') as fp:
        results = json.load(fp)
    thres = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    apg_idx, ap_thres = 0, 2.08
    # apg_idx, ap_thres = 2, 0.98
    apg_m_dense = np.ones((len(thres), len(thres)), dtype=np.float32) * -100
    apg_m_dense_color = np.zeros_like(apg_m_dense).astype(np.int32)
    for r in results:
        i, j = thres.index(r['de_occlusion']), thres.index(r['perturbation'])
        apg_m_dense[i, j] = r['apg'][apg_idx]
        if r['apg'][apg_idx] <= 0:
            apg_m_dense_color[i, j] = 0
        elif r['apg'][apg_idx] <= ap_thres:
            apg_m_dense_color[i, j] = 1
        else:
            apg_m_dense_color[i, j] = 2
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(apg_m_dense_color, cmap=ListedColormap([(0.5, 0.5, 0.5), (0.2, 0.5, 0.2), (0.5, 0.2, 0.2)]))
    ax.set_xlabel('Random Perturbation Strength (0-100)')
    ax.set_xticks(np.arange(0, len(thres)))
    ax.set_xticklabels(thres)
    ax.set_ylabel('De-Occlusion Strength (0-100)')
    ax.set_yticks(np.arange(0, len(thres)))
    ax.set_yticklabels(thres)
    for (i, j), z in np.ndenumerate(apg_m_dense):
        ax.text(j, i, '%.2f' % z, ha='center', va='center', color='white')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_results(); exit()
    # plot_results_2(); exit()
    # plot_results_3(); exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['adapt', 'partial', 'distort', 'semi', 'eval'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--model_iteration', type=int)
    parser.add_argument('--oracle_single_image', type=str, default='', choices=['', 'dense', 'sparse'])
    parser.add_argument('--tag', type=str, default='')

    parser.add_argument('--anno_day', nargs='+', default=[])
    parser.add_argument('--anno_iteration', nargs='+', default=[])
    parser.add_argument('--fp_rate', type=int, default=0)
    parser.add_argument('--fn_rate', type=int, default=0)
    parser.add_argument('--de_occlusion', type=int, default=0)
    parser.add_argument('--perturbation', type=int, default=0)
    parser.add_argument('--day1', type=bool, default=False)

    parser.add_argument('--fn_mining_num', type=int, default=-1)
    parser.add_argument('--fn_mining_thres', type=float, default=0.99)

    parser.add_argument('--cocodir', type=str)
    parser.add_argument('--smallscale', type=bool, default=False)

    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--eval_interval', type=int, default=1800)
    parser.add_argument('--image_batch_size', type=int, default=4)
    parser.add_argument('--roi_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    args.anno_day = sorted(list(map(int, set(args.anno_day))))
    args.anno_iteration = sorted(list(map(int, set(args.anno_iteration))))
    assert args.fp_rate >= 0 and args.fp_rate <= 100
    assert args.fn_rate >= 0 and args.fn_rate <= 100
    assert args.de_occlusion >= 0 and args.de_occlusion <= 100
    assert args.perturbation >= 0 and args.perturbation <= 100
    print(args)

    if args.opt == 'eval':
        evaluate_model(args)
    if args.opt == 'partial':
        adapt_oracle_partial(args)
    if args.opt == 'distort':
        adapt_oracle_distort(args)
    if args.opt == 'adapt':
        adapt(args)
    if args.opt == 'semi':
        adapt_semi(args)


'''
python adapt.py --opt eval --model r50-fpn-3x --model_iteration 0
python adapt.py --opt eval --model r101-fpn-3x-midfusion-mixup --model_iteration 1 --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth

############ test
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --anno_day 0 --anno_iteration 0 --cocodir ../../../MSCOCO2017 --iters 200 --eval_interval 101 --image_batch_size 2

############ partial
python adapt.py --opt partial --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 4 --fn_rate 0 --fp_rate 0
python adapt.py --opt partial --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 4 --fn_rate 50 --fp_rate 50 --day1 1

for X in 00 10 20 30 40 50 60 70 80 90 ; do \
    for Y in 00 10 20 30 40 50 60 70 80 90 ; do \
        [ -f "adapt_r101-fpn-3x-midfusion-mixup_manual_anno_FN${X}_FP${Y}_cocotrain.pth" ] || python adapt.py --opt partial --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 4 --fn_rate ${X} --fp_rate ${Y}; \
        rm -r __pycache__ adapt_output; \
    done; \
done

for X in 00 10 20 30 40 50 60 70 80 90 ; do \
    for Y in 00 10 20 30 40 50 60 70 80 90 ; do \
        [ -f "adapt_r101-fpn-3x-midfusion-mixup_manual_anno_day1_FN${X}_FP${Y}_cocotrain.pth" ] || python adapt.py --opt partial --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 4 --fn_rate ${X} --fp_rate ${Y} --day1 1; \
        rm -r __pycache__ adapt_output; \
    done; \
done

############ distort
python adapt.py --opt distort --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 4 --de_occlusion 50 --perturbation 10

for X in 00 10 20 30 40 50 60 70 80 90 ; do \
    for Y in 00 10 20 30 40 50 60 70 80 90 ; do \
        [ -f "adapt_r101-fpn-3x-midfusion-mixup_manual_anno_DeOcc${X}_Perturb${Y}_cocotrain.pth" ] || python adapt.py --opt distort --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --num_workers 4 --de_occlusion ${X} --perturbation ${Y}; \
        rm -r __pycache__ adapt_output; \
    done; \
done

############ semi
python adapt.py --opt semi --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --anno_day 0 --anno_iteration 0 --cocodir ../../../MSCOCO2017 --num_workers 4

rm -r __pycache__ adapt_output

############ with false negative mining
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --anno_day 0 1 2 3 --anno_iteration 0 --cocodir ../../../MSCOCO2017 --num_workers 4 --fn_mining_num 25 --tag ".A"

############ single day
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --anno_day 0 --anno_iteration 0 --cocodir ../../../MSCOCO2017 --num_workers 4

############ days combo
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth --anno_day 0 1 2 3 --anno_iteration 0 --cocodir ../../../MSCOCO2017 --num_workers 4

############ iterative
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --anno_day 0 --anno_iteration 0 --cocodir ../../../MSCOCO2017 --num_workers 4 --ckpt models/mscoco2017_remap_wdiff_midfusion_boxinpaint_r101-fpn-3x.pth

python pseudo_label.py --opt label --model r101-fpn-3x-midfusion-mixup --iteration 1 --day 0 1 --ckpt adapt_r101-fpn-3x-midfusion-mixup_refined_anno_days_0_anno_iters_0_cocotrain.pth
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --anno_day 0 1 --anno_iteration 0 1 --cocodir ../../../MSCOCO2017 --num_workers 4 --ckpt adapt_r101-fpn-3x-midfusion-mixup_refined_anno_days_0_anno_iters_0_cocotrain.pth

python pseudo_label.py --opt label --model r101-fpn-3x-midfusion-mixup --iteration 2 --day 0 1 2 --ckpt adapt_r101-fpn-3x-midfusion-mixup_refined_anno_days_0_1_anno_iters_0_1_cocotrain.pth
python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --anno_day 0 1 2 --anno_iteration 0 1 2 --cocodir ../../../MSCOCO2017 --num_workers 4 --ckpt adapt_r101-fpn-3x-midfusion-mixup_refined_anno_days_0_1_anno_iters_0_1_cocotrain.pth

python pseudo_label.py --opt label --model r101-fpn-3x-midfusion-mixup --iteration 3 --day 0 1 2 3 --ckpt 

python adapt.py --opt adapt --model r101-fpn-3x-midfusion-mixup --anno_day 0 1 2 3 --anno_iteration 0 3 --cocodir ../../../MSCOCO2017 --num_workers 4 --ckpt 

'''
