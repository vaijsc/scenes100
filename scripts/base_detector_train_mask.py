#!python3

import os
import sys
import types
import time
import datetime
import json
import copy
import math
import random
import tqdm
import glob
import gzip
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

import skimage.io
import skvideo.io
from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict
import networkx

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.engine import launch as detectron2_launcher
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import BoxMode

from base_detector_train import get_coco_dicts, simple_trainer_run_step, FinetuneTrainer


thing_classes = ['person', 'vehicle']


def train_eval(args):
    cfg_str = {
        'mask-r50-fpn-3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'mask-r101-fpn-3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'mask-x101-fpn-3x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
    }
    cfg_str_custom = {
        'mask-r18-fpn-3x': ['mask_rcnn_R_18_FPN_3x.yaml', 'mscoco2017_remap_r18-fpn-3x.pth'],
        'mask-r34-fpn-3x': ['mask_rcnn_R_34_FPN_3x.yaml', 'mscoco2017_remap_r34-fpn-3x.pth'],
        'mask-r152-fpn-3x': ['mask_rcnn_R_152_FPN_3x.yaml', 'mscoco2017_remap_r152-fpn-3x.pth'],
        'mask-x50-fpn-3x': ['mask_rcnn_X_50_32x4d_FPN_3x.yaml', 'mscoco2017_remap_x50-fpn-3x.pth'],
    }
    assert args.model in cfg_str or args.model in cfg_str_custom
    if args.model in cfg_str:
        args.det_cfg = model_zoo.get_config_file(cfg_str[args.model])
    elif args.model in cfg_str_custom:
        args.det_cfg = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_str_custom[args.model][0]))
    else:
        print('model not specified:', args.model)
        exit(1)
    output_dir = os.path.join(os.path.dirname(__file__), 'finetune_output_base_maskrcnn_' + args.model.replace('-', '_'))

    dst_train, dst_valid = get_coco_dicts(args, 'train', segment=True), get_coco_dicts(args, 'valid', segment=True)
    for im in dst_train + dst_valid:
        im['annotations'] = list(filter(lambda ann: type(ann['segmentation']) == type([]), im['annotations']))
    DatasetCatalog.register('mscoco2017_train_remap', lambda: dst_train)
    DatasetCatalog.register('mscoco2017_valid_remap', lambda: dst_valid)
    MetadataCatalog.get('mscoco2017_train_remap').thing_classes = thing_classes
    MetadataCatalog.get('mscoco2017_valid_remap').thing_classes = thing_classes

    cfg = get_cfg()
    cfg.merge_from_file(args.det_cfg)
    if args.model in cfg_str:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_str[args.model])
    elif args.model in cfg_str_custom:
        cfg.MODEL.WEIGHTS = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models', cfg_str_custom[args.model][1]))
    cfg.OUTPUT_DIR = output_dir

    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.3
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = ('mscoco2017_train_remap',)
    cfg.DATASETS.TEST = ('mscoco2017_valid_remap',)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.INPUT.MASK_FORMAT = 'polygon'
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print('- image batch size:', cfg.SOLVER.IMS_PER_BATCH)
    print('- roi batch size:', cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    print('- base lr:', cfg.SOLVER.BASE_LR)
    print('- lr warmpup iteration:', cfg.SOLVER.WARMUP_ITERS)
    print('- lr schedule gamma:', cfg.SOLVER.GAMMA)
    print('- lr schedule steps:', cfg.SOLVER.STEPS)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 180
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 250
    trainer = FinetuneTrainer(cfg)

    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.split_batch = args.split_batch
    trainer._trainer.run_step = types.MethodType(simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=args.resume)
    print('trainer initialized')

    prefix = 'mscoco2017_remap_%s' % args.model
    results_0 = OrderedDict()
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

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
    mAPs_box = [aps[i]['bbox']['AP'] for i in iter_list]
    AP50s_box = [aps[i]['bbox']['AP50'] for i in iter_list]
    mAPs_mask = [aps[i]['segm']['AP'] for i in iter_list]
    AP50s_mask = [aps[i]['segm']['AP50'] for i in iter_list]
    mAPs_box, AP50s_box, mAPs_mask, AP50s_mask = map(lambda arr: np.array([x if not math.isnan(x) else 0.0 for x in arr], dtype=np.float32), [mAPs_box, AP50s_box, mAPs_mask, AP50s_mask])

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
    plt.plot(iter_list, AP50s_box / 100, linestyle='--', marker='o', color='#FF0000')
    plt.plot(iter_list, mAPs_box / 100, linestyle='--', marker='x', color='#0000FF')
    plt.plot(iter_list, AP50s_mask / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, mAPs_mask / 100, linestyle='-', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'bbox AP50', 'bbox mAP', 'mask AP50', 'mask mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlim(-100, max(iter_list) + 100)
    plt.ylim(-0.02, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP on MSCOCO-2017 Validation Split')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Base Mask-RCNN Model on MSCOCO-2017 with Refined Classes')
    parser.add_argument('--model', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--iters', type=int)
    parser.add_argument('--eval_interval', type=int)
    parser.add_argument('--cocodir', default='MSCOCO2017', type=str)
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=256, type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--smallscale', default=False, type=bool)

    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--split_batch', type=int, default=1)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    print(args)

    if args.ddp_num_gpus <= 1:
        train_eval(args)
    else:
        from detectron2.engine import launch
        launch(train_eval, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))


'''
conda deactivate && conda activate detectron2

python base_detector_train_mask.py --model mask-r50-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4

nohup python base_detector_train_mask.py --model mask-r50-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 4 &> log_base_mask_r50.log &
nohup python base_detector_train_mask.py --model mask-r101-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 4 &> log_base_mask_r101.log &
nohup python base_detector_train_mask.py --model mask-x101-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 4 &> log_base_mask_x101.log &

CUDA_VISIBLE_DEVICES=0,1 python base_detector_train_mask.py --model mask-r18-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 2 &> log_base_mask_r18.log &
CUDA_VISIBLE_DEVICES=2,3 python base_detector_train_mask.py --model mask-r34-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 2 &> log_base_mask_r34.log &

CUDA_VISIBLE_DEVICES=0,1 python base_detector_train_mask.py --model mask-r152-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 2 &> log_base_mask_r152.log &
CUDA_VISIBLE_DEVICES=2,3 python base_detector_train_mask.py --model mask-x50-fpn-3x --iters 200000 --eval_interval 10001 --cocodir ../../MSCOCO2017 --num_workers 4 --ddp_num_gpus 2 &> log_base_mask_x50.log &
'''
