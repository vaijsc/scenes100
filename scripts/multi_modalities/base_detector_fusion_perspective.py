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
import cv2
import pycocotools.mask

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata

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
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from fusion_models import GeneralizedRCNNEarlyFusion, GeneralizedRCNNMidFusion, GeneralizedRCNNLateFusion

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'base_output_fusion_augment')


def construct_image_w_background(image, image_background):
    image_diff = (image.astype(np.float16) - image_background) # float16, [-255, 255]
    image_diff = ((image_diff + 255) * 0.5).astype(np.uint8)
    return image, image_background, image_diff


def finetune_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start
    loss_dict = self.model(data)
    loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {'total_loss': loss_dict}
    else:
        losses = sum(loss_dict.values())
    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    # If you need gradient clipping/scaling or other processing, you can wrap the optimizer with your custom `step()` method. But it is suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    self.optimizer.step()
    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


def RLEtoPolygons(rle):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    rle = pycocotools.mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
    M = pycocotools.mask.decode(rle)
    area = float((M > 0.0).sum())
    contours, _ = cv2.findContours(M, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons, valid_poly = [], 0
    for contour in contours:
        if contour.size >= 6: # Valid polygons have >= 6 coordinates (3 points)
            polygons.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    return polygons


class DatasetMapperBackgroundPerspectiveMosaic(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        _aug, _r = 'none', np.random.rand()
        if _r <= self.mosaic_p:
            patch = 2 if np.random.rand() < 0.7 else 3
            _aug = 'mosaic_%dx%d' % (patch, patch)
            dataset_dicts_mosaic = dataset_dict['mosaic_src_images']
            del dataset_dict['mosaic_src_images']
            dataset_dicts_mosaic = np.random.choice(dataset_dicts_mosaic + [dataset_dict], patch * patch).reshape(patch, patch)
            target_w, target_h = dataset_dicts_mosaic[0, 0]['width'], dataset_dicts_mosaic[0, 0]['height']
            image, image_diff = np.zeros((target_h * patch, target_w * patch, 3), dtype=np.uint8), np.zeros((target_h * patch, target_w * patch, 3), dtype=np.uint8)
            annotations_mosaic = []

            for patch_i in range(0, patch):
                for patch_j in range(0, patch):
                    image_ij = detectron2.data.detection_utils.read_image(dataset_dicts_mosaic[patch_i, patch_j]['file_name'], format=self.image_format)
                    image_background_ij = detectron2.data.detection_utils.read_image(dataset_dicts_mosaic[patch_i, patch_j]['file_name_background'], format=self.image_format)
                    image_ij, image_background_ij = cv2.resize(image_ij, [target_w, target_h]), cv2.resize(image_background_ij, [target_w, target_h])
                    assert image_background_ij.shape == image_ij.shape
                    image_ij, _, image_diff_ij = construct_image_w_background(image_ij, image_background_ij)
                    del image_background_ij
                    # print(target_w, target_h, dataset_dicts_mosaic[patch_i, patch_j]['width'], dataset_dicts_mosaic[patch_i, patch_j]['height'], image_ij.shape, image_ij.dtype)
                    image[patch_i * target_h : (patch_i + 1) * target_h, patch_j * target_w : (patch_j + 1) * target_w] = image_ij
                    image_diff[patch_i * target_h : (patch_i + 1) * target_h, patch_j * target_w : (patch_j + 1) * target_w] = image_diff_ij
                    for ann in dataset_dicts_mosaic[patch_i, patch_j]['annotations']:
                        assert ann['bbox_mode'] == BoxMode.XYWH_ABS
                        x, y, w, h = ann['bbox']
                        annotations_mosaic.append(copy.deepcopy(ann))
                        annotations_mosaic[-1]['bbox'] = [
                            x * (dataset_dicts_mosaic[patch_i, patch_j]['width'] / target_w) + patch_j * target_w,
                            y * (dataset_dicts_mosaic[patch_i, patch_j]['height'] / target_h) + patch_i * target_h,
                            w * (dataset_dicts_mosaic[patch_i, patch_j]['width'] / target_w),
                            h * (dataset_dicts_mosaic[patch_i, patch_j]['height'] / target_h)
                        ]
            dataset_dict['width'], dataset_dict['height'], dataset_dict['annotations'] = target_w * patch, target_h * patch, annotations_mosaic
        elif _r > self.mosaic_p and _r <= self.mosaic_p + self.perspective_p:
            dataset_dict['annotations_orig'] = dataset_dict['annotations']
            dataset_dict['annotations'] = []
            image_orig = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
            image_background_orig = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
            if np.random.rand() < 0.5:
                w, h = dataset_dict['width'], dataset_dict['height']
                scale_w = 1 + np.random.rand() * (self.perspective_upscale_max - 1)
                scale_h = 1 + np.random.rand() * (scale_w - 1)
                _aug = 'perspective_%.2fx%.2f' % (scale_w, scale_h)
                pt_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                pt_warped = np.float32([[0, 0], [w * scale_w, 0], [w * (scale_w + 1) / 2, h * scale_h], [w * (scale_w - 1) / 2, h * scale_h]])
                dataset_dict['width'], dataset_dict['height'] = int(dataset_dict['width'] * scale_w), int(dataset_dict['height'] * scale_h)
            else:
                w, h = dataset_dict['width'], dataset_dict['height']
                scale_w = self.perspective_downscale_min + np.random.rand() * (1 - self.perspective_downscale_min)
                scale_h = scale_w + np.random.rand() * (1 - scale_w)
                _aug = 'perspective_%.2fx%.2f' % (scale_w, scale_h)
                pt_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                pt_warped = np.float32([[w * (1 - scale_w) / 2, 0], [w * (1 + scale_w) / 2, 0], [w, h * scale_h], [0, h * scale_h]])
                dataset_dict['width'], dataset_dict['height'] = dataset_dict['width'], int(dataset_dict['height'] * scale_h)

            M_warp = cv2.getPerspectiveTransform(pt_corners, pt_warped)
            image = cv2.warpPerspective(image_orig, M_warp, (dataset_dict['width'], dataset_dict['height']))
            image_background = cv2.warpPerspective(image_background_orig, M_warp, (dataset_dict['width'], dataset_dict['height']))
            assert image_background.shape == image.shape
            image, _, image_diff = construct_image_w_background(image, image_background)
            del image_background_orig, image_background

            for ann in dataset_dict['annotations_orig']:
                if type(ann['segmentation']) == type({}):
                    ann['segmentation'] = RLEtoPolygons(ann['segmentation'])
                    print('convert RLE')
                if len(ann['segmentation']) <= 0:
                    continue
                assert ann['bbox_mode'] == BoxMode.XYWH_ABS
                dataset_dict['annotations'].append(copy.deepcopy(ann))
                del dataset_dict['annotations'][-1]['segmentation']
                pts = list(map(np.float32, ann['segmentation']))
                pts = np.concatenate(pts, axis=0).reshape(1, -1, 2)
                pts_warped = cv2.perspectiveTransform(pts, M_warp)[0]
                x1, y1, x2, y2 = max(0, pts_warped[:, 0].min()), max(0, pts_warped[:, 1].min()), min(pts_warped[:, 0].max(), dataset_dict['width']), min(pts_warped[:, 1].max(), dataset_dict['height'])
                dataset_dict['annotations'][-1]['bbox'] = list(map(float, [x1, y1, x2 - x1, y2 - y1]))
        else:
            image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
            image_background = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
            assert image_background.shape == image.shape
            image, _, image_diff = construct_image_w_background(image, image_background)
            del image_background

        # if _aug.startswith('mosaic'):
        #     import matplotlib.patches as patches
        #     _, ax = plt.subplots(1, 2); ax = ax.reshape(-1)
        #     ax[0].imshow(image_diff[:, :, ::-1])
        #     ax[0].set_axis_off()
        #     ax[1].set_title(_aug)
        #     ax[1].imshow(image)
        #     ax[1].set_axis_off()
        #     for ann in dataset_dict['annotations']:
        #         assert ann['bbox_mode'] == BoxMode.XYWH_ABS
        #         (x, y, w, h), k = ann['bbox'], ann['category_id']
        #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
        #         ax[1].add_patch(rect)
        #     plt.tight_layout()
        #     plt.show()
        # if _aug.startswith('perspective'):
        #     import matplotlib.patches as patches
        #     _, ax = plt.subplots(1, 2); ax = ax.reshape(-1)
        #     ax[0].imshow(image_orig[:, :, ::-1])
        #     ax[0].set_axis_off()
        #     for ann in dataset_dict['annotations_orig']:
        #         assert ann['bbox_mode'] == BoxMode.XYWH_ABS
        #         (x, y, w, h), k = ann['bbox'], ann['category_id']
        #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
        #         ax[0].add_patch(rect)
        #     ax[1].set_title(_aug)
        #     ax[1].imshow(image[:, :, ::-1])
        #     ax[1].set_axis_off()
        #     for ann in dataset_dict['annotations']:
        #         assert ann['bbox_mode'] == BoxMode.XYWH_ABS
        #         (x, y, w, h), k = ann['bbox'], ann['category_id']
        #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
        #         ax[1].add_patch(rect)
        #     plt.tight_layout()
        #     plt.show()

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        # USER: Remove if you don't do semantic/panoptic segmentation.
        sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_diff = transforms.apply_image(image_diff)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = np.concatenate([image, image_diff], axis=2)
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    @staticmethod
    def create_from_sup(mapper, perspective_p, perspective_upscale_max, perspective_downscale_min, mosaic_p):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperBackgroundPerspectiveMosaic
        mapper.perspective_p, mapper.perspective_upscale_max, mapper.perspective_downscale_min, mapper.mosaic_p = perspective_p, perspective_upscale_max, perspective_downscale_min, mosaic_p
        return mapper


class AdaptationTrainer(DefaultTrainer):
    def __init__(self, cfg, args):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        model = self.build_model(cfg)
        assert isinstance(model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        assert args.fusion == 'mid'
        model = GeneralizedRCNNMidFusion.create_from_sup(model, args.multitask_loss_alpha)
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

        self.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackgroundPerspectiveMosaic.create_from_sup(self.data_loader.dataset.dataset.dataset._map_func._obj, args.perspective_p, args.perspective_upscale_max, args.perspective_downscale_min, args.mosaic_p)
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []
        self._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, self._trainer)

    def build_hooks(self):
        ret = super().build_hooks()
        self.eval_results_all = {}
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = copy.deepcopy(self._last_eval_results)
            return self._last_eval_results
        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = detectron2.data.build_detection_test_loader(cfg, dataset_name)
        assert isinstance(loader.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
        loader.dataset._map_func._obj = DatasetMapperBackgroundPerspectiveMosaic.create_from_sup(loader.dataset._map_func._obj, -1, -1, -1, -1)
        return loader


def train_base_perspective(args):
    dst_cocovalid, dst_cocotrain = get_coco_dicts(args, 'valid'), get_coco_dicts(args, 'train', segment=True)
    for im in dst_cocovalid:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'val2017', os.path.basename(im['file_name'])))
    for im in dst_cocotrain:
        im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'train2017', os.path.basename(im['file_name'])))

    dst_cocotrain_aspects = {}
    for im in dst_cocotrain:
        im['aspect'] = '%.2f' % (im['width'] / im['height'])
        if not im['aspect'] in dst_cocotrain_aspects:
            dst_cocotrain_aspects[im['aspect']] = []
        dst_cocotrain_aspects[im['aspect']].append(copy.deepcopy(im))
    print('%d aspect ratio buckets' % len(dst_cocotrain_aspects))
    for im in tqdm.tqdm(dst_cocotrain, ascii=True, desc='populating mosaic sources'):
        if len(dst_cocotrain_aspects[im['aspect']]) > 0:
            im['mosaic_src_images'] = np.random.choice(dst_cocotrain_aspects[im['aspect']], 8).tolist()
        else:
            im['mosaic_src_images'] = []
    del dst_cocotrain_aspects

    assert args.fusion == 'mid'
    prefix = 'mscoco2017_remap_wdiff_midfusion_%s_perspective_p%.2f_mosaic_p%.2f' % (args.model, args.perspective_p, args.mosaic_p)
    desc_cocovalid, desc_cocotrain = 'mscoco2017_valid_remap_midfusion', 'mscoco2017_train_remap_midfusion_rand_perspective'
    args.ckpt = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth')

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
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
    cfg.DATASETS.TEST = (desc_cocovalid,)
    # print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200
    trainer = AdaptationTrainer(cfg, args)
    trainer.resume_or_load(resume=args.resume)

    results_0 = {}
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
    dst_list = {'mAP': [], 'AP50': []}
    for i in iter_list:
        dst_list['mAP'].append(aps[i]['bbox']['AP'])
        dst_list['AP50'].append(aps[i]['bbox']['AP50'])

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
    plt.plot(iter_list, np.array(dst_list['AP50']) / 100, linestyle='--', marker='x', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list['mAP']) / 100, linestyle='--', marker='x', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP'])
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
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--fusion', type=str, default='mid')
    parser.add_argument('--model', type=str, help='detection model', default='r101-fpn-3x')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--resume', type=bool, default=False)

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--smallscale', default=False, type=bool)

    parser.add_argument('--perspective_p', type=float, default=0.25)
    parser.add_argument('--perspective_upscale_max', type=float, default=2)
    parser.add_argument('--perspective_downscale_min', type=float, default=0.5)
    parser.add_argument('--mosaic_p', type=float, default=0.25)

    parser.add_argument('--multitask_loss_alpha', type=float, default=0.5, help='relative weight of multitasking losses')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    assert args.perspective_p >= 0
    assert args.mosaic_p >= 0
    assert args.perspective_p + args.mosaic_p < 0.9
    assert args.perspective_upscale_max > 1.2
    assert args.perspective_downscale_min < 0.8
    print(args)

    finetune_output = finetune_output + '_perspective_p%.2f_mosaic_p%.2f' % (args.perspective_p, args.mosaic_p)
    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    train_base_perspective(args)


'''
conda deactivate && conda activate detectron2
cd /nfs/detection/zekun/Intersections/scripts/baseline

python base_detector_fusion_perspective.py --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 160 --image_batch_size 2 --smallscale 1
python base_detector_fusion_perspective.py --cocodir ../../../MSCOCO2017 --iters 40000 --eval_interval 4001 --image_batch_size 4 --num_workers 4
'''
