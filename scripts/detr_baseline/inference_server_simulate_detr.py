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
import itertools
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

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model, default_setup
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, ImageList, Instances
from detectron2.config import get_cfg
from detr import *
from config import add_detr_config

import logging
import weakref

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


# video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
video_id_list = ['001']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_inference_server')


class AdaptativePartialTrainer(DefaultTrainer):
    def __init__(self, cfg, incremental_videos, ckpt):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order. 
        model = self.build_model(cfg)
        # model = Detr(cfg)  # TODO: edit loading here
        state_dict = torch.load(ckpt)['model']
        model.load_state_dict(state_dict, strict=False)

        self.model_teacher = deepcopy(model) # TODO: model teacher output?
        self.model_teacher.eval()
        
        trainable_modules = [model] # TODO: edit for training prompt?
        _count_all, _count_train = 0, 0
        for p in model.parameters():
            _count_all += p.numel()
            p.requires_grad = False
        for m in trainable_modules:
            for p in m.parameters():
                _count_train += p.numel()
                p.requires_grad = True
        print('only train subset of model parameters: %d/%d %.4f%%' % (_count_train, _count_all, _count_train / _count_all * 100))
        optimizer = self.build_optimizer(cfg, torch.nn.ModuleList(trainable_modules))
        if incremental_videos:
            data_loader = detectron2.data.build_detection_train_loader(cfg,
                sampler=detectron2.data.samplers.distributed_sampler.TrainingSampler(cfg.DATASETS.TRAIN_NUM_IMAGES, shuffle=False)
            )
        else:
            data_loader = self.build_train_loader(cfg)

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
        ret = super().build_hooks()
        self.eval_results_all = {}
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = deepcopy(self._last_eval_results)
            return self._last_eval_results
        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)

        def save_model_state():
            model = self.model
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                print('unwrap data parallel')
                model = model.module
            prefix = '%s.iter.%d' % (self.cfg.SOLVER.SAVE_PREFIX, self.iter)
            torch.save(model.state_dict(), prefix + '.pth')
            # mapper = {
            #     'budget': model.budget,
            #     'video_id_to_index': model.video_id_to_index,
            #     'used_indices': model.used_indices,
            #     'un_used_indices': model.un_used_indices
            # }
            # print(mapper)
            # torch.save(mapper, prefix + '.mapper.pth')
            print('saved model state to:', prefix)
        ret.append(detectron2.engine.hooks.EvalHook(self.cfg.SOLVER.SAVE_INTERVAL, save_model_state))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def finetune_ema_simple_trainer_run_step(self):
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
            pseudo_labels = self.model_teacher.forward(pseudo_inputs)
            for _i, _pred in zip(pseudo_idx, pseudo_labels):
                _mask = _pred['instances'].scores >= self.pseudo_det_min_score
                _filtered = Instances(_pred['instances']._image_size)
                _filtered.set('gt_boxes', _pred['instances'].pred_boxes[_mask])
                _filtered.set('gt_classes', _pred['instances'].pred_classes[_mask])
                data[_i]['instances'] = _filtered
                del data[_i]['image_test']

    # import matplotlib.patches as patches
    # _, axes = plt.subplots(1, len(data)); axes = axes.reshape(-1)
    # for _i, _d in enumerate(data):
    #     print(_d['image'].size(), _d['instances']._image_size)
    #     _im = _d['image'][0].detach().cpu().numpy(); axes[_i].imshow(_im)
    #     for j in range(0, len(_d['instances'])):
    #         x1, y1, x2, y2 = _d['instances'].gt_boxes.tensor[j].detach().cpu().numpy()
    #         k = _d['instances'].gt_classes[j].detach().cpu().numpy()
    #         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
    #         axes[_i].add_patch(rect)
    # plt.show()

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


class DatasetMapperPseudo(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        # for generating pseudo labels on the fly
        if 'source' in dataset_dict and dataset_dict['source'] == 'unlabeled':
            image_test = self.apply_test_transform(image)
            if image_test is not None:
                dataset_dict['image_test'] = torch.as_tensor(np.ascontiguousarray(image_test.transpose(2, 0, 1)))
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        assert self.proposal_topk is None
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)
        return dataset_dict

    def apply_test_transform(self, image):
        if not (image.dtype == np.uint8 and len(image.shape) == 3 and image.shape[2] == 3):
            return None
        h, w = image.shape[:2]
        # scale = np.random.rand() * 0.75 + 1.5 # upscale by 1.5 ~ 2.25
        scale = 1
        min_size, max_size = map(lambda x: int(x * scale), [self.min_size_test, self.max_size_test])
        newh, neww = self.get_output_shape(h, w, min_size, max_size)
        pil_image = Image.fromarray(image)
        # pil_image = pil_image.resize((neww, newh), Image.Resampling.BILINEAR)
        pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
        return np.asarray(pil_image)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
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
        assert not cfg.INPUT.CROP.ENABLED
        assert cfg.INPUT.RANDOM_FLIP == 'none'
        mapper.min_size_test, mapper.max_size_test = cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperPseudo
        return mapper


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = deepcopy(args)
    _args.cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'MSCOCO2017'))
    _args.smallscale = False
    random.seed(42)

    # validation images
    desc_manual_valid, dst_manual_valid = 'allvideos_manual', []
    for v in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', v)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            annotations = json.load(fp)
        for i in range(0, len(annotations)):
            annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
            annotations[i]['video_id'] = v
        print('manual annotation for %s: %d images, %d bboxes' % (v, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
        dst_manual_valid.extend(annotations)
    for i in range(0, len(dst_manual_valid)):
        assert 'video_id' in dst_manual_valid[i]
        dst_manual_valid[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(dst_manual_valid), sum(list(map(lambda x: len(x['annotations']), dst_manual_valid)))))

    # training images
    dst_cocotrain = get_coco_dicts(_args, 'train')
    for im in dst_cocotrain:
        im['source'] = 'coco'
        im['video_id'] = 'coco'
    random.shuffle(dst_cocotrain)

    dst_pseudo_anno, desc_pseudo_anno = [], 'allvideos_unlabeled_cocotrain'
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    for v in video_id_list:
        lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', v))
        with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
            meta = json.load(fp)
        ifilelist = meta['ifilelist']
        dict_json = []
        for i in range(0, len(ifilelist)):
            dict_json.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}], 'source': 'unlabeled', 'video_id': v})
        print('unlabeled frames of video %s at %s: %d images' % (v, lmdb_path, len(dict_json)))
        if len(dict_json) > images_per_video_cap:
            random.shuffle(dict_json)
            dict_json = dict_json[:images_per_video_cap]
            print('randomly downsampled to: %d images' % len(dict_json))
        dst_pseudo_anno.extend(dict_json)
    print('total unlabeled: %d images' % len(dst_pseudo_anno))

    if args.incremental_videos:
        # for video incremental training, make sure (# of training frames) == iterations * image-batchsize
        # 1/4 of images are from MSCOCO
        while len(dst_pseudo_anno) < args.iters * args.image_batch_size:
            dst_pseudo_anno = dst_pseudo_anno + dst_pseudo_anno
        random.shuffle(dst_pseudo_anno)
        dst_pseudo_anno = dst_pseudo_anno[: args.iters * args.image_batch_size * 3 // 4]
        dst_pseudo_anno.sort(key=lambda x: hashlib.md5(x['video_id'].encode('utf-8')).hexdigest() + os.path.basename(x['file_name']))
        dst_pseudo_anno_with_coco = []
        for i in range(0, len(dst_pseudo_anno) // 3 - 1):
            dst_pseudo_anno_with_coco.extend(dst_pseudo_anno[i * 3 : (i + 1) * 3])
            dst_pseudo_anno_with_coco.append(dst_cocotrain[i % len(dst_cocotrain)])
        while len(dst_pseudo_anno_with_coco) < (args.iters + 2) * args.image_batch_size:
            dst_pseudo_anno_with_coco.append(deepcopy(dst_pseudo_anno_with_coco[-1]))
        dst_pseudo_anno = dst_pseudo_anno_with_coco
        del dst_pseudo_anno_with_coco
        assert len(dst_pseudo_anno) >= (args.iters + 2) * args.image_batch_size
    else:
        # 1/4 of images are from MSCOCO
        dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: len(dst_pseudo_anno) // 3]
    for i in range(0, len(dst_pseudo_anno)):
        assert 'video_id' in dst_pseudo_anno[i]
        dst_pseudo_anno[i]['image_id'] = i + 1
    print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
    prefix = 'adaptive_partial_server_%s_anno_%s.%s' % ("detr", desc_pseudo_anno, args.tag)

    del _tensor, _args
    gc.collect()
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    # cfg = get_cfg_base_model(args.model)
    cfg = get_cfg()
    add_detr_config(cfg)
    # cfg.freeze()
    default_setup(cfg, args)
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    # cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    # cfg.SOLVER.GAMMA = 0.5
    # cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.WARMUP_ITERS = args.iters // 200
    cfg.SOLVER.GAMMA = 1
    cfg.SOLVER.STEPS = ()
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid,)

    cfg.SOLVER.PSEUDO_DET_MIN_SCORE = args.refine_det_score_thres
    cfg.MODEL.ADAPTIVE_BUDGET = args.budget
    cfg.MODEL.CLUSTER_CAP_FACTOR = args.cluster_cap_factor
    cfg.MODEL.BUFFER_SIZE = args.buffer_size
    cfg.MODEL.UPDATE_CLUSTER_INTERVAL = args.update_cluster_interval
    cfg.MODEL.P_BUFFER = args.p_buffer
    cfg.MODEL.P_REPLAY = args.p_replay
    cfg.SOLVER.SAVE_INTERVAL = args.save_interval
    cfg.SOLVER.SAVE_PREFIX = os.path.join(args.outputdir, prefix)
    cfg.DATASETS.TRAIN_NUM_IMAGES = len(dst_pseudo_anno)
    cfg.DATASETS.TEST_NUM_IMAGES = len(dst_manual_valid)
    cfg.CONFIG_PATH = args.config
    cfg.MODEL.SPLIT_LIST = args.split_list
    
    
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = AdaptativePartialTrainer(cfg, args.incremental_videos, args.ckpt)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_ema_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperPseudo.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, cfg)

    # manually load the parameters from ckpt
    print('loading weights from:', args.ckpt)
    # trainer.resume_or_load(resume=False)
    assert trainer.model is trainer._trainer.model
    assert trainer.model_teacher is trainer._trainer.model_teacher


    print(prefix)
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
    with open(os.path.join(args.outputdir, prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    dst_list = {'mAP': [], 'AP50': []}
    for i in iter_list:
        dst_list['mAP'].append(aps[i]['bbox']['AP'])
        dst_list['AP50'].append(aps[i]['bbox']['AP50'])

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for x in loss_history:
        for loss_key in x['loss']:
            if not loss_key in loss_history_dict:
                loss_history_dict[loss_key] = []
            loss_history_dict[loss_key].append([x['iter'], x['loss'][loss_key]])
    loss_history_dict = {loss_key: np.array(loss_history_dict[loss_key]) for loss_key in loss_history_dict}
    for loss_key in loss_history_dict:
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

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
    for loss_key in loss_history_dict:
        if color_i <= 6: # show only maximum 6 loss items 
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


class SemiRandomClient(torchdata.Dataset):
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
                im['md5'] = '%s_%s' % (video_id, im['file_name']) # for pseudo-random shuffling
                im['md5'] = hashlib.md5(im['md5'].encode('utf-8')).hexdigest()
                im['file_name'] = os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['video_id'] = video_id
            self.images.extend(_dicts)
        self.images.sort(key=lambda x: x['md5'])
        self.preloaded_images = None

    def preload(self):
        if self.preloaded_images is not None:
            return
        self.preloaded_images = []
        for i in tqdm.tqdm(range(0, len(self.images)), ascii=True, desc='preloading images'):
            self.preloaded_images.append(self.read(i))

    def __len__(self):
        return len(self.images)

    def read(self, i):
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


def simulate(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid

    mapper = torch.load(os.path.join(args.ckpts_dir, '%s.mapper.pth' % args.ckpts_tag))
    index_to_video_id = {}
    for v, i in mapper['video_id_to_index'].items():
        if not i in index_to_video_id:
            index_to_video_id[i] = [v]
        else:
            index_to_video_id[i].append(v)
    for i in sorted(list(index_to_video_id.keys())):
        print(i, sorted(index_to_video_id[i]))
    print('unused:', mapper['un_used_indices'])

    cfg = get_cfg_base_model("r101-fpn-3x")
    model = YOLOServer.create_from_sup(load_model(os.path.join(os.path.dirname(__file__),'../../configs/yolov3-custom.cfg')), mapper['budget'])
    model.video_id_to_index = mapper['video_id_to_index']
    model.used_indices = mapper['used_indices']
    model.un_used_indices = mapper['un_used_indices']

    state_dict = torch.load(os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    model.load_state_dict(state_dict)
    del state_dict

    dataset = SemiRandomClient(cfg)
    if args.preload:
        dataset.preload()
    loader = torchdata.DataLoader(dataset, batch_size=None, collate_fn=SemiRandomClient.collate, shuffle=False, num_workers=1)
    gc.collect()
    torch.cuda.empty_cache()

    detections = {v: [] for v in video_id_list}
    t_total = time.time()
    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='detecting'):
        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model.inference([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
        detections[im['video_id']].append(det)
    t_total = time.time() - t_total
    print('%d finished in %.1f seconds, throughput %.3f images/sec' % (len(dataset), t_total, len(dataset) / t_total))
    #TODO: get the base_AP for YOLO
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}
    for video_id in tqdm.tqdm(detections, ascii=True, desc='evaluating'):
        for det in detections[video_id]:
            # bbox has format [x1, y1, x2, y2]
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)
        results[video_id]['detections'] = detections
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    for video_id in results:
        AP1 = base_AP['manual_' + video_id]['results']
        AP2 = results[video_id]['results']
        for cat in categories:
            improvements[cat].append([AP2[cat][0] - AP1[cat][0], AP2[cat][1] - AP1[cat][1]])
    for cat in categories:
        improvements[cat] = np.array(improvements[cat]) * 100.0
        print('%s: mAP-G %.4f, AP50-G %.4f' % (cat, improvements[cat][:, 0].mean(), improvements[cat][:, 1].mean()))


def _run_detector(args):
    cfg, dataset, ckpt, preload = args
    if preload:
        dataset.preload()
    # model = DefaultPredictor(cfg).model
    model = load_model(os.path.join(os.path.dirname(__file__),'../../configs/yolov3-custom.cfg'), ckpt)
    loader = torchdata.DataLoader(dataset, batch_size=None, collate_fn=SemiRandomClient.collate, shuffle=False, num_workers=0)
    gc.collect()
    torch.cuda.empty_cache()
    detections = {v: [] for v in video_id_list}
    t_total = time.time()
    for inputs, im in tqdm.tqdm(loader, ascii=True, total=len(dataset), desc='detecting'):
        det = deepcopy(im)
        det['annotations'] = []
        with torch.no_grad():
            instances = model.inference([inputs])[0]['instances'].to('cpu')
            det['instances'] = {
                'bbox': instances.pred_boxes.tensor,
                'score': instances.scores,
                'label': instances.pred_classes
            }
        detections[im['video_id']].append(det)
    t_total = time.time() - t_total
    return detections, t_total

def simulate_concurrent(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    multiprocessing.set_start_method('spawn')

    cfg = get_cfg_base_model("r101-fpn-3x", ckpt=args.ckpt)
    dataset = SemiRandomClient(cfg)
    dataset_chunks = [deepcopy(dataset) for _ in range(0, args.instances)]
    for d in dataset_chunks:
        d.images = []
    for i, im in enumerate(dataset.images):
        dataset_chunks[i % len(dataset_chunks)].images.append(im)

    pool = ProcessPool(processes=args.instances)
    args_list = [(deepcopy(cfg), d, args.ckpt, args.preload) for d in dataset_chunks]
    rets = pool.map_async(_run_detector, args_list).get()
    pool.close()
    pool.join()
    detections_chunks = [r[0] for r in rets]
    t_total = [r[1] for r in rets]
    print('instances times:', t_total)
    t_total = max(t_total)
    print('%d finished in %.1f seconds, throughput %.3f images/sec' % (len(dataset), t_total, len(dataset) / t_total))

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}
    detections = detections_chunks[0]
    for chunk in detections_chunks[1:]:
        for v in chunk:
            detections[v].extend(chunk[v])
    for video_id in tqdm.tqdm(detections, ascii=True, desc='evaluating'):
        for det in detections[video_id]:
            # bbox has format [x1, y1, x2, y2]
            bbox, score, label = det['instances']['bbox'].numpy().tolist(), det['instances']['score'].numpy().tolist(), det['instances']['label'].numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            del det['instances']
            det['file_name'] = os.path.basename(det['file_name'])
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections[video_id], outputfile=None)
        results[video_id]['detections'] = detections
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    for video_id in results:
        AP1 = base_AP['manual_' + video_id]['results']
        AP2 = results[video_id]['results']
        for cat in categories:
            improvements[cat].append([AP2[cat][0] - AP1[cat][0], AP2[cat][1] - AP1[cat][1]])
    for cat in categories:
        improvements[cat] = np.array(improvements[cat]) * 100.0
        print('%s: mAP-G %.4f, AP50-G %.4f' % (cat, improvements[cat][:, 0].mean(), improvements[cat][:, 1].mean()))


def inference_throughput(args):
    mapper = torch.load(os.path.join(args.ckpts_dir, '%s.mapper.pth' % args.ckpts_tag))
    print(mapper)
    cfg = get_cfg_base_model("r101-fpn-3x")
    model = YOLOServer.create_from_sup(load_model(os.path.join(os.path.dirname(__file__),'../../configs/yolov3-custom.cfg')), mapper['budget'])
    model.video_id_to_index = mapper['video_id_to_index']
    model.used_indices = mapper['used_indices']
    model.un_used_indices = mapper['un_used_indices']
    state_dict = torch.load(os.path.join(args.ckpts_dir, '%s.pth' % args.ckpts_tag))
    model.load_state_dict(state_dict)
    del state_dict
    dataset = SemiRandomClient(cfg)
    dataset.images = list(filter(lambda x: x['video_id'] == args.id, dataset.images))
    dataset.images = sorted(dataset.images, key=lambda x: x['file_name'])[:10]
    dataset.preload()
    gc.collect()
    torch.cuda.empty_cache()
    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            model.inference([dataset[i % len(dataset)][0]])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, default='server', help='option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--config', type=str, default='../../configs/yolov5l.yaml', help='detection model config path')
    parser.add_argument('--budget', type=int)
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--id', type=str, default='', help='video ID')
    parser.add_argument('--ckpt', type=str, default="../../models/yolov5l_remap.pth", help='weights checkpoint of model')

    parser.add_argument('--incremental_videos', type=bool, default=False)
    parser.add_argument('--train_whole', type=bool, default=False)

    # buffer / replay / cluster parameters
    parser.add_argument('--clustering', type=bool, default=False)
    parser.add_argument('--cluster_cap_factor', type=float, default=1.5)
    parser.add_argument('--buffer_size', type=int, default=10)
    parser.add_argument('--update_cluster_interval', type=int, default=500)
    parser.add_argument('--p_buffer', type=float, default=0.1)
    parser.add_argument('--p_replay', type=float, default=0.25)
    parser.add_argument('--split_list', type=int, nargs='+')

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
    parser.add_argument('--lr', default=0.00334, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)

    # used for random test
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    # if not os.access(finetune_output, os.W_OK):
    #     os.mkdir(finetune_output)
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    finetune_output = args.outputdir
    # assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'adapt':
        if args.ddp_num_gpus <= 1:
            adapt(args)
        else:
            from detectron2.engine import launch
            launch(adapt, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    if args.opt == 'server':
        assert args.instances > 0
        if args.instances == 1:
            simulate(args)
        else:
            simulate_concurrent(args)
    if args.opt == 'tp':
        inference_throughput(args)


'''
python inference_server_simulate.py --model r101-fpn-3x --ckpts_dir --ckpts_tag
python inference_server_simulate.py --model r101-fpn-3x --instances 2 --ckpt
python inference_server_simulate.py --opt tp --id 001 --model r101-fpn-3x --ckpts_dir --ckpts_tag

python inference_server_simulate_yolov5.py --train_whole 1 --opt adapt --model r101-fpn-3x --ckpt ../../models/yolov5l_remap.pth --clustering 1 --update_cluster_interval 10005 --buffer_size 40 --incremental_videos 1 --tag seq.cluster.budget1 --budget 10 --iters 180000 --eval_interval 2000 --save_interval 2000 --image_batch_size 6 --num_workers 4 --p_replay 0.75 --outputdir ./yolov5_vanilla
'''
