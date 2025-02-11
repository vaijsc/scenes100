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
from detectron2.structures import BoxMode
from detectron2.structures import ImageList, Instances

import logging
import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_wdiff_earlyfusion')

from finetune import refine_annotations, get_annotation_dict, finetune_simple_trainer_run_step


# assume both images are 3-channels BGR
# def construct_image_w_background(image, image_background):
#     def image_threshold_inplace(image, thres, min_val, max_val):
#         image[np.where(image < thres)] = min_val
#         image[np.where(image >= thres)] = max_val
#     diff = (image.astype(np.float16) - image_background).mean(axis=2) # float16, [-255, 255]
#     image_diff = np.zeros_like(image)
#     image_diff[:, :, 0] = np.absolute(diff).astype(np.uint8)
#     image_diff[:, :, 1] = ((diff + 255) * 0.5).astype(np.uint8)
#     image_diff[:, :, 2] = image_diff[:, :, 0].copy()
#     image_threshold_inplace(image_diff[:, :, 2], 25, 0, 255)
#     return image, image_background, image_diff

def construct_image_w_background(image, image_background):
    image_diff = (image.astype(np.float16) - image_background) # float16, [-255, 255]
    image_diff = ((image_diff + 255) * 0.5).astype(np.uint8)
    return image, image_background, image_diff


# wrap detectron2/detectron2/data/dataset_mapper.py:DatasetMapper
class DatasetMapperBackground(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        # additional channels
        image_background = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
        assert image_background.shape == image.shape
        image, image_background, image_diff = construct_image_w_background(image, image_background)
        # print(dataset_dict['file_name'], dataset_dict['file_name_background'], image.shape, image.dtype, image_diff.shape, image_diff.dtype)
        # plt.figure(); plt.subplot(2, 2, 1); plt.imshow(image); plt.subplot(2, 2, 2); plt.imshow(image_background); plt.subplot(2, 2, 3); plt.imshow(image_diff); plt.tight_layout(); plt.show()

        # USER: Remove if you don't do semantic/panoptic segmentation.
        sem_seg_gt = None
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        # image_background = transforms.apply_image(image_background)
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
    def create_from_sup(mapper):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperBackground
        return mapper


# wrap detectron2/modeling/backbone/fpn.py:FPN
class FPNFinetuneBackground(detectron2.modeling.backbone.FPN):
    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.backbone.FPN), 'net is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.bottom_up, detectron2.modeling.backbone.ResNet), 'only support detectron2.modeling.backbone.ResNet backbone'
        input_conv = net.bottom_up.stem.conv1
        # expect: 3 -> 64, 7x7, stride 2x2, padding 3x3, no bias
        assert input_conv.bias is None and input_conv.in_channels == 3 and input_conv.out_channels == 64 and list(input_conv.kernel_size) == [7, 7] and list(input_conv.stride) == [2, 2] and list(input_conv.padding) == [3, 3]
        input_conv.in_channels = 6
        input_conv.weight.data = torch.cat([input_conv.weight.data, copy.deepcopy(input_conv.weight.data)], dim=1) / 2.0 # duplicate conv weights
        # net.bottom_up.stem.conv1 will not be in the optimizer. for a pretrained model, it is the very first layer, so it should not affect the result much.
        return net


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNFinetuneBackground(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        '''
        Normalize, pad and batch the input images.
        '''
        # print(batched_inputs[0]['image'].size(), batched_inputs[0]['image'].dtype, self.pixel_mean, self.pixel_std, self.backbone.size_divisibility)
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [torch.cat([(x[0:3] - self.pixel_mean) / self.pixel_std, (x[3:6] - self.pixel_mean) / self.pixel_std], dim=0) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        # print(images[0].size(), images[0].dtype)
        # print(self.backbone.bottom_up.stem.conv1.weight.data[0, 0, 0, 0].item(), self.backbone.bottom_up.stem.conv1.weight.data[0, 3, 0, 0].item(), self.backbone.bottom_up.res2[0].conv1.weight.data[0, 5, 0, 0].item(), self.roi_heads.box_head.fc1.weight.data[0, 0].item())
        return images

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.__class__ = GeneralizedRCNNFinetuneBackground
        net.backbone = FPNFinetuneBackground.create_from_sup(net.backbone)
        return net


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneBackgroundTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)

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
        return COCOEvaluator(dataset_name, output_dir=finetune_output)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        loader = detectron2.data.build_detection_test_loader(cfg, dataset_name)
        assert isinstance(loader.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
        loader.dataset._map_func._obj = DatasetMapperBackground.create_from_sup(loader.dataset._map_func._obj)
        return loader


def train_base(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap_wdiff_earlyfusion', get_coco_dicts(_args, 'valid')
    desc_cocotrain, dst_cocotrain = 'mscoco2017_train_remap_wdiff_earlyfusion', get_coco_dicts(_args, 'train')
    if args.coco_inpaint_type == 'mask':
        for im in dst_cocovalid:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint', 'val2017', os.path.basename(im['file_name'])))
        for im in dst_cocotrain:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint', 'train2017', os.path.basename(im['file_name'])))
    elif args.coco_inpaint_type == 'box':
        for im in dst_cocovalid:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'val2017', os.path.basename(im['file_name'])))
        for im in dst_cocotrain:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
    else:
        raise NotImplementedError
    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_cocotrain, lambda: dst_cocotrain)
    MetadataCatalog.get(desc_cocotrain).thing_classes = thing_classes

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
    cfg.DATASETS.TRAIN = (desc_cocotrain,)
    cfg.DATASETS.TEST = (desc_cocovalid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = FinetuneBackgroundTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackground.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj)
    trainer.resume_or_load(resume=False)
    assert isinstance(trainer.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
    trainer.model = GeneralizedRCNNFinetuneBackground.create_from_sup(trainer.model) # must be after resume_or_load()

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'mscoco2017_remap_wdiff_earlyfusion%s_%s' % ('' if args.coco_inpaint_type == 'mask' else '_boxinpaint', args.model)
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
    aps[iter_list[0]] = aps[iter_list[0]][desc_cocovalid]
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


def all_pseudo_manual_annotations_with_background(args):
    random.seed(42)
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    dict_json_all, count_bboxes_all, annotations_all, id_back = [], 0, [], args.id
    for v in video_id_list:
        args.id = v
        background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
        background_frame_idx = list(map(lambda x: os.path.basename(x), background_files))
        background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), background_frame_idx)))

        manual_v = get_annotation_dict(args)
        for im in manual_v:
            im['file_name_background'] = background_files[-1]
        annotations_all = annotations_all + manual_v

        dict_json_v, count_bboxes_v = refine_annotations(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            count_bboxes_v *= images_per_video_cap / len(dict_json_v)
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])

        for im in dict_json_v:
            i = os.path.basename(im['file_name'])
            i = int(i[:i.find('.')])
            im['file_name_background'] = background_files[np.absolute(background_frame_idx - i).argmin()]
        dict_json_all.append(dict_json_v)
        count_bboxes_all += count_bboxes_v
    args.id = id_back
    for i in range(0, len(annotations_all)):
        annotations_all[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(annotations_all), sum(list(map(lambda x: len(x['annotations']), annotations_all)))))
    print('all videos %d images, %d refine bboxes' % (sum(map(len, dict_json_all)), count_bboxes_all))
    return dict_json_all, annotations_all


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap_wdiff_earlyfusion', get_coco_dicts(_args, 'valid')
    if args.coco_inpaint_type == 'mask':
        for im in dst_cocovalid:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'val2017', os.path.basename(im['file_name'])))
    elif args.coco_inpaint_type == 'box':
        for im in dst_cocovalid:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'val2017', os.path.basename(im['file_name'])))
    else:
        raise NotImplementedError
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]

    if args.id in video_id_list:
        background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
        background_frame_idx = list(map(lambda x: os.path.basename(x), background_files))
        background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), background_frame_idx)))
        desc_manual_valid, dst_manual_valid = '%s_manual_wdiff_earlyfusion' % args.id, get_annotation_dict(args)
        for im in dst_manual_valid:
            im['file_name_background'] = background_files[-1] # choice of background images here does not affect training

        desc_pseudo_anno = 'refine_' + '_'.join(args.anno_models)
        dst_pseudo_anno = refine_annotations(args)[0]
        for im in dst_pseudo_anno:
            i = os.path.basename(im['file_name'])
            i = int(i[:i.find('.')])
            im['file_name_background'] = background_files[np.absolute(background_frame_idx - i).argmin()]
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(_args, 'train')
            if args.coco_inpaint_type == 'mask':
                for im in dst_cocotrain:
                    im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'train2017', os.path.basename(im['file_name'])))
            elif args.coco_inpaint_type == 'box':
                for im in dst_cocotrain:
                    im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
            else:
                raise NotImplementedError
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
        desc_pseudo_anno = desc_pseudo_anno + '_wdiff_earlyfusion'

    elif args.id == 'compound':
        import functools
        args.id = '_compound'
        desc_manual_valid, desc_pseudo_anno = '%s_manual' % args.id, 'refine_' + '_'.join(args.anno_models)
        dst_pseudo_anno, dst_manual_valid = all_pseudo_manual_annotations_with_background(args)
        dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(_args, 'train')
            if args.coco_inpaint_type == 'mask':
                for im in dst_cocotrain:
                    im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'train2017', os.path.basename(im['file_name'])))
            elif args.coco_inpaint_type == 'box':
                for im in dst_cocotrain:
                    im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_box', 'train2017', os.path.basename(im['file_name'])))
            else:
                raise NotImplementedError
            dst_cocotrain = dst_cocotrain * (len(dst_pseudo_anno) // len(dst_cocotrain) + 1)
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
        desc_pseudo_anno = desc_pseudo_anno + '_wdiff_earlyfusion'
    else:
        raise NotImplementedError

    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_cocovalid, lambda: dst_cocovalid)
    MetadataCatalog.get(desc_cocovalid).thing_classes = thing_classes
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

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
    cfg.DATASETS.TRAIN = (desc_pseudo_anno,)
    cfg.DATASETS.TEST = (desc_manual_valid, desc_cocovalid)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = FinetuneBackgroundTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackground.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj)
    assert isinstance(trainer.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
    trainer.model = GeneralizedRCNNFinetuneBackground.create_from_sup(trainer.model) # must be before resume_or_load()
    trainer.resume_or_load(resume=False)

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt%s_%s_anno_%s%s%s' % (args.id, args.model, desc_pseudo_anno, '' if args.fn_max_samples <= 0 else '_fn%.4f_%d' % (args.fn_min_score, args.fn_max_samples), '' if args.coco_inpaint_type == 'mask' else '_boxinpaint')
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
    dst_list = [desc_cocovalid, desc_manual_valid]
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
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['AP50']) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_list[desc_manual_valid]['mAP']) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'MSCOCO Valid AP50', 'MSCOCO Valid mAP', 'Manual Valid AP50', 'Manual Valid mAP'])
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


class EvaluationDatasetWithBackground(torchdata.Dataset):
    def __init__(self, image_dicts, image_pair_list):
        super(EvaluationDatasetWithBackground, self).__init__()
        assert len(image_dicts) == len(image_pair_list)
        self.image_dicts = image_dicts
        self.image_pair_list = image_pair_list
    def __len__(self):
        return len(self.image_pair_list)
    def __getitem__(self, i):
        f1, f2 = self.image_pair_list[i]
        image = detectron2.data.detection_utils.read_image(f1, format='BGR')
        image_background = detectron2.data.detection_utils.read_image(f2, format='BGR')
        return self.image_dicts[i], construct_image_w_background(image, image_background)
    @staticmethod
    def collate(batch):
        return batch


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorBackground(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNFinetuneBackground.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format
    def __call__(self, original_image, image_diff):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            assert self.input_format == 'BGR'
            height, width = original_image.shape[:2]
            tf = self.aug.get_transform(original_image)
            image = torch.as_tensor(tf.apply_image(original_image).astype('float32').transpose(2, 0, 1))
            image_diff = torch.as_tensor(tf.apply_image(image_diff).astype('float32').transpose(2, 0, 1))
            inputs = {'image': torch.cat([image, image_diff], dim=0), 'height': height, 'width': width}
            predictions = self.model([inputs])[0]
            return predictions


def evaluate(args):
    import contextlib
    import tempfile
    from evaluation import evaluate_masked, evaluate_cocovalid

    inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id))
    temp_fpath = None
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)
    if args.eval_background == 'last':
        background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
        for im in images:
            im['file_name_background'] = background_files[-1]
    elif args.eval_background == 'average':
        background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
        background_files = np.stack(list(map(skimage.io.imread, background_files)), axis=0)
        average_background = background_files.astype(np.float32).mean(axis=0).astype(np.uint8)
        _, temp_fpath = tempfile.mkstemp(suffix='.png', text=False)
        skimage.io.imsave(temp_fpath, average_background)
        for im in images:
            im['file_name_background'] = temp_fpath
    elif args.eval_background == 'dynamic':
        for im in images:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_background_lmdb', args.id, 'inpaint', im['file_name'] + '_inpaint.jpg'))
    else: raise Exception('unsupported background type: ' + args.eval_background)
    # for im in images: print(im['file_name'], im['file_name_background'])

    assert args.ckpt is not None and os.access(args.ckpt, os.R_OK)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorBackground(cfg)

    results = {}
    detections = []
    loader = torchdata.DataLoader(
        EvaluationDatasetWithBackground(
            copy.deepcopy(images),
            [(os.path.join(inputdir, 'unmasked', im['file_name']), im['file_name_background']) for im in images]
        ),
        batch_size=None, collate_fn=EvaluationDatasetWithBackground.collate, shuffle=False, num_workers=4
    )
    for im, (image, _, image_diff) in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting %s validation frames' % args.id):
        det = copy.deepcopy(im)
        det['annotations'] = []
        instances = detector(image, image_diff)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        for i in range(0, len(label)):
            det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results['manual_%s' % args.id] = evaluate_masked(args.id, detections, outputfile=args.eval_outputfile)

    if not args.eval_skip_coco:
        detections = get_coco_dicts(args, 'valid')
        for im in detections:
            im['file_name_background'] = os.path.normpath(os.path.join(os.path.dirname(im['file_name']), '..', '..', 'inpaint_mask', 'val2017', os.path.basename(im['file_name'])))
        for im in tqdm.tqdm(detections, ascii=True, desc='detecting MSCOCO2017 valid'):
            image, image_background, image_diff = read_images(im['file_name'], im['file_name_background'])
            instances = detector(image, image_diff)['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            im['annotations'] = []
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results['mscoco2017_valid'] = evaluate_cocovalid(args.cocodir, detections)
    if not temp_fpath is None:
        with open(temp_fpath, 'w') as fp:
            fp.write('')

    print(vars(args))
    for dst in results:
        print('\n            %s\n' % dst)
        print(   '             %s' % '/'.join(results[dst]['metrics']))
        for c in sorted(results[dst]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[dst]['results'][c])))
    return vars(args), results


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_background_lmdb', args.id, 'inpaint', '*inpaint.jpg'))))
    im_bg_arr = detectron2.data.detection_utils.read_image(background_files[-1], format='BGR')
    del background_files
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorBackground(cfg)
    images_tensor = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        _, _, im_diff_arr = construct_image_w_background(im_arr, im_bg_arr)
        tf = detector.aug.get_transform(im_arr)
        im_arr = torch.as_tensor(tf.apply_image(im_arr).astype('float32').transpose(2, 0, 1))
        im_diff_arr = torch.as_tensor(tf.apply_image(im_diff_arr).astype('float32').transpose(2, 0, 1))
        images_tensor.append(torch.cat([im_arr, im_diff_arr], dim=0))
    N1, N2 = 100, 900
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            detector.model.inference([{'image': images_tensor[i % len(images)], 'height': images[i % len(images)]['height'], 'width': images[i % len(images)]['width']}])
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')

    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')

    parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--smallscale', default=False, type=bool)
    parser.add_argument('--coco_inpaint_type', type=str, default='mask', choices=['mask', 'box'], help='inpaint MSCOCO2017 objects using mask/bbox annotation')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    parser.add_argument('--fn_min_score', type=float, default=0.99, help='minimum objectiveness score of false negatives')
    parser.add_argument('--fn_max_samples', type=int, default=-1, help='maximum number of false negatives per frame')
    parser.add_argument('--fn_max_samples_det_p', type=float, default=0.5, help='maximum number of false negatives per frame as percentage of number of detections')
    parser.add_argument('--fn_min_area', type=float, default=50, help='minimum area of false negative boxes')
    parser.add_argument('--fn_max_width_p', type=float, default=0.3333, help='maximum percentage width of false negative boxes')
    parser.add_argument('--fn_max_height_p', type=float, default=0.3333, help='maximum percentage height of false negative boxes')

    parser.add_argument('--eval_background', type=str, default='', choices=['', 'dynamic', 'last', 'average'], help='use inference time dynamic background or last training time background')
    parser.add_argument('--compare_ckpt_dir', type=str)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--refine_visualize_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    parser.add_argument('--eval_outputfile', default=None, type=str)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    args = parser.parse_args()
    args.anno_models = sorted(list(set(args.anno_models)))
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'adapt':
        adapt(args)
    elif args.opt == 'base':
        if args.ddp_num_gpus <= 1:
            train_base(args)
        else:
            from detectron2.engine import launch
            launch(train_base, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    elif args.opt == 'eval':
        evaluate(args)
    elif args.opt == 'tp':
        inference_throughput(args)
    else: pass
    exit(0)


'''
conda deactivate && conda activate detectron2
cd /nfs/detection/zekun/Intersections/scripts/baseline

python finetune_wdiff_earlyfusion.py --opt base --model r50-fpn-3x --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 160 --image_batch_size 2
python finetune_wdiff_earlyfusion.py --opt base --model r50-fpn-3x --cocodir ../../../MSCOCO2017 --iters 40000 --eval_interval 3000 --image_batch_size 4 --num_workers 4

python finetune_wdiff_earlyfusion.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_earlyfusion_r50-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 160 --image_batch_size 2
python finetune_wdiff_earlyfusion.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_earlyfusion_r50-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1800 --image_batch_size 4 --num_workers 4



python finetune_wdiff_earlyfusion.py --opt base --model r101-fpn-3x --coco_inpaint_type box --cocodir ../../../MSCOCO2017 --iters 40000 --eval_interval 3000 --image_batch_size 4 --num_workers 4
python finetune_wdiff_earlyfusion.py --opt adapt --id 001 --model r101-fpn-3x --coco_inpaint_type box --anno_models r50-fpn-3x r101-fpn-3x --ckpt mscoco2017_remap_wdiff_earlyfusion_boxinpaint_r101-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 140 --eval_interval 80 --image_batch_size 2



python finetune_wdiff_earlyfusion.py --opt eval --id 001 --model r50-fpn-3x --ckpt adapt001_r50-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_earlyfusion.pth --cocodir ../../../MSCOCO2017 --eval_background last
python finetune_wdiff_earlyfusion.py --opt eval --id 001 --model r50-fpn-3x --ckpt adapt001_r50-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_earlyfusion.pth --eval_skip_coco 1 --eval_background dynamic

python finetune_wdiff_earlyfusion.py --opt compare --model r101-fpn-3x --compare_ckpt_dir E:\intersections_results\background_r101 --eval_background dynamic

'''
