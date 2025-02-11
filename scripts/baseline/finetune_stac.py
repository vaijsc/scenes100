#!python3

'''
our implementation of
https://arxiv.org/pdf/2005.04757.pdf

official repo
https://github.com/google-research/ssl_detection

no tracking, no bbox refining, only self-training
set pseudo box confidence score threshold=0.9

This file should be more self-contained
'''

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


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_stac')


def gather_annotations(args):
    dst = TrainingFrames(args.id)
    imagedir = os.path.join(dst.lmdb_path, 'jpegs')
    det_file = os.path.join(dst.lmdb_path, 'detect_%s.json.gz' % args.model)
    assert os.access(det_file, os.R_OK), '%s not readable' % det_file
    dict_json, count_det_raw, count_det = [], 0, 0
    for i in range(0, len(dst)):
        dict_json.append({'file_name': os.path.join(imagedir, dst.ifilelist[i]), 'image_id': i, 'height': dst.meta['meta']['video']['H'], 'width': dst.meta['meta']['video']['W'], 'annotations': [], 'det_count': 0, 'sot_count': 0})
    with gzip.open(det_file, 'rt') as fp:
        dets = json.loads(fp.read())['dets']
    assert len(dets) == len(dict_json), 'detection & dataset mismatch'
    for i in range(0, len(dets)):
        count_det_raw += len(dets[i]['score'])
        for j in range(0, len(dets[i]['score'])):
            if dets[i]['score'][j] < args.det_score_thres:
                continue
            dict_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'det_score': dets[i]['score'][j]})
            count_det += 1
    print('finish reading from detection results of', args.id)
    print('%d images, detections %d => %d (score >= %.2f)' % (len(dict_json), count_det_raw, count_det, args.det_score_thres), flush=True)
    return dict_json, count_det

def all_pseudo_annotations(args):
    random.seed(42)
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    dict_json_all, count_bboxes_all, id_back = [], 0, args.id
    for v in video_id_list:
        args.id = v
        dict_json_v, count_bboxes_v = gather_annotations(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            count_bboxes_v *= images_per_video_cap / len(dict_json_v)
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all.append(dict_json_v)
        count_bboxes_all += count_bboxes_v
    args.id = id_back
    print('all videos %d images, %d bboxes' % (sum(map(len, dict_json_all)), count_bboxes_all), flush=True)
    return dict_json_all, count_bboxes_all


def get_annotation_dict(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))), flush=True)
    return annotations

def all_annotation_dict(args):
    annotations_all, id_back = [], args.id
    for v in video_id_list:
        args.id = v
        annotations_all = annotations_all + get_annotation_dict(args)
    args.id = id_back
    for i in range(0, len(annotations_all)):
        annotations_all[i]['image_id'] = i + 1
    print('manual annotation for all videos: %d images, %d bboxes' % (len(annotations_all), sum(list(map(lambda x: len(x['annotations']), annotations_all)))), flush=True)
    return annotations_all


# based on https://github.com/google-research/ssl_detection/blob/master/detection/utils/augmentation.py
# no color jitters because it is already performed by detectron2 dataloader
# also apply affine transformation on 1/4 of the bounding boxes instead of only 1
class RandomAugmentBBox(object):
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmenters.geometric import Affine

    CUTOUT = iaa.Cutout(nb_iterations=(1, 5), size=[0, 0.2], squared=True) # for image size == 800, 0.1 is 80.
    DEGREE = 30
    AFFINE_TRANSFORM = iaa.Sequential([
        iaa.OneOf([
            Affine(translate_percent={'x': (-0.1, 0.1)}, order=[0, 1], cval=125),
            Affine(translate_percent={'y': (-0.1, 0.1)}, order=[0, 1], cval=125),
            Affine(rotate=(-DEGREE, DEGREE), order=[0, 1], cval=125),
            Affine(shear=(-DEGREE, DEGREE), order=[0, 1], cval=125),
        ])], random_order=True)
    AFFINE_TRANSFORM_WEAK = iaa.Sequential([
        iaa.OneOf([
            Affine(translate_percent={'x': (-0.05, 0.05)}, order=[0, 1], cval=125),
            Affine(translate_percent={'y': (-0.05, 0.05)}, order=[0, 1], cval=125),
            Affine(rotate=(-10, 10), order=[0, 1], cval=125),
            Affine(shear=(-10, 10), order=[0, 1], cval=125),
        ])], random_order=True)

    def bbox_affine_transform(self, image, bounding_boxes):
        N = bounding_boxes.shape[0]
        if N < 1:
            return image, bounding_boxes
        k_list = set(np.random.choice(np.arange(0, N), max(1, N // 4)))
        image_aug = image.copy()
        for k in k_list:
            bb = bounding_boxes[k]
            # print('bbox affine', bb)
            im_crop = image[int(bb[1]) : int(bb[3]), int(bb[0]) : int(bb[2])].copy()
            im_paste = self.AFFINE_TRANSFORM_WEAK(images=[im_crop])[0]
            image_aug[int(bb[1]) : int(bb[3]), int(bb[0]) : int(bb[2])] = im_paste
        assert image.shape == image_aug.shape and image.dtype == image_aug.dtype
        return image_aug, bounding_boxes, None

    def affine_transform(self, image, bounding_boxes):
        image_aug, bbs_aug = self.AFFINE_TRANSFORM(images=[image], bounding_boxes=np.expand_dims(bounding_boxes, axis=0))
        image_aug, bbs_aug = image_aug[0], bbs_aug[0]
        assert image.shape == image_aug.shape and image.dtype == image_aug.dtype
        assert bounding_boxes.shape == bbs_aug.shape and bounding_boxes.dtype == bbs_aug.dtype
        return image_aug, bbs_aug, None

    def cutout_augment(self, image, bounding_boxes=None):
        image_aug = self.CUTOUT(image=image)
        assert image.shape == image_aug.shape and image.dtype == image_aug.dtype
        return image_aug, bounding_boxes, None

    def __init__(self, aug_type='strong', magnitude=10, weighted_inbox_selection=False):
        self.magnitude = magnitude
        self.aug_type = aug_type
        self.weighted_inbox_selection = weighted_inbox_selection
        self.augment_fn = [[self.bbox_affine_transform, self.affine_transform], [self.cutout_augment]]

    def __call__(self, image, bounding_boxes):
        for fns in self.augment_fn:
            fn = fns[np.random.randint(0, len(fns))]
            image, bounding_boxes, _ = fn(image, bounding_boxes)
        return image, bounding_boxes


######################################################
#####   many RCNN library methods are modified   #####
##### modded RCNN only tested on detectron2 v0.6 #####
#####  with models: R50-FPN, R101-FPN, X101-FPN  #####
######################################################

# wrap detectron2/detectron2/data/dataset_mapper.py:DatasetMapper
class DatasetMapperStrongAugmentation(detectron2.data.DatasetMapper):
    def __call__(self, dataset_dict):
        if not dataset_dict['strongaug']:
            # print('weak augmentations for', dataset_dict['file_name'])
            # import matplotlib.patches as patches
            # _, axes = plt.subplots(1, 2); axes = axes.reshape(-1)
            # _im=skimage.io.imread(dataset_dict['file_name']); axes[0].imshow(_im)
            # print(list(map(lambda x: x['bbox'], dataset_dict['annotations'])))
            # for ann in dataset_dict['annotations']:
            #     assert ann['bbox_mode'] == BoxMode.XYWH_ABS
            #     x1, y1, w, h = ann['bbox']
            #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='k', facecolor='none')
            #     axes[0].add_patch(rect)
            # axes[0].set_title('%s %s %s %.2f %.2f' % (dataset_dict['file_name'], _im.dtype, _im.shape, _im.min(), _im.max()))

            dataset_dict = super(DatasetMapperStrongAugmentation, self).__call__(dataset_dict)

            # _im=dataset_dict['image'].numpy().transpose(1, 2, 0); axes[1].imshow(_im)
            # print(dataset_dict['instances'].gt_boxes.tensor)
            # for x1, y1, x2, y2 in dataset_dict['instances'].gt_boxes.tensor.numpy().tolist():
            #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', facecolor='none')
            #     axes[1].add_patch(rect)
            # axes[1].set_title('weak aug %s %s %.2f %.2f' % (_im.dtype, _im.shape, _im.min(), _im.max())); plt.show()

            return dataset_dict

        # print('strong augmentations for', dataset_dict['file_name'])
        # import matplotlib.patches as patches
        # _, axes = plt.subplots(2, 2); axes = axes.reshape(-1)
        # _im=skimage.io.imread(dataset_dict['file_name']); axes[0].imshow(_im)
        # print(list(map(lambda x: x['bbox'], dataset_dict['annotations'])))
        # for ann in dataset_dict['annotations']:
        #     assert ann['bbox_mode'] == BoxMode.XYXY_ABS
        #     x1, y1, x2, y2 = ann['bbox']
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', facecolor='none')
        #     axes[0].add_patch(rect)
        # axes[0].set_title('%s %s %s %.2f %.2f' % (dataset_dict['file_name'], _im.dtype, _im.shape, _im.min(), _im.max()))

        # "default" image augmentation, which includes color augmentations
        dataset_dict = copy.deepcopy(dataset_dict)
        image = detectron2.data.detection_utils.read_image(dataset_dict['file_name'], format=self.image_format)
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)
        aug_input = detectron2.data.transforms.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2] # h, w
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if 'annotations' in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        # _im=dataset_dict['image'].numpy().transpose(1, 2, 0); axes[1].imshow(_im)
        # print(dataset_dict['instances'].gt_boxes.tensor)
        # for x1, y1, x2, y2 in dataset_dict['instances'].gt_boxes.tensor.numpy().tolist():
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', facecolor='none')
        #     axes[1].add_patch(rect)
        # axes[1].set_title('default aug %s %s %.2f %.2f' % (_im.dtype, _im.shape, _im.min(), _im.max()))

        # strong augmentation including global affine, bbox affine, cut-out
        image_np = dataset_dict['image'].numpy().transpose(1, 2, 0)
        bboxes_np = dataset_dict['instances'].gt_boxes.tensor.numpy()
        image_strong_aug, bboxes_strong_aug = self.strong_augmentor(image_np, bboxes_np)
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1)))
        dataset_dict['instances'].gt_boxes.tensor = torch.as_tensor(np.ascontiguousarray(bboxes_strong_aug))

        # _im=dataset_dict['image'].numpy().transpose(1, 2, 0); axes[2].imshow(_im)
        # print(dataset_dict['instances'].gt_boxes.tensor)
        # for x1, y1, x2, y2 in dataset_dict['instances'].gt_boxes.tensor.numpy().tolist():
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', facecolor='none')
        #     axes[2].add_patch(rect)
        # axes[2].set_title('strong aug %s %s %.2f %.2f' % (_im.dtype, _im.shape, _im.min(), _im.max())); plt.show()

        return dataset_dict

    @staticmethod
    def create_from_sup(mapper):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperStrongAugmentation
        mapper.strong_augmentor = RandomAugmentBBox()
        return mapper


# DefaultTrainer._trainer is instance of SimpleTrainer
# DefaultTrainer & SimpleTrainer are subclass of TrainerBase
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
    self.optimizer.step()

    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneTrainer(DefaultTrainer):
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


def train_eval(trainer, prefix, args):
    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    _args.smallscale = False
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap', get_coco_dicts(_args, 'valid')
    if args.not_eval_coco:
        print('use dummy MSCOCO2017-validation during training')
        dst_cocovalid = dst_cocovalid[:5] + dst_cocovalid[-5:]

    if args.id in video_id_list:
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, get_annotation_dict(args)
        desc_pseudo_anno = 'STAC'
        dst_pseudo_anno = gather_annotations(args)[0]
        for im in dst_pseudo_anno:
            im['strongaug'] = True
        random.seed(42)
        dst_cocotrain = get_coco_dicts(_args, 'train')
        for im in dst_cocotrain:
            im['strongaug'] = False
        random.shuffle(dst_cocotrain)
        dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
        desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
        print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
    elif args.id == 'compound':
        import functools
        args.id = '_compound'
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, all_annotation_dict(args)
        desc_pseudo_anno = 'STAC'
        dst_pseudo_anno = all_pseudo_annotations(args)[0]
        dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
        for im in dst_pseudo_anno:
            im['strongaug'] = True
        random.seed(42)
        dst_cocotrain = get_coco_dicts(_args, 'train')
        for im in dst_cocotrain:
            im['strongaug'] = False
        dst_cocotrain = dst_cocotrain * (len(dst_pseudo_anno) // len(dst_cocotrain) + 1)
        random.shuffle(dst_cocotrain)
        dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
        desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
        print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
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

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model(args.model)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output

    # disable random flipping & cropping
    cfg.INPUT.RANDOM_FLIP = 'none'
    cfg.INPUT.CROP.ENABLED = False

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

    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperStrongAugmentation.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj)
    trainer.resume_or_load(resume=False)

    prefix = 'adapt%s_%s_%s' % (args.id, args.model, desc_pseudo_anno)
    train_eval(trainer, prefix, args)
    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return

    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = aps.keys()
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
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')

    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--det_score_thres', type=float, default=0.9, help='minimum detection score in pseudo annotation')
    parser.add_argument('--unsupervised_loss_alpha', type=float, default=1, help='weight of unsupervised loss')

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--hold', default=0.005, type=float)

    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'adapt':
        if args.ddp_num_gpus <= 1:
            adapt(args)
        else:
            from detectron2.engine import launch
            launch(adapt, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='tcp://127.0.0.1:%d' % args.ddp_port, args=(args,))
    else:
        pass
    exit(0)


'''
conda deactivate && conda activate detectron2
'''
