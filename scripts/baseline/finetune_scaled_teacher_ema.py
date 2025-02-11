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
from sklearn.mixture import GaussianMixture

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode, Instances

import logging
import weakref
from finetune import get_annotation_dict, all_annotation_dict
from finetune_scaled_teacher import get_datasets

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_scaled_teacher_ema')


class FinetuneEMATrainer(DefaultTrainer):
    def __init__(self, cfg, train_partial=False, train_partial_input=False):
        super(DefaultTrainer, self).__init__()
        assert not (train_partial and train_partial_input)
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.model_teacher = copy.deepcopy(model)
        self.model_teacher.eval()

        if train_partial or train_partial_input:
            if train_partial:
                trainable_modules = [
                    model.backbone.bottom_up.stem,
                    model.backbone.bottom_up.res2,
                    model.proposal_generator,
                    model.roi_heads.box_predictor,
                ]
            else:
                trainable_modules = [
                    model.backbone.bottom_up.stem,
                    model.backbone.bottom_up.res2,
                ]
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
        else:
            optimizer = self.build_optimizer(cfg, model)
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
        self._trainer.ema_interval = cfg.SOLVER.EMA_INTERVAL
        self._trainer.ema_eta = cfg.SOLVER.EMA_ETA
        self._trainer.pseudo_det_min_score = cfg.SOLVER.PSEUDO_DET_MIN_SCORE

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
            pseudo_labels = self.model_teacher.inference(pseudo_inputs)
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

    # EMA update
    if self.iter % self.ema_interval == 1:
        self.optimizer.zero_grad()
        with torch.no_grad():
            sd_teacher = self.model_teacher.state_dict()
            sd_student = self.model.state_dict()
            for k in list(sd_teacher.keys()):
                p1, p2 = sd_teacher[k], sd_student[k]
                p1 = p1 * self.ema_eta + p2 * (1.0 - self.ema_eta)
                sd_teacher[k] = p1
            self.model_teacher.load_state_dict(sd_teacher)

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
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
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
        scale = np.random.rand() * 0.75 + 1.5 # upscale by 1.5 ~ 2.25
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


def teach_by_scaled(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    _args.smallscale = False
    _args.anno_scales = ['2.0']
    _args.ensemble = 'refine'
    _args.refine_iou_thres = 0.85
    _args.train_on_coco = True
    desc_manual_valid, dst_manual_valid, _, dst_pseudo_anno = get_datasets(_args)
    for im in dst_pseudo_anno:
        if im['source'] == 'psuedo':
            if np.random.rand() < args.ema_online_p:
                im['source'] = 'unlabeled'
                im['annotations'] = [{'bbox': [100, 100, 200, 200], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': 0}]
    desc_pseudo_anno = args.id + '_unlabeled_x2_cocotrain'
    del _tensor, _args
    gc.collect()
    DatasetCatalog.register(desc_manual_valid, lambda: dst_manual_valid)
    MetadataCatalog.get(desc_manual_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_pseudo_anno, lambda: dst_pseudo_anno)
    MetadataCatalog.get(desc_pseudo_anno).thing_classes = thing_classes

    if args.ckpt is not None and os.access(args.ckpt, os.R_OK):
        print('loading checkpoint:', args.ckpt)
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    else:
        cfg = get_cfg_base_model(args.model)
    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.RANDOM_FLIP = 'none'
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
    cfg.DATASETS.TEST = (desc_manual_valid,)
    cfg.SOLVER.EMA_INTERVAL = args.ema_interval
    cfg.SOLVER.EMA_ETA = args.ema_eta
    cfg.SOLVER.PSEUDO_DET_MIN_SCORE = args.refine_det_score_thres
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneEMATrainer(cfg, args.train_partial, args.train_partial_input)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_ema_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperPseudo.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, cfg)
    trainer.resume_or_load(resume=False)
    assert trainer.model is trainer._trainer.model
    assert trainer.model_teacher is trainer._trainer.model_teacher
    trainer.model_teacher.load_state_dict(trainer.model.state_dict())

    prefix = 'distill_%s_%s_ema_anno_%s' % (args.model, args.id, desc_pseudo_anno)
    if args.train_partial:
        prefix = prefix + '.partial'
    if args.train_partial_input:
        prefix = prefix + '.partial.input'
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

    m_student = trainer.model
    if isinstance(m_student, torch.nn.DataParallel) or isinstance(m_student, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m_student = m_student.module
    m_teacher = trainer.model_teacher
    if isinstance(m_teacher, torch.nn.DataParallel) or isinstance(m_teacher, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m_teacher = m_teacher.module
    for m, _postfix in [(m_student, '.student'), (m_teacher, '.teacher')]:
        if (not args.train_partial) and (not args.train_partial_input):
            torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + _postfix + '.pth'))
        else:
            if args.train_partial:
                trainable_modules = {
                    'backbone.bottom_up.stem': m.backbone.bottom_up.stem.state_dict(),
                    'backbone.bottom_up.res2': m.backbone.bottom_up.res2.state_dict(),
                    'proposal_generator': m.proposal_generator.state_dict(),
                    'roi_heads.box_predictor': m.roi_heads.box_predictor.state_dict(),
                }
                torch.save(trainable_modules, os.path.join(args.outputdir, prefix + _postfix + '.pth'))
            else:
                trainable_modules = {
                    'backbone.bottom_up.stem': m.backbone.bottom_up.stem.state_dict(),
                    'backbone.bottom_up.res2': m.backbone.bottom_up.res2.state_dict(),
                }
                torch.save(trainable_modules, os.path.join(args.outputdir, prefix + _postfix + '.pth'))

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
    # correlation(); exit()
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--train_partial', type=bool, default=False)
    parser.add_argument('--train_partial_input', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)

    # parser.add_argument('--anno_models', nargs='+', default=[])
    # parser.add_argument('--anno_scales', type=str, nargs='+', choices=['1', '1.5', '2.0', '2.5'], default=[])
    # parser.add_argument('--ensemble', type=str, choices=['refine', 'union'], default='refine')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    # parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    # parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    # parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    # parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    parser.add_argument('--ema_online_p', type=float, default=0.5)
    parser.add_argument('--ema_interval', type=int, default=10)
    parser.add_argument('--ema_eta', type=float, default=0.998)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--refine_visualize_workers', default=0, type=int)
    # parser.add_argument('--eval_skip_coco', default=False, type=bool)
    # parser.add_argument('--eval_outputfile', default=None, type=str)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    assert not (args.train_partial and args.train_partial_input)
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    assert os.path.isdir(args.outputdir)
    assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'distill':
        teach_by_scaled(args)
    if args.opt == 'eval':
        evaluate_all_videos(args)
    else:
        pass


'''
python finetune_scaled_teacher_ema.py --id 001 --opt distill --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 1 --image_batch_size 2 --iters 450 --eval_interval 151
python finetune_scaled_teacher_ema.py --id coco --opt distill --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 1 --image_batch_size 2 --iters 450 --eval_interval 151
'''
