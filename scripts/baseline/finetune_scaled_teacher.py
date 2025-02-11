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
from detectron2.structures import BoxMode

import logging
import weakref
from finetune import _graph_refine, all_pseudo_annotations, get_annotation_dict, all_annotation_dict, finetune_simple_trainer_run_step

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_scaled_teacher')


def refine_annotations(args, visualize=False):
    dst = TrainingFrames(args.id)
    imagedir = os.path.join(dst.lmdb_path, 'jpegs')
    labeldir_s1 = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label'))
    labeldir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label_scaling'))
    det_filelist = []
    for s in args.anno_scales:
        if s == '1':
            det_filelist.append(os.path.join(labeldir_s1, '%s_detect_r101-fpn-3x.json.gz' % args.id))
        else:
            det_filelist.append(os.path.join(labeldir, '%s_detect_r101-fpn-3x_s%s.json.gz' % (args.id, s)))
    for f in det_filelist:
        assert os.access(f, os.R_OK), '%s not readable' % f

    # collate bboxes from tracking & detection
    dict_json = []
    for i in range(0, len(dst)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(imagedir, dst.ifilelist[i])), 'image_id': i, 'height': dst.meta['meta']['video']['H'], 'width': dst.meta['meta']['video']['W'], 'annotations': [], 'source': 'psuedo', 'det_count': 0, 'sot_count': 0})

    for f in det_filelist:
        print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
        with gzip.open(f, 'rt') as fp:
            dets = json.loads(fp.read())['dets']
        assert len(dets) == len(dict_json), 'detection & dataset mismatch'
        for i in range(0, len(dets)):
            for j in range(0, len(dets[i]['score'])):
                if dets[i]['score'][j] < args.refine_det_score_thres:
                    continue
                dict_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'src': 'det', 'score': dets[i]['score'][j]})
                dict_json[i]['det_count'] += 1
    print('finish reading from detection results')

    count_all = {'all': 0, 'det': 0, 'sot': 0}
    for annotations in dict_json:
        count_all['det'] += annotations['det_count']
        count_all['sot'] += annotations['sot_count']
        count_all['all'] += len(annotations['annotations'])
        # annotations['annotations'] = list(filter(lambda ann: min(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) >= min_bbox_width, annotations['annotations']))
        # annotations['annotations'] = list(filter(lambda ann: max(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) <= max_bbox_width, annotations['annotations']))
    print('pseudo annotations: detection %d, tracking %d, total %d' % (count_all['det'], count_all['sot'], count_all['all']))

    pool = ProcessPool(processes=6)
    params_list, chunksize, i = [], len(dict_json) // 20, 0
    while True:
        dict_json_chunk = dict_json[i * chunksize : (i + 1) * chunksize]
        if len(dict_json_chunk) < 1:
            break
        params_list.append({'dict': dict_json_chunk, 'args': args})
        i += 1
    for i in range(0, len(params_list)):
        params_list[i]['desc'] = '%02d/%02d' % (i + 1, len(params_list))
    refine_results = pool.map_async(_graph_refine, params_list).get()
    pool.close()
    pool.join()
    dict_json, count_bboxes = [], 0
    for r in refine_results:
        dict_json = dict_json + r[0]
        count_bboxes += r[1]
    print('%d images, refine bboxes %d => %d' % (len(dict_json), count_all['all'], count_bboxes))
    return dict_json, count_bboxes


def all_pseudo_annotations(args):
    random.seed(42)
    images_per_video_cap = int(args.iters * args.image_batch_size / len(video_id_list))
    dict_json_all, count_bboxes_all, id_back = [], 0, args.id
    for v in video_id_list:
        args.id = v
        dict_json_v, count_bboxes_v = refine_annotations(args)
        if len(dict_json_v) > images_per_video_cap:
            print('randomly drop images: %d => %d' % (len(dict_json_v), images_per_video_cap))
            count_bboxes_v *= images_per_video_cap / len(dict_json_v)
            random.shuffle(dict_json_v)
            dict_json_v = dict_json_v[:images_per_video_cap]
            dict_json_v.sort(key=lambda x: x['file_name'])
        dict_json_all.append(dict_json_v)
        count_bboxes_all += count_bboxes_v
    args.id = id_back
    print('all videos %d images, %d refine bboxes' % (sum(map(len, dict_json_all)), count_bboxes_all))
    return dict_json_all, count_bboxes_all


class FinetuneTrainer(DefaultTrainer):
    def __init__(self, cfg, train_partial=False, train_partial_input=False):
        super(DefaultTrainer, self).__init__()
        assert not (train_partial and train_partial_input)
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
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


def get_datasets(args):
    if args.id in video_id_list:
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, get_annotation_dict(args)
        desc_pseudo_anno = args.id + '_' + '_'.join(['x' + s for s in args.anno_scales])
        if args.ensemble == 'refine':
            desc_pseudo_anno = desc_pseudo_anno + '_refine'
            dst_pseudo_anno = refine_annotations(args)[0]
        elif args.ensemble == 'union':
            desc_pseudo_anno = desc_pseudo_anno + '_union'
            dst_pseudo_anno = []
            _scales_back = args.anno_scales
            for s in _scales_back:
                args.anno_scales = [s]
                dst_pseudo_anno.extend(refine_annotations(args)[0])
            args.anno_scales = _scales_back
            random.shuffle(dst_pseudo_anno)
            dst_pseudo_anno = dst_pseudo_anno[: len(dst_pseudo_anno) // len(_scales_back)]
            print('downsample to %d images' % len(dst_pseudo_anno))
        else:
            raise NotImplementedError
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(args, 'train')
            for im in dst_cocotrain:
                im['source'] = 'coco'
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: len(dst_pseudo_anno) // 3]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
            del dst_cocotrain
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1

    elif args.id == 'compound':
        import functools
        args.id = '_compound'
        desc_manual_valid, dst_manual_valid = '%s_manual' % args.id, all_annotation_dict(args)
        desc_pseudo_anno = args.id + '_' + '_'.join(['x' + s for s in args.anno_scales])
        if args.ensemble == 'refine':
            desc_pseudo_anno = desc_pseudo_anno + '_refine'
            dst_pseudo_anno = all_pseudo_annotations(args)[0]
            dst_pseudo_anno = functools.reduce(lambda x, y: x + y, dst_pseudo_anno)
        elif args.ensemble == 'union':
            desc_pseudo_anno = desc_pseudo_anno + '_union'
            dst_pseudo_anno = []
            _scales_back = args.anno_scales
            for s in _scales_back:
                args.anno_scales = [s]
                _dst = all_pseudo_annotations(args)[0]
                _dst = functools.reduce(lambda x, y: x + y, _dst)
                dst_pseudo_anno.extend(_dst)
            args.anno_scales = _scales_back
            random.shuffle(dst_pseudo_anno)
            dst_pseudo_anno = dst_pseudo_anno[: len(dst_pseudo_anno) // len(_scales_back)]
            print('downsample to %d images' % len(dst_pseudo_anno))
        else:
            raise NotImplementedError
        if args.train_on_coco:
            random.seed(42)
            dst_cocotrain = get_coco_dicts(args, 'train')
            for im in dst_cocotrain:
                im['source'] = 'coco'
            dst_cocotrain = dst_cocotrain * (len(dst_pseudo_anno) // len(dst_cocotrain) + 1)
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: len(dst_pseudo_anno) // 3]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
            del dst_cocotrain
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1

    elif args.id == 'coco':
        desc_manual_valid, dst_manual_valid, id_back = 'compound_manual_downsample', [], args.id
        for v in video_id_list:
            args.id = v
            dst_manual_valid.extend(get_annotation_dict(args)[:5])
        for i in range(0, len(dst_manual_valid)):
            dst_manual_valid[i]['image_id'] = i + 1
        args.id = id_back
        desc_pseudo_anno, dst_pseudo_anno, count_det = 'cocotrain_x2', [], 0
        coco_x2_filelist = glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label_scaling', 'mscoco_train2017_detect_r101-fpn-3x_s2.0_*.json.gz')))
        for f in coco_x2_filelist:
            print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
            with gzip.open(f, 'rt') as fp:
                results = json.loads(fp.read())
            for im, det in zip(results['frames'], results['dets']):
                im['file_name'] = os.path.normpath(os.path.join(args.cocodir, 'images', 'train2017', im['file_name']))
                im['annotations'] = []
                for j in range(0, len(det['score'])):
                    if det['score'][j] < args.refine_det_score_thres:
                        continue
                    im['annotations'].append({'bbox': det['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': det['label'][j], 'src': 'det', 'score': det['score'][j]})
                    count_det += 1
                dst_pseudo_anno.append(im)
        print('MSCOCO-2017 training set x2 pseudo labels: %d images, %d bboxes' % (len(dst_pseudo_anno), count_det))
        random.seed(42)
        dst_cocotrain = get_coco_dicts(args, 'train')
        for im in dst_cocotrain:
            im['source'] = 'coco'
        random.shuffle(dst_cocotrain)
        dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[: len(dst_pseudo_anno) // 2]
        desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
        print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        del dst_cocotrain
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
    else:
        raise NotImplementedError
    return desc_manual_valid, dst_manual_valid, desc_pseudo_anno, dst_pseudo_anno


def teach_by_scaled(args):
    assert len(args.anno_scales) > 0
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    _args.smallscale = False
    desc_manual_valid, dst_manual_valid, desc_pseudo_anno, dst_pseudo_anno = get_datasets(_args)
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
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneTrainer(cfg, args.train_partial, args.train_partial_input)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)

    prefix = 'distill_%s_%s_anno_%s' % (args.model, args.id, desc_pseudo_anno)
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
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    if (not args.train_partial) and (not args.train_partial_input):
        torch.save(m.state_dict(), os.path.join(args.outputdir, prefix + '.pth'))
    else:
        if args.train_partial:
            trainable_modules = {
                'backbone.bottom_up.stem': m.backbone.bottom_up.stem.state_dict(),
                'backbone.bottom_up.res2': m.backbone.bottom_up.res2.state_dict(),
                'proposal_generator': m.proposal_generator.state_dict(),
                'roi_heads.box_predictor': m.roi_heads.box_predictor.state_dict(),
            }
            torch.save(trainable_modules, os.path.join(args.outputdir, prefix + '.pth'))
        else:
            trainable_modules = {
                'backbone.bottom_up.stem': m.backbone.bottom_up.stem.state_dict(),
                'backbone.bottom_up.res2': m.backbone.bottom_up.res2.state_dict(),
            }
            torch.save(trainable_modules, os.path.join(args.outputdir, prefix + '.pth'))

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


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        self.checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format


def evaluate_all_videos(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_r101-fpn-3x.json'), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}

    if not args.ckpt is None:
        assert args.ckpts_dir is None
        results_file = '%s_results_AP%s' % (args.ckpt, args.tag)
        print(results_file)
        if args.train_partial or args.train_partial_input:
            cfg = get_cfg_base_model(args.model)
        else:
            cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
        detector = Predictor(cfg)
        video_ckpts = [(v, None) for v in video_id_list]
        video_ckpts[0] = (video_ckpts[0][0], args.ckpt)
    else:
        assert not args.ckpts_dir is None
        results_file = os.path.join(args.ckpts_dir, 'results_AP%s' % args.tag)
        print(results_file)
        video_ckpts = sorted(glob.glob(os.path.join(args.ckpts_dir, 'distill_r101*%s*.pth' % args.ckpts_tag)))
        if len(video_ckpts) < 1:
            print('no ckpts present')
            return
        if args.train_partial or args.train_partial_input:
            cfg = get_cfg_base_model(args.model)
        else:
            cfg = get_cfg_base_model(args.model, ckpt=video_ckpts[0])
        detector = Predictor(cfg)
        video_ckpts = [(os.path.basename(f)[20 : 23], f) for f in video_ckpts]
        print('%d presented video checkpoints:' % len(video_ckpts))
        print(' '.join([v for (v, _) in video_ckpts]))
        print('missing:')
        print(' '.join(sorted(list(set(video_id_list) - set([v for (v, _) in video_ckpts])))))

    t_total, N_total = 0, 0
    for video_i, (video_id, f) in enumerate(video_ckpts):
        if not f is None:
            print('load', f)
            if args.train_partial or args.train_partial_input:
                params = torch.load(f)
                detector.model.backbone.bottom_up.stem.load_state_dict(params['backbone.bottom_up.stem'])
                detector.model.backbone.bottom_up.res2.load_state_dict(params['backbone.bottom_up.res2'])
                if args.train_partial:
                    detector.model.proposal_generator.load_state_dict(params['proposal_generator'])
                    detector.model.roi_heads.box_predictor.load_state_dict(params['roi_heads.box_predictor'])
                del params
            else:
                detector.checkpointer.load(f)
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        N_total += len(images)
        detections = []
        loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [os.path.normpath(os.path.join(inputdir, 'unmasked', im['file_name'])) for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
        torch.cuda.empty_cache()
        t0 = time.time()
        for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting %s validation frames' % video_id):
            det = copy.deepcopy(im)
            det['annotations'] = []
            instances = detector(im_arr)['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            detections.append(det)
        t_total += time.time() - t0
        print('[%d/%d finished in %.1f minutes]\n' % (video_i + 1, len(video_ckpts), t_total / 60))
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        results[video_id]['detections'] = detections
        print(   '             %s' % '/'.join(results[video_id]['metrics']))
        for c in sorted(results[video_id]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[video_id]['results'][c])))
    if not args.not_save_results_json:
        with open(results_file + '.json', 'w') as fp:
            json.dump(results, fp, indent=2)
    print('processed %d images in %.2f seconds, %.3f ms/image' % (N_total, t_total, t_total * 1000 / N_total))

    videos = sorted(list(results.keys()))
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    for video_id in videos:
        AP1 = base_AP['manual_' + video_id]['results']
        AP2 = results[video_id]['results']
        for cat in categories:
            improvements[cat].append([AP2[cat][0] - AP1[cat][0], AP2[cat][1] - AP1[cat][1]])
    for cat in categories:
        improvements[cat] = np.array(improvements[cat]) * 100.0
    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        axes[i].plot([-1, xs.max() + 1], [0, 0], 'k-')
        axes[i].plot(xs, improvements[categories[i]][:, 0], 'r.-')
        axes[i].plot(xs, improvements[categories[i]][:, 1], 'b.-')
        axes[i].legend(['0', 'mAP %.4f' % improvements[categories[i]][:, 0].mean(), 'AP50 %.4f' % improvements[categories[i]][:, 1].mean()])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        # axes[i].set_ylim(-3, 3)
        axes[i].set_ylabel('AP improvement (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    # plt.tight_layout()
    plt.suptitle('%s [%.3f ms/image]' % (results_file, t_total * 1000 / N_total))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(os.path.join(args.outputdir, results_file + '.pdf'))
    plt.close()
    print('saved to:', results_file)


def correlation():
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i1.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i2.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_x2 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\distill_faster_rcnn_x2_teach\\results_AP.json', 'r') as fp:
        results_distill = json.load(fp)
    APs_x1 = np.array([results_x1[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_x2 = np.array([results_x2[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_distill = np.array([results_distill[v]['results']['weighted'] for v in video_id_list]) * 100
    print('x1:', APs_x1.mean(), 'x2:', APs_x2.mean(), 'teached individual:', APs_distill.mean())
    plt.figure(figsize=(5, 5))
    apg_50 = APs_distill[:, 1] - APs_x1[:, 1]
    apg_m = APs_distill[:, 0] - APs_x1[:, 0]
    plt.scatter(APs_x2[:, 1] - APs_x1[:, 1], apg_50, marker='+', c='b', alpha=0.8)
    plt.scatter(APs_x2[:, 0] - APs_x1[:, 0], apg_m, marker='x', c='r', alpha=0.8)
    plt.xlabel('x2 input improvement'); plt.ylabel('x2 teached individually improvement'); plt.legend(['$AP50 \\uparrow x2 teached %+.2f$' % apg_50.mean(), '$mAP \\uparrow x2 teached %+.2f$' % apg_m.mean()])
    plt.xlim(-20, 37); plt.ylim(-20, 37); plt.grid(True)
    plt.tight_layout()
    plt.savefig('AP_faster_rcnn_x2_teach_correlation.pdf')


if __name__ == '__main__':
    # correlation(); exit()
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound', 'coco'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--train_partial', type=bool, default=False)
    parser.add_argument('--train_partial_input', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--ckpts_tag', type=str, default='')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)

    # parser.add_argument('--anno_models', nargs='+', default=[])
    parser.add_argument('--anno_scales', type=str, nargs='+', choices=['1', '1.5', '2.0', '2.5'], default=[])
    parser.add_argument('--ensemble', type=str, choices=['refine', 'union'], default='refine')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    # parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    # parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

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
python finetune_scaled_teacher.py --id 001 --opt distill --model r101-fpn-3x --anno_scales 1 1.5 2.0 2.5 --cocodir ../../../MSCOCO2017 --num_workers 1 --image_batch_size 2 --iters 450 --eval_interval 151 --train_on_coco 1
python finetune_scaled_teacher.py --id coco --opt distill --model r101-fpn-3x --anno_scales 1 1.5 2.0 2.5 --cocodir ../../../MSCOCO2017 --num_workers 1 --image_batch_size 2 --iters 450 --eval_interval 151

python finetune_scaled_teacher.py --model r101-fpn-3x --opt eval --ckpt distill_r101-fpn-3x_001_anno_r101-fpn-3x_x2_cocotrain.pth
python finetune_scaled_teacher.py --model r101-fpn-3x --opt eval --ckpts_dir cvpr24/distill_faster_rcnn_x2_teach
'''
