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
from detectron2.structures import BoxMode, Boxes
from detectron2.structures import ImageList, Instances
from fvcore.nn import sigmoid_focal_loss_jit

import logging
import weakref
from finetune_retinanet_distill import get_unlabeled_dicts, all_unlabeled_dicts
from finetune_faster_rcnn_distill import GeneralizedRCNNDistillation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base_detector_train import get_coco_dicts
from models import get_cfg_base_model


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_distill_faster_rcnn_with_roi')


# wrap detectron2/engine/defaults.py:DefaultTrainer
class DistillationTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        model = GeneralizedRCNNDistillationROI.create_from_sup(model)
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


def distillation_trainer_run_step(self):
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


def train_distillation(args):
    from finetune import get_annotation_dict
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    if args.id == 'coco':
        desc_valid, dst_valid, id_back = 'compound_manual_downsample', [], args.id
        for v in video_id_list:
            args.id = v
            dst_valid.extend(get_annotation_dict(args)[:5])
        for i in range(0, len(dst_valid)):
            dst_valid[i]['image_id'] = i + 1
        args.id = id_back
        desc_train, dst_train = 'mscoco2017_train_remap', get_coco_dicts(args, 'train')
    elif args.id in video_id_list:
        desc_valid, dst_valid = '%s_manual' % args.id, get_annotation_dict(args)
        desc_train, dst_train = '%s_unlabeled_coco' % args.id, get_unlabeled_dicts(args)
        dst_cocotrain = get_coco_dicts(args, 'train')
        random.shuffle(dst_cocotrain)
        dst_train.extend(dst_cocotrain[: len(dst_train) // 3])
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    elif args.id == 'compound':
        desc_valid, dst_valid, id_back = 'compound_manual', [], args.id
        for v in video_id_list:
            args.id = v
            dst_valid.extend(get_annotation_dict(args))
        for i in range(0, len(dst_valid)):
            dst_valid[i]['image_id'] = i + 1
        args.id = id_back
        desc_train, dst_train = 'compound_unlabeled_coco', all_unlabeled_dicts(args, args.image_batch_size * args.iters * 1.25)
        dst_cocotrain = get_coco_dicts(args, 'train')
        random.shuffle(dst_cocotrain)
        dst_train.extend(dst_cocotrain[: len(dst_train) // 3])
        for i in range(0, len(dst_train)):
            dst_train[i]['image_id'] = i + 1
    else:
        raise NotImplementedError
    del _tensor
    gc.collect()

    DatasetCatalog.register(desc_valid, lambda: dst_valid)
    MetadataCatalog.get(desc_valid).thing_classes = thing_classes
    DatasetCatalog.register(desc_train, lambda: dst_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes

    assert (args.ckpt is not None) and os.access(args.ckpt, os.R_OK)
    ckpt_roi_teacher = os.path.join(finetune_output, os.path.basename(args.ckpt) + '_roi_teacher_%.6f.pth' % time.time())
    print('copy teacher ROI head:', args.ckpt, '=>', ckpt_roi_teacher)
    state_dict = torch.load(args.ckpt)
    keys = list(state_dict.keys())
    for k in keys:
        if k.startswith('roi_heads.'):
            k_teacher = k.replace('roi_heads.', 'roi_heads_teacher.')
            if not k_teacher in state_dict:
                state_dict[k_teacher] = copy.deepcopy(state_dict[k])
    torch.save(state_dict, ckpt_roi_teacher)
    del state_dict
    cfg = get_cfg_base_model(args.model, ckpt=ckpt_roi_teacher)

    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters * 2 // 5, args.iters * 4 // 5)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_valid,)
    print(cfg)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    prefix = 'distill_%s_%s_with_roi' % (args.model, args.id)
    trainer = DistillationTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(distillation_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)
    with open(ckpt_roi_teacher, 'w') as fp:
        fp.write('empty') # reduce disk usage

    results_0 = {}
    assert len(trainer.cfg.DATASETS.TEST) == 1
    print('Evaluate on %s' % trainer.cfg.DATASETS.TEST[0])
    data_loader = trainer.build_test_loader(trainer.cfg, trainer.cfg.DATASETS.TEST[0])
    evaluator = trainer.build_evaluator(trainer.cfg, trainer.cfg.DATASETS.TEST[0])
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


class GeneralizedRCNNDistillationROI(GeneralizedRCNNDistillation):
    def _forward_roi(self, features, proposals, roi_heads):
        features = [features[f] for f in roi_heads.box_in_features]
        box_features = roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = roi_heads.box_head(box_features)
        logits, proposal_deltas = roi_heads.box_predictor(box_features)
        return box_features, logits, proposal_deltas

    def forward_distillation_loss_roi(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert self.training
        assert not torch.jit.is_scripting(), 'Not supported'

        images = self.preprocess_image(batched_inputs)
        images.tensor = torch.nn.functional.interpolate(images.tensor, scale_factor=2, mode='bilinear', align_corners=True)
        B = images.tensor.size(0)
        with torch.no_grad():
            features_x2 = self.backbone_teacher(images.tensor)
            # "pretend" image is upscaled x2 for anchors to match
            proposal_generator_teacher_training = self.proposal_generator_teacher.training
            self.proposal_generator_teacher.training = False # RPN in training state requires GT boxes
            images.image_sizes = [(2 * h, 2 * w) for (h, w) in images.image_sizes]
            proposals_x2, _ = self.proposal_generator_teacher(images, features_x2, None)
            self.proposal_generator_teacher.training = proposal_generator_teacher_training
            box_features_x2, logits_x2, proposal_deltas_x2 = self._forward_roi(features_x2, proposals_x2, self.roi_heads_teacher)

            # requiring gradient on student FPN causes VRAM OOM, too complicated to apply cropping on ROI pooling
            features = self.backbone(images.tensor)
            features = self._add_p1(features)
        box_features, logits, proposal_deltas = self._forward_roi(features, proposals_x2, self.roi_heads)

        # losses
        losses = {}
        losses['distill_roi_pool'] = torch.nn.functional.mse_loss(box_features, box_features_x2.detach(), reduction='none').mean()
        losses['distill_roi_box'] = torch.nn.functional.mse_loss(proposal_deltas, proposal_deltas_x2.detach(), reduction='none').mean()
        log_scores, log_scores_x2 = torch.nn.functional.log_softmax(logits, dim=1), torch.nn.functional.log_softmax(logits_x2, dim=1)
        losses['distill_roi_cls'] = torch.nn.functional.kl_div(log_scores, log_scores_x2.detach(), reduction='none', log_target=True).sum(dim=1).mean()
        return losses

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        else:
            # alternate between training RPN / ROI
            self.train_alternate_idx = (self.train_alternate_idx + 1) % 2
            if self.train_alternate_idx == 0:
                return self.forward_distillation_loss(batched_inputs)
            else:
                return self.forward_distillation_loss_roi(batched_inputs)

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        net.backbone_teacher = copy.deepcopy(net.backbone) # teacher model uses x2 input, 2x2 stride
        net.proposal_generator_teacher = copy.deepcopy(net.proposal_generator)
        assert not hasattr(net, 'roi_heads_teacher'), 'teacher ROI head already exists'
        net.roi_heads_teacher = copy.deepcopy(net.roi_heads)
        net.backbone.bottom_up.stem.conv1.stride = (4, 4) # student model uses x2 input, 4x4 stride
        # for generating p1 feature, stem feature has same spatial resolution as res2 so is not useful
        net.p2_to_p1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
            torch.nn.Upsample(scale_factor=2.0, mode='nearest'), torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        ).to(net.backbone.bottom_up.stem.conv1.weight.device)
        net.feature_shift_map = {
            'p2': 'p3',
            'p3': 'p4',
            'p4': 'p5',
            'p5': 'p6'
        }
        net.__class__ = GeneralizedRCNNDistillationROI
        net.train_alternate_idx = 0
        return net


class PredictorDistillation(DefaultPredictor):
    def __init__(self, cfg, anchor_downsample):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNDistillationROI.create_from_sup(self.model)
        self.model.backbone_teacher, self.model.proposal_generator_teacher, self.model.roi_heads_teacher = None, None, None # teacher is not used in inference
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        self.checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format
        self.anchor_downsample = anchor_downsample

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            assert self.input_format == 'BGR'
            height, width = original_image.shape[:2]
            tf = self.aug.get_transform(original_image)
            image = torch.as_tensor(tf.apply_image(original_image).astype('float32').transpose(2, 0, 1))
            return self.model.inference([{'image': image, 'height': height, 'width': width}], anchor_downsample=self.anchor_downsample)[0]


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
        cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
        detector = PredictorDistillation(cfg, args.anchor_downsample)
        video_ckpts = [(v, None) for v in video_id_list]
    else:
        assert not args.ckpts_dir is None
        results_file = os.path.join(args.ckpts_dir, 'results_AP%s' % args.tag)
        print(results_file)
        video_ckpts = sorted(glob.glob(os.path.join(args.ckpts_dir, 'distill_r101*.pth')))
        cfg = get_cfg_base_model(args.model, ckpt=video_ckpts[0])
        detector = PredictorDistillation(cfg, args.anchor_downsample)
        video_ckpts = [(os.path.basename(f)[20 : 23], f) for f in video_ckpts]
        print('%d presented video checkpoints:' % len(video_ckpts))
        print(' '.join([v for (v, _) in video_ckpts]))
        print('missing:')
        print(' '.join(sorted(list(set(video_id_list) - set([v for (v, _) in video_ckpts])))))

    t_total, N_total = 0, 0
    for video_i, (video_id, f) in enumerate(video_ckpts):
        if not f is None:
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


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorDistillation(cfg, args.anchor_downsample)

    inputs_list = []
    for im in images:
        im_arr = detectron2.data.detection_utils.read_image(os.path.join(inputdir, 'unmasked', im['file_name']), format='BGR')
        im_tensor = detector.aug.get_transform(im_arr).apply_image(im_arr)
        im_tensor = torch.as_tensor(im_tensor.astype('float32').transpose(2, 0, 1))
        inputs_list.append([{'image': im_tensor, 'height': im['height'], 'width': im['width']}])
    stats_all = {}
    N1, N2 = 100, 400
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N2 + N1), ascii=True):
            if i == N1: t = time.time()
            if i == N2: t = time.time() - t
            detector.model.inference(inputs_list[i % len(images)], anchor_downsample=args.anchor_downsample)
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


def correlation():
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i1.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_x1 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\feature_scaling\\results_AP_resscale_i2.00_f1.00_r101-fpn-3x.json', 'r') as fp:
        results_x2 = json.load(fp)
    with open('F:\\intersections_results\\cvpr24\\distill_faster_rcnn_roi_continue\\results_AP.json', 'r') as fp:
        results_distill = json.load(fp)
    APs_x1 = np.array([results_x1[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_x2 = np.array([results_x2[v]['results']['weighted'] for v in video_id_list]) * 100
    APs_distill = np.array([results_distill[v]['results']['weighted'] for v in video_id_list]) * 100
    print('x1:', APs_x1.mean(), 'x2:', APs_x2.mean(), 'distilled individual:', APs_distill.mean())
    plt.figure(figsize=(5, 5))
    apg_50 = APs_distill[:, 1] - APs_x1[:, 1]
    apg_m = APs_distill[:, 0] - APs_x1[:, 0]
    plt.scatter(APs_x2[:, 1] - APs_x1[:, 1], apg_50, marker='+', c='b', alpha=0.8)
    plt.scatter(APs_x2[:, 0] - APs_x1[:, 0], apg_m, marker='x', c='r', alpha=0.8)
    plt.xlabel('x2 input improvement'); plt.ylabel('distilled individually improvement'); plt.legend(['$AP50 \\uparrow distilled %+.2f$' % apg_50.mean(), '$mAP \\uparrow distilled %+.2f$' % apg_m.mean()])
    plt.xlim(-20, 37); plt.ylim(-20, 37); plt.grid(True)
    plt.tight_layout()
    plt.savefig('AP_faster_rcnn_distill_roi_correlation.pdf')


if __name__ == '__main__':
    # correlation(); exit()
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound', 'coco'], help='video ID')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--anchor_downsample', type=bool, default=False)
    parser.add_argument('--ckpts_dir', type=str, default=None, help='weights checkpoints of individual models')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--not_save_results_json', type=bool, default=False)

    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--smallscale', default=False, type=bool)

    parser.add_argument('--iters', type=int, help='total training iterations')
    parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    parser.add_argument('--image_batch_size', default=4, type=int)
    parser.add_argument('--roi_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    parser.add_argument('--eval_outputfile', default=None, type=str)
    parser.add_argument('--hold', default=0.005, type=float)
    parser.add_argument('--ddp_num_gpus', type=int, default=1)
    args = parser.parse_args()
    print(args)


    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'distill':
        if args.ddp_num_gpus <= 1:
            train_distillation(args)
        else:
            from detectron2.engine import launch
            launch(train_distillation, args.ddp_num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args,))
    if args.opt == 'tp':
        inference_throughput(args)
    if args.opt == 'eval':
        evaluate_all_videos(args)

'''
python finetune_faster_rcnn_roi_distill.py --model r101-fpn-3x --opt distill --ckpt mscoco2017_remap_r101-fpn-3x_student_teacher.pth --id compound --cocodir ../../../MSCOCO2017 --iters 60000 --eval_interval 5001 --image_batch_size 8 --num_workers 8
python finetune_faster_rcnn_roi_distill.py --model r101-fpn-3x --opt distill --ckpt mscoco2017_remap_r101-fpn-3x_student_teacher.pth --id coco --cocodir ../../../MSCOCO2017 --iters 60000 --eval_interval 5001 --image_batch_size 8 --num_workers 8

python finetune_faster_rcnn_roi_distill.py --model r101-fpn-3x --opt distill --ckpt distill_r101-fpn-3x_001.pth --id 001 --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1500 --image_batch_size 4 --num_workers 4

python finetune_faster_rcnn_roi_distill.py --model r101-fpn-3x --opt eval --ckpt distill_r101-fpn-3x_compound_coco_300000.pth
python finetune_faster_rcnn_roi_distill.py --model r101-fpn-3x --opt eval --ckpts_dir cvpr24/distill_faster_rcnn_roi
python finetune_faster_rcnn_roi_distill.py --model r101-fpn-3x --opt tp --ckpt distill_r101-fpn-3x_compound_coco_300000.pth --id 001
'''
