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
from utils import IoU, DummyWriter, bbox_inside, intersect_ratios, count_parameters
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_wdiff_midfusion_mixup')

from finetune import refine_annotations, get_annotation_dict, finetune_simple_trainer_run_step
from finetune_wdiff_earlyfusion import all_pseudo_manual_annotations_with_background, construct_image_w_background
from finetune_wdiff_midfusion import GeneralizedRCNNFinetuneBackground, FinetuneBackgroundTrainer


# wrap detectron2/detectron2/data/dataset_mapper.py:DatasetMapper
class DatasetMapperBackgroundMixup(detectron2.data.DatasetMapper):
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
        if 'mixup_src_images' in dataset_dict and random.uniform(0.0, 1.0) < self.mixup_p:
            mixup_src_dict = dataset_dict['mixup_src_images'][random.randrange(0, len(dataset_dict['mixup_src_images']))]
            src_image = detectron2.data.detection_utils.read_image(mixup_src_dict['file_name'], format=self.image_format)
            assert src_image.shape == image.shape
            src_annotations = mixup_src_dict['annotations']
            random.shuffle(src_annotations)
            src_annotations = src_annotations[: max(1, int(self.mixup_r * len(src_annotations)))]
            for ann in src_annotations:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = map(int, ann['bbox'])
                x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
                image[y1 : y2, x1 : x2] = src_image[y1 : y2, x1 : x2]
            annotations_trimmed = []
            for ann in dataset_dict['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                _trim = False
                for ann2 in src_annotations:
                    if intersect_ratios(ann['bbox'], ann2['bbox'])[0] >= self.mixup_overlap_thres or bbox_inside(ann['bbox'], ann2['bbox']):
                        _trim = True
                        break
                if not _trim:
                    annotations_trimmed.append(ann)
            for ann in src_annotations:
                annotations_trimmed.append(ann)
            dataset_dict['annotations'] = annotations_trimmed

        detectron2.data.detection_utils.check_image_size(dataset_dict, image)

        # additional channels
        image_background = detectron2.data.detection_utils.read_image(dataset_dict['file_name_background'], format=self.image_format)
        assert image_background.shape == image.shape
        image, image_background, image_diff = construct_image_w_background(image, image_background)

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
        # print(dataset_dict['file_name'], dataset_dict['file_name_background'], dataset_dict['mixup_src_images'] if 'mixup_src_images' in dataset_dict else '[]', image.shape, image.dtype)
        # plt.figure(); plt.subplot(1, 2, 1); plt.imshow(image[:, :, :3]); plt.subplot(1, 2, 2); plt.imshow(image[:, :, 3:]); plt.tight_layout(); plt.show()
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
    def create_from_sup(mapper, mixup_p, mixup_r, mixup_overlap_thres):
        assert isinstance(mapper, detectron2.data.DatasetMapper), 'mapper is not detectron2.data.DatasetMapper'
        mapper.__class__ = DatasetMapperBackgroundMixup
        mapper.mixup_p, mapper.mixup_r, mapper.mixup_overlap_thres = mixup_p, mixup_r, mixup_overlap_thres
        return mapper


def adapt(args):
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    _args = copy.deepcopy(args)
    desc_cocovalid, dst_cocovalid = 'mscoco2017_valid_remap_wdiff_midfusion_mixup', get_coco_dicts(_args, 'valid')
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
        desc_manual_valid, dst_manual_valid = '%s_manual_wdiff_midfusion_mixup' % args.id, get_annotation_dict(args)
        for im in dst_manual_valid:
            im['file_name_background'] = background_files[-1] # choice of background images here does not affect training

        desc_pseudo_anno = 'refine_' + '_'.join(args.anno_models)
        dst_pseudo_anno = refine_annotations(args)[0]
        # include sample mixup sources, this increases RAM usage
        dst_pseudo_anno_copy = copy.deepcopy(dst_pseudo_anno)
        for im in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources'):
            for _ in range(0, 3):
                im['mixup_src_images'] = [dst_pseudo_anno_copy[random.randrange(0, len(dst_pseudo_anno_copy))]]
        del dst_pseudo_anno_copy

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
        desc_pseudo_anno = desc_pseudo_anno + '_wdiff_midfusion'

    elif args.id == 'compound':
        import functools
        args.id = '_compound'
        desc_manual_valid, desc_pseudo_anno = '%s_manual_wdiff_midfusion' % args.id, 'refine_' + '_'.join(args.anno_models)
        dst_pseudo_anno, dst_manual_valid = all_pseudo_manual_annotations_with_background(args)
        # include sample mixup sources, this increases RAM usage
        for dst_v in tqdm.tqdm(dst_pseudo_anno, ascii=True, desc='populating mixup sources'):
            dst_v_copy = copy.deepcopy(dst_v)
            for im in tqdm.tqdm(dst_v, ascii=True):
                for _ in range(0, 3):
                    im['mixup_src_images'] = [dst_v_copy[random.randrange(0, len(dst_v_copy))]]
            del dst_v_copy
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
            random.shuffle(dst_cocotrain)
            dst_pseudo_anno = dst_pseudo_anno + dst_cocotrain[:len(dst_pseudo_anno)]
            desc_pseudo_anno = desc_pseudo_anno + '_cocotrain'
            print('include MSCOCO2017 training images, totally %d images' % len(dst_pseudo_anno))
        for i in range(0, len(dst_pseudo_anno)):
            dst_pseudo_anno[i]['image_id'] = i + 1
        desc_pseudo_anno = desc_pseudo_anno + '_wdiff_midfusion'
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
    detectron2.evaluation.evaluator.evaluate_interval_n = 200
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 200

    trainer = FinetuneBackgroundTrainer(cfg, args.fusion_type, args.multitask_loss_alpha)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    assert isinstance(trainer.data_loader.dataset.dataset.dataset._map_func._obj, detectron2.data.DatasetMapper), 'mapper class mismatch'
    trainer.data_loader.dataset.dataset.dataset._map_func._obj = DatasetMapperBackgroundMixup.create_from_sup(trainer.data_loader.dataset.dataset.dataset._map_func._obj, args.mixup_p, args.mixup_r, args.mixup_overlap_thres)
    trainer.resume_or_load(resume=False)

    results_0 = {}
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0[dataset_name] = inference_on_dataset(trainer.model, data_loader, evaluator)
    trainer.eval_results_all[0] = results_0
    trainer.train()

    prefix = 'adapt%s_%s_anno_%s%s%s%s_mixup' % (args.id, args.model, desc_pseudo_anno, '' if args.fn_max_samples <= 0 else '_fn%.4f_%d' % (args.fn_min_score, args.fn_max_samples), trainer.model.fusion_desc, '' if args.coco_inpaint_type == 'mask' else '_boxinpaint')
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


from finetune_wdiff_midfusion import evaluate


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

    parser.add_argument('--mixup_p', type=float, default=0.3, help='probability of applying mixup to an image')
    parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')

    parser.add_argument('--fusion_type', type=str, choices=['average', 'conv', 'attn'], default='average', help='feature pyramids fusion method')
    parser.add_argument('--multitask_loss_alpha', type=float, default=0.5, help='relative weight of multitasking losses')
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
    args = parser.parse_args()
    args.anno_models = sorted(list(set(args.anno_models)))
    assert 0 <= args.multitask_loss_alpha <= 1, str(args.multitask_loss_alpha)
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)

    if args.opt == 'adapt':
        adapt(args)
    elif args.opt == 'eval':
        evaluate(args)
    else: pass
    exit(0)


'''

python finetune_wdiff_midfusion_mixup.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusionconv_r50-fpn-3x.pth --fusion_type conv --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 300 --eval_interval 160 --image_batch_size 2 --num_workers 3

python finetune_wdiff_midfusion_mixup.py --opt eval --id 001 --model r50-fpn-3x --ckpt adapt001_r50-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_w_background.pth --cocodir ../../../MSCOCO2017 --eval_background last
python finetune_wdiff_midfusion_mixup.py --opt eval --id 001 --model r50-fpn-3x --ckpt adapt001_r50-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_w_background.pth --eval_skip_coco 1 --eval_background dynamic

python finetune_wdiff_midfusion_mixup.py --opt compare --model r101-fpn-3x --compare_ckpt_dir E:\intersections_results\background_r101 --eval_background dynamic

'''
