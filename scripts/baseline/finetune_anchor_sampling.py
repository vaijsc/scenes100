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
import skimage.transform
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
from detectron2.structures import ImageList, Instances, Boxes

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
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_multiscales')

from finetune import refine_annotations, all_pseudo_annotations, get_annotation_dict, all_annotation_dict
from finetune_mixup import DatasetMapperMixup


# wrap detectron2/modeling/meta_arch/rcnn.py:GeneralizedRCNN
class GeneralizedRCNNAnchorSampling(detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN):
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True, anchor_stride: float = 1.0):
        assert not self.training
        assert not 'proposals' in batched_inputs[0], 'pre-computed proposals not supported'
        assert detected_instances is None, 'pre-computed instances not supported'

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None, anchor_stride=anchor_stride)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            return results
        else:
            return results

    @staticmethod
    def create_from_sup(net):
        assert isinstance(net, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'network is not detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'
        assert isinstance(net.backbone, detectron2.modeling.backbone.FPN), 'backbone is not detectron2.modeling.backbone.FPN'
        assert isinstance(net.roi_heads, detectron2.modeling.roi_heads.roi_heads.StandardROIHeads), 'roi is not detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'
        assert isinstance(net.roi_heads.box_predictor, detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers), 'roi is not detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers'
        net.__class__ = GeneralizedRCNNAnchorSampling
        net.proposal_generator.__class__ = RPNAnchorSampling
        return net


# wrap detectron2/modeling/proposal_generator/rpn.py:RPN
class RPNAnchorSampling(detectron2.modeling.proposal_generator.rpn.RPN):
    def _permute_logits_deltas(self, pred_objectness_logits, pred_anchor_deltas):
        # Transpose the Hi*Wi*A dimension to the middle:
        return \
        [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ], \
        [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

    def forward(self, images: ImageList, features: Dict[str, torch.Tensor], gt_instances: Optional[List[Instances]] = None, anchor_stride: float = 1.0):
        features = [features[f] for f in self.in_features]
        anchors = [a.tensor for a in self.anchor_generator(features)] # only the actual tensor is used at inference
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        # sub sample anchors based on anchor_stride
        anchors_sub, pred_objectness_logits_sub, pred_anchor_deltas_sub = [], [], []
        if not self.training:
            for i in range(0, len(pred_objectness_logits)):
                _, A, Hi, Wi = pred_objectness_logits[i].size()
                if anchor_stride - 1.0 < 1 / max(Hi, Wi):
                    pred_objectness_logits_sub.append(pred_objectness_logits[i])
                    pred_anchor_deltas_sub.append(pred_anchor_deltas[i])
                    anchors_sub.append(anchors[i])
                else:
                    mask_Hi = torch.arange(0, Hi, anchor_stride, device=pred_objectness_logits[i].device).long()
                    mask_Wi = torch.arange(0, Wi, anchor_stride, device=pred_objectness_logits[i].device).long()
                    pred_objectness_logits_sub.append(pred_objectness_logits[i][:, :, mask_Hi][:, :, :, mask_Wi].contiguous())
                    pred_anchor_deltas_sub.append(pred_anchor_deltas[i][:, :, mask_Hi][:, :, :, mask_Wi].contiguous())
                    anchors_sub.append(anchors[i].view(Hi, Wi, A, 4)[mask_Hi][:, mask_Wi].view(-1, 4).contiguous())
        anchors, pred_objectness_logits, pred_anchor_deltas = anchors_sub, pred_objectness_logits_sub, pred_anchor_deltas_sub

        pred_objectness_logits, pred_anchor_deltas = self._permute_logits_deltas(pred_objectness_logits, pred_anchor_deltas)

        assert not self.training
        # https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/proposal_generator/proposal_utils.py#L22
        proposals = self.predict_proposals(anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes)
        return proposals, {}

    def predict_proposals(self, anchors: List[torch.Tensor], pred_objectness_logits: List[torch.Tensor], pred_anchor_deltas: List[torch.Tensor], image_sizes: List[Tuple[int, int]]):
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return detectron2.modeling.proposal_generator.proposal_utils.find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[torch.Tensor], pred_anchor_deltas: List[torch.Tensor]):
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


# wrap detectron2/engine/defaults.py:DefaultPredictor
class PredictorScaling(DefaultPredictor):
    def __init__(self, cfg, input_scale, anchor_stride):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = detectron2.modeling.build_model(self.cfg)
        assert isinstance(self.model, detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN), 'model class mismatch'
        self.model = GeneralizedRCNNAnchorSampling.create_from_sup(self.model)
        self.model.eval()
        if len(cfg.DATASETS.TEST): self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = detectron2.data.transforms.ResizeShortestEdge([int(input_scale * cfg.INPUT.MIN_SIZE_TEST), int(input_scale * cfg.INPUT.MIN_SIZE_TEST)], int(input_scale * cfg.INPUT.MAX_SIZE_TEST))
        self.input_scale, self.anchor_stride = input_scale, anchor_stride
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == 'RGB':
                original_image = original_image[:, :, ::-1]
            assert original_image.dtype == np.uint8
            height, width = original_image.shape[:2]
            image_scaleup = original_image
            image = self.aug.get_transform(image_scaleup).apply_image(image_scaleup)
            image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            predictions = self.model.inference([{'image': image, 'height': height, 'width': width}], anchor_stride=self.anchor_stride)
            return predictions[0]


def evaluate_all_videos(args):
    import contextlib
    from evaluation import evaluate_masked, evaluate_cocovalid
    from finetune import EvaluationDataset

    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    results = {}

    results_file = 'results_AP_scaling_i%.2f_a%.2f_%s' % (args.input_scale, args.anchor_stride, args.model)
    print(results_file)
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorScaling(cfg, args.input_scale, args.anchor_stride)

    t_total, N_total = 0, 0
    for video_id in video_id_list:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id)
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        N_total += len(images)
        detections = []
        loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [os.path.join(inputdir, 'unmasked', im['file_name']) for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
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
        print('[%d/%d finished in %.1f minutes]\n' % (video_id_list.index(video_id) + 1, len(video_id_list), t_total / 60))
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        del results[video_id]['raw']
        print(   '             %s' % '/'.join(results[video_id]['metrics']))
        for c in sorted(results[video_id]['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results[video_id]['results'][c])))
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

    if args.eval_skip_coco:
        return
    args.smallscale = False
    images = get_coco_dicts(args, 'valid')
    loader = torchdata.DataLoader(EvaluationDataset(copy.deepcopy(images), [im['file_name'] for im in images]),
            batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
        )
    detections = []
    for im, im_arr in tqdm.tqdm(loader, total=len(images), ascii=True, desc='detecting MSCOCO2017 valid'):
        instances = detector(im_arr)['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        im['annotations'] = []
        for i in range(0, len(label)):
            im['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(im)
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_mscoco2017_valid = evaluate_cocovalid(args.cocodir, detections)
    print(   '             %s' % '/'.join(results_mscoco2017_valid['metrics']))
    for c in sorted(results_mscoco2017_valid['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results_mscoco2017_valid['results'][c])))


def inference_throughput(args):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)[:10]
    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    detector = PredictorScaling(cfg, args.input_scale, args.anchor_stride)

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
            detector.model.inference(inputs_list[i % len(images)], anchor_stride=args.anchor_stride)
    tp = (N2 - N1) / t
    print('%.3f images/s, %.3f ms/image' % (tp, 1000 / tp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--ckpt', type=str, default=None, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.')
    parser.add_argument('--input_scale', type=float, default=1.0)
    parser.add_argument('--anchor_stride', type=float, default=1.0)

    # parser.add_argument('--anno_models', nargs='+', default=[], help='models used for pseudo annotation (detection + tracking)')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    # parser.add_argument('--not_eval_coco', type=bool, default=False, help='skip evaluation on MSCOCO2017 during training')
    # parser.add_argument('--train_on_coco', type=bool, default=False, help='include MSCOCO2017 training images in training')
    # parser.add_argument('--smallscale', default=False, type=bool)
    # parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    # parser.add_argument('--refine_iou_thres', type=float, default=0.85, help='IoU threshold to merge boxes')
    # parser.add_argument('--refine_remove_no_sot', type=bool, default=False, help='remove images without tracking results')

    # parser.add_argument('--fn_min_score', type=float, default=0.99, help='minimum objectiveness score of false negatives')
    # parser.add_argument('--fn_max_samples', type=int, default=-1, help='maximum number of false negatives per frame')
    # parser.add_argument('--fn_max_samples_det_p', type=float, default=0.5, help='maximum number of false negatives per frame as percentage of number of detections')
    # parser.add_argument('--fn_min_area', type=float, default=50, help='minimum area of false negative boxes')
    # parser.add_argument('--fn_max_width_p', type=float, default=0.3333, help='maximum percentage width of false negative boxes')
    # parser.add_argument('--fn_max_height_p', type=float, default=0.3333, help='maximum percentage height of false negative boxes')

    # parser.add_argument('--mixup_p', type=float, default=0.3, help='probability of applying mixup to an image')
    # parser.add_argument('--mixup_r', type=float, default=0.5, help='ratio of mixed-up bounding boxes')
    # parser.add_argument('--mixup_overlap_thres', type=float, default=0.65, help='above this threshold, overwritten boxes by mixup are removed')
    # parser.add_argument('--mixup_random_position', type=bool, default=False, help='randomly position patch')

    # parser.add_argument('--iters', type=int, help='total training iterations')
    # parser.add_argument('--eval_interval', type=int, help='interval for evaluation')
    # parser.add_argument('--image_batch_size', default=4, type=int)
    # parser.add_argument('--roi_batch_size', default=128, type=int)
    # parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--num_workers', default=0, type=int)
    # parser.add_argument('--refine_visualize_workers', default=0, type=int)
    parser.add_argument('--eval_skip_coco', default=False, type=bool)
    # parser.add_argument('--eval_outputfile', default=None, type=str)
    # parser.add_argument('--hold', default=0.005, type=float)

    # parser.add_argument('--ddp_num_gpus', type=int, default=1)
    # parser.add_argument('--ddp_port', type=int, default=50405)
    args = parser.parse_args()
    # args.anno_models = sorted(list(set(args.anno_models)))
    assert args.anchor_stride > 0.999
    print(args)

    # if not os.access(finetune_output, os.W_OK):
    #     os.mkdir(finetune_output)
    # assert os.path.isdir(finetune_output)
    # assert os.path.isdir(args.outputdir)
    # assert os.access(args.outputdir, os.W_OK)

    if args.opt == 'eval':
        evaluate_all_videos(args)
    elif args.opt == 'tp':
        inference_throughput(args)
    else:
        pass


'''
python finetune_anchor_sampling.py --opt eval --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --cocodir ../../../MSCOCO2017 --input_scale 1 --anchor_stride 1
python finetune_anchor_sampling.py --opt tp --model r101-fpn-3x --ckpt ../../models/mscoco2017_remap_r101-fpn-3x.pth --id 001 --input_scale 1 --anchor_stride 1
'''
