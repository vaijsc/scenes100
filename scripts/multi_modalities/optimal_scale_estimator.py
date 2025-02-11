#!python3

import os
import sys
import json
import time
import tqdm
import lmdb
import argparse
import copy
import gc
import contextlib

import imageio
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision
import torch.optim as optim
import torch.nn as nn
import detectron2.layers

import detectron2
from detectron2.structures import BoxMode

from base_detector_midfusion_scaling import AdaptationPredictor
from base_detector_fusion_mixup import construct_image_w_background
from bbox_quality_estimator import ciou_quality
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import get_cfg_base_model
from evaluation import eval_AP


thing_classes = ['person', 'vehicle']


class ScaleEstimator(nn.Module):
    scales =        [0.5, 0.75, 0.9, 1, 2, 3]
    scale_targets = [ -2,   -1,   0, 1, 2, 3] # more evenly distributed for better training

    def __init__(self, pretrained=False):
        super(ScaleEstimator, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Identity()
        self.regressor = nn.Linear(2048, 1)

    def forward(self, X):
        return self.regressor(self.resnet(X))


class EvaluationDatasetScales(torch.utils.data.Dataset):
    def __init__(self, image_dicts, image_format):
        super(EvaluationDatasetScales, self).__init__()
        self.image_dicts = image_dicts
        self.image_format = image_format
    def __len__(self):
        return len(self.image_dicts)
    def __getitem__(self, i):
        image = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name'], format=self.image_format)
        assert 'file_name_background' in self.image_dicts[i]
        image_bg = detectron2.data.detection_utils.read_image(self.image_dicts[i]['file_name_background'], format=self.image_format)
        image, _, image_diff = construct_image_w_background(image, image_bg)
        image_scales, image_diff_scales = [], []
        W0, W1, _ = image.shape
        for s in ScaleEstimator.scales:
            if s < 1:
                image_s, image_diff_s = map(lambda _im: (skimage.transform.resize(_im, (int(W0 * s), int(W1 * s))) * 255.0).astype(np.uint8), [image, image_diff])
                image_pad, image_diff_pad = np.zeros_like(image), np.zeros_like(image_diff)
                image_pad[: image_s.shape[0], : image_s.shape[1], :] = image_s
                image_diff_pad[: image_diff_s.shape[0], : image_diff_s.shape[1], :] = image_diff_s
                image_scales.append(image_pad)
                image_diff_scales.append(image_diff_pad)
            elif s == 1:
                image_scales.append(image)
                image_diff_scales.append(image_diff)
            else:
                image_s, image_diff_s = map(lambda _im: (skimage.transform.resize(_im, (W0 * s, W1 * s)) * 255.0).astype(np.uint8), [image, image_diff])
                image_scales.append(image_s)
                image_diff_scales.append(image_diff_s)
        return {'dict': copy.deepcopy(self.image_dicts[i]), 'W0': W0, 'W1': W1, 'image_scales': image_scales, 'image_diff_scales': image_diff_scales}
    @staticmethod
    def collate(batch):
        return batch


def detect_scale(detector, image, image_diff, W0, W1, s):
    im = {'annotations': []}
    assert image.dtype == np.uint8 and image_diff.dtype == np.uint8
    if s == 1:
        instances = detector(image, image_diff)[0]['instances'].to('cpu')
        bbox = instances.pred_boxes.tensor
        score = instances.scores
        label = instances.pred_classes
        for i in range(0, len(label)):
            im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
    elif s > 1:
        for patch_i in range(0, s):
            for patch_j in range(0, s):
                image_patch = image[patch_i * W0 : (patch_i + 1) * W0, patch_j * W1 : (patch_j + 1) * W1, :]
                image_diff_patch = image_diff[patch_i * W0 : (patch_i + 1) * W0, patch_j * W1 : (patch_j + 1) * W1, :]
                instances = detector(image_patch, image_diff_patch)[0]['instances'].to('cpu')
                bbox = instances.pred_boxes.tensor
                bbox = bbox / s
                bbox[:, 0] += patch_j * W1 / s
                bbox[:, 2] += patch_j * W1 / s
                bbox[:, 1] += patch_i * W0 / s
                bbox[:, 3] += patch_i * W0 / s
                score = instances.scores
                label = instances.pred_classes
                for i in range(0, len(label)):
                    im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
    else:
        instances = detector(image, image_diff)[0]['instances'].to('cpu')
        bbox = instances.pred_boxes.tensor / s
        score = instances.scores
        label = instances.pred_classes
        for i in range(0, len(label)):
            im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
    return im


def get_coco_dicts(cocodir, split):
    if split == 'valid':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_val2017.json')
        images_dir = os.path.join(cocodir, 'images', 'val2017')
    else:
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
        images_dir = os.path.join(cocodir, 'images', 'train2017')
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    thing_classes_coco, thing_classes = [['person'], ['car', 'bus', 'truck']], ['person', 'vehicle']
    category_id_remap = {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i
    coco_dicts = {}
    for im in annotations['images']:
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'file_name_background': os.path.join(cocodir, 'inpaint_mask', 'val2017' if split == 'valid' else 'train2017', im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        if ann['category_id'] in category_id_remap:
            x, y, w, h = ann['bbox']
            coco_dicts[ann['image_id']]['annotations'].append({'bbox': [x, y, x + w, y + h], 'bbox_mode': BoxMode.XYXY_ABS, 'area': ann['area'], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    return coco_dicts


def generate_optimal_scale(args, split):
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()
    assert split in ['train', 'valid']
    coco_dicts = get_coco_dicts(args.cocodir, split)
    del _tensor
    gc.collect()
    count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))

    cfg = get_cfg_base_model('r101-fpn-3x')
    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth')
    args.fusion = 'mid'
    detector = AdaptationPredictor(cfg, args)

    qualities_scales = []
    loader = torch.utils.data.DataLoader(
        EvaluationDatasetScales(coco_dicts, cfg.INPUT.FORMAT),
        batch_size=None, collate_fn=EvaluationDatasetScales.collate, shuffle=False, num_workers=args.num_workers
    )
    for batch in tqdm.tqdm(loader, total=len(coco_dicts), ascii=True, desc=split):
        im = copy.deepcopy(batch['dict'])
        lb_gt = torch.tensor([ann['category_id'] for ann in im['annotations']]).long()
        xyxy_gt = torch.tensor([ann['bbox'] for ann in im['annotations']]).float()
        xyxy_gt = [xyxy_gt[lb_gt == i_c] for i_c in range(0, len(thing_classes))]
        im['scales'] = copy.deepcopy(ScaleEstimator.scales)
        im['ciou_qualities'] = []
        for i_s, s in enumerate(ScaleEstimator.scales):
            with torch.no_grad():
                dets = detect_scale(detector, batch['image_scales'][i_s], batch['image_diff_scales'][i_s], batch['W0'], batch['W1'], s)['annotations']
            xyxy_det = torch.tensor([ann['bbox'] for ann in dets]).float()
            scores = [ann['score'] for ann in dets]
            ciou_qualities_per_box = []
            for j in range(0, xyxy_det.size(0)):
                if xyxy_gt[dets[j]['category_id']].size(0) > 0:
                    ciou_qualities_per_box.append(float(ciou_quality(xyxy_gt[dets[j]['category_id']], xyxy_det[j : j + 1].expand(xyxy_gt[dets[j]['category_id']].size(0), -1)).max()))
                else:
                    ciou_qualities_per_box.append(0.0)
                scores.append(dets[j]['score'])
            im['ciou_qualities'].append({'qualities_per_box': ciou_qualities_per_box, 'scores': scores})
        qualities_scales.append(im)
    with open('mscoco_scales_quality_%s.json' % split, 'w') as fp:
        json.dump(qualities_scales, fp)


def precompute_valid(args):
    coco_dicts = get_coco_dicts(args.cocodir, 'valid')
    count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('MSCOCO-2017 valid: %d images, %d bboxes' % (count_images, count_bboxes))
    cfg = get_cfg_base_model('r101-fpn-3x')
    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth')
    args.fusion = 'mid'
    detector = AdaptationPredictor(cfg, args)

    detections = []
    loader = torch.utils.data.DataLoader(
        EvaluationDatasetScales(coco_dicts, cfg.INPUT.FORMAT),
        batch_size=None, collate_fn=EvaluationDatasetScales.collate, shuffle=False, num_workers=args.num_workers
    )
    for batch in tqdm.tqdm(loader, total=len(coco_dicts), ascii=True):
        im = copy.deepcopy(batch['dict'])
        im['scales'] = copy.deepcopy(ScaleEstimator.scales)
        im['scales_detection'] = []
        for i_s, s in enumerate(ScaleEstimator.scales):
            with torch.no_grad():
                im['scales_detection'].append(detect_scale(detector, batch['image_scales'][i_s], batch['image_diff_scales'][i_s], batch['W0'], batch['W1'], s)['annotations'])
        detections.append(im)
    with open('mscoco_scales_detections_valid.json', 'w') as fp:
        json.dump(detections, fp)


class MSCOCOOptimalScales(data.Dataset):
    def __init__(self, split, cocodir, balanced=False):
        super(MSCOCOOptimalScales, self).__init__()
        assert split in ['train', 'valid']
        self.split, self.balanced = split, balanced
        images_dir = os.path.join(args.cocodir, 'images', 'val2017' if self.split == 'valid' else 'train2017')
        with open('mscoco_scales_quality_%s.json' % self.split, 'r') as fp:
            annotations = json.load(fp)
        self.ifilelist, self.scale_ys, self.scale_is = [], [], []
        for im in annotations:
            _weighted_qualities = []
            for i in range(0, len(ScaleEstimator.scales)):
                _q = np.array(im['ciou_qualities'][i]['qualities_per_box'])
                _s = np.array(im['ciou_qualities'][i]['scores'][: _q.shape[0]]) # there is a bug that duplicates scores in generation code
                _mask = _s > 0.75
                _q, _s = _q[_mask], _s[_mask]
                _weighted_qualities.append((_q * _s).sum() / _s.sum())
            self.scale_is.append(np.array(_weighted_qualities).argmax())
            self.scale_ys.append(ScaleEstimator.scale_targets[self.scale_is[-1]])
            self.ifilelist.append(os.path.join(images_dir, os.path.basename(im['file_name'])))
        del annotations
        self.ifilelist, self.scale_ys, self.scale_is = map(np.array, [self.ifilelist, self.scale_ys, self.scale_is])
        print('MSCOCO-2017 %s: %d images' % (self.split, self.ifilelist.shape[0]))
        for _y, _num in zip(*np.unique(self.scale_ys, return_counts=True)):
            print('%+d (x %.2f): %6d %.2f%%' % (_y, ScaleEstimator.scales[ScaleEstimator.scale_targets.index(_y)], _num, _num / self.scale_ys.shape[0] * 100))
        if self.balanced:
            self.buckets = {_y: [] for _y in ScaleEstimator.scale_targets}
            for i, _y in enumerate(self.scale_ys):
                self.buckets[_y].append(i)

        if self.split == 'train':
            self.tf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224), antialias=True),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.tf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224), antialias=True),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.ifilelist.shape[0]

    def __getitem__(self, i):
        if self.balanced:
            _y = ScaleEstimator.scale_targets[i % len(ScaleEstimator.scale_targets)]
            i = np.random.choice(self.buckets[_y])
        im_arr = skimage.io.imread(self.ifilelist[i]).astype(np.float32) / 255.0
        if len(im_arr.shape) < 3:
            im_arr = np.stack([im_arr, im_arr, im_arr], axis=2)
        assert len(im_arr.shape) == 3 and im_arr.shape[2] == 3 # RGB
        return self.tf(torch.from_numpy(im_arr.transpose(2, 0, 1))), torch.tensor(self.scale_ys[i]).float(), torch.tensor(self.scale_is[i]).long()


def train_eval(args):
    np.random.seed(0)
    torch.manual_seed(0)

    assert(torch.cuda.is_available())
    net = ScaleEstimator(pretrained=True).cuda()
    for p in net.resnet.parameters():
        p.requires_grad = False
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 3, gamma=0.1)

    loader_train = data.DataLoader(MSCOCOOptimalScales('train', args.cocodir, balanced=True), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    loader_valid = data.DataLoader(MSCOCOOptimalScales('valid', args.cocodir, balanced=False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with open('mscoco_scales_detections_valid.json', 'r') as fp:
        images_detections = json.load(fp)
    images = copy.deepcopy(images_detections)
    for im in images:
        del im['scales'], im['scales_detection']
    for i, s in enumerate(ScaleEstimator.scales):
        print('scale x %.2f:' % ScaleEstimator.scales[i])
        detections = copy.deepcopy(images_detections)
        for im in detections:
            im['annotations'] = im['scales_detection'][i]
            del im['scales'], im['scales_detection']
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_AP = eval_AP(images, detections)
        del results_AP['raw'], detections
        print('             %s' % '/'.join(results_AP['metrics']))
        for c in sorted(results_AP['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results_AP['results'][c])))
    detections = copy.deepcopy(images_detections)
    _im_i = 0
    for _, _, scale_i in tqdm.tqdm(loader_valid, ascii=True, desc='oracle'):
        for _i in scale_i:
            detections[_im_i]['annotations'] = detections[_im_i]['scales_detection'][int(_i)]
            del detections[_im_i]['scales'], detections[_im_i]['scales_detection']
            _im_i += 1
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results_AP = eval_AP(images, detections)
    del results_AP['raw'], detections
    print('             %s' % '/'.join(results_AP['metrics']))
    for c in sorted(results_AP['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), results_AP['results'][c])))
    print()

    loss_history = []
    for ep in range(0, args.epochs):
        if ep == 2:
            print('unfreeze whole network')
            for p in net.parameters():
                p.requires_grad = True

        loss_history.append({'train': [], 'valid': []})
        net.eval()
        gt, pred = [], []
        with torch.no_grad():
            for X, y, _ in tqdm.tqdm(loader_valid, ascii=True, desc='evaluating %d/%d' % (ep + 1, args.epochs)):
                gt.append(y.numpy())
                pred.append(net(X.cuda()).flatten().detach().cpu().numpy())
        gt, pred = np.concatenate(gt, axis=0), np.concatenate(pred, axis=0)
        loss_history[-1]['valid'] = ((gt - pred) ** 2).mean()

        pred = pred.reshape(-1, 1)
        scale_targets = np.array(ScaleEstimator.scale_targets).astype(np.float32).reshape(1, -1)
        pred_scale_i = np.absolute(pred - scale_targets).argmin(axis=1)
        for i, _num in zip(*np.unique(pred_scale_i, return_counts=True)):
            print('%+d (x %.2f): %d' % (ScaleEstimator.scale_targets[i], ScaleEstimator.scales[i], _num))
        detections = copy.deepcopy(images_detections)
        for i, im in enumerate(detections):
            im['annotations'] = im['scales_detection'][pred_scale_i[i]]
            del im['scales'], im['scales_detection']
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_AP = eval_AP(images, detections)
        del results_AP['raw'], detections
        print('             %s' % '/'.join(results_AP['metrics']))
        for c in sorted(results_AP['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), results_AP['results'][c])))

        net.train()
        for X, y, _ in tqdm.tqdm(loader_train, ascii=True, desc='training   %d/%d' % (ep + 1, args.epochs)):
            optimizer.zero_grad()
            y_ = net(X.cuda()).flatten()
            L = loss_fn(y_, y.cuda()).mean()
            L.backward()
            optimizer.step()
            loss_history[-1]['train'].append(L.item())
        scheduler.step()
        print('training loss = %.4f, validation loss = %.4f\n' % (np.array(loss_history[-1]['train']).mean(), loss_history[-1]['valid']))
        torch.save(net.state_dict(), 'optimalscaleestimator.pt')


def stats_wh():
    coco_dicts = get_coco_dicts('../../../MSCOCO2017', 'train')
    coco_dicts = coco_dicts + get_coco_dicts('../../../MSCOCO2017', 'valid')
    wh_counts = {}
    for im in coco_dicts:
        # _wh = 'W%dxH%d' % (im['width'], im['height'])
        _wh = 'W/H=%.2f' % (im['width'] / im['height'])
        if not _wh in wh_counts:
            wh_counts[_wh] = 0
        wh_counts[_wh] += 1
    for _wh in wh_counts:
        print('%s: %d' % (_wh, wh_counts[_wh]))
    print(len(wh_counts))


if __name__ == '__main__':
    # stats_wh(); exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['generate', 'precompute', 'train'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=2e-3, type=int)
    parser.add_argument('--cocodir', type=str)
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    print(args)
    if args.opt == 'generate':
        generate_optimal_scale(args, 'valid')
        generate_optimal_scale(args, 'train')
    elif args.opt == 'precompute':
        precompute_valid(args)
    elif args.opt == 'train':
        train_eval(args)
