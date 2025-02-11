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
import pickle
import math
import random
import tqdm
import glob
import psutil
import contextlib
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
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from models import get_cfg_base_model
from decode_training import TrainingFrames
from base_detector_train import get_coco_dicts
from evaluation import evaluate_masked, evaluate_cocovalid, eval_AP


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


class EvaluationDataset(torchdata.Dataset):
    def __init__(self, image_dicts, image_list):
        super(EvaluationDataset, self).__init__()
        assert len(image_dicts) == len(image_list)
        self.image_dicts = image_dicts
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, i):
        im_arr = skimage.io.imread(self.image_list[i])
        if len(im_arr.shape) == 2:
            im_arr = np.stack([im_arr] * 3, axis=2)
        return self.image_dicts[i], im_arr[:, :, ::-1]
    @staticmethod
    def collate(batch):
        return batch


def evaluate_base(args):
    args.cocodir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017')
    args.smallscale = False
    model = args.model
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    output_json = os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % model)

    results_AP, masked_bboxes_per_video, detections_per_video = {}, {}, {}
    cfg = get_cfg_base_model(model)
    if args.cpu:
        print('Run predictor on CPU, only for inference speed estimation')
        cfg.MODEL.DEVICE = 'cpu'
    if args.ckpt is not None:
        print('loading weights from:', args.ckpt)
        cfg.MODEL.WEIGHTS = args.ckpt
    detector = DefaultPredictor(cfg)
    results_AP[model] = {}

    images_coco = get_coco_dicts(args, 'valid')
    if args.cpu:
        images_coco = sklearn.utils.shuffle(images_coco)[:100]
    loader = torchdata.DataLoader(
        EvaluationDataset(
            copy.deepcopy(images_coco),
            [im['file_name'] for im in images_coco]
        ), batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=(1 if args.cpu else 4)
    )
    detections_coco = []
    for im, im_arr in tqdm.tqdm(loader, total=len(images_coco), ascii=True, desc='%s detecting MSCOCO2017 valid' % model):
        instances = detector(im_arr)['instances'].to('cpu')
        instances.pred_masks = instances.scores # remove masks if applicable, reduce memory usage
        im['annotations'] = []
        im['instances'] = instances
        detections_coco.append(im)
    del loader
    for im in detections_coco:
        # bbox has format [x1, y1, x2, y2]
        bbox = im['instances'].pred_boxes.tensor
        score = im['instances'].scores
        label = im['instances'].pred_classes
        for i in range(0, len(label)):
            im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
        del im['instances']

    images_manual = []
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated')
    for f in files:
        with open(os.path.join(inputdir, f['id'], 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        for im in images:
            im['video_id'] = f['id']
        images_manual = images_manual + images
    if args.cpu:
        images_manual = sklearn.utils.shuffle(images_manual)[:500]
    loader = torchdata.DataLoader(
        EvaluationDataset(
            copy.deepcopy(images_manual),
            [os.path.join(inputdir, im['video_id'], 'unmasked', im['file_name']) for im in images_manual]
        ), batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=(1 if args.cpu else 4)
    )
    detections_per_video[model] = {f['id']: [] for f in files}
    detections_manual = {f['id']: [] for f in files}
    for im, im_arr in tqdm.tqdm(loader, total=len(images_manual), ascii=True, desc='%s detecting manual annotated' % model):
        instances = detector(im_arr)['instances'].to('cpu')
        instances.pred_masks = instances.scores # remove masks if applicable, reduce memory usage
        im['annotations'] = []
        im['instances'] = instances
        detections_per_video[model][im['video_id']].append(im)
    del loader
    for vid in detections_per_video[model]:
        for im in detections_per_video[model][vid]:
            # bbox has format [x1, y1, x2, y2]
            bbox = im['instances'].pred_boxes.tensor
            score = im['instances'].scores
            label = im['instances'].pred_classes
            for i in range(0, len(label)):
                im['annotations'].append({'bbox': list(map(float, bbox[i])), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(label[i]), 'score': float(score[i])})
            del im['instances']

    del detector
    if args.cpu:
        return
    input('detection finished, start AP evaluation <Enter>')
    # AP evaluation
    masked_bboxes_per_video[model] = {}
    for f in files:
        results_id, images_id, detections_id = evaluate_masked(f['id'], copy.deepcopy(detections_per_video[model][f['id']]), return_bboxes=True)
        results_AP[model]['manual_%s' % f['id']] = results_id
        masked_bboxes_per_video[model][f['id']] = {'annotations': images_id, 'detections': detections_id}

    annotations_all_video, detections_all_video = [], []
    for f in files:
        annotations_all_video = annotations_all_video + masked_bboxes_per_video[model][f['id']]['annotations']
        detections_all_video = detections_all_video + masked_bboxes_per_video[model][f['id']]['detections']
    results_AP[model]['all_videos'] = eval_AP(annotations_all_video, detections_all_video)
    results_AP[model]['mscoco2017_valid'] = evaluate_cocovalid(args.cocodir, detections_coco)
    print('MSCOCO2017:', results_AP[model]['mscoco2017_valid']['results'])
    with open(output_json, 'w') as fp:
        json.dump(results_AP, fp)

    categories = ['person', 'vehicle', 'overall', 'weighted']
    results_AP_videos = {k[7:]: results_AP[model][k] for k in filter(lambda s: s[:7] == 'manual_', results_AP[model].keys())}
    assert len(results_AP_videos) == 100, str(results_AP_videos.keys())
    videos = sorted(list(results_AP_videos.keys()))

    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(4, 1, figsize=(28, 28))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        mAP_AP50 = np.array([results_AP_videos[v]['results'][categories[i]] for v in videos]) * 100
        mAP_AP50_allvideos = np.array(results_AP[model]['all_videos']['results'][categories[i]]) * 100
        valid_mask = mAP_AP50[:, 0] >= 0
        axes[i].plot(xs[valid_mask], mAP_AP50[valid_mask, 0], 'rx-')
        axes[i].plot(xs[valid_mask], mAP_AP50[valid_mask, 1], 'bx-')
        axes[i].legend([
            'mAP valid mean: %.4f all videos eval: %.4f' % (mAP_AP50[valid_mask, 0].mean(), mAP_AP50_allvideos[0]),
            'AP50 valid mean: %.4f all videos eval: %.4f' % (mAP_AP50[valid_mask, 1].mean(), mAP_AP50_allvideos[1])
        ])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        axes[i].set_ylim(0, 105)
        axes[i].set_ylabel('AP (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    # plt.tight_layout()
    plt.suptitle(output_json)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(output_json + '.pdf')
    plt.close()
    print('results saved to:', output_json)


def compare_with_base(model, results_file):
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % model), 'r') as fp:
        base_AP = json.load(fp)[model]
    with open(results_file, 'r') as fp:
        results_AP = json.load(fp)
    videos = sorted(list(results_AP.keys()))
    categories = ['person', 'vehicle', 'overall', 'weighted']

    improvements = {c: [] for c in categories}
    for vid in videos:
        AP1 = base_AP['manual_' + vid]['results']
        AP2 = results_AP[vid]['results']
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
    plt.suptitle(results_file)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(results_file + '.pdf')
    plt.close()


def compare_all_videos(args):
    print('scanning checkpoints in %s' % args.compare_ckpt_dir)
    ckpts = sorted(glob.glob(os.path.join(args.compare_ckpt_dir, 'adapt*.pth')))
    if len(ckpts) != 100:
        print('wrong number of checkpoints: %d, missing:' % len(ckpts))
        video_id_present = [os.path.basename(f)[5 : 8] for f in ckpts]
        with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
            files = json.load(fp)
        video_id_list = [f['id'] for f in files]
        missing = sorted(list(set(video_id_list) - set(video_id_present)))
        print(missing)
        print(' '.join(missing))
        return

    if args.arch == 'vanilla':
        from finetune import evaluate
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP.json')
        t0 = time.time()
        for i, f in enumerate(ckpts):
            args.id, args.ckpt = os.path.basename(f)[5 : 8], f
            results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(ckpts), (time.time() - t0) / 60.0))
        with open(results_AP_json, 'w') as fp:
            fp.write(json.dumps(results_AP))
        compare_with_base(args.model, results_AP_json)

    elif args.arch in ['earlyfusion', 'midfusion', 'latefusion']:
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP_%s.json' % args.eval_background)
        if args.arch == 'earlyfusion':
            from finetune_wdiff_earlyfusion import evaluate
            t0 = time.time()
            for i, f in enumerate(ckpts):
                args.id, args.ckpt = os.path.basename(f)[5 : 8], f
                results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
                print('[%d/%d finished in %.1f minutes]' % (i + 1, len(ckpts), (time.time() - t0) / 60.0))
            with open(results_AP_json, 'w') as fp:
                fp.write(json.dumps(results_AP))
            compare_with_base(args.model, results_AP_json)

        elif args.arch == 'midfusion':
            from finetune_wdiff_midfusion import evaluate
            keys = ['orig', 'merge']
            results_AP = {k: {} for k in keys}
            t0 = time.time()
            for i, f in enumerate(ckpts):
                args.id, args.ckpt = os.path.basename(f)[5 : 8], f
                _results = evaluate(args)[1]
                for k in keys:
                    results_AP[k][args.id] = _results[k]['manual_' + args.id]
                print('[%d/%d finished in %.1f minutes]' % (i + 1, len(ckpts), (time.time() - t0) / 60.0))
            for k in keys:
                results_AP_json = os.path.join(args.compare_ckpt_dir, 'results_AP_%s_%s.json' % (k, args.eval_background))
                with open(results_AP_json, 'w') as fp:
                    fp.write(json.dumps(results_AP[k]))
                compare_with_base(args.model, results_AP_json)

        if args.arch == 'latefusion':
            from finetune_wdiff_latefusion import evaluate
            keys = ['orig', 'merge']
            results_AP = {k: {} for k in keys}
            t0 = time.time()
            for i, f in enumerate(ckpts):
                args.id, args.ckpt = os.path.basename(f)[5 : 8], f
                _results = evaluate(args)[1]
                for k in keys:
                    results_AP[k][args.id] = _results[k]['manual_' + args.id]
                print('[%d/%d finished in %.1f minutes]' % (i + 1, len(ckpts), (time.time() - t0) / 60.0))
            for k in keys:
                results_AP_json = os.path.join(args.compare_ckpt_dir, 'results_AP_%s_%s.json' % (k, args.eval_background))
                with open(results_AP_json, 'w') as fp:
                    fp.write(json.dumps(results_AP[k]))
                compare_with_base(args.model, results_AP_json)

    elif args.arch in ['homography', 'radial', 'fpncorr']:
        if args.arch == 'homography':
            from finetune_homography_mixup import evaluate
        elif args.arch == 'radial':
            from finetune_radial_mixup import evaluate
        elif args.arch == 'fpncorr':
            from finetune_fpn_correlation import evaluate
        else:
            raise NotImplementedError
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP.json')
        t0 = time.time()
        for i, f in enumerate(ckpts):
            args.id, args.ckpt = os.path.basename(f)[5 : 8], f
            results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(ckpts), (time.time() - t0) / 60.0))
        with open(results_AP_json, 'w') as fp:
            fp.write(json.dumps(results_AP))
        compare_with_base(args.model, results_AP_json)

    elif args.arch in ['adapteacher', 'h2fa']:
        from finetune import evaluate
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP.json')
        t0 = time.time()
        for i, f in enumerate(ckpts):
            args.id, args.ckpt = os.path.basename(f)[20 : 23], f
            results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(ckpts), (time.time() - t0) / 60.0))
        with open(results_AP_json, 'w') as fp:
            fp.write(json.dumps(results_AP))
        compare_with_base(args.model, results_AP_json)

    else:
        raise Exception('unrecognized model modification: %s' % args.arch)


def compare_all_videos_compound(args):
    video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    print('scanning checkpoints in %s' % args.compare_ckpt_dir)
    ckpts = sorted(glob.glob(os.path.join(args.compare_ckpt_dir, 'adapt*.pth')))
    if len(ckpts) != 1:
        print('wrong number of checkpoints: %d, expect only 1' % len(ckpts))
        return

    if args.arch == 'vanilla':
        from finetune import evaluate
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP.json')
        t0 = time.time()
        for i, v in enumerate(video_id_list):
            args.id, args.ckpt = v, ckpts[0]
            results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
        with open(results_AP_json, 'w') as fp:
            fp.write(json.dumps(results_AP))
        compare_with_base(args.model, results_AP_json)

    elif args.arch in ['earlyfusion', 'midfusion', 'latefusion']:
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP_%s.json' % args.eval_background)
        if args.arch == 'earlyfusion':
            from finetune_wdiff_earlyfusion import evaluate
            t0 = time.time()
            for i, v in enumerate(video_id_list):
                args.id, args.ckpt = v, ckpts[0]
                results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
                print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
            with open(results_AP_json, 'w') as fp:
                fp.write(json.dumps(results_AP))
            compare_with_base(args.model, results_AP_json)

        elif args.arch == 'midfusion':
            from finetune_wdiff_midfusion import evaluate
            keys = ['orig', 'merge']
            results_AP = {k: {} for k in keys}
            t0 = time.time()
            for i, v in enumerate(video_id_list):
                args.id, args.ckpt = v, ckpts[0]
                _results = evaluate(args)[1]
                for k in keys:
                    results_AP[k][args.id] = _results[k]['manual_' + args.id]
                print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
            for k in keys:
                results_AP_json = os.path.join(args.compare_ckpt_dir, 'results_AP_%s_%s.json' % (k, args.eval_background))
                with open(results_AP_json, 'w') as fp:
                    fp.write(json.dumps(results_AP[k]))
                compare_with_base(args.model, results_AP_json)

        if args.arch == 'latefusion':
            from finetune_wdiff_latefusion import evaluate
            keys = ['orig', 'merge']
            results_AP = {k: {} for k in keys}
            t0 = time.time()
            for i, v in enumerate(video_id_list):
                args.id, args.ckpt = v, ckpts[0]
                _results = evaluate(args)[1]
                for k in keys:
                    results_AP[k][args.id] = _results[k]['manual_' + args.id]
                print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
            for k in keys:
                results_AP_json = os.path.join(args.compare_ckpt_dir, 'results_AP_%s_%s.json' % (k, args.eval_background))
                with open(results_AP_json, 'w') as fp:
                    fp.write(json.dumps(results_AP[k]))
                compare_with_base(args.model, results_AP_json)

    elif args.arch in ['adapteacher', 'h2fa']:
        from finetune import evaluate
        results_AP, results_AP_json = {}, os.path.join(args.compare_ckpt_dir, 'results_AP.json')
        t0 = time.time()
        for i, v in enumerate(video_id_list):
            args.id, args.ckpt = v, ckpts[0]
            results_AP[args.id] = evaluate(args)[1]['manual_' + args.id]
            print('[%d/%d finished in %.1f minutes]' % (i + 1, len(video_id_list), (time.time() - t0) / 60.0))
        with open(results_AP_json, 'w') as fp:
            fp.write(json.dumps(results_AP))
        compare_with_base(args.model, results_AP_json)

    else:
        raise Exception('unrecognized model modification: %s' % args.arch)


def compare_all_videos_precision(args):
    with open(os.path.join(os.path.dirname(__file__), 'results_AP_base_%s.json' % args.model), 'r') as fp:
        base_AP = json.load(fp)[args.model]
    with open(args.compare_pr_src_json, 'r') as fp:
        results_AP = json.load(fp)
    videos = sorted(list(results_AP.keys()))
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {}

    maxDets, recThrs, area_sup = 100, 0.8, 10000000000
    detsIdx, recIdx, areaIdx = 2, 80, 0
    for iouThrs, iouIdx in [[0.5, 0], [0.75, 5]]:
        print('comparing precisions of\nIoU      = %.2f\nrecall   = %.2f\nmax_dets = %d\nmax_area = %.1f' % (iouThrs, recThrs, maxDets, area_sup))
        improvements[iouThrs] = {c: [] for c in categories}
        for vid in tqdm.tqdm(videos, ascii=True, desc='comparing videos'):
            raw1, raw2 = base_AP['manual_' + vid]['raw'], results_AP[vid]['raw']
            w1, w2 = base_AP['manual_' + vid]['weights'], results_AP[vid]['weights']
            assert abs(w1['total'] - w2['total']) < 1 and abs(w1['classes'][0] - w2['classes'][0]) < 1e-3 and abs(w1['classes'][1] - w2['classes'][1]) < 1e-3 and abs(sum(w1['classes']) - 1) < 1e-3
            assert abs(raw1['iouThrs'][iouIdx] - iouThrs) < 1e-3 and abs(raw1['maxDets'][detsIdx] - maxDets) < 0.5 and abs(raw1['recThrs'][recIdx] - recThrs) < 1e-3 and raw1['areaRng'][areaIdx][0] < 1 and abs(raw1['areaRng'][areaIdx][1] - area_sup) < 1
            assert abs(raw2['iouThrs'][iouIdx] - iouThrs) < 1e-3 and abs(raw2['maxDets'][detsIdx] - maxDets) < 0.5 and abs(raw2['recThrs'][recIdx] - recThrs) < 1e-3 and raw2['areaRng'][areaIdx][0] < 1 and abs(raw2['areaRng'][areaIdx][1] - area_sup) < 1 # check if things match
            rec1, rec2 = np.array(raw1['recall']), np.array(raw2['recall'])
            pr1, pr2 = np.array(raw1['precision']), np.array(raw2['precision'])
            # print(vid); print(base_AP['manual_' + vid]['results']); print(results_AP[vid]['results'])
            # print(pr1[iouIdx, :, :, areaIdx, detsIdx].mean(axis=0)); print(pr2[iouIdx, :, :, areaIdx, detsIdx].mean(axis=0))
            pr_rec_1 = pr1[iouIdx, recIdx, :, areaIdx, detsIdx]
            pr_rec_2 = pr2[iouIdx, recIdx, :, areaIdx, detsIdx]
            weights = np.array(w1['classes'])
            pr_reca_w_1, pr_reca_w_2 = (pr_rec_1 * weights).sum(), (pr_rec_2 * weights).sum()
            if weights.min() < 0.1 / w1['total']:
                # print(pr_rec_1); print(pr_rec_2)
                pr_reca_avg_1, pr_reca_avg_2 = pr_reca_w_1, pr_reca_w_2 # some class is absent
            else:
                pr_reca_avg_1, pr_reca_avg_2 = pr_rec_1.mean(), pr_rec_2.mean()
            improvements[iouThrs]['person'].append(pr_rec_2[0] - pr_rec_1[0])
            improvements[iouThrs]['vehicle'].append(pr_rec_2[1] - pr_rec_1[1])
            improvements[iouThrs]['overall'].append(pr_reca_avg_2 - pr_reca_avg_1)
            improvements[iouThrs]['weighted'].append(pr_reca_w_2 - pr_reca_w_1)

    assert len(improvements) == 2, str(improvements)
    iouThrs_list = list(improvements.keys())
    for iouThrs in improvements:
        for cat in categories:
            improvements[iouThrs][cat] = np.array(improvements[iouThrs][cat]) * 100.0
    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        axes[i].plot([-1, xs.max() + 1], [0, 0], 'k-')
        axes[i].plot(xs, improvements[iouThrs_list[0]][categories[i]], 'r.-')
        axes[i].plot(xs, improvements[iouThrs_list[1]][categories[i]], 'b.-')
        axes[i].legend([
            '0',
            'Precision@Recall=%.2f IoU=%.2f %.4f' % (recThrs, iouThrs_list[0], improvements[iouThrs_list[0]][categories[i]].mean()),
            'Precision@Recall=%.2f IoU=%.2f %.4f' % (recThrs, iouThrs_list[1], improvements[iouThrs_list[1]][categories[i]].mean()),
        ])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        # axes[i].set_ylim(-3, 3)
        axes[i].set_ylabel('Precision improvement (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    # plt.tight_layout()
    plt.suptitle(args.compare_pr_src_json)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(args.compare_pr_src_json + '_precision.pdf')
    plt.close()


def compare_base_to_base(args):
    import tempfile

    with open(args.compare_b2b_src_json, 'r') as fp:
        target_AP = list(json.load(fp).values())
    assert len(target_AP) == 1
    target_AP = target_AP[0]
    target_AP_format = {}
    for k in target_AP:
        if k[:7] == 'manual_':
            target_AP_format[k[7:]] = target_AP[k]
    assert len(target_AP_format) == 100
    _, temp_fpath = tempfile.mkstemp(suffix='.json', text=True)
    with open(temp_fpath, 'w') as fp:
        json.dump(target_AP_format, fp)
    compare_with_base(args.model, temp_fpath)
    args.compare_pr_src_json = temp_fpath
    compare_all_videos_precision(args)
    print(temp_fpath)


def visualize_compare(model, vid, ckpt):
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', vid)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        images = json.load(fp)

    cfgs = [get_cfg_base_model(model), get_cfg_base_model(model, ckpt=ckpt)]
    results, vid_arrs = [], []
    outputfile = ckpt + '.mp4'
    for i in range(0, len(cfgs)):
        detector = DefaultPredictor(cfgs[i])
        detections = []
        for im in tqdm.tqdm(images, ascii=True, desc='%s detecting %s validation frames' % (model, vid)):
            det = copy.deepcopy(im)
            det['annotations'] = []
            instances = detector(skimage.io.imread(os.path.join(inputdir, 'unmasked', im['file_name']))[:, :, ::-1])['instances'].to('cpu')
            # bbox has format [x1, y1, x2, y2]
            bbox = instances.pred_boxes.tensor.numpy().tolist()
            score = instances.scores.numpy().tolist()
            label = instances.pred_classes.numpy().tolist()
            for i in range(0, len(label)):
                det['annotations'].append({'bbox': bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
            detections.append(det)
        results.append(evaluate_masked(vid, copy.deepcopy(detections), outputfile=outputfile, return_bboxes=False))
        vid_arrs.append(skvideo.io.vread(outputfile))
    print(results[0]['metrics'])
    print(results[0]['results'])
    print(results[1]['results'])

    vid_arrs = np.concatenate(vid_arrs, axis=1)
    writer = skvideo.io.FFmpegWriter(outputfile, inputdict={'-r': '1'}, outputdict={'-vcodec': 'libx265', '-r': '1', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
    for i in tqdm.tqdm(range(0, vid_arrs.shape[0]), ascii=True):
        writer.writeFrame(vid_arrs[i])
    writer.close()
    print(outputfile)


def stats(results_file):
    categories = ['person', 'vehicle', 'overall', 'weighted']
    model = 'r50-fpn-3x'
    # model = 'r101-fpn-3x'

    # with gzip.open(os.path.join(os.path.dirname(__file__), 'gmm_test_results.json.gz'), 'rt') as fp:
    #     base_AP = json.loads(fp.read())['AP'][model]
    # _APs = {cat : [] for cat in categories}
    # for vid in base_AP:
    #     if vid[: 7] != 'manual_':
    #         continue
    #     if base_AP[vid]['results']['vehicle'][0] < 0:
    #         print(vid, base_AP[vid]['results'])
    #     for cat in categories:
    #         _APs[cat].append(base_AP[vid]['results'][cat])

    with open(results_file, 'r') as fp:
        results_AP = json.load(fp)
    _APs = {cat : [] for cat in categories}
    for vid in results_AP:
        if results_AP[vid]['results']['vehicle'][0] < 0:
            print(vid, results_AP[vid]['results'])
        for cat in categories:
            _APs[cat].append(results_AP[vid]['results'][cat])

    for cat in _APs:
        mAP, AP50 = np.array(_APs[cat]).T * 100
        mAP, AP50 = mAP[mAP >= 0], AP50[AP50 >= 0]
        print(cat, mAP.shape, AP50.shape)
        bins = np.arange(0, 101, 5)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('%s mAP $%.1f\\pm %.1f$' % (cat, mAP.mean(), mAP.std()))
        plt.hist(mAP, bins=bins)
        plt.subplot(1, 2, 2)
        plt.title('%s AP50 $%.1f\\pm %.1f$' % (cat, AP50.mean(), AP50.std()))
        plt.hist(AP50, bins=bins)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str, default='', choices=['', 'eval_base', 'compare', 'compare_compound', 'compare_b2b', 'compare_pr', 'visualize'], help='script option')
    parser.add_argument('--id', type=str, help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--arch', type=str, choices=['vanilla', 'earlyfusion', 'midfusion', 'latefusion', 'homography', 'radial', 'fpncorr', 'adapteacher', 'h2fa'], help='faster RCNN architecture')
    parser.add_argument('--cocodir', type=str, help='MSCOCO2017 directory')
    parser.add_argument('--fusion_type', type=str, choices=['average', 'conv', 'attn'], default='average', help='feature pyramids fusion method')
    parser.add_argument('--eval_background', type=str, default='', choices=['', 'dynamic', 'last', 'average'], help='use inference time dynamic background, last training time background, or averaged training time background')
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--compare_ckpt_dir', type=str)
    parser.add_argument('--compare_b2b_src_json', type=str)
    parser.add_argument('--compare_pr_src_json', type=str)
    parser.add_argument('--cpu', default=False, type=bool)
    parser.add_argument('--eval_skip_coco', default=True, type=bool)
    parser.add_argument('--eval_outputfile', default=None, type=str)
    args = parser.parse_args()

    if args.opt == 'eval_base':
        evaluate_base(args)
    elif args.opt == 'visualize':
        visualize_compare(args.model, args.id, args.ckpt)
    elif args.opt == 'compare':
        compare_all_videos(args)
    elif args.opt == 'compare_compound':
        compare_all_videos_compound(args)
    elif args.opt == 'compare_b2b':
        compare_base_to_base(args)
    elif args.opt == 'compare_pr':
        compare_all_videos_precision(args)

'''
python compare_baselines.py --opt visualize --model r101-fpn-3x --id 001 --ckpt E:\\intersections_results\\baseline_crossteach_r101\\adapt001_r101-fpn-3x_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain.pth

python compare_baselines.py --model r101-fpn-3x --opt compare_b2b --compare_b2b_src_json results_AP_base_r50-fpn-3x.json

python compare_baselines.py --opt eval_base --model r101-fpn-3x

python compare_baselines.py --opt compare --model r101-fpn-3x --arch vanilla --compare_ckpt_dir E:\\intersections_results\\baseline_crossteach_r101
python compare_baselines.py --opt compare --model r101-fpn-3x --arch earlyfusion --eval_background dynamic --compare_ckpt_dir E:\\intersections_results\\object_diff_earlyfusion_mixup_r101
python compare_baselines.py --opt compare --model r101-fpn-3x --arch midfusion --eval_background dynamic --compare_ckpt_dir E:\\intersections_results\\object_diff_midfusion_mixup_r101
python compare_baselines.py --opt compare --model r101-fpn-3x --arch latefusion --eval_background dynamic --compare_ckpt_dir E:\\intersections_results\\object_diff_latefusion_r101

python compare_baselines.py --opt compare --model r101-fpn-3x --arch midfusion --fusion_type conv --eval_background dynamic --compare_ckpt_dir F:\\intersections_results\\object_diff_midfusion_conv_r101
python compare_baselines.py --opt compare --model r101-fpn-3x --arch homography --compare_ckpt_dir F:\\intersections_results\\homography_r101
'''

# stats(results_AP)
