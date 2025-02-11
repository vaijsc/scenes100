#!python3

import os
import sys
import time
import datetime
import json
import gzip
import copy
import math
import random
import tqdm
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as torchdata

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from decode_training import TrainingFrames

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def detect(args):
    import detectron2
    from detectron2.engine import DefaultPredictor
    from models import get_cfg_base_model

    cfg = get_cfg_base_model(args.model)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    for v in args.ids:
        dst = TrainingFrames(v)
        print('detect objects with %s in %s' % (args.model, dst), flush=True)
        result_json_zip = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label_scaling', '%s_detect_%s_s%.1f.json.gz' % (v, args.model, args.input_scale)))
        print('results save to:', result_json_zip)

        frame_objs, ifilelist = [], []
        for im, frame_id, fn, i in tqdm.tqdm(dst, ascii=True):
            ifilelist.append(fn)
            instances = detector(im[:, :, ::-1])['instances'].to('cpu')
            frame_objs.append({
                # bbox has format [x1, y1, x2, y2]
                'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                'score': instances.scores.numpy().tolist(),
                'label': instances.pred_classes.numpy().tolist()
            })
        with gzip.open(result_json_zip, 'wt') as fp:
            fp.write(json.dumps({'model': args.model, 'classes': thing_classes, 'frames': ifilelist, 'dets': frame_objs, 'args': vars(args)}))
        print()


def detect_coco(args):
    import detectron2
    from detectron2.engine import DefaultPredictor
    from models import get_cfg_base_model
    from base_detector_train import get_coco_dicts
    from finetune import EvaluationDataset

    cfg = get_cfg_base_model(args.model)
    cfg.INPUT.MIN_SIZE_TEST = int(args.input_scale * cfg.INPUT.MIN_SIZE_TEST)
    cfg.INPUT.MAX_SIZE_TEST = int(args.input_scale * cfg.INPUT.MAX_SIZE_TEST)
    detector = DefaultPredictor(cfg)

    args.cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    args.smallscale = False
    dst_cocotrain = get_coco_dicts(args, 'train')

    loader = torchdata.DataLoader(
        EvaluationDataset(copy.deepcopy(dst_cocotrain), [im['file_name'] for im in dst_cocotrain]),
        batch_size=None, collate_fn=EvaluationDataset.collate, shuffle=False, num_workers=1
    )
    frame_objs, ifilelist = [], []
    for i, (im, im_arr) in tqdm.tqdm(enumerate(loader), total=len(dst_cocotrain), ascii=True):
        im['file_name'] = os.path.basename(im['file_name'])
        ifilelist.append(im)
        instances = detector(im_arr)['instances'].to('cpu')
        frame_objs.append({
            # bbox has format [x1, y1, x2, y2]
            'bbox': instances.pred_boxes.tensor.numpy().tolist(),
            'score': instances.scores.numpy().tolist(),
            'label': instances.pred_classes.numpy().tolist()
        })
        if (i + 1) % 5000 == 0 or i == len(dst_cocotrain) - 1:
            result_json_zip = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label_scaling', 'mscoco_train2017_detect_%s_s%.1f_%06d-%06d.json.gz' % (args.model, args.input_scale, i + 1 - len(ifilelist), i)))
            with gzip.open(result_json_zip, 'wt') as fp:
                fp.write(json.dumps({'model': args.model, 'classes': thing_classes, 'frames': ifilelist, 'dets': frame_objs, 'args': vars(args)}))
            print('\nresults saved to:', result_json_zip)
            frame_objs, ifilelist = [], []


def check_files():
    inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_pseudo_label_scaling'))
    for s in 1.5, 2, 2.5:
        print('checking scale %.2f' % s)
        present, missing = [], []
        for v in video_id_list:
            f = os.path.join(inputdir, '%s_detect_r101-fpn-3x_s%.1f.json.gz' % (v, s))
            # print(f, end=' ')
            if not os.access(f, os.R_OK):
                missing.append(v)
                # print('not present')
            else:
                present.append(v)
                # print('present')
        print('present files %d:' % len(present), ' '.join(present))
        print('missing files %d:' % len(missing), ' '.join(missing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection & Tracking')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--ids', type=str, nargs='+')
    parser.add_argument('--input_scale', type=float, default=2.0)
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    print(args)
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()

    if args.opt == 'detect':
        detect(args)
    if args.opt == 'detect_coco':
        detect_coco(args)
    elif args.opt == 'check':
        check_files()
    else:
        pass
    del _tensor


'''
python detect_scaling.py --opt detect --model r101-fpn-3x --input_scale 2 --ids 001 003
python detect_scaling.py --opt detect_coco --model r101-fpn-3x --input_scale 2
python detect_scaling.py --opt check
'''
