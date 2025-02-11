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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from decode_training import TrainingFrames

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def convert_050():
    from detectron2.structures import BoxMode
    annodir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', '050')
    with open(os.path.join(annodir, 'instances.json'), 'r') as fp:
        annotations_coco = json.load(fp)
    images, boxes = annotations_coco['images'], annotations_coco['annotations']
    print('%d images, %d boxes' % (len(images), len(boxes)))
    category_id_remap = {1: 0, 2: 1}

    annotations_detectron2 = {im['id']: {'file_name': im['file_name'], 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []} for im in images}
    for ann in boxes:
        x, y, w, h = ann['bbox']
        annotations_detectron2[ann['image_id']]['annotations'].append({'bbox': [x, y, x + w, y + h], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': category_id_remap[ann['category_id']]})
    annotations_detectron2 = list(annotations_detectron2.values())
    for i in range(0, len(annotations_detectron2)):
        annotations_detectron2[i]['image_id'] = i + 1
    with open(os.path.join(annodir, 'annotations.json'), 'w') as fp:
        json.dump(annotations_detectron2, fp)


def detect():
    import detectron2
    from detectron2.engine import DefaultPredictor
    from models import get_cfg_base_model

    cfg = get_cfg_base_model(args.model)
    detector = DefaultPredictor(cfg)

    dst = TrainingFrames(args.id)
    print('detect objects with %s in %s' % (args.model, dst), flush=True)

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

    result_json_zip = os.path.join(dst.lmdb_path, 'detect_%s.json.gz' % args.model)
    with gzip.open(result_json_zip, 'wt') as fp:
        fp.write(json.dumps({'model': args.model, 'classes': thing_classes, 'frames': ifilelist, 'dets': frame_objs, 'args': vars(args)}))
    print('results saved to:', result_json_zip)
    print()


def track_video_multi():
    def run_tracker_multi(vid_arr, dets):
        # args.sot_min_bbox = 20
        track_bboxes = [[] for _ in range(0, len(dets))]
        bbox, scores, labels = dets[0]['bbox'], dets[0]['score'], dets[0]['label']
        _n, _t = 0, time.time()
        for j in range(0, len(scores)):
            if scores[j] < args.sot_score_thres:
                continue
            x1, y1, x2, y2 = bbox[j]
            if x2 - x1 < args.sot_min_bbox or y2 - y1 < args.sot_min_bbox:
                continue
            area_0 = (x2 - x1) * (y2 - y1)
            if random.random() > args.sot_prob:
                continue

            track_bboxes[0].append({'class': labels[j], 'bbox': [x1, y1, x2, y2], 'init_score': scores[j], 'track_length': 0})
            tracker.initialize(vid_arr[0], {'init_bbox': [x1, y1, x2 - x1, y2 - y1], 'init_object_ids': [1], 'object_ids': [1], 'sequence_object_ids': [1]})
            for i_track in range(1, len(dets)):
                out = tracker.track(vid_arr[i_track])
                _n += 1
                x, y, w, h = out['target_bbox']
                if x < 0:
                    break
                area_i = w * h
                if area_i > area_0 * 2 or area_i < area_0 / 2:
                    break
                else:
                    track_bboxes[i_track].append({'class': labels[j], 'bbox': [x, y, x + w, y + h], 'init_score': scores[j], 'track_length': i_track})

        return track_bboxes, _n, time.time() - _t

    dst = TrainingFrames(args.id)
    print('track %s detected objects in %s' % (args.model, dst))
    det_json_zip = os.path.join(dst.lmdb_path, 'detect_%s.json.gz' % args.model)
    with gzip.open(det_json_zip, 'rt') as fp:
        frame_objs = json.loads(fp.read())
    ifilelist, dets = frame_objs['frames'], frame_objs['dets']
    assert len(dets) == len(ifilelist) and len(dets) == len(dst)
    fps = dst.meta['sample_fps']

    assert args.sot_prob > 0, 'wrong args.sot_prob'
    random.seed(42)
    if args.sot_skip < 0:
        sot_every_frame = 1
    else:
        sot_every_frame = int(fps * args.sot_skip)
    sot_idx_list = np.arange(0, len(ifilelist), sot_every_frame)[2 : -2].tolist()
    sot_num_frames = int(fps * args.sot_max_length)
    print('start tracking every %d frames, total %d init points, max track %d frames, min init score %.3f, min init bbox width %d' % (sot_every_frame, len(sot_idx_list), sot_num_frames, args.sot_score_thres, args.sot_min_bbox))

    os.environ['CUDA_HOME'] = '/opt/cuda-11.3'
    sys.path.append(os.path.join(os.environ['HOME'], 'Git', 'pytracking'))
    import pytracking.tracker.dimp as DiMPTracker
    from pytracking.evaluation import Tracker as TrackerWrapper

    wrapper = TrackerWrapper('dimp', 'dimp50')
    params = wrapper.get_parameters()
    params.debug = 0
    params.tracker_name = wrapper.name
    params.param_name = wrapper.parameter_name
    params.output_not_found_box = True
    tracker = wrapper.create_tracker(params)
    tracker.initialize_features()
    print('DiMP-50 tracker initialized', flush=True)

    tracked_total, time_total = 0, 0
    boxes_forward, boxes_backward = [[] for _ in range(0, len(dets))], [[] for _ in range(0, len(dets))]
    sot_json_zip = os.path.join(dst.lmdb_path, 'detect_%s_DiMP.json.gz' % args.model)
    for init_i in tqdm.tqdm(sot_idx_list, ascii=True):
        vid_i = np.stack(list(map(lambda x: x[0], dst[init_i : init_i + sot_num_frames])), axis=0)
        dets_i = dets[init_i : init_i + sot_num_frames]

        forward_i, _n1, _t1 = run_tracker_multi(vid_i, dets_i)
        backward_i, _n2, _t2 = run_tracker_multi(vid_i[::-1], dets_i[::-1])
        backward_i = backward_i[::-1]
        tracked_total += _n1 + _n2
        time_total += _t1 + _t2
        assert len(forward_i) == len(backward_i) and len(forward_i) == len(dets_i)
        for i in range(init_i, init_i + sot_num_frames):
            boxes_forward[i] = boxes_forward[i] + forward_i[i - init_i]
            boxes_backward[i] = boxes_backward[i] + backward_i[i - init_i]

        if 0 == sot_idx_list.index(init_i) % 100:
            with gzip.open(sot_json_zip, 'wt') as fp:
                fp.write(json.dumps({'forward': boxes_forward, 'backward': boxes_backward, 'args': vars(args)}))
    with gzip.open(sot_json_zip, 'wt') as fp:
        fp.write(json.dumps({'forward': boxes_forward, 'backward': boxes_backward, 'args': vars(args)}))
    print('results saved to:', sot_json_zip)
    print('tracked %d frames in %.1f seconds, FPS=%.3f\n' % (tracked_total, time_total, tracked_total / time_total))


def show():
    import skimage.io
    import skvideo.io
    from PIL import Image, ImageDraw, ImageFont

    dst = TrainingFrames(args.id)
    print('visualize %s detected & tracked objects in %s' % (args.model, dst))
    with gzip.open(os.path.join(dst.lmdb_path, 'detect_%s.json.gz' % args.model), 'rt') as fp:
        dets = json.loads(fp.read())['dets']
    with gzip.open(os.path.join(dst.lmdb_path, 'detect_%s_DiMP.json.gz' % args.model), 'rt') as fp:
        data = json.loads(fp.read())
        sot_forward, sot_backward = data['forward'], data['backward']
    assert len(dst) == len(dets) and len(dst) == len(sot_forward) and len(dst) == len(sot_backward)

    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=18)
    output_video = os.path.join(dst.lmdb_path, 'detect_sot_%s.mp4' % args.model)
    writer = skvideo.io.FFmpegWriter(output_video, inputdict={'-r': str(dst.meta['sample_fps'])}, outputdict={'-vcodec': 'libx265', '-r': str(dst.meta['sample_fps']), '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
    for i in tqdm.tqdm(range(0, len(dst)), ascii=True):
        if len(sot_forward[i]) < 1:
            continue

        f = Image.fromarray(dst[i][0], 'RGB')
        draw = ImageDraw.Draw(f)

        # dets_i = dets[i]
        # for j in range(0, len(dets_i['score'])):
        #     if dets_i['score'][j] < 0.75:
        #         continue
        #     x1, y1, x2, y2 = dets_i['bbox'][j]
        #     c = bbox_rgbs[dets_i['label'][j]]
        #     draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=c, width=5)
        #     draw.text((x1 + 2, y1 + 2), 'D %d %s %.2f' % (dets_i['label'][j], thing_classes[dets_i['label'][j]], dets_i['score'][j]), fill=c, font=font)

        sot_i = sot_forward[i] + sot_backward[i]
        for s in sot_i:
            x1, y1, x2, y2 = s['bbox']
            c = bbox_rgbs[s['class']]
            draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=c, width=3)
            draw.text((x1 + 2, y1 + 2), 'T %d %s %.2f(+%d)' % (s['class'], thing_classes[s['class']], s['init_score'], s['track_length']), fill=c, font=font)
        writer.writeFrame(np.array(f))
    writer.close()
    print('results saved to:', output_video)
    print()


def check_files():
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    checklist = ['detect_r50-fpn-3x.json.gz', 'detect_r50-fpn-3x_DiMP.json.gz', 'detect_r101-fpn-3x.json.gz', 'detect_r101-fpn-3x_DiMP.json.gz', 'refine_r101-fpn-3x_r50-fpn-3x.mp4']
    missing = {}
    for vf in files:
        inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', vf['id'])
        if not os.access(inputdir, os.R_OK):
            missing[vf['id']] = 'NOT DECODED'
            continue
        for f in checklist:
            if not os.access(os.path.join(inputdir, f), os.R_OK):
                if not vf['id'] in missing:
                    missing[vf['id']] = []
                missing[vf['id']].append(f)
    print('missing files:')
    for f in missing:
        print(f, missing[f])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection & Tracking')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--sot_score_thres', type=float, default=0.9, help='minimum detection score to start tracking')
    parser.add_argument('--sot_min_bbox', type=int, default=50, help='minimum detection box size to start tracking')
    parser.add_argument('--sot_skip', type=float, default=-1, help='seconds interval to initialize tracking')
    parser.add_argument('--sot_max_length', type=float, default=2, help='maximum seconds of tracking')
    parser.add_argument('--sot_prob', type=float, default=1.0, help='probability of starting track')
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    print(args)
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()

    if args.opt == 'detect':
        detect()
    elif args.opt == 'sot':
        track_video_multi()
    elif args.opt == 'show':
        show()
    elif args.opt == 'check':
        check_files()
    else:
        pass
    del _tensor


'''
python baseline/detect_sot.py --opt detect --id 001 --model r101-fpn-3x
python baseline/detect_sot.py --opt sot --id 001 --model x101-fpn-3x --sot_skip 5 --sot_max_length 2
python baseline/detect_sot.py --opt sot --id 050 --model r50-fpn-3x --sot_skip 4 --sot_max_length 2
'''
