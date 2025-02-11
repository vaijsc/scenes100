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


def track_video_multi():
    def run_tracker_multi(vid_arr, dets, init_frame_i):
        track_bboxes = []
        bbox, scores, labels = dets[0]['bbox'], dets[0]['score'], dets[0]['label']
        _n, _t = 0, time.time()
        for j in range(0, len(scores)):
            if scores[j] < args.sot_score_thres:
                continue
            x1, y1, x2, y2 = bbox[j]
            if x2 - x1 < args.sot_min_bbox or y2 - y1 < args.sot_min_bbox:
                continue
            if random.random() > args.sot_prob:
                continue

            track_j = [{'class': labels[j], 'bbox': [x1, y1, x2, y2], 'init_score': scores[j], 'frame_i': init_frame_i, 'H': vid_arr.shape[1], 'W': vid_arr.shape[2]}]
            tracker.initialize(vid_arr[0], {'init_bbox': [x1, y1, x2 - x1, y2 - y1], 'init_object_ids': [1], 'object_ids': [1], 'sequence_object_ids': [1]})
            for i_track in range(1, len(dets)):
                out = tracker.track(vid_arr[i_track])
                _n += 1
                x, y, w, h = out['target_bbox']
                if x < 0:
                    break
                track_j.append({'class': labels[j], 'bbox': [x, y, x + w, y + h], 'init_score': scores[j], 'frame_i': init_frame_i + i_track, 'H': vid_arr.shape[1], 'W': vid_arr.shape[2]})
            track_bboxes.append(track_j)

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
    tracks_all = []
    sot_json_zip = os.path.join(os.path.dirname(__file__), '%s_%s_DiMP.json.gz' % (args.id, args.model))
    for init_i in tqdm.tqdm(sot_idx_list, ascii=True):
        vid_i = np.stack(list(map(lambda x: x[0], dst[init_i : init_i + sot_num_frames])), axis=0)
        dets_i = dets[init_i : init_i + sot_num_frames]

        tracks_i, _n, _t = run_tracker_multi(vid_i, dets_i, init_i)
        tracked_total += _n
        time_total += _t
        tracks_all = tracks_all + tracks_i

        if 0 == sot_idx_list.index(init_i) % 50:
            with gzip.open(sot_json_zip, 'wt') as fp:
                fp.write(json.dumps({'tracks': tracks_all, 'args': vars(args)}))
    with gzip.open(sot_json_zip, 'wt') as fp:
        fp.write(json.dumps({'tracks': tracks_all, 'args': vars(args)}))
    print('results saved to:', sot_json_zip)
    print('tracked %d frames in %.1f seconds, FPS=%.3f\n' % (tracked_total, time_total, tracked_total / time_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--sot_score_thres', type=float, default=0.9, help='minimum detection score to start tracking')
    parser.add_argument('--sot_min_bbox', type=int, default=16, help='minimum detection box size to start tracking')
    parser.add_argument('--sot_skip', type=float, default=-1, help='seconds interval to initialize tracking')
    parser.add_argument('--sot_max_length', type=float, default=2, help='maximum seconds of tracking')
    parser.add_argument('--sot_prob', type=float, default=1.0, help='probability of starting track')
    parser.add_argument('--hold', default=0.005, type=float)
    args = parser.parse_args()
    print(args)
    assert args.hold > 0
    _tensor = torch.ones(max(1, int(args.hold * 1000)), 1000, 1000, dtype=torch.int8).cuda()

    if args.opt == 'sot':
        track_video_multi()
    else:
        pass
    del _tensor

'''
python sot.py --opt sot --id 001 --model r101-fpn-3x --sot_skip 4 --sot_max_length 4 --sot_prob 0.1
'''
