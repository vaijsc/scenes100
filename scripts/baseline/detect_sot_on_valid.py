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

import skvideo.io
import skimage.io

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from decode_training import TrainingFrames

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def convert_jacksonhole():
    vfilename = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'JacksonHole_TownSquare', '20200909_1130_1316.mp4')
    assert os.access(vfilename, os.R_OK), '%s not readable' % vfilename
    fps, sample_fps, sample_interval = 30, 5, 1
    frame_skip = int(fps / sample_fps)
    annodir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', '001')
    with open(os.path.join(annodir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)

    frame_idx_list, context_idx_list = [], []
    for im in annotations:
        i = int(os.path.basename(im['file_name']).split('.')[0])
        frame_idx_list.append(i)
    frame_idx_list = sorted(frame_idx_list)
    for i in frame_idx_list:
        cntxt = np.arange(i - fps * sample_interval, i + fps * sample_interval, frame_skip).astype(int).tolist()
        if not i in cntxt:
            cntxt.append(i)
        cntxt = sorted(cntxt)
        context_idx_list.append(cntxt)

    all_decode_idx_list = []
    for l in context_idx_list:
        all_decode_idx_list = all_decode_idx_list + l
    all_decode_idx_list = set(all_decode_idx_list)
    print('%d annotated frames, %d context frames' % (len(frame_idx_list), len(all_decode_idx_list)))

    reader = skvideo.io.vreader(vfilename)
    for i in tqdm.tqdm(range(0, max(all_decode_idx_list) + 2), ascii=True, desc='decoding ...%s' % vfilename[-40:]):
        try:
            frame = next(reader)
        except StopIteration:
            break
        if i in all_decode_idx_list:
            fn = '%08d.jpg' % i
            skimage.io.imsave(os.path.join(annodir, 'context_frames', fn), frame, quality=80)
    reader.close()

    output_json = os.path.join(annodir, 'context_frames.json')
    with open(output_json, 'w') as fp:
        json.dump({'frame_idx_list': frame_idx_list, 'context_idx_list': context_idx_list}, fp)
    print('saved to:', output_json)


def detect():
    import detectron2
    from detectron2.engine import DefaultPredictor
    from models import get_cfg_base_model

    cfg = get_cfg_base_model(args.model)
    detector = DefaultPredictor(cfg)
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'context_frames.json'), 'r') as fp:
        context_idx_list = json.load(fp)['context_idx_list']
    print('detect objects with %s in %s' % (args.model, inputdir))

    frame_objs = []
    for i_list in tqdm.tqdm(context_idx_list, ascii=True):
        frame_objs.append([])
        for i in i_list:
            im = skimage.io.imread(os.path.join(inputdir, 'context_frames', '%08d.jpg' % i))
            instances = detector(im[:, :, ::-1])['instances'].to('cpu')
            frame_objs[-1].append({
                # bbox has format [x1, y1, x2, y2]
                'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                'score': instances.scores.numpy().tolist(),
                'label': instances.pred_classes.numpy().tolist()
            })

    result_json = os.path.join(inputdir, 'context_frames_detect_%s.json' % args.model)
    with open(result_json, 'w') as fp:
        json.dump({'model': args.model, 'classes': thing_classes, 'dets': frame_objs}, fp)
    print('results saved to:', result_json)


def track_video_multi():
    def run_tracker_multi(vid_arr, dets, context_idx, frame_idx):
        # print(vid_arr.shape, vid_arr.dtype, len(dets), context_idx, frame_idx, context_idx.index(frame_idx))
        track_bboxes = [[] for _ in range(0, len(dets))]
        for start_i in range(0, context_idx.index(frame_idx)):
            bbox, scores, labels = dets[start_i]['bbox'], dets[start_i]['score'], dets[start_i]['label']
            for j in range(0, len(scores)):
                if scores[j] < args.sot_score_thres:
                    continue
                x1, y1, x2, y2 = bbox[j]
                if x2 - x1 < args.sot_min_bbox or y2 - y1 < args.sot_min_bbox:
                    continue
                area_0 = (x2 - x1) * (y2 - y1)

                track_bboxes[start_i].append({'class': labels[j], 'bbox': [x1, y1, x2, y2], 'init_score': scores[j], 'track_length': 0})
                tracker.initialize(vid_arr[start_i], {'init_bbox': [x1, y1, x2 - x1, y2 - y1], 'init_object_ids': [1], 'object_ids': [1], 'sequence_object_ids': [1]})
                for i_track in range(start_i + 1, context_idx.index(frame_idx) + 1):
                    out = tracker.track(vid_arr[i_track])
                    x, y, w, h = out['target_bbox']
                    if x < 0:
                        break
                    area_i = w * h
                    if area_i > area_0 * 2 or area_i < area_0 / 2:
                        break
                    else:
                        track_bboxes[i_track].append({'class': labels[j], 'bbox': [x, y, x + w, y + h], 'init_score': scores[j], 'track_length': i_track - start_i})
        return track_bboxes

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'context_frames.json'), 'r') as fp:
        data = json.load(fp)
    frame_idx_list, context_idx_list = data['frame_idx_list'], data['context_idx_list']
    print('track %s detected objects in %s' % (args.model, inputdir))
    det_json = os.path.join(inputdir, 'context_frames_detect_%s.json' % args.model)
    with open(det_json, 'r') as fp:
        dets = json.load(fp)['dets']
    assert len(dets) == len(frame_idx_list) and len(dets) == len(context_idx_list)

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
    print('DiMP-50 tracker initialized')

    sot_json = os.path.join(inputdir, 'context_frames_detect_%s_DiMP.json' % args.model)
    sot_boxes = []
    for i in tqdm.tqdm(range(0, len(frame_idx_list)), ascii=True):
        vid_i = []
        for j in context_idx_list[i]:
            vid_i.append(skimage.io.imread(os.path.join(inputdir, 'context_frames', '%08d.jpg' % j)))
        vid_i = np.stack(vid_i, axis=0)
        dets_i = dets[i]

        forward_i = run_tracker_multi(vid_i, dets_i, context_idx_list[i], frame_idx_list[i])
        backward_i = run_tracker_multi(vid_i[::-1], dets_i[::-1], context_idx_list[i][::-1], frame_idx_list[i])[::-1]
        assert len(forward_i) == len(backward_i) and len(forward_i) == len(dets_i)
        sot_boxes.append({'forward': forward_i, 'backward': backward_i})

        with open(sot_json, 'w') as fp:
            json.dump({'tracks': sot_boxes, 'args': vars(args)}, fp)
    print('results saved to:', sot_json)


def evaluate_context():
    from detectron2.structures import BoxMode
    import networkx
    from evaluation import evaluate_masked

    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'context_frames.json'), 'r') as fp:
        data = json.load(fp)
    frame_idx_list, context_idx_list = data['frame_idx_list'], data['context_idx_list']
    with open(os.path.join(inputdir, 'context_frames_detect_%s.json' % args.model), 'r') as fp:
        dets = json.load(fp)['dets']
    with open(os.path.join(inputdir, 'context_frames_detect_%s_DiMP.json' % args.model), 'r') as fp:
        sots = json.load(fp)['tracks']
    assert len(frame_idx_list) == len(context_idx_list) == len(dets) == len(sots), '%d %d %d %d' % (len(frame_idx_list), len(context_idx_list), len(dets), len(sots))

    refined = []
    for i in range(0, len(frame_idx_list)):
        im = {'file_name': '%08d.jpg' % frame_idx_list[i], 'image_id': i, 'height': 0, 'width': 0, 'annotations': []}
        ann_i = []
        dets_i = dets[i][context_idx_list[i].index(frame_idx_list[i])]
        for j in range(0, len(dets_i['label'])):
            # if dets_i['score'][j] < args.refine_det_score_thres:
            #     continue
            ann_i.append({'bbox': dets_i['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets_i['label'][j], 'score': dets_i['score'][j]})

        sots_i = sots[i]['forward'][context_idx_list[i].index(frame_idx_list[i])]
        for s in sots_i:
            ann_i.append({'bbox': s['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': s['class'], 'score': s['init_score']})
        sots_i = sots[i]['backward'][context_idx_list[i].index(frame_idx_list[i])]
        for s in sots_i:
            ann_i.append({'bbox': s['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': s['class'], 'score': s['init_score']})

        if not args.not_refine:
            G = networkx.Graph()
            [G.add_node(i) for i in range(0, len(ann_i))]
            for i in range(0, len(ann_i)):
                for j in range(i, len(ann_i)):
                    iou_value = IoU(ann_i[i]['bbox'], ann_i[j]['bbox'])
                    if ann_i[i]['category_id'] == ann_i[j]['category_id'] and iou_value > args.refine_iou_thres:
                        G.add_edge(i, j, iou=iou_value)
            subs = list(networkx.algorithms.components.connected_components(G))

            ann_refine = []
            for sub_nodes in subs:
                max_degree, max_degree_n = -1, -1
                for n in sub_nodes:
                    D = sum(map(lambda t: t[2], list(G.edges(n, data='iou'))))
                    if D > max_degree:
                        max_degree, max_degree_n = D, n
                ann_refine.append(ann_i[max_degree_n])
            ann_i = ann_refine

        im['annotations'] = ann_i
        refined.append(im)
    results = evaluate_masked(args.id, refined, outputfile=os.path.join(inputdir, 'context_frames.%s.eval.mp4' % args.model))
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection & Tracking')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, help='video ID')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--sot_score_thres', type=float, default=0.9, help='minimum detection score to start tracking')
    parser.add_argument('--sot_min_bbox', type=int, default=50, help='minimum detection box size to start tracking')
    # parser.add_argument('--sot_skip', type=float, default=-1, help='seconds interval to initialize tracking')
    # parser.add_argument('--sot_max_length', type=float, help='maximum seconds of tracking')
    parser.add_argument('--not_refine', type=bool, default=False)
    # parser.add_argument('--refine_det_score_thres', type=float, default=0.75, help='minimum detection score in pseudo annotation')
    parser.add_argument('--refine_iou_thres', type=float, default=0.75, help='IoU threshold to merge boxes')
    args = parser.parse_args()
    print(args)

    # convert_jacksonhole()

    if args.opt == 'detect':
        detect()
    elif args.opt == 'sot':
        track_video_multi()
    elif args.opt == 'eval':
        evaluate_context()
    else:
        pass

'''
python baseline/detect_sot_on_valid.py --opt detect --id 001 --model xr101-fpn-3x
python baseline/detect_sot_on_valid.py --opt sot --id 001 --model rr50-fpn-3x
python baseline/detect_sot_on_valid.py --opt eval --id 001 --model rr50-fpn-3x --not_refine 1
'''
