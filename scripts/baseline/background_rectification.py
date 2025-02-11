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

from decode_validation import ValidationContextFrames
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
class Dummy(object): pass


def detect(args):
    import detectron2
    from detectron2.engine import DefaultPredictor
    from models import get_cfg_base_model

    dst = ValidationContextFrames(args.id)
    print('detect objects with %s in %s' % (args.model, dst))
    if not args.model in detectors:
        cfg = get_cfg_base_model(args.model)
        detectors[args.model] = DefaultPredictor(cfg)
    keys, indices = dst.context_indices()
    detect_indices_set = []
    for _idx in indices:
        detect_indices_set = detect_indices_set + [_idx[i] for i in range(0, len(_idx), int(4 * dst.meta['sample_fps']))] + _idx[-1 * int(dst.meta['sample_fps']) - 1 :]
    detect_indices_set = list(set(detect_indices_set))
    frame_objs, ifilelist = [None] * len(dst), [None] * len(dst)
    for i in tqdm.tqdm(detect_indices_set, ascii=True):
        im, _, fn, _ = dst[i]
        ifilelist[i] = fn
        instances = detectors[args.model](im[:, :, ::-1])['instances'].to('cpu')
        frame_objs[i] = {
            # bbox has format [x1, y1, x2, y2]
            'bbox': instances.pred_boxes.tensor.numpy().tolist(),
            'score': instances.scores.numpy().tolist(),
            'label': instances.pred_classes.numpy().tolist()
        }
    result_json = os.path.join(dst.lmdb_path, 'detect_%s.json' % args.model)
    with open(result_json, 'w') as fp:
        json.dump({'model': args.model, 'classes': thing_classes, 'frames': ifilelist, 'dets': frame_objs, 'args': vars(args)}, fp)
    print('results saved to:', result_json)


def track_video_multi(args):
    def run_tracker_multi(vid_arr, dets):
        track_bboxes = [[] for _ in range(0, len(dets))]
        bbox, scores, labels = dets[0]['bbox'], dets[0]['score'], dets[0]['label']
        for j in range(0, len(scores)):
            if scores[j] < args.sot_score_thres: continue
            x1, y1, x2, y2 = bbox[j]
            if x2 - x1 < args.sot_min_bbox or y2 - y1 < args.sot_min_bbox: continue
            area_0 = (x2 - x1) * (y2 - y1)
            track_bboxes[0].append({'class': labels[j], 'bbox': [x1, y1, x2, y2], 'init_score': scores[j], 'track_length': 0})
            trackers[0].initialize(vid_arr[0], {'init_bbox': [x1, y1, x2 - x1, y2 - y1], 'init_object_ids': [1], 'object_ids': [1], 'sequence_object_ids': [1]})
            for i_track in range(1, len(dets)):
                out = trackers[0].track(vid_arr[i_track])
                x, y, w, h = out['target_bbox']
                if x < 0: break
                area_i = w * h
                if area_i > area_0 * 2 or area_i < area_0 / 2: break
                else: track_bboxes[i_track].append({'class': labels[j], 'bbox': [x, y, x + w, y + h], 'init_score': scores[j], 'track_length': i_track})
        return track_bboxes

    dst = ValidationContextFrames(args.id)
    print('track %s detected objects in %s' % (args.model, dst))
    with open(os.path.join(dst.lmdb_path, 'detect_%s.json' % args.model), 'r') as fp:
        frame_objs = json.load(fp)['dets']
    keys, indices = dst.context_indices()

    os.environ['CUDA_HOME'] = '/opt/cuda-11.3'
    sys.path.append(os.path.join(os.environ['HOME'], 'Git', 'pytracking'))
    import pytracking.tracker.dimp as DiMPTracker
    from pytracking.evaluation import Tracker as TrackerWrapper
    if trackers[0] is None:
        wrapper = TrackerWrapper('dimp', 'dimp50')
        params = wrapper.get_parameters()
        params.debug = 0
        params.tracker_name = wrapper.name
        params.param_name = wrapper.parameter_name
        params.output_not_found_box = True
        trackers[0] = wrapper.create_tracker(params)
        trackers[0].initialize_features()
        print('DiMP-50 tracker initialized')

    skip = 4
    sot_boxes = [[] for _ in range(0, len(dst))]
    for _idx in tqdm.tqdm(indices, ascii=True):
        context_idx = _idx[-1 * int(dst.meta['sample_fps']) - 1 :]
        vid_arr_sample = np.stack([dst[i][0] for i in context_idx], axis=0)
        dets_sample = [frame_objs[i] for i in context_idx]
        forward = run_tracker_multi(vid_arr_sample, dets_sample)
        for j in range(0, len(context_idx)):
            sot_boxes[context_idx[j]] = sot_boxes[context_idx[j]] + forward[j]
    sot_json = os.path.normpath(os.path.join(dst.lmdb_path, 'detect_%s_DiMP.json' % args.model))
    with open(sot_json, 'w') as fp:
        json.dump({'sot_boxes': sot_boxes, 'args': vars(args)}, fp)
    print('results saved to:', sot_json)


def refine_annotations_valid(args, visualize=False):
    import contextlib
    from finetune import _graph_refine, BoxMode
    from PIL import Image, ImageDraw, ImageFont

    dst = ValidationContextFrames(args.id)
    det_list = [os.path.join(dst.lmdb_path, 'detect_%s.json' % m) for m in args.anno_models]
    for i in range(0, len(det_list)):
        with open(det_list[i], 'r') as fp:
            det_list[i] = json.load(fp)['dets']
    sot_list = [os.path.join(dst.lmdb_path, 'detect_%s_DiMP.json' % m) for m in args.anno_models]
    for i in range(0, len(sot_list)):
        with open(sot_list[i], 'r') as fp:
            sot_list[i] = json.load(fp)['sot_boxes']
    N = len(dst)
    for lst in det_list: assert len(lst) == N
    for lst in sot_list: assert len(lst) == N
    keys, indices = dst.context_indices()

    # collate bboxes from tracking & detection
    dict_json = {}
    for k in range(0, len(keys)):
        dict_json[keys[k]] = []
        for i in indices[k]:
            _detected = False
            for lst in det_list:
                if not lst[i] is None:
                    _detected = True
            if not _detected:
                continue
            annotations = []
            for lst in det_list:
                if not lst[i] is None:
                    for j in range(0, len(lst[i]['score'])):
                        if lst[i]['score'][j] < args.refine_det_score_thres:
                            continue
                        annotations.append({'bbox': lst[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': lst[i]['label'][j], 'src': 'det', 'score': lst[i]['score'][j]})
            for lst in sot_list:
                for tr in lst[i]:
                    annotations.append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
            dict_json[keys[k]].append({'file_name': '', 'dst_i': i, 'image_id': 0, 'height': None, 'width': None, 'annotations': annotations})

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        for _f in json.load(fp):
            if _f['id'] == args.id:
                meta = _f['video']
                break
    count_all, count_bboxes = 0, 0
    for key in dict_json:
        for im in dict_json[key]:
            im['width'], im['height'] = meta['W'], meta['H']
            count_all += len(im['annotations'])
        with contextlib.redirect_stderr(open(os.devnull, 'w')):
            ret = _graph_refine({'dict': dict_json[key], 'args': args, 'desc': key})
        dict_json[key] = ret[0]
        count_bboxes += ret[1]
    # print('refine bboxes %d => %d' % (count_all, count_bboxes))

    if visualize:
        font_label = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=18)
        font_title = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=60)
        output_video = os.path.join(dst.lmdb_path, 'refine_%s.mp4' % '_'.join(args.anno_models))
        writer = skvideo.io.FFmpegWriter(output_video, inputdict={'-r': str(dst.meta['sample_fps'])}, outputdict={'-vcodec': 'hevc_nvenc', '-r': str(dst.meta['sample_fps']), '-pix_fmt': 'yuv420p', '-preset': 'medium', '-rc': 'vbr', '-cq': '25'})
        for k in tqdm.tqdm(keys, ascii=True, desc='writing video'):
            for im in dict_json[k]:
                f = Image.fromarray(dst[im['dst_i']][0])
                draw = ImageDraw.Draw(f)
                for ann in im['annotations']:
                    x1, y1, x2, y2 = ann['bbox']
                    c = bbox_rgbs[ann['category_id']]
                    draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=c, width=5)
                    if ann['src'] == 'det':
                        _text = 'D/%.3f' % ann['score']
                    elif ann['src'] == 'sot':
                        _text = 'T/%.3f/%d' % (ann['init_score'], ann['track_length'])
                    else: raise Exception('unknown source')
                    draw.text((x1 + 2, y1 + 2), _text, fill=c, font=font_label)
                _text = 'Pseudo: %s' % (' + '.join(args.anno_models))
                draw.text((2, 2), _text, fill='#000000', stroke_width=3, font=font_title)
                draw.text((2, 2), _text, fill='#FFFFFF', stroke_width=1, font=font_title)
                writer.writeFrame(np.array(f))
        writer.close()
        print('refined video saved to:', output_video)
    return dict_json, keys, count_bboxes


def dynamic_background(args):
    from collections import deque
    from PIL import Image, ImageDraw, ImageFont
    from finetune import BoxMode

    dict_json, keys, _ = refine_annotations_valid(args, visualize=False)
    # dict_json, keys, _ = refine_annotations_valid(args, visualize=True)
    dst = ValidationContextFrames(args.id)
    N = len(dst)
    frames, masks, backgrounds, averageds, filenames = [], [], [], [], []
    Q = deque([], maxlen=120)
    for k in tqdm.tqdm(keys, ascii=True, desc=args.id):
        for im in dict_json[k]:
            im_arr, _, fn, _ = dst[im['dst_i']]
            Q.append({'im_arr': im_arr, 'fn': fn, 'annotations': im['annotations']})
        images_arr = np.stack(list(map(lambda x: x['im_arr'], Q))).astype(np.float16)
        frames.append(Q[-1]['im_arr'].astype(np.uint8))
        averageds.append(images_arr.mean(axis=0).astype(np.uint8))
        annotations = list(map(lambda x: x['annotations'], Q))
        weights = np.ones_like(images_arr)
        for j in range(0, len(Q)):
            for ann in annotations[j]:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = map(int, map(lambda x: max(x, 0), ann['bbox']))
                # if ann['category_id'] == 0 and x2 - x1 > 500:
                #     continue
                # if x2 - x1 > 150:
                #     continue
                images_arr[j, y1 : y2, x1 : x2] = 0.0
                weights[j, y1 : y2, x1 : x2] = 0.0
        images_arr, weights = images_arr.mean(axis=0), weights.mean(axis=0)
        M = np.zeros(shape=weights.shape, dtype=np.uint8) + 255
        for x in range(0, images_arr.shape[0]):
            for y in range(0, images_arr.shape[1]):
                if weights[x, y, 0] < 1e-5:
                    weights[x, y], images_arr[x, y] = 1, averageds[-1][x, y]
                    M[x, y] = 0
        backgrounds.append((images_arr / weights).astype(np.uint8))
        masks.append(M)
        filenames.append(k)
    frames, masks, backgrounds, averageds, filenames = np.stack(frames, axis=0), np.stack(masks, axis=0), np.stack(backgrounds, axis=0), np.stack(averageds, axis=0), np.array(filenames)

    outputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', args.id)
    np.savez_compressed(os.path.join(dst.lmdb_path, 'dynamic_background.npz'), frames=frames, masks=masks, backgrounds=backgrounds, averageds=averageds, filenames=filenames)
    font_title = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=40)
    writer = skvideo.io.FFmpegWriter(os.path.join(dst.lmdb_path, 'dynamic_background.mp4'), inputdict={'-r': '2'}, outputdict={'-vcodec': 'hevc_nvenc', '-r': '2', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-rc': 'vbr', '-cq': '25'})
    _, X, Y, C = backgrounds.shape
    for i in range(0, backgrounds.shape[0]):
        canvas = np.zeros(shape=(2 * X, 2 * Y, C), dtype=np.uint8)
        canvas[: X, : Y], canvas[: X, Y :], canvas[X :, : Y], canvas[X :, Y :] = frames[i], masks[i], backgrounds[i], averageds[i]
        canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas)
        for (pos, txt) in [((2, 2), str(filenames[i])), ((Y + 2, 2), 'mask'), ((2, X + 2), 'background'), ((Y + 2, X + 2), 'average')]:
            draw.text(pos, txt, fill='#000000', stroke_width=3, font=font_title)
            draw.text(pos, txt, fill='#FFFFFF', stroke_width=1, font=font_title)
        writer.writeFrame(np.array(canvas))
    writer.close()


def rectify(args):
    import contextlib
    import matplotlib.patches as patches
    from finetune import BoxMode
    from evaluation import evaluate_masked, eval_AP

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    def _sigmoid_inv(x):
        assert x >= 0 and x <= 1, x
        if x > 0.9999: x = 0.9999
        if x < 0.0001: x = 0.0001
        return np.log(x / (1 - x))

    dst = ValidationContextFrames(args.id)
    with open(os.path.join(dst.lmdb_path, 'detect_%s.json' % args.model), 'r') as fp:
        _data = json.load(fp)
        ifilelist, frame_objs = _data['frames'], _data['dets']
    _data = np.load(os.path.join(dst.lmdb_path, 'dynamic_background.npz'))
    frames, masks, backgrounds, filenames = _data['frames'], _data['masks'], _data['backgrounds'], list(map(str, _data['filenames']))
    _data.close()
    N = len(filenames)
    diffs, detections = [None] * N, [None] * N

    for i in range(0, N):
        dets_i = frame_objs[ifilelist.index(filenames[i])]
        score, bbox, label = dets_i['score'], dets_i['bbox'], dets_i['label']
        detections[i] = {'file_name': filenames[i], 'image_id': i, 'height': 0, 'width': 0, 'annotations': []}
        for j in range(0, len(score)):
            detections[i]['annotations'].append({'bbox': bbox[j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[j], 'score': score[j]})
        _diff = np.absolute(frames[i].astype(np.float16) - backgrounds[i].astype(np.float16) * (masks[i] / 255))
        diffs[i] = _diff.mean(axis=2) / 255.0
    diffs = np.stack(diffs, axis=0)
    diffs_mask = np.ones_like(diffs)
    diffs_mask[np.where(diffs < 0.15)] = 0

    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results1 = evaluate_masked(args.id, detections)['results']

    for i in range(0, N):
        for j in range(0, len(detections[i]['annotations'])):
            x1, y1, x2, y2 = map(lambda x: int(max(0, x)), detections[i]['annotations'][j]['bbox'])
            detections[i]['annotations'][j]['diff_mean'] = diffs[i][y1 : y2, x1 : x2].mean()
            detections[i]['annotations'][j]['diff_mask_mean'] = diffs_mask[i][y1 : y2, x1 : x2].mean()

    for i in range(0, N):
        annotations_filter = list(filter(lambda x: x['score'] > 0.3, detections[i]['annotations']))
        axes = plt.subplots(2, 2)[1].reshape(-1)
        axes[0].imshow(frames[i])
        for ann in annotations_filter:
            (x1, y1, x2, y2), c, s = map(lambda x: int(max(0, x)), ann['bbox']), bbox_rgbs[ann['category_id']], ann['score']
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=c, facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text((x1 + x2) / 2, (y1 + y2) / 2, '%.3f' % s, size=8, color=c)
        axes[0].set_title('%s %s' % (args.id, filenames[i]))
        axes[1].imshow((backgrounds[i].astype(np.float16) * (masks[i] / 255)).astype(np.uint8))
        axes[2].imshow(diffs[i])
        # for ann in annotations_filter:
        #     x1, y1, x2, y2 = map(lambda x: int(max(0, x)), ann['bbox'])
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='white', facecolor='none')
        #     axes[2].add_patch(rect)
        #     axes[2].text((x1 + x2) / 2, (y1 + y2) / 2, '%.3f' % ann['diff_mean'], size=8, color='white')
        axes[3].imshow(diffs_mask[i])
        # for ann in annotations_filter:
        #     x1, y1, x2, y2 = map(lambda x: int(max(0, x)), ann['bbox'])
        #     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='white', facecolor='none')
        #     axes[3].add_patch(rect)
        #     axes[3].text((x1 + x2) / 2, (y1 + y2) / 2, '%.3f' % ann['diff_mask_mean'], size=8, color='white')
        plt.tight_layout()
        plt.show()

    results2 = []
    for alpha in args.alphas:
        detections_rectify = copy.deepcopy(detections)
        for i in range(0, N):
            for ann in detections_rectify[i]['annotations']:
                # ann['score'] = _sigmoid(_sigmoid_inv(ann['score']) * alpha + _sigmoid_inv(ann['diff_mean']) * (1 - alpha))
                ann['score'] = _sigmoid(_sigmoid_inv(ann['score']) * alpha + _sigmoid_inv(ann['diff_mask_mean']) * (1 - alpha))

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results2.append(evaluate_masked(args.id, detections_rectify)['results'])

    return results1, results2


if __name__ == '__main__':
    # vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    # args = Dummy()
    # trackers, detectors = [None], {}
    # t0, i = time.time(), 0
    # for vid in vid_list:
    #     print('\n\n%s' % vid)
    #     for m in ['r50-fpn-3x', 'r101-fpn-3x']:
    #         args.id, args.model, args.sot_score_thres, args.sot_min_bbox = vid, m, 0.9, 12
    #         detect(args)
    #         track_video_multi(args)
    #     i += 1
    #     print('[%d/%d finished]' % (i, len(vid_list)), '[%.1f minutes]' % ((time.time() - t0) / 60.0), flush=True)

    # from multiprocessing import Pool as ProcessPool
    # vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    # params_list = []
    # for vid in vid_list:
    #     args = Dummy()
    #     args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = vid, sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.5, 0.85, False
    #     params_list.append(args)
    # procs = 2
    # pool = ProcessPool(processes=procs)
    # _ = pool.map_async(dynamic_background, params_list).get()
    # pool.close()
    # pool.join()

    # args = Dummy()
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '008', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.65, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '027', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.76, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '056', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.65, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '118', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.65, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '146', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.65, 0.85, False
    # args.id, args.anno_models, args.refine_det_score_thres, args.refine_iou_thres, args.refine_remove_no_sot = '152', sorted(['r50-fpn-3x', 'r101-fpn-3x']), 0.65, 0.85, False
    # dynamic_background(args)

    # categories = ['person', 'vehicle', 'overall', 'weighted']
    # vid_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    # alphas = np.arange(0, 1.01, 0.1).tolist()[::-1]

    # rets = []
    # for vid in tqdm.tqdm(vid_list, ascii=True):
    #     args = Dummy()
    #     args.id, args.model, args.alphas = vid, 'r101-fpn-3x', alphas
    #     rets.append(rectify(args))

    # for i in range(0, len(alphas)):
    #     improvements = {cat: [] for cat in categories}
    #     for ap1, ap2 in rets:
    #         ap2 = ap2[i]
    #         for cat in categories:
    #             if ap1[cat][0] > 0:
    #                 improvements[cat].append([ap2[cat][0] - ap1[cat][0], ap2[cat][1] - ap1[cat][1]])
    #     print('alpha =', alphas[i])
    #     for cat in categories:
    #         improvements[cat] = np.array(improvements[cat]) * 100
    #         print(cat, improvements[cat].shape, improvements[cat].mean(axis=0))

    pass
