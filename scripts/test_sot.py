#!python3

import os
import sys
import time
import datetime
import json
import copy
import math
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skvideo.io


output_dir = os.path.join(os.path.dirname(__file__), '..', 'images', 'test_det_sot')
thing_classes = ['person', 'vehicle']

def detect(v):
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'models', 'mscoco2017_remap_x101-fpn-3x.pth')
    detector = DefaultPredictor(cfg)

    npz = np.load(os.path.join(output_dir, v + '.npz'))
    frames = npz['frames']
    npz.close()

    for i in tqdm.tqdm(range(0, frames.shape[0]), ascii=True, desc='detect in ' + v):
        frame_objs = []
        instances = detector(frames[i, :, :, ::-1])['instances'].to('cpu')
        frame_objs.append({
            # bbox has format [x1, y1, x2, y2]
            'bbox': instances.pred_boxes.tensor.numpy().tolist(),
            'score': instances.scores.numpy().tolist(),
            'label': instances.pred_classes.numpy().tolist()
        })
    with open(os.path.join(output_dir, v + '.x101.json'), 'w') as fp:
        json.dump(frame_objs, fp)


def track_video_multi(v, max_length=1000):
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

    with open(os.path.join(output_dir, 'manual.json'), 'r') as fp:
        annotations_coco = json.load(fp)
    images_coco, annotations_coco, image_id_coco = annotations_coco['images'], annotations_coco['annotations'], None
    for im in images_coco:
        if im['file_name'][:3] == v:
            image_id_coco = im['id']
    annotations_coco = list(filter(lambda ann: ann['image_id'] == image_id_coco, annotations_coco))

    npz = np.load(os.path.join(output_dir, v + '.npz'))
    frames = npz['frames']
    npz.close()

    track_results = {}
    for fps in [30, 15, 6, 2]:
        frame_indices = np.arange(0, 360, 30.0 / fps).astype(int)
        vid_arr = frames[frame_indices]
        track_bboxes_all_tracks = []
        for i in tqdm.tqdm(range(0, len(annotations_coco)), ascii=True, desc='%s FPS=%d' % (v, fps)):
            x, y, w, h = annotations_coco[i]['bbox']
            track_bboxes = [None] * vid_arr.shape[0]
            track_bboxes[0] = [x, y, x + w, y + h]

            tracker.initialize(vid_arr[0], {'init_bbox': [x, y, w, h], 'init_object_ids': [1], 'object_ids': [1], 'sequence_object_ids': [1]})
            for i_track in range(1, min(vid_arr.shape[0], max_length)):
                out = tracker.track(vid_arr[i_track])
                x, y, w, h = out['target_bbox']
                if x < 0:
                    break
                else:
                    track_bboxes[i_track] = [x, y, x + w, y + h]
            track_bboxes_all_tracks.append(track_bboxes)
        track_results[fps] = [frame_indices.tolist(), track_bboxes_all_tracks]

    with open(os.path.join(output_dir, v + '.dimp.json'), 'w') as fp:
        json.dump(track_results, fp)


def draw_bbox(v):
    from PIL import Image, ImageDraw, ImageFont

    with open(os.path.join(output_dir, v + '.dimp.json'), 'r') as fp:
        tracks = json.load(fp)
    assert len(tracks) == 4
    tracks_ = {int(k): tracks[k] for k in tracks}
    tracks = tracks_
    fps_list = list(tracks.keys())
    count_tracks = len(tracks[fps_list[0]][1])
    for k in fps_list:
        assert count_tracks == len(tracks[k][1])
    color_tracks = []
    for _ in range(0, count_tracks):
        color_tracks.append(''.join(['#'] + ['0123456789ABCEDF'[random.randint(0, 15)] for _ in range(0, 6)]))

    annotations = []
    for fps in fps_list:
        frame_indices, bboxes = tracks[fps]
        ann = [{'index': frame_indices[j], 'bboxes': []} for j in range(0, len(frame_indices))]
        for i in range(0, len(bboxes)):
            for j in range(0, len(frame_indices)):
                if not bboxes[i][j] is None:
                    ann[j]['bboxes'].append({'color': color_tracks[i], 'xyxy': bboxes[i][j]})
        ann_pad = []
        for i in range(0, len(ann)):
            for _ in range(0, 30 // fps):
                ann_pad.append(ann[i])
        annotations.append(ann_pad)

    # frames = [np.ones((1080, 1920, 3), dtype=np.uint8) * 255] * 450
    npz = np.load(os.path.join(output_dir, v + '.npz'))
    frames = npz['frames']
    npz.close()

    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=100)
    writer = skvideo.io.FFmpegWriter(os.path.join(output_dir, v + '.track.mp4'), inputdict={'-r': '30'}, outputdict={'-vcodec': 'libx265', '-r': '30', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
    for i in tqdm.tqdm(range(0, len(annotations[0])), ascii=True, desc='%s %d tracks' % (v, count_tracks)):
        mosaic = []
        for fps_i in range(0, len(fps_list)):
            ann = annotations[fps_i]
            f = Image.fromarray(frames[ann[i]['index']], 'RGB')
            draw = ImageDraw.Draw(f)
            draw.text((2, 2), 'FPS=%d' % fps_list[fps_i], fill='#FFFFFF', font=font)
            bboxes = ann[i]['bboxes']
            for b in bboxes:
                x1, y1, x2, y2 = b['xyxy']
                draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=b['color'], width=5)
            mosaic.append(np.array(f))
        m1, m2 = np.concatenate([mosaic[0], mosaic[1]], axis=1), np.concatenate([mosaic[2], mosaic[3]], axis=1)
        writer.writeFrame(np.concatenate([m1, m2], axis=0))
    writer.close()


def decode():
    for i, v, start in [('001', '001.JacksonHoleTownSquare_20200910_210625.mp4', 9155), ('046', '046.SantaClausVillage_20211126_100205.hevc.mp4', 6762), ('060', '060.WestfieldParkSquare_20211130_153220.mp4', 0), ('069', '069.PanteleimonivskaSt_20211130_082715.hevc.mp4', 0), ('146', '146.CrossShinjukuBuilding_20211207_023655.hevc.mp4', 0)]:
        vfilename, frames = os.path.join('..', 'videos', v), []
        reader = skvideo.io.vreader(vfilename)
        for _ in range(0, start):
            next(reader)
        for _ in tqdm.tqdm(range(0, 360), ascii=True, desc=vfilename):
            frames.append(next(reader))
        reader.close()
        skimage.io.imsave(os.path.join(output_dir, i + '.jpg'), frames[0], quality=95)
        frames = np.stack(frames, axis=0)
        np.savez_compressed(os.path.join(output_dir, i + '.npz'), frames=frames)


if __name__ == '__main__':
    # decode()
    # detect(v)
    for v in ['001', '046', '060', '069', '146']:
        # track_video_multi(v)
        draw_bbox(v)
        pass
    # track_video_multi('001')
    # draw_bbox('001')
