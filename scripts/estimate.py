#!python3

import os
import glob
import json
import random
import time
import shutil
import tqdm

import skimage.io
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt
import imantics
from PIL import Image, ImageDraw, ImageFont

thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
mask_rgb = [0, 1, 0]


def detect():
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.config import get_cfg

    cfg = get_cfg()
    det_model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    cfg.merge_from_file(model_zoo.get_config_file(det_model))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'models', 'mscoco2017_remap_r50-fpn-3x.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    detector = DefaultPredictor(cfg)

    frame_objs = []
    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        vfiles = json.load(fp)
    for v in tqdm.tqdm(vfiles, ascii=True, desc='detecting in frames'):
        framesdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'train_200_valid_200', '%03d' % v['id'])
        with open(os.path.join(framesdir, 'frames.json'), 'r') as fp:
            frames = np.array(json.load(fp)['valid'][:200])
        frames = frames.reshape(4, -1)[:, 0].tolist()
        for f in frames:
            src = os.path.join(framesdir, f)
            tgt = os.path.join(os.path.dirname(__file__), '..', 'images', 'valid_4', v['id'] + '_' + f)
            shutil.copy2(src, tgt)

            im = skimage.io.imread(tgt)
            instances = detector(im[:, :, ::-1])['instances'].to('cpu')
            frame_objs.append({
                'image': os.path.basename(tgt),
                # bbox has format [x1, y1, x2, y2]
                'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                'score': instances.scores.numpy().tolist(),
                'label': instances.pred_classes.numpy().tolist()
            })

    with open(os.path.join(os.path.dirname(__file__), '..', 'images', 'valid_4', 'fasterrcnn_r50-fpn-3x.json'), 'w') as fp:
        json.dump({'model': det_model, 'checkpoint': cfg.MODEL.WEIGHTS, 'classes': thing_classes, 'dets': frame_objs}, fp)


def draw_bbox():
    def _check_overlap(M, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = M.shape[:2]
        x1 = min(max(x1, 0), W - 1)
        x2 = min(max(x2, 0), W - 1)
        y1 = min(max(y1, 0), H - 1)
        y2 = min(max(y2, 0), H - 1)
        return (M[y1, x1, 0] > 1e-3 or M[y1, x2, 0] > 1e-3 or M[y2, x1, 0] > 1e-3 or M[y2, x2, 0] > 1e-3)

    thres = 0.5
    alpha = 0.3
    with open(os.path.join(os.path.dirname(__file__), '..', 'masks.json'), 'r') as fp:
        masks = json.load(fp)
    masks = {m['video']: m['polygons'] for m in masks}

    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'valid_4')
    outputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'valid_4_draw')
    with open(os.path.join(inputdir, 'fasterrcnn_r50-fpn-3x.json'), 'r') as fp:
        data = json.load(fp)
    for d in tqdm.tqdm(data['dets'], ascii=True, desc='drawing bboxes'):
        image, bbox, score, label = d['image'], d['bbox'], d['score'], d['label']
        im_arr = skimage.io.imread(os.path.join(inputdir, image))
        polygons = masks[image[:3]]
        if len(polygons) > 0:
            ann = imantics.Annotation.from_polygons(polygons, image=imantics.Image.from_path(os.path.join(inputdir, image)))
            m_arr = np.expand_dims(ann.array.astype(np.float16), 2)
            m_arr = m_arr * alpha
            im_arr = ((1 - m_arr) * im_arr + m_arr * (np.array(mask_rgb) * 255).astype(np.uint8).reshape(1, 1, 3)).astype(np.uint8)
        else:
            m_arr = np.zeros_like(im_arr)[:, :, 0:1]

        fontsize, linewidth = int(im_arr.shape[0] * 0.012), int(im_arr.shape[0] / 400)
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=fontsize)
        im_arr = Image.fromarray(im_arr, 'RGB')
        draw = ImageDraw.Draw(im_arr)
        # bbox has format [x1, y1, x2, y2]
        bbox_count = 0
        for i in range(0, len(score)):
            if score[i] < thres:
                continue
            ci = label[i]
            ni = thing_classes[ci]
            if _check_overlap(m_arr, bbox[i]):
                continue
            bbox_count += 1
            x1, y1, x2, y2 = bbox[i]
            draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[ci], width=linewidth)
            draw.text((x1 + 3, y1 + 1), '%s %.3f' % (ni, score[i]), fill=bbox_rgbs[ci], font=font)
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=fontsize * 5)
        draw.text((2, 2), 'boxes: %d' % bbox_count, fill='#FFFFFF', font=font)
        im_arr = np.array(im_arr)
        skimage.io.imsave(os.path.join(outputdir, image), im_arr, quality=80)


if __name__ == '__main__':
    # detect()
    draw_bbox()
