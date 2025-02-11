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

import detectron2
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg

from models import get_cfg_base_model
from evaluation import evaluate as coco_evaluator


input_dir = os.path.join(os.path.dirname(__file__), '..', 'images', '20200909_1130_1316.mp4_finetune')
output_dir = os.path.join(os.path.dirname(__file__), 'base_model_output')
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def detect():
    cfg = get_cfg_base_model('r50-fpn-3x')
    # cfg = get_cfg_base_model('r50-c4-1x')
    # cfg = get_cfg_base_model('x101-fpn-3x')
    # cfg = get_cfg_base_model('r50-retina-3x')
    detector = DefaultPredictor(cfg)

    detections = []
    font_label = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=18)
    font_title = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=60)
    writer = skvideo.io.FFmpegWriter(os.path.join(output_dir, '001.detect.mp4'), inputdict={'-r': '5'}, outputdict={'-vcodec': 'libx265', '-r': '5', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
    ifilelist = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    for f in tqdm.tqdm(ifilelist, ascii=True, desc='detecting'):
        im = skimage.io.imread(f)
        instances = detector(im[:, :, ::-1])['instances'].to('cpu')
        # bbox has format [x1, y1, x2, y2]
        bbox = instances.pred_boxes.tensor.numpy().tolist()
        score = instances.scores.numpy().tolist()
        label = instances.pred_classes.numpy().tolist()
        im = Image.fromarray(im, 'RGB')
        draw = ImageDraw.Draw(im)
        for i in range(0, len(score)):
            if score[i] < 0.5:
                continue
            ci = label[i]
            ni = thing_classes[ci]
            x1, y1, x2, y2 = bbox[i]
            draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[ci], width=3)
            draw.text((x1 + 3, y1 + 1), '%s %.3f' % (ni, score[i]), fill=bbox_rgbs[ci], font=font_label)
        draw.text((2, 2), os.path.basename(f), fill='#FFFFFF', font=font_title)
        im = np.array(im)
        writer.writeFrame(im)

        det = {'file_name': os.path.basename(f), 'image_id': len(detections) + 1, 'height': im.shape[0], 'width': im.shape[1], 'annotations': []}
        for i in range(0, len(score)):
            det['annotations'].append({"bbox": bbox[i], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': label[i], 'score': score[i]})
        detections.append(det)
    writer.close()

    coco_evaluator(input_dir, detections)


if __name__ == '__main__':
    detect()

    # import cv2
    # f = os.path.join(output_dir, 'test.tif')
    # im_cv2 = cv2.imread(f)       # BGR
    # im_sk = skimage.io.imread(f)[:, :, :3] # RGB
    # print(im_cv2.shape, im_cv2.dtype, im_sk.shape, im_sk.dtype)
    # print(np.absolute(im_cv2 - im_sk).mean())
    # print(np.absolute(im_cv2 - im_sk[:, :, ::-1]).mean())

    pass
