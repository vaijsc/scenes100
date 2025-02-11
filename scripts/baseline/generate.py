#!python3

import os
import glob
import json
import random
import time
import shutil
import tqdm

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import imantics
from detectron2.structures import BoxMode

thing_classes = ['person', 'vehicle']
mask_rgb = [0, 1, 0]


def generate():
    inputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', '20200909_1130_1316.mp4_finetune')
    outputdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', '001')

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'masks.json'), 'r') as fp:
        mask = json.load(fp)
    mask = {m['video']: m['polygons'] for m in mask}
    mask = mask['001']
    m_ann = imantics.Annotation.from_polygons(mask, image=imantics.Image.from_path(glob.glob(os.path.join(inputdir, '*.jpg'))[0]))
    m_arr = np.expand_dims(m_ann.array.astype(np.uint8), 2)

    def _corner_not_in_mask(bbox):
        assert bbox['bbox_mode'] == BoxMode.XYXY_ABS
        x1, y1, x2, y2 = map(int, bbox['bbox'])
        H, W = m_arr.shape[:2]
        x1 = min(max(x1, 0), W - 1)
        x2 = min(max(x2, 0), W - 1)
        y1 = min(max(y1, 0), H - 1)
        y2 = min(max(y2, 0), H - 1)
        return not (m_arr[y1, x1, 0] > 0 or m_arr[y1, x2, 0] > 0 or m_arr[y2, x1, 0] > 0 or m_arr[y2, x2, 0] > 0)

    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    print('%d images %d boxes' % (len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    for im in annotations:
        for ann in im['annotations']:
            if ann['bbox_mode'] == BoxMode.XYWH_ABS:
                x1, y1, w, h = ann['bbox']
                x2, y2 = x1 + w, y1 + h
                ann['bbox_mode'] = BoxMode.XYXY_ABS
                ann['bbox'] = [x1, y1, x2, y2]
            assert ann['bbox_mode'] == BoxMode.XYXY_ABS, 'box modes %s unrecognized: %s' % (list(BoxMode), ann['bbox_mode'])
        im['annotations'] = list(filter(_corner_not_in_mask, im['annotations']))
    print('%d images %d boxes' % (len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))

    for subdir in ['unmasked', 'masked']:
        os.mkdir(os.path.join(outputdir, subdir))
    for im in tqdm.tqdm(annotations, ascii=True):
        src = os.path.join(inputdir, im['file_name'])
        shutil.copy2(src, os.path.join(outputdir, 'unmasked', im['file_name'][4:]))
        im_arr = skimage.io.imread(src)
        im_blank = ((1 - m_arr) * im_arr).astype(np.uint8)
        skimage.io.imsave(os.path.join(outputdir, 'masked', im['file_name'][4:]), im_blank, quality=80)

    for im in annotations:
        im['file_name'] = im['file_name'][4:]
    with open(os.path.join(outputdir, 'annotations.json'), 'w') as fp:
        json.dump(annotations, fp)


if __name__ == '__main__':
    generate()
