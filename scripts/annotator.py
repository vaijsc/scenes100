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
from PIL import Image, ImageDraw, ImageFont


thing_classes = ['person', 'vehicle']
mask_rgb = [0, 1, 0]

def generate_sample(estimates):
    import imantics

    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    with open(os.path.join(os.path.dirname(__file__), '..', 'masks.json'), 'r') as fp:
        masks = json.load(fp)
    masks = {m['video']: m['polygons'] for m in masks}
    s1, s2, s3 = set(map(lambda x: x[0], estimates)), set(masks.keys()), set(map(lambda x: x['id'], files))
    assert s1 == s2 and s1 == s3 # check consistency

    # plt.figure(figsize=(8, 8))
    # nums = np.array(list(map(lambda x: x[1], estimates)))
    # plt.hist(nums, bins=40)
    # plt.title('$%.2f (\\pm %.2f)$' % (nums.mean(), nums.std()))
    # plt.show()

    estimates_sort = sorted(estimates, key=lambda x: x[1])
    sample_videos = []
    for i in np.arange(3, 101, 4):
        sample_videos.append(estimates_sort[i])

    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'valid_4')
    outputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotator_sample_25')
    for v, n in sample_videos:
        image = glob.glob(os.path.join(inputdir, v + '*.jpg'))[2]
        im_arr = skimage.io.imread(image)
        polygons = masks[v]
        if len(polygons) > 0:
            ann = imantics.Annotation.from_polygons(polygons, image=imantics.Image.from_path(image))
            m_arr = np.expand_dims(ann.array.astype(np.float16), 2)
            m_arr = m_arr * 0.3
            im_arr = ((1 - m_arr) * im_arr + m_arr * (np.array(mask_rgb) * 255).astype(np.uint8).reshape(1, 1, 3)).astype(np.uint8)
        shutil.copy2(image, os.path.join(outputdir, os.path.basename(image)))
        shutil.copy2(image, os.path.join(outputdir, os.path.basename(image)))
        skimage.io.imsave(os.path.join(outputdir, os.path.basename(image)[:-4] + '_mask.jpg'), im_arr, quality=80)


def generate_annotation_images(estimates, budget, per_bbox, relaxation=1.3):
    import imantics

    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    with open(os.path.join(os.path.dirname(__file__), '..', 'masks.json'), 'r') as fp:
        masks = json.load(fp)
    masks = {m['video']: m['polygons'] for m in masks}
    s1, s2, s3 = set(map(lambda x: x[0], estimates)), set(masks.keys()), set(map(lambda x: x['id'], files))
    assert s1 == s2 and s1 == s3 # check consistency

    for v, _ in estimates:
        with open(os.path.join(os.path.dirname(__file__), '..', 'images', 'train_200_valid_200', v, 'frames.json'), 'r') as fp:
            data = json.load(fp)
        assert len(data['train']) >= 200 and len(data['valid']) >= 200, 'too few frames for %s' % v

    estimates = sorted(estimates, key=lambda x: x[1])
    estimates = list(map(list, estimates))
    budget /= len(estimates)
    for i in range(0, len(estimates)):
        estimates[i][1] *= relaxation
    for i in range(0, len(estimates)):
        estimates[i].append(budget / per_bbox / estimates[i][1])

    outputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotations')
    for subdir in ['images', 'images_mask_blank', 'images_mask_overlay', 'sparse']:
        os.mkdir(os.path.join(outputdir, subdir))
    for v, count, n_frames in tqdm.tqdm(estimates, ascii=True, desc='sampling for annotation'):
        # print(v, count, n_frames)
        framesdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'train_200_valid_200', v)
        with open(os.path.join(framesdir, 'frames.json'), 'r') as fp:
            data = json.load(fp)
            frames_train, frames_valid = data['train'], data['valid'][:200]
        i_valid = np.arange(0, len(frames_valid), len(frames_valid) / min(n_frames, len(frames_valid))).astype(int)[1:]
        frames_valid = [frames_valid[i] for i in i_valid]
        i_train = np.arange(0, len(frames_train), len(frames_train) / (5 if count > 10 else (8 if count > 5 else 17))).astype(int)[1 : -1]
        frames_train = [frames_train[i] for i in i_train]
        assert len(frames_valid) > 0 and len(frames_train) > 0

        for f in frames_train:
            shutil.copy2(os.path.join(framesdir, f), os.path.join(outputdir, 'sparse', v + '_' + f))
        if len(masks[v]) > 0:
            ann = imantics.Annotation.from_polygons(masks[v], image=imantics.Image.from_path(os.path.join(framesdir, frames_valid[0])))
            m_arr = np.expand_dims(ann.array.astype(np.float16), 2)
        else:
            m_arr = np.zeros_like(skimage.io.imread(os.path.join(framesdir, frames_valid[0])))[:, :, 0:1].astype(np.float16)
        m_arr_alpha = m_arr * 0.3
        for f in frames_valid:
            shutil.copy2(os.path.join(framesdir, f), os.path.join(outputdir, 'images', v + '_' + f))
            im_arr = skimage.io.imread(os.path.join(framesdir, f))
            im_blank = ((1 - m_arr) * im_arr).astype(np.uint8)
            skimage.io.imsave(os.path.join(outputdir, 'images_mask_blank', v + '_' + f), im_blank, quality=80)
            im_overlay = ((1 - m_arr_alpha) * im_arr + m_arr_alpha * (np.array(mask_rgb) * 255).astype(np.uint8).reshape(1, 1, 3)).astype(np.uint8)
            skimage.io.imsave(os.path.join(outputdir, 'images_mask_overlay', v + '_' + f), im_overlay, quality=80)


def verify():
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.config import get_cfg

    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotations')
    for model, url in [('r50-c4-1x', 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'), ('r50-c4-3x', 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml'), ('r50-fpn-1x', 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'), ('r50-fpn-3x', 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'), ('r101-fpn-3x', 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'), ('x101-fpn-3x', 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')]:
        print('detect with %s' % model)
        results_json = os.path.join(inputdir, 'verify_%s.json' % model)
        if os.access(results_json, os.R_OK):
            print('already done')
            continue

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(url))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(__file__), '..', 'models', 'mscoco2017_remap_%s.pth' % model)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        detector = DefaultPredictor(cfg)

        frame_objs = []
        for f in tqdm.tqdm(glob.glob(os.path.join(inputdir, 'images_mask_blank', '*.jpg')), ascii=True):
            im = skimage.io.imread(f)
            instances = detector(im[:, :, ::-1])['instances'].to('cpu')
            frame_objs.append({
                'image': f,
                'resolution': im.shape,
                # bbox has format [x1, y1, x2, y2]
                'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                'score': instances.scores.numpy().tolist(),
                'label': instances.pred_classes.numpy().tolist()
            })
        with open(results_json, 'w') as fp:
            json.dump({'model': model, 'checkpoint': cfg.MODEL.WEIGHTS, 'classes': thing_classes, 'dets': frame_objs}, fp)

    thresholds = np.arange(0.3, 1, 0.05)
    model_list = ['r50-c4-1x', 'r50-c4-3x', 'r50-fpn-1x', 'r50-fpn-3x', 'r101-fpn-3x', 'x101-fpn-3x']
    plt.figure(figsize=(4 * len(model_list), 4))
    for i in range(0, len(model_list)):
        model = model_list[i]
        with open(os.path.join(inputdir, 'verify_%s.json' % model), 'r') as fp:
            data = json.load(fp)
        dets = data['dets']
        dets_dict = {}
        for r in dets:
            v = os.path.basename(r['image'])[:3]
            if not v in dets_dict:
                dets_dict[v] = []
            dets_dict[v] = dets_dict[v] + r['score']
        for k in dets_dict:
            dets_dict[k] = np.array(dets_dict[k])

        count_min, count_max, count_mean = [], [], []
        for t in thresholds:
            counts = []
            for k in dets_dict:
                counts.append((dets_dict[k] >= t).sum())
            counts = np.array(counts)
            count_min.append(counts.min())
            count_max.append(counts.max())
            count_mean.append(counts.mean())
        plt.subplot(1, len(model_list), i + 1)
        plt.plot(thresholds, count_min, 'g-')
        plt.plot(thresholds, count_max, 'b-')
        plt.plot(thresholds, count_mean, 'r-')
        plt.xlim(thresholds.min(), thresholds.max())
        plt.ylim(0, max(count_max))
        plt.legend(['min count', 'max count', 'average count'])
        plt.grid(True)
        plt.title(model)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    estimates = list(zip(['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179'], [27.25, 10.75, 15.75, 74.5, 36, 63.25, 40, 11.75, 26, 3.25, 6.5, 35, 34, 10.5, 184.25, 26, 0.25, 14.5, 25.75, 9.75, 25.25, 25.75, 24.5, 13.25, 22.5, 24.5, 22, 51, 49, 33.5, 23.25, 6.5, 5.75, 30.5, 17.75, 29.5, 27, 20, 19.25, 23.75, 49.25, 20.25, 32.5, 8, 14.5, 7.25, 31.5, 8, 23, 21.5, 21, 53.25, 7.75, 8, 5.75, 9.5, 8, 4.25, 30, 4.5, 43.25, 8.25, 4.5, 8, 19.75, 18.5, 12.75, 51, 21.25, 23.75, 16.25, 14.75, 15, 12.25, 19.25, 54.25, 28, 15.25, 18, 18.25, 78.75, 22.25, 32, 13.25, 11, 26.5, 7.25, 39, 6, 29.25, 18.5, 6.25, 1.25, 14.5, 16.25, 11, 9.5, 49, 13.75, 18.75]))

    # generate_sample(estimates)
    generate_annotation_images(estimates, 2000, 0.02)
    # verify()
