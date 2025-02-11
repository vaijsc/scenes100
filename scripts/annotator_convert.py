#!python3

import os
import glob
import json
import random
import time
import shutil
import copy
import tqdm

import imantics
import skimage.io
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageDraw, ImageFont
from detectron2.structures import BoxMode


thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']
mask_rgb = [0, 1, 0]


def merge():
    inputjsons = []
    for d in [os.path.join('E:', 'VinAI', 'dense'), os.path.join('E:', 'VinAI', 'sparse')]:
        inputjsons = inputjsons + glob.glob(os.path.join(d, '*'))
    inputjsons = list(map(lambda x: os.path.join(x, 'annotations', 'instances_default.json'), inputjsons))
    images_all = []
    for f in tqdm.tqdm(inputjsons, ascii=True):
        with open(f, 'r') as fp:
            data = json.load(fp)
        images, annotations, categories = data['images'], data['annotations'], data['categories']
        categories = [[cat['id'], cat['name']] for cat in categories]
        assert categories[0] == [1, 'person'] and categories[1] == [2, 'vehicle']
        images_detectron2 = {im['id']: {'file_name': im['file_name'], 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []} for im in images}
        for ann in annotations:
            x, y, w, h = ann['bbox']
            images_detectron2[ann['image_id']]['annotations'].append({'bbox': [x, y, x + w, y + h], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': ann['category_id'] - 1})
        images_all = images_all + list(images_detectron2.values())
    with open(os.path.join('E:', 'VinAI', 'annotations.json'), 'w') as fp:
        json.dump(images_all, fp)


def check_overlap(M, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return (M[y1, x1, 0] > 1e-3 or M[y1, x2, 0] > 1e-3 or M[y2, x1, 0] > 1e-3 or M[y2, x2, 0] > 1e-3)


def draw_bbox(im, M, annotations, desc):
    alpha = 0.45
    M = M * alpha
    fontsize, linewidth = int(min(im.shape[0], im.shape[1]) * 0.04), int(min(im.shape[0], im.shape[1]) / 300)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf'), size=fontsize)
    counts = {x : 0 for x in range(0, len(thing_classes))}
    im = ((1 - M) * im + M * (np.array(mask_rgb) * 255).astype(np.uint8).reshape(1, 1, 3)).astype(np.uint8)
    im = Image.fromarray(im, 'RGB')
    draw = ImageDraw.Draw(im)
    for ann in annotations:
        # bbox has format [x1, y1, x2, y2]
        x1, y1, x2, y2 = ann['bbox']
        cat = ann['category_id']
        draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[cat], width=linewidth)
        counts[cat] += 1
    desc = desc + ' ' + ' '.join(['%d %s(s)' % (counts[cat], thing_classes[cat]) for cat in range(0, len(thing_classes))])
    draw.text((6, 2), desc, fill='#FFFFFF', stroke_width=3, font=font)
    draw.text((6, 2), desc, fill='#000000', stroke_width=2, font=font)
    draw.text((6, 2), desc, fill='#FFFFFF', stroke_width=1, font=font)
    im = np.array(im)
    return im


def quality_laplacian(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    s = cv2.Laplacian(im, cv2.CV_64F).var()
    return s


def convert():
    with open(os.path.join(os.path.dirname(__file__), '..', 'masks.json'), 'r') as fp:
        masks = json.load(fp)
    masks = {m['video']: m['polygons'] for m in masks}

    # read images to be annotated
    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotations', 'images')
    inputdir_sparse = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotations', 'sparse')
    images_per_video = {}
    for f in glob.glob(os.path.join(inputdir, '*.jpg')):
        f = os.path.basename(f)
        vid = f.split('_')[0]
        if not vid in images_per_video:
            images_per_video[vid] = {'dense': [], 'sparse': []}
        images_per_video[vid]['dense'].append(f)
    for f in glob.glob(os.path.join(inputdir_sparse, '*.jpg')):
        f = os.path.basename(f)
        vid = f.split('_')[0]
        images_per_video[vid]['sparse'].append(f)

    # read annotations
    with open(os.path.join(os.path.dirname(__file__), '..', 'images', 'VinAI_annotations.json'), 'r') as fp:
        annotations_all = json.load(fp)
    annotations_all = {ann['file_name']: ann for ann in annotations_all}
    for im in images_per_video.values():
        for f in im['dense']:
            assert f in annotations_all
        for f in im['sparse']:
            assert f in annotations_all

    # images_per_video = {'130': images_per_video['130'], '060': images_per_video['060'], '070': images_per_video['070'], '159': images_per_video['159'], '175': images_per_video['175']}
    # stats_all = []
    for vid in tqdm.tqdm(images_per_video, ascii=True):
        stats = {'id': vid}
        images_arr = list(map(lambda f: skimage.io.imread(os.path.join(inputdir, f)), images_per_video[vid]['dense']))
        assert len(images_arr) > 0
        H, W, C = images_arr[0].shape
        for im in images_arr:
            assert (H, W, C) == im.shape
        stats['HxWxC'] = [H, W, C]
        laplacian_score = list(map(quality_laplacian, images_arr))
        laplacian_score = np.array(laplacian_score).mean()
        stats['quality_laplacian_score'] = laplacian_score

        # compute mask
        if len(masks[vid]) > 0:
            ann = imantics.Annotation.from_polygons(masks[vid], image=imantics.Image.from_path(os.path.join(inputdir, images_per_video[vid]['dense'][0])))
            M = np.expand_dims(ann.array.astype(np.float16), 2)
        else:
            M = np.zeros_like(images_arr[0][:, :, 0:1])

        outputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotated', vid)
        assert not os.access(outputdir, os.W_OK), '%s already exists' % outputdir
        os.mkdir(outputdir)
        os.mkdir(os.path.join(outputdir, 'masked'))
        os.mkdir(os.path.join(outputdir, 'unmasked'))
        writer = skvideo.io.FFmpegWriter(os.path.join(outputdir, 'annotations.mp4'), inputdict={'-r': '1'}, outputdict={'-vcodec': 'libx265', '-r': '1', '-pix_fmt': 'yuv420p', '-preset': 'medium', '-crf': '25'})
        annotations_per_video = []
        for i in range(0, len(images_per_video[vid]['dense'])):
            im, f = images_arr[i], images_per_video[vid]['dense'][i]
            im_detectron2 = copy.deepcopy(annotations_all[f])
            assert W == im_detectron2['width'] and H == im_detectron2['height']
            annotations_filter = []
            for ann in im_detectron2['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                x1, y1, x2, y2 = ann['bbox']
                x1 = min(max(x1, 0), W - 1)
                x2 = min(max(x2, 0), W - 1)
                y1 = min(max(y1, 0), H - 1)
                y2 = min(max(y2, 0), H - 1)
                if not check_overlap(M, [x1, y1, x2, y2]):
                    annotations_filter.append(ann)
            im_detectron2['annotations'] = annotations_filter
            if len(im_detectron2['annotations']) < 1:
                continue
            im_detectron2['file_name'] = f[4:]

            shutil.copy2(os.path.join(inputdir, f), os.path.join(outputdir, 'unmasked', f[4:]))
            im_mask = ((1 - M) * im).astype(np.uint8)
            skimage.io.imsave(os.path.join(outputdir, 'masked', f[4:]), im_mask, quality=80)
            im_ann = draw_bbox(im, M, im_detectron2['annotations'], 'V %s F %s' % (vid, f[4:-4]))
            writer.writeFrame(im_ann)
            annotations_per_video.append(im_detectron2)
        writer.close()
        with open(os.path.join(outputdir, 'annotations.json'), 'w') as fp:
            json.dump(annotations_per_video, fp)
        stats['frames'] = len(annotations_per_video)
        stats['counts'] = sum(list(map(lambda x: len(x['annotations']), annotations_per_video)))

        M = np.zeros_like(M)
        annotations_per_video = []
        os.mkdir(os.path.join(outputdir, 'weaksupervised'))
        for i in range(0, len(images_per_video[vid]['sparse'])):
            f = images_per_video[vid]['sparse'][i]
            im_detectron2 = copy.deepcopy(annotations_all[f])
            assert W == im_detectron2['width'] and H == im_detectron2['height']
            for ann in im_detectron2['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            if len(im_detectron2['annotations']) < 1:
                continue
            im_detectron2['file_name'] = f[4:]

            shutil.copy2(os.path.join(inputdir_sparse, f), os.path.join(outputdir, 'weaksupervised', f[4:]))
            im = skimage.io.imread(os.path.join(inputdir_sparse, f))
            im_ann = draw_bbox(im, M, im_detectron2['annotations'], 'V %s F %s' % (vid, f[4:-4]))
            skimage.io.imsave(os.path.join(outputdir, 'weaksupervised', f[4:-4] + '_annotated.jpg'), im_ann, quality=80)
            annotations_per_video.append(im_detectron2)
        with open(os.path.join(outputdir, 'weaksupervised.json'), 'w') as fp:
            json.dump(annotations_per_video, fp)


def save_npz():
    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotated')
    stats = []
    for f in tqdm.tqdm(files, ascii=True, desc='writing NPZs'):
        with open(os.path.join(inputdir, f['id'], 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        fn_list, im_arr = [], []
        for im in images:
            fn_list.append(im['file_name'])
            im_arr.append(skimage.io.imread(os.path.join(inputdir, f['id'], 'unmasked', im['file_name'])))
        fn_list = np.array(fn_list)
        im_arr = np.stack(im_arr, axis=0)
        np.savez_compressed(os.path.join(inputdir, f['id'], 'unmasked.npz'), filenames=fn_list, images=im_arr)
    stats.sort(key=lambda x: x['objects']['all'])


def stats():
    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    inputdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'annotated')
    stats = []
    for f in files:
        with open(os.path.join(inputdir, f['id'], 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        anns = []
        for im in images:
            anns = anns + im['annotations']
        stats.append({
            'id': f['id'],
            'frames': len(images),
            'objects': {thing_classes[cat]: sum([1 if ann['category_id'] == cat else 0 for ann in anns]) for cat in range(0, len(thing_classes))},
        })
        stats[-1]['objects']['all'] = stats[-1]['objects']['person'] + stats[-1]['objects']['vehicle']
    stats.sort(key=lambda x: x['objects']['all'])

    plt.figure(figsize=(14, 8))
    legends = []
    xs = np.arange(1, len(stats) + 1, 1)
    counts = list(map(lambda x: x['objects']['all'], stats))
    legends.append('all objects $%.1f\\pm %.1f$' % (np.mean(counts), np.std(counts)))
    plt.plot(xs, counts, 'ko-')
    for i in range(0, len(counts)):
        plt.text(xs[i], counts[i] + 35, stats[i]['id'], rotation=-85, horizontalalignment='right', verticalalignment='center', size=8, color='black')
    counts = list(map(lambda x: x['objects']['person'], stats))
    legends.append('person $%.1f\\pm %.1f$' % (np.mean(counts), np.std(counts)))
    plt.plot(xs, counts, 'rx-')
    counts = list(map(lambda x: x['objects']['vehicle'], stats))
    legends.append('vehicle $%.1f\\pm %.1f$' % (np.mean(counts), np.std(counts)))
    plt.plot(xs, counts, 'bx-')
    plt.legend(legends)
    plt.ylim(0, 1750)
    plt.xlim(0, 101)
    plt.ylabel('#')
    plt.xticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # merge()
    # convert()
    # stats()
    # save_npz()
    pass
