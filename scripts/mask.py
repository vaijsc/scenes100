#!python3

import os
import glob
import json
import tqdm
import skimage.io

import numpy as np
import imantics


def blend():
    with open(os.path.join('..', 'files.json'), 'r') as fp:
        files = json.load(fp)

    for f in files:
        framesdir = os.path.join('..', 'images', 'train_200_valid_200', '%03d' % f['id'])
        with open(os.path.join(framesdir, 'frames.json'), 'r') as fp:
            frames = np.array(json.load(fp)['valid'][:200])
        frames = frames.reshape(5, -1)[:, 0].tolist()
        frames = map(lambda x: os.path.join(framesdir, x), frames)
        frames = list(map(skimage.io.imread, frames))
        frames = np.stack(frames, axis=0)
        frames_mean = frames.mean(axis=0).astype(np.uint8)
        skimage.io.imsave(os.path.join('..', 'images', 'mask_draw', '%03d.jpg' % f['id']), frames_mean, quality=80)


def convert():
    inputdir = os.path.join('..', 'images', 'mask_draw')
    images = []
    for f in ['masks0.json', 'masks1.json']:
        with open(os.path.join(inputdir, f), 'r') as fp:
            cvat_json = json.load(fp)
        images_dict = cvat_json['images']
        images_dict = {im['id']: {'image': im['file_name'], 'video': im['file_name'][:3], 'polygons': []} for im in images_dict}
        for ann in cvat_json['annotations']:
            images_dict[ann['image_id']]['polygons'] = images_dict[ann['image_id']]['polygons'] + ann['segmentation']
        images = images + list(images_dict.values())
    with open(os.path.join('..', 'masks.json'), 'w') as fp:
        json.dump(images, fp, indent=2)


if __name__ == '__main__':
    # blend()
    convert()
