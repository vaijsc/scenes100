#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import gzip
import math
import random
import tqdm
import glob
# import psutil
import argparse
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool as ProcessPool

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import skvideo.io
import networkx

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as torchdata
import torchvision

# import detectron2
# from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
# from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures import BoxMode
# from detectron2.structures import ImageList, Instances

# import logging
# import weakref
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils import IoU, DummyWriter, bbox_inside, intersect_ratios, count_parameters
# from models import get_cfg_base_model
# from decode_training import TrainingFrames
# from base_detector_train import get_coco_dicts


video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


class ConsecutiveImagePair(torchdata.Dataset):
    def __init__(self, image_dicts):
        super(ConsecutiveImagePair, self).__init__()
        self.image_dicts = image_dicts
        self.current_i = None
        self.current_im, self.next_im = None, None
        self.H, self.W = 720, 1280

    def __len__(self):
        return len(self.image_dicts) - 1

    def __getitem__(self, i):
        if self.current_im is not None:
            if i == self.current_i + 1:
                # print('use cache')
                self.current_i += 1
                self.current_im = self.next_im
                self.next_im = self.read_image(i + 1)
            elif i == self.current_i:
                pass
            else:
                raise NotImplementedError
        else:
            self.current_i = i
            self.current_im = self.read_image(i)
            self.next_im = self.read_image(i + 1)
        return self.current_im, self.next_im

    def read_image(self, i):
        r = {'fn': self.image_dicts[i]['file_name']}
        r['im'] = skimage.transform.resize(skimage.io.imread(r['fn']), (self.H, self.W)).astype(np.float32)
        r['im_norm'] = torch.from_numpy(r['im'].transpose(2, 0, 1) * 2 - 1).unsqueeze(0)
        r['im'] = (r['im'] * 255.0).astype(np.uint8)
        return r

    @staticmethod
    def collate(batch):
        return batch


def compute_flow(args):
    assert args.id in video_id_list
    lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'train_lmdb', args.id))
    with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
        meta = json.load(fp)
    ifilelist = meta['ifilelist']
    images = []
    for i in range(0, len(ifilelist)):
        images.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': []})
    images = [images[i] for i in np.arange(0, 1201)]
    print('unlabeled frames of video %s at %s: %d images' % (args.id, lmdb_path, len(images)))

    raft_L = torchvision.models.optical_flow.raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT)
    raft_L.cuda()
    raft_L.eval()
    raft_S = torchvision.models.optical_flow.raft_small(weights=torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT)
    raft_S.cuda()
    raft_S.eval()

    dataset = ConsecutiveImagePair(copy.deepcopy(images))
    loader = torchdata.DataLoader(dataset, batch_size=None, collate_fn=ConsecutiveImagePair.collate, shuffle=False, num_workers=1)

    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), '..', 'DejaVuSansCondensed.ttf'), size=(dataset.H // 20))
    writer = skvideo.io.FFmpegWriter('RAFT_flow_%s.mp4' % args.id, inputdict={'-r': '5'}, outputdict={'-vcodec': 'hevc_nvenc', '-r': '5', '-preset': 'medium'})
    for im1, im2 in tqdm.tqdm(loader, total=len(loader), ascii=True):
        with torch.no_grad():
            flow_seq = raft_L(im1['im_norm'].cuda(), im2['im_norm'].cuda())
            flow_im_L = torchvision.utils.flow_to_image(flow_seq[-1][0]).detach().cpu().numpy().transpose(1, 2, 0)
            flow_seq = raft_S(im1['im_norm'].cuda(), im2['im_norm'].cuda())
            flow_im_S = torchvision.utils.flow_to_image(flow_seq[-1][0]).detach().cpu().numpy().transpose(1, 2, 0)
        f = np.concatenate([np.concatenate([im1['im'], im2['im']], axis=1), np.concatenate([flow_im_L, flow_im_S], axis=1)], axis=0)

        f = Image.fromarray(f)
        draw = ImageDraw.Draw(f)
        draw.text((2, 2), os.path.basename(im1['fn']), fill='#000000', stroke_width=3, font=font)
        draw.text((2, 2), os.path.basename(im1['fn']), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((dataset.W + 2, 2), os.path.basename(im2['fn']), fill='#000000', stroke_width=3, font=font)
        draw.text((dataset.W + 2, 2), os.path.basename(im2['fn']), fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((2, dataset.H + 2), 'RAFT-Large', fill='#000000', stroke_width=3, font=font)
        draw.text((2, dataset.H + 2), 'RAFT-Large', fill='#FFFFFF', stroke_width=1, font=font)
        draw.text((dataset.W + 2, dataset.H + 2), 'RAFT-Small', fill='#000000', stroke_width=3, font=font)
        draw.text((dataset.W + 2, dataset.H + 2), 'RAFT-Small', fill='#FFFFFF', stroke_width=1, font=font)
        writer.writeFrame(np.array(f))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--id', type=str, default='', choices=video_id_list+['', 'compound'], help='video ID')
    parser.add_argument('--model', type=str, default='r101-fpn-3x', help='detection model')
    args = parser.parse_args()
    print(args)

    compute_flow(args)

'''
conda deactivate && conda activate detectron2

python optical_flow.py --id 001
'''
