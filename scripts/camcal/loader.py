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
import matplotlib.patches as patches

import torch
import torch.utils.data as torchdata
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IoU, DummyWriter
from decode_training import TrainingFrames


class TrackedFramesDataset(torchdata.Dataset):
    def __init__(self, video_id, track_gzs, area_thres, flip_p):
        super(TrackedFramesDataset, self).__init__()
        assert area_thres > 1.0, 'bbox area change threshold should be larger than 1'
        assert flip_p >= 0.0 and flip_p <= 1.0
        assert type(track_gzs) == type([]) and len(track_gzs) > 0
        self.area_thres, self.flip_p = area_thres, flip_p
        self.dst = TrainingFrames(video_id)
        self.tracks = []
        for f in track_gzs:
            with gzip.open(f, 'rt') as fp:
                self.tracks = self.tracks + json.loads(fp.read())['tracks']

        # remove all tracks at image edges
        def at_edge(t):
            margin = 2
            H, W = t['H'], t['W']
            x1, y1, x2, y2 = t['bbox']
            return x1 <= margin or y1 <= margin or x2 >= W - margin or y2 >= H - margin
        for i in range(0, len(self.tracks)):
            for j in range(0, len(self.tracks[i])):
                if at_edge(self.tracks[i][j]):
                    self.tracks[i] = self.tracks[i][:j]
                    break
        self.tracks = list(filter(lambda x: len(x) > 2, self.tracks))

        self.start_idx = {}
        for t in self.tracks:
            if not t[0]['frame_i'] in self.start_idx:
                self.start_idx[t[0]['frame_i']] = {'tracks': [], 'shortest': 1e10, 'longest': 0}
            self.start_idx[t[0]['frame_i']]['tracks'].append(t)
            self.start_idx[t[0]['frame_i']]['shortest'] = min(self.start_idx[t[0]['frame_i']]['shortest'], len(t))
            self.start_idx[t[0]['frame_i']]['longest'] = max(self.start_idx[t[0]['frame_i']]['longest'], len(t))

        self.pairs = []
        for k in self.start_idx:
            for l in range(min(self.start_idx[k]['shortest'], self.start_idx[k]['longest'] // 2), self.start_idx[k]['longest']):
                tracks_l = filter(lambda x: len(x) >= l, self.start_idx[k]['tracks'])
                tracks_l = map(lambda x: x[:l], tracks_l)
                tracks_l = list(map(lambda x: [x[0], x[-1]], tracks_l))
                if len(tracks_l) < 1:
                    continue
                frame_i_1, frame_i_2 = tracks_l[0][0]['frame_i'], tracks_l[0][-1]['frame_i']
                for t_l in tracks_l:
                    assert frame_i_1 == t_l[0]['frame_i'] and frame_i_2 == t_l[-1]['frame_i']
                p = {'i1': frame_i_1, 'i2': frame_i_2, 'tracks': []}
                for t_l in tracks_l:
                    x1, y1, x2, y2 = t_l[0]['bbox']
                    a1 = (x2 - x1) * (y2 - y1)
                    x1, y1, x2, y2 = t_l[-1]['bbox']
                    a2 = (x2 - x1) * (y2 - y1)
                    assert a1 > 0 and a2 > 0
                    lb = 0
                    if a2 > a1 * self.area_thres:
                        lb = +1 # move closer
                    if a2 * self.area_thres < a1:
                        lb = -1 # move further away
                    p['tracks'].append({'bbox1': t_l[0]['bbox'], 'bbox2': t_l[-1]['bbox'], 'label': lb})
                self.pairs.append(p)

        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(0.25, 0.25, 0.25),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = copy.deepcopy(self.pairs[i])
        images = [self.dst[p['i1']][0], self.dst[p['i2']][0]]
        for j in (0, 1):
            if random.random() < self.flip_p:
                images[j] = images[j][:, ::-1, :]
                for k in range(0, len(p['tracks'])):
                    x1, y1, x2, y2 = p['tracks'][k]['bbox%d' % (j + 1)]
                    p['tracks'][k]['bbox%d' % (j + 1)] = [images[j].shape[1] - x2, y1, images[j].shape[1] - x1, y2]
            images[j] = torch.from_numpy((images[j] / 255.0).astype(np.float32).transpose(2, 0, 1))
            images[j] = self.tf(images[j])
        return images[0], images[1], p['tracks']

    @staticmethod
    def collate(batch):
        im1_list = list(map(lambda t: t[0], batch))
        im2_list = list(map(lambda t: t[1], batch))
        tracks_list = list(map(lambda t: t[2], batch))
        return torch.stack(im1_list), torch.stack(im2_list), tracks_list

    def _random_show(self):
        t = self.tracks[random.randrange(0, len(self.tracks))]
        fig, axes = plt.subplots(5, 5)
        for i in range(0, 5):
            for j in range(0, 5):
                _ax = axes[i][j]
                _ax.set_axis_off()
                if i * 5 + j >= len(t):
                    continue
                im, f, _, _ = self.dst[t[i * 5 + j]['frame_i']]
                _ax.imshow(im)
                x1, y1, x2, y2 = t[i * 5 + j]['bbox']
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='green', facecolor='none')
                _ax.add_patch(rect)
                _ax.set_title('%d (%d) %.1f' % (t[i * 5 + j]['frame_i'], f, (x2 - x1) * (y2 - y1)))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dst = TrackedFramesDataset('001', ['001_r101-fpn-3x_DiMP.json.gz'], 1.25, 0.5)
    # dst = TrackedFramesDataset('050', ['050_r101-fpn-3x_DiMP.json.gz'], 1.25, 0.5)
    print(len(dst))
    # for _ in range(0, 20):
    #     dst._random_show()

    loader = torchdata.DataLoader(dst, collate_fn=TrackedFramesDataset.collate, batch_size=4, shuffle=True, num_workers=0)
    ims1, ims2, tracks = iter(loader).next()
    print(ims1.size(), ims2.size())
    assert ims1.size(0) == ims2.size(0) == len(tracks)

    for i in range(0, ims1.size(0)):
        plt.figure()
        fig, axes = plt.subplots(1, 2)
        im = ims1[i].detach().cpu().numpy().transpose(1, 2, 0)
        im -= im.min()
        im /= im.max()
        axes[0].imshow(im)
        im = ims2[i].detach().cpu().numpy().transpose(1, 2, 0)
        im -= im.min()
        im /= im.max()
        axes[1].imshow(im)
        for j in range(0, len(tracks[i])):
            b1, b2 = tracks[i][j]['bbox1'], tracks[i][j]['bbox2']
            color = ['blue', 'green', 'red']
            x1, y1, x2, y2 = b1
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color[tracks[i][j]['label']], facecolor='none')
            axes[0].add_patch(rect)
            x1, y1, x2, y2 = b2
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color[tracks[i][j]['label']], facecolor='none')
            axes[1].add_patch(rect)
        plt.tight_layout()
        plt.show()
