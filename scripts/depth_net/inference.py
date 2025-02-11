#!python3
# coding: utf-8

"""
Author: Ke Xian
Email: kexian@hust.edu.cn
Create_Date: 2019/05/21
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

import os, argparse, sys
import json
import glob
import tqdm
import math

import numpy as np
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore")
# from PIL import Image

import skimage.io
from collections import OrderedDict

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from decode_training import TrainingFrames
from DepthNet import DepthNet


class DepthNetEstimator(nn.Module):
    def __init__(self, max_width=448):
        super(DepthNetEstimator, self).__init__()
        assert (max_width % 32) == 0
        self.max_width = max_width

        self.depthnet = DepthNet(depth=50)
        f = os.path.join(os.path.dirname(__file__), 'model.pth')
        assert os.access(f, os.R_OK), 'checkpoint %s not found' % f
        _ckpt = torch.load(f)['state_dict']
        _ckpt_remap = OrderedDict()
        for k in _ckpt:
            if k[:7] == 'module.':
                _ckpt_remap[k[7:]] = _ckpt[k]
            else:
                _ckpt_remap[k] = _ckpt[k]
        self.depthnet.load_state_dict(_ckpt_remap)

    def backbone_freeze(self):
        pass
    def backbone_unfreeze(self):
        pass

    def forward(self, X):
        _, _, H, W = X.size()
        if (H % 32) == 0 and (W % 32) == 0 and (W <= self.max_width or H <= self.max_width):
            return self.depthnet(X) # no need for resizing
        if W <= self.max_width or H <= self.max_width:
            if H <= W:
                H2 = 32 * math.ceil(H / 32.0)
                W2 = (W / H) * H2
                W2 = 32 * int(W2 / 32.0)
            else:
                W2 = 32 * math.ceil(W / 32.0)
                H2 = (H / W) * W2
                H2 = 32 * int(H2 / 32.0)
        else:
            if H <= W:
                H2 = self.max_width
                W2 = (W / H) * H2
                W2 = 32 * int(W2 / 32.0)
            else:
                W2 = self.max_width
                H2 = (H / W) * W2
                H2 = 32 * int(H2 / 32.0)
        d = -1.0 * self.depthnet(nn.functional.interpolate(X, size=(H2, W2), mode='bilinear', align_corners=True))
        return nn.functional.interpolate(d, size=(H, W), mode='bilinear', align_corners=True)


def plot_depth_map(ax, depth, desc):
    assert len(depth.shape) == 2
    xs, ys = np.meshgrid(np.linspace(0, depth.shape[1], depth.shape[1]), np.linspace(0, depth.shape[0], depth.shape[0]))
    ax.imshow(depth, cmap='gray')
    cntr = ax.contour(xs, ys, depth, levels=np.arange(0, 1, 0.1), linewidths=0.5, colors='cyan')
    ax.clabel(cntr, inline=1, fontsize=10)
    ax.set_title('%s $\\mu=%.3f$ $\\sigma=%.3f$' % (desc, depth.mean(), depth.std()))


def view():
    from matplotlib.backends.backend_pdf import PdfPages

    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    tf = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = DepthNetEstimator().cuda()
    net.eval()
    with PdfPages(os.path.join(os.path.dirname(__file__), 'depth.pdf')) as pdf:
        for v in tqdm.tqdm(files, ascii=True):
            dst = TrainingFrames(v['id'])
            batch = [dst[i] for i in [100, len(dst) // 3, len(dst) // 3 * 2, len(dst) - 100]]
            images = np.stack([b[0] for b in batch])
            fns = list(map(lambda x: v['id'] + '/' + x, [b[2] for b in batch]))
            images_tensor = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            images_tensor = tf(torch.from_numpy(images_tensor))
            with torch.no_grad():
                depths = net(images_tensor.cuda())
                depths = depths.detach().cpu().data.numpy()[:, 0]

            fig, axes = plt.subplots(4, 2, figsize=(12, 16))
            for i in range(0, 4):
                axes[i][0].imshow(images[i])
                axes[i][0].set_title(fns[i])
                d_norm = depths[i] - depths[i].min()
                d_norm /= d_norm.max()
                plot_depth_map(axes[i][1], d_norm, 'normalized depth')
            plt.suptitle('video %s' % v['id'])
            # plt.tight_layout()
            # plt.show()
            # plt.savefig('depth_%s.pdf' % v['id'])
            plt.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.04)
            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    view()
