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
import psutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
import skimage.io

import sklearn.utils
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# merge feature pyramids from image/mask, keep the feature maps dimensionalities


class FeaturePyramidFusionConv(nn.Module):
    def __init__(self):
        super(FeaturePyramidFusionConv, self).__init__()
        self.keys, self.C = ('p2', 'p3', 'p4', 'p5', 'p6'), 256
        self.weights_logit = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)
        self.convs = nn.ModuleDict({
            k: nn.Sequential(
                nn.Conv2d(self.C * 2, self.C, (3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.C),
                nn.LeakyReLU(negative_slope=0.05, inplace=True),
                nn.Conv2d(self.C, self.C, (3, 3), stride=1, padding=1, bias=False),
            )
            for k in self.keys
        })

    def forward(self, fp1, fp2):
        # print(self.convs['p2'][0].weight.data[0, 0, 0, 0].item(), self.weights_logit.data)
        weights = nn.functional.softmax(self.weights_logit, dim=0)
        return {
            k: fp1[k] * weights[0] + fp2[k] * weights[1] + self.convs[k](torch.cat([fp1[k], fp2[k]], dim=1)) * weights[2]
            for k in fp1
        }


class FeaturePyramidFusionAttn(nn.Module):
    def __init__(self):
        super(FeaturePyramidFusionAttn, self).__init__()
        self.keys = ('p2', 'p3', 'p4', 'p5', 'p6')
        self.patch_sizes = {'p2': 16, 'p3': 12, 'p4': 8, 'p5': 4, 'p6': 2}
        self.weights_logit = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)
        # query: fp1, key: fp1|fp2, value: fp1|fp2
        # attention within each channel only
        self.attns = nn.ModuleDict({
            k: nn.MultiheadAttention(self.patch_sizes[k] ** 2, 1, kdim=(self.patch_sizes[k] ** 2 * 2), vdim=(self.patch_sizes[k] ** 2 * 2), dropout=0.25, batch_first=True)
            for k in self.keys
        })
        self.norms = nn.ModuleDict({
            k: nn.LayerNorm((self.patch_sizes[k] ** 2,))
            for k in self.keys
        })

    @staticmethod
    def roundup(h, w, s):
        if 0 == (h % s): h_r = h
        else: h_r = (h // s + 1) * s
        if 0 == (w % s): w_r = w
        else: w_r = (w // s + 1) * s
        return h_r, w_r, h_r // s, w_r // s

    @staticmethod
    def to_patches(fp, b, c, n_h, n_w, s): # B x C x H x W
        fp = fp.unfold(2, s, s).unfold(3, s, s) # B x C x nH x nW x S x S
        fp = fp.reshape(b, c, n_h, n_w, s * s) # B x C x nH x nW x (S x S)
        fp = fp.reshape(b, c, n_h * n_w, s * s) # B x C x (nH x nW) x (S x S)
        fp = fp.reshape(b * c, n_h * n_w, s * s) # (B x C) x (nH x nW) x (S x S)
        return fp

    @staticmethod
    def stitch_patches(fp, b, c, n_h, n_w, s):
        fp = fp.reshape(b, c, n_h * n_w, s * s) # B x C x (nH x nW) x (S x S)
        fp = fp.reshape(b, c, n_h, n_w, s * s) # B x C x nH x nW x (S x S)
        fp = fp.reshape(b, c, n_h, n_w, s, s) # B x C x nH x nW x S x S
        fp = fp.transpose(2, 4).reshape(b, c, n_w, n_h * s, s).transpose(3, 4) # B x C x nW x S x H
        fp = fp.reshape(b, c, n_w * s, n_h * s).transpose(2, 3) # B x C x H x W
        return fp

    def forward(self, fp1, fp2):
        # if self.training:
        #     print(self.attns['p2'].k_proj_weight.data[0, 0].item(), self.weights_logit.data)
        fp_fusion = {}
        for k in self.keys:
            b, c, h_, w_ = fp1[k].size()
            h, w, n_h, n_w = self.roundup(h_, w_, self.patch_sizes[k])
            fp1_resize = nn.functional.interpolate(fp1[k], size=(h, w), mode='bilinear', align_corners=True)
            fp2_resize = nn.functional.interpolate(fp2[k], size=(h, w), mode='bilinear', align_corners=True) # B x C x H x W
            fp1_resize = self.to_patches(fp1_resize, b, c, n_h, n_w, self.patch_sizes[k])
            fp2_resize = self.to_patches(fp2_resize, b, c, n_h, n_w, self.patch_sizes[k])
            fp12_resize = torch.cat([fp1_resize, fp2_resize], dim=2)
            fp12_fusion, _ = self.attns[k](fp1_resize, fp12_resize, fp12_resize)
            fp12_fusion = self.norms[k](fp12_fusion)
            fp12_fusion = self.stitch_patches(fp12_fusion, b, c, n_h, n_w, self.patch_sizes[k])
            fp_fusion[k] = nn.functional.interpolate(fp12_fusion, size=(h_, w_), mode='bilinear', align_corners=True)
        weights = nn.functional.softmax(self.weights_logit, dim=0)
        return {
            k: fp1[k] * weights[0] + fp2[k] * weights[1] + fp_fusion[k] * weights[2]
            for k in fp1
        }


def test_patchify():
    im = torch.tensor(skimage.io.imread('bird.png').transpose(2, 0, 1)[:3])
    print(im.dtype, im.size())
    im = im.unfold(1, 32, 32).unfold(2, 32, 32)
    print(im.dtype, im.size())
    im = im.reshape(3, -1, 32, 32).transpose(0, 1)
    print(im.dtype, im.size())
    N = im.size(0)
    plt.figure()
    for i in range(0, N):
        plt.subplot(1, N, i + 1)
        plt.imshow(im[i].numpy().transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    # test_patchify(); exit(0)
    fp = {
        'p2': torch.randn(7, 256, 200, 304),
        'p3': torch.rand(7, 256, 100, 152),
        'p4': torch.rand(7, 256, 50, 76),
        'p5': torch.rand(7, 256, 25, 38),
        'p6': torch.rand(7, 256, 13, 19),
    }
    # m = FeaturePyramidFusionConv()
    m = FeaturePyramidFusionAttn()
    torch.save(m.state_dict(), 'fusion.pth')
    fp_f = m(fp, fp)
    for k in fp:
        print(k, fp[k].size(), fp_f[k].size())
