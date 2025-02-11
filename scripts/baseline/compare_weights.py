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
import pickle
import math
import random
import tqdm
import glob
import psutil
import contextlib
import argparse

import numpy as np
import sklearn.utils

import torch


ckpt_base = {
    'vanilla': 'mscoco2017_remap_r101-fpn-3x.pth',
    'earlyfusion': 'mscoco2017_remap_wdiff_earlyfusion_r101-fpn-3x.pth',
    'midfusion': 'mscoco2017_remap_wdiff_midfusion_r101-fpn-3x.pth',
    'latefusion': 'mscoco2017_remap_wdiff_latefusion_r101-fpn-3x.pth'
}
# backbone.fpn_output5.weight torch.Size([256, 256, 3, 3]) torch.float32
# backbone.bottom_up.res4.22.conv3.weight torch.Size([1024, 256, 1, 1]) torch.float32
# proposal_generator.rpn_head.objectness_logits.weight torch.Size([3, 256, 1, 1]) torch.float32
# proposal_generator.rpn_head.anchor_deltas.weight torch.Size([12, 256, 1, 1]) torch.float32
# roi_heads.box_predictor.cls_score.weight torch.Size([3, 1024]) torch.float32
# roi_heads.box_predictor.bbox_pred.weight torch.Size([8, 1024]) torch.float32
layers_prefix = {
    'vanilla': {
        'backbone': 'backbone.bottom_up.res4.22.conv3.weight',
        'fpn': 'backbone.fpn_output5.weight',
        'rpn_cls': 'proposal_generator.rpn_head.objectness_logits.weight',
        'rpn_box': 'proposal_generator.rpn_head.anchor_deltas.weight',
        'roi_cls': 'roi_heads.box_predictor.cls_score.weight',
        'roi_box': 'roi_heads.box_predictor.bbox_pred.weight'
    },
    'earlyfusion': {
        'backbone': 'backbone.bottom_up.res4.22.conv3.weight',
        'fpn': 'backbone.fpn_output5.weight',
        'rpn_cls': 'proposal_generator.rpn_head.objectness_logits.weight',
        'rpn_box': 'proposal_generator.rpn_head.anchor_deltas.weight',
        'roi_cls': 'roi_heads.box_predictor.cls_score.weight',
        'roi_box': 'roi_heads.box_predictor.bbox_pred.weight'
    },
    'midfusion': {
        'backbone': 'backbone.bottom_up.res4.22.conv3.weight',
        'fpn': 'backbone.fpn_output5.weight',
        'rpn_cls': 'proposal_generator_merge.rpn_head.objectness_logits.weight',
        'rpn_box': 'proposal_generator_merge.rpn_head.anchor_deltas.weight',
        'roi_cls': 'roi_heads_merge.box_predictor.cls_score.weight',
        'roi_box': 'roi_heads_merge.box_predictor.bbox_pred.weight'
    },
    'latefusion': {
        'backbone': 'backbone.bottom_up.res4.22.conv3.weight',
        'fpn': 'backbone.fpn_output5.weight',
        'rpn_cls': 'proposal_generator.rpn_head.objectness_logits.weight',
        'rpn_box': 'proposal_generator.rpn_head.anchor_deltas.weight',
        'roi_cls': 'roi_heads_merge.box_predictor.cls_score.weight',
        'roi_box': 'roi_heads_merge.box_predictor.bbox_pred.weight'
    }
}
layers = ['backbone', 'fpn', 'rpn_cls', 'rpn_box', 'roi_cls', 'roi_box']
record_prefix = os.path.join(os.path.dirname(__file__), 'compare_weights')


def count_parameters(w):
    dims = list(w.size())
    assert len(dims) > 0
    prod = 1
    for d in dims:
        prod *= d
    return prod


def print_parameters_ckpt(args):
    assert os.access(args.ckpt, os.R_OK), '%s not readable' % args.ckpt
    state_dict = torch.load(args.ckpt, map_location='cpu')
    for k in state_dict:
        print(k, state_dict[k].size(), state_dict[k].dtype)


def count_parameters_ckpt(args):
    assert os.access(args.ckpt, os.R_OK), '%s not readable' % args.ckpt
    state_dict = torch.load(args.ckpt, map_location='cpu')
    count = 0
    for k in state_dict:
        assert state_dict[k].dtype == torch.float32
        count += count_parameters(state_dict[k])
    print(args.ckpt, count / 1000000, 'x10^6')


def weights_diff(args):
    assert args.model == 'r101-fpn-3x'
    assert len(args.tag) > 0
    layers_names = layers_prefix[args.arch]
    assert set(layers) == set(layers_names.keys())
    print('scanning checkpoints in %s' % args.ckpts_dir)
    ckpts = sorted(glob.glob(os.path.join(args.ckpts_dir, 'adapt*.pth')))
    assert len(ckpts) == 100, 'wrong number of checkpoints: %d' % len(ckpts)

    layers_params_adapt = {l: [] for l in layers_names}
    for ckpt_i in tqdm.tqdm(ckpts, ascii=True):
        sd_i = torch.load(ckpt_i, map_location='cpu')
        for l in layers_names:
            assert layers_names[l] in sd_i, '%s not found in %s' % (layers_names[l], sd_i.keys())
            layers_params_adapt[l].append(sd_i[layers_names[l]].cpu().numpy())
    for l in layers_params_adapt:
        layers_params_adapt[l] = np.stack(layers_params_adapt[l], axis=0)
    np.savez_compressed('%s_%s.npz' % (record_prefix, args.tag), **layers_params_adapt)


def plot_correlations(args):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    ckpt_base_fullpath = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', ckpt_base['vanilla']))
    print('base model ckpt:', ckpt_base_fullpath)
    sd_base = torch.load(ckpt_base_fullpath, map_location='cpu')

    tags = ['pseudo-labeling+mid-fusion+mixup', 'ST', 'AT', 'STAC', 'H2FA', 'TIA']
    print(tags)

    random.seed(0)
    colors = ['#FF0000', '#995500', '#559900', '#00FF00', '#009955', '#005599', '#0000FF']

    for l in layers:
        w = sd_base[layers_prefix['vanilla'][l]]
        weights = []
        print(l, list(w.size()))
        for t in tags:
            fp = np.load(record_prefix + '_%s.npz' % t)
            weights.append(fp[l].reshape(100, -1))
            fp.close()
        weights.append(w.cpu().numpy().reshape(1, -1))
        counts = list(map(lambda x: x.shape[0], weights))
        weights = np.concatenate(weights, axis=0)
        weights_tsne = TSNE(n_components=2, init='pca', learning_rate='auto', n_jobs=12).fit_transform(weights)
        weights_tsne -= weights_tsne.min()
        weights_tsne /= weights_tsne.max()
        print(tags, counts, weights.shape, weights_tsne.shape)

        legends = []
        plt.figure(figsize=(8, 8))
        plt.title('TSNE of layer %s %s' % (layers_prefix['vanilla'][l], list(w.size())))
        for i in range(0, len(tags)):
            weights_tsne_i = weights_tsne[sum(counts[: i]) : sum(counts[: i + 1])]
            plt.scatter(weights_tsne_i[:, 0], weights_tsne_i[:, 1], marker='o', color=colors[i], alpha=0.5)
        plt.scatter(weights_tsne[-1:, 0], weights_tsne_i[-1:, 1], marker='+', color='k', alpha=1)
        plt.legend(tags + ['base'])
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Models Weights Script')
    parser.add_argument('--opt', type=str, default='', choices=['', 'print', 'count', 'diff', 'plot'], help='script option')
    parser.add_argument('--model', type=str, help='detection model')
    parser.add_argument('--arch', type=str, choices=['vanilla', 'earlyfusion', 'midfusion', 'latefusion'], help='faster RCNN architecture')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--ckpts_dir', type=str)
    parser.add_argument('--tag', type=str)
    args = parser.parse_args()
    print(args)

    if args.opt == 'print':
        print_parameters_ckpt(args)
    elif args.opt == 'count':
        count_parameters_ckpt(args)
    elif args.opt == 'diff':
        weights_diff(args)
    elif args.opt == 'plot':
        plot_correlations(args)

'''
python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\baseline_crossteach_r101 --tag "pseudo-labeling"
python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\mixup_r101_p0.3_r0.5_overlap0.65 --tag "pseudo-labeling+mixup"
python compare_weights.py --opt diff --model r101-fpn-3x --arch midfusion --ckpts_dir F:\intersections_results\object_diff_midfusion_r101 --tag "pseudo-labeling+mid-fusion"
python compare_weights.py --opt diff --model r101-fpn-3x --arch midfusion --ckpts_dir F:\intersections_results\object_diff_midfusion_mixup_r101 --tag "pseudo-labeling+mid-fusion+mixup"

python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\cvpr22_tia_r101 --tag "TIA"
python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\cvpr22_h2fa_rcnn_r101 --tag "H2FA"
python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\cvpr22_adaptive_teacher_r101 --tag "AT"
python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\baseline_cvpr19_r101 --tag "ST"
python compare_weights.py --opt diff --model r101-fpn-3x --arch vanilla --ckpts_dir F:\intersections_results\arxiv2020_stac_r101 --tag "STAC"
'''
