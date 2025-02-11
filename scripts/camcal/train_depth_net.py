#!python3

import os
import sys
import tqdm
import json
import glob
import copy
import argparse

import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.utils.data as torchdata
from torch.optim import lr_scheduler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from loader import TrackedFramesDataset
# from DepthNetResnet50 import DepthEstimator
from depth_net.inference import DepthNetEstimator, plot_depth_map


def lr_warmup_step(iters, i):
    assert iters > 100
    i = max(0, i)
    warmup_i = iters / 10
    if i < warmup_i:
        return (i / warmup_i) ** 0.5
    if i < iters / 2:
        return 1.0
    return 0.3

'''
loss = log(1 + exp(-z1 + z2)), if lb = +1
       log(1 + exp( z1 - z2)), if lb = -1
                  (z1 - z2)^2, if lb =  0

lb = +1: object becomes larger, moves closer from 1 to 2, we want z1 > z2
lb = -1: object becomes smaller, moves further way from 1 to 2, we want z2 > z1
lb =  0: object stays similar size, we want z1 ~ z2
'''
def threeway_loss(depth1, depth2, tracks):
    bs = depth1.size(0)
    assert bs == depth2.size(0) == len(tracks)
    losses = [[] for _ in range(0, bs)]
    for i in range(0, bs):
        d1, d2 = depth1[i, 0], depth2[i, 0]
        for t in tracks[i]:
            b1, b2, lb = list(map(int, t['bbox1'])), list(map(int, t['bbox2'])), t['label']
            z1 = d1[b1[1] : b1[3], b1[0] : b1[2]].mean()
            z2 = d2[b2[1] : b2[3], b2[0] : b2[2]].mean()
            diff = z2 - z1
            if lb == 0:
                L = torch.pow(diff, 2.0)
            elif lb == 1:
                L = torch.log(1.0 + torch.exp(diff))
            elif lb == -1:
                L = torch.log(1.0 + torch.exp(-1.0 * diff))
            else:
                raise Exception('unrecognized label %s' % lb)
            losses[i].append(L)
    for i in range(0, bs):
        losses[i] = torch.stack(losses[i]).mean()
    return torch.stack(losses).mean()


def train(args):
    # net = DepthEstimator().cuda()
    net = DepthNetEstimator(max_width=448).cuda()
    net_distill = DepthNetEstimator(max_width=384).cuda()
    for p in net_distill.parameters():
        p.requires_grad = False
    net_distill.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: lr_warmup_step(args.iters, x), last_epoch=-1)

    dst = TrackedFramesDataset(args.id, args.track_gzs, args.area_thres, 0.5)
    print('%s: %d image pairs for training' % (args.id, len(dst)))
    loader = torchdata.DataLoader(dst, collate_fn=TrackedFramesDataset.collate, batch_size=1, shuffle=True, num_workers=args.num_workers)

    if os.access(args.ckpt, os.R_OK):
        print('loading checkpoint', args.ckpt)
        _ckpt = torch.load(args.ckpt)
        net.load_state_dict(_ckpt['state_dict'])
        last_iter = _ckpt['last_iter']
        loss_history = _ckpt['loss_history']
        lr_history = _ckpt['lr_history']
    else:
        last_iter, loss_history, lr_history = -1, [], []

    iter_train = iter(loader)
    def _get_batch(iter_train):
        while True:
            try:
                ims1, ims2, tracks = next(iter_train)
            except StopIteration:
                iter_train = iter(loader)
                continue
            return ims1, ims2, tracks

    net.train()
    net.backbone_freeze()
    optimizer.zero_grad()
    optimizer.step()
    for i in tqdm.tqdm(range(0, args.iters), ascii=True, desc='training'):
        if i == args.iters // 2:
            net.backbone_unfreeze()
        scheduler.step()
        if i <= last_iter:
            continue

        if i % args.save_interval == 2:
            net.eval()
            im_batch, depth_batch = [], []
            for _ in range(0, 3):
                ims1, ims2, tracks = _get_batch(iter_train)
                with torch.no_grad():
                    depth1 = net(ims1.cuda())
                im_batch.append(ims1.detach().cpu().numpy().transpose(0, 2, 3, 1)[0])
                depth_batch.append(depth1.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0])

            fig, axes = plt.subplots(3, 3, figsize=(16, 9))
            for k in range(0, 3):
                im = im_batch[k] - im_batch[k].min()
                im /= im.max()
                axes[k][0].imshow(im)
                axes[k][0].set_title('normalized image')
                dep_norm = depth_batch[k] - depth_batch[k].min()
                dep_norm /= dep_norm.max()
                plot_depth_map(axes[k][1], dep_norm, 'normalized depth')
                plot_depth_map(axes[k][2], 1 / (1 + np.exp(-1 * depth_batch[k])), 'sigmoid depth')
            # plt.tight_layout()
            plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
            plt.savefig('ckpt_%s_iter%03d.pdf' % (args.id, i))

            lr_history_np, loss_history_np = map(np.array, [lr_history, loss_history])
            loss_scale = loss_history_np[:, 1].max()
            plt.figure(figsize=(8, 8))
            plt.title('video %s @iteration-%d' % (args.id, i))
            plt.plot(lr_history_np[:, 0], lr_history_np[:, 1] / args.lr, 'k-')
            plt.plot(loss_history_np[:, 0], loss_history_np[:, 1] / loss_scale, 'g-', linewidth=0.25)
            plt.legend(['LR ($\\times$ %.1e)' % args.lr, 'loss ($\\times$ %.3f)' % loss_scale])
            plt.ylim(-0.02, 1.02)
            plt.xlim(i * -0.02, i * 1.02)
            plt.xlabel('interations ($batch=%d$)' % args.batch_size)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('ckpt_%s_loss.pdf' % args.id)

            torch.save({
                'last_iter': i, 'loss_history': loss_history, 'lr_history': lr_history,
                'state_dict': net.state_dict(), 'args': vars(args)}, os.path.join(os.path.dirname(__file__), 'ckpt_%s.pth' % args.id))
            net.train()

        Ls = []
        optimizer.zero_grad()
        for _ in range(0, args.batch_size):
            ims1, ims2, tracks = _get_batch(iter_train)
            depth1 = net(ims1.cuda())
            depth2 = net(ims2.cuda())
            with torch.no_grad():
                depth1_distill = net_distill(ims1.cuda()).detach()
                depth2_distill = net_distill(ims2.cuda()).detach()
            diff = (torch.pow(depth1_distill - depth1, 2) + torch.pow(depth2_distill - depth2, 2)) / 2.0
            L = (threeway_loss(depth1, depth2, tracks) + 0.1 * torch.pow(diff.mean(), 0.5)) / args.batch_size
            L.backward()
            Ls.append(L.item())
        optimizer.step()
        loss_history.append([i, np.array(Ls).mean()])
        lr_history.append([i, optimizer.param_groups[0]['lr']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--opt', type=str)
    parser.add_argument('--id', type=str)
    parser.add_argument('--track_gzs', nargs='+', default=[])
    parser.add_argument('--area_thres', type=float, default=1.25)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    print(args)

    if args.opt == 'train':
        train(args)
    else: pass

'''
python train_depth_net.py --opt train --id 001 --track_gzs 001_r101-fpn-3x_DiMP.json.gz --iters 255 --save_interval 50 --num_workers 3
001 003 016 017 034 050 169
005 006 007 008 009 011 012
'''
