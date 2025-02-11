#!python3

import os
import sys
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

import socket
import subprocess
from multiprocessing import Pool as ProcessPool

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils import IoU, DummyWriter
# from models import get_cfg_base_model
from decode_training import TrainingFrames


def test_gpu(args):
    print('[test]', vars(args))
    D = 1024
    vram_bytes_i = D * D * 4
    vram_bytes = float(args.hold) * (1024 ** 3)
    N = max(int(vram_bytes / vram_bytes_i * 0.77), 3)
    print(D, 'x', D, N)
    m_list = [torch.randn(D, D).type(torch.float32).cuda() for _ in range(0, N)]
    with torch.no_grad():
        while True:
            for i in range(0, N):
                m_list[i] = torch.matmul(m_list[i], m_list[i])
                m_list[i] -= m_list[i].mean()
                m_list[i] /= m_list[i].std()
            time.sleep(0.25)


def cmd_executor(cmd_list):
    t0 = time.time()
    for i in range(0, len(cmd_list)):
        c, e, o = cmd_list[i]
        # time.sleep(2)
        with open(o, 'w') as fp:
            p = subprocess.Popen(c, env=e, stdout=fp, stderr=fp)
            p.wait()
        print('[%d/%d finished]' % (i + 1, len(cmd_list)), '[%.1f hours]' % ((time.time() - t0) / 3600.0), '[%s]' % ' '.join(c), '>>>', '[%s]' % o, flush=True)


def run_crossteach(args):
    model = 'r101-fpn-3x'
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()

    commands = [[] for _ in range(0, len(args.gpus))]
    for (refine_det_score_thres, refine_iou_thres) in [(0.7, 0.85), (0.9, 0.85), (0.5, 0.75), (0.5, 0.95)]:
        outputdir = os.path.join(basedir, 'refine_ablation', 'crossteach_det%.2f_iou%.2f' % (refine_det_score_thres, refine_iou_thres))
        assert os.access(outputdir, os.W_OK)
        for i in range(0, len(args.gpus)):
            if i < len(args.gpus) - 1:
                vids_batch = vids[len(vids) // len(args.gpus) * i : len(vids) // len(args.gpus) * (i + 1)]
            else:
                vids_batch = vids[len(vids) // len(args.gpus) * i :]
            vids_batch = sorted(vids_batch)
            print(args.gpus[i], vids_batch)
            env_i = curr_env.copy()
            env_i['CUDA_VISIBLE_DEVICES'] = str(args.gpus[i])
            for v in vids_batch:
                log_i = 'log_ablation_refine_%s_det%.2f_iou%.2f_%s_GPU%s.log' % (v, refine_det_score_thres, refine_iou_thres, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
                log_i = os.path.join(basedir, log_i)
                cmd_i = [python_path, os.path.join(basedir, 'finetune.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--num_workers', '4', '--iters', '16000', '--eval_interval', '1601', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold, '--refine_det_score_thres', str(refine_det_score_thres), '--refine_iou_thres', str(refine_iou_thres), '--outputdir', outputdir]
                commands[i].append([cmd_i, env_i, log_i])
    for i in range(0, len(args.gpus)):
        commands[i].append([[python_path, os.path.join(basedir, 'run_ablation.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_ablation_refine_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])

    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_mixup(args):
    model = 'r101-fpn-3x'
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
    outputdir = os.path.join(basedir, 'mixup_ablation', 'mixup_r101_p0.3_r0.3_overlap0.65')
    assert os.access(outputdir, os.W_OK)

    commands = []
    for i in range(0, len(args.gpus)):
        commands_i = []
        if i < len(args.gpus) - 1:
            vids_batch = vids[len(vids) // len(args.gpus) * i : len(vids) // len(args.gpus) * (i + 1)]
        else:
            vids_batch = vids[len(vids) // len(args.gpus) * i :]
        vids_batch = sorted(vids_batch)
        print(args.gpus[i], vids_batch)
        env_i = curr_env.copy()
        env_i['CUDA_VISIBLE_DEVICES'] = str(args.gpus[i])
        for v in vids_batch:
            log_i = 'log_ablation_mixup_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--num_workers', '4', '--iters', '16000', '--eval_interval', '1601', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold, '--mixup_r', '0.3', '--outputdir', outputdir]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_ablation.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_ablation_mixup_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)

    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str)
    parser.add_argument('--gpus', nargs='+', default=[])
    parser.add_argument('--hold', default='0.005', type=str)
    args = parser.parse_args()
    print(args)

    if args.opt == 'refine':
        run_crossteach(args)
    elif args.opt == 'mixup':
        run_crossteach_mixup(args)
    elif args.opt == 'test':
        test_gpu(args)
