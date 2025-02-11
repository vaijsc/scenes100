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


def check_data_integrity():
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    checklist = ['detect_r50-fpn-3x.json.gz', 'detect_r50-fpn-3x_DiMP.json.gz', 'detect_r101-fpn-3x.json.gz', 'detect_r101-fpn-3x_DiMP.json.gz']
    missing, filesizes = [], {}
    for v in tqdm.tqdm(files, ascii=True, desc='checking detection & sot files'):
        dst = TrainingFrames(v['id'])
        for f in checklist:
            p = os.path.join(dst.lmdb_path, f)
            if not os.access(p, os.R_OK):
                missing.append(p)
                continue
            _s = os.path.getsize(p)
            if _s in filesizes:
                filesizes[_s].append(p)
            else:
                filesizes[_s] = [p]
    print('missing:', missing)
    print('files with same size:')
    for _s in filesizes:
        if len(filesizes[_s]) > 1:
            print(_s, filesizes[_s])

    missing = []
    for v in files:
        dst = TrainingFrames(v['id'])
        if os.access(os.path.join(os.path.join(dst.lmdb_path, 'jpegs', '00000000.jpg')), os.R_OK) and len(glob.glob(os.path.join(os.path.join(dst.lmdb_path, 'jpegs', '*.jpg')))) == 27000:
            continue
        missing.append(v['id'])
        print(dst)
    print('missing JPEGs:', ' '.join(missing))


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


def run_cvpr19(args):
    # python finetune_cvpr19.py --id 003 --opt adapt --model r101-fpn-3x --anno_models r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --train_on_coco 1 --image_batch_size 4 --hold 7.5
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_cvpr19.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_%s_cocotrain_CVPR19.pth' % (x, model, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_cvpr19_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_cvpr19.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', model, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_cvpr19_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach(args):
    # python finetune.py --id 003 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --train_on_coco 1 --image_batch_size 4 --not_use_mod_rcnn 1 --hold 18.1
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_fn_discard(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_falsenegative.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_FN_discard.pth' % (x, model)), os.R_OK), vids))
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_FN_discard_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_falsenegative.py'),  '--id', v, '--opt', 'crossteach', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1820', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_FN_discard_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_gmm(args):
    # python finetune.py --id 003 --opt adapt --model r101-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --train_on_coco 1 --image_batch_size 4 --hold 21
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_modRCNN_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_gmm_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--gmm_density_loss_weight', '0.0', '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_gmm_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_mixup(args):
    # python finetune_mixup.py --id 003 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --train_on_coco 1 --image_batch_size 4 --hold 18.1
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_mixup.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_mixup_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_mixup_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_earlyfusion(args):
    # python finetune_wdiff_earlyfusion.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_earlyfusion_r50-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1800 --image_batch_size 4 --num_workers 4
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_earlyfusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_earlyfusion.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_earlyfusion.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_earlyfusion_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_earlyfusion.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_earlyfusion_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_earlyfusion_mixup(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_earlyfusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_earlyfusion_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_earlyfusion_mixup.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_earlyfusion_mixup_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_earlyfusion_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_earlyfusion_mixup_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion(args):
    # python finetune_wdiff_midfusion.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_midfusion_r50-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1800 --image_batch_size 4 --num_workers 4
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_midfusion.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_midfusion.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_midfusion_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_midfusion.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_midfusion_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion_fn_discard(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_falsenegative.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_FN_discard_midfusion.pth' % (x, model)), os.R_OK), vids))
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_midfusion_FN_discard_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_falsenegative.py'),  '--id', v, '--opt', 'midfusion', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1820', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_midfusion_FN_discard_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion_conv(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusionconv_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_midfusion.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_midfusionconv.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_midfusionconv_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_midfusion.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--fusion_type', 'conv', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_midfusionconv_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion_attn(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusionattn_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_midfusion.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_midfusionattn.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_midfusionattn_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_midfusion.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--fusion_type', 'attn', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_midfusionattn_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion_conv_mixup(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusionconv_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_midfusion_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_midfusionconv_mixup.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_midfusionconv_mixup_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_midfusion_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--fusion_type', 'conv', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_midfusionconv_mixup_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion_attn_mixup(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusionattn_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_midfusion_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_midfusionattn_mixup.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_midfusionconv_mixup_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_midfusion_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--fusion_type', 'attn', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_midfusionattn_mixup_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_midfusion_mixup(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_midfusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_midfusion_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_midfusion_mixup.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_midfusion_mixup_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_midfusion_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--fusion_type', 'average', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_midfusion_mixup_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_latefusion(args):
    # python finetune_wdiff_latefusion.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_latefusion_r50-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1800 --image_batch_size 4 --num_workers 4
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_latefusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_latefusion.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_latefusion.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_latefusion_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_latefusion.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_latefusion_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_wdiff_latefusion_mixup(args):
    # python finetune_wdiff_latefusion_mixup.py --opt adapt --id 001 --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --ckpt ../../models/mscoco2017_remap_wdiff_latefusion_r50-fpn-3x.pth --train_on_coco 1 --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1800 --image_batch_size 4 --num_workers 4
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_wdiff_latefusion_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_wdiff_latefusion_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_wdiff_latefusion_mixup.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_wdiff_latefusion_mixup_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_wdiff_latefusion_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--coco_inpaint_type', 'mask', '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_wdiff_latefusion_mixup_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_homography(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_homography_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_homography_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_homography.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_homography_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_homography_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--mixup_p', '0', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_homography_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_radial(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_radial_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_radial_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_radial.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_radial_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_radial_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--mixup_p', '0', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_radial_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_lzu(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_lzu_mixup.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain_lzu.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_lzu_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_lzu_mixup.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--mixup_p', '0', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_lzu_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_crossteach_fpcorr(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'mscoco2017_remap_%s.pth' % args.model))
    assert os.access(os.path.join(basedir, 'finetune_fpn_correlation.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_anno_refine_r101-fpn-3x_r50-fpn-3x_cocotrain27000_fpncorr.pth' % (x, model)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_crossteach_fpcorr_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_fpn_correlation.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--anno_models', 'r50-fpn-3x', 'r101-fpn-3x', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--train_on_coco', '1', '--image_batch_size', '4', '--mixup_p', '0', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_crossteach_fpcorr_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_retinanet(args):
    basedir = os.path.dirname(__file__)
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_retinanet_r101_student_teacher.pth'))
    assert os.access(os.path.join(basedir, 'finetune_retinanet_distill.py'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_retinanet_%s.pth' % x), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_retinanet_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_retinanet_distill.py'),  '--id', v, '--opt', 'distill', '--ckpt', modelpth, '--num_workers', '6', '--iters', '10000', '--eval_interval', '1001', '--image_batch_size', '6', '--lr', '0.0002', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_retinanet_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_faster_rcnn(args):
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_r101-fpn-3x_student_teacher.pth'))
    assert os.access(os.path.join(basedir, 'finetune_faster_rcnn_distill.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK) and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_r101-fpn-3x_%s.pth' % x), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_faster_rcnn_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_faster_rcnn_distill.py'), '--model', 'r101-fpn-3x', '--id', v, '--opt', 'distill', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '26000', '--eval_interval', '2001', '--image_batch_size', '4', '--lr', '0.0005', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_faster_rcnn_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_faster_rcnn_roi(args):
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    # modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), 'mscoco2017_remap_r101-fpn-3x_student_teacher.pth'))
    assert os.access(os.path.join(basedir, 'finetune_faster_rcnn_roi_distill.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)# and os.access(modelpth, os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_r101-fpn-3x_%s_with_roi.pth' % x), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_faster_rcnn_roi_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            modelpth = os.path.normpath(os.path.join(os.path.dirname(__file__), 'distill_faster_rcnn', 'distill_r101-fpn-3x_%s.pth' % v))
            cmd_i = [python_path, os.path.join(basedir, 'finetune_faster_rcnn_roi_distill.py'), '--model', 'r101-fpn-3x', '--id', v, '--opt', 'distill', '--ckpt', modelpth, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '2001', '--image_batch_size', '4', '--lr', '0.0004', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_faster_rcnn_roi_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_faster_rcnn_x2_teach(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_scaled_teacher.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_%s_%s_anno_r101-fpn-3x_x2_cocotrain.pth' % (model, x)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_scaled_teacher_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_scaled_teacher.py'),  '--id', v, '--opt', 'distill', '--model', model, '--anno_scales', '2.0', '--cocodir', cocodir, '--num_workers', '4', '--iters', '16000', '--eval_interval', '1501', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_scaled_teacher_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_faster_rcnn_scaled_teach_ema(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_scaled_teacher_ema.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_%s_%s_ema_anno_%s_unlabeled_cocotrain.teacher.pth' % (model, x, x)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_scaled_teacher_ema_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_scaled_teacher_ema.py'),  '--id', v, '--opt', 'distill', '--model', model, '--cocodir', cocodir, '--num_workers', '4', '--iters', '16000', '--eval_interval', '1501', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_scaled_teacher_ema_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_faster_rcnn_x2_teach_output(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_scaled_teacher_partial_output.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_%s_%s_anno_r101-fpn-3x_x2_cocotrain.pth' % (model, x)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_scaled_teacher_output_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_scaled_teacher_partial_output.py'),  '--id', v, '--opt', 'distill', '--model', model, '--anno_scales', '2.0', '--cocodir', cocodir, '--num_workers', '4', '--iters', '16000', '--eval_interval', '1501', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_scaled_teacher_output_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_distill_faster_rcnn_x2_teach_homography(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_scaled_teacher_homography_input.py'), os.R_OK) and os.access(os.path.join(cocodir, 'annotations', 'instances_train2017.json'), os.R_OK)
    assert len(args.gpus) > 0
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    vids = [f['id'] for f in files]
    vids = list(filter(lambda x: x in vids, args.ids))
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'distill_%s_%s_anno_003_x2.0_refine_cocotrain_homographyinput.pth' % (model, x)), os.R_OK), vids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_distill_scaled_teacher_homography_%s_%s_%s_GPU%s.log' % (model, v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_scaled_teacher_homography_input.py'),  '--id', v, '--opt', 'distill', '--model', model, '--anno_scales', '2.0', '--cocodir', cocodir, '--num_workers', '4', '--iters', '16000', '--eval_interval', '1501', '--train_on_coco', '1', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_distill_scaled_teacher_homography_%s_999_%s_GPU%s.log' % (model, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_stac(args):
    # python finetune_stac.py --id 003 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --image_batch_size 4 --hold 18.1
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_stac.py'), os.R_OK)
    assert len(args.gpus) > 0
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_STAC_cocotrain.pth' % (x, model)), os.R_OK), args.ids))
    random.shuffle(vids)
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_adapt_STAC_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_stac.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_adapt_STAC_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_stac_ddp_2gpus(args):
    # python finetune_stac.py --id 003 --opt adapt --model r50-fpn-3x --anno_models r50-fpn-3x r101-fpn-3x --cocodir ../../../MSCOCO2017 --num_workers 4 --iters 20000 --eval_interval 1800 --image_batch_size 4 --hold 18.1 --ddp_num_gpus 2 --ddp_port 50405
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_stac.py'), os.R_OK)
    assert len(args.gpus) == 2, 'only supports 2-GPUs DDP'
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_STAC_cocotrain.pth' % (x, model)), os.R_OK), args.ids))
    vids = sorted(vids)
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
    commands = []
    curr_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    print('CUDA_VISIBLE_DEVICES=%s' % curr_env['CUDA_VISIBLE_DEVICES'], flush=True)

    for v in vids:
        commands.append(
            (
                [python_path, os.path.join(basedir, 'finetune_stac.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--cocodir', cocodir, '--num_workers', '4', '--iters', '20000', '--eval_interval', '1800', '--image_batch_size', '4', '--hold', args.hold, '--ddp_num_gpus', '2', '--ddp_port', args.ddp_port],
                curr_env,
                os.path.join(basedir, 'log_adapt_STAC_%s_%s_GPU%s.log' % (v, socket.gethostname(), curr_env['CUDA_VISIBLE_DEVICES']))
            )
        )
    commands.append(
        (
            [python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold],
            curr_env,
            os.path.join(basedir, 'log_adapt_STAC_999_%s_GPU%s.log' % (socket.gethostname(), curr_env['CUDA_VISIBLE_DEVICES']))
        )
    )
    cmd_executor(commands)


def run_tia(args):
    # python finetune_tia.py --opt adapt --id 001 --model r101-fpn-3x --cocodir ../../../MSCOCO2017 --iters 20000 --eval_interval 1800 --image_batch_size 4 --num_workers 4 --hold 18.1
    model = args.model
    basedir = os.path.dirname(__file__)
    cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MSCOCO2017'))
    assert os.access(os.path.join(basedir, 'finetune_tia.py'), os.R_OK)
    assert len(args.gpus) > 0
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_TIA.pth' % (x, model)), os.R_OK), args.ids))
    random.shuffle(vids)
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_adapt_TIA_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_tia.py'),  '--id', v, '--opt', 'adapt', '--model', model, '--cocodir', cocodir, '--num_workers', '4', '--iters', '250', '--eval_interval', '130', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_adapt_TIA_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_lods(args):
    model = args.model
    basedir = os.path.dirname(__file__)
    assert os.access(os.path.join(basedir, 'finetune_lods.py'), os.R_OK)
    assert len(args.gpus) > 0
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt%s_%s_LODS.pth' % (x, model)), os.R_OK), args.ids))
    random.shuffle(vids)
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
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
            log_i = 'log_adapt_LODS_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'finetune_lods.py'),  '--id', v, '--model', model, '--ckpt', '../../models/mscoco2017_remap_r101-fpn-3x_LODS.pth', '--num_workers', '4', '--iters', '250', '--eval_interval', '130', '--image_batch_size', '4', '--hold', args.hold]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_adapt_LODS_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def check():
    import scipy
    ckpts = sorted(glob.glob('E:\\intersections_results\\object_diff_midfusion_conv_r101\\*.pth'))
    for f in ckpts:
        sd = torch.load(f, map_location='cpu')
        alphas = sd['fusion.weights_logit'].numpy()
        ws = scipy.special.softmax(alphas, axis=0)
        print('%s, %s, %s' % (os.path.basename(f), alphas, ws))


if __name__ == '__main__':
    # check()
    # raise
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--ids', nargs='+', default=[])
    parser.add_argument('--gpus', nargs='+', default=[])
    parser.add_argument('--hold', default='0.005', type=str)
    parser.add_argument('--ddp_port', type=str, default='50137')
    args = parser.parse_args()
    print(args)

    if args.opt == 'cvpr19':
        run_cvpr19(args)
    elif args.opt == 'crossteach':
        run_crossteach(args)
    elif args.opt == 'crossteachfndiscard':
        run_crossteach_fn_discard(args)
    elif args.opt == 'gmm':
        run_crossteach_gmm(args)
    elif args.opt == 'mixup':
        run_crossteach_mixup(args)
    elif args.opt == 'earlyfusion':
        run_crossteach_wdiff_earlyfusion(args)
    elif args.opt == 'earlyfusionmixup':
        run_crossteach_wdiff_earlyfusion_mixup(args)
    elif args.opt == 'midfusion':
        run_crossteach_wdiff_midfusion(args)
    elif args.opt == 'midfusionfndiscard':
        run_crossteach_wdiff_midfusion_fn_discard(args)
    elif args.opt == 'midfusionmixup':
        run_crossteach_wdiff_midfusion_mixup(args)
    elif args.opt == 'midfusionconv':
        run_crossteach_wdiff_midfusion_conv(args)
    elif args.opt == 'midfusionattn':
        run_crossteach_wdiff_midfusion_attn(args)
    elif args.opt == 'midfusionconvmixup':
        run_crossteach_wdiff_midfusion_conv_mixup(args)
    elif args.opt == 'midfusionattnmixup':
        run_crossteach_wdiff_midfusion_attn_mixup(args)
    elif args.opt == 'latefusion':
        run_crossteach_wdiff_latefusion(args)
    elif args.opt == 'latefusionmixup':
        run_crossteach_wdiff_latefusion_mixup(args)
    elif args.opt == 'homography':
        run_crossteach_homography(args)
    elif args.opt == 'radial':
        run_crossteach_radial(args)
    elif args.opt == 'lzu':
        run_crossteach_lzu(args)
    elif args.opt == 'fpcorr':
        run_crossteach_fpcorr(args)
    elif args.opt == 'distillretinanet':
        run_distill_retinanet(args)
    elif args.opt == 'distillfasterrcnn':
        run_distill_faster_rcnn(args)
    elif args.opt == 'distillfasterrcnnroi':
        run_distill_faster_rcnn_roi(args)
    elif args.opt == 'distillx2teach':
        run_distill_faster_rcnn_x2_teach(args)
    elif args.opt == 'distillscaledteachema':
        run_distill_faster_rcnn_scaled_teach_ema(args)
    elif args.opt == 'distillx2teachoutput':
        run_distill_faster_rcnn_x2_teach_output(args)
    elif args.opt == 'distillx2teachhomography':
        run_distill_faster_rcnn_x2_teach_homography(args)

    elif args.opt == 'stac':
        run_stac(args)
    elif args.opt == 'stac_ddp2':
        run_stac_ddp_2gpus(args)
    elif args.opt == 'tia':
        run_tia(args)
    elif args.opt == 'tia_ddp2':
        run_tia_ddp_2gpus(args)
    elif args.opt == 'lods':
        run_lods(args)

    elif args.opt == 'check':
        check_data_integrity()
    elif args.opt == 'test':
        test_gpu(args)
    else: pass
    exit(0)
