#!python3

import os
import glob
import urllib
import time
import json
import random

import argparse
from multiprocessing import Pool as ProcessPool


def download_chunks(url, prefix, outputdir, clip_duration, clip_count):
    import m3u8
    import streamlink

    print('download %s -> %s' % (url, prefix))
    def get_stream():
        stream = streamlink.streams(url)['best']
        m3u8_obj = m3u8.load(stream.args['url'])
        return m3u8_obj.segments[0]

    while clip_count > 0:
        start_time, pre_time = None, None
        while True:
            segment = get_stream()
            cur_time = segment.program_date_time.strftime('%Y%m%d_%H%M%S')
            if start_time is None:
                start_time = segment.program_date_time.timestamp()
                fp = open(os.path.join(outputdir, prefix + '_' + str(cur_time) + '.ts'), 'ab+')
            if segment.program_date_time.timestamp() - start_time >= clip_duration:
                break
            if pre_time == cur_time:
                pass
            else:
                with urllib.request.urlopen(segment.uri) as response:
                    html = response.read()
                fp.write(html)
                pre_time = cur_time
        print('finished:', fp.name, flush=True)
        fp.close()
        clip_count -= 1

'''
ffmpeg -hide_banner -loglevel info -hwaccel cuda -i "SantaClausVillage_20221206_195902.ts" -an -vcodec libx265 -preset medium -crf 25 -x265-params pools=16 "SantaClausVillage_20221206_195902.hevc.mp4"
for V in  ; do ffmpeg -hide_banner -loglevel info -hwaccel cuda -i "${V}.ts" -an -vcodec libx265 -preset medium -crf 25 -x265-params pools=16 "${V}.hevc.mp4" ; done
for V in  ; do ffmpeg -hide_banner -loglevel info -hwaccel cuda -i "${V}.ts" -an -vcodec libx265 -preset medium -crf 25 -x265-params pools=16 "${V}.hevc.mp4" ; done
'''


def verify_videos():
    import skvideo.io

    basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'clips'))
    with open(os.path.join(basedir, 'clips.json'), 'r') as fp:
        clips = json.load(fp)

    for filelist in clips['days']:
        for vfilename in filelist:
            vfilename = os.path.join(basedir, vfilename + '.hevc.mp4')
            if not os.access(vfilename, os.R_OK):
                print('not found:', vfilename)
                continue
            meta = skvideo.io.ffprobe(vfilename)['video']
            FPS = meta['@r_frame_rate'].split('/')
            FPS = float(FPS[0]) / float(FPS[1])
            H, W = int(meta['@height']), int(meta['@width'])
            if not (FPS == clips['meta']['FPS'] and H == clips['meta']['H'] and W == clips['meta']['W']):
                print('video format wrong:', W, H, FPS, vfilename)
                continue
            print('passed:', vfilename)


def _decode_sample(params):
    import functools
    import tqdm
    import numpy as np
    import skvideo.io
    import skimage.io

    vfilename, framesdir, fps, sample_fps, chunk_secs, chunk_period_secs = params
    assert fps > sample_fps and chunk_secs < chunk_period_secs
    if not os.access(framesdir, os.W_OK):
        os.mkdir(framesdir)
    reader = skvideo.io.FFmpegReader(vfilename)
    F, H, W, _ = reader.getShape()
    reader.close()
    chunks_idx = list(np.arange(0, F / fps, chunk_period_secs).astype(int))
    for i in range(0, len(chunks_idx)):
        chunks_idx[i] = [chunks_idx[i] + (chunk_period_secs - chunk_secs) / 2, chunks_idx[i] + (chunk_period_secs + chunk_secs) / 2]
    chunks_idx = list(filter(lambda x: x[1] < F / fps - 1, chunks_idx))
    for i in range(0, len(chunks_idx)):
        chunks_idx[i] = (np.arange(chunks_idx[i][0], chunks_idx[i][1], 1 / sample_fps) * fps).astype(int).tolist()
    save_idx = set(functools.reduce(lambda x, y: x + y, chunks_idx))

    reader = skvideo.io.vreader(vfilename)
    for i in tqdm.tqdm(range(0, min(max(save_idx) + 100, F)), ascii=True, desc='%s => %s' % (os.path.basename(vfilename), os.path.basename(framesdir))):
        try:
            frame = next(reader)
        except StopIteration:
            break
        if i in save_idx:
            skimage.io.imsave(os.path.join(framesdir, '%08d.jpg' % i), frame, quality=80)
    reader.close()
    with open(framesdir + '.json', 'w') as fp:
        json.dump({'meta': {'W': W, 'H': H}, 'chunks': chunks_idx}, fp)

def sample_frames(day_idx, procs, sample_fps, chunk_secs, chunk_period_secs):
    basedir = os.path.normpath(os.path.dirname(__file__))
    with open(os.path.join(basedir, 'clips', 'clips.json'), 'r') as fp:
        clips = json.load(fp)
    assert day_idx >= 0 and day_idx < len(clips['days'])
    filelist = clips['days'][day_idx]
    print('decode %s with %d processes' % (len(filelist), procs))
    params = []
    for vfilename in filelist:
        params.append([os.path.join(basedir, 'clips', vfilename + '.hevc.mp4'), os.path.join(basedir, 'frames', vfilename), clips['meta']['FPS'], sample_fps, chunk_secs, chunk_period_secs])
    pool = ProcessPool(processes=procs)
    _ = pool.map_async(_decode_sample, params).get()
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, choices=['record', 'verify', 'sample', 'sample_eval'])
    parser.add_argument('--day', type=int)
    parser.add_argument('--procs', type=int)
    args = parser.parse_args()

    if args.opt == 'record':
        # 1 hr segments, resume if anything fails
        while True:
            try:
                download_chunks('https://www.youtube.com/watch?v=Cp4RRAEgpeU', 'SantaClausVillage', os.path.join(os.path.dirname(__file__), 'clips'), 3600, 240)
            except:
                pass

    if args.opt == 'verify':
        verify_videos()

    if args.opt == 'sample':
        # sample 3 min of 5 FPS frames every 10 min, 5400 frames every 1 hr video
        sample_frames(args.day, args.procs, 5, 3 * 60, 10 * 60)

    if args.opt == 'sample_eval':
        sample_frames(4, args.procs, 5, 3 * 60, 8 * 60)

'''
python long_video.py --opt record
python long_video.py --opt verify
python long_video.py --opt sample --day 0 --procs 1
python long_video.py --opt sample_eval --procs 1
'''