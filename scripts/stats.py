#!python3

import os
import glob
import json
import random
import hashlib
import time
import shutil
import tqdm

import skvideo.io
import numpy as np
import matplotlib.pyplot as plt


def probe():
    def probe_ffmpeg(vfilename):
        meta = skvideo.io.ffprobe(vfilename)['video']
        fps = meta['@r_frame_rate'].split('/')
        fps = float(fps[0]) / float(fps[1])
        H, W = int(meta['@height']), int(meta['@width'])
        reader = skvideo.io.FFmpegReader(vfilename)
        T, H2, W2, C = reader.getShape()
        reader.close()
        assert H2 == H and W2 == W
        return fps, H, W, C, T

    def sha_hashes(vfilename):
        results = {}
        for desc, hasher in [('sha1', hashlib.sha1()), ('sha512', hashlib.sha512())]:
            with open(vfilename, 'rb') as fp:
                while True:
                    content = fp.read(10 * 1024 * 1024)
                    if not content:
                        break
                    hasher.update(content)
            results[desc] = hasher.hexdigest()
        return results

    def parse_time(basename):
        timestamp = basename.split('.')[1]
        timestamp = basename.split('_')[1:]
        YYYY, MO, DD = int(timestamp[0][0:4]), int(timestamp[0][4:6]), int(timestamp[0][6:8])
        HH, MI, SS = int(timestamp[1][0:2]), int(timestamp[1][2:4]), int(timestamp[1][4:6])
        assert MO > 0 and MO < 13 and DD > 0 and DD < 32, (basename, timestamp, YYYY, MO, DD)
        assert HH >= 0 and HH <= 23 and MI >= 0 and MI <= 59 and SS >= 0 and SS <= 59, (basename, timestamp, HH, MI, S)
        return '%04d-%02d-%02d %02d:%02d:%02d UTC' % (YYYY, MO, DD, HH, MI, SS)

    with open(os.path.join(os.path.dirname(__file__), '..', 'intersections.csv'), 'r') as fp:
        lines = list(map(lambda s: s.strip(), fp.readlines()))
    lines = list(map(lambda l: l.split(','), lines))
    for i in range(0, len(lines)):
        assert lines[i][2][: 3] == '%03d' % (i + 1)

    vfilelist = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', 'videos', '*.mp4')))
    results = []
    for v in tqdm.tqdm(vfilelist, ascii=True, desc='probing files'):
        basename = os.path.basename(v)
        i = int(basename[: 3])
        fps, H, W, C, T = probe_ffmpeg(v)
        hashes = sha_hashes(v)
        results.append({
            'id': '%03d' % i, 'filename': basename,
            'file': {'bytes': os.path.getsize(v), 'sha1': hashes['sha1'], 'sha512': hashes['sha512']},
            'video': {'H': H, 'W': W, 'channel': C, 'fps': fps, 'frames': T},
            'meta': {'location': lines[i - 1][1], 'url': lines[i - 1][3], 'timestamp': parse_time(basename)},
        })

    with open(os.path.join('..', 'files.json'), 'w') as fp:
        json.dump(results, fp, indent=2)


def stats():
    fps_count, resolution_count, location_count = {}, {}, {}
    frames_all, timestamps = [], []
    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        results = json.load(fp)
    for v in results:
        frames_all.append(v['video']['frames'])
        key = v['video']['fps']
        if not key in fps_count:
            fps_count[key] = 0
        fps_count[key] += 1
        key = '%dx%d' % (v['video']['H'], v['video']['W'])
        if not key in resolution_count:
            resolution_count[key] = 0
        resolution_count[key] += 1
        key = v['meta']['location']
        timestamps.append(v['meta']['timestamp'])
        if not key in location_count:
            location_count[key] = 0
        location_count[key] += 1
    frames_all = np.array(frames_all)
    print(frames_all.mean(), frames_all.std())
    print(fps_count)
    print(resolution_count)
    print(location_count)
    print(sorted(timestamps))

# 217621.97 7793.78097261528
# {30.0: 98, 25.0: 2}
# {'1080x1920': 77, '720x1280': 20, '1080x1080': 2, '1080x1906': 1}
# {'USA': 30, 'Japan': 28, 'Canada': 6, 'Bulgaria': 1, 'UK': 3, 'Russia': 10, 'Italy': 6, 'Finland': 2, 'Ukraine': 6, 'Iceland': 1, 'Poland': 1, 'Romania': 1, 'Norway': 2, 'Netherlands': 1, 'USVI': 1, 'Switzerland': 1}


def detect(files):
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog
    from detectron2.config import get_cfg
    import skimage.io

    det_model = 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(det_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(det_model)
    obj_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    detector = DefaultPredictor(cfg)

    for v in files:
        video_id = v['id']
        framesdir = os.path.join('DetectInScenes', 'sampled_frames', '%03d' % video_id)
        with open(os.path.join(framesdir, 'frames.json'), 'r') as fp:
            ifilelist = json.load(fp)['valid']
        frame_objs = []
        for ifilename in tqdm.tqdm(ifilelist, ascii=True, desc=framesdir):
            im = skimage.io.imread(os.path.join(framesdir, ifilename))
            outputs = detector(im[:, :, ::-1])
            instances = outputs['instances'].to('cpu')
            frame_objs.append({
                # bbox has format [x1, y1, x2, y2]
                'bbox': instances.pred_boxes.tensor.numpy().tolist(),
                'score': instances.scores.numpy().tolist(),
                'label': instances.pred_classes.numpy().tolist()
            })
        assert len(ifilelist) == len(frame_objs)
        with open(framesdir + '_fasterrcnn_fpn1x.json', 'w') as fp:
            json.dump({'video': video_id, 'model': det_model, 'checkpoint': cfg.MODEL.WEIGHTS, 'classes': obj_classes, 'frames': ifilelist, 'dets': frame_objs}, fp)


def stats_obj():
    vids, opf_50, opf_75, opf_90 = [], [], [], []
    for f in glob.glob(os.path.join('DetectInScenes', 'sampled_frames', '*fpn1x.json')):
        vids.append(os.path.basename(f)[:3])
        with open(f, 'r') as fp:
            dets = json.load(fp)
        dets = dets['dets']
        scores = []
        for s in map(lambda t: t['score'], dets):
            scores = scores + s
        scores = np.array(scores)
        opf_50.append((scores >= 0.5).sum() / len(dets))
        opf_75.append((scores >= 0.75).sum() / len(dets))
        opf_90.append((scores >= 0.9).sum() / len(dets))

    tgt_dir = os.path.join('DetectInScenes', 'annotations', 'sparse')
    os.mkdir(tgt_dir)
    for v in vids:
        src_dir = os.path.join('DetectInScenes', 'sampled_frames', v)
        with open(os.path.join(src_dir, 'frames.json'), 'r') as fp:
            ifilelist = json.load(fp)['train']
        assert len(ifilelist) == 20
        for i in np.arange(0, len(ifilelist), 2):
            src = os.path.join(src_dir, ifilelist[i])
            tgt = os.path.join(tgt_dir, v + '_' + ifilelist[i])
            shutil.copy2(src, tgt)

    plt.figure(figsize=(15, 5))
    bins = np.arange(0, max(opf_50), 2)
    plt.subplot(1, 3, 1)
    plt.hist(opf_50, bins=bins)
    plt.ylim(0, 45)
    plt.title('Detected Object ($\\geq 0.5$) per Frame')
    plt.subplot(1, 3, 2)
    plt.hist(opf_75, bins=bins)
    plt.ylim(0, 45)
    plt.title('Detected Object ($\\geq 0.75$) per Frame')
    plt.subplot(1, 3, 3)
    plt.hist(opf_90, bins=bins)
    plt.ylim(0, 45)
    plt.title('Detected Object ($\\geq 0.9$) per Frame')
    plt.tight_layout()
    plt.show()

    budget = 4000
    counts = sorted(list(map(list, zip(vids, opf_50, opf_75, opf_90))))
    total = sum(map(lambda t: 1.0 / t[1], counts))
    for i in range(0, len(counts)):
        counts[i].append(int(budget / (total * counts[i][1])))
    print(sum(map(lambda t: t[-1], counts)))

    random.seed(100)
    random.shuffle(counts)
    counts_splits = [sorted(counts[i * 20 : (i + 1) * 20]) for i in range(0, 5)]

    count_vids, count_copied = 0, 0
    for s in range(0, len(counts_splits)):
        tgt_dir = os.path.join('DetectInScenes', 'annotations', 'dense%1d' % s)
        os.mkdir(tgt_dir)
        for v, _, _, _, n in counts_splits[s]:
            count_vids += 1
            src_dir = os.path.join('DetectInScenes', 'sampled_frames', v)
            with open(os.path.join(src_dir, 'frames.json'), 'r') as fp:
                ifilelist = json.load(fp)['valid']
            ifilelist = ifilelist[int(len(ifilelist) * 0.15) :]
            step = len(ifilelist) // (n - 1)
            for i in np.arange(0, len(ifilelist), step)[: n]:
                src = os.path.join(src_dir, ifilelist[i])
                tgt = os.path.join(tgt_dir, v + '_' + ifilelist[i])
                shutil.copy2(src, tgt)
                count_copied += 1
    print(count_vids, count_copied)


def stats_obj_2():
    vids, opf_50 = [], []
    for f in glob.glob(os.path.join('DetectInScenes', 'sampled_frames', '*fpn1x.json')):
        vids.append(os.path.basename(f)[:3])
        with open(f, 'r') as fp:
            dets = json.load(fp)
        dets = dets['dets']
        scores = []
        for s in map(lambda t: t['score'], dets):
            scores = scores + s
        scores = np.array(scores)
        opf_50.append((scores >= 0.5).sum() / len(dets))
    counts = sorted(list(map(list, zip(opf_50, vids))))
    assert len(counts) == 100
    print(counts)

    tgt_dir = os.path.join('DetectInScenes', 'annotator_sample_25')
    for i in range(0, 25):
        opf, vid = counts[i * 4 + random.randint(0, 3)]
        src_dir = os.path.join('DetectInScenes', 'sampled_frames', vid)
        with open(os.path.join(src_dir, 'frames.json'), 'r') as fp:
            ifilelist = json.load(fp)['train']
        random.shuffle(ifilelist)
        src = os.path.join(src_dir, ifilelist[0])
        tgt = os.path.join(tgt_dir, '%03d_%s_%s' % (int(opf), vid, ifilelist[0]))
        shutil.copy2(src, tgt)


if __name__ == '__main__':
    # probe()
    stats()

    # with open(os.path.join('DetectInScenes', 'files.json'), 'r') as fp:
    #     files = json.load(fp)
    # detect(files)

    # stats_obj()
    # stats_obj_2()

    # with open(os.path.join('..', 'files.json'), 'r') as fp:
    #     files = json.load(fp)
    # for v in files:
    #     L = v['video']['frames'] / v['video']['fps']
    #     if L / 3600 < 1.8:
    #         print(v['id'], L / 3600)
