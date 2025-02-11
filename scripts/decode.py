#!python3

import os
import glob
import json
import tqdm
import skimage.io
import skvideo.io
from multiprocessing import Pool as ProcessPool
import matplotlib.pyplot as plt


def decode_frames(meta):
    vfilename = os.path.join(os.path.dirname(__file__), '..', 'videos', meta['filename'])
    video_id = meta['id']
    if not os.access(vfilename, os.R_OK):
        print('cannot read file', vfilename)
        return

    fps, F, H, W = meta['video']['fps'], meta['video']['frames'] - 1, meta['video']['H'], meta['video']['W']
    framesdir = os.path.join(os.path.dirname(__file__), '..', 'images', 'train_200_valid_200', video_id)
    output_json = os.path.join(framesdir, 'frames.json')
    if os.access(output_json, os.R_OK):
        print('%s exists, skipped' % output_json)
        return

    if not os.access(framesdir, os.W_OK):
        os.mkdir(framesdir)
    desc = '%s... %04dx%04d %.1f fps F=%d' % (meta['filename'][:15], H, W, fps, F)

    train_frames = int(fps * 1.5 * 3600)
    valid_frames = min(F - train_frames, int(fps * 0.5 * 3600))
    interval_train, interval_valid = train_frames // 200, valid_frames // 200
    ifilelists = {'train': [], 'valid': []}

    reader = skvideo.io.vreader(vfilename)
    count = 0
    for i in tqdm.tqdm(range(0, F), ascii=True, desc=desc):
        if i > train_frames + valid_frames:
            break
        try:
            frame = next(reader)
        except StopIteration:
            break
        if i < train_frames:
            count += 1
            if count >= interval_train:
                fn = '%08d.jpg' % i
                skimage.io.imsave(os.path.join(framesdir, fn), frame, quality=80)
                ifilelists['train'].append(fn)
                count = 0
        else:
            count += 1
            if count >= interval_valid:
                fn = '%08d.jpg' % i
                skimage.io.imsave(os.path.join(framesdir, fn), frame, quality=80)
                ifilelists['valid'].append(fn)
                count = 0
    reader.close()
    with open(output_json, 'w') as fp:
        json.dump(ifilelists, fp)


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    print('%s videos' % len(files))
    pool = ProcessPool(processes=3)
    _ = pool.map_async(decode_frames, files).get()
    pool.close()
    pool.join()
