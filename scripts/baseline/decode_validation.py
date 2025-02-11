#!python3

import os
import glob
import json
import tqdm
import skimage.io
import skvideo.io
import hashlib
import numpy as np
import imageio
import lmdb
import collections.abc
from multiprocessing import Pool as ProcessPool


def decode_frames(meta):
    vfilename = os.path.join(os.path.dirname(__file__), '..', '..', 'videos', meta['filename'])
    video_id = meta['id']
    if not os.access(vfilename, os.R_OK):
        print('cannot read file', vfilename)
        return

    fps, F, H, W = meta['video']['fps'], meta['video']['frames'] - 1, meta['video']['H'], meta['video']['W']
    sample_fps, encoding, context_seconds = 5, 'utf-8', 60
    framesdir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_lmdb', video_id)
    sample_indices = np.arange(0, F - 1, fps / sample_fps).astype(int).tolist()

    annodir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'annotated', video_id, 'unmasked')
    valid_list = glob.glob(os.path.join(annodir, '*.jpg'))
    valid_list = [int(os.path.basename(f).split('.')[0]) for f in valid_list]
    valid_list = {i : list(filter(lambda x: x >= 0 and x >= i - fps * context_seconds and x <= i, sample_indices)) for i in valid_list} # context frames
    for i in valid_list:
        valid_list[i] = sorted(list(set(valid_list[i] + [i])))
    valid_context_set = []
    for i in valid_list:
        valid_context_set = valid_context_set + valid_list[i]
    valid_context_set = set(valid_context_set)

    output_json = os.path.normpath(os.path.join(framesdir, 'frames.json'))
    if os.access(output_json, os.R_OK):
        print('%s exists, skipped' % output_json)
        return
    if not os.access(framesdir, os.W_OK):
        os.mkdir(framesdir)

    lmdb_map_size_inc = 64 * 1024 * 1024
    lmdb_map_size = lmdb_map_size_inc
    saved_bytes = 0
    env = lmdb.open(framesdir, map_size=lmdb_map_size, metasync=True, sync=True)
    txn = env.begin(write=True)

    desc = '%s... %04dx%04d %.1f fps F=%d' % (meta['filename'][:15], H, W, fps, F)
    ifilelist = []
    reader = skvideo.io.vreader(vfilename)
    for i in tqdm.tqdm(range(0, F), ascii=True, desc=desc):
        try:
            frame = next(reader)
        except StopIteration:
            break
        fn = '%08d.jpg' % i
        if i in valid_context_set:
            jpeg_bytes = imageio.imwrite('<bytes>', frame, plugin='pillow', format='JPEG', quality=80)
            fn_bytes = fn.encode(encoding)
            saved_bytes += len(jpeg_bytes) + len(fn_bytes)
            if saved_bytes > lmdb_map_size * 0.95:
                txn.commit()
                env.close()
                lmdb_map_size += lmdb_map_size_inc
                env = lmdb.open(framesdir, map_size=lmdb_map_size, metasync=True, sync=True)
                txn = env.begin(write=True)
            ret = txn.put(fn_bytes, jpeg_bytes)
            assert ret, 'put data failed'
            ifilelist.append(fn)
    reader.close()
    txn.commit()
    env.close()

    with open(output_json, 'w') as fp:
        json.dump({'map_size': lmdb_map_size, 'encoding': encoding, 'ifilelist': ifilelist, 'valid_list': valid_list, 'context_seconds': context_seconds, 'sample_fps': sample_fps, 'meta': meta}, fp)
    print('%d validation frames, %d context frames, saved %s' % (len(valid_list), len(ifilelist), output_json))


class ValidationContextFrames(collections.abc.Sequence):
    def __init__(self, video_id):
        self.video_id = video_id
        self.lmdb_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_lmdb', video_id))
        assert type(self.video_id) == type('000') and len(self.video_id) == 3, '<video_id> should be the 3-digits video ID string'
        assert os.access(os.path.join(self.lmdb_path, 'frames.json'), os.R_OK), 'frames.json not found in: ' + self.lmdb_path
        assert os.access(os.path.join(self.lmdb_path, 'data.mdb'), os.R_OK) and os.access(os.path.join(self.lmdb_path, 'lock.mdb'), os.R_OK), 'LMDB files not found in: ' + self.lmdb_path
        with open(os.path.join(self.lmdb_path, 'frames.json'), 'r') as fp:
            self.meta = json.load(fp)
        self.ifilelist, self.valid_list = self.meta['ifilelist'], self.meta['valid_list']
        self.lmdb_env, self.lmdb_txn = None, None
    def __del__(self):
        if not self.lmdb_env is None:
            self.lmdb_env.close()
    def __repr__(self):
        return 'ValidationContextFrames [video %s] [%dx%d] [%d validation frames] [%d context frames] [%.1f fps] [%.2f GB]' % (self.video_id, self.meta['meta']['video']['H'], self.meta['meta']['video']['W'], len(self.valid_list), len(self), self.meta['sample_fps'], os.path.getsize(os.path.join(self.lmdb_path, 'data.mdb')) / (1024 ** 3))

    def context_indices(self):
        keys, indices = sorted(list(self.valid_list.keys())), []
        for k in keys:
            assert self.valid_list[k][-1] == int(k)
            indices.append(list(map(lambda x: self.ifilelist.index('%08d.jpg' % x), self.valid_list[k])))
        keys = list(map(lambda x: '%08d.jpg' % int(x), keys))
        assert len(keys) == len(indices)
        for _idx in indices:
            for i in _idx:
                assert i >= 0
        return keys, indices

    def __len__(self):
        return len(self.ifilelist)
    def __getitem__(self, index):
        if isinstance(index, int):
            if self.lmdb_txn is None:
                self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
                self.lmdb_txn = self.lmdb_env.begin(write=False)
            fn, frame_id = self.ifilelist[index], int(self.ifilelist[index].split('.')[0])
            jpeg_bytes = self.lmdb_txn.get(fn.encode(self.meta['encoding']))
            im = imageio.imread(jpeg_bytes, format='JPEG')
            return np.array(im), frame_id, fn, index
        if isinstance(index, slice):
            pass
        raise Exception('unsupported index: %s' % index)
    def __setitem__(self, index, item):
        raise NotImplementedError
    def __delitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        self.iter_index = -1
        return self
    def __next__(self):
        self.iter_index += 1
        if self.iter_index < len(self):
            return self[self.iter_index]
        else:
            raise StopIteration

    def __contains__(self, item):
        raise NotImplementedError
    def __reversed__(self):
        raise NotImplementedError
    def index(self, item):
        raise NotImplementedError
    def count(self, item):
        raise NotImplementedError


if __name__ == '__main__':
    # with open(os.path.join(os.path.dirname(__file__), '..', '..', 'files.json'), 'r') as fp:
    #     files = json.load(fp)
    # files = list(filter(lambda f: not os.access(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'valid_lmdb', f['id'], 'frames.json'), os.R_OK), files))
    # procs = 1
    # print('decode %s with %d processes' % (list(map(lambda f: f['id'], files)), procs))
    # pool = ProcessPool(processes=procs)
    # _ = pool.map_async(decode_frames, files).get()
    # pool.close()
    # pool.join()

    pass
