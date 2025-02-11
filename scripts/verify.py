#!python3

import os
import json
import glob
import hashlib
import tqdm
import argparse


def sha_hashes(filename):
    results = {}
    for desc, hasher in [('sha1', hashlib.sha1()), ('sha512', hashlib.sha512())]:
        with open(filename, 'rb') as fp:
            while True:
                content = fp.read(10 * 1024 * 1024)
                if not content:
                    break
                hasher.update(content)
        results[desc] = hasher.hexdigest()
    return results


def verify(opts):
    with open(os.path.join(os.path.dirname(__file__), '..', 'files.json'), 'r') as fp:
        files = json.load(fp)
    video_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'videos'))
    lmdb_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'images', 'train_lmdb'))
    model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    if 'video' in opts:
        print('verify video files in %s' % video_dir)
        passed, missing, corrupt = [], [], []
        for v in tqdm.tqdm(files, ascii=True):
            vfilename = v['filename']
            if not os.access(os.path.join(video_dir, vfilename), os.R_OK):
                missing.append(vfilename)
                continue
            hashes = sha_hashes(os.path.join(video_dir, vfilename))
            if hashes['sha1'].lower() != v['file']['sha1'].lower() or hashes['sha512'].lower() != v['file']['sha512'].lower():
                corrupt.append(vfilename)
                continue
            passed.append(vfilename)
        print('%d/%d passed' % (len(passed), len(files)))
        if len(missing) > 0:
            print('missing or unreadable files:')
            for v in missing:
                print(v)
        if len(corrupt) > 0:
            print('hash mismatched files:')
            for v in corrupt:
                print(v)

    if 'lmdb' in opts:
        print('verify LMDB files in %s' % lmdb_dir)
        passed, missing, corrupt = [], {}, {}
        for v in tqdm.tqdm(files, ascii=True):
            lmdb_path = os.path.join(lmdb_dir, v['id'])
            flag = True
            for f in ['frames.json', 'data.mdb', 'lock.mdb']:
                if not os.access(os.path.join(lmdb_path, f), os.R_OK):
                    flag = False
                    if not lmdb_path in missing:
                        missing[lmdb_path] = []
                    missing[lmdb_path].append(f)
            if not flag:
                continue
            with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
                hashes = json.load(fp)['hash']
            for f in ['data.mdb', 'lock.mdb']:
                hashes_calculated = sha_hashes(os.path.join(lmdb_path, f))
                if hashes_calculated['sha1'].lower() != hashes[f]['sha1'].lower() or hashes_calculated['sha512'].lower() != hashes[f]['sha512'].lower():
                    flag = False
                    if not lmdb_path in corrupt:
                        corrupt[lmdb_path] = []
                    corrupt[lmdb_path].append(f)
            if not flag:
                continue
            passed.append(lmdb_path)
        print('%d/%d passed' % (len(passed), len(files)))
        if len(missing) > 0:
            print('missing or unreadable files:')
            for v in sorted(list(missing.keys())):
                print('%s%s[%s]' % (v, os.path.sep, ','.join(missing[v])))
        if len(corrupt) > 0:
            print('hash mismatched files:')
            for v in sorted(list(corrupt.keys())):
                print('%s%s[%s]' % (v, os.path.sep, ','.join(corrupt[v])))

    if 'frame' in opts:
        print('verify training images in %s' % lmdb_dir)
        passed, missing, corrupt = [], [], []
        for v in tqdm.tqdm(files, ascii=True):
            lmdb_path = os.path.join(lmdb_dir, v['id'])
            if not (os.access(os.path.join(lmdb_path, 'frames.json'), os.R_OK) and os.access(os.path.join(lmdb_path, 'jpegs'), os.R_OK)):
                missing.append(lmdb_path)
                continue
            with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
                ifilelist_record = sorted(json.load(fp)['ifilelist'])
            ifilelist_present = sorted(glob.glob(os.path.join(lmdb_path, 'jpegs', '*.jpg')))
            if not (len(ifilelist_record) == len(ifilelist_present) and ifilelist_record[0] == os.path.basename(ifilelist_present[0]) and ifilelist_record[-1] == os.path.basename(ifilelist_present[-1])):
                corrupt.append(lmdb_path)
                continue
            passed.append(lmdb_path)
        print('%d/%d passed' % (len(passed), len(files)))
        if len(missing) > 0:
            print('missing or unreadable files:')
            for v in missing:
                print(v)
        if len(corrupt) > 0:
            print('hash mismatched files:')
            for v in corrupt:
                print(v)

    if 'model' in opts:
        print('verify trained base models in %s' % model_dir)
        with open(os.path.join(model_dir, 'models.json'), 'r') as fp:
            models = json.load(fp)
        passed, missing, corrupt = [], [], []
        for m in tqdm.tqdm(models, ascii=True):
            ckpt = os.path.join(model_dir, m['ckpt'])
            if not os.access(ckpt, os.R_OK):
                missing.append(ckpt)
                continue
            hashes = sha_hashes(ckpt)
            if hashes['sha1'].lower() != m['sha1'].lower() or hashes['sha512'].lower() != m['sha512'].lower():
                corrupt.append(ckpt)
                continue
            passed.append(ckpt)
        print('%d/%d passed' % (len(passed), len(models)))
        if len(missing) > 0:
            print('missing or unreadable files:')
            for m in missing:
                print(m)
        if len(corrupt) > 0:
            print('hash mismatched files:')
            for m in corrupt:
                print(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='verify dataset and model files')
    parser.add_argument('--opts', nargs='+', default=[], choices=['video', 'lmdb', 'frame', 'model'])
    args = parser.parse_args()
    verify(args.opts)
