#!python3

import os
import glob
import urllib
import m3u8
import streamlink
import time
import random
from multiprocessing import Pool as ProcessPool


def download_chunks(params):
    url, prefix, outputdir, duration, pause_second = params
    time.sleep(pause_second)
    print('download %s -> %s' % (url, prefix))
    def get_stream():
        stream = streamlink.streams(url)['best']
        m3u8_obj = m3u8.load(stream.args['url'])
        return m3u8_obj.segments[0]

    start_time, pre_time = None, None
    while True:
        segment = get_stream()
        cur_time = segment.program_date_time.strftime('%Y%m%d_%H%M%S')
        if start_time is None:
            start_time = segment.program_date_time.timestamp()
            fp = open(os.path.join(outputdir, prefix + '_' + str(cur_time) + '.ts'), 'ab+')
        if segment.program_date_time.timestamp() - start_time > duration:
            break
        if pre_time == cur_time:
            pass
        else:
            with urllib.request.urlopen(segment.uri) as response:
                html = response.read()
            fp.write(html)
            pre_time = cur_time
    fp.close()


if __name__ == '__main__':
    params_list = [
        ['https://www.youtube.com/watch?v=TmDmfkYpoQE', 'LondonBusRidesRoute12', '.', 7300, 0],
    ]
    pool = ProcessPool(processes=len(params_list))
    _ = pool.map_async(download_chunks, params_list).get()
    pool.close()
    pool.join()
