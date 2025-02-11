#!python3
'''
rem ffmpeg -i "BaoBabRestaurantLamai_20211207_061505.ts" -an -vcodec libx264 -preset slow -crf 21 -threads 16 "133.BaoBabRestaurantLamai_20211207_061505.mp4"
rem ffmpeg -i "003.HidaTakayama_20211123_235240.mp4" -an -vcodec libx265 -preset medium -crf 25 -r 30 -x265-params pools=12 "003.HidaTakayama_20211123_235240.hevc.mp4"
rem ffmpeg -i "013.Revelstoke_20211124_205130.mp4" -an -vcodec libx265 -preset medium -crf 25 -r 30 -x265-params pools=12 "013.Revelstoke_20211124_205130.hevc.mp4"
rem ffmpeg -i "041.ShiodomeRails_20211125_015235.mp4" -an -vcodec libx265 -preset medium -crf 25 -r 30 -x265-params pools=16,16 "041.ShiodomeRails_20211125_015235.hevc.mp4"
ffmpeg -i "096.PlatjadAroCenter_20211202_092450.mp4" -an -vcodec libx265 -preset medium -crf 25 -r 30 -x265-params pools=12 "096.PlatjadAroCenter_20211202_092450.hevc.mp4"

ffmpeg -hide_banner -loglevel info -hwaccel cuda -i 173.WalkingNewYorkCity_20211210_165808.hevc.mp4 -x265-params pools=16,16 -an -vcodec libx265 -vf scale=1280:720 -preset medium -crf 26 173.WalkingNewYorkCity_20211210_165808.hevc.2.mp4

ffmpeg -hide_banner -loglevel info -hwaccel cuda -i 166.SattaTougeFujisan_20211209_021749.hevc.mp4 -x265-params pools=12 -an -vcodec libx265 -vf crop=1080:1080:0:0 -preset medium -crf 25 166.SattaTougeFujisan_20211209_021749.hevc.2.mp4

'''

import os
import glob
import skvideo.io


def transcode():
    vfilelist = glob.glob('*.mp4')
    # vfilelist = list(filter(lambda f: f.find('001.Jac') < 0, vfilelist))
    # vfilelist = list(filter(lambda f: f.find('097.Su') < 0, vfilelist))
    # vfilelist = list(filter(lambda f: f.find('130.Ta') < 0, vfilelist))
    # vfilelist = list(filter(lambda f: f.find('hevc') < 0, vfilelist))
    vfilelist = [(os.path.getsize(v) / (1024 ** 3), v) for v in vfilelist]
    print(len(vfilelist))
    vfilelist = sorted(vfilelist)[::-1]
    vfilelist = vfilelist[: 5]
    print(vfilelist)
    return

    gb2s = []
    param_threads = '-x265-params pools=12'
    for gb1, input_file in vfilelist:
        meta = skvideo.io.ffprobe(input_file)['video']
        fps = meta['@r_frame_rate'].split('/')
        fps = float(fps[0]) / float(fps[1])
        if fps > 32:
            param_r = '-r 30'
        else:
            param_r = ''

        output_file = input_file[: -3] + 'hevc.mp4'
        cmd = 'ffmpeg -hide_banner -loglevel info -hwaccel cuda -i "%s" -an -vcodec libx265 -preset medium -crf 25 %s %s "%s"' % (input_file, param_r, param_threads, output_file)
        # cmd = 'ffmpeg -hide_banner -loglevel info -hwaccel cuda -i "%s" -an -vcodec hevc_nvenc -preset medium -rc vbr -cq 26 %s "%s"' % (input_file, param_r, output_file)
        print(cmd)
        os.system(cmd)

        gb2s.append((os.path.getsize(output_file) / (1024 ** 3), output_file))

    print(vfilelist)
    print(gb2s)


if __name__ == '__main__':
    transcode()
