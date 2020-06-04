import os
from itertools import cycle
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import os
import logging
import math
from multiprocessing.pool import ThreadPool
import threading
import numpy as np

"""
2020-06-03修改：增加adobe数据。
"""
rgbd_paths = ['/dataset/bg-matting/0506/1-all/input',
              '/dataset/bg-matting/0506/2/color',
              '/dataset/bg-matting/0508/1-all/input',
              '/dataset/bg-matting/0508/2-all',
              '/dataset/bg-matting/0514/1-all/input',
              ]
captured_bgs = ['/dataset/bg-matting/0506/background/captured_bg.jpg',
                '/dataset/bg-matting/0506/background/captured_bg2.jpg',
                '/dataset/bg-matting/0508/background/captured_bg.jpg',
                '/dataset/bg-matting/0508/background/captured_bg2.jpg',
                '/dataset/bg-matting/0514/background/captured_bg.jpg'
                ]
# rgbd_paths = ['/media/mozi/library1/dataset/result/background-matting-result/0506/1-all/input',
#               ]
# captured_bgs = ['/media/mozi/library1/dataset/result/background-matting-result/0506/background/captured_bg.jpg'
#                 ]
target_bg_dir = '/dataset/coco/train2017'

adobe_fg_paths = ['/dataset/Adobe_Deep_Matting_Dataset/Training_set/Adobe-licensed images/fg',
                  '/dataset/Adobe_Deep_Matting_Dataset/Training_set/Other/fg']
adobe_alpha_paths = ['/dataset/Adobe_Deep_Matting_Dataset/Training_set/Adobe-licensed images/alpha',
                     '/dataset/Adobe_Deep_Matting_Dataset/Training_set/Other/alpha']
out_csv = 'rgbd1.csv'
workers = 4
num_bgs = 10
composite_out_path = '/dataset/bg-matting/composite'

''''''


def process_rgbdours():
    target_bg_paths = [os.path.join(target_bg_dir, f) for f in sorted(os.listdir(target_bg_dir))]
    target_bg_stream = cycle(target_bg_paths)

    i = 0
    output = []
    for rgbd_path in rgbd_paths:
        color_imgs = [f for f in os.listdir(rgbd_path) if f.endswith("_img.jpg")]
        color_imgs.sort()
        for color_img in color_imgs:
            image_path = os.path.join(rgbd_path, color_img)
            mask_name = color_img.replace('_img', '_masksDL')
            mask_path = os.path.join(rgbd_path, mask_name)
            captured_bg_path = captured_bgs[i]
            target_back = next(target_bg_stream)
            target_back_img = cv2.imread(target_back, -1)
            if len(target_back_img.shape) == 2:
                print('1-channel img:{}'.format(target_back))
                while True:
                    target_back = next(target_bg_stream)
                    target_back_img = cv2.imread(target_back, -1)
                    if len(target_back_img.shape) == 3:
                        break
            fg_gt = image_path
            line = image_path + ';' + captured_bg_path + ';' + mask_path + ';' + fg_gt + ';' + target_back + ';0\n'
            # print(line)
            output.extend(line)
        i += 1
    return output


'''
处理adobe数据
'''


def process_adobe():
    def format_pbar_str(i, im_name):
        pbar_prefix = "(" + str(i) + ") "
        width = 33 - len(pbar_prefix)
        pretty_name = pbar_prefix + ("..." + im_name[-(width - 3):] if len(im_name) > width else im_name)
        return pretty_name.rjust(33)

    def fixpath(path):
        return 'Data_adobe/' + path if not os.path.isabs(path) else path

    def composite4(fg, bg, a, w, h):
        bg = bg.crop((0, 0, w, h))
        bg.paste(fg, mask=a)
        return bg

    def process_foreground_image(jobarg, fg_path, a_path, bg_path, out_path):
        worker_thread_id = int(threading.current_thread().name.rpartition("-")[-1])
        i, job = jobarg
        im_name, bg_batch = job

        im_name = im_name.replace(fg_path, '')
        im = Image.open(os.path.join(fg_path, im_name))
        if os.path.exists(os.path.join(a_path, im_name)):
            al = Image.open(os.path.join(a_path, im_name))
            al_im_name = im_name
        else:
            al_im_name = im_name.replace('.jpg', '.png')
            al = Image.open(os.path.join(a_path, al_im_name))
            al_np = np.array(al)
            if al_np.sum() < 10:
                return
            al = Image.fromarray(np.array(al) * 255)

        bbox = im.size
        w = bbox[0]
        h = bbox[1]
        if im.mode != 'RGB' and im.mode != 'RGBA':
            im = im.convert('RGB')
        if len(al.getbands()) > 0:  # take the first channel, usually R
            al = al.split()[0]

        output_lines = []
        back_name_before = ''
        with lock:
            pbar = tqdm(bg_batch, position=worker_thread_id, desc=format_pbar_str(i, im_name), leave=False)
        for b, bg_name in enumerate(pbar):
            bg = Image.open(os.path.join(bg_path, bg_name))
            if bg.mode != 'RGB':
                bg = bg.convert('RGB')

            bg_bbox = bg.size
            bw = bg_bbox[0]
            bh = bg_bbox[1]
            wratio = w / bw
            hratio = h / bh
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                bg = bg.resize((math.ceil(bw * ratio), math.ceil(bh * ratio)), Image.BICUBIC)

            try:
                out = composite4(im, bg, al, w, h)
                back_idx = i * num_bgs + b
                out_name = os.path.join(out_path, im_name[:len(im_name) - 4] + '_' + str(back_idx) + '_comp.png')
                out.save(out_name, "PNG")

                back = bg.crop((0, 0, w, h))
                back_name = os.path.join(out_path, im_name[:len(im_name) - 4] + '_' + str(back_idx) + '_back.png')
                back.save(back_name, "PNG")

                if back_name_before == '':
                    back_bf_idx = i * num_bgs + len(bg_batch) - 1
                    back_name_before = os.path.join(out_path,
                                                    im_name[:len(im_name) - 4] + '_' + str(back_bf_idx) + '_back.png')

                # line = os.path.join(fixpath(fg_path), im_name) + ';' + os.path.join(fixpath(a_path), al_im_name) + ';' \
                #        + fixpath(out_name) + ';' + fixpath(back_name) + ';' + fixpath(back_name_before) + '\n'
                image_path = out_name
                captured_bg_path = back_name
                mask_path = os.path.join(a_path, al_im_name)
                target_back = back_name_before
                fg_gt = os.path.join(fg_path, im_name)
                line = image_path + ';' + captured_bg_path + ';' + mask_path + ';' + fg_gt + ';' + target_back + ';1\n'
                output_lines.append(line)

                back_name_before = back_name
            except Exception as e:
                # logging.error("Composing %s onto %s failed! Skipping. Error: %s" % im_name, bg_name, e)
                print("Composing %s onto %s failed! Skipping. Error: %s" % im_name, bg_name, e)
            with lock:
                pbar.update()
        with lock:
            pbar.close()
        return output_lines

    output = []
    for i in range(len(adobe_fg_paths)):
        fg_path = adobe_fg_paths[i]
        a_path = adobe_alpha_paths[i]
        bg_path = target_bg_dir
        out_path = composite_out_path
        fg_files = os.listdir(adobe_fg_paths[i])
        fg_files.sort()
        # fg_files = fg_files[:3]  # todo:暂时只取2w张图
        # a_files = os.listdir(a_path)
        bg_files = os.listdir(bg_path)
        bg_batches = [bg_files[i * num_bgs:(i + 1) * num_bgs] for i in range((len(bg_files) + num_bgs - 1) // num_bgs)]

        lock = threading.Lock()
        pool = ThreadPool(workers)
        with lock:
            total_pbar = tqdm(total=len(fg_files), position=workers + 2, desc="TOTAL", leave=True, smoothing=0.0)

        def update_total_pbar(_):
            with lock:
                total_pbar.update(1)

        jobs = []
        # im_name, bg_batch, fg_path, a_path, bg_path, out_path = job
        for jobargs in enumerate(zip(fg_files, bg_batches)):
            jobs.append(pool.apply_async(process_foreground_image, args=(jobargs, fg_path, a_path, bg_path, out_path),
                                         callback=update_total_pbar))
        pool.close()
        pool.join()

        for result in jobs:
            if result.get() is None:
                continue
            output.extend(result.get())
        tqdm.write("Done composing...")
    return output


if __name__ == '__main__':
    output = []
    output += process_rgbdours()
    output += process_adobe()
    with open(out_csv, "w") as f:
        for line in output:
            f.write(line)
