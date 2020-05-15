import cv2
import numpy as np
import os


def rename_img(path):
    img_list = os.listdir(path)
    img_list.sort()
    for i in range(len(img_list)):
        in_name = os.path.join(path, img_list[i])
        try:
            idx = int(in_name.split('/')[-1].split('_')[1].split('.')[0])
            out_name = os.path.join(path, "%04d_img.jpg" % idx)
            os.rename(in_name, out_name)
        except:
            print('已重命名，跳出。')
    pass


def make_target_back():
    back_img20 = np.zeros([480, 640, 3]).astype(np.int8)
    back_img20[..., 0] = 120;
    back_img20[..., 1] = 255;
    back_img20[..., 2] = 155;
    # cv2.imshow('',back_img20)
    # cv2.waitKey()
    cv2.imwrite('./background.jpg', back_img20)
    pass


def depth_mask(color_path, depth_path):
    img_list = os.listdir(color_path)
    img_list.sort()
    for i in range(len(img_list)):
        color_name = img_list[i]
        idx = int(color_name.split('_')[0])
        depth_img_name = 'depth_' + str(idx).zfill(6) + '.png'
        # depth_img_name = color_name.replace('color', 'depth').replace('jpg', 'png')
        depth_img = cv2.imread(os.path.join(depth_path, depth_img_name), -1)
        mask = np.ones_like(depth_img) * 255
        mask[800 > depth_img] = 0
        mask[2900 < depth_img] = 0
        # cv2.imshow('',mask)
        # cv2.waitKey()
        save_name = color_name.replace('img', 'masksDL')
        cv2.imwrite(os.path.join(color_path, save_name), mask)
    pass


def composite4(fg, bg, a):
    fg = np.array(fg, np.float32)
    alpha = np.expand_dims(a / 255, axis=2)
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im


def process_zed(color_path, depth_path):
    """
    zed用的预处理。
    暂停，没有背景图。
    :param color_path:
    :param depth_path:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # image = Image.open(os.path.join(im_path,img_list[0]))
    out = cv2.VideoWriter('{0}.mp4'.format("zed_result"), fourcc, 10, (1280, 704))
    img_list = os.listdir(color_path)
    for i in range(len(img_list)):
        depth_img_name = str(i) + '.png'
        color_img_name = depth_img_name
        # depth_img_name = color_name.replace('color', 'depth').replace('jpg', 'png')
        depth_img = cv2.imread(os.path.join(depth_path, depth_img_name), -1)
        color_img = cv2.imread(os.path.join(color_path, color_img_name), -1)
        background = np.zeros_like(color_img)
        background[..., 0] = 120;
        background[..., 1] = 255;
        background[..., 2] = 155;
        alpha = np.zeros_like(depth_img)
        alpha[40 < depth_img] = 1 * 255
        alpha[:, :300] = 0
        alpha[:, 900:] = 0

        merge=composite4(color_img, background, alpha)

        # cv2.imshow('', merge)
        # cv2.waitKey()
        cv2.putText(merge, str(i), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 125, 125), 2)
        out.write(merge)
        # cv2.imwrite(os.path.join(color_path, save_name), mask)
    pass


def vismodel():
    import sys
    import torch
    import tensorwatch as tw
    from networks import ResnetConditionHR
    netM = ResnetConditionHR(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=7, n_blocks2=3)
    tw.draw_model(netM, [1, 3, 512, 512])

if __name__ == '__main__':
    # path1 = '/media/mozi/library1/dataset/result/background-matting-result/0514/color'
    # path2 = '/media/mozi/Elements SE/orbbec/dataset/rgbd采集数据/0514/1/depth'
    # rename_img(path1)
    # make_target_back()
    # depth_mask(path1, path2)
    '''zed'''
    # path1 = '/media/mozi/library1/dataset/RGBD_ours/zed/left1'
    # path2 = '/media/mozi/library1/dataset/RGBD_ours/zed/dis1'
    # process_zed(path1, path2)
    '''vis model'''
    vismodel()
    pass

# ffmpeg -r 10 -f image2 -i output/%04d_compose.jpg -vcodec libx264 -crf 15 -s 1280x720 -pix_fmt yuv420p teaser_compose.mp4
