import cv2
import numpy as np
import os
import random


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


def rename_img_wtimestamp(path, depth_path):
    img_list = os.listdir(path)
    img_list.sort()
    for i in range(len(img_list)):
        in_name = os.path.join(path, img_list[i])
        try:
            assert not img_list[i].endswith('_img.jpg')
            idx = int(in_name.split('/')[-1].split('_')[0]) // 1000
            out_name = os.path.join(path, "%06d_img.jpg" % idx)

            depth_img_name = 'depth_' + str(idx).zfill(6) + '.png'
            # depth_img_name = color_name.replace('color', 'depth').replace('jpg', 'png')
            depth_img_path = os.path.join(depth_path, depth_img_name)

            assert os.path.exists(depth_img_path)
            os.rename(in_name, out_name)
        except Exception as e:
            print(e, '已重命名、或对应深度图不存在，跳出。')
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


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True,
               random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        print(s)
        print('estimate:', m, )
        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i + 1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


def cal_plane_func(depth_bg, roi_max):
    cam = 'kinect'
    if cam == 'kinect':
        w = 1280
        h = 720
        ox = w / 2
        oy = h / 2
        fx = w / (2 * np.tan(0.5 * np.pi * 90 / 180))
        fy = h / (2 * np.tan(0.5 * np.pi * 59 / 180))

    depth_bg[roi_max < depth_bg] = 0
    # cv2.imshow('',depth_bg*255)
    # cv2.waitKey()
    gridyy, gridxx = np.mgrid[:h, :w]

    xx_cam = (gridxx - ox) / fx * depth_bg
    yy_cam = (gridyy - oy) / fy * depth_bg

    depth_bg = depth_bg[..., np.newaxis]
    bg_xyz = np.concatenate((xx_cam[..., np.newaxis], yy_cam[..., np.newaxis], depth_bg), axis=2)
    bg_xyz_resh_all = np.reshape(bg_xyz, (-1, 3))
    bg_xyz_resh = bg_xyz_resh_all[bg_xyz_resh_all[:, 2] > 0]

    # n = 100
    max_iterations = 300
    goal_inliers = bg_xyz_resh.shape[0] * 0.4
    # RANSAC
    m, _ = run_ransac(bg_xyz_resh, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    return a, b, c, d, gridyy, gridxx, ox, oy, fx, fy


def depth_mask(color_path, depth_path, cut_ground=False, bg_depth_path='', save_bg=False):
    img_list = [f for f in os.listdir(color_path) if f.endswith("_img.jpg")]
    img_list.sort()

    depth_bg = cv2.imread(bg_depth_path, -1)
    bg_depth_h, bg_depth_w = depth_bg.shape
    bg_depth_roi = depth_bg[bg_depth_h // 3:bg_depth_h * 2 // 3, bg_depth_w // 3:bg_depth_w * 2 // 3]
    roi_max = bg_depth_roi.mean()-500
    roi_min = 650

    if cut_ground:
        a, b, c, d, gridyy, gridxx, ox, oy, fx, fy = cal_plane_func(depth_bg, roi_max)
    for i in range(len(img_list)):
        color_name = img_list[i]
        idx = int(color_name.split('_')[0])
        depth_img_name = 'depth_' + str(idx).zfill(6) + '.png'
        # depth_img_name = color_name.replace('color', 'depth').replace('jpg', 'png')
        depth_img = cv2.imread(os.path.join(depth_path, depth_img_name), -1)
        mask = np.ones_like(depth_img) * 255
        mask[roi_min > depth_img] = 0
        mask[roi_max < depth_img] = 0
        mask_bg = np.zeros_like(depth_img) * 255
        mask_bg[roi_max < depth_img] = 255

        if cut_ground:
            xx_cam = (gridxx - ox) / fx * depth_img
            yy_cam = (gridyy - oy) / fy * depth_img
            dis = abs(a * xx_cam + b * yy_cam + c * depth_img + d) / ((a ** 2 + b ** 2 + c ** 2) ** 0.5)
            mask[dis < 35] = 0

        # cv2.imshow('',mask)
        # cv2.waitKey()
        '''模拟d2c'''
        # x_offset = 10
        # y_offset = 1
        # M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        # 用仿射变换实现平移
        # rows, cols = mask.shape
        # mask = cv2.warpAffine(mask, M, (cols, rows))

        save_name = color_name.replace('img', 'masksDL')

        if save_bg:
            mask_tmp = np.zeros_like(mask)
            mask_merge = np.concatenate((mask[..., np.newaxis], mask_bg[..., np.newaxis], mask_tmp[..., np.newaxis]),
                                        axis=2)
            cv2.imwrite(os.path.join(color_path, save_name), mask_merge)
        else:
            cv2.imwrite(os.path.join(color_path, save_name), mask)
        del mask, mask_bg
    pass


def depth_mask2(color_path, depth_path, cut_ground=False, bg_depth_path_list=None, save_bg=False):
    """
    基于若干背景图和当前帧的差值，判断前景。
    """
    img_list = [f for f in os.listdir(color_path) if f.endswith("_img.jpg")]
    img_list.sort()
    bg_depth_merge = np.array([])
    for bg_depth_path in bg_depth_path_list:
        bg_depth_tmp = cv2.imread(bg_depth_path, -1)
        if bg_depth_merge.size == 0:
            bg_depth_merge = bg_depth_tmp[..., np.newaxis]
        else:
            bg_depth_merge = np.concatenate((bg_depth_merge, bg_depth_tmp[..., np.newaxis]), axis=2)
    bg_depth = np.median(bg_depth_merge, axis=2)
    # cv2.imshow('',bg_depth/5000)
    # cv2.waitKey()

    if cut_ground:
        bg_depth_h, bg_depth_w = bg_depth.shape
        bg_depth_roi = bg_depth[bg_depth_h // 3:bg_depth_h * 2 // 3, bg_depth_w // 3:bg_depth_w * 2 // 3]
        roi_max = bg_depth_roi.mean()
        a, b, c, d, gridyy, gridxx, ox, oy, fx, fy = cal_plane_func(bg_depth, roi_max)
    for i in range(len(img_list)):
        color_name = img_list[i]
        idx = int(color_name.split('_')[0])
        depth_img_name = 'depth_' + str(idx).zfill(6) + '.png'
        # depth_img_name = color_name.replace('color', 'depth').replace('jpg', 'png')
        depth_img = cv2.imread(os.path.join(depth_path, depth_img_name), -1)
        dis = abs(bg_depth - depth_img)
        mask = np.zeros_like(depth_img)
        mask[dis < 300] = 255
        mask[depth_img < 50] = 0

        if cut_ground:
            xx_cam = (gridxx - ox) / fx * depth_img
            yy_cam = (gridyy - oy) / fy * depth_img
            dis = abs(a * xx_cam + b * yy_cam + c * depth_img + d) / ((a ** 2 + b ** 2 + c ** 2) ** 0.5)
            mask[dis < 30] = 0

        # cv2.imshow('',mask)
        # cv2.waitKey()
        '''模拟d2c'''
        # x_offset = 10
        # y_offset = 1
        # M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        # 用仿射变换实现平移
        # rows, cols = mask.shape
        # mask = cv2.warpAffine(mask, M, (cols, rows))

        save_name = color_name.replace('img', 'masksDL')
        save_bgname = color_name.replace('img', 'masksBG')
        cv2.imwrite(os.path.join(color_path, save_name), mask)
        # if save_bg:
        #     cv2.imwrite(os.path.join(color_path, save_bgname), mask_bg)
        del mask
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

        merge = composite4(color_img, background, alpha)

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
    # path1 = '/media/mozi/library1/dataset/result/background-matting-result/0525/yunlong-1/input'
    # path2 = '/media/mozi/library1/download/data/yunlong_0525_1/1/depth'
    # rename_img(path1)
    # rename_img_wtimestamp(path1)
    # make_target_back()
    # depth_mask(path1, path2)
    '''zed'''
    # path1 = '/media/mozi/library1/dataset/RGBD_ours/zed/left1'
    # path2 = '/media/mozi/library1/dataset/RGBD_ours/zed/dis1'
    # process_zed(path1, path2)
    '''vis model'''
    # vismodel()
    '''预处理raw'''
    # path1 = '/media/mozi/library1/dataset/result/background-matting-result/0528/12astra/input'
    # path2 = '/media/mozi/library1/download/data/yunlong-0527/12astra/depth'
    # bg_depth_path='/media/mozi/library1/download/data/yunlong-0527/12astra/depth/depth_006925.png'
    # rename_img_wtimestamp(path1, path2)
    # depth_mask(path1, path2, bg_depth_path=bg_depth_path, save_bg=True)
    '''ransac去除地面'''
    path1 = '/media/mozi/library1/download/bg-matting/0514/1-all/input'
    path2 = '/media/mozi/Elements SE/orbbec/dataset/rgbd-capture/0514/1/depth'
    bg_depth_path='/media/mozi/library1/download/bg-matting/0514/1-all/bg_depth.png'
    # rename_img(path1)
    depth_mask(path1, path2, cut_ground=True, bg_depth_path=bg_depth_path, save_bg=False)
    '''0529，另一版算前景软分割的方法。背景差值，而不是两刀切。感觉不太可行，因为深度图噪声多'''
    # path1 = '/media/mozi/library1/dataset/result/background-matting-result/0528/10/input'
    # path2 = '/media/mozi/library1/download/data/yunlong-0527/10/depth'
    # bg_depth_list = ['/media/mozi/library1/download/data/yunlong-0527/10/depth/depth_000697.png',
    #                  '/media/mozi/library1/download/data/yunlong-0527/10/depth/depth_000735.png',
    #                  '/media/mozi/library1/download/data/yunlong-0527/10/depth/depth_000759.png',
    #                  '/media/mozi/library1/download/data/yunlong-0527/10/depth/depth_000843.png']
    # depth_mask2(path1, path2, bg_depth_path_list=bg_depth_list, cut_ground=False, save_bg=False)
    pass

# ffmpeg -r 10 -f image2 -i output/%04d_compose.jpg -vcodec libx264 -crf 15 -s 1280x720 -pix_fmt yuv420p teaser_compose.mp4
