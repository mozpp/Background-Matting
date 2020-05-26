from __future__ import print_function

import os, glob, time, argparse, pdb, cv2
# import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from functions import *
from networks import ResnetConditionHR

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])

"""Parses arguments."""
parser = argparse.ArgumentParser(description='Background Matting.')
parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',
                    # choices=['real-fixed-cam', 'real-hand-held', 'syn-comp-adobe'],
                    help='Trained background matting model')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Directory to save the output results. (required)')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory to load input images. (required)')
parser.add_argument('-tb', '--target_back', type=str, help='Directory to load the target background.')
parser.add_argument('-b', '--back', type=str, default=None,
                    help='Captured background image. (only use for inference on videos with fixed camera')

args = parser.parse_args()

# input model
model_main_dir = 'Models/' + args.trained_model + '/';
# input data path
data_path = args.input_dir

if os.path.isdir(args.target_back):
    args.video = True
    print('Using video mode')
else:
    args.video = False
    print('Using image mode')
    # target background path
    back_img10 = cv2.imread(args.target_back);
    back_img10 = cv2.cvtColor(back_img10, cv2.COLOR_BGR2RGB);
    # Green-screen background
    back_img20 = np.zeros(back_img10.shape);
    back_img20[..., 0] = 120;
    back_img20[..., 1] = 255;
    back_img20[..., 2] = 155;

# initialize network
fo = glob.glob(model_main_dir + 'netG_epoch_*')
epoch_idx_max = 0
for model_name in fo:
    epoch_idx = int(model_name.split('_')[-1].split('.pth')[0])
    epoch_idx_max = epoch_idx if epoch_idx >= epoch_idx_max else epoch_idx_max
model_name1 = os.path.join(model_main_dir, 'netG_epoch_{}.pth'.format(epoch_idx_max))
netM = ResnetConditionHR(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=7, n_blocks2=3)
netM = nn.DataParallel(netM)
netM.load_state_dict(torch.load(model_name1))
netM.cuda();
netM.eval()
cudnn.benchmark = True
reso = (512, 512)  # input reoslution to the network
# reso = (288, 288)  # input reoslution to the network

# load captured background for video mode, fixed camera
if args.back is not None:
    bg_im0 = cv2.imread(args.back);
    bg_im0 = cv2.cvtColor(bg_im0, cv2.COLOR_BGR2RGB);

# Create a list of test images
test_imgs = [f for f in os.listdir(data_path) if
             os.path.isfile(os.path.join(data_path, f)) and (f.endswith('_img.png') or f.endswith('_img.jpg'))]
test_imgs.sort()

# output directory
result_path = args.output_dir

if not os.path.exists(result_path):
    os.makedirs(result_path)


def modify_alpha(alpha_out0, bg_mask=None, data_path=None, filename=None, roi=720, soft_seg=None):  # roi=608
    """
        mozi added. 基于深度值修改alpha。
        :param roi: 超参数，用于去除地面的深度值。
        :param alpha_out0: 当前范围0-255
        :param data_path:
        :return:
    """
    if soft_seg is None:
        depth_mask = cv2.imread(os.path.join(data_path, filename.replace('_img', '_masksDL')), 0)
    else:
        depth_mask = soft_seg
    depth_mask[int(roi / 720 * depth_mask.shape[0]):, ...] = 0
    # 腐蚀，膨胀，高斯滤波
    kernel = np.ones((3, 3), np.uint8)
    depth_mask = cv2.erode(depth_mask, kernel, iterations=5)  # 对于Astra，腐蚀iter=2；对于kinect，腐蚀iter=1
    # depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)
    # depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
    depth_mask = cv2.GaussianBlur(depth_mask, (7, 7), 5)

    # depth_valid = np.zeros_like(depth_mask)
    # depth_valid[np.logical_and(1500 < depth_mask, depth_mask < 2700)] = 1
    # depth_valid[depth_mask > 1] = 1

    alpha_out0 = alpha_out0[..., np.newaxis]
    depth_mask = depth_mask[..., np.newaxis]
    merge = np.concatenate((alpha_out0, depth_mask), axis=2)
    alpha_out0 = np.max(merge, axis=2)
    # alpha_out0 = depth_mask/2+alpha_out0/2

    if bg_mask is not None:
        depth_bg_mask = 255 - bg_mask
        depth_bg_mask = cv2.dilate(depth_bg_mask, kernel, iterations=2)
        alpha_out0 = alpha_out0 * (depth_bg_mask / 255)

    return alpha_out0


def modify_alpha_wbg(alpha_out0, data_path=None, filename=None, roi=720, soft_seg=None):  # roi=608
    """
        mozi added. 基于深度值修改alpha。
        :param roi: 超参数，用于去除地面的深度值。
        :param alpha_out0: 当前范围0-255
        :param data_path:
        :return:
    """
    if soft_seg is None:
        depth_bg_mask = cv2.imread(os.path.join(data_path, filename.replace('_img', '_masksBG')), 0)
    else:
        depth_bg_mask = soft_seg
    depth_bg_mask[int(roi / 720 * depth_bg_mask.shape[0]):, ...] = 0
    depth_bg_mask = 255 - depth_bg_mask
    # 腐蚀，膨胀，高斯滤波
    kernel = np.ones((3, 3), np.uint8)
    # depth_bg_mask = cv2.erode(depth_bg_mask, kernel, iterations=5)  # 对于Astra，腐蚀iter=2；对于kinect，腐蚀iter=1
    depth_bg_mask = cv2.dilate(depth_bg_mask, kernel, iterations=2)
    # depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
    depth_bg_mask = cv2.GaussianBlur(depth_bg_mask, (7, 7), 1)

    # alpha_out0 = alpha_out0[..., np.newaxis]
    # depth_bg_mask = depth_bg_mask[..., np.newaxis]
    # merge = np.concatenate((alpha_out0, depth_bg_mask), axis=2)
    # alpha_out0 = np.max(merge, axis=2)
    alpha_out0 = alpha_out0 * (depth_bg_mask/255)

    return alpha_out0

for i in range(0, len(test_imgs)):
    time0 = time.time()
    filename = test_imgs[i]
    # original image
    bgr_img = cv2.imread(os.path.join(data_path, filename));
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB);

    if args.back is None:
        # captured background image
        bg_im0 = cv2.imread(os.path.join(data_path, filename.replace('_img', '_back')));
        bg_im0 = cv2.cvtColor(bg_im0, cv2.COLOR_BGR2RGB);

    # segmentation mask
    rcnn = cv2.imread(os.path.join(data_path, filename.replace('_img', '_masksDL')), 0);

    # background mask
    bg_mask = cv2.imread(os.path.join(data_path, filename.replace('_img', '_masksBG')), 0)

    if args.video:  # if video mode, load target background frames
        # target background path
        back_img10 = cv2.imread(os.path.join(args.target_back, filename.replace('_img.png', '.png')));
        back_img10 = cv2.cvtColor(back_img10, cv2.COLOR_BGR2RGB);
        # Green-screen background
        back_img20 = np.zeros(back_img10.shape);
        back_img20[..., 0] = 120;
        back_img20[..., 1] = 255;
        back_img20[..., 2] = 155;

        # create multiple frames with adjoining frames
        gap = 20
        multi_fr_w = np.zeros((bgr_img.shape[0], bgr_img.shape[1], 4))
        idx = [i - 2 * gap, i - gap, i + gap, i + 2 * gap]
        for t in range(0, 4):
            if idx[t] < 0:
                idx[t] = len(test_imgs) + idx[t]
            elif idx[t] >= len(test_imgs):
                idx[t] = idx[t] - len(test_imgs)

            file_tmp = test_imgs[idx[t]]
            bgr_img_mul = cv2.imread(os.path.join(data_path, file_tmp));
            multi_fr_w[..., t] = cv2.cvtColor(bgr_img_mul, cv2.COLOR_BGR2GRAY);

    else:
        ## create the multi-frame
        multi_fr_w = np.zeros((bgr_img.shape[0], bgr_img.shape[1], 4))
        multi_fr_w[..., 0] = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY);
        multi_fr_w[..., 1] = multi_fr_w[..., 0]
        multi_fr_w[..., 2] = multi_fr_w[..., 0]
        multi_fr_w[..., 3] = multi_fr_w[..., 0]

    # time1 = time.time()
    # print('read', time1-time0)

    # crop tightly
    bgr_img0 = bgr_img;
    bbox = get_bbox(rcnn, R=bgr_img0.shape[0], C=bgr_img0.shape[1])

    crop_list = [bgr_img, bg_im0, rcnn, back_img10, back_img20, multi_fr_w, bg_mask]
    crop_list = crop_images(crop_list, reso, bbox)
    bgr_img = crop_list[0];
    bg_im = crop_list[1];
    rcnn = crop_list[2];
    soft_seg = rcnn
    back_img1 = crop_list[3];
    back_img2 = crop_list[4];
    multi_fr = crop_list[5]
    bg_mask = crop_list[6]

    # time2=time.time()
    # print('crop', time2-time1)

    # process segmentation mask
    kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rcnn = rcnn.astype(np.float32) / 255;
    rcnn[rcnn > 0.2] = 1;
    K = 25

    zero_id = np.nonzero(np.sum(rcnn, axis=1) == 0)
    del_id = zero_id[0][zero_id[0] > 250]
    if len(del_id) > 0:
        del_id = [del_id[0] - 2, del_id[0] - 1, *del_id]
        rcnn = np.delete(rcnn, del_id, 0)
    rcnn = cv2.copyMakeBorder(rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

    rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
    rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
    rcnn = cv2.GaussianBlur(rcnn.astype(np.float32), (31, 31), 0)
    rcnn = (255 * rcnn).astype(np.uint8)
    rcnn = np.delete(rcnn, range(reso[0], reso[0] + K), 0)

    # time3 = time.time()
    # print('process segmentation mask', time3-time2)

    # convert to torch
    img = torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0);
    img = 2 * img.float().div(255) - 1
    bg = torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0);
    bg = 2 * bg.float().div(255) - 1
    rcnn_al = torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0);
    rcnn_al = 2 * rcnn_al.float().div(255) - 1
    multi_fr = torch.from_numpy(multi_fr.transpose((2, 0, 1))).unsqueeze(0);
    multi_fr = 2 * multi_fr.float().div(255) - 1

    # time4 = time.time()
    # print('convert to torch', time4 - time3)

    with torch.no_grad():

        img, bg, rcnn_al, multi_fr = Variable(img.cuda()), Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(
            multi_fr.cuda())
        input_im = torch.cat([img, bg, rcnn_al, multi_fr], dim=1)
        time_bf_infer = time.time()
        alpha_pred, fg_pred_tmp = netM(img, bg, rcnn_al, multi_fr)

        # time5 = time.time()
        # print('infer', time5 - time_bf_infer)
        # del time5,time_bf_infer
        # if i==0:
        #     from thop import profile
        #     flops, params = profile(netM.module, inputs=(img, bg, rcnn_al, multi_fr))
        #     print("%.2fG" % (flops/1e9), "%.2fM" % (params/1e6))
        #     torch.onnx.export(netM.module, (img, bg, rcnn_al, multi_fr),'./model.onnx', opset_version=11)

        al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)

        # for regions with alpha>0.95, simply use the image as fg
        fg_pred = img * al_mask + fg_pred_tmp * (1 - al_mask)
        # time51 = time.time()
        alpha_out = to_image(alpha_pred[0, ...]);
        # time52 = time.time()
        # print('debug1', time52 - time51)
        # refine alpha with connected component
        labels = label((alpha_out > 0.05).astype(int))
        try:
            assert (labels.max() != 0)
        except:
            continue
        # time53 = time.time()
        # print('debug2', time53 - time52)
        # del time52, time51
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        alpha_out = alpha_out * largestCC

        alpha_out = (255 * alpha_out[..., 0]).astype(np.uint8)

        alpha_out = modify_alpha(alpha_out, bg_mask=bg_mask, soft_seg=soft_seg)
        # alpha_out = modify_alpha(alpha_out, data_path=data_path, filename=filename)

        fg_out = to_image(fg_pred[0, ...]);
        fg_out = fg_out * np.expand_dims((alpha_out.astype(float) / 255 > 0.01).astype(float), axis=2);
        fg_out = (255 * fg_out).astype(np.uint8)

        # Uncrop
        R0 = bgr_img0.shape[0];
        C0 = bgr_img0.shape[1]
        alpha_out0 = uncrop(alpha_out, bbox, R0, C0)
        fg_out0 = uncrop(fg_out, bbox, R0, C0)

    # alpha_out0 = modify_alpha_wbg(alpha_out0, data_path=data_path, filename=filename)
    # time_bf_compose=time.time()
    # compose
    back_img10 = cv2.resize(back_img10, (C0, R0));
    back_img20 = cv2.resize(back_img20, (C0, R0))
    comp_im_tr1 = composite4(fg_out0, back_img10, alpha_out0)
    comp_im_tr2 = composite4(fg_out0, back_img20, alpha_out0)

    time6 = time.time()
    # print('compose', time6 - time_bf_compose)
    # print('后处理', time6 - time5)

    cv2.imwrite(result_path + '/' + filename.replace('_img', '_out'), alpha_out0)
    # cv2.imwrite(result_path + '/' + filename.replace('_img', '_fg'), cv2.cvtColor(fg_out0, cv2.COLOR_BGR2RGB))
    cv2.imwrite(result_path + '/' + filename.replace('_img', '_compose'), cv2.cvtColor(comp_im_tr1, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(result_path + '/' + filename.replace('_img', '_matte').format(i),
    #             cv2.cvtColor(comp_im_tr2, cv2.COLOR_BGR2RGB))

    print("total time:", time6 - time0)
    print('Done: ' + str(i + 1) + '/' + str(len(test_imgs)))
