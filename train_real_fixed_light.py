from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
import time
import argparse
import numpy as np

from data_loader import CompositeData, RealDataWoMotion, select_datatype, RealDataAndAdobe
from functions import *
from networks import ResnetConditionHR, MultiscaleDiscriminator, conv_init, ResnetConditionHR_mo
from loss_functions import alpha_loss, compose_loss, alpha_gradient_loss, GANloss

# CUDA

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])

"""Parses arguments."""
parser = argparse.ArgumentParser(description='Training Background Matting on Adobe Dataset.')
parser.add_argument('-n', '--name', type=str, help='Name of tensorboard and model saving folders.')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch Size.')
parser.add_argument('-res', '--reso', type=int, help='Input image resolution')
parser.add_argument('-init_model', '--init_model', type=str, help='Initial model file')

parser.add_argument('-epoch', '--epoch', type=int, default=35, help='Maximum Epoch')
parser.add_argument('-n_blocks1_t', '--n_blocks1_t', type=int, default=7,
                    help='in teacher model, Number of residual blocks after Context Switching.')
parser.add_argument('-n_blocks2_t', '--n_blocks2_t', type=int, default=3,
                    help='in teacher model, Number of residual blocks for Fg and alpha each.')
parser.add_argument('-n_blocks1_s', '--n_blocks1_s', type=int, default=3,
                    help='in student model, Number of residual blocks after Context Switching.')
parser.add_argument('-n_blocks2_s', '--n_blocks2_s', type=int, default=1,
                    help='in student model, Number of residual blocks for Fg and alpha each.')

args = parser.parse_args()

debug = False
fg_color_pred = False
data_real_and_adobe = True

##Directories
localtime = time.asctime(time.localtime(time.time()))
tb_dir = os.path.join('TB_Summary/', args.name + '_' + localtime)
model_dir = os.path.join('Models/', args.name)
result_dir = os.path.join('result', args.name)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

## Input list
data_config_train = {'reso': (args.reso, args.reso)}  # if trimap is true, rcnn is used

# DATA LOADING
print('\n[Phase 1] : Data Preparation')


def collate_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Original Data
if data_real_and_adobe:
    traindata = RealDataAndAdobe(csv_file='tool/rgbd1.csv', data_config=data_config_train,
                                 transform=None)  # Write a dataloader function that can read the database provided by .csv file
else:
    traindata = RealDataWoMotion(csv_file='tool/rgbd.csv', data_config=data_config_train,
                                 transform=None)  # Write a dataloader function that can read the database provided by .csv file
train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.batch_size, collate_fn=collate_filter_none)

print('\n[Phase 2] : Initialization')

netB = ResnetConditionHR(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=args.n_blocks1_t, n_blocks2=args.n_blocks2_t)
netB = nn.DataParallel(netB)
netB.load_state_dict(torch.load(args.init_model))
netB.cuda();
netB.eval()
for param in netB.parameters():  # freeze netB
    param.requires_grad = False

netG = ResnetConditionHR_mo(input_nc=(3, 3, 1, 4), output_nc=4, n_blocks1=args.n_blocks1_s, n_blocks2=args.n_blocks2_s)
netG.apply(conv_init)
netG = nn.DataParallel(netG)
netG.cuda()
torch.backends.cudnn.benchmark = True

netD = MultiscaleDiscriminator(input_nc=3, num_D=1, norm_layer=nn.InstanceNorm2d, ndf=64)
netD.apply(conv_init)
netD = nn.DataParallel(netD)
netD.cuda()

# Loss
l1_loss = alpha_loss()
c_loss = compose_loss()
g_loss = alpha_gradient_loss()
GAN_loss = GANloss()

optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
optimizerD = optim.Adam(netD.parameters(), lr=1e-5)

log_writer = SummaryWriter(tb_dir)

print('Starting Training')
step = 50

KK = len(train_loader)

wt = 1
for epoch in range(0, args.epoch):

    netG.train();
    netD.train()

    lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    t0 = time.time();

    for i, data in enumerate(train_loader):
        # Initiating

        bg, image, seg, seg_gt, back_rnd = data['bg'], data['image'], data['seg'], data[
            'seg-gt'], data['back-rnd']
        if data_real_and_adobe:
            fg, back_tr, data_type = data['fg'], data['back_tr'], data['data_type']
        bg, image, seg, seg_gt, back_rnd = Variable(bg.cuda()), Variable(image.cuda()), Variable(
            seg.cuda()), Variable(seg_gt.cuda()), Variable(back_rnd.cuda())

        mask0 = Variable(torch.ones(seg.shape).cuda())

        tr0 = time.time()

        # pseudo-supervision
        place_holder = bg
        alpha_pred_sup, fg_pred_sup = netB(image, bg, seg, place_holder)
        mask = (alpha_pred_sup > -0.98).type(torch.cuda.FloatTensor)

        mask1 = (seg_gt > 0.95).type(torch.cuda.FloatTensor)

        if data_real_and_adobe:
            bg4input = select_datatype(back_tr, bg, data_type)
            alpha_sup = select_datatype(seg_gt, alpha_pred_sup, data_type)
            mask4compose = select_datatype(mask0, mask1, data_type)
        else:
            bg4input = bg
            alpha_sup = alpha_pred_sup
            mask4compose = mask1
        ## Train Generator
        if not fg_color_pred:
            alpha_pred = netG(image, bg4input, seg, place_holder)
            fg_pred = image  # todo:5.23消融实验,恢复fg-pred分支。
        else:
            alpha_pred, fg_pred = netG(image, bg4input, seg, place_holder)

        ##pseudo-supervised losses
        al_loss = l1_loss(alpha_sup, alpha_pred, mask0) + 2.0 * g_loss(alpha_sup, alpha_pred, mask0)
        fg_loss = l1_loss(fg_pred_sup, fg_pred, mask)

        # compose into same background
        if data_real_and_adobe:
            fg4compose = select_datatype(fg, fg_pred, data_type)
        else:
            fg4compose = fg_pred
        comp_loss = c_loss(image, alpha_pred, fg4compose, bg, mask4compose)

        # randomly permute the background, 随机改变顺序
        perm = torch.LongTensor(np.random.permutation(bg.shape[0]))
        if data_real_and_adobe:
            bg_sh = select_datatype(bg, bg[perm, :, :, :], data_type)
        else:

            bg_sh = bg[perm, :, :, :]
            # Choose the target background for composition
            # back_rnd: contains separate set of background videos captured
            # bg_sh: contains randomly permuted captured background from the same minibatch
            if np.random.random_sample() > 0.5:
                bg_sh = back_rnd

        al_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)

        # print('debug', seg.max())
        image_sh = compose_image_withshift(alpha_pred, image * al_mask + fg_pred * (1 - al_mask), bg_sh, seg)
        # print('debug', image_sh.shape)
        if i % 50 == 0:
            print('img save')
            # image_sh[0,...].numpy()
            image_sh1 = fg4compose*alpha_pred + (1-alpha_pred)*bg
            image_sh_np = to_image(image_sh1[0, ...])
            image_sh_np = (255 * image_sh_np).astype(np.uint8)
            cv2.imwrite('{}/{}debug_image_sh.jpg'.format(result_dir, epoch),
                        cv2.cvtColor(image_sh_np, cv2.COLOR_BGR2RGB))
            image_np = to_image(image[0, ...])
            image_np = (255 * image_np).astype(np.uint8)
            cv2.imwrite('{}/{}debug_image.jpg'.format(result_dir, epoch), cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            seg_np = to_image(mask4compose[0, ...])
            seg_np = (255 * seg_np).astype(np.uint8)
            cv2.imwrite('{}/{}debug_mask4compose.jpg'.format(result_dir, epoch), seg_np)
            fg_pred_mask = mask * fg_pred_sup
            fg_pred_sup_np = to_image(fg_pred_mask[0, ...])
            fg_pred_sup_np = (255 * fg_pred_sup_np).astype(np.uint8)
            cv2.imwrite('{}/{}debug_fg_pred_sup.jpg'.format(result_dir, epoch),
                        cv2.cvtColor(fg_pred_sup_np, cv2.COLOR_BGR2RGB))

        # print(image_sh.requires_grad)
        fake_response = netD(image_sh)

        loss_ganG = GAN_loss(fake_response, label_type=True)

        if fg_color_pred:
            lossG = loss_ganG + wt * (0.05 * comp_loss + 0.05 * al_loss + 0.05 * fg_loss)
        else:
            # lossG = loss_ganG + wt * (0.05 * comp_loss + 0.05 * al_loss)
            lossG = 0.01 * loss_ganG + wt * (comp_loss + al_loss)

        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()

        ##Train Discriminator

        fake_response = netD(image_sh.detach())
        real_response = netD(image)
        # real_response = netD(fg)  # todo：5.19，当前image是合成数据，而fg是真实数据，所以用fg练判别器。

        loss_ganD_fake = GAN_loss(fake_response, label_type=False)
        loss_ganD_real = GAN_loss(real_response, label_type=True)

        lossD = (loss_ganD_real + loss_ganD_fake) * 0.5

        # Update discriminator for every 5 generator update
        if i % 5 == 0:
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()

        lG += lossG.data
        lD += lossD.data
        GenL += loss_ganG.data
        DisL_r += loss_ganD_real.data
        DisL_f += loss_ganD_fake.data

        alL += al_loss.data
        fgL += fg_loss.data
        compL += comp_loss.data

        log_writer.add_scalar('Generator Loss', lossG.data, epoch * KK + i + 1)
        log_writer.add_scalar('Discriminator Loss', lossD.data, epoch * KK + i + 1)
        log_writer.add_scalar('Generator Loss: Fake', loss_ganG.data, epoch * KK + i + 1)
        log_writer.add_scalar('Discriminator Loss: Real', loss_ganD_real.data, epoch * KK + i + 1)
        log_writer.add_scalar('Discriminator Loss: Fake', loss_ganD_fake.data, epoch * KK + i + 1)

        log_writer.add_scalar('Generator Loss: Alpha', al_loss.data, epoch * KK + i + 1)
        log_writer.add_scalar('Generator Loss: Fg', fg_loss.data, epoch * KK + i + 1)
        log_writer.add_scalar('Generator Loss: Comp', comp_loss.data, epoch * KK + i + 1)

        t1 = time.time()

        elapse += t1 - t0
        elapse_run += t1 - tr0
        t0 = t1

        if i % step == (step - 1):
            print(
                '[%d, %5d] Gen-loss:  %.4f Disc-loss: %.4f Alpha-loss: %.4f Fg-loss: %.4f Comp-loss: %.4f Time-all: %.4f Time-fwbw: %.4f' % (
                    epoch + 1, i + 1, lG / step, lD / step, alL / step, fgL / step, compL / step, elapse / step,
                    elapse_run / step))
            lG, lD, GenL, DisL_r, DisL_f, alL, fgL, compL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            if i == step:
                write_tb_log(image, 'image', log_writer, i)
                write_tb_log(seg, 'seg', log_writer, i)
                write_tb_log(alpha_pred_sup, 'alpha-sup', log_writer, i)
                write_tb_log(alpha_pred, 'alpha_pred', log_writer, i)
                write_tb_log(fg_pred_sup * mask, 'fg-pred-sup', log_writer, i)
                write_tb_log(fg_pred * mask, 'fg_pred', log_writer, i)

                # composition
                alpha_pred = (alpha_pred + 1) / 2
                comp = fg_pred * alpha_pred + (1 - alpha_pred) * bg
                write_tb_log(comp, 'composite-same', log_writer, i)
                write_tb_log(image_sh, 'composite-diff', log_writer, i)

                del comp

        del mask, back_rnd, mask0, seg_gt, mask1, bg, alpha_pred, alpha_pred_sup, image, fg_pred_sup, fg_pred, seg, image_sh, bg_sh, fake_response, real_response, al_loss, fg_loss, comp_loss, lossG, lossD, loss_ganD_real, loss_ganD_fake, loss_ganG

    if (epoch % 2 == 0) and not debug:
        torch.save(netG.state_dict(), model_dir + '/netG_epoch_%d.pth' % (epoch))
        torch.save(optimizerG.state_dict(), model_dir + '/optimG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), model_dir + '/netD_epoch_%d.pth' % (epoch))
        torch.save(optimizerD.state_dict(), model_dir + '/optimD_epoch_%d.pth' % (epoch))

        # Change weight every 2 epoch to put more stress on discriminator weight and less on pseudo-supervision
        wt = wt * 0.9
