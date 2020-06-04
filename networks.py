import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class ResnetConditionHR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, nf_part=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks1=7, n_blocks2=3, padding_type='reflect'):
        assert (n_blocks1 >= 0);
        assert (n_blocks2 >= 0)
        super(ResnetConditionHR, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True

        # main encoder output 256xW/4xH/4
        model_enc1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[0], ngf, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
        model_enc1 += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2),
                       nn.ReLU(True)]
        model_enc2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4), nn.ReLU(True)]

        # back encoder output 256xW/4xH/4
        model_enc_back = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[1], ngf, kernel_size=7, padding=0, bias=use_bias),
                          norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_back += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # seg encoder output 256xW/4xH/4
        model_enc_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[2], ngf, kernel_size=7, padding=0, bias=use_bias),
                         norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_seg += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling

        # #motion encoder output 256xW/4xH/4
        model_enc_multi = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[3], ngf, kernel_size=7, padding=0, bias=use_bias),
                           norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_multi += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2), nn.ReLU(True)]

        self.model_enc1 = nn.Sequential(*model_enc1)
        self.model_enc2 = nn.Sequential(*model_enc2)
        self.model_enc_back = nn.Sequential(*model_enc_back)
        self.model_enc_seg = nn.Sequential(*model_enc_seg)
        self.model_enc_multi = nn.Sequential(*model_enc_multi)

        mult = 2 ** n_downsampling
        self.comb_back = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_seg = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_multi = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))

        # decoder
        model_res_dec = [
            nn.Conv2d(ngf * mult + 3 * nf_part, ngf * mult, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(ngf * mult), nn.ReLU(True)]
        for i in range(n_blocks1):
            model_res_dec += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_res_dec_al = []
        for i in range(n_blocks2):
            model_res_dec_al += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_res_dec_fg = []
        for i in range(n_blocks2):
            model_res_dec_fg += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_dec_al = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # model_dec_al += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]
            model_dec_al += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, stride=1, padding=1),
                             norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model_dec_al += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]

        model_dec_fg1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, int(ngf * 2), 3, stride=1, padding=1), norm_layer(int(ngf * 2)),
                         nn.ReLU(True)]
        model_dec_fg2 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, ngf, 3, stride=1, padding=1), norm_layer(ngf), nn.ReLU(True),
                         nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc - 1, kernel_size=7, padding=0)]

        self.model_res_dec = nn.Sequential(*model_res_dec)
        self.model_res_dec_al = nn.Sequential(*model_res_dec_al)
        self.model_res_dec_fg = nn.Sequential(*model_res_dec_fg)
        self.model_al_out = nn.Sequential(*model_dec_al)

        self.model_dec_fg1 = nn.Sequential(*model_dec_fg1)
        self.model_fg_out = nn.Sequential(*model_dec_fg2)

    def forward(self, image, back, seg, multi):
        img_feat1 = self.model_enc1(image)
        img_feat = self.model_enc2(img_feat1)

        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)
        # multi_feat=self.model_enc_multi(multi)

        oth_feat = torch.cat([self.comb_back(torch.cat([img_feat, back_feat], dim=1)),
                              self.comb_seg(torch.cat([img_feat, seg_feat], dim=1)),
                              self.comb_multi(torch.cat([img_feat, back_feat], dim=1))], dim=1)

        out_dec = self.model_res_dec(torch.cat([img_feat, oth_feat], dim=1))

        out_dec_al = self.model_res_dec_al(out_dec)
        al_out = self.model_al_out(out_dec_al)

        out_dec_fg = self.model_res_dec_fg(out_dec)
        out_dec_fg1 = self.model_dec_fg1(out_dec_fg)
        fg_out = self.model_fg_out(torch.cat([out_dec_fg1, img_feat1], dim=1))

        return al_out, fg_out


class ResnetConditionHR_mo(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, nf_part=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks1=7, n_blocks2=3, padding_type='reflect'):  # padding_type='reflect', 'zero'
        assert (n_blocks1 >= 0);
        assert (n_blocks2 >= 0)
        super(ResnetConditionHR_mo, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True

        # main encoder output 256xW/4xH/4
        model_enc1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[0], ngf, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
        model_enc1 += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2),
                       nn.ReLU(True)]
        model_enc2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4), nn.ReLU(True)]

        # back encoder output 256xW/4xH/4
        model_enc_back = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[1], ngf, kernel_size=7, padding=0, bias=use_bias),
                          norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_back += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # seg encoder output 256xW/4xH/4
        model_enc_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[2], ngf, kernel_size=7, padding=0, bias=use_bias),
                         norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_seg += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling

        # #motion encoder output 256xW/4xH/4
        model_enc_multi = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[3], ngf, kernel_size=7, padding=0, bias=use_bias),
                           norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_multi += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2), nn.ReLU(True)]

        self.model_enc1 = nn.Sequential(*model_enc1)
        self.model_enc2 = nn.Sequential(*model_enc2)
        self.model_enc_back = nn.Sequential(*model_enc_back)
        self.model_enc_seg = nn.Sequential(*model_enc_seg)
        self.model_enc_multi = nn.Sequential(*model_enc_multi)

        mult = 2 ** n_downsampling
        self.comb_back = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_seg = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_multi = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))

        # decoder
        model_res_dec = [
            nn.Conv2d(ngf * mult + 2 * nf_part, ngf * mult, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(ngf * mult), nn.ReLU(True)]
        for i in range(n_blocks1):
            model_res_dec += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_res_dec_al = []
        for i in range(n_blocks2):
            model_res_dec_al += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_res_dec_fg = []
        for i in range(n_blocks2):
            model_res_dec_fg += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_dec_al = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # model_dec_al += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]
            model_dec_al += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, stride=1, padding=1),
                             norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model_dec_al += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]

        model_dec_fg1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, int(ngf * 2), 3, stride=1, padding=1), norm_layer(int(ngf * 2)),
                         nn.ReLU(True)]
        model_dec_fg2 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, ngf, 3, stride=1, padding=1), norm_layer(ngf), nn.ReLU(True),
                         nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc - 1, kernel_size=7, padding=0)]

        self.model_res_dec = nn.Sequential(*model_res_dec)
        self.model_res_dec_al = nn.Sequential(*model_res_dec_al)
        self.model_res_dec_fg = nn.Sequential(*model_res_dec_fg)
        self.model_al_out = nn.Sequential(*model_dec_al)

        self.model_dec_fg1 = nn.Sequential(*model_dec_fg1)
        self.model_fg_out = nn.Sequential(*model_dec_fg2)

    def forward(self, image, back, seg, multi):
        img_feat1 = self.model_enc1(image)
        img_feat = self.model_enc2(img_feat1)

        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)

        oth_feat = torch.cat([self.comb_back(torch.cat([img_feat, back_feat], dim=1)),
                              self.comb_seg(torch.cat([img_feat, seg_feat], dim=1))], dim=1)

        out_dec = self.model_res_dec(torch.cat([img_feat, oth_feat], dim=1))

        out_dec_al = self.model_res_dec_al(out_dec)
        al_out = self.model_al_out(out_dec_al)

        # 去除fg-color预测
        # out_dec_fg = self.model_res_dec_fg(out_dec)
        # out_dec_fg1 = self.model_dec_fg1(out_dec_fg)
        # fg_out = self.model_fg_out(torch.cat([out_dec_fg1, img_feat1], dim=1))

        return al_out


class ResnetConditionHR_mo_4convert(nn.Module):
    '''
    专门为转模型测试用，把upsample换成反卷积。把reflection padding换成zeropadding。把Tanh去掉。
    '''

    def __init__(self, input_nc, output_nc, ngf=64, nf_part=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks1=7, n_blocks2=3, padding_type='zero'):  # padding_type='reflect', 'zero'
        assert (n_blocks1 >= 0);
        assert (n_blocks2 >= 0)
        super(ResnetConditionHR_mo_4convert, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True

        # main encoder output 256xW/4xH/4
        model_enc1 = [nn.Conv2d(input_nc[0], ngf, kernel_size=7, padding=3, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
        model_enc1 += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 2),
                       nn.ReLU(True)]
        model_enc2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4), nn.ReLU(True)]

        # back encoder output 256xW/4xH/4
        model_enc_back = [nn.Conv2d(input_nc[1], ngf, kernel_size=7, padding=3, bias=use_bias),
                          norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_back += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # seg encoder output 256xW/4xH/4
        model_enc_seg = [nn.Conv2d(input_nc[2], ngf, kernel_size=7, padding=3, bias=use_bias),
                         norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_seg += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling

        # #motion encoder output 256xW/4xH/4
        model_enc_multi = [nn.Conv2d(input_nc[3], ngf, kernel_size=7, padding=3, bias=use_bias),
                           norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model_enc_multi += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2), nn.ReLU(True)]

        self.model_enc1 = nn.Sequential(*model_enc1)
        self.model_enc2 = nn.Sequential(*model_enc2)
        self.model_enc_back = nn.Sequential(*model_enc_back)
        self.model_enc_seg = nn.Sequential(*model_enc_seg)
        self.model_enc_multi = nn.Sequential(*model_enc_multi)

        mult = 2 ** n_downsampling
        self.comb_back = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_seg = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_multi = nn.Sequential(
            nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))

        # decoder
        model_res_dec = [
            nn.Conv2d(ngf * mult + 2 * nf_part, ngf * mult, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(ngf * mult), nn.ReLU(True)]
        for i in range(n_blocks1):
            model_res_dec += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_res_dec_al = []
        for i in range(n_blocks2):
            model_res_dec_al += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_res_dec_fg = []
        for i in range(n_blocks2):
            model_res_dec_fg += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        model_dec_al = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_dec_al += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                                output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)),
                             nn.ReLU(True)]
            # model_dec_al += [f_upsample(scale_factor=2, mode='bilinear', align_corners=None),
            #                  nn.Conv2d(ngf * mult, int(ngf * mult / 2), 3, stride=1, padding=1),
            #                  norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model_dec_al += [nn.Conv2d(ngf, 1, kernel_size=7, padding=3)]  # nn.Tanh()

        model_dec_fg1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, int(ngf * 2), 3, stride=1, padding=1), norm_layer(int(ngf * 2)),
                         nn.ReLU(True)]
        model_dec_fg2 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, ngf, 3, stride=1, padding=1), norm_layer(ngf), nn.ReLU(True),
                         nn.Conv2d(ngf, output_nc - 1, kernel_size=7, padding=3)]

        self.model_res_dec = nn.Sequential(*model_res_dec)
        self.model_res_dec_al = nn.Sequential(*model_res_dec_al)
        self.model_res_dec_fg = nn.Sequential(*model_res_dec_fg)
        self.model_al_out = nn.Sequential(*model_dec_al)

        self.model_dec_fg1 = nn.Sequential(*model_dec_fg1)
        self.model_fg_out = nn.Sequential(*model_dec_fg2)

    def forward(self, image, back, seg):
        img_feat1 = self.model_enc1(image)
        img_feat = self.model_enc2(img_feat1)

        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)

        oth_feat = torch.cat([self.comb_back(torch.cat([img_feat, back_feat], dim=1)),
                              self.comb_seg(torch.cat([img_feat, seg_feat], dim=1))], dim=1)

        out_dec = self.model_res_dec(torch.cat([img_feat, oth_feat], dim=1))

        out_dec_al = self.model_res_dec_al(out_dec)
        al_out = self.model_al_out(out_dec_al)

        # 去除fg-color预测
        # out_dec_fg = self.model_res_dec_fg(out_dec)
        # out_dec_fg1 = self.model_dec_fg1(out_dec_fg)
        # fg_out = self.model_fg_out(torch.cat([out_dec_fg1, img_feat1], dim=1))

        return al_out


class UnetBased_mo(nn.Module):
    '''
    专门为转模型测试用，把upsample换成反卷积。把reflection padding换成zeropadding。把Tanh去掉。
    '''

    def __init__(self, input_nc, output_nc, ngf=64, nf_part=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks1=7, n_blocks2=3, padding_type='zero'):  # padding_type='reflect', 'zero'
        assert (n_blocks1 >= 0);
        assert (n_blocks2 >= 0)
        super(UnetBased_mo, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.inplanes = 64
        block = BasicBlock
        layers = [2, 2, 2, 2]
        out_channels = [128, 64, 32, 16]
        resnet_ch = [64,128,256,320]
        channel_according_dowmsample = [64] + resnet_ch
        max_downsample_separate_feat = 2
        in_channels = self.compute_channels(out_channels,resnet_ch)
        separate_feat_ch = channel_according_dowmsample[max_downsample_separate_feat - 1]
        use_bias = True

        # main encoder output 64/2xH/2
        model_enc1 = [nn.Conv2d(input_nc[0], ngf, kernel_size=7, stride=2, padding=3, bias=False),
                      norm_layer(ngf), nn.ReLU(True)]

        # back encoder output 64xW/2xH/2
        model_enc_back = [nn.Conv2d(input_nc[1], ngf, kernel_size=7, stride=2, padding=3, bias=False),
                          norm_layer(ngf), nn.ReLU(True)]

        # seg encoder output 64xW/2xH/2
        model_enc_seg = [nn.Conv2d(input_nc[2], ngf, kernel_size=7, stride=2, padding=3, bias=False),
                         norm_layer(ngf), nn.ReLU(True)]

        self.layer1 = self._make_layer(block, resnet_ch[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, resnet_ch[1], layers[1], stride=2)
        self.layer2s1 = self._make_layer(block, resnet_ch[1], layers[1], stride=1)
        self.layer3 = self._make_layer(block, resnet_ch[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, resnet_ch[3], layers[3], stride=2)
        self.inplanes = separate_feat_ch + 2 * nf_part
        self.layer2s1_comb = self._make_layer(block, resnet_ch[1], layers[1], stride=1)

        # model_enc1 = model_enc1 + layer1 + layer2
        # model_enc_back = model_enc_back + layer1 + layer2
        # model_enc_seg = model_enc_seg + layer1 + layer2

        self.model_enc1 = nn.Sequential(*model_enc1)
        self.model_enc_back = nn.Sequential(*model_enc_back)
        self.model_enc_seg = nn.Sequential(*model_enc_seg)

        # mult = 2 ** n_downsampling
        mult = 2
        # n_downsampling = 3
        self.comb_back = nn.Sequential(
            nn.Conv2d(separate_feat_ch * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))
        self.comb_seg = nn.Sequential(
            nn.Conv2d(separate_feat_ch * 2, nf_part, kernel_size=1, stride=1, padding=0, bias=False), norm_layer(ngf),
            nn.ReLU(True))

        # decoder
        self.up_layer1 = DecoderBlock(in_channels[0], out_channels[0])
        self.up_layer2 = DecoderBlock(in_channels[1], out_channels[1])
        self.up_layer3 = DecoderBlock(in_channels[2], out_channels[2])
        self.up_layer4 = DecoderBlock(in_channels[3], out_channels[3])

        model_res_dec = [
            nn.Conv2d(separate_feat_ch + 2 * nf_part, ngf * mult, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(ngf * mult), nn.ReLU(True)]
        # for i in range(n_blocks1):
        #     model_res_dec += [
        #         ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
        #                     use_bias=use_bias)]
        #
        # model_res_dec_al = []
        # for i in range(n_blocks2):
        #     model_res_dec_al += [
        #         ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
        #                     use_bias=use_bias)]
        #
        # model_res_dec_fg = []
        # for i in range(n_blocks2):
        #     model_res_dec_fg += [
        #         ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
        #                     use_bias=use_bias)]

        model_dec_al = []
        model_dec_al += [nn.Conv2d(out_channels[3], 1, kernel_size=3, padding=1), nn.Tanh()]  # nn.Tanh()

        model_dec_fg1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, int(ngf * 2), 3, stride=1, padding=1), norm_layer(int(ngf * 2)),
                         nn.ReLU(True)]
        model_dec_fg2 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                         nn.Conv2d(ngf * 4, ngf, 3, stride=1, padding=1), norm_layer(ngf), nn.ReLU(True),
                         nn.Conv2d(ngf, output_nc - 1, kernel_size=3, padding=1)]

        self.model_res_dec = nn.Sequential(*model_res_dec)
        # self.model_res_dec_al = nn.Sequential(*model_res_dec_al)
        # self.model_res_dec_fg = nn.Sequential(*model_res_dec_fg)
        self.model_al_out = nn.Sequential(*model_dec_al)

        self.model_dec_fg1 = nn.Sequential(*model_dec_fg1)
        self.model_fg_out = nn.Sequential(*model_dec_fg2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def compute_channels(self, decoder_channels, resnet_ch):
        channels=[
            resnet_ch[3] + resnet_ch[2],
            decoder_channels[0]+resnet_ch[1],
            decoder_channels[1]+resnet_ch[0],
            decoder_channels[2]+0  # 最后一层不用concat，∴加0。
        ]
        comb_idx=2
        channels[comb_idx] = decoder_channels[comb_idx-1]+64*3
        return channels


    def forward(self, image, back, seg):
        img_feat = self.model_enc1(image)
        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)

        img_feat1 = self.layer1(img_feat)
        # img_feat2 = self.layer2(img_feat)
        back_feat1 = self.layer1(back_feat)
        # back_feat2 = self.layer2(back_feat)
        seg_feat1 = self.layer1(seg_feat)
        # seg_feat2 = self.layer2(seg_feat)

        oth_feat = torch.cat([img_feat1, self.comb_back(torch.cat([img_feat1, back_feat1], dim=1)),
                              self.comb_seg(torch.cat([img_feat1, seg_feat1], dim=1))], dim=1)

        out_en1 = self.layer2s1_comb(oth_feat)
        # out_en1 = self.layer2s1(oth_feat)
        out_en2 = self.layer3(out_en1)
        out_en3 = self.layer4(out_en2)

        out_de1 = self.up_layer1([out_en3, out_en2])
        out_de2 = self.up_layer2([out_de1, out_en1])
        out_de3 = self.up_layer3([out_de2, torch.cat([img_feat, back_feat, seg_feat], dim=1)])
        out_de4 = self.up_layer4([out_de3, None])


        # out_dec_al = self.model_res_dec_al(out_dec)
        al_out = self.model_al_out(out_de4)

        # 去除fg-color预测
        # out_dec_fg = self.model_res_dec_fg(out_dec)
        # out_dec_fg1 = self.model_dec_fg1(out_dec_fg)
        # fg_out = self.model_fg_out(torch.cat([out_dec_fg1, img_feat1], dim=1))

        return al_out


############################## part ##################################

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        return x


def conv3x3_(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        # init.normal(m.weight)
        if m.bias is not None:
            init.constant(m.bias, 0)

    if classname.find('Linear') != -1:
        init.normal(m.weight)
        init.constant(m.bias, 1)

    if classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.2)
        init.constant(m.bias.data, 0.0)


class f_upsample(nn.Module):
    def __init__(self, scale_factor, mode="bilinear", align_corners=None):
        super(f_upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class conv3x3(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3s1(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3s1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv1x1(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upconv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upconv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fc, self).__init__()
        self.fullc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fullc(x)
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


##################################### Discriminators ####################################################

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
