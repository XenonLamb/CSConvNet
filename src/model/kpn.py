import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch
from torch import nn
import math, re, functools
import torch.nn.functional as F
import numpy as np
import copy
from torch.autograd import Variable


def make_model(args, parent=False):
    return KPN(args)

class LocalConv2d_No(nn.Module):

    def __init__(self, in_channels=1, out_channels=1,
                 kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(LocalConv2d_No, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        self.dilation = dilation
        self.bias = bias

    def forward(self, input, w_gen):
        '''

        Loc Conv2d implementation, naive version

        inputs:
            input: N, Ci, H, W
            w_gen: N, Co*(Ci*k*k + 1), H, W
        returns
            out: N, Co, H, W
        '''
        n, c, h, w = input.shape
        if self.kernel_size == 1:
            input_cat = input.view(n, self.kernel_size ** 2, c, h, w).contiguous()  # N, kk, Cin, H, W
            input_cat = input_cat.permute(0, 2, 1, 3, 4).contiguous().view(n, 1, c, -1, h,
                                                                           w).contiguous()  # N, Cin, kk, H, W --> N,1,Cin,kk,H,W
            if self.bias == True:

                cout = w_gen.shape[1] // (c * self.kernel_size ** 2 + 1)  # Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  # N,Co, Cin*kk+1, H, W
                b_gen = w_gen[:, :, -1, :, :]  #
                w_gen = w_gen[:, :, :-1, :, :].view(n, cout, c, -1, h, w).contiguous()  # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2)) + b_gen).contiguous()
            else:
                cout = w_gen.shape[1] // (c * self.kernel_size ** 2)  # Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  # N,Co, Cin*kk, H, W
                w_gen = w_gen[:, :, :, :, :].view(n, cout, c, -1, h, w).contiguous()  # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2))).contiguous()
        else:
            # print('Error: kernel size !=1 or 3 or 5')
            input_allpad = F.pad(input, (self.padding, self.padding, self.padding, self.padding),
                                 mode='reflect').contiguous()
            input_im2cl_list = []
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    input_im2cl_list.append(input_allpad[:, :, i:(h + i), j:(w + j)].contiguous())
            input_cat = torch.cat(input_im2cl_list, 1)
            input_cat = input_cat.view(n, self.kernel_size ** 2, c, h, w).contiguous()  # N, kk, Cin, H, W
            input_cat = input_cat.permute(0, 2, 1, 3, 4).contiguous().view(n, 1, c, -1, h,
                                                                           w).contiguous()  # N, Cin, kk, H, W --> N,1,Cin,kk,H,

            if self.bias == True:
                cout = w_gen.shape[1] // (c * self.kernel_size ** 2 + 1)  # Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  # N,Co, Cin*kk+1, H, W
                b_gen = w_gen[:, :, -1, :, :]  #
                w_gen = w_gen[:, :, :-1, :, :].view(n, cout, c, -1, h, w).contiguous()  # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2)) + b_gen).contiguous()
            else:
                cout = w_gen.shape[1] // (c * self.kernel_size ** 2)  # Co*(Cin*kk +1)/(Cin*kk) = Co
                w_gen = w_gen.view(n, cout, -1, h, w).contiguous()  # N,Co, Cin*kk, H, W
                w_gen = w_gen[:, :, :, :, :].view(n, cout, c, -1, h, w).contiguous()  # N,Co, Cin, kk, H, W
                out = (((input_cat * w_gen).sum(3).sum(2))).contiguous()

        return out


class ResidueBlock(nn.Module):
    def __init__(self, filterChannels, kernelSize):
        super(ResidueBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1)
        self.conv2 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += identity
        return out


""" using the args
class KPN(nn.Module):
    def __init__(self, args):
        super(KPN, self).__init__()
        self.args = args
        self.sourceEncoder = nn.Sequential(
            nn.Conv2d(1, args.filterChannels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(args.filterChannels, args.filterChannels, 3, padding=1)
        )

        featureExtractorList = []
        for _ in range(args.numBlocks):
            featureExtractorList.append(
                ResidueBlock(args.filterChannels, 3)
            )
        self.featureExtractor=nn.Sequential(*featureExtractorList)

        finalFilterChannels = args.KPNKernelSize ** 2
        self.kernelPredictor = nn.Sequential(
                nn.Conv2d(args.filterChannels, finalFilterChannels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(finalFilterChannels, finalFilterChannels, 1)
        )

        self.localConv = LocalConv2d_No(
            in_channels=1,
            out_channels=1,
            kernel_size=args.KPNKernelSize,
            bias=False
        )


    def forward(self, x):
        features = self.sourceEncoder(x)
        features = self.featureExtractor(features)
        kernels = self.kernelPredictor(features)
        kernels = F.softmax(kernels, dim=1)
        out = self.localConv.forward(x,kernels)
        return out
"""


class KPN(nn.Module):
    def __init__(self,args, color=False, burst_length=1, blind_est=False, sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN, self).__init__()
        self.args = args
        self.reinit = args.reinit
        kernel_size = args.kernel_size
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1)
        out_channel = (3 if color else 1) * (2 * kernel_size if sep_conv else kernel_size ** 2) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred =  LocalConv2d_No(
            in_channels=1,
            out_channels=1,
            kernel_size=args.kernel_size,
            bias=False
        )

        self.apply(self._init_weights)

    def forward(self, data, buckets=None, idx_scale=0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        if self.args.debug:
            print('kpn forwarding: ', data.shape)
        conv1 = self.conv1(data)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))
        res = self.kernel_pred.forward(data, core)
        if self.args.debug:
            print('kpn forwarding finished: ', data.shape, res.shape, core.shape)
        return res

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    def load_state_dict(self, state_dict, strict=True, reinit=False):
        if not self.reinit:
            own_state = self.state_dict()
            #if self.args.model == 'dncnn':

            ban_names = ['layers'] if reinit else []
            for name, param in state_dict.items():
                print("trying to load: ", name)
                if ((name in own_state)or(name.replace('module.','') in own_state)) and (not any(banname in name for banname in ban_names)):
                    if isinstance(param, nn.Parameter):
                        param = param.data

                    try:
                        if self.args.model == 'dncnn':
                            name = name.replace('module.','')
                        own_state[name].copy_(param)
                        print("sucessfully loaded", name)
                    except Exception:
                        if name.find('tail') == -1:
                            raise RuntimeError('While copying the parameter named {}, '
                                               'whose dimensions in the model are {} and '
                                               'whose dimensions in the checkpoint are {}.'
                                               .format(name, own_state[name].size(), param.size()))
                elif strict:
                    if name.find('tail') == -1:
                        raise KeyError('unexpected key "{}" in state_dict'
                                       .format(name))



# KPN基本网路单元
class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm


