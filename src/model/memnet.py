import torch
import torch.nn as nn

import torch
import math, re, functools
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import copy
from torch.autograd import Variable

def make_model(args, parent=False):
    return MemNet(args)


class MemNet(nn.Module):
    def __init__(self,args, num_memblock=6, num_resblock=6):
        super(MemNet, self).__init__()
        self.args = args
        self.reinit = args.reinit
        in_channels = args.n_colors
        channels = args.n_feats
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1) for i in range(num_memblock)]
        )

    def forward(self, x, buckets=None):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual

        return out

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


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock + num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.gate_unit(torch.cat(xs + ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))