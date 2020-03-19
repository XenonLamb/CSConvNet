import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init

def make_model(args, parent=False):
    return DnCNN(args)

class DnCNN(nn.Module):
    def __init__(self, args, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        self.args = args
        self.reinit = args.reinit
        if self.args.use_real:
            self.input_channels = self.args.n_colors+3
        else:
            self.input_channels = self.args.n_colors
        layers.append(nn.Conv2d(in_channels=self.input_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x, mask=None):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

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