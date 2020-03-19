
"""
Deprecated experiments, please ignore

"""

import torch
import torch.nn as nn
import model.ops as ops
from model.common import ModuleGrouper
import math


def make_model(args, parent=False):
    return Net(args)


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(64, 64)
        self.b2 = ops.ResidualBlock(64, 64)
        self.b3 = ops.ResidualBlock(64, 64)
        self.c1 = ops.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        # scale = args.scale[0]
        self.scale = args.scale[0]
        # print(self.scale)
        group = args.group
        # group = kwargs.get("group", 1)
        self.num_types = args.num_types
        self.neuf = args.neuf
        self.neufed = False
        self.part = args.part
        self.color_channels = args.n_colors
        self.n_layers = args.n_layers
        if self.n_layers ==1:
            self.first = nn.Conv2d(self.color_channels,self.color_channels,7,1,3)
        elif self.n_layers == 2:
            self.first = nn.Conv2d(self.color_channels,8,7,1,3)
            self.nonlinear = nn.PReLU()
            self.middle = nn.Conv2d(8,self.color_channels, 7, 1, 3)
        elif self.n_layers == 3:
            self.first = nn.Conv2d(self.color_channels, 16, 7, 1, 3)
            self.nonlinear = nn.PReLU()
            self.middle = nn.Conv2d(16, 16, 7, 1, 3)
            self.nonlinear2 = nn.PReLU()
            self.last = nn.Conv2d(16, self.color_channels, 7, 1, 3)
        elif self.n_layers == 5:
            self.first = nn.Conv2d(self.color_channels, 16, 5, 1, 2)
            self.nonlinear = nn.PReLU()
            self.middle = nn.Conv2d(16, 16, 3, 1, 1)
            self.nonlinear3 = nn.PReLU()
            self.third = nn.Conv2d(16, 16, 3, 1, 1)
            self.nonlinear4 = nn.PReLU()
            self.fourth = nn.Conv2d(16, 16, 3, 1, 1)
            self.nonlinear2 = nn.PReLU()
            self.last = nn.Conv2d(16, self.color_channels, 5, 1, 2)

        #if self.scale!=1:
        #    self.upsample = ops.UpsampleBlock(64, scale=self.scale,
        #                                  multi_scale=False,
        #                                  group=group)
        #self.exit = nn.Conv2d(64, 3, 3, 1, 1)
        self._initialize_weights()


    def forward(self, x, mask_bin=None, scale=2):

        #if self.neuf:
        #    out = self.body([x, mask_bin])
        #else:
        #    out = self.body(x)
        #if self.scale != 1:
        #    out = self.upsample(out, scale=self.scale)
        #out = self.exit(out)
        out = self.first(self._input_prepare([x,mask_bin]))
        if self.n_layers >1:
            out = self.nonlinear(out)
            out = self.middle(self._input_prepare([out,mask_bin]))
        if self.n_layers ==5:
            out = self.nonlinear3(out)
            out = self.third(self._input_prepare([out,mask_bin]))
            out = self.nonlinear4(out)
            out = self.fourth(self._input_prepare([out,mask_bin]))
        if self.n_layers >2:
            out = self.nonlinear2(out)
            out = self.last(self._input_prepare([out,mask_bin]))


        return out

    def _input_prepare(self, inputs):
        if self.neuf:
            return inputs
        else:
            return inputs[0]

    def load_state_dict(self, state_dict, strict=True, reinit=False):
        own_state = self.state_dict()

    def _initialize_weights(self):
        modus = [self.first]
        if self.n_layers > 1:
            modus.append(self.middle)
        if self.n_layers > 2:
            modus.append(self.last)
        for modu in modus:
            nn.init.normal_(modu.weight.data, mean=0.0,
                        std=math.sqrt(2 / (self.color_channels * modu.weight.data[0][0].numel())))
            nn.init.zeros_(modu.bias.data)


    def neufize(self):
        self.first = ModuleGrouper(self.first, self.num_types)
        if self.n_layers>1:
            self.middle = ModuleGrouper(self.middle, self.num_types)
            if self.n_layers > 2:
                self.last = ModuleGrouper(self.last, self.num_types)
                if self.n_layers > 3:

                    self.third = ModuleGrouper(self.third, self.num_types)

                    self.fourth = ModuleGrouper(self.fourth, self.num_types)

        print("neufed!")
        # print(self.state_dict().keys())
        # print(self.b1)
        # print(self.b1.state_dict().keys())
        self.neufed = True


