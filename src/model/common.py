import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ModuleGrouper(nn.Module):

    def __init__(self, orig_module, num_types=9, keep_mask=False, scaling=1):
        """

        :param orig_module: the module to be deep copied and paralleled
        :param num_types: number of copies
        :param keep_mask: if true, return both module output and the mask (for the convenience of nn.Sequential)
        """
        super(ModuleGrouper, self).__init__()
        self.num_types = num_types
        self.keep_mask = keep_mask
        self.modulelist = nn.ModuleList([copy.deepcopy(orig_module) for _ in range(self.num_types)])
        self.scale = scaling
        #print(self.modulelist[1].state_dict().keys())

    def forward(self,args):
        """

        :param x: input tensor
        :param mask: binary mask for determining which one of the module output to use
        :return:
        """
        x=args[0]
        mask=args[1]
        #print('args shape', args)
        #print('x shape', x.shape, self.keep_mask)
        #print('mask shape',mask.shape)
        B, C, H, W = mask.shape
        mask_viewed = mask.unsqueeze(3).unsqueeze(5).repeat(1,1,1,self.scale,1,self.scale).view(B, C, H*self.scale, W*self.scale)
        out_0 = self.modulelist[0].forward(x)
        #print(out_0.shape)
        #print(mask.shape)
        out_sum = out_0*(mask_viewed[:,0,:,:].unsqueeze(1))
        for i in range(self.num_types):
            if i>0:
                out_sum = out_sum + (self.modulelist[i].forward(x))*(mask_viewed[:,i,:,:].unsqueeze(1))
        if self.keep_mask:
            #print('returning shapes: ',out_sum.shape, mask.shape)
            return [out_sum, mask]
        else:
            return out_sum


