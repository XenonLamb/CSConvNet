import torch
import torch.nn as nn
import model.ops as ops
import torch
import math, re, functools
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import copy
from model.CSConv2D import CSConv2D

def make_model(args, parent=False):
    return REDNet30(args)

class REDNet30(nn.Module):
    def __init__(self, args, num_layers=15):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers
        num_features = args.n_feats
        self.args = args
        self.reinit = args.reinit
        self.num_types = args.Qstr * args.Qcohe * args.Qangle
        self.split = args.split
        self.split_skip = args.split_skip
        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(args.n_colors, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            if (not self.split_skip) or(i % 2 ==0):
                can_split=True
            else:
                can_split=False
            if self.split and can_split:
                conv_layers.append(BucketConvRelu(args, num_features, num_features))
            else:
                conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                                 nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, args.n_colors, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, buckets=None):
        residual = x
        buckets_down  = buckets[:,::2,::2].contiguous()
        conv_feats = []
        for i in range(self.num_layers):
            if (not self.split_skip) or((i+1) % 2 ==0):
                can_split=True
            else:
                can_split=False
            if (i==0)or(not self.split)or(not can_split):
                x = self.conv_layers[i](x)
            else:
                x = self.conv_layers[i](x, buckets_down)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x

    def load_state_dict(self, state_dict, strict=True, reinit=False):
        if not self.reinit:
            own_state = self.state_dict()
            #if self.args.model == 'dncnn':

            ban_names = ['layers'] if reinit else []
            if self.args.load_transfer:
                for lnum in range(self.num_layers-1):
                    if (not self.split_skip) or (lnum % 2 == 0):
                        can_split = True
                    else:
                        can_split = False
                    layernum = lnum+1
                    if can_split:
                        if ('conv_layers.'+str(layernum)+'.0.weight') in state_dict:
                                print('transfer loading: ',('conv_layers.'+str(layernum)+'.0.weight'))
                                conv_w = state_dict['conv_layers.'+str(layernum)+'.0.weight']
                                conv_b = state_dict['conv_layers.'+str(layernum)+'.0.bias']
                                w_reshape = conv_w.view(self.args.n_feats, -1).contiguous()
                                ww = torch.cat((w_reshape, conv_b.unsqueeze(-1)), -1).view(-1).contiguous().unsqueeze(0)\
                                    .repeat(own_state['conv_layers.'+str(layernum)+'.final.filter_emb.weight'].shape[0],1).contiguous()
                                own_state['conv_layers.' + str(layernum) + '.final.filter_emb.weight'].copy_(ww)
                                print('successfully loaded: ', ('conv_layers.' + str(layernum) + '.final.filter_emb.weight'))


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

class BucketConvRelu(nn.Module):
    def __init__(self,
                 args, in_channels, out_channels, mid_channels=None):
        super(BucketConvRelu, self).__init__()
        """
        if mid_channels is not None:
            self.mid_channels = mid_channels
        else:
            self.mid_channels = out_channels
        """
        self.args = args

        # self.final = BucketConvLayer(args, args.kernel_size, self.mid_channels, out_channels)
        self.final = BucketConvLayerCUDA(args, args.kernel_size, in_channels, out_channels)
        self.nonlinear = nn.ReLU(inplace=True)

        # init_weights(self.modules)

    def forward(self, x, buckets=None):

        out = self.final(x, buckets)
        out = self.nonlinear(out)
        return out


class BucketConvLayerCUDA(nn.Module):
    def __init__(self, args,kernel_size,in_channels, out_channels, bias=True):
        super(BucketConvLayerCUDA, self).__init__()

        self.kernel_size = args.kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = args.Qstr*args.Qcohe*args.Qangle
        self.kernel_size = kernel_size
        self.localconv = CSConv2D(self.kernel_size)
        if bias:
            self.filter_emb = nn.Embedding(self.num_types, self.out_channels * (
            (self.in_channels * self.kernel_size * self.kernel_size+1)))

        else:
            self.filter_emb = nn.Embedding(self.num_types, self.out_channels*((self.in_channels*self.kernel_size*self.kernel_size)))
        self.args = args
        self._initialize_weights()


    def _initialize_weights(self):

        nn.init.normal_(self.filter_emb.weight.data, mean=0.0,
                        std=0.1)
        #if self.in_channels ==1 and self.out_channels==1 and self.kernel_size == 7:
        #    self.load_from_raisr(self.args)

    def forward(self, x, buckets):
        """

        :param x: of shape (batch_size, input_channel, H, W)
        :param buckets: LongTensor of shape (batch_size, H, W)
        :return: out: FloatTensor of shape (batch_size, out_channel, H, W)

        """
        if self.args.debug:
            print('into resblock')
            print(x.shape, x.dtype)
            print(buckets.shape, buckets.dtype)
            print(buckets.max(),buckets.min())
        #if self.args.timer:
        #    torch.cuda.synchronize()
        #    self.args.timer_embedding_forward.tic()
        #    self.args.timer_total_forward.tic()
        #local_filters = self.filter_emb(buckets)
        #if self.args.timer:
            #torch.cuda.synchronize()
        #   self.args.timer_embedding_forward.hold()
        #    self.args.timer_kconv_forward.tic()
        if self.args.debug:
            print('filter retrieved')
        #local_filters = local_filters.permute(0,3,1,2).contiguous()
        #print(x.dtype, self.filter_emb.weight.dtype, buckets.dtype)
        out = self.localconv(x, self.filter_emb.weight, buckets)
        #if self.args.timer:
        #    torch.cuda.synchronize()
        #    self.args.timer_kconv_forward.hold()
        #    self.args.timer_total_forward.hold()
        if self.args.debug:
            print('local conv done')
        return out

