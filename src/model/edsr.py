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
    return Bucket_Conv(args)

class Bucket_Conv(nn.Module):
    def __init__(self, args, bias=True):
        super(Bucket_Conv, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.num_types = args.Qstr*args.Qcohe*args.Qangle
        self.kernel_size = args.kernel_size
        self.n_layers = args.n_layers
        if self.args.use_real:
            self.input_channels = self.args.n_colors+3
        else:
            self.input_channels = self.args.n_colors
        self.entry = nn.Conv2d(self.input_channels, args.n_feats, 3, 1, 1)
        self.split = args.split
        resclass = ResidualBlockBucket if self.split else ResidualBlock
        resclass2 = resclass if not args.bottom_only else ResidualBlock
        if self.n_layers == 1:
            self.res1 = resclass(args,args.n_feats,args.n_feats)
        if self.n_layers == 2:
            self.res1 = resclass(args,args.n_feats,args.n_feats)
            self.res2= resclass2(args,args.n_feats,args.n_feats)

        if self.n_layers == 3:

            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)

        elif self.n_layers==4:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)


        elif self.n_layers ==5:
            """
            self.first = BucketConvLayer(args,args.kernel_size, args.n_colors, 16)
            self.nonlinear = nn.PReLU()
            self.middle = BucketConvLayer(args,args.kernel_size, 16, 16)
            self.nonlinear2 = nn.PReLU()
            self.last = BucketConvLayer(args,args.kernel_size, 16, args.n_colors)
            """
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)

        elif self.n_layers == 8:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)
            self.res6 = resclass2(args, args.n_feats, args.n_feats)
            self.res7 = resclass2(args, args.n_feats, args.n_feats)
            self.res8 = resclass2(args, args.n_feats, args.n_feats)
        elif self.n_layers == 12:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)
            self.res6 = resclass2(args, args.n_feats, args.n_feats)
            self.res7 = resclass2(args, args.n_feats, args.n_feats)
            self.res8 = resclass2(args, args.n_feats, args.n_feats)
            self.res9 = resclass2(args, args.n_feats, args.n_feats)
            self.res10 = resclass2(args, args.n_feats, args.n_feats)
            self.res11 = resclass2(args, args.n_feats, args.n_feats)
            self.res12 = resclass2(args, args.n_feats, args.n_feats)
        elif self.n_layers == 14:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)
            self.res6 = resclass2(args, args.n_feats, args.n_feats)
            self.res7 = resclass2(args, args.n_feats, args.n_feats)
            self.res8 = resclass2(args, args.n_feats, args.n_feats)
            self.res9 = resclass2(args, args.n_feats, args.n_feats)
            self.res10 = resclass2(args, args.n_feats, args.n_feats)
            self.res11 = resclass2(args, args.n_feats, args.n_feats)
            self.res12 = resclass2(args, args.n_feats, args.n_feats)
            self.res13 = resclass2(args, args.n_feats, args.n_feats)
            self.res14 = resclass2(args, args.n_feats, args.n_feats)
        elif self.n_layers == 16:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)
            self.res6 = resclass2(args, args.n_feats, args.n_feats)
            self.res7 = resclass2(args, args.n_feats, args.n_feats)
            self.res8 = resclass2(args, args.n_feats, args.n_feats)
            self.res9 = resclass2(args, args.n_feats, args.n_feats)
            self.res10 = resclass2(args, args.n_feats, args.n_feats)
            self.res11 = resclass2(args, args.n_feats, args.n_feats)
            self.res12 = resclass2(args, args.n_feats, args.n_feats)
            self.res13 = resclass2(args, args.n_feats, args.n_feats)
            self.res14 = resclass2(args, args.n_feats, args.n_feats)
            self.res15 = resclass2(args, args.n_feats, args.n_feats)
            self.res16 = resclass2(args, args.n_feats, args.n_feats)
            if self.args.split_waist:
                self.res9 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res10 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res11 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res12 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res13 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res14 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res15 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res16 = ResidualBlock(args, args.n_feats, args.n_feats)
            if self.args.split_skip:
                self.res2 = ResidualBlock(args, args.n_feats, args.n_feats)

                self.res4 = ResidualBlock(args, args.n_feats, args.n_feats)

                self.res6 = ResidualBlock(args, args.n_feats, args.n_feats)

                self.res8 = ResidualBlock(args, args.n_feats, args.n_feats)

                self.res10 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res12 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res14 = ResidualBlock(args, args.n_feats, args.n_feats)
                self.res16 = ResidualBlock(args, args.n_feats, args.n_feats)
        elif self.n_layers == 18:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)
            self.res6 = resclass2(args, args.n_feats, args.n_feats)
            self.res7 = resclass2(args, args.n_feats, args.n_feats)
            self.res8 = resclass2(args, args.n_feats, args.n_feats)
            self.res9 = resclass2(args, args.n_feats, args.n_feats)
            self.res10 = resclass2(args, args.n_feats, args.n_feats)
            self.res11 = resclass2(args, args.n_feats, args.n_feats)
            self.res12 = resclass2(args, args.n_feats, args.n_feats)
            self.res13 = resclass2(args, args.n_feats, args.n_feats)
            self.res14 = resclass2(args, args.n_feats, args.n_feats)
            self.res15 = resclass2(args, args.n_feats, args.n_feats)
            self.res16 = resclass2(args, args.n_feats, args.n_feats)
            self.res17 = resclass2(args, args.n_feats, args.n_feats)
            self.res18 = resclass2(args, args.n_feats, args.n_feats)
        elif self.n_layers == 32:
            self.res1 = resclass(args, args.n_feats, args.n_feats)
            self.res2 = resclass2(args, args.n_feats, args.n_feats)
            self.res3 = resclass2(args, args.n_feats, args.n_feats)
            self.res4 = resclass2(args, args.n_feats, args.n_feats)
            self.res5 = resclass2(args, args.n_feats, args.n_feats)
            self.res6 = resclass2(args, args.n_feats, args.n_feats)
            self.res7 = resclass2(args, args.n_feats, args.n_feats)
            self.res8 = resclass2(args, args.n_feats, args.n_feats)
            self.res9 = resclass2(args, args.n_feats, args.n_feats)
            self.res10 = resclass2(args, args.n_feats, args.n_feats)
            self.res11 = resclass2(args, args.n_feats, args.n_feats)
            self.res12 = resclass2(args, args.n_feats, args.n_feats)
            self.res13 = resclass2(args, args.n_feats, args.n_feats)
            self.res14 = resclass2(args, args.n_feats, args.n_feats)
            self.res15 = resclass2(args, args.n_feats, args.n_feats)
            self.res16 = resclass2(args, args.n_feats, args.n_feats)
            self.res17 = resclass2(args, args.n_feats, args.n_feats)
            self.res18 = resclass2(args, args.n_feats, args.n_feats)
            self.res19 = resclass2(args, args.n_feats, args.n_feats)
            self.res20 = resclass2(args, args.n_feats, args.n_feats)
            self.res21 = resclass2(args, args.n_feats, args.n_feats)
            self.res22 = resclass2(args, args.n_feats, args.n_feats)
            self.res23 = resclass2(args, args.n_feats, args.n_feats)
            self.res24 = resclass2(args, args.n_feats, args.n_feats)
            self.res25 = resclass2(args, args.n_feats, args.n_feats)
            self.res26 = resclass2(args, args.n_feats, args.n_feats)
            self.res27 = resclass2(args, args.n_feats, args.n_feats)
            self.res28 = resclass2(args, args.n_feats, args.n_feats)
            self.res29 = resclass2(args, args.n_feats, args.n_feats)
            self.res30 = resclass2(args, args.n_feats, args.n_feats)
            self.res31 = resclass2(args, args.n_feats, args.n_feats)
            self.res32 = resclass2(args, args.n_feats, args.n_feats)

        self.exit = nn.Conv2d(args.n_feats,self.args.n_colors, 3, 1, 1)
        self.debug = args.debug
        self.reinit = args.reinit


    def load_state_dict(self, state_dict, strict=True, reinit=False):
        if not self.reinit:
            own_state = self.state_dict()
            ban_names = ['filter_emb'] if reinit else []
            if self.args.load_transfer:
                for lnum in range(self.args.n_layers):
                    layernum = lnum+1
                    if ('res'+str(layernum)+'.final.weight') in state_dict:
                        if ('res'+str(layernum)+'.final.filter_emb.weight') in own_state:
                            print('transfer loading: ',('res'+str(layernum)+'.final.filter_emb.weight'))
                            conv_w = state_dict['res'+str(layernum)+'.final.weight']
                            conv_b = state_dict['res'+str(layernum)+'.final.bias']
                            w_reshape = conv_w.view(self.args.n_feats, -1).contiguous()
                            ww = torch.cat((w_reshape, conv_b.unsqueeze(-1)), -1).view(-1).contiguous().unsqueeze(0)\
                                .repeat(own_state['res'+str(layernum)+'.final.filter_emb.weight'].shape[0],1).contiguous()
                            own_state['res' + str(layernum) + '.final.filter_emb.weight'].copy_(ww)
                            print('successfully loaded: ', ('res' + str(layernum) + '.final.filter_emb.weight'))

            for name, param in state_dict.items():
                print("trying to load: ", name)
                if (name in own_state) and (not any(banname in name for banname in ban_names)):
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    try:
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
    def forward(self, x, buckets):
        """

        :param x: of shape (batch_size, input_channel, H, W)
        :param buckets: LongTensor of shape (batch_size, H, W)
        :return: out: FloatTensor of shape (batch_size, out_channel, H, W)

        """
        if self.args.debug:
            if buckets.shape[1] > 120:
                print('forwarding x shape', x.shape)
                print('forwarding bucket shape', buckets.shape)
        out = self.entry(x)
        ent = out
        if self.n_layers ==1:
            out = self.res1(out,buckets)
        if self.n_layers ==2:
            out = self.res1(out,buckets)
            out = self.res2(out, buckets)
        if self.n_layers == 3:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
        #elif self.n_layers ==1:
        #    out = self.body(x, buckets)
        elif self.n_layers == 4:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
        elif self.n_layers == 5:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
        elif self.n_layers == 8:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
            out = self.res6(out, buckets)
            out = self.res7(out, buckets)
            out = self.res8(out, buckets)
        elif self.n_layers == 12:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
            out = self.res6(out, buckets)
            out = self.res7(out, buckets)
            out = self.res8(out, buckets)
            out = self.res9(out, buckets)
            out = self.res10(out, buckets)
            out = self.res11(out, buckets)
            out = self.res12(out, buckets)
        elif self.n_layers == 14:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
            out = self.res6(out, buckets)
            out = self.res7(out, buckets)
            out = self.res8(out, buckets)
            out = self.res9(out, buckets)
            out = self.res10(out, buckets)
            out = self.res11(out, buckets)
            out = self.res12(out, buckets)
            out = self.res13(out, buckets)
            out = self.res14(out, buckets)
        elif self.n_layers == 16:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
            out = self.res6(out, buckets)
            out = self.res7(out, buckets)
            out = self.res8(out, buckets)
            out = self.res9(out, buckets)
            out = self.res10(out, buckets)
            out = self.res11(out, buckets)
            out = self.res12(out, buckets)
            out = self.res13(out, buckets)
            out = self.res14(out, buckets)
            out = self.res15(out, buckets)
            out = self.res16(out, buckets)
        elif self.n_layers == 18:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
            out = self.res6(out, buckets)
            out = self.res7(out, buckets)
            out = self.res8(out, buckets)
            out = self.res9(out, buckets)
            out = self.res10(out, buckets)
            out = self.res11(out, buckets)
            out = self.res12(out, buckets)
            out = self.res13(out, buckets)
            out = self.res14(out, buckets)
            out = self.res15(out, buckets)
            out = self.res16(out, buckets)
            out = self.res17(out, buckets)
            out = self.res18(out, buckets)
        elif self.n_layers == 32:
            out = self.res1(out, buckets)
            out = self.res2(out, buckets)
            out = self.res3(out, buckets)
            out = self.res4(out, buckets)
            out = self.res5(out, buckets)
            out = self.res6(out, buckets)
            out = self.res7(out, buckets)
            out = self.res8(out, buckets)
            out = self.res9(out, buckets)
            out = self.res10(out, buckets)
            out = self.res11(out, buckets)
            out = self.res12(out, buckets)
            out = self.res13(out, buckets)
            out = self.res14(out, buckets)
            out = self.res15(out, buckets)
            out = self.res16(out, buckets)
            out = self.res17(out, buckets)
            out = self.res18(out, buckets)
            out = self.res19(out, buckets)
            out = self.res20(out, buckets)
            out = self.res21(out, buckets)
            out = self.res22(out, buckets)
            out = self.res23(out, buckets)
            out = self.res24(out, buckets)
            out = self.res25(out, buckets)
            out = self.res26(out, buckets)
            out = self.res27(out, buckets)
            out = self.res28(out, buckets)
            out = self.res29(out, buckets)
            out = self.res30(out, buckets)
            out = self.res31(out, buckets)
            out = self.res32(out, buckets)

        out = self.exit(out+ent)
        #out = self.exit(out)
        if self.args.debug:
            if buckets.shape[1] > 120:
                print('forwarded out shape', out.shape)
                print('forwarded bucket shape', buckets.shape)
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


class ResidualBlockBucket(nn.Module):
    def __init__(self,
                 args,in_channels, out_channels,mid_channels =None):
        super(ResidualBlockBucket, self).__init__()
        """
        if mid_channels is not None:
            self.mid_channels = mid_channels
        else:
            self.mid_channels = out_channels
        """
        self.args = args
        if args.small_res:
            #self.body = nn.Conv2d(in_channels, self.mid_channels, 3, 1, 1)
            self.body = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.PReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )
        self.res_scale = args.res_scale
        self.nonlinear = nn.PReLU()
        #self.final = BucketConvLayer(args, args.kernel_size, self.mid_channels, out_channels)
        self.final = BucketConvLayerCUDA(args, args.kernel_size, out_channels, out_channels)
        self.nonlinear2 = nn.PReLU()


        #init_weights(self.modules)

    def forward(self, x, buckets=None):
        out = self.body(x)
        out = self.nonlinear(out)
        if self.args.timer:
            torch.cuda.synchronize()
            self.args.timer_total_forward.tic()
        out = self.final(out, buckets).mul(self.res_scale)
        out = out+x
        if self.args.timer:
            torch.cuda.synchronize()
            self.args.timer_total_forward.hold()
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 args,in_channels, out_channels, mid_channels =None):
        super(ResidualBlock, self).__init__()
        """
        if mid_channels is not None:
            self.mid_channels = mid_channels
        else:
            self.mid_channels = out_channels
        """
        self.args = args
        if args.small_res:
            #self.body = nn.Conv2d(in_channels, self.mid_channels, 3, 1, 1)
            self.body = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.PReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )
        self.nonlinear = nn.PReLU()
        #self.final = nn.Conv2d(self.mid_channels, out_channels, args.kernel_size, 1, (args.kernel_size-1)//2)
        self.final = nn.Conv2d(out_channels, out_channels, args.kernel_size, 1, (args.kernel_size - 1) // 2)
        self.res_scale = args.res_scale
        self.nonlinear2 = nn.PReLU()
        #init_weights(self.modules)

    def forward(self, x, buckets=None):
        out = self.body(x)
        out = self.nonlinear(out)
        if self.args.timer:
            torch.cuda.synchronize()
            self.args.timer_total_forward.tic()
        out = self.final(out).mul(self.res_scale)
        out = out+x
        if self.args.timer:
            torch.cuda.synchronize()
            self.args.timer_total_forward.hold()
        return out

