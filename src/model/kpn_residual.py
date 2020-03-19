import torch
import torch.nn as nn
import model.ops as ops
import torch
import math, re, functools
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import copy

def make_model(args, parent=False):
    return Bucket_Conv(args)

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
        out = self.exit(out)
        if self.args.debug:
            if buckets.shape[1] > 120:
                print('forwarded out shape', out.shape)
                print('forwarded bucket shape', buckets.shape)
        return out



class BucketConvLayer(nn.Module):
    def __init__(self, args,kernel_size,in_channels, out_channels, bias=True):
        super(BucketConvLayer, self).__init__()

        self.kernel_size = args.kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = args.Qstr*args.Qcohe*args.Qangle
        self.kernel_size = kernel_size
        self.localconv = LocalConv2d_No(self.in_channels, self.out_channels, self.kernel_size)
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
        local_filters = self.filter_emb(buckets)
        if self.args.debug:
            print('filter retrieved')
        local_filters = local_filters.permute(0,3,1,2).contiguous()
        out = self.localconv(x, local_filters)
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
        self.final = BucketConvLayer(args, args.kernel_size, out_channels, out_channels)
        self.nonlinear2 = nn.PReLU()


        #init_weights(self.modules)

    def forward(self, x, buckets=None):
        out = self.body(x)
        out = self.nonlinear(out)
        out = self.final(out, buckets).mul(self.res_scale)
        out = self.nonlinear2(out+x)

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
        self.nonlinear2 = nn.PReLU()
        self.res_scale = args.res_scale

        #init_weights(self.modules)

    def forward(self, x, buckets=None):
        out = self.body(x)
        out = self.nonlinear(out)
        out = self.final(out).mul(self.res_scale)
        out = self.nonlinear2(out+x)

        return out

