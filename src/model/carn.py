import torch
import torch.nn as nn
import model.ops as ops
from model.common import ModuleGrouper
import torch.nn.functional as F

def make_model(args, parent=False):
    return Net(args)

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.EResidualBlock2(in_channels, in_channels)
        self.b2 = ops.EResidualBlock2(in_channels, in_channels)
        self.b3 = ops.EResidualBlock2(in_channels, in_channels)
        self.c1 = ops.BasicBlock(in_channels*2, in_channels, 1, 1, 0)
        self.c2 = ops.BasicBlock(in_channels*3, in_channels, 1, 1, 0)
        self.c3 = ops.BasicBlock(in_channels*4, in_channels, 1, 1, 0)

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


class BlockBucket(nn.Module):
    def __init__(self,args,
                 in_channels, out_channels,
                 group=4):
        super(BlockBucket, self).__init__()

        self.b1 = EResidualBlockBucket(args,args.n_feats, args.n_feats)
        self.b2 = EResidualBlockBucket(args,args.n_feats, args.n_feats)
        self.b3 = EResidualBlockBucket(args,args.n_feats, args.n_feats)
        self.c1 = ops.BasicBlock(args.n_feats * 2, args.n_feats, 1, 1, 0)
        self.c2 = ops.BasicBlock(args.n_feats * 3, args.n_feats, 1, 1, 0)
        self.c3 = ops.BasicBlock(args.n_feats * 4, args.n_feats, 1, 1, 0)

    def forward(self, x,buckets=None):
        c0 = o0 = x

        b1 = self.b1(o0,buckets)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1,buckets)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2,buckets)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class EResidualBlockBucket(nn.Module):
    def __init__(self,args,
                 in_channels, out_channels,
                 group=4):
        super(EResidualBlockBucket, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),)

        self.body_split=  BucketConvLayer(args, args.kernel_size, out_channels, out_channels)
        self.body2=nn.Sequential(nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)

    def forward(self, x,buckets=None):
        out = self.body1(x)
        out = self.body_split(out,buckets)
        out =self.body2(out)
        out = F.relu(out + x)
        return out

def init_weights(modules):
    pass
        

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        #scale = args.scale[0]
        self.scale = args.scale[0]
        #print(self.scale)
        multi_scale = args.multi_scale
        group = args.group
        #group = kwargs.get("group", 1)
        self.num_types = args.num_types
        self.neuf = args.neuf
        self.neufed = False
        self.part = args.part

        #self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        #self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        if self.args.use_real:
            self.input_channels = self.args.n_colors+3
        else:
            self.input_channels = self.args.n_colors
        self.entry = nn.Conv2d(self.input_channels, self.args.n_feats, 3, 1, 1)
        if self.args.split:
            self.b1 = BlockBucket(self.args,self.args.n_feats, self.args.n_feats)
            if self.args.bottom_only:
                self.b2 = Block( self.args.n_feats, self.args.n_feats)
                self.b3 = Block( self.args.n_feats, self.args.n_feats)
            else:
                self.b2 = BlockBucket(self.args,self.args.n_feats, self.args.n_feats)
                self.b3 = BlockBucket(self.args,self.args.n_feats, self.args.n_feats)
        else:
            self.b1 = Block(self.args.n_feats, self.args.n_feats)
            self.b2 = Block(self.args.n_feats, self.args.n_feats)
            self.b3 = Block(self.args.n_feats, self.args.n_feats)
        self.c1 = ops.BasicBlock(self.args.n_feats*2, self.args.n_feats, 1, 1, 0)
        self.c2 = ops.BasicBlock(self.args.n_feats*3, self.args.n_feats, 1, 1, 0)
        self.c3 = ops.BasicBlock(self.args.n_feats*4, self.args.n_feats, 1, 1, 0)
        
        if self.scale!=1:
            self.upsample = ops.UpsampleBlock(self.args.n_feats, scale=self.scale,
                                          multi_scale=multi_scale,
                                          group=group)
        else:
            self.upsample = None
        #print(self.upsample)
        self.exit = nn.Conv2d(self.args.n_feats, args.n_colors, 3, 1, 1)
                
    def forward(self, x, mask_bin=None, scale=1):
        #if not self.neufed:
        #    self.neufize()
        #print(x.shape)

        #x = self.sub_mean(x)
        x = self.entry(x)
        #print(x.shape)
        #c0 = o0 = x
        if ('b1' in self.part) and (self.neuf):
            b1 = self.b1([x,mask_bin])
        elif self.args.split:
            b1 = self.b1(x,mask_bin)
        else:
            b1 = self.b1(x)
        c1 = torch.cat([x, b1], dim=1)
        if ('c1' in self.part) and (self.neuf):
            o1 = self.c1([c1,mask_bin])
        else:
            o1 = self.c1(c1)
        if ('b2' in self.part) and (self.neuf):
            b2 = self.b2([o1,mask_bin])
        elif self.args.split and (not self.args.bottom_only):
            b2 = self.b2(o1, mask_bin)
        else:
            b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        if ('c2' in self.part) and (self.neuf):
            o2 = self.c2([c2,mask_bin])
        else:
            o2 = self.c2(c2)
        if ('b3' in self.part) and (self.neuf):
            b3 = self.b3([o2,mask_bin])
        elif self.args.split and (not self.args.bottom_only):
            b3 = self.b3(o2, mask_bin)
        else:
            b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        if ('c3' in self.part) and (self.neuf):
            o3 = self.c3([c3,mask_bin])
        else:
            o3 = self.c3(c3)
        #print(o3.shape)
        if self.scale!=1:
            out = self.upsample(o3, scale=self.scale)
        else:
            out = o3
        #print(self.scale)
        #print(out.shape)
        #print("major part complete", out.shape)
        out = self.exit(out)

        #out = self.add_mean(out)
        #print("returning", out.shape)

        return out

    def _input_prepare(self, inputs):
        if self.neuf:
            return inputs
        else:
            return inputs[0]
    def load_state_dict(self, state_dict, strict=True, reinit=False):
        own_state = self.state_dict()
        ban_names = ['b1','b2','b3','c1','c2','c3'] if reinit else []

        if self.args.load_transfer:
            bnames = ['b1','b2','b3']
            for bn_1 in bnames:
                for bn_2 in bnames:
                    source_name = bn_1+'.'+bn_2+'.'+'body.2.'
                    tgt_name = bn_1+'.'+bn_2+'.'+'body_split.filter_emb.weight'
                    if (source_name + 'weight') in state_dict:
                        if (tgt_name) in own_state:
                            print('transfer loading: ', tgt_name)
                            conv_w = state_dict[source_name + 'weight']
                            conv_b = state_dict[source_name + 'bias']

                            w_reshape = conv_w.view(self.args.n_feats, -1).contiguous()
                            ww = torch.cat((w_reshape, conv_b.unsqueeze(-1)), -1).view(-1).contiguous().unsqueeze(0) \
                                .repeat(own_state[tgt_name].shape[0],
                                        1).contiguous()
                            own_state[tgt_name].copy_(ww)
                            print('successfully loaded: ', tgt_name)
                    source_name2 = bn_1 + '.' + bn_2 + '.' + 'body.'
                    tgt_name2 = bn_1 + '.' + bn_2 + '.' + 'body'
                    own_state[tgt_name2+'1.0.weight'].copy_(state_dict[source_name2+'0.weight'])
                    own_state[tgt_name2 + '1.0.bias'].copy_(state_dict[source_name2 + '0.bias'])
                    own_state[tgt_name2 + '2.1.weight'].copy_(state_dict[source_name2 + '4.weight'])
                    own_state[tgt_name2 + '2.1.bias'].copy_(state_dict[source_name2 + '4.bias'])

        if self.scale ==1:
            ban_names.append('upsample')
        for name, param in state_dict.items():
            if (name in own_state) and(not any(banname in name for banname in ban_names)):
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
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

    def neufize(self):
        if 'b1' in self.part:
            self.b1 = ModuleGrouper(self.b1, self.num_types)
        if 'b2' in self.part:
            self.b2 = ModuleGrouper(self.b2, self.num_types)
        if 'b3' in self.part:
            self.b3 = ModuleGrouper(self.b3, self.num_types)
        if 'c1' in self.part:
            self.c1 = ModuleGrouper(self.c1, self.num_types)
        if 'c2' in self.part:
            self.c2 = ModuleGrouper(self.c2, self.num_types)
        if 'c3' in self.part:
            self.c3 = ModuleGrouper(self.c3, self.num_types)
        print("neufed!")
        #print(self.state_dict().keys())
        #print(self.b1)
        #print(self.b1.state_dict().keys())
        self.neufed = True


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
        self.nonlinear = nn.PReLU()
        #self.final = BucketConvLayer(args, args.kernel_size, self.mid_channels, out_channels)
        self.final = BucketConvLayer(args, args.kernel_size, out_channels, out_channels)
        self.nonlinear2 = nn.PReLU()


        #init_weights(self.modules)

    def forward(self, x, buckets=None):
        out = self.body(x)
        out = self.nonlinear(out)
        out = self.final(out, buckets)
        out = self.nonlinear2(out+x)

        return out


