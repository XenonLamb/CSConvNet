""" Full assembly of the parts to form the complete network """


import torch.nn.functional as F

from .unet_parts import *


def make_model(args, parent=False):
    return UNet(args)

class UNet(nn.Module):
    def __init__(self,args, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        if args.predict_groups:
            self.n_classes = args.Qstr*args.Qcohe*args.Qangle
        else:
            self.n_classes = n_classes
        self.bilinear = bilinear
        self.args = args
        if args.use_stats:
            n_channels = args.n_colors+3
        else:
            n_channels = args.n_colors

        if self.args.unetsize =='tiny':
            self.inc = NewDoubleConvG2(n_channels, 8)
            self.down1 = NewDownG2(8, 8)
            self.down2 = NewDownG2(8, 16)
            self.down3 = NewDownG2(16, 16)
            self.ext1 = NewDoubleConvG2(16,8)
            self.ext2 = NewDoubleConvG2(8,16)
            self.up1 = NewUpG2(32, 8, bilinear)
            self.up2 = NewUpG2(16, 8, bilinear)
            self.up3 = NewUpG2(16, 8, bilinear)
            self.outc1 = OutConv(8, args.Qangle)
            self.outc2 = OutConv(8, args.Qstr)
            self.outc3 = OutConv(8, args.Qcohe)
        elif self.args.unetsize =='small':
            self.inc = NewDoubleConvG2(n_channels, 8)
            self.down1 = NewDownG2(8, 16)
            self.down2 = NewDownG2(16, 32)
            self.down3 = NewDownG2(32, 32)
            self.ext1 = NewResG2(32, 32)
            self.ext2 = NewResG2(32, 32)
            self.up1 = NewUpG2(64, 16, bilinear)
            self.up2 = NewUpG2(32, 8, bilinear)
            self.up3 = NewUpG2(16, 8, bilinear)
            self.outc1 = OutConv(8, args.Qangle)
            self.outc2 = OutConv(8, args.Qstr)
            self.outc3 = OutConv(8, args.Qcohe)
        else:
            self.inc = NewDoubleConvG2(n_channels, 16)
            self.down1 = NewDownG2(16, 32)
            self.down2 = NewDownG2(32, 64)
            self.down3 = NewDownG2(64, 64)
            self.up1 = NewUpG2(128, 32, bilinear)
            self.up2 = NewUpG2(64, 16, bilinear)
            self.up3 = NewUpG2(32, 16, bilinear)
            #self.outc = OutConv(16, self.n_classes)

            self.outc1 = OutConv(16, args.Qangle)
            self.outc2 = OutConv(16, args.Qstr)
            self.outc3 = OutConv(16, args.Qcohe)

    def forward(self, x, mask=None):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.ext1(x4)
        x4 = self.ext2(x4)
        #x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        #x = self.up1(x5, x4)
        #x = self.up2(x, x3)
        #x = self.up3(x, x2)
        #x = self.up4(x, x1)
        logits1 = self.outc1(x)
        logits2 = self.outc2(x)
        logits3 = self.outc3(x)
        # if self.args.debug:
        #    print("3 logits shape:")
        #    print(logits1.shape, logits2.shape, logits3.shape)
        # logits = torch.cat([logits1.unsqueeze(-1),logits2.unsqueeze(-1),logits3.unsqueeze(-1)],dim=-1)
        logits = [logits1, logits2, logits3]
        return logits

    def load_state_dict(self, state_dict, strict=True, reinit=False):
        own_state = self.state_dict()
        ban_names = ['inc','down','up','outc'] if reinit else []
        for name, param in state_dict.items():
            if (name in own_state) and (not any(banname in name for banname in ban_names)) :
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                    print('successfully loaded', name)
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
