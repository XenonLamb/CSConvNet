import torch
import torch.nn as nn
from model.common import ModuleGrouper
import math

def make_model(args, parent=False):
    model = Net(num_channels=args.n_colors, upscale_factor=args.scale, args=args)
    model.weight_init(mean=0.0, std=0.2)
    return model

class Net(torch.nn.Module):
    def __init__(self, num_channels, upscale_factor, d=56, s=12, m=4, args=None):
        super(Net, self).__init__()
        self.neufed = False
        self.num_types = args.num_types
        self.scale = args.scale[0]
        self.part = args.part



        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=self.scale, padding=9 // 2,
                                            output_padding=self.scale-1)
        self._initialize_weights()

        self.neuf = args.neuf

        # Deconvolution
        #if args.neuf:
        #    self.neufize()
            #print(self.mid_part)

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x, mask=None):
        if ((self.neuf)and (self.part =='first')):
            out = self.first_part([x, mask])
        else:
            out = self.first_part(x)
        if ((self.neuf)and (self.part in ['default', 'body'])):
        #    print('mask shape at forward',mask.shape)
        #    print('input shape',x.shape)
            out = self.mid_part([out, mask])
        else:
            out = self.mid_part(out)
        if ((self.neuf)and (self.part =='up')):
            out = self.last_part([out,mask])
        else:
            out = self.last_part(out)
        #out = self.last_part(out)
        return out

    def load_state_dict(self, state_dict, strict=True, reinit=False):
        own_state = self.state_dict()
        ban_names = ['first_part','mid_part','last_part'] if reinit else ['first_part','last_part']
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

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()

    def neufize(self):
        if self.part == 'default':
            body_neuf = [ModuleGrouper(block, num_types = self.num_types, keep_mask=True) for block in self.mid_part[:-1]]
            body_neuf.append(ModuleGrouper(self.mid_part[-1], num_types = self.num_types, keep_mask=False))
            self.mid_part = nn.Sequential(*body_neuf)
        elif self.part == 'up':
            self.last_part = ModuleGrouper(self.last_part, scaling=self.scale)
        elif self.part == 'body':
            self.mid_part = ModuleGrouper(self.mid_part)
        elif self.part == 'first':
            self.first_part = ModuleGrouper(self.first_part)
        self.neufed = True
        print(self.state_dict().keys())
        print("model neufed!")