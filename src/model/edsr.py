from model import common
from model.common import ModuleGrouper

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.num_types = args.num_types
        self.neuf = args.neuf
        self.neufed = False
        self.args =args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        #self.sub_mean = common.MeanShift(args.rgb_range)
        #self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        if self.args.use_real:
            self.input_channels = self.args.n_colors+3
        else:
            self.input_channels = self.args.n_colors
        m_head = [conv(self.input_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            #common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, mask=None):
            #print(self.body)
            #print(self.body[0].modulelist)
        #x = self.sub_mean(x)
        x = self.head(x)
        if self.neuf:
            res = self.body([x,mask])
        else:
            res = self.body(x)
        res += x

        x = self.tail(res)
        #x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True, reinit=False):
        own_state = self.state_dict()
        ban_names = ['body'] if reinit else []
        for name, param in state_dict.items():
            if (name in own_state) and (not any(banname in name for banname in ban_names)) :
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
        body_neuf = [ModuleGrouper(block, num_types = self.num_types, keep_mask=True) for block in self.body[:-1]]
        body_neuf.append(ModuleGrouper(self.body[-1], num_types = self.num_types, keep_mask=False))
        self.body = nn.Sequential(*body_neuf)
        self.neufed = True
        print(self.state_dict().keys())
        print("model neufed!")

