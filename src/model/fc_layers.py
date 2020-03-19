import torch
import torch.nn as nn
import numpy as np


class Q1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Q1, self).__init__()

        self.mask = torch.from_numpy(np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=1, kernel_size=3)

    def forward(self, x):
        self.mask=self.mask.to(self.conv1.weight.device)
        self.conv1.weight.data = self.conv1.weight * self.mask
        x = self.conv1(x)
        return x


class Q2(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(Q2, self).__init__()

        self.mask = torch.from_numpy(np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dilated_value,
                               dilation=dilated_value)

    def forward(self, x):
        self.mask=self.mask.to(self.conv1.weight.device)
        self.conv1.weight.data = self.conv1.weight * self.mask
        x = self.conv1(x)
        return x


class E1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(E1, self).__init__()

        self.mask = torch.from_numpy(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=1, kernel_size=3)

    def forward(self, x):
        self.mask=self.mask.to(self.conv1.weight.device)
        self.conv1.weight.data = self.conv1.weight * self.mask
        x = self.conv1(x)
        return x


class E2(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(E2, self).__init__()

        self.mask = torch.from_numpy(np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dilated_value,
                               dilation=dilated_value)

    def forward(self, x):
        self.mask=self.mask.to(self.conv1.weight.device)
        self.conv1.weight.data = self.conv1.weight * self.mask
        x = self.conv1(x)
        return x


class D1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D1, self).__init__()

        self.mask = torch.from_numpy(np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=1, kernel_size=3)

    def forward(self, x):
        self.mask=self.mask.to(self.conv1.weight.device)
        self.conv1.weight.data = self.conv1.weight * self.mask
        x = self.conv1(x)
        return x


class D2(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(D2, self).__init__()

        self.mask = torch.from_numpy(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=np.float32)).cuda()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dilated_value,
                               dilation=dilated_value)

    def forward(self, x):
        self.mask=self.mask.to(self.conv1.weight.device)
        self.conv1.weight.data = self.conv1.weight * self.mask
        x = self.conv1(x)
        return x


class QED_first_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QED_first_layer, self).__init__()

        self.q1 = Q1(in_ch, out_ch)
        self.e1 = E1(in_ch, out_ch)
        self.d1 = D1(in_ch, out_ch)

    def forward(self, x):
        outputs = []

        outputs.append(self.q1(x))
        outputs.append(self.e1(x))
        outputs.append(self.d1(x))

        return outputs


class QED_layer(nn.Module):
    def __init__(self, in_ch, out_ch, dilated_value):
        super(QED_layer, self).__init__()

        self.q2_prelu = nn.PReLU(in_ch, 0).cuda()
        self.e2_prelu = nn.PReLU(in_ch, 0).cuda()
        self.d2_prelu = nn.PReLU(in_ch, 0).cuda()

        self.q2 = Q2(in_ch, out_ch, dilated_value)
        self.e2 = E2(in_ch, out_ch, dilated_value)
        self.d2 = D2(in_ch, out_ch, dilated_value)

    def forward(self, inputs):
        outputs = []

        out_q2 = self.q2_prelu(inputs[0])
        out_e2 = self.e2_prelu(inputs[1])
        out_d2 = self.d2_prelu(inputs[2])

        outputs.append(self.q2(out_q2))
        outputs.append(self.e2(out_e2))
        outputs.append(self.d2(out_d2))

        return outputs


class Average_layer(nn.Module):
    def __init__(self, in_ch):
        super(Average_layer, self).__init__()

        self.prelu = nn.PReLU(in_ch, 0).cuda()

    def forward(self, inputs):
        mean = torch.mean(torch.stack(inputs), dim=0)
        #         mean = torch.mean(inputs, dim=0, keepdim = True)
        output = self.prelu(mean)

        return output


class Residual_module(nn.Module):
    def __init__(self, in_ch):
        super(Residual_module, self).__init__()

        self.prelu1 = nn.PReLU(in_ch, 0).cuda()
        self.prelu2 = nn.PReLU(in_ch, 0).cuda()
        #         self.prelu3 = nn.PReLU(in_ch).cuda()

        self.conv1_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv2_1by1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)

    def forward(self, input):
        #         output = self.prelu1(input)

        output_residual = self.conv1_1by1(input)
        output_residual = self.prelu1(output_residual)
        output_residual = self.conv2_1by1(output_residual)

        output = torch.mean(torch.stack([input, output_residual]), dim=0)
        output = self.prelu2(output)

        return output

        # class QED_first_layer(nn.Module):
        #     def __init__(self, in_ch, out_ch):
        #         super(QED_first_layer, self).__init__()

        #         self.q1 = Q1(in_ch,out_ch)
        #         self.e1 = E1(in_ch,out_ch)
        #         self.d1 = D1(in_ch,out_ch)

        #     def forward(self, x):

        #         outputs = []

        #         outputs = torch.cat((self.q1(x), self.e1(x), self.d1(x)), )
        #         return outputs


        # class QED_layer(nn.Module):
        #     def __init__(self, in_ch, out_ch, dilated_value):
        #         super(QED_layer, self).__init__()

        #         self.q2_prelu = nn.PReLU(in_ch,0).cuda()
        #         self.e2_prelu = nn.PReLU(in_ch,0).cuda()
        #         self.d2_prelu = nn.PReLU(in_ch,0).cuda()

        #         self.q2 = Q2(in_ch,out_ch,dilated_value)
        #         self.e2 = E2(in_ch,out_ch,dilated_value)
        #         self.d2 = D2(in_ch,out_ch,dilated_value)

        #     def forward(self, inputs):

        #         out_q2 = self.q2_prelu(self.q2(inputs[:1]))
        #         out_e2 = self.e2_prelu(self.e2(inputs[1:2]))
        #         out_d2 = self.d2_prelu(self.d2(inputs[2:3]))

        #         outputs = torch.cat((out_q2, out_e2, out_d2), )

        #         return outputs

        # class Average_layer(nn.Module):
        #     def __init__(self, in_ch):
        #         super(Average_layer, self).__init__()

        #         self.prelu = nn.PReLU(in_ch,0).cuda()

        #     def forward(self, inputs):

        #         mean = torch.mean(torch.stack(inputs), dim=0)
        # #         mean = torch.mean(inputs, dim=0, keepdim = True)
        #         output = self.prelu(mean)

        #         return output