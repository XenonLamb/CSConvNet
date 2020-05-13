#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Inherited from the code by Shangchen Zhou <shangchenzhou@gmail.com>
import torch
from torch import nn
from torch.autograd import Function
import CSKernelConv2D
import random

class CSKernelConv2DFunction(Function):
    #def __init__(self, kernel_size=3):
    #    super(CSKernelConv2DFunction, self).__init__()
    #    self.kernel_size = kernel_size
    @staticmethod
    def forward( ctx, input, kernel_bank, buckets, kernel_size):
        #print('forwarding')
        ctx.kernel_size = kernel_size
        assert(input.is_contiguous() == True)
        assert(kernel_bank.is_contiguous() == True)
        #self.mark_non_differentiable(buckets)
        ctx.save_for_backward(input, kernel_bank, buckets)
        #assert (self.kernel_size == int((kernel.size(1)/(input.size(1)**2))**0.5))
        intKernelSize = kernel_size
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intOutputHeight = buckets.size(1)
        intOutputWidth = buckets.size(2)

        assert(intInputHeight - intKernelSize == intOutputHeight - 1)
        assert(intInputWidth - intKernelSize == intOutputWidth - 1)

        with torch.cuda.device_of(input):
            output = input.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()
            if input.is_cuda == True:
                CSKernelConv2D.forward(input, kernel_bank, intKernelSize, output, buckets)
            elif input.is_cuda == False:
                raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED
                print(5)
        #print('forward successfully')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #print('backwarding, ',grad_output[:2,:2])

        #print('backwarding')
        input, kernel_bank, buckets  = ctx.saved_tensors
        #print(buckets.requires_grad)
        buckets.requires_grad = False
        intKernelSize = ctx.kernel_size
        grad_output = grad_output.contiguous()
        with torch.cuda.device_of(input):
            grad_input = input.new().resize_(input.size()).zero_()
            grad_kernel = kernel_bank.new().resize_(kernel_bank.size()).zero_()
            grad_buckets = buckets.new().resize_(buckets.size()).zero_()
            if grad_output.is_cuda == True:
                CSKernelConv2D.backward(input, kernel_bank, intKernelSize, grad_output, grad_input, grad_kernel, buckets)

            elif grad_output.is_cuda == False:
                raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

        return grad_input, grad_kernel, grad_buckets, None

"""
def gradient_check():
    kernel_size_list = [1, 3]
    len_list = [8, 10]
    for i in range(10):
        B = random.randint(1,4)
        C = i + 1
        K = random.choice(kernel_size_list)
        H = random.choice(len_list)
        W = random.choice(len_list)
        input = torch.randn(B,C,H+K-1,W+K-1, requires_grad=True).cuda()
        kernel = torch.randn(B,C*K*K,H,W, requires_grad=True).cuda()
        # linear function, thus eps set to 1e-1
        print(torch.autograd.gradcheck(KernelConv2DFunction(K),(input,kernel),eps=1e-1, atol=1e-5, rtol=1e-3, raise_exception=True))
"""
class CSConv2D(nn.Module):
    def __init__(self, kernel_size):
        super(CSConv2D, self).__init__()
        assert(kernel_size%2 == 1)
        self.kernel_size = kernel_size
        self.pad = torch.nn.ZeroPad2d([(kernel_size-1)//2, (kernel_size-1)//2, (kernel_size-1)//2, (kernel_size-1)//2])
        #self.csconv = CSKernelConv2DFunction(self.kernel_size)
    def forward(self, input, kernel_bank, buckets):
        input_pad = self.pad(input)
        #print('CSConv: input: ',input.shape, 'pad: ', input_pad.shape, 'buckets: ', buckets.shape )
        return CSKernelConv2DFunction.apply(input_pad, kernel_bank, buckets, self.kernel_size)