import os
import sys
import numpy as np
#import math
#from math import floor, atan2, pi, isnan, sqrt

import torch

#import pycuda.driver as cuda
#from pycuda.compiler import SourceModule
#import pycuda.autoinit
#from pycuda import gpuarray
import time
import skimage.color as sc

import raisr_hash

"""
with open('raisr_hash_function.cu', 'r',encoding='utf-8') as myfile:
    data = myfile.read()
    mod =SourceModule(data)


def raisr_compute_grads_cuda(img, hsize=4,  step=1):
    row, col, _ = img.shape

    lr = (sc.rgb2ycbcr(img).astype(np.float32) / 255.)[:, :, 0]
    # lr = gaussian_filter(lr, sigma=0.8)
    in_GX, in_GY = np.gradient(lr)

    p_hsize = hsize
    bx =16
    by =16
    row_gpu = np.int32(row)
    col_gpu = np.int32(col)
    out_tensor = np.zeros(img.shape, dtype=np.float32)
    hsize_gpu = np.int32(hsize)

    gdimX = (int)((col + bx-1) / bx)
    gdimY = (int)((row + by-1) / by)
    print(gdimX, gdimY, gdimX*gdimY)
    tensor_phi = np.zeros((row,col), dtype=np.float32)
    tensor_lambda = np.zeros((row, col), dtype=np.float32)
    tensor_mu = np.zeros((row, col), dtype=np.float32)
    tensor_phi_gpu = gpuarray.to_gpu(tensor_phi.astype(np.float32))
    tensor_lambda_gpu = gpuarray.to_gpu(tensor_lambda.astype(np.float32))
    tensor_mu_gpu = gpuarray.to_gpu(tensor_mu.astype(np.float32))

    in_GX_gpu = gpuarray.to_gpu(in_GX.astype(np.float32))
    in_GY_gpu = gpuarray.to_gpu(in_GY.astype(np.float32))


    func = mod.get_function("raisr_hash_cuda")
    func(in_GX_gpu, in_GY_gpu,hsize_gpu,
               row_gpu,col_gpu,tensor_phi_gpu,tensor_lambda_gpu, tensor_mu_gpu, block=(bx,by,1), grid = (gdimX, gdimY))

    tensor_phi = tensor_phi_gpu.get()
    tensor_lambda = tensor_lambda_gpu.get()
    tensor_mu = tensor_mu_gpu.get()
    out_tensor[:,:,0] = tensor_phi
    out_tensor[:, :, 1] = tensor_lambda
    out_tensor[:, :, 2] = tensor_mu

    return out_tensor

"""
def raisr_compute_grads_cuda(img, hsize=4,  step=1,device=None):
    if len(img.shape) == 3:
        row, col, c = img.shape
        if c == 3:
            lr = (sc.rgb2ycbcr(img).astype(np.float32) / 255.)[:, :, 0]
        else:
            lr = img[:,:,0]
    else:
        row, col = img.shape
        c=1
        lr = img


    # lr = gaussian_filter(lr, sigma=0.8)
    in_GX, in_GY = np.gradient(lr)
    p_hsize = hsize
    #bx =16
    #by =16
    #row_gpu = np.int32(row)
    #col_gpu = np.int32(col)
    out_tensor = np.zeros((row,col,3), dtype=np.float32)
    #hsize_gpu = np.int32(hsize)
    tensor_phi = np.zeros((row,col), dtype=np.float32)
    tensor_lambda = np.zeros((row, col), dtype=np.float32)
    tensor_mu = np.zeros((row, col), dtype=np.float32)
    tensor_phi_gpu = torch.from_numpy(tensor_phi).to(device)
    tensor_lambda_gpu = torch.from_numpy(tensor_lambda).to(device)
    tensor_mu_gpu = torch.from_numpy(tensor_mu).to(device)
    in_GX_gpu = torch.from_numpy(in_GX).to(device)
    in_GY_gpu = torch.from_numpy(in_GY).to(device)
    raisr_hash.forward(in_GX_gpu,in_GY_gpu,hsize,row,col,tensor_phi_gpu,tensor_lambda_gpu,tensor_mu_gpu)
    out_tensor[:, :, 0] = tensor_phi_gpu
    out_tensor[:, :, 1] = tensor_lambda_gpu
    out_tensor[:, :, 2] = tensor_mu_gpu
    #print( out_tensor[:, :, 0].min(),  out_tensor[:, :, 0].max())
    #print(out_tensor[:, :, 1].min(), out_tensor[:, :, 1].max())
    #print(out_tensor[:, :, 2].min(), out_tensor[:, :, 2].max())
    """
    print("By cuda:   ")
    print(out_tensor[10:13, 10:13, 0])
    print(out_tensor[10:13, 10:13, 1])
    print(out_tensor[10:13, 10:13, 2])
    """
    #out_tensor[:,:,0] = tensor_phi_gpu.cpu().numpy()
    #out_tensor[:, :, 1] = tensor_lambda.cpu().numpy()
    #out_tensor[:, :, 2] = tensor_mu.cpu().numpy()
    return out_tensor
