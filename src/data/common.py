import random
from data.isp_pipeline import RawProcess
import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False, fix=False,recon=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size
    if fix:
        ix=600
        iy=100
    else:
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    #ret = [
    #    args[0][iy:iy + ip, ix:ix + ip, :],
    #    (*[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:-1]]), args[-1][iy:iy + ip, ix:ix + ip, :]
    #]
    ret = [
        (*[a[iy:iy + ip, ix:ix + ip, :]for a in args])
    ]

    return ret
def rgb2yuv(rgb):
    out = np.zeros_like(rgb)
    out[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    out[:, :, 1] = -0.1687 * rgb[:, :, 0] - 0.3313 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] + 0.5
    out[:, :, 2] = 0.5 * rgb[:, :, 0] - 0.4187 * rgb[:, :, 1] - 0.0813 * rgb[:, :, 2] + 0.5
    return out
def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    if len(args)==3:
        res = [_set_channel(a) for a in args[:-1]]
        res.append(args[-1])
    else:
        res = [_set_channel(a) for a in args]
    return res

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor
    def _np2Tensor_mask(img):
        if img.ndim ==2:
             #print(img.shape)
             tensor = torch.from_numpy(img).long()
        else:
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose.astype(float)).float()

        return tensor

    retlist = [_np2Tensor(a) for a in args[:-1]]
    retlist.append(_np2Tensor_mask(args[-1]))

    return retlist

def into_groups(mask,Qcohe, Qstr):
    H,W = mask.shape
    temp = mask
    mask_group = np.zeros((H,W,3),dtype=np.int32)
    mask_group[:,:,2] = temp % Qcohe
    temp = temp // Qcohe
    mask_group[:,:,1] = temp % Qstr
    temp = temp // Qstr
    mask_group[:,:,0] = temp

    return mask_group

def augment(*args, hflip=True, rot=True,argss=None):
    hflip = hflip and random.random() < 0.5
    #vflip = rot and random.random() < 0.5
    vflip = False
    rot90 = rot and random.random() < 0.5
    Qcohe = argss.Qcohe
    Qstr = argss.Qstr
    def _augment(img,raisrmask=False):
        if hflip:
            img = img[::-1,:,:][:, ::-1, :].copy()
            if img.shape[2]==9:
                #print('transforming mask patch')
                img = img[:,:, np.array([0,8,7,6,5,4,3,2,1])]
        if vflip:
            img = img[::-1, :, :].copy()
        if rot90:
            img = img.transpose(1, 0, 2)[:,::-1,:].copy()
            if raisrmask:
                img_groups = into_groups(img[:,:,0].astype(np.int32),Qcohe, Qstr)
                img_angles = img_groups[:,:,0]
                img_angles = (img_angles + 4)%8
                img_groups[:,:,0] = img_angles
                img = np.expand_dims(((img_groups[:,:,0]* Qstr * Qcohe) + img_groups[:,:,1] * Qcohe + img_groups[:,:,2]),-1).astype(np.float32)
                #print(img.shape)
                #print(img[10:20,10:20,0])
            if img.shape[2]==9:
                img = img[:,:,np.array([0,5,6,7,8,1,2,3,4])]
        
        return img

    retlist =  [_augment(a) for a in args[:-1]]
    retlist.append(_augment(args[-1],raisrmask=argss.from_regression))
    return retlist

