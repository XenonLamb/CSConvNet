"""
main code for handling datasets/data loading
"""

import os
import glob
import random
import pickle
from math import floor, atan2, pi, isnan, sqrt
from data import common
from data.isp_pipeline import RawProcess
from scipy.ndimage import gaussian_filter
import numba as nb
import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import scipy.io
from data.utils.utils import *

@nb.jit(nopython=True)
def rgb2yuv(rgb):
    out = np.zeros_like(rgb)
    out[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    out[:, :, 1] = -0.1687 * rgb[:, :, 0] - 0.3313 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] + 0.5
    out[:, :, 2] = 0.5 * rgb[:, :, 0] - 0.4187 * rgb[:, :, 1] - 0.0813 * rgb[:, :, 2] + 0.5
    return out

@nb.jit(nopython=True)
def hash_table(patchX, patchY, weight, Qangle, Qstrength, Qcoherence, stre, cohe):
    assert (len(stre) == Qstrength- 1) and (len(cohe) == Qcoherence - 1), "Quantization number should be equal"
    gx = patchX.ravel()
    gy = patchY.ravel()
    G = np.vstack((gx, gy)).T
    if weight is not None:
        x0 = np.dot(G.T, weight)
        x = np.dot(x0, G)
    else:
        x = np.dot(G.T, G)
    x = x / (patchX.size)
    w,v = np.linalg.eig(x)
    index = w.argsort()[::-1]
    w = w[index]
    v = v[:, index]
    theta = atan2(v[0, 0], v[1, 0])
    if theta < 0:
        theta = theta + pi
    theta = floor(theta / (pi / Qangle))
    lamda = sqrt(abs(w[0]))
    u = (sqrt(w[0]) - sqrt(w[1])) / ((sqrt(w[0])) + sqrt(w[1]) + 0.00000000000001)
    if isnan(u):
        u = 1
    if theta > Qangle - 1:
        theta = Qangle - 1
    if theta < 0:
        theta = 0
    lamda = np.searchsorted(stre, lamda)
    u = np.searchsorted(cohe, u)
    return theta, lamda, u



## for real noises, please ignore
def load_CRF():
    CRF = scipy.io.loadmat('matdata/201_CRF_data.mat')
    iCRF = scipy.io.loadmat('matdata/dorfCurvesInv.mat')
    B_gl = CRF['B']
    I_gl = CRF['I']
    B_inv_gl = iCRF['invB']
    I_inv_gl = iCRF['invI']

    if os.path.exists('matdata/201_CRF_iCRF_function.mat')==0:
        CRF_para = np.array(CRF_function_transfer(I_gl, B_gl))
        iCRF_para = 1. / CRF_para
        scipy.io.savemat('matdata/201_CRF_iCRF_function.mat', {'CRF':CRF_para, 'iCRF':iCRF_para})
    else:
        Bundle = scipy.io.loadmat('matdata/201_CRF_iCRF_function.mat')
        CRF_para = Bundle['CRF']
        iCRF_para = Bundle['iCRF']

    #print('finished loading CRF:')
    #print(CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl)

    return CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        self.buffer = {}
        self._set_filesystem(args.dir_data)
        npys = np.load('./trans_prob.npz')
        self.trans_prob = npys['probs']
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        if self.args.real_isp or self.args.use_real:
            self.CRF_para, self.iCRF_para, self.I_gl, self.B_gl, self.I_inv_gl, self.B_inv_gl = load_CRF()

        self.mask_type = args.mask_type
        ##RAISR specific configs
        self.h_hsize = args.h_hsize
        self.Qstr = args.Qstr
        self.Qcohe = args.Qcohe
        self.Qangle = args.Qangle
        self.stre = np.linspace(0, 0.2, args.Qstr + 1)[1:-1]
        self.cohe = np.linspace(0, 1, args.Qcohe + 1)[1:-1]
        if train:
            list_hr, list_lr = self._scan(half_mode=args.halfing)
        else:
            list_hr, list_lr = self._scan()
        self.origin_lr = list_lr
        if (1 > 0):
            if self.args.compute_grads:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
            else:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
            for s in self.scale:
                if self.args.task == 'predenoise':
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            ('X{}_noise'+self.args.noiselevel).format(s)
                        ),
                        exist_ok=True
                    )
                else:
                    if self.args.compute_grads:
                        os.makedirs(
                            os.path.join(
                                self.dir_lr.replace(self.apath, path_bin),
                                'X{}'.format(s)
                            ),
                            exist_ok=True
                        )
                    else:
                        os.makedirs(
                            os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                            ),
                            exist_ok=True
                        )

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            self.masks_lr = [[] for _ in self.scale]
            self.masks_hr = []
            for h in list_hr:
                if self.args.compute_grads:
                    b = h.replace(self.apath, path_bin)
                else:
                    b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                b2 = b.replace('.pt', '_mask.npz')
                self.images_hr.append(b)
                if self.args.mask_type == 'neuf_gt':
                    self.masks_hr.append(b2)
                self._check_and_load(args.ext, h, b, (b2 if self.args.mask_type== 'neuf_gt' else None), verbose=True)
            for i, ll in enumerate(list_lr):
                if self.args.mask_type == 'neuf_gt':
                    self.masks_lr[i]=self.masks_hr
                for l in ll:
                    if self.args.compute_grads:
                        b = l.replace(self.apath, path_bin)
                    else:
                        b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    b2 = b.replace('.pt', '_mask.npz')
                    self.images_lr[i].append(b)
                    if self.args.mask_type != 'neuf_gt':
                        if self.args.compute_grads:
                            self.masks_lr[i].append(b2.replace('bin','bin_grads').replace('.npz', '_grads'+ str(self.args.h_hsize) + '.npz'))
                        else:
                            self.masks_lr[i].append(b2)
                    self._check_and_load(args.ext, l, b, (b2 if self.args.mask_type!= 'neuf_gt' else None), verbose=True)
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self, half_mode = 'full'):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        total_num = len(names_hr)
        if half_mode == 'head':
            names_hr = names_hr[0:(total_num // 2)]
        elif half_mode == 'tail':
            names_hr = names_hr[(total_num // 2):total_num]


        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s == 1:
                    if self.args.task == 'predenoise' :#or self.args.task =='denoise':
                        names_lr[si].append(os.path.join(
                            self.dir_lr, ('X{}_noise'+self.args.noiselevel+'/{}{}').format(
                                s, filename, self.ext[1]
                            )))
                    else:
                        names_lr[si].append(os.path.join(
                            self.dir_lr, 'X{}/{}{}'.format(
                                s, filename, self.ext[1]
                            )))
                else:
                    names_lr[si].append(os.path.join(
                        self.dir_lr, 'X{}/{}x{}{}'.format(
                            s, filename, s, self.ext[1]
                        )
                    ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, f2=None, verbose=True, gtimg = None):
        if not os.path.isfile(f):
            _img = imageio.imread(img)
            with open(f, 'wb') as _f:
                pickle.dump(_img, _f)
        if self.mask_type == 'neuf':
            if ((f2 is not None) and (not os.path.isfile(f2))) or ext.find('reset') >= 0:
                if f2 is not None:
                    _img = imageio.imread(img)
                    masks, masks_bin = self._compute_masks(_img)
                    np.savez(
                        f2,
                        masks=masks)
                    np.savez(
                        f2.replace('.npz', '_bool.npz'),
                        masks_bool=masks_bin.astype(bool))
            else:
                if f2 is not None:
                    if (os.path.isfile(f2)):
                        if (not os.path.isfile(f2.replace('.npz', '_bool.npz'))):
                            npys = np.load((f2))
                            masks_bin = npys['masks_bin'].astype(bool)
                            np.savez(
                                f2.replace('.npz', '_bool.npz'),
                                masks_bool=masks_bin)
        elif self.mask_type =='neuf_gt':
            if f2 is not None:
                f3 = f2.replace('.npz','_gt.npz')
            else:
                f3 = None
            if ((f3 is not None) and (not os.path.isfile(f3))) or ext.find('reset') >= 0:
                if f3 is not None:
                    _img = imageio.imread(img)
                    masks, masks_bin = self._compute_masks(_img)
                    masks_bin = self.downmask(masks_bin, self.args.scale[0])
                    np.savez(
                        f3,
                        masks=masks)
                    np.savez(
                        f3.replace('.npz', '_bool.npz'),
                        masks_bool=masks_bin.astype(bool))
        elif (f2 is not None) and (self.args.compute_grads):
            f3 = f2.replace('bin', 'bin_grads').replace('.npz', '_grads' + str(
                self.args.h_hsize) + '.npz')
            #print(f3)
            if (f3 is not None) and (not os.path.isfile(f3)):
                print("computing raisr stats", f3)
                _img = imageio.imread(img)
                grads = self._compute_grads(_img)
                np.savez(
                    f3,
                    grads=grads)
        elif (f2 is not None) and (self.mask_type == 'raisr') and (self.args.pre_raisr):
            if self.args.from_regression:
                if self.args.Qstr ==2:
                    f3 = f2.replace('X1_noise'+self.args.noiselevel, 'X1').replace('X1', 'X1_noise'+self.args.noiselevel).replace('_mask.npz',
                                                                                            '_SR822' + self.args.postfix + '.npz')
                else:
                    f3 = f2.replace('X1_noise'+self.args.noiselevel,'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                            '_SR833'+self.args.postfix+'.npz')
            else:
                f3 = f2.replace('.npz', (str(self.args.Qstr) + str(self.args.Qcohe) + str(self.args.Qangle) + str(self.args.h_hsize) + '_raisr.npz'))
            if (f3 is not None) and (not os.path.isfile(f3)) and (not self.args.predict_groups):
                print("computing raisr cache", f3)
                _img = imageio.imread(img)
                buckets = self.get_raisr_buckets(_img)
                np.savez(
                    f3,
                    buckets=buckets)


    def downmask(self, masks, scale):
        #print(masks.shape)

        H, W, C = masks.shape
        h = (H // scale) * scale
        w = (W //scale) * scale
        masks_cut = masks[:h,:w,:]
        masks_new = masks_cut.reshape((H//scale, scale, W//scale, scale, C))
        masks_new = masks_new[:,2,:,2,:]

        return masks_new

    def _compute_masks(self, _img, step=1):
        lr = (rgb2yuv(_img.astype(np.float32) / 255.)[:, :, 0])
        #lr = gaussian_filter(lr, sigma=0.8)
        gx, gy = np.gradient(lr)
        H, W = gx.shape
        p_hsize = self.h_hsize
        masks_bin = np.zeros((H, W, 9), dtype=np.float)
        masks = np.zeros((H, W), dtype=np.float)

        for i1 in range(p_hsize, H - p_hsize, step):
            for j1 in range(p_hsize, W - p_hsize, step):
                hash_idx1 = (slice(i1 - p_hsize, i1 + p_hsize + 1), slice(j1 - p_hsize, j1 + p_hsize + 1))
                patchX = gx[hash_idx1]
                patchY = gy[hash_idx1]
                theta, lamda, u = self.grad_patch(patchX, patchY)
                x1 = (0 if lamda < 0.2 else 1)
                if x1 == 1:
                    if (theta < 0.125):
                        x1 = 1
                    elif theta < 0.25:
                        x1 = 2
                    elif theta < 0.375:
                        x1 = 3
                    elif theta < 0.5:
                        x1 = 4
                    elif theta < 0.625:
                        x1 = 5
                    elif theta < 0.75:
                        x1 = 6
                    elif theta < 0.875:
                        x1 = 7
                    else:
                        x1 = 8
                masks_bin[i1, j1, x1] = 1
                if (i1 < (2 * p_hsize)):
                    masks_bin[i1 - p_hsize, j1, x1] = 1
                    if (j1 < (2 * p_hsize))and():
                        masks_bin[i1 - p_hsize, j1 - p_hsize, x1] = 1
                    if (j1 >= (H - 2 * p_hsize)):
                        masks_bin[i1 - p_hsize, j1 + p_hsize, x1] = 1
                if (i1 >= (H - 2 * p_hsize)):
                    masks_bin[i1 + p_hsize, j1, x1] = 1
                    if (j1 < (2 * p_hsize)):
                        masks_bin[i1 + p_hsize, j1 - p_hsize, x1] = 1
                    if (j1 >= (H - 2 * p_hsize)):
                        masks_bin[i1 + p_hsize, j1 + p_hsize, x1] = 1
                if (j1 < (2 * p_hsize)):
                    masks_bin[i1, j1 - p_hsize, x1] = 1
                if (j1 >= (H - 2 * p_hsize - 1)):
                    masks_bin[i1, j1 + p_hsize, x1] = 1
                masks[i1, j1] = x1

        return masks, masks_bin

    def _compute_grads(self, _img, step=1):
        lr = (sc.rgb2ycbcr(_img).astype(np.float32) / 255.)[:, :, 0]
        #lr = gaussian_filter(lr, sigma=0.8)
        gx, gy = np.gradient(lr)
        H, W = gx.shape
        p_hsize = self.h_hsize
        #masks_bin = np.zeros((H, W, 9), dtype=np.float)
        grads = np.zeros((H, W,3), dtype=np.float)

        for i1 in range(p_hsize, H - p_hsize, step):
            for j1 in range(p_hsize, W - p_hsize, step):
                hash_idx1 = (slice(i1 - p_hsize, i1 + p_hsize + 1), slice(j1 - p_hsize, j1 + p_hsize + 1))
                patchX = gx[hash_idx1]
                patchY = gy[hash_idx1]
                theta, lamda, u = self.grad_patch(patchX, patchY)
                grads[i1,j1,0] = theta
                grads[i1, j1, 1] = lamda
                grads[i1, j1, 2] = u

        return grads

    def grad_patch(self, patch_x, patch_y):
        gx = patch_x.ravel()
        gy = patch_y.ravel()
        G = np.vstack((gx, gy)).T
        x = np.dot(G.T, G)
        x = x / (patch_x.size)
        w, v = np.linalg.eig(x)
        index = w.argsort()[::-1]
        w = w[index]
        v = v[:, index]
        theta = atan2(v[1, 0], v[0, 0])
        if theta < 0:
            theta = theta + pi
        theta = theta / pi
        lamda = sqrt(abs(w[0]))
        u = (sqrt(abs(w[0])) - sqrt(abs(w[1]))) / ((sqrt(abs(w[0]))) + sqrt(abs(w[1])) + 0.00000000000000001)
        return theta, lamda, u

    def get_raisr_buckets(self, in_img, step=1, ratio=1):
        lr_blur = in_img
        lr = (sc.rgb2ycbcr(lr_blur).astype(np.float32) / 255.)[:, :, 0]
        in_GX, in_GY = np.gradient(lr)
        H, W = lr.shape
        p_hsize = self.h_hsize
        buckets = np.zeros((H, W), dtype=np.int32)
        for i1 in range(p_hsize, H - p_hsize, step):
            for j1 in range(p_hsize, W - p_hsize, step):
                hash_idx1 = (slice(i1 - p_hsize, i1 + p_hsize + 1), slice(j1 - p_hsize, j1 + p_hsize + 1))
                filter_idx1 = (slice(i1 - p_hsize, i1 + p_hsize + 1), slice(j1 - p_hsize, j1 + p_hsize + 1))
                patch = in_img[filter_idx1]
                patchX = in_GX[hash_idx1]
                patchY = in_GY[hash_idx1]
                theta, lamda, u = hash_table(patchX, patchY, None, self.Qangle, self.Qstr, self.Qcohe, self.stre,
                                             self.cohe)
                j = (theta * self.Qstr * self.Qcohe) + lamda * self.Qcohe + u
                jx = np.int(j)
                buckets[i1, j1] = jx
                if (i1 < (2 * p_hsize)):
                    buckets[i1 - p_hsize, j1] = jx
                    if (j1 < (2 * p_hsize)):
                        buckets[i1 - p_hsize, j1 - p_hsize] = jx
                    if (j1 >= (H - 2 * p_hsize)):
                        buckets[i1 - p_hsize, j1 + p_hsize] = jx
                if (i1 >= (H - 2 * p_hsize)):
                    buckets[i1 + p_hsize, j1] = jx
                    if (j1 < (2 * p_hsize)):
                        buckets[i1 + p_hsize, j1 - p_hsize] = jx
                    if (j1 >= (H - 2 * p_hsize)):
                        buckets[i1 + p_hsize, j1 + p_hsize] = jx
                if (j1 < (2 * p_hsize)):
                    buckets[i1, j1 - p_hsize] = jx
                if (j1 >= (H - 2 * p_hsize - 1)):
                    buckets[i1, j1 + p_hsize] = jx

        return buckets
    def into_groups(self,mask):
        H,W = mask.shape
        temp = mask
        mask_group = np.zeros((H,W,3),dtype=np.int32)
        mask_group[:,:,2] = temp % self.Qcohe
        temp = temp // self.Qcohe
        mask_group[:,:,1] = temp % self.Qstr
        temp = temp // self.Qstr
        mask_group[:,:,0] = temp

        return mask_group

    def merge_grads(self,grads, stre, cohe):
        H, W, C = grads.shape
        grad_map = np.zeros((H, W), dtype=np.int32)
        for i in range(H):
            for j in range(W):
                tempu = grads[i, j, 0] * self.args.Qangle
                if tempu < 0:
                    tempu = 0
                if tempu > self.args.Qangle - 1:
                    tempu = self.args.Qangle - 1
                lamda =  np.searchsorted(stre, grads[i, j, 1])
                mu = np.searchsorted(cohe, grads[i, j, 2])
                grad_map[i,j] = (tempu * self.Qstr * self.Qcohe) + lamda * self.Qcohe + mu


        return grad_map

    def drop_mask(self,mask):
        temp_mask = mask
        H,W,C = mask.shape
        for x in range(H):
            for y in range(W):
                replace_prob = random.random()
                if replace_prob < self.args.drop_mask_prob:
                    source_group = int(temp_mask[x,y,0])
                    if self.trans_prob.sum(1)[source_group]==0:
                        temp_mask[x,y,0] = float(np.random.choice(self.args.Qcohe*self.args.Qangle*self.args.Qstr,1)[0])
                    else:
                        temp_mask[x, y, 0] = float(np.random.choice(self.args.Qcohe*self.args.Qangle*self.args.Qstr, 1, p=self.trans_prob[source_group,:])[0])
        if self.args.debug:
            print('replaced mask generated')
            print(temp_mask.shape)
            print(temp_mask[10:15,10:15,:])
        return temp_mask


    def __getitem__(self, idx):
        if self.args.use_stats:
            lr, hr, mask,mask_grads, filename = self._load_file(idx)
        elif self.args.use_real and (self.name in ['SIDD','SIDDVAL','NAM']):
            lr, hr, mask, noisemap, filename = self._load_file(idx)
        else:
            lr, hr, mask, filename = self._load_file(idx)

        if self.args.debug:
        #    if self.args.compute_grads:
           print('got item')
           print(lr.shape, hr.shape,mask.shape)
        if self.args.predict_groups:
            mask = self.into_groups(mask)
        if (not(self.args.compute_grads and ((not self.train) or lr.shape[0]*lr.shape[1]<2000*2000))) and (not(self.args.predict_groups and ((not self.train) or lr.shape[0]*lr.shape[1]<2000*2000))):
            if (self.mask_type == 'raisr') and (not self.args.pre_raisr):
                pair = self.get_patch(lr, hr)
                mask = self.get_raisr_buckets(pair[1])
            else:
                if self.args.pre_raisr and (not self.args.compute_grads)and (not self.args.predict_groups):
                    mask = np.expand_dims(mask, -1)
                if self.args.use_stats:
                    lr, hr,mask_grads, mask = self.get_patch(lr, hr,mask_grad=mask_grads, mask=mask)
                elif self.args.use_real and (self.name in ['SIDD','SIDDVAL','NAM']):
                    lr, hr, noisemap, mask = self.get_patch(lr, hr, mask_grad=noisemap, mask=mask)
                else:
                    lr,hr,mask = self.get_patch(lr, hr, mask=mask)

                if self.args.debug:
                    print('data augmented!')
                    print(mask.shape)
                    print(mask.dtype)
                    print(mask[10:20,10:20,0])
                if self.train and self.args.drop_mask:
                    mask = self.drop_mask(mask)
                pair= (lr,hr,mask)
        else:
            pair = (lr,hr,mask)
        if self.args.debug:
            print(pair[0].shape,pair[1].shape, pair[2].shape)
        if self.args.real_isp or self.args.use_real:
            if not (self.name in ['SIDD','SIDDVAL','NAM']):
                lr, noisemap = AddRealNoise((hr.astype(np.float32))/255., self.CRF_para, self.iCRF_para, self.I_gl, self.B_gl, self.I_inv_gl, self.B_inv_gl)
                lr = lr * 255.
            else:
                if self.args.model == 'unet_noisemap':
                    _, noisemap = AddRealNoise(hr, self.CRF_para, self.iCRF_para, self.I_gl, self.B_gl, self.I_inv_gl, self.B_inv_gl)
            if self.args.real_isp:
                mask = np.concatenate((mask,noisemap),axis=-1)
            if self.args.debug:
                print('noisemap and concatenate shape: ', noisemap.shape, noisemap.dtype, mask.shape, mask.dtype)
        pair = (lr,hr,mask)

        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        if self.args.debug:
            print('channels set!')
            print(pair[0].shape, pair[1].shape,pair[2].shape)
        if (self.mask_type == 'raisr') and (not self.args.pre_raisr):
            pair.append(mask)
        if (self.args.pre_raisr):
            if not (self.args.compute_grads or self.args.predict_groups):
                pair[2] = pair[2].squeeze(-1)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        if self.args.use_stats:
            np_transpose = np.ascontiguousarray(mask_grads.transpose((2, 0, 1)))
            mask_grads = torch.from_numpy(np_transpose).float()
            lr_grads = torch.cat((pair_t[0],mask_grads),0)

        if self.args.use_real:
            np_transpose = np.ascontiguousarray(noisemap.transpose((2, 0, 1)))
            noisemap = torch.from_numpy(np_transpose).float()
            lr_grads = torch.cat((pair_t[0],noisemap),0)

        if self.args.debug:
            if self.args.compute_grads or self.args.predict_groups:
                print(pair_t[0].shape, pair_t[1].shape,pair_t[2].shape)
        if self.args.use_stats or self.args.use_real:
            return lr_grads, pair_t[1], pair_t[2], filename
        else:
            return pair_t[0], pair_t[1], pair_t[2], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        f_mask = self.masks_lr[self.idx_scale][idx]
        if self.args.debug:
            print('loading:   ',f_hr,f_mask)
        if self.args.use_real and (self.name in ['SIDD','SIDDVAL','NAM']):
            npys = np.load((f_mask.replace('X1_noise'+self.args.noiselevel,'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                '_SRnoisemap.npz')), allow_pickle=True)
            noisemap = npys['noisemap']
        if self.args.no_gt:
            masks_bin = np.zeros((5,5),dtype=np.int32)
        elif self.args.compute_grads:
            npys = np.load(f_mask.replace('X1_noise'+self.args.noiselevel,'X1'), allow_pickle=True)
            masks_bin = npys['grads']

            if self.args.use_stats:
                npys = np.load((f_mask.replace('bin_grads','bin').replace('bin','bin_grads').replace('X1_noise'+self.args.noiselevel,'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                        '_mask_grads4.npz')), allow_pickle=True)
                masks_grad = npys['grads']
        elif self.args.pre_raisr:
            if self.args.use_stats:
                npys = np.load((f_mask.replace('bin_grads','bin').replace('bin','bin_grads').replace('X1_noise'+self.args.noiselevel,'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                        '_mask_grads4.npz')), allow_pickle=True)
                masks_grad = npys['grads']
            if self.args.predict_groups:
                npys = np.load((f_mask.replace('bin_grads','bin').replace('bin','bin_grads').replace('X1_noise'+self.args.noiselevel, 'X1').replace('_mask.npz',
                                                                           '_GT833.npz')), allow_pickle=True)
                masks_bin = npys['grads']
            elif self.args.gtmask:
                npys = np.load(
                    (f_mask.replace('bin_grads', 'bin').replace('bin', 'bin_grads').replace('X1_noise'+self.args.noiselevel,'X1').replace('_mask.npz',
                                                                                                      '_GT833.npz')),
                    allow_pickle=True)
                masks_bin = npys['grads']
            elif self.args.task == 'denoise':
                if self.args.from_regression:
                    if self.args.Qangle == 4:
                        npys = np.load(
                            (f_mask.replace('X1_noise'+self.args.noiselevel, 'X1').replace('X1', 'X1_noise'+self.args.noiselevel).replace('_mask.npz',
                                                                                                    '_SR433.npz')),
                            allow_pickle=True)
                    elif self.args.Qstr ==3:
                        npys = np.load((f_mask.replace('X1_noise'+self.args.noiselevel, 'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                                                                                   '_SR833.npz')), allow_pickle=True)
                    else:
                        npys = np.load((f_mask.replace('X1_noise'+self.args.noiselevel, 'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                            '_SR.npz')), allow_pickle=True)
                    masks_bin = npys['grads']
                else:
                    npys = np.load((f_mask.replace('X1_noise'+self.args.noiselevel, 'X1').replace('.npz', (
                            str(self.args.Qstr) + str(self.args.Qcohe) + str(self.args.Qangle) + str(
                        self.args.h_hsize) + '_raisr.npz'))), allow_pickle=True)
                    masks_bin = npys['buckets']
            else:
                if self.args.from_regression:
                    if self.args.gtmask:
                        npys = np.load((f_mask.replace('bin_grads','bin').replace('bin','bin_grads').replace('_mask.npz',
                                                       '_GT833.npz')), allow_pickle=True)
                    elif self.args.noisemask:
                        npys = np.load((f_mask.replace('bin_grads','bin').replace('bin','bin_grads').replace('X1_noise'+self.args.noiselevel, 'X1').replace('X1','X1_noise'+self.args.noiselevel).replace('_mask.npz',
                                                       '_NOI833.npz')), allow_pickle=True)
                    elif self.args.Qangle ==4:
                        npys = np.load((f_mask.replace('_mask.npz',
                                                       '_SR433' + self.args.postfix + '.npz')), allow_pickle=True)
                    elif self.args.Qstr == 3:
                        npys = np.load((f_mask.replace('_mask.npz',
                                                       '_SR833'+self.args.postfix+'.npz')), allow_pickle=True)
                    elif self.args.Qstr == 2:
                        npys = np.load((f_mask.replace('_mask.npz',
                                                       '_SR822' + self.args.postfix + '.npz')), allow_pickle=True)
                    else:
                        npys = np.load((f_mask.replace('_mask.npz',
                        '_SR.npz')), allow_pickle=True)
                    masks_bin = npys['grads']
                else:
                    npys = np.load((f_mask.replace('.npz', (
                    str(self.args.Qstr) + str(self.args.Qcohe) + str(self.args.Qangle)+ str(self.args.h_hsize) + '_raisr.npz'))), allow_pickle=True)
                    masks_bin = npys['buckets']
        elif self.args.mask_type =='neuf_gt':
            npys = np.load((f_mask.replace('.npz', '_gt_bool.npz')))
            masks_bin = npys['masks_bool']
        else:
            npys = np.load((f_mask.replace('.npz', '_bool.npz')))
            masks_bin = npys['masks_bool']
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        with open(f_hr, 'rb') as _f:
            hr = pickle.load(_f)
        with open(f_lr, 'rb') as _f:
            lr = pickle.load(_f)
        if self.args.use_stats:
            return lr, hr, masks_bin, masks_grad, filename
        elif self.args.use_real and (self.name in ['SIDD','SIDDVAL','NAM']):
            return lr, hr, masks_bin, noisemap, filename
        else:
            return lr, hr, masks_bin, filename




    def get_patch(self, lr, hr, mask_grad=None, mask=None):
        scale = self.scale[0]
        if self.train:
            if (mask is None) :#or (not self.args.split):
                lr, hr = common.get_patch(
                    lr, hr,
                    patch_size=self.args.patch_size,
                    scale=scale,
                    multi=(len(self.scale) > 1),
                    input_large=self.input_large
                )
            else:
                if mask_grad is not None:
                    lr, hr, mask_grad, mask = common.get_patch(
                        lr, hr,mask_grad, mask,
                        patch_size=self.args.patch_size,
                        scale=scale,
                        multi=(len(self.scale) > 1),
                        input_large=self.input_large
                    )
                else:
                    lr, hr, mask = common.get_patch(
                        lr, hr, mask,
                        patch_size=self.args.patch_size,
                        scale=scale,
                        multi=(len(self.scale) > 1),
                        input_large=self.input_large
                    )

            if not (self.args.no_augment or (self.args.pre_raisr and (not self.args.from_regression))):
                if (mask is None)or (not self.args.split):
                    lr, hr = common.augment(lr, hr)
                else:
                    if self.args.debug:
                        print('data augmenting!')
                        print(mask.shape)
                        print(mask.dtype)
                        print(mask[10:20, 10:20, 0])
                    lr, hr, mask = common.augment(lr, hr, mask,argss=self.args)
        else:

            if lr.shape[1]>800:
                if not self.args.compute_grads:
                    split_size=400
                    overlap_size=100
                    if mask is None:
                        return lr,hr
                    elif mask_grad is not None:
                        return lr,hr,mask_grad,mask
                    else:
                        return lr,hr,mask
                else:

                    if self.args.n_layers ==1:
                        psize = 300
                    elif self.args.n_layers ==2:
                        psize = 300
                    elif self.args.n_layers == 3:
                        psize=300
                    else:
                        psize = 300

                    if (mask is None) or (not self.args.split):
                        lr, hr = common.get_patch(
                            lr, hr,
                            patch_size=psize,
                            scale=scale,
                            multi=(len(self.scale) > 1),
                            input_large=self.input_large,
                            fix=True

                        )
                    else:

                        if mask_grad is not None:
                            lr, hr, mask_grad, mask = common.get_patch(
                                lr, hr, mask_grad, mask,
                                patch_size=psize,
                                scale=scale,
                                multi=(len(self.scale) > 1),
                                input_large=self.input_large,
                                fix=True
                            )
                        else:
                            lr, hr, mask = common.get_patch(
                                lr, hr, mask,
                                patch_size=psize,
                                scale=scale,
                                multi=(len(self.scale) > 1),
                                input_large=self.input_large,
                                fix=True
                            )

            else:
                ih, iw = lr.shape[:2]
                hr = hr[0:ih * scale, 0:iw * scale]
        if mask is None:
            return lr, hr
        else:
            if mask_grad is not None:
                return lr, hr, mask_grad, mask
            else:
                return lr, hr, mask

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)





