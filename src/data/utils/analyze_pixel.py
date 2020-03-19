import pickle
import os
import cv2
import argparse
import numpy as np
from termcolor import cprint

from utils import *
from process_functions import hash_table, grad_patch
from quantization_values import quantization_value_1

parser = argparse.ArgumentParser()

parser.add_argument('--test_folder', type=str, default='../data/kodak', help="folder path for testing dataset")
parser.add_argument('--model-type', type=str, choices=['sr', 'demo', 'remo', 'remo_sony'], default='remo',
                    help="Analyzed filter model type, must in 'sr', 'demo' and 'remo'")
parser.add_argument('--model-folder', type=str, default='../models/remo_models/gt_y7x7',
                    help="folder path for training results")
parser.add_argument('--pattern', type=str, choices=['rggb', 'gbrg', 'grbg', 'bggr'], default='rggb')
parser.add_argument('--quantization45', type=str, default='yy7x7_45', help="")

args = parser.parse_args()

with open(os.path.join(args.model_folder, "filters"), "rb") as fp:
    filters = pickle.load(fp)
with open(os.path.join(args.model_folder, "args"), "rb") as ap:
    model_args = pickle.load(ap)

Qangle = model_args.theta_num
Qstrength = model_args.s_num
Qcoherence = model_args.mu_num
bp = args.pattern
quant_type = model_args.quantization
f_hsize = model_args.f_hsize
h_hsize = model_args.h_hsize
s_thres = quantization_value_1[model_args.quantization]['s_thres']
mu_thres = quantization_value_1[model_args.quantization]['mu_thres']
s_thres_45 = quantization_value_1[args.quantization45]['s_thres']
mu_thres_45 = quantization_value_1[args.quantization45]['mu_thres']


if __name__ == '__main__':
    img_list = make_image_list(args.test_folder)
    print(s_thres)
    print(mu_thres)

    cprint('begin to process images:', 'red')
    for img_count, img_path in enumerate(img_list[:1]):
        img_name = os.path.split(img_path)[-1]
        print('\r', end='')
        print('' * 60, end='')

        print('\r Processing ' + str(img_count + 1) + '/' + str(len(img_list)) + ' image (' + img_name + ')')
        img_uint8 = cv2.imread(img_path)
        rgb_uint8 = cv2.cvtColor(img_uint8.copy(), cv2.COLOR_BGR2RGB)
        img = im2double(img_uint8)
        rgb = im2double(rgb_uint8)
        quad_raw = rgb2quadraw(rgb, bayer_pattern=bp)
        raw = rgb2raw(rgb, bayer_pattern=bp)
        y_rough = quad2y_linear(quad_raw, bp)

        gx, gy = np.gradient(y_rough)
        g45, g135 = gradient_45(y_rough)
        out_img = quadraw2raw(quad_raw, bp)

        H, W = quad_raw.shape
        p_hsize = max(h_hsize, f_hsize)
        rgb_patch = np.zeros((1 + 2 * h_hsize, 1 + 2 * h_hsize, 3))
        res_img = np.zeros((5 + 4 * h_hsize, 5 + 4 * h_hsize, 3))

        for h in range(p_hsize, H - p_hsize, 55):
            for w in range(p_hsize, W - p_hsize, 55):
                idx1 = (slice(h - h_hsize, h + h_hsize + 1), slice(w - h_hsize, w + h_hsize + 1))
                patchX = gx[idx1]
                patchY = gy[idx1]
                patch45 = g45[idx1]
                patch135 = g135[idx1]
                theta1, lamda1, u1 = hash_table(patchX, patchY, None, Qangle, Qstrength, Qcoherence, s_thres, mu_thres)
                theta2, lamda2, u2 = hash_table(patch45, patch135, None, Qangle, Qstrength, Qcoherence, s_thres_45, mu_thres_45)
                #print(int(theta1), int(lamda1), int(u1), int(theta2), int(lamda2), int(u2))

                t1, s1, c1 = grad_patch(patchX, patchY)
                t2, s2, c2 = grad_patch(patch45, patch135)
                #print(t1, t2, s1, s2, c1, c2)
                print(s1/s2)

                if s1/s2 < 0.35:

                    y_patch = y_rough[idx1]
                    gt_patch = raw[idx1]
                    rgb_patch[:, :, 0] = img[:, :, 0][idx1]
                    rgb_patch[:, :, 1] = img[:, :, 1][idx1]
                    rgb_patch[:, :, 2] = img[:, :, 2][idx1]

                    res_img[4+2*h_hsize:, :1+2*h_hsize, :] = rgb_patch
                    res_img[:1+2*h_hsize, :1+2*h_hsize, 0] = y_patch
                    res_img[:1 + 2 * h_hsize, :1 + 2 * h_hsize, 1] = y_patch
                    res_img[:1 + 2 * h_hsize, :1 + 2 * h_hsize, 2] = y_patch
                    res_img[:1 + 2 * h_hsize, 4+2*h_hsize:, 0] = gt_patch
                    res_img[:1 + 2 * h_hsize, 4 + 2 * h_hsize:, 1] = gt_patch
                    res_img[:1 + 2 * h_hsize, 4 + 2 * h_hsize:, 2] = gt_patch

                    window_name = 'theta:%d_s:%d_mu:%d||theta:%d_s:%d_mu:%d' % (theta1, lamda1, u1, theta2, lamda2, u2)
                    cv2.namedWindow(window_name, 0)
                    cv2.resizeWindow(window_name, 1440, 800)
                    checkimage(res_img, window_name)
                    cv2.destroyWindow(window_name)







