import os
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model1-path', type=str, default='../models/remo_models/dj_quad2y2_yy7x7_gbrg_8_8_8.model', help="folder path for training results")
parser.add_argument('--model2-folder', type=str, default='../models/remo_hdr_models/yy7x7_noise_gbrg_8_8_8', help="folder path for training results")
parser.add_argument('--export-path', type=str, default='../export/filterTablesLinearYDJ.cpp', help="folder path to save testing results")

args = parser.parse_args()

with open(args.model1_path, "rb") as fm:
    model1 = pickle.load(fm)

model1.display_model_info()

theta_num = model1.Qtheta
s_num = model1.Qstre
mu_num = model1.Qcohe
filters1 = model1.filters
filter_size = model1.f_hsize * 2 + 1

with open(args.export_path, 'w') as ep:
    ep.write("float filterTablesLinearYDJ[%d][%d][%d][%d][%d]={" % (theta_num, s_num, mu_num, 16, filter_size ** 2))
    for angle in range(theta_num):
        for s in range(s_num):
            for mu in range(mu_num):
                for pos in range(16):
                    mu_tmp = mu
                    filter_index = angle * s_num * mu_num + s * mu_num + mu_tmp
                    filter_i = filters1[pos, filter_index]
                    for f in filter_i:
                        ep.write("%.8ff," % f)
                    ep.write('\n')
    ep.write("};")


