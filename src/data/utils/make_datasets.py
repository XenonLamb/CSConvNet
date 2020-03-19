import os
import cv2

from .utils import *


if __name__ == '__main__':
    path = '/data/datasets/Demosaic-net_input_output/'
    raw_subpath = 'meanraw'
    png_subpath = 'out_tiff'
    raw_outsubpath = 'bayer_aug'
    quad_subpath = 'quad_aug'
    raw_path = os.path.join(path, raw_subpath)
    png_path = os.path.join(path, png_subpath)
    quad_path = os.path.join(path, quad_subpath)
    raw_outpath = os.path.join(path, raw_outsubpath)
    bp = 'gbrg'
    os.makedirs(quad_path, exist_ok=True)
    os.makedirs(raw_outpath, exist_ok=True)

    img_list = []
    path_list = os.listdir(raw_path)
    for img_count, img_path in enumerate(path_list):
        if os.path.splitext(img_path)[-1] == '.raw':
            img_list.append(img_path)

    h_list = [0, 1, 1, 0, 1, 0, 0, 1]
    w_list = [1, 1, 1, 1, 0, 0, 0, 0]

    print(img_list)
    for img_path in img_list:
        image_name = os.path.splitext(img_path)[0]
        raw = read_raw(os.path.join(raw_path, img_path), 'uint16', (3000, 4000))
        png_name = image_name + '_mit0.tiff'

        png = cv2.imread(os.path.join(png_path, png_name), -1)
        png = cv2.cvtColor(png, cv2.COLOR_BGR2RGB)

        for i in range(8):
            bayer_name = image_name + '_%d.png' % i
            png_tmp = data_aug(png, i)
            raw_tmp = data_aug(raw, i)

            h = h_list[i]
            w = w_list[i]

            raw_tmp = raw_tmp[h:, w:]
            png_tmp = png_tmp[h:, w:, :]
            raw_tmp = modcrop(raw_tmp, 4)
            png_tmp = modcrop(png_tmp, 4)
            print(raw_tmp.shape)
            print(png_tmp.shape)
            assert(raw_tmp.shape == png_tmp.shape[:2])

            quad = rgb2quadraw(png_tmp, bp)
            print(np.max(quad))
            cv2.imwrite(os.path.join(quad_path, bayer_name), quad)
            cv2.imwrite(os.path.join(raw_outpath, bayer_name), raw_tmp)

        #cv2.imwrite('./test.png', raw*64)
        #cv2.imwrite('./res.png', png*64)
        #quad = rgb2quadraw(png, bp)
        #print(np.max(quad))
        #cv2.imwrite(os.path.join(quad_path, bayer_name), quad)
        #cv2.imwrite(os.path.join(raw_outpath, bayer_name), raw)





