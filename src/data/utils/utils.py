import binascii
from math import ceil, sqrt, floor
from skimage.measure import compare_psnr
import numpy as np
import cv2
import os
import math
from scipy.optimize import curve_fit

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.bin', '.tiff'
                  ]
RAW_EXTENSIONS = ['.bin', '.raw', '.dng', '.RAW', '.RAWPLAIN16', '.RAWMIPI']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_raw_file(filename):
    return any(filename.endswith(extension) for extension in RAW_EXTENSIONS)


def make_image_list(dir, is_raw=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_raw:
                if is_raw_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
            else:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def is_greyimage(im):
    if len(im.shape) == 2:
        return True
    #x = abs(im[:, :, 0] - im[:, :, 1])
    #print(x)
    #y = np.linalg.norm(x)
    #print(y)
    #if y == 0:
    #    return True
    else:
        return False


def readimg(path):
    if os.path.splitext(path)[-1] == '.bin':
        img = read_bin(path, np.uint8)
    else:
        img = cv2.imread(path)
        if is_greyimage(img):
            return img[:, :, 0]
    return img


def modcrop(img, num):
    if len(img.shape) == 2:
        h, w = img.shape
        h -= h % num
        w -= w % num
        img = img[:h, :w]
    else:
        h, w, _ = img.shape
        h -= h % num
        w -= w % num
        img = img[:h, :w, :]
    return img


def checkimage(image, window_name='test'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


# Python opencv library (cv2) cv2.COLOR_BGR2YCrCb has different parameters with MATLAB color convertion.
# In order to have a fair comparison with the benchmark, we wrote these functions.
def BGR2YCbCr(im):
    mat = np.array([[24.966, 128.553, 65.481], [112, -74.203, -37.797], [-18.214, -93.786, 112]])
    mat = mat.T
    offset = np.array([[[16, 128, 128]]])
    if im.dtype == 'uint8':
        mat = mat / 255
        out = np.dot(im, mat) + offset
        out = np.clip(out, 0, 255)
        out = np.rint(out).astype('uint8')
    elif im.dtype == 'float':
        mat = mat / 255
        offset = offset / 255
        out = np.dot(im, mat) + offset
        out = np.clip(out, 0, 1)
    else:
        assert False
    return out


def YCbCr2BGR(im):
    mat = np.array([[24.966, 128.553, 65.481], [112, -74.203, -37.797], [-18.214, -93.786, 112]])
    mat = mat.T
    mat = np.linalg.inv(mat)
    offset = np.array([[[16, 128, 128]]])
    if im.dtype == 'uint8':
        mat = mat * 255
        out = np.dot((im - offset), mat)
        out = np.clip(out, 0, 255)
        out = np.rint(out).astype('uint8')
    elif im.dtype == 'float':
        mat = mat * 255
        offset = offset / 255
        out = np.dot((im - offset), mat)
        out = np.clip(out, 0, 1)
    else:
        assert False
    return out


def rgb2y(rgb):
    out = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return out


def im2double(im):
    out = im.copy()
    if im.dtype == 'uint8':
        out = out.astype('float') / 255.
    elif im.dtype == 'uint16':
        out = out.astype('float') / 65535.
    elif im.dtype == 'float':
        print("input already float!")
    else:
        assert False
    out = np.clip(out, 0, 1)
    return out


def im2int(im, out_type):
    out = np.clip(im.copy(), 0, 1)
    if out_type == 'uint8':
        out = (out * 255.).astype('uint8')
    elif out_type == 'uint16':
        out = (out * 65535.).astype('uint16')
    else:
        pass
    return out


def read_bin(filename, datatype):
    '''
        read the binary file (bayer or quad) for Sony simulator
    :param filename:
        filename for the binary file
    :param datatype:
    :return:
        img: a [m, n] image of quad or bayer pattern. The image is 10 bit w/ max value of 1023.
            We don't do any normalization here.
    '''
    with open(filename, 'rb') as f:
        myArr = binascii.hexlify(f.read())
        w = int(myArr[2:4] + myArr[0:2], 16)
        h = int(myArr[6:8] + myArr[4:6], 16)

        img = np.zeros([h, w], dtype=datatype)
        pixelIdx = 8

        # read img data
        for i in range(h):
            if i % 1000 == 0:
                print(str(round(i * 100 / h)) + '%')
            for j in range(w):
                byte = myArr[pixelIdx + 2: pixelIdx + 4] + myArr[pixelIdx: pixelIdx + 2]
                b = int(byte, 16)
                pixel = int(b / 4.)
                if pixel < 0:
                    pixel = 0
                img[i, j] = pixel
                pixelIdx += 4

    return img


def read_raw(path, img_type, img_shape):
    imgData = np.fromfile(path, dtype=img_type)
    imgData = np.reshape(imgData, img_shape)

    print(np.min(imgData), np.max(imgData))
    return imgData


def read_vivo_mipi10bayer(path, img_type, img_shape):
    imgData = np.fromfile(path, dtype='uint8')
    h, w = img_shape
    unpack_w = int(w / 4 * 5)
    imgData = np.reshape(imgData, (h, unpack_w))
    outData_tmp = imgData.copy()
    obj = np.linspace(5, unpack_w, unpack_w / 5) - 1
    # print(obj)

    outData_tmp = np.delete(outData_tmp, obj, axis=1).astype('uint16')
    outData_offset = np.zeros(outData_tmp.shape, dtype='uint16')

    imgData_offset = imgData[:, 4::5]
    outData_offset[:, 0::4] = np.bitwise_and(imgData_offset, 3)
    outData_offset[:, 1::4] = np.bitwise_and(imgData_offset, 12) / 4
    outData_offset[:, 2::4] = np.bitwise_and(imgData_offset, 48) / 16
    outData_offset[:, 3::4] = np.bitwise_and(imgData_offset, 192) / 64
    outData_offset = outData_offset.astype('uint16')
    # print(outData_offset[:5, :20])
    # print(imgData[:5, 4:25:5])
    outData_tmp = outData_tmp * 4 + outData_offset

    outData = np.zeros(img_shape, dtype=img_type)
    outData[:, :] = outData_tmp[:, :]
    outData[0::4, 1::4] = outData_tmp[0::4, 2::4]
    outData[0::4, 2::4] = outData_tmp[0::4, 1::4]
    outData[1::4, 0::4] = outData_tmp[2::4, 0::4]
    outData[2::4, 0::4] = outData_tmp[1::4, 0::4]
    outData[1::4, 3::4] = outData_tmp[2::4, 3::4]
    outData[2::4, 3::4] = outData_tmp[1::4, 3::4]
    outData[3::4, 1::4] = outData_tmp[3::4, 2::4]
    outData[3::4, 2::4] = outData_tmp[3::4, 1::4]
    outData[1::4, 1::4] = outData_tmp[2::4, 2::4]
    outData[2::4, 2::4] = outData_tmp[1::4, 1::4]
    outData[1::4, 2::4] = outData_tmp[2::4, 1::4]
    outData[2::4, 1::4] = outData_tmp[1::4, 2::4]

    print(np.min(outData), np.max(outData))
    return outData


def read_mi_mipi10bayer(path, img_type, img_shape):
    imgData = np.fromfile(path, dtype='uint8')
    h, w = img_shape
    unpack_w = int(w / 4 * 5)
    imgData = np.reshape(imgData, (h, unpack_w))
    outData = imgData.copy()
    obj = np.linspace(5, unpack_w, unpack_w / 5) - 1
    # print(obj)

    outData = np.delete(outData, obj, axis=1).astype('uint16')
    outData_offset = np.zeros(outData.shape, dtype='uint16')

    imgData_offset = imgData[:, 4::5]
    outData_offset[:, 0::4] = np.bitwise_and(imgData_offset, 3)
    outData_offset[:, 1::4] = np.bitwise_and(imgData_offset, 12) / 4
    outData_offset[:, 2::4] = np.bitwise_and(imgData_offset, 48) / 16
    outData_offset[:, 3::4] = np.bitwise_and(imgData_offset, 192) / 64
    outData_offset = outData_offset.astype('uint16')
    # print(outData_offset[:5, :20])
    # print(imgData[:5, 4:25:5])
    outData = outData * 4 + outData_offset

    print(np.min(outData), np.max(outData))
    return outData


def cal_psnr(a, b, crop=0, maxval=1.0):
    """Computes PSNR on a cropped version of a,b"""
    if len(a.shape) == 1:
        return compare_psnr(a, b)

    if crop > 0:
        if len(a.shape) == 2:
            aa = a[crop:-crop, crop:-crop]
            bb = b[crop:-crop, crop:-crop]
        else:
            aa = a[crop:-crop, crop:-crop, :]
            bb = b[crop:-crop, crop:-crop, :]
    else:
        aa = a
        bb = b

    # d = np.mean(np.square(aa - bb))
    # d = -10 * np.log10(d / (maxval * maxval))
    d = compare_psnr(aa, bb)
    return d


def data_aug(img, mode=0):
    # data augmentation
    img = img.copy()
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def my_sqrt(array):
    array[array < 0] = 0
    res = np.sqrt(array)
    return res


def disk_filter(rad):
    crad = ceil(rad - 0.5)
    crad_grid = np.array(range(crad * 2 + 1)) - rad
    x, y = np.meshgrid(crad_grid, crad_grid)
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(y))
    # print(maxxy)
    # print(minxy)
    m1 = (rad ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) \
         + (rad ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * my_sqrt(rad ** 2 - (maxxy + 0.5) ** 2)
    # print(m1)
    m2 = (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) \
         + (rad ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * my_sqrt(rad ** 2 - (maxxy - 0.5) ** 2)
    # print(m2)
    sgrid = (rad ** 2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) + 0.25 * (
            np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) - (maxxy - 0.5) * (m2 - m1) + (
                     m1 - minxy + 0.5)) * ((((rad ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) & (
            rad ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)) | (
                                                    (minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad))))
    # print(sgrid)
    sgrid += ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < rad ** 2)
    # print(sgrid)
    sgrid[crad, crad] = min(np.pi * rad ** 2, np.pi / 2)
    # print(sgrid)

    if crad > 0 and rad > crad - 0.5 and rad ** 2 < (crad - 0.5) ** 2 + 0.25:
        m1 = sqrt(rad ** 2 - (crad - 0.5) ** 2)
        m1n = m1 / rad
        sg0 = 2 * (rad ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] -= sg0
        sgrid[crad, 2 * crad - 1] -= sg0
        sgrid[crad, 1] -= sg0
        sgrid[1, crad] -= sg0
    sgrid[crad, crad] = min(sgrid[crad, crad], 1)
    h = sgrid / np.sum(sgrid)
    return h


def circle_blur(img, rad=2, value=0.5):
    kernel = disk_filter(rad)
    kernel *= value
    kernel[rad, rad] += (1 - value)
    # checkimage(kernel*25.)
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def gradient_45(array):
    H, W = array.shape
    array_large = np.zeros((H+2, W+2), dtype=array.dtype)
    array_large[1:-1, 1:-1] = array
    array_large[1:-1, 0] = array[:, 0]
    array_large[1:-1, -1] = array[:, -1]
    array_large[0, 1:-1] = array[0, :]
    array_large[-1, 1:-1] = array[-1, :]
    array_large[0, 0] = array[0, 0]
    array_large[0, -1] = array[0, -1]
    array_large[-1, 0] = array[-1, 0]
    array_large[-1, -1] = array[-1, -1]
    g45 = array_large[0:-2, 0:-2] - array_large[2:, 2:]
    g135 = array_large[0:-2, 2:] - array_large[2:, 0:-2]

    return g45, g135


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ReadImg(filename):
    img = cv2.imread(filename)
    img = img[:, :, ::-1] / 255.0
    img = np.array(img).astype('float32')

    return img


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


####################################################
#################### noise model ###################
####################################################

def func(x, a):
    return np.power(x, a)


def CRF_curve_fit(I, B):
    popt, pcov = curve_fit(func, I, B)
    return popt


def CRF_function_transfer(x, y):
    para = []
    for crf in range(201):
        temp_x = np.array(x[crf, :])
        temp_y = np.array(y[crf, :])
        para.append(CRF_curve_fit(temp_x, temp_y))
    return para


def mosaic_bayer(rgb, pattern, noiselevel):
    w, h, c = rgb.shape
    if pattern == 1:
        num = [1, 2, 0, 1]
    elif pattern == 2:
        num = [1, 0, 2, 1]
    elif pattern == 3:
        num = [2, 1, 1, 0]
    elif pattern == 4:
        num = [0, 1, 1, 2]
    elif pattern == 5:
        return rgb

    mosaic = np.zeros((w, h, 3))
    mask = np.zeros((w, h, 3))
    B = np.zeros((w, h))

    B[0:w:2, 0:h:2] = rgb[0:w:2, 0:h:2, num[0]]
    B[0:w:2, 1:h:2] = rgb[0:w:2, 1:h:2, num[1]]
    B[1:w:2, 0:h:2] = rgb[1:w:2, 0:h:2, num[2]]
    B[1:w:2, 1:h:2] = rgb[1:w:2, 1:h:2, num[3]]

    gauss = np.random.normal(0, noiselevel / 255., (w, h))
    gauss = gauss.reshape(w, h)
    B = B + gauss

    return (B, mask, mosaic)


def ICRF_Map(Img, I, B):
    w, h, c = Img.shape
    output_Img = Img.copy()
    prebin = I.shape[0]
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    for i in range(w):
        for j in range(h):
            for k in range(c):
                temp = output_Img[i, j, k]
                start_bin = 0
                if temp > min_tiny_bin:
                    start_bin = math.floor(temp / tiny_bin - 1) - 1
                for b in range(start_bin, prebin):
                    tempB = B[b]
                    if tempB >= temp:
                        index = b
                        if index > 0:
                            comp1 = tempB - temp
                            comp2 = temp - B[index - 1]
                            if comp2 < comp1:
                                index = index - 1
                        output_Img[i, j, k] = I[index]
                        break

    return output_Img


def CRF_Map(Img, I, B):
    w, h, c = Img.shape
    output_Img = Img.copy()
    prebin = I.shape[0]
    tiny_bin = 9.7656e-04
    min_tiny_bin = 0.0039
    for i in range(w):
        for j in range(h):
            for k in range(c):
                temp = output_Img[i, j, k]

                if temp < 0:
                    temp = 0
                    Img[i, j, k] = 0
                elif temp > 1:
                    temp = 1
                    Img[i, j, k] = 1
                start_bin = 0
                if temp > min_tiny_bin:
                    start_bin = math.floor(temp / tiny_bin - 1) - 1

                for b in range(start_bin, prebin):
                    tempB = I[b]
                    if tempB >= temp:
                        index = b
                        if index > 0:
                            comp1 = tempB - temp
                            comp2 = temp - B[index - 1]
                            if comp2 < comp1:
                                index = index - 1
                        output_Img[i, j, k] = B[index]
                        break
    return output_Img


def CRF_Map_opt(Img, popt):
    w, h, c = Img.shape
    output_Img = Img.copy()

    output_Img = func(output_Img, *popt)
    return output_Img


def Demosaic(B_b, pattern):
    B_b = B_b * 255
    B_b = B_b.astype(np.uint16)

    if pattern == 1:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerGB2BGR)
    elif pattern == 2:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerGR2BGR)
    elif pattern == 3:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerBG2BGR)
    elif pattern == 4:
        lin_rgb = cv2.demosaicing(B_b, cv2.COLOR_BayerRG2BGR)
    elif pattern == 5:
        lin_rgb = B_b

    lin_rgb = lin_rgb[:, :, ::-1] / 255.
    return lin_rgb


def AddNoiseMosai(x, CRF_para, iCRF_para, I, B, Iinv, Binv, sigma_s, sigma_c, crf_index, pattern, opt=1):
    w, h, c = x.shape
    temp_x = CRF_Map_opt(x, iCRF_para[crf_index])

    sigma_s = np.reshape(sigma_s, (1, 1, c))
    noise_s_map = np.multiply(sigma_s, temp_x)
    noise_s = np.random.randn(w, h, c) * noise_s_map
    temp_x_n = temp_x + noise_s

    noise_c = np.zeros((w, h, c))
    for chn in range(3):
        noise_c[:, :, chn] = np.random.normal(0, sigma_c[chn], (w, h))

    temp_x_n = temp_x_n + noise_c
    temp_x_n = np.clip(temp_x_n, 0.0, 1.0)
    temp_x_n = CRF_Map_opt(temp_x_n, CRF_para[crf_index])

    if opt == 1:
        temp_x = CRF_Map_opt(temp_x, CRF_para[crf_index])

    B_b_n = mosaic_bayer(temp_x_n[:, :, ::-1], pattern, 0)[0]
    lin_rgb_n = Demosaic(B_b_n, pattern)
    result = lin_rgb_n
    if opt == 1:
        B_b = mosaic_bayer(temp_x[:, :, ::-1], pattern, 0)[0]
        lin_rgb = Demosaic(B_b, pattern)
        diff = lin_rgb_n - lin_rgb
        result = x + diff

    return result


def AddRealNoise(image, CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl):
    sigma_s = np.random.uniform(0.0, 0.16, (3,))
    sigma_c = np.random.uniform(0.0, 0.06, (3,))
    CRF_index = np.random.choice(201)
    pattern = np.random.choice(4) + 1
    noise_img = AddNoiseMosai(image, CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl, sigma_s, sigma_c, CRF_index,
                              pattern, 0)
    noise_level = sigma_s * np.power(image, 0.5) + sigma_c

    return noise_img, noise_level


# for debug only
if __name__ == '__main__':
    # img_path = '/home/SENSETIME/chenruobing/project/python-raisr/imx586_benchmark_dataset/corridor/corridor_48M_bayer.bin'
    # img = read_bin(img_path, np.uint8)
    # img = raw2rgb(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # checkimage(img)
    img_path = '../data/kodak/kodim04.png'
    img = cv2.imread(img_path)
    checkimage(img)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]
    y = y.astype('float') / 255.
    checkimage(y)
    g45, g135 = gradient_45(y)
    checkimage(g45)
    checkimage(g135)
    print(y.shape, g45.shape, g135.shape)
    pass



