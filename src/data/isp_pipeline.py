from data.utils.utils import *
import numba as nb

class RawProcess:
    def __init__(self, digital_gain=None, rgb_gain=None, ccm=None, bp='rggb',):
        self.digital_gain_value = digital_gain
        if bp not in ['rggb', 'gbrg', 'grbg', 'bggr']:
            raise RuntimeError("bayer pattern incorrect!")
        self.pattern = bp
        self.rgb_gain = rgb_gain
        self.ccm = ccm

    def digital_gain(self, img):
        if self.digital_gain_value is None:
            return img / np.random.normal(0.8, 0.1)
        return img * self.digital_gain_value

    def inverse_digital_gain(self, img):
        if self.digital_gain_value is None:
            return img * np.random.normal(0.8, 0.1)
        return img / self.digital_gain_value

    def white_balance(self, img):
        if self.rgb_gain is None:
            r_gain = np.random.uniform(1.9, 2.4)
            b_gain = np.random.uniform(1.5, 1.9)
            g_gain = 1
        else:
            r_gain, g_gain, b_gain = self.rgb_gain
        out = np.zeros_like(img)
        out[:, :, 0] = img[:, :, 0] * r_gain
        out[:, :, 1] = img[:, :, 1] * g_gain
        out[:, :, 2] = img[:, :, 2] * b_gain
        return out

    @staticmethod
    def _wb_transform(data, thres, gain):
        out = np.zeros_like(data)
        if gain <= 1:
            out[:] = data / gain
        else:
            data_max = np.max(data)
            data = data / data_max
            mask = data > thres
            out[mask == False] = data[mask == False] / gain
            alpha = ((data[mask] - thres) / (1 - thres)) ** 2
            out[mask] = (1 - alpha) * (data[mask] / gain) + alpha * data[mask]
        return out

    def inverse_white_balance(self, img):
        if self.rgb_gain is None:
            r_gain = np.random.uniform(1.9, 2.4)
            b_gain = np.random.uniform(1.5, 1.9)
            g_gain = 1
        else:
            r_gain, g_gain, b_gain = self.rgb_gain
        threshold = 0.9
        out = np.zeros_like(img)
        out[:, :, 0] = self._wb_transform(img[:, :, 0], threshold, r_gain)
        out[:, :, 1] = self._wb_transform(img[:, :, 1], threshold, g_gain)
        out[:, :, 2] = self._wb_transform(img[:, :, 2], threshold, b_gain)
        return out

    def color_correction(self, img):
        if self.ccm is None:
            return img
        out = np.dot(img, self.ccm)
        return out

    def inverse_color_correction(self, img):
        if self.ccm is None:
            return img
        ccm_inv = np.linalg.inv(self.ccm)
        out = np.dot(img, ccm_inv)
        return out

    @staticmethod
    def gamma_correction(img):
        eps = 1e-8
        out = np.power(np.maximum(img, eps), 1/2.2)
        return out

    @staticmethod
    def inverse_gamma_correction(img):
        eps = 1e-8
        out = np.power(np.maximum(img, eps), 2.2)
        return out

    @staticmethod
    def tone_mapping(img):
        return 3 * img**2 - 2 * img**3

    @staticmethod
    def inverse_tone_mapping(img):
        return 0.5 - np.sin(np.arcsin(1 - 2 * img) / 3)

    @staticmethod
    def rgb2yuv(rgb):
        out = np.zeros_like(rgb)
        out[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        out[:, :, 1] = -0.1687 * rgb[:, :, 0] - 0.3313 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] + 0.5
        out[:, :, 2] = 0.5 * rgb[:, :, 0] - 0.4187 * rgb[:, :, 1] - 0.0813 * rgb[:, :, 2] + 0.5
        return out

    @staticmethod
    def yuv2rgb(yuv):
        out = np.zeros_like(yuv)
        out[:, :, 0] = yuv[:, :, 0] + 1.402 * (yuv[:, :, 2] - 0.5)
        out[:, :, 1] = yuv[:, :, 0] - 0.34414 * (yuv[:, :, 1] - 0.5) - 0.71414 * (yuv[:, :, 2] - 0.5)
        out[:, :, 2] = yuv[:, :, 0] + 1.772 * (yuv[:, :, 1] - 0.5)
        return out


    def rgb2bayer(self, img):
        h_shift, w_shift = {'rggb': [0, 0],
                            'grbg': [0, 1],
                            'gbrg': [1, 0],
                            'bggr': [1, 1]}[self.pattern]
        h, w, c = img.shape
        out = img[:, :, 1].copy()
        out[h_shift:h:2, w_shift:w:2] = img[h_shift:h:2, w_shift:w:2, 0]
        out[1 - h_shift:h:2, 1 - w_shift:w:2] = img[1 - h_shift:h:2, 1 - w_shift:w:2, 2]
        return out

    def rgb2quadbayer(self, img):
        h_shift, w_shift = {'rggb': [0, 0],
                            'grbg': [0, 2],
                            'gbrg': [2, 0],
                            'bggr': [2, 2]}[self.pattern]
        h, w, c = img.shape
        out = img[:, :, 1].copy()
        out[h_shift:h:4, w_shift:w:4] = img[h_shift:h:4, w_shift:w:4, 0]
        out[h_shift + 1:h:4, w_shift:w:4] = img[h_shift + 1:h:4, w_shift:w:4, 0]
        out[h_shift:h:4, w_shift + 1:w:4] = img[h_shift:h:4, w_shift + 1:w:4, 0]
        out[h_shift + 1:h:4, w_shift + 1:w:4] = img[h_shift + 1:h:4, w_shift + 1:w:4, 0]
        h_shift = 2 - h_shift
        w_shift = 2 - w_shift
        out[h_shift:h:4, w_shift:w:4] = img[h_shift:h:4, w_shift:w:4, 2]
        out[h_shift + 1:h:4, w_shift:w:4] = img[h_shift + 1:h:4, w_shift:w:4, 2]
        out[h_shift:h:4, w_shift + 1:w:4] = img[h_shift:h:4, w_shift + 1:w:4, 2]
        out[h_shift + 1:h:4, w_shift + 1:w:4] = img[h_shift + 1:h:4, w_shift + 1:w:4, 2]
        return out

    def bayer2rgb(self, bayer):
        cvt_func = {'rggb': cv2.COLOR_BAYER_RG2BGR,
                    'grbg': cv2.COLOR_BAYER_GR2BGR,
                    'gbrg': cv2.COLOR_BAYER_GB2BGR,
                    'bggr': cv2.COLOR_BAYER_BG2BGR,}[self.pattern]
        img = bayer.copy()
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        img_type = bayer.dtype.type
        if img_type != 'uint8' and img_type != 'uint16':
            raw_uint16 = im2int(img, 'uint16')
            rgb_uint16 = cv2.cvtColor(raw_uint16, cvt_func)
            rgb = im2double(rgb_uint16)
        else:
            rgb = cv2.cvtColor(img, cvt_func)
        return rgb

    def bayer_addgain(self, bayer):
        if self.rgb_gain is None:
            r_gain = np.random.uniform(1.9, 2.4)
            b_gain = np.random.uniform(1.5, 1.9)
            g_gain = 1
        else:
            r_gain, g_gain, b_gain = self.rgb_gain
        h_shift, w_shift = {'rggb': [0, 0],
                            'grbg': [0, 1],
                            'gbrg': [1, 0],
                            'bggr': [1, 1]}[self.pattern]
        h, w = bayer.shape
        out = bayer.copy() * g_gain
        out[h_shift:h:2, w_shift:w:2] = bayer[h_shift:h:2, w_shift:w:2] * r_gain
        out[1 - h_shift:h:2, 1 - w_shift:w:2] = bayer[1 - h_shift:h:2, 1 - w_shift:w:2] * b_gain
        if bayer.dtype == 'uint8':
            out = np.clip(out, 0, 255)
        elif bayer.dtype == 'uint16':
            out = np.clip(out, 0, 65535)
        out = out.astype(bayer.dtype)
        return out

    def quadbayer_addgain(self, quadbayer):
        if self.rgb_gain is None:
            r_gain = np.random.uniform(1.9, 2.4)
            b_gain = np.random.uniform(1.5, 1.9)
            g_gain = 1
        else:
            r_gain, g_gain, b_gain = self.rgb_gain
        h_shift, w_shift = {'rggb': [0, 0],
                            'grbg': [0, 2],
                            'gbrg': [2, 0],
                            'bggr': [2, 2]}[self.pattern]
        h, w = quadbayer.shape
        out = quadbayer.copy() * g_gain
        out[h_shift:h:4, w_shift:w:4] = quadbayer[h_shift:h:4, w_shift:w:4] * r_gain
        out[h_shift + 1:h:4, w_shift:w:4] = quadbayer[h_shift + 1:h:4, w_shift:w:4] * r_gain
        out[h_shift:h:4, w_shift + 1:w:4] = quadbayer[h_shift:h:4, w_shift + 1:w:4] * r_gain
        out[h_shift + 1:h:4, w_shift + 1:w:4] = quadbayer[h_shift + 1:h:4, w_shift + 1:w:4] * r_gain
        h_shift = 2 - h_shift
        w_shift = 2 - w_shift
        out[h_shift:h:4, w_shift:w:4] = quadbayer[h_shift:h:4, w_shift:w:4] * b_gain
        out[h_shift + 1:h:4, w_shift:w:4] = quadbayer[h_shift + 1:h:4, w_shift:w:4] * b_gain
        out[h_shift:h:4, w_shift + 1:w:4] = quadbayer[h_shift:h:4, w_shift + 1:w:4] * b_gain
        out[h_shift + 1:h:4, w_shift + 1:w:4] = quadbayer[h_shift + 1:h:4, w_shift + 1:w:4] * b_gain
        if quadbayer.dtype == 'uint8':
            out = np.clip(out, 0, 255)
        elif quadbayer.dtype == 'uint16':
            out = np.clip(out, 0, 65535)
        out = out.astype(quadbayer.dtype)
        return out

    def quad2bayer(self, quadbayer):
        bayer_rough = quadbayer.copy()
        bayer_rough[0::4, 1::4] = quadbayer[0::4, 2::4]
        bayer_rough[0::4, 2::4] = quadbayer[0::4, 1::4]
        bayer_rough[3::4, 1::4] = quadbayer[3::4, 2::4]
        bayer_rough[3::4, 2::4] = quadbayer[3::4, 1::4]
        bayer_rough[1::4, 0::4] = quadbayer[2::4, 0::4]
        bayer_rough[2::4, 0::4] = quadbayer[1::4, 0::4]
        bayer_rough[1::4, 3::4] = quadbayer[2::4, 3::4]
        bayer_rough[2::4, 3::4] = quadbayer[1::4, 3::4]
        if self.pattern == 'rggb' or self.pattern == 'bggr':
            bayer_rough[1::4, 1::4] = quadbayer[2::4, 2::4]
            bayer_rough[2::4, 2::4] = quadbayer[1::4, 1::4]
        else:
            bayer_rough[1::4, 2::4] = quadbayer[2::4, 1::4]
            bayer_rough[2::4, 1::4] = quadbayer[1::4, 2::4]
        return bayer_rough

    def quad2rgb_linear(self, quadbayer):
        H, W = quadbayer.shape
        # print(H, W)
        bayer_large = np.zeros((H + 2, W + 2))
        bayer_large[1:-1, 1:-1] = quadbayer
        bayer_large[0, :] = bayer_large[3, :]
        bayer_large[-1, :] = bayer_large[-4, :]
        bayer_large[:, 0] = bayer_large[:, 3]
        bayer_large[:, -1] = bayer_large[:, -4]

        out = np.zeros((H, W, 3))

        if self.pattern == 'rggb' or self.pattern == 'bggr':
            out[0::4, 0::4, 0] = quadbayer[0::4, 0::4]
            out[0::4, 1::4, 0] = quadbayer[0::4, 1::4]
            out[1::4, 0::4, 0] = quadbayer[1::4, 0::4]
            out[1::4, 1::4, 0] = quadbayer[1::4, 1::4]

            out[2::4, 2::4, 2] = quadbayer[2::4, 2::4]
            out[2::4, 3::4, 2] = quadbayer[2::4, 3::4]
            out[3::4, 2::4, 2] = quadbayer[3::4, 2::4]
            out[3::4, 3::4, 2] = quadbayer[3::4, 3::4]

            out[0::4, 2::4, 1] = quadbayer[0::4, 2::4]
            out[0::4, 3::4, 1] = quadbayer[0::4, 3::4]
            out[1::4, 2::4, 1] = quadbayer[1::4, 2::4]
            out[1::4, 3::4, 1] = quadbayer[1::4, 3::4]
            out[2::4, 0::4, 1] = quadbayer[2::4, 0::4]
            out[2::4, 1::4, 1] = quadbayer[2::4, 1::4]
            out[3::4, 0::4, 1] = quadbayer[3::4, 0::4]
            out[3::4, 1::4, 1] = quadbayer[3::4, 1::4]
            # R
            out[0::4, 2::4, 0] = bayer_large[1:-1:4, 2::4] / 3 * 2 + bayer_large[1:-1:4, 5::4] / 3 * 1
            out[0::4, 3::4, 0] = bayer_large[1:-1:4, 2::4] / 3 * 1 + bayer_large[1:-1:4, 5::4] / 3 * 2
            out[1::4, 2::4, 0] = bayer_large[2::4, 2::4] / 3 * 2 + bayer_large[2::4, 5::4] / 3 * 1
            out[1::4, 3::4, 0] = bayer_large[2::4, 2::4] / 3 * 1 + bayer_large[2::4, 5::4] / 3 * 2

            out[2::4, 0::4, 0] = bayer_large[2::4, 1:-1:4] / 3 * 2 + bayer_large[5::4, 1:-1:4] / 3 * 1
            out[2::4, 1::4, 0] = bayer_large[2::4, 2::4] / 3 * 2 + bayer_large[5::4, 2::4] / 3 * 1
            out[3::4, 0::4, 0] = bayer_large[2::4, 1:-1:4] / 3 * 1 + bayer_large[5::4, 1:-1:4] / 3 * 2
            out[3::4, 1::4, 0] = bayer_large[2::4, 2::4] / 3 * 1 + bayer_large[5::4, 2::4] / 3 * 2

            out[2::4, 2::4, 0] = bayer_large[2::4, 2::4] / 9 * 4 + bayer_large[2::4, 5::4] / 9 * 2 + bayer_large[5::4,
                                                                                                     2::4] / 9 * 2 + bayer_large[
                                                                                                                     5::4,
                                                                                                                     5::4] / 9 * 1
            out[2::4, 3::4, 0] = bayer_large[2::4, 2::4] / 9 * 2 + bayer_large[2::4, 5::4] / 9 * 4 + bayer_large[5::4,
                                                                                                     2::4] / 9 * 1 + bayer_large[
                                                                                                                     5::4,
                                                                                                                     5::4] / 9 * 2
            out[3::4, 2::4, 0] = bayer_large[2::4, 2::4] / 9 * 2 + bayer_large[2::4, 5::4] / 9 * 1 + bayer_large[5::4,
                                                                                                     2::4] / 9 * 4 + bayer_large[
                                                                                                                     5::4,
                                                                                                                     5::4] / 9 * 2
            out[3::4, 3::4, 0] = bayer_large[2::4, 2::4] / 9 * 1 + bayer_large[2::4, 5::4] / 9 * 2 + bayer_large[5::4,
                                                                                                     2::4] / 9 * 2 + bayer_large[
                                                                                                                     5::4,
                                                                                                                     5::4] / 9 * 4
            # B
            out[0::4, 0::4, 2] = bayer_large[0:-2:4, 0:-2:4] / 9 * 4 + bayer_large[3::4, 0:-2:4] / 9 * 2 + bayer_large[
                                                                                                           0:-2:4,
                                                                                                           3::4] / 9 * 2 + bayer_large[
                                                                                                                           3::4,
                                                                                                                           3::4] / 9 * 1
            out[0::4, 1::4, 2] = bayer_large[0:-2:4, 0:-2:4] / 9 * 2 + bayer_large[3::4, 0:-2:4] / 9 * 1 + bayer_large[
                                                                                                           0:-2:4,
                                                                                                           3::4] / 9 * 4 + bayer_large[
                                                                                                                           3::4,
                                                                                                                           3::4] / 9 * 2
            out[1::4, 0::4, 2] = bayer_large[0:-2:4, 0:-2:4] / 9 * 2 + bayer_large[3::4, 0:-2:4] / 9 * 4 + bayer_large[
                                                                                                           0:-2:4,
                                                                                                           3::4] / 9 * 1 + bayer_large[
                                                                                                                           3::4,
                                                                                                                           3::4] / 9 * 2
            out[1::4, 1::4, 2] = bayer_large[0:-2:4, 0:-2:4] / 9 * 1 + bayer_large[3::4, 0:-2:4] / 9 * 2 + bayer_large[
                                                                                                           0:-2:4,
                                                                                                           3::4] / 9 * 2 + bayer_large[
                                                                                                                           3::4,
                                                                                                                           3::4] / 9 * 4

            out[0::4, 2::4, 2] = bayer_large[0:-2:4, 3::4] / 3 * 2 + bayer_large[3::4, 3::4] / 3 * 1
            out[0::4, 3::4, 2] = bayer_large[0:-2:4, 4::4] / 3 * 2 + bayer_large[3::4, 4::4] / 3 * 1
            out[1::4, 2::4, 2] = bayer_large[0:-2:4, 3::4] / 3 * 1 + bayer_large[3::4, 3::4] / 3 * 2
            out[1::4, 3::4, 2] = bayer_large[0:-2:4, 4::4] / 3 * 1 + bayer_large[3::4, 4::4] / 3 * 2

            out[2::4, 0::4, 2] = bayer_large[3::4, 0:-2:4] / 3 * 2 + bayer_large[3::4, 3::4] / 3 * 1
            out[2::4, 1::4, 2] = bayer_large[3::4, 0:-2:4] / 3 * 1 + bayer_large[3::4, 3::4] / 3 * 2
            out[3::4, 0::4, 2] = bayer_large[4::4, 0:-2:4] / 3 * 2 + bayer_large[4::4, 3::4] / 3 * 1
            out[3::4, 1::4, 2] = bayer_large[4::4, 0:-2:4] / 3 * 1 + bayer_large[4::4, 3::4] / 3 * 2
            # G
            out[0::4, 0::4, 1] = bayer_large[0:-2:4, 1:-1:4] / 6 * 2 + bayer_large[3::4, 1:-1:4] / 6 * 1 + bayer_large[
                                                                                                           1:-1:4,
                                                                                                           0:-2:4] / 6 * 2 + bayer_large[
                                                                                                                             1:-1:4,
                                                                                                                             3::4] / 6 * 1
            out[0::4, 1::4, 1] = bayer_large[0:-2:4, 2:-1:4] / 6 * 2 + bayer_large[3::4, 2:-1:4] / 6 * 1 + bayer_large[
                                                                                                           1:-1:4,
                                                                                                           0:-2:4] / 6 * 1 + bayer_large[
                                                                                                                             1:-1:4,
                                                                                                                             3::4] / 6 * 2
            out[1::4, 0::4, 1] = bayer_large[0:-2:4, 1:-1:4] / 6 * 1 + bayer_large[3::4, 1:-1:4] / 6 * 2 + bayer_large[
                                                                                                           2:-1:4,
                                                                                                           0:-2:4] / 6 * 2 + bayer_large[
                                                                                                                             2:-1:4,
                                                                                                                             3::4] / 6 * 1
            out[1::4, 1::4, 1] = bayer_large[0:-2:4, 2:-1:4] / 6 * 1 + bayer_large[3::4, 2:-1:4] / 6 * 2 + bayer_large[
                                                                                                           2:-1:4,
                                                                                                           0:-2:4] / 6 * 1 + bayer_large[
                                                                                                                             2:-1:4,
                                                                                                                             3::4] / 6 * 2

            out[2::4, 2::4, 1] = bayer_large[2::4, 3::4] / 6 * 2 + bayer_large[5::4, 3::4] / 6 * 1 + bayer_large[3::4,
                                                                                                     2::4] / 6 * 2 + bayer_large[
                                                                                                                     3::4,
                                                                                                                     5::4] / 6 * 1
            out[2::4, 3::4, 1] = bayer_large[2::4, 4::4] / 6 * 2 + bayer_large[5::4, 4::4] / 6 * 1 + bayer_large[3::4,
                                                                                                     2::4] / 6 * 1 + bayer_large[
                                                                                                                     3::4,
                                                                                                                     5::4] / 6 * 2
            out[3::4, 2::4, 1] = bayer_large[2::4, 3::4] / 6 * 1 + bayer_large[5::4, 3::4] / 6 * 2 + bayer_large[4::4,
                                                                                                     2::4] / 6 * 2 + bayer_large[
                                                                                                                     4::4,
                                                                                                                     5::4] / 6 * 1
            out[3::4, 3::4, 1] = bayer_large[2::4, 4::4] / 6 * 1 + bayer_large[5::4, 4::4] / 6 * 2 + bayer_large[4::4,
                                                                                                     2::4] / 6 * 1 + bayer_large[
                                                                                                                     4::4,
                                                                                                                     5::4] / 6 * 2

        if self.pattern == 'gbrg' or self.pattern == 'grbg':
            out[0::4, 0::4, 1] = quadbayer[0::4, 0::4]
            out[0::4, 1::4, 1] = quadbayer[0::4, 1::4]
            out[1::4, 0::4, 1] = quadbayer[1::4, 0::4]
            out[1::4, 1::4, 1] = quadbayer[1::4, 1::4]
            out[2::4, 2::4, 1] = quadbayer[2::4, 2::4]
            out[2::4, 3::4, 1] = quadbayer[2::4, 3::4]
            out[3::4, 2::4, 1] = quadbayer[3::4, 2::4]
            out[3::4, 3::4, 1] = quadbayer[3::4, 3::4]

            out[0::4, 2::4, 2] = quadbayer[0::4, 2::4]
            out[0::4, 3::4, 2] = quadbayer[0::4, 3::4]
            out[1::4, 2::4, 2] = quadbayer[1::4, 2::4]
            out[1::4, 3::4, 2] = quadbayer[1::4, 3::4]
            out[2::4, 0::4, 0] = quadbayer[2::4, 0::4]
            out[2::4, 1::4, 0] = quadbayer[2::4, 1::4]
            out[3::4, 0::4, 0] = quadbayer[3::4, 0::4]
            out[3::4, 1::4, 0] = quadbayer[3::4, 1::4]
            # R
            out[0::4, 0::4, 0] = bayer_large[0:-2:4, 1:-1:4] / 3 * 2 + bayer_large[3::4, 1:-1:4] / 3 * 1
            out[0::4, 1::4, 0] = bayer_large[0:-2:4, 2:-1:4] / 3 * 2 + bayer_large[3::4, 2:-1:4] / 3 * 1
            out[1::4, 0::4, 0] = bayer_large[0:-2:4, 1:-1:4] / 3 * 1 + bayer_large[3::4, 1:-1:4] / 3 * 2
            out[1::4, 1::4, 0] = bayer_large[0:-2:4, 2:-1:4] / 3 * 1 + bayer_large[3::4, 2:-1:4] / 3 * 2

            out[2::4, 2::4, 0] = bayer_large[3::4, 2::4] / 3 * 2 + bayer_large[3::4, 5::4] / 3 * 1
            out[2::4, 3::4, 0] = bayer_large[3::4, 2::4] / 3 * 1 + bayer_large[3::4, 5::4] / 3 * 2
            out[3::4, 2::4, 0] = bayer_large[4::4, 2::4] / 3 * 2 + bayer_large[4::4, 5::4] / 3 * 1
            out[3::4, 3::4, 0] = bayer_large[4::4, 2::4] / 3 * 1 + bayer_large[4::4, 5::4] / 3 * 2

            out[0::4, 2::4, 0] = bayer_large[0:-2:4, 2::4] / 9 * 4 + bayer_large[0:-2:4, 5::4] / 9 * 2 + bayer_large[
                                                                                                         3::4,
                                                                                                         2::4] / 9 * 2 + bayer_large[
                                                                                                                         3::4,
                                                                                                                         5::4] / 9 * 1
            out[0::4, 3::4, 0] = bayer_large[0:-2:4, 2::4] / 9 * 2 + bayer_large[0:-2:4, 5::4] / 9 * 4 + bayer_large[
                                                                                                         3::4,
                                                                                                         2::4] / 9 * 1 + bayer_large[
                                                                                                                         3::4,
                                                                                                                         5::4] / 9 * 2
            out[1::4, 2::4, 0] = bayer_large[0:-2:4, 2::4] / 9 * 2 + bayer_large[0:-2:4, 5::4] / 9 * 1 + bayer_large[
                                                                                                         3::4,
                                                                                                         2::4] / 9 * 4 + bayer_large[
                                                                                                                         3::4,
                                                                                                                         5::4] / 9 * 2
            out[1::4, 3::4, 0] = bayer_large[0:-2:4, 2::4] / 9 * 1 + bayer_large[0:-2:4, 5::4] / 9 * 2 + bayer_large[
                                                                                                         3::4,
                                                                                                         2::4] / 9 * 2 + bayer_large[
                                                                                                                         3::4,
                                                                                                                         5::4] / 9 * 4
            # B
            out[2::4, 0::4, 2] = bayer_large[2::4, 0:-2:4] / 9 * 4 + bayer_large[2::4, 3::4] / 9 * 2 + bayer_large[5::4,
                                                                                                       0:-2:4] / 9 * 2 + bayer_large[
                                                                                                                         5::4,
                                                                                                                         3::4] / 9 * 1
            out[2::4, 1::4, 2] = bayer_large[2::4, 0:-2:4] / 9 * 2 + bayer_large[2::4, 3::4] / 9 * 4 + bayer_large[5::4,
                                                                                                       0:-2:4] / 9 * 1 + bayer_large[
                                                                                                                         5::4,
                                                                                                                         3::4] / 9 * 2
            out[3::4, 0::4, 2] = bayer_large[2::4, 0:-2:4] / 9 * 2 + bayer_large[2::4, 3::4] / 9 * 1 + bayer_large[5::4,
                                                                                                       0:-2:4] / 9 * 4 + bayer_large[
                                                                                                                         5::4,
                                                                                                                         3::4] / 9 * 2
            out[3::4, 1::4, 2] = bayer_large[2::4, 0:-2:4] / 9 * 1 + bayer_large[2::4, 3::4] / 9 * 2 + bayer_large[5::4,
                                                                                                       0:-2:4] / 9 * 2 + bayer_large[
                                                                                                                         5::4,
                                                                                                                         3::4] / 9 * 4

            out[0::4, 0::4, 2] = bayer_large[1:-1:4, 0:-2:4] / 3 * 2 + bayer_large[1:-1:4, 3::4] / 3 * 1
            out[0::4, 1::4, 2] = bayer_large[1:-1:4, 0:-2:4] / 3 * 1 + bayer_large[1:-1:4, 3::4] / 3 * 2
            out[1::4, 0::4, 2] = bayer_large[2:-1:4, 0:-2:4] / 3 * 2 + bayer_large[2:-1:4, 3::4] / 3 * 1
            out[1::4, 1::4, 2] = bayer_large[2:-1:4, 0:-2:4] / 3 * 1 + bayer_large[2:-1:4, 3::4] / 3 * 2

            out[2::4, 2::4, 2] = bayer_large[2::4, 3::4] / 3 * 2 + bayer_large[5::4, 3::4] / 3 * 1
            out[2::4, 3::4, 2] = bayer_large[2::4, 4::4] / 3 * 2 + bayer_large[5::4, 4::4] / 3 * 1
            out[3::4, 2::4, 2] = bayer_large[2::4, 3::4] / 3 * 1 + bayer_large[5::4, 3::4] / 3 * 2
            out[3::4, 3::4, 2] = bayer_large[2::4, 4::4] / 3 * 1 + bayer_large[5::4, 4::4] / 3 * 2
            # G
            out[0::4, 2::4, 1] = bayer_large[0:-2:4, 3::4] / 6 * 2 + bayer_large[3::4, 3:-1:4] / 6 * 1 + bayer_large[
                                                                                                         1:-1:4,
                                                                                                         2:-2:4] / 6 * 2 + bayer_large[
                                                                                                                           1:-1:4,
                                                                                                                           5::4] / 6 * 1
            out[0::4, 3::4, 1] = bayer_large[0:-2:4, 4::4] / 6 * 2 + bayer_large[3::4, 4:-1:4] / 6 * 1 + bayer_large[
                                                                                                         1:-1:4,
                                                                                                         2:-2:4] / 6 * 1 + bayer_large[
                                                                                                                           1:-1:4,
                                                                                                                           5::4] / 6 * 2
            out[1::4, 2::4, 1] = bayer_large[0:-2:4, 3::4] / 6 * 1 + bayer_large[3::4, 3:-1:4] / 6 * 2 + bayer_large[
                                                                                                         2:-1:4,
                                                                                                         2:-2:4] / 6 * 2 + bayer_large[
                                                                                                                           2:-1:4,
                                                                                                                           5::4] / 6 * 1
            out[1::4, 3::4, 1] = bayer_large[0:-2:4, 4::4] / 6 * 1 + bayer_large[3::4, 4:-1:4] / 6 * 2 + bayer_large[
                                                                                                         2:-1:4,
                                                                                                         2:-2:4] / 6 * 1 + bayer_large[
                                                                                                                           2:-1:4,
                                                                                                                           5::4] / 6 * 2

            out[2::4, 0::4, 1] = bayer_large[2::4, 1:-1:4] / 6 * 2 + bayer_large[5::4, 1:-1:4] / 6 * 1 + bayer_large[
                                                                                                         3::4,
                                                                                                         0:-2:4] / 6 * 2 + bayer_large[
                                                                                                                           3::4,
                                                                                                                           3::4] / 6 * 1
            out[2::4, 1::4, 1] = bayer_large[2::4, 2:-1:4] / 6 * 2 + bayer_large[5::4, 2:-1:4] / 6 * 1 + bayer_large[
                                                                                                         3::4,
                                                                                                         0:-2:4] / 6 * 1 + bayer_large[
                                                                                                                           3::4,
                                                                                                                           3::4] / 6 * 2
            out[3::4, 0::4, 1] = bayer_large[2::4, 1:-1:4] / 6 * 1 + bayer_large[5::4, 1:-1:4] / 6 * 2 + bayer_large[
                                                                                                         4::4,
                                                                                                         0:-2:4] / 6 * 2 + bayer_large[
                                                                                                                           4::4,
                                                                                                                           3::4] / 6 * 1
            out[3::4, 1::4, 1] = bayer_large[2::4, 2:-1:4] / 6 * 1 + bayer_large[5::4, 2:-1:4] / 6 * 2 + bayer_large[
                                                                                                         4::4,
                                                                                                         0:-2:4] / 6 * 1 + bayer_large[
                                                                                                                           4::4,
                                                                                                                           3::4] / 6 * 2

        if self.pattern == 'bggr' or self.pattern == 'grbg':
            out = out[:, :, ::-1]

        return out

    def quad2rgb_nn(self):
        pass

    def quad2rgb_mix(self):
        pass

    def forward(self, image):
        assert len(image.shape) == 3
        image_after_dg = self.digital_gain(image)
        image_after_wb = self.white_balance(image_after_dg).clip(0, 1)
        image_after_cc = self.color_correction(image_after_wb)
        #image_after_gamma = self.gamma_correction(image_after_cc)
        #image_after_tm = self.tone_mapping(image_after_gamma)
        image_out = image_after_cc.clip(0, 1)
        return image_out

    def backward(self, image):
        assert len(image.shape) == 3
        #image_before_tm = self.inverse_tone_mapping(image)
        #image_before_gamma = self.inverse_gamma_correction(image_before_tm)
        image_before_cc = self.inverse_color_correction(image)
        image_before_wb = self.inverse_white_balance(image_before_cc)
        image_before_dg = self.inverse_digital_gain(image_before_wb)
        image_out = image_before_dg.clip(0, 1)
        return image_out


if __name__ == '__main__':
    ccm = np.array([[1.44, -0.4, -0.04],
                    [-0.17, 1.35, -0.18],
                    [0, -0.53, 1.53]])
    bp = 'gbrg'
    path = '/data/datasets/raw/result/0731gain'
    for f in os.listdir(path):
        if is_raw_file(f):
            image = read_raw(path + '/' + f, 'uint16', (3456, 4608))
            image = image[::3, ::3]
            raw = image.astype('float') / 1023.
            process = RawProcess(1.25, (2.0, 1., 1.6), ccm.T, 'gbrg')
            rgb = process.bayer2rgb(raw)
            res = process.forward(rgb)
            res = (res[:, :, ::-1])
            checkimage(res)
