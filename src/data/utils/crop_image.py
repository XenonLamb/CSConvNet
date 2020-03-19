import argparse

from .utils import *

kodak_dict = {
    101: ['kodim03', 220, 320, 350, 450],
    201: ['kodim07', 280, 380, 140, 240],
    202: ['kodim01', 300, 400, 400, 500],
    203: ['kodim08',  390, 490, 560, 660],
    204: ['kodim09',  360, 460, 370, 470],
    205: ['kodim12',  160, 260, 210, 310],
    206: ['kodim14',  200, 300, 200, 300],
    207: ['kodim05',  220, 320, 160, 260],
    208: ['kodim20',  200, 300, 150, 250],
    209: ['kodim24',  250, 350, 400, 500],
    301: ['kodim04',  270, 370, 180, 280],
    302: ['kodim06',  160, 260, 70, 170],
    303: ['kodim08',  390, 490, 560, 660],
    304: ['kodim13',  270, 370, 100, 200],
    305: ['kodim18',  260, 360, 240, 340],
    306: ['kodim20',  230, 330, 250, 350],
    307: ['kodim23',  160, 260, 130, 230],
    308: ['kodim24',  50, 150, 310, 410],
    401: ['kodim19',  450, 550, 250, 350],
    402: ['kodim19',  420, 520, 400, 500],
    501: ['kodim03',  100, 200, 450, 550],
    502: ['kodim06',  20, 120, 200, 300],
}

imx586_dict = {
    101: ['parking',  3452, 3552, 3872, 3972],
    102: ['garage',  3732, 3832, 3560, 3660],
    103: ['package',  1772, 1872, 4836, 4936],
    104: ['kanji',  1492, 1592, 4620, 4720],
    105: ['kanji',  4152, 4252, 5712, 5812],
    201: ['parking',  3532, 3632, 3872, 3972],
    202: ['floorplan',  3180, 3280, 2212, 2312],
    203: ['floorplan',  2780, 2880, 4452, 4552],
    204: ['garage',  2460, 2560, 6010, 6110],
    205: ['gate',  2836, 2936, 2760, 2860],
    206: ['parking',  1772, 1872, 1512, 1612],
    301: ['garage',  3620, 3720, 3560, 3660],
    302: ['kanji',  2252, 2352, 2732, 2832],
    401: ['kanji',  1772, 1872, 1292, 1392],
    402: ['kanji',  1892, 1992, 1540, 1640],
    403: ['kanji',  2272, 2372, 1252, 1352],
    404: ['kanji',  2372, 2472, 512, 612],
    501: ['floorplan',  2792, 2892, 3860, 3960],
    502: ['kanji',  1680, 1780, 3264, 3364],
    503: ['corridor',  2800, 2900, 4144, 4244],
    504: ['corridor',  4700, 4800, 2600, 2700],
    505: ['garage',  2112, 2212, 3600, 3700],
    506: ['kanji',  2160, 2260, 1532, 1632],
}

sony_dict = {
    101: ['IMX586proto_Shirts_macbeth',  2752, 2852, 2752, 2852],
    102: ['IMX586proto_Nenpyo0',  3400, 3500, 2560, 2660],
    103: ['IMX586proto_Nenpyo1',  3140, 3240, 2372, 2472],
    104: ['IMX376Proto_BlindOpen',  1552, 1652, 2400, 2500],
    105: ['IMX586proto_BlindClose',  2752, 2852, 3772, 3872],
    106: ['IMX586proto_BlindOpen',  3152, 3252, 3720, 3820],
    107: ['IMX586proto_BlindOpen',  2752, 2852, 3760, 3860],
    201: ['IMX376Proto_BlindClose',  1600, 1700, 2200, 2300],
    202: ['IMX376Proto_BlindOpen',  1680, 1780, 2900, 3000],
    203: ['IMX586proto_BlindClose',  2852, 2952, 3300, 3400],
    204: ['IMX376Proto_Nenpyo',  2132, 2232, 2720, 2820],
    301: ['IMX376Proto_Nenpyo',  2020, 2120, 1252, 1352],
    302: ['IMX376Proto_Nenpyo',  2152, 2252, 2812, 2912],
    401: ['IMX586proto_Shirts_macbeth',  3100, 3200, 2792, 2892],
    501: ['IMX586proto_BlindClose',  3100, 3200, 3764, 3864],
    502: ['IMX586proto_BlindClose',  3100, 3200, 3764, 3864],
    503: ['IMX376Proto_BlindClose',  1800, 1900, 2400, 2500],
}

parser = argparse.ArgumentParser()

parser.add_argument('--image-folder', type=str, default='/data/datasets/RAISR_remosaic_result/real_rgb/y7x7_hdr', help="folder path for testing dataset")
parser.add_argument('--result-folder', type=str, default='/data/datasets/RAISR_remosaic_result/crop_result/sony/y7x7_hdr',
                    help="folder path to save testing results")
parser.add_argument('--post', type=str, default='_48M_quad_result0', help='add post-name.')
parser.add_argument('--pool-num', type=int, default=8, help="number of multiprocessing pool")

args = parser.parse_args()
# kodak:    _result0
# imx586:   _48M_quad_result0
# sony:     _result0
img_dict = imx586_dict


def crop_image(img_path, x1, x2, y1, y2):
    img_data = cv2.imread(img_path)

    if len(img_data.shape) == 2:
        img_crop = img_data[x1:x2, y1:y2]
    else:
        img_crop = img_data[x1:x2, y1:y2, :]
    #print(img_crop.shape)
    img_data = cv2.rectangle(img_data, (y1-25, x1-25), (y2+25, x2+25), (0, 0, 255), 30)
    return img_data, img_crop


def crop_vivo_chaobin(input_path, output_path):
    for file in os.listdir(input_path):
        if not is_image_file(file):
            continue
        print(file)

        filename = file.split('.png')[0]
        img = cv2.imread(os.path.join(input_path, file), -1)
        if '20190802_114129' in file:
            cv2.imwrite(os.path.join(output_path, filename + '_1.png'), img[1000:1160, 4471:4607])
        if '20190802_114211' in file:
            cv2.imwrite(os.path.join(output_path, filename + '_1.png'), img[2400:2500, 3079:3170])
        if '20190802_142513' in file:
            cv2.imwrite(os.path.join(output_path, filename + '_1.png'), img[2330:2450, 1060:1160])
            cv2.imwrite(os.path.join(output_path, filename + '_2.png'), img[620:760, 2990:3100])
        if '20190802_141733' in file:
            cv2.imwrite(os.path.join(output_path, filename + '_1.png'), img[1950:2050, 1450:1550])
            cv2.imwrite(os.path.join(output_path, filename + '_2.png'), img[1940:2040, 1215:1315])
            cv2.imwrite(os.path.join(output_path, filename + '_3.png'), img[0:430, 2310:2650])
            cv2.imwrite(os.path.join(output_path, filename + '_4.png'), img[1950:2050, 1625:1725])
            cv2.imwrite(os.path.join(output_path, filename + '_5.png'), img[1960:2020, 1570:1870])


if __name__ == '__main__':
    for key in img_dict.keys():
        img_name = img_dict[key][0]
        x1, x2, y1, y2 = img_dict[key][1:]
        img_in_path = img_name + args.post + '.png'
        img_in_path = os.path.join(args.image_folder, img_in_path)
        print(img_in_path)
        img, img_crop = crop_image(img_in_path, x1, x2, y1, y2)
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_NEAREST)
        #checkimage(img)

        os.makedirs(args.result_folder, exist_ok=True)

        img_out_path = str(key) + '.png'
        img_out_path = os.path.join(args.result_folder, img_out_path)
        cv2.imwrite(img_out_path, img_crop)


