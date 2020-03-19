import argparse

from termcolor import cprint
from multiprocessing import Pool

from utils.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--test_folder', type=str, default='/data/datasets/remosaic_bin/imx586_benchmark_dataset', help="folder path for testing dataset")
parser.add_argument('--result-folder', type=str, default='./QBC', help="folder path to save testing results")
parser.add_argument('--pool-num', type=int, default=4, help="number of multiprocessing pool")

args = parser.parse_args()


def test_image(image_count, image_path):
    img_name = os.path.split(image_path)[-1]
    print('\r', end='')
    print('' * 60, end='')

    print('\r Processing ' + str(image_count + 1) + '/' + str(len(img_list)) + ' image (' + img_name + ')')
    start = time.time()
    img_uint8 = readimg(image_path)
    # imx586 & low_light
    #img_uint8 = img_uint8[2:-1, 4:-4]
    #cv2.imwrite(os.path.join(args.result_folder, os.path.splitext(img_name)[0] + '.png'), img_uint8)
    end = time.time()
    print(' time cost: %0.4fs' % float(end - start))


if __name__ == '__main__':
    img_list = make_image_list(args.test_folder)

    os.makedirs(args.result_folder, exist_ok=True)

    cprint('begin to process images:', 'red')
    p = Pool(args.pool_num)
    pool_result = []
    for img_count, img_path in enumerate(img_list):
        #r = p.apply_async(test_image, args=(img_count, img_path,))
        test_image(img_count, img_path)
    p.close()
    p.join()



