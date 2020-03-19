"""
options and argument parser
"""

import argparse
import template
import utility
from math import floor
import random




parser = argparse.ArgumentParser(description='Pixel-wise classification and denoising')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--timer', action='store_true',
                    help='display running time')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=16,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=4,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# general specifications
parser.add_argument('--ckpt_path', type=str, default='../dataset',
                    help='path to checkpoint')
#parser.add_argument('--split_ckpt_path', type=str, default='../dataset',
#                    help='path to splitted checkpoint')

parser.add_argument('--test_every', type=int, default=4000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=350,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--from_scratch', action='store_true',
                    help='train from scratch')
parser.add_argument('--compute_grads', action='store_true',
                    help='split')
parser.add_argument('--predict_groups', action='store_true',
                    help='split')
parser.add_argument('--from_regression', action='store_true',
                    help='use predicted mask')
parser.add_argument('--no_gt', action='store_true',
                    help='no loading gt mask')
parser.add_argument('--drop_mask', action='store_true',
                    help='randomly replace mask')
parser.add_argument('--drop_mask_prob', type=float, default=0.1,
                    help='learning rate decay factor for step decay')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--dir_data', type=str, default='../dataset',
                    help='dataset directory')
parser.add_argument('--postfix', type=str, default='',
                    help='dataset directory')
parser.add_argument('--halfing', type=str, default='full',
                    help='which half of training data')
parser.add_argument('--noiselevel', type=str, default='15',
                    help='noise level')
parser.add_argument('--outgrads', action='store_true',
                    help='output grad statistics when using unet')
parser.add_argument('--from_legacy', action='store_true',
                    help='load checkpoint from original carn ckpt')



# Optimization specifications
parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='20-40-60-250',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')


## model options

parser.add_argument('--model', default='bucket_residual',
                    help='model name')
parser.add_argument('--n_layers', type=int, default=5,
                    help='number of layers')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='conv kernel size')
parser.add_argument('--split', action='store_true',
                    help='use CSConv')
parser.add_argument('--small_res', action='store_true',
                    help='2 convs per resblock')
parser.add_argument('--gtmask', action='store_true',
                    help='use classes produced from RAISR on ground truth images')
parser.add_argument('--noisemask', action='store_true',
                    help='use classes produced from RAISR on noisy images')
parser.add_argument('--split_skip', action='store_true',
                    help='have split convolution only at bottom')
parser.add_argument('--split', action='store_true',
                    help='use CSConv')
parser.add_argument('--small_res', action='store_true',
                    help='2 convs per resblock')
parser.add_argument('--part', type=str, default='body',
                    help='where to split (deprecated)')
parser.add_argument('--load_transfer', action='store_true',
                    help='use non-CSConv model for initialization')



## specific options for unet (PCN)
parser.add_argument('--postfix', type=str, default='',
                    help='postfix of the .npy class files')
parser.add_argument('--unetsize', type=str, default='tiny',
                    help='model size of unet (for pixel-wise classification)')
parser.add_argument('--use_stats', action='store_true',
                    help='use gradient stats in input')


# Option for RAISR splitting
parser.add_argument('--Qstr', type=int, default=3,
                    help='#of groups for strength')
parser.add_argument('--Qcohe', type=int, default=3,
                    help='#of groups for coherence')
parser.add_argument('--Qangle', type=int, default=8,
                    help='#of groups for directions')
parser.add_argument('--h_hsize', type=int, default=4,
                    help='patch size for raisr')
parser.add_argument('--pre_raisr', action='store_true',
                    help='store raisr split into npz files (disable augmentation when doing so)')

## dataset options

parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K+CBSD500+ADOBE5K+WED',
                    help='train dataset name')
#parser.add_argument('--data_test', type=str, default='DIV2K',
#                    help='test dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K+CBSD68',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--data_range_adobe', type=str, default='1-5000/001-002',
                    help='train/test data range for ADOBE5K')
parser.add_argument('--data_range_hdr', type=str, default='1-3500/001-002',
                    help='train/test data range for hdrplus')
parser.add_argument('--data_range_wed', type=str, default='1-4700/4701-4744',
                    help='train/test data range for wed')
parser.add_argument('--data_range_sidd', type=str, default='1-320/1-2',
                    help='train/test data range for sidd training set')
parser.add_argument('--data_range_siddval', type=str, default='1-2/1-1280',
                    help='train/test data range for sidd validation set')
parser.add_argument('--data_range_nam', type=str, default='1-100/1-100',
                    help='train/test data range for nam dataset')

#some custom datasets, please ignore
parser.add_argument('--data_range_bsdwed', type=str, default='1-1200/1-1200',
                    help='train/test data range')
parser.add_argument('--data_range_bsdwed4', type=str, default='1-2450/1-2450',
                    help='train/test data range')





##options used for real noise, you can ignore
parser.add_argument('--real_isp', action='store_true',
                    help='use simulated real noise')
parser.add_argument('--use_real', action='store_true',
                    help='use simulated real noise')





#parser.add_argument('--chop', action='store_true',
#                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# ignore these

parser.add_argument('--task', type=str, default='sr',
                    help='sr or denoise')
parser.add_argument('--mask_type', type=str, default='raisr',
                    help='neuf or raisr')
#parser.add_argument('--num_types', type=int, default=9,
#                    help='#of groups for raisr')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
#parser.add_argument('--n_resblocks', type=int, default=16,
#                    help='number of residual blocks')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

#original options by EDSR/CARN, you can ignore them
parser.add_argument('--scale', type=str, default='1',
                    help='super resolution scale')

# Option for CARN, deprecated
parser.add_argument('--group', type=int, default=1,
                    help='#of groups for group conv')
parser.add_argument('--multi_scale', action='store_true',
                    help='multiscale model')





####IGNORE BELOW

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--reinit', action='store_true',
                    help='reintialize the weights related to split')

parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--noise_eval', type=int, default=25,
                    help='noise level added in evaluation')


args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

if args.timer:
    args.timer_kconv_forward = utility.timer()
    #args.timer_kconv_backward = utility.timer()
    args.timer_embedding_forward = utility.timer()
    args.timer_total_forward = utility.timer()
    #args.timer_total_backward = utility.timer()

