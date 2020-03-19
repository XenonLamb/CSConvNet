import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F

class timer:
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint:
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.ckpt_path = args.ckpt_path
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def merge_grads(self,grads_tensor):
        if self.args.Qstr ==2:
            stre = np.array([0.05],dtype=np.float32)
        else:
            stre = np.linspace(0, 0.2, self.args.Qstr + 1)[1:-1]
        if self.args.Qcohe ==2:
            cohe = np.array([0.3], dtype=np.float32)
        else:
            cohe = np.linspace(0, 1, self.args.Qcohe + 1)[1:-1]
        grads = grads_tensor.cpu().numpy()
        H, W, C = grads.shape
        if self.args.debug:
            print('merging grads:')
            print(stre, cohe)
            print(grads.shape, grads.dtype)
        tempus = np.clip(np.floor(grads[:,:,0]*self.args.Qangle),0,self.args.Qangle-1)
        lamdas = np.clip(np.searchsorted(stre,grads[:,:,1]),0,self.args.Qstr)
        mus = np.clip(np.searchsorted(cohe,grads[:,:,2]),0,self.args.Qcohe)
        grad_map = (tempus * self.args.Qstr * self.args.Qcohe) + lamdas * self.args.Qcohe + mus

        if self.args.debug:
            print('grad_map:')
            print(grad_map.shape)
            print(grad_map[10:12,10:12])

        return grad_map

    def get_path(self, *subdir, from_legacy=False):
        if from_legacy:
            return self.ckpt_path
        else:
            return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(str(log))
        self.log_file.write('\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch+1, epoch+1)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            if self.args.compute_grads or self.args.predict_groups:
                filename = self.get_path(
                    'results-{}'.format(dataset.dataset.name),
                    filename
                )
            else:
                filename = self.get_path(
                    'results-{}'.format(dataset.dataset.name),
                    '{}_x{}_'.format(filename, scale)
                )
            if self.args.compute_grads or self.args.predict_groups:
                postfix = ('SR', 'HR')
                for v, p in zip(save_list, postfix):
                    #normalized = v[0].mul(255 / self.args.rgb_range)
                    # print(normalized.shape)
                    #tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                    if self.args.debug:
                        print("in saving results:")
                        print(v.shape, v.dtype)
                    if self.args.real_isp and p == 'SR':
                        # the saving list will be the concatenation of [grads, noisemap] over the last dimension
                        noisemap = v[:,:,3:6].cpu().numpy()
                        if self.args.debug:
                            print('noisemap: ',noisemap.shape, noisemap.dtype)
                        v = self.merge_grads(v[:,:,0:3])
                        if self.args.debug:
                            print('v: ',v.shape, v.dtype)
                        np.savez(('{}_{}noisemap.npz'.format(filename, p)), noisemap=noisemap)
                    if not(self.args.compute_grads and (p=='SR')and (len(v.shape)==2)):
                        v = v.cpu().numpy()
                    if self.args.model == 'unet_g2ext'or self.args.model == 'unet_noisemap':
                        if self.args.Qangle == 4:
                            np.savez(('{}_{}433ext.npz'.format(filename, p)), grads=v)
                        elif self.args.outgrads:
                            np.savez(('{}_{}gradsext.npz'.format(filename, p)), grads=v)
                        elif self.args.Qangle == 1:
                            np.savez(('{}_{}9ext.npz'.format(filename, p)), grads=v)
                        elif self.args.Qstr == 2:
                            np.savez(('{}_{}822ext.npz'.format(filename, p)), grads=v)
                        else:
                            np.savez(('{}_{}833ext.npz'.format(filename, p)), grads=v)
                    elif self.args.use_stats:
                        if self.args.Qstr == 2:
                            np.savez(('{}_{}822.npz'.format(filename, p)), grads=v)
                        elif self.args.unetsize =='tiny':
                            np.savez(('{}_{}833tinystat.npz'.format(filename, p)), grads=v)
                        elif self.args.unetsize =='small':
                            np.savez(('{}_{}833smallstat.npz'.format(filename, p)), grads=v)
                    elif self.args.predict_groups:
                        np.savez(('{}_{}833classext.npz'.format(filename, p)), grads=v)
                    else:
                        if self.args.Qstr == 2:
                            np.savez(('{}_{}822.npz'.format(filename, p)), grads=v)
                        else:
                            np.savez(('{}_{}833small.npz'.format(filename,p)),grads=v)
                    # print(tensor_cpu.shape)
                    #self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
                    # cv2.imwrite('resulting.png', yy)
            else:

                postfix = ('SR', 'LR', 'HR')
                for v, p in zip(save_list, postfix):
                    normalized = v[0].mul(255 / self.args.rgb_range)
                    #print(normalized.shape)
                    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                    if tensor_cpu.shape[2] == 1:
                        tensor_cpu = tensor_cpu[:,:,0]
                    #print(tensor_cpu.shape)
                    #self.queue.put(('{}{}_{}.png'.format(filename, p,self.args.model), tensor_cpu))
                    if self.args.debug:
                        print('Pushing QUEUE: {}{}.png'.format(filename, p))
                    self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
                    #cv2.imwrite('resulting.png', yy)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def expand_celoss(self, pred, ground):
    if self.args.debug:
        print("loss expand function:")
        print(pred.shape, ground.shape)
    pred = pred.permute(0, 2, 3, 1)
    ncs = pred.shape[3]
    pred = pred.view(-1, ncs)
    mask_flat = ground.view(-1)
    lossp = F.cross_entropy(pred, mask_flat)

    return lossp.data[0]

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale+6
        #if diff.size(1) > 1:
        #    gray_coeffs = [65.738, 129.057, 25.064]
        #    convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        #    diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def calc_mse(sr, hr):
    if hr.nelement() == 1: return 0

    diff = (sr - hr)
    shave = 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return mse

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):

            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

