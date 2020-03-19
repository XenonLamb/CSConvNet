"""
main training script
"""


import os
import math
from decimal import Decimal

import utility

import numpy as np
from math import floor
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import torch.nn.functional as F


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.losstype = args.loss
        self.task = args.task
        self.noise_eval = args.noise_eval
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        self.noiseL_B = [0, 55]  # ingnored when opt.mode=='S'
        self.error_last = 1e8

        self.ckp.write_log("-------options----------")
        self.ckp.write_log(args)

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model, timer_backward = utility.timer(), utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, mask, _,) in enumerate(self.loader_train):
            if self.args.debug:
                print("into train loop")
                print(lr.shape,hr.shape,mask.shape)

            if self.task =='denoise' and (not self.args.real_isp):
            #if self.args.compute_grads or self.args.predict_groups:

                noise = torch.randn(lr.size())*(self.noise_eval)
                lr = torch.clamp((lr + noise),0,255)

            lr, hr, mask = self.prepare(lr, hr, mask)
            # lr = lr.to(self.model.device)
            # hr = hr.to(self.model.device)
            # mask = mask.to(self.model.device)
            mask.requires_grad_(False)
            if self.args.debug:
                print('lr shape:', lr.shape)
                print("hr shape: ", hr.shape)
                print("mask shape", mask.shape)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model(lr, mask, 0)
            if self.args.debug:
                print("training forwarded: ")
                print(lr.shape,hr.shape,mask.shape)
                print(mask.dtype)
                print(mask[:,10:12,10:12])
            if self.args.compute_grads:
                loss = self.loss(sr, mask)
            elif self.args.predict_groups:
                mask = mask.long()
                loss = self.loss_expand(sr[0],mask[:,0,:,:])+self.loss_expand(sr[1],mask[:,1,:,:])+self.loss_expand(sr[2],mask[:,2,:,:])
            else:
                if 'WeightedL1' in self.losstype:
                    loss = self.loss(sr, hr, mask)
                else:
                    loss = self.loss(sr, hr)
            timer_backward.tic()
            loss.backward()
            timer_backward.hold()
            if self.args.debug:
                print("loss backwarded: ")
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                if self.args.timer:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.3f}+{:.3f}+{:.3f}s\t{:.3f}+{:.3f}+{:.3f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release(),
                        timer_backward.release(),
                        self.args.timer_total_forward.release(),
                        self.args.timer_embedding_forward.release(),
                        self.args.timer_kconv_forward.release(),
                    ))
                else:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release(),
                    timer_backward.release()))


            timer_data.tic()


        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def loss_expand(self,pred,ground):
        if self.args.debug:
            print("loss expand function:")
            print(pred.shape, ground.shape)
        pred = pred.permute(0, 2, 3, 1)
        ncs = pred.shape[3]
        pred = pred.view(-1, ncs)
        mask_flat = ground.view(-1)
        lossp = self.loss(pred, mask_flat)

        return lossp
    """
    def merge_grads(self,grads_tensor):
        stre = np.linspace(0, 0.2, self.args.Qstr + 1)[1:-1]
        cohe = np.linspace(0, 1, self.args.Qcohe + 1)[1:-1]
        grads = grads_tensor.cpu().numpy()
        H, W, C = grads.shape
        if self.args.debug:
            print('merging grads:')
            print(stre, cohe)
            print(grads.shape, grads.dtype)
        grad_map = np.zeros((H, W), dtype=np.int32)
        for i in range(H):
            for j in range(W):
                if self.args.debug:
                    print('grads:')
                    print(grads[i,j,0],grads[i,j,1],grads[i,j,2])
                tempu = floor(grads[i, j, 0] * self.args.Qangle)
                if tempu < 0:
                    tempu = 0
                if tempu > self.args.Qangle - 1:
                    tempu = self.args.Qangle - 1
                if self.args.debug:
                    print('tempu:')
                    print(tempu)
                lamda =  np.searchsorted(stre, grads[i, j, 1])
                mu = np.searchsorted(cohe, grads[i, j, 2])
                if self.args.debug:
                    print('lamda&mu:')
                    print(lamda, mu)
                grad_map[i,j] = (tempu * self.args.Qstr * self.args.Qcohe) + lamda * self.args.Qcohe + mu

        if self.args.debug:
            print('grad_map:')
            print(grad_map[10:12,10:12])


        return grad_map
    """

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
    def stride_cut(self,lr,hr,mask=None, split_size=400, overlap_size=100):
        if self.args.debug:
            print('stride cutting the following tensors: ')
            print('lr: ',lr.shape)
            print('hr: ',hr.shape)
            if mask is not None:
                print('mask: ',mask.shape)
            print('split_size: ',split_size,'  overlap_size: ',overlap_size)
        stride = split_size - overlap_size
            ## 构建图像块的索引
        orig_shape = (lr.shape[2], lr.shape[3])

        imhigh = lr.shape[2]
        imwidth = lr.shape[3]
        range_y = np.arange(0, imhigh - split_size, stride)
        range_x = np.arange(0, imwidth - split_size, stride)
        if self.args.debug:
            print(range_x)
            print(range_y)
        if range_y[-1] != imhigh - split_size:
            range_y = np.append(range_y, imhigh - split_size)
        if range_x[-1] != imwidth - split_size:
            range_x = np.append(range_x, imwidth - split_size)

        sz = len(range_y) * len(range_x)  ## 图像块的数量
        if self.args.debug:
            print('sz: ',sz)
        res_lr=  torch.zeros((sz,lr.shape[1], split_size,split_size),dtype=lr.dtype)
        res_hr = torch.zeros((sz, hr.shape[1], split_size, split_size), dtype = hr.dtype)
        res_mask = torch.zeros((sz, split_size, split_size), dtype = mask.dtype)
        if self.args.debug:
            print(range_x)
            print(range_y)
            print('sz: ',sz, res_lr.shape,res_lr.dtype,res_hr.shape,res_mask.shape,res_mask.dtype)
        index = 0
        for y in range_y:
            for x in range_x:
                res_lr[index,:,:,:] = lr[0,:,y:y + split_size, x:x + split_size]
                res_hr[index, :, :, :] = hr[0, :, y:y + split_size, x:x + split_size]
                res_mask[index, :, :] = mask[0, y:y + split_size, x:x + split_size]
                index = index + 1

        if self.args.debug:
            print('finished cutting: ', res_lr.shape,res_hr.shape,res_mask.shape)

        return res_lr,res_hr,res_mask

    def recon_from_cols(self,sr_cols,imsize, stride=300,split_size=400):

        sr_recon = torch.zeros((1,sr_cols.shape[1],imsize[0],imsize[1]),dtype = sr_cols.dtype)

        w = torch.zeros((1,sr_cols.shape[1],imsize[0],imsize[1]),dtype = sr_cols.dtype)
        if self.args.debug:
            print('reconstructing patches: ', sr_recon.shape, w.shape)
        range_y = np.arange(0, imsize[0] - split_size, stride)
        range_x = np.arange(0, imsize[1] - split_size, stride)
        if range_y[-1] != imsize[0] - split_size:
            range_y = np.append(range_y, imsize[0] - split_size)
        if range_x[-1] != imsize[1] - split_size:
            range_x = np.append(range_x, imsize[1] - split_size)
        if self.args.debug:
            print('range x and y: ', range_x, range_y)
        index = 0
        for y in range_y:
            for x in range_x:
                sr_recon[0,:,y:y + split_size, x:x + split_size] = sr_recon[0,:,y:y + split_size, x:x + split_size] + sr_cols[
                    index,:,:,:]
                w[0,:,y:y + split_size, x:x + split_size] = w[0,:,y:y + split_size, x:x + split_size] + 1
                index = index + 1
        if self.args.debug:
            print('reconstruction finished: ', sr_recon.shape,sr_recon.max(), w.shape)

        return sr_recon / w



    def test(self):
        torch.set_grad_enabled(False)
        #print(self.loader_test)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        #self.model.to
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        if self.args.debug:
            print("Eval loop")
        for idx_data, d in enumerate(self.loader_test):
            if self.args.debug:
                print("idx data retrieved")
            for idx_scale, scale in enumerate(self.scale):
                if self.args.debug:
                    print("scale fetched")
                d.dataset.set_scale(idx_scale)
                torch.manual_seed(2)
                for lr, hr, mask, filename in tqdm(d, ncols=80):
                    if self.args.debug:
                        print("eval started")

                    #print("prepared")
                    if self.task =='denoise' and (not self.args.real_isp):
                    #if self.args.compute_grads or self.args.predict_groups:
                        noise = (torch.randn(lr.size())*(self.noise_eval))
                        lr = torch.clamp((lr + noise),0,255)

                    if self.args.predict_groups:

                        mask = mask.long()

                    if self.args.debug:
                        print('lr shape:', lr.shape)
                        print("hr shape: ", hr.shape)
                        print("mask shape", mask.shape)
                    #print("noise added", lr.shape, hr.shape, mask.shape)
                    #print(hr.shape)
                    if lr.shape[2]*lr.shape[3]>250*250 and(not (self.args.compute_grads or self.args.predict_groups)):
                        if self.args.model == 'kpn':
                            if self.args.debug:
                                print('oversized input: ', lr.shape, hr.shape, mask.shape)

                            orig_shape = (lr.shape[2], lr.shape[3])
                            lr_cols, hr_cols, mask_cols = self.stride_cut(lr, hr, mask, split_size=96,
                                                                          overlap_size=48)
                            sr_cols = torch.zeros(hr_cols.shape, dtype=hr_cols.dtype)
                            for colidx in range(lr_cols.shape[0]):
                                lr_slice = lr_cols[colidx, :, :, :].unsqueeze(0)
                                hr_slice = hr_cols[colidx, :, :, :].unsqueeze(0)
                                mask_slice = mask_cols[colidx, :, :].unsqueeze(0)
                                lr_slice, hr_slice, mask_slice = self.prepare(lr_slice, hr_slice, mask_slice)
                                if self.args.debug:
                                    print('cut forwarding: ', colidx, lr_slice.shape)
                                sr_slice = self.model(lr_slice, mask_slice, idx_scale)
                                sr_cols[colidx, :, :, :] = sr_slice[0, :, :, :]
                            if self.args.debug:
                                print('finished patch forwarding: ', sr_cols.shape)
                            sr = self.recon_from_cols(sr_cols, orig_shape, stride=48, split_size=96)
                        else:

                            if self.args.debug:
                                print('oversized input: ',lr.shape, hr.shape, mask.shape)

                            orig_shape = (lr.shape[2],lr.shape[3])
                            lr_cols, hr_cols, mask_cols = self.stride_cut(lr, hr, mask, split_size=200, overlap_size=100)
                            sr_cols = torch.zeros(hr_cols.shape, dtype = hr_cols.dtype)
                            for colidx in range(lr_cols.shape[0]):
                                lr_slice= lr_cols[colidx,:,:,:].unsqueeze(0)
                                hr_slice= hr_cols[colidx,:,:,:].unsqueeze(0)
                                mask_slice = mask_cols[colidx, :, :].unsqueeze(0)
                                lr_slice,hr_slice,mask_slice = self.prepare(lr_slice,hr_slice,mask_slice)
                                sr_slice = self.model(lr_slice, mask_slice,idx_scale)
                                sr_cols[colidx,:,:,:] = sr_slice[0,:,:,:]
                            if self.args.debug:
                                print('finished patch forwarding: ',sr_cols.shape)
                            sr = self.recon_from_cols(sr_cols,orig_shape, stride=100,split_size=200)


                    else:
                        lr, hr, mask = self.prepare(lr, hr, mask)
                        if self.args.debug:
                            print('prepared: ', lr.shape, hr.shape,mask.shape)
                        sr = self.model(lr, mask, idx_scale)
                    if self.args.debug:
                        print("forwarded")
                        print("eval sr shape & mask shape:")
                        if self.args.predict_groups:
                            print(sr[0].shape, mask.shape)
                        else:
                            print(sr.shape, mask.shape)
                    if not (self.args.compute_grads or self.args.predict_groups):
                        sr = utility.quantize(sr, self.args.rgb_range)

                    if self.args.predict_groups:


                        _, sr_predict0 = torch.max(sr[0], 1)
                        _, sr_predict1 = torch.max(sr[1], 1)
                        _, sr_predict2 = torch.max(sr[2], 1)
                        sr_predict = torch.cat((sr_predict0.unsqueeze(-1),sr_predict1.unsqueeze(-1),sr_predict2.unsqueeze(-1)),dim=-1)
                        sr_predict = sr_predict.to(dtype=torch.int32)
                        if self.args.debug:
                            print("calculate prediction")
                            print("prediction shape:")
                            print(sr_predict.shape)
                            print(sr_predict.dtype)
                        save_list = [((sr_predict.squeeze(0)[:,:,0]*self.args.Qcohe*self.args.Qstr)+ (sr_predict.squeeze(0)[:,:,1]*self.args.Qcohe)+(sr_predict.squeeze(0)[:,:,2]))]
                        if self.args.debug:
                            print("savelist saved")
                            print("savelist shape:")
                            print(save_list[0].shape)
                    elif self.args.compute_grads:
                        if self.args.debug:
                            print("saving mask: ")
                        if self.args.real_isp:

                            save_list = [sr.permute(0, 2, 3, 1).contiguous().squeeze(0)]

                        elif self.args.outgrads:
                            save_list = [sr.permute(0, 2, 3, 1).contiguous().squeeze(0)]
                        else:
                            save_list = [self.merge_grads(sr.permute(0,2,3,1).contiguous().squeeze(0))]
                        #save_list = [sr.permute(0, 2, 3, 1).contiguous().squeeze(0)]
                    else:

                        save_list = [sr]


                    if self.args.compute_grads:
                        if self.args.debug:
                            print("calculate mse")
                            print(sr.shape, sr.dtype, mask.shape, mask.dtype)
                        if self.args.no_gt:
                            self.ckp.log[-1, idx_data, idx_scale] += 0
                        else:
                            self.ckp.log[-1, idx_data, idx_scale] += utility.calc_mse(
                                sr, mask
                            )
                    elif self.args.predict_groups:
                        #print('permulting')
                        #print(mask.permute(0,2,3,1).contiguous().shape)
                        #mask_per = mask.permute(0,2,3,1).contiguous()
                        #print('permulted')
                        #print(mask_per.shape,mask_per.dtype)
                        #print(sr_predict.shape,sr_predict.dtype)

                        #diffs = (sr_predict==mask.permute(0,2,3,1).contiguous())
                        #if self.args.debug:
                        #    print("calculate difference:")
                        #    print(diffs.shape,diffs.dtype)
                        #    print("number of elements:")
                        #    print(mask.numel())
                        #    print("correct prediction:")
                        #    print(torch.sum((sr_predict==mask.permute(0,2,3,1)).float()))
                        #acc = torch.sum((sr_predict[1]==mask.permute(0,2,3,1)).float())/mask.numel()
                        if self.args.debug:
                            print(sr_predict.shape, sr_predict.dtype)
                            print(mask.shape, mask.dtype)
                        acc = torch.sum((sr_predict[:,:,:,1] == mask.to(dtype=torch.int32)[:,1,:,:]).float()) / mask[:,1,:,:].numel()
                        if self.args.debug:
                            print("calculate accuracy:")
                            print(acc)
                        self.ckp.log[-1, idx_data, idx_scale] += acc

                    else:
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                    if self.args.debug:
                        print("loss logged!")

                    if self.args.save_gt:
                        if self.args.compute_grads or self.args.predict_groups:
                            save_list.extend([mask.permute(0,2,3,1).contiguous().squeeze(0)])
                        else:
                            if self.args.use_real:
                                if self.args.debug:
                                    print('saving lr image when using real isp',lr[:,0:self.args.n_colors,:,:].shape)
                                save_list.extend([lr[:,0:self.args.n_colors,:,:], hr])
                            else:
                                save_list.extend([lr, hr])
                    if self.args.debug:
                        print("SAVING RESULTS!")
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    if self.args.debug:
                        print("results saved!")

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args, test=False):
        device = torch.device('cpu' if self.args.cpu or test else 'cuda')
        self.model = self.model.to(device)
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs



