#!/usr/bin/env bash
# EDSR baseline model (x2) + JPEG augmentation

srun -p Pixel --mpi=pmi2 python add_noise.py

python main.py --model carn --scale 4 --patch_size 96 --save carn_baseline_x4 --multi_scale --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75
srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_partial --scale 1 --patch_size 72 --save bucket_splittop_5layer5x5_x1_inmask_ydenoise25_divbsd --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_test DIV2K+CBSD68 --data_train DIV2K+CBSD500 --reinit --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --n_layers 5
# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]
srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model singlelayer --scale 1 --patch_size 96 --save single_1layer7x7_x1_ydenoise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 16 --save_results --pre_raisr --mask_type raisr --data_test DIV2K+CBSD68 --reinit --task denoise --n_threads 20 --save_gt --n_GPUs 2 --n_layers 1

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model singlelayer --scale 4 --save single_split_x4_gtmaskmiddle --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --neuf --n_GPUs 2 --save_gt --mask_type neuf_gt

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]
--Qstr 3 --Qcohe 2
# EDSR in the paper (x2)
srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model singlelayer --scale 1 --patch_size 72 --save single_3layer5x5_split_x1_inmask_ydenoise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --neuf --save_results --pre_raisr --mask_type neuf --data_test DIV2K+CBSD68 --reinit --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --n_layers 3 --lr 5e-4


srun -p Pixel --mpi=pmi2 --gres=gpu:1 python main.py --model bucket_conv --scale 1 --patch_size 96 --save bucket_5layer5x5_x1_inmask_ydenoise25_testtrain --from_legacy --ckpt_path ../experiment/bucket_5layer5x5888_x1_inmask_denoise25_testbsd/model/model_best.pt --warm_load --split_ckpt_path ../experiment/bucket_5layer5x5888_x1_inmask_denoise25_testbsd/model/model_best.pt  --batch_size 2 --pre_raisr --save_results --mask_type raisr --data_test DIV2K+CBSD68+CBSD500 --task predenoise --n_threads 20 --save_gt --n_GPUs 1 --h_hsize 3 --n_layers 5 --lr 1e-7 --data_range 1-800/1-800


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save bucket_5layer5x5_x1_gt_ydenoise25_divbsd_testinput --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_test DIV2K+CBSD68+CBSD500 --reinit --task denoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5

python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --data_range 300-400/801-810 --task denoise --n_threads 20 --save_gt

python main.py --model bucket_conv --scale 1 --patch_size 96 --save bucket_1layer7x7_888_x1_ydenoise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --data_test DIV2K --reinit --task denoise --n_threads 20 --save_gt --n_GPUs 2


python main.py --model bucket_conv --scale 1 --patch_size 96 --save bucket_888_x1_denoise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 64 --save_results --pre_raisr --reinit --Qstr 8 --Qcohe 8 --data_test DIV2K --task denoise --n_threads 20 --save_gt --n_GPUs 2

python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_result --n_GPUs 1 --task denoise --pre_raisr --data_range 300-400/801-810

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --save carn_split_x1_wl1_denoise75_resume --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 16 --save_results --neuf --loss '1*WeightedL1'--n_GPUs 2 --data_test DIV2K --part b1b2b3 --noise_eval 75 --warm_load --split_ckpt_path ../experiment/carn_split_x1_wl1_denoise75/model/model_latest.pt --save_gt
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]
python main.py --model carn --scale 4 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 16 --save_results --neuf --mask_type neuf_gt --save_gt --data_range 001-200/801-810

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]
srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 4 --save carn_split_x4_gtmask --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --neuf --n_GPUs 2 --part b1b2b3 --save_gt --mask_type neuf_gt

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models
srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model singlelayer --scale 1 --patch_size 72 --save single_5layer5x5_split_x1_inmask_ydenoise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --neuf --save_results --mask_type neuf --data_test DIV2K+CBSD68 --reinit --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --n_layers 5
# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt


### generate mask for ADOBE5K

 srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train ADOBE5K --data_test DIV2K --Qangle 8 --Qstr 5 --Qcohe 6 --h_hsize 4 --data_range_adobe 1000-1500/147-148 --task predenoise --n_threads 20 --save_gt

 ### residual blocks

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask_ydenoise25_divbsd --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68 --reinit --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 8 --Qcohe 8 --kernel_size 3 --n_feats 16 --from_regression --halfing tail

### residual_predict
 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask833small_ydenoise25_divbsdadobe5000_drop01 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --drop_mask --no_augment --lr 2e-4


  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask833small_test --from_legacy --ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833small_divbsdadobewed_test/model/model_best.pt --warm_load --split_ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833small_divbsdadobewed_test/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix small



 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask_ydenoise25_divbsdadobe5000_drop01_augment --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 8 --Qcohe 8 --kernel_size 3 --n_feats 16 --from_regression --halfing full --drop_mask

## small residual predict

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketsmallres_5layer_x1_3x3split_predmask833_ydenoise25_divbsdadobe4960_tinystat --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-4960/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --drop_mask --no_augment --lr 2e-4 --small_res --postfix tinystat


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketsmallres_5layer_x1_3x3split_predmask833_test --from_legacy --ckpt_path ../experiment/bucketsmallres_5layer_x1_3x3split_predmask833_ydenoise25_divbsdadobe5000_test/model/model_best.pt --warm_load --split_ckpt_path ../experiment/bucketsmallres_5layer_x1_3x3split_predmask833_ydenoise25_divbsdadobe5000_test/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --small_res

### residual with hand-craft mask

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_3layer_x1_inmask_ydenoise25_divbsdadobe_evenraisr --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD68+ADOBE5K --data_test DIV2K+CBSD68 --Qstr 5 --Qcohe 6 --reinit --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 4 --n_layers 3 --kernel_size 3

 #### DNCNN

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model dncnn --scale 1 --patch_size 96 --save dncnn_x1_ydenoise25_divbsdadobewed_pretrained_test_finetune --from_legacy --ckpt_path ../experiment/dncnn_x1_ydenoise25_divbsdadobewed_pretrained/model/model_best.pt --warm_load --split_ckpt_path ../experiment/dncnn_x1_ydenoise25_divbsdadobewed_pretrained/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --Qstr 8 --Qcohe 3 Qcohe 3 --lr 8e-5 --no_augment

####EDSR_Denoise

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr_denoise --scale 1 --patch_size 96 --save edsr_x1_ydenoise25_divbsd_pretrained --from_legacy --ckpt_path ../checkpoint/net.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --Qstr 8 --Qcohe 8


###save grads

   srun -p Pixel --mpi=pmi2  python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --data_range 300-400/801-810 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4
  ### residual 1*1 split, +ADOBE5K

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer1x1split_x1_inmask_ydenoise25_divbsdadobe --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_test DIV2K+CBSD68 --reinit --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --Qstr 8 --Qcohe 8 --h_hsize 3 --n_layers 5


  #### unet

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_med --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 20*MSE
  #### unet l1

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_small_l1mse_divbsdadobe --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-2 --data_range 1-800/801-810 --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 4*L1+4*MSE
    #### unet tiny with stats

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_tiny_l1mse_divbsd_withstat --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/801-810 --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 1*L1+1*MSE --lr 1e-4 --use_stats

          srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_new --scale 1 --patch_size 96 --save newunet2_regression_h4_small_l1_divbsdadobewed_withstat --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range_wed 1-4700/4701-4744 --data_range 1-800/801-810 --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 1*L1+1*MSE --lr 1e-4 --use_stats --unetsize small --no_augment

  #### unet_test

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_tiny_testtrain --from_legacy --ckpt_path ../experiment/unet_regression_h4_tinynew_saveeval/model/model_latest.pt --warmload --split_ckpt_path ../experiment/unet_regression_h4_tinynew_saveeval/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test DIV2K+CBSD500+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 20*MSE


     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_med_l1_testtrain --from_legacy --ckpt_path ../experiment/unet_regression_h4_med_l1/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/unet_regression_h4_med_l1/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test DIV2K+CBSD500+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 1*L1


  #### unet classification

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_classification --scale 1 --patch_size 96 --save unet_classification_h3_large --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --h_hsize 3 --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 5*CE --predict_groups

  #### unet classification

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_classification --scale 1 --patch_size 96 --save unet_classification_h3_med_testtrain --from_legacy --ckpt_path ../experiment/unet_classification_h3_tiny/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/unet_classification_h3_tiny/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --h_hsize 3 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test DIV2K+CBSD500+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 5*CE --predict_groups --lr 1e-4


### generate mask for ADOBE5K

 srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train ADOBE5K --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 3 --data_range_adobe 1600-2000/801-810 --task predenoise --n_threads 20 --save_gt


### generate grads for HDRPLUS

 srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train HDRPLUS --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 4 --data_range_hdr 1000-1500/147-148 --task predenoise --n_threads 20 --save_gt --split --compute_grads

### generate grads for WED

 srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train WED --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 4 --data_range_wed 2250-2500/147-148 --task predenoise --n_threads 20 --save_gt --split --compute_grads


  srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train ADOBE5K --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 4 --data_range_adobe 1-250/147-148 --task predenoise --n_threads 20 --save_gt --split --compute_grads


  srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train DIV2K --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 4 --data_range 400-810/147-148 --task predenoise --n_threads 20 --save_gt --split --compute_grads

### generate grads for SIDD

  srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train SIDD --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 4 --data_range_sidd 1-30/301-305 --task denoise --n_threads 20 --save_gt --split --compute_grads

    srun -p Pixel --mpi=pmi2 python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train SIDDVAL --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 4 --data_range_siddval 1-640/301-305 --task denoise --n_threads 20 --save_gt --split --compute_grads

###unet l1 predictionmask test

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_med_l1_halfed_testadobe  --from_legacy --ckpt_path ../experiment/unet_regression_h4_med_l1_divbsd_halfed/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/unet_regression_h4_med_l1_divbsd_halfed/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 2 --Qcohe 2 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test ADOBE5K+DIV2K+CBSD68+CBSD500 --data_range_adobe 1-2/1-2800  --split --n_GPUs 2 --test_every 1000 --loss 1*L1 --lr 1e-5 --halfing full

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_lesssmall_l1_halfed  --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68  --split --n_GPUs 2 --test_every 2000 --loss 10*L1 --halfing head



     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_small_l1_halfed_test  --from_legacy --ckpt_path ../experiment/unet_regression_h4_small_l1_divbsd_halfed/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/unet_regression_h4_small_l1_divbsd_halfed/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 8 --Qcohe 8 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test DIV2K+CBSD68+CBSD500  --split --n_GPUs 2 --test_every 2000 --loss 1*L1 --lr 1e-5 --halfing full


  #### unet classification 833

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_classification --scale 1 --patch_size 96 --save unet_classification_833_small --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --h_hsize 4 --data_train DIV2K+CBSD500+ADOBE5K --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 1*CE --predict_groups --no_augment

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_classification --scale 1 --patch_size 96 --save unet_classification_833_small_divbsd --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --reinit --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --h_hsize 4 --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 1*CE --predict_groups --no_augment


  #### unet classification 833 test

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_classification --scale 1 --patch_size 96 --save unet_classification_833_small_divbsd_testadobe --from_legacy --ckpt_path ../experiment/unet_classification_833_small_divbsd2/model/model_best.pt --warm_load --split_ckpt_path ../experiment/unet_classification_833_small_divbsd2/model/model_best.pt  --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task predenoise --n_threads 20 --save_gt --h_hsize 4 --data_train DIV2K+CBSD500 --data_test DIV2K+CBSD68+CBSD500+ADOBE5K --data_range 1-800/1-810 --data_range_adobe 1-10/1-2800 --split --n_GPUs 2 --test_every 2000 --loss 1*CE --predict_groups --no_augment --lr 1e-6






../experiment/bucketres_5layer_x1_3x3split_predmask833_ydenoise25_divbsdadobe5000_drop00_augment/model/model_latest.pt

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask833big_test --from_legacy --ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833_ydenoise25_divbsdadobe5000_drop00_augment/model/model_best.pt --warm_load --split_ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833_ydenoise25_divbsdadobe5000_drop00_augment/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5

 ##### 822 denoising


  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask822_ydenoise25_divbsdadobewed_test --from_legacy --ckpt_path  ../experiment/bucketres_5layer_x1_3x3split_predmask822_ydenoise25_divbsdadobewed/model/model_best.pt --warm_load --split_ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask822_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 2 --Qcohe 2 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat


    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask833ext_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_16feats_833class5 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res


                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_16feats_833class5_scale02 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res --res_scale 0.2


                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_16feats_833smallstat --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res


                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_16feats_822class5_scale02 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 2 --Qcohe 2 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res --res_scale 0.2

              srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_8feats_822class5_scale04 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 2 --Qcohe 2 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res --res_scale 0.4



                                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_8layer_16feats --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 20 --save_gt  --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res


        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask833ext_splitskip --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res --split_skip


                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask833ext_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer

                  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_833classification_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix smallstat --small_res --load_transfer

                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_16feats_833gt --from_legacy --ckpt_path ../experiment/smallbucketres_16layer_x1_3x3split_predmask833ext_loadtransfer/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --gtmask

                      srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_16feats_833noise --from_legacy  --ckpt_path ../checkpoint/carn.pth --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --noisemask

                  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask833ext_test --from_legacy --ckpt_path ../experiment/smallbucketres_16layer_x1_3x3split_predmask833ext_loadtransfer/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext --small_res


                                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask822ext_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 2 --Qcohe 2 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer

                                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask833ext_loadtransfer_scale02 --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer --res_scale 0.2


    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_5layer_x1_3x3split_predmask833ext_ydenoise25_divbsdadobewed_test --from_legacy --ckpt_path ../experiment/smallbucketres_5layer_x1_3x3split_predmask833ext_ydenoise25_divbsdadobewed2/model/model_best.pt --warm_load --split_ckpt_path ../experiment/smallbucketres_5layer_x1_3x3split_predmask833ext_ydenoise25_divbsdadobewed2/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext --small_res



        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_x1_ydenoise25_divbsdadobewed_test --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext --small_res




        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask833smallstat_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask833ext_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-2800/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask833smallstat_divbsdadobewed_test --from_legacy --ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833smallnew_ydenoise25_divbsdadobewed_drop00_new2/model/model_best.pt --warm_load --split_ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833smallnew_ydenoise25_divbsdadobewed_drop00_new2/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat


                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_x1_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res


  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_8layer_x1_3x3split_predmask833smallstat_8feats_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 4 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat

    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_5layer_x1_3x3split_predmask833smallstat_8feats_test --from_legacy --ckpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833smallstat_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --warm_load --split_skpt_path ../experiment/bucketres_5layer_x1_3x3split_predmask833smallstat_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 4 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat


    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save bucketres_8layer_x1_3x3split_predmask833smallstat_8feats_test --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 4 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat

    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save res_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../experiment/res_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --warm_load --split_ckpt_path ../experiment/res_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 4 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save res_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 4 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat

                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save res_5layer_x1_3x3_16feats_divbsdadobewed_test --from_legacy --ckpt_path ../experiment/res_5layer_x1_3x3_16feats_ydenoise25_divbsd/model/model_best.pt --warm_load --split_ckpt_path ../experiment/res_5layer_x1_3x3_16feats_ydenoise25_divbsd/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 4 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_x1_3x3_8feats_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 10 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res



                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_5layer_x1_3x3_8feats_test --from_legacy --ckpt_path ../experiment/smallres_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --warm_load --split_ckpt_path ../experiment/smallres_5layer_x1_3x3_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 10 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat --small_res


                   @@@@@             srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_x1_833smallstat_3x3_8feats_test --from_legacy --ckpt_path ../experiment/smallbucketres_8layer_x1_3x3split_predmask833smallstat_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --warm_load --split_ckpt_path ../experiment/smallbucketres_8layer_x1_3x3split_predmask833smallstat_8feats_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat --small_res --split



                srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3_predmask833smallstat_8feats_ydenoise25_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 10 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat --small_res --split

                                srun -p Pixel --mpi=pmi2 --gres=gpu:3 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3_predmask833smallstat_16feats_frombaseline --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 3 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 3e-4 --postfix smallstat --small_res --split


         srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_x1_ydenoise25_divbsdadobewed_test_finetune --from_legacy --ckpt_path ../experiment/edsr_x1_ydenoise25_divbsdadobewed/model/model_best.pt --warm_load --split_ckpt_path ../experiment/edsr_x1_ydenoise25_divbsdadobewed/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 4e-5 --postfix ext
                  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn_demotest --from_legacy  --ckpt_path ../experiment/carn_x1_ydenoise25_divbsdadobewed/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/carn_x1_ydenoise25_divbsdadobewed/model/model_latest.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --test_every 100 --postfix smallstat
                                    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_2 --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext
                              srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_test --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-6 --postfix ext
 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_test --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext
srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833smallstat_divbsdadobewed --from_legacy  --ckpt_path ../experiment/bucketcarn_predmask833smallstat_group4_feat16_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-5 --postfix smallstat

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_finetune4 --from_legacy  --ckpt_path ../experiment/bucketcarn_group4_feat16_833class5_loadtransfer3/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-4 --postfix ext


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_loadtransfer2 --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_2/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833classification_loadtransfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_2/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix smallstat --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833gt --from_legacy  --ckpt_path ../experiment/bucketcarn_group4_feat16_833class5_loadtransfer2/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --gtmask

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833noise --from_legacy  --ckpt_path ../experiment/bucketcarn_group4_feat16_833class5_loadtransfer2/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --noisemask


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_finetune --from_legacy  --ckpt_path ../experiment/bucketcarn_group4_feat16_833class5_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_822class5_divbsdadobewed --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 2 --Qcohe 2 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext

                                                                        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn_group4_feat64_divbsdadobewed --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat
#### regression checkpoints



../experiment/unet_regression_h4_small_l1mse_divbsd/model/model_latest.pt


     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_small_l1mse_stat_halfed_testadobe  --from_legacy --ckpt_path ../experiment/unet_regression_h4_small_l1mse_divbsd_withstat/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/unet_regression_h4_small_l1mse_divbsd_withstat/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test ADOBE5K+DIV2K+CBSD68+CBSD500 --data_range_adobe 1-2/1-2800  --split --n_GPUs 2 --test_every 2000 --loss 1*L1 --lr 1e-5 --halfing full --unetsize small --use_stats --no_augment


../experiment/unet_regression_h4_tiny_l1mse_divbsd_withstat/model/model_latest.pt

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_raisr --scale 1 --patch_size 96 --save unet_regression_h4_tiny_l1mse_stat_halfed_testadobe  --from_legacy --ckpt_path ../experiment/unet_regression_h4_tiny_l1mse_divbsd_withstat/model/model_latest.pt --warm_load --split_ckpt_path ../experiment/unet_regression_h4_tiny_l1mse_divbsd_withstat/model/model_latest.pt --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500 --data_range 1-800/1-810 --data_test ADOBE5K+DIV2K+CBSD68+CBSD500 --data_range_adobe 1-2/1-2800  --split --n_GPUs 2 --test_every 2000 --loss 1*L1 --lr 1e-5 --halfing full --unetsize tiny --use_stats --no_augment


../experiment/unet_regression_h4_small_l1mse_divbsd_withstat/model/model_latest.pt


for name in ./results-HDRPLUS/*SR833small.npz; do mv "$name" "${name#small}" done
source_dir = './results-HDRPLUS'
fileslist = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
for f in fileslist:
    os.rename(join(source_dir, f),join(source_dir, f.replace('small','')))



     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_new --scale 1 --patch_size 96 --save newunet2_regression_h4_833small_divbsdwed_l1_stat  --from_legacy --ckpt_path ../checkpoint/carn.pth  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/1-810 --data_test DIV2K+CBSD68+WED --data_range_wed 1-4700/4701-4744  --split --n_GPUs 2 --test_every 2000 --loss 1*L1 --lr 2e-4 --halfing full --unetsize small --use_stats --no_augment

          srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 2 --Qcohe 2 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/801-810 --data_test DIV2K+CBSD68+WED --data_range_wed 1-4700/4701-4744  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 2e-4 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment

          srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_regression_h4_833n15_divbsdwed_l1_stat  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/801-810 --data_test CBSD68+WED --data_range_wed 1-4700/4701-4744  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-4 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment

                    srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_regression_h4_833n50_divbsdwed_l1_stat_test  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833n50_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/1-810 --data_test DIV2K+CBSD68+WED+CBSD500+ADOBE5K  --data_range_adobe 1-2/1-5000 --data_range_wed 1-4700/1-4744  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-5 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment --test



                    srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat_test  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 2 --Qcohe 2 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/1-810 --data_test DIV2K+CBSD68+WED+CBSD500+ADOBE5K --data_range_wed 1-4700/1-4744 --data_range_adobe 1-2/1-5000  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-5 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment

   #### 422

     srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat_test  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 4 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/1-810 --data_test DIV2K+Urban100+Set12+CBSD68+WED+CBSD500+ADOBE5K --data_range_wed 1-4700/1-4744 --data_range_adobe 1-2/1-5000  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-5 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment


                       srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_regression_h4_833smallstat_real_l1_stat  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train SIDD+DIV2K --data_range 1-400/1-810 --data_test SIDDEVAL --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-4 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment


















##### JIMMY SIDE


### single resblock, split, gtmask,16feats,smallres

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_1layer_7x7split_gt833_16feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 1 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 16 --from_regression --halfing full --gtmask --no_augment --lr 2e-4 --small_res --gtmask


  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_1layer_7x7_16feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 1 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --small_res


  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_2layer_7x7split_gt833_8feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 2 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --small_res --gtmask

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_2layer_7x7_16feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 2 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --small_res

    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_3layer_7x7_16feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 3 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --small_res

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_3layer_7x7split_gt833_8feats_divbsdadobe5000_bottomonly --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 3 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --small_res --gtmask --bottom_only

    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_4layer_7x7split_gt833_8feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 4 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --small_res --gtmask

    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_2layer_3x3split_gt833_8feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 2 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --small_res --gtmask

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_4layer_7x7_16feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 4 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --small_res

    srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_5layer_7x7split_gt833_8feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --small_res --gtmask

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_5layer_7x7_16feats_divbsdadobe5000 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --small_res

###### BIG KERNEL 1 CHANNEL SIDE
     srun -p Pixel --mpi=pmi2 --gres=gpu:1 python main.py --model multi_raisr --scale 1 --patch_size 108 --save multiraisr_3layer_7x7split_pred833smallstat_1feats_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-3600/1-10 --data_range_wed 1-4700/4701-4744 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 1 --h_hsize 3 --n_layers 3 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 1 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat

          srun -p Pixel --mpi=pmi2 --gres=gpu:1 python main.py --model multi_raisr --scale 1 --patch_size 108 --save multiraisr_1layer_7x7split_gt833_1feats_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-3600/1-10 --data_range_wed 1-4700/4701-4744 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 1 --h_hsize 3 --n_layers 1 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 1 --from_regression --halfing full --no_augment --lr 2e-4 --gtmask


     srun -p Pixel --mpi=pmi2 --gres=gpu:1 python main.py --model multi_raisr_res --scale 1 --patch_size 108 --save multiraisrres_3layer_7x7split_pred833smallstat_8feats_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-3600/1-10 --data_range_wed 1-4700/4701-4744 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 1 --h_hsize 3 --n_layers 3 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat

     srun -p Pixel --mpi=pmi2 --gres=gpu:1 python main.py --model multi_raisr_res --scale 1 --patch_size 108 --save multiraisrres_1layer_7x7_8feats_divbsdadobewed --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-3600/1-10 --data_range_wed 1-4700/4701-4744 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 1 --h_hsize 1 --n_layers 3 --Qstr 3 --Qcohe 3 --kernel_size 7 --n_feats 8 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat








######## REAL NOISE ###############

         srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_64feats_real --from_legacy --ckpt_path ../experiment/edsr_64feats_real/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext

            srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_64feats_real_scratch --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_real --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_real --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model dncnn --scale 1 --patch_size 96 --save dncnn_real --from_legacy --ckpt_path ../experiment/dncnn_x1_ydenoise25_divbsdadobewed_pretrained/model/model_best.pt --warm_load --split_ckpt_path ../experiment/dncnn_x1_ydenoise25_divbsdadobewed_pretrained/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --Qstr 8 --Qcohe 3 Qcohe 3 --lr 2e-4 --no_augment --postfix ext --from_regression


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_real_loadtransfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_real/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_real --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed_test/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_real_833class5_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_real/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train SIDD --data_range_sidd 1-320/1-10 --data_test SIDDVAL --data_range_siddval 1-2/1-1280 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer

######## NOISE 50 ##########


      srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_feats64_noise50_finetune --from_legacy --ckpt_path ../experiment/edsr_x1_ydenoise25_divbsdadobewed/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext

         srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_feats64_noise50_finetune_test --from_legacy --ckpt_path ../experiment/edsr_feats64_noise50_finetune/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_noise50 --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed_test/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix smallstat --small_res

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_noise50_test --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext --small_res


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_noise50 --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix smallstat

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_noise50_test --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_noise50 --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix smallstat

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_noise50_test --from_legacy  --ckpt_path ../experiment/carn2_group4_feat64_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-5 --postfix smallstat

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model dncnn --scale 1 --patch_size 96 --save dncnn_noise50_pretrained --from_legacy --ckpt_path ../experiment/dncnn_x1_ydenoise25_divbsdadobewed_pretrained_test_finetune/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --Qstr 8 --Qcohe 3 Qcohe 3 --lr 1e-4 --no_augment --from_regression --postfix ext --halfing full


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_noise50_833class5_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_noise50_loadtransfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_noise50/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer



########## UNET Noisemap


srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_noisemap --scale 1 --patch_size 96 --save unetnoisemap_h4_833class5_l1  --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/801-810 --data_test DIV2K+CBSD68+SIDDVAL --data_range_wed 1-4700/1-2 --data_range_adobe 1-2/1-5000  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-4 --batch_size 1 --halfing full --unetsize small --real_isp --no_augment --n_colors 3


srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_noisemap --scale 1 --patch_size 96 --save unetnoisemap_h4_833class5_l1_test  --from_legacy --ckpt_path ../experiment/unetnoisemap_h4_833class5_l1/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/801-810 --data_test SIDD+SIDDVAL --data_range_wed 1-4700/1-2 --data_range_adobe 1-2/1-5000 --data_range_sidd 1-2/1-320 --data_range_siddval 1-2/1-1280  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-6 --batch_size 1 --halfing full --unetsize small --real_isp --no_augment --n_colors 3

      srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_feats64_real_noisemap --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED+SIDD --data_range_adobe 1-5000/1-10 --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280 --data_test DIV2K+CBSD68+SIDDVAL --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --use_real --n_colors 3

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_real_noisemap --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED+SIDD --data_range_adobe 1-5000/1-10 --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280 --data_test DIV2K+CBSD68+SIDDVAL --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --small_res --use_real --n_colors 3

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_real_noisemap --from_legacy  --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED+SIDD --data_range_adobe 1-5000/1-10 --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280 --data_test DIV2K+CBSD68+SIDDVAL --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --use_real --n_colors 3

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_real_noisemap --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED+SIDD --data_range_adobe 1-5000/1-10 --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280 --data_test DIV2K+CBSD68+SIDDVAL --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext --use_real --n_colors 3


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_16feats_real_noisemap_833class5_transfer --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_real_noisemap/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+SIDD+CBSD500+SIDD+WED+SIDD --data_range_adobe 1-5000/1-10 --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280 --data_test CBSD68+SIDDVAL --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16  --split --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --use_real --n_colors 3 --load_transfer


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_real_noisemap_833class5_transfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_real_noisemap/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+SIDD+CBSD500+SIDD+WED+SIDD --data_range_adobe 1-5000/1-10 --data_range_sidd 1-320/1-2 --data_range_siddval 1-2/1-1280 --data_test CBSD68+SIDDVAL --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --use_real --n_colors 3 --split --load_transfer






 ######## NOISE 15 ##########


      srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_feats64_noise15_finetune --from_legacy --ckpt_path ../experiment/edsr_x1_ydenoise25_divbsdadobewed/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext

         srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model edsr --scale 1 --patch_size 96 --save edsr_feats64_noise15_finetune_test --from_legacy --ckpt_path ../experiment/edsr_feats64_noise15_finetune/model/model_best.pt  --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 8 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_noise15 --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed_test/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_noise15_2  --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-3 --postfix ext --small_res

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_noise15_test --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext --small_res


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_noise15 --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_noise15_2 --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+WED+CBSD500+DIV2K+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-3 --postfix ext

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat16_noise15_test --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_noise15 --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 2e-4 --postfix ext

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat64_noise15_test --from_legacy  --ckpt_path ../experiment/carn2_group4_feat64_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Urban100+Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-5 --postfix ext

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model dncnn --scale 1 --patch_size 96 --save dncnn_noise15_pretrained --from_legacy --ckpt_path ../experiment/dncnn_x1_ydenoise25_divbsdadobewed_pretrained_test_finetune/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --Qstr 8 --Qcohe 3 Qcohe 3 --lr 1e-4 --no_augment --from_regression --postfix ext --halfing full


 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_noise15_833class5_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise15/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_noise15_833class5_loadtransfer2 --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise15_2/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --load_transfer

   srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_noise15_833class5_loadtransfer040 --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise15_4/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1.5e-4 --postfix ext --small_res --load_transfer


srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_noise15_loadtransfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_noise15/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_noise15_loadtransfer2 --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_noise15_2/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --load_transfer


########ABLATION########


#### 4 ANGLES #####

         srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_3x3split_predmask433ext_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qangle 4 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer

        srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_433class5_loadtransfer2 --from_legacy  --ckpt_path ../experiment/carn2_group4_feat16_2/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qangle 4 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer

####CHANGE NUMBER OF RES BLOCKS#######

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_12layer_64feats_noise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 12 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

   srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_8layer_64feats_noise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 64 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_12layer_16feats_noise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 12 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_8layer_16feats_noise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_12layer_x1_3x3split_predmask833ext_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_12layer_16feats_noise25/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 12 --Qangle 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_x1_3x3split_predmask833ext_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_8layer_16feats_noise25/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qangle 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_8layer_x1_3x3split_predmask833ext_loadtransfer2 --from_legacy --ckpt_path ../experiment/smallbucketres_8layer_x1_3x3split_predmask833ext_loadtransfer/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 8 --Qangle 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res


#####CHANGE NUMBER OF FEATURES

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_32feats_noise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 32 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_8feats_noise25 --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_8feats_833class5_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_8feats_noise25/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qangle 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_x1_32feats_833class5_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_32feats_noise25/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qangle 8 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 32 --from_regression --halfing full --no_augment --lr 8e-5 --postfix ext --small_res --load_transfer



 srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat32 --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 32 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext
  srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save carn2_group4_feat8 --from_legacy  --ckpt_path ../checkpoint/net.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat8_833class5_loadtransfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat8/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 8 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat32_833class5_loadtransfer --from_legacy  --ckpt_path ../experiment/carn2_group4_feat32/model/model_latest.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 32 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --load_transfer



##### classification mask ######

 srun -p Pixel --mpi=pmi2 --gres=gpu:2  python main.py --model unet_extclass --scale 1 --patch_size 96 --save unet_classification_833ext --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 1 --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --h_hsize 3 --data_train DIV2K+CBSD500+WED --data_test WED+CBSD68 --split --n_GPUs 2 --test_every 2000 --loss 1*CE --predict_groups --lr 1e-4 --unetsize small --use_stats --no_augment




 ####### FAC conv #######


   srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual_fac --scale 1 --patch_size 96 --save smallbucketres_16layer_833class5_fac --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res



      srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model bucket_residual_kpn --scale 1 --patch_size 96 --save smallbucketres_16layer_833class5_kpn --from_legacy --ckpt_path ../checkpoint/carn.pth --reinit --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --unetsize small

srun -p Pixel --mpi=pmi2 --gres=gpu:2 python main.py --model carn --scale 1 --patch_size 96 --save bucketcarn_group4_feat16_833class5_loadtransfer8 --from_legacy  --ckpt_path ../experiment/bucketcarn_group4_feat16_833class5_transfer04/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+BSDWED4+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_range_bsdwed4 1-2450/1-2450 --data_test Set12+CBSD68 --task predenoise --n_threads 16 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 5 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 4e-5 --postfix ext --noiselevel 25