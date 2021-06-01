# Class-specific Convolution for Image Denoising

## Requirements

- numpy
- scipy
- Pillow
- matplotlib
- Pytorch 
- opencv-python
- imageio
- numba

## Setup some CUDA packages

There are some CUDA based packages for fast GPU computation of some of our operations.

- `raisr_hash_cuda` is the CUDA implementation of getting RAISR values from an image.

- `CSKernelConv2D` is the CUDA implementation of class-specific convolution

To setup the packages, run
```angular2
sh install.sh
```
in their respective folders.

## Run the code

`main.py` is for training and testing. 

`option.py` contains the command-line options and their definitions.

The `model` folder contains different models used for our method and the baselines. `bucket_residual.py
`, `carn.py` and `red.py` can be used for class-specific convolution. 

### Prepare the data and train a model

Before training, download your datasets (e.g. DIV2K) as .png images, where the corresponding dataset directory should be  `dataset/DIV2K/`. In that directory, put your noise-free images into `/HR`, and the noisy images into `/LR_bicubic/X1_noise<noise level>`.

For faster training, store the the predicted classes into .npz files.

First, to get ground truth RAISR values (gradient statistics) of a dataset (for example, DIV2K), run
```
python main.py --model bucket_conv --scale 1 --patch_size 96 --save waste --from_legacy --ckpt_path ../checkpoint/carn.pth --batch_size 32 --save_results --pre_raisr --mask_type raisr --reinit --data_train DIV2K --data_test DIV2K --Qangle 8 --Qstr 8 --Qcohe 8 --h_hsize 3 --data_range 1-800/801-810 --task predenoise --n_threads 20 --save_gt --compute_grads
```

Then you may want to train a network to predict the RAISR values from noisy images, so run
```
python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_833_l1mse  --from_legacy --ckpt_path ../experiment/unetg2extres_regression_h4_833smallstat_divbsdwed_l1_stat/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 2 --Qcohe 2 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/801-810 --data_test DIV2K+CBSD68+WED --data_range_wed 1-4700/4701-4744  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-4 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment noiselevel 25
```


Then run the following to generate the predicted classes for all your datasets:
```
python main.py --model unet_g2ext --scale 1 --patch_size 96 --save unetg2extres_test  --from_legacy --ckpt_path ../experiment/unetg2extres_833_l1mse/model/model_latest.pt  --save_results --pre_raisr --mask_type raisr --Qangle 8 --Qstr 3 --Qcohe 3 --task denoise --n_threads 20 --save_gt --compute_grads --h_hsize 4 --data_train DIV2K+CBSD500+WED --data_range 1-800/1-810 --data_test DIV2K+CBSD68+WED+CBSD500+ADOBE5K --data_range_wed 1-4700/1-4744 --data_range_adobe 1-2/1-5000  --split --n_GPUs 2 --test_every 2000 --loss 1*L1+0.2*MSE --lr 1e-5 --batch_size 1 --halfing full --unetsize small --use_stats --no_augment --noiselevel 25
```

Then to use the predicted classes for denoising with CSConv, just move the `XXXX_SR833ext.npz` files in the `results-XXXX` folders into the `bin/LR_bicubic/X1_noiseXX` folders in the dataset directories. 

After that you can train a model with CSConv. For efficient convergence, we can first train a baseline model without CSConv:
```
python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallres_16layer_16feats_noise50 --from_legacy --ckpt_path ../experiment/smallres_16layer_x1_ydenoise25_divbsdadobewed_test/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix smallstat --small_res --noiselevel 50
```

Then use the learned convolution layers as initialization for CSConv training (with the option `load_transfer`):
```
python main.py --model bucket_residual --scale 1 --patch_size 96 --save smallbucketres_16layer_noise50_833class5_loadtransfer --from_legacy --ckpt_path ../experiment/smallres_16layer_16feats_noise50/model/model_best.pt --batch_size 4 --pre_raisr --save_results --mask_type raisr --data_train DIV2K+CBSD500+ADOBE5K+WED --data_range_adobe 1-5000/1-10 --data_test DIV2K+CBSD68 --task predenoise --n_threads 20 --save_gt --split --n_GPUs 2 --h_hsize 3 --n_layers 16 --Qstr 3 --Qcohe 3 --kernel_size 3 --n_feats 16 --from_regression --halfing full --no_augment --lr 1e-4 --postfix ext --small_res --load_transfer --noiselevel 50
```





