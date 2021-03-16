#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/CUFSF_RF1G --name face2sketch_pix2pix_CUFSF_RF1G --model pix2pix_three --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned_three --no_lsgan --norm batch --pool_size 0
