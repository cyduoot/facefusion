#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/CUFSF --name face2sketch_pix2pix_CUFSF --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
