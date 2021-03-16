#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/bc/Work/MUNIT/datasets/edge2sketchshoes --name edge2sketchshoes_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
