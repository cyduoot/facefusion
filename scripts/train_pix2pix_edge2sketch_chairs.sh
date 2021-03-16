#!/usr/bin/env bash

python train.py --dataroot /home/bc/Work/MUNIT/datasets/edge2sketchchairs --name edge2sketchchairs_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
