#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./datasets/CUHK_RF1G --name face2sketch_pix2pix_CUHK_RF1G --model pix2pix_three --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned_three --norm batch --loadSize 256
