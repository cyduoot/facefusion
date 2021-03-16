#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python test.py --dataroot ./datasets/CUHK --name face2sketch_pix2pix_CUHK --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch
