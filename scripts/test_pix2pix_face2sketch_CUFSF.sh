#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python test.py --dataroot ./datasets/CUFSF --name face2sketch_pix2pix_CUFSF --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --loadSize 256
