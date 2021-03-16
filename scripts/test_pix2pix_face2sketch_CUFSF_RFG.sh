#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python test.py --dataroot ./datasets/CUFSF_RFG --name face2sketch_pix2pix_CUFSF_RFG --model pix2pix_three --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned_three --norm batch --loadSize 256
