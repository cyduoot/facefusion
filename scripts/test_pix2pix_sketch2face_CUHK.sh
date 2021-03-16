#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./datasets/CUHK_gen --name sketch2face_pix2pix_CUHK --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode aligned --norm batch
