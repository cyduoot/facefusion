#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python test.py \
                              --dataroot ./datasets/AR_RFG \
                              --name face2sketch_pix2pix_AR_RFG \
                              --model pix2pix_attn \
                              --which_model_netG resnet_9blocks_attn \
                              --which_direction AtoB \
                              --dataset_mode aligned_three \
                              --norm batch \
                              --loadSize 256
