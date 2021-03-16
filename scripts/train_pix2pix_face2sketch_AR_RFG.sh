#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py \
                              --dataroot ./datasets/AR_RFG \
                              --name face2sketch_pix2pix_AR_RFG \
                              --model pix2pix_attn \
                              --which_model_netD n_layers_attn \
                              --which_model_netG resnet_9blocks_attn \
                              --n_layers_D 5 \
                              --which_direction AtoB \
                              --lambda_A 100 \
                              --dataset_mode aligned_three \
                              --no_lsgan \
                              --batchSize 2 \
                              --norm batch \
                              --pool_size 0 \
                              --niter 250 \
                              --niter_decay 150

#if [ $? -eq 0 ]; then
#        python test.py \
#              --dataroot ./datasets/AR_RFG \
#              --name face2sketch_pix2pix_AR_RFG \
#              --model pix2pix_attn \
#              --which_model_netG resnet_9blocks_attn \
#              --which_direction AtoB \
#              --dataset_mode aligned_three \
#              --norm batch \
#              --loadSize 256
#fi
