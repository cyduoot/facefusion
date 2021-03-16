#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python train.py --dataroot ./datasets/CUHK_CycleGAN --name CUHK_cyclegan --model cycle_gan --pool_size 50 --no_dropout
