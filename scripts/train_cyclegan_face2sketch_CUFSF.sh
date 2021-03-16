#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/CUFSF_CycleGAN --name CUFSF_cyclegan --model cycle_gan --pool_size 50 --no_dropout
