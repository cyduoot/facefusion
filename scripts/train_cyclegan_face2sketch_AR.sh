#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/AR_CycleGAN --name AR_cyclegan --model cycle_gan --pool_size 50 --no_dropout
