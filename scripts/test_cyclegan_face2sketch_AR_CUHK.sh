#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./datasets/AR_CUHK_CycleGAN --name AR_CUHK_cyclegan --model cycle_gan --phase test --no_dropout