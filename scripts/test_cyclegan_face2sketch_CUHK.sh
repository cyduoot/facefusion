#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./datasets/CUHK_CycleGAN --name CUHK_cyclegan --model cycle_gan --phase test --no_dropout --loadSize 256
