#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./datasets/CUFSF_CycleGAN --name CUFSF_cyclegan --model cycle_gan --phase test --no_dropout --loadSize 256
