#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python test.py --dataroot ./datasets/AR_CycleGAN --name AR_cyclegan --model cycle_gan --phase test --no_dropout
