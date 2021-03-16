#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/bc/Work/MUNIT/datasets/edge2sketchchairs --name edge2sketchchairs_cyclegan --model cycle_gan --pool_size 50 --no_dropout
