#!/usr/bin/env bash
python train.py --dataroot ./datasets/AR --name e2e_AR --model e2e --which_direction AtoB --lambda_A 100 --dataset_mode paralleled --no_lsgan --norm batch --pool_size 0 --gpu_ids=2 --runname trysmall
