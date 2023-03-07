#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export OMP_NUM_THREADS=8 # you can change this value according to your number of cpu cores


py -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/tusimple.py
# python train.py configs/tusimple.py
# py train.py --config configs/tusimple.py