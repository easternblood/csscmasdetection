#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export OMP_NUM_THREADS=8 # you can change this value according to your number of cpu cores


py -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/tusimple.py
# py train.py --config configs/tusimple.py
# py test.py --config configs/tusimple.py --test_model 20220204_165429_lr_4e-04_b_128/ep099.pth --test_work_dir ./tmp
all
py test.py --config configs/tusimple.py --test_model 0323_1824_b102_zbt_mas_DSCSA/ep119.pth --test_work_dir ./

py demo.py --config configs/tusimple.py --test_model 0323_1824_b102_zbt_mas_DSCSA/ep119.pth --test_work_dir ./

py train.py --config configs/tusimple.py
py test.py --config configs/tusimple.py --test_model 0323_1824_b102_zbt_mas_DSCSA/ep199.pth --test_work_dir ./tmp

py test.py --config configs/tusimple.py --test_model 0314_1445_b102_zbt_mas_DSCSA/ep099.pth --test_work_dir ./tmp

py test.py --config configs/tusimple.py --test_model defaults/0311_2015_b102_zbt_DSCB/ep099.pth --test_work_dir ./tmp



py demo.py --config configs/tusimple.py --test_model 0310_1032_b102_RESNET/ep099.pth --test_work_dir ./

py demo.py --config configs/tusimple.py --test_model shuffle/0312_1501_b102_zbt_DSCSA/ep099.pth --test_work_dir ./

py demo.py --config configs/tusimple.py --test_model pools/0310_2102_b102_zbt_SOFT/ep099.pth --test_work_dir ./

py demo.py --config configs/tusimple.py --test_model defaults/0312_1154_b102_zbt_DSCC1/ep099.pth --test_work_dir ./

py demo.py --config configs/tusimple.py --test_model shuffle/0312_1501_b102_zbt_DSCSA/ep099.pth --test_work_dir ./

tensorboard --logdir 0316_1144_b102_zbt_mas_DSCSA --bind_all