#!/bin/bash

gpu=7
imb_factor=0.1
savename=cub200lt_if0.1
nohup python -u main.py --loss margin --batch_mining distance --log_online \
    --project DML_project --group cub200lt --seed 0 \
    --gpu $gpu --bs 112 --data_sampler class_random --samples_per_class 2 \
    --arch resnet50_frozen_normalize --source /hdd/hdd3/kjb/dataset \
    --n_epochs 150 --lr 0.00001 --embed_dim 128 --evaluate_on_gpu \
    --imb_factor $imb_factor --savename $savename \
    > test.out 2>&1 &

