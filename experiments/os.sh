#!/bin/bash


gpu=6
imb_factor=0.1
savename=final_if0.1_sr0.3
nohup python -u main.py --loss b_triplet --batch_mining distance --log_online \
    --project DML_project --group cub200lt --seed 0 --proto tail \
    --gpu $gpu --bs 112 --data_sampler lt_sampler --samples_per_class 2 \
    --arch resnet50_frozen_normalize --source /hdd/hdd3/kjb/dataset \
    --n_epochs 150 --lr 0.00001 --embed_dim 128 --evaluate_on_gpu \
    --imb_factor $imb_factor --savename $savename --sampling_ratio 0.3 \
    > outputs/${savename}.out 2>&1 &

gpu=7
imb_factor=0.1
savename=final_if0.1_sr0.1
nohup python -u main.py --loss b_triplet --batch_mining distance --log_online \
    --project DML_project --group cub200lt --seed 0 --proto tail \
    --gpu $gpu --bs 112 --data_sampler lt_sampler --samples_per_class 2 \
    --arch resnet50_frozen_normalize --source /hdd/hdd3/kjb/dataset \
    --n_epochs 150 --lr 0.00001 --embed_dim 128 --evaluate_on_gpu \
    --imb_factor $imb_factor --savename $savename --sampling_ratio 0.1 \
    > outputs/${savename}.out 2>&1 &

