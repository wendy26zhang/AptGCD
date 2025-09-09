#!/bin/bash

set -e
set -x

#CUDA_VISIBLE_DEVICES=3
#nohup \
python train.py \
    --dataset_name 'cub' \
    --batch_size 64 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 0 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.12 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --thr 0.7 \
    --exp_name cub_aptgcd \
#    > train_cub_0125_k+v.log
#    --classnum 100
