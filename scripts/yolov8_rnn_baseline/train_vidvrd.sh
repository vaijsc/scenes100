#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python yolov8_rnn.py \
    --opt adapt \
    --dataset vidvrd \
    --ckpt ../../models/yolov8s_vidvrd.pth \
    --config ../../configs/yolov8s_vidvrd.yaml \
    --tag test \
    --iters 20000 \
    --eval_interval 5000 \
    --save_interval 5000 \
    --image_batch_size 10 \
    --num_workers 0 \
    --outputdir ./yolov8s_rnn_bs10_lr0.0001_vidvrd_test \
    --lr 1e-4 \
    # --seed $2
    # --train_whole 0 \