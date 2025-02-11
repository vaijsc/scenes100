#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python yolov8_rnn_new.py \
    --train_whole 1 \
    --opt adapt \
    --dataset scenes100 \
    --ckpt ../../models/yolov8s_remap.pth \
    --config ../../configs/yolov8s.yaml \
    --tag scenes100 \
    --iters 4000 \
    --eval_interval 1000 \
    --save_interval 1000 \
    --image_batch_size 10 \
    --num_workers 0 \
    --outputdir ./new_init_scenes100_train_whole/yolov8s_rnn_bs10_lr0.0001_innerlr0.00001_teacherx2_conf0.4_scenes100_vid007 \
    --lr 1e-4 \
    --agg_lr 1e-5 \
    --id '007' \
    # --seed $2

    # ./yolov8s_rnn_bs10_lr0.0001_innerlr0.00001_teacherx2_conf0.4_scenes100_vid005
    # --full_ckpt ./yolov8s_rnn_bs10_lr0.0001_innerlr0.00001_teacherx2_conf0.4_scenes100_vid007/adaptive_partial_server_yolov8s_rnn_anno_allvideos_unlabeled_cocotrain.scenes100.iter.4000.pth \
    