#!/bin/bash

# CUDA_VISIBLE_DEVICES=$1 python yolov8_rnn_new.py \
#     --opt adapt \
#     --dataset imagenet_vid \
#     --ckpt ../../models/yolov8s_imagenet_vid.pth \
#     --config ../../configs/yolov8s_imagenet_vid.yaml \
#     --tag imagenet_vid \
#     --iters 200 \
#     --eval_interval 20 \
#     --save_interval 20 \
#     --image_batch_size 10 \
#     --num_workers 0 \
#     --outputdir ./test13 \
#     --lr 1e-4 \
#     --agg_lr 1e-5 \

    # --seed $2
    # ./yolov8s_rnn_bs10_lr0.0001_imagenet_vid_train_whole
    # --train_whole 1 \


for id in {192..192}
do
    CUDA_VISIBLE_DEVICES=0 python yolov8_rnn_test.py \
    --opt adapt \
    --dataset imagenet_vid \
    --ckpt ../../models/yolov8s_imagenet_vid_fgfa.pth \
    --config ../../configs/yolov8s_imagenet_vid.yaml \
    --tag imagenet_vid \
    --iters 100 \
    --eval_interval 20 \
    --save_interval 20 \
    --image_batch_size 10 \
    --num_workers 0 \
    --outputdir ./test_trained_192 \
    --lr 1e-4 \
    --agg_lr 1e-5 \
    --id $id
done

 # --full_ckpt ./test_imagenet_vid_1vid_fgfa_shuffled_valided/test$((id+1))/best.pth \