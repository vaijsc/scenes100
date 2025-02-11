#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python yolov8_rnn.py \
#     --opt adapt \
#     --dataset imagenet_vid \
#     --ckpt ../../models/yolov8s_imagenet_vid_fgfa.pth \
#     --config ../../configs/yolov8s_imagenet_vid.yaml \
#     --tag imagenet_vid \
#     --iters 40000 \
#     --eval_interval 100 \
#     --save_interval 100 \
#     --image_batch_size 10 \
#     --num_workers 0 \
#     --outputdir ./test_imagenet_allvid \
#     --lr 1e-4 \
#     --agg_lr 1e-5 \

# --seed $2
# ./yolov8s_rnn_bs10_lr0.0001_imagenet_vid_train_whole
# --train_whole 1 \



CUDA_VISIBLE_DEVICES=0 python yolov8_rnn_new_pred_multistep.py \
    --opt adapt \
    --dataset imagenet_vid \
    --ckpt ../../models/yolov8s_imagenet_vid_fgfa.pth \
    --config ../../configs/yolov8s_imagenet_vid.yaml \
    --tag imagenet_vid \
    --iters 2000 \
    --eval_interval 20 \
    --save_interval 20 \
    --image_batch_size 10 \
    --num_workers 0 \
    --outputdir ./test \
    --lr 1e-3 \
    --agg_lr 1e-4 \
    --num_step 1 \
    --id 48 \
    # --outputdir ./test49_target_gt_new_pred_new_G_new_init_outerlr10_3_unlockF \
# --full_ckpt test49_valid_shuffled_fgfa_longtrain/adaptive_partial_server_yolov8s_rnn_anno_allvideos_valid.imagenet_vid.iter.600.pth \