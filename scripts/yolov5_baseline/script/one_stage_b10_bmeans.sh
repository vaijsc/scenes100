#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov5_all.py \
    --train_whole 1 \
    --opt adapt \
    --model r101-fpn-3x \
    --ckpt ../../models/yolov5s_remap.pth \
    --config ../../configs/yolov5s.yaml \
    --tag seq.cluster.budget10 \
    --budget 10 \
    --iters 40000 \
    --eval_interval 1000 \
    --save_interval 2000 \
    --image_batch_size 28 \
    --num_workers 4 \
    --outputdir ./yolov5s_bs28_lr0.0001_teacherx1.5_conf0.4_b10_onestage40k_bmeans_fpn_new$2 \
    --lr 1e-4 \
    --split_list 0 1 2 3 4 -1 \
    --mapper yolov5s_bs28_lr0.0001_teacherx1.5_conf0.4/adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.23999.10means.fpn.$2.new.frombase.mapper.pth

    # change mapper