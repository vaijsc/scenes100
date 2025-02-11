#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov5_all.py \
        --opt cluster \
        --model r101-fpn-3x \
        --ckpt ../../models/yolov5s_remap.pth \
        --ckpts_dir ./yolov5s_bs28_lr0.0001_teacherx1.5_conf0.4 \
        --ckpts_tag adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.23999 \
        --config ../../configs/yolov5s.yaml \
        --budget 10 \
        --image_batch_size 28 \
        --split_list 0 1 2 -1 \
        --from_base 1 \
        # --random 1 \
        # --seed 10 20 30 \
        # --outputdir ./random_mapping


