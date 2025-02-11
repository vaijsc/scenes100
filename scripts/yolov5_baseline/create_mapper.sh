#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov5_clustering.py \
        --opt cluster \
        --model r101-fpn-3x \
        --ckpt ../../models/b1_x2_iters24kbase.pth \
        --ckpts_dir ./yolov5s_bs28_lr0.0001_teacherx2_conf0.4_b1 \
        --ckpts_tag adaptive_partial_server_yolov3_anno_allvideos_unlabeled_cocotrain.seq.cluster.budget1.iter.23999 \
        --config ../../configs/yolov5s.yaml \
        --budget 10 \
        --image_batch_size 28 \
        --split_list 0 1 2 3 4 -1 \
        
        # --random 1 \
        # --seed 10 20 30 \
        # --outputdir ./random_mapping
        # --from_base 1 \
