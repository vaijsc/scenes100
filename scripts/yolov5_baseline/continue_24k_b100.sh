#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python inference_server_simulate_yolov5_from_one_branch.py \
                    --train_whole 1 \
                    --opt adapt \
                    --model r101-fpn-3x \
                    --ckpt ../../models/b1_x2_iters24kbase.pth \
                    --config ../../configs/yolov5s.yaml \
                    --buffer_size 40 \
                    --tag seq.cluster.budget100 \
                    --budget 100 \
                    --iters 16000 \
                    --eval_interval 1000 \
                    --save_interval 2000 \
                    --image_batch_size 28 \
                    --num_workers 4 \
                    --outputdir ./yolov5s_bs28_lr0.0001_teacherx2_conf0.4_b100_continue_24k \
                    --lr 1e-4 \
                    --split_list 0 1 2 3 4 -1