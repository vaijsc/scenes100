# python inference_server_simulate_dino.py \
#     --train_whole 1 \
#     --opt adapt \
#     --model r101-fpn-3x \
#     --ckpt ../../models/dino_5scale_remap_orig.pth \
#     --config ../../configs/dino_5scale.yaml \
#     --tag budget1 \
#     --budget 1 \
#     --iters 32000 \
#     --eval_interval 2000 \
#     --save_interval 2000 \
#     --image_batch_size 2 \
#     --num_workers 4 \
#     --outputdir ./dino_x2_split_b1 \
#     --lr 1e-4 \
#     --refine_det_score_thres 0.3 \
    # --generic_query_ratio 0


python inference_server_simulate_dino_interm.py \
    --train_whole 1 \
    --opt adapt \
    --model r101-fpn-3x \
    --ckpt ../../models/dino_b1_x2_split_iters8kbase.pth \
    --mapper ./dino_x2_split_b1/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.budget1.10means.fpn.p5.mapper.pth \
    --config ../../configs/dino_5scale.yaml \
    --tag budget10 \
    --iters 2000 \
    --eval_interval 2000 \
    --save_interval 2000 \
    --image_batch_size 2 \
    --num_workers 4 \
    --outputdir ./dino_x2_split_b10_2stage_interm_run3 \
    --lr 1e-4 \
    --refine_det_score_thres 0.3 \
    --budget 10 \
#     --id 001
    # --mapper ./dino_x2_split_b1/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.budget1.frombase.10means.fpn.p5.mapper.pth \
    # --mapper random_mapping/mapper_random_20_b10.pth \

# python inference_server_simulate_dino_interm.py \
#     --train_whole 1 \
#     --opt adapt \
#     --model r101-fpn-3x \
#     --config ../../configs/dino_5scale.yaml \
#     --ckpt ../../models/dino_b1_x2_split_iters8kbase.pth \
#     --tag budget100 \
#     --iters 10000 \
#     --eval_interval 2000 \
#     --save_interval 2000 \
#     --image_batch_size 2 \
#     --num_workers 4 \
#     --outputdir ./dino_x2_split_b100_2stage_interm_run4 \
#     --lr 1e-4 \
#     --refine_det_score_thres 0.3 \
#     --budget 100 \
    # --id 001
#     # --mapper random_mapping/mapper_random_150_b10.pth \

#     # --mapper ./dino_x1_base/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.vanilla.frombase.10means.fpn.p4.p5.mapper.pth
#     # 
