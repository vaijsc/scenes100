# python inference_server_simulate_dino.py \
#     --train_whole 1 \
#     --opt adapt \
#     --model r101-fpn-3x \
#     --ckpt ../../models/dino_5scale_remap_orig.pth \
#     --config ../../configs/dino_5scale.yaml \
#     --tag vanilla \
#     --iters 40000 \
#     --eval_interval 2000 \
#     --save_interval 2000 \
#     --image_batch_size 2 \
#     --num_workers 4 \
#     --outputdir ./dino_x1_b100_test_001 \
#     --lr 1e-4 \
#     --refine_det_score_thres 0.3 \
#     --budget 100


# python inference_server_simulate_dino.py \
#     --train_whole 1 \
#     --opt adapt \
#     --model r101-fpn-3x \
#     --ckpt ../../models/dino_5scale_remap_orig.pth \
#     --config ../../configs/dino_5scale.yaml \
#     --tag budget10 \
#     --iters 32000 \
#     --eval_interval 32000 \
#     --save_interval 32000 \
#     --image_batch_size 2 \
#     --num_workers 4 \
#     --outputdir ./dino_x1_b10_1stage_p4p5_run4 \
#     --lr 1e-4 \
#     --refine_det_score_thres 0.3 \
#     --budget 10 \
#     --mapper ./dino_x1_base/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.vanilla.frombase.10means.fpn.p4.p5.mapper.pth \
    # --mapper random_mapping/mapper_random_40_b10.pth \


python inference_server_simulate_dino.py \
    --train_whole 1 \
    --opt adapt \
    --model r101-fpn-3x \
    --config ../../configs/dino_5scale.yaml \
    --ckpt ../../models/dino_5scale_remap_orig.pth \
    --tag budget100 \
    --iters 32000 \
    --eval_interval 32000 \
    --save_interval 32000 \
    --image_batch_size 2 \
    --num_workers 4 \
    --outputdir ./dino_x1_b100_1stage_run4 \
    --lr 1e-4 \
    --refine_det_score_thres 0.3 \
    --budget 100 \
    # --id 001 \
    # --mapper random_mapping/mapper_random_150_b10.pth \
    # --mapper ./dino_x1_base/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.vanilla.frombase.10means.fpn.p4.p5.mapper.pth
