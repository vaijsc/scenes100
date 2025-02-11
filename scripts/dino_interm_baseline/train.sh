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


python inference_server_simulate_dino.py \
    --train_whole 1 \
    --opt adapt \
    --model r101-fpn-3x \
    --ckpt ../../models/dino_b1_x1_iters24kbase.pth \
    --mapper ../dino_baseline/dino_x1_base/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.vanilla.iter.23999.10means.fpn.p4.p5.mapper.pth \
    --config ../../configs/dino_5scale.yaml \
    --tag budget10 \
    --iters 8000 \
    --eval_interval 8000 \
    --save_interval 8000 \
    --image_batch_size 2 \
    --num_workers 4 \
    --outputdir ./dino_x1_b10_2stage_interm_run7 \
    --lr 1e-4 \
    --refine_det_score_thres 0.3 \
    --budget 10 
    # --mapper random_mapping/mapper_random_60_b10.pth \

# python inference_server_simulate_dino.py \
#     --train_whole 1 \
#     --opt adapt \
#     --model r101-fpn-3x \
#     --config ../../configs/dino_5scale.yaml \
#     --ckpt ../../models/dino_b1_x1_iters24kbase.pth \
#     --tag budget100 \
#     --iters 8000 \
#     --eval_interval 8000 \
#     --save_interval 8000 \
#     --image_batch_size 2 \
#     --num_workers 4 \
#     --outputdir ./dino_x1_b100_2stage_interm_run6 \
#     --lr 1e-4 \
#     --refine_det_score_thres 0.3 \
#     --budget 100 \
    # --mapper random_mapping/mapper_random_150_b10.pth \
    # --id 001
    # --mapper ./dino_x1_base/adaptive_partial_server_detr_anno_allvideos_unlabeled_cocotrain.vanilla.frombase.10means.fpn.p4.p5.mapper.pth
    