#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/eval_lora.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --arch lora \
    --dataset scenes100 \
    --shot 1 \
    --prompt "traffic object ." \
    --setting unseen \
    --ckpt lora/scenes100_lora_unseen_p1/gdinoswint_scenes100_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . vehicle ." \
    --setting unseen \
    --ckpt lora/scenes100_lora_unseen_p2/gdinoswint_scenes100_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . pedestrian . vehicle . automobile . car ." \
    --setting unseen \
    --ckpt lora/scenes100_lora_unseen_p3/gdinoswint_scenes100_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen object ." \
    --setting unseen \
    --ckpt lora/egoper_lora_unseen_p1/gdinoswint_egoper_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --iters 2000 \
    --num_workers 2 \
    --arch lora \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen . cooking . human body ." \
    --setting unseen \
    --ckpt lora/egoper_lora_unseen_p2/gdinoswint_egoper_1shot_unseen_lora.pth \
    --lora_r 16
    
python finetune_groundingdino_accv.py \
    --opt eval \
    --iters 2000 \
    --num_workers 2 \
    --arch lora \
    --dataset egoper \
    --shot 1 \
    --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
    --setting unseen \
    --ckpt lora/egoper_lora_unseen_p3/gdinoswint_egoper_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset hoist \
    --shot 1 \
    --prompt "common object ." \
    --setting unseen \
    --ckpt lora/hoist_lora_unseen_p1/gdinoswint_hoist_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset hoist \
    --shot 1 \
    --prompt "object in hand ." \
    --setting unseen \
    --ckpt lora/hoist_lora_unseen_p2/gdinoswint_hoist_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset hoist \
    --shot 1 \
    --prompt "hand-held object ." \
    --setting unseen \
    --ckpt lora/hoist_lora_unseen_p3/gdinoswint_hoist_1shot_unseen_lora.pth \
    --lora_r 16


python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --ckpt lora/ovcoco_lora_unseen_p1/gdinoswint_ovcoco_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset ovcoco \
    --shot 1 \
    --prompt "open vocabulary coco classes ." \
    --setting unseen \
    --ckpt lora/ovcoco_lora_unseen_p2/gdinoswint_ovcoco_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset ovcoco \
    --shot 1 \
    --prompt "common objects ." \
    --setting unseen \
    --ckpt lora/ovcoco_lora_unseen_p3/gdinoswint_ovcoco_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --ckpt lora/rareplanes_lora_unseen_p1/gdinoswint_rareplanes_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset rareplanes \
    --shot 1 \
    --prompt "aircrafts ." \
    --setting unseen \
    --ckpt lora/rareplanes_lora_unseen_p2/gdinoswint_rareplanes_1shot_unseen_lora.pth \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset rareplanes \
    --shot 1 \
    --prompt "airplane . aeroplane . airliner . aircraft ." \
    --setting unseen \
    --ckpt lora/rareplanes_lora_unseen_p3/gdinoswint_rareplanes_1shot_unseen_lora.pth \
    --lora_r 16

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --arch bert \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --ckpt lora/rareplanes_bert_unseen_p3/gdinoswint_rareplanes_1shot_unseen_bert.pth                                                                