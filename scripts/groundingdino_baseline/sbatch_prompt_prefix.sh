#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/prompt_prefix.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "traffic object ." \
#     --setting unseen \
#     --savedir prompt_prefix/scenes100_prompt_unseen_p1 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "person . vehicle ." \
#     --setting unseen \
#     --savedir prompt_prefix/scenes100_prompt_unseen_p2 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "person . pedestrian . vehicle . automobile . car ." \
#     --setting unseen \
#     --savedir prompt_prefix/scenes100_prompt_unseen_p3 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "kitchen object ." \
#     --setting unseen \
#     --savedir prompt_prefix/egoper_prompt_unseen_p1 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "kitchen . cooking . human body ." \
#     --setting unseen \
#     --savedir prompt_prefix/egoper_prompt_unseen_p2 \

    
# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
#     --setting unseen \
#     --savedir prompt_prefix/egoper_prompt_unseen_p3 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 20000 \
#     --eval_interval 5001 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "common object ." \
#     --setting unseen \
#     --savedir prompt_prefix/hoist_prompt_unseen_p1 \


python finetune_groundingdino_accv_prompt_prefix.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch prompt \
    --dataset hoist \
    --shot 1 \
    --prompt "object in hand ." \
    --setting unseen \
    --savedir prompt_prefix/hoist_prompt_unseen_p2 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 20000 \
#     --eval_interval 5001 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "hand-held object ." \
#     --setting unseen \
#     --savedir prompt_prefix/hoist_prompt_unseen_p3 \



# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 10000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset ovcoco \
#     --shot 1 \
#     --prompt "coco objects ." \
#     --setting unseen \
#     --savedir prompt_prefix/ovcoco_prompt_unseen_p1 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 10000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset ovcoco \
#     --shot 1 \
#     --prompt "open vocabulary coco classes ." \
#     --setting unseen \
#     --savedir prompt_prefix/ovcoco_prompt_unseen_p2 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 10000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset ovcoco \
#     --shot 1 \
#     --prompt "common objects ." \
#     --setting unseen \
#     --savedir prompt_prefix/ovcoco_prompt_unseen_p3 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "planes ." \
#     --setting unseen \
#     --savedir prompt_prefix/rareplanes_prompt_unseen_p1 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "aircrafts ." \
#     --setting unseen \
#     --savedir prompt_prefix/rareplanes_prompt_unseen_p2 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch prompt \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --setting unseen \
#     --savedir prompt_prefix/rareplanes_prompt_unseen_p3 \


# python finetune_groundingdino_accv_prompt_prefix.py \
#     --opt eval \
#     --arch bert \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --ckpt rareplanes_bert_unseen_p3/gdinoswint_rareplanes_1shot_unseen_bert.pth                                                                