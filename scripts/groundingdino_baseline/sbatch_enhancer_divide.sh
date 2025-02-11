#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/enhancer_divide.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "traffic object ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/scenes100_enhancer_unseen_p1 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2



# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "person . vehicle ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/scenes100_enhancer_unseen_p2 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "person . pedestrian . vehicle . automobile . car ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/scenes100_enhancer_unseen_p3 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "kitchen object ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/egoper_enhancer_unseen_p1 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "kitchen . cooking . human body ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/egoper_enhancer_unseen_p2 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2

    
# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 2000 \
#     --eval_interval 501 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/egoper_enhancer_unseen_p3 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 20000 \
#     --eval_interval 5001 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "common object ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/hoist_enhancer_unseen_p1 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 20000 \
#     --eval_interval 5001 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "object in hand ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/hoist_enhancer_unseen_p2 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt train \
#     --iters 20000 \
#     --eval_interval 5001 \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "hand-held object ." \
#     --setting unseen \
#     --savedir enhancer_lr_same${1}_divide$2/hoist_enhancer_unseen_p3 \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1 \
#     --divide $2



python finetune_groundingdino_accv_supp.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --savedir enhancer_lr_same${1}_divide$2/ovcoco_enhancer_unseen_p1 \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 \
    --divide $2


python finetune_groundingdino_accv_supp.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "open vocabulary coco classes ." \
    --setting unseen \
    --savedir enhancer_lr_same${1}_divide$2/ovcoco_enhancer_unseen_p2 \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 \
    --divide $2


python finetune_groundingdino_accv_supp.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "common objects ." \
    --setting unseen \
    --savedir enhancer_lr_same${1}_divide$2/ovcoco_enhancer_unseen_p3 \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 \
    --divide $2


python finetune_groundingdino_accv_supp.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --savedir enhancer_lr_same${1}_divide$2/rareplanes_enhancer_unseen_p1 \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 \
    --divide $2


python finetune_groundingdino_accv_supp.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "aircrafts ." \
    --setting unseen \
    --savedir enhancer_lr_same${1}_divide$2/rareplanes_enhancer_unseen_p2 \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 \
    --divide $2


python finetune_groundingdino_accv_supp.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "airplane . aeroplane . airliner . aircraft ." \
    --setting unseen \
    --savedir enhancer_lr_same${1}_divide$2/rareplanes_enhancer_unseen_p3 \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 \
    --divide $2


# python finetune_groundingdino_accv_supp.py \
#     --opt eval \
#     --arch bert \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --ckpt rareplanes_bert_unseen_p3/gdinoswint_rareplanes_1shot_unseen_bert.pth                                                                