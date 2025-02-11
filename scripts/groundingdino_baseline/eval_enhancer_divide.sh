#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/eval_enhancer_divide8.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here

for i in {000..007}; do
python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . vehicle ." \
    --setting unseen \
    --ckpt enhancer_lr_same${1}_divide$2/scenes100_enhancer_unseen_p2/gdinoswint_scenes100_1shot_unseen_enhancer.$i.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1 
done

for i in {000..007}; do
python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen object ." \
    --setting unseen \
    --ckpt enhancer_lr_same${1}_divide$2/egoper_enhancer_unseen_p1/gdinoswint_egoper_1shot_unseen_enhancer.$i.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1
done

for i in {000..007}; do
python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "hand-held object ." \
    --setting unseen \
    --ckpt enhancer_lr_same${1}_divide$2/hoist_enhancer_unseen_p3/gdinoswint_hoist_1shot_unseen_enhancer.$i.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1
done

for i in {000..007}; do
python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --ckpt enhancer_lr_same${1}_divide$2/ovcoco_enhancer_unseen_p1/gdinoswint_ovcoco_1shot_unseen_enhancer.$i.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1
done

for i in {000..007}; do
python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --ckpt enhancer_lr_same${1}_divide$2/rareplanes_enhancer_unseen_p1/gdinoswint_rareplanes_1shot_unseen_enhancer.$i.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1
done

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --arch bert \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --ckpt enhancer_lr_same${1}_divide$2/rareplanes_bert_unseen_p3/gdinoswint_rareplanes_1shot_unseen_bert.pth                                                                