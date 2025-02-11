#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/losa.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset scenes100 \
    --shot 1 \
    --prompt "traffic object ." \
    --setting unseen \
    --savedir scenes100_losa_unseen_p1 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . vehicle ." \
    --setting unseen \
    --savedir scenes100_losa_unseen_p2 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . pedestrian . vehicle . automobile . car ." \
    --setting unseen \
    --savedir scenes100_losa_unseen_p3 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen object ." \
    --setting unseen \
    --savedir egoper_losa_unseen_p1 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen . cooking . human body ." \
    --setting unseen \
    --savedir egoper_losa_unseen_p2 \
    --losa_r 16
    
python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset egoper \
    --shot 1 \
    --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
    --setting unseen \
    --savedir egoper_losa_unseen_p3 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset hoist \
    --shot 1 \
    --prompt "common object ." \
    --setting unseen \
    --savedir hoist_losa_unseen_p1 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset hoist \
    --shot 1 \
    --prompt "object in hand ." \
    --setting unseen \
    --savedir hoist_losa_unseen_p2 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset hoist \
    --shot 1 \
    --prompt "hand-held object ." \
    --setting unseen \
    --savedir hoist_losa_unseen_p3 \
    --losa_r 16


python finetune_groundingdino_accv.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --savedir ovcoco_losa_unseen_p1 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset ovcoco \
    --shot 1 \
    --prompt "open vocabulary coco classes ." \
    --setting unseen \
    --savedir ovcoco_losa_unseen_p2 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset ovcoco \
    --shot 1 \
    --prompt "common objects ." \
    --setting unseen \
    --savedir ovcoco_losa_unseen_p3 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --savedir rareplanes_losa_unseen_p1 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset rareplanes \
    --shot 1 \
    --prompt "aircrafts ." \
    --setting unseen \
    --savedir rareplanes_losa_unseen_p2 \
    --losa_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch losa \
    --dataset rareplanes \
    --shot 1 \
    --prompt "airplane . aeroplane . airliner . aircraft ." \
    --setting unseen \
    --savedir rareplanes_losa_unseen_p3 \
    --losa_r 16

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --arch bert \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --ckpt rareplanes_bert_unseen_p3/gdinoswint_rareplanes_1shot_unseen_bert.pth                                                                