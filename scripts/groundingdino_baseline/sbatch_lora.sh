#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/lora.txt       
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
    --arch lora \
    --dataset scenes100 \
    --shot 1 \
    --prompt "traffic object ." \
    --setting unseen \
    --savedir lora/scenes100_lora_unseen_p1 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . vehicle ." \
    --setting unseen \
    --savedir lora/scenes100_lora_unseen_p2 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . pedestrian . vehicle . automobile . car ." \
    --setting unseen \
    --savedir lora/scenes100_lora_unseen_p3 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen object ." \
    --setting unseen \
    --savedir lora/egoper_lora_unseen_p1 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen . cooking . human body ." \
    --setting unseen \
    --savedir lora/egoper_lora_unseen_p2 \
    --lora_r 16
    
python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset egoper \
    --shot 1 \
    --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
    --setting unseen \
    --savedir lora/egoper_lora_unseen_p3 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset hoist \
    --shot 1 \
    --prompt "common object ." \
    --setting unseen \
    --savedir lora/hoist_lora_unseen_p1 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset hoist \
    --shot 1 \
    --prompt "object in hand ." \
    --setting unseen \
    --savedir lora/hoist_lora_unseen_p2 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 20000 \
    --eval_interval 5001 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset hoist \
    --shot 1 \
    --prompt "hand-held object ." \
    --setting unseen \
    --savedir lora/hoist_lora_unseen_p3 \
    --lora_r 16


python finetune_groundingdino_accv.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --savedir lora/ovcoco_lora_unseen_p1 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset ovcoco \
    --shot 1 \
    --prompt "open vocabulary coco classes ." \
    --setting unseen \
    --savedir lora/ovcoco_lora_unseen_p2 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 10000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset ovcoco \
    --shot 1 \
    --prompt "common objects ." \
    --setting unseen \
    --savedir lora/ovcoco_lora_unseen_p3 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --savedir lora/rareplanes_lora_unseen_p1 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset rareplanes \
    --shot 1 \
    --prompt "aircrafts ." \
    --setting unseen \
    --savedir lora/rareplanes_lora_unseen_p2 \
    --lora_r 16

python finetune_groundingdino_accv.py \
    --opt train \
    --iters 2000 \
    --eval_interval 501 \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch lora \
    --dataset rareplanes \
    --shot 1 \
    --prompt "airplane . aeroplane . airliner . aircraft ." \
    --setting unseen \
    --savedir lora/rareplanes_lora_unseen_p3 \
    --lora_r 16

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --arch bert \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --ckpt rareplanes_bert_unseen_p3/gdinoswint_rareplanes_1shot_unseen_bert.pth                                                                