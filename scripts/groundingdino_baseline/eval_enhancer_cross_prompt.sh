#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/eval_enhancer_cross_prompt.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --arch enhancer \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "traffic object ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p1/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "traffic object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p2/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "traffic object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p3/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "kitchen object ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p1/gdinoswint_egoper_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --iters 2000 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p2/gdinoswint_egoper_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1
    
python finetune_groundingdino_accv.py \
    --opt eval \
    --iters 2000 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p3/gdinoswint_egoper_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "common object ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p1/gdinoswint_hoist_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "common object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p2/gdinoswint_hoist_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "common object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p3/gdinoswint_hoist_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1


# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset ovcoco \
#     --shot 1 \
#     --prompt "coco objects ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p1/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p2/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "coco objects ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p3/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "planes ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p1/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p2/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "planes ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p3/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# P2

#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/eval_enhancer.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . vehicle ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p1/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "person . vehicle ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p2/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . vehicle ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p3/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen . cooking . human body ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p1/gdinoswint_egoper_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --iters 2000 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "kitchen . cooking . human body ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p2/gdinoswint_egoper_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1
    
python finetune_groundingdino_accv.py \
    --opt eval \
    --iters 2000 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "kitchen . cooking . human body ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p3/gdinoswint_egoper_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "object in hand ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p1/gdinoswint_hoist_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "object in hand ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p2/gdinoswint_hoist_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "object in hand ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p3/gdinoswint_hoist_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1


python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "open vocabulary coco classes ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p1/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset ovcoco \
#     --shot 1 \
#     --prompt "open vocabulary coco classes ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p2/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "open vocabulary coco classes ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p3/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "aircrafts ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p1/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "aircrafts ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p2/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "aircrafts ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p3/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1


# P3

#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=eval_log/eval_enhancer.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here


python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . pedestrian . vehicle . automobile . car ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p1/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset scenes100 \
    --shot 1 \
    --prompt "person . pedestrian . vehicle . automobile . car ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p2/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset scenes100 \
#     --shot 1 \
#     --prompt "person . pedestrian . vehicle . automobile . car ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/scenes100_enhancer_unseen_p3/gdinoswint_scenes100_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p1/gdinoswint_egoper_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --iters 2000 \
    --num_workers 2 \
    --arch enhancer \
    --dataset egoper \
    --shot 1 \
    --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p2/gdinoswint_egoper_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1
    
# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --iters 2000 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset egoper \
#     --shot 1 \
#     --prompt "appliance . utensil . cutlery . seasoning . ingredient . food . hand ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/egoper_enhancer_unseen_p3/gdinoswint_egoper_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "hand-held object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p1/gdinoswint_hoist_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset hoist \
    --shot 1 \
    --prompt "hand-held object ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p2/gdinoswint_hoist_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset hoist \
#     --shot 1 \
#     --prompt "hand-held object ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/hoist_enhancer_unseen_p3/gdinoswint_hoist_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1


python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "common objects ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p1/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset ovcoco \
    --shot 1 \
    --prompt "common objects ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p2/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset ovcoco \
#     --shot 1 \
#     --prompt "common objects ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/ovcoco_enhancer_unseen_p3/gdinoswint_ovcoco_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "airplane . aeroplane . airliner . aircraft ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p1/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

python finetune_groundingdino_accv.py \
    --opt eval \
    --image_batch_size 2 \
    --num_workers 2 \
    --arch enhancer \
    --dataset rareplanes \
    --shot 1 \
    --prompt "airplane . aeroplane . airliner . aircraft ." \
    --setting unseen \
    --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p2/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
    --enhancer_r_qkv $1 \
    --enhancer_r_ff $1

# python finetune_groundingdino_accv.py \
#     --opt eval \
#     --image_batch_size 2 \
#     --num_workers 2 \
#     --arch enhancer \
#     --dataset rareplanes \
#     --shot 1 \
#     --prompt "airplane . aeroplane . airliner . aircraft ." \
#     --setting unseen \
#     --ckpt enhancer_lr_same$1/rareplanes_enhancer_unseen_p3/gdinoswint_rareplanes_1shot_unseen_enhancer.pth \
#     --enhancer_r_qkv $1 \
#     --enhancer_r_ff $1
                                                             