#!/bin/bash
#SBATCH --job-name=truongvq1         
#SBATCH --output=train_all_vid_imagenet_vid.txt       
#SBATCH --gpus=1         
#SBATCH --cpus-per-gpu=32       
#SBATCH --mem-per-gpu=36GB                 
#SBATCH --partition=research        

# Your commands here
bash train_imagenet_vid_1.sh 