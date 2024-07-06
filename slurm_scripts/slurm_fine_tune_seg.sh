#!/bin/bash
#SBATCH --job-name=geobench-seg-training

#SBATCH --gres=gpu:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=32G 


#############################################################################################################
# slurm script for finetuning on a single segmentation datasets with one pretraining model
# we ensure the smoothing argument is 0. 
# we also ensure that the model has an additional "unet" word in the model name
#############################################################################################################


python -m  main_finetune \
            --model convnextv2_unet_atto \
            --batch_size 16 \
            --update_freq 2 \
            --blr 1e-2 \
            --epochs 200 \
            --warmup_epochs 0 \
            --layer_decay_type 'single' \
            --layer_decay 0.9 \
            --weight_decay 0.3 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0. \
            --cutmix 0. \
            --smoothing 0. \
            --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-s2rgb-patch16-ip112-newstem/checkpoint-199.pth \
            --output_dir "/projects/dereeco/data/global-lr/ConvNeXt-V2/results/testing" \
            --data_set "m-cashew-plantation" \
            --linear_probe True \
            --pretraining testing\
            --wandb True \
            --auto_resume False \
            --patch_size 16 \
            --input_size 112 \
            --use_orig_stem False \
            --visualize False \
            --eval False \
            --save_ckpt False \
	        --run_on_test True
