#!/bin/bash
#SBATCH --job-name=pre-training

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-24:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/pretraining-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=64G 



#############################################################################################################
# slurm script for pretraining the MP-MAE 
#############################################################################################################
# params = ${1:-}
python  main_pretrain.py \
        --model convnextv2_atto \
        --batch_size 256 \
        --update_freq 16 \
        --blr 1.5e-4 \
        --epochs 200 \
        --warmup_epochs 40 \
        --data_dir /projects/dereeco/data/global-lr/data_100k_v001/ \
        --output_dir /projects/dereeco/data/global-lr/ConvNeXt-V2/results_v001/pt-100k-v001/ \
        --wandb True \
        --wandb_run_name 100k_taster \
        --wandb_project mmearth-v001 \
        --loss_aggr uncertainty \
        --auto_resume True \
        --norm_pix_loss True \
        --num_workers 4 \
        --patch_size 16 \
        --input_size 112 \
        --use_orig_stem False \
        --save_ckpt True \
        --distributed False \

