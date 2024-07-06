#!/bin/bash
#SBATCH --job-name=pre-training

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=14:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/pretraining-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=64G 


#############################################################################################################
# slurm script for pretraining the MP-MAE 
#############################################################################################################
params = ${1:-}
python  python --nproc_per_node=2 main_pretrain.py \
        --model convnextv2_pico \
        --batch_size 256 \
        --update_freq 8 \
        --blr 1.5e-4 \
        --epochs 200 \
        --warmup_epochs 40 \
        --data_dir /projects/dereeco/data/global-lr/data_1M_130_new \
        --output_dir /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-sanity_check_atto \
        --wandb True \
        --wandb_run_name sanity_check_atto \
        --wandb_project global-lr \
        --loss_aggr uncertainty \
        --auto_resume False \
        --norm_pix_loss True \
        --num_workers 15 \
        --patch_size 16 \
        --input_size 112 \
        --use_orig_stem False \
        --save_ckpt True $params

