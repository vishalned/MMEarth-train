#!/bin/bash
#SBATCH --job-name=pre-training

#SBATCH --gres=gpu:titanrtx:2
#SBATCH --tasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/pretraining-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=64G 


#############################################################################################################
# slurm script for pretraining the MP-MAE 
#############################################################################################################

python  -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
        --model convnextv2_pico \
        --batch_size 256 \
        --update_freq 8 \
        --blr 1.5e-4 \
        --epochs 200 \
        --warmup_epochs 40 \
        --data_path /projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new.h5 \
        --output_dir /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-sanity_check_atto \
        --wandb True \
        --wandb_run_name sanity_check_atto \
        --wandb_project global-lr \
        --loss_aggr uncertainty \
        --auto_resume False \
        --norm_pix_loss True \
        --num_workers 8 \
        --patch_size 16 \
        --input_size 112 \
        --random_crop True \
        --use_orig_stem False \
        --save_ckpt True

