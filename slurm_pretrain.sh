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
        --loss_full False \
        --auto_resume True \
        --norm_pix_loss True \
        --num_workers 8 \
        --patch_size 16 \
        --input_size 112 \
        --random_crop True \
        --use_orig_stem False \
        --IMNET False \
        --save_ckpt True



# python  -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
#         --model convnextv2_atto \
#         --batch_size 32 \
#         --update_freq 2 \
#         --blr 1.5e-4 \
#         --epochs 200 \
#         --warmup_epochs 40 \
#         --data_path /projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new.h5 \
#         --output_dir /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-all_mod_uncertainty_norm_pix_all \
#         --wandb False \
#         --wandb_run_name all_mod_uncertainty_norm_pix_all \
#         --loss_type L2 \
#         --loss_aggr unweighted \
#         --loss_full False \
#         --auto_resume False \
#         --norm_pix_loss True \
#         --num_workers 8 \
#         --patch_size 16 \
#         --input_size 112 \
#         --random_crop True \
#         --use_orig_stem False \
#         --IMNET False \
#         --save_ckpt False


        # python   -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
        # --model convnextv2_atto \
        # --batch_size 16 \
        # --update_freq 2 \
        # --blr 1.5e-4 \
        # --epochs 200 \
        # --warmup_epochs 40 \
        # --data_path /home/qbk152/vishal/global-lr/data/data_1M_130_new/data_1M_130_new.h5 \
        # --output_dir /projects/dereeco/data/global-lr/ConvNeXt-V2/results/testing \
        # --wandb True \
        # --wandb_run_name testing  \
        # --loss_type L2 \
        # --loss_aggr uncertainty \
        # --loss_full False \
        # --auto_resume False \
        # --norm_pix_loss True \
        # --num_workers 8 \
        # --patch_size 16 \
        # --input_size 112 \
        # --random_crop True \
        # --use_orig_stem False \
        # --IMNET False \
        # --save_ckpt False