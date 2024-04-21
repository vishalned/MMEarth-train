#!/bin/bash
#SBATCH --job-name=geobench-seg-training

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=32G 
#SBATCH --array=1-2  # Set the array size according to the number of jobs you want to run (2 datasets, 2 tasks each)



datasets=("geobench.m-cashew-plantation" "geobench.m-SA-crop-type")
pretraining="pt-all_mod_tiny_v100"


task_type="unet-lp&ft"
linear_probe=True
blr=1e-2


output_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/results"

dataset_idx=$((SLURM_ARRAY_TASK_ID - 1))

output_dir="${output_dir_base}/${task_type}-${datasets[$dataset_idx]}-${pretraining}"
log_dir="${log_dir_base}/${task_type}-${datasets[$dataset_idx]}-${pretraining}"
dataset="${datasets[$dataset_idx]}"





python -m  main_finetune \
            --model convnextv2_unet_atto \
            --batch_size 32 \
            --update_freq 1 \
            --blr $blr  \
            --epochs 200 \
            --warmup_epochs 0 \
            --layer_decay_type 'single' \
            --layer_decay 0.9 \
            --weight_decay 0.3 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0. \
            --cutmix 0. \
            --smoothing 0 \
            --finetune "/projects/dereeco/data/global-lr/ConvNeXt-V2/results/${pretraining}/checkpoint-199.pth" \
            --output_dir "$output_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe"\
            --auto_resume False \
            --pretraining $pretraining \
            --wandb True \
            --wandb_run_name "$task_type--$dataset--$pretraining" \
            --wandb_project "tiny-geobench" \
            --patch_size 16 \
            --input_size 112 \
            --use_orig_stem False \
            --run_on_test True 



            # python -m  main_finetune \
            # --model convnextv2_unet_atto \
            # --batch_size 4 \
            # --update_freq 1 \
            # --blr 2e-4 \
            # --epochs 200 \
            # --warmup_epochs 0 \
            # --layer_decay_type 'single' \
            # --layer_decay 0.9 \
            # --weight_decay 0.3 \
            # --drop_path 0.1 \
            # --reprob 0.25 \
            # --mixup 0. \
            # --cutmix 0. \
            # --smoothing 0.2 \
            # --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-s2rgb-patch16-ip112-newstem/checkpoint-199.pth \
            # --output_dir "/projects/dereeco/data/global-lr/ConvNeXt-V2/results/testing" \
            # --data_set "geobench.m-cashew-plantation" \
            # --linear_probe True \
            # --pretraining testing\
            # --wandb False \
            # --auto_resume False \
            # --patch_size 16 \
            # --input_size 112 \
            # --use_orig_stem False \
            # --visualize False \
            # --eval False \
            # --save_ckpt False \
# /projects/dereeco/data/global-lr/ConvNeXt-V2/results/original_convnext_atto/convnextv2_atto_1k_224_fcmae.pt
# /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-s2rgb-patch32-ip128-origstem/checkpoint-199.pth