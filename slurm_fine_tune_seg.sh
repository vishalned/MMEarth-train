#!/bin/bash
#SBATCH --job-name=geobench-seg-training

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=16G 
#SBATCH --array=1-6



datasets="geobench.m-cashew-plantation"
linear_probe=False # always false for segmentation
task_type=unet-ft-blr_sweep # for segmentation tasks this is just a random name

idx=$((SLURM_ARRAY_TASK_ID - 1))
pretraining="s2rgb-patch16-ip112-newstem-${idx}"
output_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/results"
log_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/logs"


output_dir="${output_dir_base}/${task_type}-${datasets}-${pretraining}"
log_dir="${log_dir_base}/${task_type}-${datasets}-${pretraining}"
dataset="${datasets}"

blr=(1e-2 5e-2 1e-3 5e-3 1e-4 5e-4)





python -m  main_finetune \
            --model convnextv2_unet_atto \
            --batch_size 32 \
            --update_freq 8 \
            --blr ${blr[$idx]}  \
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
            --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-s2rgb-patch16-ip112-newstem/checkpoint-199.pth \
            --output_dir "$output_dir" \
            --log_dir "$log_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe"\
            --auto_resume False \
            --pretraining $pretraining \
            --wandb True \
            --wandb_run_name "$task_type--$dataset--$pretraining--$idx" \
            --patch_size 16 \
            --input_size 112 \
            --use_orig_stem False \
            --run_on_test True



#             python -m  main_finetune \
#             --model convnextv2_unet_atto \
#             --batch_size 4 \
#             --update_freq 1 \
#             --blr 2e-4 \
#             --epochs 200 \
#             --warmup_epochs 0 \
#             --layer_decay_type 'single' \
#             --layer_decay 0.9 \
#             --weight_decay 0.3 \
#             --drop_path 0.1 \
#             --reprob 0.25 \
#             --mixup 0. \
#             --cutmix 0. \
#             --smoothing 0.2 \
#             --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/original_convnext_atto/convnextv2_atto_1k_224_fcmae.pt \
#             --output_dir "/projects/dereeco/data/global-lr/ConvNeXt-V2/results/testing" \
#             --data_set "geobench.m-cashew-plantation" \
#             --linear_probe False \
#             --pretraining testing\
#             --wandb False \
#             --auto_resume False \
#             --patch_size 32 \
#             --input_size 224 \
#             --use_orig_stem True \
#             --visualize False \
#             --eval False
# /projects/dereeco/data/global-lr/ConvNeXt-V2/results/original_convnext_atto/convnextv2_atto_1k_224_fcmae.pt
# /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-s2rgb-patch32-ip128-origstem/checkpoint-199.pth