#!/bin/bash
#SBATCH --job-name=eurosat

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-07:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=32G 



dataset=geobench.m-eurosat
linear_probe=True
task_type=lp # lp for linear probe, ft for fine-tuning
pretraining=pt-imnet-patch32-ip224-origstem--full

output_dir="/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/${task_type}-${dataset}_${pretraining}"




blr=(1e-2 5e-2 1e-3 5e-3 1e-4 5e-4 1e-5 5e-5)


python -m  main_finetune \
            --model convnextv2_atto \
            --batch_size 32 \
            --update_freq 1 \
            --blr 2e-4 \
            --epochs 100 \
            --warmup_epochs 0 \
            --layer_decay_type 'single' \
            --layer_decay 0.9 \
            --weight_decay 0.3 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0. \
            --cutmix 0. \
            --smoothing 0.2 \
            --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-imnet-patch32-ip224-origstem/checkpoint-199.pth \
            --output_dir "$output_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe" \
            --pretraining "$pretraining"\
            --wandb False \
            --wandb_run_name "$task_type--$dataset--$pretraining" \
            --patch_size 32 \
            --input_size 224 \
            --use_orig_stem True \
            --run_on_test True \
            --save_ckpt True



python -m  main_finetune \
            --model convnextv2_unet_atto \
            --batch_size 32 \
            --update_freq 2 \
            --blr 2e-4 \
            --epochs 200 \
            --warmup_epochs 0 \
            --layer_decay_type 'single' \
            --layer_decay 0.9 \
            --weight_decay 0.3 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0. \
            --cutmix 0. \
            --smoothing 0.2 \
            --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-all_mod_uncertainty/checkpoint-199.pth \
            --output_dir "/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/testing" \
            --data_set "geobench.m-bigearthnet" \
            --linear_probe True \
            --pretraining testing \
            --wandb False \
            --auto_resume False \
            --patch_size 16 \
            --input_size 112 \
            --use_orig_stem False \
            --save_ckpt False \
            --num_samples 50000

# python -m  main_finetune \
#             --model resnet50_unet \
#             --batch_size 32 \
#             --update_freq 2 \
#             --blr 1e-2 \
#             --epochs 200 \
#             --warmup_epochs 0 \
#             --layer_decay_type 'single' \
#             --layer_decay 0.9 \
#             --weight_decay 0.3 \
#             --drop_path 0.1 \
#             --reprob 0.25 \
#             --mixup 0. \
#             --cutmix 0. \
#             --smoothing 0 \
#             --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/satlas-pretrain/sentinel2_resnet50_si_rgb.pth \
#             --output_dir "/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/testing" \
#             --data_set "geobench.m-cashew-plantation" \
#             --linear_probe True \
#             --pretraining testing\
#             --wandb False \
#             --auto_resume False \
#             --patch_size 16 \
#             --input_size 112 \
#             --use_orig_stem False \
#             --save_ckpt False


#gassl_pretrain/moco_tp.pth.tar
#seco-pretrain/seco_resnet50_1m.pt
#satlas-pretrain/sentinel2_resnet50_si_rgb.pth



