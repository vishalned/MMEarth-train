#!/bin/bash
#SBATCH --job-name=sweep

#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
# PATH TO SAVE SLURM LOGS
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=32G 
# NUMBER OF JOBS TO RUN
#SBATCH --array=1-9



datasets=("geobench.m-bigearthnet" "geobench.m-so2sat" "geobench.m-eurosat")
linear_probe=True
task_type=lp # lp for linear probe, ft for fine-tuning
pretraining=pt-all_mod_tiny_v100
# pretraining=pt-all_mod_uncertainty



# percents=(0.005 0.05 0.5)
samples=(100 1000 10000)
dataset_idx=$(( (SLURM_ARRAY_TASK_ID - 1) / 3))
idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
dataset=${datasets[$dataset_idx]}
num_samples=${samples[$idx]}
# percent=0.05
blr=2e-4

output_dir="/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/${task_type}-${dataset}_${pretraining}_${num_samples}"



python -m  main_finetune \
            --model convnextv2_atto \
            --batch_size 32 \
            --update_freq 1 \
            --blr $blr \
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
            --finetune "/projects/dereeco/data/global-lr/ConvNeXt-V2/results/${pretraining}/checkpoint-199.pth" \
            --output_dir "$output_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe" \
            --pretraining "$pretraining"\
            --wandb True \
            --wandb_run_name "subsample--$task_type--$percent--$dataset--$pretraining" \
            --auto_resume False \
            --patch_size 32 \
            --input_size 224 \
            --use_orig_stem True \
            --run_on_test True \
            --save_ckpt True \
            --num_samples $num_samples \
            # --percent $percent



