#!/bin/bash
#SBATCH --job-name=dist-shift
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-14:00:00
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
#SBATCH --mem=64G
#SBATCH --array=1-4



# model="pt-imnet-patch32-ip224-origstem"
# model="pt-s2rgb-patch32-ip128-newstem"
# model="pt-s2rgb-patch16-ip128-newstem"
# model="pt-s2rgb-patch16-ip112-newstem"
# model="pt-s2_12-patch16-ip112-newstem"
model="pt-all_mod_uncertainty"
patch_size=16
input_size=112
origstem=False

pretraining=$model
datasets=("geobench.m-bigearthnet" "geobench.m-so2sat")
linear_probe=False
output_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/results"
log_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/logs"

dataset_idx=$(((SLURM_ARRAY_TASK_ID - 1) / 2))
task_type_idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 2 ))

# Retrieve dataset and task type from indices
dataset=${datasets[$dataset_idx]}
if [ $task_type_idx -eq 0 ]; then
    task_type="lp"
    linear_probe=True
else
    task_type="ft"
fi

output_dir="${output_dir_base}/${task_type}-${dataset}-${pretraining}-dist"


echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "dataset: $dataset"
echo "task_type: $task_type"
echo "output_dir: $output_dir"
echo "log_dir: $log_dir"


# Run python script for linear probe or fine-tuning
python -m main_finetune \
    --model convnextv2_atto \
    --batch_size 64 \
    --update_freq 16 \
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
    --finetune "/projects/dereeco/data/global-lr/ConvNeXt-V2/results/${pretraining}/checkpoint-199.pth" \
    --output_dir "$output_dir" \
    --data_set "$dataset" \
    --linear_probe "$linear_probe" \
    --pretraining "$pretraining"\
    --wandb True \
    --wandb_run_name "dist-$task_type--$dataset--$pretraining" \
    --auto_resume False \
    --patch_size $patch_size \
    --input_size $input_size \
    --use_orig_stem $origstem \
    --run_on_test True 
