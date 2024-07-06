#!/bin/bash
#SBATCH --job-name=benchmarking
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
#SBATCH --mem=64G
#SBATCH --array=1-4  # Set the array size according to the number of jobs you want to run (4 datasets * 2 tasks = 8 jobs)

############################################################################################################
# slurm scripts for fine-tuning on all other SOTA benchmarking models
############################################################################################################


pretraining=gassl-resnet50
datasets=("m-bigearthnet" "m-so2sat")
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

output_dir="${output_dir_base}/${task_type}-${dataset}-${pretraining}"
log_dir="${log_dir_base}/${task_type}-${dataset}-${pretraining}"


echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "dataset: $dataset"
echo "task_type: $task_type"
echo "output_dir: $output_dir"
echo "log_dir: $log_dir"


# Run python script for linear probe or fine-tuning
python -m main_finetune \
    --model resnet50 \
    --batch_size 512 \
    --update_freq 2 \
    --lr 1e-3 \
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
    --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/gassl_pretrain/moco_tp.pth.tar \
    --output_dir "$output_dir" \
    --log_dir "$log_dir" \
    --data_set "$dataset" \
    --linear_probe "$linear_probe" \
    --pretraining "$pretraining"\
    --wandb True \
    --wandb_run_name "$task_type--$dataset--$pretraining" \
    --auto_resume False \
    --patch_size 32 \
    --input_size 112 \
    --use_orig_stem True \
    --run_on_test True 

