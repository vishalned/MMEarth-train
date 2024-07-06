#!/bin/bash
#SBATCH --job-name=geobench-training
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/qbk152/vishal/slurm_logs/slurm-training-%A_%a_%x.out
#SBATCH --mem=32G
#SBATCH --array=1  # Set the array size according to the number of jobs you want to run (4 datasets * 2 tasks = 8 jobs)

############################################################################################################
# slurm script for finetuning on all geobench datasets with one pretraining model
############################################################################################################


pretraining=s2_12-patch16-ip112-newstem
datasets=("m-so2sat" "m-bigearthnet")
linear_probe=True
output_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/results"
log_dir_base="/projects/dereeco/data/global-lr/ConvNeXt-V2/logs"

dataset_idx=$(((SLURM_ARRAY_TASK_ID - 1) / 2))
task_type_idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 2 ))

# Retrieve dataset and task type from indices
dataset=${datasets[$dataset_idx]}
if [ $task_type_idx -eq 0 ]; then
    task_type="ft"
    linear_probe=False
else
    task_type="lp"
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
    --model convnextv2_atto \
    --batch_size 32 \
    --update_freq 1 \
    --blr 1e-4 \
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
    --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-s2rgb-patch16-ip112-newstem/checkpoint-199.pth \
    --output_dir "$output_dir" \
    --data_set "$dataset" \
    --linear_probe "$linear_probe" \
    --pretraining "$pretraining"\
    --wandb True \
    --wandb_run_name "testing-$task_type--$dataset--$pretraining" \
    --auto_resume False \
    --patch_size 16 \
    --input_size 112 \
    --use_orig_stem False \
    --run_on_test False \
    --save_ckpt False

