#!/bin/bash
#SBATCH --job-name=geobench-training
#SBATCH --array=1-4  # Set the array size according to the number of jobs you want to run (3 datasets * 2 tasks = 6 jobs)
#SBATCH -p page --gres=gpu:1 --constraint="V100"
#SBATCH --tasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
# PATH TO SAVE SLURM LOGS
##SBATCH --output=./slurm_logs/pretraining_v100-%A_%a_%x.out
# TOTAL MEMORY PER NODE
#SBATCH --mem=50G

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak@di.ku.dk

##export SCRATCH=$SCRATCH/$SLURM_ARRAY_TASK_ID
##mkdir -p $SCRATCH


#############################################################################################################
# slurm script for finetuning on all classification datasets with one pretraining model on a v100 for the tiny model
#############################################################################################################


cleanup() {
    rm -rf $SCRATCH/*
    exit 0
}

trap 'cleanup' SIGTERM

hostname
echo $CUDA_VISIBLE_DEVICES
which python

mkdir $SCRATCH/tmp
export TMPDIR=$SCRATCH/tmp

time cp --sparse=always -r /groups/scienceai/ankit/benchmarking_data/ $SCRATCH

export GEO_BENCH_DIR=$SCRATCH/benchmarking_data

model="pt-all_mod_tiny_v100"
patch_size=16
input_size=112
origstem=False


pretraining=$model
datasets=("m-bigearthnet" "m-so2sat")
linear_probe=False
output_dir_base="./results"
log_dir_base="./logs"


dataset_idx=$(((SLURM_ARRAY_TASK_ID - 1) / 2))
task_type_idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 2))

# Retrieve dataset and task type from indices
dataset=${datasets[$dataset_idx]}
if [ $task_type_idx -eq 0 ]; then
    task_type="lp"
    linear_probe=True
    blr=1e-2
else
    task_type="ft"
    blr=1e-4
fi



output_dir="${output_dir_base}/${task_type}-${dataset}-${pretraining}"
log_dir="${log_dir_base}/${task_type}-${dataset}-${pretraining}"

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "dataset: $dataset"
echo "task_type: $task_type"
echo "output_dir: $output_dir"
echo "log_dir: $log_dir"


# Run python script for linear probe or fine-tuning
WANDB__SERVICE_WAIT=300 python -m main_finetune \
    --model convnextv2_tiny \
    --batch_size 128 \
    --update_freq 8 \
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
    --finetune "./results/${pretraining}/checkpoint-199.pth" \
    --output_dir "$output_dir" \
    --data_set "$dataset" \
    --linear_probe "$linear_probe" \
    --pretraining "$pretraining"\
    --wandb True \
    --wandb_run_name "$task_type--$dataset--$pretraining" \
    --wandb_project "tiny-geobench" \
    --auto_resume False \
    --patch_size $patch_size \
    --input_size $input_size \
    --use_orig_stem $origstem \
    --run_on_test True \
    --test_scores_dir "./test_scores_tiny/" \

cleanup
