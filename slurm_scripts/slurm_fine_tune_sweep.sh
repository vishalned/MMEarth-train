#!/bin/bash
#SBATCH --job-name=geobench-sweep
#SBATCH --array=1-8  # Set the array size according to the number of jobs you want to run (3 datasets * 2 tasks = 6 jobs)
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



datasets="geobench.m-bigearthnet"
linear_probe=False # always false for segmentation
task_type=unet-ft-blr_sweep # for segmentation tasks this is just a random name
model="pt-all_mod_tiny_v100"
patch_size=16
input_size=112
origstem=False
pretraining=$model
output_dir_base="./results"
log_dir_base="./logs"

idx=$((SLURM_ARRAY_TASK_ID - 1))
output_dir="${output_dir_base}/${task_type}-${datasets}-${pretraining}"
dataset="${datasets}"

blrs=(1e-2 5e-2 1e-3 5e-3 1e-4 5e-4 1e-5 5e-5)
blr=(${blrs[$idx]})


echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "dataset: $dataset"
echo "task_type: $task_type"
echo "output_dir: $output_dir"
echo "log_dir: $log_dir"


WANDB__SERVICE_WAIT=300 python -m main_finetune \
            --model convnextv2_tiny \
            --batch_size 64 \
            --update_freq 16 \
            --blr ${blr} \
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
            --finetune ./results/pt-all_mod_tiny_v100/checkpoint-199.pth \
            --output_dir "$output_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe"\
            --auto_resume False \
            --pretraining $pretraining \
            --wandb True \
            --wandb_run_name "sweep-$task_type--$dataset--$pretraining--$blr" \
            --patch_size $patch_size \
            --input_size $input_size \
            --use_orig_stem $origstem \
            --run_on_test True \
            --test_scores_dir ./test_scores_sweep

cleanup



