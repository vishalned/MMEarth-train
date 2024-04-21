#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=1-12  # Set the array size according to the number of jobs you want to run (3 datasets * 2 tasks = 6 jobs)
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


datasets=("geobench.m-bigearthnet" "geobench.m-so2sat" "geobench.m-eurosat")
linear_probe=True
task_type="lp" # lp for linear probe, ft for fine-tuning
pretraining="pt-imagenet_tiny"
# pretraining=pt-all_mod_uncertainty



# percents=(0.005 0.05 0.5)
samples=(100 1000 10000 50000)
dataset_idx=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
dataset=${datasets[$dataset_idx]}
num_samples=${samples[$idx]}
# percent=0.05
blr=2e-4

output_dir="./results/${task_type}-${dataset}_${pretraining}_${num_samples}"



echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "dataset: $dataset"
echo "task_type: $task_type"
echo "output_dir: $output_dir"


WANDB__SERVICE_WAIT=300 python -m  main_finetune \
            --model convnextv2_tiny \
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
            --finetune "./results/pt-imagenet_tiny/convnextv2_tiny_1k_224_fcmae.pt" \
            --output_dir "$output_dir" \
            --data_set "$dataset" \
            --linear_probe "$linear_probe" \
            --pretraining "$pretraining"\
            --wandb True \
            --wandb_run_name "subsample--$task_type--$num_samples--$dataset--$pretraining" \
            --wandb_project "tiny-geobench" \
            --auto_resume False \
            --patch_size 32 \
            --input_size 224 \
            --use_orig_stem True \
            --run_on_test True \
            --save_ckpt True \
            --num_samples $num_samples \ 
            --baseline True



#model="pt-all_mod_tiny_v100"
#pretraining=$model
#output_dir_base="./results"
#log_dir_base="./logs"
#
#datasets=("geobench.m-bigearthnet" "geobench.m-so2sat")
#linear_probe=True
#task_type="lp" # lp for linear probe, ft for fine-tuning
#
#samples=(100 1000 10000)
#dataset_idx=$(( (SLURM_ARRAY_TASK_ID - 1) / 3))
#idx=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))
#dataset=${datasets[$dataset_idx]}
#num_samples=${samples[$idx]}
## percent=0.05
#blr=2e-4
#
#
#
#output_dir="${output_dir_base}/${task_type}-${dataset}_${pretraining}_${num_samples}"
#log_dir="${log_dir_base}/${task_type}-${dataset}-${pretraining}"
#
#echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
#echo "dataset: $dataset"
#echo "task_type: $task_type"
#echo "output_dir: $output_dir"
#echo "log_dir: $log_dir"
#
#
## Run python script for linear probe or fine-tuning
#WANDB__SERVICE_WAIT=300 python -m main_finetune \
#            --model convnextv2_tiny \
#            --batch_size 32 \
#            --update_freq 1 \
#            --blr $blr \
#            --epochs 100 \
#            --warmup_epochs 0 \
#            --layer_decay_type 'single' \
#            --layer_decay 0.9 \
#            --weight_decay 0.3 \
#            --drop_path 0.1 \
#            --reprob 0.25 \
#            --mixup 0. \
#            --cutmix 0. \
#            --smoothing 0.2 \
#            --finetune "./results/${pretraining}/checkpoint-199.pth" \
#            --output_dir "$output_dir" \
#            --data_set "$dataset" \
#            --linear_probe "$linear_probe" \
#            --pretraining "$pretraining"\
#            --wandb True \
#            --wandb_run_name "subsample--$task_type--$num_samples--$dataset--$pretraining" \
#            --wandb_project "tiny-geobench" \
#            --auto_resume False \
#            --patch_size 16 \
#            --input_size 112 \
#            --use_orig_stem False \
#            --run_on_test True \
#            --save_ckpt True \
#            --num_samples $num_samples \
#
cleanup
