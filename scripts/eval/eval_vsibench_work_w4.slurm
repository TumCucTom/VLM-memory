#!/bin/bash
#SBATCH --job-name=eval_vsibench_work_w4
#SBATCH --output=logs/eval/vsibench_work_w4_%j.out
#SBATCH --error=logs/eval/vsibench_work_w4_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

source ~/miniforge3/bin/activate
conda activate pytorch-cu128

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

cd thinking-in-space
mkdir -p logs

export PYTHONWARNINGS=ignore
export LLAVA_DEBUG_GRAD=1
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${SLURM_JOB_GPUS:-1}

bash eval_vsibench_work_w4.sh
