#!/bin/bash
#SBATCH --job-name=eval_vstibench_both_ws8_es16
#SBATCH --output=logs/eval/vstibench_both_ws8_es16_%j.out
#SBATCH --error=logs/eval/vstibench_both_ws8_es16_%j.err
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
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${SLURM_JOB_GPUS:-1}

bash eval_vstibench_both_ws8_es16.sh
