#!/bin/bash
#SBATCH --job-name=eval_vsti_joint_qmem
#SBATCH --output=logs/eval/vstibench_joint_qmem_%j.out
#SBATCH --error=logs/eval/vstibench_joint_qmem_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ~/miniforge3/bin/activate
conda activate pytorch-cu128

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

export PYTHONWARNINGS=ignore
export LLAVA_DEBUG_GRAD=0
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export JOINT_QMEM_CKPT="${JOINT_QMEM_CKPT:-$PROJECT_DIR/work_dirs/vlm2-joint-vsti-query-memory-lora-128to32-4467965}"
export MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
export MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-128}"
export QUERY_SELECTION_NUM_SELECT="${QUERY_SELECTION_NUM_SELECT:-32}"
export QUERY_SELECTION_TEMPERATURE="${QUERY_SELECTION_TEMPERATURE:-1.0}"
export QUERY_SELECTION_USE_GUMBEL="${QUERY_SELECTION_USE_GUMBEL:-False}"
export QUERY_SELECTION_PROJECT_SELECTED="${QUERY_SELECTION_PROJECT_SELECTED:-False}"
export LOG_SUFFIX="${LOG_SUFFIX:-joint_qmem_vsti_4467965_${SLURM_JOB_ID}}"

cd thinking-in-space
mkdir -p logs

bash eval_vlm_3r_vstibench_joint_qmem.sh

echo "End Time: $(date)"
echo "Job completed"
