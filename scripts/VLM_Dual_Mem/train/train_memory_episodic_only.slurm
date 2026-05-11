#!/bin/bash
#SBATCH --job-name=em_only_Le32
#SBATCH --output=logs/train/episodic_memory_only_%j.out
#SBATCH --error=logs/train/episodic_memory_only_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export BNB_CUDA_VERSION=126
export BITSANDBYTES_NOWELCOME=1

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

source ~/miniforge3/bin/activate
conda activate vlm3r-2

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"

mkdir -p logs/train
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export NCCL_TIMEOUT=86400
export TORCH_NCCL_BLOCKING_WAIT=1
export DATA_YAML="${DATA_YAML:-scripts/VLM_3R/vsibench_data_no_route_plan.yaml}"
export MASTER_PORT=${MASTER_PORT:-$((30500 + SLURM_JOB_ID % 1000))}
# 16 nodes * 4 GPUs/node * batch 1 * grad_accum 2 = effective batch 128.
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}

export MEMORY_MODE=episodic_only
export MEMORY_LABEL=episodic-only
export RUN_NAME="${RUN_NAME:-vlm2-episodic-memory-only-Le32-${SLURM_JOB_ID}}"

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
  export NNODES=$SLURM_NNODES
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c 'export NODE_RANK=$SLURM_PROCID; bash scripts/VLM_Dual_Mem/train/train_memory_mode.sh'
else
  bash scripts/VLM_Dual_Mem/train/train_memory_mode.sh
fi

echo "End Time: $(date)"
echo "Job completed"
