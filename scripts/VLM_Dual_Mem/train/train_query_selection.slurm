#!/bin/bash
#SBATCH --job-name=query_sel_128to32
#SBATCH --output=logs/train/query_selection_%j.out
#SBATCH --error=logs/train/query_selection_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

# Add CUDA library path for bitsandbytes (CUDA libraries are in targets/sbsa-linux/lib)
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Set CUDA paths for bitsandbytes
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CUDA_HOME: ${CUDA_HOME:-<unset>}"
echo "GCC: $(gcc --version 2>/dev/null | head -1 || echo 'default')"

export WANDB_API_KEY

source ~/miniforge3/bin/activate
conda activate vlm3r-2

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"

# Set environment variables
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Keep async CUDA execution for full training; override to 1 when debugging a crash.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

# Set bitsandbytes CUDA version and paths (CUDA 12.6)
export BNB_CUDA_VERSION=126
# Help bitsandbytes find CUDA libraries
export BITSANDBYTES_NOWELCOME=1
# Point to the actual CUDA library location
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

mkdir -p logs

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export NCCL_TIMEOUT=86400
export TORCH_NCCL_BLOCKING_WAIT=1

# Use filtered vsibench (scannet + scannetpp + route_plan); only samples with existing videos
export DATA_YAML="scripts/VLM_3R/vsibench_data_no_route_plan.yaml"

# Multinode: first node is master; srun runs one task per node so each gets correct NODE_RANK
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
  export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
  export NNODES=$SLURM_NNODES
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c 'export NODE_RANK=$SLURM_PROCID; bash scripts/VLM_Dual_Mem/train/train_query_selection.sh'
else
  bash scripts/VLM_Dual_Mem/train/train_query_selection.sh
fi

echo "End Time: $(date)"
echo "Job completed"
