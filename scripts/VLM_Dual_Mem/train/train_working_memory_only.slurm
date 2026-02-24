#!/bin/bash
#SBATCH --job-name=wm_only_Lw8
#SBATCH --output=logs/train/working_memory_only_%j.out
#SBATCH --error=logs/train/working_memory_only_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

# Load CUDA (need toolkit with headers for DeepSpeed FusedAdam JIT build)
module load cuda/12.6 2>/dev/null || module load cuda/12.4.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -f "${CUDA_HOME}/include/cuda.h" ]; then
  for d in /opt/cray/pe/cuda/12.6 /opt/cray/pe/cuda/12.4 /opt/cray/pe/cuda/11.8 /opt/cuda /usr/local/cuda; do
    [ -f "${d}/include/cuda.h" ] && export CUDA_HOME="$d" && break
  done
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64" 2>/dev/null || true
if [ -f "${CUDA_HOME}/include/cuda.h" ]; then
  export CPATH="${CUDA_HOME}/include:${CPATH}"
  export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
fi
# PyTorch/DeepSpeed JIT builds need GCC 9+
module load gcc/11 2>/dev/null || module load gcc/10 2>/dev/null || module load gcc/9 2>/dev/null || true

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CUDA_HOME: ${CUDA_HOME:-<unset>}"
echo "GCC: $(gcc --version 2>/dev/null | head -1 || echo 'default')"

source ~/miniforge3/bin/activate
conda activate vlm3r

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"

# Sync fixed llava files from local workspace to lustre
LUSTRE_DIR="/lus/lfs1aip2/scratch/u5fj/trvbale.u5fj/VLM-memory"
LOCAL_WORKSPACE="/home/u5fj/trvbale.u5fj/scratch/VLM-memory"
if [ -d "$LOCAL_WORKSPACE" ]; then
    echo "Syncing llava/ files to $LUSTRE_DIR..."
    cp -f "$LOCAL_WORKSPACE/llava/model/llava_arch.py" "$LUSTRE_DIR/llava/model/llava_arch.py"
    cp -f "$LOCAL_WORKSPACE/llava/train/llava_trainer.py" "$LUSTRE_DIR/llava/train/llava_trainer.py"
    cp -f "$LOCAL_WORKSPACE/llava/train/train.py" "$LUSTRE_DIR/llava/train/train.py"
fi

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
  export MASTER_PORT=${MASTER_PORT:-29500}
  export NNODES=$SLURM_NNODES
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c 'export NODE_RANK=$SLURM_PROCID; bash scripts/VLM_Dual_Mem/train/train_working_memory_only.sh'
else
  bash scripts/VLM_Dual_Mem/train/train_working_memory_only.sh
fi

echo "End Time: $(date)"
echo "Job completed"
