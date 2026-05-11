#!/bin/bash
#SBATCH --job-name=mem_lora_vsti_Lw4Le32
#SBATCH --output=logs/train/memory_lora_vsti_Lw4Le32_%j.out
#SBATCH --error=logs/train/memory_lora_vsti_Lw4Le32_%j.err
#SBATCH --time=24:00:00
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

export PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"

mkdir -p logs/train
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export NCCL_TIMEOUT=86400
export TORCH_NCCL_BLOCKING_WAIT=1
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export JOB_CACHE_BASE="${PROJECT_DIR}/.cache/${SLURM_JOB_ID}"
mkdir -p "${JOB_CACHE_BASE}"

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
  export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
  export NNODES=$SLURM_NNODES
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c 'export NODE_RANK=$SLURM_PROCID; export RANK_CACHE_DIR="${JOB_CACHE_BASE}/rank_${SLURM_PROCID}"; export TMPDIR="${RANK_CACHE_DIR}/tmp"; export TRITON_CACHE_DIR="${RANK_CACHE_DIR}/triton"; export TORCHINDUCTOR_CACHE_DIR="${RANK_CACHE_DIR}/torchinductor"; export TORCH_EXTENSIONS_DIR="${RANK_CACHE_DIR}/torch_extensions"; mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"; bash scripts/VLM_Dual_Mem/train/train_memory_lora_vsti_Lw4_Le32_continue.sh'
else
  export RANK_CACHE_DIR="${JOB_CACHE_BASE}/rank_0"
  export TMPDIR="${RANK_CACHE_DIR}/tmp"
  export TRITON_CACHE_DIR="${RANK_CACHE_DIR}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${RANK_CACHE_DIR}/torchinductor"
  export TORCH_EXTENSIONS_DIR="${RANK_CACHE_DIR}/torch_extensions"
  mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
  bash scripts/VLM_Dual_Mem/train/train_memory_lora_vsti_Lw4_Le32_continue.sh
fi

echo "End Time: $(date)"
echo "Job completed"
