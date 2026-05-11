#!/bin/bash
#SBATCH --job-name=joint_vsti_qmem_lora
#SBATCH --output=logs/train/joint_vsti_query_memory_lora_%j.out
#SBATCH --error=logs/train/joint_vsti_query_memory_lora_%j.err
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
echo "CUDA_HOME: ${CUDA_HOME:-<unset>}"
echo "GCC: $(gcc --version 2>/dev/null | head -1 || echo 'default')"

source ~/miniforge3/bin/activate
conda activate vlm3r-2

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

export PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/train

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export NCCL_TIMEOUT=86400
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export VLM_NCCL_TIMEOUT_HOURS="${VLM_NCCL_TIMEOUT_HOURS:-24}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-86400}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SCRATCH_CACHE_BASE="${PROJECT_DIR}/.cache/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_CACHE_BASE}"

export DATA_YAML="${DATA_YAML:-scripts/VLM_3R/vstibench_train_data.yaml}"
export VIDEO_FOLDER="${VIDEO_FOLDER:-data/processed_data}"
export TASK_LORA_PATH="${TASK_LORA_PATH:-$PROJECT_DIR/work_dirs/vlm2-query-selection-lora-vsti-memory400-raw-128to32-4460032/checkpoint-2050}"
export CHECKPOINT_ADAPTER="${CHECKPOINT_ADAPTER:-$PROJECT_DIR/work_dirs/vlm2-full-memory-vsti-no-salience-Lw8-Le32-32f-4461527/checkpoint-1000}"
export MASTER_PORT=${MASTER_PORT:-$((31600 + SLURM_JOB_ID % 1000))}
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export FRAMES_UPBOUND="${FRAMES_UPBOUND:-128}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export QUERY_SELECTION_NUM_SELECT="${QUERY_SELECTION_NUM_SELECT:-32}"
export QUERY_SELECTION_PROJECT_SELECTED="${QUERY_SELECTION_PROJECT_SELECTED:-False}"
export QUERY_SELECTION_DISABLE_DUAL_MEMORY=False
export USE_QUERY_SELECTION=True
export USE_DUAL_MEMORY=True
export MEMORY_MODE=both
export EPISODIC_MEMORY_GATED_ATTENTION=False
export RUN_NAME="${RUN_NAME:-vlm2-joint-vsti-query-memory-lora-128to32-${SLURM_JOB_ID}}"

echo "Task LoRA warm start: ${TASK_LORA_PATH}"
echo "Memory overlay checkpoint: ${CHECKPOINT_ADAPTER}"
echo "Data YAML: ${DATA_YAML}"
echo "Output run name: ${RUN_NAME}"

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
  export NNODES=$SLURM_NNODES
  srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c 'export NODE_RANK=$SLURM_PROCID; export LOCAL_JOB_TMP="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"; export RANK_CACHE_DIR="${LOCAL_JOB_TMP}/rank_${SLURM_PROCID}"; export TMPDIR="${RANK_CACHE_DIR}/tmp"; export TMP="${TMPDIR}"; export TEMP="${TMPDIR}"; export TEMPDIR="${TMPDIR}"; export XDG_CACHE_HOME="${RANK_CACHE_DIR}/xdg_cache"; export PYTHONPYCACHEPREFIX="${RANK_CACHE_DIR}/pycache"; export TRITON_CACHE_DIR="${RANK_CACHE_DIR}/triton"; export TORCHINDUCTOR_CACHE_DIR="${RANK_CACHE_DIR}/torchinductor"; export TORCH_EXTENSIONS_DIR="${RANK_CACHE_DIR}/torch_extensions"; export TORCH_HOME="${SCRATCH_CACHE_BASE}/torch_home"; mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCH_HOME}"; bash scripts/VLM_Dual_Mem/train/train_joint_vsti_query_memory_lora.sh'
else
  export LOCAL_JOB_TMP="${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}"
  export RANK_CACHE_DIR="${LOCAL_JOB_TMP}/rank_0"
  export TMPDIR="${RANK_CACHE_DIR}/tmp"
  export TMP="${TMPDIR}"
  export TEMP="${TMPDIR}"
  export TEMPDIR="${TMPDIR}"
  export XDG_CACHE_HOME="${RANK_CACHE_DIR}/xdg_cache"
  export PYTHONPYCACHEPREFIX="${RANK_CACHE_DIR}/pycache"
  export TRITON_CACHE_DIR="${RANK_CACHE_DIR}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${RANK_CACHE_DIR}/torchinductor"
  export TORCH_EXTENSIONS_DIR="${RANK_CACHE_DIR}/torch_extensions"
  export TORCH_HOME="${SCRATCH_CACHE_BASE}/torch_home"
  mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCH_HOME}"
  bash scripts/VLM_Dual_Mem/train/train_joint_vsti_query_memory_lora.sh
fi

echo "End Time: $(date)"
echo "Job completed"
