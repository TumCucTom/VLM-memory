#!/bin/bash
#SBATCH --job-name=eval_vsi_ckpt
#SBATCH --output=logs/eval/vsibench_checkpoint_%j.out
#SBATCH --error=logs/eval/vsibench_checkpoint_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

# Load CUDA
module load cuda/12.6 2>/dev/null || module load cuda/11.8 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -f "${CUDA_HOME}/include/cuda.h" ]; then
  for d in /opt/cray/pe/cuda/12.6 /opt/cray/pe/cuda/11.8 /opt/cuda /usr/local/cuda; do
    [ -f "${d}/include/cuda.h" ] && export CUDA_HOME="$d" && break
  done
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64" 2>/dev/null || true
[ -f "${CUDA_HOME}/include/cuda.h" ] && export CPATH="${CUDA_HOME}/include:${CPATH}" && export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH}" && export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"

echo "Job ID: $SLURM_JOB_ID" && echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate vlm3r

export PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

# Checkpoint to eval (set EVAL_CHECKPOINT before sbatch to override). Default: working-memory training run (use run dir so non_lora_trainables.bin is found; or use run_dir/checkpoint-N). Eval uses adapter-overlay (base+LoRA + overlay). Set EVAL_USE_DUAL_MEMORY=False for no-memory checkpoints.
# For untrained-memory baseline (keep base ~60%%): set EVAL_OVERLAY_MEMORY_ONLY=True so only working/episodic memory weights are overlaid; otherwise full checkpoint overlay can overwrite fusion_block and drop to ~1%%.
export EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-work_dirs/vlm2-working-memory-only-Lw8}"
export EVAL_USE_DUAL_MEMORY="${EVAL_USE_DUAL_MEMORY:-True}"
export EVAL_OVERLAY_MEMORY_ONLY="${EVAL_OVERLAY_MEMORY_ONLY:-False}"
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export NUM_GPUS=$NUM_GPUS_PER_NODE
export NNODES=1

cd thinking-in-space
mkdir -p logs

bash eval_vlm_3r_vsibench_checkpoint.sh

echo "End Time: $(date)" && echo "Job completed"
