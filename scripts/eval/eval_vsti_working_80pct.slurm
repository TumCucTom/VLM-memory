#!/bin/bash
#SBATCH --job-name=eval_vsti_working_80pct
#SBATCH --output=logs/eval/vsti_working_80pct_%j.out
#SBATCH --error=logs/eval/vsti_working_80pct_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

module load cuda/12.6 2>/dev/null || module load cuda/11.8 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -f "${CUDA_HOME}/include/cuda.h" ]; then
  for d in /opt/cray/pe/cuda/12.6 /opt/cray/pe/cuda/11.8 /opt/cuda /usr/local/cuda; do
    [ -f "${d}/include/cuda.h" ] && export CUDA_HOME="$d" && break
  done
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64" 2>/dev/null || true
if [ -f "${CUDA_HOME}/include/cuda.h" ]; then
  export CPATH="${CUDA_HOME}/include:${CPATH}"
  export C_INCLUDE_PATH="${CUDA_HOME}/include:${C_INCLUDE_PATH}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
fi

source ~/miniforge3/bin/activate
conda activate vsibench

cd /home/u5hv/trvbale.u5hv/scratch/VLM-memory
mkdir -p logs/eval
cd thinking-in-space

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

bash eval_vsti_working_80pct.sh
