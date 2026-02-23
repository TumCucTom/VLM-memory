#!/bin/bash
#SBATCH --job-name=eval_vsibench
#SBATCH --output=logs/eval/vsibench_%j.out
#SBATCH --error=logs/eval/vsibench_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

# Load CUDA (cuda/12.6 and cuda/11.8 available; need toolkit with headers for Triton JIT)
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
echo "CUDA_HOME: ${CUDA_HOME:-<unset>}"

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ~/miniforge3/bin/activate
conda activate vlm3r

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Numpy, Scipy versions: $(python -c "import numpy; import scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)")"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

cd thinking-in-space
mkdir -p logs

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

bash eval_vlm_3r_vsibench.sh

echo "End Time: $(date)"
echo "Job completed"
