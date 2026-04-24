#!/bin/bash
#SBATCH --job-name=eval_vstibench
#SBATCH --output=logs/eval/vstibench_%j.out
#SBATCH --error=logs/eval/vstibench_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

# Multi-node: get head node IP for distributed launch
# scontrol is instant (controller query), vs srun which spawns across all nodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
# Verify MASTER_ADDR resolves on all nodes - check within 10s timeout
if ! timeout 10 bash -c "echo > /dev/tcp/$MASTER_ADDR/29500" 2>/dev/null; then
  # Fallback: try resolving to IP if hostname doesn't work across nodes
  MASTER_IP=$(getent hosts "$MASTER_ADDR" | awk '{print $1}')
  [ -n "$MASTER_IP" ] && export MASTER_ADDR="$MASTER_IP"
fi
export NNODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=0
export MASTER_PORT=29500
export NUM_GPUS=4

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ~/miniforge3/bin/activate
conda activate pytorch-cu128

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
python -c "import torch; print(f'{torch.__version__}, CUDA:{torch.version.cuda}, Available:{torch.cuda.is_available()}, Device:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

cd thinking-in-space
mkdir -p logs

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export GPUS_PER_NODE=4
export NUM_GPUS=4

bash eval_vlm_3r_vstibench.sh

echo "End Time: $(date)"
echo "Job completed"
