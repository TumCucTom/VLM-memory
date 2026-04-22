#!/bin/bash
#SBATCH --job-name=setup_vsibench_gpu
#SBATCH --output=logs/setup_vsibench_gpu_%j.out
#SBATCH --error=logs/setup_vsibench_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Load CUDA
module load cuda/12.6

# Set CUDA paths so pip wheel gets detected properly
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH

# Create conda env
cd /home/b6ar/trvbale.b6ar/scratch/VLM-memory
conda create --name vsibench-1 python=3.10 -y 2>&1 | tail -5
conda activate vsibench-1

echo "=== Installing PyTorch 2.1.1 CUDA 12.1 via pip ==="
pip install --upgrade pip
pip install torch==2.1.1 torchvision==0.16.1 --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -10

echo "=== Verifying PyTorch CUDA ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

echo "=== Installing thinking-in-space ==="
cd thinking-in-space
pip install -e . --no-deps 2>&1 | tail -5

echo "=== Installing s2wrapper ==="
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales 2>&1 | tail -5

echo "=== Installing flash-attn (pip wheel) ==="
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 2>&1 | tail -5

echo "=== Installing transformers and peft ==="
pip install transformers==4.40.0 peft==0.10.0 2>&1 | tail -5

echo "=== Installing google packages ==="
pip install google-generativeai google-genai 2>&1 | tail -5

echo "=== Installing huggingface hub with hf_xet ==="
pip install "huggingface_hub[hf_xet]" 2>&1 | tail -5

echo "=== Final verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

echo "End Time: $(date)"