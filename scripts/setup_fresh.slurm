#!/bin/bash
#SBATCH --job-name=setup_fresh
#SBATCH --output=logs/setup_fresh_%j.out
#SBATCH --error=logs/setup_fresh_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Load CUDA module
module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi -L

source ~/miniforge3/bin/activate

# =============================================================================
# 1. Remove old environments
# =============================================================================
echo ""
echo "=== Removing old environments ==="
conda env remove -n vlm3rtest2 --yes 2>/dev/null || true
conda env remove -n vsibenchtest2 --yes 2>/dev/null || true
echo "Old envs removed"

# =============================================================================
# 2. Create vlm3rtest2 environment (for training)
# =============================================================================
echo ""
echo "=== Setting up vlm3rtest2 environment ==="

conda create -n vlm3rtest2 python=3.10 -y 2>&1 | tail -3

source ~/miniforge3/bin/activate vlm3rtest2
pip install --upgrade pip

# Install PyTorch from conda-forge (has aarch64 support)
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 2>&1 | tail -10

echo "PyTorch installed via conda"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, arch={torch._C._cuda_getArchFlags() if hasattr(torch._C, \"_cuda_getArchFlags\") else \"N/A\"}')"

# Install flash-attention via conda-forge
conda install -y flash-attn -c conda-forge 2>&1 | tail -10

# Project dependencies
pip install -e /home/b6ar/trvbale.b6ar/scratch/VLM-memory 2>&1 | tail -5

# scipy
pip install scipy 2>&1 | tail -3

# trl for DPOTrainer
pip install trl 2>&1 | tail -3

# Verify
echo ""
echo "=== Verifying vlm3rtest2 ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')"
python -c "import flash_attn; print('flash_attn imported')" 2>&1 || echo "flash_attn not importable"

echo "vlm3rtest2 setup complete"

# =============================================================================
# 3. GPU Test
# =============================================================================
echo ""
echo "=== GPU Test ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Arch list: {torch.cuda.get_arch_list() if torch.cuda.is_available() else \"N/A\"}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(10, device='cuda')
    t.zero_()
    print('tensor.zero_() OK')
    t.fill_(1.0)
    print('tensor.fill_() OK')
    print('All CUDA ops passed!')
"

echo ""
echo "End Time: $(date)"
echo "Setup complete"