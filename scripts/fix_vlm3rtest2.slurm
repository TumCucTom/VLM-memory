#!/bin/bash
#SBATCH --job-name=fix_vlm3rtest2
#SBATCH --output=logs/fix_vlm3rtest2_%j.out
#SBATCH --error=logs/fix_vlm3rtest2_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

module load cuda/12.6
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH

source ~/miniforge3/bin/activate vlm3rtest2

echo "=== Adding missing packages to vlm3rtest2 ==="

# Install missing demo dependencies via pip (these don't depend on torch)
pip install matplotlib pillow opencv-python-headless einops 2>&1 | tail -5

# Install pyav via conda (no pip version for aarch64)
conda install -y pyav decord -c conda-forge 2>&1 | tail -5

echo ""
echo "=== Verifying vlm3rtest2 ==="
python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')
import matplotlib; print(f'matplotlib={matplotlib.__version__}')
import cv2; print(f'opencv={cv2.__version__}')
import llava; print('llava OK')
import transformers; print(f'transformers={transformers.__version__}')
"

echo ""
echo "=== GPU Test ==="
python -c "
import torch
t = torch.zeros(10, device='cuda')
t.zero_()
print('CUDA tensor operations OK')
"

echo "Done!"