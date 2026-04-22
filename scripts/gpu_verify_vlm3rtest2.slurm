#!/bin/bash
#SBATCH --job-name=gpu_verify_vlm3rtest2
#SBATCH --output=logs/gpu_verify_vlm3rtest2_%j.out
#SBATCH --error=logs/gpu_verify_vlm3rtest2_%j.err
#SBATCH --time=00:10:00
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

echo "=== vlm3rtest2 GPU Test ==="
python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(10, device='cuda')
    t.zero_()
    print('tensor.zero_() OK')
"

echo ""
echo "=== vlm3rtest2 Demo Packages ==="
python -c "
import matplotlib; print(f'matplotlib={matplotlib.__version__}')
import cv2; print(f'opencv={cv2.__version__}')
import llava; print('llava imported OK')
"