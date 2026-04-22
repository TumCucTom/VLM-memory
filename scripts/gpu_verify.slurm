#!/bin/bash
#SBATCH --job-name=gpu_verify
#SBATCH --output=logs/gpu_verify_%j.out
#SBATCH --error=logs/gpu_verify_%j.err
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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "CUDA_HOME: $CUDA_HOME"

source ~/miniforge3/bin/activate vlm3rtest2

echo ""
echo "=== Environment Info ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "torch._C._cuda_getArchFlags(): $(python -c 'import torch; print(torch._C._cuda_getArchFlags() if hasattr(torch._C, "_cuda_getArchFlags") else "N/A")')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Device count: $(python -c 'import torch; print(torch.cuda.device_count())')"

if python -c 'import torch; torch.cuda.is_available()'; then
    echo ""
    echo "=== GPU Test ==="
    python -c "
import torch
print(f'Device: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'Compute capability: {props.major}.{props.minor}')
print(f'Total memory: {props.total_memory / 1024**3:.2f} GB')

# Test basic CUDA operations
t = torch.zeros(100, 100, device='cuda')
t.fill_(1.0)
result = torch.matmul(t, t)
print(f'Matmul result shape: {result.shape}')
print('All CUDA operations passed!')
"
fi

echo ""
echo "End Time: $(date)"