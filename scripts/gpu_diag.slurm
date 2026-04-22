#!/bin/bash
#SBATCH --job-name=gpu_diag
#SBATCH --output=logs/gpu_diag_%j.out
#SBATCH --error=logs/gpu_diag_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "=== nvidia-smi ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv
echo ""
echo "=== CUDA versions ==="
nvidia-smi | grep "CUDA Version"
echo ""
echo "=== PyTorch info ==="
source ~/miniforge3/bin/activate vsibench
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'Available GPUs: {torch.cuda.device_count()}')
print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
if props:
    print(f'Device major: {props.major}, minor: {props.minor}')
    print(f'Device total memory: {props.total_memory / 1024**3:.2f} GB')
print(f'Arch list: {torch.cuda.get_arch_list()}')
"
echo ""
echo "=== Test basic CUDA ops ==="
source ~/miniforge3/bin/activate vsibench
python -c "
import torch
t = torch.zeros(10, device='cuda')
t.zero_()
print('tensor.zero_() OK')
t.fill_(1.0)
print('tensor.fill_() OK')
print('All basic ops passed!')
"
