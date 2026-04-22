#!/bin/bash
#SBATCH --job-name=setup_cu128
#SBATCH --output=logs/setup_cu128_%j.out
#SBATCH --error=logs/setup_cu128_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate

conda env remove -n pytorch-cu128 -y 2>/dev/null || true
conda create -y -n pytorch-cu128 python=3.10
conda activate pytorch-cu128

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch
import torch.nn.functional as F
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

with torch.cuda.device('cuda:0'):
    t1 = torch.randn(128, 128, device='cuda:0', dtype=torch.float32)
    t2 = torch.randn(128, 128, device='cuda:0', dtype=torch.float32)
    r = torch.mm(t1, t2)
    print('torch.mm OK')
    inp = torch.randn(1, 3, 64, 64, device='cuda:0', dtype=torch.float32)
    w = torch.randn(16, 3, 7, 7, device='cuda:0', dtype=torch.float32)
    r = F.conv2d(inp, w, None, 2, 3, 1, 1)
    print('F.conv2d OK')
print('All CUDA ops passed!')
"

echo "End Time: $(date)"
