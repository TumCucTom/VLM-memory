#!/bin/bash
#SBATCH --job-name=gpu_test_pytorch21
#SBATCH --output=logs/gpu_test_pytorch21_%j.out
#SBATCH --error=logs/gpu_test_pytorch21_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
nvidia-smi -L

source ~/miniforge3/bin/activate

# Create temp env with PyTorch 2.1.1+cu121
conda create -n pytorch21test python=3.10 -y 2>&1 | tail -2
conda activate pytorch21test
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

echo ""
echo "=== PyTorch Info ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Arch list: {torch.cuda.get_arch_list()}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Device compute: {props.major}.{props.minor}')
"

echo ""
echo "=== CUDA ops test ==="
python -c "
import torch
t = torch.zeros(10, device='cuda')
t.zero_()
print('zero_() OK')
t.fill_(1.0)
print('fill_() OK')
t = torch.randn(100, 100, device='cuda')
r = torch.matmul(t, t)
print(f'matmul OK, result shape: {r.shape}')
print('All CUDA ops passed!')
"

echo ""
echo "End Time: $(date)"

conda deactivate
conda env -n pytorch21test --remove -y 2>&1 | tail -2