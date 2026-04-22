#!/bin/bash
#SBATCH --job-name=verify_vsibench4
#SBATCH --output=logs/verify_vsibench4_%j.out
#SBATCH --error=logs/verify_vsibench4_%j.err
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

source ~/miniforge3/bin/activate vsibench4

echo "=== Verifying vsibench4 ==="
python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')
import transformers; print(f'transformers={transformers.__version__}')
"

echo ""
echo "=== GPU Test ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(10, device='cuda')
    t.zero_()
    print('tensor.zero_() OK')
"

echo ""
echo "=== Testing av ==="
python -c "import av; print(f'av={av.__version__}')" 2>&1

echo ""
echo "=== Testing decord ==="
python -c "import decord; print(f'decord={decord.__version__}')" 2>&1
