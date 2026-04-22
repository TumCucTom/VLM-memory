#!/bin/bash
#SBATCH --job-name=check_pyav
#SBATCH --output=logs/check_pyav_%j.out
#SBATCH --error=logs/check_pyav_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=workq

module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

source ~/miniforge3/bin/activate vlm3rtest2

echo "=== Checking pyav ==="
python -c "import av; print(f'av imported successfully, version: {av.__version__}')" 2>&1

echo ""
echo "=== Checking decord ==="
python -c "import decord; print(f'decord imported successfully, version: {decord.__version__}')" 2>&1

echo ""
echo "=== GPU Check ==="
python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    t = torch.zeros(10, device='cuda')
    t.zero_()
    print('CUDA OK')
"