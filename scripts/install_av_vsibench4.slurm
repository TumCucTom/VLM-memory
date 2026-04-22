#!/bin/bash
#SBATCH --job-name=install_av_vsibench4
#SBATCH --output=logs/install_av_vsibench4_%j.out
#SBATCH --error=logs/install_av_vsibench4_%j.err
#SBATCH --time=00:15:00
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

echo "=== Installing av (pyav) in vsibench4 ==="
pip install av 2>&1 | tail -5

echo ""
echo "=== Verifying ==="
python -c "import av; print(f'av OK: {av.__version__}')"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')"