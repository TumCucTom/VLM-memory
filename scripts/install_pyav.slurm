#!/bin/bash
#SBATCH --job-name=install_pyav
#SBATCH --output=logs/install_pyav_%j.out
#SBATCH --error=logs/install_pyav_%j.err
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

source ~/miniforge3/bin/activate vlm3rtest2

echo "=== Installing pyav and decord via conda-forge ==="
conda install -y pyav decord -c conda-forge 2>&1 | tail -10

echo ""
echo "=== Verifying ==="
python -c "import av; print(f'av OK: {av.__version__}')"
python -c "import decord; print(f'decord OK: {decord.__version__}')"
