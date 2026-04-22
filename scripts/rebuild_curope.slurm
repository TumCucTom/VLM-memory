#!/bin/bash
#SBATCH --job-name=rebuild_curope
#SBATCH --output=logs/rebuild_curope_%j.out
#SBATCH --error=logs/rebuild_curope_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

source ~/miniforge3/bin/activate vlm3r-2

echo "=== Current curope build ==="
ls -la CUT3R/src/croco/models/curope/build/lib*/curope*.so 2>/dev/null || echo "No existing .so found"

echo "=== Rebuilding curope for sm_80 + sm_90 ==="
cd CUT3R/src/croco/models/curope/
rm -rf build *.so
python setup.py build_ext --inplace 2>&1

echo "=== Checking compiled kernels ==="
cd ..
find . -name "*.ptx" -o -name "*.cubin" 2>/dev/null | head -5
ls -la build/temp.linux-aarch64-cpython-310/*.o 2>/dev/null | head -5

echo "Done."
echo "End Time: $(date)"