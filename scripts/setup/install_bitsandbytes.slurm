#!/bin/bash
#SBATCH --job-name=install_bitsandbytes
#SBATCH --output=logs/install_bitsandbytes_%j.out
#SBATCH --error=logs/install_bitsandbytes_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=workq

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Load CUDA module (use 12.6 which is available on this system)
module load cuda/12.6

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate vlm3r

echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not found')"

# Install build dependencies
echo "Installing build dependencies..."
pip install packaging wheel setuptools --quiet

# Uninstall existing bitsandbytes if present
echo "Uninstalling existing bitsandbytes..."
pip uninstall bitsandbytes -y --quiet 2>/dev/null || true

# Build bitsandbytes from source
BUILD_DIR=/tmp/bitsandbytes-build-$$
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Cloning bitsandbytes repository..."
git clone https://github.com/TimDettmers/bitsandbytes.git .
cd bitsandbytes

# Set CUDA version for compilation (CUDA 12.6)
export CUDA_VERSION=126
export BNB_CUDA_VERSION=126

# Set compiler flags for aarch64
export CC=gcc
export CXX=g++
export NVCC=$CUDA_HOME/bin/nvcc

# Verify CUDA compiler
if [ ! -f "$NVCC" ]; then
    echo "ERROR: nvcc not found at $NVCC"
    exit 1
fi

echo "CUDA compiler: $NVCC"
echo "CUDA compiler version: $($NVCC --version | grep release || echo 'Unable to get version')"

# Build and install bitsandbytes
echo "Building bitsandbytes from source with CUDA 12.6 support..."
echo "This may take 15-30 minutes..."

# Install with CUDA support
# The setup.py should detect CUDA automatically, but we'll specify it explicitly
CUDA_VERSION=126 pip install . --no-build-isolation --verbose

# Verify installation
echo "Verifying bitsandbytes installation..."
if python -c "import bitsandbytes; from bitsandbytes.cextension import COMPILED_WITH_CUDA; print(f'bitsandbytes compiled with CUDA: {COMPILED_WITH_CUDA}')" 2>/dev/null; then
    echo "SUCCESS: bitsandbytes installed with CUDA support!"
    python -c "import bitsandbytes; print(f'bitsandbytes version: {bitsandbytes.__version__}')" 2>/dev/null || true
else
    echo "WARNING: bitsandbytes installation verification failed"
    echo "Attempting to import bitsandbytes..."
    python -c "import bitsandbytes; print('bitsandbytes imported successfully')" || {
        echo "ERROR: bitsandbytes import failed"
        exit 1
    }
fi

# Clean up
cd /
rm -rf $BUILD_DIR

echo "End Time: $(date)"
echo "Bitsandbytes installation completed!"
