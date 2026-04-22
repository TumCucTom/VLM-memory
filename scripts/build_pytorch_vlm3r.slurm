#!/bin/bash
#SBATCH --job-name=build_pytorch_vlm3r
#SBATCH --output=logs/build_pytorch_vlm3r_%j.out
#SBATCH --error=logs/build_pytorch_vlm3r_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
nvidia-smi -L

source ~/miniforge3/bin/activate

CONDA_ENV=vlm3r
conda activate $CONDA_ENV

# =============================================================================
# Step 1: Uninstall old PyTorch
# =============================================================================
echo ""
echo "=== Uninstalling old PyTorch ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip uninstall -y cuda-bindings cuda-toolkit cuda-pathfinder 2>/dev/null || true
echo "Old PyTorch removed"

# =============================================================================
# Step 2: Create stub CUDA headers (HPC SDK missing cuBLAS/cuSPARSE/cuSOLVER headers)
# =============================================================================
echo ""
echo "=== Creating stub CUDA headers ==="
STUB_DIR=$HOME/miniforge3/envs/$CONDA_ENV/include/nvidia/cu12
mkdir -p $STUB_DIR

# stub cublas_v2.h
cat > $STUB_DIR/cublas_v2.h << 'STUBEOF'
#pragma once
#include <stdint.h>
typedef struct cublasContext* cublasHandle_t;
typedef enum cublasFillMode_t { CUBLAS_FILL_MODE_LOWER=0, CUBLAS_FILL_MODE_UPPER=1 } cublasFillMode_t;
typedef enum cublasOperation_t { CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2 } cublasOperation_t;
typedef enum cublasDiagType_t { CUBLAS_DIAG_NON_UNIT=0, CUBLAS_DIAG_UNIT=1 } cublasDiagType_t;
typedef enum cublasSideMode_t { CUBLAS_SIDE_LEFT=0, CUBLAS_SIDE_RIGHT=1 } cublasSideMode_t;
typedef enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST=0, CUBLAS_POINTER_MODE_DEVICE=1 } cublasPointerMode_t;
typedef enum cublasStatus_t { CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_NOT_SUPPORTED=0x7 } cublasStatus_t;
STUBEOF

# stub cusparse.h
cat > $STUB_DIR/cusparse.h << 'STUBEOF'
#pragma once
#include <stdint.h>
typedef struct cusparseContext* cusparseHandle_t;
typedef enum { CUSPARSE_STATUS_SUCCESS=0 } cusparseStatus_t;
typedef enum { CUSPARSE_OPERATION_NON_TRANSPOSE=0, CUSPARSE_OPERATION_TRANSPOSE=1, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE=2 } cusparseOperation_t;
typedef enum { CUSPARSE_INDEX_BASE_ZERO=0, CUSPARSE_INDEX_BASE_ONE=1 } cusparseIndexBase_t;
typedef enum { CUSPARSE_MATRIX_TYPE_GENERAL=0 } cusparseMatrixType_t;
const char* cusparseGetErrorString(cusparseStatus_t status);
STUBEOF

# stub cusolverDn.h
cat > $STUB_DIR/cusolverDn.h << 'STUBEOF'
#pragma once
#include <stdint.h>
typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef enum { CUSOLVER_STATUS_SUCCESS=0 } cusolverStatus_t;
STUBEOF

# stub cusolver_common.h (must define cusolverStatus_t)
cat > $STUB_DIR/cusolver_common.h << 'STUBEOF'
#pragma once
#include <stdint.h>
typedef enum { CUSOLVER_STATUS_SUCCESS=0 } cusolverStatus_t;
const char* cusolverGetErrorMessage(cusolverStatus_t status);
STUBEOF

# stub cublasLt.h
cat > $STUB_DIR/cublasLt.h << 'STUBEOF'
#pragma once
#include "cublas_v2.h"
typedef void* cublasLtHandle_t;
STUBEOF

echo "Stub headers created at $STUB_DIR"

# =============================================================================
# Step 3: Set up build environment
# =============================================================================
echo ""
echo "=== Setting up build environment ==="

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compiler: GCC 13 for ARM
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export LDFLAGS="-Wl,-rpath,/home/b6ar/trvbale.b6ar/miniforge3/envs/$CONDA_ENV/lib"

# ARM + CUDA linker optimization
export USE_PRIORITIZED_TEXT_FOR_LD=1

# Add stub headers to include path (HPC SDK missing cuBLAS/cuSPARSE/cuSOLVER headers)
export CPATH=$STUB_DIR:$CPATH
export CPLUS_INCLUDE_PATH=$STUB_DIR:$CPLUS_INCLUDE_PATH

# Build flags
export MAX_JOBS=8
export OMP_NUM_THREADS=8

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $($CUDA_HOME/bin/nvcc --version | grep release)"
echo "GCC: $(/usr/bin/gcc-13 --version | head -1)"

# =============================================================================
# Step 4: Clone PyTorch 2.5.1 with submodules
# =============================================================================
echo ""
echo "=== Cloning PyTorch 2.5.1 ==="
PYTORCH_BUILD_DIR=$HOME/pytorch_build/$CONDA_ENV
mkdir -p $PYTORCH_BUILD_DIR
cd $PYTORCH_BUILD_DIR
rm -rf pytorch
git clone --depth=1 --branch v2.5.1 --recurse-submodules https://github.com/pytorch/pytorch.git 2>&1 | tail -5
cd pytorch
echo "Build directory: $(pwd)"

# =============================================================================
# Step 5: Install build dependencies
# =============================================================================
echo ""
echo "=== Installing build dependencies ==="
pip install "cmake==3.28.1" ninja pybind11 2>&1 | tail -3
pip install -r requirements.txt 2>&1 | tail -5

# =============================================================================
# Step 6: Build PyTorch from source
# =============================================================================
echo ""
echo "=== Building PyTorch 2.5.1 (this will take ~2-3 hours) ==="

BUILD_TYPE=Release \
TORCH_CUDA_ARCH_LIST="9.0" \
USE_CUDNN=0 \
USE_NCCL=0 \
USE_CUDA=1 \
USE_CUBLAS=0 \
USE_CUSPARSE=0 \
USE_CUSOLVER=0 \
USE_DISTRIBUTED=0 \
USE_MKLDNN=0 \
USE_OPENCV=0 \
USE_QNNPACK=0 \
USE_PYTORCH_QNNPACK=0 \
USE_XNNPACK=0 \
USE_NNPACK=0 \
USE_FLASH_ATTN=0 \
USE_MEM_EFF_ATTENTION=0 \
BUILD_TEST=0 \
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
python setup.py install 2>&1 | tee $HOME/pytorch_build/$CONDA_ENV/build.log

BUILD_STATUS=${PIPESTATUS[0]}

# Show last 100 lines of build log for diagnostics
echo ""
echo "=== Last 100 lines of build log ==="
tail -100 $HOME/pytorch_build/$CONDA_ENV/build.log

echo ""
echo "Build exit status: $BUILD_STATUS"

if [ $BUILD_STATUS -ne 0 ]; then
    echo "ERROR: PyTorch build failed!"
    exit $BUILD_STATUS
fi

# =============================================================================
# Step 7: Verify installation
# =============================================================================
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Arch list: {torch.cuda.get_arch_list()}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Device major.minor: {props.major}.{props.minor}')
    print(f'Total memory: {props.total_memory / 1024**3:.2f} GB')
"

# Test basic CUDA operations
python -c "
import torch
t = torch.zeros(10, device='cuda')
t.zero_()
print('tensor.zero_() OK')
t.fill_(1.0)
print('tensor.fill_() OK')
print('All basic CUDA ops passed!')
"

echo ""
echo "End Time: $(date)"
echo "Build complete for vlm3r"