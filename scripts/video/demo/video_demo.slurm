#!/bin/bash
#SBATCH --job-name=vlm3r_video_demo
#SBATCH --output=logs/video_demo_memory_%j.out
#SBATCH --error=logs/video_demo_memory_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

# Load CUDA module (use 12.6 which is available on this system)
module load cuda/12.6

# Add CUDA library path for bitsandbytes (CUDA libraries are in targets/sbsa-linux/lib)
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Set CUDA paths for bitsandbytes
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Activate conda environment (which has decord installed)
source ~/miniforge3/bin/activate
conda activate vlm3r

# Change to project directory
cd /home/u5fj/trvbale.u5fj/scratch/VLM-memory

# Install missing dependencies
echo "Installing missing dependencies..."
pip install omegaconf --quiet

# Try to fix bitsandbytes CUDA support by installing from conda-forge
# conda-forge may have aarch64 builds with CUDA support
echo "Checking bitsandbytes CUDA support..."
if python -c "import bitsandbytes; from bitsandbytes.cextension import COMPILED_WITH_CUDA; exit(0 if COMPILED_WITH_CUDA else 1)" 2>/dev/null; then
    echo "bitsandbytes has CUDA support"
else
    echo "WARNING: bitsandbytes compiled without CUDA support. Installing from conda-forge..."
    pip uninstall bitsandbytes -y --quiet 2>/dev/null || true
    # Try installing from conda-forge which may have aarch64 CUDA builds
    conda install -c conda-forge bitsandbytes -y --quiet || \
    echo "NOTE: conda-forge installation failed, may need manual compilation"
fi

# Install llava package in editable mode if not already installed
if ! python -c "import llava; from llava.model import LlavaLlamaForCausalLM" 2>/dev/null; then
    echo "Installing llava package in editable mode..."
    pip install -e . --quiet
fi

# Verify Python and GPU
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Enable CUDA debugging to get better error messages
export CUDA_LAUNCH_BLOCKING=1

# Set bitsandbytes CUDA version and paths (CUDA 12.6)
export BNB_CUDA_VERSION=126
# Help bitsandbytes find CUDA libraries
export BITSANDBYTES_NOWELCOME=1
# Point to the actual CUDA library location
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

# Run the exact command from README.md (lines 153-157)
# Note: $IDX is not set in the shell script, so --chunk-idx will be -1, but we'll fix that
# by setting IDX=1 to match default behavior
export IDX=1

# Set the prompt for the video demo
export PROMPT="If I am standing by the stool and facing the stove, is the sofa to my left, right, or back?\nAn object is to my back if I would have to turn at least 135 degrees in order to face it."

bash scripts/video/demo/video_demo.sh \
    Journey9ni/vlm-3r-llava-qwen2-lora \
    qwen_1_5 64 2 average grid True \
    playground/demo/47334096.mp4 \
    lmms-lab/LLaVA-NeXT-Video-7B-Qwen2

# Print completion time
echo "End Time: $(date)"
echo "Job completed"

