#!/bin/bash
#SBATCH --job-name=vlm3r_video_demo
#SBATCH --output=logs/video_demo_memory_%j.out
#SBATCH --error=logs/video_demo_memory_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

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
conda activate vlm3r-2

# Change to project directory
cd /home/b6ar/trvbale.b6ar/scratch/VLM-memory


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

# Set the prompt for the video demo
export PROMPT="If I am standing by the stool and facing the stove, is the sofa to my left, right, or back?"
export IDX=1

python3 playground/demo/video_demo.py \
    --model-path Journey9ni/vlm-3r-llava-qwen2-lora \
    --model-base lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
    --video_path playground/demo/47334096.mp4 \
    --output_dir ./work_dirs/video_demo/vlm-3r-llava-qwen2-lora_qwen_1_5_frames_64_stride_2 \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite True \
    --mm_spatial_pool_stride 2 \
    --for_get_frames_num 16 \
    --conv-mode qwen_1_5 \
    --mm_spatial_pool_mode average \
    --mm_newline_position grid \
    --use_av True \
    --prompt "${PROMPT}. An object is to my back if I would have to turn at least 135 degrees in order to face it."

# Print completion time
echo "End Time: $(date)"
echo "Job completed"

