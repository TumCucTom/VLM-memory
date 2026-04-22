#!/bin/bash
#SBATCH --job-name=vlm3r_video_demo
#SBATCH --output=logs/video_demo_memory_%j.out
#SBATCH --error=logs/video_demo_memory_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

# Add CUDA library path for bitsandbytes
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

source ~/miniforge3/bin/activate
conda activate vlm3r-2

cd /home/b6ar/trvbale.b6ar/scratch/VLM-memory

mkdir -p logs

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export BNB_CUDA_VERSION=126
export BITSANDBYTES_NOWELCOME=1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Install llava package in editable mode if not already installed
if ! python -c "import llava; from llava.model import LlavaLlamaForCausalLM" 2>/dev/null; then
    echo "Installing llava package in editable mode..."
    pip install -e . --quiet
fi

# Run the exact command from authors' script
bash scripts/video/demo/video_demo.sh \
    Journey9ni/vlm-3r-llava-qwen2-lora \
    qwen_1_5 64 2 average grid True \
    playground/demo/47334096.mp4 \
    lmms-lab/LLaVA-NeXT-Video-7B-Qwen2

echo "End Time: $(date)"
echo "Done."