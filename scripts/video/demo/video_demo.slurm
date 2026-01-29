#!/bin/bash
#SBATCH --job-name=vlm3r_video_demo
#SBATCH --output=logs/video_demo_%j.out
#SBATCH --error=logs/video_demo_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985

# Load CUDA module for bitsandbytes compatibility
module load cuda/12.4.1

# Set LD_LIBRARY_PATH to include CUDA libraries
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/software/spack/linux-rocky8-broadwell/gcc-12.3.0/cuda-12.4.1-ebv6/lib64:$LD_LIBRARY_PATH

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Activate Python virtual environment
source ~/venvs/vlm3r/bin/activate

# Verify Python and GPU
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd ~/VLM-memory

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Enable CUDA debugging to get better error messages
export CUDA_LAUNCH_BLOCKING=1

# Run the exact command from README.md (lines 153-157)
# Note: $IDX is not set in the shell script, so --chunk-idx will be -1, but we'll fix that
# by setting IDX=1 to match default behavior
export IDX=1

bash scripts/video/demo/video_demo.sh \
    Journey9ni/vlm-3r-llava-qwen2-lora \
    qwen_1_5 32 2 average grid True \
    playground/demo/47334096.mp4 \
    lmms-lab/LLaVA-NeXT-Video-7B-Qwen2

# Print completion time
echo "End Time: $(date)"
echo "Job completed"

