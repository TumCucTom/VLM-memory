#!/bin/bash
#SBATCH --job-name=vlm3r_memory_phase1
#SBATCH --output=logs/train_memory_phase1_%j.out
#SBATCH --error=logs/train_memory_phase1_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100-sxm4-40gb:4
# For 1 GPU version, change above to:
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=100G
# #SBATCH --gres=gpu:a100-sxm4-40gb:1
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
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Activate Python virtual environment
source ~/venvs/vlm3r/bin/activate

# Verify Python and GPU
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
for i in $(seq 0 $(($SLURM_GPUS_ON_NODE - 1))); do
    echo "CUDA device $i: $(python -c 'import torch; print(torch.cuda.get_device_name('$i') if torch.cuda.is_available() else "N/A")')"
done
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Change to project directory
cd ~/VLM-memory

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

# Enable CUDA debugging to get better error messages (optional, can disable for performance)
# export CUDA_LAUNCH_BLOCKING=1

# Set NUM_GPUS_PER_NODE to match allocated GPUs
# For 4 GPUs: export NUM_GPUS_PER_NODE=4
# For 1 GPU: export NUM_GPUS_PER_NODE=1
# The training script will use this environment variable if set, otherwise defaults to 4
export NUM_GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}

# Run the Phase 1 training script
bash scripts/VLM_Dual_Mem/train_memory_phase1.sh

# Print completion time
echo "End Time: $(date)"
echo "Job completed"

