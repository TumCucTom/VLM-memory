#!/bin/bash
#SBATCH --job-name=test_base_model
#SBATCH --output=logs/test_base_%j.out
#SBATCH --error=logs/test_base_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export BNB_CUDA_VERSION=126
export BITSANDBYTES_NOWELCOME=1

source ~/miniforge3/bin/activate vlm3r-2

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

cd /home/b6ar/trvbale.b6ar/scratch/VLM-memory

mkdir -p logs

export IDX=1
export PROMPT="If I am standing by the stool and facing the stove, is the sofa to my left, right, or back? An object is to my back if I would have to turn at least 135 degrees in order to face it."

# Test WITHOUT LoRA - use base model for both
python3 playground/demo/video_demo.py \
    --model-path lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
    --model-base lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
    --video_path playground/demo/47334096.mp4 \
    --output_dir ./work_dirs/video_demo/test_base_model \
    --output_name pred \
    --chunk-idx $(($IDX - 1)) \
    --overwrite True \
    --mm_spatial_pool_stride 2 \
    --for_get_frames_num 64 \
    --conv-mode qwen_1_5 \
    --mm_spatial_pool_mode average \
    --mm_newline_position grid \
    --use_av True \
    --prompt "${PROMPT}"

echo "End Time: $(date)"
echo "Done."