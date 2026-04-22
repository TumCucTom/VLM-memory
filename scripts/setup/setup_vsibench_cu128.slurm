#!/bin/bash
#SBATCH --job-name=setup_vsibench_cu128
#SBATCH --output=logs/setup_vsibench_cu128_%j.out
#SBATCH --error=logs/setup_vsibench_cu128_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate pytorch-cu128

PROJECT_DIR="$HOME/scratch/VLM-memory"
cd "$PROJECT_DIR"

# Install using constraints file to protect torch/cuda packages from being upgraded
CONSTRAINTS="$PROJECT_DIR/constraints-cu128.txt"

echo "=== Installing core packages (protected by constraints) ==="
pip install transformers peft accelerate bitsandbytes huggingface-hub --constraints "$CONSTRAINTS"

echo "=== Installing decord (conda, has bundled ffmpeg) ==="
conda install -y -c conda-forge decord

echo "=== Installing thinking-in-space ==="
cd thinking-in-space
pip install -e . --no-deps

echo "=== Installing CUT3R dependencies (protected by constraints) ==="
cd ../CUT3R
pip install -r requirements.txt --constraints "$CONSTRAINTS"

echo "=== Installing lmms-eval dependencies (protected by constraints) ==="
pip install loguru nltk jsonlines ftfy protobuf==3.20 pybind11 sentencepiece==0.1.99 numexpr \
    pytablewriter sqlitedict==2.1.0 tenacity==8.3.0 timm pycocoevalcap evaluate datasets \
    huggingface-hub[hf_xet] transformers-stream-generator yt-dlp zss zstandard wandb \
    tqdm-multiprocess openai opencv-python-headless black==24.1.0 isort==5.13.2 av \
    hf_transfer openpyxl pre-commit pydantic tiktoken distro jiter rfc3986[idna2008] \
    joblib threadpoolctl sniffio \
    --constraints "$CONSTRAINTS"
pip install httpx==0.23.3 --constraints "$CONSTRAINTS"

echo "=== Verifying PyTorch and key packages ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0)}')

import torch.nn.functional as F
with torch.cuda.device('cuda:0'):
    t1 = torch.randn(128, 128, device='cuda:0', dtype=torch.float32)
    t2 = torch.randn(128, 128, device='cuda:0', dtype=torch.float32)
    print('torch.mm:', torch.mm(t1, t2).shape)
    inp = torch.randn(1, 3, 64, 64, device='cuda:0', dtype=torch.float32)
    w = torch.randn(16, 3, 7, 7, device='cuda:0', dtype=torch.float32)
    print('F.conv2d:', F.conv2d(inp, w, None, 2, 3, 1, 1).shape)

# Verify transformers loads (exercises most lmms-eval deps)
from transformers import AutoModel
print('transformers AutoModel: OK')

# Verify accelerate works
from accelerate import Accelerator
print('accelerate: OK')
print('All checks passed!')
"

echo "End Time: $(date)"
