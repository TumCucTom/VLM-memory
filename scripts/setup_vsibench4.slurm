#!/bin/bash
#SBATCH --job-name=setup_vsibench4
#SBATCH --output=logs/setup_vsibench4_%j.out
#SBATCH --error=logs/setup_vsibench4_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

module load cuda/12.6
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi -L

source ~/miniforge3/bin/activate

# =============================================================================
# Create vsibench4 environment with conda-forge PyTorch (matching vlm3rtest2 approach)
# =============================================================================
echo ""
echo "=== Creating vsibench4 environment ==="

conda create -n vsibench4 python=3.10 -y 2>&1 | tail -3

conda activate vsibench4

# Install PyTorch from conda-forge with CUDA 12.1 (same as vlm3rtest2 which works)
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 2>&1 | tail -10

# Install flash-attention via conda-forge
conda install -y flash-attn -c conda-forge 2>&1 | tail -5

# Install remaining packages via pip (NOT torch/torchvision)
pip install --upgrade pip

# Install non-torch dependencies via pip
pip install pyyaml scipy h5py 2>&1 | tail -3
pip install transformers==4.40.0 peft==0.10.0 accelerate==0.29.1 huggingface_hub 2>&1 | tail -5
pip install pytablewriter scikit-learn sentencepiece tiktoken timm tqdm-multiprocess 2>&1 | tail -5
pip install transformers-stream-generator wandb yt-dlp zss zstandard 2>&1 | tail -5
pip install httpx==0.23.3 tenacity==8.3.0 2>&1 | tail -3
pip install opencv-python-headless openpyxl pycocoevalcap numexpr openai 2>&1 | tail -5
pip install pillow matplotlib einops 2>&1 | tail -3

# Install pyav and decord via conda-forge (no pip wheels for aarch64)
conda install -y pyav decord -c conda-forge 2>&1 | tail -5

# Install lmms_eval and project dependencies
pip install -e /home/b6ar/trvbale.b6ar/scratch/VLM-memory/thinking-in-space 2>&1 | tail -5
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales 2>&1 | tail -3

# Verify
echo ""
echo "=== Verifying vsibench4 ==="
python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')
import transformers; print(f'transformers={transformers.__version__}')
"

echo ""
echo "=== GPU Test vsibench4 ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(10, device='cuda')
    t.zero_()
    print('tensor.zero_() OK')
"

echo ""
echo "End Time: $(date)"
