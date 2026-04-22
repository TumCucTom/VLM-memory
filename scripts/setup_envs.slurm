#!/bin/bash
#SBATCH --job-name=setup_vlm3r_envs
#SBATCH --output=logs/setup_envs_%j.out
#SBATCH --error=logs/setup_envs_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate

# =============================================================================
# Environment setup following:
#   - Isambard docs (GH200/Hopper/sm_90): PyTorch 2.6+ with CUDA 12.8 (cu128) [but 2.6.0 unavailable, using 2.11.0]
#   - README.md: other package versions
# =============================================================================

# -----------------------------------------------------------------------------
# 1. vsibench environment (for evaluation)
# -----------------------------------------------------------------------------
echo ""
echo "=== Setting up vsibench environment ==="

conda create -n vsibench python=3.10 -y 2>&1 | tail -5

source ~/miniforge3/bin/activate vsibench

pip install --upgrade pip

# PyTorch 2.11.0 with cu128 (required for GH200/Hopper sm_90 per Isambard docs)
# Note: 2.6.0 was not available on cu128 index, using 2.11.0
pip install torch==2.11.0 torchvision==0.21.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5

python -c "import torch; print(f'vsibench PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, arch: {torch.cuda.get_arch_list()}')"

# lmms_eval and project dependencies
pip install -e /home/b6ar/trvbale.b6ar/scratch/VLM-memory/thinking-in-space 2>&1 | tail -5

# s2wrapper
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales 2>&1 | tail -3

# transformers and related (per README)
pip install transformers==4.40.0 peft==0.10.0 google-generativeai google-genai huggingface_hub[hf_xet] 2>&1 | tail -5

# NOTE: flash-attn wheel from README (v2.7.3 for cu12torch2.1) is NOT compatible
# with GH200 sm_90 or PyTorch 2.11+cu128. We skip it and rely on SDPA attention.
# If flash-attn is explicitly needed, it must be built from source with matching CUDA.
echo "Skipping flash-attn wheel (incompatible with GH200/PyTorch 2.11+cu128 - use SDPA attention)"

# scipy: pin to 1.14.0 to avoid numpy/ufunc issues with transformers 4.40.0
pip install scipy==1.14.0 2>&1 | tail -3

# Additional eval dependencies
pip install pytablewriter scikit-learn sentencepiece tiktoken timm tqdm-multiprocess \
    transformers-stream-generator wandb yt-dlp zss zstandard \
    httpx==0.23.3 tenacity==8.3.0 protobuf==3.20 \
    opencv-python-headless openpyxl pycocoevalcap numexpr openai 2>&1 | tail -5

echo "vsibench setup complete"

# -----------------------------------------------------------------------------
# 2. vlm3r environment (for training)
# -----------------------------------------------------------------------------
echo ""
echo "=== Setting up vlm3r environment ==="

conda create -n vlm3r python=3.10 -y 2>&1 | tail -5

source ~/miniforge3/bin/activate vlm3r

pip install --upgrade pip

# PyTorch 2.11.0 with cu128 (required for GH200/Hopper sm_90 per Isambard docs)
# Note: 2.6.0 was not available on cu128 index, using 2.11.0
pip install torch==2.11.0 torchvision==0.21.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5

python -c "import torch; print(f'vlm3r PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, arch: {torch.cuda.get_arch_list()}')"

# Project dependencies (per README)
pip install -e /home/b6ar/trvbale.b6ar/scratch/VLM-memory 2>&1 | tail -5

# NOTE: flash-attn wheel from README is NOT compatible with GH200
# We rely on SDPA attention instead
echo "Skipping flash-attn wheel (incompatible with GH200/PyTorch 2.11+cu128 - use SDPA attention)"

# scipy: pinned to 1.14.0 for numpy 1.x compatibility (per README: numpy<2)
pip install scipy==1.14.0 2>&1 | tail -3

# trl (for DPOTrainer used in llava_trainer) - pin to 0.9.6 for transformers 4.40.0 compatibility
pip install trl==0.9.6 2>&1 | tail -3

# Pin transformers to 4.40.0 per README
pip install transformers==4.40.0 2>&1 | tail -3

# verify imports
python -c "
import torch, numpy, scipy, transformers
print(f'torch={torch.__version__}, numpy={numpy.__version__}, scipy={scipy.__version__}, transformers={transformers.__version__}')
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
print('Core imports OK')
"

echo "vlm3r setup complete"

# -----------------------------------------------------------------------------
# 3. Verify both environments have working CUDA
# -----------------------------------------------------------------------------
echo ""
echo "=== Verifying CUDA on login node (expected: False) ==="
source ~/miniforge3/bin/activate vsibench
python -c "import torch; print(f'vsibench CUDA available: {torch.cuda.is_available()}')"

source ~/miniforge3/bin/activate vlm3r
python -c "import torch; print(f'vlm3r CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "End Time: $(date)"
echo "Setup complete. Both envs ready for GPU jobs."
