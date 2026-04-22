#!/bin/bash
#SBATCH --job-name=setup_test_envs
#SBATCH --output=logs/setup_test_envs_%j.out
#SBATCH --error=logs/setup_test_envs_%j.err
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
# 1. vsibenchtest environment (for evaluation)
# =============================================================================
echo ""
echo "=== Setting up vsibenchtest environment ==="

conda create -n vsibenchtest python=3.10 -y 2>&1 | tail -3

source ~/miniforge3/bin/activate vsibenchtest
pip install --upgrade pip

# PyTorch 2.1.1 with CUDA 12.1 per README
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

# lmms_eval and project dependencies
pip install -e /home/b6ar/trvbale.b6ar/scratch/VLM-memory/thinking-in-space 2>&1 | tail -5

# s2wrapper
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales 2>&1 | tail -3

# transformers per README (skip flash-attn — incompatible with GH200)
pip install transformers==4.40.0 peft==0.10.0 google-generativeai google-genai huggingface_hub[hf_xet] 2>&1 | tail -5

# scipy
pip install scipy==1.14.0 2>&1 | tail -3

# Additional eval dependencies
pip install pytablewriter scikit-learn sentencepiece tiktoken timm tqdm-multiprocess \
    transformers-stream-generator wandb yt-dlp zss zstandard \
    httpx==0.23.3 tenacity==8.3.0 protobuf==3.20 \
    opencv-python-headless openpyxl pycocoevalcap numexpr openai 2>&1 | tail -5

# Verify
python -c "import torch; print(f'vsibenchtest: torch={torch.__version__}, cuda={torch.version.cuda}, arch={torch.cuda.get_arch_list() if torch.cuda.is_available() else \"N/A\"}')"

echo "vsibenchtest setup complete"

# =============================================================================
# 2. vlm3rtest environment (for training)
# =============================================================================
echo ""
echo "=== Setting up vlm3rtest environment ==="

conda create -n vlm3rtest python=3.10 -y 2>&1 | tail -3

source ~/miniforge3/bin/activate vlm3rtest
pip install --upgrade pip

# PyTorch 2.1.1 with CUDA 12.1 per README
pip install torch==2.1.1 torchvision==0.16.1 --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

# Project dependencies (per README)
pip install -e /home/b6ar/trvbale.b6ar/scratch/VLM-memory 2>&1 | tail -5

# scipy per README
pip install scipy==1.14.0 2>&1 | tail -3

# trl for DPOTrainer
pip install trl==0.9.6 2>&1 | tail -3

# Pin transformers per README
pip install transformers==4.40.0 2>&1 | tail -3

# Verify
python -c "import torch; print(f'vlm3rtest: torch={torch.__version__}, cuda={torch.version.cuda}, arch={torch.cuda.get_arch_list() if torch.cuda.is_available() else \"N/A\"}')"

echo "vlm3rtest setup complete"

# =============================================================================
# 3. Quick GPU test
# =============================================================================
echo ""
echo "=== GPU Test ==="
source ~/miniforge3/bin/activate vsibenchtest
nvidia-smi -L
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Arch list: {torch.cuda.get_arch_list()}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(10, device='cuda')
    t.zero_()
    print('tensor.zero_() OK')
"

echo ""
echo "End Time: $(date)"
echo "Setup complete"