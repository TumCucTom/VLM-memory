#!/bin/bash
#SBATCH --job-name=fix_pytorch_cuda
#SBATCH --output=logs/fix_pytorch_cuda_%j.out
#SBATCH --error=logs/fix_pytorch_cuda_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "GPU check:"
nvidia-smi -L

source ~/miniforge3/bin/activate

# Fix vlm3r environment
echo "=== Fixing vlm3r environment ==="
conda activate vlm3r
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 'numpy<2' 2>&1 | tail -10
python -c "import torch; print(f'vlm3r PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}')"

# Fix vsibench environment
echo "=== Fixing vsibench environment ==="
conda activate vsibench
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 'numpy<2' 2>&1 | tail -10
python -c "import torch; print(f'vsibench PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, available: {torch.cuda.is_available()}')"

echo "End Time: $(date)"