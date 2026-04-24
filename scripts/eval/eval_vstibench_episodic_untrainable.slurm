#!/bin/bash
#SBATCH --job-name=eval_vstibench_episodic_untrainable
#SBATCH --output=logs/eval/vstibench_episodic_untrainable_%j.out
#SBATCH --error=logs/eval/vstibench_episodic_untrainable_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ~/miniforge3/bin/activate
conda activate pytorch-cu128

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
python -c "import torch; print(f'{torch.__version__}, CUDA:{torch.version.cuda}, Available:{torch.cuda.is_available()}, Device:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

cd thinking-in-space
mkdir -p logs

export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${SLURM_JOB_GPUS:-1}

bash eval_vlm_3r_vstibench_episodic_untrainable.sh

echo "End Time: $(date)"
echo "Job completed"
