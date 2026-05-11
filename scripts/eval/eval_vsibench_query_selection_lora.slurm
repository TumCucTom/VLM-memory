#!/bin/bash
#SBATCH --job-name=eval_vsi_qsel_lora
#SBATCH --output=logs/eval/vsibench_query_selection_lora_%j.out
#SBATCH --error=logs/eval/vsibench_query_selection_lora_%j.err
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
echo "Numpy, Scipy versions: $(python -c "import numpy; import scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)")"
python -c "import torch; print(f'{torch.__version__}, CUDA:{torch.version.cuda}, Available:{torch.cuda.is_available()}, Device:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

cd thinking-in-space
mkdir -p logs

export PYTHONWARNINGS=ignore
export LLAVA_DEBUG_GRAD=0
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}

export QUERY_SELECTION_LORA_CKPT="${QUERY_SELECTION_LORA_CKPT:-$PROJECT_DIR/work_dirs/vlm2-query-selection-lora-128to32-4448070}"
export MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
export MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-128}"
export QUERY_SELECTION_NUM_SELECT="${QUERY_SELECTION_NUM_SELECT:-32}"
export QUERY_SELECTION_TEMPERATURE="${QUERY_SELECTION_TEMPERATURE:-1.0}"
export QUERY_SELECTION_USE_GUMBEL="${QUERY_SELECTION_USE_GUMBEL:-False}"
export LOG_SUFFIX="${LOG_SUFFIX:-query_selection_lora_128to32_4448070_${SLURM_JOB_ID}}"

bash eval_vlm_3r_vsibench_query_selection_lora.sh

echo "End Time: $(date)"
echo "Job completed"
