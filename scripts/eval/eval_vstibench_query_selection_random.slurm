#!/bin/bash
#SBATCH --job-name=eval_vsti_qsel_rand
#SBATCH --output=logs/eval/vstibench_query_selection_random_%j.out
#SBATCH --error=logs/eval/vstibench_query_selection_random_%j.err
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
export LLAVA_DEBUG_GRAD=0
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}

export VSTI_LORA_PATH="${VSTI_LORA_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora-vstibench}"
export MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
export MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-128}"
export QUERY_SELECTION_NUM_SELECT="${QUERY_SELECTION_NUM_SELECT:-32}"
export QUERY_SELECTION_TEMPERATURE="${QUERY_SELECTION_TEMPERATURE:-1.0}"
export QUERY_SELECTION_USE_GUMBEL="${QUERY_SELECTION_USE_GUMBEL:-False}"
export QUERY_SELECTION_PROJECT_SELECTED="${QUERY_SELECTION_PROJECT_SELECTED:-True}"

PROJECT_TAG="projected"
if [[ "$(printf '%s' "${QUERY_SELECTION_PROJECT_SELECTED}" | tr '[:upper:]' '[:lower:]')" =~ ^(false|0|no)$ ]]; then
  PROJECT_TAG="raw"
fi
export LOG_SUFFIX="${LOG_SUFFIX:-query_selection_random_128to32_${PROJECT_TAG}_${SLURM_JOB_ID}}"

bash eval_vlm_3r_vstibench_query_selection_random.sh

echo "End Time: $(date)"
echo "Job completed"
