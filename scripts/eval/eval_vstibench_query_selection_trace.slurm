#!/bin/bash
#SBATCH --job-name=trace_vsti_qsel
#SBATCH --output=logs/eval/vstibench_query_selection_trace_%j.out
#SBATCH --error=logs/eval/vstibench_query_selection_trace_%j.err
#SBATCH --time=01:00:00
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
python -c "import torch; print(f'{torch.__version__}, CUDA:{torch.version.cuda}, Available:{torch.cuda.is_available()}, Device:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/VLM-memory}"
REPORT_DIR="${REPORT_DIR:-$HOME/scratch/Dissertation_template_main}"
cd "$PROJECT_DIR"
mkdir -p logs/eval

cd thinking-in-space
mkdir -p logs/query_selection_traces

export PYTHONWARNINGS=ignore
export LLAVA_DEBUG_GRAD=0
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER="accelerate"
export NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}

export LMMS_LIMIT="${LMMS_LIMIT:-6}"
export QUERY_SELECTION_TRACE_LIMIT="${QUERY_SELECTION_TRACE_LIMIT:-$LMMS_LIMIT}"
export LOG_SUFFIX="${LOG_SUFFIX:-query_selection_trace_${SLURM_JOB_ID}}"
export QUERY_SELECTION_TRACE_PATH="${QUERY_SELECTION_TRACE_PATH:-$PWD/logs/query_selection_traces/${LOG_SUFFIX}.jsonl}"
export OUTPUT_PATH="${OUTPUT_PATH:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench_${LOG_SUFFIX}}"
export FIGURE_OUTPUT="${FIGURE_OUTPUT:-$REPORT_DIR/figures/query_selection_trace_doc0.png}"

echo "Trace path: ${QUERY_SELECTION_TRACE_PATH}"
echo "Figure output: ${FIGURE_OUTPUT}"
echo "lmms-eval limit: ${LMMS_LIMIT}"

bash eval_vlm_3r_vstibench_query_selection_lora.sh

TRACE_INPUT="$QUERY_SELECTION_TRACE_PATH"
if [ ! -s "$TRACE_INPUT" ]; then
    TRACE_ROOT="${QUERY_SELECTION_TRACE_PATH%.*}"
    TRACE_EXT="${QUERY_SELECTION_TRACE_PATH##*.}"
    COMBINED_TRACE="${TRACE_ROOT}.combined.${TRACE_EXT}"
    rm -f "$COMBINED_TRACE"
    for rank_trace in "${TRACE_ROOT}".rank*."${TRACE_EXT}"; do
        if [ -s "$rank_trace" ]; then
            cat "$rank_trace" >> "$COMBINED_TRACE"
        fi
    done
    TRACE_INPUT="$COMBINED_TRACE"
fi

python "$PROJECT_DIR/scripts/analysis/plot_query_selection_trace.py" \
    "$TRACE_INPUT" \
    --index 0 \
    --output "$FIGURE_OUTPUT"

echo "End Time: $(date)"
echo "Job completed"
