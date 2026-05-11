# Eval VSTI with a local checkpoint.
# Set EVAL_CHECKPOINT to the checkpoint path. Uses adapter-overlay (base+LoRA + overlay) for all checkpoints.
# Set EVAL_USE_DUAL_MEMORY=False for no-memory checkpoints.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
CHECKPOINT="${EVAL_CHECKPOINT:?Set EVAL_CHECKPOINT to checkpoint path}"
USE_DUAL_MEMORY="${EVAL_USE_DUAL_MEMORY:-True}"
[ "${CHECKPOINT#/}" = "$CHECKPOINT" ] && CHECKPOINT="${PROJECT_DIR:-..}/${CHECKPOINT}"

EVAL_LORA_PATH="${EVAL_LORA_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora}"
EVAL_MODEL_BASE="${EVAL_MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
OVERLAY_MEMORY_ONLY="${EVAL_OVERLAY_MEMORY_ONLY:-False}"
USE_QUERY_SELECTION="${EVAL_USE_QUERY_SELECTION:-False}"
QUERY_SELECTION_NUM_SELECT="${EVAL_QUERY_SELECTION_NUM_SELECT:-32}"
QUERY_SELECTION_TEMPERATURE="${EVAL_QUERY_SELECTION_TEMPERATURE:-1.0}"
QUERY_SELECTION_USE_GUMBEL="${EVAL_QUERY_SELECTION_USE_GUMBEL:-False}"
QUERY_SELECTION_PROJECT_SELECTED="${EVAL_QUERY_SELECTION_PROJECT_SELECTED:-True}"
FRAMES_UPBOUND="${EVAL_FRAMES_UPBOUND:-32}"
FORCE_SAMPLE="${EVAL_FORCE_SAMPLE:-True}"
LOG_SUFFIX="${LOG_SUFFIX:-wm_checkpoint}"
OUTPUT_PATH="${OUTPUT_PATH:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench_${LOG_SUFFIX}}"

if [ "$NUM_GPUS" -gt 1 ]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
else
  export CUDA_VISIBLE_DEVICES=0
fi
export LMMS_EVAL_LAUNCHER="accelerate"

TOTAL_PROCESSES=$((NNODES * NUM_GPUS))
LAUNCH_ARGS=(--num_processes="$TOTAL_PROCESSES")
if [ "$NNODES" -gt 1 ]; then
  LAUNCH_ARGS+=(
    --num_machines="$NNODES"
    --machine_rank="$NODE_RANK"
    --main_process_ip="$MASTER_ADDR"
    --main_process_port="$MASTER_PORT"
  )
fi

# Do not set model_name so builder uses LoRA branch (non_lora_trainables.bin from HF)
MODEL_ARGS="pretrained=${EVAL_LORA_PATH},model_base=${EVAL_MODEL_BASE},checkpoint_adapter=${CHECKPOINT},conv_template=qwen_1_5,max_frames_num=32,use_dual_memory=${USE_DUAL_MEMORY},overlay_memory_only=${OVERLAY_MEMORY_ONLY},use_query_selection=${USE_QUERY_SELECTION},query_selection_num_select=${QUERY_SELECTION_NUM_SELECT},query_selection_temperature=${QUERY_SELECTION_TEMPERATURE},query_selection_use_gumbel=${QUERY_SELECTION_USE_GUMBEL},query_selection_project_selected=${QUERY_SELECTION_PROJECT_SELECTED},frames_upbound=${FRAMES_UPBOUND},force_sample=${FORCE_SAMPLE}"
echo "Running VSTiBench checkpoint eval"
echo "Checkpoint: ${CHECKPOINT}"
echo "Model args: ${MODEL_ARGS}"
echo "Output path: ${OUTPUT_PATH}"
accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    -m lmms_eval \
    --model vlm_3r \
    --model_args "${MODEL_ARGS}" \
    --tasks vstibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX}" \
    --output_path "${OUTPUT_PATH}"

if ! find "${OUTPUT_PATH}" -type f -name "results*.json" -print -quit | grep -q .; then
    echo "ERROR: evaluation completed without a results JSON under ${OUTPUT_PATH}" >&2
    exit 1
fi
