#!/bin/bash
set -euo pipefail

# VSTI eval with the real VSTI LoRA checkpoint and freshly initialized
# query-selection MLPs. No query-selection checkpoint is loaded here.
NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
if [ "$NUM_GPUS" -gt 1 ]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
else
  export CUDA_VISIBLE_DEVICES=0
fi
export LMMS_EVAL_LAUNCHER="accelerate"

VSTI_LORA_PATH=${VSTI_LORA_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora-vstibench}
MODEL_BASE=${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}
MAX_FRAMES_NUM=${MAX_FRAMES_NUM:-128}
QUERY_SELECTION_NUM_SELECT=${QUERY_SELECTION_NUM_SELECT:-32}
QUERY_SELECTION_TEMPERATURE=${QUERY_SELECTION_TEMPERATURE:-1.0}
QUERY_SELECTION_USE_GUMBEL=${QUERY_SELECTION_USE_GUMBEL:-False}
QUERY_SELECTION_PROJECT_SELECTED=${QUERY_SELECTION_PROJECT_SELECTED:-True}
LOG_SUFFIX=${LOG_SUFFIX:-query_selection_random_128to32_project_${QUERY_SELECTION_PROJECT_SELECTED}}
OUTPUT_PATH=${OUTPUT_PATH:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench_${LOG_SUFFIX}}

MODEL_ARGS="pretrained=${VSTI_LORA_PATH},model_base=${MODEL_BASE},model_name=llava_qwen_lora,conv_template=qwen_1_5,max_frames_num=${MAX_FRAMES_NUM},use_dual_memory=False,use_query_selection=True,query_selection_num_select=${QUERY_SELECTION_NUM_SELECT},query_selection_temperature=${QUERY_SELECTION_TEMPERATURE},query_selection_use_gumbel=${QUERY_SELECTION_USE_GUMBEL},query_selection_project_selected=${QUERY_SELECTION_PROJECT_SELECTED},frames_upbound=${MAX_FRAMES_NUM},force_sample=True"

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

echo "Running VSTiBench random query-selection eval"
echo "VSTI LoRA: ${VSTI_LORA_PATH}"
echo "Model base: ${MODEL_BASE}"
echo "Random query MLPs: no query-selection checkpoint is loaded"
echo "Project selected tokens: ${QUERY_SELECTION_PROJECT_SELECTED}"
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
