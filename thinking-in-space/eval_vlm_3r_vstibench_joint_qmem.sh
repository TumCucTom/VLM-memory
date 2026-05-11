#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-bf16}

if [ "$NUM_GPUS" -gt 1 ]; then
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
else
  export CUDA_VISIBLE_DEVICES=0
fi
export LMMS_EVAL_LAUNCHER="accelerate"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

JOINT_QMEM_CKPT=${JOINT_QMEM_CKPT:-/home/b6ar/trvbale.b6ar/scratch/VLM-memory/work_dirs/vlm2-joint-vsti-query-memory-lora-128to32-4467965}
MODEL_BASE=${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}
MAX_FRAMES_NUM=${MAX_FRAMES_NUM:-128}
QUERY_SELECTION_NUM_SELECT=${QUERY_SELECTION_NUM_SELECT:-32}
QUERY_SELECTION_TEMPERATURE=${QUERY_SELECTION_TEMPERATURE:-1.0}
QUERY_SELECTION_USE_GUMBEL=${QUERY_SELECTION_USE_GUMBEL:-False}
QUERY_SELECTION_PROJECT_SELECTED=${QUERY_SELECTION_PROJECT_SELECTED:-False}
LOG_SUFFIX=${LOG_SUFFIX:-joint_qmem_vsti_4467965}
OUTPUT_PATH=${OUTPUT_PATH:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench_${LOG_SUFFIX}}

MODEL_ARGS="pretrained=${JOINT_QMEM_CKPT},model_base=${MODEL_BASE},model_name=llava_qwen_lora,conv_template=qwen_1_5,max_frames_num=${MAX_FRAMES_NUM},use_dual_memory=True,memory_mode=both,use_query_selection=True,query_selection_num_select=${QUERY_SELECTION_NUM_SELECT},query_selection_temperature=${QUERY_SELECTION_TEMPERATURE},query_selection_use_gumbel=${QUERY_SELECTION_USE_GUMBEL},query_selection_project_selected=${QUERY_SELECTION_PROJECT_SELECTED},frames_upbound=${MAX_FRAMES_NUM},force_sample=True"

TOTAL_PROCESSES=$((NNODES * NUM_GPUS))
LAUNCH_ARGS=(--num_processes="$TOTAL_PROCESSES" --mixed_precision="$ACCELERATE_MIXED_PRECISION")
if [ "$NNODES" -gt 1 ]; then
  LAUNCH_ARGS+=(
    --num_machines="$NNODES"
    --machine_rank="$NODE_RANK"
    --main_process_ip="$MASTER_ADDR"
    --main_process_port="$MASTER_PORT"
  )
fi

echo "Running VSTiBench joint query+memory eval"
echo "Checkpoint: ${JOINT_QMEM_CKPT}"
echo "Model base: ${MODEL_BASE}"
echo "Model args: ${MODEL_ARGS}"
echo "PyTorch CUDA alloc conf: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Accelerate mixed precision: ${ACCELERATE_MIXED_PRECISION}"
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
