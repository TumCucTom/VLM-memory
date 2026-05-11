#!/bin/bash
set -euo pipefail

# Use NUM_GPUS from environment (e.g. set by SLURM) or default to 1 (per-node)
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

QUERY_SELECTION_CKPT=${QUERY_SELECTION_CKPT:-/home/b6ar/trvbale.b6ar/scratch/VLM-memory/work_dirs/vlm2-query-selection-128to32-4443888}
MAX_FRAMES_NUM=${MAX_FRAMES_NUM:-128}
QUERY_SELECTION_NUM_SELECT=${QUERY_SELECTION_NUM_SELECT:-32}
QUERY_SELECTION_TEMPERATURE=${QUERY_SELECTION_TEMPERATURE:-1.0}
QUERY_SELECTION_USE_GUMBEL=${QUERY_SELECTION_USE_GUMBEL:-False}
QUERY_SELECTION_PROJECT_SELECTED=${QUERY_SELECTION_PROJECT_SELECTED:-True}
LOG_SUFFIX=${LOG_SUFFIX:-query_selection_128to32_4443888}
OUTPUT_PATH=${OUTPUT_PATH:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_${LOG_SUFFIX}}

MODEL_ARGS="pretrained=${QUERY_SELECTION_CKPT},model_name=llava_qwen,conv_template=qwen_1_5,max_frames_num=${MAX_FRAMES_NUM},use_dual_memory=False,use_query_selection=True,query_selection_num_select=${QUERY_SELECTION_NUM_SELECT},query_selection_temperature=${QUERY_SELECTION_TEMPERATURE},query_selection_use_gumbel=${QUERY_SELECTION_USE_GUMBEL},query_selection_project_selected=${QUERY_SELECTION_PROJECT_SELECTED},frames_upbound=${MAX_FRAMES_NUM},force_sample=True"

# Multinode: total processes = NNODES * NUM_GPUS; single-node uses NUM_GPUS only
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

echo "Running VSiBench query-selection eval"
echo "Checkpoint: ${QUERY_SELECTION_CKPT}"
echo "Model args: ${MODEL_ARGS}"
echo "Output path: ${OUTPUT_PATH}"

accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    -m lmms_eval \
    --model vlm_3r \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${LOG_SUFFIX}" \
    --output_path "${OUTPUT_PATH}"
