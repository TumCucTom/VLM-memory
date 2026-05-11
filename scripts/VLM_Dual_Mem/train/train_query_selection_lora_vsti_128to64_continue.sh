#!/bin/bash
# Continue VSTI training with:
#   1. trainable VSTI task LoRA
#   2. query-selection module unfrozen
#
# Warm start:
#   - load VSTI task LoRA as trainable
#   - overlay the current query-selection checkpoint on top

set -e
export OMP_NUM_THREADS=8

export NCCL_IB_GID_INDEX=3
unset NCCL_SOCKET_IFNAME 2>/dev/null || true
export NCCL_DEBUG=INFO
export VLM_NCCL_TIMEOUT_HOURS="${VLM_NCCL_TIMEOUT_HOURS:-24}"

MODEL_PATH="${MODEL_PATH:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
TASK_LORA_PATH="${TASK_LORA_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora-vstibench}"
CHECKPOINT_ADAPTER="${CHECKPOINT_ADAPTER:-$HOME/scratch/VLM-memory/work_dirs/vlm2-query-selection-vsti-raw-128to32-4461526/checkpoint-800}"
CHECKPOINT_OVERLAY_MEMORY_ONLY="${CHECKPOINT_OVERLAY_MEMORY_ONLY:-False}"

VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-google/siglip-so400m-patch14-384}"
SPATIAL_TOWER="${SPATIAL_TOWER:-cut3r}"
SPATIAL_TOWER_SELECT_FEATURE="${SPATIAL_TOWER_SELECT_FEATURE:-patch_tokens}"
SPATIAL_FEATURE_DIM="${SPATIAL_FEATURE_DIM:-768}"
FUSION_BLOCK="${FUSION_BLOCK:-cross_attention}"

USE_QUERY_SELECTION="${USE_QUERY_SELECTION:-True}"
QUERY_SELECTION_NUM_SELECT="${QUERY_SELECTION_NUM_SELECT:-64}"
QUERY_SELECTION_TEMPERATURE="${QUERY_SELECTION_TEMPERATURE:-1.0}"
QUERY_SELECTION_USE_GUMBEL="${QUERY_SELECTION_USE_GUMBEL:-True}"
QUERY_SELECTION_PROJECT_SELECTED="${QUERY_SELECTION_PROJECT_SELECTED:-False}"
QUERY_SELECTION_DISABLE_DUAL_MEMORY="${QUERY_SELECTION_DISABLE_DUAL_MEMORY:-True}"

USE_DUAL_MEMORY="${USE_DUAL_MEMORY:-False}"
MEMORY_MODE="${MEMORY_MODE:-both}"
MEMORY_L_W="${MEMORY_L_W:-8}"
MEMORY_L_E="${MEMORY_L_E:-32}"
EPISODIC_MEMORY_GATED_ATTENTION="${EPISODIC_MEMORY_GATED_ATTENTION:-False}"

FRAMES_UPBOUND="${FRAMES_UPBOUND:-128}"
FORCE_SAMPLE="${FORCE_SAMPLE:-True}"

NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-0}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-${NUM_GPUS:-1}}

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_EFFECTIVE_BATCH="${TARGET_EFFECTIVE_BATCH:-128}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
    WORLD_MICRO_BATCH=$((NNODES * NUM_GPUS_PER_NODE * BATCH_SIZE))
    GRADIENT_ACCUMULATION_STEPS=$(((TARGET_EFFECTIVE_BATCH + WORLD_MICRO_BATCH - 1) / WORLD_MICRO_BATCH))
    if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]]; then
        GRADIENT_ACCUMULATION_STEPS=1
    fi
fi
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
MAX_STEPS="${MAX_STEPS:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ALLOW_EXISTING_OUTPUT_DIR="${ALLOW_EXISTING_OUTPUT_DIR:-False}"

DATA_YAML="${DATA_YAML:-scripts/VLM_3R/vstibench_train_data.yaml}"
IMAGE_FOLDER="${IMAGE_FOLDER:-data/vlm_3r_data}"
VIDEO_FOLDER="${VIDEO_FOLDER:-data/processed_data}"

RUN_STAMP="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-vlm2-query-selection-lora-vsti-128to64-${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/${RUN_NAME}}"

ALLOW_EXISTING_OUTPUT_DIR_NORMALIZED="$(printf '%s' "${ALLOW_EXISTING_OUTPUT_DIR}" | tr '[:upper:]' '[:lower:]')"
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
   && [[ -z "${RESUME_FROM_CHECKPOINT}" ]] \
   && [[ "${ALLOW_EXISTING_OUTPUT_DIR_NORMALIZED}" != "true" ]] \
   && [[ "${ALLOW_EXISTING_OUTPUT_DIR_NORMALIZED}" != "1" ]] \
   && [[ "${ALLOW_EXISTING_OUTPUT_DIR_NORMALIZED}" != "yes" ]]; then
    echo "ERROR: OUTPUT_DIR already exists and is non-empty: ${OUTPUT_DIR}" >&2
    echo "Set RESUME_FROM_CHECKPOINT=<path|true> to resume, choose a new RUN_NAME/OUTPUT_DIR, or set ALLOW_EXISTING_OUTPUT_DIR=True." >&2
    exit 1
fi

MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
    MAX_STEPS_ARGS=(--max_steps "${MAX_STEPS}")
fi

echo "=========================================="
echo "VSTI query-selection continuation: LoRA + query selection"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Task LoRA warm start: ${TASK_LORA_PATH}"
echo "Checkpoint adapter: ${CHECKPOINT_ADAPTER} (memory_only=${CHECKPOINT_OVERLAY_MEMORY_ONLY})"
echo "Query selection: use=${USE_QUERY_SELECTION}, num_select=${QUERY_SELECTION_NUM_SELECT}, temp=${QUERY_SELECTION_TEMPERATURE}, project_selected=${QUERY_SELECTION_PROJECT_SELECTED}, disable_dual_memory=${QUERY_SELECTION_DISABLE_DUAL_MEMORY}"
echo "Dual memory: use=${USE_DUAL_MEMORY}"
echo "Frame sampling: frames_upbound=${FRAMES_UPBOUND}, force_sample=${FORCE_SAMPLE}"
echo "Data: ${DATA_YAML}"
echo "Distributed: nnodes=${NNODES}, gpus_per_node=${NUM_GPUS_PER_NODE}, batch=${BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, target_effective_batch=${TARGET_EFFECTIVE_BATCH}"
echo "Training: epochs=${NUM_TRAIN_EPOCHS}, lr=${LEARNING_RATE}, save_steps=${SAVE_STEPS}, save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "Checkpointing: save_only_model=${SAVE_ONLY_MODEL}, resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

ACCELERATE_CPU_AFFINITY=0 torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-29500} \
    llava/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --lora_weight_path "${TASK_LORA_PATH}" \
    --lora_enable True \
    --version qwen_1_5 \
    --data_path "${DATA_YAML}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --spatial_tower ${SPATIAL_TOWER} \
    --spatial_tower_select_feature ${SPATIAL_TOWER_SELECT_FEATURE} \
    --spatial_feature_dim ${SPATIAL_FEATURE_DIM} \
    --fusion_block ${FUSION_BLOCK} \
    --checkpoint_adapter "${CHECKPOINT_ADAPTER}" \
    --checkpoint_overlay_memory_only ${CHECKPOINT_OVERLAY_MEMORY_ONLY} \
    --use_dual_memory ${USE_DUAL_MEMORY} \
    --memory_mode ${MEMORY_MODE} \
    --memory_L_w ${MEMORY_L_W} \
    --memory_L_e ${MEMORY_L_E} \
    --episodic_memory_gated_attention ${EPISODIC_MEMORY_GATED_ATTENTION} \
    --tune_spatial_tower False \
    --tune_fusion_block False \
    --tune_mm_mlp_adapter False \
    --tune_mm_vision_resampler False \
    --unfreeze_mm_vision_tower False \
    --unfreeze_language_model False \
    --use_query_selection ${USE_QUERY_SELECTION} \
    --query_selection_num_select ${QUERY_SELECTION_NUM_SELECT} \
    --query_selection_temperature ${QUERY_SELECTION_TEMPERATURE} \
    --query_selection_use_gumbel ${QUERY_SELECTION_USE_GUMBEL} \
    --query_selection_project_selected ${QUERY_SELECTION_PROJECT_SELECTED} \
    --query_selection_disable_dual_memory ${QUERY_SELECTION_DISABLE_DUAL_MEMORY} \
    --mm_tunable_parts "query_selection" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    $([[ -n "${RESUME_FROM_CHECKPOINT}" ]] && echo --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}") \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    "${MAX_STEPS_ARGS[@]}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --save_only_model ${SAVE_ONLY_MODEL} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --lazy_preprocess True \
    --force_sample ${FORCE_SAMPLE} \
    --report_to none \
    --torch_compile False \
    --dataloader_drop_last True \
    --frames_upbound ${FRAMES_UPBOUND} \
    --mm_newline_position grid

echo "=========================================="
echo "Training finished. Checkpoints: ${OUTPUT_DIR}"
echo "=========================================="
