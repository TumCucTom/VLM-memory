#!/bin/bash
# Train with everything frozen except memory. Trainable: only memory (mm_tunable_parts=dual_memory).
# working_only = working memory only; episodic and fusion disabled.

set -e
export OMP_NUM_THREADS=8

export NCCL_IB_GID_INDEX=3
unset NCCL_SOCKET_IFNAME 2>/dev/null || true
export NCCL_DEBUG=INFO
# Multi-GPU: process group timeout (hours); train.py inits with this before DeepSpeed (default 24)
export VLM_NCCL_TIMEOUT_HOURS="${VLM_NCCL_TIMEOUT_HOURS:-24}"
# torch_compile: disabled (PyTorch NCCL watchdog has hardcoded 10min; first step can exceed that with 4 GPUs)

# Model: base VLM + task LoRA (VLM-3R) merged before training — training uses base + LoRA, not base-only
MODEL_PATH="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
TASK_LORA_PATH="${TASK_LORA_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
SPATIAL_TOWER="cut3r"
SPATIAL_TOWER_SELECT_FEATURE="patch_tokens"
SPATIAL_FEATURE_DIM=768
FUSION_BLOCK="cross_attention"

# Memory: working memory only, size 8 (episodic/dual fusion disabled)
MEMORY_MODE="working_only"
MEMORY_L_W=8
MEMORY_L_E=32
MEMORY_NUM_HEADS=8
MEMORY_DROPOUT=0.1

# Training: hyperparams for 8-node (32 GPU) run. Effective batch = num_gpus * batch_size * grad_accum = 32*1*4 = 128.
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=1e-3
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_TOTAL_LIMIT=3
# Resume: set RESUME_FROM_CHECKPOINT=true (or path to checkpoint) to resume; default is no resume
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# Data (same layout as phase1 – adjust if your paths differ; slurm can override via DATA_YAML)
DATA_YAML="${DATA_YAML:-scripts/VLM_3R/vsibench_data.yaml}"
IMAGE_FOLDER="data/vlm_3r_data"
VIDEO_FOLDER="data/vlm_3r_data"

RUN_NAME="vlm2-working-memory-only-Lw8"
OUTPUT_DIR="work_dirs/${RUN_NAME}"

echo "=========================================="
echo "Working memory only (L_w=${MEMORY_L_W}), dual/episodic disabled"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Memory mode: ${MEMORY_MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root (script is scripts/VLM_Dual_Mem/train/*.sh)
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

# Multinode: set NNODES and NODE_RANK (e.g. by Slurm srun; train_working_memory_only.slurm uses 2 nodes = 8 GPUs)
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# Disable CPU affinity to avoid requiring pynvml on compute nodes (can set to 1 if pynvml is available later)
ACCELERATE_CPU_AFFINITY=0 torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE:-${NUM_GPUS:-1}} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-29500} \
    llava/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --lora_weight_path "${TASK_LORA_PATH}" \
    --lora_enable False \
    --version qwen_1_5 \
    --data_path "${DATA_YAML}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --spatial_tower ${SPATIAL_TOWER} \
    --spatial_tower_select_feature ${SPATIAL_TOWER_SELECT_FEATURE} \
    --spatial_feature_dim ${SPATIAL_FEATURE_DIM} \
    --fusion_block ${FUSION_BLOCK} \
    --tune_spatial_tower False \
    --tune_fusion_block False \
    --tune_mm_mlp_adapter False \
    --tune_mm_vision_resampler False \
    --unfreeze_mm_vision_tower False \
    --unfreeze_language_model False \
    --memory_mode ${MEMORY_MODE} \
    --memory_L_w ${MEMORY_L_W} \
    --memory_L_e ${MEMORY_L_E} \
    --memory_num_heads ${MEMORY_NUM_HEADS} \
    --memory_dropout ${MEMORY_DROPOUT} \
    --mm_tunable_parts "dual_memory" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    $([[ -n "${RESUME_FROM_CHECKPOINT}" ]] && echo --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}") \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound ${FRAMES_UPBOUND:-32} \
    --mm_newline_position grid

echo "=========================================="
echo "Training finished. Checkpoints: ${OUTPUT_DIR}"
echo "=========================================="
