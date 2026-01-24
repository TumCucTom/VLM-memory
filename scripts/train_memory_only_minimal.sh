#!/bin/bash
# Minimal memory-only training script (for experiments)
# Trains ONLY memory modules, freezes everything else including projector

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
MODEL_PATH="lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"
DATA_YAML="scripts/VLM_3R/vsibench_data.yaml"  # CHANGE THIS
IMAGE_FOLDER="data/vlm_3r_data"  # CHANGE THIS
VIDEO_FOLDER="data/vlm_3r_data"  # CHANGE THIS
OUTPUT_DIR="work_dirs/memory-only-minimal"
NUM_GPUS=1

# ============================================
# MEMORY CONFIGURATION (VLM² recommended)
# ============================================
MEMORY_L_W=8
MEMORY_L_E=32

# ============================================
# TRAINING CONFIGURATION
# ============================================
NUM_EPOCHS=3
LEARNING_RATE=1e-4
BATCH_SIZE=1
GRAD_ACCUM=16

echo "=========================================="
echo "Memory-Only Fine-Tuning (MINIMAL)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Memory: L_w=${MEMORY_L_W}, L_e=${MEMORY_L_E}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "FROZEN: Vision, Language Model, Spatial, Fusion, Projector"
echo "TRAINED: Dual-Memory modules ONLY (~12M params)"
echo ""
echo "WARNING: This may be suboptimal because:"
echo "  - Projector was trained for original features"
echo "  - Memory outputs may need different projection"
echo "  - Consider using train_memory_only_simple.sh instead"
echo "=========================================="

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    llava/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version qwen_1_5 \
    --data_path ${DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --video_folder ${VIDEO_FOLDER} \
    --spatial_tower cut3r \
    --spatial_tower_select_feature patch_tokens \
    --spatial_feature_dim 768 \
    --fusion_block cross_attention \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --evaluation_strategy no \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True \
    --report_to wandb \
    --run_name "memory-only-minimal-${MEMORY_L_W}-${MEMORY_L_E}" \
    \
    --use_dual_memory True \
    --memory_L_w ${MEMORY_L_W} \
    --memory_L_e ${MEMORY_L_E} \
    --memory_num_heads 8 \
    --memory_dropout 0.1 \
    --tune_dual_memory True \
    --mm_tunable_parts "dual_memory" \
    --tune_spatial_tower False \
    --tune_fusion_block False \
    --tune_mm_mlp_adapter False \
    --tune_mm_vision_resampler False \
    --unfreeze_mm_vision_tower False \
    --unfreeze_language_model False \
    --lora_enable False

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoint: ${OUTPUT_DIR}"
echo "=========================================="

