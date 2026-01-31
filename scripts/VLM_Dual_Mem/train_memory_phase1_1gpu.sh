#!/bin/bash

# Phase 1: Memory Component Training
# Based on TRAINING_DUAL_MEMORY.md Phase 1 recommendations
# Trains only memory components (attention modules fully, MLPs fully)
# Freezes: Base VLM-3R (vision tower, language model, spatial tower)
#          Existing fusion block and MM projector (keep pre-trained weights)

# set training parameters
# IMPORTANT: Set NUM_GPUS_PER_NODE to the number of GPUs you want to use on this machine.
export NUM_GPUS_PER_NODE=1
MASTER_ADDR="localhost" # For single-node multi-GPU

MASTER_PORT=30000

# Set up the data folder
IMAGE_FOLDER="data/vlm_3r_data"
VIDEO_FOLDER="data/vlm_3r_data"
DATA_YAML="scripts/VLM_3R/vsibench_data.yaml" # e.g exp.yaml
# Add suffix based on number of GPUs
SUFFIX="vlm_3r_memory_phase1_salience_gated_1gpu"
NUM_TRAIN_EPOCHS=5
SAVE_TOTAL_LIMIT=5
SPATIAL_TOWER="cut3r"
FUSION_BLOCK="cross_attention"
SPATIAL_TOWER_SELECT_FEATURE="all_tokens"
SPATIAL_FEATURE_DIM="768"
# Phase 1: Freeze fusion block and MM projector
TUNE_MM_MLP_ADAPTER=False
TUNE_FUSION_BLOCK=False
# Enable memory component training
TUNE_MEMORY_COMPONENTS=True
# Adjust gradient accumulation: bs=128 total
# For 4 GPUs: 32 * 4 = 128, for 1 GPU: 128 * 1 = 128
GRADIENT_ACCUMULATION_STEPS=128 # bs=128, num_gpus=1

################ Arnold Jobs ################
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llava_video_7b_qwen2_${SUFFIX}"
PREV_STAGE_CHECKPOINT="checkpoints/LLaVA-Video-7B-Qwen2"

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --spatial_tower $SPATIAL_TOWER \
    --spatial_tower_select_feature $SPATIAL_TOWER_SELECT_FEATURE \
    --spatial_feature_dim $SPATIAL_FEATURE_DIM \
    --fusion_block $FUSION_BLOCK \
    --tune_spatial_tower False \
    --tune_fusion_block $TUNE_FUSION_BLOCK \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --tune_memory_components $TUNE_MEMORY_COMPONENTS \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $SUFFIX \
    --output_dir work_dirs_auto_eval/$MID_RUN_NAME \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 1e-4 \
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
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;

