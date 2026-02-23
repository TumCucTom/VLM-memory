# Eval VSI with a local checkpoint.
# Set EVAL_CHECKPOINT to the checkpoint path.
# Always uses adapter-overlay: load base+LoRA (good baseline) then overlay adapter+memory from checkpoint,
# so memory performance is on top of the same 60.85% base. Set EVAL_USE_DUAL_MEMORY=False for no-memory checkpoints.

NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
CHECKPOINT="${EVAL_CHECKPOINT:?Set EVAL_CHECKPOINT to checkpoint path}"
USE_DUAL_MEMORY="${EVAL_USE_DUAL_MEMORY:-True}"
# Resolve relative path against PROJECT_DIR (slurm sets PROJECT_DIR to repo root)
[ "${CHECKPOINT#/}" = "$CHECKPOINT" ] && CHECKPOINT="${PROJECT_DIR:-..}/${CHECKPOINT}"

EVAL_LORA_PATH="${EVAL_LORA_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora}"
EVAL_MODEL_BASE="${EVAL_MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"

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

# Base+LoRA + overlay adapter/memory from checkpoint (good base for all checkpoints)
# Do not set model_name so builder gets name from pretrained path (contains 'lora') and uses LoRA branch with non_lora_trainables.bin from HF
MODEL_ARGS="pretrained=${EVAL_LORA_PATH},model_base=${EVAL_MODEL_BASE},checkpoint_adapter=${CHECKPOINT},conv_template=qwen_1_5,max_frames_num=32,use_dual_memory=${USE_DUAL_MEMORY}"
accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    -m lmms_eval \
    --model vlm_3r \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix wm_checkpoint \
    --output_path logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench
