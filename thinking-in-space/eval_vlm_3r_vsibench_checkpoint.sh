# Eval VSI with a local full checkpoint (e.g. working-memory checkpoint-4).
# Set EVAL_CHECKPOINT to the checkpoint dir (absolute path recommended).
# Uses model_name=llava_qwen, no model_base, use_dual_memory=True so the repo llava loads the full model.

NUM_GPUS=${NUM_GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
CHECKPOINT="${EVAL_CHECKPOINT:?Set EVAL_CHECKPOINT to checkpoint path (e.g. work_dirs/vlm2-working-memory-only-Lw8/checkpoint-4)}"
# Resolve relative path against PROJECT_DIR (slurm sets PROJECT_DIR to repo root)
[ "${CHECKPOINT#/}" = "$CHECKPOINT" ] && CHECKPOINT="${PROJECT_DIR:-..}/${CHECKPOINT}"
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

# Full checkpoint: pretrained=path, model_name=llava_qwen, no model_base, use_dual_memory=True
accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    -m lmms_eval \
    --model vlm_3r \
    --model_args "pretrained=${CHECKPOINT},model_name=llava_qwen,conv_template=qwen_1_5,max_frames_num=32,use_dual_memory=True" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix wm_checkpoint \
    --output_path logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench
