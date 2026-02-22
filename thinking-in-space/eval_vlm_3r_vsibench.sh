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

accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    -m lmms_eval \
    --model vlm_3r \
    --model_args pretrained=Journey9ni/vlm-3r-llava-qwen2-lora,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32,use_dual_memory=False \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vlm_3r_7b_qwen2_lora \
    --output_path logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench