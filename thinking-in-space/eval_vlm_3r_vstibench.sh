# Use GPUS_PER_NODE from environment (set by SLURM) or default to 1
GPUS_PER_NODE=${GPUS_PER_NODE:-${NUM_GPUS:-1}}
if [ "$GPUS_PER_NODE" -gt 1 ]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
else
  export CUDA_VISIBLE_DEVICES=0
fi
export LMMS_EVAL_LAUNCHER="accelerate"

# NNODES from environment (set by SLURM script), defaults to 1 for single-node
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# TOTAL_PROCESSES = total GPUs across all nodes
TOTAL_PROCESSES=$((NNODES * GPUS_PER_NODE))
NUM_GPUS=$GPUS_PER_NODE

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
    --model_args pretrained=Journey9ni/vlm-3r-llava-qwen2-lora-vstibench,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32,use_dual_memory=False \
    --tasks vstibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vlm_3r_7b_qwen2_lora \
    --output_path logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench