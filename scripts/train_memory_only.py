#!/usr/bin/env python3
"""
Standalone training script for memory-only fine-tuning.
Freezes everything except memory modules.

Usage:
    python scripts/train_memory_only.py \
        --model_path "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2" \
        --data_path "path/to/data.yaml" \
        --output_dir "work_dirs/memory-only"
"""

import argparse
import torch
import transformers
from transformers import TrainingArguments
from llava.train.train import (
    ModelArguments, DataArguments, TrainingArguments as LLaVATrainingArguments,
    train, get_model
)
from llava.utils import rank0_print


def create_memory_only_args():
    """Create arguments for memory-only training"""
    parser = argparse.ArgumentParser(description="Memory-only fine-tuning")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--vision_tower", type=str, 
                       default="google/siglip-so400m-patch14-384",
                       help="Vision encoder")
    
    # Spatial encoder
    parser.add_argument("--spatial_tower", type=str, default="cut3r",
                       choices=["cut3r", "vggt", "spann3r"],
                       help="Spatial encoder type")
    parser.add_argument("--spatial_tower_select_feature", type=str,
                       default="patch_tokens",
                       help="Which spatial features to use")
    parser.add_argument("--spatial_feature_dim", type=int, default=768,
                       help="Spatial feature dimension")
    
    # Fusion
    parser.add_argument("--fusion_block", type=str, default="cross_attention",
                       help="Fusion block type")
    parser.add_argument("--train_fusion_block", action="store_true",
                       help="Also train fusion block (recommended)")
    
    # Memory configuration
    parser.add_argument("--memory_mode", type=str, default="working_only",
                       choices=["working_only", "episodic_only", "both"],
                       help="Memory training mode: 'working_only', 'episodic_only', or 'both' (default: working_only)")
    parser.add_argument("--memory_L_w", type=int, default=8,
                       help="Working memory capacity")
    parser.add_argument("--memory_L_e", type=int, default=32,
                       help="Episodic memory capacity")
    parser.add_argument("--memory_num_heads", type=int, default=8,
                       help="Attention heads for memory")
    parser.add_argument("--memory_dropout", type=float, default=0.1,
                       help="Dropout for memory")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to data YAML or JSON")
    parser.add_argument("--image_folder", type=str, required=True,
                       help="Path to image folder")
    parser.add_argument("--video_folder", type=str, required=True,
                       help="Path to video folder")
    
    # Training
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for memory modules")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs")
    
    return parser.parse_args()


def setup_memory_only_training(args):
    """Setup model arguments for memory-only training"""
    
    # Model arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_path,
        vision_tower=args.vision_tower,
        spatial_tower=args.spatial_tower,
        spatial_tower_select_feature=args.spatial_tower_select_feature,
        spatial_feature_dim=args.spatial_feature_dim,
        fusion_block=args.fusion_block,
        tune_spatial_tower=False,  # Freeze spatial encoder
        tune_fusion_block=args.train_fusion_block,  # Optional
        tune_mm_mlp_adapter=False,  # Freeze projector
        use_dual_memory=True,  # Enable memory
        memory_mode=args.memory_mode,  # Memory mode: working_only, episodic_only, or both
        memory_L_w=args.memory_L_w,
        memory_L_e=args.memory_L_e,
        memory_num_heads=args.memory_num_heads,
        memory_dropout=args.memory_dropout,
        tune_memory_components=True,  # Train memory components
        mm_tunable_parts="dual_memory" + (",fusion_block" if args.train_fusion_block else ""),
        version="qwen_1_5",
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
    )
    
    # Data arguments
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        video_folder=args.video_folder,
        lazy_preprocess=True,
        data_args=None,  # Will be set by training script
    )
    
    # Training arguments
    training_args = LLaVATrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="no",
        bf16=True,
        tf32=True,
        model_max_length=32768,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="wandb",
        torch_compile=True,
        torch_compile_backend="inductor",
        dataloader_drop_last=True,
        attn_implementation="flash_attention_2",
        group_by_modality_length=True,
        image_aspect_ratio="anyres_max_9",
        image_grid_pinpoints="(1x1),...,(6x6)",
        mm_patch_merge_type="spatial_unpad",
        frames_upbound=32,
        mm_newline_position="grid",
    )
    
    return model_args, data_args, training_args


def verify_memory_setup(model):
    """Verify memory module is properly set up for training"""
    rank0_print("\n" + "="*50)
    rank0_print("Memory Setup Verification")
    rank0_print("="*50)
    
    # Check if memory module exists
    if model.get_model().dual_memory is None:
        rank0_print("❌ ERROR: Dual-memory module is None!")
        rank0_print("   Check that --use_dual_memory=True is set")
        return False
    else:
        rank0_print("✅ Dual-memory module initialized")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_params = sum(p.numel() for n, p in model.named_parameters() 
                       if 'dual_memory' in n and p.requires_grad)
    
    rank0_print(f"\nTotal parameters: {total_params/1e9:.2f}B")
    rank0_print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    rank0_print(f"Memory parameters: {memory_params/1e6:.2f}M")
    
    # Check what's frozen
    frozen_components = []
    if not any(p.requires_grad for n, p in model.named_parameters() if 'vision_tower' in n):
        frozen_components.append("Vision Tower")
    if not any(p.requires_grad for n, p in model.named_parameters() if 'language_model' in n or 'model' in n):
        frozen_components.append("Language Model")
    if not any(p.requires_grad for n, p in model.named_parameters() if 'spatial_tower' in n):
        frozen_components.append("Spatial Tower")
    if not any(p.requires_grad for n, p in model.named_parameters() if 'mm_projector' in n):
        frozen_components.append("Projector")
    
    rank0_print(f"\nFrozen components: {', '.join(frozen_components)}")
    
    # Check what's trainable
    trainable_components = []
    if any(p.requires_grad for n, p in model.named_parameters() if 'dual_memory' in n):
        trainable_components.append("Dual Memory")
    if any(p.requires_grad for n, p in model.named_parameters() if 'fusion_block' in n):
        trainable_components.append("Fusion Block")
    
    rank0_print(f"Trainable components: {', '.join(trainable_components)}")
    rank0_print("="*50 + "\n")
    
    return True


def main():
    args = create_memory_only_args()
    
    rank0_print("="*60)
    rank0_print("Memory-Only Fine-Tuning")
    rank0_print("="*60)
    rank0_print(f"Model: {args.model_path}")
    rank0_print(f"Memory Mode: {args.memory_mode}")
    rank0_print(f"Memory: L_w={args.memory_L_w}, L_e={args.memory_L_e}")
    rank0_print(f"Training: Memory modules {'+ Fusion' if args.train_fusion_block else 'only'}")
    rank0_print("="*60 + "\n")
    
    # Setup arguments
    model_args, data_args, training_args = setup_memory_only_training(args)
    
    # Verify setup before training
    # (This would be done in the actual training function)
    
    rank0_print("Starting training...")
    rank0_print("Note: Memory modules will be trained, everything else is frozen.\n")
    
    # The actual training is handled by the train() function
    # This script just sets up the arguments
    # You would call: train() with these arguments
    
    # For now, print the command that would be used
    rank0_print("\nTo run training, use the shell script:")
    rank0_print("  bash scripts/train_memory_only.sh")
    rank0_print("\nOr modify this script to call train() directly.")


if __name__ == "__main__":
    main()

