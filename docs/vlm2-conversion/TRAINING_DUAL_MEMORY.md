# Training Strategy for Dual Memory System with VLM-3R

## Overview

This document outlines the training strategy for integrating the dual memory system (working memory + episodic memory) from VLM² into VLM-3R. The dual memory system introduces new trainable components that need to be fine-tuned to work effectively with the existing VLM-3R architecture.

## Current VLM-3R Training Setup

Based on `scripts/VLM_3R/train_vsibench.sh` and the README:

### What VLM-3R Trains (Without Dual Memory)

1. **LoRA Adaptation** (Low-Rank Adaptation):
   - `lora_r=128`, `lora_alpha=256`
   - Applied to fusion block and projection layers
   - Freezes base model weights, only trains adapter parameters

2. **Trainable Components**:
   - **3D Fusion Block** (`tune_fusion_block=True`): Cross-attention mechanism fusing visual and spatial features
   - **MM Projector** (`tune_mm_mlp_adapter=True`): MLP adapter aligning visual features with language model
   - **Spatial Tower**: Frozen (`tune_spatial_tower=False`) - CUT3R encoder is pre-trained

3. **Frozen Components**:
   - Vision tower (SigLIP encoder)
   - Language model (Qwen2 base)
   - Spatial tower (CUT3R)

4. **Training Data**:
   - **VLM-3R-DATA** (HuggingFace): 200K+ instruction tuning QA pairs
   - **VSiBench training data**: Spatial reasoning tasks
   - **VSTiBench training data**: Spatio-temporal reasoning
   - Datasets: ScanNet, ScanNet++, ARKitScenes (videos need to be downloaded separately)

5. **Training Objective**:
   - Same as LLaVA-NeXT-Video: standard language modeling loss
   - Instruction tuning format: question-answer pairs

## Dual Memory System Components Requiring Training

### 1. **Memory Attention Mechanisms** (Critical - Must Train)

**Components:**
- `self.working_attention` (MultiheadAttention)
- `self.episodic_attention` (MultiheadAttention)

**Why Training is Needed:**
- These attention modules learn to retrieve relevant information from memory buffers
- They need to learn which memory elements are most relevant for the current query (H_t)
- Without training, attention weights may be random/uninformative

**Current Status:** 
- Initialized randomly (not pre-trained)
- Currently used in inference but not trained

**Training Strategy:**
- **Include in LoRA**: Add to LoRA target modules
- **Alternative**: Train full attention weights (smaller than base model)
- **Recommended**: Use LoRA with `r=64-128` for memory attention modules

### 2. **Memory Fusion MLP** (Critical - Must Train)

**Component:**
- `self.memory_fusion_mlp` (Sequential: Linear → ReLU → Linear → Sigmoid)

**Why Training is Needed:**
- Learns to combine working memory (short-term) and episodic memory (long-term) outputs
- The gate γ_t determines the balance between immediate context and long-term knowledge
- Critical for effective memory integration

**Current Status:**
- Initialized randomly
- Outputs gate values in [0,1] but may not be optimal

**Training Strategy:**
- **Full training** (not LoRA): Small MLP, can afford full parameter training
- **Or LoRA**: If memory constrained, use LoRA with `r=32-64`

### 3. **Salience Gate (Episodic Memory)** (Optional but Recommended)

**Component:**
- `self.episodic_memory.salience_gate` (Sequential: Linear → ReLU → Linear → Sigmoid)

**Why Training is Needed:**
- Learns to identify which frames/features are "salient" enough for long-term storage
- Prevents episodic memory from filling with redundant information
- Important for memory efficiency

**Current Status:**
- Initialized randomly
- Can be disabled (`use_gated_attention=False`)

**Training Strategy:**
- **Full training**: Small MLP, can train fully
- **Or disable**: If not training, set `use_gated_attention=False` in config

### 4. **Memory Buffers** (No Training Needed)

**Components:**
- `WorkingMemory._buffer_list`: FIFO buffer (not trainable)
- `EpisodicMemory._buffer_list`: Similarity-based buffer (not trainable)

**Why No Training:**
- These are data structures, not neural network parameters
- Update mechanisms (FIFO, similarity-based replacement) are deterministic algorithms
- No learnable parameters

## Recommended Training Strategy

### Phase 1: Memory Component Training (Recommended First Step)

**Trainable Parameters:**
1. Memory attention modules (`working_attention`, `episodic_attention`)
2. Memory fusion MLP (`memory_fusion_mlp`)
3. Salience gate (if enabled)

**Frozen Components:**
- Base VLM-3R model (vision tower, language model, spatial tower)
- Existing fusion block and MM projector (keep pre-trained weights)
- All other VLM-3R components

**Training Approach:**
- **LoRA for attention modules**: `lora_r=64-128`, `lora_alpha=128-256`
- **Full training for MLPs**: Memory fusion MLP and salience gate (small, can train fully)
- **Learning rate**: 1e-4 to 2e-5 (lower than base model training)
- **Warmup**: 0.03 ratio

**Data Requirements:**
- Use existing VLM-3R training data (VLM-3R-DATA)
- Focus on long-video sequences (benefit from memory)
- Ensure videos have temporal dependencies requiring memory

### Phase 2: Joint Fine-Tuning (Optional, More Expensive)

**Trainable Parameters:**
- Memory components (from Phase 1)
- MM projector (fine-tune with memory)
- Fusion block (fine-tune with memory)

**Frozen Components:**
- Vision tower
- Language model
- Spatial tower

**Training Approach:**
- Lower learning rate: 5e-6 to 1e-5
- Longer training: 3-5 epochs
- Use gradient accumulation for stability

### Phase 3: End-to-End Fine-Tuning (Advanced, Requires More Resources)

**Trainable Parameters:**
- All memory components
- MM projector
- Fusion block
- Optionally: LoRA on language model for memory-aware generation

**Training Approach:**
- Very low learning rate: 1e-6 to 5e-6
- Careful monitoring to avoid catastrophic forgetting
- Use validation set to track performance

## Datasets and Data Requirements

### Primary Datasets (Same as VLM-3R)

1. **VLM-3R-DATA** (HuggingFace: Journey9ni/VLM-3R-DATA)
   - 200K+ instruction tuning QA pairs
   - Includes VSiBench and VSTiBench training data
   - Spatial reasoning tasks

2. **Video Sources** (Need to download separately):
   - ScanNet videos
   - ScanNet++ videos
   - ARKitScenes videos

### Dataset Considerations for Memory Training

**Important Characteristics:**
- **Long videos**: Memory system benefits from videos with 32+ frames
- **Temporal dependencies**: Questions requiring information from earlier frames
- **Cross-frame reasoning**: Tasks like "count objects throughout video", "track object movement"

**Data Format:**
- Same format as VLM-3R training data
- Video paths: `data/vlm_3r_data/{dataset}/videos/{scene}.mp4`
- QA pairs in YAML format (see `scripts/VLM_3R/vsibench_data.yaml`)

### Additional Datasets (Optional, for Memory-Specific Tasks)

1. **Long-video QA datasets**:
   - Videos with 64+ frames
   - Questions requiring long-term memory

2. **Temporal reasoning benchmarks**:
   - Tasks explicitly testing memory capabilities
   - Cross-temporal object tracking
   - Event sequence understanding

## Implementation Details

### Modifying Training Script

**File**: `scripts/VLM_3R/train_vsibench.sh` or create new `train_dual_memory.sh`

**Key Changes Needed:**

1. **Add LoRA targets for memory modules**:
   ```bash
   # In training script, ensure LoRA targets include:
   # - working_attention
   # - episodic_attention
   # (memory_fusion_mlp can be trained fully or with LoRA)
   ```

2. **Add training flags**:
   ```bash
   --tune_memory_components True
   --memory_lora_r 64
   --memory_lora_alpha 128
   ```

3. **Modify LoRA config** (in `llava/train/train.py`):
   - Add memory attention modules to LoRA target modules
   - Ensure memory fusion MLP is trainable

### Code Modifications Required

**File**: `llava/train/train.py`

1. **Add memory components to trainable parameters**:
   ```python
   # After LoRA setup, ensure memory components are trainable
   if hasattr(model.get_model(), 'working_attention'):
       model.get_model().working_attention.requires_grad_(True)
   if hasattr(model.get_model(), 'episodic_attention'):
       model.get_model().episodic_attention.requires_grad_(True)
   if hasattr(model.get_model(), 'memory_fusion_mlp'):
       model.get_model().memory_fusion_mlp.requires_grad_(True)
   ```

2. **Add memory components to LoRA targets** (if using LoRA):
   ```python
   # In get_lora_module_names or similar function
   lora_target_modules = [
       # ... existing targets ...
       "working_attention",
       "episodic_attention",
       # Optionally: "memory_fusion_mlp" (or train fully)
   ]
   ```

**File**: `llava/model/llava_arch.py`

- No changes needed for training (already integrated)
- Ensure memory components are properly initialized
- Memory buffers are automatically handled (not trainable)

## Training Configuration Recommendations

### Recommended Hyperparameters

**For Phase 1 (Memory Components Only):**
```bash
--learning_rate 1e-4
--lora_r 64
--lora_alpha 128
--num_train_epochs 3-5
--warmup_ratio 0.03
--gradient_accumulation_steps 16
--per_device_train_batch_size 1
--bf16 True
--gradient_checkpointing True
```

**For Phase 2 (Joint Fine-Tuning):**
```bash
--learning_rate 5e-6
--lora_r 128
--lora_alpha 256
--num_train_epochs 5
--warmup_ratio 0.03
--gradient_accumulation_steps 16
```

### Memory Considerations

- **Working Memory**: L_w=8 (fixed, no training needed)
- **Episodic Memory**: L_e=32 (fixed, no training needed)
- **Memory buffers**: Stored on GPU during forward pass, cleared between videos

### Computational Cost

**Phase 1 (Memory Components Only):**
- Small increase: ~5-10% more parameters than base VLM-3R
- Training time: Similar to base VLM-3R (memory components are small)

**Phase 2 (Joint Fine-Tuning):**
- Moderate increase: Includes MM projector and fusion block
- Training time: 1.2-1.5x base training time

## Evaluation Strategy

### Metrics to Track

1. **Memory Utilization**:
   - Working memory fill rate
   - Episodic memory fill rate
   - Memory retrieval effectiveness

2. **Task Performance**:
   - VSiBench accuracy
   - VSTiBench accuracy
   - Long-video QA performance

3. **Ablation Studies**:
   - With/without memory
   - With/without salience gate
   - Different memory sizes (L_w, L_e)

### Validation Set

- Use VSTiBench test set
- Long-video subsets
- Memory-dependent tasks

## Potential Challenges and Solutions

### Challenge 1: Memory Components Not Learning

**Symptoms**: Memory attention outputs similar to inputs, fusion gate not varying

**Solutions**:
- Increase learning rate for memory components
- Use separate learning rates (higher for memory, lower for base model)
- Add explicit loss terms encouraging memory usage

### Challenge 2: Catastrophic Forgetting

**Symptoms**: Base model performance degrades after memory training

**Solutions**:
- Use LoRA for memory components (isolates changes)
- Lower learning rates
- Include base model tasks in training data

### Challenge 3: Memory Not Being Utilized

**Symptoms**: Episodic memory stays empty, working memory underutilized

**Solutions**:
- Ensure training data has long videos
- Add tasks explicitly requiring memory
- Monitor memory usage during training

## Next Steps

1. **Implement training code modifications**:
   - Add memory components to trainable parameters
   - Update LoRA configuration
   - Add training flags

2. **Prepare training data**:
   - Ensure long-video sequences are included
   - Verify data format compatibility

3. **Run Phase 1 training**:
   - Start with memory components only
   - Monitor memory utilization and task performance

4. **Evaluate and iterate**:
   - Compare with/without memory
   - Tune hyperparameters
   - Proceed to Phase 2 if Phase 1 successful

## References

- VLM-3R Paper: https://arxiv.org/abs/2505.20279
- VLM² Paper: https://arxiv.org/pdf/2511.20644
- VLM-3R Training Script: `scripts/VLM_3R/train_vsibench.sh`
- VLM-3R Data: https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA


