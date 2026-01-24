# VLM² Implementation Todo List

This is a comprehensive todo list for adapting VLM-3R to VLM² architecture based on the adaptation guide.

## Part 1: Enhanced Semantic-Geometric Fusion

### 1.1 Viewpoint-Aware Geometry Alignment Module

- [ ] **Create new file**: `llava/model/multimodal_fusion_block/viewpoint_aware_fusion.py`
- [ ] **Implement ViewpointAwareAlignment class**:
  - [ ] View token encoding network (from camera tokens)
  - [ ] Geometry-view alignment attention mechanism
  - [ ] Feature modulation layers
  - [ ] Forward pass: encode view → align geometry → return view-aligned features
- [ ] **Modify**: `llava/model/multimodal_fusion_block/builder.py`
  - [ ] Add viewpoint alignment as preprocessing step
  - [ ] Insert between spatial encoder output and fusion block input
- [ ] **Test**: Viewpoint alignment with different camera poses

### 1.2 3D Coordinate Injection with Learnable Gating

- [ ] **Create new file**: `llava/model/multimodal_fusion_block/coordinate_injection.py`
- [ ] **Implement CoordinateInjection class**:
  - [ ] 3D coordinate encoder (MLP: 3 → d_visual)
  - [ ] Learnable gating mechanism (sigmoid/softmax gate)
  - [ ] Coordinate injection into visual tokens
  - [ ] Forward pass: encode coords → compute gates → inject into visual tokens
- [ ] **Modify**: `llava/model/llava_arch.py` in `encode_images()` method
  - [ ] Extract 3D points from spatial encoder (`pts3d_in_other_view` from CUT3R)
  - [ ] Reshape points to match visual token spatial dimensions
  - [ ] Apply coordinate injection after visual feature extraction, before fusion
- [ ] **Test**: Coordinate injection with various point configurations

### 1.3 Enhanced Cross-Attention Fusion

- [ ] **Modify**: `llava/model/multimodal_fusion_block/cross_attention_mlp.py`
  - [ ] Add viewpoint-aware geometry features as input
  - [ ] Incorporate coordinate-injected visual tokens
  - [ ] Ensure view-consistent fusion
  - [ ] Update input signature: `(visual_tokens_with_coords, aligned_geometry_tokens)`
- [ ] **Test**: Enhanced fusion with aligned geometry and coordinate-injected tokens

---

## Part 2: Dual-Memory System

### 2.1 Working Memory (Sliding Window)

- [x] **Create new file**: `llava/model/memory/working_memory.py`
- [x] **Implement WorkingMemory class**:
  - [x] Fixed-size buffer with capacity `L_w=8` (recommended from ablation study)
  - [x] FIFO queue mechanism (remove oldest when full)
  - [x] `get_buffer()` method to retrieve current buffer
  - [x] `clear()` method to reset buffer
  - [x] `update(H_t)` method: add if not full, else remove oldest then add (Algorithm 1 Lines 8-14)
- [ ] **Test**: FIFO buffer management with capacity limits
- [ ] **Test**: Buffer update mechanism with various frame sequences

### 2.2 Episodic Memory (Long-term Storage)

- [x] **Create new file**: `llava/model/memory/episodic_memory.py`
- [x] **Implement EpisodicMemory class**:
  - [x] Memory bank with fixed capacity `L_e=32` (recommended from ablation study)
  - [x] `get_buffer()` method to retrieve current buffer
  - [x] `clear()` method to reset buffer
  - [x] `_compute_similarity(M_t, E_i)` method: cosine similarity computation
  - [x] `update(M_t)` method: similarity-based replacement (Algorithm 1 Lines 15-24)
    - [x] If not full: add M_t
    - [x] If full: find most similar, delete it, add M_t
- [ ] **Test**: Similarity-based replacement mechanism
- [ ] **Test**: Memory consolidation with various feature patterns

### 2.3 Dual-Memory Module (Main Integration)

- [x] **Create new file**: `llava/model/memory/dual_memory.py`
- [x] **Implement DualMemoryModule class** (Algorithm 1):
  - [x] Initialize with recommended sizes: `L_w=8, L_e=32`
  - [x] Initialize `WorkingMemory` and `EpisodicMemory` instances
  - [x] Working Attention: `nn.MultiheadAttention` for retrieval (Algorithm 1 Line 1)
  - [x] Episodic Attention: `nn.MultiheadAttention` for retrieval (Algorithm 1 Line 2)
  - [x] Gated fusion MLP: `γ_t ← MLP([M_t^w; M_t^e])` (Algorithm 1 Line 5)
  - [x] Forward pass implementing Algorithm 1:
    - [x] Line 1: `M_t^w ← Working Attention(Q=H_t, KV=W_t)`
    - [x] Line 2: `M_t^e ← Episodic Attention(Q=H_t, KV=E_t)`
    - [x] Line 5: `γ_t ← MLP([M_t^w; M_t^e])`
    - [x] Lines 6-7: `M_t ← γ_t ⊙ M_t^w + (1-γ_t) ⊙ M_t^e`
    - [x] Lines 8-14: Update Working Memory
    - [x] Lines 15-24: Update Episodic Memory
  - [x] Return: `(M_t, W_{t+1}, E_{t+1})`
- [ ] **Test**: Complete forward pass with Algorithm 1
- [ ] **Test**: Attention-based retrieval mechanisms
- [ ] **Test**: Gated fusion with various memory states

### 2.4 Memory Integration into Model Architecture

- [x] **Modify**: `llava/model/llava_arch.py`
  - [x] **In `LlavaMetaModel.__init__`**:
    - [x] Initialize `DualMemoryModule` if `config.use_dual_memory` is True
    - [x] Initialize `_video_memories` dict to store memory states per video
  - [x] **In `LlavaMetaForCausalLM.encode_images`**:
    - [x] Add `is_video` and `video_id` parameters
    - [x] Call `_apply_memory_to_features()` after fusion block, before `mm_projector`
  - [x] **Create `_apply_memory_to_features()` method**:
    - [x] Handle sequential frame processing
    - [x] Retrieve/update `W_t` and `E_t` from `_video_memories` for given `video_id`
    - [x] Iterate through frames, call `dual_memory.forward(H_t, W_t, E_t)`
    - [x] Update memory state in `_video_memories`
    - [x] Return memory-enhanced features
  - [x] **Create `clear_video_memory()` method**:
    - [x] Clear memory state for specific `video_id` or all videos
  - [x] **Modify `prepare_inputs_labels_for_multimodal`**:
    - [x] Pass `is_video` and `video_id` to `encode_images`
    - [x] Call `clear_video_memory(video_id)` at start of new video sequence
- [ ] **Test**: Sequential frame processing following Algorithm 1
- [ ] **Test**: Memory persistence across frames within same video
- [ ] **Test**: Memory reset at video boundaries
- [ ] **Test**: Batch processing with multiple videos (separate memory per video)

### 2.5 Memory Module Package

- [x] **Create**: `llava/model/memory/__init__.py`
  - [x] Export `WorkingMemory`, `EpisodicMemory`, `DualMemoryModule`
- [ ] **Create**: `llava/model/memory/memory_utils.py` (optional helper functions)
  - [ ] Helper functions for memory operations if needed

---

## Part 3: Architecture Modifications

### 3.1 Model Builder Updates

- [ ] **Modify**: `llava/model/builder.py`
  - [ ] Register memory modules
  - [ ] Add memory initialization logic
  - [ ] Handle memory config parameters

### 3.2 Configuration Updates

- [x] **Add memory configuration parameters**:
  - [x] `use_dual_memory`: bool (default: False)
  - [x] `memory_L_w`: int (default: 8, recommended from ablation study)
  - [x] `memory_L_e`: int (default: 32, recommended from ablation study)
  - [x] `memory_feature_dim`: int (should match fusion output dim, default: 1152)
  - [x] `memory_num_heads`: int (default: 8, for attention mechanisms)
  - [x] `memory_dropout`: float (default: 0.1)
- [ ] **Add fusion configuration parameters**:
  - [ ] `use_viewpoint_alignment`: bool (default: True)
  - [ ] `use_coordinate_injection`: bool (default: True)
  - [ ] `coordinate_gate_type`: str (default: "sigmoid")

---

## Part 4: Training Integration

### 4.1 Training Script Updates

- [x] **Modify**: `llava/train/train.py`
  - [x] **Add to `ModelArguments`**:
    - [x] `use_dual_memory`: bool field
    - [x] `memory_L_w`: int field
    - [x] `memory_L_e`: int field
    - [x] `memory_num_heads`: int field
    - [x] `memory_dropout`: float field
    - [x] `tune_dual_memory`: bool field
  - [x] **In model initialization**:
    - [x] Pass memory config to model
    - [x] Move memory module to correct device/dtype
  - [x] **In trainability setup**:
    - [x] Set memory parameters to trainable if `tune_dual_memory=True`
    - [x] Support `mm_tunable_parts` with "dual_memory"
  - [x] **Handle memory state during training**:
    - [x] Ensure memory resets at video boundaries
    - [x] Handle memory state in data collator if needed
- [ ] **Test**: Memory module trainability
- [ ] **Test**: Gradient flow through memory modules

### 4.2 Training Strategy

- [x] **Decide on training approach**:
  - [x] Option A: Reset memory for each training sample (IMPLEMENTED)
  - [ ] Option B: Maintain memory across batch (if same video)
  - [ ] Option C: Use teacher forcing with ground truth memory states
- [ ] **Consider memory-specific losses** (optional):
  - [ ] Memory consistency loss
  - [ ] Memory utilization loss
- [ ] **Verify gradient flow**:
  - [ ] Check gradients flow through memory modules
  - [ ] Ensure memory parameters are trainable
  - [ ] Test backward pass

---

## Part 5: Testing and Validation

### 5.1 Unit Tests

- [ ] **Test viewpoint alignment**:
  - [ ] Different camera poses
  - [ ] View-dependent feature modulation
- [ ] **Test coordinate injection**:
  - [ ] Various point configurations
  - [ ] Gating mechanism
  - [ ] Coordinate encoding
- [ ] **Test working memory**:
  - [ ] Buffer capacity limits
  - [ ] FIFO update mechanism
  - [ ] Buffer retrieval
- [ ] **Test episodic memory**:
  - [ ] Similarity computation
  - [ ] Similarity-based replacement
  - [ ] Memory consolidation
- [ ] **Test dual-memory module**:
  - [ ] Attention-based retrieval
  - [ ] Gated fusion
  - [ ] Complete Algorithm 1 forward pass

### 5.2 Integration Tests

- [ ] **Test full pipeline**:
  - [ ] Single video processing
  - [ ] Memory persistence across frames
  - [ ] Memory reset at boundaries
- [ ] **Test with different video lengths**:
  - [ ] Short videos (< L_w frames)
  - [ ] Medium videos (L_w to L_e frames)
  - [ ] Long videos (> L_e frames)
- [ ] **Test batch processing**:
  - [ ] Multiple videos in batch
  - [ ] Separate memory per video
  - [ ] Memory state isolation

### 5.3 Validation

- [ ] **Compare outputs**:
  - [ ] With/without memory
  - [ ] With/without viewpoint alignment
  - [ ] With/without coordinate injection
- [ ] **Verify memory utilization**:
  - [ ] Are memories being used?
  - [ ] Memory retrieval statistics
  - [ ] Memory update frequency
- [ ] **Check for issues**:
  - [ ] Memory leaks or unbounded growth
  - [ ] Gradient flow problems
  - [ ] Performance degradation
- [ ] **Validate on datasets**:
  - [ ] VSI-Bench
  - [ ] VSTI-Bench
  - [ ] Compare to VLM² paper results

---

## Part 6: Documentation

### 6.1 Code Documentation

- [ ] **Add docstrings** to all new classes and methods
- [ ] **Document Algorithm 1** implementation in code comments
- [ ] **Document memory state management**
- [ ] **Document configuration parameters**

### 6.2 Usage Documentation

- [ ] **Create training guide**:
  - [ ] How to enable memory
  - [ ] Configuration examples
  - [ ] Training command examples
- [ ] **Create API documentation**:
  - [ ] Memory module API
  - [ ] Integration points
  - [ ] Configuration options

---

## Part 7: Optimization and Refinement

### 7.1 Performance Optimization

- [ ] **Optimize attention mechanisms**:
  - [ ] Use efficient attention (Flash Attention if available)
  - [ ] Consider sparse memory updates
- [ ] **Optimize memory operations**:
  - [ ] Efficient similarity computation
  - [ ] Batch memory operations if possible
- [ ] **Profile and benchmark**:
  - [ ] Memory overhead
  - [ ] Training speed impact
  - [ ] Inference speed impact

### 7.2 Ablation Studies

- [ ] **Test different memory sizes**:
  - [ ] L_w variations (4, 8, 16)
  - [ ] L_e variations (16, 32, 64)
  - [ ] Compare to recommended L_w=8, L_e=32
- [ ] **Test component combinations**:
  - [ ] Only viewpoint alignment
  - [ ] Only coordinate injection
  - [ ] Only working memory
  - [ ] Only episodic memory
  - [ ] Full system
- [ ] **Compare training strategies**:
  - [ ] Memory-only training
  - [ ] Memory + projector training
  - [ ] Memory + fusion + projector training

---

## Part 8: Known Issues and Challenges

### 8.1 Memory State Management

- [x] **Handle memory across different videos in batch**:
  - [x] Maintain separate memory instances per video
  - [x] Reset at video boundaries
  - [ ] Handle edge cases (empty videos, single frame videos)

### 8.2 Computational Cost

- [ ] **Monitor memory overhead**:
  - [ ] Memory usage during training
  - [ ] Memory usage during inference
  - [ ] Compare to baseline VLM-3R
- [ ] **Optimize if needed**:
  - [ ] Limit memory capacity if too expensive
  - [ ] Consider sparse updates
  - [ ] Use efficient attention

### 8.3 Gradient Flow

- [ ] **Verify gradients**:
  - [ ] Flow through memory modules
  - [ ] Flow through attention mechanisms
  - [ ] Flow through similarity computation
- [ ] **Fix if needed**:
  - [ ] Add residual connections if needed
  - [ ] Ensure differentiability
  - [ ] Consider straight-through estimators if needed

### 8.4 Point Coordinate Extraction

- [ ] **Align points with visual tokens**:
  - [ ] Spatial alignment (resize/interpolate)
  - [ ] Patch-level aggregation
  - [ ] Attention-based matching
- [ ] **Handle edge cases**:
  - [ ] Missing points
  - [ ] Invalid coordinates
  - [ ] Dimension mismatches

---

## Part 9: Future Enhancements (Optional)

### 9.1 Advanced Features

- [ ] **Dynamic memory capacity** (if needed)
- [ ] **Memory importance scoring** (if needed)
- [ ] **Multi-scale memory** (if needed)
- [ ] **Memory compression** (if needed)

### 9.2 Additional Improvements

- [ ] **Memory visualization tools**
- [ ] **Memory debugging utilities**
- [ ] **Memory analysis scripts**

---

## Summary Checklist

### Critical Path (Must Complete)

- [ ] Part 1: Enhanced Fusion (viewpoint alignment + coordinate injection)
- [x] Part 2: Dual-Memory System (working + episodic + dual module)
- [x] Part 2.4: Memory Integration into model architecture
- [x] Part 4.1: Training script updates
- [ ] Part 5.2: Integration tests

### Important (Should Complete)

- [ ] Part 3: Architecture modifications
- [ ] Part 4.2: Training strategy
- [ ] Part 5.1: Unit tests
- [ ] Part 5.3: Validation

### Nice to Have (Optional)

- [ ] Part 6: Documentation
- [ ] Part 7: Optimization
- [ ] Part 8: Issue resolution
- [ ] Part 9: Future enhancements

---

## Notes

- **Recommended memory sizes**: L_w=8, L_e=32 (from ablation study, Table 7)
- **Performance target**: Avg. 68.8 on VSI-Bench (Numerical: 68.2, Multiple-Choice: 69.4)
- **Feature dimension**: Should match fusion output dim (typically 1152)
- **Algorithm 1**: Must be followed exactly for dual-memory module

