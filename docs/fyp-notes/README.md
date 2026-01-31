# Running demo

Created a slurm script at /scripts/video/demo/video_demo.slurm to run the demo script
- This needs larger GPU then just 2080ti - using A100
- Setup and load python env
- Have to use scratch work space
- Running into tokenizer vocab mismatch error
    - Made change to Builder.py 
    - Use actual size rather than hardcoded 152064

- `CUT3R/src/croco/models/pos_embed.py` - Fixed CUDA indexing error in RoPE2D (position clamping)
- `llava/model/language_model/llava_qwen.py` - Fixed MultiheadAttention initialisation

# Added working memory
Run inference by using the demo script to do inital test

Created scripts: 
- `llava/model/memory/working_memory.py` - FIFO working memory module (L_w=8 capacity)

Updated scripts: 
- `llava/model/llava_arch.py` - Integrated working memory with attention retrieval and per-frame updates
- `playground/demo/video_demo.py` - Added memory clearing at start of each video
- `llava/model/language_model/llava_qwen.py` - added debug prints
- `llava/model/builder.py` - Added debug prints for model loading

# Added episodic memory
Integrated episodic memory according to VLM² paper (Algorithm 1 Lines 15-24)

Created scripts:
- `llava/model/memory/episodic_memory.py` - Episodic memory module with similarity-based replacement (L_e=32 capacity)
  - Implements cosine similarity-based replacement when memory is full
  - Optional gated attention mechanism to filter salient information (enabled by default)
  - Stores critical long-term information for spatial reasoning

Updated scripts:
- `llava/model/llava_arch.py` - Added episodic memory initialization and attention mechanism
- `playground/demo/video_demo.py` - Updated to clear both working and episodic memory at video boundaries

Key features:
- Capacity: L_e=32 (recommended from paper ablation study)
- Update mechanism: Similarity-based replacement (replaces most similar element when full)
- Salience filtering: Optional gated attention to identify critical information
- Storage: Uses fused memory representation M_t (not raw H_t) per Algorithm 1

# Dual memory system
Implemented complete dual-memory module following VLM² Algorithm 1

Architecture:
- **Working Memory** (L_w=8): FIFO sliding window for immediate context
- **Episodic Memory** (L_e=32): Similarity-based replacement for long-term critical information
- **Gated Memory Fusion**: Learned gate γ_t combines working and episodic memory outputs

Algorithm 1 Implementation:
1. **Line 1**: M_t^w ← Working Attention(Q = H_t, KV = W_t)
2. **Line 2**: M_t^e ← Episodic Attention(Q = H_t, KV = E_t)
3. **Lines 5-7**: Gated Memory Fusion
   - γ_t = σ(MLP(Concat[M_t^w; M_t^e]))
   - M_t = γ_t ⊙ M_t^w + (1-γ_t) ⊙ M_t^e
4. **Lines 8-14**: Update Working Memory (FIFO) with H_t
5. **Lines 15-24**: Update Episodic Memory (similarity-based) with M_t

Updated scripts:
- `llava/model/llava_arch.py` - Replaced `_apply_working_memory` with `_apply_dual_memory` implementing full Algorithm 1
  - Memory fusion MLP for gated combination
  - Per-frame processing for video sequences
  - Correct update mechanisms: Working memory uses H_t, Episodic memory uses M_t

## Implementation Choices (Not Specified in Paper):

The VLM² paper specifies the mathematical formulation but does not provide exact MLP architectures. Our implementation uses:

**What Makes It "Gated"?**
The "gated" terminology refers to the MLP's **output acting as a gate** that controls information flow, not internal gates within the MLP itself. The MLP is a standard feedforward network that outputs gate values (via sigmoid activation), which are then used to:
- **Weight information** (memory fusion): Gate value γ_t controls the weighted combination of working and episodic memory
- **Filter information** (salience gate): Gate value determines whether information passes through to episodic memory

This is different from "gated architectures" like GRU/LSTM which have internal gates (forget gate, input gate, etc.).

1. **Memory Fusion MLP** (`llava/model/llava_arch.py`, lines 97-102):
   - Architecture: 2-layer MLP with ReLU activation
   - Input: Concatenated [M_t^w; M_t^e] → [2*D]
   - Hidden dimension: `config.hidden_size` (default) or `memory_fusion_hidden_dim` if specified
   - Output: Gate values γ_t in [0, 1] via Sigmoid
   - Structure: Linear(2*D → hidden_dim) → ReLU → Linear(hidden_dim → D) → Sigmoid
   - **How it gates**: The output γ_t is used as a weight: `M_t = γ_t ⊙ M_t^w + (1-γ_t) ⊙ M_t^e` (line 312 in code)
   - Matches paper equation: γ_t = σ(MLP(Concat[M_t^w; M_t^e]))

```
self.memory_fusion_mlp = nn.Sequential(
    nn.Linear(config.hidden_size * 2, fusion_hidden_dim),  # Input: [M_t^w; M_t^e]
    nn.ReLU(),
    nn.Linear(fusion_hidden_dim, config.hidden_size),  # Output: gate values
    nn.Sigmoid()  # Gate in [0, 1]
)
```

2. **Salience Gate MLP** (`llava/model/memory/episodic_memory.py`, lines 41-46):
   - Architecture: 2-layer MLP with ReLU activation
   - Input: Feature vector H_t → [D]
   - Hidden dimension: `feature_dim // 2`
   - Output: Single scalar salience score in [0, 1] via Sigmoid
   - Structure: Linear(D → D/2) → ReLU → Linear(D/2 → 1) → Sigmoid
   - **How it gates**: The output salience score is compared to a threshold (default 0.5); if below threshold, information is filtered out (not stored in episodic memory) - see lines 152-155 in code
   - Purpose: Filters which information is "critical" enough for episodic memory storage
   - Can be disabled via `use_gated_attention=False` in config

```
self.salience_gate = nn.Sequential(
    nn.Linear(feature_dim, feature_dim // 2),
    nn.ReLU(),
    nn.Linear(feature_dim // 2, 1),
    nn.Sigmoid()  # Output: salience score in [0, 1]
)
```

These choices follow common MLP design patterns and are consistent with the paper's mathematical formulation.

Testing:
- Run inference using demo script: `bash scripts/video/demo/video_demo.slurm`
- Memory clearing happens automatically at video boundaries

- Remove salience gate and increased number of sampled franes to checked episodic memory replacing

# Training

See the guide [here](../vlm2-conversion/TRAINING_DUAL_MEMORY.md)

## Phase 1 Training Setup

Prepared the system for Phase 1 memory component training (memory components only, with salience-gated fusion).

**Created scripts:**
- `scripts/VLM_Dual_Mem/train_memory_phase1.sh` - Phase 1 training script based on `train_vsibench.sh`
  - Trains only memory components (attention modules + MLPs)
  - Freezes base VLM-3R (vision tower, language model, spatial tower)
  - Freezes fusion block and MM projector (keeps pre-trained weights)
  - Learning rate: 1e-4 (per Phase 1 recommendations)
  - Salience gate enabled by default

**Updated scripts:**
- `llava/train/train.py` - Added memory component training support:
  - Added `tune_memory_components` flag to `ModelArguments`
  - Modified `find_all_linear_names()` to exclude memory components from LoRA targets (trained fully)
  - Added Phase 1 training logic:
    - When `tune_memory_components=True`, freezes entire model then makes memory components trainable:
      - `working_attention`: Full training (small MultiheadAttention module)
      - `episodic_attention`: Full training (small MultiheadAttention module)
      - `memory_fusion_mlp`: Full training (small MLP)
      - `salience_gate`: Full training (if enabled, small MLP)
    - Ensures fusion block and MM projector remain frozen
    - Prints debug messages showing trainable/frozen components

**Training configuration:**
- **Trainable**: Memory components only (attention modules + MLPs)
- **Frozen**: Vision tower, language model, spatial tower, fusion block, MM projector
- **Learning rate**: 1e-4 (higher than base model training, per Phase 1 recommendations)
- **Epochs**: 5
- **LoRA**: Applied to language model only (r=64, alpha=128)
- **Salience gate**: Enabled by default

**Usage:**
```bash
bash scripts/VLM_Dual_Mem/train_memory_phase1.sh
```

The system is now ready for Phase 1 training with salience-gated fusion enabled.

