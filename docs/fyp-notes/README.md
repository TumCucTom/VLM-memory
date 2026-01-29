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

Testing:
- Run inference using demo script: `bash scripts/video/demo/video_demo.slurm`
- Memory clearing happens automatically at video boundaries