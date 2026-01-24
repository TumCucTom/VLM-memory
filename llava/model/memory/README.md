# Dual-Memory System Implementation

This directory contains the implementation of the dual-memory system from VLM² (Algorithm 1).

## Files

- `working_memory.py`: Working Memory with FIFO mechanism (Algorithm 1 Lines 8-14)
- `episodic_memory.py`: Episodic Memory with similarity-based replacement (Algorithm 1 Lines 15-24)
- `dual_memory.py`: Complete Dual-Memory Module implementing Algorithm 1
- `__init__.py`: Module exports

## Usage

```python
from llava.model.memory import DualMemoryModule

# Initialize with recommended sizes (L_w=8, L_e=32)
dual_memory = DualMemoryModule(
    L_w=8,           # Working memory capacity
    L_e=32,          # Episodic memory capacity
    feature_dim=1152 # Feature dimension
)

# For each frame t:
# H_t is the current input features [B, N, D]
M_t, W_t_1, E_t_1 = dual_memory(H_t)

# M_t is the fused memory output to use for language model
# W_t_1 and E_t_1 are updated memory buffers
```

## Algorithm 1 Implementation

The `DualMemoryModule.forward()` method implements Algorithm 1 exactly:

1. **Line 1**: `M_t^w ← Working Attention(Q = H_t, KV = W_t)`
2. **Line 2**: `M_t^e ← Episodic Attention(Q = H_t, KV = E_t)`
3. **Line 5**: `γ_t ← MLP([M_t^w; M_t^e])`
4. **Lines 6-7**: `M_t ← γ_t ⊙ M_t^w + (1-γ_t) ⊙ M_t^e`
5. **Lines 8-14**: Update Working Memory (FIFO)
6. **Lines 15-24**: Update Episodic Memory (similarity-based replacement)

## Memory Sizes

Recommended values from ablation study (Table 7):
- **L_w = 8**: Working memory window size
- **L_e = 32**: Episodic memory capacity
- **Performance**: Avg. 68.8 on VSI-Bench

## Notes

- Memories are cleared at video boundaries
- Each video should have its own memory instance
- Memory buffers are stored as lists of tensors internally
- The module handles different input tensor shapes automatically

