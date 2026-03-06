"""
Episodic Memory Module for VLM²
Implements similarity-based replacement mechanism as per Algorithm 1 (Lines 15-24)

Modified: removed salience gate - stores all frames without filtering
"""
import torch
import torch.nn as nn
from typing import List, Optional


class EpisodicMemory(nn.Module):
    """
    Episodic Memory with similarity-based replacement
    
    According to VLM² paper (https://arxiv.org/pdf/2511.20644):
    - "episodic memory that consolidates and stores critical long-term information"
    - When full, replaces the most similar existing memory element (similarity-based replacement)
    
    Modified: Removed gated attention for salience filtering - stores all frames equally
    """
    
    def __init__(self, L_e: int = 32, feature_dim: int = 1152):
        """
        Args:
            L_e: Maximum capacity of episodic memory
                Recommended: 32 (from ablation study, Table 7)
            feature_dim: Dimension of features stored in memory
        """
        super().__init__()
        self.L_e = L_e
        self.feature_dim = feature_dim
        
        # Buffer to store episodic memory elements
        self.register_buffer(_buffer_size, torch.tensor(0))
    
    def get_buffer(self) -> List[torch.Tensor]:
        """Get current episodic memory buffer"""
        if not hasattr(self, _buffer_list):
            self._buffer_list = []
        return self._buffer_list
    
    def clear(self):
        """Clear the episodic memory buffer"""
        if hasattr(self, _buffer_list):
            self._buffer_list = []
        # In-place so _buffer_size stays on same device as module (avoids CPU tensor when model is on GPU)
        self._buffer_size.zero_()
    
    def _compute_similarity(self, H_t: torch.Tensor, memory_elem: torch.Tensor) -> float:
        """
        Compute cosine similarity between H_t and a memory element
        
        Args:
            H_t: Current input features [D] or [N, D]
            memory_elem: Memory element [D] or [N, D]
        
        Returns:
            Similarity score (scalar)
        """
        # Flatten to 1D if needed
        if H_t.dim() > 1:
            H_t = H_t.flatten()
        if memory_elem.dim() > 1:
            memory_elem = memory_elem.flatten()
        
        # Ensure same device
        memory_elem = memory_elem.to(H_t.device)
        
        # Compute cosine similarity
        H_t_norm = H_t / (torch.norm(H_t) + 1e-8)
        mem_norm = memory_elem / (torch.norm(memory_elem) + 1e-8)
        similarity = torch.dot(H_t_norm, mem_norm).item()
        
        return similarity
    
    def update(self, H_t: torch.Tensor) -> List[torch.Tensor]:
        """
        Update Episodic Memory (Algorithm 1 Lines 15-24)
        
        Modified: Removed salience gate - stores all frames without filtering
        
        Args:
            H_t: Current input features [B, N, D] or [N, D] or [D]
                where D is feature_dim
        
        Returns:
            Updated episodic memory buffer E_{t+1}
        """
        buffer = self.get_buffer()
        
        # Ensure H_t is a tensor and handle different input shapes
        if not isinstance(H_t, torch.Tensor):
            H_t = torch.tensor(H_t)
        
        # Store a copy for storage
        H_t_storage = H_t.detach().clone()
        
        # Algorithm 1 Line 15: if |E_t| < L_e then
        if len(buffer) < self.L_e:
            # Algorithm 1 Line 16: E_{t+1} ← E_t ∪ {M_t}
            buffer.append(H_t_storage)
        else:
            # Algorithm 1 Lines 17-23: Find most similar element and replace
            similarities = []
            for mem_elem in buffer:
                sim = self._compute_similarity(H_t_storage, mem_elem)
                similarities.append(sim)
            
            # Algorithm 1 Line 18: j* ← argmax_{j} similarity(H_t, E_t[j])
            max_sim_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            
            # Algorithm 1 Line 19: E_{t+1}[j*] ← H_t
            buffer[max_sim_idx] = H_t_storage
        
        # In-place so _buffer_size stays on same device as module
        self._buffer_size.fill_(len(buffer))
        return buffer
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert buffer list to tensor for attention operations
        
        Args:
            device: Target device for tensor
        
        Returns:
            Tensor of shape [L_e, *feature_shape] or [|E_t|, *feature_shape]
        """
        buffer = self.get_buffer()
        
        if len(buffer) == 0:
            # Return empty tensor with correct shape
            return torch.zeros(0, self.feature_dim, device=device)
        
        # Stack all elements
        stacked = torch.stack(buffer, dim=0)  # [|E_t|, *feature_shape]
        
        if device is not None:
            stacked = stacked.to(device)
        
        return stacked
    
    def __len__(self) -> int:
        """Return current size of episodic memory"""
        return len(self.get_buffer())
    
    def is_full(self) -> bool:
        """Check if episodic memory is at capacity"""
        return len(self.get_buffer()) >= self.L_e
