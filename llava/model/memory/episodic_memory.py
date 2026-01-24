"""
Episodic Memory Module for VLM²
Implements similarity-based replacement mechanism as per Algorithm 1 (Lines 15-24)
"""
import torch
import torch.nn as nn
from typing import List, Optional
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """
    Episodic Memory with similarity-based replacement (Algorithm 1 Lines 15-24)
    
    Maintains long-term memory with capacity L_e.
    When full, replaces the most similar element (cosine similarity).
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
        self.register_buffer('_buffer_size', torch.tensor(0))
    
    def get_buffer(self) -> List[torch.Tensor]:
        """Get current episodic memory buffer"""
        if not hasattr(self, '_buffer_list'):
            self._buffer_list = []
        return self._buffer_list
    
    def clear(self):
        """Clear the episodic memory buffer"""
        if hasattr(self, '_buffer_list'):
            self._buffer_list = []
        self._buffer_size = torch.tensor(0)
    
    def compute_similarity(self, M_t: torch.Tensor, E_i: torch.Tensor) -> float:
        """
        Compute cosine similarity between M_t and E_i (Algorithm 1 Line 18)
        
        Args:
            M_t: Current fused memory [*feature_shape]
            E_i: Memory element [*feature_shape]
        
        Returns:
            Cosine similarity score
        """
        # Flatten to 1D for similarity computation
        M_t_flat = M_t.flatten()
        E_i_flat = E_i.flatten()
        
        # Compute cosine similarity
        # s_i = (M_t · E_i) / (||M_t|| * ||E_i||)
        dot_product = torch.dot(M_t_flat, E_i_flat)
        norm_M = torch.norm(M_t_flat)
        norm_E = torch.norm(E_i_flat)
        
        if norm_M == 0 or norm_E == 0:
            return 0.0
        
        similarity = dot_product / (norm_M * norm_E)
        return similarity.item()
    
    def update(self, M_t: torch.Tensor) -> List[torch.Tensor]:
        """
        Update Episodic Memory (Algorithm 1 Lines 15-24)
        
        Args:
            M_t: Fused memory output from gated fusion [*feature_shape]
        
        Returns:
            Updated episodic memory buffer E_{t+1}
        """
        buffer = self.get_buffer()
        
        # Ensure M_t is a tensor
        if not isinstance(M_t, torch.Tensor):
            M_t = torch.tensor(M_t)
        
        # Store a copy to avoid reference issues
        M_t = M_t.detach().clone()
        
        # Algorithm 1 Line 15: if |E_t| < L_e then
        if len(buffer) < self.L_e:
            # Algorithm 1 Line 16: E_{t+1} ← E_t ∪ {M_t}
            buffer.append(M_t)
        else:
            # Algorithm 1 Lines 17-23: Replace most similar element
            # Line 18: Compute cosine similarity for each element
            similarities = []
            for E_i in buffer:
                s_i = self.compute_similarity(M_t, E_i)
                similarities.append(s_i)
            
            # Line 19: Find index of maximum similarity
            # i_t^* ← arg max_{i∈{1,...,L_e}} s_i
            max_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            
            # Line 20-22: Delete most similar element and add M_t
            del buffer[max_idx]
            buffer.append(M_t)
        
        self._buffer_size = torch.tensor(len(buffer))
        return buffer
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert buffer list to tensor for attention operations
        
        Args:
            device: Target device for tensor
        
        Returns:
            Tensor of shape [L_e, *feature_shape] or [|E_t|, *feature_shape]
                Padded with zeros if buffer not full
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

