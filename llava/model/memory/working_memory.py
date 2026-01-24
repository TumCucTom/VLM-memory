"""
Working Memory Module for VLM²
Implements FIFO sliding window mechanism as per Algorithm 1 (Lines 8-14)
"""
import torch
import torch.nn as nn
from typing import List, Optional


class WorkingMemory(nn.Module):
    """
    Working Memory with FIFO mechanism (Algorithm 1 Lines 8-14)
    
    Maintains a sliding window of recent frames with capacity L_w.
    When full, removes oldest element before adding new one (FIFO).
    """
    
    def __init__(self, L_w: int = 8, feature_dim: int = 1152):
        """
        Args:
            L_w: Maximum capacity of working memory (window size)
                Recommended: 8 (from ablation study, Table 7)
            feature_dim: Dimension of features stored in memory
        """
        super().__init__()
        self.L_w = L_w
        self.feature_dim = feature_dim
        
        # Buffer to store working memory elements
        # Will be initialized as empty list, converted to tensor when needed
        self.register_buffer('_buffer_size', torch.tensor(0))
    
    def get_buffer(self) -> List[torch.Tensor]:
        """Get current working memory buffer"""
        if not hasattr(self, '_buffer_list'):
            self._buffer_list = []
        return self._buffer_list
    
    def clear(self):
        """Clear the working memory buffer"""
        if hasattr(self, '_buffer_list'):
            self._buffer_list = []
        self._buffer_size = torch.tensor(0)
    
    def update(self, H_t: torch.Tensor) -> List[torch.Tensor]:
        """
        Update Working Memory (Algorithm 1 Lines 8-14)
        
        Args:
            H_t: Current input features [B, N, D] or [N, D] or [D]
                where D is feature_dim
        
        Returns:
            Updated working memory buffer W_{t+1}
        """
        buffer = self.get_buffer()
        
        # Ensure H_t is a tensor and handle different input shapes
        if not isinstance(H_t, torch.Tensor):
            H_t = torch.tensor(H_t)
        
        # Store a copy to avoid reference issues
        H_t = H_t.detach().clone()
        
        # Algorithm 1 Line 8: if |W_t| < L_w then
        if len(buffer) < self.L_w:
            # Algorithm 1 Line 9: W_{t+1} ← W_t ∪ {H_t}
            buffer.append(H_t)
        else:
            # Algorithm 1 Lines 10-13: Remove oldest element (FIFO)
            # Remove first element (oldest)
            buffer.pop(0)
            # Add new element
            buffer.append(H_t)
        
        self._buffer_size = torch.tensor(len(buffer))
        return buffer
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert buffer list to tensor for attention operations
        
        Args:
            device: Target device for tensor
        
        Returns:
            Tensor of shape [L_w, *feature_shape] or [|W_t|, *feature_shape]
                Padded with zeros if buffer not full
        """
        buffer = self.get_buffer()
        
        if len(buffer) == 0:
            # Return empty tensor with correct shape
            # Try to infer shape from feature_dim
            return torch.zeros(0, self.feature_dim, device=device)
        
        # Stack all elements
        # Handle different shapes: [B, N, D], [N, D], or [D]
        stacked = torch.stack(buffer, dim=0)  # [|W_t|, *feature_shape]
        
        if device is not None:
            stacked = stacked.to(device)
        
        return stacked
    
    def __len__(self) -> int:
        """Return current size of working memory"""
        return len(self.get_buffer())
    
    def is_full(self) -> bool:
        """Check if working memory is at capacity"""
        return len(self.get_buffer()) >= self.L_w

