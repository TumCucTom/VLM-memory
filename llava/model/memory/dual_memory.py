"""
Dual-Memory Module for VLM²
Implements Algorithm 1: Dual-Memory Module exactly as specified in the paper

Modified to use non-learnable cross-attention and simple 0.5/0.5 fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory


def cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout: float = 0.0
) -> torch.Tensor:
    """
    Explicit cross-attention without learnable projections.
    
    Args:
        query: [B, N, D]
        key: [B, L, D]
        value: [B, L, D]
    
    Returns:
        output: [B, N, D]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout)
    
    return torch.matmul(attn_weights, value)


class DualMemoryModule(nn.Module):
    """
    Dual-Memory Module implementing Algorithm 1 from VLM² paper
    
    This module:
    1. Retrieves from working and episodic memories using attention
    2. Fuses retrieved memories with simple 0.5/0.5 weighting
    3. Updates both memories according to their update strategies
    """
    
    def __init__(
        self,
        L_w: int = 8,
        L_e: int = 32,
        feature_dim: int = 1152,
        dropout: float = 0.0
    ):
        """
        Args:
            L_w: Working memory capacity (window size)
                Recommended: 8 (from ablation study, Table 7)
            L_e: Episodic memory capacity
                Recommended: 32 (from ablation study, Table 7)
            feature_dim: Dimension of input features H_t
            dropout: Dropout rate for attention (default 0 for no dropout)
        """
        super().__init__()
        self.L_w = L_w
        self.L_e = L_e
        self.feature_dim = feature_dim
        self.dropout = dropout
        
        # Initialize memory modules
        self.working_memory = WorkingMemory(L_w=L_w, feature_dim=feature_dim)
        self.episodic_memory = EpisodicMemory(L_e=L_e, feature_dim=feature_dim)
    
    def _prepare_for_attention(
        self,
        query: torch.Tensor,
        memory_buffer: list,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare query and memory for attention operation
        
        Args:
            query: Query tensor H_t [B, N, D] or [N, D]
            memory_buffer: List of memory elements
        
        Returns:
            query_tensor: Prepared query [B, N, D]
            memory_tensor: Prepared memory [B, L, D] where L is memory length
        """
        if device is None:
            device = query.device
        
        # Ensure query is 3D [B, N, D]
        if query.dim() == 1:
            query = query.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        elif query.dim() == 2:
            query = query.unsqueeze(0)  # [1, N, D]
        
        # Convert memory buffer to tensor
        if len(memory_buffer) == 0:
            # Empty memory - return zero tensor
            B = query.shape[0]
            memory_tensor = torch.zeros(B, 0, self.feature_dim, device=device, dtype=query.dtype)
        else:
            # Stack memory elements
            memory_list = []
            for mem_elem in memory_buffer:
                mem_elem = mem_elem.to(device)
                # Ensure same dimensionality as query
                if mem_elem.dim() == 1:
                    mem_elem = mem_elem.unsqueeze(0)  # [1, D]
                elif mem_elem.dim() == 2:
                    pass  # Already [N, D]
                elif mem_elem.dim() == 3:
                    mem_elem = mem_elem.squeeze(0)  # Remove batch dim if present
                
                # Average pool if needed to match query sequence length
                if mem_elem.shape[0] != query.shape[1]:
                    if mem_elem.shape[0] > query.shape[1]:
                        # Downsample
                        mem_elem = F.adaptive_avg_pool1d(
                            mem_elem.transpose(0, 1).unsqueeze(0),
                            query.shape[1]
                        ).squeeze(0).transpose(0, 1)
                    else:
                        # Upsample by repeating
                        repeat_factor = query.shape[1] // mem_elem.shape[0] + 1
                        mem_elem = mem_elem.repeat(repeat_factor, 1)[:query.shape[1]]
                
                memory_list.append(mem_elem)
            
            memory_tensor = torch.stack(memory_list, dim=0)  # [L, N, D]
            # Average over sequence dimension: [L, N, D] -> [L, D]
            if memory_tensor.dim() == 3:
                memory_tensor = memory_tensor.mean(dim=1)  # [L, D]
            
            # Add batch dimension: [L, D] -> [B, L, D]
            B = query.shape[0]
            if memory_tensor.dim() == 2:
                memory_tensor = memory_tensor.unsqueeze(0).expand(B, -1, -1)
        
        return query, memory_tensor
    
    def forward(
        self,
        H_t: torch.Tensor,
        W_t: Optional[list] = None,
        E_t: Optional[list] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Algorithm 1: Dual-Memory Module forward pass
        
        Args:
            H_t: Current input features [B, N, D] or [N, D] or [D]
            W_t: Current working memory buffer (optional, uses internal if None)
            E_t: Current episodic memory buffer (optional, uses internal if None)
        
        Returns:
            M_t: Fused memory output [B, N, D]
            W_{t+1}: Updated working memory buffer
            E_{t+1}: Updated episodic memory buffer
        """
        device = H_t.device
        
        # Get memory buffers (use internal if not provided)
        if W_t is None:
            W_t = self.working_memory.get_buffer()
        if E_t is None:
            E_t = self.episodic_memory.get_buffer()
        
        # Working Memory retrieval: M_t^w = CrossAttn(Q=H_t, KV=W_t)
        if len(W_t) > 0:
            query_w, key_value_w = self._prepare_for_attention(H_t, W_t, device)
            M_t_w = cross_attention(
                query_w,
                key_value_w,
                key_value_w,
                dropout=self.dropout
            )  # [B, N, D]
        else:
            # Empty working memory - use zero tensor
            if H_t.dim() == 1:
                M_t_w = torch.zeros(1, 1, self.feature_dim, device=device, dtype=H_t.dtype)
            elif H_t.dim() == 2:
                M_t_w = torch.zeros(1, H_t.shape[0], self.feature_dim, device=device, dtype=H_t.dtype)
            else:
                M_t_w = torch.zeros_like(H_t)
        
        # Episodic Memory retrieval: M_t^e = CrossAttn(Q=H_t, KV=E_t)
        if len(E_t) > 0:
            query_e, key_value_e = self._prepare_for_attention(H_t, E_t, device)
            M_t_e = cross_attention(
                query_e,
                key_value_e,
                key_value_e,
                dropout=self.dropout
            )  # [B, N, D]
        else:
            # Empty episodic memory - use zero tensor
            if H_t.dim() == 1:
                M_t_e = torch.zeros(1, 1, self.feature_dim, device=device, dtype=H_t.dtype)
            elif H_t.dim() == 2:
                M_t_e = torch.zeros(1, H_t.shape[0], self.feature_dim, device=device, dtype=H_t.dtype)
            else:
                M_t_e = torch.zeros_like(H_t)
        
        # Ensure M_t_w and M_t_e have same shape
        if M_t_w.shape != M_t_e.shape:
            target_shape = M_t_w.shape
            if M_t_e.numel() > 0:
                M_t_e = F.interpolate(
                    M_t_e.transpose(1, 2),
                    size=target_shape[1],
                    mode=linear,
                    align_corners=False
                ).transpose(1, 2)
            else:
                M_t_e = torch.zeros_like(M_t_w)
        
        # Simple 0.5/0.5 fusion (no learnable gate)
        # M_t = 0.5 * M_t^w + 0.5 * M_t^e
        M_t = 0.5 * M_t_w + 0.5 * M_t_e  # [B, N, D]
        
        # Algorithm 1 Lines 8-14: Update Working Memory
        # W_{t+1} ← Update Working Memory with H_t
        W_t_1 = self.working_memory.update(H_t)
        
        # Algorithm 1 Lines 15-24: Update Episodic Memory
        # E_{t+1} ← Update Episodic Memory with M_t
        if M_t.dim() == 3:
            M_t_for_storage = M_t.mean(dim=1)  # [B, D] -> average over sequence
            if M_t_for_storage.dim() == 2 and M_t_for_storage.shape[0] == 1:
                M_t_for_storage = M_t_for_storage.squeeze(0)  # [D]
        else:
            M_t_for_storage = M_t
        
        E_t_1 = self.episodic_memory.update(M_t_for_storage)
        
        return M_t, W_t_1, E_t_1
    
    def clear_memories(self):
        """Clear both working and episodic memories"""
        self.working_memory.clear()
        self.episodic_memory.clear()
    
    def get_memory_states(self) -> Tuple[list, list]:
        """Get current memory states"""
        return self.working_memory.get_buffer(), self.episodic_memory.get_buffer()
