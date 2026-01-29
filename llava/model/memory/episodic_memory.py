"""
Episodic Memory Module for VLM²
Implements similarity-based replacement mechanism with gated attention for salient information
as per Algorithm 1 (Lines 15-24) and paper Section 3.2
"""
import torch
import torch.nn as nn
from typing import List, Optional


class EpisodicMemory(nn.Module):
    """
    Episodic Memory with similarity-based replacement and gated attention for salient information
    
    According to VLM² paper (https://arxiv.org/pdf/2511.20644):
    - "episodic memory that consolidates and stores critical long-term information"
    - Uses attention mechanism to identify and retain salient spatial information
    - When full, replaces the most similar existing memory element (similarity-based replacement)
    
    The gated attention mechanism helps identify which information is salient enough
    to be stored in episodic memory, rather than storing all frames equally.
    """
    
    def __init__(self, L_e: int = 32, feature_dim: int = 1152, use_gated_attention: bool = True):
        """
        Args:
            L_e: Maximum capacity of episodic memory
                Recommended: 32 (from ablation study, Table 7)
            feature_dim: Dimension of features stored in memory
            use_gated_attention: Whether to use gated attention to select salient information
                If True, only stores information that passes the salience gate
        """
        super().__init__()
        self.L_e = L_e
        self.feature_dim = feature_dim
        self.use_gated_attention = use_gated_attention
        
        # Gated attention mechanism for selecting salient information
        # This helps identify which features are "critical" enough to store in episodic memory
        if use_gated_attention:
            self.salience_gate = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()  # Output: salience score in [0, 1]
            )
        
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
    
    def _compute_salience(self, H_t: torch.Tensor) -> torch.Tensor:
        """
        Compute salience score for H_t using gated attention
        
        According to VLM² paper, episodic memory should "consolidate and store critical 
        long-term information". This gate helps identify which information is salient.
        
        Args:
            H_t: Current input features [D]
        
        Returns:
            Salience score in [0, 1] - higher means more salient/critical
        """
        if not self.use_gated_attention:
            return torch.tensor(1.0, device=H_t.device)  # Always store if gate disabled
        
        # Compute salience score
        salience = self.salience_gate(H_t)  # [1]
        return salience.squeeze()
    
    def update(self, H_t: torch.Tensor, salience_threshold: float = 0.5) -> List[torch.Tensor]:
        """
        Update Episodic Memory (Algorithm 1 Lines 15-24)
        
        According to VLM² paper:
        - Episodic memory "consolidates and stores critical long-term information"
        - Uses similarity-based replacement when full
        - Should use gated attention to identify salient information
        
        Args:
            H_t: Current input features [B, N, D] or [N, D] or [D]
                where D is feature_dim
            salience_threshold: Minimum salience score to store in episodic memory
                Only used if use_gated_attention=True
        
        Returns:
            Updated episodic memory buffer E_{t+1}
        """
        buffer = self.get_buffer()
        
        # Ensure H_t is a tensor and handle different input shapes
        if not isinstance(H_t, torch.Tensor):
            H_t = torch.tensor(H_t)
        
        # Store a copy to avoid reference issues
        H_t = H_t.detach().clone()
        
        # Extract a single representation: mean pool over sequence if needed
        if H_t.dim() > 1:
            # If [B, N, D] or [N, D], mean pool to [D]
            H_t = H_t.mean(dim=tuple(range(H_t.dim() - 1)))
        
        # Gated attention: Check if information is salient enough to store
        if self.use_gated_attention:
            salience_score = self._compute_salience(H_t)
            salience_val = salience_score.item() if isinstance(salience_score, torch.Tensor) else salience_score
            
            # DEBUG: Log salience scores (only for first few frames or when filtered)
            if len(buffer) < 3 or salience_val < salience_threshold:
                if salience_val < salience_threshold:
                    print(f"[Episodic Memory DEBUG] Frame filtered by salience gate: score={salience_val:.4f} < threshold={salience_threshold}")
            
            if salience_val < salience_threshold:
                # Not salient enough - skip storing in episodic memory
                # This helps retain only critical information
                return buffer
        
        # Algorithm 1 Line 15: if |E_t| < L_e then
        if len(buffer) < self.L_e:
            # Algorithm 1 Line 16: E_{t+1} ← E_t ∪ {H_t}
            buffer.append(H_t)
        else:
            # Algorithm 1 Lines 17-23: Find most similar element and replace
            # This similarity-based replacement helps maintain diversity in episodic memory
            # by replacing redundant (similar) information with new information
            
            # Compute similarities with all existing memory elements
            similarities = []
            for mem_elem in buffer:
                sim = self._compute_similarity(H_t, mem_elem)
                similarities.append(sim)
            
            # Algorithm 1 Line 18: j* ← argmax_{j} similarity(H_t, E_t[j])
            # Find index of most similar element (most redundant)
            max_sim_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            max_sim_val = similarities[max_sim_idx]
            min_sim_val = min(similarities)
            avg_sim_val = sum(similarities) / len(similarities)
            
            # DEBUG: Log similarity statistics for replacement
            if len(buffer) < 3:
                print(f"[Episodic Memory DEBUG] Similarity-based replacement: "
                      f"replacing idx {max_sim_idx} (sim={max_sim_val:.4f}), "
                      f"min_sim={min_sim_val:.4f}, avg_sim={avg_sim_val:.4f}")
            
            # Algorithm 1 Line 19: E_{t+1}[j*] ← H_t
            # Replace the most similar (redundant) element with the new one
            # This ensures episodic memory maintains diverse, salient information
            old_elem = buffer[max_sim_idx].clone()
            buffer[max_sim_idx] = H_t
            
            # DEBUG: Verify replacement happened
            new_elem = buffer[max_sim_idx]
            replacement_verified = not torch.allclose(old_elem, new_elem, atol=1e-5)
            if not replacement_verified:
                print(f"[Episodic Memory DEBUG] Replacement verification failed! "
                      f"Element at idx {max_sim_idx} unchanged.")
        
        self._buffer_size = torch.tensor(len(buffer))
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
