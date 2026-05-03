"""
Query-Based Selection Module
Implements Section 3.3 from VideoStreaming paper (arxiv 2405.16009)

This module performs query-conditioned frame selection:
1. Encode question text to H_Q
2. Compute cosine similarity between H_Q and each clip indicator H_hat_k
3. Gumbel-Topk selection of top V memories
4. Return selected clip features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QueryBasedSelection(nn.Module):
    """
    Query-Based Selection following VideoStreaming Section 3.3

    Architecture from the paper:
    - Streaming encoder MLP: 4096→2560→2560 (projects clip features to shared intermediate space)
    - LLM input MLP: 2560→4096→4096 (projects selected clips back to LLM dimension)

    The question encoder is also a 2-layer MLP that projects the text query into
    the same 2560-dimensional intermediate space for cosine similarity computation.
    """
    # Intermediate dimension from paper (2560) used for similarity computation
    INTERMEDIATE_DIM = 2560

    def __init__(
        self,
        feature_dim: int = 4096,
        llm_dim: int = 4096,
        num_select: int = 4,
        temperature: float = 1.0,
        use_gumbel: bool = True,
    ):
        """
        Args:
            feature_dim: Dimension of clip features from vision encoder (after mm_projector)
            llm_dim: Dimension of LLM hidden states (also used as output dimension for this model)
            num_select: Number of clips to select (V from paper, default 4)
            temperature: Gumbel temperature for differentiable selection
            use_gumbel: Whether to use Gumbel-Topk (True) or hard Topk (False)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.llm_dim = llm_dim
        self.num_select = num_select
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        inter_dim = self.INTERMEDIATE_DIM

        # Streaming encoder MLP (Section 3.3): projects clip features to intermediate space
        # Paper: "Streaming encoder MLP: 4096→2560→2560" - adapted to model's actual dimensions
        self.clip_projector = nn.Sequential(
            nn.Linear(feature_dim, inter_dim),
            nn.GELU(),
            nn.Linear(inter_dim, inter_dim),
        )

        # Question encoder MLP (Section 3.3): projects question to same intermediate space
        # Same architecture - both use 2-layer MLP with GELU
        self.question_encoder = nn.Sequential(
            nn.Linear(llm_dim, inter_dim),
            nn.GELU(),
            nn.Linear(inter_dim, inter_dim),
        )

        # LLM input MLP (Section 3.3): projects selected clips back to LLM dimension
        # Paper: "LLM input MLP: 2560→4096→4096" - using model's actual llm_dim (3584 for this model)
        self.llm_input_projector = nn.Sequential(
            nn.Linear(inter_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(
        self,
        clip_features: torch.Tensor,
        question_embeds: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            clip_features: [B, T, N, D] or [B, T, D] - T clips, N tokens per clip, D=feature_dim
            question_embeds: [B, L, D] - L tokens in question, D=llm_dim
            question_mask: [B, L] - attention mask for question (optional)

        Returns:
            selected: [B, V, N, llm_dim] or [B, V, llm_dim] - V selected clips projected to LLM dim
        """
        if clip_features.dim() not in (3, 4):
            raise ValueError(f"clip_features must be [B, T, D] or [B, T, N, D], got {tuple(clip_features.shape)}")

        B, T = clip_features.shape[:2]
        if T == 0:
            return clip_features
        k = min(self.num_select, T)

        # Create clip indicators Ĥ_k [B, T, inter_dim] via mean pooling then projection
        if clip_features.dim() == 4:
            # [B, T, N, D] -> mean pool -> [B, T, D]
            H_hat = clip_features.mean(dim=2)
        else:
            # Already [B, T, D]
            H_hat = clip_features

        # Project clip features to intermediate space: H_hat_k
        H_hat = self.clip_projector(H_hat)  # [B, T, inter_dim]

        # Encode question: pool question tokens to get H_Q, then project to intermediate space
        if question_mask is not None:
            # Masked mean pooling
            question_mask_expanded = question_mask.unsqueeze(-1).to(
                device=question_embeds.device, dtype=question_embeds.dtype
            )  # [B, L, 1]
            H_Q = (question_embeds * question_mask_expanded).sum(dim=1) / (
                question_mask_expanded.sum(dim=1) + 1e-8
            )  # [B, llm_dim]
        else:
            # Simple mean pooling
            H_Q = question_embeds.mean(dim=1)  # [B, llm_dim]

        # Project question to intermediate space: H_Q
        H_Q = self.question_encoder(H_Q)  # [B, inter_dim]

        # Normalize for cosine similarity
        H_Q_norm = F.normalize(H_Q, dim=-1)  # [B, inter_dim]
        H_hat_norm = F.normalize(H_hat, dim=-1)  # [B, T, inter_dim]

        # Cosine similarity: sim(H_Q, H_hat_k) [B, T]
        similarities = torch.bmm(
            H_Q_norm.unsqueeze(1), H_hat_norm.transpose(1, 2)
        ).squeeze(1)

        # Gumbel-Topk selection (Section 3.3 from VideoStreaming)
        selected_weights = None
        if self.use_gumbel and self.training:
            indices, selected_weights = self._gumbel_topk(similarities, k=k)
        else:
            # Hard selection for inference
            indices = similarities.topk(k, dim=-1).indices

        # Gather selected clip features
        if clip_features.dim() == 4:
            # [B, T, N, D] - get selected clips
            _, _, N_tokens, D = clip_features.shape
            gather_idx = indices[:, :, None, None].expand(-1, -1, N_tokens, D)
            gathered = clip_features.gather(dim=1, index=gather_idx)  # [B, V, N, D]
            selected_flat = gathered.reshape(B * k * N_tokens, D)
            selected_flat = self.clip_projector(selected_flat)  # [B*V*N, 2560]
            selected_flat = self.llm_input_projector(selected_flat)  # [B*V*N, llm_dim]
            selected = selected_flat.view(B, k, N_tokens, -1)
            if selected_weights is not None:
                selected = selected * selected_weights[:, :, None, None]
        else:
            # [B, V, D]
            D = clip_features.shape[-1]
            gather_idx = indices[:, :, None].expand(-1, -1, D)
            selected = clip_features.gather(dim=1, index=gather_idx)
            # Project selected clips back to LLM dimension via LLM input MLP
            selected = selected.reshape(B * k, D)
            selected = self.clip_projector(selected)  # [B*V, 2560]
            selected = self.llm_input_projector(selected)  # [B*V, llm_dim]
            selected = selected.view(B, k, -1)  # [B, V, llm_dim]
            if selected_weights is not None:
                selected = selected * selected_weights[:, :, None]

        return selected

    def _gumbel_topk(self, logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Straight-through Gumbel-Topk selection.

        The forward pass uses hard top-k indices. The returned weights have value
        1.0 on selected clips, but their backward pass follows the softmax scores
        so the question/streaming encoders receive gradients.
        """
        temperature = max(float(self.temperature), 1e-6)
        uniform = torch.rand(logits.shape, device=logits.device, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
        gumbel_noise = -torch.log(-torch.log(uniform)).to(dtype=logits.dtype)
        gumbel_logits = (logits + gumbel_noise) / temperature

        # Top-k
        _, indices = gumbel_logits.topk(k, dim=-1)
        soft_weights = torch.softmax(gumbel_logits, dim=-1)
        selected_soft_weights = soft_weights.gather(dim=-1, index=indices)
        selected_weights = torch.ones_like(selected_soft_weights) + selected_soft_weights - selected_soft_weights.detach()
        return indices, selected_weights


class QuerySelectionConfig:
    """Configuration for Query-Based Selection"""

    def __init__(
        self,
        use_query_selection: bool = False,
        query_selection_num_select: int = 4,
        query_selection_temperature: float = 1.0,
        query_selection_use_gumbel: bool = True,
    ):
        self.use_query_selection = use_query_selection
        self.query_selection_num_select = query_selection_num_select
        self.query_selection_temperature = query_selection_temperature
        self.query_selection_use_gumbel = query_selection_use_gumbel
