"""
multi_agent_model.py  — GTNet v3
═══════════════════════════════════════════════════════════════════════════════

CẢI TIẾN so với v2 (dựa trên phân tích 39-epoch GTNet_Full):

[IMP-1] TemporalSelfAttention (mới)
    Lightweight single-head attention trên toàn bộ GRU output sequence.
    Cho phép model chú ý tới những timestep quan trọng trong lịch sử
    (phanh gấp, chuyển hướng) thay vì chỉ dùng hidden[-1].
    Kích hoạt bằng config.use_temporal_attention = True.

[IMP-2] MultimodalDecoder — shared GRU + mode embeddings
    Thay thế K GRUCell độc lập bằng một GRUCell chung + K learned mode
    embeddings kích thước mode_embed_dim (default 64).
    • Ít tham số hơn (đặc biệt khi K lớn): K × hidden_dim²×5 → hidden_dim²×5
    • Mode embedding được nối vào GRU input ở TỪNG bước decode
      → mode specialize thông qua context liên tục, không chỉ initial state
    • Thêm score_head: Linear(hidden_dim → K) để dự đoán mode probability
      (dùng cho NLL loss hoặc confidence-weighted inference sau này)

[IMP-3] Encoder dropout (mới)
    Dropout trên hidden state sau GRU encoder (config.encoder_dropout).
    Giúp giảm chênh lệch train/val ADE (hiện tại ~3.5×).

[IMP-4] Config backward-compatible
    from_json / to_json tự động handle missing keys → có thể load
    checkpoint cũ mà không cần thay đổi training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MultiAgentModelConfig:
    # ── Baseline ─────────────────────────────────────────────────────────────
    input_dim: int = 6
    hidden_dim: int = 256
    graph_layers: int = 3
    future_steps: int = 30
    dropout: float = 0.1

    # ── GAT ──────────────────────────────────────────────────────────────────
    enable_gat: bool = False
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    attention_concat_mode: str = "concat"  # "concat" or "average"

    # ── Multimodal ───────────────────────────────────────────────────────────
    enable_multimodal: bool = False
    num_modes: int = 3

    # ── Adaptive radius ──────────────────────────────────────────────────────
    enable_adaptive_radius: bool = False
    radius_base: float = 20.0
    radius_alpha: float = 0.5

    # ── [IMP-1] Temporal self-attention ──────────────────────────────────────
    use_temporal_attention: bool = False   # Apply after GRU encoder

    # ── [IMP-2] Mode embedding dim in MultimodalDecoder ──────────────────────
    mode_embed_dim: int = 64              # Shared GRU input: 2 + mode_embed_dim

    # ── [IMP-3] Encoder output dropout ───────────────────────────────────────
    encoder_dropout: float = 0.0          # 0 = disabled (safe default)

    def __post_init__(self) -> None:
        if self.num_attention_heads < 1:
            raise ValueError(f"num_attention_heads >= 1, got {self.num_attention_heads}")
        if self.attention_concat_mode not in ("concat", "average"):
            raise ValueError(f"attention_concat_mode must be 'concat'/'average'")
        if self.num_modes < 1:
            raise ValueError(f"num_modes >= 1, got {self.num_modes}")
        if self.radius_base <= 0:
            raise ValueError(f"radius_base > 0, got {self.radius_base}")
        if self.radius_alpha < 0:
            raise ValueError(f"radius_alpha >= 0, got {self.radius_alpha}")
        if self.mode_embed_dim < 1:
            raise ValueError(f"mode_embed_dim >= 1, got {self.mode_embed_dim}")

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, data: dict) -> "MultiAgentModelConfig":
        """Backward-compatible: silently ignore unknown keys, use defaults for missing keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def full_config(cls, hidden_dim: int = 256) -> "MultiAgentModelConfig":
        return cls(
            hidden_dim=hidden_dim,
            enable_gat=True,
            enable_multimodal=True,
            enable_adaptive_radius=True,
        )

    @classmethod
    def gat_config(cls, hidden_dim: int = 256) -> "MultiAgentModelConfig":
        return cls(hidden_dim=hidden_dim, enable_gat=True)

    @classmethod
    def multimodal_config(cls, hidden_dim: int = 256) -> "MultiAgentModelConfig":
        return cls(hidden_dim=hidden_dim, enable_multimodal=True)


# ══════════════════════════════════════════════════════════════════════════════
# [IMP-1] Temporal Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSelfAttention(nn.Module):
    """
    Lightweight single-head self-attention over a GRU output sequence.

    Instead of only using hidden[-1], this module attends over the full
    history sequence and produces a context vector that emphasises the
    most informative past timesteps (e.g. a sudden braking event).

    Input:  seq  [BN, T, H]  — GRU output sequence
            mask [BN, T]     — bool, True = valid timestep
    Output: [BN, H]           — context vector (weighted sum + projection)

    Designed to be applied AFTER the GRU encoder and BEFORE the graph blocks.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:  [BN, T, H]
            mask: [BN, T]  bool — True = valid
        Returns:
            context: [BN, H]
        """
        BN, T, H = seq.shape

        Q = self.q(seq)  # [BN, T, H]
        K = self.k(seq)
        V = self.v(seq)

        # Scaled dot-product scores
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [BN, T, T]

        # Mask padding timesteps in the key dimension
        if mask is not None:
            # True = valid, so invert for masking
            pad_mask = ~mask.unsqueeze(1)  # [BN, 1, T]
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # rows with all -inf → uniform 0
        attn = self.drop(attn)

        out = torch.bmm(attn, V)  # [BN, T, H]

        # Use the last valid timestep as the residual anchor
        # (fallback to seq[:, -1] which is always populated after GRU)
        anchor = seq[:, -1, :]  # [BN, H]

        # Weighted context = attention output at the query position of anchor
        # Simpler: use global average of attention-weighted values + residual
        ctx = out[:, -1, :]  # [BN, H] — context from the "now" query position
        return self.norm(anchor + self.proj(ctx))  # [BN, H]


# ══════════════════════════════════════════════════════════════════════════════
# Graph interaction blocks (GCN & GAT)
# ══════════════════════════════════════════════════════════════════════════════

class GraphInteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = agent_mask.to(dtype=h.dtype)
        adj = adj.to(dtype=h.dtype) * mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = adj + torch.eye(
            adj.shape[-1], device=adj.device, dtype=adj.dtype
        ).unsqueeze(0) * mask.unsqueeze(1)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        message = torch.bmm(adj / degree, h)
        updated = self.update(torch.cat([h, message], dim=-1))
        return self.norm(h + updated) * mask.unsqueeze(-1)


class GATLayer(nn.Module):
    """
    Multi-head Graph Attention Network layer.

    [FIX-7] Uses boolean masking to avoid fp16 -1e9 overflow.
    Renormalizes attention weights after dropout so padded agents
    never leak NaN into valid agents across stacked layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads

        if concat_heads:
            assert hidden_dim % num_heads == 0, (
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads}) when concat_heads=True"
            )
            self.head_dim = hidden_dim // num_heads
        else:
            self.head_dim = hidden_dim

        self.W = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim, bias=False) for _ in range(num_heads)]
        )
        self.attention = nn.ModuleList(
            [nn.Linear(2 * self.head_dim, 1, bias=False) for _ in range(num_heads)]
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        if not concat_heads:
            self.head_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = h.shape
        device, dtype = h.device, h.dtype

        mask_bool = agent_mask.to(dtype=torch.bool)
        h = torch.where(mask_bool.unsqueeze(-1), h, torch.zeros_like(h))

        adj_bool = (adj > 0) & mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        adj_bool = adj_bool | (eye & mask_bool.unsqueeze(-1))

        head_outputs = []
        for head_idx in range(self.num_heads):
            h_t = self.W[head_idx](h)  # [B, N, head_dim]
            h_i = h_t.unsqueeze(2).expand(B, N, N, self.head_dim)
            h_j = h_t.unsqueeze(1).expand(B, N, N, self.head_dim)
            h_cat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2*head_dim]

            e = self.attention[head_idx](h_cat).squeeze(-1)  # [B, N, N]
            e = self.leaky_relu(e)

            mask_val = torch.finfo(e.dtype).min
            e_masked = e.masked_fill(~adj_bool, mask_val)
            alpha = torch.softmax(e_masked.float(), dim=-1).to(dtype=e.dtype)

            alpha = alpha * adj_bool.to(dtype=alpha.dtype)
            alpha_denom = alpha.sum(dim=-1, keepdim=True)
            alpha = torch.where(
                alpha_denom > 0,
                alpha / alpha_denom.clamp_min(torch.finfo(alpha.dtype).eps),
                torch.zeros_like(alpha),
            )
            alpha = self.dropout_layer(alpha)
            h_prime = torch.bmm(alpha, h_t)  # [B, N, head_dim]
            head_outputs.append(h_prime)

        if self.concat_heads:
            h_out = torch.cat(head_outputs, dim=-1)
        else:
            h_out = torch.stack(head_outputs, dim=0).mean(dim=0)
            h_out = self.head_projection(h_out)

        h_out = self.norm(h + h_out)
        return torch.where(mask_bool.unsqueeze(-1), h_out, torch.zeros_like(h_out))


# ══════════════════════════════════════════════════════════════════════════════
# [IMP-2] Multimodal Decoder — shared GRU + mode embeddings
# ══════════════════════════════════════════════════════════════════════════════

class MultimodalDecoder(nn.Module):
    """
    Multimodal decoder with K trajectory hypotheses.

    ARCHITECTURE CHANGE (v2 → v3):
      Old: K independent GRUCells (each K×(hidden_dim²×5) params)
      New: 1 shared GRUCell + K learned mode embeddings (mode_embed_dim)

    How it works:
      At every decoding step t, the GRU receives:
          input = cat([prev_delta (2-D), mode_embedding_k (mode_embed_dim)])
      This means mode identity flows through the GRU at EVERY step,
      not just as an initial state offset — modes diverge more naturally.

    Additional score_head outputs K logits from the initial hidden state,
    representing the model's confidence in each mode. These can be used for:
      • Confidence-weighted averaging at inference time
      • NLL (Gaussian Mixture) loss in future training versions
      (Currently NOT used in loss — stored for compatibility)

    Args:
        hidden_dim:     Encoder output dimension
        num_modes:      K — number of trajectory hypotheses
        future_steps:   Prediction horizon
        mode_embed_dim: Dimension of per-mode conditioning embeddings
        dropout:        Dropout in delta_head MLP
    """

    def __init__(
        self,
        hidden_dim: int,
        num_modes: int = 3,
        future_steps: int = 30,
        mode_embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.mode_embed_dim = mode_embed_dim

        # Learned per-mode conditioning embeddings
        self.mode_embeddings = nn.Embedding(num_modes, mode_embed_dim)

        # Shared decoder: input = [prev_delta(2), mode_embed(mode_embed_dim)]
        self.decoder_cell = nn.GRUCell(
            input_size=2 + mode_embed_dim,
            hidden_size=hidden_dim,
        )

        # Shared delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, 2),
        )

        # Mode score head: predict relative confidence of each mode
        # Input: initial hidden state h → K logits
        # NOTE: not currently used in the training loss.
        #       Access via model.multimodal_decoder.score_head(h_flat)
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def forward(
        self,
        h: torch.Tensor,           # [B, N, H]
        last_pos: torch.Tensor,    # [B, N, 2]
        agent_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:             # [B, N, K, T, 2]
        """
        Predict K alternative future trajectories.

        Returns:
            trajectories: [B, N, K, T, 2]  (same shape as v2, backward compatible)
        """
        B, N, H = h.shape
        BN = B * N
        device, dtype = h.device, h.dtype

        flat_h    = h.reshape(BN, H)
        flat_pos  = last_pos.reshape(BN, 2)
        flat_mask = agent_mask.reshape(BN).to(dtype=dtype).unsqueeze(-1)

        # Zero out positions for padded agents
        flat_pos = flat_pos * flat_mask

        # Precompute mode embeddings for all K modes: [K, E]
        mode_ids = torch.arange(self.num_modes, device=device)
        mode_embeds = self.mode_embeddings(mode_ids)  # [K, E]

        all_mode_preds = []

        for k in range(self.num_modes):
            # Each mode starts from the SAME initial hidden state h
            # (diversity emerges from different mode_embed at every step)
            state     = flat_h.clone()       # [BN, H]
            prev_pos  = flat_pos.clone()     # [BN, 2]
            prev_delta = torch.zeros(BN, 2, device=device, dtype=dtype)

            # Broadcast mode embedding: [BN, E]
            mode_emb_k = mode_embeds[k].unsqueeze(0).expand(BN, -1)

            mode_preds = []
            for _ in range(self.future_steps):
                # GRU input: concat prev_delta with mode conditioning
                gru_in = torch.cat([prev_delta, mode_emb_k], dim=-1)  # [BN, 2+E]
                state  = self.decoder_cell(gru_in, state)               # [BN, H]

                delta    = self.delta_head(state) * flat_mask           # [BN, 2]
                prev_pos = prev_pos + delta
                mode_preds.append(prev_pos)
                prev_delta = delta

            # [BN, T, 2] → store
            mode_preds_t = torch.stack(mode_preds, dim=1)
            all_mode_preds.append(mode_preds_t)

        # Stack K modes: [BN, K, T, 2] → reshape → [B, N, K, T, 2]
        all_modes = torch.stack(all_mode_preds, dim=1)          # [BN, K, T, 2]
        return all_modes.reshape(B, N, self.num_modes, self.future_steps, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Top-level predictor
# ══════════════════════════════════════════════════════════════════════════════

class MultiAgentTrajectoryPredictor(nn.Module):
    """
    GRU encoder  →  [TemporalAttention]  →  Graph blocks  →  Decoder

    [IMP-1] Optional TemporalSelfAttention between encoder and graph blocks.
    [IMP-2] MultimodalDecoder now uses shared GRU + mode embeddings.
    [IMP-3] Optional dropout on encoder output (config.encoder_dropout).
    """

    def __init__(self, config: Optional[MultiAgentModelConfig] = None) -> None:
        super().__init__()
        self.config = config or MultiAgentModelConfig()
        cfg = self.config

        # ── GRU encoder ──────────────────────────────────────────────────────
        self.encoder = nn.GRU(
            input_size=int(cfg.input_dim),
            hidden_size=int(cfg.hidden_dim),
            batch_first=True,
        )

        # [IMP-1] Temporal self-attention (optional)
        if cfg.use_temporal_attention:
            self.temporal_attn = TemporalSelfAttention(
                hidden_dim=int(cfg.hidden_dim),
                dropout=float(cfg.dropout),
            )
        else:
            self.temporal_attn = None  # type: ignore[assignment]

        # [IMP-3] Encoder output dropout (optional)
        self.enc_dropout = (
            nn.Dropout(p=float(cfg.encoder_dropout))
            if cfg.encoder_dropout > 0.0
            else nn.Identity()
        )

        # ── Graph interaction blocks ──────────────────────────────────────────
        if cfg.enable_gat:
            self.graph_blocks = nn.ModuleList([
                GATLayer(
                    hidden_dim=int(cfg.hidden_dim),
                    num_heads=int(cfg.num_attention_heads),
                    dropout=float(cfg.attention_dropout),
                    concat_heads=(cfg.attention_concat_mode == "concat"),
                )
                for _ in range(max(0, int(cfg.graph_layers)))
            ])
        else:
            self.graph_blocks = nn.ModuleList([
                GraphInteractionBlock(
                    hidden_dim=int(cfg.hidden_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(max(0, int(cfg.graph_layers)))
            ])

        # ── Decoder ──────────────────────────────────────────────────────────
        if cfg.enable_multimodal:
            # [IMP-2] Shared GRU + mode embeddings
            self.multimodal_decoder = MultimodalDecoder(
                hidden_dim=int(cfg.hidden_dim),
                num_modes=int(cfg.num_modes),
                future_steps=int(cfg.future_steps),
                mode_embed_dim=int(cfg.mode_embed_dim),
                dropout=float(cfg.dropout),
            )
        else:
            self.decoder_cell = nn.GRUCell(
                input_size=2, hidden_size=int(cfg.hidden_dim)
            )
            self.delta_head = nn.Sequential(
                nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(cfg.hidden_dim), 2),
            )

    @staticmethod
    def _last_valid_positions(
        x: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract last valid (x, y) position from history for each agent."""
        valid_counts = x_mask.long().sum(dim=-1).clamp_min(1)
        last_idx = (valid_counts - 1).view(
            x.shape[0], x.shape[1], 1, 1
        ).expand(-1, -1, 1, 2)
        return torch.gather(x[..., :2], dim=2, index=last_idx).squeeze(2)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          [B, N, T_hist, input_dim]
            adj:        [B, N, N]
            x_mask:     [B, N, T_hist]  bool (True = valid timestep)
            agent_mask: [B, N]          bool (True = valid agent)

        Returns:
            Unimodal:   [B, N, T_future, 2]
            Multimodal: [B, N, K, T_future, 2]
        """
        if x_mask is None:
            x_mask = torch.ones(x.shape[:3], dtype=torch.bool, device=x.device)
        if agent_mask is None:
            agent_mask = x_mask.any(dim=-1)

        B, N, T_hist, input_dim = x.shape
        BN = B * N
        H = int(self.config.hidden_dim)

        # ── Encode history ─────────────────────────────────────────────────────
        flat_x    = x.reshape(BN, T_hist, input_dim)
        flat_mask = x_mask.reshape(BN, T_hist).to(dtype=flat_x.dtype)
        flat_x    = flat_x * flat_mask.unsqueeze(-1)

        encoded_seq, hidden = self.encoder(flat_x)  # encoded_seq: [BN, T, H]

        # [IMP-1] Optional temporal attention
        if self.temporal_attn is not None:
            flat_mask_bool = x_mask.reshape(BN, T_hist)
            h_flat = self.temporal_attn(encoded_seq, flat_mask_bool)  # [BN, H]
        else:
            h_flat = hidden[-1]  # [BN, H] — standard: use final hidden state

        # [IMP-3] Encoder dropout
        h_flat = self.enc_dropout(h_flat)

        h = h_flat.reshape(B, N, H)
        h = h * agent_mask.to(dtype=h.dtype).unsqueeze(-1)

        # ── Graph interaction ──────────────────────────────────────────────────
        for block in self.graph_blocks:
            h = block(h, adj=adj, agent_mask=agent_mask)

        # ── Decode future trajectory ───────────────────────────────────────────
        last_pos = self._last_valid_positions(x, x_mask)  # [B, N, 2]

        if self.config.enable_multimodal:
            return self.multimodal_decoder(h, last_pos, agent_mask)

        # Unimodal autoregressive decode
        state      = h.reshape(BN, H)
        prev_pos   = last_pos.reshape(BN, 2)
        prev_delta = torch.zeros_like(prev_pos)
        flat_mask2 = (
            agent_mask.reshape(BN).to(dtype=state.dtype).unsqueeze(-1)
        )

        preds = []
        for _ in range(int(self.config.future_steps)):
            state    = self.decoder_cell(prev_delta, state)
            delta    = self.delta_head(state) * flat_mask2
            prev_pos = prev_pos + delta
            preds.append(prev_pos.reshape(B, N, 2))
            prev_delta = delta

        return torch.stack(preds, dim=2)  # [B, N, T, 2]


# ══════════════════════════════════════════════════════════════════════════════
# Loss & metrics
# ══════════════════════════════════════════════════════════════════════════════

def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    mask = (y_mask & agent_mask.unsqueeze(-1)).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, target, reduction="none")
    denom = mask.to(dtype=pred.dtype).sum().clamp_min(1.0) * pred.shape[-1]
    return (diff * mask.to(dtype=pred.dtype)).sum() / denom


def wta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Winner-Takes-All loss for multimodal predictions.

    Args:
        pred:       [B, N, K, T, 2]
        target:     [B, N, T, 2]
        y_mask:     [B, N, T]
        agent_mask: [B, N]
    """
    B, N, K, T, C = pred.shape
    target_exp = target.unsqueeze(2).expand(B, N, K, T, C)

    valid_agent = agent_mask & y_mask.any(dim=-1)
    valid = (y_mask & valid_agent.unsqueeze(-1)).unsqueeze(2).unsqueeze(-1)
    valid_f = valid.to(dtype=pred.dtype)

    per_mode_loss = F.smooth_l1_loss(pred, target_exp, reduction="none")
    per_mode_loss = per_mode_loss * valid_f
    valid_steps = valid_f.squeeze(-1).sum(dim=-1).clamp_min(1.0)
    per_mode_loss = per_mode_loss.sum(dim=(-1, -2)) / (valid_steps * C)

    wta, _ = per_mode_loss.min(dim=-1)  # [B, N]

    mask = valid_agent.float()
    return (wta * mask).sum() / mask.sum().clamp_min(1.0)


def masked_ade_fde(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> Tuple[float, float]:
    valid = y_mask & agent_mask.unsqueeze(-1)
    displacement = torch.linalg.norm(pred - target, dim=-1)
    valid_f = valid.to(dtype=pred.dtype)
    ade = (displacement * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    T = displacement.shape[-1]
    flipped = valid.flip(dims=[-1])
    last_valid_idx = (T - 1) - flipped.long().argmax(dim=-1)
    final_disp = displacement.gather(-1, last_valid_idx.unsqueeze(-1)).squeeze(-1)
    final_valid = valid.any(dim=-1)
    fde = (
        final_disp * final_valid.to(dtype=pred.dtype)
    ).sum() / final_valid.to(dtype=pred.dtype).sum().clamp_min(1.0)

    return float(ade.detach().cpu().item()), float(fde.detach().cpu().item())


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint_with_compatibility(
    checkpoint_path: str,
    device: "torch.device | str" = "cpu",
    target_config: Optional[MultiAgentModelConfig] = None,
) -> "tuple[MultiAgentTrajectoryPredictor, dict]":
    """
    Load checkpoint with full backward / forward compatibility.

    Changes in v3:
      • MultimodalDecoder keys changed (decoder_cells.{k}.* → decoder_cell.*)
        → handled by strict=False with informative printing
      • New keys (temporal_attn.*, enc_dropout.*) initialized randomly
      • MultiAgentModelConfig.from_json now silently ignores unknown fields
        and uses defaults for missing fields → no KeyError on old checkpoints
    """
    if isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg_dict = checkpoint.get("model_config", None)

    if cfg_dict is None:
        print("⚠  Legacy checkpoint (no model_config). Assuming baseline config.")
        ckpt_cfg = MultiAgentModelConfig()
    else:
        ckpt_cfg = MultiAgentModelConfig.from_json(cfg_dict)
        print(
            f"✓ Checkpoint config: GAT={ckpt_cfg.enable_gat}, "
            f"Multimodal={ckpt_cfg.enable_multimodal}, "
            f"AdaptiveRadius={ckpt_cfg.enable_adaptive_radius}, "
            f"TemporalAttn={ckpt_cfg.use_temporal_attention}"
        )

    use_cfg = target_config if target_config is not None else ckpt_cfg
    model = MultiAgentTrajectoryPredictor(config=use_cfg).to(device)

    state = checkpoint["model_state_dict"]
    result = model.load_state_dict(state, strict=False)

    if result.missing_keys:
        print(f"  ↳ {len(result.missing_keys)} new keys initialized randomly:")
        for k in result.missing_keys[:8]:
            print(f"    + {k}")
        if len(result.missing_keys) > 8:
            print(f"    ... and {len(result.missing_keys) - 8} more")
    if result.unexpected_keys:
        print(f"  ↳ {len(result.unexpected_keys)} old keys ignored:")
        for k in result.unexpected_keys[:8]:
            print(f"    - {k}")

    return model, checkpoint