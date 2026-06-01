"""
multi_agent_model.py  — GTNet v4
═══════════════════════════════════════════════════════════════════════════════

CẢI TIẾN so với v3:

[V4-A] Adaptive Radius thực sự hoạt động
    Implement _compute_adaptive_adj() trong forward() của MultiAgentTrajectoryPredictor.
    Khi config.enable_adaptive_radius = True, ma trận adj gốc từ dataset
    sẽ bị ghi đè bằng ma trận vật lý mới: bán kính tương tác mở rộng dựa
    trên vận tốc của từng xe.

[V4-B] Edge-aware GATLayer (RelativeEdgeEncoder)
    GATLayer v4 nhận thêm edge_feat [B, N, N, edge_dim] từ RelativeEdgeEncoder.
    Thêm thông tin không gian tương đối (dx, dy, distance) vào Attention.
    Nếu config.gat_edge_dim = 0 → backward-compatible với v3 (mù không gian).

CẢI TIẾN TỪ v3:
[IMP-1] TemporalSelfAttention
[IMP-2] MultimodalDecoder (shared GRU + mode embeddings)
[IMP-3] Encoder dropout
[IMP-4] Config backward-compatible (Strict=False loading)
"""

from __future__ import annotations

import math
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
    
    # [V4-B] GAT Edge Feature Dim (0 = disable)
    gat_edge_dim: int = 32

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
        if self.gat_edge_dim < 0:
            raise ValueError(f"gat_edge_dim >= 0, got {self.gat_edge_dim}")

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, data: dict) -> "MultiAgentModelConfig":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def gat_config(cls, hidden_dim: int = 256) -> "MultiAgentModelConfig":
        return cls(hidden_dim=hidden_dim, enable_gat=True)

    @classmethod
    def multimodal_config(cls, hidden_dim: int = 256) -> "MultiAgentModelConfig":
        return cls(hidden_dim=hidden_dim, enable_multimodal=True)

    @classmethod
    def full_config(cls, hidden_dim: int = 256) -> "MultiAgentModelConfig":
        return cls(
            hidden_dim=hidden_dim,
            enable_gat=True,
            enable_multimodal=True,
            enable_adaptive_radius=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# [IMP-1] Temporal Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSelfAttention(nn.Module):
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
        BN, T, H = seq.shape

        Q = self.q(seq)
        K = self.k(seq)
        V = self.v(seq)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  

        if mask is not None:
            pad_mask = ~mask.unsqueeze(1)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        out = torch.bmm(attn, V)

        anchor = seq[:, -1, :]
        ctx = out[:, -1, :]
        return self.norm(anchor + self.proj(ctx))


# ══════════════════════════════════════════════════════════════════════════════
# [V4-B] Edge Feature Encoder
# ══════════════════════════════════════════════════════════════════════════════

class RelativeEdgeEncoder(nn.Module):
    """
    Maps relative coordinates (dx, dy, distance) between agents to an edge feature.
    Input: last valid positions [B, N, 2]
    Output: edge features [B, N, N, edge_dim]
    """
    def __init__(self, edge_dim: int):
        super().__init__()
        self.edge_dim = edge_dim
        # input dim = 3: dx, dy, Euclidean distance
        self.mlp = nn.Sequential(
            nn.Linear(3, edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(edge_dim, edge_dim)
        )

    def forward(self, pos: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = pos.shape
        # Create [B, N, N, 2] distance vectors
        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        diff = pos_j - pos_i  # relative vector from i to j (dx, dy)
        
        # Calculate Euclidean distance [B, N, N, 1]
        dist = torch.norm(diff, dim=-1, keepdim=True)
        
        # Combine [B, N, N, 3]
        inputs = torch.cat([diff, dist], dim=-1)
        
        # Apply MLP to get edge features [B, N, N, edge_dim]
        edge_feat = self.mlp(inputs)
        
        # Mask out invalid pairs
        mask_bool = agent_mask.to(torch.bool)
        valid_pair = mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2) # [B, N, N]
        
        return edge_feat * valid_pair.unsqueeze(-1).to(edge_feat.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Graph interaction blocks (GCN & GAT v4)
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
        edge_feat: Optional[torch.Tensor] = None, # Ignore edge_feat for baseline GCN
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
    [V4-B] Multi-head Graph Attention Network layer (Edge-aware).
    Receives edge_feat. If edge_dim > 0, concatenates it before calculating attention.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True,
        edge_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim

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
        
        # Attention now takes (2 * head_dim + edge_dim)
        self.attention = nn.ModuleList(
            [nn.Linear(2 * self.head_dim + edge_dim, 1, bias=False) for _ in range(num_heads)]
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
        edge_feat: Optional[torch.Tensor] = None,
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
            
            # [V4-B] Concat edge features if enabled
            if self.edge_dim > 0 and edge_feat is not None:
                h_cat = torch.cat([h_i, h_j, edge_feat], dim=-1)  # [B, N, N, 2*head_dim + edge_dim]
            else:
                h_cat = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2*head_dim] (Backward compatible v3)

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

        self.mode_embeddings = nn.Embedding(num_modes, mode_embed_dim)

        self.decoder_cell = nn.GRUCell(
            input_size=2 + mode_embed_dim,
            hidden_size=hidden_dim,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, 2),
        )
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def forward(
        self,
        h: torch.Tensor,
        last_pos: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, H = h.shape
        BN = B * N
        device, dtype = h.device, h.dtype

        flat_h    = h.reshape(BN, H)
        flat_pos  = last_pos.reshape(BN, 2)
        flat_mask = agent_mask.reshape(BN).to(dtype=dtype).unsqueeze(-1)

        flat_pos = flat_pos * flat_mask

        mode_ids = torch.arange(self.num_modes, device=device)
        mode_embeds = self.mode_embeddings(mode_ids)

        all_mode_preds = []

        for k in range(self.num_modes):
            state     = flat_h.clone()
            prev_pos  = flat_pos.clone()
            prev_delta = torch.zeros(BN, 2, device=device, dtype=dtype)

            mode_emb_k = mode_embeds[k].unsqueeze(0).expand(BN, -1)

            mode_preds = []
            for _ in range(self.future_steps):
                gru_in = torch.cat([prev_delta, mode_emb_k], dim=-1)
                state  = self.decoder_cell(gru_in, state)

                delta    = self.delta_head(state) * flat_mask
                prev_pos = prev_pos + delta
                mode_preds.append(prev_pos)
                prev_delta = delta

            mode_preds_t = torch.stack(mode_preds, dim=1)
            all_mode_preds.append(mode_preds_t)

        all_modes = torch.stack(all_mode_preds, dim=1)
        return all_modes.reshape(B, N, self.num_modes, self.future_steps, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Top-level predictor
# ══════════════════════════════════════════════════════════════════════════════

class MultiAgentTrajectoryPredictor(nn.Module):
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

        # [IMP-1] Temporal self-attention
        if cfg.use_temporal_attention:
            self.temporal_attn = TemporalSelfAttention(
                hidden_dim=int(cfg.hidden_dim),
                dropout=float(cfg.dropout),
            )
        else:
            self.temporal_attn = None

        # [IMP-3] Encoder output dropout
        self.enc_dropout = (
            nn.Dropout(p=float(cfg.encoder_dropout))
            if cfg.encoder_dropout > 0.0
            else nn.Identity()
        )

        # ── [V4-B] Edge Feature Encoder ─────────────────────────────────────────
        if cfg.enable_gat and cfg.gat_edge_dim > 0:
            self.edge_encoder = RelativeEdgeEncoder(edge_dim=int(cfg.gat_edge_dim))
        else:
            self.edge_encoder = None

        # ── Graph interaction blocks ──────────────────────────────────────────
        if cfg.enable_gat:
            self.graph_blocks = nn.ModuleList([
                GATLayer(
                    hidden_dim=int(cfg.hidden_dim),
                    num_heads=int(cfg.num_attention_heads),
                    dropout=float(cfg.attention_dropout),
                    concat_heads=(cfg.attention_concat_mode == "concat"),
                    edge_dim=int(cfg.gat_edge_dim),  # [V4-B]
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
        
    @staticmethod
    def _last_valid_velocities(
        x: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract last valid (vx, vy) velocity from history for each agent."""
        # Assuming feature dim 2 and 3 are vx and vy
        if x.shape[-1] < 4:
            return torch.zeros((x.shape[0], x.shape[1], 2), dtype=x.dtype, device=x.device)
            
        valid_counts = x_mask.long().sum(dim=-1).clamp_min(1)
        last_idx = (valid_counts - 1).view(
            x.shape[0], x.shape[1], 1, 1
        ).expand(-1, -1, 1, 2)
        return torch.gather(x[..., 2:4], dim=2, index=last_idx).squeeze(2)

    def _compute_adaptive_adj(
        self, 
        last_pos: torch.Tensor, 
        last_vel: torch.Tensor, 
        agent_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        [V4-A] Dynamically compute physics-based adjacency matrix.
        Radius = base_radius + speed * alpha
        """
        B, N, _ = last_pos.shape
        device, dtype = last_pos.device, last_pos.dtype
        
        # Calculate speed for each agent [B, N]
        speed = torch.norm(last_vel, dim=-1)
        
        # Calculate dynamic radius [B, N]
        radius = self.config.radius_base + speed * self.config.radius_alpha
        
        # Create distance matrix [B, N, N]
        pos_i = last_pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = last_pos.unsqueeze(1).expand(B, N, N, 2)
        dist = torch.norm(pos_j - pos_i, dim=-1)
        
        # Check if distance is within radius (agent i observing agent j)
        # radius is [B, N] -> expand to [B, N, N] where radius depends on agent i
        radius_i = radius.unsqueeze(2).expand(B, N, N)
        adj_dynamic = (dist <= radius_i).to(dtype)
        
        # Mask out invalid agents
        mask_float = agent_mask.to(dtype)
        valid_pair = mask_float.unsqueeze(1) * mask_float.unsqueeze(2)
        
        return adj_dynamic * valid_pair

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        encoded_seq, hidden = self.encoder(flat_x)

        if self.temporal_attn is not None:
            flat_mask_bool = x_mask.reshape(BN, T_hist)
            h_flat = self.temporal_attn(encoded_seq, flat_mask_bool)
        else:
            h_flat = hidden[-1]

        h_flat = self.enc_dropout(h_flat)

        h = h_flat.reshape(B, N, H)
        h = h * agent_mask.to(dtype=h.dtype).unsqueeze(-1)
        
        # Get final position and velocity for spatial logic
        last_pos = self._last_valid_positions(x, x_mask)  # [B, N, 2]
        last_vel = self._last_valid_velocities(x, x_mask) # [B, N, 2]
        
        # ── [V4-A] Adaptive Radius Logic ───────────────────────────────────────
        if self.config.enable_adaptive_radius:
            adj = self._compute_adaptive_adj(last_pos, last_vel, agent_mask)
            
        # ── [V4-B] Edge Feature Encoding ───────────────────────────────────────
        edge_feat = None
        if self.edge_encoder is not None:
            edge_feat = self.edge_encoder(last_pos, agent_mask)

        # ── Graph interaction ──────────────────────────────────────────────────
        for block in self.graph_blocks:
            h = block(h, adj=adj, agent_mask=agent_mask, edge_feat=edge_feat)

        # ── Decode future trajectory ───────────────────────────────────────────
        if self.config.enable_multimodal:
            return self.multimodal_decoder(h, last_pos, agent_mask)

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

        return torch.stack(preds, dim=2)


# ══════════════════════════════════════════════════════════════════════════════
# Loss & metrics (unchanged)
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
    B, N, K, T, C = pred.shape
    target_exp = target.unsqueeze(2).expand(B, N, K, T, C)

    valid_agent = agent_mask & y_mask.any(dim=-1)
    valid = (y_mask & valid_agent.unsqueeze(-1)).unsqueeze(2).unsqueeze(-1)
    valid_f = valid.to(dtype=pred.dtype)

    per_mode_loss = F.smooth_l1_loss(pred, target_exp, reduction="none")
    per_mode_loss = per_mode_loss * valid_f
    valid_steps = valid_f.squeeze(-1).sum(dim=-1).clamp_min(1.0)
    per_mode_loss = per_mode_loss.sum(dim=(-1, -2)) / (valid_steps * C)

    wta, _ = per_mode_loss.min(dim=-1)

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
    if isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg_dict = checkpoint.get("model_config", None)

    if cfg_dict is None:
        print("⚠  Legacy checkpoint (no model_config). Assuming baseline config.")
        ckpt_cfg = MultiAgentModelConfig()
    else:
        ckpt_cfg = MultiAgentModelConfig.from_json(cfg_dict)
        edge_dim = getattr(ckpt_cfg, 'gat_edge_dim', 0)
        print(
            f"✓ Checkpoint config: GAT={ckpt_cfg.enable_gat}, "
            f"Multimodal={ckpt_cfg.enable_multimodal}, "
            f"AdaptiveRadius={ckpt_cfg.enable_adaptive_radius}, "
            f"TemporalAttn={ckpt_cfg.use_temporal_attention}, "
            f"EdgeDim={edge_dim}"
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
