from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultiAgentModelConfig:
    input_dim: int = 6
    hidden_dim: int = 128
    graph_layers: int = 2
    future_steps: int = 30
    dropout: float = 0.1


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

    def forward(self, h: torch.Tensor, adj: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        mask = agent_mask.to(dtype=h.dtype)
        adj = adj.to(dtype=h.dtype) * mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = adj + torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype).unsqueeze(0) * mask.unsqueeze(1)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        message = torch.bmm(adj / degree, h)
        updated = self.update(torch.cat([h, message], dim=-1))
        return self.norm(h + updated) * mask.unsqueeze(-1)


class MultiAgentTrajectoryPredictor(nn.Module):
    """GRU encoder + graph interaction + autoregressive GRU decoder baseline."""

    def __init__(self, config: Optional[MultiAgentModelConfig] = None) -> None:
        super().__init__()
        self.config = config or MultiAgentModelConfig()
        self.encoder = nn.GRU(
            input_size=int(self.config.input_dim),
            hidden_size=int(self.config.hidden_dim),
            batch_first=True,
        )
        self.graph_blocks = nn.ModuleList(
            [
                GraphInteractionBlock(
                    hidden_dim=int(self.config.hidden_dim),
                    dropout=float(self.config.dropout),
                )
                for _ in range(max(0, int(self.config.graph_layers)))
            ]
        )
        self.decoder_cell = nn.GRUCell(input_size=2, hidden_size=int(self.config.hidden_dim))
        self.delta_head = nn.Sequential(
            nn.Linear(int(self.config.hidden_dim), int(self.config.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.config.hidden_dim), 2),
        )

    @staticmethod
    def _last_valid_positions(x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, H, D], x_mask: [B, N, H]
        valid_counts = x_mask.long().sum(dim=-1).clamp_min(1)
        last_indices = (valid_counts - 1).view(x.shape[0], x.shape[1], 1, 1).expand(-1, -1, 1, 2)
        return torch.gather(x[..., :2], dim=2, index=last_indices).squeeze(2)

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

        batch_size, max_agents, history_len, input_dim = x.shape
        flat_x = x.reshape(batch_size * max_agents, history_len, input_dim)
        flat_mask = x_mask.reshape(batch_size * max_agents, history_len).to(dtype=flat_x.dtype)
        flat_x = flat_x * flat_mask.unsqueeze(-1)

        _encoded_seq, hidden = self.encoder(flat_x)
        h = hidden[-1].reshape(batch_size, max_agents, int(self.config.hidden_dim))
        h = h * agent_mask.to(dtype=h.dtype).unsqueeze(-1)

        for block in self.graph_blocks:
            h = block(h, adj=adj, agent_mask=agent_mask)

        state = h.reshape(batch_size * max_agents, int(self.config.hidden_dim))
        prev_pos = self._last_valid_positions(x, x_mask).reshape(batch_size * max_agents, 2)
        prev_delta = torch.zeros_like(prev_pos)
        flat_agent_mask = agent_mask.reshape(batch_size * max_agents).to(dtype=state.dtype).unsqueeze(-1)

        preds = []
        for _ in range(int(self.config.future_steps)):
            state = self.decoder_cell(prev_delta, state)
            delta = self.delta_head(state) * flat_agent_mask
            prev_pos = prev_pos + delta
            preds.append(prev_pos.reshape(batch_size, max_agents, 2))
            prev_delta = delta
        return torch.stack(preds, dim=2)


def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    mask = (y_mask & agent_mask.unsqueeze(-1)).unsqueeze(-1)
    diff = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    denom = mask.to(dtype=pred.dtype).sum().clamp_min(1.0) * pred.shape[-1]
    return (diff * mask.to(dtype=pred.dtype)).sum() / denom


def masked_ade_fde(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[float, float]:
    valid = y_mask & agent_mask.unsqueeze(-1)
    displacement = torch.linalg.norm(pred - target, dim=-1)
    ade = (displacement * valid.to(dtype=pred.dtype)).sum() / valid.to(dtype=pred.dtype).sum().clamp_min(1.0)

    final_valid = valid[..., -1]
    fde = (
        displacement[..., -1] * final_valid.to(dtype=pred.dtype)
    ).sum() / final_valid.to(dtype=pred.dtype).sum().clamp_min(1.0)
    return float(ade.detach().cpu().item()), float(fde.detach().cpu().item())

