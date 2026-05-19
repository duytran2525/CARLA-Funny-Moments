from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultiAgentModelConfig:
    # Baseline parameters
    input_dim: int = 6
    hidden_dim: int = 128
    graph_layers: int = 2
    future_steps: int = 30
    dropout: float = 0.1
    
    # GAT parameters
    enable_gat: bool = False
    num_attention_heads: int = 4
    attention_dropout: float = 0.1
    attention_concat_mode: str = "concat"  # "concat" or "average"
    
    # Multimodal parameters
    enable_multimodal: bool = False
    num_modes: int = 3
    
    # Adaptive radius parameters
    enable_adaptive_radius: bool = False
    radius_base: float = 20.0
    radius_alpha: float = 0.5
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_attention_heads < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {self.num_attention_heads}")
        if self.attention_concat_mode not in ["concat", "average"]:
            raise ValueError(
                f"attention_concat_mode must be 'concat' or 'average', got '{self.attention_concat_mode}'"
            )
        if self.num_modes < 1:
            raise ValueError(f"num_modes must be >= 1, got {self.num_modes}")
        if self.radius_base <= 0:
            raise ValueError(f"radius_base must be > 0, got {self.radius_base}")
        if self.radius_alpha < 0:
            raise ValueError(f"radius_alpha must be >= 0, got {self.radius_alpha}")
    
    def to_json(self) -> dict:
        """Serialize configuration to JSON-compatible dictionary."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "graph_layers": self.graph_layers,
            "future_steps": self.future_steps,
            "dropout": self.dropout,
            "enable_gat": self.enable_gat,
            "num_attention_heads": self.num_attention_heads,
            "attention_dropout": self.attention_dropout,
            "attention_concat_mode": self.attention_concat_mode,
            "enable_multimodal": self.enable_multimodal,
            "num_modes": self.num_modes,
            "enable_adaptive_radius": self.enable_adaptive_radius,
            "radius_base": self.radius_base,
            "radius_alpha": self.radius_alpha,
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "MultiAgentModelConfig":
        """Deserialize configuration from JSON-compatible dictionary."""
        return cls(**data)
    
    @classmethod
    def gat_config(cls) -> "MultiAgentModelConfig":
        """Factory method: GAT-enabled configuration."""
        return cls(enable_gat=True)
    
    @classmethod
    def multimodal_config(cls) -> "MultiAgentModelConfig":
        """Factory method: Multimodal-enabled configuration."""
        return cls(enable_multimodal=True)
    
    @classmethod
    def full_config(cls) -> "MultiAgentModelConfig":
        """Factory method: Full configuration with all improvements enabled."""
        return cls(
            enable_gat=True,
            enable_multimodal=True,
            enable_adaptive_radius=True
        )


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


class GATLayer(nn.Module):
    """Graph Attention Network layer with multi-head attention.
    
    Implements attention-based message aggregation where each agent learns
    to weight the importance of its neighbors. Supports multiple attention
    heads to capture diverse interaction patterns.
    
    Args:
        hidden_dim: Feature dimension per node
        num_heads: Number of parallel attention heads (default: 4)
        dropout: Dropout probability for attention weights (default: 0.1)
        concat_heads: If True, concatenate heads; else average (default: True)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_heads = concat_heads
        
        # When concatenating, each head produces hidden_dim // num_heads features
        # When averaging, each head produces hidden_dim features
        if concat_heads:
            assert hidden_dim % num_heads == 0, \
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}) when concat_heads=True"
            self.head_dim = hidden_dim // num_heads
            self.out_dim = hidden_dim
        else:
            self.head_dim = hidden_dim
            self.out_dim = hidden_dim
        
        # Learnable weight matrices for each head
        # W transforms node features: h_i -> W * h_i
        self.W = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        ])
        
        # Attention mechanism: a^T [W*h_i || W*h_j]
        # a is a learnable vector of size 2 * head_dim
        self.attention = nn.ModuleList([
            nn.Linear(2 * self.head_dim, 1, bias=False)
            for _ in range(num_heads)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Residual connection and layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # If not concatenating, we need a projection to combine heads
        if not concat_heads:
            self.head_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(
        self,
        h: torch.Tensor,  # [batch_size, max_agents, hidden_dim]
        adj: torch.Tensor,  # [batch_size, max_agents, max_agents]
        agent_mask: torch.Tensor  # [batch_size, max_agents]
    ) -> torch.Tensor:  # [batch_size, max_agents, hidden_dim]
        """
        Compute attention-weighted message aggregation.
        
        Args:
            h: Node features [batch_size, max_agents, hidden_dim]
            adj: Adjacency matrix [batch_size, max_agents, max_agents]
                 1 if connected, 0 otherwise
            agent_mask: Valid agent mask [batch_size, max_agents]
                       True for valid agents, False for padding
        
        Returns:
            Updated node features with same shape as input h
        """
        batch_size, max_agents, _ = h.shape
        device = h.device
        dtype = h.dtype
        
        # Convert masks to float for computation
        mask = agent_mask.to(dtype=dtype)  # [B, N]
        adj_float = adj.to(dtype=dtype)  # [B, N, N]
        
        # Mask adjacency: zero out connections to/from invalid agents
        adj_masked = adj_float * mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]
        
        # Add self-connections for valid agents
        eye = torch.eye(max_agents, device=device, dtype=dtype).unsqueeze(0)  # [1, N, N]
        adj_masked = adj_masked + eye * mask.unsqueeze(1)  # [B, N, N]
        
        # Compute attention for each head
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            # Transform features: [B, N, hidden_dim] -> [B, N, head_dim]
            h_transformed = self.W[head_idx](h)  # [B, N, head_dim]
            
            # Compute attention scores for all pairs
            # e_ij = a^T [W*h_i || W*h_j]
            
            # Expand for broadcasting: [B, N, 1, head_dim] and [B, 1, N, head_dim]
            h_i = h_transformed.unsqueeze(2)  # [B, N, 1, head_dim]
            h_j = h_transformed.unsqueeze(1)  # [B, 1, N, head_dim]
            
            # Concatenate: [B, N, N, 2*head_dim]
            h_concat = torch.cat([
                h_i.expand(batch_size, max_agents, max_agents, self.head_dim),
                h_j.expand(batch_size, max_agents, max_agents, self.head_dim)
            ], dim=-1)
            
            # Compute attention coefficients: [B, N, N, 2*head_dim] -> [B, N, N, 1] -> [B, N, N]
            e = self.attention[head_idx](h_concat).squeeze(-1)  # [B, N, N]
            e = self.leaky_relu(e)  # [B, N, N]
            
            # Mask attention scores: set to -inf where no edge exists
            # This ensures softmax gives 0 weight to non-neighbors
            mask_value = -1e9
            e_masked = torch.where(adj_masked > 0, e, torch.tensor(mask_value, dtype=dtype, device=device))
            
            # Apply softmax to get attention weights (normalized over neighbors)
            alpha = torch.softmax(e_masked, dim=-1)  # [B, N, N]
            
            # Apply dropout to attention weights
            alpha = self.dropout_layer(alpha)  # [B, N, N]
            
            # Aggregate messages: weighted sum of neighbor features
            # [B, N, N] @ [B, N, head_dim] -> [B, N, head_dim]
            h_prime = torch.bmm(alpha, h_transformed)  # [B, N, head_dim]
            
            head_outputs.append(h_prime)
        
        # Combine multi-head outputs
        if self.concat_heads:
            # Concatenate: [B, N, num_heads * head_dim] = [B, N, hidden_dim]
            h_out = torch.cat(head_outputs, dim=-1)  # [B, N, hidden_dim]
        else:
            # Average: [B, N, hidden_dim]
            h_out = torch.stack(head_outputs, dim=0).mean(dim=0)  # [B, N, hidden_dim]
            h_out = self.head_projection(h_out)  # [B, N, hidden_dim]
        
        # Residual connection and layer normalization
        h_out = self.norm(h + h_out)  # [B, N, hidden_dim]
        
        # Mask output for invalid agents
        h_out = h_out * mask.unsqueeze(-1)  # [B, N, hidden_dim]
        
        return h_out


class MultimodalDecoder(nn.Module):
    """Multimodal decoder with K parallel decoder heads.
    
    Predicts K alternative future trajectories using separate GRU decoder heads
    and MLP position delta predictors for each mode. Each mode operates
    independently during autoregressive decoding.
    
    Args:
        hidden_dim: Encoder output dimension
        num_modes: Number of trajectory hypotheses (K)
        future_steps: Prediction horizon length
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_modes: int = 3,
        future_steps: int = 30
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        
        # K separate GRUCell decoders (one per mode)
        self.decoder_cells = nn.ModuleList([
            nn.GRUCell(input_size=2, hidden_size=hidden_dim)
            for _ in range(num_modes)
        ])
        
        # K separate MLP heads for position delta prediction
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2),
            )
            for _ in range(num_modes)
        ])
    
    def forward(
        self,
        h: torch.Tensor,  # [batch_size, max_agents, hidden_dim]
        last_pos: torch.Tensor,  # [batch_size, max_agents, 2]
        agent_mask: torch.Tensor  # [batch_size, max_agents]
    ) -> torch.Tensor:  # [batch_size, max_agents, num_modes, future_steps, 2]
        """
        Predict K alternative future trajectories.
        
        Args:
            h: Encoded hidden states [batch_size, max_agents, hidden_dim]
            last_pos: Last valid position from history [batch_size, max_agents, 2]
            agent_mask: Valid agent mask [batch_size, max_agents]
        
        Returns:
            Multimodal predictions with shape [B, N, K, T, 2]
        """
        batch_size, max_agents, _ = h.shape
        device = h.device
        dtype = h.dtype
        
        # Flatten batch and agents for processing
        # [B, N, H] -> [B*N, H]
        state_init = h.reshape(batch_size * max_agents, self.hidden_dim)
        prev_pos_init = last_pos.reshape(batch_size * max_agents, 2)
        flat_agent_mask = agent_mask.reshape(batch_size * max_agents).to(dtype=dtype).unsqueeze(-1)
        
        # Zero out initial positions for masked agents
        prev_pos_init = prev_pos_init * flat_agent_mask  # [B*N, 2]
        
        # Store predictions for all modes
        all_mode_preds = []  # Will be list of [B*N, T, 2] tensors
        
        # Decode each mode independently
        for mode_idx in range(self.num_modes):
            # Initialize state and position for this mode
            state = state_init.clone()  # [B*N, H]
            prev_pos = prev_pos_init.clone()  # [B*N, 2]
            prev_delta = torch.zeros_like(prev_pos)  # [B*N, 2]
            
            mode_preds = []  # Will store [B*N, 2] for each timestep
            
            # Autoregressive decoding for this mode
            for _ in range(self.future_steps):
                # Update state with previous delta
                state = self.decoder_cells[mode_idx](prev_delta, state)  # [B*N, H]
                
                # Predict position delta
                delta = self.delta_heads[mode_idx](state)  # [B*N, 2]
                
                # Mask invalid agents
                delta = delta * flat_agent_mask  # [B*N, 2]
                
                # Update position
                prev_pos = prev_pos + delta  # [B*N, 2]
                
                # Store prediction
                mode_preds.append(prev_pos)
                
                # Update previous delta for next step
                prev_delta = delta
            
            # Stack timesteps: [B*N, T, 2]
            mode_preds_tensor = torch.stack(mode_preds, dim=1)  # [B*N, T, 2]
            all_mode_preds.append(mode_preds_tensor)
        
        # Stack modes: [K, B*N, T, 2]
        all_modes_tensor = torch.stack(all_mode_preds, dim=0)  # [K, B*N, T, 2]
        
        # Reshape to [B, N, K, T, 2]
        output = all_modes_tensor.permute(1, 0, 2, 3)  # [B*N, K, T, 2]
        output = output.reshape(batch_size, max_agents, self.num_modes, self.future_steps, 2)
        
        return output


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
        
        # Use GAT layers if enabled, otherwise use GraphInteractionBlock
        if self.config.enable_gat:
            self.graph_blocks = nn.ModuleList(
                [
                    GATLayer(
                        hidden_dim=int(self.config.hidden_dim),
                        num_heads=int(self.config.num_attention_heads),
                        dropout=float(self.config.attention_dropout),
                        concat_heads=(self.config.attention_concat_mode == "concat"),
                    )
                    for _ in range(max(0, int(self.config.graph_layers)))
                ]
            )
        else:
            self.graph_blocks = nn.ModuleList(
                [
                    GraphInteractionBlock(
                        hidden_dim=int(self.config.hidden_dim),
                        dropout=float(self.config.dropout),
                    )
                    for _ in range(max(0, int(self.config.graph_layers)))
                ]
            )
        
        # Use MultimodalDecoder if enabled, otherwise use single decoder
        if self.config.enable_multimodal:
            self.multimodal_decoder = MultimodalDecoder(
                hidden_dim=int(self.config.hidden_dim),
                num_modes=int(self.config.num_modes),
                future_steps=int(self.config.future_steps),
            )
        else:
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

        # Get last valid positions for decoder initialization
        last_pos = self._last_valid_positions(x, x_mask)  # [B, N, 2]
        
        # Use multimodal decoder if enabled, otherwise use single decoder
        if self.config.enable_multimodal:
            # Multimodal prediction: [B, N, K, T, 2]
            return self.multimodal_decoder(h, last_pos, agent_mask)
        else:
            # Unimodal prediction: [B, N, T, 2]
            state = h.reshape(batch_size * max_agents, int(self.config.hidden_dim))
            prev_pos = last_pos.reshape(batch_size * max_agents, 2)
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


def wta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    """Winner-Takes-All loss for multimodal trajectory prediction.
    
    Computes per-mode errors using smooth L1 loss, selects the best mode per agent
    (argmin over modes), and applies loss only to the best-matching mode.
    
    Args:
        pred: Predicted trajectories [batch_size, max_agents, num_modes, future_steps, 2]
        target: Ground truth trajectories [batch_size, max_agents, future_steps, 2]
        y_mask: Valid future frames [batch_size, max_agents, future_steps]
        agent_mask: Valid agents [batch_size, max_agents]
    
    Returns:
        Scalar loss tensor
    
    **Validates: Requirements 2.3, 2.4, 2.5**
    """
    # pred shape: [B, N, K, T, 2]
    # target shape: [B, N, T, 2]
    batch_size, max_agents, num_modes, future_steps, _ = pred.shape
    
    # Expand target to match pred shape: [B, N, T, 2] -> [B, N, K, T, 2]
    target_expanded = target.unsqueeze(2).expand(-1, -1, num_modes, -1, -1)
    
    # Compute mask: [B, N, T] -> [B, N, 1, T, 1] for broadcasting
    mask = (y_mask & agent_mask.unsqueeze(-1)).unsqueeze(2).unsqueeze(-1)
    
    # Compute per-mode smooth L1 loss: [B, N, K, T, 2]
    per_mode_errors = torch.nn.functional.smooth_l1_loss(
        pred, target_expanded, reduction="none"
    )
    
    # Apply mask and sum over timesteps and coordinates: [B, N, K]
    # Shape: [B, N, K, T, 2] * [B, N, 1, T, 1] -> [B, N, K, T, 2] -> sum -> [B, N, K]
    masked_errors = per_mode_errors * mask.to(dtype=pred.dtype)
    per_mode_total_error = masked_errors.sum(dim=(3, 4))  # Sum over T and 2
    
    # Select best mode per agent (argmin over modes): [B, N]
    best_mode_indices = torch.argmin(per_mode_total_error, dim=2)
    
    # Gather the loss from the best mode for each agent
    # best_mode_indices shape: [B, N] -> [B, N, 1] for gather
    best_mode_indices_expanded = best_mode_indices.unsqueeze(2)
    
    # Gather: [B, N, K] -> [B, N, 1] -> [B, N]
    best_mode_errors = torch.gather(per_mode_total_error, dim=2, index=best_mode_indices_expanded).squeeze(2)
    
    # Apply agent mask and normalize
    agent_mask_float = agent_mask.to(dtype=pred.dtype)
    masked_best_errors = best_mode_errors * agent_mask_float
    
    # Count valid elements for normalization
    # Each valid agent contributes (num_valid_timesteps * 2) elements
    valid_elements_per_agent = mask.squeeze(2).squeeze(-1).to(dtype=pred.dtype).sum(dim=2)  # [B, N]
    total_valid_elements = (valid_elements_per_agent * agent_mask_float).sum().clamp_min(1.0)
    
    # Return normalized loss
    return masked_best_errors.sum() / total_valid_elements


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


def load_checkpoint_with_compatibility(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    target_config: Optional[MultiAgentModelConfig] = None,
) -> tuple[MultiAgentTrajectoryPredictor, dict]:
    """Load checkpoint with backward compatibility for legacy checkpoints.
    
    This function handles loading both legacy checkpoints (without GAT/multimodal)
    and new checkpoints. It detects the checkpoint version and converts legacy
    checkpoints to the new architecture by initializing missing components with
    random weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to (default: "cpu")
        target_config: Optional target configuration. If provided, the model will
                      be created with this config. If None, uses config from checkpoint.
    
    Returns:
        Tuple of (model, checkpoint_dict) where:
        - model: Loaded MultiAgentTrajectoryPredictor with state restored
        - checkpoint_dict: Full checkpoint dictionary (for optimizer/scheduler state)
    
    **Validates: Requirements 10.1, 10.2, 10.3**
    
    Example:
        >>> # Load legacy checkpoint and upgrade to GAT+multimodal
        >>> target_cfg = MultiAgentModelConfig.full_config()
        >>> model, ckpt = load_checkpoint_with_compatibility("old_model.pt", target_config=target_cfg)
        >>> # New GAT and multimodal components are initialized with random weights
        
        >>> # Load new checkpoint normally
        >>> model, ckpt = load_checkpoint_with_compatibility("new_model.pt")
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint
    checkpoint_config_dict = checkpoint.get("model_config", None)
    
    # Detect checkpoint version
    if checkpoint_config_dict is None:
        # Legacy checkpoint without model_config field
        # Assume baseline configuration (all improvements disabled)
        print("⚠️  Legacy checkpoint detected (no model_config field)")
        print("    Assuming baseline configuration (enable_gat=False, enable_multimodal=False)")
        checkpoint_config = MultiAgentModelConfig()
        checkpoint_version = "1.0"
    else:
        # New checkpoint with model_config field
        checkpoint_config = MultiAgentModelConfig.from_json(checkpoint_config_dict)
        checkpoint_version = checkpoint_config_dict.get("version", "1.0")
        print(f"✓ Checkpoint version: {checkpoint_version}")
        print(f"  Config: GAT={checkpoint_config.enable_gat}, "
              f"Multimodal={checkpoint_config.enable_multimodal}, "
              f"AdaptiveRadius={checkpoint_config.enable_adaptive_radius}")
    
    # Determine target configuration
    if target_config is None:
        # Use checkpoint config as-is
        target_config = checkpoint_config
        print("  Using checkpoint configuration")
    else:
        # User specified a target config (e.g., upgrading legacy checkpoint)
        print(f"  Target config: GAT={target_config.enable_gat}, "
              f"Multimodal={target_config.enable_multimodal}, "
              f"AdaptiveRadius={target_config.enable_adaptive_radius}")
    
    # Create model with target configuration
    model = MultiAgentTrajectoryPredictor(config=target_config)
    model = model.to(device)
    
    # Load state dict with compatibility handling
    state_dict = checkpoint["model_state_dict"]
    
    # Check if we need to handle architecture mismatch
    needs_conversion = (
        checkpoint_config.enable_gat != target_config.enable_gat or
        checkpoint_config.enable_multimodal != target_config.enable_multimodal
    )
    
    if needs_conversion:
        print("⚠️  Architecture mismatch detected - performing conversion")
        
        # Load compatible parameters (strict=False allows missing/unexpected keys)
        load_result = model.load_state_dict(state_dict, strict=False)
        
        # Report what was loaded and what was initialized
        if load_result.missing_keys:
            print(f"  ✓ Initialized new components with random weights:")
            for key in load_result.missing_keys:
                print(f"    - {key}")
        
        if load_result.unexpected_keys:
            print(f"  ⚠️  Ignored old components (not in target architecture):")
            for key in load_result.unexpected_keys:
                print(f"    - {key}")
        
        print("  ✓ Conversion complete")
    else:
        # Architectures match - load normally with strict=True
        model.load_state_dict(state_dict, strict=True)
        print("  ✓ Loaded checkpoint (architectures match)")
    
    return model, checkpoint

