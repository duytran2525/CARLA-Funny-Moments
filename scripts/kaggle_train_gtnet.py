"""
kaggle_train_gtnet.py  â€” v3 (GTNet improved)
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Cáº¢I TIáº¾N so vá»›i v2:

[IMP-A] WarmupCosineScheduler (má»›i)
    Linear warmup trong --warmup-epochs epoch Ä‘áº§u â†’ cosine decay.
    TrĂ¡nh LR Ä‘á»™t ngá»™t cao ngay epoch 1 gĂ¢y instability.
    Há»— trá»£ nhiá»u param groups (GAT dual-LR) tá»± Ä‘á»™ng.
    Thay tháº¿ CosineAnnealingLR khi --cosine-lr Ä‘Æ°á»£c báº­t.

[IMP-B] Diversity loss (má»›i)
    Penalize cĂ¡c mode cĂ³ endpoint quĂ¡ gáº§n nhau (khoáº£ng cĂ¡ch < margin).
    NgÄƒn mode collapse khi dĂ¹ng WTA vá»›i K=5.
    KĂ­ch hoáº¡t báº±ng --diversity-weight > 0 (khuyáº¿n nghá»‹ 0.05â€“0.1).

[IMP-C] Data augmentation (má»›i)
    _augment_batch(): Ă¡p dá»¥ng trong training loop:
      â€¢ Random in-plane rotation (Â±rot_std_degÂ°) trĂªn x, y, vx, vy, heading
        vĂ  target y â†’ tÄƒng cÆ°á»ng tĂ­nh invariance vá»›i gĂ³c quay ego
      â€¢ History frame dropout: ngáº«u nhiĂªn mask má»™t sá»‘ frame lá»‹ch sá»­
        (giá»¯ nguyĂªn anchor frame cuá»‘i) â†’ mĂ´ hĂ¬nh robust hÆ¡n vá»›i missing obs.
    KĂ­ch hoáº¡t báº±ng --augment.

[IMP-D] Early stopping dá»±a trĂªn val_ADE (thay vĂ¬ val_loss)
    val_ADE = metric trá»±c tiáº¿p pháº£n Ă¡nh cháº¥t lÆ°á»£ng dá»± Ä‘oĂ¡n quá»¹ Ä‘áº¡o.
    val_loss bá»‹ áº£nh hÆ°á»Ÿng bá»Ÿi scale cá»§a smooth-L1 â†’ khĂ´ng luĂ´n monotone
    vá»›i ADE. ThĂªm --early-stop-metric (loss|ade, default: ade).
    LÆ°u riĂªng best_ade.pt vĂ  best_loss.pt.

[IMP-E] Best result tracking chĂ­nh xĂ¡c
    best_val_ade vĂ  best_val_fde trong káº¿t quáº£ giá» Ä‘Æ°á»£c láº¥y tá»« CĂ™NG epoch
    vá»›i best checkpoint, khĂ´ng pháº£i min() Ä‘á»™c láº­p qua toĂ n history.

[IMP-F] New config fields forwarded to build_model()
    --use-temporal-attention, --mode-embed-dim, --encoder-dropout
    â†’ tá»± Ä‘á»™ng forward vĂ o MultiAgentModelConfig.

KHAI THĂC Tá»ª PHĂ‚N TĂCH 39 EPOCHS:
  â€¢ Train/val ADE gap = 3.5Ă— â†’ [IMP-C] augmentation + [IMP-F] encoder_dropout
  â€¢ Val plateau sau epoch 24 â†’ [IMP-A] warmup giĂºp giai Ä‘oáº¡n Ä‘áº§u tá»‘t hÆ¡n
  â€¢ K=5 WTA risk collapse â†’ [IMP-B] diversity loss
  â€¢ LR khĂ´ng warmup â†’ [IMP-A]
  â€¢ Early stop dá»±a loss â†’ [IMP-D] dá»±a ADE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# â”€â”€ AMP compatibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def _make_scaler(enabled: bool) -> _GradScaler:
        return _GradScaler("cuda", enabled=enabled, init_scale=2.0 ** 12)

    def _make_autocast(enabled: bool) -> _autocast:
        return _autocast("cuda", enabled=enabled)

except (ImportError, TypeError):
    from torch.cuda.amp import GradScaler as _LegacyScaler       # type: ignore
    from torch.cuda.amp import autocast as _LegacyAutocast        # type: ignore

    def _make_scaler(enabled: bool) -> _LegacyScaler:             # type: ignore
        return _LegacyScaler(enabled=enabled, init_scale=2.0 ** 12)

    def _make_autocast(enabled: bool) -> _LegacyAutocast:         # type: ignore
        return _LegacyAutocast(enabled=enabled)


# â”€â”€ sys.path setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _candidate_project_roots() -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get("GTNET_PROJECT_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser())

    roots.extend([PROJECT_ROOT, PROJECT_ROOT.parent, Path("/kaggle/working")])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        roots.append(kaggle_input)
        for pattern in ("*", "*/*", "*/*/*"):
            for candidate in kaggle_input.glob(pattern):
                if candidate.is_dir() and (candidate / "core_perception").exists():
                    roots.append(candidate)
    return roots


for _root in _candidate_project_roots():
    try:
        root = _root.resolve()
    except OSError:
        root = _root
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from core_perception.multi_agent_dataset import (  # noqa: E402
    MultiAgentTrajectoryDataset,
    collate_multi_agent_trajectory,
    split_sample_paths,
)
from core_perception.multi_agent_model import (  # noqa: E402
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
    masked_ade_fde,
    masked_smooth_l1_loss,
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Multimodal-aware loss & metrics (unchanged from v2)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    diversity_weight: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """
    Compute training loss.

    Unimodal  [B, A, T, 2]   â†’ masked smooth-L1
    Multimodal [B, A, K, T, 2] â†’ Winner-Takes-All + optional diversity term

    [IMP-B] diversity_weight > 0 adds a pairwise endpoint repulsion loss
    that penalizes modes whose final positions are too close together.
    This prevents mode collapse when K is large (e.g. K=5).

    Returns:
        (total_loss, div_loss_scalar) â€” div_loss_scalar is 0.0 for unimodal
        or when diversity_weight == 0. Caller uses it for separate logging.
    """
    if pred.dim() == 4:
        return masked_smooth_l1_loss(pred, target, y_mask, agent_mask), 0.0

    # â”€â”€ Multimodal WTA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    B, A, K, T, C = pred.shape
    target_exp = target.unsqueeze(2).expand(B, A, K, T, C)

    valid_agent = agent_mask & y_mask.any(dim=-1)
    valid = (y_mask & valid_agent.unsqueeze(-1)).unsqueeze(2).unsqueeze(-1)
    valid_f = valid.to(dtype=pred.dtype)

    per_mode_loss = F.smooth_l1_loss(pred, target_exp, reduction="none")
    per_mode_loss = per_mode_loss * valid_f
    valid_steps = valid_f.squeeze(-1).sum(dim=-1).clamp_min(1.0)
    per_mode_loss = per_mode_loss.sum(dim=(-1, -2)) / (valid_steps * C)

    wta_loss, _ = per_mode_loss.min(dim=-1)
    mask = valid_agent.float()
    n_valid = mask.sum().clamp_min(1.0)
    main_loss = (wta_loss * mask).sum() / n_valid

    if diversity_weight > 0.0 and K > 1:
        div = _diversity_loss(pred, agent_mask, y_mask)
        return main_loss + diversity_weight * div, float(div.detach().cpu().item())
    return main_loss, 0.0


def _last_valid_displacement(
    displacement: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    valid = y_mask & agent_mask.unsqueeze(-1)
    T = displacement.shape[-1]
    last_valid_idx = (T - 1) - valid.flip(dims=[-1]).long().argmax(dim=-1)

    if displacement.dim() == 3:
        return displacement.gather(-1, last_valid_idx.unsqueeze(-1)).squeeze(-1)

    B, A, K, _ = displacement.shape
    idx = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, A, K, 1)
    return displacement.gather(-1, idx).squeeze(-1)


def _compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    miss_threshold: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ADE, FDE, and MissRate.

    Returns:
        (ade, fde, miss_rate) where miss_rate is the fraction of valid
        agents whose best-mode FDE exceeds *miss_threshold* metres.
    """
    valid_agent = agent_mask & y_mask.any(dim=-1)

    if pred.dim() == 4:
        disp = torch.norm(pred - target, dim=-1)
        valid = y_mask & agent_mask.unsqueeze(-1)
        valid_f = valid.to(dtype=pred.dtype)
        ade = (disp * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        fde_per_agent = _last_valid_displacement(disp, y_mask, agent_mask)
        mask = valid_agent.to(dtype=pred.dtype)
        n_valid = mask.sum().clamp_min(1.0)
        fde = (fde_per_agent * mask).sum() / n_valid
        miss = ((fde_per_agent > miss_threshold) & valid_agent).float().sum() / n_valid
        return ade, fde, miss

    B, A, K, T, C = pred.shape
    target_exp = target.unsqueeze(2).expand(B, A, K, T, C)
    disp = torch.norm(pred - target_exp, dim=-1)

    valid = (y_mask & agent_mask.unsqueeze(-1)).unsqueeze(2)
    valid_f = valid.to(dtype=pred.dtype)
    ade_per_mode = (disp * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)
    fde_per_mode = _last_valid_displacement(disp, y_mask, agent_mask)

    best_mode = ade_per_mode.argmin(dim=-1, keepdim=True)
    min_ade = ade_per_mode.gather(-1, best_mode).squeeze(-1)
    min_fde = fde_per_mode.gather(-1, best_mode).squeeze(-1)

    mask = valid_agent.float()
    n_valid = mask.sum().clamp_min(1.0)
    miss = ((min_fde > miss_threshold) & valid_agent).float().sum() / n_valid
    return (min_ade * mask).sum() / n_valid, (min_fde * mask).sum() / n_valid, miss


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# [IMP-B] Mode diversity loss
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _diversity_loss(
    pred: torch.Tensor,
    agent_mask: torch.Tensor,
    y_mask: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Pairwise endpoint repulsion loss for multimodal predictions.

    Penalizes pairs of modes whose final predicted positions are within
    `margin` metres of each other. This prevents mode collapse â€” a
    common failure mode for WTA loss when K is large.

    Loss per pair (i, j): ReLU(margin - ||endpoint_i - endpoint_j||)
    Averaged over all K*(K-1)/2 pairs and all valid agents.

    Args:
        pred:       [B, A, K, T, 2]
        agent_mask: [B, A]
        y_mask:     [B, A, T]
        margin:     minimum desired endpoint separation in metres (default 0.5)

    Returns:
        Scalar diversity loss (0 if K <= 1)
    """
    if pred.dim() != 5:
        return pred.new_tensor(0.0)
    B, A, K, T, C = pred.shape
    if K <= 1:
        return pred.new_tensor(0.0)

    valid = (agent_mask & y_mask.any(dim=-1)).float()  # [B, A]
    n_valid = valid.sum().clamp_min(1.0)

    # [FIX-3] Use last *valid* timestep per agent (not always index -1).
    # y_mask: [B, A, T] â€” True = valid future step.
    # For agents with all-False mask, last_idx falls back to T-1 (harmless,
    # they are excluded from the loss via the valid weight).
    valid_counts = y_mask.long().sum(dim=-1).clamp_min(1)   # [B, A]
    last_idx = (valid_counts - 1)                            # [B, A]
    # Expand to [B, A, K, 1, 2] for gather on pred [B, A, K, T, 2]
    last_idx_exp = last_idx.view(B, A, 1, 1, 1).expand(B, A, K, 1, C)
    endpoints = pred.gather(3, last_idx_exp).squeeze(3)      # [B, A, K, 2]

    total = pred.new_tensor(0.0)
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.norm(
                endpoints[:, :, i] - endpoints[:, :, j], dim=-1
            )  # [B, A]
            repulsion = F.relu(margin - dist)
            total = total + (repulsion * valid).sum() / n_valid
            n_pairs += 1

    return total / max(1, n_pairs)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# [IMP-C] Data augmentation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _augment_batch(
    batch: Dict,
    rot_std_deg: float = 10.0,
    hist_dropout: float = 0.1,
) -> Dict:
    """
    Apply random in-plane rotation and history frame dropout.

    Random rotation:
        Rotates features in ego-frame by Â±rot_std_degÂ° (sampled per sample).
        Applied consistently to all feature channels:
          x[..., 0:2]  â€” local position (x, y)
          x[..., 2:4]  â€” local velocity (vx, vy)   [if feature_dim >= 4]
          x[..., 4:6]  â€” heading unit vector        [if feature_dim >= 6]
          y[..., 0:2]  â€” target position (x, y)
        Adjacency matrix is unchanged (distances are rotation-invariant).
        Rationale: CARLA ego can spawn at any yaw; augmentation forces the
        model to be invariant to small ego orientation perturbations.

    History frame dropout:
        Randomly masks hist_dropout fraction of valid history frames
        (excluding the anchor frame at index -1).
        Rationale: simulates partial observability / sensor gaps, reduces
        the train/val ADE gap by making the model more robust to missing obs.

    Args:
        batch:         collated batch dict
        rot_std_deg:   standard deviation of rotation noise in degrees
        hist_dropout:  probability of dropping any single history frame

    Returns:
        Augmented batch dict (shallow copy, tensors cloned as needed)
    """
    batch = dict(batch)  # shallow copy so we can replace keys
    B = batch["x"].shape[0]
    device = batch["x"].device

    # â”€â”€ Random rotation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if rot_std_deg > 0.0:
        angles = torch.randn(B, device=device) * math.radians(rot_std_deg)
        cos_a = angles.cos()  # [B]
        sin_a = angles.sin()

        def _rot2d(t: torch.Tensor, ch: int) -> torch.Tensor:
            """Rotate 2D channels (ch, ch+1) of tensor t. Returns clone."""
            t = t.clone()
            shape = (-1,) + (1,) * (t.dim() - 2)  # broadcast to leading dims
            c = cos_a.view(shape)
            s = sin_a.view(shape)
            x_ch = t[..., ch].clone()
            y_ch = t[..., ch + 1].clone()
            t[..., ch]     = c * x_ch - s * y_ch
            t[..., ch + 1] = s * x_ch + c * y_ch
            return t

        feat_dim = batch["x"].shape[-1]
        x = batch["x"].clone()
        x = _rot2d(x, 0)           # channels 0-1: local position (always present)
        if feat_dim >= 4:
            # channels 2-3: local velocity (vx, vy) in new 6-ch format, OR
            # heading unit vector (hx, hy) in legacy 4-ch format.
            # Both are 2D vectors in ego-frame â†’ rotation applies identically.
            x = _rot2d(x, 2)
        if feat_dim >= 6:
            x = _rot2d(x, 4)       # channels 4-5: heading unit vector (new 6-ch format)
        batch["x"] = x

        # Rotate targets (always 2D position)
        batch["y"] = _rot2d(batch["y"].clone(), 0)

    # â”€â”€ History frame dropout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if hist_dropout > 0.0 and batch["x_mask"].shape[-1] > 1:
        x_mask = batch["x_mask"].clone()
        # Bernoulli dropout on currently-valid frames, but NEVER drop t=-1
        drop = (torch.rand_like(x_mask.float()) < hist_dropout) & x_mask
        drop[..., -1] = False       # protect anchor frame
        batch["x_mask"] = x_mask & ~drop

    return batch


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# [IMP-A] Warmup + Cosine scheduler
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class WarmupCosineScheduler:
    """
    Linear LR warmup then cosine annealing, with optional per-group start epoch.

    Drop-in replacement for CosineAnnealingLR. Works with multi-group
    optimizers (e.g. the GAT dual-LR setup) by scaling each group's
    initial LR independently.

    Schedule per group:
        epoch < start_epoch              : LR = 0  (group is frozen/inactive)
        start_epoch .. start_epoch+warmup: LR ramps linearly 0 â†’ base_lr
        after warmup                     : cosine decay from base_lr â†’ min_lr
                                           (cosine runs from start_epoch to total_epochs)

    Args:
        optimizer:         PyTorch optimizer (single or multi group)
        warmup_epochs:     default warmup epochs for all groups (0 = no warmup)
        total_epochs:      total training epochs (cosine endpoint)
        min_lr_ratio:      min LR = base_lr Ă— min_lr_ratio (default 0.01)
        group_start_epochs: dict {group_name: start_epoch} â€” groups whose name
                            matches will not receive any LR until start_epoch,
                            then get their own warmup from that epoch onward.
                            Groups not listed start at epoch 0 (normal behaviour).

    [FIX-3] group_start_epochs lets the GAT param group start warming up only
    after gat_freeze_epochs, so its cosine budget is calculated correctly and it
    never gets decayed before it has even started training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01,
        group_start_epochs: Optional[Dict[str, int]] = None,
    ) -> None:
        self.optimizer          = optimizer
        self.warmup_epochs      = warmup_epochs
        self.total_epochs       = total_epochs
        self.min_lr_ratio       = min_lr_ratio
        self.group_start_epochs = group_start_epochs or {}
        # Snapshot initial LRs for each group at construction time.
        self.base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        self._epoch   = 0

    def _group_start(self, pg: dict) -> int:
        """Return the start epoch for a given param group (0 if not listed)."""
        name = str(pg.get("name", ""))
        return self.group_start_epochs.get(name, 0)

    def step(self) -> None:
        self._epoch += 1
        e = self._epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            min_lr     = base_lr * self.min_lr_ratio
            start_ep   = self._group_start(pg)

            if e <= start_ep:
                # Group not yet active â€” keep LR at 0 (params are frozen anyway,
                # but setting it makes logs correct after unfreeze).
                pg["lr"] = 0.0
                continue

            # Local epoch relative to this group's start
            local_e = e - start_ep
            if self.warmup_epochs > 0 and local_e <= self.warmup_epochs:
                lr = base_lr * local_e / self.warmup_epochs
            else:
                # Cosine decays from start_ep to total_epochs
                cos_e   = local_e - self.warmup_epochs
                cos_max = max(1, self.total_epochs - start_ep - self.warmup_epochs)
                progress = cos_e / cos_max
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            pg["lr"] = max(float(lr), min_lr)

    def get_last_lr(self) -> List[float]:
        return [float(pg["lr"]) for pg in self.optimizer.param_groups]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Ablation variant definitions
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AblationVariant(NamedTuple):
    name: str
    code: str
    enable_gat: bool
    enable_multimodal: bool
    enable_adaptive_radius: bool
    description: str


ABLATION_VARIANTS: List[AblationVariant] = [
    AblationVariant("Baseline",   "000", False, False, False,
                    "GCN + unimodal + fixed radius (baseline)"),
    AblationVariant("GAT_only",   "100", True,  False, False,
                    "Adds Graph Attention â€” learns edge weights between agents"),
    AblationVariant("Multi_only", "010", False, True,  False,
                    "Adds multimodal output â€” K future trajectory hypotheses"),
    AblationVariant("AdpR_only",  "001", False, False, True,
                    "Adds adaptive radius â€” interaction range scales with speed"),
    AblationVariant("GAT_Multi",  "110", True,  True,  False,
                    "GAT + multimodal"),
    AblationVariant("GAT_AdpR",   "101", True,  False, True,
                    "GAT + adaptive radius"),
    AblationVariant("Multi_AdpR", "011", False, True,  True,
                    "Multimodal + adaptive radius"),
    AblationVariant("GTNet_Full", "111", True,  True,  True,
                    "All improvements enabled â€” full GTNet"),
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GTNet Kaggle trainer v3 (improved).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    required=True, nargs="+")
    p.add_argument("--out-dir",     default="/kaggle/working/models")
    p.add_argument("--mode", choices=["baseline", "full", "ablation", "per-town"],
                   default="full")

    # â”€â”€ Hyper-params â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument("--epochs",              type=int,   default=50)
    p.add_argument("--batch-size",          type=int,   default=64)
    p.add_argument("--accum-steps",         type=int,   default=1)
    p.add_argument("--learning-rate",       type=float, default=1e-3)
    p.add_argument("--weight-decay",        type=float, default=1e-4)
    p.add_argument("--train-ratio",         type=float, default=0.8)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--num-workers",         type=int,   default=2)

    # â”€â”€ Model architecture â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument("--hidden-dim",           type=int,   default=256)
    p.add_argument("--graph-layers",         type=int,   default=3)
    p.add_argument("--dropout",              type=float, default=0.1)
    p.add_argument("--num-modes",            type=int,   default=3)
    p.add_argument("--num-attention-heads",  type=int,   default=4)
    # [IMP-F] New architecture flags
    p.add_argument("--use-temporal-attention", action="store_true",
                   help="[IMP-1] Apply self-attention over GRU history sequence.")
    p.add_argument("--mode-embed-dim",  type=int,   default=64,
                   help="[IMP-2] Mode embedding dim in shared-GRU decoder.")
    p.add_argument("--encoder-dropout", type=float, default=0.0,
                   help="[IMP-3] Dropout on GRU encoder output. Reduces train/val gap.")
    # [V4-B] GAT edge feature dimension
    p.add_argument("--gat-edge-dim", type=int, default=32,
                   help="[V4-B] GAT edge feature dimension (0 = disable). "
                        "Enables RelativeEdgeEncoder for spatial awareness.")

    # â”€â”€ Feature flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument("--enable-gat",             action="store_true")
    p.add_argument("--enable-multimodal",      action="store_true")
    p.add_argument("--enable-adaptive-radius", action="store_true")

    # â”€â”€ Training config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p.add_argument("--grad-clip",                type=float, default=1.0)
    p.add_argument("--early-stopping-patience",  type=int,   default=10)
    p.add_argument("--lr-patience",              type=int,   default=4)
    p.add_argument("--cosine-lr",                action="store_true",
                   help="Use WarmupCosineScheduler instead of ReduceLROnPlateau.")
    # [IMP-A]
    p.add_argument("--warmup-epochs",  type=int,   default=5,
                   help="[IMP-A] Linear LR warmup epochs (0 = no warmup). "
                        "Only effective when --cosine-lr is set.")
    # [IMP-D]
    p.add_argument("--early-stop-metric", choices=["loss", "ade"], default="ade",
                   help="[IMP-D] Metric to monitor for early stopping and "
                        "best-checkpoint selection.")
    # [IMP-B]
    p.add_argument("--diversity-weight", type=float, default=0.0,
                   help="[IMP-B] Weight for mode diversity loss (0 = disabled). "
                        "Recommended: 0.05â€“0.1 when --enable-multimodal + K>=3.")
    p.add_argument("--diversity-ramp-epochs", type=int, default=20,
                   help="[IMP-D] Ramp diversity weight from 0 â†’ target over this "
                        "many epochs. Prevents diversity loss from destabilising "
                        "WTA training in early epochs. 0 = no ramp (full weight "
                        "from epoch 1).")
    # [IMP-C]
    p.add_argument("--augment",          action="store_true",
                   help="[IMP-C] Enable online data augmentation (rotation + "
                        "history dropout).")
    p.add_argument("--aug-rot-std",      type=float, default=10.0,
                   help="Std dev of rotation noise in degrees for augmentation.")
    p.add_argument("--aug-hist-dropout", type=float, default=0.1,
                   help="Probability of dropping each history frame (excl. anchor).")

    p.add_argument("--log-every",   type=int,   default=50)
    p.add_argument("--no-amp",      action="store_true")
    # GAT stability
    # [FIX-1] gat_lr_scale 0.1â†’0.5 so GAT actually moves during 30-epoch runs.
    # [FIX-1] gat_clip_scale 0.5â†’1.0 so the backbone is no longer clipped at 0.4.
    p.add_argument("--gat-lr-scale",   type=float, default=0.5,
                   help="GAT param-group LR = base_lr Ă— gat_lr_scale.")
    p.add_argument("--gat-clip-scale", type=float, default=1.0,
                   help="Unused when --gat-per-group-clip is set; kept for compat.")
    p.add_argument("--gat-per-group-clip", action="store_true",
                   help="[FIX-1] Clip backbone and GAT params separately "
                        "(backbone=grad_clip, GAT=grad_clipĂ—0.5 post-unfreeze).")
    # [FIX-2] Freeze GAT for first N epochs so backbone stabilises first.
    p.add_argument("--gat-freeze-epochs", type=int, default=5,
                   help="[FIX-2] Freeze GAT graph_blocks for this many epochs "
                        "before unfreezing with its own LR warmup. 0 = disabled.")
    p.add_argument("--nan-patience",   type=int,   default=3)
    p.add_argument("--debug-gat",      action="store_true")
    # Misc
    p.add_argument("--limit-samples",  type=int,  default=0)
    p.add_argument("--town-filter",    type=str,  default="")
    p.add_argument("--quick-ablation", action="store_true")
    return p.parse_args()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float(v: object) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)  # type: ignore[arg-type]


def make_target_metrics(
    min_ade: float,
    min_fde: float,
    miss_rate: float,
    latency_ms: float,
) -> Dict[str, float]:
    return {
        "minADE": float(min_ade),
        "minFDE": float(min_fde),
        "MissRate": float(miss_rate),
        "inference_latency_ms": float(latency_ms),
    }


def move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = dict(batch)
    for k in ("x", "y", "adj", "x_mask", "y_mask", "agent_mask"):
        moved[k] = batch[k].to(device, non_blocking=True)
    return moved


def config_to_dict(cfg: object) -> dict:
    if hasattr(cfg, "to_json") and callable(cfg.to_json):
        return cfg.to_json()
    raw = cfg.__dict__ if hasattr(cfg, "__dict__") else {}
    return {
        k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        for k, v in raw.items()
    }


def _is_gat_param_name(name: str) -> bool:
    return ".graph_blocks." in f".{name}."


def _grad_norms(model: nn.Module) -> Dict[str, float]:
    sums = {"total": 0.0, "gat": 0.0, "other": 0.0}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        if not torch.isfinite(g).all():
            norm_sq = float("nan")
        else:
            norm = float(g.norm(2).detach().cpu().item())
            norm_sq = norm * norm
        sums["total"] += norm_sq
        sums["gat" if _is_gat_param_name(name) else "other"] += norm_sq
    return {k: (math.sqrt(v) if not math.isnan(v) else float("nan"))
            for k, v in sums.items()}


def _print_separator(title: str = "", width: int = 72) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_sample_paths(data_dirs: List[Path], limit: int = 0) -> List[Path]:
    all_paths: List[Path] = []
    for d in data_dirs:
        ds = MultiAgentTrajectoryDataset(d)
        all_paths.extend(ds.sample_paths)
    if limit > 0:
        all_paths = all_paths[:limit]
    return all_paths


def filter_by_town(paths: List[Path], town: str) -> List[Path]:
    if not town:
        return paths
    return [p for p in paths if town.lower() in str(p).lower()]


def _build_path_root_map(
    paths: List[Path], data_dirs: List[Path]
) -> Dict[Path, Path]:
    mapping: Dict[Path, Path] = {}
    for path in paths:
        for d in data_dirs:
            try:
                path.relative_to(d)
                mapping[path] = d
                break
            except ValueError:
                pass
        else:
            mapping[path] = data_dirs[0]
    return mapping


def make_dataset(
    paths: List[Path], data_dirs: List[Path], root_map: Dict[Path, Path]
) -> MultiAgentTrajectoryDataset:
    unique = {root_map[p] for p in paths}
    root = unique.pop() if len(unique) == 1 else data_dirs[0]
    return MultiAgentTrajectoryDataset(root, sample_files=paths)


def build_loaders(
    train_paths: List[Path],
    val_paths: List[Path],
    data_dirs: List[Path],
    root_map: Dict[Path, Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, MultiAgentTrajectoryDataset]:
    train_ds = make_dataset(train_paths, data_dirs, root_map)
    val_ds   = make_dataset(val_paths,   data_dirs, root_map)
    kw = dict(
        batch_size=max(1, batch_size),
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_multi_agent_trajectory,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        train_ds,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(
    input_dim: int,
    future_steps: int,
    args: argparse.Namespace,
    variant: Optional[AblationVariant] = None,
) -> Tuple[MultiAgentTrajectoryPredictor, MultiAgentModelConfig]:
    gat   = variant.enable_gat             if variant else args.enable_gat
    multi = variant.enable_multimodal      if variant else args.enable_multimodal
    adp_r = variant.enable_adaptive_radius if variant else args.enable_adaptive_radius

    base_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=max(16, args.hidden_dim),
        graph_layers=max(0, args.graph_layers),
        future_steps=future_steps,
        dropout=args.dropout,
    )
    gtnet_kwargs = dict(
        enable_gat=gat,
        enable_multimodal=multi,
        enable_adaptive_radius=adp_r,
        num_modes=args.num_modes,
        num_attention_heads=args.num_attention_heads,
    )
    # [IMP-F] Forward new architecture params if the model supports them
    new_kwargs = {}
    if hasattr(MultiAgentModelConfig, "use_temporal_attention"):
        new_kwargs["use_temporal_attention"] = getattr(
            args, "use_temporal_attention", False
        )
    if hasattr(MultiAgentModelConfig, "mode_embed_dim"):
        new_kwargs["mode_embed_dim"] = getattr(args, "mode_embed_dim", 64)
    if hasattr(MultiAgentModelConfig, "encoder_dropout"):
        new_kwargs["encoder_dropout"] = getattr(args, "encoder_dropout", 0.0)
    # [V4-B] Forward GAT edge dimension
    if hasattr(MultiAgentModelConfig, "gat_edge_dim"):
        new_kwargs["gat_edge_dim"] = getattr(args, "gat_edge_dim", 32)

    try:
        cfg = MultiAgentModelConfig(**base_kwargs, **gtnet_kwargs, **new_kwargs)
    except TypeError:
        try:
            cfg = MultiAgentModelConfig(**base_kwargs, **gtnet_kwargs)
        except TypeError:
            cfg = MultiAgentModelConfig(**base_kwargs)

    model = MultiAgentTrajectoryPredictor(cfg)
    return model, cfg


def build_optimizer(
    model: nn.Module,
    *,
    base_lr: float,
    gat_lr: float,
    weight_decay: float,
    is_gat_variant: bool,
) -> torch.optim.Optimizer:
    if not is_gat_variant:
        return torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    gat_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (gat_params if _is_gat_param_name(name) else other_params).append(param)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "name": "base"})
    if gat_params:
        groups.append({"params": gat_params,   "lr": gat_lr,  "name": "gat"})
    if not groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _clip_gradients(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    per_group: bool = False,
    gat_clip_ratio: float = 0.5,
) -> None:
    """
    Clip gradients for all or per param-group.

    [FIX-1] per_group=True clips backbone and GAT separately:
      - backbone ("base" group): clipped at grad_clip
      - GAT ("gat" group):       clipped at grad_clip × gat_clip_ratio
    This prevents large GAT gradients (especially right after unfreeze)
    from destabilising the backbone through the global norm calculation.

    per_group=False: standard global clip_grad_norm_ (backward-compatible).
    """
    if grad_clip <= 0:
        return
    if not per_group:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        return

    for pg in optimizer.param_groups:
        name  = str(pg.get("name", ""))
        clip  = grad_clip * (gat_clip_ratio if name == "gat" else 1.0)
        params = [p for p in pg["params"] if p.grad is not None]
        if params:
            nn.utils.clip_grad_norm_(params, clip)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference latency measurement
# ═══════════════════════════════════════════════════════════════════════════════

def measure_inference_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 100,
    use_amp: bool = True,
) -> float:
    """Measure average inference latency in milliseconds per sample.

    Runs *num_batches* forward passes (after a short warm-up) and returns
    the mean wall-clock time per sample.  GPU synchronisation is inserted
    when running on CUDA so the measurement is accurate.
    """
    model.eval()
    latencies: List[float] = []

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= num_batches:
                break

            batch = move_batch(raw_batch, device)
            actual_batch_size = batch["x"].shape[0]

            # Warm-up (first batch only)
            if device.type == "cuda" and batch_idx == 0:
                for _ in range(5):
                    _ = model(
                        x=batch["x"], adj=batch["adj"],
                        x_mask=batch["x_mask"], agent_mask=batch["agent_mask"],
                    )
                torch.cuda.synchronize()

            amp_ctx = _make_autocast(enabled=use_amp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with amp_ctx:
                _ = model(
                    x=batch["x"], adj=batch["adj"],
                    x_mask=batch["x_mask"], agent_mask=batch["agent_mask"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms / max(1, actual_batch_size))

    return sum(latencies) / len(latencies) if latencies else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Training epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
    log_every: int = 0,
    use_amp: bool = True,
    debug_gat: bool = False,
    per_group_clip: bool = False,       # [FIX-1] per-group gradient clipping
    # [IMP-B, IMP-C]
    diversity_weight: float = 0.0,
    augment: bool = False,
    aug_rot_std: float = 10.0,
    aug_hist_dropout: float = 0.1,
    # [IMP-D] Progressive diversity weight annealing
    current_epoch: int = 0,
    total_epochs: int = 1,
    diversity_ramp_epochs: int = 20,
) -> Tuple[float, float, float, float, float]:
    """
    Run one epoch.  Returns (mean_loss, mean_ADE, mean_FDE, mean_div_loss, mean_MissRate).

    [IMP-B] diversity_weight > 0 adds mode repulsion to loss (multimodal only).
    [IMP-C] augment=True applies random rotation + history dropout per batch.
    [IMP-D] diversity_weight is progressively ramped from 0 to its target value
            over diversity_ramp_epochs to avoid destabilising early training.
            effective_div_weight = diversity_weight * min(1, epoch / ramp_epochs)
    """
    training = optimizer is not None
    model.train(training)

    # [IMP-D] Anneal diversity weight: ramp from 0 → target over ramp_epochs
    # so early epochs focus on WTA loss before mode separation is enforced.
    if training and diversity_weight > 0.0 and diversity_ramp_epochs > 0:
        ramp_frac = min(1.0, float(max(1, current_epoch)) / float(diversity_ramp_epochs))
        effective_div_weight = diversity_weight * ramp_frac
    else:
        effective_div_weight = diversity_weight

    total_loss = total_ade = total_fde = total_div_loss = total_miss = 0.0
    n_valid = nan_batches = nonfinite_preds = 0
    # [BUG-6 FIX] Track accumulation position explicitly so a NaN batch in the
    # middle of an accumulation window doesn't silently mix a reset gradient with
    # gradients from the previous (valid) mini-batches in the same window.
    accum_pos = 0  # counts valid backward steps in the current window

    if training:
        optimizer.zero_grad(set_to_none=True)

    for idx, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)

        # [IMP-C] Augmentation (training only)
        if training and augment:
            batch = _augment_batch(batch, aug_rot_std, aug_hist_dropout)

        amp_ctx = _make_autocast(enabled=use_amp and training)
        with torch.set_grad_enabled(training), amp_ctx:
            pred = model(
                x=batch["x"],
                adj=batch["adj"],
                x_mask=batch["x_mask"],
                agent_mask=batch["agent_mask"],
            )

            # NaN safety
            pred_finite = torch.isfinite(pred)
            if not bool(pred_finite.all().detach().cpu().item()):
                nonfinite_preds += 1
                if debug_gat and nonfinite_preds <= 3:
                    bad_ratio = 1.0 - float(
                        pred_finite.float().mean().detach().cpu().item()
                    )
                    print(
                        f"    [WARN] non-finite pred (ratio={bad_ratio:.6f}) "
                        f"before sanitization"
                    )

            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

            # [IMP-B] Pass effective (ramped) diversity_weight into loss
            loss, div_loss_val = _compute_loss(
                pred=pred,
                target=batch["y"],
                y_mask=batch["y_mask"],
                agent_mask=batch["agent_mask"],
                diversity_weight=effective_div_weight if training else 0.0,
            )
            loss_scaled = loss / accum_steps

        loss_val = float(loss.detach().cpu().item())

        if training and (math.isnan(loss_val) or math.isinf(loss_val)):
            nan_batches += 1
            if log_every > 0 and idx % log_every == 0:
                print(
                    f"    [WARN] step={idx}/{len(loader)}: "
                    f"loss={loss_val} — skipping, resetting scaler"
                )
            # [BUG-6 FIX] Reset the entire accumulation window: discard any
            # gradients accumulated from valid batches earlier in this window,
            # then reset the position counter so the next batch starts fresh.
            # This is safer than letting a partial window reach optimizer.step().
            optimizer.zero_grad(set_to_none=True)
            accum_pos = 0
            # [SCALER FIX] Do NOT touch scaler._scale (private API). The
            # GradScaler will reduce its scale automatically via scaler.update()
            # the next time scaler.step() encounters inf/NaN gradients. No
            # manual intervention needed here; just skip this batch cleanly.
            continue

        if training:
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_pos += 1
            # [BUG-6 FIX] Step only when the accumulation window is full (based
            # on valid-batch count, not raw batch index). This ensures NaN-skipped
            # batches don't shift the optimizer step off-boundary.
            if accum_pos >= accum_steps:
                accum_pos = 0
                if scaler is not None:
                    if grad_clip > 0 or debug_gat:
                        scaler.unscale_(optimizer)
                    if debug_gat and log_every > 0 and idx % log_every == 0:
                        norms = _grad_norms(model)
                        print(
                            f"    grad_norm total={norms['total']:.3e} "
                            f"gat={norms['gat']:.3e} other={norms['other']:.3e}"
                        )
                    if grad_clip > 0:
                        _clip_gradients(model, optimizer, grad_clip,
                                        per_group=per_group_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if debug_gat and log_every > 0 and idx % log_every == 0:
                        norms = _grad_norms(model)
                        print(
                            f"    grad_norm total={norms['total']:.3e} "
                            f"gat={norms['gat']:.3e} other={norms['other']:.3e}"
                        )
                    if grad_clip > 0:
                        _clip_gradients(model, optimizer, grad_clip,
                                        per_group=per_group_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ade, fde, miss = _compute_metrics(
                pred=pred.detach(),
                target=batch["y"],
                y_mask=batch["y_mask"],
                agent_mask=batch["agent_mask"],
            )

        ade_val = _to_float(ade)
        fde_val = _to_float(fde)
        miss_val = _to_float(miss)

        if not (math.isnan(ade_val) or math.isnan(fde_val)):
            total_loss     += loss_val
            total_ade      += ade_val
            total_fde      += fde_val
            total_div_loss += div_loss_val
            total_miss     += miss_val
            n_valid        += 1

        if training and log_every > 0 and idx % log_every == 0:
            div_info = (
                f" div={div_loss_val:.4f}(w={effective_div_weight:.3f})"
                if effective_div_weight > 0.0 else ""
            )
            print(
                f"    step={idx}/{len(loader)} loss={loss_val:.4f} "
                f"ADE={ade_val:.3f} FDE={fde_val:.3f} "
                f"MissRate={miss_val:.1%}{div_info}"
            )

    if training and nan_batches > 0:
        print(f"    [INFO] {nan_batches}/{len(loader)} batches skipped (NaN/Inf loss)")
    if nonfinite_preds > 0:
        phase = "train" if training else "val"
        print(
            f"    [INFO] {phase}: {nonfinite_preds}/{len(loader)} batches had "
            f"non-finite predictions before sanitization"
        )

    d = max(1, n_valid)
    avg_loss     = total_loss     / d if n_valid > 0 else float("nan")
    avg_ade      = total_ade      / d if n_valid > 0 else float("nan")
    avg_fde      = total_fde      / d if n_valid > 0 else float("nan")
    avg_div_loss = total_div_loss / d if n_valid > 0 else 0.0
    avg_miss     = total_miss     / d if n_valid > 0 else float("nan")
    return avg_loss, avg_ade, avg_fde, avg_div_loss, avg_miss


# ═══════════════════════════════════════════════════════════════════════════════
# Single training run
# ═══════════════════════════════════════════════════════════════════════════════

def train_single(
    *,
    out_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds_len: int,
    val_ds_len: int,
    input_dim: int,
    future_steps: int,
    data_dirs: List[Path],
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    variant: Optional[AblationVariant] = None,
    run_tag: str = "run",
) -> dict:
    """
    Full train + val loop for one configuration.

    Changes vs v2:
    [IMP-A] Uses WarmupCosineScheduler (linear warmup + cosine) when --cosine-lr.
    [IMP-D] best checkpoint selected by ADE or loss (--early-stop-metric).
            best_val_ade / best_val_fde reported from SAME epoch as best ckpt.
    [IMP-B] Passes diversity_weight to run_epoch.
    [IMP-C] Passes augmentation args to run_epoch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = build_model(input_dim, future_steps, args, variant)

    n_gpus = torch.cuda.device_count()
    if device.type == "cuda" and n_gpus > 1:
        print(f"  [GPU] DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    model = model.to(device)

    is_gat  = variant.enable_gat if variant else args.enable_gat
    base_lr = args.learning_rate
    gat_lr  = base_lr * (args.gat_lr_scale if is_gat else 1.0)
    # [FIX-1] eff_clip = grad_clip always (gat_clip_scale removed from global clip).
    # Per-group clip is handled inside _clip_gradients when --gat-per-group-clip.
    eff_clip     = args.grad_clip
    per_grp_clip = getattr(args, "gat_per_group_clip", False) and is_gat

    # [CRITICAL FIX] Build optimizer BEFORE freezing so GAT params are
    # registered in AdamW's state. If we froze first, build_optimizer would
    # skip requires_grad=False params → GAT never gets Adam moments → never
    # updated after unfreeze. Frozen params with grad=None are safely skipped
    # by AdamW internally, so having them in the group is harmless.
    optimizer = build_optimizer(
        model, base_lr=base_lr, gat_lr=gat_lr,
        weight_decay=args.weight_decay, is_gat_variant=is_gat,
    )

    # [FIX-2] Freeze GAT graph_blocks for the first gat_freeze_epochs epochs.
    # Must happen AFTER build_optimizer (see above).
    freeze_epochs = int(getattr(args, "gat_freeze_epochs", 0)) if is_gat else 0
    gat_frozen    = freeze_epochs > 0

    def _set_gat_frozen(frozen: bool) -> None:
        raw = model.module if hasattr(model, "module") else model
        for name, param in raw.named_parameters():
            if _is_gat_param_name(name):
                param.requires_grad_(not frozen)

    if gat_frozen:
        _set_gat_frozen(True)
        print(
            f"  [GAT-freeze] Freezing graph_blocks for {freeze_epochs} epochs; "
            f"will unfreeze at epoch {freeze_epochs + 1} with own warmup."
        )

    if is_gat:
        clip_desc = "per-group" if per_grp_clip else f"{eff_clip:.2f}"
        print(
            f"  [GAT] LR groups: base={base_lr:.2e}, gat={gat_lr:.2e} "
            f"({args.gat_lr_scale:g}×) | clip={clip_desc}"
        )

    # [FIX-3] Build scheduler with GAT group_start_epochs so the GAT group's
    # cosine budget is measured from unfreeze_epoch, not from epoch 0.
    warmup_ep = getattr(args, "warmup_epochs", 0)
    if args.cosine_lr:
        # GAT group starts its warmup AFTER freeze phase ends.
        gat_start  = freeze_epochs  # epoch at which GAT unfreezes
        group_starts: Optional[Dict[str, int]] = (
            {"gat": gat_start} if (is_gat and gat_start > 0) else None
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_ep,
            total_epochs=epochs,
            min_lr_ratio=0.001,  # [IMP-F] 0.01→0.001: exploit cosine tail better
            group_start_epochs=group_starts,
        )
        sched_mode = f"warmup({warmup_ep})+cosine"
        if group_starts:
            sched_mode += f" [GAT starts ep {gat_start + 1}+warmup]"
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.lr_patience,
        )
        sched_mode = "plateau"

    use_amp = not args.no_amp and device.type == "cuda"
    scaler  = _make_scaler(enabled=use_amp)

    # [IMP-D] Early-stop metric choice
    stop_metric = getattr(args, "early_stop_metric", "ade")
    print(
        f"  AMP={'ON' if use_amp else 'OFF'} | GPUs={n_gpus} | "
        f"eff_batch={args.batch_size * args.accum_steps * max(1, n_gpus)} | "
        f"LR={base_lr:.2e} | sched={sched_mode} | "
        f"early_stop_on={stop_metric} | "
        f"freeze_epochs={freeze_epochs} | "
        f"diversity_w={getattr(args, 'diversity_weight', 0.0):.3f} | "
        f"augment={getattr(args, 'augment', False)}"
    )

    best_metric  = math.inf  # tracks the monitored metric
    stale        = 0
    nan_epochs   = 0
    best_path    = out_dir / f"{run_tag}_best.pt"       # best by monitored metric
    best_ade_path = out_dir / f"{run_tag}_best_ade.pt"  # always track best ADE
    last_path    = out_dir / f"{run_tag}_last.pt"
    history: List[dict] = []

    best_ade_metric = math.inf  # independent ADE tracker
    best_ade_record: dict = {}  # epoch record at best ADE

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # [FIX-2/4] Unfreeze GAT at the designated epoch and reset early-stop.
        if gat_frozen and epoch > freeze_epochs:
            gat_frozen = False
            _set_gat_frozen(False)
            # Reset early-stop so the post-unfreeze phase gets a full budget.
            stale = 0
            # [FIX-5] Reset best_metric so the frozen-phase checkpoint (with
            # random GAT weights) doesn't block saving the real best after unfreeze.
            best_metric = math.inf
            # [BUG-4 FIX] Also reset best_ade_metric and best_ade_record so the
            # independent ADE tracker doesn't keep a frozen-phase checkpoint
            # (which has untrained GAT) as the "best ADE" model after unfreeze.
            # Before wiping, archive the frozen-best checkpoint under a distinct
            # name so it remains retrievable for comparison.
            if best_ade_record:
                frozen_best_path = out_dir / f"{run_tag}_frozen_best_ade.pt"
                if best_ade_path.exists():
                    import shutil
                    shutil.copy2(best_ade_path, frozen_best_path)
                    print(
                        f"  [GAT-unfreeze] Archived frozen-phase best-ADE checkpoint "
                        f"→ {frozen_best_path.name} "
                        f"(ADE={best_ade_record.get('val_ade', float('nan')):.4f})"
                    )
            best_ade_metric = math.inf
            best_ade_record = {}
            print(
                f"  [GAT-unfreeze] Epoch {epoch}: GAT graph_blocks unfrozen. "
                f"Early-stop counter, best_metric, and best_ade_metric reset."
            )

        tr_loss, tr_ade, tr_fde, tr_div, tr_miss = run_epoch(
            model, train_loader, device,
            optimizer=optimizer, scaler=scaler,
            accum_steps=args.accum_steps,
            grad_clip=eff_clip,
            log_every=args.log_every,
            use_amp=use_amp,
            debug_gat=args.debug_gat and is_gat,
            per_group_clip=per_grp_clip,
            diversity_weight=getattr(args, "diversity_weight", 0.0),
            augment=getattr(args, "augment", False),
            aug_rot_std=getattr(args, "aug_rot_std", 10.0),
            aug_hist_dropout=getattr(args, "aug_hist_dropout", 0.1),
            # [IMP-D] Pass epoch info so diversity weight ramps gradually
            current_epoch=epoch,
            total_epochs=epochs,
            diversity_ramp_epochs=getattr(args, "diversity_ramp_epochs", 20),
        )
        val_loss, val_ade, val_fde, _, val_miss = run_epoch(
            model, val_loader, device, use_amp=use_amp
        )

        # Scheduler step
        if args.cosine_lr:
            scheduler.step()
        else:
            if not math.isnan(val_loss):
                scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_groups = {
            str(g.get("name", f"g{i}")): float(g["lr"])
            for i, g in enumerate(optimizer.param_groups)
        }
        lr_text = " ".join(f"{n}={lr:.2e}" for n, lr in lr_groups.items())

        record = dict(
            epoch=epoch,
            tr_loss=tr_loss, tr_ade=tr_ade, tr_fde=tr_fde, tr_div=tr_div,
            tr_miss=tr_miss,
            val_loss=val_loss, val_ade=val_ade, val_fde=val_fde,
            val_miss=val_miss,
            lr=optimizer.param_groups[0]["lr"], lr_groups=lr_groups,
        )
        history.append(record)

        # NaN early stop
        if math.isnan(tr_loss) or math.isnan(val_loss):
            nan_epochs += 1
        else:
            nan_epochs = 0
        if nan_epochs >= args.nan_patience:
            print(
                f"  [NaN-stop] {nan_epochs} consecutive NaN epochs "
                f"(patience={args.nan_patience}). Aborting."
            )
            break

        # [IMP-D] Determine the metric we're optimizing
        if stop_metric == "ade":
            current = val_ade
        else:
            current = val_loss
        improved = (not math.isnan(current)) and current < best_metric

        # [IMP-I] Log diversity loss when active
        div_info = (
            f" div={tr_div:.4f}" if tr_div > 0.0 else ""
        )
        marker = " âœ“" if improved else ""
        print(
            f"  [{run_tag}] epoch={epoch:03d}/{epochs} lr={lr_text} t={elapsed:.0f}s | "
            f"tr_loss={tr_loss:.4f} ADE={tr_ade:.3f} FDE={tr_fde:.3f} "
            f"MissRate={tr_miss:.1%}{div_info} | "
            f"val_loss={val_loss:.4f} ADE={val_ade:.3f} FDE={val_fde:.3f} "
            f"MissRate={val_miss:.1%}{marker}"
        )

        raw_model = model.module if hasattr(model, "module") else model
        ckpt = dict(
            model_state_dict=raw_model.state_dict(),
            model_config=config_to_dict(cfg),
            epoch=epoch,
            val_loss=val_loss, val_ade=val_ade, val_fde=val_fde,
            val_miss=val_miss,
        )
        torch.save(ckpt, last_path)

        if improved:
            best_metric = current
            stale = 0
            torch.save(ckpt, best_path)
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={args.early_stopping_patience}, "
                    f"metric={stop_metric})"
                )
                break

        # [IMP-E] Always track best ADE independently (saved separately)
        if not math.isnan(val_ade) and val_ade < best_ade_metric:
            best_ade_metric = val_ade
            best_ade_record = record.copy()
            torch.save(ckpt, best_ade_path)

    (out_dir / f"{run_tag}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # [IMP-E] Report best ADE/FDE from the SAME epoch as best_ade checkpoint
    best_ade_ep = best_ade_record.get("val_ade", float("nan"))
    best_fde_ep = best_ade_record.get("val_fde", float("nan"))
    best_miss_ep = best_ade_record.get("val_miss", float("nan"))

    # Measure inference latency on the best checkpoint
    inference_latency_ms = 0.0
    try:
        raw_model = model.module if hasattr(model, "module") else model
        if best_ade_path.exists():
            ckpt_best = torch.load(best_ade_path, map_location=device, weights_only=True)
            raw_model.load_state_dict(ckpt_best["model_state_dict"])
        inference_latency_ms = measure_inference_latency(
            raw_model, val_loader, device,
            num_batches=min(100, len(val_loader)),
            use_amp=use_amp,
        )
        print(f"  Inference latency: {inference_latency_ms:.2f} ms/sample")
    except Exception as exc:
        print(f"  [WARN] Inference latency measurement failed: {exc}")

    target_metrics = make_target_metrics(
        best_ade_ep,
        best_fde_ep,
        best_miss_ep,
        inference_latency_ms,
    )
    results = dict(
        run_tag=run_tag,
        best_val_loss=(
            best_metric if stop_metric == "loss" and not math.isinf(best_metric)
            else best_ade_record.get("val_loss", float("nan"))
        ),
        best_val_ade=best_ade_ep,
        best_val_fde=best_fde_ep,
        best_val_miss_rate=best_miss_ep,
        inference_latency_ms=inference_latency_ms,
        minADE=target_metrics["minADE"],
        minFDE=target_metrics["minFDE"],
        MissRate=target_metrics["MissRate"],
        latency_ms_per_sample=target_metrics["inference_latency_ms"],
        metrics=target_metrics,
        target_metrics=target_metrics,
        best_val_ade_epoch=best_ade_record.get("epoch", -1),
        best_path=str(best_path),
        best_ade_path=str(best_ade_path),
    )
    (out_dir / f"{run_tag}_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Ablation runner
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_ablation(
    *,
    out_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds_len: int,
    val_ds_len: int,
    input_dim: int,
    future_steps: int,
    data_dirs: List[Path],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    epochs = max(1, args.epochs // 3) if args.quick_ablation else args.epochs
    if args.quick_ablation:
        print(f"  Quick ablation: {epochs} epochs per variant")

    all_results = []
    _print_separator("ABLATION STUDY â€” 8 VARIANTS")
    print(
        "\n  Variant   â”‚ Code â”‚ GAT â”‚ Multi â”‚ AdpR â”‚ Description\n"
        "  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"
    )
    for v in ABLATION_VARIANTS:
        g = "âœ“" if v.enable_gat else "-"
        m = "âœ“" if v.enable_multimodal else "-"
        r = "âœ“" if v.enable_adaptive_radius else "-"
        print(f"  {v.name:<10}â”‚  {v.code}  â”‚  {g}  â”‚   {m}   â”‚  {r}   â”‚ {v.description}")
    print()

    for i, variant in enumerate(ABLATION_VARIANTS, start=1):
        _print_separator(f"Variant {i}/8 â€” {variant.name} [{variant.code}]")
        variant_dir = out_dir / f"ablation_{variant.code}_{variant.name}"
        # [FIX] Reset RNG so every variant gets the same random model init.
        # Without this, the 8 variants receive progressively different weight
        # initialisations, making ADE comparisons unfair.
        set_seed(args.seed)
        try:
            result = train_single(
                out_dir=variant_dir,
                train_loader=train_loader, val_loader=val_loader,
                train_ds_len=train_ds_len, val_ds_len=val_ds_len,
                input_dim=input_dim, future_steps=future_steps,
                data_dirs=data_dirs, device=device, args=args,
                epochs=epochs, variant=variant, run_tag=f"ablation_{variant.code}",
            )
        except Exception as exc:
            print(f"\n  [ERROR] Variant {variant.name} raised: {exc}")
            traceback.print_exc()
            failed_metrics = make_target_metrics(float("nan"), float("nan"), float("nan"), 0.0)
            result = dict(
                run_tag=f"ablation_{variant.code}",
                best_val_loss=float("nan"),
                best_val_ade=float("nan"),
                best_val_fde=float("nan"),
                best_val_miss_rate=float("nan"),
                inference_latency_ms=0.0,
                minADE=failed_metrics["minADE"],
                minFDE=failed_metrics["minFDE"],
                MissRate=failed_metrics["MissRate"],
                latency_ms_per_sample=failed_metrics["inference_latency_ms"],
                metrics=failed_metrics,
                target_metrics=failed_metrics,
                best_path="N/A",
                error=str(exc),
            )

        result["variant"]     = variant.name
        result["code"]        = variant.code
        result["description"] = variant.description
        all_results.append(result)

    _print_separator("ABLATION RESULTS SUMMARY")
    header_sep = "-"
    print(
        f"\n  {'Variant':<14} {'Code':>4}  {'val_ADE':>8}  {'val_FDE':>8}  "
        f"{'MissRate':>8}  {'Latency':>8}  "
        f"{'val_loss':>10}  Status\n"
        f"  {header_sep*14}  {header_sep*4}  {header_sep*8}  {header_sep*8}"
        f"  {header_sep*8}  {header_sep*8}  {header_sep*10}  {header_sep*8}"
    )

    def _sort_key(r: dict) -> tuple:
        v = r["best_val_ade"]
        return (0, v) if not math.isnan(v) else (1, 0)

    for r in sorted(all_results, key=_sort_key):
        ade_s  = f"{r['best_val_ade']:8.3f}"  if not math.isnan(r["best_val_ade"])  else "     NaN"
        fde_s  = f"{r['best_val_fde']:8.3f}"  if not math.isnan(r["best_val_fde"])  else "     NaN"
        mr_val = r.get("best_val_miss_rate", float("nan"))
        mr_s   = f"{mr_val:7.1%}"  if not math.isnan(mr_val) else "     NaN"
        lat_val = r.get("inference_latency_ms", 0.0)
        lat_s  = f"{lat_val:7.1f}ms" if lat_val > 0 else "     N/A"
        loss_s = f"{r['best_val_loss']:10.4f}" if not math.isnan(r["best_val_loss"]) else "       NaN"
        status = "FAILED" if "error" in r else ("NaN" if math.isnan(r["best_val_ade"]) else "OK")
        print(f"  {r['variant']:<14}  {r['code']:>4}  {ade_s}  {fde_s}  {mr_s}  {lat_s}  {loss_s}  {status}")

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  [OK] Summary: {summary_path}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Per-town runner
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07"]


def run_per_town(
    *,
    all_paths: List[Path],
    data_dirs: List[Path],
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    all_results = []
    for town in TOWNS:
        town_paths = filter_by_town(all_paths, town)
        if not town_paths:
            print(f"  [SKIP] {town}: no samples.")
            continue
        _print_separator(f"Per-town: {town} ({len(town_paths)} samples)")
        root_map = _build_path_root_map(town_paths, data_dirs)
        train_p, val_p = split_sample_paths(
            town_paths, train_ratio=args.train_ratio, seed=args.seed
        )
        if not val_p:
            val_p   = train_p[-1:]
            train_p = train_p[:-1] or val_p
        tr_loader, val_loader, tr_ds = build_loaders(
            train_p, val_p, data_dirs, root_map,
            args.batch_size, args.num_workers, device,
        )
        first = tr_ds[0]
        result = train_single(
            out_dir=out_dir / town,
            train_loader=tr_loader, val_loader=val_loader,
            train_ds_len=len(tr_ds), val_ds_len=len(val_p),
            input_dim=int(first["x"].shape[2]),
            future_steps=int(first["y"].shape[1]),
            data_dirs=data_dirs, device=device, args=args,
            epochs=args.epochs,
            variant=ABLATION_VARIANTS[-1],  # GTNet_Full
            run_tag=town,
        )
        result["town"] = town
        all_results.append(result)

    summary_path = out_dir / "per_town_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    _print_separator("PER-TOWN SUMMARY")
    for r in all_results:
        mr_val = r.get("best_val_miss_rate", float("nan"))
        mr_s = f"{mr_val:.1%}" if not math.isnan(mr_val) else "N/A"
        lat_val = float(r.get("inference_latency_ms", 0.0))
        lat_s = f"{lat_val:.1f}ms" if lat_val > 0.0 else "N/A"
        print(
            f"  {r['town']:8s}  minADE={r['best_val_ade']:.3f}  "
            f"minFDE={r['best_val_fde']:.3f}  MissRate={mr_s}  "
            f"Latency={lat_s}"
        )
    print(f"\n  [OK] Summary: {summary_path}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# main
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus    = torch.cuda.device_count()
    data_dirs = [Path(d).resolve() for d in args.data_dir]
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_separator("GTNet Kaggle Trainer v3 (improved)")
    print(f"  Mode          : {args.mode}")
    print(f"  Device        : {device}  ({n_gpus} GPU(s))")
    print(f"  AMP           : {'OFF' if args.no_amp else 'ON'}")
    print(f"  Scheduler     : {'warmup+cosine' if args.cosine_lr else 'plateau'}")
    if args.cosine_lr:
        print(f"  Warmup epochs : {args.warmup_epochs}")
    print(f"  Early stop on : {getattr(args, 'early_stop_metric', 'ade')}")
    print(f"  Diversity wt  : {getattr(args, 'diversity_weight', 0.0):.3f}")
    print(f"  Augmentation  : {getattr(args, 'augment', False)}")
    print(f"  Temporal attn : {getattr(args, 'use_temporal_attention', False)}")
    print(f"  Encoder drop  : {getattr(args, 'encoder_dropout', 0.0):.2f}")
    print(f"  GAT edge dim  : {getattr(args, 'gat_edge_dim', 32)}")
    print(f"  Data dirs     : {[str(d) for d in data_dirs]}")

    all_paths = load_all_sample_paths(data_dirs, limit=args.limit_samples)
    if args.town_filter and args.mode != "per-town":
        all_paths = filter_by_town(all_paths, args.town_filter)
    if not all_paths:
        raise RuntimeError("No sample files found. Check --data-dir path.")
    print(f"  Samples       : {len(all_paths)} total")

    if args.mode == "per-town":
        run_per_town(
            all_paths=all_paths, data_dirs=data_dirs,
            out_dir=out_dir, device=device, args=args,
        )
        return 0

    root_map = _build_path_root_map(all_paths, data_dirs)
    train_p, val_p = split_sample_paths(
        all_paths, train_ratio=args.train_ratio, seed=args.seed
    )
    if not val_p:
        val_p   = train_p[-1:]
        train_p = train_p[:-1] or val_p

    tr_loader, val_loader, tr_ds = build_loaders(
        train_p, val_p, data_dirs, root_map,
        args.batch_size, args.num_workers, device,
    )
    first        = tr_ds[0]
    future_steps = int(first["y"].shape[1])
    input_dim    = int(first["x"].shape[2])
    print(f"  Split         : train={len(train_p)}  val={len(val_p)}")

    if args.mode == "ablation":
        run_ablation(
            out_dir=out_dir,
            train_loader=tr_loader, val_loader=val_loader,
            train_ds_len=len(train_p), val_ds_len=len(val_p),
            input_dim=input_dim, future_steps=future_steps,
            data_dirs=data_dirs, device=device, args=args,
        )
        return 0

    variant: Optional[AblationVariant] = None
    if args.mode == "full":
        variant = ABLATION_VARIANTS[-1]  # GTNet_Full (111)

    run_tag = f"gtnet_{args.mode}"
    result  = train_single(
        out_dir=out_dir / run_tag,
        train_loader=tr_loader, val_loader=val_loader,
        train_ds_len=len(train_p), val_ds_len=len(val_p),
        input_dim=input_dim, future_steps=future_steps,
        data_dirs=data_dirs, device=device, args=args,
        epochs=args.epochs, variant=variant, run_tag=run_tag,
    )

    _print_separator("DONE")
    print(f"  minADE       : {result['minADE']:.3f}")
    print(f"  minFDE       : {result['minFDE']:.3f}")
    mr_val = result.get('best_val_miss_rate', float('nan'))
    print(f"  Miss Rate    : {mr_val:.1%}" if not math.isnan(mr_val) else "  Miss Rate    : N/A")
    print(f"  Latency      : {result['latency_ms_per_sample']:.2f} ms/sample")
    print(f"  Best at epoch: {result.get('best_val_ade_epoch', '?')}")
    print(f"  Checkpoint   : {result['best_path']}")
    print(f"  Best ADE ckpt: {result['best_ade_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
