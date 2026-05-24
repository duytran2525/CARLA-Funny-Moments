"""
kaggle_train_gtnet.py  — v3 (GTNet improved)
═════════════════════════════════════════════════════════════════════════════

CẢI TIẾN so với v2:

[IMP-A] WarmupCosineScheduler (mới)
    Linear warmup trong --warmup-epochs epoch đầu → cosine decay.
    Tránh LR đột ngột cao ngay epoch 1 gây instability.
    Hỗ trợ nhiều param groups (GAT dual-LR) tự động.
    Thay thế CosineAnnealingLR khi --cosine-lr được bật.

[IMP-B] Diversity loss (mới)
    Penalize các mode có endpoint quá gần nhau (khoảng cách < margin).
    Ngăn mode collapse khi dùng WTA với K=5.
    Kích hoạt bằng --diversity-weight > 0 (khuyến nghị 0.05–0.1).

[IMP-C] Data augmentation (mới)
    _augment_batch(): áp dụng trong training loop:
      • Random in-plane rotation (±rot_std_deg°) trên x, y, vx, vy, heading
        và target y → tăng cường tính invariance với góc quay ego
      • History frame dropout: ngẫu nhiên mask một số frame lịch sử
        (giữ nguyên anchor frame cuối) → mô hình robust hơn với missing obs.
    Kích hoạt bằng --augment.

[IMP-D] Early stopping dựa trên val_ADE (thay vì val_loss)
    val_ADE = metric trực tiếp phản ánh chất lượng dự đoán quỹ đạo.
    val_loss bị ảnh hưởng bởi scale của smooth-L1 → không luôn monotone
    với ADE. Thêm --early-stop-metric (loss|ade, default: ade).
    Lưu riêng best_ade.pt và best_loss.pt.

[IMP-E] Best result tracking chính xác
    best_val_ade và best_val_fde trong kết quả giờ được lấy từ CÙNG epoch
    với best checkpoint, không phải min() độc lập qua toàn history.

[IMP-F] New config fields forwarded to build_model()
    --use-temporal-attention, --mode-embed-dim, --encoder-dropout
    → tự động forward vào MultiAgentModelConfig.

KHAI THÁC TỪ PHÂN TÍCH 39 EPOCHS:
  • Train/val ADE gap = 3.5× → [IMP-C] augmentation + [IMP-F] encoder_dropout
  • Val plateau sau epoch 24 → [IMP-A] warmup giúp giai đoạn đầu tốt hơn
  • K=5 WTA risk collapse → [IMP-B] diversity loss
  • LR không warmup → [IMP-A]
  • Early stop dựa loss → [IMP-D] dựa ADE
"""

from __future__ import annotations

import argparse
import json
import math
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

# ── AMP compatibility ──────────────────────────────────────────────────────────
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


# ── sys.path setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_PARENT  = Path("/kaggle/input/datasets/dungnguyentrung/f23456")
for _root in (CORE_PARENT, PROJECT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

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


# ══════════════════════════════════════════════════════════════════════════════
# Multimodal-aware loss & metrics (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    diversity_weight: float = 0.0,
) -> torch.Tensor:
    """
    Compute training loss.

    Unimodal  [B, A, T, 2]   → masked smooth-L1
    Multimodal [B, A, K, T, 2] → Winner-Takes-All + optional diversity term

    [IMP-B] diversity_weight > 0 adds a pairwise endpoint repulsion loss
    that penalizes modes whose final positions are too close together.
    This prevents mode collapse when K is large (e.g. K=5).
    """
    if pred.dim() == 4:
        return masked_smooth_l1_loss(pred, target, y_mask, agent_mask)

    # ── Multimodal WTA ────────────────────────────────────────────────────────
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
        return main_loss + diversity_weight * div

    return main_loss


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_agent = agent_mask & y_mask.any(dim=-1)

    if pred.dim() == 4:
        disp = torch.norm(pred - target, dim=-1)
        valid = y_mask & agent_mask.unsqueeze(-1)
        valid_f = valid.to(dtype=pred.dtype)
        ade = (disp * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        fde_per_agent = _last_valid_displacement(disp, y_mask, agent_mask)
        mask = valid_agent.to(dtype=pred.dtype)
        fde = (fde_per_agent * mask).sum() / mask.sum().clamp_min(1.0)
        return ade, fde

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
    return (min_ade * mask).sum() / n_valid, (min_fde * mask).sum() / n_valid


# ══════════════════════════════════════════════════════════════════════════════
# [IMP-B] Mode diversity loss
# ══════════════════════════════════════════════════════════════════════════════

def _diversity_loss(
    pred: torch.Tensor,
    agent_mask: torch.Tensor,
    y_mask: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Pairwise endpoint repulsion loss for multimodal predictions.

    Penalizes pairs of modes whose final predicted positions are within
    `margin` metres of each other. This prevents mode collapse — a
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

    # Use last timestep as "endpoint" for efficiency
    endpoints = pred[:, :, :, -1, :]  # [B, A, K, 2]

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


# ══════════════════════════════════════════════════════════════════════════════
# [IMP-C] Data augmentation
# ══════════════════════════════════════════════════════════════════════════════

def _augment_batch(
    batch: Dict,
    rot_std_deg: float = 10.0,
    hist_dropout: float = 0.1,
) -> Dict:
    """
    Apply random in-plane rotation and history frame dropout.

    Random rotation:
        Rotates features in ego-frame by ±rot_std_deg° (sampled per sample).
        Applied consistently to all feature channels:
          x[..., 0:2]  — local position (x, y)
          x[..., 2:4]  — local velocity (vx, vy)   [if feature_dim >= 4]
          x[..., 4:6]  — heading unit vector        [if feature_dim >= 6]
          y[..., 0:2]  — target position (x, y)
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

    # ── Random rotation ──────────────────────────────────────────────────────
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
        x = _rot2d(x, 0)           # position
        if feat_dim >= 4:
            x = _rot2d(x, 2)       # velocity (new format) or heading (old)
        if feat_dim >= 6:
            x = _rot2d(x, 4)       # heading (new format)
        batch["x"] = x

        # Rotate targets (always 2D position)
        batch["y"] = _rot2d(batch["y"].clone(), 0)

    # ── History frame dropout ────────────────────────────────────────────────
    if hist_dropout > 0.0 and batch["x_mask"].shape[-1] > 1:
        x_mask = batch["x_mask"].clone()
        # Bernoulli dropout on currently-valid frames, but NEVER drop t=-1
        drop = (torch.rand_like(x_mask.float()) < hist_dropout) & x_mask
        drop[..., -1] = False       # protect anchor frame
        batch["x_mask"] = x_mask & ~drop

    return batch


# ══════════════════════════════════════════════════════════════════════════════
# [IMP-A] Warmup + Cosine scheduler
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Linear LR warmup then cosine annealing.

    Drop-in replacement for CosineAnnealingLR. Works with multi-group
    optimizers (e.g. the GAT dual-LR setup) by scaling each group's
    initial LR independently.

    Schedule:
        epoch 1..warmup_epochs : LR ramps linearly from 0 → base_lr
        epoch warmup_epochs+1.. : cosine decay from base_lr → min_lr

    Args:
        optimizer:      PyTorch optimizer (single or multi group)
        warmup_epochs:  number of linear warmup epochs (0 = no warmup)
        total_epochs:   total training epochs (for cosine end point)
        min_lr_ratio:   min LR = base_lr × min_lr_ratio (default 0.01)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr_ratio  = min_lr_ratio
        # Snapshot initial LRs for each group at construction time
        self.base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        self._epoch   = 0

    def step(self) -> None:
        self._epoch += 1
        e = self._epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            min_lr = base_lr * self.min_lr_ratio
            if self.warmup_epochs > 0 and e <= self.warmup_epochs:
                lr = base_lr * e / self.warmup_epochs
            else:
                cos_e   = e - self.warmup_epochs
                cos_max = max(1, self.total_epochs - self.warmup_epochs)
                progress = cos_e / cos_max
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            pg["lr"] = max(float(lr), min_lr)

    def get_last_lr(self) -> List[float]:
        return [float(pg["lr"]) for pg in self.optimizer.param_groups]


# ══════════════════════════════════════════════════════════════════════════════
# Ablation variant definitions
# ══════════════════════════════════════════════════════════════════════════════

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
                    "Adds Graph Attention — learns edge weights between agents"),
    AblationVariant("Multi_only", "010", False, True,  False,
                    "Adds multimodal output — K future trajectory hypotheses"),
    AblationVariant("AdpR_only",  "001", False, False, True,
                    "Adds adaptive radius — interaction range scales with speed"),
    AblationVariant("GAT_Multi",  "110", True,  True,  False,
                    "GAT + multimodal"),
    AblationVariant("GAT_AdpR",   "101", True,  False, True,
                    "GAT + adaptive radius"),
    AblationVariant("Multi_AdpR", "011", False, True,  True,
                    "Multimodal + adaptive radius"),
    AblationVariant("GTNet_Full", "111", True,  True,  True,
                    "All improvements enabled — full GTNet"),
]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GTNet Kaggle trainer v3 (improved).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",    required=True, nargs="+")
    p.add_argument("--out-dir",     default="/kaggle/working/models")
    p.add_argument("--mode", choices=["baseline", "full", "ablation", "per-town"],
                   default="full")

    # ── Hyper-params ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",              type=int,   default=50)
    p.add_argument("--batch-size",          type=int,   default=64)
    p.add_argument("--accum-steps",         type=int,   default=1)
    p.add_argument("--learning-rate",       type=float, default=1e-3)
    p.add_argument("--weight-decay",        type=float, default=1e-4)
    p.add_argument("--train-ratio",         type=float, default=0.8)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--num-workers",         type=int,   default=2)

    # ── Model architecture ────────────────────────────────────────────────────
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

    # ── Feature flags ─────────────────────────────────────────────────────────
    p.add_argument("--enable-gat",             action="store_true")
    p.add_argument("--enable-multimodal",      action="store_true")
    p.add_argument("--enable-adaptive-radius", action="store_true")

    # ── Training config ───────────────────────────────────────────────────────
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
                        "Recommended: 0.05–0.1 when --enable-multimodal + K>=3.")
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
    p.add_argument("--gat-lr-scale",  type=float, default=0.1)
    p.add_argument("--gat-clip-scale", type=float, default=0.5)
    p.add_argument("--nan-patience",   type=int,   default=3)
    p.add_argument("--debug-gat",      action="store_true")
    # Misc
    p.add_argument("--limit-samples",  type=int,  default=0)
    p.add_argument("--town-filter",    type=str,  default="")
    p.add_argument("--quick-ablation", action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float(v: object) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)  # type: ignore[arg-type]


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


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Training epoch
# ══════════════════════════════════════════════════════════════════════════════

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
    # [IMP-B, IMP-C]
    diversity_weight: float = 0.0,
    augment: bool = False,
    aug_rot_std: float = 10.0,
    aug_hist_dropout: float = 0.1,
) -> Tuple[float, float, float]:
    """
    Run one epoch.  Returns (mean_loss, mean_ADE, mean_FDE).

    [IMP-B] diversity_weight > 0 adds mode repulsion to loss (multimodal only).
    [IMP-C] augment=True applies random rotation + history dropout per batch.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = total_ade = total_fde = 0.0
    n_valid = nan_batches = nonfinite_preds = 0

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

            # [IMP-B] Pass diversity_weight into loss
            loss = _compute_loss(
                pred=pred,
                target=batch["y"],
                y_mask=batch["y_mask"],
                agent_mask=batch["agent_mask"],
                diversity_weight=diversity_weight if training else 0.0,
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
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                try:
                    scaler._scale = torch.tensor(2.0 ** 12, device=device)
                except Exception:
                    pass
            continue

        if training:
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if idx % accum_steps == 0:
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
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ade, fde = _compute_metrics(
                pred=pred.detach(),
                target=batch["y"],
                y_mask=batch["y_mask"],
                agent_mask=batch["agent_mask"],
            )

        ade_val = _to_float(ade)
        fde_val = _to_float(fde)

        if not (math.isnan(ade_val) or math.isnan(fde_val)):
            total_loss += loss_val
            total_ade  += ade_val
            total_fde  += fde_val
            n_valid    += 1

        if training and log_every > 0 and idx % log_every == 0:
            print(
                f"    step={idx}/{len(loader)} loss={loss_val:.4f} "
                f"ADE={ade_val:.3f} FDE={fde_val:.3f}"
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
    avg_loss = total_loss / d if n_valid > 0 else float("nan")
    avg_ade  = total_ade  / d if n_valid > 0 else float("nan")
    avg_fde  = total_fde  / d if n_valid > 0 else float("nan")
    return avg_loss, avg_ade, avg_fde


# ══════════════════════════════════════════════════════════════════════════════
# Single training run
# ══════════════════════════════════════════════════════════════════════════════

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
    eff_clip = args.grad_clip * (args.gat_clip_scale if is_gat else 1.0)
    if is_gat:
        print(
            f"  [GAT] LR groups: base={base_lr:.2e}, gat={gat_lr:.2e} "
            f"({args.gat_lr_scale:g}×) | clip={args.grad_clip:.2f}→{eff_clip:.2f}"
        )

    optimizer = build_optimizer(
        model, base_lr=base_lr, gat_lr=gat_lr,
        weight_decay=args.weight_decay, is_gat_variant=is_gat,
    )

    # [IMP-A] Scheduler selection
    warmup_ep = getattr(args, "warmup_epochs", 0)
    if args.cosine_lr:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_ep,
            total_epochs=epochs,
            min_lr_ratio=0.01,
        )
        sched_mode = f"warmup({warmup_ep})+cosine"
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
        tr_loss, tr_ade, tr_fde = run_epoch(
            model, train_loader, device,
            optimizer=optimizer, scaler=scaler,
            accum_steps=args.accum_steps,
            grad_clip=eff_clip,
            log_every=args.log_every,
            use_amp=use_amp,
            debug_gat=args.debug_gat and is_gat,
            diversity_weight=getattr(args, "diversity_weight", 0.0),
            augment=getattr(args, "augment", False),
            aug_rot_std=getattr(args, "aug_rot_std", 10.0),
            aug_hist_dropout=getattr(args, "aug_hist_dropout", 0.1),
        )
        val_loss, val_ade, val_fde = run_epoch(
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
            tr_loss=tr_loss, tr_ade=tr_ade, tr_fde=tr_fde,
            val_loss=val_loss, val_ade=val_ade, val_fde=val_fde,
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

        marker = " ✓" if improved else ""
        print(
            f"  [{run_tag}] epoch={epoch:03d}/{epochs} lr={lr_text} t={elapsed:.0f}s | "
            f"tr_loss={tr_loss:.4f} ADE={tr_ade:.3f} FDE={tr_fde:.3f} | "
            f"val_loss={val_loss:.4f} ADE={val_ade:.3f} FDE={val_fde:.3f}{marker}"
        )

        raw_model = model.module if hasattr(model, "module") else model
        ckpt = dict(
            model_state_dict=raw_model.state_dict(),
            model_config=config_to_dict(cfg),
            epoch=epoch,
            val_loss=val_loss, val_ade=val_ade, val_fde=val_fde,
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

    results = dict(
        run_tag=run_tag,
        best_val_loss=(
            best_metric if stop_metric == "loss" and not math.isinf(best_metric)
            else best_ade_record.get("val_loss", float("nan"))
        ),
        best_val_ade=best_ade_ep,
        best_val_fde=best_fde_ep,
        best_val_ade_epoch=best_ade_record.get("epoch", -1),
        best_path=str(best_path),
        best_ade_path=str(best_ade_path),
    )
    (out_dir / f"{run_tag}_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Ablation runner
# ══════════════════════════════════════════════════════════════════════════════

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
    _print_separator("ABLATION STUDY — 8 VARIANTS")
    print(
        "\n  Variant   │ Code │ GAT │ Multi │ AdpR │ Description\n"
        "  ──────────┼──────┼─────┼───────┼──────┼──────────────────────────────"
    )
    for v in ABLATION_VARIANTS:
        g = "✓" if v.enable_gat else "-"
        m = "✓" if v.enable_multimodal else "-"
        r = "✓" if v.enable_adaptive_radius else "-"
        print(f"  {v.name:<10}│  {v.code}  │  {g}  │   {m}   │  {r}   │ {v.description}")
    print()

    for i, variant in enumerate(ABLATION_VARIANTS, start=1):
        _print_separator(f"Variant {i}/8 — {variant.name} [{variant.code}]")
        variant_dir = out_dir / f"ablation_{variant.code}_{variant.name}"
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
            result = dict(
                run_tag=f"ablation_{variant.code}",
                best_val_loss=float("nan"),
                best_val_ade=float("nan"),
                best_val_fde=float("nan"),
                best_path="N/A",
                error=str(exc),
            )

        result["variant"]     = variant.name
        result["code"]        = variant.code
        result["description"] = variant.description
        all_results.append(result)

    _print_separator("ABLATION RESULTS SUMMARY")
    print(
        f"\n  {'Variant':<14} {'Code':>4}  {'val_ADE':>8}  {'val_FDE':>8}  "
        f"{'val_loss':>10}  Status\n"
        f"  {'─'*14}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}"
    )

    def _sort_key(r: dict) -> tuple:
        v = r["best_val_ade"]
        return (0, v) if not math.isnan(v) else (1, 0)

    for r in sorted(all_results, key=_sort_key):
        ade_s  = f"{r['best_val_ade']:8.3f}"  if not math.isnan(r["best_val_ade"])  else "     NaN"
        fde_s  = f"{r['best_val_fde']:8.3f}"  if not math.isnan(r["best_val_fde"])  else "     NaN"
        loss_s = f"{r['best_val_loss']:10.4f}" if not math.isnan(r["best_val_loss"]) else "       NaN"
        status = "FAILED" if "error" in r else ("NaN" if math.isnan(r["best_val_ade"]) else "OK")
        print(f"  {r['variant']:<14}  {r['code']:>4}  {ade_s}  {fde_s}  {loss_s}  {status}")

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  [OK] Summary: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-town runner
# ══════════════════════════════════════════════════════════════════════════════

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
        print(f"  {r['town']:8s}  ADE={r['best_val_ade']:.3f}  FDE={r['best_val_fde']:.3f}")
    print(f"\n  [OK] Summary: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

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
    print(f"  Best val ADE : {result['best_val_ade']:.3f}")
    print(f"  Best val FDE : {result['best_val_fde']:.3f}")
    print(f"  Best at epoch: {result.get('best_val_ade_epoch', '?')}")
    print(f"  Checkpoint   : {result['best_path']}")
    print(f"  Best ADE ckpt: {result['best_ade_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())