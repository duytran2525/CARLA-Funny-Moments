#!/usr/bin/env python3
"""Validation script for GTNet target metrics verification.

This script evaluates a trained multi-agent trajectory prediction model on a held-out
test set and verifies that target performance metrics are achieved:
- minADE < 1.5 meters (target: 1.3m, baseline: 2.5m)
- minFDE < 2.7 meters (target: 2.5m, baseline: 4.5m)
- MissRate < 0.20 (target: 0.15, baseline: 0.35)
- inference_latency_ms < 25 milliseconds (target: 20ms)

The script generates a validation report with overall metrics, per-town breakdown,
and inference latency measurements on both GPU and CPU (if available).

**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 8.9**

Changelog (bug fixes):
  B1: load_checkpoint() now uses getattr(..., default) for advanced GTNet fields
      (enable_gat / enable_multimodal / enable_adaptive_radius / num_modes /
      num_attention_heads) that are absent from the baseline MultiAgentModelConfig.
      Previously caused AttributeError on every baseline checkpoint.
  B2: validation_report dict uses same getattr pattern — same crash path as B1.
  B3: enable_multimodal flag resolved via getattr before evaluate_model() call —
      was also AttributeError on baseline checkpoints.
  B4: Inference latency is now reported *per sample* (divided by actual batch size)
      instead of per batch. The old code reported numbers 16-32× too large.
  B5: FDE in compute_multimodal_metrics() now uses the last *valid* future timestep
      per agent (argmax over y_mask) instead of hard-coding index -1. When
      --allow-missing is active (the default), the last timestep may be masked for
      many agents, making the old displacement[..., -1] meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_dataset import (  # noqa: E402
    MultiAgentTrajectoryDataset,
    collate_multi_agent_trajectory,
    split_sample_paths,
)
from core_perception.multi_agent_model import (  # noqa: E402
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
    masked_ade_fde,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate trained model against target metrics."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        nargs="+",
        help="One or more processed dataset dirs containing manifest.csv and .pt samples.",
    )
    parser.add_argument(
        "--out-file",
        default="validation_report.json",
        help="Output path for validation report JSON file",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for test set (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test set split",
    )
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=100,
        help="Number of batches for latency measurement",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use: auto (detect), cpu, or cuda",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device from command-line argument."""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    """Move batch tensors to specified device."""
    moved = dict(batch)
    for key in ("x", "y", "adj", "x_mask", "y_mask", "agent_mask"):
        moved[key] = batch[key].to(device, non_blocking=True)  # type: ignore[union-attr]
    return moved


def _last_valid_displacement(
    displacement: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-agent FDE using the last *valid* future timestep.

    BUG FIX B5: When --allow-missing is active (dataset default), the very last
    future frame may be masked for many agents.  Hard-coding index -1 then yields
    the displacement at a missing step (i.e. interpolated/zero position) rather
    than the true final observed position.  We instead find the last timestep
    where y_mask is True per agent and fall back to index -1 only when all steps
    are masked (edge-case guard).

    Args:
        displacement: [B, N, T] or [B, N, K, T] — L2 distance tensor.
        y_mask:       [B, N, T] bool — valid future frames.
        agent_mask:   [B, N]    bool — valid agents.

    Returns:
        Tensor of shape [B, N] (or [B, N, K]) with per-agent last-valid FDE.
    """
    # Work on the last dim regardless of whether a mode dim is present.
    leading = displacement.shape[:-1]  # (B, N) or (B, N, K)
    T = displacement.shape[-1]

    # Build a [B, N, T] index mask: True only at the last valid step.
    # y_mask shape: [B, N, T]. We need to broadcast over a possible K dim later.
    B, N = y_mask.shape[0], y_mask.shape[1]

    # last_valid_idx[b, n] ∈ {0 .. T-1}, defaults to T-1 if no valid step.
    # flip along T to find the last True, take argmax, then un-flip.
    flipped = y_mask.flip(dims=[-1])              # [B, N, T]
    last_valid_idx = (T - 1) - flipped.long().argmax(dim=-1)  # [B, N]
    # When all steps are masked, argmax returns 0 → T-1 after un-flip (safe).

    if displacement.ndim == 3:
        # Unimodal path: [B, N, T]
        idx = last_valid_idx.unsqueeze(-1)        # [B, N, 1]
        return displacement.gather(-1, idx).squeeze(-1)  # [B, N]
    else:
        # Multimodal path: [B, N, K, T]
        K = displacement.shape[-2]
        idx = last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, 1)
        return displacement.gather(-1, idx).squeeze(-1)   # [B, N, K]


def compute_multimodal_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> dict:
    """Compute minADE, minFDE, MissRate for multimodal (or unimodal) predictions.

    Args:
        pred:       [B, N, K, T, 2] or [B, N, T, 2] for unimodal
        target:     [B, N, T, 2]
        y_mask:     [B, N, T] bool — valid future frames
        agent_mask: [B, N]    bool — valid agents

    Returns:
        dict with minADE (float), minFDE (float), MissRate (float)
    """
    # Normalise to [B, N, K, T, 2]
    if pred.ndim == 4:
        pred = pred.unsqueeze(2)

    batch_size, max_agents, num_modes, future_steps, _ = pred.shape

    # Expand target: [B, N, 1, T, 2]
    target_expanded = target.unsqueeze(2)

    # Per-step displacement: [B, N, K, T]
    displacement = torch.linalg.norm(pred - target_expanded, dim=-1)

    # Valid mask: [B, N, T] → [B, N, 1, T] for broadcasting over K
    valid = (y_mask & agent_mask.unsqueeze(-1)).unsqueeze(2)          # [B, N, 1, T]
    valid_float = valid.to(dtype=pred.dtype)

    # ADE per mode: [B, N, K]
    per_mode_ade = (displacement * valid_float).sum(dim=-1) / valid_float.sum(dim=-1).clamp_min(1.0)

    # FDE per mode: [B, N, K]  — BUG FIX B5: use last *valid* timestep
    per_mode_fde = _last_valid_displacement(displacement, y_mask, agent_mask)  # [B, N, K]

    # Best mode selection by minimum ADE: [B, N]
    best_mode_indices = torch.argmin(per_mode_ade, dim=2)
    best_idx_exp = best_mode_indices.unsqueeze(2)                     # [B, N, 1]

    min_ade_per_agent = per_mode_ade.gather(2, best_idx_exp).squeeze(2)   # [B, N]
    min_fde_per_agent = per_mode_fde.gather(2, best_idx_exp).squeeze(2)   # [B, N]

    # Aggregate over valid agents
    agent_mask_float = agent_mask.to(dtype=pred.dtype)
    num_valid_agents = agent_mask_float.sum().clamp_min(1.0)

    min_ade = (min_ade_per_agent * agent_mask_float).sum() / num_valid_agents
    min_fde = (min_fde_per_agent * agent_mask_float).sum() / num_valid_agents

    # MissRate: agents where minFDE > 2.0 m
    miss_threshold = 2.0
    miss_mask = (min_fde_per_agent > miss_threshold) & agent_mask
    miss_rate = miss_mask.to(dtype=pred.dtype).sum() / num_valid_agents

    return {
        "minADE": float(min_ade.detach().cpu()),
        "minFDE": float(min_fde.detach().cpu()),
        "MissRate": float(miss_rate.detach().cpu()),
    }


def evaluate_model(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    enable_multimodal: bool,
    per_town_metrics: bool = True,
) -> tuple[dict, dict]:
    """Evaluate model on test set and compute metrics.

    Returns:
        (overall_metrics dict, per_town_metrics dict)
    """
    model.eval()

    total_metrics: Dict[str, float] = {"minADE": 0.0, "minFDE": 0.0, "MissRate": 0.0}
    total_batches = 0

    town_metrics: Dict[str, Dict] = defaultdict(
        lambda: {"minADE": 0.0, "minFDE": 0.0, "MissRate": 0.0, "count": 0}
    )

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)

            pred = model(
                x=batch["x"],            # type: ignore[arg-type]
                adj=batch["adj"],         # type: ignore[arg-type]
                x_mask=batch["x_mask"],   # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )

            if enable_multimodal:
                metrics = compute_multimodal_metrics(
                    pred=pred,
                    target=batch["y"],        # type: ignore[arg-type]
                    y_mask=batch["y_mask"],   # type: ignore[arg-type]
                    agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                )
            else:
                # Unimodal: use masked_ade_fde + compute MissRate
                ade, fde = masked_ade_fde(
                    pred=pred,
                    target=batch["y"],        # type: ignore[arg-type]
                    y_mask=batch["y_mask"],   # type: ignore[arg-type]
                    agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                )

                # MissRate: last-valid FDE > 2.0 m per agent
                y_mask_bool: torch.Tensor = batch["y_mask"]      # type: ignore[assignment]
                agent_mask_bool: torch.Tensor = batch["agent_mask"]  # type: ignore[assignment]

                displacement = torch.linalg.norm(
                    pred - batch["y"], dim=-1  # type: ignore[arg-type]
                )  # [B, N, T]

                # BUG FIX B5: last *valid* timestep, not always index -1
                final_disp = _last_valid_displacement(
                    displacement, y_mask_bool, agent_mask_bool
                )  # [B, N]

                # Gate on: agent is valid AND last step is valid
                last_valid = (y_mask_bool & agent_mask_bool.unsqueeze(-1))[..., -1]  # [B, N]
                miss_mask = (final_disp > 2.0) & last_valid
                denom = last_valid.to(dtype=pred.dtype).sum().clamp_min(1.0)
                miss_rate = miss_mask.to(dtype=pred.dtype).sum() / denom

                metrics = {
                    "minADE": ade,
                    "minFDE": fde,
                    "MissRate": float(miss_rate.detach().cpu()),
                }

            for key, value in metrics.items():
                total_metrics[key] += float(value)
            total_batches += 1

            # Per-town accumulation
            if per_town_metrics and "towns" in raw_batch:
                towns = raw_batch["towns"]  # type: ignore[index]
                unique_towns = set(towns) if isinstance(towns, (list, tuple)) else {str(towns)}
                for town in unique_towns:
                    for key, value in metrics.items():
                        town_metrics[str(town)][key] += float(value)
                    town_metrics[str(town)]["count"] += 1

    denom = max(1, total_batches)
    overall_result = {k: v / denom for k, v in total_metrics.items()}

    per_town_result: Dict[str, dict] = {}
    for town, tdata in town_metrics.items():
        cnt = max(1, tdata["count"])
        per_town_result[town] = {
            "minADE": tdata["minADE"] / cnt,
            "minFDE": tdata["minFDE"] / cnt,
            "MissRate": tdata["MissRate"] / cnt,
            "num_batches": tdata["count"],
        }

    return overall_result, per_town_result


def measure_inference_latency(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> float:
    """Measure average inference latency in milliseconds **per sample**.

    BUG FIX B4: The original implementation timed a full batch and reported
    that as "ms/sample". We now divide by the actual batch size so the number
    is comparable across different --batch-size settings and matches the
    real-time budget (one sample = one scene at one timestep).

    Args:
        num_samples: Number of *batches* to time (same as before — the arg
                     name kept for CLI compatibility).

    Returns:
        Mean latency in milliseconds per **sample** (not per batch).
    """
    model.eval()
    latencies_ms: List[float] = []

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= num_samples:
                break

            batch = move_batch_to_device(raw_batch, device)
            actual_batch_size: int = batch["x"].shape[0]  # type: ignore[union-attr]

            # GPU warm-up on first batch
            if device.type == "cuda" and batch_idx == 0:
                for _ in range(5):
                    model(
                        x=batch["x"],          # type: ignore[arg-type]
                        adj=batch["adj"],       # type: ignore[arg-type]
                        x_mask=batch["x_mask"],  # type: ignore[arg-type]
                        agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                    )
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(
                x=batch["x"],          # type: ignore[arg-type]
                adj=batch["adj"],       # type: ignore[arg-type]
                x_mask=batch["x_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # BUG FIX B4: divide by batch size → per-sample latency
            latencies_ms.append(elapsed_ms / max(1, actual_batch_size))

    return sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MultiAgentTrajectoryPredictor, dict]:
    """Load model checkpoint.

    BUG FIX B1: All advanced GTNet fields (enable_gat, enable_multimodal,
    enable_adaptive_radius, num_modes, num_attention_heads) are read via
    getattr() with safe defaults.  Baseline checkpoints produced by
    train_multi_agent_trajectory.py use the plain MultiAgentModelConfig that
    does NOT contain these fields, so direct attribute access crashed.

    Returns:
        (model, checkpoint_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_config' field")

    config_dict = checkpoint["model_config"]
    model_config = MultiAgentModelConfig(**config_dict)

    model = MultiAgentTrajectoryPredictor(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # BUG FIX B1: use getattr for optional GTNet-extended fields
    print("Model configuration:")
    print(f"  hidden_dim:              {model_config.hidden_dim}")
    print(f"  graph_layers:            {getattr(model_config, 'graph_layers', '?')}")
    print(f"  future_steps:            {getattr(model_config, 'future_steps', '?')}")
    print(f"  enable_gat:              {getattr(model_config, 'enable_gat', False)}")
    print(f"  enable_multimodal:       {getattr(model_config, 'enable_multimodal', False)}")
    print(f"  enable_adaptive_radius:  {getattr(model_config, 'enable_adaptive_radius', False)}")
    print(f"  num_modes:               {getattr(model_config, 'num_modes', 1)}")
    print(f"  num_attention_heads:     {getattr(model_config, 'num_attention_heads', 1)}")

    return model, checkpoint


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    data_dirs = [Path(p).resolve() for p in args.data_dir]
    out_file = Path(args.out_file).resolve()
    device = resolve_device(str(args.device))

    print("=" * 80)
    print("GTNet Target Metrics Validation")
    print("=" * 80)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Data directories: {len(data_dirs)}")
    for d in data_dirs:
        print(f"  - {d}")
    print(f"Output file:      {out_file}")
    print(f"Device:           {device}")
    print(f"Test ratio:       {args.test_ratio}")
    print(f"Batch size:       {args.batch_size}")
    print()

    # Load model ---------------------------------------------------------------
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = model.config

    # BUG FIX B3: safe reads for all optional fields
    enable_multimodal: bool = getattr(model_config, "enable_multimodal", False)

    # Load data ----------------------------------------------------------------
    print("Loading datasets...")
    loaded_datasets = [MultiAgentTrajectoryDataset(d) for d in data_dirs]
    sample_paths = [p for ds in loaded_datasets for p in ds.sample_paths]

    if not sample_paths:
        print("Error: No samples found in datasets")
        return 1

    print(f"Total samples: {len(sample_paths)}")

    train_paths, test_paths = split_sample_paths(
        sample_paths,
        train_ratio=1.0 - float(args.test_ratio),
        seed=int(args.seed),
    )

    if not test_paths:
        print("Error: No test samples after split")
        return 1

    print(f"Test samples:  {len(test_paths)}")

    dataset_root = data_dirs[0]
    test_dataset = MultiAgentTrajectoryDataset(dataset_root, sample_files=test_paths)

    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multi_agent_trajectory,
    }
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Evaluate -----------------------------------------------------------------
    print("\nEvaluating model on test set...")
    overall_metrics, per_town_result = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        enable_multimodal=enable_multimodal,
        per_town_metrics=True,
    )

    print("\nOverall Test Metrics:")
    print(f"  minADE:   {overall_metrics['minADE']:.4f} m")
    print(f"  minFDE:   {overall_metrics['minFDE']:.4f} m")
    print(f"  MissRate: {overall_metrics['MissRate']:.4f}")

    if per_town_result:
        print("\nPer-Town Metrics:")
        for town in sorted(per_town_result.keys()):
            td = per_town_result[town]
            print(f"  {town}: minADE={td['minADE']:.4f} m  "
                  f"minFDE={td['minFDE']:.4f} m  "
                  f"MissRate={td['MissRate']:.4f}  "
                  f"batches={td['num_batches']}")

    # Latency ------------------------------------------------------------------
    print(f"\nMeasuring inference latency ({args.latency_samples} batches)...")
    latency_ms = measure_inference_latency(
        model=model,
        loader=test_loader,
        device=device,
        num_samples=int(args.latency_samples),
    )
    latency_results = {str(device.type): latency_ms}
    print(f"  {device.type.upper()} latency: {latency_ms:.3f} ms/sample")

    if device.type == "cuda":
        print("  Measuring CPU latency...")
        cpu_device = torch.device("cpu")
        cpu_model = model.to(cpu_device)
        cpu_loader_kwargs = {**loader_kwargs, "pin_memory": False}
        cpu_loader = DataLoader(test_dataset, shuffle=False, **cpu_loader_kwargs)
        cpu_latency_ms = measure_inference_latency(
            model=cpu_model, loader=cpu_loader, device=cpu_device,
            num_samples=int(args.latency_samples),
        )
        latency_results["cpu"] = cpu_latency_ms
        print(f"  CPU latency: {cpu_latency_ms:.3f} ms/sample")
        model = cpu_model.to(device)

    # Target check -------------------------------------------------------------
    TARGET_MIN_ADE    = 1.5
    TARGET_MIN_FDE    = 2.7
    TARGET_MISS_RATE  = 0.20
    TARGET_LATENCY_MS = 25.0

    print("\n" + "=" * 80)
    print("Target Metrics Verification")
    print("=" * 80)

    minADE_met   = overall_metrics["minADE"]   < TARGET_MIN_ADE
    minFDE_met   = overall_metrics["minFDE"]   < TARGET_MIN_FDE
    missRate_met = overall_metrics["MissRate"] < TARGET_MISS_RATE
    primary_lat  = latency_results.get("cuda", latency_results.get("cpu", 0.0))
    latency_met  = primary_lat < TARGET_LATENCY_MS
    targets_met  = minADE_met and minFDE_met and missRate_met and latency_met

    print(f"minADE:    {overall_metrics['minADE']:.4f} < {TARGET_MIN_ADE:.4f} m  "
          f"{'[PASS]' if minADE_met else '[FAIL]'}")
    print(f"minFDE:    {overall_metrics['minFDE']:.4f} < {TARGET_MIN_FDE:.4f} m  "
          f"{'[PASS]' if minFDE_met else '[FAIL]'}")
    print(f"MissRate:  {overall_metrics['MissRate']:.4f} < {TARGET_MISS_RATE:.4f}     "
          f"{'[PASS]' if missRate_met else '[FAIL]'}")
    print(f"Latency:   {primary_lat:.3f} < {TARGET_LATENCY_MS:.1f} ms/sample  "
          f"{'[PASS]' if latency_met else '[FAIL]'}")
    print("=" * 80)
    print("\n[OK] All target metrics achieved!" if targets_met
          else "\n[FAIL] Some target metrics not achieved")

    # Report -------------------------------------------------------------------
    # BUG FIX B2: all optional model_config fields use getattr
    validation_report = {
        "checkpoint": str(checkpoint_path),
        "data_dirs": [str(d) for d in data_dirs],
        "test_samples": len(test_paths),
        "device": str(device),
        "model_config": {
            "hidden_dim":              model_config.hidden_dim,
            "graph_layers":            getattr(model_config, "graph_layers", None),
            "future_steps":            getattr(model_config, "future_steps", None),
            "enable_gat":              getattr(model_config, "enable_gat", False),
            "enable_multimodal":       getattr(model_config, "enable_multimodal", False),
            "enable_adaptive_radius":  getattr(model_config, "enable_adaptive_radius", False),
            "num_modes":               getattr(model_config, "num_modes", 1),
            "num_attention_heads":     getattr(model_config, "num_attention_heads", 1),
        },
        "overall_metrics": overall_metrics,
        "per_town_metrics": per_town_result,
        "inference_latency_ms_per_sample": latency_results,
        "target_metrics": {
            "minADE_m": TARGET_MIN_ADE,
            "minFDE_m": TARGET_MIN_FDE,
            "MissRate": TARGET_MISS_RATE,
            "latency_ms_per_sample": TARGET_LATENCY_MS,
        },
        "targets_met": {
            "minADE":               minADE_met,
            "minFDE":               minFDE_met,
            "MissRate":             missRate_met,
            "inference_latency_ms": latency_met,
            "all":                  targets_met,
        },
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(validation_report, fh, indent=2)

    print(f"\n[OK] Validation report saved to: {out_file}")

    if not targets_met:
        print("\n[ERROR] Target metrics not achieved. Exiting with error code 1.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())