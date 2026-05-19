#!/usr/bin/env python3
"""Ablation study script for GTNet improvements.

This script trains 8 model variants (all combinations of 3 binary flags) to measure
the individual contribution of each improvement:
- enable_gat: Graph Attention Networks
- enable_multimodal: Multimodal prediction with WTA loss
- enable_adaptive_radius: Adaptive interaction radius

All variants use identical training data, hyperparameters, and random seeds.
Results are logged to ablation_results.json with metrics and timing information.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.10**
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict

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
    masked_smooth_l1_loss,
    wta_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation study for GTNet improvements."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        nargs="+",
        help="One or more processed dataset dirs containing manifest.csv and .pt samples.",
    )
    parser.add_argument(
        "--out-dir",
        default="ablation_results",
        help="Output directory for ablation results and checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per variant")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use: auto (detect), cpu, or cuda",
    )
    parser.add_argument(
        "--quick-ablation",
        action="store_true",
        help="Quick smoke test mode with reduced epochs and samples",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Limit number of samples for testing (0 = no limit)",
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def compute_multimodal_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> dict:
    """Compute minADE, minFDE, MissRate for multimodal predictions.
    
    Args:
        pred: Predicted trajectories [batch_size, max_agents, num_modes, future_steps, 2]
        target: Ground truth trajectories [batch_size, max_agents, future_steps, 2]
        y_mask: Valid future frames [batch_size, max_agents, future_steps]
        agent_mask: Valid agents [batch_size, max_agents]
    
    Returns:
        Dictionary with minADE, minFDE, MissRate
    """
    batch_size, max_agents, num_modes, future_steps, _ = pred.shape
    
    # Expand target to match pred shape: [B, N, 1, T, 2]
    target_expanded = target.unsqueeze(2)
    
    # Compute displacement for all modes: [B, N, K, T]
    displacement = torch.linalg.norm(pred - target_expanded, dim=-1)
    
    # Valid mask: [B, N, T] -> [B, N, 1, T] for broadcasting
    valid = (y_mask & agent_mask.unsqueeze(-1)).unsqueeze(2)
    valid_float = valid.to(dtype=pred.dtype)
    
    # Compute ADE per mode: [B, N, K]
    per_mode_ade = (displacement * valid_float).sum(dim=3) / valid_float.sum(dim=3).clamp_min(1.0)
    
    # Compute FDE per mode: [B, N, K]
    per_mode_fde = displacement[..., -1]  # [B, N, K]
    
    # Select best mode per agent (minimum ADE)
    best_mode_indices = torch.argmin(per_mode_ade, dim=2)  # [B, N]
    
    # Gather minADE and minFDE
    best_mode_indices_expanded = best_mode_indices.unsqueeze(2)  # [B, N, 1]
    min_ade_per_agent = torch.gather(per_mode_ade, dim=2, index=best_mode_indices_expanded).squeeze(2)
    min_fde_per_agent = torch.gather(per_mode_fde, dim=2, index=best_mode_indices_expanded).squeeze(2)
    
    # Apply agent mask and compute mean
    agent_mask_float = agent_mask.to(dtype=pred.dtype)
    num_valid_agents = agent_mask_float.sum().clamp_min(1.0)
    
    min_ade = (min_ade_per_agent * agent_mask_float).sum() / num_valid_agents
    min_fde = (min_fde_per_agent * agent_mask_float).sum() / num_valid_agents
    
    # Compute MissRate (fraction where minFDE > 2.0m)
    miss_threshold = 2.0
    miss_mask = (min_fde_per_agent > miss_threshold) & agent_mask
    miss_rate = miss_mask.to(dtype=pred.dtype).sum() / num_valid_agents
    
    return {
        "minADE": float(min_ade.detach().cpu().item()),
        "minFDE": float(min_fde.detach().cpu().item()),
        "MissRate": float(miss_rate.detach().cpu().item()),
    }


def run_epoch(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    enable_multimodal: bool = False,
) -> tuple[float, dict]:
    """Run one epoch of training or validation.
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_metrics = {}
    total_batches = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred = model(
                x=batch["x"],  # type: ignore[arg-type]
                adj=batch["adj"],  # type: ignore[arg-type]
                x_mask=batch["x_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )
            
            # Use WTA loss for multimodal, smooth L1 for unimodal
            if enable_multimodal:
                loss = wta_loss(
                    pred=pred,
                    target=batch["y"],  # type: ignore[arg-type]
                    y_mask=batch["y_mask"],  # type: ignore[arg-type]
                    agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                )
            else:
                loss = masked_smooth_l1_loss(
                    pred=pred,
                    target=batch["y"],  # type: ignore[arg-type]
                    y_mask=batch["y_mask"],  # type: ignore[arg-type]
                    agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                )

        if training:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        # Compute metrics
        if enable_multimodal:
            metrics = compute_multimodal_metrics(
                pred=pred,
                target=batch["y"],  # type: ignore[arg-type]
                y_mask=batch["y_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )
        else:
            ade, fde = masked_ade_fde(
                pred=pred,
                target=batch["y"],  # type: ignore[arg-type]
                y_mask=batch["y_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )
            metrics = {"ADE": ade, "FDE": fde, "MissRate": 0.0}
        
        total_loss += float(loss.detach().cpu().item())
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        total_batches += 1

    denom = max(1, total_batches)
    avg_metrics = {key: value / denom for key, value in total_metrics.items()}
    return total_loss / denom, avg_metrics


def measure_inference_latency(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 100,
) -> float:
    """Measure inference latency in milliseconds per sample.
    
    Args:
        model: The trajectory prediction model
        loader: DataLoader for the dataset
        device: Device to run on
        num_samples: Number of samples to measure
    
    Returns:
        Average inference latency in milliseconds per sample
    """
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= num_samples:
                break
            
            batch = move_batch_to_device(raw_batch, device)
            
            # Warm-up for GPU
            if device.type == "cuda" and batch_idx == 0:
                for _ in range(5):
                    _ = model(
                        x=batch["x"],  # type: ignore[arg-type]
                        adj=batch["adj"],  # type: ignore[arg-type]
                        x_mask=batch["x_mask"],  # type: ignore[arg-type]
                        agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                    )
                torch.cuda.synchronize()
            
            # Measure inference time
            start_time = time.perf_counter()
            _ = model(
                x=batch["x"],  # type: ignore[arg-type]
                adj=batch["adj"],  # type: ignore[arg-type]
                x_mask=batch["x_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)
    
    return sum(latencies) / len(latencies) if latencies else 0.0


def train_variant(
    variant_name: str,
    enable_gat: bool,
    enable_multimodal: bool,
    enable_adaptive_radius: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    input_dim: int,
    future_steps: int,
) -> dict:
    """Train a single model variant and return results.
    
    Args:
        variant_name: Name of the variant (e.g., "baseline", "gat_only")
        enable_gat: Enable Graph Attention Networks
        enable_multimodal: Enable multimodal prediction
        enable_adaptive_radius: Enable adaptive radius
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        args: Command-line arguments
        input_dim: Input feature dimension
        future_steps: Number of future timesteps
    
    Returns:
        Dictionary with metrics and timing information
    """
    print(f"\n{'='*80}")
    print(f"Training variant: {variant_name}")
    print(f"  GAT: {enable_gat}, Multimodal: {enable_multimodal}, Adaptive Radius: {enable_adaptive_radius}")
    print(f"{'='*80}")
    
    # Create model configuration
    model_config = MultiAgentModelConfig(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        graph_layers=int(args.graph_layers),
        future_steps=future_steps,
        dropout=float(args.dropout),
        enable_gat=enable_gat,
        num_attention_heads=4,
        enable_multimodal=enable_multimodal,
        num_modes=3,
        enable_adaptive_radius=enable_adaptive_radius,
    )
    
    # Create model and optimizer
    model = MultiAgentTrajectoryPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    
    # Training loop
    best_val_metric = math.inf
    stale_epochs = 0
    train_start_time = time.time()
    
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=float(args.grad_clip),
            enable_multimodal=enable_multimodal,
        )
        
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                enable_multimodal=enable_multimodal,
            )
        
        # Use minADE for early stopping if multimodal, else use ADE
        if enable_multimodal:
            val_metric = val_metrics.get("minADE", val_loss)
        else:
            val_metric = val_metrics.get("ADE", val_loss)
        
        scheduler.step(val_metric)
        
        improved = val_metric < best_val_metric
        if improved:
            best_val_metric = val_metric
            stale_epochs = 0
        else:
            stale_epochs += 1
        
        # Log progress
        if epoch % 5 == 0 or improved:
            marker = " [BEST]" if improved else ""
            print(f"  Epoch {epoch:03d}: val_metric={val_metric:.4f}{marker}")
        
        # Early stopping
        if stale_epochs >= int(args.early_stopping_patience):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    train_time = time.time() - train_start_time
    
    # Final evaluation
    with torch.no_grad():
        _, final_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            enable_multimodal=enable_multimodal,
        )
    
    # Measure inference latency
    inference_latency = measure_inference_latency(
        model=model,
        loader=val_loader,
        device=device,
        num_samples=100,
    )
    
    # Prepare results
    results = {
        "variant_name": variant_name,
        "enable_gat": enable_gat,
        "enable_multimodal": enable_multimodal,
        "enable_adaptive_radius": enable_adaptive_radius,
        "train_time_seconds": train_time,
        "inference_latency_ms": inference_latency,
    }
    
    # Add metrics (use consistent naming)
    if enable_multimodal:
        results["minADE"] = final_metrics.get("minADE", 0.0)
        results["minFDE"] = final_metrics.get("minFDE", 0.0)
        results["MissRate"] = final_metrics.get("MissRate", 0.0)
    else:
        results["minADE"] = final_metrics.get("ADE", 0.0)
        results["minFDE"] = final_metrics.get("FDE", 0.0)
        results["MissRate"] = final_metrics.get("MissRate", 0.0)
    
    print(f"\nVariant {variant_name} Results:")
    print(f"  minADE: {results['minADE']:.4f}")
    print(f"  minFDE: {results['minFDE']:.4f}")
    print(f"  MissRate: {results['MissRate']:.4f}")
    print(f"  Train Time: {train_time:.1f}s")
    print(f"  Inference Latency: {inference_latency:.2f}ms")
    
    return results


def generate_comparison_table(results: dict, baseline_results: dict) -> str:
    """Generate comparison table showing improvements over baseline.
    
    Args:
        results: Dictionary mapping variant names to their results
        baseline_results: Results for the baseline variant
    
    Returns:
        Formatted comparison table as string
    """
    baseline_ade = baseline_results["minADE"]
    baseline_fde = baseline_results["minFDE"]
    baseline_miss = baseline_results["MissRate"]
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("="*100)
    lines.append(f"{'Variant':<25} {'minADE':>10} {'Δ ADE':>10} {'minFDE':>10} {'Δ FDE':>10} {'MissRate':>10} {'Δ Miss':>10}")
    lines.append("-"*100)
    
    # Sort variants: baseline first, then alphabetically
    sorted_variants = sorted(results.keys(), key=lambda x: (x != "baseline", x))
    
    for variant_name in sorted_variants:
        variant_results = results[variant_name]
        ade = variant_results["minADE"]
        fde = variant_results["minFDE"]
        miss = variant_results["MissRate"]
        
        # Compute deltas (negative = improvement)
        delta_ade = ade - baseline_ade
        delta_fde = fde - baseline_fde
        delta_miss = miss - baseline_miss
        
        # Format deltas with color indicators
        delta_ade_str = f"{delta_ade:+.4f}" if variant_name != "baseline" else "-"
        delta_fde_str = f"{delta_fde:+.4f}" if variant_name != "baseline" else "-"
        delta_miss_str = f"{delta_miss:+.4f}" if variant_name != "baseline" else "-"
        
        lines.append(
            f"{variant_name:<25} {ade:>10.4f} {delta_ade_str:>10} "
            f"{fde:>10.4f} {delta_fde_str:>10} "
            f"{miss:>10.4f} {delta_miss_str:>10}"
        )
    
    lines.append("="*100)
    lines.append("\nTiming Information:")
    lines.append("-"*100)
    lines.append(f"{'Variant':<25} {'Train Time (s)':>20} {'Inference (ms)':>20}")
    lines.append("-"*100)
    
    for variant_name in sorted_variants:
        variant_results = results[variant_name]
        train_time = variant_results["train_time_seconds"]
        inference_time = variant_results["inference_latency_ms"]
        lines.append(f"{variant_name:<25} {train_time:>20.1f} {inference_time:>20.2f}")
    
    lines.append("="*100)
    
    return "\n".join(lines)


def main() -> int:
    """Main function to run ablation study."""
    args = parse_args()
    
    # Apply quick-ablation mode settings
    if args.quick_ablation:
        print("Quick ablation mode enabled: reducing epochs and samples")
        args.epochs = 5
        if args.limit_samples == 0:
            args.limit_samples = 200
    
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    data_dirs = [Path(path).resolve() for path in args.data_dir]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ablation Study Configuration:")
    print(f"  Data directories: {len(data_dirs)}")
    for data_dir in data_dirs:
        print(f"    - {data_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Quick ablation: {args.quick_ablation}")
    
    # Load datasets
    loaded_datasets = [MultiAgentTrajectoryDataset(data_dir) for data_dir in data_dirs]
    sample_paths = [path for dataset in loaded_datasets for path in dataset.sample_paths]
    
    if int(args.limit_samples) > 0:
        sample_paths = sample_paths[: int(args.limit_samples)]
    
    if not sample_paths:
        print("Error: No samples found")
        return 1
    
    print(f"  Total samples: {len(sample_paths)}")
    
    # Split into train/val with fixed seed for reproducibility
    train_paths, val_paths = split_sample_paths(
        sample_paths,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )
    if not val_paths:
        val_paths = train_paths[-1:]
        train_paths = train_paths[:-1] or val_paths
    
    print(f"  Train samples: {len(train_paths)}")
    print(f"  Val samples: {len(val_paths)}")
    
    # Create datasets
    dataset_root = data_dirs[0]
    train_dataset = MultiAgentTrajectoryDataset(dataset_root, sample_files=train_paths)
    val_dataset = MultiAgentTrajectoryDataset(dataset_root, sample_files=val_paths)
    
    # Get input dimensions from first sample
    first_sample = train_dataset[0]
    future_steps = int(first_sample["y"].shape[1])  # type: ignore[index]
    input_dim = int(first_sample["x"].shape[2])  # type: ignore[index]
    
    # Create data loaders
    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multi_agent_trajectory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    # Define all 8 variants (all combinations of 3 binary flags)
    variants = [
        ("baseline", False, False, False),
        ("gat_only", True, False, False),
        ("multimodal_only", False, True, False),
        ("adaptive_radius_only", False, False, True),
        ("gat_multimodal", True, True, False),
        ("gat_adaptive", True, False, True),
        ("multimodal_adaptive", False, True, True),
        ("full", True, True, True),
    ]
    
    print(f"\nTraining {len(variants)} variants...")
    
    # Train all variants
    all_results = {}
    for variant_name, enable_gat, enable_multimodal, enable_adaptive_radius in variants:
        # Reset seed before each variant for reproducibility
        set_seed(int(args.seed))
        
        variant_results = train_variant(
            variant_name=variant_name,
            enable_gat=enable_gat,
            enable_multimodal=enable_multimodal,
            enable_adaptive_radius=enable_adaptive_radius,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
            input_dim=input_dim,
            future_steps=future_steps,
        )
        
        all_results[variant_name] = variant_results
    
    # Save results to JSON
    results_path = out_dir / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[OK] Results saved to: {results_path}")
    
    # Generate and print comparison table
    baseline_results = all_results["baseline"]
    comparison_table = generate_comparison_table(all_results, baseline_results)
    print(comparison_table)
    
    # Save comparison table to text file
    table_path = out_dir / "comparison_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(comparison_table)
    
    print(f"[OK] Comparison table saved to: {table_path}")
    
    # Verify full model achieves target metrics (if not in quick mode)
    if not args.quick_ablation:
        full_results = all_results["full"]
        target_minADE = 1.5
        target_minFDE = 2.7
        target_MissRate = 0.20
        
        print("\nTarget Metrics Validation:")
        print(f"  minADE: {full_results['minADE']:.4f} (target: < {target_minADE})")
        print(f"  minFDE: {full_results['minFDE']:.4f} (target: < {target_minFDE})")
        print(f"  MissRate: {full_results['MissRate']:.4f} (target: < {target_MissRate})")
        
        meets_targets = (
            full_results['minADE'] < target_minADE and
            full_results['minFDE'] < target_minFDE and
            full_results['MissRate'] < target_MissRate
        )
        
        if meets_targets:
            print("\n✓ Full model meets all target metrics!")
        else:
            print("\n✗ Full model does not meet all target metrics")
            print("  Note: This may be due to limited training data or epochs")
    
    print("\n[OK] Ablation study complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
