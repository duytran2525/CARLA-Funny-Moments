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
        help="Number of samples for latency measurement",
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


def compute_multimodal_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> dict:
    """Compute minADE, minFDE, MissRate for multimodal predictions.
    
    Args:
        pred: Predicted trajectories [batch_size, max_agents, num_modes, future_steps, 2]
              or [batch_size, max_agents, future_steps, 2] for unimodal
        target: Ground truth trajectories [batch_size, max_agents, future_steps, 2]
        y_mask: Valid future frames [batch_size, max_agents, future_steps]
        agent_mask: Valid agents [batch_size, max_agents]
    
    Returns:
        Dictionary with minADE, minFDE, MissRate
    """
    # Handle both unimodal and multimodal predictions
    if pred.ndim == 4:
        # Unimodal: [B, N, T, 2] -> add mode dimension [B, N, 1, T, 2]
        pred = pred.unsqueeze(2)
    
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


def evaluate_model(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    enable_multimodal: bool,
    per_town_metrics: bool = True,
) -> tuple[dict, dict]:
    """Evaluate model on test set and compute metrics.
    
    Args:
        model: The trajectory prediction model
        loader: DataLoader for the test set
        device: Device to run on
        enable_multimodal: Whether model uses multimodal prediction
        per_town_metrics: Whether to compute per-town breakdown
    
    Returns:
        Tuple of (overall_metrics, per_town_metrics_dict)
    """
    model.eval()
    
    # Accumulators for overall metrics
    total_metrics = {"minADE": 0.0, "minFDE": 0.0, "MissRate": 0.0}
    total_batches = 0
    
    # Accumulators for per-town metrics
    town_metrics = defaultdict(lambda: {"minADE": 0.0, "minFDE": 0.0, "MissRate": 0.0, "count": 0})
    
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            
            # Run inference
            pred = model(
                x=batch["x"],  # type: ignore[arg-type]
                adj=batch["adj"],  # type: ignore[arg-type]
                x_mask=batch["x_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )
            
            # Compute metrics
            if enable_multimodal:
                metrics = compute_multimodal_metrics(
                    pred=pred,
                    target=batch["y"],  # type: ignore[arg-type]
                    y_mask=batch["y_mask"],  # type: ignore[arg-type]
                    agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                )
            else:
                # For unimodal, use standard ADE/FDE and compute MissRate
                ade, fde = masked_ade_fde(
                    pred=pred,
                    target=batch["y"],  # type: ignore[arg-type]
                    y_mask=batch["y_mask"],  # type: ignore[arg-type]
                    agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
                )
                # Compute MissRate for unimodal
                agent_mask_bool = batch["agent_mask"]  # type: ignore[assignment]
                y_mask_bool = batch["y_mask"]  # type: ignore[assignment]
                valid = y_mask_bool & agent_mask_bool.unsqueeze(-1)  # type: ignore[union-attr]
                final_valid = valid[..., -1]
                displacement = torch.linalg.norm(pred - batch["y"], dim=-1)  # type: ignore[arg-type]
                final_displacement = displacement[..., -1]
                miss_mask = (final_displacement > 2.0) & final_valid
                miss_rate = miss_mask.to(dtype=pred.dtype).sum() / final_valid.to(dtype=pred.dtype).sum().clamp_min(1.0)
                
                metrics = {
                    "minADE": ade,
                    "minFDE": fde,
                    "MissRate": float(miss_rate.detach().cpu().item()),
                }
            
            # Accumulate overall metrics
            for key, value in metrics.items():
                total_metrics[key] += value
            total_batches += 1
            
            # Accumulate per-town metrics if requested
            if per_town_metrics and "towns" in raw_batch:
                towns = raw_batch["towns"]  # type: ignore[index]
                if isinstance(towns, (list, tuple)):
                    # Batch may contain multiple towns
                    for town in set(towns):
                        for key, value in metrics.items():
                            town_metrics[town][key] += value
                        town_metrics[town]["count"] += 1
                else:
                    # Single town for entire batch
                    town = str(towns)
                    for key, value in metrics.items():
                        town_metrics[town][key] += value
                    town_metrics[town]["count"] += 1
    
    # Compute average overall metrics
    denom = max(1, total_batches)
    overall_metrics = {key: value / denom for key, value in total_metrics.items()}
    
    # Compute average per-town metrics
    per_town_results = {}
    for town, town_data in town_metrics.items():
        count = max(1, town_data["count"])
        per_town_results[town] = {
            "minADE": town_data["minADE"] / count,
            "minFDE": town_data["minFDE"] / count,
            "MissRate": town_data["MissRate"] / count,
            "num_batches": town_data["count"],
        }
    
    return overall_metrics, per_town_results


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


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[MultiAgentTrajectoryPredictor, dict]:
    """Load model checkpoint and return model and metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    if "model_config" in checkpoint:
        config_dict = checkpoint["model_config"]
        model_config = MultiAgentModelConfig(**config_dict)
    else:
        raise ValueError("Checkpoint missing 'model_config' field")
    
    # Create model and load weights
    model = MultiAgentTrajectoryPredictor(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model configuration:")
    print(f"  enable_gat: {model_config.enable_gat}")
    print(f"  enable_multimodal: {model_config.enable_multimodal}")
    print(f"  enable_adaptive_radius: {model_config.enable_adaptive_radius}")
    print(f"  hidden_dim: {model_config.hidden_dim}")
    print(f"  num_modes: {model_config.num_modes}")
    
    return model, checkpoint


def main() -> int:
    """Main function to run validation."""
    args = parse_args()
    
    # Resolve paths and device
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    data_dirs = [Path(path).resolve() for path in args.data_dir]
    out_file = Path(args.out_file).resolve()
    device = resolve_device(str(args.device))
    
    print("="*80)
    print("GTNet Target Metrics Validation")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directories: {len(data_dirs)}")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")
    print(f"Output file: {out_file}")
    print(f"Device: {device}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load model
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = model.config
    enable_multimodal = model_config.enable_multimodal
    
    # Load datasets and create test split
    print("Loading datasets...")
    loaded_datasets = [MultiAgentTrajectoryDataset(data_dir) for data_dir in data_dirs]
    sample_paths = [path for dataset in loaded_datasets for path in dataset.sample_paths]
    
    if not sample_paths:
        print("Error: No samples found in datasets")
        return 1
    
    print(f"Total samples: {len(sample_paths)}")
    
    # Split into train/test (we only use test set)
    train_paths, test_paths = split_sample_paths(
        sample_paths,
        train_ratio=1.0 - float(args.test_ratio),
        seed=int(args.seed),
    )
    
    if not test_paths:
        print("Error: No test samples after split")
        return 1
    
    print(f"Test samples: {len(test_paths)}")
    
    # Create test dataset and loader
    dataset_root = data_dirs[0]
    test_dataset = MultiAgentTrajectoryDataset(dataset_root, sample_files=test_paths)
    
    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multi_agent_trajectory,
    }
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    overall_metrics, per_town_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        enable_multimodal=enable_multimodal,
        per_town_metrics=True,
    )
    
    print("\nOverall Test Metrics:")
    print(f"  minADE: {overall_metrics['minADE']:.4f} meters")
    print(f"  minFDE: {overall_metrics['minFDE']:.4f} meters")
    print(f"  MissRate: {overall_metrics['MissRate']:.4f}")
    
    if per_town_metrics:
        print("\nPer-Town Metrics:")
        for town in sorted(per_town_metrics.keys()):
            town_data = per_town_metrics[town]
            print(f"  {town}:")
            print(f"    minADE: {town_data['minADE']:.4f} meters")
            print(f"    minFDE: {town_data['minFDE']:.4f} meters")
            print(f"    MissRate: {town_data['MissRate']:.4f}")
            print(f"    num_batches: {town_data['num_batches']}")
    
    # Measure inference latency
    print(f"\nMeasuring inference latency ({args.latency_samples} samples)...")
    
    # Measure on current device
    latency_ms = measure_inference_latency(
        model=model,
        loader=test_loader,
        device=device,
        num_samples=int(args.latency_samples),
    )
    
    latency_results = {
        str(device.type): latency_ms
    }
    
    print(f"  {device.type.upper()} latency: {latency_ms:.2f} ms/sample")
    
    # Measure on CPU if we're currently on GPU
    if device.type == "cuda":
        print("  Measuring CPU latency...")
        cpu_device = torch.device("cpu")
        model_cpu = model.to(cpu_device)
        
        # Create CPU loader
        cpu_loader_kwargs = dict(loader_kwargs)
        cpu_loader_kwargs["pin_memory"] = False
        cpu_loader = DataLoader(test_dataset, shuffle=False, **cpu_loader_kwargs)
        
        cpu_latency_ms = measure_inference_latency(
            model=model_cpu,
            loader=cpu_loader,
            device=cpu_device,
            num_samples=int(args.latency_samples),
        )
        
        latency_results["cpu"] = cpu_latency_ms
        print(f"  CPU latency: {cpu_latency_ms:.2f} ms/sample")
        
        # Move model back to GPU
        model = model.to(device)
    
    # Define target metrics
    target_minADE = 1.5
    target_minFDE = 2.7
    target_MissRate = 0.20
    target_latency_ms = 25.0
    
    # Check if targets are met
    print("\n" + "="*80)
    print("Target Metrics Verification")
    print("="*80)
    
    targets_met = True
    
    # Check minADE
    minADE_met = overall_metrics["minADE"] < target_minADE
    status_ade = "✓ PASS" if minADE_met else "✗ FAIL"
    print(f"minADE: {overall_metrics['minADE']:.4f} < {target_minADE:.4f} ... {status_ade}")
    targets_met = targets_met and minADE_met
    
    # Check minFDE
    minFDE_met = overall_metrics["minFDE"] < target_minFDE
    status_fde = "✓ PASS" if minFDE_met else "✗ FAIL"
    print(f"minFDE: {overall_metrics['minFDE']:.4f} < {target_minFDE:.4f} ... {status_fde}")
    targets_met = targets_met and minFDE_met
    
    # Check MissRate
    MissRate_met = overall_metrics["MissRate"] < target_MissRate
    status_miss = "✓ PASS" if MissRate_met else "✗ FAIL"
    print(f"MissRate: {overall_metrics['MissRate']:.4f} < {target_MissRate:.4f} ... {status_miss}")
    targets_met = targets_met and MissRate_met
    
    # Check inference latency (use GPU latency if available, else CPU)
    primary_latency = latency_results.get("cuda", latency_results.get("cpu", 0.0))
    latency_met = primary_latency < target_latency_ms
    status_latency = "✓ PASS" if latency_met else "✗ FAIL"
    print(f"Inference Latency: {primary_latency:.2f} < {target_latency_ms:.2f} ms ... {status_latency}")
    targets_met = targets_met and latency_met
    
    print("="*80)
    
    if targets_met:
        print("\n✓ All target metrics achieved!")
    else:
        print("\n✗ Some target metrics not achieved")
    
    # Generate validation report
    validation_report = {
        "checkpoint": str(checkpoint_path),
        "data_dirs": [str(d) for d in data_dirs],
        "test_samples": len(test_paths),
        "device": str(device),
        "model_config": {
            "enable_gat": model_config.enable_gat,
            "enable_multimodal": model_config.enable_multimodal,
            "enable_adaptive_radius": model_config.enable_adaptive_radius,
            "hidden_dim": model_config.hidden_dim,
            "num_modes": model_config.num_modes,
            "num_attention_heads": model_config.num_attention_heads,
        },
        "overall_metrics": {
            "minADE": overall_metrics["minADE"],
            "minFDE": overall_metrics["minFDE"],
            "MissRate": overall_metrics["MissRate"],
        },
        "per_town_metrics": per_town_metrics,
        "inference_latency_ms": latency_results,
        "target_metrics": {
            "minADE": target_minADE,
            "minFDE": target_minFDE,
            "MissRate": target_MissRate,
            "inference_latency_ms": target_latency_ms,
        },
        "targets_met": {
            "minADE": minADE_met,
            "minFDE": minFDE_met,
            "MissRate": MissRate_met,
            "inference_latency_ms": latency_met,
            "all": targets_met,
        },
    }
    
    # Save validation report
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n[OK] Validation report saved to: {out_file}")
    
    # Exit with error code if targets not met
    if not targets_met:
        print("\n[ERROR] Target metrics not achieved. Exiting with error code 1.")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
