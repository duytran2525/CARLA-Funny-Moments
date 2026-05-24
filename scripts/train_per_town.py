from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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
        description="Train per-town multi-agent trajectory predictor with Kaggle optimization."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        nargs="+",
        help="One or more processed dataset dirs containing manifest.csv and .pt samples.",
    )
    parser.add_argument(
        "--town-filter",
        nargs="+",
        default=None,
        help="Train on specific towns only (e.g., Town01 Town02). If not specified, trains on all towns.",
    )
    parser.add_argument(
        "--out-dir",
        default="models/multi_agent",
        help="Base checkpoint/output directory. Town-specific subdirs will be created.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm (default: 1.0)")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Early stopping patience in epochs (default: 8)",
    )
    parser.add_argument("--log-every", type=int, default=20, help="Log metrics every N batches")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use: auto (detect), cpu, or cuda",
    )
    parser.add_argument("--limit-samples", type=int, default=0, help="Optional smoke-test sample cap.")
    
    # Model configuration flags
    parser.add_argument("--enable-gat", action="store_true", help="Enable Graph Attention Networks")
    parser.add_argument("--enable-multimodal", action="store_true", help="Enable multimodal prediction")
    parser.add_argument("--enable-adaptive-radius", action="store_true", help="Enable adaptive radius")
    parser.add_argument("--num-modes", type=int, default=3, help="Number of trajectory modes (default: 3)")
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=4,
        help="Number of attention heads for GAT (default: 4)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = dict(batch)
    for key in ("x", "y", "adj", "x_mask", "y_mask", "agent_mask"):
        moved[key] = batch[key].to(device, non_blocking=True)  # type: ignore[union-attr]
    return moved


def filter_samples_by_town(sample_paths: list[Path], town_filter: list[str] | None) -> list[Path]:
    """Filter sample paths to include only specified towns.
    
    Args:
        sample_paths: List of paths to .pt sample files
        town_filter: List of town names to include (e.g., ["Town01", "Town02"]), or None for all
    
    Returns:
        Filtered list of sample paths
    """
    if town_filter is None:
        return sample_paths
    
    # Load each sample and check its town metadata
    filtered_paths = []
    for sample_path in sample_paths:
        try:
            sample = torch.load(sample_path, map_location="cpu")
            town = sample.get("town", "")
            if town in town_filter:
                filtered_paths.append(sample_path)
        except Exception as e:
            print(f"Warning: Failed to load sample {sample_path}: {e}")
            continue
    
    return filtered_paths


def compute_multimodal_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mask: torch.Tensor,
    agent_mask: torch.Tensor,
) -> dict:
    """Compute minADE, minFDE, and per-mode metrics for multimodal predictions.
    
    Args:
        pred: Predicted trajectories [batch_size, max_agents, num_modes, future_steps, 2]
        target: Ground truth trajectories [batch_size, max_agents, future_steps, 2]
        y_mask: Valid future frames [batch_size, max_agents, future_steps]
        agent_mask: Valid agents [batch_size, max_agents]
    
    Returns:
        Dictionary with minADE, minFDE, and per-mode metrics
    
    **Validates: Requirements 4.1, 4.2, 4.6, 4.8**
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
    # ADE = mean displacement over valid timesteps
    per_mode_ade = (displacement * valid_float).sum(dim=3) / valid_float.sum(dim=3).clamp_min(1.0)
    
    # Compute FDE per mode: [B, N, K]
    # FDE = displacement at final timestep
    final_valid = valid[..., -1]  # [B, N, 1]
    per_mode_fde = displacement[..., -1]  # [B, N, K]
    
    # Select best mode per agent (minimum ADE)
    best_mode_indices = torch.argmin(per_mode_ade, dim=2)  # [B, N]
    
    # Gather minADE and minFDE
    best_mode_indices_expanded = best_mode_indices.unsqueeze(2)  # [B, N, 1]
    min_ade_per_agent = torch.gather(per_mode_ade, dim=2, index=best_mode_indices_expanded).squeeze(2)  # [B, N]
    min_fde_per_agent = torch.gather(per_mode_fde, dim=2, index=best_mode_indices_expanded).squeeze(2)  # [B, N]
    
    # Apply agent mask and compute mean
    agent_mask_float = agent_mask.to(dtype=pred.dtype)
    num_valid_agents = agent_mask_float.sum().clamp_min(1.0)
    
    min_ade = (min_ade_per_agent * agent_mask_float).sum() / num_valid_agents
    min_fde = (min_fde_per_agent * agent_mask_float).sum() / num_valid_agents
    
    # Compute per-mode metrics for analysis
    per_mode_metrics = {}
    for mode_idx in range(num_modes):
        mode_ade = (per_mode_ade[:, :, mode_idx] * agent_mask_float).sum() / num_valid_agents
        mode_fde = (per_mode_fde[:, :, mode_idx] * agent_mask_float).sum() / num_valid_agents
        per_mode_metrics[f"mode_{mode_idx}_ADE"] = float(mode_ade.detach().cpu().item())
        per_mode_metrics[f"mode_{mode_idx}_FDE"] = float(mode_fde.detach().cpu().item())
    
    return {
        "minADE": float(min_ade.detach().cpu().item()),
        "minFDE": float(min_fde.detach().cpu().item()),
        **per_mode_metrics,
    }


def load_checkpoint(
    checkpoint_path: Path,
    model: MultiAgentTrajectoryPredictor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, list]:
    """Load checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
    
    Returns:
        Tuple of (start_epoch, best_val_metric, metrics_history)
    
    **Validates: Requirements 10.6, 10.7**
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Get training state
    start_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
    best_val_metric = checkpoint.get("best_val_minADE", checkpoint.get("val_loss", math.inf))
    metrics_history = checkpoint.get("metrics_history", [])
    
    print(f"Resuming from epoch {start_epoch}, best_val_metric={best_val_metric:.4f}")
    
    return start_epoch, best_val_metric, metrics_history


def run_epoch(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    log_every: int = 0,
    enable_multimodal: bool = False,
) -> tuple[float, dict]:
    """Run one epoch of training or validation.
    
    Args:
        model: The trajectory prediction model
        loader: DataLoader for the dataset
        device: Device to run on
        optimizer: Optimizer for training (None for validation)
        grad_clip: Gradient clipping max norm
        log_every: Log metrics every N batches (0 to disable)
        enable_multimodal: Whether multimodal prediction is enabled
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_metrics = {}
    total_batches = 0

    for batch_idx, raw_batch in enumerate(loader, start=1):
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
            metrics = {"ADE": ade, "FDE": fde}
        
        total_loss += float(loss.detach().cpu().item())
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        total_batches += 1

        if training and log_every > 0 and batch_idx % int(log_every) == 0:
            metric_str = " ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
            print(f"  batch={batch_idx}/{len(loader)} loss={float(loss.item()):.4f} {metric_str}")

    denom = max(1, total_batches)
    avg_metrics = {key: value / denom for key, value in total_metrics.items()}
    return total_loss / denom, avg_metrics


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    data_dirs = [Path(path).resolve() for path in args.data_dir]
    
    # Determine town name for output directory
    if args.town_filter and len(args.town_filter) == 1:
        town_name = args.town_filter[0]
    elif args.town_filter:
        town_name = "_".join(sorted(args.town_filter))
    else:
        town_name = "all_towns"
    
    # Create town-specific output directory
    base_out_dir = Path(args.out_dir).resolve()
    out_dir = base_out_dir / town_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets and filter by town
    loaded_datasets = [MultiAgentTrajectoryDataset(data_dir) for data_dir in data_dirs]
    sample_paths = [path for dataset in loaded_datasets for path in dataset.sample_paths]
    
    # Filter samples by town if specified
    if args.town_filter:
        print(f"Filtering samples for towns: {args.town_filter}")
        sample_paths = filter_samples_by_town(sample_paths, args.town_filter)
        print(f"Found {len(sample_paths)} samples matching town filter")
    
    if int(args.limit_samples) > 0:
        sample_paths = sample_paths[: int(args.limit_samples)]

    if not sample_paths:
        print("Error: No samples found matching the specified criteria")
        return 1

    train_paths, val_paths = split_sample_paths(
        sample_paths,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )
    if not val_paths:
        val_paths = train_paths[-1:]
        train_paths = train_paths[:-1] or val_paths

    dataset_root = data_dirs[0]
    train_dataset = MultiAgentTrajectoryDataset(dataset_root, sample_files=train_paths)
    val_dataset = MultiAgentTrajectoryDataset(dataset_root, sample_files=val_paths)
    first_sample = train_dataset[0]
    future_steps = int(first_sample["y"].shape[1])  # type: ignore[index]
    input_dim = int(first_sample["x"].shape[2])  # type: ignore[index]

    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multi_agent_trajectory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Create model configuration
    model_config = MultiAgentModelConfig(
        input_dim=input_dim,
        hidden_dim=max(16, int(args.hidden_dim)),
        graph_layers=max(0, int(args.graph_layers)),
        future_steps=future_steps,
        dropout=float(args.dropout),
        enable_gat=args.enable_gat,
        num_attention_heads=int(args.num_attention_heads),
        enable_multimodal=args.enable_multimodal,
        num_modes=int(args.num_modes),
        enable_adaptive_radius=args.enable_adaptive_radius,
    )
    
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

    # Save training configuration
    metadata = {
        "data_dirs": [str(path) for path in data_dirs],
        "town_filter": args.town_filter,
        "town_name": town_name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": str(device),
        "model_config": model_config.to_json(),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "args": vars(args),
    }
    (out_dir / "train_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Town: {town_name}")
    print(f"Dataset: train={len(train_dataset)} val={len(val_dataset)} data_dirs={len(data_dirs)}")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")
    print(f"Model: hidden={model_config.hidden_dim} graph_layers={model_config.graph_layers} future={future_steps}")
    print(f"  GAT: {model_config.enable_gat} (heads={model_config.num_attention_heads})")
    print(f"  Multimodal: {model_config.enable_multimodal} (modes={model_config.num_modes})")
    print(f"  Adaptive Radius: {model_config.enable_adaptive_radius}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    best_val = math.inf
    stale_epochs = 0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    
    # Track metrics history
    metrics_history = []
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            print(f"Error: Resume checkpoint not found at {resume_path}")
            return 1
        
        start_epoch, best_val, metrics_history = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, max(1, int(args.epochs)) + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
            enable_multimodal=args.enable_multimodal,
        )
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                enable_multimodal=args.enable_multimodal,
            )
        
        # Use minADE for early stopping if multimodal, else use loss
        if args.enable_multimodal:
            val_metric = val_metrics.get("minADE", val_loss)
        else:
            val_metric = val_loss
        
        scheduler.step(val_metric)

        # Build checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_minADE": best_val if args.enable_multimodal else math.inf,
            "model_config": model_config.to_json(),
            "train_config": metadata,
            "metrics_history": metrics_history,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }
        torch.save(checkpoint, last_path)

        # Track metrics history
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metrics_history.append(epoch_metrics)

        improved = val_metric < best_val
        if improved:
            best_val = val_metric
            stale_epochs = 0
            torch.save(checkpoint, best_path)
        else:
            stale_epochs += 1

        lr = optimizer.param_groups[0]["lr"]
        marker = " saved_best" if improved else ""
        
        # Format metrics for display
        train_metric_str = " ".join([f"train_{k}={v:.3f}" for k, v in train_metrics.items()])
        val_metric_str = " ".join([f"val_{k}={v:.3f}" for k, v in val_metrics.items()])
        
        print(
            f"epoch={epoch:03d} lr={lr:.2e} "
            f"train_loss={train_loss:.4f} {train_metric_str} "
            f"val_loss={val_loss:.4f} {val_metric_str}{marker}"
        )

        if stale_epochs >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch}.")
            break

    # Save metrics history
    (out_dir / "metrics_history.json").write_text(
        json.dumps(metrics_history, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[OK] Best checkpoint: {best_path}")
    print(f"[OK] Last checkpoint: {last_path}")
    print(f"[OK] Metrics history: {out_dir / 'metrics_history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
