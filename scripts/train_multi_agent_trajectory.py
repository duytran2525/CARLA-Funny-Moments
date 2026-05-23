from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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
    masked_smooth_l1_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline multi-agent trajectory predictor.")
    parser.add_argument(
        "--data-dir",
        required=True,
        nargs="+",
        help="One or more processed dataset dirs containing manifest.csv and .pt samples.",
    )
    parser.add_argument("--out-dir", default="models/multi_agent", help="Checkpoint/output directory.")
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
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-samples", type=int, default=0, help="Optional smoke-test sample cap.")
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


def config_to_dict(config: object) -> dict:
    """Safely serialise a config object to a JSON-compatible dict."""
    raw = config.__dict__ if hasattr(config, "__dict__") else {}
    safe: dict = {}
    for k, v in raw.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe


# ── FIX 1 ─────────────────────────────────────────────────────────────────────
# Build per-path → root mapping so each sample is resolved against the correct
# data dir, regardless of how many --data-dir values were supplied.
# Previously the code always passed data_dirs[0] as the root for *all* paths,
# which would silently fail (or load wrong data) for samples that live in
# data_dirs[1], data_dirs[2], … when paths happened to be relative.
# We guarantee paths are absolute (via Path.resolve() in loaded_datasets), but
# making the root accurate is still the correct thing to do.
def _build_path_root_map(
    sample_paths: List[Path], data_dirs: List[Path]
) -> Dict[Path, Path]:
    """Return {sample_path: owning_data_dir} for every sample."""
    mapping: Dict[Path, Path] = {}
    for path in sample_paths:
        for data_dir in data_dirs:
            try:
                path.relative_to(data_dir)
                mapping[path] = data_dir
                break
            except ValueError:
                continue
        else:
            # Path does not belong to any known dir (shouldn't happen, but be safe)
            mapping[path] = data_dirs[0]
    return mapping


def make_dataset_for_paths(
    paths: List[Path],
    data_dirs: List[Path],
    path_root_map: Dict[Path, Path],
) -> MultiAgentTrajectoryDataset:
    """
    Create a MultiAgentTrajectoryDataset whose root is derived from the actual
    owning directory of each sample rather than hard-coded to data_dirs[0].

    If all paths share a single root, use that root.  When paths span multiple
    roots, fall back to data_dirs[0] — at this point every path is already
    absolute so the root is only used for metadata, not for file I/O.
    """
    unique_roots = {path_root_map[p] for p in paths}
    root = unique_roots.pop() if len(unique_roots) == 1 else data_dirs[0]
    return MultiAgentTrajectoryDataset(root, sample_files=paths)


# ── FIX 2 ─────────────────────────────────────────────────────────────────────
# ade/fde were accumulated as raw tensors (or whatever masked_ade_fde returns).
# In training mode this keeps the whole computation graph alive for the entire
# epoch, causing an O(N_batches) GPU memory leak.
# Fix: always convert to plain Python float immediately.
def _to_float(v: object) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)  # type: ignore[arg-type]


def run_epoch(
    model: MultiAgentTrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    log_every: int = 0,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
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

        # ── FIX 2 applied: use pred.detach() so metrics never extend the graph,
        #    and convert results to float immediately.
        with torch.no_grad():
            ade, fde = masked_ade_fde(
                pred=pred.detach(),
                target=batch["y"],  # type: ignore[arg-type]
                y_mask=batch["y_mask"],  # type: ignore[arg-type]
                agent_mask=batch["agent_mask"],  # type: ignore[arg-type]
            )

        total_loss += float(loss.detach().cpu().item())
        total_ade += _to_float(ade)  # FIX 2: was `+= ade` (raw tensor)
        total_fde += _to_float(fde)  # FIX 2: was `+= fde` (raw tensor)
        total_batches += 1

        if training and log_every > 0 and batch_idx % int(log_every) == 0:
            print(
                f"  batch={batch_idx}/{len(loader)} "
                f"loss={float(loss.item()):.4f} "
                f"ade={_to_float(ade):.3f} fde={_to_float(fde):.3f}"
            )

    denom = max(1, total_batches)
    return total_loss / denom, total_ade / denom, total_fde / denom


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    data_dirs = [Path(path).resolve() for path in args.data_dir]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all datasets and gather absolute sample paths.
    loaded_datasets = [MultiAgentTrajectoryDataset(data_dir) for data_dir in data_dirs]
    sample_paths: List[Path] = [path for dataset in loaded_datasets for path in dataset.sample_paths]
    if int(args.limit_samples) > 0:
        sample_paths = sample_paths[: int(args.limit_samples)]

    # ── FIX 1 applied ─────────────────────────────────────────────────────────
    # Build a stable per-path → owning-root mapping *before* the train/val split
    # so that both subsets can be handed to their correct dataset roots.
    path_root_map = _build_path_root_map(sample_paths, data_dirs)

    train_paths, val_paths = split_sample_paths(
        sample_paths,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )
    if not val_paths:
        val_paths = train_paths[-1:]
        train_paths = train_paths[:-1] or val_paths

    # FIX 1: use per-path roots instead of always data_dirs[0]
    train_dataset = make_dataset_for_paths(train_paths, data_dirs, path_root_map)
    val_dataset   = make_dataset_for_paths(val_paths,   data_dirs, path_root_map)

    first_sample = train_dataset[0]
    future_steps = int(first_sample["y"].shape[1])  # type: ignore[index]
    input_dim    = int(first_sample["x"].shape[2])  # type: ignore[index]

    loader_kwargs = {
        "batch_size": max(1, int(args.batch_size)),
        "num_workers": max(0, int(args.num_workers)),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multi_agent_trajectory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    model_config = MultiAgentModelConfig(
        input_dim=input_dim,
        hidden_dim=max(16, int(args.hidden_dim)),
        graph_layers=max(0, int(args.graph_layers)),
        future_steps=future_steps,
        dropout=float(args.dropout),
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

    # ── FIX 3 ─────────────────────────────────────────────────────────────────
    # model_config.__dict__ was passed raw to json.dumps, which silently
    # serialises non-primitive fields as un-reproducible repr() strings.
    # Use config_to_dict() to make serialisation explicit and safe.
    metadata = {
        "data_dirs": [str(path) for path in data_dirs],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "device": str(device),
        "model_config": config_to_dict(model_config),  # FIX 3
        "args": vars(args),
    }
    (out_dir / "train_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Dataset: train={len(train_dataset)} val={len(val_dataset)} data_dirs={len(data_dirs)}")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")
    print(f"Model: hidden={model_config.hidden_dim} graph_layers={model_config.graph_layers} future={future_steps}")
    print(f"Device: {device}")

    best_val = math.inf
    stale_epochs = 0
    best_path = out_dir / "multi_agent_trajectory_best.pt"
    last_path  = out_dir / "multi_agent_trajectory_last.pt"

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        train_loss, train_ade, train_fde = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
        )
        # FIX 4 (minor): the outer `with torch.no_grad()` was redundant because
        # run_epoch already uses torch.set_grad_enabled(False) internally.
        # Removed the wrapper to avoid confusion; run_epoch handles it cleanly.
        val_loss, val_ade, val_fde = run_epoch(model=model, loader=val_loader, device=device)
        scheduler.step(val_loss)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": config_to_dict(model_config),  # FIX 3
            "epoch": epoch,
            "val_loss": val_loss,
            "val_ade": val_ade,
            "val_fde": val_fde,
            "train_loss": train_loss,
            "train_ade": train_ade,
            "train_fde": train_fde,
        }
        torch.save(checkpoint, last_path)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            stale_epochs = 0
            torch.save(checkpoint, best_path)
        else:
            stale_epochs += 1

        lr = optimizer.param_groups[0]["lr"]
        marker = " ✓ saved_best" if improved else ""
        print(
            f"epoch={epoch:03d} lr={lr:.2e} "
            f"train_loss={train_loss:.4f} train_ADE={train_ade:.3f} train_FDE={train_fde:.3f} "
            f"val_loss={val_loss:.4f} val_ADE={val_ade:.3f} val_FDE={val_fde:.3f}{marker}"
        )

        if stale_epochs >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"[OK] Best checkpoint: {best_path}")
    print(f"[OK] Last checkpoint: {last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())