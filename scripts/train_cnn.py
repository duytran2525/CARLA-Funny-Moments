from __future__ import annotations

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import random
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

# FIX KAGGLE OOM LOG: Tắt sạch các warning tự động để tránh treo Jupyter
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core_perception.cnn_model import WaypointPredictor

try:
    from core_perception.dataset import WaypointCarlaDataset
except ImportError:  # pragma: no cover
    WaypointCarlaDataset = None


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA unavailable.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Detected GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")


def collate_waypoint_batch(batch):
    images, waypoints, commands, recovery_flags, speeds = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(waypoints).float(),
        torch.stack(commands).long(),
        torch.stack(recovery_flags).float()
        if torch.is_tensor(recovery_flags[0])
        else torch.tensor(recovery_flags, dtype=torch.float32),
        torch.stack(speeds).float(),
    )


def _get_recovery_flags(dataset) -> Optional[Iterable[int]]:
    if hasattr(dataset, "get_recovery_flags"):
        return dataset.get_recovery_flags()
    if hasattr(dataset, "recovery_flags"):
        return dataset.recovery_flags
    return None


def _build_recovery_sampler(flags: Optional[Iterable[int]], recovery_weight: float) -> Optional[WeightedRandomSampler]:
    if flags is None:
        return None
    weights = [recovery_weight if int(flag) == 1 else 1.0 for flag in flags]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _load_training_dataframe(data_root: Path, csv_path: Optional[Path]) -> tuple[pd.DataFrame, str]:
    resolved_root = data_root.resolve()

    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        if "dataset_subdir" not in df.columns:
            try:
                relative_parent = csv_path.resolve().parent.relative_to(resolved_root)
            except ValueError:
                relative_parent = None
            if relative_parent is not None:
                rel_text = relative_parent.as_posix()
                if rel_text not in {"", "."}:
                    df = df.copy()
                    df["dataset_subdir"] = rel_text
        return df, str(csv_path)

    town_csvs = sorted(path for path in resolved_root.glob("*/driving_log.csv") if path.is_file())
    if town_csvs:
        frames = []
        for town_csv in town_csvs:
            town_df = pd.read_csv(town_csv)
            rel_text = town_csv.parent.relative_to(resolved_root).as_posix()
            town_df = town_df.copy()
            town_df["dataset_subdir"] = rel_text
            frames.append(town_df)
        merged = pd.concat(frames, ignore_index=True)
        return merged, f"{resolved_root} (merged {len(town_csvs)} sub-datasets)"

    missing_path = csv_path if csv_path is not None else (resolved_root / "driving_log.csv")
    raise FileNotFoundError(
        "Cannot find training CSV. Checked "
        f"'{missing_path}' and subdirectories matching '{resolved_root}/*/driving_log.csv'."
    )


def _stratified_split(df: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    working_df = df.copy()
    command_series = pd.to_numeric(working_df.get("command", 0), errors="coerce").fillna(-1).astype(int).astype(str)
    if "dataset_subdir" in working_df.columns:
        subdir_series = working_df["dataset_subdir"].fillna("root").astype(str)
    else:
        subdir_series = pd.Series(["root"] * len(working_df), index=working_df.index, dtype="object")
    working_df["_split_key"] = subdir_series + "|" + command_series

    train_parts = []
    val_parts = []
    for _, group in working_df.groupby("_split_key", sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed)
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue

        split_idx = int(round(float(train_ratio) * len(shuffled)))
        split_idx = min(max(1, split_idx), len(shuffled) - 1)
        train_parts.append(shuffled.iloc[:split_idx])
        val_parts.append(shuffled.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else working_df.iloc[0:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else working_df.iloc[0:0].copy()

    if val_df.empty and len(train_df) > 1:
        val_df = train_df.tail(1).copy()
        train_df = train_df.iloc[:-1].copy()

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = train_df.drop(columns=["_split_key"], errors="ignore")
    val_df = val_df.drop(columns=["_split_key"], errors="ignore")
    return train_df, val_df


def _print_split_summary(label: str, df: pd.DataFrame) -> None:
    print(f"{label}: {len(df)} rows")
    if "command" in df.columns:
        counts = pd.to_numeric(df["command"], errors="coerce").fillna(-1).astype(int).value_counts().sort_index().to_dict()
        print(f"  command_counts={counts}")
    if "dataset_subdir" in df.columns:
        counts = df["dataset_subdir"].fillna("root").astype(str).value_counts().sort_index().to_dict()
        print(f"  dataset_counts={counts}")


def _maybe_cap_rows(df: pd.DataFrame, limit: int, *, seed: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df
    return df.sample(n=int(limit), random_state=seed).reset_index(drop=True)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir / "configs" / "train_params.yaml")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    print(f"Seed: {seed}")
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_root = Path(config.get("data_root", root_dir / "data"))
    if not data_root.is_absolute():
        data_root = (root_dir / data_root).resolve()

    csv_path_cfg = config.get("csv_path") or config.get("data_csv")
    csv_path = Path(csv_path_cfg) if csv_path_cfg else data_root / "driving_log.csv"
    if not csv_path.is_absolute():
        csv_path = (root_dir / csv_path).resolve()

    model_save_path = Path(config.get("model_save_path", root_dir / "models" / "waypoint_predictor.pth"))
    if not model_save_path.is_absolute():
        model_save_path = (root_dir / model_save_path).resolve()
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["batch_size"])

    print("\n" + "=" * 70)
    print("GPU SETUP")
    print("=" * 70)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print_gpu_info()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    requested_multi_gpu = config.get("use_multi_gpu")
    if num_gpus > 0:
        primary_device = torch.device("cuda:0")
        if requested_multi_gpu is None:
            use_multi_gpu = num_gpus > 1 and batch_size >= 48
        else:
            use_multi_gpu = num_gpus > 1 and _to_bool(requested_multi_gpu, True)
        print(f"Using device: {num_gpus} GPU(s)")
        if use_multi_gpu:
            print("Mode: DataParallel")
        elif num_gpus > 1:
            print("Mode: Single GPU (DataParallel disabled by config/heuristic)")
    else:
        primary_device = torch.device("cpu")
        use_multi_gpu = False
        print("Using CPU")

    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)

    transform = transforms.Compose([transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)])

    df, df_source = _load_training_dataframe(data_root, csv_path)
    print(f"Loaded metadata from: {df_source}")
    print(f"Total rows before split: {len(df)}")

    default_work_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else str(data_root)
    work_dir = Path(config.get("work_dir", default_work_dir))
    if not work_dir.is_absolute():
        work_dir = (root_dir / work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = work_dir / "train_split_log.csv"
    val_csv_path = work_dir / "val_split_log.csv"
    if train_csv_path.exists():
        train_csv_path.unlink()
    if val_csv_path.exists():
        val_csv_path.unlink()

    train_split_ratio = float(config.get("train_split", 0.75))
    train_df, val_df = _stratified_split(df, train_split_ratio, seed)
    train_df = _maybe_cap_rows(train_df, int(config.get("max_train_rows", 0) or 0), seed=seed)
    val_df = _maybe_cap_rows(val_df, int(config.get("max_val_rows", 0) or 0), seed=seed)

    _print_split_summary("Train split", train_df)
    _print_split_summary("Val split", val_df)

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    if WaypointCarlaDataset is None:
        raise RuntimeError(
            "WaypointCarlaDataset is unavailable in core_perception.dataset. "
            "Fix the dataset module before training."
        )

    include_side_cameras_train = _to_bool(config.get("include_side_cameras_train"), True)
    train_dataset = WaypointCarlaDataset(
        csv_file=train_csv_path,
        root_dir=data_root,
        transform=transform,
        is_training=True,
        geometric_offset=float(config.get("geometric_offset", 0.35)),
        include_side_cameras=include_side_cameras_train,
    )
    val_dataset = WaypointCarlaDataset(
        csv_file=val_csv_path,
        root_dir=data_root,
        transform=transform,
        is_training=False,
        geometric_offset=float(config.get("geometric_offset", 0.35)),
        include_side_cameras=False,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after path resolution/filtering. Check CSV schema and image layout.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after split/path resolution. Adjust data split or dataset root.")

    num_workers_cfg = config.get("num_workers")
    if num_workers_cfg is None:
        num_workers = min(4, max(1, os.cpu_count() or 2))
    else:
        num_workers = max(0, int(num_workers_cfg))
    pin_mem = torch.cuda.is_available()
    prefetch_factor = max(2, int(config.get("prefetch_factor", 2)))
    log_every_batches = max(0, int(config.get("log_every_batches", 0)))
    val_log_every_batches = max(0, int(config.get("val_log_every_batches", 0)))

    recovery_weight = float(config.get("recovery_weight", 2.0))
    sampler = _build_recovery_sampler(_get_recovery_flags(train_dataset), recovery_weight)
    generator = torch.Generator().manual_seed(seed)

    train_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "collate_fn": collate_waypoint_batch,
        "persistent_workers": num_workers > 0,
        "generator": generator,
    }
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "collate_fn": collate_waypoint_batch,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("DataLoader produced zero iterations. Reduce batch size or inspect dataset sizes.")

    print(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    print(f"  train_steps={len(train_loader)} val_steps={len(val_loader)}")

    print("\n" + "=" * 70)
    print("MODEL SETUP")
    print("=" * 70)

    model = WaypointPredictor().to(primary_device)
    if primary_device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if use_multi_gpu:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"Wrapped model with DataParallel ({num_gpus} GPUs)")
    else:
        print("Single-device mode")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params} ({trainable_params} trainable)")

    print("\n" + "=" * 70)
    print("TRAINING SETUP")
    print("=" * 70)

    huber_loss = nn.SmoothL1Loss(reduction="mean")
    base_lr = float(config["learning_rate"])
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_patience = int(config.get("lr_patience", 3))
    lr_factor = float(config.get("lr_factor", 0.5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience,
    )
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)
    lambda_wp = float(config.get("loss_lambda_wp", 1.0))
    lambda_gnll = float(config.get("loss_lambda_gnll", 0.05))
    lambda_smoothness = float(config.get("loss_lambda_smoothness", 0.1))

    # GNLL stability params
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    sigma_min = float(config.get("sigma_min", 0.01))
    sigma_max = float(config.get("sigma_max", 10.0))
    warmup_epochs = int(config.get("warmup_epochs", 1))
    aggressive_memory_cleanup = _to_bool(config.get("aggressive_memory_cleanup"), False)

    print(f"Optimizer: Adam (lr={base_lr})")
    lambda_speed = float(config.get("loss_lambda_speed", 0.5))
    print(f"Loss weights: waypoint={lambda_wp} gnll={lambda_gnll} smoothness={lambda_smoothness} speed={lambda_speed}")
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")
    print(f"Grad clip: {grad_clip_norm}, sigma range: [{sigma_min}, {sigma_max}]")
    print(f"Warmup epochs: {warmup_epochs}")

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")

    epochs = int(config["epochs"])
    early_stopping_patience = int(config.get("early_stopping_patience", 8))
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Linear LR warmup
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        model.train()
        running_loss = 0.0
        nan_batches = 0

        for i, (images, waypoints, commands, recovery_flags, speeds) in enumerate(train_loader):
            del recovery_flags
            images = images.to(primary_device, non_blocking=True)
            if primary_device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            commands = commands.to(primary_device, non_blocking=True)
            waypoints = waypoints.to(primary_device, non_blocking=True)
            speeds = speeds.to(primary_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                outputs = model(images, commands, speeds)
                pred_wp = outputs[:, :10].view(-1, 5, 2)
                # Model outputs sigma; GNLL uses variance for numerical consistency.
                pred_sigma = outputs[:, 10:15].view(-1, 5, 1).expand(-1, 5, 2).clamp(sigma_min, sigma_max)
                pred_var = pred_sigma.pow(2).clamp(sigma_min**2, sigma_max**2)
                target_wp = waypoints.view(-1, 5, 2)

                loss_wp = huber_loss(pred_wp, target_wp)
                loss_gnll = 0.5 * ((target_wp - pred_wp) ** 2 / pred_var + torch.log(pred_var))

                vel = pred_wp[:, 1:] - pred_wp[:, :-1]
                accel = vel[:, 1:] - vel[:, :-1]
                smoothness_loss = accel.pow(2).mean()

                # Speed prediction loss (CILRS-style auxiliary regularizer)
                pred_speed = outputs[:, 15]
                loss_speed = torch.nn.functional.mse_loss(pred_speed, speeds)

                loss = (lambda_wp * loss_wp
                        + lambda_gnll * loss_gnll.mean()
                        + lambda_smoothness * smoothness_loss
                        + lambda_speed * loss_speed)

            # Skip NaN/Inf batches to protect optimizer state
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            # Unscale before clipping (required for correct AMP gradient clipping)
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            if log_every_batches > 0 and (i == 0 or (i + 1) % log_every_batches == 0):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Train] Epoch {epoch + 1} - Batch {i + 1}/{len(train_loader)} "
                    f"| loss={loss.item():.4f} | lr={current_lr:.2e}"
                )

        n_valid = max(1, len(train_loader) - nan_batches)
        train_loss = running_loss / n_valid
        if nan_batches > 0:
            print(f"  Skipped {nan_batches} NaN/Inf batches")

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0

        with torch.no_grad():
            for i, (images, waypoints, commands, recovery_flags, speeds) in enumerate(val_loader):
                del recovery_flags
                images = images.to(primary_device, non_blocking=True)
                if primary_device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                commands = commands.to(primary_device, non_blocking=True)
                waypoints = waypoints.to(primary_device, non_blocking=True)
                speeds = speeds.to(primary_device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    outputs = model(images, commands, speeds)
                    pred_wp = outputs[:, :10].view(-1, 5, 2)
                    pred_sigma = outputs[:, 10:15].view(-1, 5, 1).expand(-1, 5, 2).clamp(sigma_min, sigma_max)
                    pred_var = pred_sigma.pow(2).clamp(sigma_min**2, sigma_max**2)
                    target_wp = waypoints.view(-1, 5, 2)

                    loss_wp = huber_loss(pred_wp, target_wp)
                    loss_gnll = 0.5 * ((target_wp - pred_wp) ** 2 / pred_var + torch.log(pred_var))

                    vel = pred_wp[:, 1:] - pred_wp[:, :-1]
                    accel = vel[:, 1:] - vel[:, :-1]
                    smoothness_loss = accel.pow(2).mean()

                    # Speed prediction loss (CILRS-style auxiliary regularizer)
                    pred_speed = outputs[:, 15]
                    loss_speed = torch.nn.functional.mse_loss(pred_speed, speeds)

                    loss = (lambda_wp * loss_wp
                            + lambda_gnll * loss_gnll.mean()
                            + lambda_smoothness * smoothness_loss
                            + lambda_speed * loss_speed)

                bs = images.size(0)
                val_loss += float(loss.item()) * bs
                val_mae += float((pred_wp - target_wp).abs().mean().item()) * bs
                val_samples += bs
                if val_log_every_batches > 0 and (i == 0 or (i + 1) % val_log_every_batches == 0):
                    print(f"  [Val] Epoch {epoch + 1} - Batch {i + 1}/{len(val_loader)}")

        val_loss = val_loss / max(1, val_samples)
        val_mae = val_mae / max(1, val_samples)

        if torch.cuda.is_available() and num_gpus > 0:
            mem_allocated = torch.cuda.memory_allocated(primary_device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(primary_device) / 1024**3
            mem_str = f"| Mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB"
        else:
            mem_str = ""

        print(
            f"Epoch [{epoch + 1:2d}/{epochs}] | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} (MAE={val_mae:.4f}) {mem_str}"
        )

        # Only step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
            # Save with metadata (backward compatible via unwrap_state_dict)
            torch.save(
                {"model_state_dict": model_state, "epoch": epoch + 1,
                 "val_loss": val_loss, "val_mae": val_mae},
                model_save_path,
            )
            print(f"  Saved best -> {model_save_path} (val={val_loss:.4f}, MAE={val_mae:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping after epoch {epoch + 1}")
                break

        if aggressive_memory_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {model_save_path}")


if __name__ == "__main__":
    main()
