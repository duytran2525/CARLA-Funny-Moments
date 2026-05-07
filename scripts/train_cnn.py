from __future__ import annotations

import os
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
    images, waypoints, commands, recovery_flags = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(waypoints).float(),
        torch.stack(commands).long(),
        torch.stack(recovery_flags).float()
        if torch.is_tensor(recovery_flags[0])
        else torch.tensor(recovery_flags, dtype=torch.float32),
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


def main():
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir / "configs" / "train_params.yaml")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    print(f"Seed: {seed}")

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

    print("\n" + "=" * 70)
    print("GPU SETUP")
    print("=" * 70)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print_gpu_info()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    if num_gpus > 0:
        primary_device = torch.device("cuda:0")
        use_multi_gpu = num_gpus > 1
        print(f"Using device: {num_gpus} GPU(s)")
        if use_multi_gpu:
            print("Mode: DataParallel")
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

    _print_split_summary("Train split", train_df)
    _print_split_summary("Val split", val_df)

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    if WaypointCarlaDataset is None:
        raise RuntimeError(
            "WaypointCarlaDataset is unavailable in core_perception.dataset. "
            "Fix the dataset module before training."
        )

    train_dataset = WaypointCarlaDataset(
        csv_file=train_csv_path,
        root_dir=data_root,
        transform=transform,
        is_training=True,
        geometric_offset=float(config.get("geometric_offset", 0.35)),
    )
    val_dataset = WaypointCarlaDataset(
        csv_file=val_csv_path,
        root_dir=data_root,
        transform=transform,
        is_training=False,
        geometric_offset=float(config.get("geometric_offset", 0.35)),
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after path resolution/filtering. Check CSV schema and image layout.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after split/path resolution. Adjust data split or dataset root.")

    # FIX KAGGLE SYSTEM RAM OOM: Ép chặt num_workers = 0 và pin_memory = False
    num_workers = 0
    pin_mem = False
    batch_size = int(config["batch_size"])

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
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["prefetch_factor"] = 2

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
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    lr_patience = int(config.get("lr_patience", 3))
    lr_factor = float(config.get("lr_factor", 0.7))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)
    lambda_wp = float(config.get("loss_lambda_wp", 1.0))
    lambda_gnll = float(config.get("loss_lambda_gnll", 0.1))

    print(f"Optimizer: Adam (lr={float(config['learning_rate'])})")
    print(f"Loss weights: waypoint={lambda_wp} gnll={lambda_gnll}")
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")

    epochs = int(config["epochs"])
    early_stopping_patience = int(config.get("early_stopping_patience", 10))
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, waypoints, commands, recovery_flags) in enumerate(train_loader):
            del recovery_flags
            images = images.to(primary_device, non_blocking=True)
            commands = commands.to(primary_device, non_blocking=True)
            waypoints = waypoints.to(primary_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                outputs = model(images, commands)
                pred_wp = outputs[:, :10].view(-1, 5, 2)
                pred_sigma = outputs[:, 10:].view(-1, 5, 1).expand(-1, 5, 2).clamp_min(1e-4)
                target_wp = waypoints.view(-1, 5, 2)

                loss_wp = huber_loss(pred_wp, target_wp)
                loss_gnll = 0.5 * ((target_wp - pred_wp) ** 2 / pred_sigma + torch.log(pred_sigma))
                loss = lambda_wp * loss_wp + lambda_gnll * loss_gnll.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

            # FIX KAGGLE INTRA-EPOCH OOM: Dọn rác
            del images, waypoints, commands, outputs, pred_wp, pred_sigma, target_wp, loss_wp, loss_gnll, loss
            if i % 200 == 0:
                import gc
                gc.collect()
                # Giữ im lặng để tránh IOStream.flush timed out trên Kaggle

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i, (images, waypoints, commands, recovery_flags) in enumerate(val_loader):
                del recovery_flags
                images = images.to(primary_device, non_blocking=True)
                commands = commands.to(primary_device, non_blocking=True)
                waypoints = waypoints.to(primary_device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                    outputs = model(images, commands)
                    pred_wp = outputs[:, :10].view(-1, 5, 2)
                    pred_sigma = outputs[:, 10:].view(-1, 5, 1).expand(-1, 5, 2).clamp_min(1e-4)
                    target_wp = waypoints.view(-1, 5, 2)

                    loss_wp = huber_loss(pred_wp, target_wp)
                    loss_gnll = 0.5 * ((target_wp - pred_wp) ** 2 / pred_sigma + torch.log(pred_sigma))
                    loss = lambda_wp * loss_wp + lambda_gnll * loss_gnll.mean()

                val_loss += float(loss.item())

                # FIX KAGGLE VALIDATION OOM: Dọn rác
                del images, waypoints, commands, outputs, pred_wp, pred_sigma, target_wp, loss_wp, loss_gnll, loss
                if i % 200 == 0:
                    import gc
                    gc.collect()

        val_loss = val_loss / len(val_loader)

        if torch.cuda.is_available() and num_gpus > 0:
            mem_allocated = torch.cuda.memory_allocated(primary_device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(primary_device) / 1024**3
            mem_str = f"| Mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB"
        else:
            mem_str = ""

        print(
            f"Epoch [{epoch + 1:2d}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} {mem_str}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
            torch.save(model_state, model_save_path)
            print(f"  Saved best model -> {model_save_path} (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping after epoch {epoch + 1}")
                break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {model_save_path}")


if __name__ == "__main__":
    main()
