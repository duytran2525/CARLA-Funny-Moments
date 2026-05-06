from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def _resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root_dir / path).resolve()


def _validate_data_config(data_config_path: Path) -> None:
    if not data_config_path.exists():
        raise FileNotFoundError(f"YOLO dataset config not found: {data_config_path}")

    with data_config_path.open("r", encoding="utf-8") as fp:
        data_cfg = yaml.safe_load(fp) or {}

    dataset_root = _resolve_path(data_config_path.parent, str(data_cfg.get("path", ".")))
    for split_name in ("train", "val"):
        split_value = data_cfg.get(split_name)
        if not split_value:
            raise RuntimeError(f"Missing '{split_name}' entry in {data_config_path}")
        split_path = _resolve_path(dataset_root, str(split_value))
        if not split_path.exists():
            raise FileNotFoundError(
                f"YOLO {split_name} path does not exist: {split_path}. "
                "Update configs/data.yaml to match your dataset layout."
            )

    names = data_cfg.get("names")
    if not isinstance(names, (dict, list)) or len(names) == 0:
        raise RuntimeError(f"Invalid or empty 'names' section in {data_config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector on the configured dataset.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo11s.pt", help="Base Ultralytics model checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="models/yolo", help="Output directory for Ultralytics runs.")
    parser.add_argument("--name", default="retrain_results", help="Run name under --project.")
    return parser.parse_args()


def train_model(args: argparse.Namespace) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_config_path = _resolve_path(root_dir, args.data)
    project_path = _resolve_path(root_dir, args.project)
    model_path = _resolve_path(root_dir, args.model)

    _validate_data_config(data_config_path)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training YOLO on: {'GPU' if device == 0 else 'CPU'}")
    print(f"Dataset config: {data_config_path}")

    model_source = str(model_path) if model_path.exists() else str(args.model)
    model = YOLO(model_source)
    model.train(
        data=str(data_config_path),
        epochs=int(args.epochs),
        batch=int(args.batch),
        imgsz=int(args.imgsz),
        device=device,
        project=str(project_path),
        name=str(args.name),
    )
    print("YOLO training completed.")


if __name__ == "__main__":
    train_model(parse_args())
