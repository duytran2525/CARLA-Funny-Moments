from pathlib import Path

import torch
from ultralytics import YOLO


def train_model():
    root_dir = Path(__file__).resolve().parent.parent
    data_config_path = root_dir / "configs" / "data.yaml"

    if not data_config_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file cau hinh dataset cho YOLO: {data_config_path}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Bat dau huan luyen tren thiet bi: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO("yolo11s.pt")
    model.train(
        data=str(data_config_path),
        epochs=100,
        batch=8,
        imgsz=640,
        device=device,
        project=str(root_dir / "models" / "yolo"),
        name="retrain_results",
    )
    print("Huan luyen hoan tat!")


if __name__ == "__main__":
    train_model()
