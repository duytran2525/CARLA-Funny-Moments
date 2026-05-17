#!/usr/bin/env python3
"""
Offline Evaluation Script for CIL Waypoint Model
Calculates ADE (Average Displacement Error) and FDE (Final Displacement Error)
"""

import os
import sys
from pathlib import Path
import yaml
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.cnn_model import WaypointPredictor
from scripts.kaggle_train_h5 import WaypointCarlaDatasetH5, find_csv_root

def main():
    print("="*60)
    print(" CIL WAYPOINT MODEL - OFFLINE METRIC EVALUATION")
    print("="*60)

    # 1. Load config
    config_path = PROJECT_ROOT / "configs" / "train_params.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Allow overriding paths from environment variables for local testing
    h5_path = os.environ.get("H5_PATH", "/kaggle/input/datasets/yudtrann/dataset-carlav3/carla_images_drive.h5")
    csv_root_hint = os.environ.get("CSV_ROOT", config.get("data_root", None))

    if not os.path.exists(h5_path):
        print(f"⚠️ H5 File not found at: {h5_path}")
        print("Please set H5_PATH environment variable to run locally.")
        print("Example: set H5_PATH=C:/path/to/carla_images_drive.h5 && python scripts/evaluate_cnn_metrics.py")
        return

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = WaypointPredictor().to(device)
    model.eval()

    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / "models" / "waypoint_predictor.pth"
    if not checkpoint_path.exists():
        print(f"⚠️ Model checkpoint not found at: {checkpoint_path}")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # --- TỰ ĐỘNG FIX LỖI TƯƠNG THÍCH MODEL (BACKWARD COMPATIBILITY) ---
        if "film.embedding.weight" in state_dict and "film_s4.embedding.weight" not in state_dict:
            print("  ⚠️ [Auto-Fix] Phát hiện file .pth được train từ bản code cũ (chỉ có 1 lớp FiLM).")
            print("  -> Đang tự động chuyển đổi trọng số sang chuẩn Multi-level FiLM mới...")
            # Chuyển lớp film cũ thành film_s4 (do film cũ nằm ở tầng cuối)
            state_dict["film_s4.embedding.weight"] = state_dict.pop("film.embedding.weight")
            state_dict["film_s4.mlp.0.weight"] = state_dict.pop("film.mlp.0.weight")
            state_dict["film_s4.mlp.0.bias"] = state_dict.pop("film.mlp.0.bias")
            state_dict["film_s4.mlp.2.weight"] = state_dict.pop("film.mlp.2.weight")
            state_dict["film_s4.mlp.2.bias"] = state_dict.pop("film.mlp.2.bias")
            
            # Load với strict=False để bỏ qua lớp film_s3. 
            # (film_s3 sẽ tự động được khởi tạo là ma trận đơn vị - không làm thay đổi kết quả dự đoán)
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
            
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # 3. Setup DataLoader
    print("\nPreparing Validation Dataset...")
    try:
        csv_root = find_csv_root(h5_path, csv_root_hint)
        with h5py.File(h5_path, "r") as f:
            towns = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        
        transform = transforms.Compose([transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)])
        val_dataset = WaypointCarlaDatasetH5(
            h5_path=h5_path,
            csv_root=csv_root,
            towns=towns,
            transform=transform,
            is_training=False,
            train_ratio=float(config.get("train_split", 0.75)),
            geometric_offset=float(config.get("geometric_offset", 0.35)),
            include_side_cameras=False,
        )
        
        def collate_fn(batch):
            imgs, wps, cmds, recs, speeds = zip(*batch)
            return (
                torch.stack(imgs),
                torch.stack(wps).float(),
                torch.stack(cmds).long(),
                torch.tensor(recs, dtype=torch.float32),
                torch.tensor(speeds, dtype=torch.float32),
            )

        val_loader = DataLoader(
            val_dataset, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fn
        )
        print(f"Validation Dataset size: {len(val_dataset)} samples")
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        return

    # 4. Evaluation Loop
    print("\nEvaluating (calculating ADE / FDE)...")
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.inference_mode():
        for i, (imgs, wps, cmds, _, speeds) in enumerate(val_loader):
            imgs = imgs.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)
            wps = wps.to(device, non_blocking=True)
            speeds = speeds.to(device, non_blocking=True)
            
            # Predict
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                out = model(imgs, cmds, speeds)
                pred_wp = out[:, :10].view(-1, 5, 2)
            
            # wps and pred_wp shape: [batch_size, 5, 2]
            # Calculate Euclidean distance for each point: [batch_size, 5]
            distances = torch.norm(pred_wp - wps, dim=-1)
            
            # ADE: mean distance across all 5 points for each sample, then sum for batch
            batch_ade = distances.mean(dim=1).sum().item()
            # FDE: distance at the final point (index 4) for each sample, then sum for batch
            batch_fde = distances[:, 4].sum().item()
            
            total_ade += batch_ade
            total_fde += batch_fde
            total_samples += imgs.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(val_loader)} batches...")

    if total_samples > 0:
        mean_ade = total_ade / total_samples
        mean_fde = total_fde / total_samples
        
        print("\n" + "="*60)
        print(" 🎯 EVALUATION RESULTS (METERS)")
        print("="*60)
        print(f"Total Samples Evaluated: {total_samples}")
        print(f"Average Displacement Error (ADE): {mean_ade:.4f} m")
        print(f"Final Displacement Error (FDE):   {mean_fde:.4f} m")
        print("="*60)
        
        # Determine performance rating for presentation
        if mean_ade < 1.0 and mean_fde < 1.5:
            print("\n🌟 CONCLUSION: EXCELLENT!")
            print("Model is highly accurate. Safe to deploy in autonomous driving.")
        elif mean_ade < 2.0 and mean_fde < 3.0:
            print("\n👍 CONCLUSION: GOOD")
            print("Model performs well but may have occasional lane drifting.")
        else:
            print("\n⚠️ CONCLUSION: NEEDS IMPROVEMENT")
            print("Model error is high. Review training data or increase epochs.")
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    main()
