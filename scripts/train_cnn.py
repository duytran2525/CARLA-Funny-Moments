from __future__ import annotations

import os
import sys
from typing import Iterable, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core_perception.cnn_model import WaypointPredictor

try:
    from core_perception.dataset import WaypointCarlaDataset
except ImportError:  # pragma: no cover
    WaypointCarlaDataset = None

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def print_gpu_info():
    """In thông tin GPU details"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA không available!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Số GPU detect: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"      - Compute Capability: {props.major}.{props.minor}")

def collate_waypoint_batch(batch):
    """
    Collate batch cho waypoint dataset.

    __getitem__ kỳ vọng trả về: (image, waypoints, command, recovery_flag)
    """
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

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(ROOT_DIR, 'configs', 'train_params.yaml'))
    data_root = config.get('data_root', os.path.join(ROOT_DIR, 'data'))
    if not os.path.isabs(data_root):
        data_root = os.path.join(ROOT_DIR, str(data_root))

    csv_path = config.get('csv_path') or config.get('data_csv')
    if csv_path is None:
        csv_path = os.path.join(data_root, 'driving_log.csv')
    elif not os.path.isabs(csv_path):
        csv_path = os.path.join(ROOT_DIR, str(csv_path))

    DATA_DIR = data_root
    CSV_PATH = csv_path
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'waypoint_predictor.pth')
    
    # ═══════════════════════════════════════════════════════════
    # PHẦN 1: GPU SETUP (OPTIMIZED CHO MULTI-GPU)
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🚀 PHẦN 1: GPU SETUP")
    print("="*70)
    
    # Detect GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print_gpu_info()
    
    # Chọn device
    if num_gpus > 0:
        primary_device = torch.device("cuda:0")  # Use GPU 0 as primary
        use_multi_gpu = num_gpus > 1
        print(f"\n✅ Sử dụng GPU: {num_gpus} GPU(s)")
        if use_multi_gpu:
            print(f"   Mode: DataParallel (sử dụng cả {num_gpus} GPU)")
    else:
        primary_device = torch.device("cpu")
        use_multi_gpu = False
        print("\n⚠️ CUDA không available, dùng CPU (SLOW!)")
    
    # ═══════════════════════════════════════════════════════════
    # PHẦN 2: DATA LOADING (OPTIMIZED NUM_WORKERS)
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("📊 PHẦN 2: DATA LOADING")
    print("="*70)

    # Cấu hình Transform: Vì dataset.py tự chuyển thành Tensor [9, H, W] rồi
    # Ta chỉ cần tạo bộ Normalize cho 9 channels.
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)
    ])

    print("Đang phân chia tập Train và Validation từ file CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Xóa file CSV tạm cũ để tránh load lại steering đã bị normalize
    default_work_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else DATA_DIR
    work_dir = config.get("work_dir", default_work_dir)
    if not os.path.isabs(work_dir):
        work_dir = os.path.join(ROOT_DIR, str(work_dir))
    os.makedirs(work_dir, exist_ok=True)

    train_csv_path = os.path.join(work_dir, 'train_split_log.csv')
    val_csv_path = os.path.join(work_dir, 'val_split_log.csv')
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
        print(f"Xóa file cache cũ: {train_csv_path}")
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)
        print(f"Xóa file cache cũ: {val_csv_path}")
    
    # Tính toán chỉ số chia (vd: 75% train, 25% val)
    train_split_ratio = config.get('train_split', 0.75)
    split_idx = int(train_split_ratio * len(df))
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    
    # Lưu ra 2 file CSV tạm thời
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    # Bóp chết Dataframe để cứu 2-3GB RAM cực kỳ quan trọng
    import gc
    del df, train_df, val_df
    gc.collect()

    if WaypointCarlaDataset is None:
        raise RuntimeError(
            "WaypointCarlaDataset chưa có trong core_perception.dataset. "
            "Hãy triển khai dataset waypoint trước khi train."
        )

    train_dataset = WaypointCarlaDataset(
        csv_file=train_csv_path,
        root_dir=DATA_DIR,
        transform=transform,
        is_training=True,
        geometric_offset=float(config.get("geometric_offset", 0.35)),
    )

    val_dataset = WaypointCarlaDataset(
        csv_file=val_csv_path,
        root_dir=DATA_DIR,
        transform=transform,
        is_training=False,
        geometric_offset=float(config.get("geometric_offset", 0.35)),
    )
    
    # ⭐ FIX KAGGLE OOM CUỐI CÙNG:
    num_workers = 0
    pin_mem = False  # BẮT BUỘC FALSE: PyTorch Pin_Memory làm rò rỉ RAM trên luồng chính
    
    batch_size = config['batch_size']
    
    recovery_weight = float(config.get("recovery_weight", 2.0))
    sampler = _build_recovery_sampler(_get_recovery_flags(train_dataset), recovery_weight)

    # Đưa vào DataLoader với tối ưu
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_waypoint_batch,
        persistent_workers=(num_workers > 0),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_waypoint_batch,
        persistent_workers=(num_workers > 0),
    )
    
    print(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    print(f"  Iterations per epoch (train): {len(train_loader)}")
    print(f"  Iterations per epoch (val): {len(val_loader)}")
    
    # ═══════════════════════════════════════════════════════════
    # PHẦN 3: MODEL SETUP (MULTI-GPU SUPPORT)
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🧠 PHẦN 3: MODEL SETUP")
    print("="*70)
    
    # Tạo model trên primary device
    model = WaypointPredictor().to(primary_device)
    
    # ⭐ WRAP với DataParallel nếu có multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"✅ Model wrapped với DataParallel ({num_gpus} GPUs)")
    else:
        print(f"ℹ️ Single GPU mode")
    
    # In model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params} ({trainable_params} trainable)")
    
    # ═══════════════════════════════════════════════════════════
    # PHẦN 4: TRAINING SETUP
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("⚙️ PHẦN 4: TRAINING SETUP")
    print("="*70)
    
    huber_loss = nn.SmoothL1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    
    # Learning rate scheduler
    lr_patience = config.get('lr_patience', 3)
    lr_factor = config.get('lr_factor', 0.7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )
    
    # Mixed-precision scaler
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda' if use_amp else 'cpu', enabled=use_amp)
    
    print(f"Optimizer: Adam (lr={float(config['learning_rate'])})")
    print("Loss function: Huber + GNLL")
    print(f"LR Scheduler: ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience})")
    print(f"Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # ═══════════════════════════════════════════════════════════
    # PHẦN 5: TRAINING LOOP (MEMORY TRACKING)
    # ═══════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🔥 PHẦN 5: TRAINING")
    print("="*70 + "\n")
    
    epochs = config['epochs']
    early_stopping_patience = config.get('early_stopping_patience', 10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # ──────────────────────────────────
        # Training phase
        # ──────────────────────────────────
        model.train()
        running_loss = 0.0
        
        for i, (images, waypoints, commands, recovery_flags) in enumerate(train_loader):
            images = images.to(primary_device, non_blocking=True)
            commands = commands.to(primary_device, non_blocking=True)
            waypoints = waypoints.to(primary_device, non_blocking=True)
            
            # Xóa sạch đồ thị tính toán cũ, cắt đứt con trỏ tham chiếu
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                outputs = model(images, commands)
                pred_wp = outputs[:, :10].view(-1, 5, 2)
                pred_sigma = outputs[:, 10:].view(-1, 5, 1).expand(-1, 5, 2)
                target_wp = waypoints.view(-1, 5, 2)

                loss_wp = huber_loss(pred_wp, target_wp)
                loss_gnll = 0.5 * ((target_wp - pred_wp) ** 2 / pred_sigma + torch.log(pred_sigma))
                loss_gnll = loss_gnll.mean()

                lambda_wp = float(config.get("loss_lambda_wp", 1.0))
                lambda_gnll = float(config.get("loss_lambda_gnll", 0.1))
                loss = lambda_wp * loss_wp + lambda_gnll * loss_gnll
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

            # 🔥 XÓA RÁC TẠI MỖI BATCH: Tiêu diệt Tensor tránh phình RAM
            del images, waypoints, commands, recovery_flags
            del outputs, pred_wp, pred_sigma, target_wp, loss_wp, loss_gnll, loss
            
            # Ép máy dọn rác cực mạnh mỗi 200 batch để chống tràn RAM trên Kaggle
            if i % 200 == 0:
                import gc
                gc.collect()
        
        train_loss = running_loss / len(train_loader)
        
        # ──────────────────────────────────
        # Validation phase
        # ──────────────────────────────────
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, waypoints, commands, recovery_flags in val_loader:
                images = images.to(primary_device, non_blocking=True)
                commands = commands.to(primary_device, non_blocking=True)
                waypoints = waypoints.to(primary_device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                    outputs = model(images, commands)
                    pred_wp = outputs[:, :10].view(-1, 5, 2)
                    pred_sigma = outputs[:, 10:].view(-1, 5, 1).expand(-1, 5, 2)
                    target_wp = waypoints.view(-1, 5, 2)

                    loss_wp = huber_loss(pred_wp, target_wp)
                    loss_gnll = 0.5 * ((target_wp - pred_wp) ** 2 / pred_sigma + torch.log(pred_sigma))
                    loss_gnll = loss_gnll.mean()

                    lambda_wp = float(config.get("loss_lambda_wp", 1.0))
                    lambda_gnll = float(config.get("loss_lambda_gnll", 0.1))
                    loss = lambda_wp * loss_wp + lambda_gnll * loss_gnll
                
                val_loss += loss.item()

                # 🔥 XÓA RÁC Ở NHÁNH VALIDATION
                del images, waypoints, commands, recovery_flags
                del outputs, pred_wp, pred_sigma, target_wp, loss_wp, loss_gnll, loss
                
                # Ép máy dọn rác
                import gc
                gc.collect()
        
        val_loss = val_loss / len(val_loader)
        
        # ──────────────────────────────────
        # Memory info & logging
        # ──────────────────────────────────
        if torch.cuda.is_available() and num_gpus > 0:
            mem_allocated = torch.cuda.memory_allocated(primary_device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(primary_device) / 1024**3
            mem_str = f"| Mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB"
        else:
            mem_str = ""
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {mem_str}")
        
        # ──────────────────────────────────
        # Dọn rác System RAM (Kaggle Life-saver)
        # ──────────────────────────────────
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ──────────────────────────────────
        # Scheduler step
        # ──────────────────────────────────
        scheduler.step(val_loss)
        
        # ──────────────────────────────────
        # Model checkpoint
        # ──────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save model (handle DataParallel wrapper)
            model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
            torch.save(model_state, MODEL_SAVE_PATH)
            print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n🛑 Early stopping sau {epoch+1} epochs")
                print(f"   (không cải thiện trong {early_stopping_patience} epochs)")
                break

    print("\n" + "="*70)
    print("✅ HOÀN TẤT TRAINING!")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
