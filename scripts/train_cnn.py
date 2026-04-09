import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_perception.dataset import CarlaDataset, CILCarlaDataset
from core_perception.cnn_model import CIL_NvidiaCNN as NvidiaCNN

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

def collate_cil_batch(batch):
    """
    Custom collate_fn để fix dtype casting issue trong DataLoader.
    
    DataLoader mặc định collate bằng cách stack tất cả tensors cùng dtype,
    gây ra casting nhầm (commands float32→cần long, steerings long→cần float32).
    
    __getitem__ trả về: (image, steering, speed, command)
    
    Returns:
        (images, steerings, speeds, commands) với dtype chính xác
    """
    images, steerings, speeds, commands = zip(*batch)
    
    return (
        torch.stack(images),                                    # float32
        torch.stack(steerings).float(),                         # ← FORCE float32
        torch.stack(speeds),                                    # float32
        torch.stack(commands).long(),                           # ← FORCE long
    )

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(ROOT_DIR, 'configs', 'train_params.yaml'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    CSV_PATH = os.path.join(DATA_DIR, 'driving_log.csv')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'cnn_steering.pth')
    
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Đang phân chia tập Train và Validation từ file CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Xóa file CSV tạm cũ để tránh load lại steering đã bị normalize
    train_csv_path = os.path.join(DATA_DIR, 'train_split_log.csv')
    val_csv_path = os.path.join(DATA_DIR, 'val_split_log.csv')
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

    # Khởi tạo Dataset từ 2 file CSV riêng biệt
    train_dataset = CILCarlaDataset(
        csv_file=train_csv_path, 
        root_dir=DATA_DIR, 
        transform=transform, 
        steering_correction=config['steering_correction'], 
        is_training=True  # Sẽ gọi hàm balance và augmentation
    )
    
    val_dataset = CILCarlaDataset(
        csv_file=val_csv_path, 
        root_dir=DATA_DIR, 
        transform=transform, 
        steering_correction=config['steering_correction'], 
        is_training=False # Không augmentation, giữ nguyên data thực tế để đánh giá
    )
    
    # ⭐ OPTIMIZE: num_workers dựa trên CPU cores
    num_workers = min(8, (os.cpu_count() or 1) // 2)  # Use half CPU cores
    pin_mem = torch.cuda.is_available()
    
    batch_size = config['batch_size']
    
    # Đưa vào DataLoader với tối ưu
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,      # ← OPTIMIZED
        pin_memory=pin_mem, 
        collate_fn=collate_cil_batch,
        persistent_workers=(num_workers > 0)  # ← NEW: Keep workers alive
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,      # ← OPTIMIZED
        pin_memory=pin_mem, 
        collate_fn=collate_cil_batch,
        persistent_workers=(num_workers > 0)  # ← NEW
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
    model = NvidiaCNN().to(primary_device)
    
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
    
    criterion = nn.MSELoss()
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
    print(f"Loss function: MSELoss")
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
        
        for i, (images, steerings, speeds, commands) in enumerate(train_loader):
            images = images.to(primary_device, non_blocking=True)
            speeds = speeds.to(primary_device, non_blocking=True).float()
            commands = commands.to(primary_device, non_blocking=True)
            steerings = steerings.to(primary_device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                outputs = model(images, speeds, commands)
                loss = criterion(outputs, steerings)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # ──────────────────────────────────
        # Validation phase
        # ──────────────────────────────────
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, steerings, speeds, commands in val_loader:
                images = images.to(primary_device, non_blocking=True)
                speeds = speeds.to(primary_device, non_blocking=True)
                commands = commands.to(primary_device, non_blocking=True)
                steerings = steerings.to(primary_device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                    outputs = model(images, speeds, commands)
                    loss = criterion(outputs, steerings)
                
                val_loss += loss.item()
        
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
