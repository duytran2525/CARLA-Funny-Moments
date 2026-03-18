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
from core_perception.cnn_model import NvidiaCNNV2 as NvidiaCNN      

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(ROOT_DIR, 'configs', 'train_params.yaml'))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    CSV_PATH = os.path.join(DATA_DIR, 'driving_log.csv')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'cnn_steering.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu Train trên thiết bị: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Đang phân chia tập Train và Validation từ file CSV...")
    df = pd.read_csv(CSV_PATH, header=0, names=['img_id', 'steering', 'throttle', 'brake', 'speed', 'command'])
    
    
    # Tính toán chỉ số chia (vd: 80% train, 20% val)
    train_split_ratio = config.get('train_split', 0.8)
    split_idx = int(train_split_ratio * len(df))
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Lưu ra 2 file CSV tạm thời
    train_csv_path = os.path.join(DATA_DIR, 'train_split_log.csv')
    val_csv_path = os.path.join(DATA_DIR, 'val_split_log.csv')
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # Khởi tạo Dataset từ 2 file CSV riêng biệt
    train_dataset = CarlaDataset(
        csv_file=train_csv_path, 
        root_dir=DATA_DIR, 
        transform=transform, 
        steering_correction=config['steering_correction'], 
        is_training=True  # Sẽ gọi hàm balance và augmentation
    )
    
    val_dataset = CarlaDataset(
        csv_file=val_csv_path, 
        root_dir=DATA_DIR, 
        transform=transform, 
        steering_correction=config['steering_correction'], 
        is_training=False # Không augmentation, giữ nguyên data thực tế để đánh giá
    )
    
    # Đưa vào DataLoader (Đã gỡ bỏ lớp Subset gây lỗi)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = NvidiaCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    # Learning rate scheduler: halve LR when val loss plateaus
    lr_patience = config.get('lr_patience', 3)
    lr_factor = config.get('lr_factor', 0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )
    # Mixed-precision scaler (disabled automatically on CPU)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = config['epochs']
    early_stopping_patience = config.get('early_stopping_patience', 5)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, steerings) in enumerate(train_loader):
            images, steerings = images.to(device, non_blocking=True), steerings.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # TÍNH TOÁN BẰNG MIXED PRECISION
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, steerings)
                
            # BACKWARD THÔNG QUA SCALER
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, steerings in val_loader:
                images, steerings = images.to(device, non_blocking=True), steerings.to(device, non_blocking=True)
                # Validation cũng có thể dùng autocast để đánh giá nhanh hơn
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, steerings)
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Step the learning rate scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Đã lưu trọng số mới tốt nhất tại epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping sau {epoch+1} epochs (không cải thiện trong {early_stopping_patience} epochs liên tiếp).")
                break

    print("Hoàn tất quá trình huấn luyện!")

if __name__ == "__main__":
    main()