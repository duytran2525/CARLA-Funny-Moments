import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_perception.dataset import CarlaDataset
from core_perception.cnn_model import NvidiaCNN

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(ROOT_DIR, 'configs', 'train_params.yaml'))
    
    CSV_PATH = os.path.join(ROOT_DIR, 'data', 'raw_driving', 'driving_log.csv')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'cnn_steering.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu Train trên thiết bị: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

    # 1. KHỞI TẠO 2 DATASET RIÊNG BIỆT ĐỂ BẢO VỆ TẬP VAL
    full_train_dataset = CarlaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=transform, steering_correction=config['steering_correction'], is_training=True)
    full_val_dataset = CarlaDataset(csv_file=CSV_PATH, root_dir=ROOT_DIR, transform=transform, steering_correction=config['steering_correction'], is_training=False)
    
    # Tạo danh sách index và xáo trộn
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # Chia 80/20
    train_split = int(np.floor(config['train_split'] * dataset_size))
    train_indices, val_indices = indices[:train_split], indices[train_split:]
    
    # Áp dụng index vào đúng Dataset tương ứng
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    
    # 2. BẬT PIN_MEMORY ĐỂ TĂNG TỐC ĐẨY DỮ LIỆU LÊN GPU
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = NvidiaCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    # 3. KHỞI TẠO SCALER CHO MIXED PRECISION (AMP)
    scaler = torch.cuda.amp.GradScaler()

    epochs = config['epochs']
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, steerings) in enumerate(train_loader):
            images, steerings = images.to(device, non_blocking=True), steerings.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # TÍNH TOÁN BẰNG MIXED PRECISION
            with torch.cuda.amp.autocast():
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
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, steerings)
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Đã lưu trọng số mới tốt nhất tại epoch {epoch+1}")

    print("Hoàn tất quá trình huấn luyện!")

if __name__ == "__main__":
    main()