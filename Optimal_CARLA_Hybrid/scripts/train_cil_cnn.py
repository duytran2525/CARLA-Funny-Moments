import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_perception.dataset import CarlaDataset, CILCarlaDataset
from core_perception.cil_cnn_model import NvidiaCNNV2, CIL_NvidiaCNN


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def _build_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def _split_csv(csv_path, data_dir, train_ratio, prefix=''):
    """
    Đọc CSV, chia train/val và lưu ra 2 file tạm trong ``data_dir``.

    Ghi chú: Các file tạm thời (``{prefix}train_split_log.csv`` và
    ``{prefix}val_split_log.csv``) được giữ lại sau khi training kết thúc
    để có thể tái tạo kết quả. Chúng có thể bị ghi đè ở lần chạy tiếp theo.
    """
    df = pd.read_csv(csv_path)
    split_idx = int(train_ratio * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    train_csv = os.path.join(data_dir, f'{prefix}train_split_log.csv')
    val_csv = os.path.join(data_dir, f'{prefix}val_split_log.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    return train_csv, val_csv


def _make_loaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def setup_phase1(config, data_dir, device):
    """
    Chuẩn bị model, dataset và optimizer cho Phase 1 (Behavioral Cloning).

    Sử dụng ``NvidiaCNNV2`` và ``CarlaDataset``.
    Đầu vào: ảnh → Đầu ra: góc lái.

    Returns
    -------
    model, criterion, optimizer, train_loader, val_loader
    """
    csv_path = os.path.join(data_dir, 'raw_driving', 'driving_log.csv')
    transform = _build_transform()
    train_csv, val_csv = _split_csv(
        csv_path, data_dir, config.get('train_split', 0.8), prefix=''
    )
    print("Đang phân chia tập Train và Validation từ file CSV...")

    train_dataset = CarlaDataset(
        csv_file=train_csv, root_dir=data_dir, transform=transform,
        steering_correction=config['steering_correction'], is_training=True
    )
    val_dataset = CarlaDataset(
        csv_file=val_csv, root_dir=data_dir, transform=transform,
        steering_correction=config['steering_correction'], is_training=False
    )

    train_loader, val_loader = _make_loaders(
        train_dataset, val_dataset, config['batch_size']
    )
    model = NvidiaCNNV2().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    return model, criterion, optimizer, train_loader, val_loader


def setup_phase2(config, data_dir, device):
    """
    Chuẩn bị model, dataset và optimizer cho Phase 2 (CIL).

    Sử dụng ``CIL_NvidiaCNN`` và ``CILCarlaDataset``.
    Đầu vào: ảnh + vận tốc + lệnh GPS → Đầu ra: góc lái.

    CSV phải có cột: ``img_id, steering, throttle, brake, speed, command``.

    Returns
    -------
    model, criterion, optimizer, train_loader, val_loader
    """
    csv_path = os.path.join(data_dir, 'raw_driving', 'driving_log.csv')
    transform = _build_transform()
    print("Phase 2 – CIL Mode: Đang phân chia dữ liệu Train/Val...")
    train_csv, val_csv = _split_csv(
        csv_path, data_dir, config.get('train_split', 0.8), prefix='cil_'
    )

    train_dataset = CILCarlaDataset(
        csv_file=train_csv, root_dir=data_dir, transform=transform,
        steering_correction=config['steering_correction'], is_training=True
    )
    val_dataset = CILCarlaDataset(
        csv_file=val_csv, root_dir=data_dir, transform=transform,
        steering_correction=config['steering_correction'], is_training=False
    )

    train_loader, val_loader = _make_loaders(
        train_dataset, val_dataset, config['batch_size']
    )
    model = CIL_NvidiaCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    return model, criterion, optimizer, train_loader, val_loader


def run_training_loop(
    model, criterion, optimizer,
    train_loader, val_loader,
    config, model_save_path, device,
    is_cil: bool = False,
):
    """
    Vòng lặp huấn luyện chung cho Phase 1 và Phase 2.

    Tính năng
    ---------
    * Mixed-precision (AMP) khi có GPU.
    * Learning rate scheduler ``ReduceLROnPlateau``.
    * Early stopping khi val loss không cải thiện.
    * Lưu checkpoint tốt nhất vào ``model_save_path``.

    Parameters
    ----------
    is_cil : bool
        ``True`` → batch có 4 phần tử ``(image, steering, speed, command)``.
        ``False`` → batch có 2 phần tử ``(image, steering)``.
    """
    lr_patience = config.get('lr_patience', 3)
    lr_factor = config.get('lr_factor', 0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = config['epochs']
    early_stopping_patience = config.get('early_stopping_patience', 5)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            if is_cil:
                images, steerings, speeds, commands = batch
                images = images.to(device, non_blocking=True)
                steerings = steerings.to(device, non_blocking=True)
                speeds = speeds.to(device, non_blocking=True)
                commands = commands.to(device, non_blocking=True)
            else:
                images, steerings = batch
                images = images.to(device, non_blocking=True)
                steerings = steerings.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images, speeds, commands) if is_cil else model(images)
                loss = criterion(outputs, steerings)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if is_cil:
                    images, steerings, speeds, commands = batch
                    images = images.to(device, non_blocking=True)
                    steerings = steerings.to(device, non_blocking=True)
                    speeds = speeds.to(device, non_blocking=True)
                    commands = commands.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(images, speeds, commands)
                        loss = criterion(outputs, steerings)
                else:
                    images, steerings = batch
                    images = images.to(device, non_blocking=True)
                    steerings = steerings.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, steerings)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Đã lưu trọng số mới tốt nhất tại epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping sau {epoch+1} epochs "
                    f"(không cải thiện trong {early_stopping_patience} epochs liên tiếp)."
                )
                break


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện CIL CNN steering model cho CARLA tự lái (Phase 2)."
    )
    parser.add_argument(
        '--mode', choices=['phase1', 'phase2'], default='phase2',
        help=(
            "Chế độ huấn luyện:\n"
            "  phase1 – NvidiaCNNV2 (Behavioral Cloning, ảnh → góc lái)\n"
            "  phase2 – CIL_NvidiaCNN (CIL, ảnh+vận tốc+lệnh GPS → góc lái)"
        )
    )
    parser.add_argument(
        '--config', default=None,
        help="Đường dẫn tới train_params.yaml (mặc định: configs/train_params.yaml)."
    )
    args = parser.parse_args()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = args.config or os.path.join(ROOT_DIR, 'configs', 'train_params.yaml')
    config = load_config(config_path)
    data_dir = os.path.join(ROOT_DIR, 'data')
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu Train [{args.mode.upper()}] trên thiết bị: {device}")

    if args.mode == 'phase2':
        model_save_path = os.path.join(ROOT_DIR, 'models', 'cil_cnn_steering.pth')
        model, criterion, optimizer, train_loader, val_loader = setup_phase2(
            config, data_dir, device
        )
        is_cil = True
    else:
        model_save_path = os.path.join(ROOT_DIR, 'models', 'cnn_steering.pth')
        model, criterion, optimizer, train_loader, val_loader = setup_phase1(
            config, data_dir, device
        )
        is_cil = False

    run_training_loop(
        model, criterion, optimizer,
        train_loader, val_loader,
        config, model_save_path, device,
        is_cil=is_cil,
    )
    print("Hoàn tất quá trình huấn luyện!")


if __name__ == "__main__":
    main()
