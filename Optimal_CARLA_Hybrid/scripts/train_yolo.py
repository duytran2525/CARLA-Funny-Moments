from ultralytics import YOLO
import torch

def train_model():
    # Kiểm tra phần cứng: Ưu tiên dùng card rời NVIDIA
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Bắt đầu huấn luyện trên thiết bị: {'GPU (RTX 3050)' if device == 0 else 'CPU'}")

    # Khởi tạo mô hình lõi (bạn dùng bản small 18MB)
    model = YOLO('yolo11s.pt') 

    # Bắt đầu quá trình huấn luyện
    results = model.train(
        data='config/data.yaml',  # Đảm bảo bạn có file này trong thư mục config
        epochs=100,
        batch=8,                  # RTX 3050 4GB VRAM nên để batch 8 cho an toàn
        imgsz=640,
        device=device,
        project='models/yolo',    # Lưu thẳng kết quả vào thư mục models
        name='retrain_results'
    )
    print("Huấn luyện hoàn tất!")

if __name__ == '__main__':
    train_model()