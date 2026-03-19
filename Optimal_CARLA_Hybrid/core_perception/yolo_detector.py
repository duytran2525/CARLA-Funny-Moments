import cv2
import torch
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path=r""):  #thg bò đưa link file vào đây
        """
        Hàm khởi tạo: Chỉ chạy 1 lần duy nhất khi khởi động xe CARLA.
        Sẽ nạp cục 'best.pt' nặng 18.7MB thẳng vào VRAM của con RTX 3050.
        """
        # Tự động nhận diện card đồ họa
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Perception] Đang khởi động YOLO trên: {self.device.upper()}")
        
        # Load mô hình
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Hàm này sẽ được gọi liên tục ở mỗi khung hình camera từ CARLA (30-60 lần/giây).
        """
        # Chạy suy luận (Inference), ép verbose=False để terminal không bị spam lag máy
        # conf=0.5: Chỉ lấy các dự đoán có độ tự tin trên 50% để tránh nhận diện rác
        results = self.model(frame, device=self.device, verbose=False, conf=0.5)
        
        # 1. Vẽ khung chữ nhật lên ảnh để hiển thị (Debug)
        annotated_frame = results[0].plot()
        
        # 2. Bóc tách dữ liệu thô (Tọa độ x, y, width, height, class_id)
        # Đây là thứ mà xe tự lái CẦN để tính toán khoảng cách và ra quyết định phanh/đánh lái
        detections = results[0].boxes.data.tolist() if results[0].boxes else []
        
        return annotated_frame, detections