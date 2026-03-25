import torch
import numpy as np
import cv2
from ultralytics import YOLO
import math
import os

class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        
        # 1. Kiểm tra sự tồn tại của file model TRƯỚC TIÊN
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model YOLO: {model_path}")

        # 2. Thiết lập thiết bị chạy (Ưu tiên GPU CUDA cho RTX 3050)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 3. Load model YOLO 1 lần duy nhất
        print(f"[YOLO DETECTOR] Đang load model từ {model_path} lên thiết bị {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.class_names = self.model.names

        self.display_classes = ['pedestrian', 'vehicle', 'two_wheeler', 'traffic_light', 'traffic_sign']
        self.target_classes = ['pedestrian', 'vehicle', 'two_wheeler'] 
        print(f"[YOLO DETECTOR] Hiển thị: {self.display_classes} | Phanh gấp: {self.brake_classes}")

        # TỪ ĐIỂN THAM SỐ VẬT THỂ (MÁY CHUẨN)
        # Bao gồm: Chiều cao thực tế (real_h), Chiều rộng thực tế (real_w) 
        # và ngưỡng để chuyển đổi phương pháp (ratio_threshold)
        self.reference_objects = {
            'pedestrian': {'real_h': 1.6, 'real_w': 0.45}, # Người
            'vehicle': {'real_h': 1.5, 'real_w': 1.8},    # Xe con (rộng)
            'two_wheeler': {'real_h': 1.1, 'real_w': 0.6},        
        }

    def estimate_distance_optimized(self, class_name, box, height, width):
        """
        Hàm ước lượng khoảng cách tối ưu sử dụng cả chiều cao và chiều dài.
        """
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # Tránh lỗi chia cho 0
        if box_height == 0 or box_width == 0: return float('inf')

        # Lấy thông số tham số vật thể
        obj_params = self.reference_objects.get(class_name, {'real_h': 1.5, 'real_w': 1.8})
        real_h = obj_params['real_h']
        real_w = obj_params['real_w']
        
        # TÍNH FOCAL LENGTH CHUẨN (Dùng width)
        # Giả định FOV ngang là 90 độ (Bạn nên thay bằng FOV thực tế từ carla_manager.py nếu có thể)
        fov_deg = 90.0 
        fov_rad = math.radians(fov_deg)
        focal_length = (width / 2) / math.tan(fov_rad / 2)

        # CÁCH 1: Tính theo Chiều cao (Chuẩn lý thuyết)
        d_height = (real_h * focal_length) / box_height
        
        # CÁCH 2: Tính theo Chiều dài (Ổn định khi ở gần)
        d_width = (real_w * focal_length) / box_width

        # HỢP NHẤT: Lấy khoảng cách ngắn nhất để an toàn tối đa
        final_distance = min(d_height, d_width)

        # Debug nhỏ để xem kết quả:
        # if final_distance < 10:
        #     print(f"[{class_name}] H:{d_height:.1f}m | W:{d_width:.1f}m -> {final_distance:.1f}m")

        return final_distance

    
    def detect_and_evaluate(self, raw_image, distance_threshold=5.0):
        """
        Hàm chính: Nhận diện và ước lượng khoảng cách.
        """
        # --- BƯỚC 1: Xử lý hình ảnh ---
        if raw_image.shape[2] == 4: # Nếu là BGRA (CARLA default)
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else: # Nếu đã là 3 kênh
            img = raw_image.copy()

        height, width, _ = img.shape

        # --- BƯỚC 2: Chạy Inference YOLO ---
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        # --- BƯỚC 3: Xử lý kết quả và Tính toán diện tích ---
        emergency_flag = False
        processed_detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes 

            for box in boxes:
                cls_id = int(box.cls[0]) 
                class_name = self.class_names[cls_id] 

                # 1. Lọc class để hiển thị (cho phép tất cả 5 class đi qua)
                if class_name not in self.display_classes:
                    continue 

                # Lọc confidence
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                # Lấy tọa độ bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Tính toán khoảng cách
                distance = self.estimate_distance_optimized(class_name, xyxy, height, width)

                # Thu thập thông tin vật thể để vẽ Bounding Box
                x1, y1, x2, y2 = map(int, xyxy)
                det_info = {
                    'box': (x1, y1, x2, y2),
                    'class_name': class_name,
                    'confidence': conf,
                    'distance': distance
                }
                processed_detections.append(det_info)

                # 2. Logic phanh khẩn cấp: CHỈ xét những vật cản nằm trong brake_classes
                if class_name in self.brake_classes and distance < distance_threshold:
                    emergency_flag = True

        return processed_detections, emergency_flag