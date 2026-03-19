import torch
import numpy as np
import cv2
import math
from ultralytics import YOLO
import os

class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.45):
        self.conf_threshold = conf_threshold

        # 1. Kiểm tra file model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model YOLO: {model_path}")

        # 2. Thiết bị chạy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 3. Load model YOLO
        print(f"[YOLO DETECTOR] Đang load model từ {model_path} lên {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.class_names = self.model.names

        # 4. Định nghĩa các class cần chú ý
        self.target_classes = ['pedestrian', 'vehicle', 'bike', 'motobike'] 
        
        # >>> THÊM MỚI: Từ điển chiều cao thực tế (mét) để tính khoảng cách <<<
        self.real_heights = {
            'pedestrian': 1.7, 
            'vehicle': 1.5,
            'bike': 1.5,
            'motobike': 1.5
        }
        # FOV camera mặc định của CARLA là 90 độ
        self.camera_fov = 90 

    def detect_and_evaluate(self, raw_image, distance_threshold=5.0):
        """
        Hàm chính: Nhận diện và tính khoảng cách thực tế.
        distance_threshold: Khoảng cách nguy hiểm để phanh (mặc định 5.0 mét).
        """
        # --- BƯỚC 1: Xử lý hình ảnh ---
        if raw_image.shape[2] == 4:
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else:
            img = raw_image.copy()

        height, width, _ = img.shape

        # >>> THÊM MỚI: Tính tiêu cự (focal length) của camera CARLA <<<
        fov_rad = math.radians(self.camera_fov)
        focal_length = (height / 2) / math.tan(fov_rad / 2)

        # --- BƯỚC 2: Chạy Inference YOLO ---
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        # --- BƯỚC 3: Phân tích và Ra quyết định ---
        emergency_flag = False
        processed_detections = []
        
        if len(results) > 0:
            boxes = results[0].boxes 

            for box in boxes:
                cls_id = int(box.cls[0]) 
                class_name = self.class_names[cls_id] 

                if class_name not in self.target_classes:
                    continue 

                # Lấy tọa độ bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # >>> THAY ĐỔI: Tính khoảng cách thay vì diện tích <<<
                box_height = y2 - y1
                real_h = self.real_heights.get(class_name, 1.5)
                
                # Tránh lỗi chia cho 0
                if box_height > 0:
                    distance_m = (real_h * focal_length) / box_height
                else:
                    distance_m = 999.0

                # Thu thập thông tin
                det_info = {
                    'box': (x1, y1, x2, y2),
                    'class_name': class_name,
                    'confidence': float(box.conf[0]),
                    'distance_m': distance_m
                }
                processed_detections.append(det_info)

                # Ra quyết định phanh dựa trên số MÉT thực tế
                if distance_m < distance_threshold:
                    emergency_flag = True
                    print(f"[CRITICAL] Có {class_name} cách xe {distance_m:.1f}m! ĐẠP PHANH GẤP!")
                    break 

        return processed_detections, emergency_flag