import torch
import numpy as np
import cv2
from ultralytics import YOLO
import os

class YoloDetector:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

        # Tìm đường dẫn đến best.pt
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(current_dir, '..', 'best.pt') 
        
        # 1. Kiểm tra sự tồn tại của file model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model YOLO tại đường dẫn: {model_path}. Vui lòng kiểm tra lại vị trí file best.pt")

        # 2. Thiết lập thiết bị chạy (Ưu tiên GPU CUDA cho RTX 3050)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 3. Load model YOLO 1 lần duy nhất
        print(f"[YOLO DETECTOR] Đang load model từ {model_path} lên thiết bị {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        
        # 4. Lấy tên các class từ model để tiện lọc
        self.class_names = self.model.names
        print(f"[YOLO DETECTOR] Model đã load xong. Số lượng class nhận diện được: {len(self.class_names)}")
        # print(f"[YOLO DETECTOR] Chi tiết các class: {self.class_names}") # Có thể uncomment để xem chi tiết

        # Định nghĩa các class cần phanh gấp
        self.target_classes = ['pedestrian', 'vehicle', 'bike', 'motobike'] 
        print(f"[YOLO DETECTOR] Chỉ lọc các đối tượng nguy hiểm thuộc nhóm: {self.target_classes}")

    
    def detect_and_evaluate(self, raw_image, area_threshold=0.3):
        """
        Hàm chính: Nhận ảnh từ CARLA, nhận diện đối tượng, 
        và tính toán diện tích để trả về cờ phanh khẩn cấp.
        """
        # --- BƯỚC 1: Xử lý hình ảnh ---
        if raw_image.shape[2] == 4: # Nếu là BGRA (CARLA default)
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else: # Nếu đã là 3 kênh
            img = raw_image.copy()

        # Lấy kích thước ảnh camera
        height, width, _ = img.shape
        img_area = height * width

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

                # Lọc class
                if class_name not in self.target_classes:
                    continue 

                # Lọc confidence
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                # Lấy tọa độ bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # >> ĐÃ SỬA LỖI TÍNH TOÁN DIỆN TÍCH <<
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Tính tỷ lệ diện tích
                area_ratio = box_area / img_area if img_area > 0 else 0

                # Thu thập thông tin vật thể
                det_info = {
                    'box': (x1, y1, x2, y2),
                    'class_name': class_name,
                    'confidence': conf,
                    'area_ratio': area_ratio
                }
                processed_detections.append(det_info)

                # Ra quyết định phanh
                if area_ratio > area_threshold:
                    emergency_flag = True
                    print(f"[CRITICAL WARNING] {class_name} quá gần! Tỷ lệ diện tích: {area_ratio:.2f}")
                    # Nên giữ break ở đây để tối ưu hiệu năng, vì chỉ cần 1 vật quá gần là đủ để đạp phanh rồi
                    break 

        return processed_detections, emergency_flag