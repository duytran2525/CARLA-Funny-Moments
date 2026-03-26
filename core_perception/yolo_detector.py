import math
import os

import cv2
import torch
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Khong tim thay file model YOLO: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[YOLO DETECTOR] Dang load model tu {model_path} len thiet bi {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.class_names = self.model.names

        self.display_classes = [
            "pedestrian",
            "vehicle",
            "two_wheeler",
            "traffic_light",
            "traffic_sign",
        ]
        self.target_classes = ["pedestrian", "vehicle", "two_wheeler"]
        self.brake_classes = set(self.target_classes)
        print(
            f"[YOLO DETECTOR] Hien thi: {self.display_classes} | "
            f"Phanh gap: {sorted(self.brake_classes)}"
        )

        self.reference_objects = {
            "pedestrian": {"real_h": 1.6, "real_w": 0.45},
            "vehicle": {"real_h": 1.5, "real_w": 1.8},
            "two_wheeler": {"real_h": 1.1, "real_w": 0.6},
        }

    def estimate_distance_optimized(self, class_name, box, height, width):
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        if box_height <= 0 or box_width <= 0:
            return float("inf")

        obj_params = self.reference_objects.get(class_name, {"real_h": 1.5, "real_w": 1.8})
        real_h = obj_params["real_h"]
        real_w = obj_params["real_w"]

        fov_deg = 90.0
        fov_rad = math.radians(fov_deg)
        focal_length = (width / 2) / math.tan(fov_rad / 2)

        d_height = (real_h * focal_length) / box_height
        d_width = (real_w * focal_length) / box_width
        return min(d_height, d_width)

    def detect_and_evaluate(self, raw_image, distance_threshold=5.0):
        if raw_image is None or raw_image.ndim != 3 or raw_image.shape[2] not in (3, 4):
            raise ValueError("raw_image phai la anh HxWx3 hoac HxWx4.")

        if raw_image.shape[2] == 4:
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else:
            img = raw_image.copy()

        height, width, _ = img.shape
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        emergency_flag = False
        processed_detections = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]

                if class_name not in self.display_classes:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                distance = self.estimate_distance_optimized(class_name, xyxy, height, width)

                x1, y1, x2, y2 = map(int, xyxy)
                processed_detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "confidence": conf,
                        "distance": distance,
                    }
                )

                if class_name in self.brake_classes and distance < distance_threshold:
                    emergency_flag = True

        return processed_detections, emergency_flag
