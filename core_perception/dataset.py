from __future__ import annotations

import os
import random
import sys
from typing import Iterable

import cv2
cv2.setNumThreads(0)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _safe_print(message: object = "") -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        safe_message = str(message).encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        print(safe_message)


def _image_id(value: object) -> str:
    stem = str(value).strip()
    if "." in stem:
        stem = stem.split(".", 1)[0]
    return stem.zfill(8) + ".jpg"


def _filter_stationary_rows(
    data_df: pd.DataFrame,
    *,
    min_speed_kmh: float,
    min_wp5_x_m: float,
) -> pd.DataFrame:
    """Drop stopped samples whose future trajectory barely moves forward."""
    required_columns = {"speed", "wp_5_x"}
    if not required_columns.issubset(set(data_df.columns)):
        return data_df

    speed = pd.to_numeric(data_df["speed"], errors="coerce")
    wp5_x = pd.to_numeric(data_df["wp_5_x"], errors="coerce")
    stationary_mask = (speed < min_speed_kmh) & (wp5_x < min_wp5_x_m)
    dropped = int(stationary_mask.sum())
    if dropped <= 0:
        return data_df

    kept_df = data_df.loc[~stationary_mask].reset_index(drop=True)
    _safe_print(
        "Stationary filter: "
        f"{len(data_df)} -> {len(kept_df)} rows "
        f"(dropped {dropped}, speed < {min_speed_kmh:.1f} km/h "
        f"and wp_5_x < {min_wp5_x_m:.1f} m)"
    )
    return kept_df


class WaypointCarlaDataset(Dataset):
    """
    Dataset V4.0 - Dự đoán Quỹ đạo Không-Thời gian (Spatio-Temporal Waypoints).
    Đã vá lỗi: Hỗ trợ Cấu trúc Thư mục mới /TownXX/, và Filter xe đứng yên.
    """
    def __init__(self, csv_file, root_dir, transform=None, is_training=True, geometric_offset=0.35, filter_stationary=True, min_speed_kmh=1.0, min_wp5_x_m=3.0):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = bool(is_training)
        self.geometric_offset = float(geometric_offset)
        self.samples: list[dict[str, object]] = []
        self.recovery_flags: list[int] = []

        self.data_df = pd.read_csv(csv_file)
        
        # Áp dụng bộ lọc thông minh của bạn
        if filter_stationary:
            self.data_df = _filter_stationary_rows(
                self.data_df,
                min_speed_kmh=float(min_speed_kmh),
                min_wp5_x_m=float(min_wp5_x_m),
            )
            
        self._prepare_data()

        del self.data_df
    def _prepare_data(self):
        _safe_print(f"Đang phân tích cấu trúc dữ liệu đa luồng (Multi-frame) từ {len(self.data_df)} dòng...")
        
        missing_count = 0
        
        for _, row in self.data_df.iterrows():
            command = int(row.get('command', 0))
            recovery_flag = float(row.get('recovery_flag', 0.0))
            
            # Đọc 10 giá trị Waypoint (5 cặp X, Y)
            wp_car = np.array([
                row['wp_1_x'], row['wp_1_y'],
                row['wp_2_x'], row['wp_2_y'],
                row['wp_3_x'], row['wp_3_y'],
                row['wp_4_x'], row['wp_4_y'],
                row['wp_5_x'], row['wp_5_y']
            ], dtype=np.float32).reshape(5, 2)
            
            # LẤY ĐƯỜNG DẪN ẢNH CHUẨN TỪ CSV (Giữ nguyên bản vá quan trọng)
            if 'center_camera' not in row or pd.isna(row['center_camera']):
                continue
                
            center_path_full = row['center_camera']
            town_folder = center_path_full.replace('\\', '/').split('/')[0] 
            
            img_t0_name = _image_id(row['img_id'])
            img_t1_name = _image_id(row['img_id_tm03'])
            img_t2_name = _image_id(row['img_id_tm06'])
            
            configs = [
                ('images_center', 0.0),
                ('images_left', self.geometric_offset),
                ('images_right', -self.geometric_offset)
            ]
            
            for camera_type, offset in configs:
                if not self.is_training and camera_type != 'images_center':
                    continue
                    
                path_t0 = os.path.join(self.root_dir, town_folder, camera_type, img_t0_name)
                path_t1 = os.path.join(self.root_dir, town_folder, camera_type, img_t1_name)
                path_t2 = os.path.join(self.root_dir, town_folder, camera_type, img_t2_name)
                
                if os.path.exists(path_t0) and os.path.exists(path_t1) and os.path.exists(path_t2):
                    wp_cam = wp_car.copy()
                    wp_cam[:, 1] += offset
                    
                    self.samples.append({
                        'paths': (path_t0, path_t1, path_t2),
                        'waypoints': wp_cam,
                        'command': command,
                        'recovery_flag': recovery_flag
                    })
                    self.recovery_flags.append(int(recovery_flag))
                else:
                    missing_count += 1
                    
        _safe_print(f"✅ Hoàn tất! Đã nạp thành công {len(self.samples)} mẫu dữ liệu Tensor.")
        if missing_count > 0:
            _safe_print(f"⚠️ Bỏ qua {missing_count} mẫu do không tìm thấy đủ 3 khung hình ảnh trên ổ cứng.")

    def __len__(self) -> int:
        return len(self.samples)

    def _random_brightness(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + (random.random() - 0.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_shadow(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        pts = np.array(
            [
                [random.randint(0, width), 0],
                [random.randint(0, width), height],
                [random.randint(0, width), height],
                [random.randint(0, width), 0],
            ],
            np.int32,
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = np.zeros_like(hsv[:, :, 2])
        cv2.fillPoly(mask, [pts], 255)
        hsv[:, :, 2][mask == 255] = hsv[:, :, 2][mask == 255] * 0.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_blur(self, image: np.ndarray) -> np.ndarray:
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _random_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _random_contrast(self, image: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
        factor = random.uniform(low, high)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _random_cutout(
        self,
        image: np.ndarray,
        min_size_ratio: float = 0.1,
        max_size_ratio: float = 0.3,
    ) -> np.ndarray:
        height, width, _channels = image.shape
        cut_h = int(height * random.uniform(min_size_ratio, max_size_ratio))
        cut_w = int(width * random.uniform(min_size_ratio, max_size_ratio))
        center_y = random.randint(0, height)
        center_x = random.randint(0, width)
        y1, y2 = np.clip(center_y - cut_h // 2, 0, height), np.clip(center_y + cut_h // 2, 0, height)
        x1, x2 = np.clip(center_x - cut_w // 2, 0, width), np.clip(center_x + cut_w // 2, 0, width)
        image[y1:y2, x1:x2, :] = 0
        return image

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        paths = sample["paths"]
        waypoints = np.asarray(sample["waypoints"], dtype=np.float32).copy()
        command = int(sample["command"])
        recovery_flag = float(sample["recovery_flag"])

        images = []
        do_flip = self.is_training and random.random() > 0.5
        do_brightness = self.is_training and random.random() > 0.5
        do_shadow = self.is_training and random.random() > 0.5
        do_blur = self.is_training and random.random() > 0.5
        do_noise = self.is_training and random.random() > 0.7
        do_contrast = self.is_training and random.random() > 0.5
        do_cutout = self.is_training and random.random() > 0.5

        for path in paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((600, 800, 3), dtype=np.uint8) 
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            img = img[int(h * 0.45):, :, :]
            
            if self.is_training:
                if do_brightness:
                    img = self._random_brightness(img)
                if do_shadow:
                    img = self._random_shadow(img)
                if do_blur:
                    img = self._random_blur(img)
                if do_noise:
                    img = self._random_noise(img)
                if do_contrast:
                    img = self._random_contrast(img)
                if do_cutout:
                    img = self._random_cutout(img)
                if do_flip:
                    img = cv2.flip(img, 1)

            img = cv2.resize(img, (200, 66))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            images.append(img)

        stacked_image = np.concatenate(images, axis=-1)
        stacked_tensor = torch.from_numpy(stacked_image).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            stacked_tensor = self.transform(stacked_tensor)

        if do_flip:
            waypoints[:, 1] = -waypoints[:, 1]
            if command == 1:
                command = 2
            elif command == 2:
                command = 1

        waypoint_tensor = torch.tensor(waypoints, dtype=torch.float32)
        command_tensor = torch.tensor(command, dtype=torch.long)
        recovery_tensor = torch.tensor(recovery_flag, dtype=torch.float32)
        
        return stacked_tensor, waypoint_tensor, command_tensor, recovery_tensor