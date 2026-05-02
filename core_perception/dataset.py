from __future__ import annotations

import os
import random
import sys
from typing import Iterable

import cv2
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
    Đã vá lỗi: Hỗ trợ WeightedRandomSampler, chuẩn hóa Shape [5, 2], tích hợp Transform.
    """
    def __init__(self, csv_file, root_dir, transform=None, is_training=True, geometric_offset=0.35):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = bool(is_training)
        self.geometric_offset = float(geometric_offset)
        self.samples: list[dict[str, object]] = []
        self.recovery_flags: list[int] = []

        self.data_df = pd.read_csv(csv_file)
        if filter_stationary:
            self.data_df = _filter_stationary_rows(
                self.data_df,
                min_speed_kmh=float(min_speed_kmh),
                min_wp5_x_m=float(min_wp5_x_m),
            )
        self._validate_columns()
        self._prepare_data()

    def _prepare_data(self):
        print(f"Đang phân tích cấu trúc dữ liệu đa luồng (Multi-frame) từ {len(self.data_df)} dòng...")
        
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
            
            img_id_t0 = str(row['img_id']).split('.')[0].zfill(8) + '.jpg'
            img_id_t1 = str(row['img_id_tm03']).split('.')[0].zfill(8) + '.jpg'
            img_id_t2 = str(row['img_id_tm06']).split('.')[0].zfill(8) + '.jpg'
            
            configs = [
                ('images_center', 0.0),
                ('images_left', self.geometric_offset),
                ('images_right', -self.geometric_offset)
            ]
            
            for sub_dir, offset in configs:
                if not self.is_training and sub_dir != 'images_center':
                    continue
                    
                path_t0 = os.path.join(self.root_dir, sub_dir, img_id_t0)
                path_t1 = os.path.join(self.root_dir, sub_dir, img_id_t1)
                path_t2 = os.path.join(self.root_dir, sub_dir, img_id_t2)
                
                if os.path.exists(path_t0) and os.path.exists(path_t1) and os.path.exists(path_t2):
                    wp_cam = wp_car.copy()
                    wp_cam[:, 1] += offset
                    
                    self.samples.append({
                        'paths': (path_t0, path_t1, path_t2),
                        'waypoints': wp_cam,
                        'command': command,
                        'recovery_flag': recovery_flag
                    })
                    # [FIX A] Cập nhật danh sách cờ tương ứng với từng mẫu dữ liệu
                    self.recovery_flags.append(int(recovery_flag))
                    
        print(f"✅ Hoàn tất! Đã nạp thành công {len(self.samples)} khối lượng Tensor.")

    def __len__(self) -> int:
        return len(self.samples)

    # ... (CÁC HÀM AUGMENTATION GIỮ NGUYÊN NHƯ BẢN TRƯỚC) ...
    def _random_brightness(self, image):
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            img = img[int(h * 0.45):, :, :]
            
            if self.is_training:
                if do_brightness:
                    image = self._random_brightness(image)
                if do_shadow:
                    image = self._random_shadow(image)
                if do_blur:
                    image = self._random_blur(image)
                if do_noise:
                    image = self._random_noise(image)
                if do_contrast:
                    image = self._random_contrast(image)
                if do_cutout:
                    image = self._random_cutout(image)
                if do_flip:
                    image = cv2.flip(image, 1)

            image = cv2.resize(image, (200, 66))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            images.append(image)

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


class CarlaDataset(Dataset):
    """Single-frame steering dataset kept for older behavioral-cloning code."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        steering_correction: float = 0.2,
        is_training: bool = True,
        filter_stationary: bool = True,
        min_speed_kmh: float = 1.0,
        min_wp5_x_m: float = 3.0,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.steering_correction = float(steering_correction)
        self.is_training = bool(is_training)
        self.samples: list[tuple[str, float]] = []

        self.data_df = pd.read_csv(csv_file)
        if filter_stationary:
            self.data_df = _filter_stationary_rows(
                self.data_df,
                min_speed_kmh=float(min_speed_kmh),
                min_wp5_x_m=float(min_wp5_x_m),
            )
        self._prepare_data()

    def _prepare_data(self) -> None:
        _safe_print("Đang rà soát file ảnh từ 3 camera...")
        max_steer = 1.0

        for _, row in self.data_df.iterrows():
            steering = float(row["steering"])
            image_id = _image_id(row["img_id"])
            configs = (
                ("images_center", 0.0),
                ("images_left", self.steering_correction),
                ("images_right", -self.steering_correction),
            )

            for sub_dir, correction in configs:
                if not self.is_training and sub_dir != "images_center":
                    continue
                image_path = os.path.join(self.root_dir, sub_dir, image_id)
                if os.path.exists(image_path):
                    self.samples.append((image_path, (steering + correction) / max_steer))

        _safe_print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")
        _safe_print(f"   Steering được chuẩn hóa bằng: MAX_STEER = {max_steer} (symmetric scaling)")
        if self.is_training:
            self._balance_steering_distribution()

    def _balance_steering_distribution(self, bins: int = 25, max_per_bin: int | None = None) -> None:
        if not self.samples:
            return

        steerings = [steering for _path, steering in self.samples]
        hist, bin_edges = np.histogram(steerings, bins=bins)
        if max_per_bin is None:
            max_per_bin = max(1, int(np.mean(hist)))

        balanced_samples = []
        bin_counts = {index: 0 for index in range(bins)}
        random.shuffle(self.samples)

        for image_path, steering in self.samples:
            bin_idx = min(np.digitize(steering, bin_edges[1:-1]), bins - 1)
            if bin_counts[bin_idx] < max_per_bin:
                balanced_samples.append((image_path, steering))
                bin_counts[bin_idx] += 1

        _safe_print(f"Cân bằng dữ liệu: {len(self.samples)} -> {len(balanced_samples)} mẫu")
        self.samples = balanced_samples

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

    def _random_translate(
        self,
        image: np.ndarray,
        steering: float,
        range_x: int = 20,
        range_y: int = 10,
    ) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        trans_x = range_x * (random.random() - 0.5)
        trans_y = range_y * (random.random() - 0.5)
        steering += trans_x * 0.004
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering

    def _random_blur(self, image: np.ndarray) -> np.ndarray:
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _random_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _random_rotation(
        self,
        image: np.ndarray,
        steering: float,
        max_angle: float = 5,
    ) -> tuple[np.ndarray, float]:
        angle = random.uniform(-max_angle, max_angle)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height))
        steering -= angle * 0.01
        return image, steering

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

    def _process_image_and_steering(self, image_path: str, steering: float):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Khong doc duoc anh dataset: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height = image.shape[0]
        image = image[int(height * 0.45) :, :, :]

        if self.is_training:
            if random.random() > 0.5:
                image, steering = self._random_translate(image, steering)
            if random.random() > 0.5:
                image = self._random_brightness(image)
            if random.random() > 0.5:
                image = self._random_shadow(image)
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                steering = -steering
            if random.random() > 0.5:
                image = self._random_blur(image)
            if random.random() > 0.7:
                image = self._random_noise(image)
            if random.random() > 0.5:
                image, steering = self._random_rotation(image, steering)
            if random.random() > 0.5:
                image = self._random_contrast(image)
            if random.random() > 0.5:
                image = self._random_cutout(image)

        image = cv2.resize(image, (200, 66))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if self.transform:
            image = self.transform(image)

        steering_tensor = torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32)
        return image, steering_tensor

    def __getitem__(self, idx: int):
        image_path, steering = self.samples[idx]
        return self._process_image_and_steering(image_path, steering)


class CILCarlaDataset(CarlaDataset):
    """Single-frame steering dataset with speed and high-level command labels."""

    MAX_SPEED_KMH = 120.0

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        steering_correction: float = 0.2,
        is_training: bool = True,
        filter_stationary: bool = True,
        min_speed_kmh: float = 1.0,
        min_wp5_x_m: float = 3.0,
    ) -> None:
        raw_df = pd.read_csv(csv_file)
        if "command" not in raw_df.columns:
            import warnings

            warnings.warn(
                "CSV file does not contain a 'command' column; defaulting all "
                "commands to 0 (Follow Lane).",
                UserWarning,
                stacklevel=2,
            )
            raw_df["command"] = 0
        if "speed" not in raw_df.columns:
            import warnings

            warnings.warn(
                "CSV file does not contain a 'speed' column; defaulting all speeds to 0.0.",
                UserWarning,
                stacklevel=2,
            )
            raw_df["speed"] = 0.0

        if filter_stationary:
            raw_df = _filter_stationary_rows(
                raw_df,
                min_speed_kmh=float(min_speed_kmh),
                min_wp5_x_m=float(min_wp5_x_m),
            )

        self._raw_df = raw_df
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            transform=transform,
            steering_correction=steering_correction,
            is_training=is_training,
            filter_stationary=False,
            min_speed_kmh=min_speed_kmh,
            min_wp5_x_m=min_wp5_x_m,
        )
        self.data_df = self._raw_df

    def _prepare_data(self) -> None:
        _safe_print("Đang rà soát file ảnh từ 3 camera (CIL mode)...")
        max_steer = 1.0
        source_df = self._raw_df if hasattr(self, "_raw_df") else self.data_df
        self.samples = []

        for _, row in source_df.iterrows():
            steering = float(row["steering"])
            speed_norm = float(row.get("speed", 0.0)) / self.MAX_SPEED_KMH
            command = int(row.get("command", 0))
            image_id = _image_id(row["img_id"])
            configs = (
                ("images_center", 0.0),
                ("images_left", self.steering_correction),
                ("images_right", -self.steering_correction),
            )

            for sub_dir, correction in configs:
                if not self.is_training and sub_dir != "images_center":
                    continue
                image_path = os.path.join(self.root_dir, sub_dir, image_id)
                if os.path.exists(image_path):
                    self.samples.append(
                        (image_path, (steering + correction) / max_steer, speed_norm, command)
                    )

        _safe_print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")
        _safe_print(f"   Steering được chuẩn hóa bằng: MAX_STEER = {max_steer} (symmetric scaling)")
        if self.is_training:
            self._balance_steering_distribution()

    def _balance_steering_distribution(self, bins: int = 25, max_per_bin: int | None = None) -> None:
        if not self.samples:
            return

        steerings = [steering for _path, steering, _speed, _command in self.samples]
        hist, bin_edges = np.histogram(steerings, bins=bins)
        if max_per_bin is None:
            max_per_bin = max(1, int(np.mean(hist)))

        balanced_samples = []
        bin_counts = {index: 0 for index in range(bins)}
        random.shuffle(self.samples)

        for sample in self.samples:
            steering = sample[1]
            bin_idx = min(np.digitize(steering, bin_edges[1:-1]), bins - 1)
            if bin_counts[bin_idx] < max_per_bin:
                balanced_samples.append(sample)
                bin_counts[bin_idx] += 1

        _safe_print(f"Cân bằng dữ liệu: {len(self.samples)} -> {len(balanced_samples)} mẫu")
        self.samples = balanced_samples

    def __getitem__(self, idx: int):
        image_path, steering, speed_norm, command = self.samples[idx]
        image, steering_tensor = self._process_image_and_steering(image_path, steering)
        speed_tensor = torch.tensor(speed_norm, dtype=torch.float32)
        command_tensor = torch.tensor(command, dtype=torch.long)
        return image, steering_tensor, speed_tensor, command_tensor
