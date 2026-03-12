import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, steering_correction=0.2, is_training = True):
        self.data_df = pd.read_csv(csv_file, header=0, names=['img_id', 'steering', 'throttle', 'brake', 'speed'])
        self.root_dir = root_dir
        self.transform = transform
        self.steering_correction = steering_correction
        self.is_training = is_training
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        print("Đang rà soát file ảnh từ 3 camera...")
        for _, row in self.data_df.iterrows():
            steering = float(row['steering'])
            img_id_str = str(row['img_id']).strip()
            if '.' in img_id_str: 
                img_id_str = img_id_str.split('.')[0] # Bỏ phần thập phân nếu có
            img_id = img_id_str.zfill(8) + '.png'

            configs = [
                ('images_center', 0),
                ('images_left', self.steering_correction),
                ('images_right', -self.steering_correction)
            ]

            for sub_dir, correction in configs:
                if not self.is_training and sub_dir != 'images_center':
                    continue
                img_path = os.path.join(self.root_dir, sub_dir, img_id)
                if os.path.exists(img_path):
                    self.samples.append((img_path, steering + correction))
        
        print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")
        if self.is_training:
            self._balance_steering_distribution()

    def _balance_steering_distribution(self, bins=25, max_per_bin=None):
        """
        Giảm số lượng mẫu đi thẳng (steering ≈ 0) để cân bằng dữ liệu.
        Ý tưởng: chia steering thành các bin, giới hạn mỗi bin tối đa = trung bình.
        """
        steerings = [s for _, s in self.samples]
        hist, bin_edges = np.histogram(steerings, bins=bins)

        if max_per_bin is None:
            max_per_bin = int(np.mean(hist))

        balanced_samples = []
        bin_counts = {i: 0 for i in range(bins)}

        random.shuffle(self.samples)

        for img_path, steering in self.samples:
            bin_idx = min(np.digitize(steering, bin_edges[1:-1]), bins - 1)
            if bin_counts[bin_idx] < max_per_bin:
                balanced_samples.append((img_path, steering))
                bin_counts[bin_idx] += 1

        print(f"Cân bằng dữ liệu: {len(self.samples)} → {len(balanced_samples)} mẫu")
        self.samples = balanced_samples
        
        print(f"Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # --- CÁC HÀM AUGMENTATION ---

    def _random_brightness(self, image):
        """Thay đổi độ sáng ngẫu nhiên bằng cách chuyển sang hệ màu HSV"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Tỉ lệ sáng từ 0.5 (tối đi một nửa) đến 1.5 (sáng rực)
        ratio = 1.0 + (random.random() - 0.5) 
        hsv[:,:,2] = np.clip(hsv[:,:,2] * ratio, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_shadow(self, image):
        """Tạo bóng râm ngẫu nhiên trên mặt đường"""
        h, w = image.shape[:2]
        # Tạo 4 điểm ngẫu nhiên để vẽ một đa giác (polygon) làm bóng râm
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        x3, y3 = random.randint(0, w), h
        x4, y4 = random.randint(0, w), 0
        
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = np.zeros_like(hsv[:, :, 2])
        cv2.fillPoly(mask, [pts], 255)
        
        # Giảm độ sáng (kênh V) tại vùng có mask xuống 50%
        hsv[:, :, 2][mask == 255] = hsv[:, :, 2][mask == 255] * 0.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_translate(self, image, steering, range_x=20, range_y=10):
        """Dịch chuyển ảnh ngẫu nhiên và bù trừ góc lái tương ứng"""
        h, w = image.shape[:2]
        trans_x = range_x * (random.random() - 0.5)
        trans_y = range_y * (random.random() - 0.5)
        
        # Nếu xe bị lệch sang phải (trans_x > 0), phải đánh lái thêm về bên phải để đưa xe về giữa
        steering += trans_x * 0.004 
        
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        image = cv2.warpAffine(image, trans_m, (w, h))
        return image, steering
    # GAUSSIAN BLUR
    def _random_blur(self, image):
        """Thêm Gaussian blur ngẫu nhiên để mô phỏng mờ camera"""
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    # =====================================================
    # [MỚI #2] GAUSSIAN NOISE — robust với nhiễu sensor
    # =====================================================
    def _random_noise(self, image):
        """Thêm nhiễu Gaussian ngẫu nhiên"""
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image

    # =====================================================
    # [MỚI #5] RANDOM ROTATION — robust với mặt đường nghiêng
    # =====================================================
    def _random_rotation(self, image, steering, max_angle=5):
        """Xoay ảnh nhẹ ngẫu nhiên và bù steering tương ứng"""
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        # Xoay ảnh theo chiều kim đồng hồ (angle âm) ≈ xe nghiêng phải → cần lái trái
        steering -= angle * 0.01
        return image, steering

    # =====================================================
    # [MỚI #8] RANDOM CONTRAST — robust với điều kiện thời tiết
    # =====================================================
    def _random_contrast(self, image, low=0.7, high=1.3):
        """Thay đổi contrast ngẫu nhiên"""
        factor = random.uniform(low, high)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return image
    def __getitem__(self, idx):
        img_path, steering = self.samples[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width, _ = image.shape
        image = image[int(height*0.45):, :, :]

        # 2. DATA AUGMENTATION NGẪU NHIÊN
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
            # --- Augmentation MỚI ---
            if random.random() > 0.5:                                  # [MỚI #2]
                image = self._random_blur(image)
            if random.random() > 0.7:                                  # [MỚI #3] xác suất thấp hơn
                image = self._random_noise(image)
            if random.random() > 0.5:                                  # [MỚI #5]
                image, steering = self._random_rotation(image, steering)
            if random.random() > 0.5:                                  # [MỚI #8]
                image = self._random_contrast(image)

        image = cv2.resize(image, (200, 66))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if self.transform:
            image = self.transform(image)
        
        steering = torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32)
        
        return image, steering


class CILCarlaDataset(CarlaDataset):
    """
    Phase 2 Dataset: Conditional Imitation Learning.
    Extends CarlaDataset to also return normalised speed and high-level command
    alongside the image and steering label.

    The driving_log CSV must contain the columns:
        img_id, steering, throttle, brake, speed, command
    where `command` is an integer:
        0 = Follow Lane  (no intersection)
        1 = Turn Left
        2 = Turn Right
        3 = Go Straight  (at intersection)

    If the CSV does not contain a `command` column, command defaults to 0
    (Follow Lane) for every sample so Phase-1 weights can still be loaded.
    """

    # Maximum speed (km/h) used for normalisation. CARLA's default speed limit on
    # urban roads is 30–90 km/h; 120 km/h provides headroom for highway scenarios
    # while keeping normalised values comfortably in [0, 1].
    MAX_SPEED_KMH = 120.0

    def __init__(self, csv_file, root_dir, transform=None,
                 steering_correction=0.2, is_training=True):
        # Read the extended CSV *before* calling super().__init__ so that the
        # speed/command columns are available when _prepare_data runs.
        raw_df = pd.read_csv(csv_file)
        if 'command' not in raw_df.columns:
            import warnings
            warnings.warn(
                "CSV file does not contain a 'command' column – defaulting all "
                "commands to 0 (Follow Lane). Add a 'command' column to enable "
                "full CIL training for Phase 2.",
                UserWarning, stacklevel=2,
            )
            raw_df['command'] = 0
        if 'speed' not in raw_df.columns:
            import warnings
            warnings.warn(
                "CSV file does not contain a 'speed' column – defaulting all "
                "speeds to 0.0. Add a 'speed' column to enable speed-conditioned "
                "CIL training for Phase 2.",
                UserWarning, stacklevel=2,
            )
            raw_df['speed'] = 0.0

        # Persist as an attribute that _prepare_data (inside super().__init__)
        # can also reference.
        self._raw_df = raw_df

        super().__init__(csv_file=csv_file, root_dir=root_dir,
                         transform=transform,
                         steering_correction=steering_correction,
                         is_training=is_training)

    # ------------------------------------------------------------------
    # Override _prepare_data to capture speed/command alongside each sample
    # ------------------------------------------------------------------
    def _prepare_data(self):
        """Build self.samples as (img_path, steering, speed_norm, command)."""
        print("Đang rà soát file ảnh từ 3 camera (CIL mode)...")

        # Use the raw df that already has 'speed' and 'command'
        source_df = self._raw_df if hasattr(self, '_raw_df') else self.data_df

        self.samples = []  # reset in case called a second time
        for _, row in source_df.iterrows():
            steering = float(row['steering'])
            speed_norm = float(row.get('speed', 0.0)) / self.MAX_SPEED_KMH
            command = int(row.get('command', 0))

            img_id_str = str(row['img_id']).strip()
            if '.' in img_id_str:
                img_id_str = img_id_str.split('.')[0]
            img_id = img_id_str.zfill(8) + '.png'

            configs = [
                ('images_center', 0),
                ('images_left', self.steering_correction),
                ('images_right', -self.steering_correction),
            ]

            for sub_dir, correction in configs:
                if not self.is_training and sub_dir != 'images_center':
                    continue
                img_path = os.path.join(self.root_dir, sub_dir, img_id)
                if os.path.exists(img_path):
                    self.samples.append(
                        (img_path, steering + correction, speed_norm, command)
                    )

        print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")
        if self.is_training:
            self._balance_steering_distribution()

    # ------------------------------------------------------------------
    # Override _balance_steering_distribution to work with 4-tuple samples
    # ------------------------------------------------------------------
    def _balance_steering_distribution(self, bins=25, max_per_bin=None):
        steerings = [s for _, s, _, _ in self.samples]
        hist, bin_edges = np.histogram(steerings, bins=bins)

        if max_per_bin is None:
            max_per_bin = int(np.mean(hist))

        balanced_samples = []
        bin_counts = {i: 0 for i in range(bins)}

        random.shuffle(self.samples)

        for sample in self.samples:
            steering = sample[1]
            bin_idx = min(np.digitize(steering, bin_edges[1:-1]), bins - 1)
            if bin_counts[bin_idx] < max_per_bin:
                balanced_samples.append(sample)
                bin_counts[bin_idx] += 1

        print(f"Cân bằng dữ liệu: {len(self.samples)} → {len(balanced_samples)} mẫu")
        self.samples = balanced_samples
        print(f"Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")

    # ------------------------------------------------------------------
    # Override __getitem__ to return (image, steering, speed, command)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path, steering, speed_norm, command = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape
        image = image[int(height * 0.45):, :, :]

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

        image = cv2.resize(image, (200, 66))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if self.transform:
            image = self.transform(image)

        steering = torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32)
        speed = torch.tensor(speed_norm, dtype=torch.float32)
        command = torch.tensor(command, dtype=torch.long)

        return image, steering, speed, command


