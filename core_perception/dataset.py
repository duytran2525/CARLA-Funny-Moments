import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np

class WaypointCarlaDataset(Dataset):
    """
    Dataset V4.0 - Dự đoán Quỹ đạo Không-Thời gian (Spatio-Temporal Waypoints).
    Đã vá lỗi: Hỗ trợ WeightedRandomSampler, chuẩn hóa Shape [5, 2], tích hợp Transform.
    """
    def __init__(self, csv_file, root_dir, transform=None, is_training=True, geometric_offset=0.35):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.geometric_offset = geometric_offset
        
        self.data_df = pd.read_csv(csv_file)
        self.samples = []
        
        # [FIX A] Mảng lưu trữ Cờ cứu xe cho WeightedRandomSampler
        self.recovery_flags = [] 
        
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

    def __len__(self):
        return len(self.samples)

    # ... (CÁC HÀM AUGMENTATION GIỮ NGUYÊN NHƯ BẢN TRƯỚC) ...
    def _random_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + (random.random() - 0.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_shadow(self, image):
        h, w = image.shape[:2]
        pts = np.array([[random.randint(0, w), 0], [random.randint(0, w), h],
                        [random.randint(0, w), h], [random.randint(0, w), 0]], np.int32)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = np.zeros_like(hsv[:, :, 2])
        cv2.fillPoly(mask, [pts], 255)
        hsv[:, :, 2][mask == 255] = hsv[:, :, 2][mask == 255] * 0.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_blur(self, image):
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _random_noise(self, image):
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _random_contrast(self, image, low=0.7, high=1.3):
        factor = random.uniform(low, high)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _random_cutout(self, image, min_size_ratio=0.1, max_size_ratio=0.3):
        h, w, c = image.shape
        cut_h = int(h * random.uniform(min_size_ratio, max_size_ratio))
        cut_w = int(w * random.uniform(min_size_ratio, max_size_ratio))
        cy = random.randint(0, h)
        cx = random.randint(0, w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        image[y1:y2, x1:x2, :] = 0
        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        paths = sample['paths']
        waypoints = sample['waypoints'].copy()
        command = sample['command']
        recovery_flag = sample['recovery_flag']
        
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
                if do_brightness: img = self._random_brightness(img)
                if do_shadow: img = self._random_shadow(img)
                if do_blur: img = self._random_blur(img)
                if do_noise: img = self._random_noise(img)
                if do_contrast: img = self._random_contrast(img)
                if do_cutout: img = self._random_cutout(img)
                if do_flip:
                    img = cv2.flip(img, 1)
                    
            img = cv2.resize(img, (200, 66))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            images.append(img)
            
        stacked_img = np.concatenate(images, axis=-1)
        
        # Ép kiểu thành Tensor [9, 66, 200] và Scale [0, 1]
        stacked_tensor = torch.from_numpy(stacked_img).permute(2, 0, 1).float() / 255.0
        
        # [FIX C] Áp dụng Transform nếu có cấu hình từ file train_cnn.py
        if self.transform:
            stacked_tensor = self.transform(stacked_tensor)
        
        if do_flip:
            waypoints[:, 1] = -waypoints[:, 1]
            if command == 1: command = 2
            elif command == 2: command = 1
            
        # [FIX B] Giữ nguyên Shape [5, 2] cho PyTorch
        wp_tensor = torch.tensor(waypoints, dtype=torch.float32)
        cmd_tensor = torch.tensor(command, dtype=torch.long)
        rec_tensor = torch.tensor(recovery_flag, dtype=torch.float32)
        
        return stacked_tensor, wp_tensor, cmd_tensor, rec_tensor