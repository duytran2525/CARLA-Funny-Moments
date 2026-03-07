import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, steering_correction=0.2, is_training = True):
        self.data_df = pd.read_csv(csv_file, names=['img_id', 'steering', 'throttle', 'brake', 'speed'])
        self.root_dir = root_dir
        self.transform = transform
        self.steering_correction = steering_correction
        
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        print("Đang rà soát file ảnh từ 3 camera...")
        for _, row in self.data_df.iterrows():
            steering = float(row['steering'])
            img_id = str(row['img_id']).strip()
            if not img_id.endswith('.png'): img_id += '.png'

            configs = [
                ('img_center', 0),
                ('img_left', self.steering_correction),
                ('img_right', -self.steering_correction)
            ]

            for sub_dir, correction in configs:
                img_path = os.path.join(self.root_dir, sub_dir, img_id)
                if os.path.exists(img_path):
                    self.samples.append((img_path, steering + correction))
        
        print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")

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

        image = cv2.resize(image, (200, 66))
        
        if self.transform:
            image = self.transform(image)
        
        steering = torch.tensor(steering, dtype=torch.float32)
        
        return image, steering