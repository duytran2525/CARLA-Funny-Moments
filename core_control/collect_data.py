# [FILE: core_perception/collect_data.py]
import csv
import logging
import queue
import threading # 🌟 [CẬP NHẬT] Thư viện để chạy luồng ngầm
import time
from pathlib import Path
import cv2
import numpy as np

class DataCollector:
    # 🌟 [CẬP NHẬT] Thêm cờ use_multi_camera và các Queue đồng bộ
    def __init__(
        self,
        output_dir,
        enabled: bool = False,
        use_multi_camera: bool = False, # Cờ quyết định 1 hay 3 camera
        jpeg_quality: int = 95,
        save_every_n: int = 1,
    ) -> None:
        self.enabled = enabled
        self.use_multi_camera = use_multi_camera
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.csv_path = self.output_dir / "driving_log.csv"
        self.save_every_n = max(1, save_every_n)
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        
        # Hàng đợi (Queue) để giao tiếp giữa luồng CARLA và luồng Ghi file
        self._sensor_queue = queue.Queue()
        self._state_queue = queue.Queue()
        
        # Bộ nhớ đệm để ghép cặp dữ liệu theo tick
        self._pending_images = {}
        self._pending_states = {}
        
        # Điều khiển luồng ngầm
        self._is_running = False
        self._worker_thread = None

    def start(self) -> None:
        if not self.enabled: return
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        append = self.csv_path.exists()
        self._csv_file = self.csv_path.open("a" if append else "w", newline="", encoding="utf-8")
        
        # 🌟 [CẬP NHẬT] Header chuẩn 9 cột của Phase 2
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=['tick', 'image', 'image_left', 'image_right', 'steer', 'throttle', 'brake', 'speed_kmh', 'command']
        )
        if not append:
            self._writer.writeheader()

        # Khởi động Background Worker
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logging.info(f"Data Collector (Multi-Cam: {self.use_multi_camera}) started on Background Thread.")

    def make_sensor_callback(self, camera_side: str):
        """Callback gắn vào camera CARLA, đẩy ảnh thô vào Queue ngay lập tức."""
        def _callback(image) -> None:
            if not self.enabled: return
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            bgra = array.reshape((image.height, image.width, 4))
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            # 🌟 [CẬP NHẬT] Ném ảnh vào hàng đợi và return ngay
            self._sensor_queue.put((image.frame, camera_side, rgb))
        return _callback

    # 🌟 [CẬP NHẬT] Hàm add() return siêu tốc, không chặn luồng chính
    def add(self, tick: int, steer: float, throttle: float, brake: float, speed_kmh: float, command: int, image_center=None) -> None:
        if not self.enabled: return
        
        # Tương thích ngược: Nếu Agent truyền thẳng ảnh center vào hàm add()
        if image_center is not None:
            self._sensor_queue.put((tick, 'center', image_center))
            
        self._state_queue.put((tick, {
            'steer': steer, 'throttle': throttle, 'brake': brake,
            'speed_kmh': speed_kmh, 'command': command
        }))

    # 🌟 [CẬP NHẬT] TRÁI TIM CỦA C2: Luồng ngầm tự động ghép cặp và ghi ổ cứng
    def _worker_loop(self):
        required_cams = {"center", "left", "right"} if self.use_multi_camera else {"center"}
        
        while self._is_running or not self._state_queue.empty() or not self._sensor_queue.empty():
            # 1. Rút ảnh từ Queue
            while not self._sensor_queue.empty():
                tick, cam_name, img = self._sensor_queue.get_nowait()
                if tick not in self._pending_images: self._pending_images[tick] = {}
                self._pending_images[tick][cam_name] = img
                
            # 2. Rút trạng thái xe từ Queue
            while not self._state_queue.empty():
                tick, state = self._state_queue.get_nowait()
                self._pending_states[tick] = state
                
            # 3. Ghép cặp và Ghi file
            common_ticks = sorted(list(set(self._pending_images.keys()) & set(self._pending_states.keys())))
            for t in common_ticks:
                images = self._pending_images[t]
                state = self._pending_states[t]
                
                # Kiểm tra xem đã gom đủ số lượng camera yêu cầu chưa
                if required_cams.issubset(images.keys()):
                    if t % self.save_every_n == 0:
                        row_data = {'tick': t, **state, 'image': '', 'image_left': '', 'image_right': ''}
                        
                        # Lưu ảnh và ghi đường dẫn tương đối
                        for cam in required_cams:
                            img_name = f"{cam}_frame_{t:08d}.jpg"
                            cv2.imwrite(str(self.images_dir / img_name), cv2.cvtColor(images[cam], cv2.COLOR_RGB2BGR), self._encode_params)
                            
                            col_name = 'image' if cam == 'center' else f'image_{cam}'
                            row_data[col_name] = f"images/{img_name}"
                        
                        self._writer.writerow(row_data)
                    
                    # Xóa khỏi bộ đệm sau khi xử lý xong
                    del self._pending_images[t]
                    del self._pending_states[t]

            # Dọn rác các tick bị khuyết dữ liệu quá lâu (quá 30 tick)
            current_max = max(self._pending_states.keys()) if self._pending_states else 0
            for t in list(self._pending_images.keys()):
                if current_max - t > 30: del self._pending_images[t]
            for t in list(self._pending_states.keys()):
                if current_max - t > 30: del self._pending_states[t]

            time.sleep(0.005) # Tránh ăn 100% CPU của Core chạy ngầm

    def close(self) -> None:
        if self.enabled and self._is_running:
            self._is_running = False
            if self._worker_thread:
                self._worker_thread.join(timeout=2.0) # Đợi luồng ngầm ghi nốt file
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            logging.info("Background Data Collector stopped safely.")