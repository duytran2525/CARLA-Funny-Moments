from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CAMERA_DIR_ALIASES = {
    "center": ("images_center", "center"),
    "left": ("images_left", "left"),
    "right": ("images_right", "right"),
}


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


def _normalize_relative_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalize_image_filename(value: object) -> str:
    text = _normalize_relative_path(value)
    if not text or text.lower() == "nan":
        return ""
    leaf = Path(text).name
    return leaf if Path(leaf).suffix else _image_id(leaf)


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
    """Temporal CARLA dataset that supports both legacy and current collector layouts."""

    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        is_training=True,
        geometric_offset=0.35,
        filter_stationary=True,
        min_speed_kmh=1.0,
        min_wp5_x_m=3.0,
        include_side_cameras=True,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.is_training = bool(is_training)
        self.geometric_offset = float(geometric_offset)
        self.include_side_cameras = bool(include_side_cameras)
        self.samples: list[dict[str, object]] = []
        self.recovery_flags: list[int] = []
        self.csv_file = Path(csv_file).resolve()
        self._resolved_image_cache: dict[tuple[str, str, str], Optional[str]] = {}

        self.data_df = pd.read_csv(self.csv_file)
        self._inject_dataset_subdir_from_csv_path()

        if filter_stationary:
            self.data_df = _filter_stationary_rows(
                self.data_df,
                min_speed_kmh=float(min_speed_kmh),
                min_wp5_x_m=float(min_wp5_x_m),
            )

        self._prepare_data()

    def _inject_dataset_subdir_from_csv_path(self) -> None:
        if "dataset_subdir" in self.data_df.columns:
            return
        try:
            relative_parent = self.csv_file.parent.relative_to(self.root_dir)
        except ValueError:
            return
        relative_text = relative_parent.as_posix()
        if relative_text in {"", "."}:
            return
        self.data_df = self.data_df.copy()
        self.data_df["dataset_subdir"] = relative_text

    def _candidate_dataset_roots(self, row: dict[str, object]) -> list[Path]:
        candidates: list[Path] = []

        dataset_subdir = _normalize_relative_path(row.get("dataset_subdir", ""))
        if dataset_subdir:
            candidates.append((self.root_dir / dataset_subdir).resolve())

        center_ref = _normalize_relative_path(row.get("center_camera", ""))
        if center_ref:
            parts = [part for part in center_ref.split("/") if part]
            camera_dir_names = {name for names in CAMERA_DIR_ALIASES.values() for name in names}
            for marker in CAMERA_DIR_ALIASES["center"]:
                if marker not in parts:
                    continue
                marker_idx = parts.index(marker)
                prefix_parts = parts[:marker_idx]
                for start_idx in range(len(prefix_parts)):
                    candidate_rel = Path(*prefix_parts[start_idx:])
                    candidates.append((self.root_dir / candidate_rel).resolve())
            if parts:
                first = parts[0]
                if first not in camera_dir_names:
                    candidates.append((self.root_dir / first).resolve())

        candidates.append(self.root_dir)

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _filename_for_time(self, row: dict[str, object], time_key: str) -> str:
        if time_key == "t0":
            direct_keys = ("image_filename", "center_camera", "img_id")
        elif time_key == "tm03":
            direct_keys = ("image_filename_tm03", "img_id_tm03")
        else:
            direct_keys = ("image_filename_tm06", "img_id_tm06")

        for key in direct_keys:
            value = row.get(key)
            filename = _normalize_image_filename(value)
            if filename:
                return filename
        return ""

    def _resolve_image_path(
        self,
        base_dirs: list[Path],
        camera_type: str,
        filename: str,
    ) -> Optional[str]:
        if not filename:
            return None

        for base_dir in base_dirs:
            for dir_name in CAMERA_DIR_ALIASES[camera_type]:
                cache_key = (str(base_dir), dir_name, filename)
                cached = self._resolved_image_cache.get(cache_key)
                if cached is not None:
                    return cached

                candidate = base_dir / dir_name / filename
                if candidate.exists():
                    resolved = str(candidate)
                    self._resolved_image_cache[cache_key] = resolved
                    return resolved
                self._resolved_image_cache[cache_key] = None

        if camera_type == "center":
            center_ref = _normalize_relative_path(filename)
            direct_candidate = Path(center_ref)
            if direct_candidate.is_absolute() and direct_candidate.exists():
                return str(direct_candidate)

        return None

    def _resolve_triplet_paths(self, row: dict[str, object], camera_type: str) -> Optional[tuple[str, str, str]]:
        base_dirs = self._candidate_dataset_roots(row)
        filenames = {
            "t0": self._filename_for_time(row, "t0"),
            "tm03": self._filename_for_time(row, "tm03"),
            "tm06": self._filename_for_time(row, "tm06"),
        }
        if not all(filenames.values()):
            return None

        path_t0 = self._resolve_image_path(base_dirs, camera_type, filenames["t0"])
        path_tm03 = self._resolve_image_path(base_dirs, camera_type, filenames["tm03"])
        path_tm06 = self._resolve_image_path(base_dirs, camera_type, filenames["tm06"])

        if path_t0 and path_tm03 and path_tm06:
            return path_t0, path_tm03, path_tm06
        return None

    @staticmethod
    def _extract_waypoints(row: dict[str, object]) -> Optional[np.ndarray]:
        keys = (
            "wp_1_x",
            "wp_1_y",
            "wp_2_x",
            "wp_2_y",
            "wp_3_x",
            "wp_3_y",
            "wp_4_x",
            "wp_4_y",
            "wp_5_x",
            "wp_5_y",
        )
        try:
            values = [float(row[key]) for key in keys]
        except (KeyError, TypeError, ValueError):
            return None
        return np.asarray(values, dtype=np.float32).reshape(5, 2)

    def _prepare_data(self) -> None:
        _safe_print(f"Preparing temporal dataset from {len(self.data_df)} rows...")

        missing_triplets = 0
        invalid_rows = 0
        camera_configs = [("center", 0.0)]
        if self.is_training and self.include_side_cameras:
            camera_configs.extend(
                [
                    ("left", self.geometric_offset),
                    ("right", -self.geometric_offset),
                ]
            )

        for row_idx, row in enumerate(self.data_df.to_dict("records")):
            waypoints = self._extract_waypoints(row)
            if waypoints is None:
                invalid_rows += 1
                continue

            try:
                command = int(row.get("command", 0))
            except (TypeError, ValueError):
                command = 0
            try:
                recovery_flag = float(row.get("recovery_flag", 0.0))
            except (TypeError, ValueError):
                recovery_flag = 0.0

            for camera_type, lateral_offset in camera_configs:
                paths = self._resolve_triplet_paths(row, camera_type)
                if paths is None:
                    missing_triplets += 1
                    continue

                wp_cam = waypoints.copy()
                wp_cam[:, 1] += float(lateral_offset)
                self.samples.append(
                    {
                        "paths": paths,
                        "waypoints": wp_cam,
                        "command": command,
                        "recovery_flag": recovery_flag,
                        "row_idx": int(row_idx),
                        "camera_type": camera_type,
                        "speed": float(row.get("speed", 0.0)),
                    }
                )
                self.recovery_flags.append(int(recovery_flag))

        _safe_print(f"Loaded {len(self.samples)} temporal samples.")
        if missing_triplets > 0:
            _safe_print(f"Skipped {missing_triplets} camera triplets with missing files.")
        if invalid_rows > 0:
            _safe_print(f"Skipped {invalid_rows} rows with invalid waypoint metadata.")
        if not self.samples:
            _safe_print(
                "Dataset is empty. Expected either legacy columns "
                "('center_camera', 'img_id*') or current collector columns "
                "('image_filename*', optional 'dataset_subdir')."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get_recovery_flags(self) -> Iterable[int]:
        return list(self.recovery_flags)

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
        speed = float(sample["speed"])
        speed_tensor = torch.tensor(speed / 120.0, dtype=torch.float32)

        images = []
        do_flip = self.is_training and random.random() > 0.5
        do_brightness = self.is_training and random.random() > 0.5
        do_shadow = self.is_training and random.random() > 0.5
        do_blur = self.is_training and random.random() > 0.5
        do_noise = self.is_training and random.random() > 0.7
        do_contrast = self.is_training and random.random() > 0.5
        do_cutout = self.is_training and random.random() > 0.5

        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Failed to read dataset image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, _w = img.shape[:2]
            img = img[int(h * 0.45) :, :, :]

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
            image = self.transform(image)

        steering = torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32)

        return image, steering
    
    def __getitem__(self, idx):
        img_path, steering = self.samples[idx]
        return self._process_image_and_steering(img_path, steering)


class CILCarlaDataset(CarlaDataset):
    """
    Dataset Phase 2 - Conditional Imitation Learning (CIL).

    Kế thừa ``CarlaDataset`` và mở rộng để trả về thêm **vận tốc đã chuẩn
    hoá** và **lệnh điều hướng cấp cao** bên cạnh ảnh và nhãn góc lái.

    Sự khác biệt so với ``CarlaDataset``
    -------------------------------------
    * ``__getitem__`` trả về ``(image, steering, speed, command)`` thay vì
      ``(image, steering)``.
    * ``_prepare_data`` lưu ``(img_path, steering, speed_norm, command)``
      trong mỗi phần tử của ``self.samples``.
    * ``_balance_steering_distribution`` được ghi đè để làm việc với
      tuple 4 phần tử thay vì 2.
    * Vận tốc được chuẩn hoá về [0, 1] bằng cách chia cho
      ``MAX_SPEED_KMH = 120``.

    Định dạng CSV
    -------------
    CSV phải có 6 cột: ``img_id, steering, throttle, brake, speed, command``.
    Nếu thiếu ``command`` hoặc ``speed``, một cảnh báo sẽ được phát và giá
    trị mặc định (0) được điền vào - cho phép tải trọng số Phase 1 mà không
    cần sửa file CSV.

    Lệnh điều hướng
    ---------------
    =====================  =====
    Lệnh                   Index
    =====================  =====
    Follow Lane            0
    Turn Left              1
    Turn Right             2
    Go Straight            3
    =====================  =====
    """

    # Maximum speed (km/h) used for normalisation. CARLA's default speed limit on
    # urban roads is 30-90 km/h; 120 km/h provides headroom for highway scenarios
    # while keeping normalised values comfortably in [0, 1].
    MAX_SPEED_KMH = 120.0

    def __init__(self, csv_file, root_dir, transform=None,
                 steering_correction=0.2, is_training=True):
        """
        Khởi tạo CILCarlaDataset.

        Trước khi gọi ``super().__init__``, đọc CSV bằng pandas để kiểm tra
        sự hiện diện của các cột ``speed`` và ``command``:

        * Nếu thiếu ``command`` → cảnh báo + điền giá trị mặc định ``0``
          (Follow Lane).
        * Nếu thiếu ``speed``   → cảnh báo + điền giá trị mặc định ``0.0``.

        DataFrame đã xử lý được lưu vào ``self._raw_df`` để phương thức
        ``_prepare_data`` (được gọi bên trong ``super().__init__``) có thể
        truy cập các cột bổ sung này.

        Parameters
        ----------
        csv_file : str
            Đường dẫn file CSV driving log (Phase 2).
        root_dir : str
            Thư mục gốc chứa ba thư mục con camera.
        transform : callable, optional
            Transform tuỳ chọn áp lên ảnh đã xử lý.
        steering_correction : float, optional
            Bù góc lái cho camera trái/phải.  Mặc định ``0.2``.
        is_training : bool, optional
            Bật/tắt augmentation và sử dụng 3 camera.  Mặc định ``True``.
        """
        # Read the extended CSV *before* calling super().__init__ so that the
        # speed/command columns are available when _prepare_data runs.
        raw_df = pd.read_csv(csv_file)
        if 'command' not in raw_df.columns:
            import warnings
            warnings.warn(
                "CSV file does not contain a 'command' column - defaulting all "
                "commands to 0 (Follow Lane). Add a 'command' column to enable "
                "full CIL training for Phase 2.",
                UserWarning, stacklevel=2,
            )
            raw_df['command'] = 0
        if 'speed' not in raw_df.columns:
            import warnings
            warnings.warn(
                "CSV file does not contain a 'speed' column - defaulting all "
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

    def _prepare_data(self):
        """
        Xây dựng ``self.samples`` dưới dạng tuple 4 phần tử
        ``(img_path, steering, speed_norm, command)``.

        Tương tự ``CarlaDataset._prepare_data`` nhưng đọc thêm ``speed`` và
        ``command`` từ ``self._raw_df``.  Vận tốc được chuẩn hoá ngay tại
        đây bằng cách chia cho ``MAX_SPEED_KMH``.

        Thứ tự ưu tiên nguồn dữ liệu:

        1. ``self._raw_df`` - DataFrame đã có cột ``speed`` và ``command``
           (được tạo trong ``__init__`` của class này).
        2. ``self.data_df`` - fallback nếu ``_raw_df`` chưa được gán (không
           nên xảy ra trong luồng bình thường).
        """
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
            img_id = img_id_str.zfill(8) + '.jpg'

            configs = [
                ('images_center', 0),
                ('images_left',  self.steering_correction),
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

    def _balance_steering_distribution(self, bins=25, max_per_bin=None):
        """
        Cân bằng phân phối steering cho các mẫu 4-tuple của CIL dataset.

        Ghi đè phiên bản cha vì mỗi phần tử trong ``self.samples`` là
        tuple ``(img_path, steering, speed_norm, command)`` thay vì
        tuple 2 phần tử.

        Thuật toán giống hệt ``CarlaDataset._balance_steering_distribution``:
        histogram → tính ``max_per_bin`` → shuffle → giữ lại tối đa
        ``max_per_bin`` mẫu mỗi bin.

        Parameters
        ----------
        bins : int, optional
            Số bin phân phối steering.  Mặc định ``25``.
        max_per_bin : int, optional
            Ngưỡng tối đa mẫu mỗi bin.  Nếu ``None`` dùng
            ``int(mean(histogram))``.
        """
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

    def __getitem__(self, idx):
        """
        Trả về một mẫu dữ liệu CIL đã qua toàn bộ pipeline xử lý.

        Pipeline xử lý ảnh giống hệt ``CarlaDataset.__getitem__`` (crop,
        augmentation, resize 200x66, chuyển YUV, transform).  Ngoài ra còn
        đóng gói **vận tốc** và **lệnh điều hướng** thành tensor.

        Parameters
        ----------
        idx : int
            Chỉ số mẫu trong ``self.samples``.

        Returns
        -------
        image : torch.Tensor hoặc np.ndarray
            Ảnh đã xử lý, shape ``(3, 66, 200)`` nếu dùng ``ToTensor``.
        steering : torch.Tensor
            Góc lái scalar, dtype ``float32``, trong [-1, 1].
        speed : torch.Tensor
            Vận tốc đã chuẩn hoá, scalar, dtype ``float32``, trong [0, 1].
        command : torch.Tensor
            Lệnh điều hướng, scalar, dtype ``long`` (int64).
            Giá trị hợp lệ: 0 (Follow Lane), 1 (Turn Left),
            2 (Turn Right), 3 (Go Straight).
        """
        # 1. Lấy thông tin từ tuple 4 phần tử
        img_path, steering, speed_norm, command = self.samples[idx]

        # 2. Nhờ hàm của lớp cha xử lý giùm phần Ảnh và Vô lăng
        image, steering_tensor = self._process_image_and_steering(img_path, steering)

        # 3. Chỉ tập trung xử lý phần mở rộng (Speed và Command)
        speed = torch.tensor(speed_norm, dtype=torch.float32)
        command = torch.tensor(command, dtype=torch.long)

        return image, steering_tensor, speed, command
