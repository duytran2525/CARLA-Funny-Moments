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
