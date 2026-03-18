from __future__ import annotations

import csv
import logging
import queue
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


class DataCollector:
    """Collect synchronized center/left/right camera frames and vehicle states."""

    def __init__(
        self,
        output_dir: str | Path,
        enabled: bool = False,
        jpeg_quality: int = 95,
        save_every_n: int = 1,
        flush_every_n: int = 50,
        resize_width: int = 400,
        resize_height: int = 200,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.images_center_dir = self.output_dir / "images_center"
        self.images_left_dir = self.output_dir / "images_left"
        self.images_right_dir = self.output_dir / "images_right"
        self.csv_path = self.output_dir / "driving_log.csv"
        self.jpeg_quality = jpeg_quality
        self.save_every_n = max(1, save_every_n)
        self.flush_every_n = max(1, flush_every_n)
        self.resize_width = max(1, resize_width)
        self.resize_height = max(1, resize_height)
        self._csv_file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows_since_flush = 0
        self._next_img_index = 1
        self._sensor_queue: "queue.Queue[tuple[int, str, Any]]" = queue.Queue()
        self._pending_images: Dict[int, Dict[str, Any]] = {}
        self._pending_states: Dict[int, Dict[str, float | int]] = {}
        self._saved_frames = 0
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] if cv2 else []

    def start(self) -> None:
        if not self.enabled:
            return
        if cv2 is None:
            raise RuntimeError("opencv-python is required for data collection.")
        if np is None:
            raise RuntimeError("numpy is required for data collection.")
        self.images_center_dir.mkdir(parents=True, exist_ok=True)
        self.images_left_dir.mkdir(parents=True, exist_ok=True)
        self.images_right_dir.mkdir(parents=True, exist_ok=True)

        append = self.csv_path.exists()
        self._csv_file = self.csv_path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["img_id", "steering", "throttle", "brake", "speed", "command"],
        )
        if not append:
            self._writer.writeheader()

        # Continue image id numbering from existing JPG files in center folder.
        max_existing = 0
        for image_path in self.images_center_dir.glob("*.jpg"):
            stem = image_path.stem
            if stem.isdigit():
                max_existing = max(max_existing, int(stem))
        self._next_img_index = max_existing + 1

        self._rows_since_flush = 0
        self._saved_frames = 0
        self._pending_images.clear()
        self._pending_states.clear()
        while not self._sensor_queue.empty():
            try:
                self._sensor_queue.get_nowait()
            except queue.Empty:
                break

        logging.info(
            "Data collector started at %s (jpeg_q=%d, every=%d, resize=%dx%d)",
            self.output_dir,
            self.jpeg_quality,
            self.save_every_n,
            self.resize_width,
            self.resize_height,
        )

    @staticmethod
    def _camera_name_from_side(side: str) -> str:
        side_norm = side.strip().lower()
        if side_norm not in {"center", "left", "right"}:
            raise ValueError(f"Invalid camera side '{side}'. Expected center/left/right.")
        return side_norm

    def make_sensor_callback(self, camera_side: str):
        camera_name = self._camera_name_from_side(camera_side)

        def _callback(image) -> None:
            if not self.enabled or cv2 is None or np is None:
                return
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            bgra = array.reshape((image.height, image.width, 4))
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            self._sensor_queue.put((int(image.frame), camera_name, rgb))

        return _callback

    def add_vehicle_state(
        self,
        frame_id: int,
        steer: float,
        throttle: float,
        brake: float,
        speed_kmh: float,
        command: int = 0,
    ) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            raise RuntimeError("DataCollector.start() must be called before add_vehicle_state().")

        self._drain_sensor_queue()
        self._pending_states[frame_id] = {
            "steering": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "speed": float(speed_kmh),
            "command": int(command),
        }
        self._finalize_ready_frames(max_frame_id=frame_id)
        self._cleanup_old_frames(current_frame_id=frame_id)

    def _finalize_ready_frames(self, max_frame_id: int | None = None) -> None:
        common = set(self._pending_images.keys()) & set(self._pending_states.keys())
        if max_frame_id is not None:
            common = {fid for fid in common if fid <= max_frame_id}
        for fid in sorted(common):
            self._try_finalize_frame(fid)

    def _drain_sensor_queue(self) -> None:
        while True:
            try:
                frame_id, camera_name, rgb_image = self._sensor_queue.get_nowait()
            except queue.Empty:
                break
            frame_images = self._pending_images.setdefault(frame_id, {})
            frame_images[camera_name] = rgb_image

    def _resize_for_storage(self, rgb_image):
        return cv2.resize(
            rgb_image,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA,
        )

    def _save_jpeg(self, image_dir: Path, img_id: str, rgb_image) -> None:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        image_path = image_dir / f"{img_id}.jpg"
        cv2.imwrite(str(image_path), bgr, self._encode_params)

    def _try_finalize_frame(self, frame_id: int) -> None:
        images = self._pending_images.get(frame_id)
        state = self._pending_states.get(frame_id)
        if images is None or state is None:
            return
        if not all(side in images for side in ("center", "left", "right")):
            return

        # Save only configured sampling rate; still consume complete triplets for skipped frames.
        if frame_id % self.save_every_n != 0:
            self._pending_images.pop(frame_id, None)
            self._pending_states.pop(frame_id, None)
            return

        img_id = f"{self._next_img_index:08d}"
        self._next_img_index += 1

        center = self._resize_for_storage(images["center"])
        left = self._resize_for_storage(images["left"])
        right = self._resize_for_storage(images["right"])

        self._save_jpeg(self.images_center_dir, img_id, center)
        self._save_jpeg(self.images_left_dir, img_id, left)
        self._save_jpeg(self.images_right_dir, img_id, right)

        self._writer.writerow(
            {
                "img_id": img_id,
                "steering": round(float(state["steering"]), 5),
                "throttle": round(float(state["throttle"]), 4),
                "brake": round(float(state["brake"]), 4),
                "speed": round(float(state["speed"]), 3),
                "command": int(state["command"]),
            }
        )
        self._saved_frames += 1
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n and self._csv_file is not None:
            self._csv_file.flush()
            self._rows_since_flush = 0

        self._pending_images.pop(frame_id, None)
        self._pending_states.pop(frame_id, None)

    def _cleanup_old_frames(self, current_frame_id: int) -> None:
        # Protect memory if a frame was incomplete due to sensor glitches.
        keep_after = current_frame_id - 32
        old_image_frames = [f for f in self._pending_images.keys() if f < keep_after]
        old_state_frames = [f for f in self._pending_states.keys() if f < keep_after]
        for frame_id in old_image_frames:
            self._pending_images.pop(frame_id, None)
        for frame_id in old_state_frames:
            self._pending_states.pop(frame_id, None)

    def close(self) -> None:
        if self.enabled:
            self._drain_sensor_queue()
            # Attempt to flush any now-complete frames before closing.
            self._finalize_ready_frames()
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._writer = None
        logging.info("Data collector stopped. Saved %d synchronized samples.", self._saved_frames)
