from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class DataRecord:
    tick: int
    image_name: str
    steer: float
    throttle: float
    brake: float
    speed_kmh: float


class DataCollector:
    """Save driving frames and controls for model training/debugging."""

    def __init__(
        self,
        output_dir: str | Path,
        enabled: bool = False,
        jpeg_quality: int = 85,
        save_every_n: int = 1,
        flush_every_n: int = 50,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.csv_path = self.output_dir / "driving_log.csv"
        self.jpeg_quality = jpeg_quality
        self.save_every_n = max(1, save_every_n)
        self.flush_every_n = max(1, flush_every_n)
        self._csv_file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows_since_flush = 0
        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] if cv2 else []

    def start(self) -> None:
        if not self.enabled:
            return
        if cv2 is None:
            raise RuntimeError("opencv-python is required for data collection.")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["tick", "image", "steer", "throttle", "brake", "speed_kmh"],
        )
        self._writer.writeheader()
        self._rows_since_flush = 0
        logging.info("Data collector started at %s (jpeg_q=%d, every=%d)",
                     self.output_dir, self.jpeg_quality, self.save_every_n)

    def add(
        self,
        tick: int,
        rgb_frame,
        steer: float,
        throttle: float,
        brake: float,
        speed_kmh: float,
    ) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            raise RuntimeError("DataCollector.start() must be called before add().")
        if tick % self.save_every_n != 0:
            return

        image_name = f"frame_{tick:06d}.jpg"
        image_path = self.images_dir / image_name
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), bgr, self._encode_params)

        self._writer.writerow(
            {
                "tick": tick,
                "image": f"images/{image_name}",
                "steer": round(steer, 5),
                "throttle": round(throttle, 4),
                "brake": round(brake, 4),
                "speed_kmh": round(speed_kmh, 2),
            }
        )
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n:
            self._csv_file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._writer = None
            logging.info("Data collector stopped.")
