from __future__ import annotations

import csv
import logging
import math
import queue
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from core_control.sync_data import build_synchronized_data, normalize_camera_side

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


BUFFER_MAXLEN = 150
HISTORY_LOOKUPS: tuple[tuple[str, float], ...] = (
    ("tm06", -0.6),
    ("tm03", -0.3),
)
FUTURE_OFFSETS_S: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
HISTORY_NEAREST_TOLERANCE_S = 0.05
EPSILON_S = 1e-6
TURN_DENSE_STRIDE_DIVISOR = 5


def _build_csv_fieldnames() -> list[str]:
    fieldnames = [
        "frame_id",
        "timestamp",
        "img_id",
        "img_id_tm06",
        "img_id_tm03",
        "frame_id_tm06",
        "frame_id_tm03",
        "timestamp_tm06",
        "timestamp_tm03",
        "speed",
        "command",
        "steering",
        "throttle",
        "brake",
        "pitch",
        "roll",
        "yaw",
        "x",
        "y",
        "z",
        "has_crash",
        "recovery_flag",
    ]
    for index, _ in enumerate(FUTURE_OFFSETS_S, start=1):
        fieldnames.extend((f"wp_{index}_x", f"wp_{index}_y"))
    return fieldnames


@dataclass(slots=True)
class FrameRecord:
    frame_id: int
    timestamp: float
    steering: float
    throttle: float
    brake: float
    speed: float
    command: int
    pitch: float
    roll: float
    yaw: float
    x: float
    y: float
    z: float
    has_crash: bool
    is_recovering: bool
    images: Dict[str, Any]


class DataCollector:
    """
    Temporal dataset collector for pixel-to-waypoint learning.

    A frame is only written after enough future telemetry exists to build
    waypoint labels at T+0.5, 1.0, 1.5, 2.0, and 2.5 seconds.
    """

    CSV_FIELDNAMES = _build_csv_fieldnames()

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
        self.jpeg_quality = int(jpeg_quality)
        self.save_every_n = max(1, int(save_every_n))
        self.flush_every_n = max(1, int(flush_every_n))
        self.resize_width = max(1, int(resize_width))
        self.resize_height = max(1, int(resize_height))
        self.turn_save_every_n = max(1, self.save_every_n // TURN_DENSE_STRIDE_DIVISOR)

        self._csv_file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows_since_flush = 0
        self._next_img_index = 1

        self._sensor_queue: "queue.Queue[tuple[int, str, Any]]" = queue.Queue()
        self._pending_images: Dict[int, Dict[str, Any]] = {}
        self._pending_states: Dict[int, Dict[str, Any]] = {}

        self._frame_buffer: deque[FrameRecord] = deque(maxlen=BUFFER_MAXLEN)
        self._frame_lookup: Dict[int, FrameRecord] = {}
        self._target_frame_ids: deque[int] = deque()
        self._saved_image_ids: Dict[int, str] = {}

        self._saved_samples = 0
        self._dropped_startup = 0
        self._dropped_crash = 0
        self._dropped_geometry = 0
        self._dropped_evicted = 0
        self._dropped_unresolved = 0
        self._saved_command_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        self._encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality] if cv2 else []

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
        if append:
            self._validate_existing_csv_schema()

        self._csv_file = self.csv_path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDNAMES)
        if not append:
            self._writer.writeheader()

        self._next_img_index = self._resolve_next_image_index()
        self._rows_since_flush = 0

        self._pending_images.clear()
        self._pending_states.clear()
        self._frame_buffer.clear()
        self._frame_lookup.clear()
        self._target_frame_ids.clear()
        self._saved_image_ids.clear()
        self._saved_samples = 0
        self._dropped_startup = 0
        self._dropped_crash = 0
        self._dropped_geometry = 0
        self._dropped_evicted = 0
        self._dropped_unresolved = 0
        self._saved_command_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        while not self._sensor_queue.empty():
            try:
                self._sensor_queue.get_nowait()
            except queue.Empty:
                break

        logging.info(
            (
                "Data collector started at %s "
                "(buffer=%d, every=%d, turn_every=%d, resize=%dx%d, future_horizon=%.1fs)"
            ),
            self.output_dir,
            BUFFER_MAXLEN,
            self.save_every_n,
            self.turn_save_every_n,
            self.resize_width,
            self.resize_height,
            FUTURE_OFFSETS_S[-1],
        )

    def _validate_existing_csv_schema(self) -> None:
        if not self.csv_path.exists():
            return
        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fieldnames = list(reader.fieldnames or [])
        if existing_fieldnames != self.CSV_FIELDNAMES:
            raise RuntimeError(
                "Existing driving_log.csv schema does not match DataCollector V2. "
                f"Expected {self.CSV_FIELDNAMES}, found {existing_fieldnames}. "
                "Use a fresh output directory or remove the old CSV."
            )

    def _resolve_next_image_index(self) -> int:
        max_existing = 0
        for image_path in self.images_center_dir.glob("*.jpg"):
            stem = image_path.stem
            if stem.isdigit():
                max_existing = max(max_existing, int(stem))
        return max_existing + 1

    @staticmethod
    def _camera_name_from_side(side: str) -> str:
        return normalize_camera_side(side)

    def get_synchronized_data(self, frame_id: int) -> Optional[Dict[str, Any]]:
        self._drain_sensor_queue()
        return build_synchronized_data(
            frame_id=frame_id,
            pending_images=self._pending_images,
            pending_states=self._pending_states,
        )

    def make_sensor_callback(self, camera_side: str):
        camera_name = self._camera_name_from_side(camera_side)

        def _callback(image) -> None:
            if not self.enabled or cv2 is None or np is None:
                return
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            bgra = array.reshape((image.height, image.width, 4))
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            rgb = self._resize_for_storage(rgb)
            self._sensor_queue.put((int(image.frame), camera_name, rgb))

        return _callback

    def add_vehicle_state(
        self,
        frame_id: int,
        timestamp: float,
        steer: float,
        throttle: float,
        brake: float,
        speed_kmh: float,
        x: float,
        y: float,
        z: float,
        has_crash: bool,
        is_recovering: bool,
        command: int = 0,
        pitch: float = 0.0,
        roll: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            raise RuntimeError("DataCollector.start() must be called before add_vehicle_state().")

        self._drain_sensor_queue()
        self._pending_states[int(frame_id)] = {
            "timestamp": float(timestamp),
            "steering": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "speed": float(speed_kmh),
            "command": int(command),
            "pitch": float(pitch),
            "roll": float(roll),
            "yaw": float(yaw),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "has_crash": bool(has_crash),
            "is_recovering": bool(is_recovering),
        }
        self._ingest_ready_frames(max_frame_id=int(frame_id))
        self._cleanup_pending(current_frame_id=int(frame_id))

    def _resize_for_storage(self, rgb_image: Any) -> Any:
        return cv2.resize(
            rgb_image,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA,
        )

    def _drain_sensor_queue(self) -> None:
        while True:
            try:
                frame_id, camera_name, rgb_image = self._sensor_queue.get_nowait()
            except queue.Empty:
                break
            frame_images = self._pending_images.setdefault(int(frame_id), {})
            frame_images[camera_name] = rgb_image

    def _ingest_ready_frames(self, max_frame_id: int | None = None) -> None:
        common = set(self._pending_images.keys()) & set(self._pending_states.keys())
        if max_frame_id is not None:
            common = {frame_id for frame_id in common if frame_id <= max_frame_id}
        for frame_id in sorted(common):
            synchronized = self.get_synchronized_data(frame_id)
            if synchronized is None:
                continue
            state = synchronized["state"]
            record = FrameRecord(
                frame_id=int(frame_id),
                timestamp=float(state["timestamp"]),
                steering=float(state["steering"]),
                throttle=float(state["throttle"]),
                brake=float(state["brake"]),
                speed=float(state["speed"]),
                command=int(state["command"]),
                pitch=float(state["pitch"]),
                roll=float(state["roll"]),
                yaw=float(state["yaw"]),
                x=float(state["x"]),
                y=float(state["y"]),
                z=float(state["z"]),
                has_crash=bool(state["has_crash"]),
                is_recovering=bool(state["is_recovering"]),
                images=dict(synchronized["images"]),
            )
            self._append_frame_record(record)
            self._pending_images.pop(frame_id, None)
            self._pending_states.pop(frame_id, None)

        self._process_ready_targets()

    def _append_frame_record(self, record: FrameRecord) -> None:
        if len(self._frame_buffer) == self._frame_buffer.maxlen and self._frame_buffer:
            evicted = self._frame_buffer[0]
            self._frame_lookup.pop(evicted.frame_id, None)
        self._frame_buffer.append(record)
        self._frame_lookup[record.frame_id] = record
        if self._should_mark_target_frame(record):
            self._target_frame_ids.append(record.frame_id)

    def _should_mark_target_frame(self, record: FrameRecord) -> bool:
        if record.is_recovering or int(record.command) in (1, 2, 3):
            return int(record.frame_id) % self.turn_save_every_n == 0
        return int(record.frame_id) % self.save_every_n == 0

    def _process_ready_targets(self) -> None:
        if not self._frame_buffer:
            return

        latest_timestamp = float(self._frame_buffer[-1].timestamp)
        required_future = FUTURE_OFFSETS_S[-1]

        while self._target_frame_ids:
            target_frame_id = self._target_frame_ids[0]
            target_frame = self._frame_lookup.get(target_frame_id)
            if target_frame is None:
                self._target_frame_ids.popleft()
                self._dropped_evicted += 1
                continue

            if latest_timestamp + EPSILON_S < target_frame.timestamp + required_future:
                break

            self._target_frame_ids.popleft()
            self._finalize_target_frame(target_frame)

    def _finalize_target_frame(self, target_frame: FrameRecord) -> None:
        sample_row, reason = self._build_sample_row(target_frame)
        if sample_row is None:
            if reason == "startup_or_history":
                self._dropped_startup += 1
            elif reason == "crash_window":
                self._dropped_crash += 1
            else:
                self._dropped_geometry += 1
            return

        self._writer.writerow(sample_row)
        self._saved_samples += 1
        command_value = int(target_frame.command)
        if command_value not in self._saved_command_counts:
            self._saved_command_counts[command_value] = 0
        self._saved_command_counts[command_value] += 1
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n and self._csv_file is not None:
            self._csv_file.flush()
            self._rows_since_flush = 0

    def _build_sample_row(
        self,
        target_frame: FrameRecord,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if np is None:
            return None, "geometry"

        records = list(self._frame_buffer)
        if not records:
            return None, "geometry"

        timestamps = np.fromiter((record.timestamp for record in records), dtype=np.float64)
        xs = np.fromiter((record.x for record in records), dtype=np.float64)
        ys = np.fromiter((record.y for record in records), dtype=np.float64)
        crashes = np.fromiter((1 if record.has_crash else 0 for record in records), dtype=np.int8)

        target_ts = float(target_frame.timestamp)
        history_start_ts = target_ts + HISTORY_LOOKUPS[0][1]
        future_end_ts = target_ts + FUTURE_OFFSETS_S[-1]

        if timestamps[0] > history_start_ts + EPSILON_S:
            return None, "startup_or_history"
        if timestamps[-1] + EPSILON_S < future_end_ts:
            return None, "geometry"

        history_matches = self._resolve_history_records(records, timestamps, target_ts)
        if history_matches is None:
            return None, "startup_or_history"

        crash_mask = (timestamps >= history_start_ts - EPSILON_S) & (timestamps <= future_end_ts + EPSILON_S)
        if bool(np.any(crashes[crash_mask] > 0)):
            return None, "crash_window"

        future_waypoints = self._interpolate_future_waypoints(
            timestamps=timestamps,
            xs=xs,
            ys=ys,
            target_timestamp=target_ts,
            ego_frame=target_frame,
        )
        if future_waypoints is None:
            return None, "geometry"

        hist_tm06 = history_matches["tm06"]
        hist_tm03 = history_matches["tm03"]
        img_id_tm06 = self._ensure_frame_images_saved(hist_tm06)
        img_id_tm03 = self._ensure_frame_images_saved(hist_tm03)
        img_id_t0 = self._ensure_frame_images_saved(target_frame)

        row: Dict[str, Any] = {
            "frame_id": int(target_frame.frame_id),
            "timestamp": f"{target_frame.timestamp:.6f}",
            "img_id": img_id_t0,
            "img_id_tm06": img_id_tm06,
            "img_id_tm03": img_id_tm03,
            "frame_id_tm06": int(hist_tm06.frame_id),
            "frame_id_tm03": int(hist_tm03.frame_id),
            "timestamp_tm06": f"{hist_tm06.timestamp:.6f}",
            "timestamp_tm03": f"{hist_tm03.timestamp:.6f}",
            "speed": round(float(target_frame.speed), 3),
            "command": int(target_frame.command),
            "steering": round(float(target_frame.steering), 5),
            "throttle": round(float(target_frame.throttle), 4),
            "brake": round(float(target_frame.brake), 4),
            "pitch": round(float(target_frame.pitch), 5),
            "roll": round(float(target_frame.roll), 5),
            "yaw": round(float(target_frame.yaw), 5),
            "x": round(float(target_frame.x), 6),
            "y": round(float(target_frame.y), 6),
            "z": round(float(target_frame.z), 6),
            "has_crash": int(bool(target_frame.has_crash)),
            "recovery_flag": int(bool(target_frame.is_recovering)),
        }

        for index, (wp_x, wp_y) in enumerate(future_waypoints, start=1):
            row[f"wp_{index}_x"] = round(float(wp_x), 6)
            row[f"wp_{index}_y"] = round(float(wp_y), 6)

        return row, None

    def _resolve_history_records(
        self,
        records: list[FrameRecord],
        timestamps: Any,
        target_timestamp: float,
    ) -> Optional[Dict[str, FrameRecord]]:
        history_records: Dict[str, FrameRecord] = {}
        for label, offset_s in HISTORY_LOOKUPS:
            desired_timestamp = float(target_timestamp + offset_s)
            index = int(np.searchsorted(timestamps, desired_timestamp, side="left"))

            candidate_indices: list[int] = []
            if index < len(records):
                candidate_indices.append(index)
            if index > 0:
                candidate_indices.append(index - 1)
            if not candidate_indices:
                return None

            best_index = min(
                candidate_indices,
                key=lambda item_index: abs(float(timestamps[item_index]) - desired_timestamp),
            )
            if abs(float(timestamps[best_index]) - desired_timestamp) > HISTORY_NEAREST_TOLERANCE_S + EPSILON_S:
                return None
            history_records[label] = records[best_index]
        return history_records

    def _interpolate_future_waypoints(
        self,
        timestamps: Any,
        xs: Any,
        ys: Any,
        target_timestamp: float,
        ego_frame: FrameRecord,
    ) -> Optional[list[tuple[float, float]]]:
        yaw_rad = math.radians(float(ego_frame.yaw))
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        ego_waypoints: list[tuple[float, float]] = []
        for offset_s in FUTURE_OFFSETS_S:
            desired_timestamp = float(target_timestamp + offset_s)
            interp_position = self._interpolate_position_at_timestamp(
                timestamps=timestamps,
                xs=xs,
                ys=ys,
                desired_timestamp=desired_timestamp,
            )
            if interp_position is None:
                return None
            global_x, global_y = interp_position
            dx = float(global_x - ego_frame.x)
            dy = float(global_y - ego_frame.y)
            ego_x = cos_yaw * dx + sin_yaw * dy
            ego_y = -sin_yaw * dx + cos_yaw * dy
            ego_waypoints.append((ego_x, ego_y))

        return ego_waypoints

    @staticmethod
    def _interpolate_position_at_timestamp(
        timestamps: Any,
        xs: Any,
        ys: Any,
        desired_timestamp: float,
    ) -> Optional[tuple[float, float]]:
        index = int(np.searchsorted(timestamps, desired_timestamp, side="left"))

        if index < len(timestamps) and abs(float(timestamps[index]) - desired_timestamp) <= EPSILON_S:
            return float(xs[index]), float(ys[index])

        if index <= 0 or index >= len(timestamps):
            return None

        prev_index = index - 1
        next_index = index
        t0 = float(timestamps[prev_index])
        t1 = float(timestamps[next_index])
        if t1 <= t0:
            return None

        alpha = (desired_timestamp - t0) / (t1 - t0)
        x_value = float(xs[prev_index] + alpha * (xs[next_index] - xs[prev_index]))
        y_value = float(ys[prev_index] + alpha * (ys[next_index] - ys[prev_index]))
        return x_value, y_value

    def _ensure_frame_images_saved(self, frame_record: FrameRecord) -> str:
        existing = self._saved_image_ids.get(frame_record.frame_id)
        if existing is not None:
            return existing

        image_id = f"{self._next_img_index:08d}"
        self._next_img_index += 1

        self._save_jpeg(self.images_center_dir, image_id, frame_record.images["center"])
        self._save_jpeg(self.images_left_dir, image_id, frame_record.images["left"])
        self._save_jpeg(self.images_right_dir, image_id, frame_record.images["right"])

        self._saved_image_ids[frame_record.frame_id] = image_id
        return image_id

    def _save_jpeg(self, image_dir: Path, image_id: str, rgb_image: Any) -> None:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        image_path = image_dir / f"{image_id}.jpg"
        cv2.imwrite(str(image_path), bgr, self._encode_params)

    def _cleanup_pending(self, current_frame_id: int) -> None:
        keep_after = int(current_frame_id) - 64
        for frame_id in [frame for frame in self._pending_images if frame < keep_after]:
            self._pending_images.pop(frame_id, None)
        for frame_id in [frame for frame in self._pending_states if frame < keep_after]:
            self._pending_states.pop(frame_id, None)

    def close(self) -> None:
        if self.enabled:
            self._drain_sensor_queue()
            self._ingest_ready_frames()
            self._process_ready_targets()
            self._dropped_unresolved += len(self._target_frame_ids)
            self._target_frame_ids.clear()

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

        logging.info(
            (
                "Data collector stopped. "
                "saved=%d dropped_startup=%d dropped_crash=%d "
                "dropped_geometry=%d dropped_evicted=%d dropped_unresolved=%d "
                "saved_cmd_hist={0:%d,1:%d,2:%d,3:%d}"
            ),
            self._saved_samples,
            self._dropped_startup,
            self._dropped_crash,
            self._dropped_geometry,
            self._dropped_evicted,
            self._dropped_unresolved,
            int(self._saved_command_counts.get(0, 0)),
            int(self._saved_command_counts.get(1, 0)),
            int(self._saved_command_counts.get(2, 0)),
            int(self._saved_command_counts.get(3, 0)),
        )
