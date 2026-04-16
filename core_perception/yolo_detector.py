from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core_perception.object_tracker import KalmanObjectTracker
from core_perception.spatial_math import CameraIntrinsics, DynamicIPM


class YoloDetector:
    """
    YOLO + Kalman Tracker + Dynamic IPM pipeline.

    This module intentionally does NOT compute emergency brake flag.
    """

    DEFAULT_DISPLAY_CLASSES: Sequence[str] = (
        "pedestrian",
        "vehicle",
        "two_wheeler",
        "traffic_sign",
        "traffic_light_red",
        "traffic_light_green",
        "stop_line",
    )

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        display_classes: Optional[Sequence[str]] = None,
        # Compatibility args used by run_agents.py
        camera_fov_deg: float = 90.0,
        obstacle_base_distance_m: float = 8.0,
        camera_mount_x_m: float = 1.5,
        camera_mount_y_m: float = 0.0,
        camera_mount_z_m: float = 2.2,
        camera_pitch_deg: float = -8.0,
        camera_roll_deg: float = 0.0,
        # Tracker tuning.
        tracker_iou_threshold: float = 0.25,
        tracker_max_age: int = 30,
        tracker_min_hits: int = 1,
        tracker_process_noise: float = 1.0,
        tracker_measurement_noise: float = 10.0,
    ) -> None:
        _ = obstacle_base_distance_m  # retained only for backward compatibility
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        self.conf_threshold = float(conf_threshold)
        self.camera_fov_deg = float(camera_fov_deg)
        self.camera_mount_x_m = float(camera_mount_x_m)
        self.camera_mount_y_m = float(camera_mount_y_m)
        self.camera_mount_z_m = float(camera_mount_z_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.camera_roll_deg = float(camera_roll_deg)

        model_ext = os.path.splitext(model_path)[1].lower()
        self._is_exported_model = model_ext in {".engine", ".onnx", ".openvino", ".xml", ".tflite"}

        if model_ext == ".engine":
            self._ensure_tensorrt_module_alias()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path, task="detect")
        if not self._is_exported_model:
            self.model = self.model.to(self.device)
        self.class_names = self.model.names

        self.display_classes = set(
            self._normalize_class_name(name)
            for name in (display_classes or self.DEFAULT_DISPLAY_CLASSES)
        )
        self.class_aliases = {
            "trafficlight_red": "traffic_light_red",
            "red_traffic_light": "traffic_light_red",
            "trafficlight_green": "traffic_light_green",
            "green_traffic_light": "traffic_light_green",
            "stopline": "stop_line",
            "stop-line": "stop_line",
            "stop_line_marking": "stop_line",
        }

        self._tracker = KalmanObjectTracker(
            iou_threshold=float(tracker_iou_threshold),
            max_age=int(tracker_max_age),
            min_hits=int(tracker_min_hits),
            process_noise=float(tracker_process_noise),
            measurement_noise=float(tracker_measurement_noise),
        )

        self._spatial: Optional[DynamicIPM] = None
        self._spatial_resolution: Optional[Tuple[int, int]] = None
        self._last_debug_info: Dict[str, Any] = {}

    @staticmethod
    def _ensure_tensorrt_module_alias() -> None:
        """Map tensorrt_bindings to tensorrt when NVIDIA meta package is unavailable."""
        try:
            import tensorrt  # type: ignore  # noqa: F401
            return
        except Exception:
            pass

        try:
            import tensorrt_bindings as trt  # type: ignore

            if "tensorrt" not in sys.modules:
                sys.modules["tensorrt"] = trt
        except Exception:
            # Keep default import behavior; Ultralytics will raise a clear error if TRT is unavailable.
            pass

    @staticmethod
    def _normalize_class_name(class_name: Any) -> str:
        return str(class_name).strip().lower().replace(" ", "_").replace("-", "_")

    def _resolve_class_name(self, cls_id: int) -> str:
        if isinstance(self.class_names, dict):
            raw_name = self.class_names.get(cls_id, str(cls_id))
        else:
            if 0 <= cls_id < len(self.class_names):
                raw_name = self.class_names[cls_id]
            else:
                raw_name = str(cls_id)
        normalized = self._normalize_class_name(raw_name)
        return self.class_aliases.get(normalized, normalized)

    def _ensure_spatial(self, width: int, height: int) -> None:
        key = (int(width), int(height))
        if self._spatial is not None and self._spatial_resolution == key:
            return

        intr = CameraIntrinsics.from_fov(width, height, self.camera_fov_deg)
        self._spatial = DynamicIPM(
            intrinsics=intr,
            camera_mount_x_m=self.camera_mount_x_m,
            camera_mount_y_m=self.camera_mount_y_m,
            camera_mount_z_m=self.camera_mount_z_m,
            base_pitch_deg=self.camera_pitch_deg,
            base_roll_deg=self.camera_roll_deg,
        )
        self._spatial_resolution = key

    def _prepare_bgr(self, raw_image: np.ndarray) -> np.ndarray:
        if raw_image is None or raw_image.ndim != 3 or raw_image.shape[2] not in (3, 4):
            raise ValueError("raw_image must be HxWx3 or HxWx4.")
        if raw_image.shape[2] == 4:
            return cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        return raw_image.copy()

    def detect(self, raw_image: np.ndarray) -> List[Dict[str, Any]]:
        image_bgr = self._prepare_bgr(raw_image)
        if self._is_exported_model:
            infer_device: Any = 0 if torch.cuda.is_available() else "cpu"
            results = self.model.predict(
                source=image_bgr,
                conf=self.conf_threshold,
                verbose=False,
                device=infer_device,
            )
        else:
            results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []

        if len(results) == 0:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self._resolve_class_name(cls_id)
            if class_name not in self.display_classes:
                continue

            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                }
            )
        return detections

    def process_frame(
        self,
        raw_image: np.ndarray,
        speed_kmh: Optional[float] = None,
        imu_pitch_deg: float = 0.0,
        imu_roll_deg: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        image_bgr = self._prepare_bgr(raw_image)
        height, width = image_bgr.shape[:2]
        self._ensure_spatial(width, height)

        ts = time.time() if timestamp is None else float(timestamp)
        raw_detections = self.detect(image_bgr)
        tracked = self._tracker.update(raw_detections, timestamp=ts)

        if self._spatial is None:
            spatial_objects: List[Dict[str, Any]] = []
        else:
            spatial_objects = self._spatial.project_to_bev(
                tracked,
                pitch=float(imu_pitch_deg),
                roll=float(imu_roll_deg),
                ego_speed_kmh=0.0 if speed_kmh is None else float(speed_kmh),
                timestamp=ts,
            )

        return {
            "raw_detections": raw_detections,
            "tracked_objects": tracked,
            "spatial_objects": spatial_objects,
            "frame_size": (height, width),
            "timestamp": ts,
        }

    def _to_runtime_detections(
        self,
        spatial_objects: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for obj in spatial_objects:
            bbox = obj.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            class_name = str(obj.get("class", "unknown"))
            confidence = float(obj.get("conf", 0.0))
            distance = float(obj.get("distance_m", float("inf")))
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            bev_xy = obj.get("bev_xy_m")
            ground_point_vehicle = None
            if isinstance(bev_xy, (list, tuple)) and len(bev_xy) == 2:
                bx, by = float(bev_xy[0]), float(bev_xy[1])
                if np.isfinite(bx) and np.isfinite(by):
                    ground_point_vehicle = (bx, by, 0.0)

            processed.append(
                {
                    "box": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": confidence,
                    "distance": distance,
                    "distance_source": "dynamic_ipm",
                    "center": (center_x, center_y),
                    "roi_zone": None,
                    "ground_point": (center_x, float(y2)),
                    "ground_point_vehicle": ground_point_vehicle,
                    "in_danger_roi": False,
                    "path_check_mode": "dynamic_ipm",
                    "danger_match": False,
                    "track_id": obj.get("track_id"),
                    "velocity_kmh": float(obj.get("velocity_kmh", 0.0)),
                    "relative_velocity_kmh": float(obj.get("relative_velocity_kmh", 0.0)),
                }
            )
        return processed

    def detect_and_evaluate(
        self,
        raw_image: np.ndarray,
        distance_threshold: Optional[float] = None,
        depth_map_m: Optional[np.ndarray] = None,
        vehicle_steer: Optional[float] = None,
        speed_kmh: Optional[float] = None,
        imu_pitch_deg: float = 0.0,
        imu_roll_deg: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        _ = (distance_threshold, depth_map_m, vehicle_steer)
        pipeline = self.process_frame(
            raw_image=raw_image,
            speed_kmh=speed_kmh,
            imu_pitch_deg=imu_pitch_deg,
            imu_roll_deg=imu_roll_deg,
            timestamp=time.time(),
        )

        spatial_objects = pipeline["spatial_objects"]
        processed = self._to_runtime_detections(spatial_objects)

        self._last_debug_info = {
            "roi_regions": [],
            "obstacle_danger_roi": {"label": "Obstacle corridor", "polygon": []},
            "obstacle_path_model": {"mode": "dynamic_ipm"},
            "road_plane": {"valid": False},
            "depth_available": depth_map_m is not None,
            "active_roi_zone": None,
            "locked_zone": None,
            "last_locked_zone": None,
            "zone_lock_acquire_counter": {},
            "zone_lock_missing_counter": 0,
            "zone_lock_acquire_frames": 0,
            "zone_lock_release_missing_frames": 0,
            "obstacle_emergency": False,
            "obstacle_reason": "",
            "red_light_frame_trigger": False,
            "green_light_frame_release": False,
            "red_light_active": False,
            "green_immunity_counter": 0,
            "green_immunity_zone": None,
            "turn_phase_active": False,
            "turn_red_suppressed": False,
            "turn_confirm_counter": 0,
            "turn_hold_counter": 0,
            "turn_green_grace_counter": 0,
            "turn_green_grace_zone": None,
            "zone_evaluations": {},
            "decision_reason": "stopping_logic_removed",
            "pipeline_counts": {
                "raw": len(pipeline["raw_detections"]),
                "tracked": len(pipeline["tracked_objects"]),
                "spatial": len(spatial_objects),
            },
            "pipeline_timestamp": float(pipeline["timestamp"]),
        }
        # Emergency-stop policy was intentionally removed in this module.
        is_emergency = False
        return processed, is_emergency

    def get_last_debug_info(self) -> Dict[str, Any]:
        return self._last_debug_info

