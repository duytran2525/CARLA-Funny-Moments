from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR

from core_perception.spatial_math import CameraIntrinsics, DynamicIPM

class YoloDetector:
    """
    YOLO + Ultralytics tracker (BoT-SORT/ByteTrack) + Dynamic IPM pipeline.

    This module intentionally does NOT compute emergency brake flag.
    """

    DEFAULT_DISPLAY_CLASSES: Sequence[str] = (
        "vehicle",
        "two_wheeler",
        "traffic_light_red",
        "traffic_sign",
        "pedestrian",
        "traffic_light_green",
        "stop_line",
    )
    METRICS_EVAL_CLASSES: Sequence[str] = DEFAULT_DISPLAY_CLASSES
    CLASS_ALIASES = {
        "bike": "two_wheeler",
        "bicycle": "two_wheeler",
        "cyclist": "two_wheeler",
        "motobike": "two_wheeler",
        "motorbike": "two_wheeler",
        "motorcycle": "two_wheeler",
        "trafficlight_red": "traffic_light_red",
        "red_traffic_light": "traffic_light_red",
        "trafficlight_green": "traffic_light_green",
        "green_traffic_light": "traffic_light_green",
        "stopline": "stop_line",
        "stop-line": "stop_line",
        "stop_line_marking": "stop_line",
    }

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        display_classes: Optional[Sequence[str]] = None,
        inference_imgsz: Optional[int | Tuple[int, int]] = None,
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
        tracker_config: str = "botsort.yaml",
        enable_tracking_metrics_logging: bool = False,
    ) -> None:
        _ = (
            obstacle_base_distance_m,
            tracker_iou_threshold,
            tracker_max_age,
            tracker_min_hits,
            tracker_process_noise,
            tracker_measurement_noise,
        )  # retained only for backward compatibility
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        self.conf_threshold = float(conf_threshold)
        self.camera_fov_deg = float(camera_fov_deg)
        self.camera_mount_x_m = float(camera_mount_x_m)
        self.camera_mount_y_m = float(camera_mount_y_m)
        self.camera_mount_z_m = float(camera_mount_z_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.camera_roll_deg = float(camera_roll_deg)
        self.inference_imgsz = inference_imgsz
        self.uses_depth_input = False

        model_ext = os.path.splitext(model_path)[1].lower()
        self._is_exported_model = model_ext in {".engine", ".onnx", ".openvino", ".xml", ".tflite"}

        if model_ext == ".engine":
            self._ensure_tensorrt_module_alias()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._predict_device: Any = 0 if self.device.type == "cuda" else "cpu"
        self._use_half_precision = self.device.type == "cuda" and not self._is_exported_model
        self._warmed_up = False
        
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if "rtdetr" in model_path.lower():
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path, task="detect")
        if not self._is_exported_model:
            self.model = self.model.to(self.device)
        self.class_names = self.model.names

        self.display_classes = set(
            self._normalize_class_name(name)
            for name in (display_classes or self.DEFAULT_DISPLAY_CLASSES)
        )
        self.class_aliases = dict(self.CLASS_ALIASES)
        self.tracker_config = str(tracker_config or "botsort.yaml")
        self._track_persist = True
        self._enable_tracking_metrics_logging = bool(enable_tracking_metrics_logging)
        self._metrics_eval_classes = set(self.METRICS_EVAL_CLASSES)
        self.current_frame_id = 0
        self.tracking_logs: List[str] = []

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
        if raw_image.flags.c_contiguous:
            return raw_image
        return np.ascontiguousarray(raw_image)

    @staticmethod
    def _extract_static_export_hw(exc: BaseException) -> Optional[Tuple[int, int]]:
        message = str(exc)
        match = re.search(r"max model size\s*\(\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", message)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _imgsz_from_hw(height: int, width: int) -> int | Tuple[int, int]:
        return int(height) if int(height) == int(width) else (int(height), int(width))

    def _predict_once(self, image_bgr: np.ndarray):
        predict_kwargs: Dict[str, Any] = {
            "source": image_bgr,
            "conf": self.conf_threshold,
            "verbose": False,
            "device": self._predict_device,
        }
        if self.inference_imgsz is not None:
            predict_kwargs["imgsz"] = self.inference_imgsz
        if self._use_half_precision:
            predict_kwargs["half"] = True
        return self.model.predict(**predict_kwargs)

    def _predict(self, image_bgr: np.ndarray):
        try:
            return self._predict_once(image_bgr)
        except AssertionError as exc:
            if not self._is_exported_model:
                raise

            static_hw = self._extract_static_export_hw(exc)
            if static_hw is None:
                raise

            corrected_imgsz = self._imgsz_from_hw(*static_hw)
            if self.inference_imgsz == corrected_imgsz:
                raise

            logging.warning(
                "Overriding incompatible inference_imgsz=%s with exported model input size %s for %s.",
                self.inference_imgsz,
                corrected_imgsz,
                os.path.basename(str(self.model)),
            )
            self.inference_imgsz = corrected_imgsz
            return self._predict_once(image_bgr)

    def _track_once(self, image_bgr: np.ndarray):
        track_kwargs: Dict[str, Any] = {
            "source": image_bgr,
            "conf": self.conf_threshold,
            "verbose": False,
            "device": self._predict_device,
            "persist": self._track_persist,
            "tracker": self.tracker_config,
        }
        if self.inference_imgsz is not None:
            track_kwargs["imgsz"] = self.inference_imgsz
        if self._use_half_precision:
            track_kwargs["half"] = True
        return self.model.track(**track_kwargs)

    def _track(self, image_bgr: np.ndarray):
        try:
            return self._track_once(image_bgr)
        except AssertionError as exc:
            if not self._is_exported_model:
                raise

            static_hw = self._extract_static_export_hw(exc)
            if static_hw is None:
                raise

            corrected_imgsz = self._imgsz_from_hw(*static_hw)
            if self.inference_imgsz == corrected_imgsz:
                raise

            logging.warning(
                "Overriding incompatible inference_imgsz=%s with exported model input size %s.",
                self.inference_imgsz,
                corrected_imgsz,
            )
            self.inference_imgsz = corrected_imgsz
            return self._track_once(image_bgr)

    def warmup(self, width: int, height: int) -> None:
        if self._warmed_up:
            return
        warm_h = max(32, int(height))
        warm_w = max(32, int(width))
        warm_frame = np.zeros((warm_h, warm_w, 3), dtype=np.uint8)
        self._predict(warm_frame)
        self._warmed_up = True

    def _detections_from_results(
        self,
        results: Any,
        *,
        include_tracking: bool = False,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.round().to(dtype=torch.int32).cpu().numpy()
        confs = boxes.conf.float().cpu().numpy()
        class_ids = boxes.cls.to(dtype=torch.int32).cpu().numpy()
        if include_tracking and boxes.id is not None:
            track_ids: Sequence[Any] = boxes.id.to(dtype=torch.int32).cpu().numpy()
        else:
            track_ids = [None] * len(boxes)

        for coords, conf, cls_id, t_id in zip(xyxy, confs, class_ids, track_ids):
            class_name = self._resolve_class_name(int(cls_id))
            if class_name not in self.display_classes:
                continue

            conf = float(conf)
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = [int(v) for v in coords.tolist()]
            if x2 <= x1 or y2 <= y1:
                continue

            track_id: Optional[int] = None
            if t_id is not None:
                try:
                    track_id = int(t_id)
                except Exception:
                    track_id = None

            det: Dict[str, Any] = {
                "class": class_name,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
            }
            if include_tracking:
                det["raw_bbox"] = [x1, y1, x2, y2]
                det["track_id"] = track_id

            detections.append(det)
        return detections

    def detect(self, raw_image: np.ndarray) -> List[Dict[str, Any]]:
        image_bgr = self._prepare_bgr(raw_image)
        results = self._predict(image_bgr)
        return self._detections_from_results(results)

    def detect_and_track(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Use Ultralytics built-in tracker (BoT-SORT/ByteTrack)."""
        frame_bgr = self._prepare_bgr(image_bgr)
        if self.current_frame_id <= 0:
            self.current_frame_id = 1
        results = self._track(frame_bgr)
        tracked_objects = self._detections_from_results(results, include_tracking=True)

        if not tracked_objects:
            predicted_objects = self.detect(frame_bgr)
            tracked_objects = [
                {
                    **obj,
                    "raw_bbox": list(obj["bbox"]),
                    "track_id": None,
                }
                for obj in predicted_objects
            ]
            if tracked_objects:
                logging.debug(
                    "Ultralytics track() returned no usable boxes; falling back to predict() for %d boxes.",
                    len(tracked_objects),
                )

        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            class_name = str(obj.get("class", "unknown"))
            conf = float(obj.get("conf", 0.0))
            track_id = obj.get("track_id")
            track_id_val = int(track_id) if track_id is not None else -1

            if self._enable_tracking_metrics_logging and class_name in self._metrics_eval_classes:
                # MOTChallenge tracking-results format plus a repo-local class column:
                # <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>,<class>
                # The first 10 columns stay MOT-compatible; the class column lets
                # repo-local metrics avoid matching e.g. a vehicle prediction to pedestrian GT.
                w = int(x2 - x1)
                h = int(y2 - y1)
                log_line = (
                    f"{int(self.current_frame_id)},{int(track_id_val)},{int(x1)},{int(y1)},"
                    f"{int(w)},{int(h)},{float(conf):.4f},-1,-1,-1,{class_name}"
                )
                self.tracking_logs.append(log_line)
        return tracked_objects

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
        self.current_frame_id += 1
        tracked = self.detect_and_track(image_bgr)

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
            "raw_detections": tracked,
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

    def save_tracking_metrics_log(self, save_path: str = "tracker_predictions.txt") -> None:
        """Save tracker predictions as MOT-format txt for post-run evaluation."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_obj:
            for line in self.tracking_logs:
                file_obj.write(f"{line}\n")
        logging.info(
            "[Metrics] Saved %d tracking detections to %s",
            len(self.tracking_logs),
            path,
        )
        self.tracking_logs = []
        self.current_frame_id = 0

