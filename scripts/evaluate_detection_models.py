from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - optional CARLA dependency
    carla = None

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    yaml = None

try:
    from core_control.carla_manager import CarlaManager, SpectatorConfig
except Exception as exc:  # pragma: no cover - import validated at runtime
    logging.warning("Failed to import CarlaManager: %s", exc)
    CarlaManager = None
    SpectatorConfig = None

try:
    from core_perception.yolo_detector import YoloDetector
except Exception as exc:  # pragma: no cover - import validated at runtime
    logging.warning("Failed to import YoloDetector: %s", exc)
    YoloDetector = None

try:
    from agents.navigation.basic_agent import BasicAgent  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional CARLA PythonAPI dependency
    BasicAgent = None


EVAL_CLASS_ORDER = [
    "vehicle",
    "two_wheeler",
    "pedestrian",
    "traffic_light_red",
    "traffic_light_green",
]

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
    "stop_line_marking": "stop_line",
}

AP_IOU_THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]


@dataclass
class ModelConfig:
    name: str
    path: Path
    model_type: str = "yolo"


@dataclass
class CarlaEvalConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    timeout: float = 60.0
    sync: bool = True
    fixed_delta: float = 1.0 / 30.0
    no_rendering: bool = False
    map_name: str = "Town03"
    vehicle_filter: str = "vehicle.tesla.model3"
    spawn_point: int = 1
    destination_point: int = 50
    camera_width: int = 640
    camera_height: int = 360
    camera_fov: float = 90.0
    weather_preset: str = "ClearNoon"
    npc_vehicle_count: int = 15
    npc_bike_count: int = 5
    npc_motorbike_count: int = 5
    npc_pedestrian_count: int = 5
    npc_enable_autopilot: bool = True
    seed: Optional[int] = None


@dataclass
class GroundTruthFilterConfig:
    max_distance_m: float = 40.0
    min_bbox_dim_px: int = 10
    min_bbox_area_px: int = 600
    min_visible_area_ratio: float = 0.50
    min_depth_visible_ratio: float = 0.20
    max_bbox_aspect_ratio: float = 6.0


@dataclass
class DetectionEvalConfig:
    models: List[ModelConfig] = field(default_factory=list)
    conf_threshold: float = 0.5
    iou_threshold: float = 0.3
    class_iou_thresholds: Dict[str, float] = field(default_factory=dict)
    max_frames: int = 2000
    inference_imgsz: Optional[int] = 640
    eval_classes: List[str] = field(default_factory=lambda: list(EVAL_CLASS_ORDER))
    output_dir: Path = PROJECT_ROOT / "outputs" / "detection_test_eval"
    save_frames: bool = False
    save_frame_stride: int = 30
    target_speed_kmh: float = 35.0
    log_interval_frames: int = 30
    traffic_light_gt_mode: str = "level_bbs"
    traffic_light_actor_fallback: bool = True
    traffic_light_head_max_extent_m: float = 2.5
    traffic_light_head_max_volume_m3: float = 3.0
    traffic_light_actor_match_radius_m: float = 10.0
    traffic_light_min_bbox_dim_px: int = 4
    traffic_light_min_bbox_area_px: int = 40
    traffic_light_min_visible_area_ratio: float = 0.25
    traffic_light_max_bbox_aspect_ratio: float = 8.0
    gt_filters: GroundTruthFilterConfig = field(default_factory=GroundTruthFilterConfig)


@dataclass
class RunConfig:
    carla_cfg: CarlaEvalConfig
    eval_cfg: DetectionEvalConfig
    source_config_path: Path


@dataclass
class DetectionRecord:
    frame: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0
    object_id: int = -1


@dataclass
class ProjectedBox:
    bbox: Tuple[float, float, float, float]
    projected_points: List[Tuple[float, float, float]]
    distance_m: Optional[float] = None


def normalize_class_name(class_name: Any) -> str:
    text = str(class_name).strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(text, text)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_") or "model"


def cfg_get(config: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def nested_cfg_get(config: Dict[str, Any], path: Sequence[str], default: Any) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def resolve_repo_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def parse_run_config(config_path: Path) -> RunConfig:
    raw = load_yaml(config_path)
    test_detection = raw.get("test_detection", {})
    if not isinstance(test_detection, dict):
        test_detection = {}

    models: List[ModelConfig] = []
    for entry in test_detection.get("models", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or Path(str(entry.get("path", ""))).stem).strip()
        path_text = str(entry.get("path", "")).strip()
        if not name or not path_text:
            continue
        models.append(
            ModelConfig(
                name=name,
                path=resolve_repo_path(path_text),
                model_type=str(entry.get("type", "yolo")).strip().lower() or "yolo",
            )
        )

    gt_filters_raw = test_detection.get("gt_filters", {})
    if not isinstance(gt_filters_raw, dict):
        gt_filters_raw = {}

    gt_filters = GroundTruthFilterConfig(
        max_distance_m=float(gt_filters_raw.get("max_distance_m", 40.0)),
        min_bbox_dim_px=int(gt_filters_raw.get("min_bbox_dim_px", 10)),
        min_bbox_area_px=int(gt_filters_raw.get("min_bbox_area_px", 600)),
        min_visible_area_ratio=float(gt_filters_raw.get("min_visible_area_ratio", 0.50)),
        min_depth_visible_ratio=float(gt_filters_raw.get("min_depth_visible_ratio", 0.20)),
        max_bbox_aspect_ratio=float(gt_filters_raw.get("max_bbox_aspect_ratio", 6.0)),
    )

    class_thresholds = {}
    raw_class_thresholds = test_detection.get("class_iou_thresholds", {})
    if isinstance(raw_class_thresholds, dict):
        for key, value in raw_class_thresholds.items():
            class_thresholds[normalize_class_name(key)] = float(value)

    inference_imgsz_raw = test_detection.get("inference_imgsz", 640)
    inference_imgsz = int(inference_imgsz_raw) if inference_imgsz_raw is not None else 0
    if inference_imgsz <= 0:
        inference_imgsz_value: Optional[int] = None
    else:
        inference_imgsz_value = inference_imgsz

    eval_classes = [
        normalize_class_name(item)
        for item in test_detection.get("eval_classes", EVAL_CLASS_ORDER)
        if str(item).strip()
    ]
    if not eval_classes:
        eval_classes = list(EVAL_CLASS_ORDER)

    carla_cfg = CarlaEvalConfig(
        host=str(cfg_get(raw, "carla", "host", "127.0.0.1")),
        port=int(cfg_get(raw, "carla", "port", 2000)),
        tm_port=int(cfg_get(raw, "carla", "tm_port", 8000)),
        timeout=float(cfg_get(raw, "carla", "timeout", 60.0)),
        sync=to_bool(cfg_get(raw, "carla", "sync", True), True),
        fixed_delta=float(cfg_get(raw, "carla", "fixed_delta", 1.0 / 30.0)),
        no_rendering=to_bool(cfg_get(raw, "carla", "no_rendering", False), False),
        map_name=str(cfg_get(raw, "carla", "map", "Town03")),
        vehicle_filter=str(cfg_get(raw, "vehicle", "filter", "vehicle.tesla.model3")),
        spawn_point=int(cfg_get(raw, "vehicle", "spawn_point", 1)),
        destination_point=int(cfg_get(raw, "vehicle", "destination_point", 50)),
        camera_width=int(cfg_get(raw, "camera", "width", 640)),
        camera_height=int(cfg_get(raw, "camera", "height", 360)),
        camera_fov=float(cfg_get(raw, "camera", "fov", 90.0)),
        weather_preset=str(cfg_get(raw, "weather", "preset", "ClearNoon")),
        npc_vehicle_count=int(cfg_get(raw, "traffic_spawn", "vehicle_count", 15)),
        npc_bike_count=int(cfg_get(raw, "traffic_spawn", "bike_count", 5)),
        npc_motorbike_count=int(cfg_get(raw, "traffic_spawn", "motorbike_count", 5)),
        npc_pedestrian_count=int(cfg_get(raw, "traffic_spawn", "pedestrian_count", 5)),
        npc_enable_autopilot=to_bool(
            cfg_get(raw, "traffic_spawn", "npc_enable_autopilot", True),
            True,
        ),
        seed=(
            int(nested_cfg_get(raw, ["runtime", "seed"], 0))
            if nested_cfg_get(raw, ["runtime", "seed"], None) is not None
            else None
        ),
    )

    eval_cfg = DetectionEvalConfig(
        models=models,
        conf_threshold=float(test_detection.get("conf_threshold", 0.5)),
        iou_threshold=float(test_detection.get("iou_threshold", 0.3)),
        class_iou_thresholds=class_thresholds,
        max_frames=int(test_detection.get("max_frames", 2000)),
        inference_imgsz=inference_imgsz_value,
        eval_classes=eval_classes,
        output_dir=resolve_repo_path(test_detection.get("output_dir", "outputs/detection_test_eval")),
        save_frames=to_bool(test_detection.get("save_frames", False), False),
        save_frame_stride=max(1, int(test_detection.get("save_frame_stride", 30))),
        target_speed_kmh=float(test_detection.get("target_speed_kmh", 35.0)),
        log_interval_frames=max(1, int(test_detection.get("log_interval_frames", 30))),
        traffic_light_gt_mode=str(test_detection.get("traffic_light_gt_mode", "level_bbs")).strip().lower(),
        traffic_light_actor_fallback=to_bool(test_detection.get("traffic_light_actor_fallback", True), True),
        traffic_light_head_max_extent_m=float(test_detection.get("traffic_light_head_max_extent_m", 2.5)),
        traffic_light_head_max_volume_m3=float(test_detection.get("traffic_light_head_max_volume_m3", 3.0)),
        traffic_light_actor_match_radius_m=float(test_detection.get("traffic_light_actor_match_radius_m", 10.0)),
        traffic_light_min_bbox_dim_px=int(test_detection.get("traffic_light_min_bbox_dim_px", 4)),
        traffic_light_min_bbox_area_px=int(test_detection.get("traffic_light_min_bbox_area_px", 40)),
        traffic_light_min_visible_area_ratio=float(test_detection.get("traffic_light_min_visible_area_ratio", 0.25)),
        traffic_light_max_bbox_aspect_ratio=float(test_detection.get("traffic_light_max_bbox_aspect_ratio", 8.0)),
        gt_filters=gt_filters,
    )

    return RunConfig(carla_cfg=carla_cfg, eval_cfg=eval_cfg, source_config_path=config_path.resolve())


def make_output_dir(root: Path, dry_run: bool = False) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = "dry_run_" if dry_run else ""
    out_dir = root / f"{prefix}{stamp}"
    counter = 1
    while out_dir.exists():
        out_dir = root / f"{prefix}{stamp}_{counter:02d}"
        counter += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def ensure_navigation_agent_imports() -> None:
    global BasicAgent
    if BasicAgent is not None:
        return

    candidates: List[Path] = []
    env_pythonapi = os.environ.get("CARLA_PYTHONAPI", "").strip()
    if env_pythonapi:
        candidates.append(Path(env_pythonapi))
    env_carla_root = os.environ.get("CARLA_ROOT", "").strip()
    if env_carla_root:
        candidates.append(Path(env_carla_root) / "PythonAPI")
        candidates.append(Path(env_carla_root) / "WindowsNoEditor" / "PythonAPI")
    candidates.extend(
        [
            PROJECT_ROOT / "PythonAPI",
            PROJECT_ROOT.parent / "PythonAPI",
            Path("D:/CARLA/PythonAPI"),
            Path("D:/CARLA/WindowsNoEditor/PythonAPI"),
            Path("C:/CARLA/PythonAPI"),
            Path("C:/CARLA/WindowsNoEditor/PythonAPI"),
        ]
    )

    for base in candidates:
        if not base.exists():
            continue
        base_str = str(base)
        if base_str not in sys.path:
            sys.path.append(base_str)
        carla_sub = base / "carla"
        if carla_sub.exists() and str(carla_sub) not in sys.path:
            sys.path.append(str(carla_sub))
        dist_dir = base / "carla" / "dist"
        if dist_dir.exists():
            for egg in dist_dir.glob("*.egg"):
                egg_str = str(egg)
                if egg_str not in sys.path:
                    sys.path.append(egg_str)

    try:
        BasicAgent = importlib.import_module("agents.navigation.basic_agent").BasicAgent
    except Exception:
        BasicAgent = None


def set_navigation_destination(nav_agent: Any, current_location: Any, destination_location: Any) -> None:
    setter = getattr(nav_agent, "set_destination", None)
    if setter is None:
        raise AttributeError("Navigation agent does not expose set_destination().")

    for args in (
        (destination_location,),
        (current_location, destination_location),
        (destination_location, current_location),
    ):
        try:
            setter(*args)
            return
        except TypeError:
            continue
    setter(destination_location)


def apply_weather_preset(world: Any, preset: str) -> None:
    if carla is None:
        return
    preset_lower = str(preset).lower().replace(" ", "").replace("_", "")
    presets = {
        "clearnoon": carla.WeatherParameters.ClearNoon,
        "cloudynoon": carla.WeatherParameters.CloudyNoon,
        "wetnoon": carla.WeatherParameters.WetNoon,
        "wetcloudynoon": carla.WeatherParameters.WetCloudyNoon,
        "softrainnoon": carla.WeatherParameters.SoftRainNoon,
        "midrainnoon": carla.WeatherParameters.MidRainyNoon,
        "hardrainnoon": carla.WeatherParameters.HardRainNoon,
        "clearsunset": carla.WeatherParameters.ClearSunset,
        "cloudysunset": carla.WeatherParameters.CloudySunset,
        "wetsunset": carla.WeatherParameters.WetSunset,
        "wetcloudysunset": carla.WeatherParameters.WetCloudySunset,
        "softrainsunset": carla.WeatherParameters.SoftRainSunset,
        "midrainsunset": carla.WeatherParameters.MidRainSunset,
        "hardrainsunset": carla.WeatherParameters.HardRainSunset,
    }
    world.set_weather(presets.get(preset_lower, carla.WeatherParameters.ClearNoon))
    logging.info("Applied weather preset: %s", preset)


class EgoRouteDriver:
    def __init__(self, world: Any, vehicle: Any, carla_cfg: CarlaEvalConfig, eval_cfg: DetectionEvalConfig) -> None:
        self.world = world
        self.vehicle = vehicle
        self.carla_cfg = carla_cfg
        self.eval_cfg = eval_cfg
        self.spawn_points = list(world.get_map().get_spawn_points())
        self.nav_agent = None
        self.destination_index: Optional[int] = None
        self.fixed_destination_consumed = False
        self.tm_autopilot = False

    def start(self) -> None:
        if self.vehicle is None:
            return
        ensure_navigation_agent_imports()
        if BasicAgent is not None and self.spawn_points:
            try:
                self.nav_agent = BasicAgent(
                    self.vehicle,
                    target_speed=max(10.0, float(self.eval_cfg.target_speed_kmh)),
                )
                self._set_new_destination()
                logging.info("Ego route driver using BasicAgent.")
                return
            except Exception as exc:
                logging.info("BasicAgent unavailable, falling back to Traffic Manager autopilot: %s", exc)
                self.nav_agent = None

        self._enable_tm_autopilot()
        logging.info("Ego route driver using Traffic Manager autopilot.")

    def _enable_tm_autopilot(self) -> None:
        try:
            self.vehicle.set_autopilot(True, self.carla_cfg.tm_port)
        except TypeError:
            self.vehicle.set_autopilot(True)
        self.tm_autopilot = True

    def _nearest_spawn_index(self, location: Any) -> Optional[int]:
        best_idx = None
        best_dist = float("inf")
        for idx, transform in enumerate(self.spawn_points):
            try:
                dist = float(location.distance(transform.location))
            except Exception:
                continue
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _choose_destination_index(self, current_location: Any) -> int:
        if self.carla_cfg.destination_point >= 0 and not self.fixed_destination_consumed:
            self.fixed_destination_consumed = True
            return int(self.carla_cfg.destination_point) % len(self.spawn_points)

        current_idx = self._nearest_spawn_index(current_location)
        candidates: List[int] = []
        for idx, transform in enumerate(self.spawn_points):
            if current_idx is not None and idx == current_idx:
                continue
            if self.destination_index is not None and idx == self.destination_index:
                continue
            try:
                if current_location.distance(transform.location) < 80.0:
                    continue
            except Exception:
                pass
            candidates.append(idx)
        if not candidates:
            candidates = [idx for idx in range(len(self.spawn_points)) if idx != current_idx]
        if not candidates:
            return int(current_idx or 0)
        return int(random.choice(candidates))

    def _set_new_destination(self) -> None:
        if self.nav_agent is None or not self.spawn_points:
            return
        current_location = self.vehicle.get_location()
        dest_idx = self._choose_destination_index(current_location)
        destination = self.spawn_points[dest_idx].location
        self.destination_index = dest_idx
        set_navigation_destination(self.nav_agent, current_location, destination)
        logging.info(
            "Ego destination set: spawn=%d target=(%.1f, %.1f, %.1f)",
            dest_idx,
            float(destination.x),
            float(destination.y),
            float(destination.z),
        )

    def run_step(self) -> None:
        if self.nav_agent is None or self.vehicle is None:
            return
        try:
            if bool(self.nav_agent.done()):
                self._set_new_destination()
        except Exception:
            pass
        try:
            control = self.nav_agent.run_step()
            self.vehicle.apply_control(control)
        except Exception as exc:
            logging.warning("BasicAgent run_step failed, switching to TM autopilot: %s", exc)
            self.nav_agent = None
            self._enable_tm_autopilot()

    def cleanup(self) -> None:
        if self.vehicle is not None and self.tm_autopilot:
            try:
                self.vehicle.set_autopilot(False, self.carla_cfg.tm_port)
            except Exception:
                pass


def decode_carla_depth_to_meters(image: Any) -> Any:
    if np is None:
        return None
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((image.height, image.width, 4)).astype(np.float32)
    normalized = (
        bgra[:, :, 2] + bgra[:, :, 1] * 256.0 + bgra[:, :, 0] * 65536.0
    ) / 16777215.0
    return normalized * 1000.0


class SensorFrameBuffer:
    def __init__(self, require_depth: bool = True, max_keep_frames: int = 120) -> None:
        self.require_depth = bool(require_depth)
        self.max_keep_frames = int(max_keep_frames)
        self._condition = threading.Condition()
        self._rgb_by_frame: Dict[int, Any] = {}
        self._depth_by_frame: Dict[int, Any] = {}

    def on_rgb(self, image: Any) -> None:
        if np is None or cv2 is None:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = array.reshape((image.height, image.width, 4))
        frame_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        frame_id = int(getattr(image, "frame", 0))
        with self._condition:
            self._rgb_by_frame[frame_id] = frame_bgr
            self._trim_locked(self._rgb_by_frame)
            self._condition.notify_all()

    def on_depth(self, image: Any) -> None:
        depth_m = decode_carla_depth_to_meters(image)
        frame_id = int(getattr(image, "frame", 0))
        with self._condition:
            self._depth_by_frame[frame_id] = depth_m
            self._trim_locked(self._depth_by_frame)
            self._condition.notify_all()

    def _trim_locked(self, store: Dict[int, Any]) -> None:
        if len(store) <= self.max_keep_frames:
            return
        for frame_id in sorted(store)[: max(0, len(store) - self.max_keep_frames)]:
            store.pop(frame_id, None)

    @staticmethod
    def _get_nearest_frame(store: Dict[int, Any], frame_id: int) -> Tuple[Optional[int], Any]:
        if frame_id in store:
            return frame_id, store[frame_id]
        newer = [key for key in store if key >= frame_id]
        if newer:
            key = min(newer)
            return key, store[key]
        if store:
            key = max(store)
            return key, store[key]
        return None, None

    def wait_for_frame(self, frame_id: int, timeout_s: float) -> Tuple[Any, Any]:
        deadline = time.time() + max(0.1, float(timeout_s))
        with self._condition:
            while time.time() < deadline:
                rgb_id, rgb = self._get_nearest_frame(self._rgb_by_frame, frame_id)
                depth_id, depth = self._get_nearest_frame(self._depth_by_frame, frame_id)
                if rgb is not None and (not self.require_depth or depth is not None):
                    if rgb_id is not None:
                        self._rgb_by_frame.pop(rgb_id, None)
                    if depth_id is not None:
                        self._depth_by_frame.pop(depth_id, None)
                    return rgb, depth
                self._condition.wait(max(0.01, deadline - time.time()))
        return None, None


def spawn_camera_pair(world: Any, vehicle: Any, carla_cfg: CarlaEvalConfig, buffer: SensorFrameBuffer) -> Tuple[Any, Any]:
    if carla is None:
        raise RuntimeError("CARLA package is required to spawn sensors.")
    bp_lib = world.get_blueprint_library()
    camera_transform = carla.Transform(
        carla.Location(x=1.5, y=0.0, z=2.2),
        carla.Rotation(pitch=-8.0, yaw=0.0, roll=0.0),
    )

    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(carla_cfg.camera_width))
    rgb_bp.set_attribute("image_size_y", str(carla_cfg.camera_height))
    rgb_bp.set_attribute("fov", str(carla_cfg.camera_fov))
    # if rgb_bp.has_attribute("sensor_tick") and carla_cfg.sync:
    #     rgb_bp.set_attribute("sensor_tick", str(carla_cfg.fixed_delta))
    rgb_camera = world.spawn_actor(rgb_bp, camera_transform, attach_to=vehicle)
    rgb_camera.listen(buffer.on_rgb)

    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(carla_cfg.camera_width))
    depth_bp.set_attribute("image_size_y", str(carla_cfg.camera_height))
    depth_bp.set_attribute("fov", str(carla_cfg.camera_fov))
    # if depth_bp.has_attribute("sensor_tick") and carla_cfg.sync:
    #     depth_bp.set_attribute("sensor_tick", str(carla_cfg.fixed_delta))
    depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
    depth_camera.listen(buffer.on_depth)

    logging.info("Attached synchronized RGB and depth cameras.")
    return rgb_camera, depth_camera


def destroy_sensor(sensor: Any) -> None:
    if sensor is None:
        return
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        sensor.destroy()
    except Exception:
        pass


def camera_intrinsics_matrix(width: int, height: int, fov_deg: float) -> Any:
    if np is None:
        return None
    fov_rad = math.radians(float(fov_deg))
    fx = (float(width) / 2.0) / max(math.tan(fov_rad / 2.0), 1e-6)
    fy = fx
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_world_location_to_image(
    world_location: Any,
    world_to_camera: Any,
    intrinsics: Any,
) -> Optional[Tuple[float, float, float]]:
    if np is None:
        return None
    point_world = np.array(
        [float(world_location.x), float(world_location.y), float(world_location.z), 1.0],
        dtype=np.float64,
    )
    point_camera = world_to_camera @ point_world
    point_camera = np.array(
        [point_camera[1], -point_camera[2], point_camera[0]],
        dtype=np.float64,
    )
    depth = float(point_camera[2])
    if depth <= 1e-3:
        return None
    point_img = intrinsics @ point_camera
    if abs(float(point_img[2])) <= 1e-6:
        return None
    u = float(point_img[0] / point_img[2])
    v = float(point_img[1] / point_img[2])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return u, v, depth


def projected_depth_visibility_ratio(
    projected_points: List[Tuple[float, float, float]],
    depth_map_m: Any,
    image_w: int,
    image_h: int,
) -> Optional[float]:
    if np is None or depth_map_m is None or not projected_points:
        return None
    try:
        depth_array = np.asarray(depth_map_m)
    except Exception:
        return None
    if depth_array.ndim < 2:
        return None

    depth_h = int(depth_array.shape[0])
    depth_w = int(depth_array.shape[1])
    if depth_h <= 0 or depth_w <= 0 or image_w <= 0 or image_h <= 0:
        return None

    visible = 0
    checked = 0
    for u, v, expected_depth in projected_points:
        if not (
            math.isfinite(float(u))
            and math.isfinite(float(v))
            and math.isfinite(float(expected_depth))
            and expected_depth > 0.0
        ):
            continue
        if u < 0.0 or v < 0.0 or u >= float(image_w) or v >= float(image_h):
            continue

        px = int(round(float(u) * float(depth_w - 1) / max(1.0, float(image_w - 1))))
        py = int(round(float(v) * float(depth_h - 1) / max(1.0, float(image_h - 1))))
        px = max(0, min(depth_w - 1, px))
        py = max(0, min(depth_h - 1, py))
        try:
            measured_depth = float(depth_array[py, px])
        except Exception:
            continue
        if not math.isfinite(measured_depth) or measured_depth <= 0.0:
            continue

        checked += 1
        tolerance_m = max(1.5, 0.10 * float(expected_depth))
        if float(expected_depth) <= measured_depth + tolerance_m:
            visible += 1

    if checked <= 0:
        return None
    return float(visible) / float(checked)


def project_vertices_to_box(
    vertices: Sequence[Any],
    center_location: Optional[Any],
    world_to_camera: Any,
    intrinsics: Any,
    image_w: int,
    image_h: int,
    filters: GroundTruthFilterConfig,
    depth_map_m: Any = None,
) -> Optional[ProjectedBox]:
    projected_points: List[Tuple[float, float, float]] = []
    for vertex in vertices:
        projected = project_world_location_to_image(vertex, world_to_camera, intrinsics)
        if projected is not None:
            projected_points.append(projected)

    if center_location is not None:
        projected = project_world_location_to_image(center_location, world_to_camera, intrinsics)
        if projected is not None:
            projected_points.append(projected)

    if len(projected_points) < 2:
        return None

    xs = [point[0] for point in projected_points]
    ys = [point[1] for point in projected_points]
    raw_x1 = min(xs)
    raw_y1 = min(ys)
    raw_x2 = max(xs)
    raw_y2 = max(ys)
    raw_area = max(0.0, raw_x2 - raw_x1) * max(0.0, raw_y2 - raw_y1)

    x1 = max(0.0, raw_x1)
    y1 = max(0.0, raw_y1)
    x2 = min(float(image_w - 1), raw_x2)
    y2 = min(float(image_h - 1), raw_y2)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    if width < filters.min_bbox_dim_px or height < filters.min_bbox_dim_px:
        return None
    if width * height < filters.min_bbox_area_px:
        return None
    aspect = max(width, height) / max(min(width, height), 1.0)
    if aspect > filters.max_bbox_aspect_ratio:
        return None

    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    if cx < 0.0 or cx >= float(image_w) or cy < 0.0 or cy >= float(image_h):
        return None

    if raw_area > 1.0:
        visible_area_ratio = (width * height) / raw_area
        if visible_area_ratio < filters.min_visible_area_ratio:
            return None

    depth_visibility = projected_depth_visibility_ratio(
        projected_points=projected_points,
        depth_map_m=depth_map_m,
        image_w=image_w,
        image_h=image_h,
    )
    if depth_visibility is not None and depth_visibility < filters.min_depth_visible_ratio:
        return None

    return ProjectedBox(bbox=(x1, y1, width, height), projected_points=projected_points)


def traffic_light_projection_filters(eval_cfg: DetectionEvalConfig) -> GroundTruthFilterConfig:
    base = eval_cfg.gt_filters
    return GroundTruthFilterConfig(
        max_distance_m=base.max_distance_m,
        min_bbox_dim_px=max(1, int(eval_cfg.traffic_light_min_bbox_dim_px)),
        min_bbox_area_px=max(1, int(eval_cfg.traffic_light_min_bbox_area_px)),
        min_visible_area_ratio=float(eval_cfg.traffic_light_min_visible_area_ratio),
        min_depth_visible_ratio=base.min_depth_visible_ratio,
        max_bbox_aspect_ratio=max(1.0, float(eval_cfg.traffic_light_max_bbox_aspect_ratio)),
    )


def infer_actor_gt_class(actor: Any) -> Optional[str]:
    type_id = str(getattr(actor, "type_id", "")).lower()
    if type_id.startswith("walker."):
        return "pedestrian"
    if type_id.startswith("vehicle."):
        two_wheel_tokens = (
            "bike",
            "bicycle",
            "motorcycle",
            "motorbike",
            "vespa",
            "yamaha",
            "harley",
            "kawasaki",
            "diamondback",
            "gazelle",
            "bh",
            "crossbike",
            "cyclist",
            "yzf",
            "ninja",
            "zx125",
            "low_rider",
            "century",
            "omafiets",
        )
        if any(token in type_id for token in two_wheel_tokens):
            return "two_wheeler"
        return "vehicle"
    return None


def traffic_light_state_class(traffic_light_actor: Any) -> Optional[str]:
    if traffic_light_actor is None or carla is None:
        return None
    try:
        state = traffic_light_actor.get_state()
    except Exception:
        return None
    if state == carla.TrafficLightState.Red:
        return "traffic_light_red"
    if state == carla.TrafficLightState.Green:
        return "traffic_light_green"
    return None


def is_traffic_light_head_bbox(level_bbox: Any, eval_cfg: DetectionEvalConfig) -> bool:
    try:
        extent = level_bbox.extent
        dims = [
            abs(float(extent.x)) * 2.0,
            abs(float(extent.y)) * 2.0,
            abs(float(extent.z)) * 2.0,
        ]
    except Exception:
        return True
    max_dim = max(dims)
    volume = max(0.0, dims[0] * dims[1] * dims[2])
    if max_dim > float(eval_cfg.traffic_light_head_max_extent_m):
        return False
    if volume > float(eval_cfg.traffic_light_head_max_volume_m3):
        return False
    return True


def nearest_traffic_light_actor(level_bbox: Any, traffic_lights: Sequence[Any], max_radius_m: float) -> Optional[Any]:
    try:
        bbox_location = level_bbox.location
    except Exception:
        return None

    best_actor = None
    best_distance = float("inf")
    for actor in traffic_lights:
        try:
            distance = float(bbox_location.distance(actor.get_location()))
        except Exception:
            continue
        if distance < best_distance:
            best_distance = distance
            best_actor = actor
    if best_distance <= float(max_radius_m):
        return best_actor
    return None


def level_bbox_vertices(level_bbox: Any) -> List[Any]:
    if carla is None:
        return []
    try:
        return list(level_bbox.get_world_vertices(carla.Transform()))
    except Exception:
        pass
    try:
        return list(level_bbox.get_local_vertices())
    except Exception:
        return []


def generate_ground_truth(
    world: Any,
    camera: Any,
    ego_vehicle: Any,
    depth_map_m: Any,
    frame_id: int,
    image_size: Tuple[int, int],
    carla_cfg: CarlaEvalConfig,
    eval_cfg: DetectionEvalConfig,
) -> List[DetectionRecord]:
    if np is None:
        return []
    image_w, image_h = image_size
    intrinsics = camera_intrinsics_matrix(image_w, image_h, carla_cfg.camera_fov)
    if intrinsics is None:
        return []

    try:
        world_to_camera = np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float64)
    except Exception:
        return []

    try:
        actors = world.get_actors()
    except Exception:
        return []

    try:
        ego_id = int(ego_vehicle.id)
    except Exception:
        ego_id = -1
    try:
        ego_location = ego_vehicle.get_location()
    except Exception:
        ego_location = None

    eval_class_set = set(eval_cfg.eval_classes)
    records: List[DetectionRecord] = []

    for actor in actors:
        try:
            actor_id = int(actor.id)
        except Exception:
            continue
        if actor_id == ego_id:
            continue

        class_name = infer_actor_gt_class(actor)
        if class_name is None or class_name not in eval_class_set:
            continue

        if ego_location is not None:
            try:
                if float(ego_location.distance(actor.get_location())) > eval_cfg.gt_filters.max_distance_m:
                    continue
            except Exception:
                pass

        bbox_3d = getattr(actor, "bounding_box", None)
        if bbox_3d is None:
            continue

        try:
            vertices = list(bbox_3d.get_world_vertices(actor.get_transform()))
            center = actor.get_location()
        except Exception:
            continue

        projected = project_vertices_to_box(
            vertices=vertices,
            center_location=center,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            image_w=image_w,
            image_h=image_h,
            filters=eval_cfg.gt_filters,
            depth_map_m=depth_map_m,
        )
        if projected is None:
            continue

        records.append(
            DetectionRecord(
                frame=frame_id,
                class_name=class_name,
                bbox=projected.bbox,
                confidence=1.0,
                object_id=actor_id,
            )
        )

    traffic_light_records = generate_traffic_light_ground_truth(
        world=world,
        actors=actors,
        camera=camera,
        ego_location=ego_location,
        frame_id=frame_id,
        image_size=image_size,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        depth_map_m=depth_map_m,
        eval_cfg=eval_cfg,
    )
    records.extend(record for record in traffic_light_records if record.class_name in eval_class_set)
    return records


def generate_traffic_light_ground_truth(
    world: Any,
    actors: Any,
    camera: Any,
    ego_location: Any,
    frame_id: int,
    image_size: Tuple[int, int],
    world_to_camera: Any,
    intrinsics: Any,
    depth_map_m: Any,
    eval_cfg: DetectionEvalConfig,
) -> List[DetectionRecord]:
    image_w, image_h = image_size
    records: List[DetectionRecord] = []
    tl_filters = traffic_light_projection_filters(eval_cfg)
    try:
        traffic_lights = list(actors.filter("traffic.traffic_light*"))
    except Exception:
        traffic_lights = []

    mode = str(eval_cfg.traffic_light_gt_mode or "level_bbs").lower()
    used_level_bbs = False
    if mode in {"level_bbs", "level_bbs_then_actor", "both"} and carla is not None:
        try:
            level_bbs = list(world.get_level_bbs(carla.CityObjectLabel.TrafficLight))
        except Exception:
            level_bbs = []
        for index, level_bbox in enumerate(level_bbs):
            if not is_traffic_light_head_bbox(level_bbox, eval_cfg):
                continue
            matched_actor = nearest_traffic_light_actor(
                level_bbox,
                traffic_lights,
                eval_cfg.traffic_light_actor_match_radius_m,
            )
            class_name = traffic_light_state_class(matched_actor)
            if class_name is None:
                continue
            if ego_location is not None:
                try:
                    if ego_location.distance(level_bbox.location) > eval_cfg.gt_filters.max_distance_m:
                        continue
                except Exception:
                    pass
            vertices = level_bbox_vertices(level_bbox)
            if not vertices:
                continue
            projected = project_vertices_to_box(
                vertices=vertices,
                center_location=getattr(level_bbox, "location", None),
                world_to_camera=world_to_camera,
                intrinsics=intrinsics,
                image_w=image_w,
                image_h=image_h,
                filters=tl_filters,
                depth_map_m=depth_map_m,
            )
            if projected is None:
                continue
            records.append(
                DetectionRecord(
                    frame=frame_id,
                    class_name=class_name,
                    bbox=projected.bbox,
                    confidence=1.0,
                    object_id=1_000_000 + index,
                )
            )
            used_level_bbs = True

    should_use_actor_bbox = mode in {"actor_bbox", "both"} or (
        eval_cfg.traffic_light_actor_fallback and not used_level_bbs
    )
    if not should_use_actor_bbox:
        return records

    for actor in traffic_lights:
        class_name = traffic_light_state_class(actor)
        if class_name is None:
            continue
        if ego_location is not None:
            try:
                if ego_location.distance(actor.get_location()) > eval_cfg.gt_filters.max_distance_m:
                    continue
            except Exception:
                pass
        bbox_3d = getattr(actor, "bounding_box", None)
        if bbox_3d is None:
            continue
        try:
            vertices = list(bbox_3d.get_world_vertices(actor.get_transform()))
            center = actor.get_location()
            object_id = int(actor.id)
        except Exception:
            continue
        projected = project_vertices_to_box(
            vertices=vertices,
            center_location=center,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            image_w=image_w,
            image_h=image_h,
            filters=tl_filters,
            depth_map_m=depth_map_m,
        )
        if projected is None:
            continue
        records.append(
            DetectionRecord(
                frame=frame_id,
                class_name=class_name,
                bbox=projected.bbox,
                confidence=1.0,
                object_id=object_id,
            )
        )
    return records


def normalize_detector_output(
    detections: Iterable[Dict[str, Any]],
    frame_id: int,
    eval_classes: Sequence[str],
) -> List[DetectionRecord]:
    eval_class_set = set(eval_classes)
    records: List[DetectionRecord] = []
    for det_idx, det in enumerate(detections):
        raw_class = det.get("class", det.get("class_name", ""))
        class_name = normalize_class_name(raw_class)
        if class_name not in eval_class_set:
            continue

        box = det.get("bbox", det.get("box"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(det.get("conf", det.get("confidence", 0.0)))
        if not math.isfinite(confidence):
            confidence = 0.0
        records.append(
            DetectionRecord(
                frame=frame_id,
                class_name=class_name,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                confidence=confidence,
                object_id=det_idx + 1,
            )
        )
    return records


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def flatten_records(records_by_frame: Dict[int, List[DetectionRecord]]) -> List[DetectionRecord]:
    records: List[DetectionRecord] = []
    for frame_id in sorted(records_by_frame):
        records.extend(records_by_frame[frame_id])
    return records


def match_counts_at_threshold(
    predictions_by_frame: Dict[int, List[DetectionRecord]],
    ground_truth_by_frame: Dict[int, List[DetectionRecord]],
    eval_classes: Sequence[str],
    iou_threshold: float,
    class_iou_thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    class_stats: Dict[str, Dict[str, Any]] = {
        class_name: {
            "gt_detections": 0,
            "pred_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "iou_sum": 0.0,
        }
        for class_name in eval_classes
    }

    frames = sorted(set(predictions_by_frame) | set(ground_truth_by_frame))
    for frame_id in frames:
        frame_gt = ground_truth_by_frame.get(frame_id, [])
        frame_pred = predictions_by_frame.get(frame_id, [])

        for class_name in eval_classes:
            gt_cls = [record for record in frame_gt if record.class_name == class_name]
            pred_cls = [record for record in frame_pred if record.class_name == class_name]
            stat = class_stats[class_name]
            stat["gt_detections"] += len(gt_cls)
            stat["pred_detections"] += len(pred_cls)

            threshold = float(class_iou_thresholds.get(class_name, iou_threshold))
            candidates: List[Tuple[float, int, int]] = []
            for gt_idx, gt_record in enumerate(gt_cls):
                for pred_idx, pred_record in enumerate(pred_cls):
                    iou_value = bbox_iou(gt_record.bbox, pred_record.bbox)
                    if iou_value >= threshold:
                        candidates.append((iou_value, gt_idx, pred_idx))
            candidates.sort(reverse=True, key=lambda item: item[0])

            matched_gt = set()
            matched_pred = set()
            for iou_value, gt_idx, pred_idx in candidates:
                if gt_idx in matched_gt or pred_idx in matched_pred:
                    continue
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                stat["true_positives"] += 1
                stat["iou_sum"] += float(iou_value)

            stat["false_positives"] += len(pred_cls) - len(matched_pred)
            stat["false_negatives"] += len(gt_cls) - len(matched_gt)

    for class_name, stat in class_stats.items():
        tp = int(stat["true_positives"])
        fp = int(stat["false_positives"])
        fn = int(stat["false_negatives"])
        gt_count = int(stat["gt_detections"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / gt_count if gt_count > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        stat["precision"] = precision
        stat["recall"] = recall
        stat["f1"] = f1
        stat["mean_iou"] = float(stat["iou_sum"]) / tp if tp > 0 else 0.0
    return class_stats


def all_point_interpolated_ap(recalls: List[float], precisions: List[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return float(ap)


def compute_ap_for_class(
    predictions_by_frame: Dict[int, List[DetectionRecord]],
    ground_truth_by_frame: Dict[int, List[DetectionRecord]],
    class_name: str,
    iou_threshold: float,
) -> Tuple[float, int]:
    gt_by_frame: Dict[int, List[DetectionRecord]] = {}
    for frame_id, records in ground_truth_by_frame.items():
        filtered = [record for record in records if record.class_name == class_name]
        if filtered:
            gt_by_frame[frame_id] = filtered
    total_gt = sum(len(records) for records in gt_by_frame.values())
    if total_gt <= 0:
        return 0.0, 0

    predictions = [
        record
        for records in predictions_by_frame.values()
        for record in records
        if record.class_name == class_name
    ]
    predictions.sort(key=lambda record: record.confidence, reverse=True)
    if not predictions:
        return 0.0, total_gt

    matched_gt_by_frame: Dict[int, set[int]] = {frame_id: set() for frame_id in gt_by_frame}
    tp_values: List[int] = []
    fp_values: List[int] = []

    for pred in predictions:
        frame_gt = gt_by_frame.get(pred.frame, [])
        best_iou = 0.0
        best_gt_idx = -1
        matched_for_frame = matched_gt_by_frame.setdefault(pred.frame, set())
        for gt_idx, gt_record in enumerate(frame_gt):
            if gt_idx in matched_for_frame:
                continue
            iou_value = bbox_iou(pred.bbox, gt_record.bbox)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= float(iou_threshold):
            matched_for_frame.add(best_gt_idx)
            tp_values.append(1)
            fp_values.append(0)
        else:
            tp_values.append(0)
            fp_values.append(1)

    cum_tp = 0
    cum_fp = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for tp, fp in zip(tp_values, fp_values):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / total_gt)
        precisions.append(cum_tp / max(1, cum_tp + cum_fp))

    return all_point_interpolated_ap(recalls, precisions), total_gt


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(pct) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compute_detection_metrics(
    predictions_by_frame: Dict[int, List[DetectionRecord]],
    ground_truth_by_frame: Dict[int, List[DetectionRecord]],
    eval_classes: Sequence[str],
    iou_threshold: float,
    class_iou_thresholds: Dict[str, float],
    inference_times_ms: Sequence[float],
    frames_processed: int,
) -> Dict[str, Any]:
    class_stats = match_counts_at_threshold(
        predictions_by_frame=predictions_by_frame,
        ground_truth_by_frame=ground_truth_by_frame,
        eval_classes=eval_classes,
        iou_threshold=iou_threshold,
        class_iou_thresholds=class_iou_thresholds,
    )

    valid_ap50_values: List[float] = []
    valid_map_values: List[float] = []
    for class_name in eval_classes:
        ap50, total_gt = compute_ap_for_class(
            predictions_by_frame,
            ground_truth_by_frame,
            class_name,
            iou_threshold=0.50,
        )
        ap_values = []
        for ap_threshold in AP_IOU_THRESHOLDS:
            ap_value, _ = compute_ap_for_class(
                predictions_by_frame,
                ground_truth_by_frame,
                class_name,
                iou_threshold=ap_threshold,
            )
            ap_values.append(ap_value)
        map_value = sum(ap_values) / len(ap_values) if ap_values else 0.0
        class_stats[class_name]["ap@0.5"] = ap50
        class_stats[class_name]["mAP@0.5:0.95"] = map_value
        if total_gt > 0:
            valid_ap50_values.append(ap50)
            valid_map_values.append(map_value)

    total_gt = sum(int(stat["gt_detections"]) for stat in class_stats.values())
    total_pred = sum(int(stat["pred_detections"]) for stat in class_stats.values())
    total_tp = sum(int(stat["true_positives"]) for stat in class_stats.values())
    total_fp = sum(int(stat["false_positives"]) for stat in class_stats.values())
    total_fn = sum(int(stat["false_negatives"]) for stat in class_stats.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    import statistics
    times = [float(value) for value in inference_times_ms]
    avg_ms = statistics.fmean(times) if times else 0.0
    metrics: Dict[str, Any] = {
        "method": "strict_class_iou_detection",
        "iou_threshold": float(iou_threshold),
        "frames": int(frames_processed),
        "gt_detections": total_gt,
        "pred_detections": total_pred,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP@0.5": sum(valid_ap50_values) / len(valid_ap50_values) if valid_ap50_values else 0.0,
        "mAP@0.5:0.95": sum(valid_map_values) / len(valid_map_values) if valid_map_values else 0.0,
        "avg_inference_ms": avg_ms,
        "p50_inference_ms": percentile(times, 50.0),
        "p95_inference_ms": percentile(times, 95.0),
        "fps": 1000.0 / avg_ms if avg_ms > 0.0 else 0.0,
        "per_class": class_stats,
    }
    return metrics


def write_mot_txt(path: Path, records_by_frame: Dict[int, List[DetectionRecord]], is_gt: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for frame_id in sorted(records_by_frame):
            for row_idx, record in enumerate(records_by_frame[frame_id], start=1):
                object_id = int(record.object_id)
                if object_id < 0:
                    object_id = row_idx
                x, y, w, h = record.bbox
                confidence = 1.0 if is_gt else float(record.confidence)
                file_obj.write(
                    f"{int(record.frame)},{object_id},"
                    f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{confidence:.4f},-1,-1,-1,{record.class_name}\n"
                )


def metric_row_from_stats(scope: str, class_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scope": scope,
        "class": class_name,
        "frames": int(metrics.get("frames", 0)),
        "gt_detections": int(metrics.get("gt_detections", 0)),
        "pred_detections": int(metrics.get("pred_detections", 0)),
        "true_positives": int(metrics.get("true_positives", 0)),
        "false_positives": int(metrics.get("false_positives", 0)),
        "false_negatives": int(metrics.get("false_negatives", 0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "mean_iou": float(metrics.get("mean_iou", 0.0)),
        "mAP@0.5": float(metrics.get("mAP@0.5", metrics.get("ap@0.5", 0.0))),
        "mAP@0.5:0.95": float(metrics.get("mAP@0.5:0.95", 0.0)),
        "avg_inference_ms": float(metrics.get("avg_inference_ms", 0.0)),
        "p50_inference_ms": float(metrics.get("p50_inference_ms", 0.0)),
        "p95_inference_ms": float(metrics.get("p95_inference_ms", 0.0)),
        "fps": float(metrics.get("fps", 0.0)),
    }


SUMMARY_FIELDS = [
    "scope",
    "class",
    "frames",
    "gt_detections",
    "pred_detections",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_iou",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "avg_inference_ms",
    "p50_inference_ms",
    "p95_inference_ms",
    "fps",
]


def write_model_metrics_outputs(model_dir: Path, model_name: str, metrics: Dict[str, Any], eval_classes: Sequence[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    csv_path = model_dir / "detection_metrics_summary.csv"
    txt_path = model_dir / "detection_metrics_summary.txt"

    overall_row = metric_row_from_stats("overall", "__overall__", metrics)
    rows = [overall_row]
    per_class = metrics.get("per_class", {})
    for class_name in eval_classes:
        class_metrics = dict(per_class.get(class_name, {}))
        class_metrics["frames"] = metrics.get("frames", 0)
        rows.append(metric_row_from_stats("class", class_name, class_metrics))

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        f"Detection Metrics Summary: {model_name}",
        f"method: {metrics.get('method', '')}",
        f"iou_threshold: {float(metrics.get('iou_threshold', 0.0)):.3f}",
        f"frames: {int(metrics.get('frames', 0))}",
        f"gt_detections: {int(metrics.get('gt_detections', 0))}",
        f"pred_detections: {int(metrics.get('pred_detections', 0))}",
        f"precision: {float(metrics.get('precision', 0.0)):.6f}",
        f"recall: {float(metrics.get('recall', 0.0)):.6f}",
        f"f1: {float(metrics.get('f1', 0.0)):.6f}",
        f"mAP@0.5: {float(metrics.get('mAP@0.5', 0.0)):.6f}",
        f"mAP@0.5:0.95: {float(metrics.get('mAP@0.5:0.95', 0.0)):.6f}",
        f"avg_inference_ms: {float(metrics.get('avg_inference_ms', 0.0)):.3f}",
        "",
        "Per Class",
    ]
    for class_name in eval_classes:
        stat = per_class.get(class_name, {})
        lines.append(
            "class={class_name} | gt={gt} | pred={pred} | tp={tp} | fp={fp} | fn={fn} | "
            "precision={precision:.6f} | recall={recall:.6f} | f1={f1:.6f} | "
            "ap50={ap50:.6f} | map={map_value:.6f}".format(
                class_name=class_name,
                gt=int(stat.get("gt_detections", 0)),
                pred=int(stat.get("pred_detections", 0)),
                tp=int(stat.get("true_positives", 0)),
                fp=int(stat.get("false_positives", 0)),
                fn=int(stat.get("false_negatives", 0)),
                precision=float(stat.get("precision", 0.0)),
                recall=float(stat.get("recall", 0.0)),
                f1=float(stat.get("f1", 0.0)),
                ap50=float(stat.get("ap@0.5", 0.0)),
                map_value=float(stat.get("mAP@0.5:0.95", 0.0)),
            )
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_best_model(metrics_by_model: Dict[str, Dict[str, Any]], metric_name: str, lower_is_better: bool = False) -> str:
    candidates: List[Tuple[float, str]] = []
    for model_name, metrics in metrics_by_model.items():
        try:
            value = float(metrics.get(metric_name, 0.0))
        except Exception:
            continue
        candidates.append((value, model_name))
    if not candidates:
        return ""
    if lower_is_better:
        return min(candidates, key=lambda item: item[0])[1]
    return max(candidates, key=lambda item: item[0])[1]


def write_comparison_outputs(
    out_dir: Path,
    metrics_by_model: Dict[str, Dict[str, Any]],
    eval_classes: Sequence[str],
) -> None:
    comparison_csv = out_dir / "detection_comparison.csv"
    comparison_txt = out_dir / "detection_comparison.txt"
    per_class_csv = out_dir / "detection_comparison_per_class.csv"

    model_names = list(metrics_by_model)
    comparison_metrics = [
        "frames",
        "gt_detections",
        "pred_detections",
        "precision",
        "recall",
        "f1",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "avg_inference_ms",
        "p50_inference_ms",
        "p95_inference_ms",
        "fps",
    ]
    lower_is_better_metrics = {"avg_inference_ms", "p50_inference_ms", "p95_inference_ms"}
    no_best_metrics = {"frames", "gt_detections", "pred_detections"}

    with comparison_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["metric"] + model_names + ["best_model"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in comparison_metrics:
            row: Dict[str, Any] = {"metric": metric_name}
            for model_name in model_names:
                row[model_name] = metrics_by_model[model_name].get(metric_name, "")
            row["best_model"] = (
                ""
                if metric_name in no_best_metrics
                else pick_best_model(
                    metrics_by_model,
                    metric_name,
                    lower_is_better=metric_name in lower_is_better_metrics,
                )
            )
            writer.writerow(row)

    per_class_fields = ["class", "metric"] + model_names + ["best_model"]
    with per_class_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=per_class_fields)
        writer.writeheader()
        for class_name in eval_classes:
            for metric_name in [
                "gt_detections",
                "pred_detections",
                "precision",
                "recall",
                "f1",
                "ap@0.5",
                "mAP@0.5:0.95",
            ]:
                row = {"class": class_name, "metric": metric_name}
                temp_metrics: Dict[str, Dict[str, Any]] = {}
                for model_name in model_names:
                    per_class = metrics_by_model[model_name].get("per_class", {})
                    value = per_class.get(class_name, {}).get(metric_name, "")
                    row[model_name] = value
                    temp_metrics[model_name] = {metric_name: value}
                row["best_model"] = (
                    ""
                    if metric_name in {"gt_detections", "pred_detections"}
                    else pick_best_model(temp_metrics, metric_name)
                )
                writer.writerow(row)

    lines = ["Detection Model Comparison", ""]
    for metric_name in comparison_metrics:
        values = []
        for model_name in model_names:
            value = metrics_by_model[model_name].get(metric_name, "")
            if isinstance(value, float):
                values.append(f"{model_name}={value:.6f}")
            else:
                values.append(f"{model_name}={value}")
        best = ""
        if metric_name not in no_best_metrics:
            best = " | best=" + pick_best_model(
                metrics_by_model,
                metric_name,
                lower_is_better=metric_name in lower_is_better_metrics,
            )
        lines.append(f"{metric_name}: " + " | ".join(values) + best)
    lines.extend(["", f"CSV: {comparison_csv}", f"Per-class CSV: {per_class_csv}"])
    comparison_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


ALL_RUNS_SUMMARY_FILENAME = "all_runs_summary.csv"

ALL_RUNS_BASE_FIELDS = [
    "run_timestamp",
    "run_dir",
    "dry_run",
    "map_name",
    "weather",
    "npc_vehicles",
    "npc_pedestrians",
    "model_name",
    "model_type",
    "frames",
    "conf_threshold",
    "iou_threshold",
    "gt_detections",
    "pred_detections",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "avg_inference_ms",
    "p50_inference_ms",
    "p95_inference_ms",
    "fps",
]


def _all_runs_summary_fields(eval_classes: Sequence[str]) -> List[str]:
    """Build ordered field list including per-class metric columns."""
    fields = list(ALL_RUNS_BASE_FIELDS)
    for class_name in eval_classes:
        safe = safe_name(class_name)
        fields.append(f"{safe}_precision")
        fields.append(f"{safe}_recall")
        fields.append(f"{safe}_f1")
        fields.append(f"{safe}_ap50")
        fields.append(f"{safe}_map")
    return fields


def append_all_runs_summary(
    out_dir: Path,
    run_config: RunConfig,
    metrics_by_model: Dict[str, Dict[str, Any]],
    frames_processed: int,
    dry_run: bool,
) -> Path:
    """Append one row per model to the cumulative all-runs summary CSV.

    The CSV is placed in the *parent* of the per-run output directory
    (i.e. ``outputs/detection_test_eval/all_runs_summary.csv``) so that
    every run appends to the same file regardless of its timestamp folder.
    """
    summary_path = out_dir.parent / ALL_RUNS_SUMMARY_FILENAME
    eval_classes = run_config.eval_cfg.eval_classes
    fieldnames = _all_runs_summary_fields(eval_classes)

    run_timestamp = out_dir.name

    needs_header = not summary_path.exists()
    if not needs_header:
        try:
            with summary_path.open("r", newline="", encoding="utf-8") as check_file:
                reader = csv.reader(check_file)
                existing_header = next(reader, None)
            if existing_header is not None and list(existing_header) != fieldnames:
                merged = list(existing_header)
                for field_name in fieldnames:
                    if field_name not in merged:
                        merged.append(field_name)
                fieldnames = merged
        except Exception:
            pass

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()

        carla_cfg = run_config.carla_cfg
        total_npc_vehicles = (
            carla_cfg.npc_vehicle_count
            + carla_cfg.npc_bike_count
            + carla_cfg.npc_motorbike_count
        )

        for model_cfg in run_config.eval_cfg.models:
            model_name = model_cfg.name
            metrics = metrics_by_model.get(model_name, {})
            per_class = metrics.get("per_class", {})

            row: Dict[str, Any] = {
                "run_timestamp": run_timestamp,
                "run_dir": str(out_dir),
                "dry_run": str(dry_run),
                "map_name": carla_cfg.map_name,
                "weather": carla_cfg.weather_preset,
                "npc_vehicles": total_npc_vehicles,
                "npc_pedestrians": carla_cfg.npc_pedestrian_count,
                "model_name": model_name,
                "model_type": model_cfg.model_type,
                "frames": int(metrics.get("frames", frames_processed)),
                "conf_threshold": float(run_config.eval_cfg.conf_threshold),
                "iou_threshold": float(run_config.eval_cfg.iou_threshold),
                "gt_detections": int(metrics.get("gt_detections", 0)),
                "pred_detections": int(metrics.get("pred_detections", 0)),
                "true_positives": int(metrics.get("true_positives", 0)),
                "false_positives": int(metrics.get("false_positives", 0)),
                "false_negatives": int(metrics.get("false_negatives", 0)),
                "precision": f"{float(metrics.get('precision', 0.0)):.6f}",
                "recall": f"{float(metrics.get('recall', 0.0)):.6f}",
                "f1": f"{float(metrics.get('f1', 0.0)):.6f}",
                "mAP@0.5": f"{float(metrics.get('mAP@0.5', 0.0)):.6f}",
                "mAP@0.5:0.95": f"{float(metrics.get('mAP@0.5:0.95', 0.0)):.6f}",
                "avg_inference_ms": f"{float(metrics.get('avg_inference_ms', 0.0)):.3f}",
                "p50_inference_ms": f"{float(metrics.get('p50_inference_ms', 0.0)):.3f}",
                "p95_inference_ms": f"{float(metrics.get('p95_inference_ms', 0.0)):.3f}",
                "fps": f"{float(metrics.get('fps', 0.0)):.1f}",
            }

            for class_name in eval_classes:
                safe = safe_name(class_name)
                cls_stats = per_class.get(class_name, {})
                row[f"{safe}_precision"] = f"{float(cls_stats.get('precision', 0.0)):.6f}"
                row[f"{safe}_recall"] = f"{float(cls_stats.get('recall', 0.0)):.6f}"
                row[f"{safe}_f1"] = f"{float(cls_stats.get('f1', 0.0)):.6f}"
                row[f"{safe}_ap50"] = f"{float(cls_stats.get('ap@0.5', 0.0)):.6f}"
                row[f"{safe}_map"] = f"{float(cls_stats.get('mAP@0.5:0.95', 0.0)):.6f}"

            writer.writerow(row)

    logging.info(
        "Appended %d model rows to cumulative summary: %s",
        len(run_config.eval_cfg.models),
        summary_path,
    )
    return summary_path


def write_run_metadata(
    out_dir: Path,
    run_config: RunConfig,
    frames_processed: int,
    dry_run: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    metadata = {
        "dry_run": bool(dry_run),
        "frames_processed": int(frames_processed),
        "source_config_path": str(run_config.source_config_path),
        "carla": {
            "host": run_config.carla_cfg.host,
            "port": run_config.carla_cfg.port,
            "tm_port": run_config.carla_cfg.tm_port,
            "map": run_config.carla_cfg.map_name,
            "sync": run_config.carla_cfg.sync,
            "fixed_delta": run_config.carla_cfg.fixed_delta,
            "weather_preset": run_config.carla_cfg.weather_preset,
        },
        "camera": {
            "width": run_config.carla_cfg.camera_width,
            "height": run_config.carla_cfg.camera_height,
            "fov": run_config.carla_cfg.camera_fov,
            "mount": {"x": 1.5, "y": 0.0, "z": 2.2, "pitch": -8.0},
        },
        "test_detection": {
            "models": [
                {"name": model.name, "path": str(model.path), "type": model.model_type}
                for model in run_config.eval_cfg.models
            ],
            "conf_threshold": run_config.eval_cfg.conf_threshold,
            "iou_threshold": run_config.eval_cfg.iou_threshold,
            "class_iou_thresholds": run_config.eval_cfg.class_iou_thresholds,
            "max_frames": run_config.eval_cfg.max_frames,
            "inference_imgsz": run_config.eval_cfg.inference_imgsz,
            "eval_classes": run_config.eval_cfg.eval_classes,
            "traffic_light_gt_mode": run_config.eval_cfg.traffic_light_gt_mode,
            "gt_filters": run_config.eval_cfg.gt_filters.__dict__,
        },
    }
    if extra:
        metadata.update(extra)
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def update_run_status(
    out_dir: Path,
    run_config: RunConfig,
    frames_processed: int,
    stage: str,
    status: str = "running",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {"status": status, "stage": stage}
    if extra:
        payload.update(extra)
    write_run_metadata(
        out_dir=out_dir,
        run_config=run_config,
        frames_processed=frames_processed,
        dry_run=False,
        extra=payload,
    )


def attach_run_file_logger(out_dir: Path) -> logging.Handler:
    log_path = out_dir / "run.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)
    return handler


def write_failure_status(
    out_dir: Path,
    run_config: RunConfig,
    frames_processed: int,
    error: str,
    stage: str,
    skipped_sensor_frames: int = 0,
) -> None:
    """Persist a readable failure marker when a full run stops before metrics."""
    message = (
        "Detection evaluation stopped before metrics were written.\n"
        f"stage: {stage}\n"
        f"frames_processed: {int(frames_processed)}\n"
        f"skipped_sensor_frames: {int(skipped_sensor_frames)}\n"
        f"error: {error}\n"
    )
    (out_dir / "run_failure.txt").write_text(message, encoding="utf-8")
    write_run_metadata(
        out_dir=out_dir,
        run_config=run_config,
        frames_processed=frames_processed,
        dry_run=False,
        extra={
            "status": "failed",
            "failure_stage": stage,
            "partial_error": error,
            "skipped_sensor_frames": int(skipped_sensor_frames),
            "note": "No metrics were written because no CARLA frames were processed.",
        },
    )


def load_detectors(
    eval_cfg: DetectionEvalConfig,
    strict_models: bool = True,
    camera_fov_deg: float = 90.0,
) -> Dict[str, Any]:
    if YoloDetector is None:
        raise RuntimeError("Cannot import YoloDetector. Install ultralytics and check core_perception/yolo_detector.py.")
    detectors: Dict[str, Any] = {}
    for model_cfg in eval_cfg.models:
        if not model_cfg.path.exists():
            message = f"Model file not found for {model_cfg.name}: {model_cfg.path}"
            if strict_models:
                raise FileNotFoundError(message)
            logging.warning("%s", message)
            continue
        detector = YoloDetector(
            str(model_cfg.path),
            conf_threshold=eval_cfg.conf_threshold,
            display_classes=eval_cfg.eval_classes,
            inference_imgsz=eval_cfg.inference_imgsz,
            camera_fov_deg=float(camera_fov_deg),
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
        )
        detectors[model_cfg.name] = detector
        logging.info("Loaded detector: %s -> %s", model_cfg.name, model_cfg.path)
    if strict_models and len(detectors) != len(eval_cfg.models):
        raise RuntimeError(f"Loaded {len(detectors)}/{len(eval_cfg.models)} detectors.")
    return detectors


def write_all_outputs(
    out_dir: Path,
    run_config: RunConfig,
    ground_truth_by_frame: Dict[int, List[DetectionRecord]],
    predictions_by_model: Dict[str, Dict[int, List[DetectionRecord]]],
    inference_times_by_model: Dict[str, List[float]],
    frames_processed: int,
    dry_run: bool,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    eval_cfg = run_config.eval_cfg
    write_mot_txt(out_dir / "ground_truth.txt", ground_truth_by_frame, is_gt=True)

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for model_cfg in eval_cfg.models:
        model_name = model_cfg.name
        model_dir = out_dir / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_predictions = predictions_by_model.get(model_name, {})
        write_mot_txt(model_dir / "predictions.txt", model_predictions, is_gt=False)
        metrics = compute_detection_metrics(
            predictions_by_frame=model_predictions,
            ground_truth_by_frame=ground_truth_by_frame,
            eval_classes=eval_cfg.eval_classes,
            iou_threshold=eval_cfg.iou_threshold,
            class_iou_thresholds=eval_cfg.class_iou_thresholds,
            inference_times_ms=inference_times_by_model.get(model_name, []),
            frames_processed=frames_processed,
        )
        metrics_by_model[model_name] = metrics
        write_model_metrics_outputs(model_dir, model_name, metrics, eval_cfg.eval_classes)

    write_comparison_outputs(out_dir, metrics_by_model, eval_cfg.eval_classes)
    write_run_metadata(
        out_dir=out_dir,
        run_config=run_config,
        frames_processed=frames_processed,
        dry_run=dry_run,
        extra=extra_metadata,
    )
    append_all_runs_summary(
        out_dir=out_dir,
        run_config=run_config,
        metrics_by_model=metrics_by_model,
        frames_processed=frames_processed,
        dry_run=dry_run,
    )
    return metrics_by_model


def make_synthetic_eval_data(
    run_config: RunConfig,
) -> Tuple[Dict[int, List[DetectionRecord]], Dict[str, Dict[int, List[DetectionRecord]]], Dict[str, List[float]]]:
    gt = {
        1: [
            DetectionRecord(1, "vehicle", (10, 10, 40, 30), 1.0, 101),
            DetectionRecord(1, "pedestrian", (120, 40, 18, 42), 1.0, 102),
        ],
        2: [
            DetectionRecord(2, "two_wheeler", (60, 50, 30, 30), 1.0, 201),
            DetectionRecord(2, "traffic_light_red", (300, 20, 18, 32), 1.0, 202),
        ],
        3: [
            DetectionRecord(3, "traffic_light_green", (310, 25, 18, 32), 1.0, 301),
        ],
    }
    predictions_by_model: Dict[str, Dict[int, List[DetectionRecord]]] = {}
    times_by_model: Dict[str, List[float]] = {}
    for index, model_cfg in enumerate(run_config.eval_cfg.models):
        offset = float(index)
        predictions_by_model[model_cfg.name] = {
            1: [
                DetectionRecord(1, "vehicle", (10 + offset, 10, 40, 30), 0.95, 1),
                DetectionRecord(1, "pedestrian", (122, 40, 18, 42), 0.90, 2),
            ],
            2: [
                DetectionRecord(2, "two_wheeler", (60, 51 + offset, 30, 30), 0.88, 3),
                DetectionRecord(2, "traffic_light_red", (300, 20, 18, 32), 0.80, 4),
            ],
            3: [
                DetectionRecord(3, "traffic_light_green", (310 + offset, 25, 18, 32), 0.83, 5),
            ],
        }
        if index == 0:
            predictions_by_model[model_cfg.name][3].append(
                DetectionRecord(3, "vehicle", (500, 200, 20, 20), 0.30, 6)
            )
        times_by_model[model_cfg.name] = [12.0 + index * 2.0, 13.0 + index * 2.0, 11.5 + index * 2.0]
    return gt, predictions_by_model, times_by_model


def run_dry_run(run_config: RunConfig, strict_models: bool) -> Path:
    out_dir = make_output_dir(run_config.eval_cfg.output_dir, dry_run=True)
    gt, predictions_by_model, times_by_model = make_synthetic_eval_data(run_config)
    model_smoke: Dict[str, Any] = {}

    if np is not None and run_config.eval_cfg.models:
        existing_models = [model for model in run_config.eval_cfg.models if model.path.exists()]
        missing_models = [model for model in run_config.eval_cfg.models if not model.path.exists()]
        for model in missing_models:
            message = f"Dry-run model smoke skipped missing file: {model.path}"
            if strict_models:
                raise FileNotFoundError(message)
            logging.warning("%s", message)
            model_smoke[model.name] = {"status": "missing", "path": str(model.path)}
        if existing_models:
            detectors = load_detectors(
                DetectionEvalConfig(
                    models=existing_models,
                    conf_threshold=run_config.eval_cfg.conf_threshold,
                    iou_threshold=run_config.eval_cfg.iou_threshold,
                    class_iou_thresholds=run_config.eval_cfg.class_iou_thresholds,
                    max_frames=run_config.eval_cfg.max_frames,
                    inference_imgsz=run_config.eval_cfg.inference_imgsz,
                    eval_classes=run_config.eval_cfg.eval_classes,
                ),
                strict_models=strict_models,
                camera_fov_deg=run_config.carla_cfg.camera_fov,
            )
            fake_frame = np.zeros(
                (run_config.carla_cfg.camera_height, run_config.carla_cfg.camera_width, 3),
                dtype=np.uint8,
            )
            for model_name, detector in detectors.items():
                try:
                    warmup = getattr(detector, "warmup", None)
                    if callable(warmup):
                        warmup(run_config.carla_cfg.camera_width, run_config.carla_cfg.camera_height)
                    detections = detector.detect(fake_frame)
                    model_smoke[model_name] = {"status": "ok", "detections": len(detections)}
                except Exception as exc:
                    if strict_models:
                        raise
                    logging.warning("Dry-run detector smoke failed for %s: %s", model_name, exc)
                    model_smoke[model_name] = {"status": "failed", "error": str(exc)}

    write_all_outputs(
        out_dir=out_dir,
        run_config=run_config,
        ground_truth_by_frame=gt,
        predictions_by_model=predictions_by_model,
        inference_times_by_model=times_by_model,
        frames_processed=3,
        dry_run=True,
        extra_metadata={"model_smoke": model_smoke},
    )
    logging.info("Dry-run finished. Outputs: %s", out_dir)
    return out_dir


def run_full_evaluation(run_config: RunConfig, skip_warmup: bool) -> Path:
    if carla is None:
        raise RuntimeError("Python package 'carla' is required for full evaluation. Use --dry-run to test metrics only.")
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for full evaluation.")
    if CarlaManager is None or SpectatorConfig is None:
        raise RuntimeError("Cannot import CarlaManager from core_control.carla_manager.")

    eval_cfg = run_config.eval_cfg
    carla_cfg = run_config.carla_cfg
    if not eval_cfg.models:
        raise ValueError("No test_detection.models configured.")
    if eval_cfg.max_frames <= 0:
        logging.warning("max_frames <= 0 means run until interrupted.")
    missing_models = [model for model in eval_cfg.models if not model.path.exists()]
    if missing_models:
        missing_text = ", ".join(f"{model.name}={model.path}" for model in missing_models)
        raise FileNotFoundError(f"Missing configured model files: {missing_text}")

    if carla_cfg.seed is not None:
        random.seed(int(carla_cfg.seed))

    out_dir = make_output_dir(eval_cfg.output_dir, dry_run=False)
    file_log_handler = attach_run_file_logger(out_dir)
    stage = "created_output_dir"
    update_run_status(
        out_dir=out_dir,
        run_config=run_config,
        frames_processed=0,
        stage=stage,
        status="starting",
    )
    manager = None
    rgb_camera = None
    depth_camera = None
    driver = None
    frames_dir = out_dir / "frames"
    ground_truth_by_frame: Dict[int, List[DetectionRecord]] = {}
    predictions_by_model: Dict[str, Dict[int, List[DetectionRecord]]] = {
        model.name: {} for model in eval_cfg.models
    }
    inference_times_by_model: Dict[str, List[float]] = {model.name: [] for model in eval_cfg.models}
    frames_processed = 0
    partial_error: Optional[str] = None
    output_write_error: Optional[str] = None
    outputs_written = False
    skipped_sensor_frames = 0
    max_initial_sensor_misses = max(30, int(eval_cfg.log_interval_frames))

    try:
        stage = "creating_carla_manager"
        update_run_status(out_dir, run_config, frames_processed, stage)
        manager = CarlaManager(
            host=carla_cfg.host,
            port=carla_cfg.port,
            tm_port=carla_cfg.tm_port,
            timeout=carla_cfg.timeout,
            map_name=carla_cfg.map_name,
            sync=carla_cfg.sync,
            fixed_delta=carla_cfg.fixed_delta,
            no_rendering=carla_cfg.no_rendering,
            vehicle_filter=carla_cfg.vehicle_filter,
            spawn_point=carla_cfg.spawn_point,
            spectator_cfg=SpectatorConfig(lock_on_spawn=True),
            npc_vehicle_count=carla_cfg.npc_vehicle_count,
            npc_bike_count=carla_cfg.npc_bike_count,
            npc_motorbike_count=carla_cfg.npc_motorbike_count,
            npc_pedestrian_count=carla_cfg.npc_pedestrian_count,
            npc_enable_autopilot=carla_cfg.npc_enable_autopilot,
            seed=carla_cfg.seed,
        )
        stage = "starting_carla_manager"
        update_run_status(out_dir, run_config, frames_processed, stage)
        manager.start()
        if manager.world is None or manager.ego_vehicle is None:
            raise RuntimeError("CARLA manager did not provide world/ego vehicle.")
        stage = "applying_weather"
        update_run_status(out_dir, run_config, frames_processed, stage)
        apply_weather_preset(manager.world, carla_cfg.weather_preset)

        stage = "spawning_sensors"
        update_run_status(out_dir, run_config, frames_processed, stage)
        frame_buffer = SensorFrameBuffer(require_depth=True)
        rgb_camera, depth_camera = spawn_camera_pair(manager.world, manager.ego_vehicle, carla_cfg, frame_buffer)
        stage = "starting_route_driver"
        update_run_status(out_dir, run_config, frames_processed, stage)
        driver = EgoRouteDriver(manager.world, manager.ego_vehicle, carla_cfg, eval_cfg)
        driver.start()

        stage = "loading_detectors"
        update_run_status(out_dir, run_config, frames_processed, stage)
        detectors = load_detectors(
            eval_cfg,
            strict_models=True,
            camera_fov_deg=carla_cfg.camera_fov,
        )
        if not skip_warmup:
            stage = "warming_detectors"
            update_run_status(out_dir, run_config, frames_processed, stage)
            for model_name, detector in detectors.items():
                warmup = getattr(detector, "warmup", None)
                if callable(warmup):
                    t0 = time.perf_counter()
                    warmup(carla_cfg.camera_width, carla_cfg.camera_height)
                    logging.info(
                        "Detector warmup completed: %s %.1f ms",
                        model_name,
                        (time.perf_counter() - t0) * 1000.0,
                    )

        image_size = (int(carla_cfg.camera_width), int(carla_cfg.camera_height))
        if eval_cfg.save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        stage = "processing_frames"
        update_run_status(out_dir, run_config, frames_processed, stage)
        while eval_cfg.max_frames <= 0 or frames_processed < eval_cfg.max_frames:
            manager.tick()
            snapshot_frame = int(manager.world.get_snapshot().frame)
            wait_timeout = max(2.0, float(carla_cfg.fixed_delta) * 10.0)
            frame_bgr, depth_map_m = frame_buffer.wait_for_frame(snapshot_frame, wait_timeout)
            if frame_bgr is None:
                skipped_sensor_frames += 1
                logging.warning("Skipping CARLA frame %d because RGB frame did not arrive.", snapshot_frame)
                driver.run_step()
                if frames_processed <= 0 and skipped_sensor_frames >= max_initial_sensor_misses:
                    raise RuntimeError(
                        "No RGB/depth sensor frames arrived after "
                        f"{skipped_sensor_frames} CARLA ticks. Check CARLA synchronous mode, "
                        "camera spawning, and whether the simulator is ticking."
                    )
                continue

            frames_processed += 1
            skipped_sensor_frames = 0
            eval_frame_id = frames_processed
            update_run_status(
                out_dir,
                run_config,
                frames_processed,
                stage,
                extra={"last_carla_frame": snapshot_frame},
            )

            gt_records = generate_ground_truth(
                world=manager.world,
                camera=rgb_camera,
                ego_vehicle=manager.ego_vehicle,
                depth_map_m=depth_map_m,
                frame_id=eval_frame_id,
                image_size=image_size,
                carla_cfg=carla_cfg,
                eval_cfg=eval_cfg,
            )
            ground_truth_by_frame[eval_frame_id] = gt_records

            for model_name, detector in detectors.items():
                try:
                    t0 = time.perf_counter()
                    raw_detections = detector.detect(frame_bgr)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    logging.debug("Detector %s finished in %.1f ms", model_name, elapsed_ms)
                    inference_times_by_model[model_name].append(elapsed_ms)
                    predictions_by_model[model_name][eval_frame_id] = normalize_detector_output(
                        raw_detections,
                        frame_id=eval_frame_id,
                        eval_classes=eval_cfg.eval_classes,
                    )
                except BaseException:
                    logging.exception("Detector %s failed on frame %d", model_name, eval_frame_id)
                    raise

            if eval_cfg.save_frames and eval_frame_id % eval_cfg.save_frame_stride == 0:
                cv2.imwrite(str(frames_dir / f"frame_{eval_frame_id:06d}.jpg"), frame_bgr)

            driver.run_step()

            if eval_frame_id % eval_cfg.log_interval_frames == 0:
                gt_count = sum(len(rows) for rows in ground_truth_by_frame.values())
                pred_counts = {
                    model_name: sum(len(rows) for rows in model_predictions.values())
                    for model_name, model_predictions in predictions_by_model.items()
                }
                logging.info(
                    "Processed %d frames | GT=%d | predictions=%s",
                    eval_frame_id,
                    gt_count,
                    pred_counts,
                )
                update_run_status(
                    out_dir,
                    run_config,
                    frames_processed,
                    stage,
                    extra={"gt_detections": gt_count, "predictions": pred_counts},
                )
    except KeyboardInterrupt:
        partial_error = "Interrupted by user."
        logging.warning("Interrupted by user. Writing partial results.")
    except BaseException as exc:
        partial_error = str(exc)
        logging.error("Evaluation failed: %s", exc, exc_info=True)
    finally:
        if frames_processed > 0:
            stage = "writing_outputs"
            update_run_status(out_dir, run_config, frames_processed, stage)
            try:
                write_all_outputs(
                    out_dir=out_dir,
                    run_config=run_config,
                    ground_truth_by_frame=ground_truth_by_frame,
                    predictions_by_model=predictions_by_model,
                    inference_times_by_model=inference_times_by_model,
                    frames_processed=frames_processed,
                    dry_run=False,
                    extra_metadata=(
                        {"status": "completed", "partial_error": partial_error}
                        if partial_error
                        else {"status": "completed"}
                    ),
                )
                outputs_written = True
                logging.info("Evaluation outputs written to: %s", out_dir)
            except BaseException as exc:
                output_write_error = str(exc)
                logging.error("Failed to write evaluation outputs: %s", exc, exc_info=True)

        cleanup_errors: List[str] = []
        for label, cleanup_fn in (
            ("rgb_camera", lambda: destroy_sensor(rgb_camera)),
            ("depth_camera", lambda: destroy_sensor(depth_camera)),
            ("driver", lambda: driver.cleanup() if driver is not None else None),
            ("manager", lambda: manager.cleanup() if manager is not None else None),
        ):
            try:
                cleanup_fn()
            except BaseException as exc:
                cleanup_errors.append(f"{label}: {exc}")
                logging.warning("Cleanup failed for %s: %s", label, exc, exc_info=True)

        if cleanup_errors and outputs_written:
            write_run_metadata(
                out_dir=out_dir,
                run_config=run_config,
                frames_processed=frames_processed,
                dry_run=False,
                extra={
                    "status": "completed_with_cleanup_errors",
                    "cleanup_errors": cleanup_errors,
                    "partial_error": partial_error,
                },
            )

        logging.getLogger().removeHandler(file_log_handler)
        file_log_handler.close()

    if frames_processed <= 0:
        error = partial_error or "No frames were processed; no metrics written."
        write_failure_status(
            out_dir=out_dir,
            run_config=run_config,
            frames_processed=frames_processed,
            error=error,
            stage=stage,
            skipped_sensor_frames=skipped_sensor_frames,
        )
        raise RuntimeError(f"No frames were processed; see failure details in: {out_dir}")

    if output_write_error:
        raise RuntimeError(f"Failed to write detection outputs after {frames_processed} frames: {output_write_error}")
    if partial_error:
        raise RuntimeError(f"Evaluation stopped early after {frames_processed} frames: {partial_error}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple detection models on the same CARLA-generated test frames."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "carla_env_test_detection.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override test_detection.max_frames.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Override test_detection.output_dir root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run synthetic metrics/output checks without CARLA.",
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="In dry-run, fail if configured model files are missing or detector smoke fails.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip detector warmup in full CARLA evaluation.",
    )
    parser.add_argument(
        "--only-model",
        default="",
        help="Comma-separated model name filter for debugging one detector at a time.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_config = parse_run_config(resolve_repo_path(args.config))
    if args.max_frames is not None:
        run_config.eval_cfg.max_frames = int(args.max_frames)
    if args.out_dir:
        run_config.eval_cfg.output_dir = resolve_repo_path(args.out_dir)
    if args.only_model.strip():
        wanted = {item.strip() for item in args.only_model.split(",") if item.strip()}
        run_config.eval_cfg.models = [
            model for model in run_config.eval_cfg.models if model.name in wanted
        ]
        if not run_config.eval_cfg.models:
            raise ValueError(f"--only-model did not match any configured model names: {sorted(wanted)}")

    try:
        if args.dry_run:
            out_dir = run_dry_run(run_config, strict_models=bool(args.strict_models))
        else:
            out_dir = run_full_evaluation(run_config, skip_warmup=bool(args.skip_warmup))
    except BaseException as exc:
        import traceback
        import sys
        print("CRITICAL: Exception caught in main:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        return 1

    print(f"[OK] Detection evaluation outputs: {out_dir}")

    comparison_csv = out_dir / "detection_comparison.csv"
    if comparison_csv.exists():
        print("\n--- DETECTION COMPARISON (CSV) ---")
        print(comparison_csv.read_text(encoding="utf-8").strip())
        print("----------------------------------\n")

    per_class_csv = out_dir / "detection_comparison_per_class.csv"
    if per_class_csv.exists():
        print("--- PER-CLASS COMPARISON (CSV) ---")
        print(per_class_csv.read_text(encoding="utf-8").strip())
        print("----------------------------------\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
