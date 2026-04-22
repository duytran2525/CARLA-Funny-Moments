from __future__ import annotations

import argparse
import csv
import inspect
import importlib
import logging
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

try:
    import carla
except ImportError:
    carla = None

try:
    from agents.navigation.basic_agent import BasicAgent  # type: ignore[import-not-found]
except Exception:
    BasicAgent = None

try:
    from agents.navigation.behavior_agent import BehaviorAgent  # type: ignore[import-not-found]
except Exception:
    BehaviorAgent = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from core_perception.cnn_model import CIL_NvidiaCNN, NvidiaCNN, NvidiaCNNV2
except Exception:
    CIL_NvidiaCNN = None
    NvidiaCNN = None
    NvidiaCNNV2 = None

try:
    from core_perception.yolo_detector import YoloDetector
except Exception as exc:
    logging.warning("Failed to import YoloDetector: %s", exc)
    YoloDetector = None

try:
    from core_control.traffic_supervisor import TrafficSupervisor
except Exception as exc:
    logging.warning("Failed to import TrafficSupervisor: %s", exc)
    TrafficSupervisor = None

from core_control.carla_manager import CarlaManager, SpectatorConfig
from core_control.cil_route_planner import CILRoutePlanner
from core_control.collect_data import DataCollector
from core_control.pid_manager import SpeedPIDController

try:
    from utils.visualizer import DrivingVisualizer, RouteMapVisualizer
except Exception as exc:
    logging.warning("Failed to import visualizers: %s", exc)
    DrivingVisualizer = None
    RouteMapVisualizer = None


def ensure_navigation_agent_imports() -> None:
    """Try to import CARLA navigation agents, including dynamic PythonAPI paths."""
    global BasicAgent, BehaviorAgent
    if BasicAgent is not None and BehaviorAgent is not None:
        return

    project_root = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_pythonapi = os.environ.get("CARLA_PYTHONAPI", "").strip()
    if env_pythonapi:
        candidates.append(Path(env_pythonapi))

    env_carla_root = os.environ.get("CARLA_ROOT", "").strip()
    if env_carla_root:
        candidates.append(Path(env_carla_root) / "PythonAPI")
        candidates.append(Path(env_carla_root) / "WindowsNoEditor" / "PythonAPI")

    candidates.extend(
        [
            project_root / "PythonAPI",
            project_root.parent / "PythonAPI",
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
        if carla_sub.exists():
            carla_sub_str = str(carla_sub)
            if carla_sub_str not in sys.path:
                sys.path.append(carla_sub_str)

        dist_dir = base / "carla" / "dist"
        if dist_dir.exists():
            for egg in dist_dir.glob("*.egg"):
                egg_str = str(egg)
                if egg_str not in sys.path:
                    sys.path.append(egg_str)

    if BasicAgent is None:
        try:
            BasicAgent = importlib.import_module("agents.navigation.basic_agent").BasicAgent
        except Exception:
            pass
    if BehaviorAgent is None:
        try:
            BehaviorAgent = importlib.import_module("agents.navigation.behavior_agent").BehaviorAgent
        except Exception:
            pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def decode_carla_depth_to_meters(image) -> Any:
    """Decode CARLA depth camera BGRA buffer into meters."""
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((image.height, image.width, 4)).astype(np.float32)
    normalized = (
        bgra[:, :, 2] + bgra[:, :, 1] * 256.0 + bgra[:, :, 0] * 65536.0
    ) / 16777215.0
    return normalized * 1000.0


def _camera_to_vehicle_rotation(camera_pitch_deg: float) -> Any:
    pitch_rad = math.radians(-float(camera_pitch_deg))
    rot_y = np.array(
        [
            [math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
        ],
        dtype=np.float32,
    )
    base = np.array(
        [
            [0.0, 0.0, 1.0],   # z_cam -> x_vehicle
            [1.0, 0.0, 0.0],   # x_cam -> y_vehicle
            [0.0, -1.0, 0.0],  # -y_cam -> z_vehicle
        ],
        dtype=np.float32,
    )
    return rot_y @ base


def _project_vehicle_to_image(
    point_vehicle: Any,
    frame_width: int,
    frame_height: int,
    camera_fov_deg: float,
    camera_mount_xyz: tuple[float, float, float],
    camera_pitch_deg: float,
) -> Optional[tuple[int, int]]:
    if np is None:
        return None

    fx = (frame_width / 2.0) / max(math.tan(math.radians(camera_fov_deg) / 2.0), 1e-6)
    fy = fx
    cx = frame_width / 2.0
    cy = frame_height / 2.0

    r_c2v = _camera_to_vehicle_rotation(camera_pitch_deg)
    r_v2c = r_c2v.T
    t = np.array(camera_mount_xyz, dtype=np.float32)
    p_rel = point_vehicle.astype(np.float32) - t
    p_cam = r_v2c @ p_rel
    if p_cam[2] <= 0.15:
        return None

    u = fx * (p_cam[0] / p_cam[2]) + cx
    v = fy * (p_cam[1] / p_cam[2]) + cy
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return (int(round(u)), int(round(v)))

def _draw_yellow_danger_corridor(
    frame_bgr: Any,
    debug_info: Dict[str, Any],
    supervisor_debug: Dict[str, Any],
) -> bool:
    """
    Vẽ Yellow Danger Corridor từ Traffic Supervisor
    
    Input:
    - supervisor_debug: từ traffic_supervisor.get_debug_info()
    - debug_info: từ yolo_detector.get_last_debug_info()
    
    Output:
    - bool: True nếu vẽ thành công, False nếu không có polygon
    """
    if np is None or cv2 is None:
        return False
    
    # Lấy polygon từ supervisor (ưu tiên)
    danger_polygon = supervisor_debug.get("danger_polygon")
    if danger_polygon is None or len(danger_polygon) < 3:
        # Fallback: thử lấy từ YOLO debug info
        obstacle_roi = debug_info.get("obstacle_danger_roi", {})
        danger_polygon = obstacle_roi.get("polygon", [])
        if not danger_polygon or len(danger_polygon) < 3:
            return False
    
    # Vẽ polygon
    try:
        points = np.array(danger_polygon, dtype=np.int32).reshape((-1, 1, 2))
        
        # Màu vàng
        corridor_color = (0, 255, 255)  # BGR: Yellow
        
        # Vẽ fill nhạt
        overlay = frame_bgr.copy()
        cv2.fillPoly(overlay, [points], (30, 200, 255))  # Fill màu cam nhạt
        cv2.addWeighted(overlay, 0.25, frame_bgr, 0.75, 0.0, frame_bgr)
        
        # Vẽ edge đậm
        cv2.polylines(frame_bgr, [points], True, corridor_color, 3)
        
        # Vẽ center line (trắng)
        center_color = (255, 255, 255)
        # Tính center points (trung bình của left + right)
        center_indices = len(danger_polygon) // 2
        center_pts = []
        for i in range(center_indices):
            left_idx = i
            right_idx = len(danger_polygon) - 1 - i
            if left_idx < len(danger_polygon) and right_idx >= 0:
                center_x = (danger_polygon[left_idx][0] + danger_polygon[right_idx][0]) // 2
                center_y = (danger_polygon[left_idx][1] + danger_polygon[right_idx][1]) // 2
                center_pts.append([center_x, center_y])
        
        if len(center_pts) >= 2:
            center_arr = np.array(center_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_bgr, [center_arr], False, center_color, 2)
        
        # Vẽ label
        if len(danger_polygon) >= 2:
            label_x = int(danger_polygon[0][0])
            label_y = max(20, int(danger_polygon[0][1]) - 15)
            cv2.putText(
                frame_bgr,
                "DANGER ZONE (Curved)",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                corridor_color,
                2,
            )
        
        return True
    except Exception as exc:
        logging.warning("Failed to draw yellow corridor: %s", exc)
        return False

def _draw_curved_obstacle_path(
    frame_bgr: Any,
    debug_info: Dict[str, Any],
    camera_fov_deg: float,
    camera_mount_xyz: tuple[float, float, float] = (1.5, 0.0, 2.2),
    camera_pitch_deg: float = -8.0,
) -> bool:
    if np is None or cv2 is None:
        return False

    path_cfg = debug_info.get("obstacle_path_model", {}) or {}
    road_plane = debug_info.get("road_plane", {}) or {}
    if path_cfg.get("mode") != "curved_3d":
        return False
    if not bool(road_plane.get("valid", False)):
        return False

    normal_raw = road_plane.get("normal", [0.0, 0.0, 1.0])
    if not isinstance(normal_raw, list) or len(normal_raw) != 3:
        return False
    normal = np.array(
        [float(normal_raw[0]), float(normal_raw[1]), float(normal_raw[2])],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return False
    normal = normal / norm
    if abs(float(normal[2])) < 1e-4:
        return False
    d = float(road_plane.get("d", 0.0))

    steer = float(path_cfg.get("steer", 0.0))
    curvature = float(path_cfg.get("curvature", 0.0))
    horizon_m = float(path_cfg.get("horizon_m", 22.0))
    min_forward_m = float(path_cfg.get("min_forward_m", 0.8))
    base_half_width_m = float(path_cfg.get("base_half_width_m", 1.1))
    width_growth_per_m = float(path_cfg.get("width_growth_per_m", 0.035))
    curve_width_gain = float(path_cfg.get("curve_width_gain", 0.55))
    max_half_width_m = float(path_cfg.get("max_half_width_m", 2.8))

    horizon_m = max(min_forward_m + 2.0, min(horizon_m, 40.0))
    sample_count = max(28, int(horizon_m * 2.5))
    forward_values = np.linspace(min_forward_m, horizon_m, sample_count, dtype=np.float32)

    left_pixels: list[tuple[int, int]] = []
    right_pixels: list[tuple[int, int]] = []
    center_pixels: list[tuple[int, int]] = []
    frame_h, frame_w = frame_bgr.shape[:2]

    for x in forward_values:
        center_y = 0.5 * curvature * float(x) * float(x)
        half_w = min(
            max_half_width_m,
            base_half_width_m + width_growth_per_m * float(x) + curve_width_gain * abs(curvature) * float(x),
        )
        y_left = center_y - half_w
        y_right = center_y + half_w

        z_center = -(float(normal[0]) * float(x) + float(normal[1]) * center_y + d) / float(normal[2])
        z_left = -(float(normal[0]) * float(x) + float(normal[1]) * y_left + d) / float(normal[2])
        z_right = -(float(normal[0]) * float(x) + float(normal[1]) * y_right + d) / float(normal[2])

        p_center = np.array([float(x), float(center_y), float(z_center)], dtype=np.float32)
        p_left = np.array([float(x), float(y_left), float(z_left)], dtype=np.float32)
        p_right = np.array([float(x), float(y_right), float(z_right)], dtype=np.float32)

        center_uv = _project_vehicle_to_image(
            p_center,
            frame_w,
            frame_h,
            camera_fov_deg,
            camera_mount_xyz,
            camera_pitch_deg,
        )
        left_uv = _project_vehicle_to_image(
            p_left,
            frame_w,
            frame_h,
            camera_fov_deg,
            camera_mount_xyz,
            camera_pitch_deg,
        )
        right_uv = _project_vehicle_to_image(
            p_right,
            frame_w,
            frame_h,
            camera_fov_deg,
            camera_mount_xyz,
            camera_pitch_deg,
        )

        if center_uv is not None:
            center_pixels.append(center_uv)
        if left_uv is not None:
            left_pixels.append(left_uv)
        if right_uv is not None:
            right_pixels.append(right_uv)

    if len(left_pixels) < 4 or len(right_pixels) < 4:
        return False

    corridor = np.array(left_pixels + right_pixels[::-1], dtype=np.int32).reshape((-1, 1, 2))
    overlay = frame_bgr.copy()
    fill_color = (30, 190, 255)
    edge_color = (0, 255, 255)
    center_color = (255, 255, 255)
    cv2.fillPoly(overlay, [corridor], fill_color)
    cv2.addWeighted(overlay, 0.22, frame_bgr, 0.78, 0.0, frame_bgr)
    cv2.polylines(frame_bgr, [corridor], True, edge_color, 2)

    if len(center_pixels) >= 2:
        center_arr = np.array(center_pixels, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_bgr, [center_arr], False, center_color, 2)

    label = (
        f"Curved path | steer={steer:+.2f} | k={curvature:+.3f} 1/m | "
        f"h={horizon_m:.1f}m"
    )
    anchor = center_pixels[0] if center_pixels else left_pixels[0]
    text_x = max(8, min(frame_w - 320, int(anchor[0]) - 40))
    text_y = max(24, min(frame_h - 8, int(anchor[1]) - 10))
    cv2.putText(
        frame_bgr,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        edge_color,
        2,
    )
    return True


def map_road_option_to_command(road_option) -> int:
    if road_option is None:
        return 0

    # Numeric fallback for enum values seen in some CARLA API variants.
    raw_value = getattr(road_option, "value", road_option)
    try:
        numeric_value = int(raw_value)
        numeric_map = {
            1: 1,  # LEFT
            2: 2,  # RIGHT
            3: 3,  # STRAIGHT
            5: 1,  # CHANGELANELEFT
            6: 2,  # CHANGELANERIGHT
        }
        if numeric_value in numeric_map:
            return numeric_map[numeric_value]
    except (TypeError, ValueError):
        pass

    option_name = getattr(road_option, "name", str(road_option)).lower()
    option_name = option_name.replace(" ", "")
    if "change_lane_left" in option_name or "changelaneleft" in option_name:
        return 1
    if "change_lane_right" in option_name or "changelaneright" in option_name:
        return 2
    if "left" in option_name:
        return 1
    if "right" in option_name:
        return 2
    if "straight" in option_name:
        return 3
    return 0


def set_navigation_destination(nav_agent: Any, current_location: Any, destination_location: Any) -> None:
    """Call set_destination with the right argument order across CARLA API variants."""
    setter = getattr(nav_agent, "set_destination", None)
    if setter is None:
        raise AttributeError("Navigation agent does not expose set_destination().")

    try:
        signature = inspect.signature(setter)
        param_names = [param.name.lower() for param in signature.parameters.values()]
    except (TypeError, ValueError):
        param_names = []

    # API variants:
    # - set_destination(start_location, end_location)
    # - set_destination(end_location, start_location=None)
    if len(param_names) >= 2:
        first, second = param_names[0], param_names[1]
        if "start" in first and ("end" in second or "dest" in second):
            setter(current_location, destination_location)
            return
        if ("end" in first or "dest" in first) and "start" in second:
            setter(destination_location, current_location)
            return

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

    # Final fallback for unusual signatures.
    setter(destination_location)


def apply_random_weather(world) -> str:
    # 40% clear day, 20% sunset, 20% night, 20% rain.
    roll = random.random()
    if roll < 0.40:
        preset_name = "clear_day"
        weather = carla.WeatherParameters(
            cloudiness=8.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=8.0,
            fog_density=0.0,
            wetness=0.0,
            sun_azimuth_angle=30.0,
            sun_altitude_angle=70.0,
        )
    elif roll < 0.60:
        preset_name = "sunset"
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=10.0,
            fog_density=2.0,
            wetness=0.0,
            sun_azimuth_angle=15.0,
            sun_altitude_angle=8.0,
        )
    elif roll < 0.80:
        preset_name = "night"
        weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=6.0,
            fog_density=3.0,
            wetness=0.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=-35.0,
        )
    else:
        preset_name = "rain"
        weather = carla.WeatherParameters(
            cloudiness=85.0,
            precipitation=70.0,
            precipitation_deposits=60.0,
            wind_intensity=35.0,
            fog_density=12.0,
            wetness=75.0,
            sun_azimuth_angle=25.0,
            sun_altitude_angle=45.0,
        )

    world.set_weather(weather)
    return preset_name


def apply_weather_preset(world, preset: str) -> None:
    """Apply a specific weather preset to the world."""
    if carla is None:
        return
    preset_lower = preset.lower().replace(" ", "").replace("_", "")
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
    weather = presets.get(preset_lower, carla.WeatherParameters.ClearNoon)
    world.set_weather(weather)
    logging.info("Applied weather preset: %s", preset)


def resolve_model_path(model_path: str) -> Path:
    project_root = Path(__file__).resolve().parent
    if model_path.lower() != "auto":
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    models_dir = project_root / "models"
    if not models_dir.exists():
        return (project_root / "models" / "cnn_steering.pth").resolve()

    def newest(pattern: str) -> list[Path]:
        return sorted(
            models_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    preferred = newest("cnn_steering TL=*.pth")
    if preferred:
        return preferred[0]

    steering_models = newest("cnn_steering*.pth")
    if steering_models:
        return steering_models[0]

    any_pth = newest("*.pth")
    if any_pth:
        return any_pth[0]

    return (models_dir / "cnn_steering.pth").resolve()


def resolve_cil_model_path(model_path: str) -> Path:
    project_root = Path(__file__).resolve().parent
    if model_path.lower() != "auto":
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    models_dir = project_root / "models"
    if not models_dir.exists():
        return (project_root / "models" / "cil_nvidia.pth").resolve()

    def newest(pattern: str) -> list[Path]:
        return sorted(
            models_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    preferred = newest("cil*.pth")
    if preferred:
        return preferred[0]

    preferred = newest("*cil*.pth")
    if preferred:
        return preferred[0]

    any_pth = newest("*.pth")
    if any_pth:
        return any_pth[0]

    return (models_dir / "cil_nvidia.pth").resolve()


def resolve_yolo_model_path(model_path: str) -> Path:
    project_root = Path(__file__).resolve().parent
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


@dataclass
class RunConfig:
    env_config_path: str  # Path to environment config file (.env, .json, .yaml)
    host: str  # CARLA server host (for example: 127.0.0.1)
    port: int  # CARLA RPC port (default: 2000)
    tm_port: int  # Traffic Manager port (default: 8000)
    timeout: float  # RPC timeout in seconds
    sync: bool  # Use synchronous mode to avoid frame drops
    fixed_delta: float  # Fixed step duration per frame (e.g. 0.05 = 20 FPS)
    no_rendering: bool  # Disable rendering on server for higher simulation speed
    map_name: str  # CARLA map name (for example: Town01, Town04)
    vehicle_filter: str  # Ego vehicle blueprint filter
    spawn_point: int  # Ego spawn point index
    destination_point: int  # Destination spawn index for route target (negative means random)
    ticks: int  # Max simulation steps to run
    tick_interval: float  # Sleep interval between ticks in dry-run mode
    dry_run: bool  # Run without connecting to CARLA
    seed: Optional[int]  # Random seed for reproducibility
    model_path: str  # Path to model weights (.pth, .pt)
    cil_model_path: str  # Path to CIL model weights (.pth)
    yolo_model_path: str  # Path to YOLO weights (.pt)
    model_device: str  # Inference device (cpu or cuda)
    target_speed_kmh: float  # Target ego speed in km/h
    max_throttle: float  # Max throttle command in range [0.0, 1.0]
    max_brake: float  # Max brake command in range [0.0, 1.0]
    steer_smoothing: float  # Steering smoothing factor
    camera_width: int  # Camera capture width in pixels
    camera_height: int  # Camera capture height in pixels
    camera_fov: float  # Camera field of view in degrees
    lock_spectator_on_spawn: bool  # Lock spectator near ego on spawn
    spectator_reapply_each_tick: bool  # Re-apply spectator transform each tick
    spectator_follow_distance: float  # Spectator follow distance from ego
    spectator_height: float  # Spectator height relative to ego
    spectator_pitch: float  # Spectator camera pitch in degrees
    collect_data: bool  # Enable synchronized dataset collection
    collect_data_dir: str  # Dataset output folder
    save_every_n: int  # Keep one sample every N frames
    image_prefix: str
    npc_vehicle_count: int
    npc_bike_count: int
    npc_motorbike_count: int
    npc_pedestrian_count: int
    npc_enable_autopilot: bool
    record_video: bool
    video_output_path: str
    video_fps: float
    video_duration_sec: int
    video_codec: str
    random_weather: bool
    weather_preset: str  # Weather preset from config (e.g., ClearNoon)
    recovery_interval_frames: int
    recovery_duration_frames: int
    recovery_steer_offset: float
    nav_agent_type: str
    yolo_disable_autopilot_red_light: bool
    cil_route_lookahead_m: float
    cil_command_prep_time_s: float
    cil_command_trigger_min_m: float
    cil_command_trigger_max_m: float
    cil_command_retarget_window_s: float

class BaseSession:
    """Shared interface for CARLA and dry-run sessions."""

    def start(self) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError

    @property
    def ego_vehicle(self):
        return None

    @property
    def world(self):
        return None


class CarlaSession(BaseSession):
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._manager: Optional[CarlaManager] = None

    @property
    def ego_vehicle(self):
        if self._manager is None:
            return None
        return self._manager.ego_vehicle

    @property
    def world(self):
        if self._manager is None:
            return None
        return self._manager.world

    @property
    def traffic_manager(self):
        if self._manager is None:
            return None
        return self._manager.tm

    def start(self) -> None:
        assert carla is not None
        self._manager = CarlaManager(
            host=self.config.host,
            port=self.config.port,
            tm_port=self.config.tm_port,
            timeout=self.config.timeout,
            map_name=self.config.map_name,
            sync=self.config.sync,
            fixed_delta=self.config.fixed_delta,
            no_rendering=self.config.no_rendering,
            vehicle_filter=self.config.vehicle_filter,
            spawn_point=self.config.spawn_point,
            spectator_cfg=SpectatorConfig(
                lock_on_spawn=self.config.lock_spectator_on_spawn,
                keep_reapply_each_tick=self.config.spectator_reapply_each_tick,
                follow_distance=self.config.spectator_follow_distance,
                height=self.config.spectator_height,
                pitch=self.config.spectator_pitch,
            ),
            npc_vehicle_count=self.config.npc_vehicle_count,
            npc_bike_count=self.config.npc_bike_count,
            npc_motorbike_count=self.config.npc_motorbike_count,
            npc_pedestrian_count=self.config.npc_pedestrian_count,
            npc_enable_autopilot=self.config.npc_enable_autopilot,
        )
        self._manager.start()
        logging.info("CARLA session is ready.")

    def tick(self) -> None:
        if self._manager is None:
            return
        self._manager.tick()

    def cleanup(self) -> None:
        if self._manager is not None:
            self._manager.cleanup()


class DryRunSession(BaseSession):
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.step = 0

    def start(self) -> None:
        logging.info("Starting dry-run mode (CARLA not required).")

    def tick(self) -> None:
        self.step += 1
        time.sleep(self.config.tick_interval)

    def cleanup(self) -> None:
        logging.info("Dry-run session ended after %d ticks.", self.step)


class BaseAgent:
    name = "base"

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.session: Optional[BaseSession] = None

    def setup(self, session: BaseSession) -> None:
        self.session = session

    def run_step(self, step_idx: int) -> None:
        raise NotImplementedError

    def teardown(self) -> None:
        return

    def should_stop(self) -> bool:
        return False


class AutopilotAgent(BaseAgent):
    name = "autopilot"

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._video_camera = None
        self._video_writer = None
        self._video_output: Optional[Path] = None
        self._video_frames_written = 0
        self._video_max_frames = 0
        self._stop_requested = False
        self._latest_rgb = None
        self._frame_lock = threading.Lock()
        self._waiting_frame_logged = False
        self._data_cameras = []
        self._collector: Optional[DataCollector] = None
        self._nav_agent = None
        self._spawn_points = []
        self._recovery_start_frame = -1
        self._recovery_direction = 1.0
        self._tm_fallback_mode = False

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        if vehicle is None or session.world is None:
            logging.info("No ego vehicle in this session; autopilot setup skipped.")
            return

        self._init_navigation_agent(session.world, vehicle)

        self._start_video_recording(session.world, vehicle)
        self._start_data_collection(session.world, vehicle)
        logging.info("Autopilot enabled for ego vehicle.")

    def _init_navigation_agent(self, world, vehicle) -> None:
        self._spawn_points = world.get_map().get_spawn_points()
        if not self._spawn_points:
            raise RuntimeError("No spawn points found to build navigation route.")

        ensure_navigation_agent_imports()

        nav_type = self.config.nav_agent_type.lower()
        try:
            if nav_type == "behavior":
                if BehaviorAgent is None:
                    raise RuntimeError("BehaviorAgent missing")
                self._nav_agent = BehaviorAgent(vehicle, behavior="normal")
            else:
                if BasicAgent is None:
                    raise RuntimeError("BasicAgent missing")
                self._nav_agent = BasicAgent(vehicle, target_speed=max(10.0, self.config.target_speed_kmh))
        except Exception:
            self._nav_agent = None
            self._tm_fallback_mode = True
            try:
                vehicle.set_autopilot(True, self.config.tm_port)
            except TypeError:
                vehicle.set_autopilot(True)
            logging.info(
                "Using TM autopilot fallback with heuristic command labels (set CARLA_PYTHONAPI to enable BasicAgent/BehaviorAgent)."
            )
            return

        self._set_new_destination(vehicle)
        logging.info("Navigation agent initialized: %s", self.config.nav_agent_type)

    def _set_new_destination(self, vehicle) -> None:
        if not self._spawn_points or self._nav_agent is None:
            return
        if self.config.destination_point >= 0:
            destination = self._spawn_points[self.config.destination_point % len(self._spawn_points)].location
        else:
            destination = random.choice(self._spawn_points).location
        current_loc = vehicle.get_location()
        set_navigation_destination(self._nav_agent, current_loc, destination)

    def _extract_current_command(self) -> int:
        vehicle = self.session.ego_vehicle if self.session is not None else None
        world = self.session.world if self.session is not None else None

        if self._tm_fallback_mode and vehicle is not None and world is not None:
            m = world.get_map()
            waypoint = m.get_waypoint(vehicle.get_location(), project_to_road=True)
            if waypoint is None:
                return 0
            # Heuristic command labels at junctions when route planner API is unavailable.
            if waypoint.is_junction:
                steer = float(vehicle.get_control().steer)
                if steer < -0.10:
                    return 1
                if steer > 0.10:
                    return 2
                return 3
            return 0

        if self._nav_agent is None:
            return 0
        planner = None
        if hasattr(self._nav_agent, "get_local_planner"):
            planner = self._nav_agent.get_local_planner()
        if planner is None:
            return 0

        road_option = getattr(planner, "target_road_option", None)
        if road_option is None:
            road_option = getattr(planner, "_target_road_option", None)

        # Fallback: peek first waypoint command in planner queue.
        if road_option is None:
            queue_attr = getattr(planner, "_waypoints_queue", None)
            if queue_attr and len(queue_attr) > 0:
                road_option = queue_attr[0][1]
        return map_road_option_to_command(road_option)

    def _recovery_offset(self, frame_id: int) -> float:
        if not self.config.collect_data:
            return 0.0
        if self._tm_fallback_mode:
            # Keep TM autopilot untouched when navigation agent API is unavailable.
            return 0.0

        interval = max(1, self.config.recovery_interval_frames)
        duration = max(1, self.config.recovery_duration_frames)
        if frame_id % interval == 0:
            self._recovery_start_frame = frame_id
            self._recovery_direction = random.choice([-1.0, 1.0])

        if self._recovery_start_frame >= 0 and frame_id < self._recovery_start_frame + duration:
            return self._recovery_direction * self.config.recovery_steer_offset
        return 0.0

    def _start_video_recording(self, world, vehicle) -> None:
        if not self.config.record_video:
            return
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video recording.")

        output_path = Path(self.config.video_output_path)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(self.config.video_fps),
            (int(self.config.camera_width), int(self.config.camera_height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick") and self.config.sync:
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        self._video_writer = writer
        self._video_camera = camera
        self._video_output = output_path
        self._video_max_frames = max(1, int(self.config.video_duration_sec * self.config.video_fps))
        self._video_frames_written = 0
        self._stop_requested = False
        camera.listen(self._on_video_frame)
        logging.info(
            "Autopilot video recording started: %s (fps=%.2f, duration=%ds, max_frames=%d)",
            output_path,
            self.config.video_fps,
            self.config.video_duration_sec,
            self._video_max_frames,
        )

    def _start_data_collection(self, world, vehicle) -> None:
        if not self.config.collect_data:
            return

        self._collector = DataCollector(
            output_dir=self.config.collect_data_dir,
            enabled=True,
            save_every_n=self.config.save_every_n,
        )
        self._collector.start()

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick"):
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_setups = [
            ("center", carla.Transform(carla.Location(x=1.5, y=0.0, z=2.2), carla.Rotation(pitch=-8.0))),
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(yaw=-25.0, pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(yaw=25.0, pitch=-8.0))),
        ]
        for side, transform in camera_setups:
            sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            sensor.listen(self._collector.make_sensor_callback(side))
            self._data_cameras.append(sensor)

        logging.info("Autopilot data collector started with synchronized center/left/right cameras.")

    def _on_video_frame(self, image) -> None:
        if np is None or cv2 is None:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = array.reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        with self._frame_lock:
            self._latest_rgb = rgb

    def _read_latest_video_frame(self):
        with self._frame_lock:
            frame = self._latest_rgb
            self._latest_rgb = None
        return frame

    def run_step(self, step_idx: int) -> None:
        frame = self._read_latest_video_frame()
        vehicle = self.session.ego_vehicle if self.session is not None else None
        world = self.session.world if self.session is not None else None
        if frame is not None:
            if self._video_writer is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self._video_writer.write(bgr)
                self._video_frames_written += 1
                if self._video_frames_written >= self._video_max_frames:
                    self._stop_requested = True
        elif (self._video_writer is not None or self._collector is not None) and not self._waiting_frame_logged:
            logging.info("Autopilot waiting for first camera frame...")
            self._waiting_frame_logged = True

        if vehicle is not None and self._nav_agent is not None:
            try:
                if self._nav_agent.done():
                    self._set_new_destination(vehicle)
            except Exception:
                pass

            nav_control = self._nav_agent.run_step()
            frame_id_for_control = step_idx
            if world is not None:
                frame_id_for_control = int(world.get_snapshot().frame)
            recovery_delta = self._recovery_offset(frame_id_for_control)
            nav_control.steer = float(clamp(nav_control.steer + recovery_delta, -1.0, 1.0))
            vehicle.apply_control(nav_control)
        elif vehicle is not None and self._tm_fallback_mode:
            frame_id_for_control = step_idx
            if world is not None:
                frame_id_for_control = int(world.get_snapshot().frame)
            recovery_delta = self._recovery_offset(frame_id_for_control)
            if abs(recovery_delta) > 1e-6:
                control = vehicle.get_control()
                control.steer = float(clamp(control.steer + recovery_delta, -1.0, 1.0))
                vehicle.apply_control(control)

        if self._collector is not None and vehicle is not None:
            velocity = vehicle.get_velocity()
            speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            control = vehicle.get_control()
            rotation = vehicle.get_transform().rotation
            frame_id = step_idx
            if world is not None:
                frame_id = int(world.get_snapshot().frame)
            command = self._extract_current_command()
            self._collector.add_vehicle_state(
                frame_id=frame_id,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                command=command,
                pitch=rotation.pitch,
                roll=rotation.roll,
                yaw=rotation.yaw,
            )

        if step_idx % 20 == 0:
            logging.info("Autopilot tick %d", step_idx)

    def teardown(self) -> None:
        if self.session is None:
            return
        vehicle = self.session.ego_vehicle
        if vehicle is None:
            return
        if self._tm_fallback_mode:
            try:
                vehicle.set_autopilot(False, self.config.tm_port)
            except TypeError:
                vehicle.set_autopilot(False)
        self._nav_agent = None
        if self._collector is not None:
            self._collector.close()
            self._collector = None
        for sensor in self._data_cameras:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self._data_cameras = []
        if self._video_camera is not None:
            self._video_camera.stop()
            self._video_camera.destroy()
            self._video_camera = None
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logging.info(
                "Saved video to %s (%d frames).",
                self._video_output,
                self._video_frames_written,
            )

    def should_stop(self) -> bool:
        return self._stop_requested


class LaneFollowAgent(BaseAgent):
    name = "lane_follow"

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._enabled = False
        self._model = None
        self._device = None
        self._camera = None
        self._depth_camera = None
        self._latest_rgb = None
        self._latest_depth_m = None
        self._frame_lock = threading.Lock()
        self._last_steer = 0.0
        self._waiting_frame_logged = False
        self._collector: Optional[DataCollector] = None
        self._data_cameras = []
        self._video_writer = None
        self._video_output: Optional[Path] = None
        self._video_frames_written = 0
        self._video_max_frames = 0
        self._stop_requested = False
        self._visualizer = None
        # YOLO integration
        self._yolo_detector = None
        self._yolo_window_name = "Lane Follow + YOLO Detection"
        self._yolo_enabled = False
        # Speed control (PID)
        self._speed_controller = SpeedPIDController(
            target_speed_kmh=config.target_speed_kmh,
            max_throttle=config.max_throttle,
            max_brake=config.max_brake,
        )

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        if vehicle is None or session.world is None:
            logging.info("No CARLA vehicle/world available; lane_follow runs as noop.")
            return

        self._model = self._load_model()
        self._camera = self._spawn_camera(session.world, vehicle)
        self._camera.listen(self._on_camera_frame)
        try:
            self._depth_camera = self._spawn_depth_camera(session.world, vehicle)
            self._depth_camera.listen(self._on_depth_frame)
        except Exception as exc:
            self._depth_camera = None
            logging.warning(
                "Depth camera unavailable for lane_follow, fallback to bbox distance. Reason: %s",
                exc,
            )

        self._collector = DataCollector(
            output_dir=self.config.collect_data_dir,
            enabled=self.config.collect_data,
            save_every_n=self.config.save_every_n,
        )
        self._collector.start()
        self._start_data_collection_cameras(session.world, vehicle)
        self._init_video_writer()

        # Load YOLO detector if model path is provided
        self._load_yolo_detector()
        if DrivingVisualizer is not None:
            window_name = self._yolo_window_name if self._yolo_enabled else "Lane Follow HUD"
            self._visualizer = DrivingVisualizer(window_name=window_name)

        self._enabled = True
        logging.info("Lane-follow agent is ready.")

    def _start_data_collection_cameras(self, world, vehicle) -> None:
        if not self.config.collect_data or self._collector is None:
            return

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick"):
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_setups = [
            ("center", carla.Transform(carla.Location(x=1.5, y=0.0, z=2.2), carla.Rotation(pitch=-8.0))),
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(yaw=-25.0, pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(yaw=25.0, pitch=-8.0))),
        ]
        for side, transform in camera_setups:
            sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            sensor.listen(self._collector.make_sensor_callback(side))
            self._data_cameras.append(sensor)

        logging.info("Lane-follow data collection cameras are attached (center/left/right).")

    def _init_video_writer(self) -> None:
        if not self.config.record_video:
            return
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video recording.")

        output_path = Path(self.config.video_output_path)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(self.config.video_fps),
            (int(self.config.camera_width), int(self.config.camera_height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")

        self._video_writer = writer
        self._video_output = output_path
        self._video_max_frames = max(1, int(self.config.video_duration_sec * self.config.video_fps))
        logging.info(
            "Video recording started: %s (fps=%.2f, duration=%ds, max_frames=%d)",
            output_path,
            self.config.video_fps,
            self.config.video_duration_sec,
            self._video_max_frames,
        )

    def _load_yolo_detector(self) -> None:
        """Load YOLO detector if yolo_model_path is provided."""
        if not self.config.yolo_model_path:
            logging.info("No YOLO model path provided, YOLO detection disabled.")
            return

        if YoloDetector is None:
            logging.warning("YoloDetector not available. Install ultralytics to enable YOLO detection.")
            return

        model_path = resolve_yolo_model_path(self.config.yolo_model_path)
        if not model_path.exists():
            logging.warning("YOLO model file not found: %s. YOLO detection disabled.", model_path)
            return

        self._yolo_detector = YoloDetector(
            str(model_path),
            camera_fov_deg=self.config.camera_fov,
            obstacle_base_distance_m=8.0,
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
        )
        self._yolo_enabled = True
        logging.info("YOLO detection integrated with lane_follow. Model: %s", model_path)

    def _load_model(self):
        if torch is None:
            raise RuntimeError("PyTorch is required for lane_follow agent.")
        if cv2 is None:
            raise RuntimeError("opencv-python is required for lane_follow agent.")
        if np is None:
            raise RuntimeError("numpy is required for lane_follow agent.")
        if NvidiaCNN is None:
            raise RuntimeError("Cannot import NvidiaCNN from core_perception.cnn_model.")

        model_path = resolve_model_path(self.config.model_path)
        if self.config.model_path.lower() == "auto":
            logging.info("Auto-selected model path: %s", model_path)
        if not model_path.exists():
            models_dir = Path(__file__).resolve().parent / "models"
            existing = ", ".join(str(p.name) for p in sorted(models_dir.glob("*.pth")))
            if not existing:
                existing = "no .pth file found in models/"
            raise RuntimeError(f"Model file not found: {model_path}. Available: {existing}")

        device_name = self.config.model_device.lower()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            device_name = "cpu"

        self._device = torch.device(device_name)

        checkpoint = torch.load(model_path, map_location=self._device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise RuntimeError("Unsupported checkpoint format. Expected state_dict.")

        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {
                key.replace("module.", "", 1): value for key, value in state_dict.items()
            }

        # Auto-detect V2 architecture (has BatchNorm layers)
        is_v2 = any("running_mean" in k for k in state_dict.keys())
        if is_v2:
            logging.info("Detected V2 architecture (BatchNorm) for %s", model_path)
            model = NvidiaCNNV2().to(self._device)
        else:
            model = NvidiaCNN().to(self._device)

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        logging.info("Loaded model from %s on %s", model_path, self._device)
        return model

    def _spawn_camera(self, world, vehicle):
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick") and self.config.sync:
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logging.info("Attached RGB camera to ego vehicle.")
        return camera

    def _spawn_depth_camera(self, world, vehicle):
        bp_lib = world.get_blueprint_library()
        depth_bp = bp_lib.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(self.config.camera_width))
        depth_bp.set_attribute("image_size_y", str(self.config.camera_height))
        depth_bp.set_attribute("fov", str(self.config.camera_fov))
        if depth_bp.has_attribute("sensor_tick") and self.config.sync:
            depth_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        logging.info("Attached depth camera to ego vehicle.")
        return camera

    def _on_camera_frame(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = array.reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        with self._frame_lock:
            self._latest_rgb = rgb

    def _on_depth_frame(self, image) -> None:
        depth_m = decode_carla_depth_to_meters(image)
        with self._frame_lock:
            self._latest_depth_m = depth_m

    def _read_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_rgb
            if frame is None:
                return None, None
            depth_m = self._latest_depth_m
            self._latest_rgb = None
            self._latest_depth_m = None
        return frame, depth_m

    def _write_video_frame(self, rgb_frame) -> None:
        if self._video_writer is None:
            return
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self._video_writer.write(bgr)
        self._video_frames_written += 1
        if self._video_frames_written >= self._video_max_frames:
            self._stop_requested = True

    def _predict_steering(self, rgb_frame) -> float:
        height = rgb_frame.shape[0]
        cropped = rgb_frame[int(height * 0.45) :, :, :]
        resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)

        # Keep inference color space identical to training (RGB -> YUV).
        yuv_image = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        tensor = torch.from_numpy(yuv_image).permute(2, 0, 1).float().div_(255.0)
        tensor.sub_(0.5).div_(0.5)
        tensor.unsqueeze_(0)
        tensor = tensor.to(self._device, non_blocking=True)
        with torch.inference_mode():
            steering = self._model(tensor).item()
        return clamp(steering, -1.0, 1.0)

    def _current_speed_kmh(self) -> float:
        velocity = self.session.ego_vehicle.get_velocity()
        speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_mps * 3.6

    def _longitudinal_control(self, speed_kmh: float) -> tuple[float, float]:
        """Compute throttle and brake using PID controller."""
        self._speed_controller.set_target_speed(self.config.target_speed_kmh)
        throttle, brake = self._speed_controller.compute(speed_kmh)
        return throttle, brake

    def _run_yolo_detection(
        self,
        frame,
        depth_map_m,
        step_idx: int,
        current_steer: Optional[float] = None,
        speed_kmh: Optional[float] = None,
    ) -> tuple[bool, Dict[str, Any], Any]:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # distance_threshold applies to dynamic obstacles (pedestrian/vehicle/two_wheeler).
        detections, is_emergency = self._yolo_detector.detect_and_evaluate(
            frame_bgr,
            distance_threshold=None,
            depth_map_m=depth_map_m,
            vehicle_steer=current_steer,
            speed_kmh=speed_kmh,
        )
        debug_info = {}
        if hasattr(self._yolo_detector, "get_last_debug_info"):
            debug_info = self._yolo_detector.get_last_debug_info() or {}

        annotated_frame = frame_bgr.copy()
        for roi_region in debug_info.get("roi_regions", []):
            x1, y1, x2, y2 = roi_region["box"]
            is_active = bool(roi_region.get("active", False))
            roi_color = (255, 180, 0) if is_active else (80, 80, 80)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), roi_color, 2)
            roi_label = (
                f"ROI {roi_region['label']} < {roi_region['max_distance_m']:.0f}m"
            )
            cv2.putText(
                annotated_frame,
                roi_label,
                (x1 + 4, min(y2 - 6, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                roi_color,
                1,
            )

        obstacle_roi = debug_info.get("obstacle_danger_roi", {})
        drew_curved = _draw_curved_obstacle_path(
            annotated_frame,
            debug_info,
            camera_fov_deg=float(self.config.camera_fov),
            camera_mount_xyz=(1.5, 0.0, 2.2),
            camera_pitch_deg=-8.0,
        )
        obstacle_polygon = obstacle_roi.get("polygon", [])
        if (not drew_curved) and np is not None and len(obstacle_polygon) >= 3:
            points = np.array(obstacle_polygon, dtype=np.int32).reshape((-1, 1, 2))
            roi_color = (0, 255, 255)
            cv2.polylines(annotated_frame, [points], True, roi_color, 2)
            label = (
                f"{obstacle_roi.get('label', 'Obstacle corridor')} < "
                f"{float(obstacle_roi.get('distance_threshold_m', 5.0)):.1f}m"
            )
            anchor_x = int(obstacle_polygon[0][0])
            anchor_y = max(18, int(obstacle_polygon[0][1]) - 8)
            cv2.putText(
                annotated_frame,
                label,
                (anchor_x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                roi_color,
                1,
            )

        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class_name']
            conf = det['confidence']
            distance = det['distance']
            distance_source = det.get("distance_source", "bbox")
            roi_zone = det.get('roi_zone')
            in_danger_roi = bool(det.get("in_danger_roi", False))
            danger_match = bool(det.get("danger_match", False))
            path_check_mode = det.get("path_check_mode")

            if class_name == "traffic_light_red":
                color = (0, 0, 255)
            elif class_name == "traffic_light_green":
                color = (0, 255, 0)
            elif danger_match:
                color = (0, 0, 255)
            elif in_danger_roi and distance < 10.0:
                color = (0, 165, 255)
            elif distance < 5.0:
                color = (255, 200, 0)
            elif distance < 10.0:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f} ({distance:.1f}m)"
            label = f"{label} [{distance_source}]"
            if roi_zone is not None:
                label = f"{label} [{roi_zone}]"
            if in_danger_roi:
                label = f"{label} [path]"
            if path_check_mode:
                label = f"{label} [{path_check_mode}]"
            if danger_match:
                label = f"{label} [BRAKE]"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if debug_info.get("red_light_active"):
            status_text = "RED LIGHT STOP"
            if debug_info.get("decision_reason"):
                status_text += f" | {debug_info['decision_reason']}"
            status_color = (0, 0, 255)
        elif debug_info.get("obstacle_emergency"):
            status_text = "EMERGENCY BRAKE (OBSTACLE)"
            status_color = (0, 0, 255)
        else:
            status_text = "Normal"
            status_color = (0, 255, 0)

        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        lock_zone = debug_info.get("locked_zone")
        if lock_zone:
            lock_text = (
                f"LOCK={lock_zone} | immunity={debug_info.get('green_immunity_counter', 0)}"
            )
        else:
            lock_text = "LOCK=None"
        cv2.putText(
            annotated_frame,
            lock_text,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        turn_text = (
            f"TURN={bool(debug_info.get('turn_phase_active', False))} | "
            f"grace={int(debug_info.get('turn_green_grace_counter', 0))} | "
            f"suppress={bool(debug_info.get('turn_red_suppressed', False))}"
        )
        cv2.putText(
            annotated_frame,
            turn_text,
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            2,
        )

        if step_idx % 20 == 0 and detections:
            logging.info(
                "YOLO detections: %d objects | Emergency: %s | Reason: %s",
                len(detections),
                is_emergency,
                debug_info.get("decision_reason", "n/a"),
            )

        return is_emergency, debug_info, annotated_frame

    def run_step(self, step_idx: int) -> None:
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("Lane-follow agent waiting for CARLA runtime.")
            return

        frame, depth_map_m = self._read_latest_frame()
        if frame is None:
            if not self._waiting_frame_logged:
                logging.info("Waiting for first camera frame...")
                self._waiting_frame_logged = True
            return

        speed_kmh = self._current_speed_kmh()

        # Run YOLO detection and display if enabled
        is_emergency = False
        yolo_debug_info: Dict[str, Any] = {}
        annotated_yolo_frame = None
        if self._yolo_enabled and self._yolo_detector is not None:
            is_emergency, yolo_debug_info, annotated_yolo_frame = self._run_yolo_detection(
                frame,
                depth_map_m,
                step_idx,
                current_steer=self._last_steer,
                speed_kmh=speed_kmh,
            )

        self._write_video_frame(frame)
        if self._stop_requested:
            logging.info("Video duration target reached, stopping agent loop.")
            return

        steering_raw = self._predict_steering(frame)
        alpha = clamp(self.config.steer_smoothing, 0.0, 0.99)
        steering = alpha * self._last_steer + (1.0 - alpha) * steering_raw
        self._last_steer = steering

        throttle, brake = self._longitudinal_control(speed_kmh)

        # Apply emergency braking if YOLO detects danger
        if is_emergency:
            throttle = 0.0
            # Red light and obstacle use the same brake cap in lane-follow mode.
            brake = max(brake, self.config.max_brake)

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(clamp(steering, -1.0, 1.0)),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )
        self.session.ego_vehicle.apply_control(control)
        rotation = self.session.ego_vehicle.get_transform().rotation
        yaw_deg = float(rotation.yaw)

        if self._collector is not None:
            frame_id = step_idx
            if self.session.world is not None:
                frame_id = int(self.session.world.get_snapshot().frame)
            self._collector.add_vehicle_state(
                frame_id=frame_id,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                command=0,
                pitch=rotation.pitch,
                roll=rotation.roll,
                yaw=rotation.yaw,
            )

        emergency_reason = yolo_debug_info.get("decision_reason", "none") if is_emergency else "none"
        hud_metrics = {
            "agent": "lane_follow",
            "tick": step_idx,
            "speed_kmh": speed_kmh,
            "target_speed_kmh": self.config.target_speed_kmh,
            "steer": control.steer,
            "steer_raw": steering_raw,
            "throttle": control.throttle,
            "brake": control.brake,
            "yaw_deg": yaw_deg,
            "emergency": is_emergency,
            "reason": emergency_reason,
        }
        if self._visualizer is not None:
            extra_lines = [f"Reason: {emergency_reason}"] if is_emergency else None
            if annotated_yolo_frame is not None:
                self._visualizer.show_bgr(annotated_yolo_frame, hud_metrics, extra_lines=extra_lines)
            else:
                self._visualizer.show_rgb(frame, hud_metrics, extra_lines=extra_lines)
        elif annotated_yolo_frame is not None:
            cv2.imshow(self._yolo_window_name, annotated_yolo_frame)
            cv2.waitKey(1)

        if step_idx % 20 == 0:
            logging.info(
                "lane_follow tick=%d speed=%.1f km/h steer=%.3f throttle=%.2f brake=%.2f emergency=%s reason=%s",
                step_idx,
                speed_kmh,
                control.steer,
                control.throttle,
                control.brake,
                is_emergency,
                emergency_reason,
            )

    def teardown(self) -> None:
        if self._collector is not None:
            self._collector.close()
            self._collector = None
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logging.info(
                "Saved video to %s (%d frames).",
                self._video_output,
                self._video_frames_written,
            )
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
            self._camera = None
            logging.info("Destroyed lane-follow RGB camera.")
        if self._depth_camera is not None:
            self._depth_camera.stop()
            self._depth_camera.destroy()
            self._depth_camera = None
            logging.info("Destroyed lane-follow depth camera.")
        for sensor in self._data_cameras:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self._data_cameras = []
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None
        # Clean up YOLO window
        if self._yolo_enabled:
            try:
                cv2.destroyWindow(self._yolo_window_name)
            except Exception:
                pass
            self._yolo_enabled = False
            logging.info("Closed YOLO detection window.")

    def should_stop(self) -> bool:
        return self._stop_requested


class CILAgent(BaseAgent):
    name = "cil"
    CIL_MAX_SPEED_KMH = 120.0
    COMMAND_PREP_TIME_S = 1.8
    COMMAND_TRIGGER_MIN_M = 8.0
    COMMAND_TRIGGER_MAX_M = 25.0
    COMMAND_WARMUP_DISTANCE_M = 32.0
    COMMAND_RESET_CLEAR_FRAMES = 5
    COMMAND_MAX_LATCH_FRAMES = 220
    CIL_STEER_DEADBAND = 0.01
    CIL_MAX_STEER_RATE_PER_S = 2.5
    CIL_TURN_SPEED_CAP_KMH = 12.0
    CIL_STRAIGHT_JUNCTION_SPEED_CAP_KMH = 18.0
    CIL_CLEAR_PHASE_SPEED_CAP_KMH = 22.0
    CIL_ROUTE_FALLBACK_MIN_VALID = 0.55

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._enabled = False
        self._model = None
        self._device = None
        self._camera = None
        self._latest_rgb = None
        self._frame_lock = threading.Lock()
        self._last_steer = 0.0
        self._waiting_frame_logged = False
        self._collector: Optional[DataCollector] = None
        self._data_cameras = []
        self._video_writer = None
        self._video_output: Optional[Path] = None
        self._video_frames_written = 0
        self._video_max_frames = 0
        self._stop_requested = False
        self._visualizer = None
        self._route_map = None
        self._telemetry_fp = None
        self._telemetry_writer = None
        self._nav_agent = None
        self._spawn_points = []
        self._route_start_location = None
        self._route_destination_location = None
        self._route_history_xy: list[tuple[float, float]] = []
        self._cached_route_locations: list[Any] = []
        self._cached_route_tick: int = -1
        self._arrival_distance_m = 3.0
        self._destination_reached_logged = False
        self._last_speed_kmh = 0.0
        self._last_speed_plan: Dict[str, float] = {}
        self._route_lookahead_m = float(config.cil_route_lookahead_m)
        self._command_prep_time_s = float(config.cil_command_prep_time_s)
        self._command_trigger_min_m = float(config.cil_command_trigger_min_m)
        self._command_trigger_max_m = float(config.cil_command_trigger_max_m)
        self._command_retarget_window_s = float(config.cil_command_retarget_window_s)
        self._active_navigation_command = 0
        self._active_command_source = "none"
        self._command_phase = "cruise"
        self._command_latch_frames = 0
        self._command_entered_junction = False
        self._command_clear_frames = 0
        self._last_completed_turn_command = 0
        self._last_completed_turn_tick = -10000
        self._last_completed_turn_location = None
        self._last_command_debug: Dict[str, Any] = {}
        self._last_route_context: Dict[str, float] = {}
        self._last_replan_tick = -10000
        self._route_planner = CILRoutePlanner(
            route_lookahead_m=self._route_lookahead_m,
            arrival_distance_m=self._arrival_distance_m,
        )
        # Speed control (PID)
        self._speed_controller = SpeedPIDController(
            target_speed_kmh=config.target_speed_kmh,
            max_throttle=config.max_throttle,
            max_brake=config.max_brake,
        )

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        world = session.world
        if vehicle is None or world is None:
            logging.info("No CARLA vehicle/world available; CIL agent runs as noop.")
            return

        self._force_configured_spawn_pose(world, vehicle)
        self._ensure_vehicle_on_driving_lane(world, vehicle)

        self._model = self._load_cil_model()
        self._camera = self._spawn_camera(world, vehicle)
        self._camera.listen(self._on_camera_frame)
        self._init_navigation_agent(world, vehicle)

        self._collector = DataCollector(
            output_dir=self.config.collect_data_dir,
            enabled=self.config.collect_data,
            save_every_n=self.config.save_every_n,
        )
        self._collector.start()
        self._start_data_collection_cameras(world, vehicle)
        self._init_video_writer()
        if DrivingVisualizer is not None:
            self._visualizer = DrivingVisualizer(window_name="CIL Driving HUD")
        if RouteMapVisualizer is not None:
            self._route_map = RouteMapVisualizer(window_name="CIL Route Map")
        self._init_telemetry_logger()

        self._enabled = True
        logging.info("CIL agent is ready.")

    def _init_telemetry_logger(self) -> None:
        telemetry_path = (Path(__file__).resolve().parent / "outputs" / "cil_route_debug.csv")
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry_fp = telemetry_path.open("w", newline="", encoding="utf-8")
        self._telemetry_writer = csv.writer(self._telemetry_fp)
        self._telemetry_writer.writerow(
            [
                "tick",
                "speed_kmh",
                "target_speed_kmh",
                "route_valid",
                "target_x_m",
                "target_y_m",
                "distance_to_turn_m",
                "distance_to_junction_m",
                "turn_urgency",
                "command",
                "command_phase",
                "command_source",
                "steer",
                "throttle",
                "brake",
                "low_confidence_limit",
            ]
        )
        self._telemetry_fp.flush()

    def _zero_vehicle_motion(self, vehicle) -> None:
        if carla is None:
            return
        try:
            if hasattr(vehicle, "set_target_velocity"):
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            if hasattr(vehicle, "set_target_angular_velocity"):
                vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:
            pass
        try:
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=False,
                    reverse=False,
                )
            )
        except Exception:
            pass

    def _tick_world_for_spawn_alignment(self, world) -> None:
        try:
            if self.config.sync:
                world.tick()
            else:
                world.wait_for_tick(self.config.timeout)
        except Exception as exc:
            logging.warning("Could not tick world after CIL spawn alignment: %s", exc)

    def _apply_spawn_locked_spectator(self, world, spawn_transform) -> None:
        if carla is None or not self.config.lock_spectator_on_spawn:
            return

        try:
            forward = spawn_transform.get_forward_vector()
            loc = spawn_transform.location
            spectator_loc = carla.Location(
                x=float(loc.x) - float(forward.x) * float(self.config.spectator_follow_distance),
                y=float(loc.y) - float(forward.y) * float(self.config.spectator_follow_distance),
                z=float(loc.z) + float(self.config.spectator_height),
            )
            spectator_tf = carla.Transform(
                spectator_loc,
                carla.Rotation(
                    pitch=float(self.config.spectator_pitch),
                    yaw=float(spawn_transform.rotation.yaw),
                    roll=0.0,
                ),
            )
            world.get_spectator().set_transform(spectator_tf)
        except Exception as exc:
            logging.debug("Could not update spectator after CIL spawn alignment: %s", exc)

    def _force_configured_spawn_pose(self, world, vehicle) -> None:
        if carla is None or int(self.config.spawn_point) < 0:
            return

        self._spawn_points = world.get_map().get_spawn_points()
        if not self._spawn_points:
            raise RuntimeError("No spawn points found; cannot place CIL ego at route S.")

        spawn_idx = int(self.config.spawn_point) % len(self._spawn_points)
        configured_tf = self._spawn_points[spawn_idx]
        configured_loc = configured_tf.location
        current_loc = vehicle.get_location()
        initial_offset_m = math.hypot(
            float(current_loc.x - configured_loc.x),
            float(current_loc.y - configured_loc.y),
        )

        if initial_offset_m <= 0.35:
            self._zero_vehicle_motion(vehicle)
            self._route_start_location = configured_loc
            return

        spawn_location = carla.Location(
            x=float(configured_loc.x),
            y=float(configured_loc.y),
            z=float(configured_loc.z) + 0.20,
        )
        target_tf = carla.Transform(spawn_location, configured_tf.rotation)

        physics_disabled = False
        try:
            if hasattr(vehicle, "set_simulate_physics"):
                vehicle.set_simulate_physics(False)
                physics_disabled = True
            vehicle.set_transform(target_tf)
            self._zero_vehicle_motion(vehicle)
            self._tick_world_for_spawn_alignment(world)
            if hasattr(vehicle, "set_location"):
                after_tf_loc = vehicle.get_location()
                after_tf_offset_m = math.hypot(
                    float(after_tf_loc.x - configured_loc.x),
                    float(after_tf_loc.y - configured_loc.y),
                )
                if after_tf_offset_m > 0.75:
                    vehicle.set_location(spawn_location)
                    self._tick_world_for_spawn_alignment(world)
            if physics_disabled:
                vehicle.set_simulate_physics(True)
                self._zero_vehicle_motion(vehicle)
                self._tick_world_for_spawn_alignment(world)
        except Exception as exc:
            if physics_disabled:
                try:
                    vehicle.set_simulate_physics(True)
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to move CIL ego to configured route S spawn_point={spawn_idx}: {exc}"
            ) from exc

        verified_loc = vehicle.get_location()
        verified_offset_m = math.hypot(
            float(verified_loc.x - configured_loc.x),
            float(verified_loc.y - configured_loc.y),
        )
        if verified_offset_m > 0.75:
            raise RuntimeError(
                "CIL ego is not at configured route S after spawn alignment "
                f"(spawn_point={spawn_idx}, offset={verified_offset_m:.2f}m, "
                f"ego=({verified_loc.x:.1f}, {verified_loc.y:.1f}, {verified_loc.z:.1f}), "
                f"S=({configured_loc.x:.1f}, {configured_loc.y:.1f}, {configured_loc.z:.1f}))."
            )

        self._route_start_location = configured_loc
        self._apply_spawn_locked_spectator(world, target_tf)
        logging.warning(
            "CIL aligned ego to route S spawn_point=%d S=(%.1f, %.1f, %.1f) "
            "initial offset=%.2fm final offset=%.2fm.",
            spawn_idx,
            float(configured_loc.x),
            float(configured_loc.y),
            float(configured_loc.z),
            initial_offset_m,
            verified_offset_m,
        )

    def _ensure_vehicle_on_driving_lane(self, world, vehicle) -> None:
        if carla is None:
            return

        try:
            world_map = world.get_map()
            current_loc = vehicle.get_location()
            waypoint = world_map.get_waypoint(current_loc, project_to_road=True)
        except Exception:
            return

        if waypoint is None or not hasattr(waypoint, "transform"):
            return

        lane_type = getattr(waypoint, "lane_type", None)
        lane_is_driving = True
        try:
            lane_is_driving = bool(lane_type == carla.LaneType.Driving)
        except Exception:
            lane_is_driving = "driving" in str(lane_type).lower()

        wp_loc = waypoint.transform.location
        lateral_dist = math.hypot(float(current_loc.x - wp_loc.x), float(current_loc.y - wp_loc.y))
        has_fixed_spawn = int(self.config.spawn_point) >= 0
        if has_fixed_spawn:
            # Respect fixed spawn point exactly; do not relocate ego after spawn.
            should_snap = False
        else:
            should_snap = (not lane_is_driving) or lateral_dist > 4.0
        if not should_snap:
            return

        snap_location = carla.Location(x=float(wp_loc.x), y=float(wp_loc.y), z=float(wp_loc.z) + 0.30)
        snap_rotation = waypoint.transform.rotation
        try:
            vehicle.set_transform(carla.Transform(snap_location, snap_rotation))
            if hasattr(vehicle, "set_target_velocity"):
                vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            if hasattr(vehicle, "set_target_angular_velocity"):
                vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            logging.warning(
                "CIL snapped ego to nearest driving lane (off-lane spawn detected, dist=%.2fm).",
                lateral_dist,
            )
        except Exception:
            return

    def _init_navigation_agent(self, world, vehicle) -> None:
        self._spawn_points = world.get_map().get_spawn_points()

        if self._spawn_points and int(self.config.spawn_point) >= 0 and carla is not None:
            spawn_idx = int(self.config.spawn_point) % len(self._spawn_points)
            configured_tf = self._spawn_points[spawn_idx]
            ego_loc = vehicle.get_location()
            spawn_offset_m = math.hypot(
                float(ego_loc.x - configured_tf.location.x),
                float(ego_loc.y - configured_tf.location.y),
            )
            if spawn_offset_m > 1.5:
                try:
                    vehicle.set_transform(configured_tf)
                    if hasattr(vehicle, "set_target_velocity"):
                        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    if hasattr(vehicle, "set_target_angular_velocity"):
                        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    logging.warning(
                        "CIL force-aligned ego to configured spawn_point=%d (offset=%.2fm).",
                        spawn_idx,
                        spawn_offset_m,
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to force-align ego to configured spawn_point=%d: %s",
                        spawn_idx,
                        exc,
                    )

        endpoint_warnings = self._route_planner.configure_endpoints(
            spawn_points=self._spawn_points,
            vehicle_location=vehicle.get_location(),
            configured_spawn_index=int(self.config.spawn_point),
            configured_destination_index=int(self.config.destination_point),
        )
        self._route_start_location = self._route_planner.start_location
        self._route_destination_location = self._route_planner.destination_location
        for message in endpoint_warnings:
            logging.warning(message)

        if self._route_start_location is not None and self._route_destination_location is not None:
            ego_loc = vehicle.get_location()
            start_offset_m = math.hypot(
                float(ego_loc.x - self._route_start_location.x),
                float(ego_loc.y - self._route_start_location.y),
            )
            logging.info(
                "CIL route anchors | S=(%.1f, %.1f) D=(%.1f, %.1f) ego=(%.1f, %.1f) S-ego=%.2fm",
                float(self._route_start_location.x),
                float(self._route_start_location.y),
                float(self._route_destination_location.x),
                float(self._route_destination_location.y),
                float(ego_loc.x),
                float(ego_loc.y),
                start_offset_m,
            )

        if not self._spawn_points:
            logging.warning("No spawn points found to build navigation route for CIL command extraction.")
            self._nav_agent = None
            return

        ensure_navigation_agent_imports()
        nav_type = self.config.nav_agent_type.lower()
        try:
            if nav_type == "behavior":
                if BehaviorAgent is None:
                    raise RuntimeError("BehaviorAgent missing")
                self._nav_agent = BehaviorAgent(vehicle, behavior="normal")
            else:
                if BasicAgent is None:
                    raise RuntimeError("BasicAgent missing")
                self._nav_agent = BasicAgent(vehicle, target_speed=max(10.0, self.config.target_speed_kmh))
            self._set_configured_destination(vehicle)
            logging.info("CIL navigation planner initialized: %s", self.config.nav_agent_type)
        except Exception as exc:
            self._nav_agent = None
            logging.warning(
                "CIL navigation planner unavailable; command injection will stay at follow-lane (0). Reason: %s",
                exc,
            )

    def _set_configured_destination(self, vehicle) -> None:
        self._route_destination_location = self._route_planner.destination_location
        if self._nav_agent is None or self._route_destination_location is None:
            return
        start_loc = (
            self._route_start_location
            if self._route_start_location is not None
            else vehicle.get_location()
        )
        set_navigation_destination(self._nav_agent, start_loc, self._route_destination_location)

    def _maybe_replan_route(self, step_idx: int, vehicle) -> None:
        if self._nav_agent is None or self._route_destination_location is None:
            return
        if step_idx - int(self._last_replan_tick) < 30:
            return

        route_locations = self._collect_route_locations(max_points=24)
        if len(route_locations) >= 6:
            return

        try:
            current_loc = vehicle.get_location()
            set_navigation_destination(self._nav_agent, current_loc, self._route_destination_location)
            self._last_replan_tick = int(step_idx)
            logging.info("CIL replanned route (planner points=%d).", len(route_locations))
        except Exception as exc:
            self._last_replan_tick = int(step_idx)
            logging.debug("CIL replan attempt failed: %s", exc)

    def _maybe_replan_route_from_context(
        self,
        step_idx: int,
        vehicle,
        route_context: Optional[Dict[str, float]],
    ) -> None:
        if self._nav_agent is None or self._route_destination_location is None:
            return
        if not isinstance(route_context, dict):
            return
        if step_idx - int(self._last_replan_tick) < 12:
            return

        route_valid = clamp(float(route_context.get("route_valid", 0.0)), 0.0, 1.0)
        lateral_abs = abs(float(route_context.get("target_y_m", 0.0)))
        heading_abs = abs(float(route_context.get("heading_error_deg", 0.0)))

        is_fallback = float(route_context.get("is_fallback", 0.0))

        # CHỈ REPLAN KHI THỰC SỰ CẦN THIẾT (Không replan khi đang ở chế độ dự phòng)
        if (route_valid >= 0.20 and lateral_abs <= 5.0) or is_fallback > 0.5:
            return

        try:
            current_loc = vehicle.get_location()
            set_navigation_destination(self._nav_agent, current_loc, self._route_destination_location)
            self._last_replan_tick = int(step_idx)
            logging.info(
                "CIL replanned route (unstable context: valid=%.2f lateral=%.1fm heading=%.1fdeg).",
                route_valid,
                lateral_abs,
                heading_abs,
            )
        except Exception as exc:
            self._last_replan_tick = int(step_idx)
            logging.debug("CIL context-based replan failed: %s", exc)

    def _distance_to_destination(self, vehicle_location: Any) -> Optional[float]:
        return self._route_planner.distance_to_destination(vehicle_location)

    def _distance_from_route_start_m(self) -> float:
        if self._route_start_location is None:
            return float("inf")
        if self.session is None or self.session.ego_vehicle is None:
            return float("inf")
        ego_loc = self.session.ego_vehicle.get_location()
        return math.hypot(
            float(ego_loc.x - self._route_start_location.x),
            float(ego_loc.y - self._route_start_location.y),
        )

    def _distance_from_last_completed_turn_m(self) -> float:
        if self._last_completed_turn_location is None:
            return float("inf")
        if self.session is None or self.session.ego_vehicle is None:
            return float("inf")
        ego_loc = self.session.ego_vehicle.get_location()
        return math.hypot(
            float(ego_loc.x - self._last_completed_turn_location.x),
            float(ego_loc.y - self._last_completed_turn_location.y),
        )

    def _current_waypoint(self):
        if self.session is None or self.session.world is None or self.session.ego_vehicle is None:
            return None
        world = self.session.world
        vehicle = self.session.ego_vehicle
        try:
            return world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
        except Exception:
            return None

    def _is_in_junction(self) -> bool:
        waypoint = self._current_waypoint()
        return bool(waypoint is not None and getattr(waypoint, "is_junction", False))

    def _command_lookahead_m(self) -> float:
        speed = max(0.0, float(self._last_speed_kmh))
        return clamp(4.0 + 0.22 * speed, 4.0, 12.0)

    @staticmethod
    def _sample_polyline_point(
        points_xy: list[tuple[float, float]],
        cumulative_s: list[float],
        target_s: float,
    ) -> Optional[tuple[float, float]]:
        if not points_xy or not cumulative_s:
            return None
        if len(points_xy) == 1:
            return points_xy[0]

        target = max(0.0, float(target_s))
        if target <= cumulative_s[0]:
            return points_xy[0]
        if target >= cumulative_s[-1]:
            return points_xy[-1]

        for idx in range(1, len(cumulative_s)):
            if cumulative_s[idx] < target:
                continue
            s0 = cumulative_s[idx - 1]
            s1 = cumulative_s[idx]
            p0 = points_xy[idx - 1]
            p1 = points_xy[idx]
            ratio = (target - s0) / max(1e-6, s1 - s0)
            x = p0[0] + ratio * (p1[0] - p0[0])
            y = p0[1] + ratio * (p1[1] - p0[1])
            return (float(x), float(y))
        return points_xy[-1]

    @staticmethod
    def _signed_curvature_3pts(
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        ax = float(p1[0] - p0[0])
        ay = float(p1[1] - p0[1])
        bx = float(p2[0] - p1[0])
        by = float(p2[1] - p1[1])
        cx = float(p2[0] - p0[0])
        cy = float(p2[1] - p0[1])

        a = math.hypot(ax, ay)
        b = math.hypot(bx, by)
        c = math.hypot(cx, cy)
        if a <= 1e-4 or b <= 1e-4 or c <= 1e-4:
            return 0.0

        cross = ax * cy - ay * cx
        return float((2.0 * cross) / max(1e-6, a * b * c))

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        return float(wrapped)

    @staticmethod
    def _planner_item_to_waypoint(item: Any):
        if hasattr(item, "transform"):
            return item
        if isinstance(item, (tuple, list)) and len(item) >= 1 and hasattr(item[0], "transform"):
            return item[0]
        return None

    @staticmethod
    def _road_option_to_turn_command(road_option: Any) -> int:
        if road_option is None:
            return 0

        raw_value = getattr(road_option, "value", road_option)
        try:
            numeric_value = int(raw_value)
            if numeric_value in (5, 6):
                # Ignore lane-change hints for CIL command injection.
                return 0
        except (TypeError, ValueError):
            pass

        option_name = getattr(road_option, "name", str(road_option)).lower().replace(" ", "")
        if "change_lane" in option_name or "changelane" in option_name:
            return 0

        command = map_road_option_to_command(road_option)
        return int(command) if command in (1, 2, 3) else 0

    def _extract_upcoming_turn_signal(self) -> tuple[int, float]:
        if self._nav_agent is None:
            return 0, float("inf")
        planner = None
        if hasattr(self._nav_agent, "get_local_planner"):
            planner = self._nav_agent.get_local_planner()
        if planner is None:
            return 0, float("inf")

        distance_to_junction_m = self._distance_to_next_junction_m()
        if not math.isfinite(distance_to_junction_m):
            distance_to_junction_m = float("inf")

        vehicle_loc = None
        if self.session is not None and self.session.ego_vehicle is not None:
            vehicle_loc = self.session.ego_vehicle.get_location()

        direct_road_option = getattr(planner, "target_road_option", None)
        command = self._road_option_to_turn_command(direct_road_option)
        if command != 0:
            return command, float(distance_to_junction_m)

        direct_road_option = getattr(planner, "_target_road_option", None)
        command = self._road_option_to_turn_command(direct_road_option)
        if command != 0:
            return command, float(distance_to_junction_m)

        queue_attr = getattr(planner, "_waypoints_queue", None)
        if queue_attr:
            try:
                planner_items = list(queue_attr)
            except Exception:
                planner_items = []
            cumulative_distance_m = 0.0
            prev_loc = vehicle_loc
            for item in planner_items[:96]:
                waypoint = self._planner_item_to_waypoint(item)
                wp_loc = waypoint.transform.location if waypoint is not None else None
                if wp_loc is not None and prev_loc is not None:
                    cumulative_distance_m += math.hypot(
                        float(wp_loc.x - prev_loc.x),
                        float(wp_loc.y - prev_loc.y),
                    )
                if wp_loc is not None:
                    prev_loc = wp_loc

                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                command = self._road_option_to_turn_command(item[1])
                if command != 0:
                    command_distance_m = float(cumulative_distance_m)
                    if not math.isfinite(command_distance_m) or command_distance_m <= 0.0:
                        command_distance_m = float(distance_to_junction_m)
                    return command, command_distance_m
        return 0, float("inf")

    def _extract_upcoming_turn_command(self) -> int:
        command, _ = self._extract_upcoming_turn_signal()
        return int(command)

    def _distance_to_next_junction_m(self, max_probe_m: float = 70.0, step_m: float = 1.5) -> float:
        waypoint = self._current_waypoint()
        if waypoint is None:
            return float("inf")
        if getattr(waypoint, "is_junction", False):
            return 0.0

        travelled = 0.0
        probe_wp = waypoint
        step = max(0.5, float(step_m))
        max_steps = max(1, int(max_probe_m / step))

        for _ in range(max_steps):
            try:
                next_wps = probe_wp.next(step)
            except Exception:
                next_wps = []
            if not next_wps:
                break

            if len(next_wps) == 1:
                probe_wp = next_wps[0]
            else:
                base_yaw = float(probe_wp.transform.rotation.yaw)
                probe_wp = min(
                    next_wps,
                    key=lambda wp: abs(
                        self._normalize_angle_deg(float(wp.transform.rotation.yaw) - base_yaw)
                    ),
                )

            travelled += step
            if getattr(probe_wp, "is_junction", False):
                return float(travelled)

        return float("inf")

    def _command_trigger_distance_m(self, speed_kmh: float) -> float:
        speed_mps = max(0.0, float(speed_kmh)) / 3.6
        return clamp(
            speed_mps * self._command_prep_time_s,
            self._command_trigger_min_m,
            self._command_trigger_max_m,
        )

    def _fallback_command_from_route_context(
        self,
        route_context: Optional[Dict[str, float]],
        in_junction: bool,
        distance_to_junction_m: float,
        trigger_distance_m: float,
    ) -> int:
        if not isinstance(route_context, dict):
            return 0

        route_valid = float(route_context.get("route_valid", 0.0))
        heading_error_deg = float(route_context.get("heading_error_deg", 0.0))
        curvature_1pm = float(route_context.get("curvature_1pm", 0.0))
        target_x_m = float(route_context.get("target_x_m", self._route_lookahead_m))
        target_y_m = float(route_context.get("target_y_m", 0.0))
        turn_urgency = float(route_context.get("turn_urgency", 0.0))
        distance_to_turn_m = float(route_context.get("distance_to_turn_m", 90.0))
        junction_proximity = float(route_context.get("junction_proximity", 0.0))

        heading_abs = abs(heading_error_deg)
        lateral_abs = abs(target_y_m)
        near_junction_from_map = (
            math.isfinite(distance_to_junction_m)
            and distance_to_junction_m <= (trigger_distance_m + 4.0)
        )
        near_for_fallback = in_junction or near_junction_from_map
        emergency_recovery_case = (
            route_valid < 0.75
            and heading_abs >= 55.0
            and lateral_abs >= 10.0
            and distance_to_turn_m <= 8.0
        )

        if route_valid < self.CIL_ROUTE_FALLBACK_MIN_VALID and not emergency_recovery_case:
            return 0
        if not near_for_fallback and not emergency_recovery_case:
            return 0

        inferred_from_route = int(self._route_planner.command_from_context(route_context))
        if inferred_from_route in (1, 2, 3):
            if inferred_from_route in (1, 2):
                if abs(heading_error_deg) < 14.0 and distance_to_turn_m > max(8.0, 1.15 * trigger_distance_m):
                    return 0
            return inferred_from_route

        lateral_ratio = clamp(target_y_m / max(3.0, abs(target_x_m)), -1.0, 1.0)
        signed_turn = heading_error_deg + math.degrees(math.atan(curvature_1pm * 8.0)) + 10.0 * lateral_ratio

        if turn_urgency >= 0.35 and distance_to_turn_m <= max(10.0, 1.35 * trigger_distance_m):
            if signed_turn > 8.0:
                return 1
            if signed_turn < -8.0:
                return 2

        if in_junction and turn_urgency >= 0.50 and distance_to_turn_m <= 10.0 and abs(signed_turn) <= 8.0:
            return 3

        if turn_urgency >= 0.35 and signed_turn > 11.0:
            return 1
        if turn_urgency >= 0.35 and signed_turn < -11.0:
            return 2
        return 0

    def _update_distance_based_command(
        self,
        speed_kmh: float,
        route_context: Optional[Dict[str, float]] = None,
        step_idx: Optional[int] = None,
    ) -> tuple[int, Dict[str, Any]]:
        current_step = int(step_idx) if step_idx is not None else -1
        in_junction = self._is_in_junction()
        upcoming_command, distance_to_upcoming_turn_m = self._extract_upcoming_turn_signal()
        command_source = "planner" if upcoming_command in (1, 2, 3) else "none"
        distance_to_junction_m = self._distance_to_next_junction_m()
        trigger_distance_m = self._command_trigger_distance_m(speed_kmh)
        reset_distance_m = max(8.0, 0.55 * trigger_distance_m)
        if not math.isfinite(distance_to_upcoming_turn_m):
            distance_to_upcoming_turn_m = float("inf")
        route_distance_to_turn_m = float("inf")
        if isinstance(route_context, dict):
            route_distance_to_turn_m = float(route_context.get("distance_to_turn_m", float("inf")))
            if not math.isfinite(route_distance_to_turn_m):
                route_distance_to_turn_m = float("inf")
        if in_junction:
            command_distance_m = min(distance_to_upcoming_turn_m, route_distance_to_turn_m)
        else:
            command_distance_m = distance_to_upcoming_turn_m
            if math.isfinite(route_distance_to_turn_m):
                command_distance_m = min(command_distance_m, route_distance_to_turn_m)

        route_valid = 0.0
        route_heading_abs = 0.0
        route_lateral_abs = 0.0
        route_turn_urgency = 0.0
        route_inferred_command = 0
        planner_command_suppressed = False
        planner_suppression_reason = "none"
        if isinstance(route_context, dict):
            route_valid = clamp(float(route_context.get("route_valid", 0.0)), 0.0, 1.0)
            route_heading_abs = abs(float(route_context.get("heading_error_deg", 0.0)))
            route_lateral_abs = abs(float(route_context.get("target_y_m", 0.0)))
            route_turn_urgency = clamp(float(route_context.get("turn_urgency", 0.0)), 0.0, 1.0)
            route_inferred_command = int(self._route_planner.command_from_context(route_context))

        if command_source == "planner" and upcoming_command in (1, 2):
            planner_conflict = (
                route_inferred_command in (1, 2)
                and route_inferred_command != upcoming_command
                and route_heading_abs >= 16.0
                and route_lateral_abs >= 3.0
            )
            low_confidence_planner = (
                route_valid < 0.55
                and not in_junction
                and command_distance_m > max(10.0, 0.95 * trigger_distance_m)
            )
            if planner_conflict or low_confidence_planner:
                planner_command_suppressed = True
                planner_suppression_reason = "conflict" if planner_conflict else "low_confidence"
                upcoming_command = 0
                command_source = "planner_suppressed"

        distance_from_start_m = self._distance_from_route_start_m()
        warmup_block_commands = (
            math.isfinite(distance_from_start_m)
            and distance_from_start_m < self.COMMAND_WARMUP_DISTANCE_M
        )
        if warmup_block_commands:
            upcoming_command = 0
            command_source = "none"
            if self._active_navigation_command != 0:
                self._active_navigation_command = 0
                self._active_command_source = "none"
                self._command_phase = "cruise"
                self._command_latch_frames = 0
                self._command_entered_junction = False
                self._command_clear_frames = 0

        if self._nav_agent is not None and upcoming_command == 0 and not warmup_block_commands:
            inferred_command = self._fallback_command_from_route_context(
                route_context=route_context,
                in_junction=in_junction,
                distance_to_junction_m=distance_to_junction_m,
                trigger_distance_m=trigger_distance_m,
            )
            if inferred_command in (1, 2, 3):
                upcoming_command = inferred_command
                command_source = "route_fallback"
                command_distance_m = (
                    route_distance_to_turn_m
                    if in_junction
                    else min(distance_to_junction_m, route_distance_to_turn_m)
                )

        command_retargeted = False
        if self._active_navigation_command != 0 and upcoming_command in (1, 2, 3):
            dt_s = max(1e-3, float(self.config.fixed_delta))
            retarget_window_frames = max(
                20,
                int(self._command_retarget_window_s / max(1e-3, float(self.config.fixed_delta))),
            )
            in_junction_swap_window = max(8, int(0.9 / dt_s))
            if (
                upcoming_command != self._active_navigation_command
                and self._command_latch_frames <= retarget_window_frames
            ):
                if not self._command_entered_junction:
                    self._active_navigation_command = int(upcoming_command)
                    self._active_command_source = command_source
                    command_retargeted = True
                elif (
                    self._active_navigation_command in (1, 2)
                    and upcoming_command in (1, 2)
                    and self._command_latch_frames <= in_junction_swap_window
                ):
                    # If we latched the wrong left/right direction right at
                    # junction entry, allow one quick swap to avoid locking
                    # into the opposite turn.
                    self._active_navigation_command = int(upcoming_command)
                    self._active_command_source = command_source
                    command_retargeted = True
                elif self._active_navigation_command == 3 and upcoming_command in (1, 2):
                    # Upgrade a previously ambiguous straight command to a clear
                    # left/right decision once route context becomes reliable.
                    self._active_navigation_command = int(upcoming_command)
                    self._active_command_source = command_source
                    command_retargeted = True

        if self._active_navigation_command == 0:
            self._command_phase = "cruise"
            self._command_latch_frames = 0
            self._command_clear_frames = 0
            self._command_entered_junction = False

            near_enough_for_latch = command_distance_m <= max(10.0, 1.10 * trigger_distance_m)
            in_junction_with_near_turn = in_junction and command_distance_m <= max(12.0, 1.30 * trigger_distance_m)
            fallback_latch_distance = max(
                14.0,
                (1.40 if in_junction else 1.20) * trigger_distance_m,
            )
            allow_source_latch = command_source == "planner" or (
                command_source == "route_fallback"
                and command_distance_m <= fallback_latch_distance
            )
            recent_opposite_turn_block = False
            if (
                upcoming_command in (1, 2)
                and self._last_completed_turn_command in (1, 2)
                and upcoming_command != self._last_completed_turn_command
            ):
                elapsed_frames = (
                    current_step - int(self._last_completed_turn_tick)
                    if current_step >= 0
                    else 10**9
                )
                moved_since_turn_m = self._distance_from_last_completed_turn_m()
                if (
                    elapsed_frames < 180
                    and moved_since_turn_m < 42.0
                    and not in_junction
                    and route_turn_urgency < 0.90
                ):
                    recent_opposite_turn_block = True

            should_latch = allow_source_latch and upcoming_command in (1, 2, 3) and (
                near_enough_for_latch or in_junction_with_near_turn
            )
            if recent_opposite_turn_block:
                should_latch = False
            if should_latch:
                self._active_navigation_command = int(upcoming_command)
                self._active_command_source = str(command_source)
                self._command_phase = "in_junction" if in_junction else "armed"
                self._command_latch_frames = 1
                self._command_entered_junction = bool(in_junction)
        else:
            self._command_latch_frames += 1
            if in_junction:
                self._command_phase = "in_junction"
                self._command_entered_junction = True
                self._command_clear_frames = 0
            else:
                if self._command_entered_junction:
                    self._command_phase = "clear"
                    self._command_clear_frames += 1
                elif command_distance_m > trigger_distance_m + 20.0:
                    # If we armed too early but never entered a junction, release command.
                    self._active_navigation_command = 0
                    self._active_command_source = "none"
                    self._command_phase = "cruise"
                    self._command_latch_frames = 0
                    self._command_clear_frames = 0

            should_reset = False
            stale_fallback_frames = max(8, int(1.0 / max(1e-3, float(self.config.fixed_delta))))
            if (
                self._active_command_source == "route_fallback"
                and command_source == "none"
                and (not self._command_entered_junction)
                and route_distance_to_turn_m > (trigger_distance_m + 16.0)
                and self._command_latch_frames >= stale_fallback_frames
            ):
                should_reset = True
            if self._command_entered_junction and not in_junction:
                if (
                    self._command_clear_frames >= self.COMMAND_RESET_CLEAR_FRAMES
                    and distance_to_junction_m > reset_distance_m
                ):
                    if self._active_navigation_command in (1, 2):
                        self._last_completed_turn_command = int(self._active_navigation_command)
                        self._last_completed_turn_tick = int(current_step)
                        if self.session is not None and self.session.ego_vehicle is not None:
                            self._last_completed_turn_location = self.session.ego_vehicle.get_location()
                    should_reset = True
            if self._command_latch_frames >= self.COMMAND_MAX_LATCH_FRAMES:
                should_reset = True

            if should_reset:
                self._active_navigation_command = 0
                self._active_command_source = "none"
                self._command_phase = "cruise"
                self._command_latch_frames = 0
                self._command_entered_junction = False
                self._command_clear_frames = 0

        command_debug: Dict[str, Any] = {
            "phase": self._command_phase,
            "upcoming_command": int(upcoming_command),
            "active_command": int(self._active_navigation_command),
            "active_source": str(self._active_command_source),
            "upcoming_source": command_source,
            "retargeted": bool(command_retargeted),
            "in_junction": bool(in_junction),
            "distance_to_turn_m": float(command_distance_m),
            "route_distance_to_turn_m": float(route_distance_to_turn_m),
            "distance_to_junction_m": float(distance_to_junction_m),
            "trigger_distance_m": float(trigger_distance_m),
            "reset_distance_m": float(reset_distance_m),
            "latch_frames": int(self._command_latch_frames),
            "distance_from_start_m": float(distance_from_start_m),
            "warmup_block_commands": bool(warmup_block_commands),
            "route_inferred_command": int(route_inferred_command),
            "planner_command_suppressed": bool(planner_command_suppressed),
            "planner_suppression_reason": planner_suppression_reason,
        }
        self._last_command_debug = command_debug
        return int(self._active_navigation_command), command_debug

    def _fallback_route_context_from_destination(
        self,
        base_context: Dict[str, float],
        vehicle_loc: Any,
        forward: Any,
        right: Any,
        lookahead_m: float,
    ) -> Dict[str, float]:
        if self._route_destination_location is None:
            self._last_route_context = base_context
            return base_context

        dx = float(self._route_destination_location.x - vehicle_loc.x)
        dy = float(self._route_destination_location.y - vehicle_loc.y)
        dist_xy = math.hypot(dx, dy)

        target_x = float(forward.x) * dx + float(forward.y) * dy
        target_y = -(float(right.x) * dx + float(right.y) * dy)
        target_x = clamp(target_x, -80.0, 80.0)
        target_y = clamp(target_y, -40.0, 40.0)

        x_for_heading = target_x if abs(target_x) > 1e-3 else (1e-3 if target_y >= 0.0 else -1e-3)
        heading_error_deg = math.degrees(math.atan2(target_y, x_for_heading))
        curvature_1pm = math.tan(math.radians(heading_error_deg)) / max(6.0, float(lookahead_m))

        heading_urgency = clamp((abs(heading_error_deg) - 4.0) / 22.0, 0.0, 1.0)
        turn_urgency = max(heading_urgency, clamp(abs(curvature_1pm) / 0.10, 0.0, 1.0))
        junction_proximity = 0.45 if self._is_near_junction(lookahead_m=lookahead_m) else 0.0

        fallback = dict(base_context)
        fallback.update(
            {
                "route_valid": 0.6,
                "target_x_m": float(target_x),
                "target_y_m": float(target_y),
                "heading_error_deg": float(heading_error_deg),
                "curvature_1pm": float(curvature_1pm),
                "distance_to_turn_m": float(max(2.0, min(90.0, 0.35 * dist_xy))),
                "distance_to_junction_m": float(12.0 if junction_proximity > 0.0 else 90.0),
                "turn_urgency": float(turn_urgency),
                "junction_proximity": float(junction_proximity),
            }
        )
        self._last_route_context = fallback
        return fallback

    def _compute_route_context(self, lookahead_m: Optional[float] = None) -> Dict[str, float]:
        if lookahead_m is None:
            lookahead_m = max(self._command_lookahead_m(), self._route_lookahead_m)
        context: Dict[str, float] = {
            "route_valid": 0.0,
            "target_x_m": float(max(4.0, lookahead_m)),
            "target_y_m": 0.0,
            "heading_error_deg": 0.0,
            "curvature_1pm": 0.0,
            "distance_to_turn_m": 90.0,
            "distance_to_junction_m": 90.0,
            "turn_urgency": 0.0,
            "junction_proximity": 0.0,
        }

        if self.session is None or self.session.ego_vehicle is None:
            self._last_route_context = context
            self._route_planner.last_route_context = context
            return context

        vehicle = self.session.ego_vehicle
        vehicle_loc = vehicle.get_location()
        transform = vehicle.get_transform()
        route_locations = self._collect_route_locations(max_points=80)
        world_map = self.session.world.get_map() if self.session is not None and self.session.world is not None else None
        near_junction = self._is_near_junction(lookahead_m=float(lookahead_m))

        context = self._route_planner.compute_route_context(
            vehicle_location=vehicle_loc,
            vehicle_transform=transform,
            route_locations=route_locations,
            world_map=world_map,
            lookahead_m=float(lookahead_m),
            near_junction=near_junction,
        )
        self._last_route_context = context
        return context

    def _is_near_junction(self, lookahead_m: float = 12.0) -> bool:
        waypoint = self._current_waypoint()
        if waypoint is None:
            return False
        if waypoint.is_junction:
            return True

        probe_dists = sorted(
            {
                max(3.0, 0.35 * float(lookahead_m)),
                max(5.0, 0.65 * float(lookahead_m)),
                max(8.0, float(lookahead_m)),
            }
        )

        for probe_dist in probe_dists:
            try:
                next_wps = waypoint.next(probe_dist)
            except Exception:
                next_wps = []
            for wp in next_wps:
                if getattr(wp, "is_junction", False):
                    return True
        return False

    def _request_stop_at_destination(self, reason: str, distance_m: Optional[float] = None) -> None:
        if self._stop_requested:
            return

        if self.session is not None and self.session.ego_vehicle is not None and carla is not None:
            stop_control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=float(max(0.85, self.config.max_brake)),
                hand_brake=False,
                reverse=False,
            )
            self.session.ego_vehicle.apply_control(stop_control)

        self._stop_requested = True
        if not self._destination_reached_logged:
            if distance_m is None:
                logging.info("CIL reached destination D (%s). Vehicle stopping.", reason)
            else:
                logging.info(
                    "CIL reached destination D (%s, distance=%.2fm). Vehicle stopping.",
                    reason,
                    distance_m,
                )
            self._destination_reached_logged = True

    def _refresh_planner_state(self) -> None:
        if self._nav_agent is None:
            return

        planner = None
        if hasattr(self._nav_agent, "get_local_planner"):
            planner = self._nav_agent.get_local_planner()

        if planner is not None and hasattr(planner, "run_step"):
            try:
                planner.run_step()
                return
            except TypeError:
                try:
                    planner.run_step(debug=False)
                    return
                except Exception:
                    pass
            except Exception:
                pass

        if hasattr(self._nav_agent, "run_step"):
            try:
                self._nav_agent.run_step()
            except Exception:
                pass

    def _collect_route_locations(self, max_points: int = 260, step_idx: Optional[int] = None) -> list[Any]:
        anchor_location = None
        if self.session is not None and self.session.ego_vehicle is not None:
            anchor_location = self.session.ego_vehicle.get_location()

        route_locations = self._route_planner.collect_route_locations(
            nav_agent=self._nav_agent,
            max_points=max_points,
            anchor_location=anchor_location,
        )
        if route_locations:
            self._cached_route_locations = list(route_locations)
            if step_idx is not None:
                self._cached_route_tick = int(step_idx)
            return route_locations

        if self._cached_route_locations:
            return list(self._cached_route_locations[: max(1, int(max_points))])
        return []

    def _build_lane_center_hint(self, vehicle_location: Any, vehicle_transform: Any) -> Dict[str, float]:
        if self.session is None or self.session.world is None:
            return {"valid": 0.0}
        if vehicle_location is None or vehicle_transform is None:
            return {"valid": 0.0}

        try:
            waypoint = self.session.world.get_map().get_waypoint(vehicle_location, project_to_road=True)
        except Exception:
            waypoint = None
        if waypoint is None:
            return {"valid": 0.0}

        lane_center = waypoint.transform.location
        forward = vehicle_transform.get_forward_vector()
        right = vehicle_transform.get_right_vector()
        dx = float(lane_center.x - vehicle_location.x)
        dy = float(lane_center.y - vehicle_location.y)
        target_x = float(forward.x) * dx + float(forward.y) * dy
        target_y_left = -(float(right.x) * dx + float(right.y) * dy)

        heading_error = self._normalize_angle_deg(
            float(waypoint.transform.rotation.yaw) - float(vehicle_transform.rotation.yaw)
        )
        lane_width = float(getattr(waypoint, "lane_width", 3.5))
        lane_half_width = max(1.2, 0.5 * lane_width)

        return {
            "valid": 1.0,
            "target_x_m": float(clamp(target_x, 2.5, 30.0)),
            "target_y_m": float(clamp(target_y_left, -12.0, 12.0)),
            "heading_error_deg": float(heading_error),
            "lane_half_width_m": float(lane_half_width),
        }

    def _update_route_history(self, location: Any) -> None:
        self._route_planner.update_route_history(location)
        self._route_history_xy = list(self._route_planner.route_history_xy)

    def _draw_route_debug_overlay(self, step_idx: int, route_locations: list[Any], vehicle_location: Any) -> None:
        if self.session is None or self.session.world is None or carla is None:
            return
        if step_idx % 5 != 0:
            return

        debug = self.session.world.debug
        life_time = max(0.20, self.config.fixed_delta * 4.0 if self.config.sync else 0.35)
        lift = 0.35

        try:
            for idx in range(0, max(0, len(route_locations) - 1)):
                loc_a = route_locations[idx]
                loc_b = route_locations[idx + 1]
                p0 = carla.Location(x=float(loc_a.x), y=float(loc_a.y), z=float(loc_a.z) + lift)
                p1 = carla.Location(x=float(loc_b.x), y=float(loc_b.y), z=float(loc_b.z) + lift)
                debug.draw_line(
                    p0,
                    p1,
                    thickness=0.08,
                    color=carla.Color(r=0, g=210, b=255),
                    life_time=life_time,
                    persistent_lines=False,
                )

            if self._route_start_location is not None:
                s = carla.Location(
                    x=float(self._route_start_location.x),
                    y=float(self._route_start_location.y),
                    z=float(self._route_start_location.z) + 0.55,
                )
                debug.draw_point(s, size=0.12, color=carla.Color(r=40, g=230, b=40), life_time=life_time, persistent_lines=False)
                debug.draw_string(s, "START", False, carla.Color(r=40, g=230, b=40), life_time, False)

            if self._route_destination_location is not None:
                d = carla.Location(
                    x=float(self._route_destination_location.x),
                    y=float(self._route_destination_location.y),
                    z=float(self._route_destination_location.z) + 0.55,
                )
                debug.draw_point(d, size=0.12, color=carla.Color(r=255, g=80, b=80), life_time=life_time, persistent_lines=False)
                debug.draw_string(d, "DEST", False, carla.Color(r=255, g=80, b=80), life_time, False)

            if vehicle_location is not None:
                e = carla.Location(
                    x=float(vehicle_location.x),
                    y=float(vehicle_location.y),
                    z=float(vehicle_location.z) + 0.55,
                )
                debug.draw_point(e, size=0.13, color=carla.Color(r=255, g=255, b=0), life_time=life_time, persistent_lines=False)
                debug.draw_string(e, "EGO", False, carla.Color(r=255, g=255, b=0), life_time, False)
        except Exception:
            pass

    def _load_cil_model(self):
        if torch is None:
            raise RuntimeError("PyTorch is required for CIL agent.")
        if cv2 is None:
            raise RuntimeError("opencv-python is required for CIL agent.")
        if np is None:
            raise RuntimeError("numpy is required for CIL agent.")
        if CIL_NvidiaCNN is None:
            raise RuntimeError("Cannot import CIL_NvidiaCNN from core_perception.cnn_model.")

        model_path = resolve_cil_model_path(self.config.cil_model_path)
        if self.config.cil_model_path.lower() == "auto":
            logging.info("Auto-selected CIL model path: %s", model_path)
        if not model_path.exists():
            models_dir = Path(__file__).resolve().parent / "models"
            existing = ", ".join(str(p.name) for p in sorted(models_dir.glob("*.pth")))
            if not existing:
                existing = "no .pth file found in models/"
            raise RuntimeError(f"CIL model file not found: {model_path}. Available: {existing}")

        device_name = self.config.model_device.lower()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            device_name = "cpu"

        self._device = torch.device(device_name)

        checkpoint = torch.load(model_path, map_location=self._device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise RuntimeError("Unsupported CIL checkpoint format. Expected state_dict.")

        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {
                key.replace("module.", "", 1): value for key, value in state_dict.items()
            }

        model = CIL_NvidiaCNN().to(self._device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        logging.info("Loaded CIL model from %s on %s", model_path, self._device)
        return model

    def _spawn_camera(self, world, vehicle):
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick") and self.config.sync:
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logging.info("Attached RGB camera to CIL agent ego vehicle.")
        return camera

    def _start_data_collection_cameras(self, world, vehicle) -> None:
        if not self.config.collect_data or self._collector is None:
            return

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick"):
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_setups = [
            ("center", carla.Transform(carla.Location(x=1.5, y=0.0, z=2.2), carla.Rotation(pitch=-8.0))),
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(yaw=-25.0, pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(yaw=25.0, pitch=-8.0))),
        ]
        for side, transform in camera_setups:
            sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            sensor.listen(self._collector.make_sensor_callback(side))
            self._data_cameras.append(sensor)

        logging.info("CIL data collection cameras are attached (center/left/right).")

    def _init_video_writer(self) -> None:
        if not self.config.record_video:
            return
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video recording.")

        output_path = Path(self.config.video_output_path)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(self.config.video_fps),
            (int(self.config.camera_width), int(self.config.camera_height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")

        self._video_writer = writer
        self._video_output = output_path
        self._video_max_frames = max(1, int(self.config.video_duration_sec * self.config.video_fps))
        logging.info(
            "CIL video recording started: %s (fps=%.2f, duration=%ds, max_frames=%d)",
            output_path,
            self.config.video_fps,
            self.config.video_duration_sec,
            self._video_max_frames,
        )

    def _on_camera_frame(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = array.reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        with self._frame_lock:
            self._latest_rgb = rgb

    def _read_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_rgb
            self._latest_rgb = None
        return frame

    def _write_video_frame(self, rgb_frame) -> None:
        if self._video_writer is None:
            return
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self._video_writer.write(bgr)
        self._video_frames_written += 1
        if self._video_frames_written >= self._video_max_frames:
            self._stop_requested = True

    def _current_speed_kmh(self) -> float:
        velocity = self.session.ego_vehicle.get_velocity()
        speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_mps * 3.6

    def _longitudinal_control(
        self,
        speed_kmh: float,
        route_context: Dict[str, float],
        destination_distance_m: Optional[float],
        command: int = 0,
        command_phase: str = "cruise",
    ) -> tuple[float, float, float, Dict[str, float]]:
        """Compute throttle/brake from PID with adaptive target-speed planning."""
        adaptive_target_kmh, speed_plan = self._route_planner.compute_adaptive_target_speed_kmh(
            base_target_speed_kmh=self.config.target_speed_kmh,
            current_speed_kmh=speed_kmh,
            route_context=route_context,
            destination_distance_m=destination_distance_m,
            dt_s=(self.config.fixed_delta if self.config.sync else (1.0 / 20.0)),
        )

        route_turn_urgency = clamp(float(route_context.get("turn_urgency", 0.0)), 0.0, 1.0)
        route_junction_proximity = clamp(float(route_context.get("junction_proximity", 0.0)), 0.0, 1.0)

        phase = str(command_phase).lower()
        command_cap_kmh: Optional[float] = None
        if int(command) in (1, 2):
            if phase in {"armed", "in_junction"}:
                command_cap_kmh = clamp(27.0 - 15.0 * route_turn_urgency, 12.0, 25.0)
            elif phase == "clear":
                command_cap_kmh = clamp(30.0 - 10.0 * route_turn_urgency, self.CIL_CLEAR_PHASE_SPEED_CAP_KMH, 30.0)
        elif int(command) == 3:
            if phase in {"armed", "in_junction"}:
                command_cap_kmh = clamp(30.0 - 8.0 * route_junction_proximity, self.CIL_STRAIGHT_JUNCTION_SPEED_CAP_KMH, 30.0)
            elif phase == "clear":
                command_cap_kmh = clamp(31.0 - 8.0 * route_junction_proximity, self.CIL_CLEAR_PHASE_SPEED_CAP_KMH, 31.0)

        if command_cap_kmh is not None:
            adaptive_target_kmh = min(float(adaptive_target_kmh), float(command_cap_kmh))
            speed_plan["command_cap_kmh"] = float(command_cap_kmh)
        else:
            speed_plan["command_cap_kmh"] = float(adaptive_target_kmh)

        heading_abs = abs(float(route_context.get("heading_error_deg", 0.0)))
        lateral_abs = abs(float(route_context.get("target_y_m", 0.0)))
        turn_urgency = route_turn_urgency
        route_valid = clamp(float(route_context.get("route_valid", 0.0)), 0.0, 1.0)
        is_fallback = float(route_context.get("is_fallback", 0.0)) > 0.5
        recovery_cap_kmh = float(adaptive_target_kmh)
        has_reliable_route = route_valid >= 0.72 and not is_fallback
        if has_reliable_route:
            if int(command) == 0:
                should_limit_lane_follow = (
                    heading_abs >= 26.0 or lateral_abs >= 5.5 or turn_urgency >= 0.75
                )
                if should_limit_lane_follow:
                    recovery_severity = max(
                        clamp((heading_abs - 26.0) / 70.0, 0.0, 1.0),
                        clamp((lateral_abs - 5.5) / 16.0, 0.0, 1.0),
                        clamp((turn_urgency - 0.75) / 0.35, 0.0, 1.0),
                    )
                    recovery_cap_kmh = clamp(36.0 - 10.0 * recovery_severity, 24.0, 38.0)
                    adaptive_target_kmh = min(float(adaptive_target_kmh), float(recovery_cap_kmh))
            elif heading_abs >= 20.0 or lateral_abs >= 3.0 or turn_urgency >= 0.70:
                recovery_severity = max(
                    clamp((heading_abs - 20.0) / 80.0, 0.0, 1.0),
                    clamp((lateral_abs - 3.0) / 14.0, 0.0, 1.0),
                    clamp((turn_urgency - 0.55) / 0.45, 0.0, 1.0),
                )
                if route_valid < 0.85:
                    recovery_severity = clamp(
                        recovery_severity + clamp((0.85 - route_valid) / 0.35, 0.0, 1.0) * 0.35,
                        0.0,
                        1.0,
                    )
                recovery_cap_kmh = clamp(24.0 - 12.0 * recovery_severity, 10.0, 24.0)
                adaptive_target_kmh = min(float(adaptive_target_kmh), float(recovery_cap_kmh))
        # Emergency speed cap when route is completely lost
        elif route_valid < 0.30 and not is_fallback and (heading_abs > 30.0 or lateral_abs > 10.0):
            lost_severity = max(
                clamp((heading_abs - 30.0) / 50.0, 0.0, 1.0),
                clamp((lateral_abs - 10.0) / 20.0, 0.0, 1.0),
            )
            recovery_cap_kmh = clamp(15.0 - 10.0 * lost_severity, 5.0, 15.0)
            adaptive_target_kmh = min(float(adaptive_target_kmh), float(recovery_cap_kmh))
        speed_plan["recovery_cap_kmh"] = float(recovery_cap_kmh)

        self._speed_controller.set_target_speed(adaptive_target_kmh)
        throttle, brake = self._speed_controller.compute(speed_kmh)

        overspeed_kmh = max(0.0, float(speed_kmh) - float(adaptive_target_kmh))
        if overspeed_kmh > 2.5:
            feedforward_brake = clamp((overspeed_kmh - 2.5) / 12.0, 0.0, self.config.max_brake)
            if feedforward_brake > brake:
                brake = float(feedforward_brake)
                throttle = 0.0

        # Emergency full stop when car is completely lost (any command)
        completely_lost = (
            route_valid < 0.15
            and not is_fallback
            and (heading_abs > 45.0 or lateral_abs > 20.0)
            and speed_kmh < 8.0
        )
        if completely_lost:
            throttle = 0.0
            lost_brake = clamp(
                max((heading_abs - 40.0) / 40.0, (lateral_abs - 15.0) / 25.0),
                0.3,
                self.config.max_brake,
            )
            brake = max(float(brake), float(lost_brake))

        # In low-confidence route fallback, avoid pushing hard throttle into walls.
        # Applies regardless of active command when route is unreliable
        low_confidence_risk = (
            route_valid < 0.55
            and not is_fallback
            and (heading_abs > 25.0 or lateral_abs > 8.0)
        )
        if low_confidence_risk:
            risk_severity = max(
                clamp((heading_abs - 25.0) / 55.0, 0.0, 1.0),
                clamp((lateral_abs - 8.0) / 12.0, 0.0, 1.0),
            )
            throttle = min(float(throttle), 0.35 - 0.25 * risk_severity)
            if speed_kmh > 12.0:
                brake = max(float(brake), 0.05 + 0.20 * risk_severity)
        speed_plan["low_confidence_limit"] = 1.0 if (low_confidence_risk or completely_lost) else 0.0

        self._last_speed_plan = speed_plan
        return throttle, brake, float(adaptive_target_kmh), speed_plan

    def _predict_cil_steering(self, rgb_frame, speed_kmh: float, command: int) -> float:
        height = rgb_frame.shape[0]
        cropped = rgb_frame[int(height * 0.45) :, :, :]
        resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)

        yuv_image = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        image_tensor = torch.from_numpy(yuv_image).permute(2, 0, 1).float().div_(255.0)
        image_tensor.sub_(0.5).div_(0.5)
        image_tensor.unsqueeze_(0)

        speed_norm = clamp(speed_kmh / self.CIL_MAX_SPEED_KMH, 0.0, 1.0)
        command_idx = max(0, min(3, int(command)))
        speed_tensor = torch.tensor([speed_norm], dtype=torch.float32)
        command_tensor = torch.tensor([command_idx], dtype=torch.long)

        image_tensor = image_tensor.to(self._device, non_blocking=True)
        speed_tensor = speed_tensor.to(self._device, non_blocking=True)
        command_tensor = command_tensor.to(self._device, non_blocking=True)

        with torch.inference_mode():
            steering_value = float(self._model(image_tensor, speed_tensor, command_tensor).item())
        return clamp(steering_value, -1.0, 1.0)

    def run_step(self, step_idx: int) -> None:
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("CIL agent waiting for CARLA runtime.")
            return

        frame = self._read_latest_frame()
        if frame is None:
            if not self._waiting_frame_logged:
                logging.info("CIL waiting for first camera frame...")
                self._waiting_frame_logged = True
            return

        vehicle = self.session.ego_vehicle if self.session is not None else None
        if vehicle is None:
            return

        speed_kmh = self._current_speed_kmh()
        self._last_speed_kmh = speed_kmh

        self._write_video_frame(frame)
        if self._stop_requested:
            logging.info("CIL video duration target reached, stopping agent loop.")
            return

        destination_distance_m = self._distance_to_destination(vehicle.get_location())
        if destination_distance_m is not None and destination_distance_m <= self._arrival_distance_m:
            self._request_stop_at_destination("distance_threshold", destination_distance_m)
            return

        if self._nav_agent is not None:
            try:
                if self._nav_agent.done():
                    self._request_stop_at_destination("planner_done", destination_distance_m)
                    return
            except Exception:
                pass
            self._refresh_planner_state()
            self._maybe_replan_route(step_idx, vehicle)

        route_context = self._compute_route_context()
        self._maybe_replan_route_from_context(step_idx, vehicle, route_context)
        command, command_debug = self._update_distance_based_command(
            speed_kmh=speed_kmh,
            route_context=route_context,
            step_idx=step_idx,
        )
        lane_hint = self._build_lane_center_hint(
            vehicle_location=vehicle.get_location(),
            vehicle_transform=vehicle.get_transform(),
        )
        model_steer = self._predict_cil_steering(frame, speed_kmh, command)
        route_heading_abs = abs(float(route_context.get("heading_error_deg", 0.0)))
        route_lateral_abs = abs(float(route_context.get("target_y_m", 0.0)))
        route_turn_urgency = clamp(float(route_context.get("turn_urgency", 0.0)), 0.0, 1.0)
        route_valid = clamp(float(route_context.get("route_valid", 0.0)), 0.0, 1.0)
        is_fallback = float(route_context.get("is_fallback", 0.0)) > 0.5
        command_active_source = str(command_debug.get("active_source", "none"))
        command_distance_to_turn = float(command_debug.get("distance_to_turn_m", float("inf")))
        steer_gain = 1.0
        steer_gain += 0.45 * clamp((route_heading_abs - 14.0) / 70.0, 0.0, 1.0)
        steer_gain += 0.35 * clamp((route_lateral_abs - 2.0) / 14.0, 0.0, 1.0)
        if command in (1, 2, 3):
            steer_gain += 0.20 * route_turn_urgency
        if command in (1, 2) and command_active_source == "planner":
            if command_distance_to_turn <= 8.0:
                steer_gain += 0.45
            elif command_distance_to_turn <= 14.0:
                steer_gain += 0.25
        if route_valid >= 0.72 and route_valid < 0.90:
            steer_gain += 0.12 * clamp((0.90 - route_valid) / 0.18, 0.0, 1.0)
        if command_active_source == "planner":
            steer_gain_limit = 1.25
        elif route_valid >= 0.72:
            steer_gain_limit = 1.15
        elif route_valid >= 0.30 and not is_fallback:
            steer_gain_limit = 1.05
        else:
            # Don't amplify noisy CNN output when route is completely lost
            steer_gain_limit = 1.0
        steer_gain = clamp(steer_gain, 1.0, steer_gain_limit)
        steering_raw = clamp(model_steer * steer_gain, -1.0, 1.0)

        target_x_m = float(route_context.get("target_x_m", self._route_lookahead_m))
        target_y_m = float(route_context.get("target_y_m", 0.0))
        lateral_ratio = clamp(target_y_m / max(4.0, abs(target_x_m)), -1.0, 1.0)
        route_steer_correction = clamp(
            -(float(route_context.get("heading_error_deg", 0.0)) / 55.0) - 0.45 * lateral_ratio,
            -0.85,
            0.85,
        )
        route_assist_weight = 0.0
        # Allow emergency route assist even at low route_valid when car is way off-course
        if route_valid >= 0.72 or (route_heading_abs >= 30.0 and route_valid >= 0.25 and not is_fallback):
            valid_scale = 1.0 if route_valid >= 0.72 else clamp((route_valid - 0.2) / 0.52, 0.2, 0.7)
            route_assist_weight = max(
                clamp((route_heading_abs - 18.0) / 70.0, 0.0, 0.35),
                clamp((route_lateral_abs - 2.5) / 14.0, 0.0, 0.35),
            )
            route_assist_weight *= valid_scale
            if command in (1, 2):
                route_assist_weight += 0.12
                if command_active_source == "planner" and command_distance_to_turn <= 10.0:
                    route_assist_weight += 0.18
            elif command == 3:
                # STRAIGHT: chỉ dùng route assist rất nhẹ, để CNN tự lái thẳng
                route_assist_weight *= 0.3
        route_assist_weight = clamp(route_assist_weight, 0.0, 0.55)
        if route_assist_weight > 0.0:
            steering_raw = clamp(
                (1.0 - route_assist_weight) * steering_raw
                + route_assist_weight * route_steer_correction,
                -1.0,
                1.0,
            )

        lane_hint_valid = float(lane_hint.get("valid", 0.0))
        if int(command) == 0 and lane_hint_valid > 0.5 and route_turn_urgency < 0.60:
            lane_heading = float(lane_hint.get("heading_error_deg", 0.0))
            lane_target_y = float(lane_hint.get("target_y_m", 0.0))
            lane_half_width = max(1.2, float(lane_hint.get("lane_half_width_m", 1.75)))
            lane_ratio = clamp(lane_target_y / lane_half_width, -1.4, 1.4)
            lane_correction = clamp(
                -(lane_heading / 55.0) - 0.15 * lane_ratio,
                -0.25,
                0.25,
            )
            lane_weight = clamp(
                0.08
                + 0.12 * clamp(abs(lane_target_y) - 0.15, 0.0, 2.0)
                + 0.01 * clamp(abs(lane_heading) - 2.0, 0.0, 20.0),
                0.08,
                0.34,
            )
            steering_raw = clamp(
                (1.0 - lane_weight) * steering_raw + lane_weight * lane_correction,
                -1.0,
                1.0,
            )

        if int(command) == 0:
            if route_valid < 0.72:
                # Adaptive clamp: allow more steering when heading error is large
                # If fallback, allow CNN to steer more freely since route is unreliable
                if is_fallback:
                    low_valid_limit = 0.50
                else:
                    heading_factor = clamp(route_heading_abs / 45.0, 0.0, 1.0)
                    low_valid_limit = 0.12 + 0.38 * heading_factor
                steering_raw = clamp(steering_raw, -low_valid_limit, low_valid_limit)
            elif route_turn_urgency < 0.45 and command_distance_to_turn > 25.0:
                steering_raw = clamp(steering_raw, -0.34, 0.34)
            if route_valid < 0.60 and route_lateral_abs > 10.0 and route_heading_abs < 25.0 and not is_fallback:
                steering_raw = clamp(0.35 * steering_raw, -0.15, 0.15)

        steering_target = 0.0 if abs(steering_raw) < self.CIL_STEER_DEADBAND else steering_raw
        alpha_base = clamp(self.config.steer_smoothing, 0.0, 0.99)
        speed_ratio = clamp(speed_kmh / 50.0, 0.0, 1.0)
        alpha = clamp(alpha_base + 0.05 * speed_ratio, 0.0, 0.85)
        steering_smooth = alpha * self._last_steer + (1.0 - alpha) * steering_target
        dt_s = self.config.fixed_delta if self.config.sync else (1.0 / 20.0)
        max_steer_step = clamp(self.CIL_MAX_STEER_RATE_PER_S * dt_s, 0.01, 0.18)
        steering = clamp(
            steering_smooth,
            self._last_steer - max_steer_step,
            self._last_steer + max_steer_step,
        )
        self._last_steer = steering

        throttle, brake, adaptive_target_kmh, speed_plan = self._longitudinal_control(
            speed_kmh=speed_kmh,
            route_context=route_context,
            destination_distance_m=destination_distance_m,
            command=command,
            command_phase=str(command_debug.get("phase", "cruise")),
        )
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(clamp(steering, -1.0, 1.0)),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )
        vehicle.apply_control(control)
        vehicle_location = vehicle.get_location()
        rotation = vehicle.get_transform().rotation
        yaw_deg = float(rotation.yaw)
        self._update_route_history(vehicle_location)
        route_locations = self._collect_route_locations(max_points=80, step_idx=step_idx)
        # Disabled: CARLA debug line drawing is extremely expensive (~10 FPS penalty)
        # self._draw_route_debug_overlay(step_idx, route_locations, vehicle_location)

        destination_distance_m = self._distance_to_destination(vehicle_location)
        if destination_distance_m is not None and destination_distance_m <= self._arrival_distance_m:
            self._request_stop_at_destination("distance_threshold", destination_distance_m)

        if self._collector is not None:
            frame_id = step_idx
            if self.session.world is not None:
                frame_id = int(self.session.world.get_snapshot().frame)
            self._collector.add_vehicle_state(
                frame_id=frame_id,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                command=command,
                pitch=rotation.pitch,
                roll=rotation.roll,
                yaw=rotation.yaw,
            )

        if self._visualizer is not None:
            extra_lines = []
            if destination_distance_m is not None:
                extra_lines.append(f"Dist to D: {destination_distance_m:.1f} m")
            command_label = {0: "LANE_FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}.get(int(command), f"CMD_{command}")
            extra_lines.append(f"Command: {command_label} ({command})")
            self._visualizer.show_rgb(
                frame,
                {
                    "agent": "cil",
                    "tick": step_idx,
                    "speed_kmh": speed_kmh,
                    "target_speed_kmh": adaptive_target_kmh,
                    "steer": control.steer,
                    "throttle": control.throttle,
                    "brake": control.brake,
                    "command": command,
                },
                extra_lines=extra_lines if extra_lines else None,
            )

        if self._route_map is not None:
            self._route_map.show(
                route_points=route_locations,
                current_location=vehicle_location,
                start_location=self._route_start_location,
                destination_location=self._route_destination_location,
                heading_yaw_deg=yaw_deg,
                trajectory_points=self._route_history_xy,
                command=command,
            )

        if step_idx % 20 == 0:
            logging.info(
                "cil tick=%d speed=%.1f km/h target=%.1f cmd=%d phase=%s next=%d src=%s retarget=%s warmup=%s s_from_start=%.1f d_turn=%.1f d_route=%.1f d_junc=%.1f trigger=%.1f steer=%.3f model=%.3f gain=%.2f assist=%.2f throttle=%.2f brake=%.2f",
                step_idx,
                speed_kmh,
                adaptive_target_kmh,
                command,
                str(command_debug.get("phase", "cruise")),
                int(command_debug.get("upcoming_command", 0)),
                str(command_debug.get("upcoming_source", "none")),
                str(command_debug.get("retargeted", False)),
                str(command_debug.get("warmup_block_commands", False)),
                float(command_debug.get("distance_from_start_m", float("inf"))),
                float(command_debug.get("distance_to_turn_m", float("inf"))),
                float(command_debug.get("route_distance_to_turn_m", float("inf"))),
                float(command_debug.get("distance_to_junction_m", float("inf"))),
                float(command_debug.get("trigger_distance_m", 0.0)),
                control.steer,
                model_steer,
                steer_gain,
                route_assist_weight,
                control.throttle,
                control.brake,
            )

        if self._telemetry_writer is not None:
            self._telemetry_writer.writerow(
                [
                    int(step_idx),
                    f"{float(speed_kmh):.3f}",
                    f"{float(adaptive_target_kmh):.3f}",
                    f"{float(route_context.get('route_valid', 0.0)):.3f}",
                    f"{float(route_context.get('target_x_m', 0.0)):.3f}",
                    f"{float(route_context.get('target_y_m', 0.0)):.3f}",
                    f"{float(route_context.get('distance_to_turn_m', float('inf'))):.3f}",
                    f"{float(route_context.get('distance_to_junction_m', float('inf'))):.3f}",
                    f"{float(route_context.get('turn_urgency', 0.0)):.3f}",
                    int(command),
                    str(command_debug.get("phase", "cruise")),
                    str(command_debug.get("active_source", "none")),
                    f"{float(control.steer):.4f}",
                    f"{float(control.throttle):.4f}",
                    f"{float(control.brake):.4f}",
                    int(float(speed_plan.get("low_confidence_limit", 0.0)) > 0.5),
                ]
            )
            if step_idx % 20 == 0 and self._telemetry_fp is not None:
                self._telemetry_fp.flush()

    def teardown(self) -> None:
        if self._collector is not None:
            self._collector.close()
            self._collector = None
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logging.info(
                "Saved CIL video to %s (%d frames).",
                self._video_output,
                self._video_frames_written,
            )
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
            self._camera = None
            logging.info("Destroyed CIL RGB camera.")
        for sensor in self._data_cameras:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self._data_cameras = []
        self._nav_agent = None
        self._route_history_xy = []
        self._cached_route_locations = []
        self._cached_route_tick = -1
        self._route_planner.reset_runtime_state()
        self._last_route_context = {}
        self._last_speed_plan = {}
        self._active_navigation_command = 0
        self._active_command_source = "none"
        self._command_phase = "cruise"
        self._command_latch_frames = 0
        self._command_entered_junction = False
        self._command_clear_frames = 0
        self._last_command_debug = {}
        self._last_completed_turn_command = 0
        self._last_completed_turn_tick = -10000
        self._last_completed_turn_location = None
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None
        if self._route_map is not None:
            self._route_map.close()
            self._route_map = None
        if self._telemetry_fp is not None:
            self._telemetry_fp.close()
            self._telemetry_fp = None
            self._telemetry_writer = None

    def should_stop(self) -> bool:
        return self._stop_requested


class YoloDetectAgent(BaseAgent):
    name = "yolo_detect"

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._enabled = False
        self._camera = None
        self._depth_camera = None
        self._latest_rgb = None
        self._latest_depth_m = None
        self._frame_lock = threading.Lock()
        self._waiting_frame_logged = False
        self._detector = None
        self._window_name = "CARLA YOLO Detections"
        self._tm_autopilot_enabled = False
        self._nav_agent = None
        self._spawn_points = []
        self._tm_fallback_mode = False
        self._traffic_supervisor = None
        self._last_supervisor_debug_info: Dict[str, Any] = {}
        self._last_control_ts: Optional[float] = None

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        world = session.world
        if vehicle is None or world is None:
            logging.info("No CARLA vehicle/world available; yolo_detect runs as noop.")
            return

        if cv2 is None or np is None:
            raise RuntimeError("opencv-python and numpy are required for yolo_detect agent.")
        if YoloDetector is None:
            raise RuntimeError(
                "Cannot import YoloDetector. Install ultralytics and verify core_perception/yolo_detector.py."
            )

        model_path = resolve_yolo_model_path(self.config.yolo_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        self._detector = YoloDetector(
            str(model_path),
            camera_fov_deg=self.config.camera_fov,
            obstacle_base_distance_m=8.0,
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
        )
        if TrafficSupervisor is None:
            logging.warning("TrafficSupervisor unavailable. yolo_detect will run without supervisor brake fusion.")
            self._traffic_supervisor = None
        else:
            try:
                self._traffic_supervisor = TrafficSupervisor(self._build_supervisor_config())
                logging.info("TrafficSupervisor integrated into yolo_detect control loop.")
            except Exception as exc:
                self._traffic_supervisor = None
                logging.warning("Failed to initialize TrafficSupervisor: %s", exc)
        self._camera = self._spawn_camera(world, vehicle)
        self._camera.listen(self._on_camera_frame)
        try:
            self._depth_camera = self._spawn_depth_camera(world, vehicle)
            self._depth_camera.listen(self._on_depth_frame)
        except Exception as exc:
            self._depth_camera = None
            logging.warning(
                "Depth camera unavailable for yolo_detect, fallback to bbox distance. Reason: %s",
                exc,
            )
        self._init_navigation_agent(world, vehicle)
        self._enabled = True
        if self._nav_agent is not None:
            logging.info(
                "YOLO detection enabled with %s planner autopilot. Model: %s",
                self.config.nav_agent_type,
                model_path,
            )
        else:
            logging.info("YOLO detection enabled with TM autopilot fallback. Model: %s", model_path)

    def _init_navigation_agent(self, world, vehicle) -> None:
        self._spawn_points = world.get_map().get_spawn_points()
        if not self._spawn_points:
            logging.warning("No spawn points found for YOLO route planner, using TM autopilot fallback.")
            self._enable_tm_autopilot(vehicle)
            self._tm_fallback_mode = True
            return

        ensure_navigation_agent_imports()
        nav_type = self.config.nav_agent_type.lower()
        try:
            if nav_type == "behavior":
                if BehaviorAgent is None:
                    raise RuntimeError("BehaviorAgent missing")
                self._nav_agent = BehaviorAgent(vehicle, behavior="normal")
            else:
                if BasicAgent is None:
                    raise RuntimeError("BasicAgent missing")
                self._nav_agent = BasicAgent(vehicle, target_speed=max(10.0, self.config.target_speed_kmh))
            self._configure_nav_agent_traffic_lights()
            self._set_new_destination(vehicle)
            self._tm_fallback_mode = False
        except Exception as exc:
            self._nav_agent = None
            self._enable_tm_autopilot(vehicle)
            self._tm_fallback_mode = True
            logging.info(
                "YOLO planner unavailable, using TM autopilot fallback (set CARLA_PYTHONAPI to enable BasicAgent/BehaviorAgent). Reason: %s",
                exc,
            )

    def _enable_tm_autopilot(self, vehicle) -> None:
        try:
            vehicle.set_autopilot(True, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(True)
        self._configure_tm_traffic_lights(vehicle)
        self._tm_autopilot_enabled = True

    def _configure_nav_agent_traffic_lights(self) -> None:
        if not self.config.yolo_disable_autopilot_red_light:
            return
        if self._nav_agent is None:
            return

        configured = False
        ignore_fn = getattr(self._nav_agent, "ignore_traffic_lights", None)
        if callable(ignore_fn):
            try:
                ignore_fn(True)
                configured = True
            except TypeError:
                try:
                    ignore_fn()
                    configured = True
                except Exception:
                    configured = False
            except Exception:
                configured = False

        if not configured and hasattr(self._nav_agent, "_ignore_traffic_lights"):
            try:
                setattr(self._nav_agent, "_ignore_traffic_lights", True)
                configured = True
            except Exception:
                configured = False

        if configured:
            logging.info("YOLO planner autopilot configured to ignore traffic lights (supervisor decides stop/go).")
        else:
            logging.warning("Could not disable traffic-light handling on planner autopilot; supervisor may compete with planner stops.")

    def _configure_tm_traffic_lights(self, vehicle) -> None:
        if not self.config.yolo_disable_autopilot_red_light:
            return
        if self.session is None:
            return

        tm = getattr(self.session, "traffic_manager", None)
        if tm is None:
            logging.warning("TrafficManager unavailable; cannot apply ignore_lights_percentage for YOLO fallback autopilot.")
            return

        try:
            tm.ignore_lights_percentage(vehicle, 100.0)
            logging.info("YOLO TM fallback configured to ignore all traffic lights (100%%).")
        except Exception as exc:
            logging.warning("Failed to configure TM ignore_lights_percentage for YOLO fallback: %s", exc)

    def _set_new_destination(self, vehicle) -> None:
        if not self._spawn_points or self._nav_agent is None:
            return
        if self.config.destination_point >= 0:
            destination = self._spawn_points[self.config.destination_point % len(self._spawn_points)].location
        else:
            destination = random.choice(self._spawn_points).location
        current_loc = vehicle.get_location()
        set_navigation_destination(self._nav_agent, current_loc, destination)

    def _spawn_camera(self, world, vehicle):
        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick") and self.config.sync:
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logging.info("Attached RGB camera for YOLO detection.")
        return camera

    def _spawn_depth_camera(self, world, vehicle):
        bp_lib = world.get_blueprint_library()
        depth_bp = bp_lib.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(self.config.camera_width))
        depth_bp.set_attribute("image_size_y", str(self.config.camera_height))
        depth_bp.set_attribute("fov", str(self.config.camera_fov))
        if depth_bp.has_attribute("sensor_tick") and self.config.sync:
            depth_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0)
        )
        camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        logging.info("Attached depth camera for YOLO detection.")
        return camera

    def _on_camera_frame(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = array.reshape((image.height, image.width, 4))
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        with self._frame_lock:
            self._latest_rgb = rgb

    def _on_depth_frame(self, image) -> None:
        depth_m = decode_carla_depth_to_meters(image)
        with self._frame_lock:
            self._latest_depth_m = depth_m

    def _read_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_rgb
            if frame is None:
                return None, None
            depth_m = self._latest_depth_m
            self._latest_rgb = None
            self._latest_depth_m = None
        return frame, depth_m

    @staticmethod
    def _build_supervisor_config() -> Dict[str, Any]:
        return {
            "confidence_threshold": 0.5,
            "temporal_filter_frames": 3,
            "red_light_distance_threshold": 30.0,
            "obstacle_distance_threshold": 5.0,
            "max_stopped_time": 30.0,
        }

    @staticmethod
    def _to_supervisor_detections(detections: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        supervisor_inputs: list[Dict[str, Any]] = []
        for det in detections:
            box = det.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            distance_m = float(det.get("distance", float("inf")))
            if not math.isfinite(distance_m):
                distance_m = float("inf")

            supervisor_inputs.append(
                {
                    "class_name": str(det.get("class_name", "unknown")),
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox": (x1, y1, w, h),
                    "distance_m": distance_m,
                    "relative_velocity_kmh": float(det.get("relative_velocity_kmh", 0.0)),
                }
            )
        return supervisor_inputs

    def run_step(self, step_idx: int) -> None:
        """
        Main detection & control loop for YOLO agent.
        
        Workflow:
        1. Read frame + depth from camera
        2. Run YOLO detection
        3. Build danger_polygon from TrafficSupervisor
        4. Compute supervisor brake signal
        5. Apply control (nav_agent or TM autopilot)
        6. Draw annotations including YELLOW CORRIDOR
        7. Display & log
        """
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("YOLO agent waiting for CARLA runtime.")
            return

        # ─────────────────────────────────────────────────────────
        # Step 1: Read Camera Frames
        # ─────────────────────────────────────────────────────────
        frame, depth_map_m = self._read_latest_frame()
        if frame is None:
            if not self._waiting_frame_logged:
                logging.info("YOLO waiting for first camera frame...")
                self._waiting_frame_logged = True
            return

        # ─────────────────────────────────────────────────────────
        # Step 2: Get Vehicle State
        # ─────────────────────────────────────────────────────────
        vehicle = self.session.ego_vehicle if self.session is not None else None
        current_steer = None
        speed_kmh = None
        
        if vehicle is not None:
            try:
                current_steer = float(vehicle.get_control().steer)
            except Exception:
                current_steer = None
            try:
                velocity = vehicle.get_velocity()
                speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            except Exception:
                speed_kmh = None

        # ─────────────────────────────────────────────────────────
        # Step 3: Run YOLO Detection
        # ─────────────────────────────────────────────────────────
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detections, detector_emergency = self._detector.detect_and_evaluate(
            frame_bgr,
            distance_threshold=None,
            depth_map_m=depth_map_m,
            vehicle_steer=current_steer,
            speed_kmh=speed_kmh,
        )
        debug_info = {}
        if hasattr(self._detector, "get_last_debug_info"):
            debug_info = self._detector.get_last_debug_info() or {}

        # ─────────────────────────────────────────────────────────
        # Step 4: Build Danger Polygon & Compute Supervisor Brake
        # ─────────────────────────────────────────────────────────
        supervisor_brake = 0.0
        supervisor_state = "n/a"
        supervisor_reason = "n/a"
        hard_supervisor_emergency = False
        sup_debug = {}
        
        if self._traffic_supervisor is not None:
            now_ts = time.time()
            if self._last_control_ts is None:
                dt = self.config.fixed_delta if self.config.sync else (1.0 / 30.0)
            elif self.config.sync:
                dt = self.config.fixed_delta
            else:
                dt = max(1e-3, now_ts - float(self._last_control_ts))
            self._last_control_ts = now_ts

            try:
                # 🔧 BUILD DANGER POLYGON từ supervisor (CRITICAL!)
                if current_steer is None:
                    current_steer = 0.0
                if speed_kmh is None:
                    speed_kmh = 0.0
                    
                danger_polygon = self._traffic_supervisor._build_obstacle_danger_polygon(
                    image_shape=frame_bgr.shape,
                    vehicle_steer=float(current_steer),
                    vehicle_speed_kmh=float(speed_kmh)
                )
                
                # Store polygon để vẽ sau (cực kỳ quan trọng!)
                self._traffic_supervisor.last_danger_polygon = danger_polygon
                
                # Compute supervisor brake signal
                sup_dets = self._to_supervisor_detections(detections)
                supervisor_brake = float(
                    self._traffic_supervisor.compute(
                        detections=sup_dets,
                        current_speed=0.0 if speed_kmh is None else (speed_kmh / 3.6),
                        image_shape=frame_bgr.shape,
                        distance_threshold=None,
                        vehicle_steer=current_steer,
                        dt=dt,
                        danger_polygon=danger_polygon,
                    )
                )
                supervisor_brake = clamp(supervisor_brake, 0.0, 1.0)
                sup_debug = self._traffic_supervisor.get_debug_info()
                self._last_supervisor_debug_info = sup_debug
                supervisor_state = str(sup_debug.get("state", "n/a"))
                supervisor_reason = str(sup_debug.get("selected_target_type", "none"))
                hard_supervisor_emergency = supervisor_brake >= 0.95
                
            except Exception as exc:
                sup_debug = {}
                self._last_supervisor_debug_info = {}
                supervisor_brake = 0.0
                logging.warning("TrafficSupervisor compute failed: %s", exc)

        is_emergency = bool(detector_emergency or hard_supervisor_emergency)

        # ─────────────────────────────────────────────────────────
        # Step 5: Apply Control (Navigation Agent or TM Autopilot)
        # ─────────────────────────────────────────────────────────
        if vehicle is not None and self._nav_agent is not None:
            try:
                if self._nav_agent.done():
                    self._set_new_destination(vehicle)
            except Exception:
                pass

            nav_control = self._nav_agent.run_step()
            if is_emergency or supervisor_brake > 0.0:
                nav_control.throttle = 0.0
                emergency_floor = 1.0 if is_emergency else 0.0
                nav_control.brake = float(
                    clamp(
                        max(float(nav_control.brake), supervisor_brake, emergency_floor),
                        0.0,
                        1.0,
                    )
                )
            vehicle.apply_control(nav_control)

        # Emergency override on top of CARLA autopilot (TM fallback)
        elif is_emergency and vehicle is not None and self._nav_agent is None:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = float(0.0 if current_steer is None else current_steer)
            control.brake = float(clamp(max(1.0, supervisor_brake), 0.0, 1.0))
            control.hand_brake = False
            vehicle.apply_control(control)
            logging.warning(
                "[TICK %d] EMERGENCY BRAKE! Reason: detector=%s supervisor_state=%s target=%s",
                step_idx,
                debug_info.get("decision_reason", "dangerous object nearby"),
                supervisor_state,
                supervisor_reason,
            )
        elif supervisor_brake > 0.0 and vehicle is not None and self._nav_agent is None:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = float(0.0 if current_steer is None else current_steer)
            control.brake = float(supervisor_brake)
            control.hand_brake = False
            vehicle.apply_control(control)

        # ─────────────────────────────────────────────────────────
        # Step 6: Prepare Annotation Frame
        # ─────────────────────────────────────────────────────────
        annotated_frame = frame_bgr.copy()

        # ════════════════════════════════════════════════════════════════
        # 🎨 VẼ YELLOW DANGER CORRIDOR (LUÔN HIỂN THỊ) - PHẦN QUAN TRỌNG
        # ════════════════════════════════════════════════════════════════
        yellow_drew = _draw_yellow_danger_corridor(
            annotated_frame,
            debug_info,
            sup_debug,
        )

        if not yellow_drew:
            logging.debug("[TICK %d] Yellow corridor not drawn (no polygon available)", step_idx)
        else:
            logging.debug("[TICK %d] Yellow corridor drawn successfully", step_idx)

        # ─────────────────────────────────────────────────────────
        # Vẽ ROI regions (từ YOLO detector)
        # ─────────────────────────────────────────────────────────
        for roi_region in debug_info.get("roi_regions", []):
            x1, y1, x2, y2 = roi_region["box"]
            is_active = bool(roi_region.get("active", False))
            roi_color = (255, 180, 0) if is_active else (80, 80, 80)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), roi_color, 2)
            roi_label = f"ROI {roi_region['label']} < {roi_region['max_distance_m']:.0f}m"
            cv2.putText(
                annotated_frame,
                roi_label,
                (x1 + 4, min(y2 - 6, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                roi_color,
                1,
            )

        # ─────────────────────────────────────────────────────────
        # Vẽ Detection Bounding Boxes
        # ─────────────────────────────────────────────────────────
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            distance = det["distance"]
            distance_source = det.get("distance_source", "bbox")
            roi_zone = det.get("roi_zone")
            in_danger_roi = bool(det.get("in_danger_roi", False))
            danger_match = bool(det.get("danger_match", False))
            path_check_mode = det.get("path_check_mode")

            # Build label
            label = f"{class_name} {confidence:.2f} ({distance:.1f}m)"
            label = f"{label} [{distance_source}]"
            if roi_zone is not None:
                label = f"{label} [{roi_zone}]"
            if in_danger_roi:
                label = f"{label} [path]"
            if path_check_mode:
                label = f"{label} [{path_check_mode}]"
            if danger_match:
                label = f"{label} [BRAKE]"

            # Determine color
            if class_name == "traffic_light_red":
                color = (0, 0, 255)  # Red
            elif class_name == "traffic_light_green":
                color = (0, 255, 0)  # Green
            elif danger_match:
                color = (0, 0, 255)  # Red (brake)
            elif in_danger_roi and distance < 10.0:
                color = (0, 165, 255)  # Orange
            elif distance < 5.0:
                color = (255, 200, 0)  # Cyan
            else:
                color = (0, 255, 0)  # Green

            bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(
                annotated_frame,
                label,
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Draw ground touch point
            cv2.circle(annotated_frame, (int((bx1 + bx2) / 2), by2), 5, color, -1)

        # ─────────────────────────────────────────────────────────
        # Draw Status Text Overlays
        # ─────────────────────────────────────────────────────────
        if supervisor_brake > 0.0:
            status_text = f"SUPERVISOR BRAKE {supervisor_brake:.2f} ({supervisor_reason})"
            status_color = (0, 0, 255)
        elif is_emergency:
            status_text = "EMERGENCY BRAKE"
            status_color = (0, 0, 255)
        elif self._traffic_supervisor is not None:
            status_text = f"Supervisor {str(sup_debug.get('state', 'cruising')).upper()}"
            status_color = (0, 255, 0)
        else:
            status_text = "Normal"
            status_color = (0, 255, 0)

        cv2.putText(
            annotated_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2,
        )

        lock_zone = sup_debug.get("locked_zone", debug_info.get("locked_zone"))
        if lock_zone:
            lock_text = (
                f"LOCK={lock_zone} | immunity="
                f"{sup_debug.get('green_immunity_counter', debug_info.get('green_immunity_counter', 0))}"
            )
        else:
            lock_text = "LOCK=None"

        cv2.putText(
            annotated_frame,
            lock_text,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        turn_text = (
            f"TURN={bool(sup_debug.get('in_turn_phase', debug_info.get('turn_phase_active', False)))} | "
            f"grace={int(sup_debug.get('turn_grace_counter', debug_info.get('turn_green_grace_counter', 0)))} | "
            f"state={str(sup_debug.get('state', 'n/a')).upper()}"
        )
        cv2.putText(
            annotated_frame,
            turn_text,
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            2,
        )

        # ─────────────────────────────────────────────────────────
        # Step 7: Display Frame
        # ─────────────────────────────────────────────────────────
        cv2.imshow(self._window_name, annotated_frame)
        cv2.waitKey(1)

        # ─────────────────────────────────────────────────────────
        # Step 8: Log Information
        # ─────────────────────────────────────────────────────────
        if step_idx % 20 == 0:
            logging.info(
                "yolo_detect tick=%d detections=%d | emergency=%s | supervisor_brake=%.2f | "
                "state=%s | reason=%s | yellow_polygon=%s",
                step_idx,
                len(detections),
                is_emergency,
                supervisor_brake,
                supervisor_state,
                supervisor_reason,
                "drawn" if yellow_drew else "missing",
            )

    def teardown(self) -> None:
        if self._tm_autopilot_enabled and self.session is not None and self.session.ego_vehicle is not None:
            try:
                self.session.ego_vehicle.set_autopilot(False, self.config.tm_port)
            except TypeError:
                self.session.ego_vehicle.set_autopilot(False)
            self._tm_autopilot_enabled = False
        self._nav_agent = None
        self._traffic_supervisor = None
        self._last_supervisor_debug_info = {}
        self._last_control_ts = None

        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
            self._camera = None
            logging.info("Destroyed YOLO camera.")
        if self._depth_camera is not None:
            self._depth_camera.stop()
            self._depth_camera.destroy()
            self._depth_camera = None
            logging.info("Destroyed YOLO depth camera.")

        if cv2 is not None:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass


class NoopAgent(BaseAgent):
    name = "noop"

    def run_step(self, step_idx: int) -> None:
        if step_idx % 50 == 0:
            logging.info("Noop agent alive at tick %d", step_idx)


AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    AutopilotAgent.name: AutopilotAgent,
    LaneFollowAgent.name: LaneFollowAgent,
    CILAgent.name: CILAgent,
    YoloDetectAgent.name: YoloDetectAgent,
    NoopAgent.name: NoopAgent,
}


def load_env_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        logging.warning("Config file not found: %s. Using CLI/default values.", path)
        return {}
    if yaml is None:
        logging.warning("PyYAML is not installed; skipping config file %s", path)
        return {}

    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp) or {}
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid yaml root in {path}. Expected mapping/object.")
    logging.info("Loaded environment config from %s", path)
    return loaded


def _cfg_get(env_cfg: Dict[str, Any], section: str, key: str, fallback: Any) -> Any:
    data = env_cfg.get(section, {})
    if isinstance(data, dict):
        return data.get(key, fallback)
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARLA agent loop.")
    parser.add_argument(
        "--config",
        default="configs/carla_env.yaml",
        help="Path to environment YAML config file.",
    )
    parser.add_argument(
        "--agent",
        choices=sorted(AGENT_REGISTRY),
        default="lane_follow",
        help="Choose lane_follow/cil/autopilot/yolo_detect agent mode.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=1000.0)
    parser.add_argument("--sync", action="store_true", help="Enable synchronous mode.")
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
        help="Fixed delta seconds (used when --sync is enabled).",
    )
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument(
        "--map",
        default="Town03",
        help="CARLA map to load before spawning vehicle (default: Town03).",
    )
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=-1,
        help="Spawn-point index. Negative means random.",
    )
    parser.add_argument(
        "--destination-point",
        type=int,
        default=-1,
        help="Destination point index B. Negative means random destination.",
    )
    parser.add_argument("--npc-vehicle-count", type=int, default=30)
    parser.add_argument("--npc-bike-count", type=int, default=10)
    parser.add_argument("--npc-motorbike-count", type=int, default=10)
    parser.add_argument("--npc-pedestrian-count", type=int, default=50)
    parser.add_argument(
        "--disable-npc-autopilot",
        action="store_true",
        help="Spawn NPCs but do not enable autopilot for them.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=1000,
        help="Number of ticks to run. Use 0 or negative for infinite loop.",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=0.05,
        help="Sleep interval per tick in dry-run mode.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--model-path",
        default="auto",
        help="Path to .pth model for lane_follow agent (or 'auto').",
    )
    parser.add_argument(
        "--cil-model-path",
        default="auto",
        help="Path to .pth model for CIL agent (or 'auto').",
    )
    parser.add_argument(
        "--yolo-model-path",
        default="best.pt",
        help="Path to YOLO .pt model used by yolo_detect agent.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for lane_follow/cil.",
    )
    parser.add_argument("--target-speed-kmh", type=float, default=30.0)
    parser.add_argument("--max-throttle", type=float, default=0.2)
    parser.add_argument("--max-brake", type=float, default=0.60)
    parser.add_argument(
        "--steer-smoothing",
        type=float,
        default=0.35,
        help="0 means no smoothing, closer to 1 means smoother steering.",
    )
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=600)
    parser.add_argument("--camera-fov", type=float, default=90.0)

    parser.add_argument(
        "--lock-spectator-on-spawn",
        dest="lock_spectator_on_spawn",
        action="store_true",
        default=None,
        help="Lock spectator camera to spawn viewpoint.",
    )
    parser.add_argument(
        "--no-lock-spectator-on-spawn",
        dest="lock_spectator_on_spawn",
        action="store_false",
        help="Disable spectator spawn lock.",
    )
    parser.add_argument(
        "--spectator-reapply-each-tick",
        action="store_true",
        help="Re-apply spectator transform every tick.",
    )
    parser.add_argument("--spectator-follow-distance", type=float, default=9.0)
    parser.add_argument("--spectator-height", type=float, default=4.5)
    parser.add_argument("--spectator-pitch", type=float, default=-18.0)

    parser.add_argument("--collect-data", action="store_true")
    parser.add_argument("--collect-data-dir", default="data/collected")
    parser.add_argument("--image-prefix", default="", help="Image filename prefix (e.g. town01_beochan)")
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=50,
        help="Only save a frame every N ticks (default: 50 for diverse scenes).",
    )
    parser.add_argument(
        "--nav-agent-type",
        choices=["basic", "behavior"],
        default="basic",
        help="Navigation agent used by autopilot mode for route command extraction.",
    )
    parser.add_argument(
        "--yolo-disable-autopilot-red-light",
        dest="yolo_disable_autopilot_red_light",
        action="store_true",
        default=None,
        help="In yolo_detect mode, force planner/TM autopilot to ignore traffic lights so supervisor decides braking.",
    )
    parser.add_argument(
        "--no-yolo-disable-autopilot-red-light",
        dest="yolo_disable_autopilot_red_light",
        action="store_false",
        help="In yolo_detect mode, keep autopilot traffic-light handling enabled.",
    )
    parser.add_argument(
        "--no-random-weather",
        action="store_true",
        help="Disable random weather preset selection at start.",
    )
    parser.add_argument(
        "--recovery-interval-frames",
        type=int,
        default=100,
        help="Apply recovery steering disturbance every N frames.",
    )
    parser.add_argument(
        "--recovery-duration-frames",
        type=int,
        default=10,
        help="Number of frames to keep disturbance active.",
    )
    parser.add_argument(
        "--recovery-steer-offset",
        type=float,
        default=0.3,
        help="Steering offset added during disturbance window.",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-output-path", default="outputs/town_drive_10m.mp4")
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-duration-sec", type=int, default=600)
    parser.add_argument("--video-codec", default="mp4v")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--cil-route-lookahead-m",
        type=float,
        default=9.0,
        help="Nominal route lookahead used for route context and speed planning in CIL mode.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    env_cfg = load_env_config(args.config)

    lock_from_yaml = bool(_cfg_get(env_cfg, "spectator", "lock_on_spawn", True))
    lock_spectator = (
        args.lock_spectator_on_spawn
        if args.lock_spectator_on_spawn is not None
        else lock_from_yaml
    )

    sync = args.sync or bool(_cfg_get(env_cfg, "carla", "sync", False))
    fixed_delta = (
        args.fixed_delta
        if args.fixed_delta != 0.05
        else float(_cfg_get(env_cfg, "carla", "fixed_delta", args.fixed_delta))
    )
    if args.collect_data and not sync:
        logging.warning("Data collection requires synchronous mode. Forcing sync=True.")
        sync = True
    if args.collect_data and fixed_delta <= 0.0:
        fixed_delta = 0.05
        logging.warning("Invalid fixed_delta for data collection. Using fixed_delta=0.05.")
    video_enabled = args.record_video or bool(
        _cfg_get(env_cfg, "recording", "enabled", False)
    )
    video_output_path = (
        args.video_output_path
        if args.video_output_path != "outputs/town_drive_10m.mp4"
        else str(_cfg_get(env_cfg, "recording", "output_path", args.video_output_path))
    )
    video_fps = (
        args.video_fps
        if args.video_fps != 20.0
        else float(_cfg_get(env_cfg, "recording", "fps", args.video_fps))
    )
    video_duration_sec = (
        args.video_duration_sec
        if args.video_duration_sec != 600
        else int(_cfg_get(env_cfg, "recording", "duration_sec", args.video_duration_sec))
    )
    video_codec = (
        args.video_codec
        if args.video_codec != "mp4v"
        else str(_cfg_get(env_cfg, "recording", "codec", args.video_codec))
    )

    npc_vehicle_count = (
        args.npc_vehicle_count
        if args.npc_vehicle_count != 30
        else int(_cfg_get(env_cfg, "traffic_spawn", "vehicle_count", args.npc_vehicle_count))
    )
    npc_bike_count = (
        args.npc_bike_count
        if args.npc_bike_count != 10
        else int(_cfg_get(env_cfg, "traffic_spawn", "bike_count", args.npc_bike_count))
    )
    npc_motorbike_count = (
        args.npc_motorbike_count
        if args.npc_motorbike_count != 10
        else int(_cfg_get(env_cfg, "traffic_spawn", "motorbike_count", args.npc_motorbike_count))
    )
    npc_pedestrian_count = (
        args.npc_pedestrian_count
        if args.npc_pedestrian_count != 50
        else int(_cfg_get(env_cfg, "traffic_spawn", "pedestrian_count", args.npc_pedestrian_count))
    )
    npc_enable_autopilot = bool(
        _cfg_get(env_cfg, "traffic_spawn", "npc_enable_autopilot", not args.disable_npc_autopilot)
    ) and not args.disable_npc_autopilot

    ticks = args.ticks
    if video_enabled and args.ticks == 1000:
        if sync:
            required_ticks = max(1, int(video_duration_sec / max(fixed_delta, 1e-3)))
        else:
            required_ticks = max(1, int(video_duration_sec * video_fps))
        ticks = required_ticks
        logging.info("Auto-set ticks=%d from video duration=%ds.", ticks, video_duration_sec)

    spawn_point_cfg = (
        args.spawn_point
        if args.spawn_point != -1
        else int(_cfg_get(env_cfg, "vehicle", "spawn_point", args.spawn_point))
    )
    destination_point_cfg = (
        args.destination_point
        if args.destination_point != -1
        else int(_cfg_get(env_cfg, "vehicle", "destination_point", args.destination_point))
    )

    if args.agent == "cil" and spawn_point_cfg < 0:
        logging.warning(
            "CIL route map requires deterministic S point. vehicle.spawn_point < 0, forcing spawn_point=0."
        )
        spawn_point_cfg = 0

    if args.agent == "cil" and destination_point_cfg < 0:
        destination_point_cfg = spawn_point_cfg + 1
        logging.warning(
            "CIL route map requires deterministic D point. vehicle.destination_point < 0, "
            "forcing destination_point=%d.",
            destination_point_cfg,
        )

    cil_route_lookahead_m = (
        args.cil_route_lookahead_m
        if args.cil_route_lookahead_m != 9.0
        else float(_cfg_get(env_cfg, "cil", "route_lookahead_m", args.cil_route_lookahead_m))
    )
    cil_command_prep_time_s = float(
        _cfg_get(env_cfg, "cil", "command_prep_time_s", CILAgent.COMMAND_PREP_TIME_S)
    )
    cil_command_trigger_min_m = float(
        _cfg_get(env_cfg, "cil", "command_trigger_min_m", CILAgent.COMMAND_TRIGGER_MIN_M)
    )
    cil_command_trigger_max_m = float(
        _cfg_get(env_cfg, "cil", "command_trigger_max_m", CILAgent.COMMAND_TRIGGER_MAX_M)
    )
    cil_command_retarget_window_s = float(
        _cfg_get(env_cfg, "cil", "command_retarget_window_s", 3.0)
    )
    cil_command_trigger_min_m = max(3.0, cil_command_trigger_min_m)
    cil_command_trigger_max_m = max(cil_command_trigger_min_m + 1.0, cil_command_trigger_max_m)
    cil_command_prep_time_s = max(0.8, cil_command_prep_time_s)
    cil_command_retarget_window_s = max(0.8, cil_command_retarget_window_s)
    yolo_disable_autopilot_red_light = (
        args.yolo_disable_autopilot_red_light
        if args.yolo_disable_autopilot_red_light is not None
        else bool(_cfg_get(env_cfg, "yolo", "disable_autopilot_red_light", False))
    )

    return RunConfig(
        env_config_path=args.config,
        host=args.host if args.host != "127.0.0.1" else _cfg_get(env_cfg, "carla", "host", args.host),
        port=args.port if args.port != 2000 else int(_cfg_get(env_cfg, "carla", "port", args.port)),
        tm_port=args.tm_port if args.tm_port != 8000 else int(_cfg_get(env_cfg, "carla", "tm_port", args.tm_port)),
        timeout=(
            args.timeout
            if args.timeout != 1000.0
            else float(_cfg_get(env_cfg, "carla", "timeout", args.timeout))
        ),
        sync=sync,
        fixed_delta=fixed_delta,
        no_rendering=args.no_rendering or bool(_cfg_get(env_cfg, "carla", "no_rendering", False)),
        map_name=args.map if args.map != "Town03" else _cfg_get(env_cfg, "carla", "map", args.map),
        vehicle_filter=_cfg_get(env_cfg, "vehicle", "filter", args.vehicle_filter),
        spawn_point=spawn_point_cfg,
        destination_point=destination_point_cfg,
        ticks=ticks,
        tick_interval=args.tick_interval,
        dry_run=args.dry_run,
        seed=args.seed,
        model_path=args.model_path,
        cil_model_path=args.cil_model_path,
        yolo_model_path=args.yolo_model_path,
        model_device=args.device,
        target_speed_kmh=args.target_speed_kmh,
        max_throttle=args.max_throttle,
        max_brake=args.max_brake,
        steer_smoothing=args.steer_smoothing,
        camera_width=(
            args.camera_width
            if args.camera_width != 800
            else int(_cfg_get(env_cfg, "camera", "width", args.camera_width))
        ),
        camera_height=(
            args.camera_height
            if args.camera_height != 600
            else int(_cfg_get(env_cfg, "camera", "height", args.camera_height))
        ),
        camera_fov=(
            args.camera_fov
            if args.camera_fov != 90.0
            else float(_cfg_get(env_cfg, "camera", "fov", args.camera_fov))
        ),
        lock_spectator_on_spawn=lock_spectator,
        spectator_reapply_each_tick=(
            args.spectator_reapply_each_tick
            or bool(_cfg_get(env_cfg, "spectator", "keep_reapply_each_tick", False))
        ),
        spectator_follow_distance=float(
            _cfg_get(env_cfg, "spectator", "follow_distance", args.spectator_follow_distance)
        ),
        spectator_height=float(_cfg_get(env_cfg, "spectator", "height", args.spectator_height)),
        spectator_pitch=float(_cfg_get(env_cfg, "spectator", "pitch", args.spectator_pitch)),
        collect_data=args.collect_data,
        collect_data_dir=args.collect_data_dir,
        save_every_n=args.save_every_n,
        image_prefix=args.image_prefix,
        npc_vehicle_count=npc_vehicle_count,
        npc_bike_count=npc_bike_count,
        npc_motorbike_count=npc_motorbike_count,
        npc_pedestrian_count=npc_pedestrian_count,
        npc_enable_autopilot=npc_enable_autopilot,
        record_video=video_enabled,
        video_output_path=video_output_path,
        video_fps=video_fps,
        video_duration_sec=video_duration_sec,
        video_codec=video_codec,
        random_weather=not args.no_random_weather,
        weather_preset=str(_cfg_get(env_cfg, "weather", "preset", "ClearNoon")),
        recovery_interval_frames=max(1, args.recovery_interval_frames),
        recovery_duration_frames=max(1, args.recovery_duration_frames),
        recovery_steer_offset=abs(args.recovery_steer_offset),
        nav_agent_type=args.nav_agent_type,
        yolo_disable_autopilot_red_light=bool(yolo_disable_autopilot_red_light),
        cil_route_lookahead_m=max(4.0, cil_route_lookahead_m),
        cil_command_prep_time_s=cil_command_prep_time_s,
        cil_command_trigger_min_m=cil_command_trigger_min_m,
        cil_command_trigger_max_m=cil_command_trigger_max_m,
        cil_command_retarget_window_s=cil_command_retarget_window_s,
    )


def build_session(config: RunConfig) -> BaseSession:
    if config.dry_run:
        return DryRunSession(config)
    if carla is None:
        raise RuntimeError(
            "Python package 'carla' is not installed. Use --dry-run or install CARLA API."
        )
    return CarlaSession(config)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(args)
    if config.seed is not None:
        random.seed(config.seed)

    session = build_session(config)
    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls(config)
    tick_limit = config.ticks

    try:
        session.start()
        if not config.dry_run and session.world is not None:
            if config.random_weather:
                preset = apply_random_weather(session.world)
                logging.info("Applied random weather preset: %s", preset)
            else:
                apply_weather_preset(session.world, config.weather_preset)
        agent.setup(session)
        step = 0
        while tick_limit <= 0 or step < tick_limit:
            step += 1
            session.tick()
            agent.run_step(step)
            if agent.should_stop():
                logging.info("Agent requested stop at tick %d.", step)
                break
        logging.info("Finished %d ticks.", step)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logging.error("Agent run failed: %s", exc, exc_info=True)
        return 1
    finally:
        try:
            agent.teardown()
        finally:
            session.cleanup()


if __name__ == "__main__":
    sys.exit(main())
