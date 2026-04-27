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
from core_control.navigation_command import (
    build_global_reference_route,
    DEFAULT_COMMAND_MAX_ARMED_FRAMES,
    DEFAULT_COMMAND_PREP_TIME_S,
    DEFAULT_COMMAND_TRIGGER_MAX_M,
    DEFAULT_COMMAND_TRIGGER_MIN_M,
    NavigationCommandOracle,
    map_road_option_to_command as shared_map_road_option_to_command,
    snapshot_planner_route,
)
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
    return shared_map_road_option_to_command(road_option)


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
    fps_log_interval_ticks: int
    cil_enable_hud: bool
    cil_enable_route_map: bool
    cil_enable_telemetry_csv: bool
    cil_profile_tick_timing: bool
    cil_profile_log_interval_ticks: int
    recovery_interval_frames: int
    recovery_duration_frames: int
    recovery_steer_offset: float
    nav_agent_type: str
    yolo_disable_autopilot_red_light: bool
    cil_command_prep_time_s: float
    cil_command_trigger_min_m: float
    cil_command_trigger_max_m: float

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


class TickFpsProfiler:
    """Aggregate runtime FPS and per-stage timing to locate bottlenecks."""

    def __init__(self, sync: bool, fixed_delta: float, log_interval_ticks: int) -> None:
        self.sync = bool(sync)
        self.fixed_delta = float(fixed_delta)
        self.log_interval_ticks = max(1, int(log_interval_ticks))
        self._target_fps = (1.0 / self.fixed_delta) if (self.sync and self.fixed_delta > 0.0) else 0.0
        self._reset_window()

    def _reset_window(self) -> None:
        self._count = 0
        self._sum_total_s = 0.0
        self._sum_session_s = 0.0
        self._sum_agent_s = 0.0
        self._max_tick_s = 0.0

    def record(self, step_idx: int, session_s: float, agent_s: float, total_s: float) -> None:
        session_s = max(0.0, float(session_s))
        agent_s = max(0.0, float(agent_s))
        total_s = max(1e-9, float(total_s))

        self._count += 1
        self._sum_session_s += session_s
        self._sum_agent_s += agent_s
        self._sum_total_s += total_s
        self._max_tick_s = max(self._max_tick_s, total_s)

        if self._count >= self.log_interval_ticks:
            self._log_window(step_idx)

    def flush(self, step_idx: int) -> None:
        if self._count > 0:
            self._log_window(step_idx)

    def _log_window(self, step_idx: int) -> None:
        avg_tick_s = self._sum_total_s / max(1, self._count)
        avg_session_s = self._sum_session_s / max(1, self._count)
        avg_agent_s = self._sum_agent_s / max(1, self._count)
        avg_fps = self._count / max(self._sum_total_s, 1e-9)
        session_share = 100.0 * self._sum_session_s / max(self._sum_total_s, 1e-9)
        agent_share = 100.0 * self._sum_agent_s / max(self._sum_total_s, 1e-9)

        target_text = f"{self._target_fps:.2f}" if self._target_fps > 0.0 else "unbounded"
        logging.info(
            "[FPS] tick=%d avg_fps=%.2f target_fps=%s avg_tick=%.2fms avg_session=%.2fms(%.1f%%) avg_agent=%.2fms(%.1f%%) worst_tick=%.2fms",
            step_idx,
            avg_fps,
            target_text,
            avg_tick_s * 1000.0,
            avg_session_s * 1000.0,
            session_share,
            avg_agent_s * 1000.0,
            agent_share,
            self._max_tick_s * 1000.0,
        )

        self._reset_window()


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
        self._reference_route_plan: list[Dict[str, Any]] = []
        self._route_destination_location = None
        self._recovery_start_frame = -1
        self._recovery_direction = 1.0
        self._tm_fallback_mode = False
        self._command_oracle = NavigationCommandOracle(
            get_planner=self._get_local_planner,
            get_current_waypoint=self._current_waypoint,
            get_vehicle_location=self._vehicle_location,
            get_reference_route=self._get_reference_route_plan,
            prep_time_s=self.config.cil_command_prep_time_s,
            trigger_min_m=self.config.cil_command_trigger_min_m,
            trigger_max_m=self.config.cil_command_trigger_max_m,
            max_armed_frames=DEFAULT_COMMAND_MAX_ARMED_FRAMES,
        )

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        if vehicle is None or session.world is None:
            logging.info("No ego vehicle in this session; autopilot setup skipped.")
            return

        self._init_navigation_agent(session.world, vehicle)
        self._command_oracle.reset()

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
                "Using TM autopilot fallback. Shared intersection-only CIL commands require planner access, so collector commands stay at follow-lane (0)."
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
        self._route_destination_location = destination
        set_navigation_destination(self._nav_agent, current_loc, destination)
        self._cache_reference_route_plan(force=True)
        self._command_oracle.reset()

    def _vehicle_location(self):
        if self.session is None or self.session.ego_vehicle is None:
            return None
        return self.session.ego_vehicle.get_location()

    def _current_waypoint(self):
        if self.session is None or self.session.world is None or self.session.ego_vehicle is None:
            return None
        try:
            return self.session.world.get_map().get_waypoint(
                self.session.ego_vehicle.get_location(),
                project_to_road=True,
            )
        except Exception:
            return None

    def _get_local_planner(self):
        if self._nav_agent is None or not hasattr(self._nav_agent, "get_local_planner"):
            return None
        try:
            return self._nav_agent.get_local_planner()
        except Exception:
            return None

    def _get_reference_route_plan(self):
        return list(self._reference_route_plan)

    def _cache_reference_route_plan(self, force: bool = False) -> int:
        if self._tm_fallback_mode:
            self._reference_route_plan = []
            return 0
        if self._reference_route_plan and not force:
            return int(len(self._reference_route_plan))
        world = self.session.world if self.session is not None else None
        world_map = world.get_map() if world is not None else None
        current_loc = self._vehicle_location()
        destination_loc = self._route_destination_location

        reference_route = build_global_reference_route(
            world_map=world_map,
            start_location=current_loc,
            destination_location=destination_loc,
        )
        route_source = "global"
        if not reference_route:
            planner = self._get_local_planner()
            reference_route = snapshot_planner_route(planner)
            route_source = "planner_snapshot"
        if reference_route or force:
            self._reference_route_plan = list(reference_route)
        if reference_route:
            logging.info(
                "Autopilot cached fixed route plan with %d points from %s.",
                len(reference_route),
                route_source,
            )
        return int(len(self._reference_route_plan))

    def _extract_current_command(self, speed_kmh: float = 0.0) -> int:
        command, _ = self._command_oracle.update(speed_kmh=speed_kmh)
        return int(command)

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
            if not self._reference_route_plan:
                self._cache_reference_route_plan()
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
            command = self._extract_current_command(speed_kmh=speed_kmh)
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
        self._command_oracle.reset()
        self._nav_agent = None
        self._reference_route_plan = []
        self._route_destination_location = None
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
    COMMAND_PREP_TIME_S = DEFAULT_COMMAND_PREP_TIME_S
    COMMAND_TRIGGER_MIN_M = DEFAULT_COMMAND_TRIGGER_MIN_M
    COMMAND_TRIGGER_MAX_M = DEFAULT_COMMAND_TRIGGER_MAX_M
    COMMAND_MAX_ARMED_FRAMES = DEFAULT_COMMAND_MAX_ARMED_FRAMES
    REPLAN_MIN_QUEUE_SIZE = 1
    REPLAN_GUARD_JUNCTION_M = 30.0
    REPLAN_GUARD_DESTINATION_M = 35.0

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._enabled = False
        self._model = None
        self._device = None
        self._camera = None
        self._latest_rgb = None
        self._frame_lock = threading.Lock()
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
        self._reference_route_plan: list[Dict[str, Any]] = []
        self._route_start_location = None
        self._route_destination_location = None
        self._route_history_xy: list[tuple[float, float]] = []
        self._arrival_distance_m = 3.0
        self._destination_reached_logged = False
        self._last_speed_kmh = 0.0
        self._last_steer = 0.0
        self._max_observed_speed_kmh = 0.0
        self._blocked_frames = 0
        self._command_prep_time_s = float(config.cil_command_prep_time_s)
        self._command_trigger_min_m = float(config.cil_command_trigger_min_m)
        self._command_trigger_max_m = float(config.cil_command_trigger_max_m)
        self._active_navigation_command = 0
        self._active_command_source = "none"
        self._command_phase = "cruise"
        self._command_latch_frames = 0
        self._command_entered_junction = False
        self._last_command_debug: Dict[str, Any] = {}
        self._last_route_assist_debug: Dict[str, Any] = {}
        self._last_route_curve_debug: Dict[str, Any] = {}
        self._command_oracle = NavigationCommandOracle(
            get_planner=self._get_local_planner,
            get_current_waypoint=self._current_waypoint,
            get_vehicle_location=self._vehicle_location,
            get_reference_route=self._get_reference_route_plan,
            prep_time_s=self._command_prep_time_s,
            trigger_min_m=self._command_trigger_min_m,
            trigger_max_m=self._command_trigger_max_m,
            max_armed_frames=self.COMMAND_MAX_ARMED_FRAMES,
        )
        self._last_replan_tick = -10000
        self._route_planner = CILRoutePlanner(arrival_distance_m=self._arrival_distance_m)
        self._timing_window_ticks = 0
        self._timing_sums: Dict[str, float] = {
            "read": 0.0,
            "nav": 0.0,
            "model": 0.0,
            "control": 0.0,
            "viz": 0.0,
            "telemetry": 0.0,
            "total": 0.0,
        }
        # HUD debug drawing
        self._hud_ema_fps: Optional[float] = None
        self._hud_last_tick_time: Optional[float] = None
        self._route_overlay_bounds: Optional[tuple[float, float, float, float]] = None
        # Speed control (PID)
        self._speed_controller = SpeedPIDController(
            target_speed_kmh=config.target_speed_kmh,
            max_throttle=config.max_throttle,
            max_brake=config.max_brake,
        )

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        self._route_overlay_bounds = None
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
        self._command_oracle.reset()

        self._collector = DataCollector(
            output_dir=self.config.collect_data_dir,
            enabled=self.config.collect_data,
            save_every_n=self.config.save_every_n,
        )
        self._collector.start()
        self._start_data_collection_cameras(world, vehicle)
        self._init_video_writer()
        if self.config.cil_enable_telemetry_csv:
            self._init_telemetry_logger()

        # Initialize the separate route-map OpenCV window.
        if RouteMapVisualizer is not None:
            self._route_map = RouteMapVisualizer(
                window_name="CIL Route Map",
                canvas_size=620,
            )
            logging.info("RouteMapVisualizer window enabled.")

        self._enabled = True
        logging.info(
            "CIL agent is ready (hud=%s, route_map=%s, telemetry_csv=%s).",
            self.config.cil_enable_hud,
            self.config.cil_enable_route_map,
            self.config.cil_enable_telemetry_csv,
        )

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
                "distance_to_turn_m",
                "distance_to_junction_m",
                "command",
                "command_phase",
                "command_source",
                "upcoming_command",
                "reset_reason",
                "steer_model",
                "steer_assist",
                "route_lateral_error_m",
                "route_heading_error_deg",
                "route_curve_strength",
                "steer",
                "throttle",
                "brake",
            ]
        )
        self._telemetry_fp.flush()

    def _accumulate_tick_timing(self, stage_s: Dict[str, float], step_idx: int) -> None:
        if not self.config.cil_profile_tick_timing:
            return
        for key in self._timing_sums.keys():
            self._timing_sums[key] += float(stage_s.get(key, 0.0))
        self._timing_window_ticks += 1

        report_every = max(1, int(self.config.cil_profile_log_interval_ticks))
        if self._timing_window_ticks < report_every:
            return

        n = float(max(1, self._timing_window_ticks))
        avg_total_s = self._timing_sums["total"] / n
        avg_fps = 1.0 / max(1e-9, avg_total_s)
        nav_share = 100.0 * self._timing_sums["nav"] / max(1e-9, self._timing_sums["total"])
        model_share = 100.0 * self._timing_sums["model"] / max(1e-9, self._timing_sums["total"])
        viz_share = 100.0 * self._timing_sums["viz"] / max(1e-9, self._timing_sums["total"])

        logging.info(
            "[CIL profile] tick=%d avg_fps=%.2f read=%.2fms nav=%.2fms(%.1f%%) model=%.2fms(%.1f%%) control=%.2fms viz=%.2fms(%.1f%%) telemetry=%.2fms",
            step_idx,
            avg_fps,
            1000.0 * self._timing_sums["read"] / n,
            1000.0 * self._timing_sums["nav"] / n,
            nav_share,
            1000.0 * self._timing_sums["model"] / n,
            model_share,
            1000.0 * self._timing_sums["control"] / n,
            1000.0 * self._timing_sums["viz"] / n,
            viz_share,
            1000.0 * self._timing_sums["telemetry"] / n,
        )

        self._timing_window_ticks = 0
        for key in self._timing_sums.keys():
            self._timing_sums[key] = 0.0

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
        try:
            if BasicAgent is None:
                raise RuntimeError("BasicAgent missing")
            if self.config.nav_agent_type.lower() != "basic":
                logging.warning("CIL command injection uses BasicAgent only; ignoring nav_agent_type=%s.", self.config.nav_agent_type)
            self._nav_agent = BasicAgent(vehicle, target_speed=max(10.0, self.config.target_speed_kmh))
            self._set_configured_destination(vehicle)
            logging.info("CIL navigation planner initialized: shared intersection-only BasicAgent command source.")
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
        self._cache_reference_route_plan(force=True)
        self._command_oracle.reset()

    def _get_reference_route_plan(self):
        return list(self._reference_route_plan)

    def _cache_reference_route_plan(self, force: bool = False) -> int:
        if self._nav_agent is None:
            self._reference_route_plan = []
            return 0
        if self._reference_route_plan and not force:
            return int(len(self._reference_route_plan))
        world = self.session.world if self.session is not None else None
        world_map = world.get_map() if world is not None else None
        start_loc = self._route_start_location
        if start_loc is None:
            start_loc = self._vehicle_location()
        destination_loc = self._route_destination_location

        reference_route = build_global_reference_route(
            world_map=world_map,
            start_location=start_loc,
            destination_location=destination_loc,
        )
        route_source = "global"
        if not reference_route:
            planner = self._get_local_planner()
            reference_route = snapshot_planner_route(planner)
            route_source = "planner_snapshot"
        if reference_route or force:
            self._reference_route_plan = list(reference_route)
        if reference_route:
            logging.info(
                "CIL cached fixed route plan with %d points from %s.",
                len(reference_route),
                route_source,
            )
        return int(len(self._reference_route_plan))

    def _maybe_replan_route(self, step_idx: int, vehicle) -> None:
        if self._nav_agent is None or self._route_destination_location is None:
            return
        if self._reference_route_plan:
            return
        if step_idx - int(self._last_replan_tick) < 30:
            return

        planner = self._get_local_planner()
        queue_size = self._planner_queue_size(planner)
        if queue_size is None or queue_size > self.REPLAN_MIN_QUEUE_SIZE:
            return

        distance_to_junction_m = self._distance_to_next_junction_m()
        upcoming_command, _distance_to_turn_m = self._extract_upcoming_turn_signal()
        if (
            self._is_in_junction()
            or upcoming_command in (1, 2, 3)
            or (
                math.isfinite(distance_to_junction_m)
                and distance_to_junction_m <= float(self.REPLAN_GUARD_JUNCTION_M)
            )
        ):
            logging.debug(
                "CIL skipped route replan near junction/turn (queue_size=%s, cmd=%s, d_junc=%.1f).",
                queue_size,
                upcoming_command,
                float(distance_to_junction_m),
            )
            return

        try:
            current_loc = vehicle.get_location()
            destination_distance_m = self._distance_to_destination(current_loc)
            if (
                destination_distance_m is not None
                and destination_distance_m <= float(self.REPLAN_GUARD_DESTINATION_M)
            ):
                logging.debug(
                    "CIL skipped route replan near destination (queue_size=%s, d_dest=%.1f).",
                    queue_size,
                    float(destination_distance_m),
                )
                return
            set_navigation_destination(self._nav_agent, current_loc, self._route_destination_location)
            self._last_replan_tick = int(step_idx)
            logging.info("CIL replanned BasicAgent route (queue_size=%s).", queue_size)
        except Exception as exc:
            self._last_replan_tick = int(step_idx)
            logging.debug("CIL replan attempt failed: %s", exc)

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

    def _vehicle_location(self):
        if self.session is None or self.session.ego_vehicle is None:
            return None
        return self.session.ego_vehicle.get_location()

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

    def _get_local_planner(self):
        if self._nav_agent is None or not hasattr(self._nav_agent, "get_local_planner"):
            return None
        try:
            return self._nav_agent.get_local_planner()
        except Exception:
            return None

    @staticmethod
    def _planner_queue_size(planner: Any) -> Optional[int]:
        if planner is None:
            return None
        for attr_name in ("_waypoints_queue", "_waypoint_buffer"):
            items = getattr(planner, attr_name, None)
            if items is None:
                continue
            try:
                return int(len(items))
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        return float(wrapped)

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
        planner = self._get_local_planner()
        if planner is None:
            return 0, float("inf")

        distance_to_junction_m = self._distance_to_next_junction_m()
        if not math.isfinite(distance_to_junction_m):
            distance_to_junction_m = float("inf")

        for attr_name in ("target_road_option", "_target_road_option"):
            command = self._road_option_to_turn_command(getattr(planner, attr_name, None))
            if command != 0:
                return command, float(distance_to_junction_m)

        for queue_name in ("_waypoint_buffer", "_waypoints_queue"):
            queue_attr = getattr(planner, queue_name, None)
            if not queue_attr:
                continue
            try:
                planner_items = list(queue_attr)
            except Exception:
                planner_items = []
            for item in planner_items[:96]:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                command = self._road_option_to_turn_command(item[1])
                if command != 0:
                    return command, float(distance_to_junction_m)
        return 0, float("inf")

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

    def _update_distance_based_command(
        self,
        speed_kmh: float,
        step_idx: Optional[int] = None,
    ) -> tuple[int, Dict[str, Any]]:
        del step_idx
        command, command_debug = self._command_oracle.update(
            speed_kmh=speed_kmh,
            route_start_location=self._route_start_location,
        )
        self._last_command_debug = command_debug
        return int(command), command_debug

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

    def _update_route_history(self, location: Any) -> None:
        self._route_planner.update_route_history(location)
        self._route_history_xy = list(self._route_planner.route_history_xy)

    @staticmethod
    def _overlay_xy(value: Optional[Any]) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        try:
            return (float(value.x), float(value.y))
        except Exception:
            pass
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            try:
                return (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _downsample_overlay_points(
        points_xy: list[tuple[float, float]],
        max_points: int,
    ) -> list[tuple[float, float]]:
        if max_points <= 1 or len(points_xy) <= max_points:
            return list(points_xy)
        step = max(1, int(math.ceil(len(points_xy) / float(max_points))))
        sampled = list(points_xy[::step])
        if sampled[-1] != points_xy[-1]:
            sampled.append(points_xy[-1])
        return sampled

    @staticmethod
    def _compute_route_overlay_bounds(
        points_xy: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        x_values = [pt[0] for pt in points_xy]
        y_values = [pt[1] for pt in points_xy]
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        margin_x = max(4.0, span_x * 0.15)
        margin_y = max(4.0, span_y * 0.15)
        return (min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y)

    def _update_route_overlay_bounds(
        self,
        points_xy: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        new_bounds = self._compute_route_overlay_bounds(points_xy)
        if self._route_overlay_bounds is None:
            self._route_overlay_bounds = new_bounds
            return new_bounds

        min_x, max_x, min_y, max_y = self._route_overlay_bounds
        new_min_x, new_max_x, new_min_y, new_max_y = new_bounds
        expand_margin = 1.0
        if new_min_x < (min_x + expand_margin):
            min_x = new_min_x
        if new_max_x > (max_x - expand_margin):
            max_x = new_max_x
        if new_min_y < (min_y + expand_margin):
            min_y = new_min_y
        if new_max_y > (max_y - expand_margin):
            max_y = new_max_y
        self._route_overlay_bounds = (min_x, max_x, min_y, max_y)
        return self._route_overlay_bounds

    def _get_spectator_overlay_basis(self) -> Optional[Dict[str, Any]]:
        if self.session is None or self.session.world is None or carla is None:
            return None
        try:
            spectator_tf = self.session.world.get_spectator().get_transform()
        except Exception:
            return None

        rot = spectator_tf.rotation
        pitch_rad = math.radians(float(rot.pitch))
        yaw_rad = math.radians(float(rot.yaw))
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        return {
            "origin": spectator_tf.location,
            "vfwd": (cos_p * cos_y, cos_p * sin_y, sin_p),
            "vup": (-sin_p * cos_y, -sin_p * sin_y, cos_p),
            "vright": (-sin_y, cos_y, 0.0),
        }

    def _overlay_plane_location(
        self,
        basis: Dict[str, Any],
        dist_s: float,
        offset_right: float,
        offset_up: float,
    ) -> Any:
        origin = basis["origin"]
        fx, fy, fz = basis["vfwd"]
        rx, ry, rz = basis["vright"]
        ux, uy, uz = basis["vup"]
        return carla.Location(
            x=float(origin.x) + fx * dist_s + rx * offset_right + ux * offset_up,
            y=float(origin.y) + fy * dist_s + ry * offset_right + uy * offset_up,
            z=float(origin.z) + fz * dist_s + rz * offset_right + uz * offset_up,
        )

    def _draw_overlay_panel(
        self,
        debug: Any,
        basis: Dict[str, Any],
        dist_s: float,
        left: float,
        top: float,
        width: float,
        height: float,
        life_time: float,
        fill_color: Any,
        border_color: Any,
        fill_rows: int = 14,
    ) -> None:
        if carla is None:
            return
        bottom = top - height
        rows = max(4, int(fill_rows))
        fill_thickness = max(0.02, min(0.05, height / max(8.0, float(rows))))
        for idx in range(rows + 1):
            y = bottom + (height * idx / rows)
            p0 = self._overlay_plane_location(basis, dist_s, left, y)
            p1 = self._overlay_plane_location(basis, dist_s, left + width, y)
            debug.draw_line(
                p0,
                p1,
                thickness=fill_thickness,
                color=fill_color,
                life_time=life_time,
                persistent_lines=False,
            )

        border_pts = [
            self._overlay_plane_location(basis, dist_s, left, top),
            self._overlay_plane_location(basis, dist_s, left + width, top),
            self._overlay_plane_location(basis, dist_s, left + width, bottom),
            self._overlay_plane_location(basis, dist_s, left, bottom),
        ]
        for idx in range(4):
            debug.draw_line(
                border_pts[idx],
                border_pts[(idx + 1) % 4],
                thickness=0.03,
                color=border_color,
                life_time=life_time,
                persistent_lines=False,
            )

    def _draw_route_map_overlay(
        self,
        debug: Any,
        basis: Dict[str, Any],
        vehicle_location: Any,
        heading_yaw_deg: float,
        route_locations: list[Any],
        command: int,
        life_time: float,
    ) -> None:
        if carla is None:
            return

        route_xy = [pt for pt in (self._overlay_xy(loc) for loc in route_locations) if pt is not None]
        trajectory_xy = [pt for pt in (self._overlay_xy(loc) for loc in self._route_history_xy[-180:]) if pt is not None]
        start_xy = self._overlay_xy(self._route_start_location)
        dest_xy = self._overlay_xy(self._route_destination_location)
        current_xy = self._overlay_xy(vehicle_location)

        bounds_points = list(route_xy)
        bounds_points.extend(trajectory_xy[-45:])
        for pt in (start_xy, dest_xy, current_xy):
            if pt is not None:
                bounds_points.append(pt)
        if len(bounds_points) < 2:
            return

        min_x, max_x, min_y, max_y = self._update_route_overlay_bounds(bounds_points)
        span_x = max(1e-3, max_x - min_x)
        span_y = max(1e-3, max_y - min_y)

        panel_left = 0.90
        panel_top = 2.05
        panel_width = 2.75
        panel_height = 2.02
        panel_bottom = panel_top - panel_height
        dist_s = 4.25
        self._draw_overlay_panel(
            debug=debug,
            basis=basis,
            dist_s=dist_s,
            left=panel_left,
            top=panel_top,
            width=panel_width,
            height=panel_height,
            life_time=life_time,
            fill_color=carla.Color(r=18, g=18, b=18),
            border_color=carla.Color(r=105, g=105, b=105),
            fill_rows=18,
        )

        command_labels = {0: "LANE_FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
        legend_lines = [
            ("Route Map (Navigator vs Vehicle)", (230, 230, 230), panel_top - 0.12),
            ("Cyan: planner | Orange: trail", (205, 205, 205), panel_top - 0.30),
            (f"Command: {command_labels.get(int(command), f'CMD_{command}')}", (255, 230, 120), panel_top - 0.48),
        ]
        for text, (r, g, b), y in legend_lines:
            debug.draw_string(
                self._overlay_plane_location(basis, dist_s, panel_left + 0.08, y),
                text,
                False,
                carla.Color(r=r, g=g, b=b),
                life_time,
                False,
            )

        plot_left = panel_left + 0.10
        plot_right = panel_left + panel_width - 0.10
        plot_bottom = panel_bottom + 0.10
        plot_top = panel_top - 0.62
        plot_width = plot_right - plot_left
        plot_height = plot_top - plot_bottom

        def project_to_overlay(pt_xy: tuple[float, float]) -> Any:
            u = clamp((pt_xy[0] - min_x) / span_x, 0.0, 1.0)
            v = clamp((pt_xy[1] - min_y) / span_y, 0.0, 1.0)
            return self._overlay_plane_location(
                basis,
                dist_s,
                plot_left + u * plot_width,
                plot_bottom + v * plot_height,
            )

        for ratio in (0.0, 0.25, 0.50, 0.75, 1.0):
            y = plot_bottom + ratio * plot_height
            p0 = self._overlay_plane_location(basis, dist_s, plot_left, y)
            p1 = self._overlay_plane_location(basis, dist_s, plot_right, y)
            debug.draw_line(
                p0,
                p1,
                thickness=0.02,
                color=carla.Color(r=42, g=42, b=42),
                life_time=life_time,
                persistent_lines=False,
            )
            x = plot_left + ratio * plot_width
            p2 = self._overlay_plane_location(basis, dist_s, x, plot_bottom)
            p3 = self._overlay_plane_location(basis, dist_s, x, plot_top)
            debug.draw_line(
                p2,
                p3,
                thickness=0.02,
                color=carla.Color(r=42, g=42, b=42),
                life_time=life_time,
                persistent_lines=False,
            )

        def draw_polyline(
            points_xy: list[tuple[float, float]],
            color: Any,
            max_points: int,
            thickness: float,
        ) -> None:
            sampled = self._downsample_overlay_points(points_xy, max_points=max_points)
            if len(sampled) < 2:
                return
            prev_loc = project_to_overlay(sampled[0])
            for pt_xy in sampled[1:]:
                cur_loc = project_to_overlay(pt_xy)
                debug.draw_line(
                    prev_loc,
                    cur_loc,
                    thickness=thickness,
                    color=color,
                    life_time=life_time,
                    persistent_lines=False,
                )
                prev_loc = cur_loc

        draw_polyline(route_xy, carla.Color(r=0, g=220, b=255), max_points=48, thickness=0.035)
        draw_polyline(trajectory_xy, carla.Color(r=255, g=165, b=0), max_points=36, thickness=0.032)

        def offset_loc(base_loc: Any, dx: float, dy: float) -> Any:
            rx, ry, rz = basis["vright"]
            ux, uy, uz = basis["vup"]
            return carla.Location(
                x=float(base_loc.x) + rx * dx + ux * dy,
                y=float(base_loc.y) + ry * dx + uy * dy,
                z=float(base_loc.z) + rz * dx + uz * dy,
            )

        def draw_marker(pt_xy: Optional[tuple[float, float]], label: str, color: Any, size: float) -> None:
            if pt_xy is None:
                return
            loc = project_to_overlay(pt_xy)
            debug.draw_point(loc, size=size, color=color, life_time=life_time, persistent_lines=False)
            debug.draw_string(
                offset_loc(loc, 0.06, 0.06),
                label,
                False,
                color,
                life_time,
                False,
            )

        draw_marker(start_xy, "S", carla.Color(r=40, g=220, b=40), 0.07)
        draw_marker(dest_xy, "D", carla.Color(r=255, g=40, b=40), 0.07)

        if current_xy is not None:
            current_loc = project_to_overlay(current_xy)
            debug.draw_point(
                current_loc,
                size=0.08,
                color=carla.Color(r=255, g=255, b=0),
                life_time=life_time,
                persistent_lines=False,
            )
            yaw_rad = math.radians(float(heading_yaw_deg))
            heading_xy = (
                current_xy[0] + 4.0 * math.cos(yaw_rad),
                current_xy[1] + 4.0 * math.sin(yaw_rad),
            )
            heading_loc = project_to_overlay(heading_xy)
            debug.draw_line(
                current_loc,
                heading_loc,
                thickness=0.03,
                color=carla.Color(r=255, g=255, b=0),
                life_time=life_time,
                persistent_lines=False,
            )

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

    def _longitudinal_control_simple(
        self,
        speed_kmh: float,
        destination_distance_m: Optional[float] = None,
        command: int = 0,
        command_phase: str = "cruise",
        distance_to_turn_m: float = float("inf"),
        route_curve_strength: float = 0.0,
    ) -> tuple[float, float]:
        adaptive_target_kmh = max(5.0, float(self.config.target_speed_kmh))
        phase = str(command_phase).lower()
        distance_to_turn_m = float(distance_to_turn_m)
        route_curve_strength = clamp(float(route_curve_strength), 0.0, 1.0)

        if int(command) in (1, 2, 3) and phase in {"armed", "in_junction"}:
            if int(command) in (1, 2):
                entry_speed_kmh = 12.0
                comfort_decel_ms2 = 2.4
            else:
                entry_speed_kmh = 16.0
                comfort_decel_ms2 = 2.0

            if math.isfinite(distance_to_turn_m):
                entry_speed_mps = float(entry_speed_kmh) / 3.6
                approach_speed_kmh = (
                    math.sqrt(
                        max(
                            0.0,
                            entry_speed_mps * entry_speed_mps
                            + 2.0 * float(comfort_decel_ms2) * max(0.0, distance_to_turn_m),
                        )
                    )
                    * 3.6
                )
                adaptive_target_kmh = min(
                    adaptive_target_kmh,
                    clamp(float(approach_speed_kmh), float(entry_speed_kmh), adaptive_target_kmh),
                )
            else:
                adaptive_target_kmh = min(adaptive_target_kmh, float(entry_speed_kmh) + 6.0)

            if phase == "in_junction":
                adaptive_target_kmh = min(adaptive_target_kmh, float(entry_speed_kmh) + 2.0)

        # Speed shaping should follow route geometry, not tiny steering noise on
        # straight roads. Route curvature gives a stable slowdown signal.
        if route_curve_strength > 0.12:
            if int(command) == 0 and phase == "cruise":
                curve_cap_kmh = 50.0 - 26.0 * route_curve_strength
                adaptive_target_kmh = min(
                    adaptive_target_kmh,
                    clamp(curve_cap_kmh, 22.0, adaptive_target_kmh),
                )
            else:
                curve_cap_kmh = 42.0 - 18.0 * route_curve_strength
                adaptive_target_kmh = min(
                    adaptive_target_kmh,
                    clamp(curve_cap_kmh, 18.0, adaptive_target_kmh),
                )

        if destination_distance_m is not None and math.isfinite(float(destination_distance_m)):
            distance_m = max(0.0, float(destination_distance_m))
            stop_buffer = max(3.6, self._arrival_distance_m + 0.8)
            if distance_m <= 45.0:
                comfort_decel_ms2 = 2.8
                available = max(0.0, distance_m - stop_buffer)
                v_stop_kmh = math.sqrt(max(0.0, 2.0 * comfort_decel_ms2 * available)) * 3.6
                adaptive_target_kmh = min(adaptive_target_kmh, clamp(v_stop_kmh, 9.0, adaptive_target_kmh))
            if distance_m < 18.0:
                adaptive_target_kmh = min(
                    adaptive_target_kmh,
                    clamp(4.5 + 0.65 * distance_m, 4.0, adaptive_target_kmh),
                )

        self._speed_controller.set_target_speed(adaptive_target_kmh)
        throttle, brake = self._speed_controller.compute(speed_kmh)

        overspeed_kmh = max(0.0, float(speed_kmh) - float(adaptive_target_kmh))
        if overspeed_kmh > 2.5:
            feedforward_brake = clamp((overspeed_kmh - 2.5) / 10.0, 0.0, self.config.max_brake)
            if feedforward_brake > brake:
                brake = float(feedforward_brake)
                throttle = 0.0

        self._max_observed_speed_kmh = max(self._max_observed_speed_kmh, float(speed_kmh))
        blocked_condition = (
            self._max_observed_speed_kmh > 15.0
            and adaptive_target_kmh > 20.0
            and float(speed_kmh) < 1.0
            and float(throttle) > 0.45
            and float(brake) < 0.05
        )
        if blocked_condition:
            self._blocked_frames += 1
        else:
            self._blocked_frames = max(0, self._blocked_frames - 2)

        if self._blocked_frames == 12:
            self._speed_controller.reset()
            logging.warning(
                "CIL blocked-state detected at low speed; overriding throttle to avoid pushing into a wall/obstacle."
            )
        if self._blocked_frames >= 12:
            throttle = 0.0
            brake = max(float(brake), min(float(self.config.max_brake), 0.45))

        return throttle, brake

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

    def _stabilize_cil_steering(
        self,
        steering_raw: float,
        speed_kmh: float,
        command: int,
        command_phase: str,
    ) -> float:
        # ──────────────────────────────────────────────────────────────
        # [TEST] Bypass all smoothing – use 100% raw CNN model steering.
        # Comment block below and uncomment original logic to restore.
        # ──────────────────────────────────────────────────────────────
        self._last_steer = clamp(float(steering_raw), -1.0, 1.0)
        return float(self._last_steer)

        # ──────────────────────────────────────────────────────────────
        # [ORIGINAL] EMA smoothing + slew-rate limit (DISABLED FOR TEST)
        # ──────────────────────────────────────────────────────────────
        # phase = str(command_phase).lower()
        # alpha = clamp(self.config.steer_smoothing, 0.0, 0.98)
        #
        # # Keep stronger smoothing on lane-follow, slightly relax it around turns
        # # so the model can still commit to the intersection trajectory.
        # if int(command) in (1, 2, 3) and phase in {"armed", "in_junction"}:
        #     alpha = min(alpha, 0.76)
        #
        # smoothed = alpha * float(self._last_steer) + (1.0 - alpha) * float(steering_raw)
        #
        # # Per-tick steering slew-rate limit. High speed gets tighter limits to
        # # suppress left-right hunting; turn phases get a bit more headroom.
        # speed_ratio = clamp(
        #     float(speed_kmh) / max(10.0, float(self.config.target_speed_kmh)),
        #     0.0,
        #     1.4,
        # )
        # max_delta = 0.11 - 0.05 * min(speed_ratio, 1.0)
        # if phase == "armed":
        #     max_delta += 0.015
        # elif phase == "in_junction":
        #     max_delta += 0.030
        # max_delta = clamp(max_delta, 0.05, 0.14)
        #
        # smoothed = clamp(
        #     smoothed,
        #     float(self._last_steer) - max_delta,
        #     float(self._last_steer) + max_delta,
        # )
        # self._last_steer = clamp(smoothed, -1.0, 1.0)
        # return float(self._last_steer)

    @staticmethod
    def _xy_distance(a: Any, b: Any) -> float:
        return math.hypot(float(a.x - b.x), float(a.y - b.y))

    def _route_reference_state(
        self,
        vehicle_location: Any,
        route_locations: list[Any],
        speed_kmh: float,
    ) -> tuple[Any, float] | tuple[None, None]:
        if vehicle_location is None:
            return None, None

        if route_locations and len(route_locations) >= 2:
            probe = route_locations[: min(120, len(route_locations))]
            nearest_idx = min(
                range(len(probe)),
                key=lambda idx: self._xy_distance(probe[idx], vehicle_location),
            )
            lookahead_steps = int(clamp(float(speed_kmh) / 8.0, 2.0, 10.0))
            ref_idx = min(len(route_locations) - 1, nearest_idx + lookahead_steps)
            prev_idx = max(0, ref_idx - 2)
            ref_loc = route_locations[nearest_idx]
            next_loc = route_locations[ref_idx]
            prev_loc = route_locations[prev_idx]
            heading_yaw_deg = math.degrees(
                math.atan2(
                    float(next_loc.y - prev_loc.y),
                    float(next_loc.x - prev_loc.x),
                )
            )
            return ref_loc, float(heading_yaw_deg)

        waypoint = self._current_waypoint()
        if waypoint is None:
            return None, None
        return waypoint.transform.location, float(waypoint.transform.rotation.yaw)

    def _compute_route_curve_strength(
        self,
        vehicle_location: Any,
        route_locations: list[Any],
        speed_kmh: float,
    ) -> tuple[float, Dict[str, Any]]:
        if vehicle_location is None or len(route_locations) < 4:
            debug = {"curve_strength": 0.0, "curve_heading_delta_deg": 0.0}
            self._last_route_curve_debug = debug
            return 0.0, debug

        probe = route_locations[: min(140, len(route_locations))]
        nearest_idx = min(
            range(len(probe)),
            key=lambda idx: self._xy_distance(probe[idx], vehicle_location),
        )
        near_idx = min(
            len(route_locations) - 2,
            nearest_idx + int(clamp(float(speed_kmh) / 18.0, 2.0, 5.0)),
        )
        far_idx = min(
            len(route_locations) - 1,
            nearest_idx + int(clamp(float(speed_kmh) / 6.0, 6.0, 16.0)),
        )
        if near_idx <= nearest_idx or far_idx <= near_idx:
            debug = {"curve_strength": 0.0, "curve_heading_delta_deg": 0.0}
            self._last_route_curve_debug = debug
            return 0.0, debug

        base_loc = route_locations[nearest_idx]
        near_loc = route_locations[near_idx]
        far_loc = route_locations[far_idx]
        heading_near_deg = math.degrees(
            math.atan2(
                float(near_loc.y - base_loc.y),
                float(near_loc.x - base_loc.x),
            )
        )
        heading_far_deg = math.degrees(
            math.atan2(
                float(far_loc.y - near_loc.y),
                float(far_loc.x - near_loc.x),
            )
        )
        heading_delta_deg = abs(
            self._normalize_angle_deg(float(heading_far_deg) - float(heading_near_deg))
        )
        curve_strength = clamp((float(heading_delta_deg) - 4.0) / 26.0, 0.0, 1.0)
        debug = {
            "curve_strength": float(curve_strength),
            "curve_heading_delta_deg": float(heading_delta_deg),
        }
        self._last_route_curve_debug = debug
        return float(curve_strength), debug

    def _compute_route_centering_assist(
        self,
        vehicle: Any,
        speed_kmh: float,
        route_locations: list[Any],
        command: int,
        command_phase: str,
    ) -> tuple[float, Dict[str, Any]]:
        transform = vehicle.get_transform()
        vehicle_location = transform.location
        vehicle_yaw_deg = float(transform.rotation.yaw)
        ref_loc, ref_yaw_deg = self._route_reference_state(
            vehicle_location=vehicle_location,
            route_locations=route_locations,
            speed_kmh=speed_kmh,
        )
        if ref_loc is None or ref_yaw_deg is None:
            debug = {"assist": 0.0, "lateral_error_m": 0.0, "heading_error_deg": 0.0}
            self._last_route_assist_debug = debug
            return 0.0, debug

        yaw_rad = math.radians(float(ref_yaw_deg))
        dx = float(vehicle_location.x - ref_loc.x)
        dy = float(vehicle_location.y - ref_loc.y)

        # CARLA/UE uses +Y to the vehicle's right. We expose telemetry in a more
        # intuitive frame here: positive lateral/heading error means the vehicle
        # sits or points to the route's left, which requires a positive (right)
        # steering correction.
        lateral_error_m = math.sin(yaw_rad) * dx - math.cos(yaw_rad) * dy
        heading_error_deg = self._normalize_angle_deg(float(ref_yaw_deg) - float(vehicle_yaw_deg))
        speed_mps = max(0.5, float(speed_kmh) / 3.6)

        crosstrack_term = math.atan2(2.8 * float(lateral_error_m), speed_mps + 2.0)
        heading_term = 0.90 * math.radians(float(heading_error_deg))
        assist = crosstrack_term + heading_term

        phase = str(command_phase).lower()
        assist_cap = 0.28
        if int(command) in (1, 2, 3) and phase in {"armed", "in_junction"}:
            assist_cap = 0.16
        assist = clamp(float(assist), -assist_cap, assist_cap)

        debug = {
            "assist": float(assist),
            "lateral_error_m": float(lateral_error_m),
            "heading_error_deg": float(heading_error_deg),
        }
        self._last_route_assist_debug = debug
        return float(assist), debug

    def _update_hud_fps(self) -> float:
        """Track HUD FPS using exponential moving average."""
        now = time.perf_counter()
        if self._hud_last_tick_time is None:
            self._hud_last_tick_time = now
            return 0.0
        dt = now - self._hud_last_tick_time
        self._hud_last_tick_time = now
        if dt <= 1e-6:
            return float(self._hud_ema_fps) if self._hud_ema_fps is not None else 0.0
        instant_fps = 1.0 / dt
        if self._hud_ema_fps is None:
            self._hud_ema_fps = instant_fps
        else:
            self._hud_ema_fps = 0.90 * self._hud_ema_fps + 0.10 * instant_fps
        return self._hud_ema_fps

    def _draw_hud_on_screen(
        self,
        step_idx: int,
        speed_kmh: float,
        target_speed_kmh: float,
        steer: float,
        throttle: float,
        brake: float,
        command: int,
        fps: float,
        destination_distance_m: Optional[float],
        route_locations: list,
    ) -> None:
        """Draw telemetry HUD and route lines directly in the CARLA viewport."""
        if self.session is None or self.session.world is None or carla is None:
            return
        if self.session.ego_vehicle is None:
            return
        show_hud = bool(self.config.cil_enable_hud)
        show_route_map = bool(self.config.cil_enable_route_map)
        if not show_hud and not show_route_map:
            return

        vehicle = self.session.ego_vehicle
        world = self.session.world
        debug = world.debug
        life_time = max(0.08, self.config.fixed_delta * 2.5 if self.config.sync else 0.12)

        transform = vehicle.get_transform()
        loc = transform.location
        basis = self._get_spectator_overlay_basis()
        if basis is None:
            return

        command_labels = {0: "LANE_FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
        cmd_label = command_labels.get(int(command), f"CMD_{command}")
        dist_text = f"{destination_distance_m:.1f}m" if destination_distance_m is not None else "N/A"

        hud_lines = [
            f"Speed: {speed_kmh:.1f} / {target_speed_kmh:.1f} km/h",
            f"Steer: {steer:+.3f}",
            f"Thr: {throttle:.2f}  Brk: {brake:.2f}",
            f"Cmd: {cmd_label}  FPS: {fps:.0f}",
            f"Dist: {dist_text}",
        ]

        hud_dist = 4.25
        left_off = -3.55
        top_off = 2.02
        line_h = 0.23
        hud_life_time = max(
            0.015,
            min(0.05, (0.70 * self.config.fixed_delta) if self.config.sync else 0.05),
        )
        if show_hud:
            self._draw_overlay_panel(
                debug=debug,
                basis=basis,
                dist_s=hud_dist,
                left=left_off - 0.16,
                top=top_off + 0.10,
                width=2.05,
                height=1.30,
                life_time=hud_life_time,
                fill_color=carla.Color(r=20, g=20, b=20),
                border_color=carla.Color(r=92, g=92, b=92),
                fill_rows=12,
            )

            for i, line in enumerate(hud_lines):
                debug.draw_string(
                    self._overlay_plane_location(basis, hud_dist, left_off, top_off - i * line_h),
                    line,
                    False,
                    carla.Color(r=255, g=255, b=0),
                    hud_life_time,
                    False,
                )

        # ── Route visualization (3D lines in the world) ──
        if show_route_map and route_locations and len(route_locations) >= 2:
            lift = 0.40
            step = max(1, len(route_locations) // 40)
            for idx in range(0, len(route_locations) - step, step):
                loc_a = route_locations[idx]
                loc_b = route_locations[min(idx + step, len(route_locations) - 1)]
                p0 = carla.Location(x=float(loc_a.x), y=float(loc_a.y), z=float(loc_a.z) + lift)
                p1 = carla.Location(x=float(loc_b.x), y=float(loc_b.y), z=float(loc_b.z) + lift)
                debug.draw_line(
                    p0, p1,
                    thickness=0.08,
                    color=carla.Color(r=0, g=220, b=255),
                    life_time=life_time,
                    persistent_lines=False,
                )

        if show_route_map and route_locations:
            self._draw_route_map_overlay(
                debug=debug,
                basis=basis,
                vehicle_location=loc,
                heading_yaw_deg=float(transform.rotation.yaw),
                route_locations=route_locations,
                command=command,
                life_time=hud_life_time,
            )

        # Draw destination marker
        if show_route_map and self._route_destination_location is not None:
            d_loc = self._route_destination_location
            d_point = carla.Location(
                x=float(d_loc.x), y=float(d_loc.y), z=float(d_loc.z) + 2.0,
            )
            debug.draw_point(d_point, size=0.20, color=carla.Color(r=255, g=50, b=50), life_time=life_time, persistent_lines=False)
            debug.draw_string(d_point, "  DEST", False, carla.Color(r=255, g=50, b=50), life_time, False)

    def _update_spectator_follow(self) -> None:
        """Lock the spectator camera to follow behind the ego vehicle each tick."""
        if self.session is None or self.session.world is None or self.session.ego_vehicle is None:
            return
        if carla is None:
            return

        vehicle = self.session.ego_vehicle
        transform = vehicle.get_transform()
        forward = transform.get_forward_vector()
        loc = transform.location

        follow_dist = float(self.config.spectator_follow_distance)
        height = float(self.config.spectator_height)
        pitch = float(self.config.spectator_pitch)

        spectator_loc = carla.Location(
            x=float(loc.x) - float(forward.x) * follow_dist,
            y=float(loc.y) - float(forward.y) * follow_dist,
            z=float(loc.z) + height,
        )
        spectator_rot = carla.Rotation(
            pitch=pitch,
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        )
        try:
            self.session.world.get_spectator().set_transform(
                carla.Transform(spectator_loc, spectator_rot)
            )
        except Exception:
            pass

    def run_step(self, step_idx: int) -> None:
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("CIL agent waiting for CARLA runtime.")
            return

        tick_t0 = time.perf_counter()
        stage_times = {
            "read": 0.0,
            "nav": 0.0,
            "model": 0.0,
            "control": 0.0,
            "viz": 0.0,
            "telemetry": 0.0,
            "total": 0.0,
        }
        read_t0 = tick_t0

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
        stage_times["read"] = time.perf_counter() - read_t0

        self._write_video_frame(frame)
        if self._stop_requested:
            logging.info("CIL video duration target reached, stopping agent loop.")
            return

        destination_distance_m = self._distance_to_destination(vehicle.get_location())
        if destination_distance_m is not None and destination_distance_m <= self._arrival_distance_m:
            self._request_stop_at_destination("distance_threshold", destination_distance_m)
            return

        nav_t0 = time.perf_counter()

        if self._nav_agent is not None:
            if not self._reference_route_plan:
                self._cache_reference_route_plan()
            try:
                if self._nav_agent.done():
                    self._request_stop_at_destination("planner_done", destination_distance_m)
                    return
            except Exception:
                pass
            self._refresh_planner_state()
            self._maybe_replan_route(step_idx, vehicle)

        vehicle_location = vehicle.get_location()
        if self._reference_route_plan:
            route_locations = self._route_planner.collect_reference_route_locations(
                self._reference_route_plan,
                anchor_location=vehicle_location,
            )
        else:
            route_locations = self._route_planner.collect_route_locations(
                self._nav_agent,
                anchor_location=vehicle_location,
            )

        command, command_debug = self._update_distance_based_command(
            speed_kmh=speed_kmh,
            step_idx=step_idx,
        )

        stage_times["nav"] = time.perf_counter() - nav_t0

        model_t0 = time.perf_counter()
        model_steer = self._predict_cil_steering(frame, speed_kmh, command)
        stage_times["model"] = time.perf_counter() - model_t0

        control_t0 = time.perf_counter()
        steering = self._stabilize_cil_steering(
            steering_raw=model_steer,
            speed_kmh=speed_kmh,
            command=command,
            command_phase=str(command_debug.get("phase", "cruise")),
        )
        route_assist, route_assist_debug = self._compute_route_centering_assist(
            vehicle=vehicle,
            speed_kmh=speed_kmh,
            route_locations=route_locations,
            command=command,
            command_phase=str(command_debug.get("phase", "cruise")),
        )
        route_curve_strength, route_curve_debug = self._compute_route_curve_strength(
            vehicle_location=vehicle_location,
            route_locations=route_locations,
            speed_kmh=speed_kmh,
        )
        # [TEST] Route centering assist disabled – using 100% raw model steering.
        # steering = clamp(steering + float(route_assist), -1.0, 1.0)

        throttle, brake = self._longitudinal_control_simple(
            speed_kmh=speed_kmh,
            destination_distance_m=destination_distance_m,
            command=command,
            command_phase=str(command_debug.get("phase", "cruise")),
            distance_to_turn_m=float(command_debug.get("distance_to_turn_m", float("inf"))),
            route_curve_strength=float(route_curve_strength),
        )
        adaptive_target_kmh = float(self._speed_controller.target_speed_kmh)
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
        self._update_route_history(vehicle_location)

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

        stage_times["control"] = time.perf_counter() - control_t0

        viz_t0 = time.perf_counter()

        hud_fps = self._update_hud_fps()
        self._update_spectator_follow()
        self._draw_hud_on_screen(
            step_idx, speed_kmh, adaptive_target_kmh,
            control.steer, control.throttle, control.brake,
            command, hud_fps, destination_distance_m, route_locations,
        )

        # ── Separate OpenCV route-map window ──
        if self._route_map is not None and step_idx % 3 == 0:
            vehicle_location = vehicle.get_location()
            heading_yaw = float(vehicle.get_transform().rotation.yaw)
            # Use the FULL reference route plan (S→D) instead of the clipped
            # route_locations so the entire planned path is always visible.
            full_route_locs = [
                entry["location"]
                for entry in self._reference_route_plan
                if entry.get("location") is not None
            ] if self._reference_route_plan else route_locations
            self._route_map.show(
                route_points=full_route_locs,
                current_location=vehicle_location,
                start_location=self._route_start_location,
                destination_location=self._route_destination_location,
                heading_yaw_deg=heading_yaw,
                trajectory_points=self._route_history_xy,
                command=command,
            )

        stage_times["viz"] = time.perf_counter() - viz_t0

        telemetry_t0 = time.perf_counter()

        if step_idx % 20 == 0:
            logging.info(
                "cil tick=%d speed=%.1f km/h target=%.1f cmd=%d phase=%s next=%d src=%s reset=%s s_from_start=%.1f d_turn=%.1f d_junc=%.1f trigger=%.1f steer=%.3f model=%.3f assist=%.3f lat=%.2f hdg=%.1f curve=%.2f throttle=%.2f brake=%.2f",
                step_idx,
                speed_kmh,
                adaptive_target_kmh,
                command,
                str(command_debug.get("phase", "cruise")),
                int(command_debug.get("upcoming_command", 0)),
                str(command_debug.get("upcoming_source", "none")),
                str(command_debug.get("reset_reason", "none")),
                float(command_debug.get("distance_from_start_m", float("inf"))),
                float(command_debug.get("distance_to_turn_m", float("inf"))),
                float(command_debug.get("distance_to_junction_m", float("inf"))),
                float(command_debug.get("trigger_distance_m", 0.0)),
                control.steer,
                model_steer,
                float(route_assist_debug.get("assist", 0.0)),
                float(route_assist_debug.get("lateral_error_m", 0.0)),
                float(route_assist_debug.get("heading_error_deg", 0.0)),
                float(route_curve_debug.get("curve_strength", 0.0)),
                control.throttle,
                control.brake,
            )

        if self._telemetry_writer is not None:
            self._telemetry_writer.writerow(
                [
                    int(step_idx),
                    f"{float(speed_kmh):.3f}",
                    f"{float(adaptive_target_kmh):.3f}",
                    f"{float(command_debug.get('distance_to_turn_m', float('inf'))):.3f}",
                    f"{float(command_debug.get('distance_to_junction_m', float('inf'))):.3f}",
                    int(command),
                    str(command_debug.get("phase", "cruise")),
                    str(command_debug.get("active_source", "none")),
                    int(command_debug.get("upcoming_command", 0)),
                    str(command_debug.get("reset_reason", "none")),
                    f"{float(model_steer):.4f}",
                    f"{float(route_assist_debug.get('assist', 0.0)):.4f}",
                    f"{float(route_assist_debug.get('lateral_error_m', 0.0)):.4f}",
                    f"{float(route_assist_debug.get('heading_error_deg', 0.0)):.4f}",
                    f"{float(route_curve_debug.get('curve_strength', 0.0)):.4f}",
                    f"{float(control.steer):.4f}",
                    f"{float(control.throttle):.4f}",
                    f"{float(control.brake):.4f}",
                ]
            )
            if step_idx % 20 == 0 and self._telemetry_fp is not None:
                self._telemetry_fp.flush()

        stage_times["telemetry"] = time.perf_counter() - telemetry_t0
        stage_times["total"] = time.perf_counter() - tick_t0
        self._accumulate_tick_timing(stage_times, step_idx)

    def teardown(self) -> None:
        self._route_overlay_bounds = None
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
        self._reference_route_plan = []
        self._route_history_xy = []
        self._route_planner.reset_runtime_state()
        self._command_oracle.reset()
        self._active_navigation_command = 0
        self._active_command_source = "none"
        self._command_phase = "cruise"
        self._command_latch_frames = 0
        self._command_entered_junction = False
        self._last_command_debug = {}
        self._last_steer = 0.0
        self._max_observed_speed_kmh = 0.0
        self._blocked_frames = 0
        self._last_route_assist_debug = {}
        self._last_route_curve_debug = {}
        self._hud_ema_fps = None
        self._hud_last_tick_time = None
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
    DESTINATION_APPROACH_OFFSET_M = 12.0
    MIN_RANDOM_DEST_DISTANCE_M = 80.0

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
        self._route_destination_index: Optional[int] = None
        self._route_destination_location = None
        self._fixed_destination_consumed = False
        self._tm_fallback_mode = False
        self._traffic_supervisor = None
        self._last_supervisor_debug_info: Dict[str, Any] = {}
        self._last_control_ts: Optional[float] = None
        self._steer_ema_alpha = 0.85
        self._smoothed_steer = 0.0
        self._has_smoothed_steer = False
        self._use_depth_camera = False
        self._hud_ema_fps: Optional[float] = None
        self._hud_last_tick_time: Optional[float] = None
        self._vehicle_max_steer_angle_deg: Optional[float] = None

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        self._smoothed_steer = 0.0
        self._has_smoothed_steer = False
        self._route_destination_index = None
        self._route_destination_location = None
        self._fixed_destination_consumed = False
        self._hud_ema_fps = None
        self._hud_last_tick_time = None
        self._vehicle_max_steer_angle_deg = None
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
            inference_imgsz=448,
            camera_fov_deg=self.config.camera_fov,
            obstacle_base_distance_m=8.0,
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
        )
        self._use_depth_camera = bool(getattr(self._detector, "uses_depth_input", False))
        warmup_fn = getattr(self._detector, "warmup", None)
        if callable(warmup_fn):
            warmup_t0 = time.perf_counter()
            warmup_fn(self.config.camera_width, self.config.camera_height)
            logging.info(
                "YOLO detector warmup completed in %.1f ms.",
                (time.perf_counter() - warmup_t0) * 1000.0,
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
        if self._use_depth_camera:
            try:
                self._depth_camera = self._spawn_depth_camera(world, vehicle)
                self._depth_camera.listen(self._on_depth_frame)
            except Exception as exc:
                self._depth_camera = None
                logging.warning(
                    "Depth camera unavailable for yolo_detect, fallback to bbox distance. Reason: %s",
                    exc,
                )
        else:
            self._depth_camera = None
            logging.info("YOLO detector does not consume depth_map_m; skipping depth camera for higher FPS.")
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
        world = self.session.world if self.session is not None else None
        world_map = world.get_map() if world is not None else None
        current_loc = vehicle.get_location()
        current_idx = self._nearest_spawn_index(current_loc)

        if self.config.destination_point >= 0 and not self._fixed_destination_consumed:
            dest_idx = self.config.destination_point % len(self._spawn_points)
        else:
            dest_idx = self._choose_random_destination_index(current_loc, current_idx)

        raw_destination = self._spawn_points[dest_idx].location
        destination = self._refine_destination_location(world_map, current_loc, raw_destination)
        self._route_destination_index = int(dest_idx)
        self._route_destination_location = destination
        set_navigation_destination(self._nav_agent, current_loc, destination)
        logging.info(
            "YOLO planner destination set: spawn=%d raw=(%.1f, %.1f) target=(%.1f, %.1f)",
            dest_idx,
            float(raw_destination.x),
            float(raw_destination.y),
            float(destination.x),
            float(destination.y),
        )

    def _nearest_spawn_index(self, location) -> Optional[int]:
        if location is None or not self._spawn_points:
            return None
        best_idx = None
        best_dist = float("inf")
        for idx, transform in enumerate(self._spawn_points):
            dist = location.distance(transform.location)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _choose_random_destination_index(self, current_loc, current_idx: Optional[int]) -> int:
        if not self._spawn_points:
            raise RuntimeError("No spawn points available for YOLO destination selection.")

        candidates: list[int] = []
        min_distance = float(self.MIN_RANDOM_DEST_DISTANCE_M)
        for idx, transform in enumerate(self._spawn_points):
            if current_idx is not None and idx == current_idx:
                continue
            if self._route_destination_index is not None and idx == self._route_destination_index:
                continue
            if current_loc is not None and current_loc.distance(transform.location) < min_distance:
                continue
            candidates.append(idx)

        if not candidates:
            for idx, _transform in enumerate(self._spawn_points):
                if current_idx is not None and idx == current_idx:
                    continue
                if self._route_destination_index is not None and idx == self._route_destination_index:
                    continue
                candidates.append(idx)

        if not candidates:
            return int((current_idx or 0) % len(self._spawn_points))
        return int(random.choice(candidates))

    def _refine_destination_location(self, world_map, current_loc, destination_loc):
        if carla is None or world_map is None or current_loc is None or destination_loc is None:
            return destination_loc
        try:
            dest_wp = world_map.get_waypoint(
                destination_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            dest_wp = None
        if dest_wp is None:
            return destination_loc

        offset = float(self.DESTINATION_APPROACH_OFFSET_M)
        candidates = []
        try:
            candidates.extend(dest_wp.previous(offset))
        except Exception:
            pass
        try:
            candidates.extend(dest_wp.next(offset))
        except Exception:
            pass

        if not candidates:
            return dest_wp.transform.location

        best_wp = min(
            candidates,
            key=lambda wp: current_loc.distance(wp.transform.location),
        )
        return best_wp.transform.location

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
        frame_bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        with self._frame_lock:
            self._latest_rgb = frame_bgr

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

    def _update_hud_fps(self) -> float:
        now = time.perf_counter()
        if self._hud_last_tick_time is None:
            self._hud_last_tick_time = now
            return 0.0
        dt = now - self._hud_last_tick_time
        self._hud_last_tick_time = now
        if dt <= 1e-6:
            return float(self._hud_ema_fps) if self._hud_ema_fps is not None else 0.0
        instant_fps = 1.0 / dt
        if self._hud_ema_fps is None:
            self._hud_ema_fps = instant_fps
        else:
            self._hud_ema_fps = 0.90 * self._hud_ema_fps + 0.10 * instant_fps
        return self._hud_ema_fps

    def _resolve_vehicle_max_steer_angle_deg(self, vehicle: Any) -> Optional[float]:
        if vehicle is None:
            return None
        if self._vehicle_max_steer_angle_deg is not None:
            return self._vehicle_max_steer_angle_deg
        try:
            physics_control = vehicle.get_physics_control()
            wheels = list(getattr(physics_control, "wheels", []))
        except Exception:
            return None

        candidate_angles: list[float] = []
        if len(wheels) >= 2:
            candidate_angles.extend(float(getattr(wheel, "max_steer_angle", 0.0)) for wheel in wheels[:2])
        candidate_angles = [angle for angle in candidate_angles if angle > 0.0]
        if not candidate_angles:
            candidate_angles = [
                float(getattr(wheel, "max_steer_angle", 0.0))
                for wheel in wheels
                if float(getattr(wheel, "max_steer_angle", 0.0)) > 0.0
            ]
        if not candidate_angles:
            return None

        self._vehicle_max_steer_angle_deg = max(candidate_angles)
        return self._vehicle_max_steer_angle_deg

    def _steer_to_angle_deg(self, vehicle: Any, steer_value: Optional[float]) -> Optional[float]:
        if steer_value is None:
            return None
        max_steer_angle_deg = self._resolve_vehicle_max_steer_angle_deg(vehicle)
        if max_steer_angle_deg is None:
            return None
        return clamp(float(steer_value), -1.0, 1.0) * max_steer_angle_deg

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
        frame_bgr, depth_map_m = self._read_latest_frame()
        if frame_bgr is None:
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
        display_steer = 0.0
        display_throttle = 0.0
        display_brake = 0.0
        
        if vehicle is not None:
            try:
                # Smooth raw autopilot steer command to avoid corridor jitter between frames.
                control_snapshot = vehicle.get_control()
                raw_steer = float(control_snapshot.steer)
                display_throttle = float(control_snapshot.throttle)
                display_brake = float(control_snapshot.brake)
                alpha = clamp(float(self._steer_ema_alpha), 0.0, 0.98)
                if not self._has_smoothed_steer:
                    self._smoothed_steer = raw_steer
                    self._has_smoothed_steer = True
                else:
                    self._smoothed_steer = (
                        alpha * self._smoothed_steer + (1.0 - alpha) * raw_steer
                    )
                current_steer = float(self._smoothed_steer)
                display_steer = current_steer
            except Exception:
                current_steer = float(self._smoothed_steer)
                display_steer = current_steer
            try:
                velocity = vehicle.get_velocity()
                speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            except Exception:
                speed_kmh = None

        # ─────────────────────────────────────────────────────────
        # Step 3: Run YOLO Detection
        # ─────────────────────────────────────────────────────────
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
                if current_steer is None:
                    current_steer = 0.0
                if speed_kmh is None:
                    speed_kmh = 0.0

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
                    if self.config.destination_point >= 0 and not self._fixed_destination_consumed:
                        self._fixed_destination_consumed = True
                        logging.info(
                            "YOLO planner reached configured destination spawn=%s; switching to roaming destinations.",
                            self._route_destination_index,
                        )
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
            display_steer = float(nav_control.steer)
            display_throttle = float(nav_control.throttle)
            display_brake = float(nav_control.brake)

        # Emergency override on top of CARLA autopilot (TM fallback)
        elif is_emergency and vehicle is not None and self._nav_agent is None:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = float(0.0 if current_steer is None else current_steer)
            control.brake = float(clamp(max(1.0, supervisor_brake), 0.0, 1.0))
            control.hand_brake = False
            vehicle.apply_control(control)
            display_steer = float(control.steer)
            display_throttle = float(control.throttle)
            display_brake = float(control.brake)
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
            display_steer = float(control.steer)
            display_throttle = float(control.throttle)
            display_brake = float(control.brake)

        # ─────────────────────────────────────────────────────────
        # Step 6: Prepare Annotation Frame
        # ─────────────────────────────────────────────────────────
        annotated_frame = frame_bgr.copy()
        hud_fps = self._update_hud_fps()
        steer_angle_deg = self._steer_to_angle_deg(vehicle, display_steer)
        speed_text = f"{float(speed_kmh):.1f} km/h" if speed_kmh is not None else "n/a"
        steer_text = f"STEER={display_steer:+.3f}"
        if steer_angle_deg is not None:
            steer_text = f"{steer_text} ({steer_angle_deg:+.1f} deg)"

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

        demo_text = f"{steer_text} | FPS={hud_fps:.1f} | SPEED={speed_text}"
        cv2.putText(
            annotated_frame,
            demo_text,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (40, 220, 255),
            2,
        )

        control_text = f"THR={display_throttle:.2f} | BRK={display_brake:.2f} | DETS={len(detections)}"
        cv2.putText(
            annotated_frame,
            control_text,
            (10, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
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
                "yolo_detect tick=%d fps=%.1f speed=%s steer=%.3f angle=%s throttle=%.2f brake=%.2f "
                "detections=%d emergency=%s supervisor_brake=%.2f state=%s reason=%s yellow_polygon=%s",
                step_idx,
                hud_fps,
                speed_text,
                display_steer,
                f"{steer_angle_deg:+.1f}deg" if steer_angle_deg is not None else "n/a",
                display_throttle,
                display_brake,
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
        self._smoothed_steer = 0.0
        self._has_smoothed_steer = False
        self._route_destination_index = None
        self._route_destination_location = None
        self._fixed_destination_consumed = False
        self._hud_ema_fps = None
        self._hud_last_tick_time = None
        self._vehicle_max_steer_angle_deg = None

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


def _to_bool(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return bool(fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(fallback)


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
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--tm-port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--sync",
        dest="sync",
        action="store_true",
        default=None,
        help="Enable synchronous mode.",
    )
    parser.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        help="Disable synchronous mode.",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=None,
        help="Fixed delta seconds (used when --sync is enabled).",
    )
    parser.add_argument("--no-rendering", action="store_true", default=None)
    parser.add_argument(
        "--map",
        default=None,
        help="CARLA map to load before spawning vehicle (default: Town03).",
    )
    parser.add_argument("--vehicle-filter", default=None)
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=None,
        help="Spawn-point index. Negative means random.",
    )
    parser.add_argument(
        "--destination-point",
        type=int,
        default=None,
        help="Destination point index B. Negative means random destination.",
    )
    parser.add_argument("--npc-vehicle-count", type=int, default=None)
    parser.add_argument("--npc-bike-count", type=int, default=None)
    parser.add_argument("--npc-motorbike-count", type=int, default=None)
    parser.add_argument("--npc-pedestrian-count", type=int, default=None)
    parser.add_argument(
        "--disable-npc-autopilot",
        action="store_true",
        default=None,
        help="Spawn NPCs but do not enable autopilot for them.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Number of ticks to run. Use 0 or negative for infinite loop.",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=None,
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
    parser.add_argument("--target-speed-kmh", type=float, default=None)
    parser.add_argument("--max-throttle", type=float, default=None)
    parser.add_argument("--max-brake", type=float, default=None)
    parser.add_argument(
        "--steer-smoothing",
        type=float,
        default=None,
        help="0 means no smoothing, closer to 1 means smoother steering.",
    )
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fov", type=float, default=None)

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
        default=None,
        help="Disable spectator spawn lock.",
    )
    parser.add_argument(
        "--spectator-reapply-each-tick",
        action="store_true",
        default=None,
        help="Re-apply spectator transform every tick.",
    )
    parser.add_argument("--spectator-follow-distance", type=float, default=None)
    parser.add_argument("--spectator-height", type=float, default=None)
    parser.add_argument("--spectator-pitch", type=float, default=None)

    parser.add_argument("--collect-data", action="store_true", default=None)
    parser.add_argument("--collect-data-dir", default=None)
    parser.add_argument("--image-prefix", default=None, help="Image filename prefix (e.g. town01_beochan)")
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=None,
        help="Only save a frame every N ticks (default: 50 for diverse scenes).",
    )
    parser.add_argument(
        "--nav-agent-type",
        choices=["basic", "behavior"],
        default=None,
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
        default=None,
        help="In yolo_detect mode, keep autopilot traffic-light handling enabled.",
    )
    parser.add_argument(
        "--no-random-weather",
        action="store_true",
        default=None,
        help="Disable random weather preset selection at start.",
    )
    parser.add_argument(
        "--recovery-interval-frames",
        type=int,
        default=None,
        help="Apply recovery steering disturbance every N frames.",
    )
    parser.add_argument(
        "--recovery-duration-frames",
        type=int,
        default=None,
        help="Number of frames to keep disturbance active.",
    )
    parser.add_argument(
        "--recovery-steer-offset",
        type=float,
        default=None,
        help="Steering offset added during disturbance window.",
    )
    parser.add_argument("--record-video", action="store_true", default=None)
    parser.add_argument("--video-output-path", default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-duration-sec", type=int, default=None)
    parser.add_argument("--video-codec", default=None)

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    env_cfg = load_env_config(args.config)

    def pick(cli_value: Any, section: str, key: str, fallback: Any) -> Any:
        if cli_value is not None:
            return cli_value
        return _cfg_get(env_cfg, section, key, fallback)

    host = str(pick(args.host, "carla", "host", "127.0.0.1"))
    port = int(pick(args.port, "carla", "port", 2000))
    tm_port = int(pick(args.tm_port, "carla", "tm_port", 8000))
    timeout = float(pick(args.timeout, "carla", "timeout", 60.0))
    map_name = str(pick(args.map, "carla", "map", "Town03"))
    vehicle_filter = str(pick(args.vehicle_filter, "vehicle", "filter", "vehicle.tesla.model3"))

    sync = _to_bool(args.sync, _to_bool(_cfg_get(env_cfg, "carla", "sync", False), False))
    fixed_delta = float(pick(args.fixed_delta, "carla", "fixed_delta", 0.05))
    if sync and fixed_delta <= 0.0:
        fixed_delta = 1.0 / 30.0
        logging.warning("Invalid fixed_delta in sync mode. Using fixed_delta=%.5f", fixed_delta)

    no_rendering = _to_bool(
        args.no_rendering,
        _to_bool(_cfg_get(env_cfg, "carla", "no_rendering", False), False),
    )

    lock_spectator = _to_bool(
        args.lock_spectator_on_spawn,
        _to_bool(_cfg_get(env_cfg, "spectator", "lock_on_spawn", True), True),
    )
    spectator_reapply_each_tick = _to_bool(
        args.spectator_reapply_each_tick,
        _to_bool(_cfg_get(env_cfg, "spectator", "keep_reapply_each_tick", False), False),
    )
    spectator_follow_distance = float(
        pick(args.spectator_follow_distance, "spectator", "follow_distance", 9.0)
    )
    spectator_height = float(pick(args.spectator_height, "spectator", "height", 4.5))
    spectator_pitch = float(pick(args.spectator_pitch, "spectator", "pitch", -18.0))

    camera_width = int(pick(args.camera_width, "camera", "width", 800))
    camera_height = int(pick(args.camera_height, "camera", "height", 600))
    camera_fov = float(pick(args.camera_fov, "camera", "fov", 90.0))

    collect_data = _to_bool(
        args.collect_data,
        _to_bool(_cfg_get(env_cfg, "data_collection", "enabled", False), False),
    )
    collect_data_dir = str(pick(args.collect_data_dir, "data_collection", "output_dir", "data/collected"))
    save_every_n = max(1, int(pick(args.save_every_n, "data_collection", "save_every_n", 50)))
    image_prefix = str(pick(args.image_prefix, "data_collection", "image_prefix", ""))

    if collect_data and not sync:
        logging.warning("Data collection requires synchronous mode. Forcing sync=True.")
        sync = True
    if collect_data and fixed_delta <= 0.0:
        fixed_delta = 1.0 / 30.0
        logging.warning("Invalid fixed_delta for data collection. Using fixed_delta=%.5f.", fixed_delta)

    video_enabled = _to_bool(
        args.record_video,
        _to_bool(_cfg_get(env_cfg, "recording", "enabled", False), False),
    )
    video_output_path = str(
        pick(args.video_output_path, "recording", "output_path", "outputs/town_drive_10m.mp4")
    )
    video_fps = float(pick(args.video_fps, "recording", "fps", 30.0))
    video_duration_sec = int(pick(args.video_duration_sec, "recording", "duration_sec", 600))
    video_codec = str(pick(args.video_codec, "recording", "codec", "mp4v"))

    npc_vehicle_count = max(0, int(pick(args.npc_vehicle_count, "traffic_spawn", "vehicle_count", 30)))
    npc_bike_count = max(0, int(pick(args.npc_bike_count, "traffic_spawn", "bike_count", 10)))
    npc_motorbike_count = max(0, int(pick(args.npc_motorbike_count, "traffic_spawn", "motorbike_count", 10)))
    npc_pedestrian_count = max(0, int(pick(args.npc_pedestrian_count, "traffic_spawn", "pedestrian_count", 50)))
    npc_enable_autopilot = _to_bool(
        _cfg_get(env_cfg, "traffic_spawn", "npc_enable_autopilot", True),
        True,
    )
    if args.disable_npc_autopilot:
        npc_enable_autopilot = False

    runtime_ticks_cfg = _cfg_get(env_cfg, "runtime", "ticks", None)
    if args.ticks is not None:
        ticks = int(args.ticks)
    elif runtime_ticks_cfg is not None:
        ticks = int(runtime_ticks_cfg)
    else:
        ticks = 1000

    tick_interval = float(pick(args.tick_interval, "runtime", "tick_interval", 0.05))

    if video_enabled and args.ticks is None and runtime_ticks_cfg is None:
        if sync:
            required_ticks = max(1, int(video_duration_sec / max(fixed_delta, 1e-3)))
        else:
            required_ticks = max(1, int(video_duration_sec * max(video_fps, 1e-3)))
        ticks = required_ticks
        logging.info("Auto-set ticks=%d from video duration=%ds.", ticks, video_duration_sec)

    spawn_point_cfg = int(pick(args.spawn_point, "vehicle", "spawn_point", -1))
    destination_point_cfg = int(pick(args.destination_point, "vehicle", "destination_point", -1))

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

    seed = args.seed if args.seed is not None else _cfg_get(env_cfg, "runtime", "seed", None)
    if seed is not None:
        seed = int(seed)

    target_speed_kmh = float(pick(args.target_speed_kmh, "cil", "target_speed_kmh", 30.0))
    max_throttle = float(pick(args.max_throttle, "cil", "max_throttle", 0.20))
    max_brake = float(pick(args.max_brake, "cil", "max_brake", 0.60))
    steer_smoothing = float(pick(args.steer_smoothing, "cil", "steer_smoothing", 0.35))

    cil_command_prep_time_s = float(
        _cfg_get(env_cfg, "cil", "command_prep_time_s", CILAgent.COMMAND_PREP_TIME_S)
    )
    cil_command_trigger_min_m = float(
        _cfg_get(env_cfg, "cil", "command_trigger_min_m", CILAgent.COMMAND_TRIGGER_MIN_M)
    )
    cil_command_trigger_max_m = float(
        _cfg_get(env_cfg, "cil", "command_trigger_max_m", CILAgent.COMMAND_TRIGGER_MAX_M)
    )
    cil_command_trigger_min_m = max(3.0, cil_command_trigger_min_m)
    cil_command_trigger_max_m = max(cil_command_trigger_min_m + 1.0, cil_command_trigger_max_m)
    cil_command_prep_time_s = max(0.8, cil_command_prep_time_s)

    nav_agent_type = str(pick(args.nav_agent_type, "cil", "nav_agent_type", "basic")).lower()
    if nav_agent_type not in {"basic", "behavior"}:
        logging.warning("Unsupported nav_agent_type=%s. Falling back to 'basic'.", nav_agent_type)
        nav_agent_type = "basic"

    yolo_disable_autopilot_red_light = _to_bool(
        args.yolo_disable_autopilot_red_light,
        _to_bool(_cfg_get(env_cfg, "yolo", "disable_autopilot_red_light", False), False),
    )

    weather_preset = str(_cfg_get(env_cfg, "weather", "preset", "ClearNoon"))
    yaml_random_weather = _to_bool(_cfg_get(env_cfg, "weather", "random", False), False)
    if yaml_random_weather and not _to_bool(args.no_random_weather, False):
        logging.warning(
            "weather.random=true is ignored for deterministic testing. Using fixed weather preset '%s'.",
            weather_preset,
        )

    recovery_interval_frames = max(
        1,
        int(pick(args.recovery_interval_frames, "cil", "recovery_interval_frames", 100)),
    )
    recovery_duration_frames = max(
        1,
        int(pick(args.recovery_duration_frames, "cil", "recovery_duration_frames", 10)),
    )
    recovery_steer_offset = abs(
        float(pick(args.recovery_steer_offset, "cil", "recovery_steer_offset", 0.3))
    )

    fps_log_interval_ticks = max(
        1,
        int(_cfg_get(env_cfg, "runtime", "fps_log_interval_ticks", 30)),
    )
    cil_enable_hud = _to_bool(_cfg_get(env_cfg, "runtime", "cil_enable_hud", False), False)
    cil_enable_route_map = _to_bool(_cfg_get(env_cfg, "runtime", "cil_enable_route_map", False), False)
    cil_enable_telemetry_csv = _to_bool(
        _cfg_get(env_cfg, "runtime", "cil_enable_telemetry_csv", False),
        False,
    )
    cil_profile_tick_timing = _to_bool(
        _cfg_get(env_cfg, "runtime", "cil_profile_tick_timing", True),
        True,
    )
    cil_profile_log_interval_ticks = max(
        1,
        int(_cfg_get(env_cfg, "runtime", "cil_profile_log_interval_ticks", 60)),
    )

    return RunConfig(
        env_config_path=args.config,
        host=host,
        port=port,
        tm_port=tm_port,
        timeout=timeout,
        sync=sync,
        fixed_delta=fixed_delta,
        no_rendering=no_rendering,
        map_name=map_name,
        vehicle_filter=vehicle_filter,
        spawn_point=spawn_point_cfg,
        destination_point=destination_point_cfg,
        ticks=ticks,
        tick_interval=tick_interval,
        dry_run=args.dry_run,
        seed=seed,
        model_path=args.model_path,
        cil_model_path=args.cil_model_path,
        yolo_model_path=args.yolo_model_path,
        model_device=args.device,
        target_speed_kmh=target_speed_kmh,
        max_throttle=max_throttle,
        max_brake=max_brake,
        steer_smoothing=steer_smoothing,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fov=camera_fov,
        lock_spectator_on_spawn=lock_spectator,
        spectator_reapply_each_tick=spectator_reapply_each_tick,
        spectator_follow_distance=spectator_follow_distance,
        spectator_height=spectator_height,
        spectator_pitch=spectator_pitch,
        collect_data=collect_data,
        collect_data_dir=collect_data_dir,
        save_every_n=save_every_n,
        image_prefix=image_prefix,
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
        random_weather=False,
        weather_preset=weather_preset,
        fps_log_interval_ticks=fps_log_interval_ticks,
        cil_enable_hud=cil_enable_hud,
        cil_enable_route_map=cil_enable_route_map,
        cil_enable_telemetry_csv=cil_enable_telemetry_csv,
        cil_profile_tick_timing=cil_profile_tick_timing,
        cil_profile_log_interval_ticks=cil_profile_log_interval_ticks,
        recovery_interval_frames=recovery_interval_frames,
        recovery_duration_frames=recovery_duration_frames,
        recovery_steer_offset=recovery_steer_offset,
        nav_agent_type=nav_agent_type,
        yolo_disable_autopilot_red_light=bool(yolo_disable_autopilot_red_light),
        cil_command_prep_time_s=cil_command_prep_time_s,
        cil_command_trigger_min_m=cil_command_trigger_min_m,
        cil_command_trigger_max_m=cil_command_trigger_max_m,
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
    fps_profiler = TickFpsProfiler(
        sync=config.sync,
        fixed_delta=config.fixed_delta,
        log_interval_ticks=config.fps_log_interval_ticks,
    )

    try:
        session.start()
        if not config.dry_run and session.world is not None:
            apply_weather_preset(session.world, config.weather_preset)
        agent.setup(session)
        step = 0
        while tick_limit <= 0 or step < tick_limit:
            step += 1

            tick_t0 = time.perf_counter()
            session.tick()
            session_dt = time.perf_counter() - tick_t0

            agent_t0 = time.perf_counter()
            agent.run_step(step)
            agent_dt = time.perf_counter() - agent_t0

            total_dt = time.perf_counter() - tick_t0
            fps_profiler.record(
                step_idx=step,
                session_s=session_dt,
                agent_s=agent_dt,
                total_s=total_dt,
            )

            if agent.should_stop():
                logging.info("Agent requested stop at tick %d.", step)
                break
        fps_profiler.flush(step)
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
