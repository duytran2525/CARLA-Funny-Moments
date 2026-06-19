from __future__ import annotations

import argparse
import csv
import inspect
import importlib
import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

try:
    import carla
except ImportError:
    carla = None

try:
    from agents.navigation.basic_agent import BasicAgent
except Exception:
    BasicAgent = None

try:
    from agents.navigation.behavior_agent import BehaviorAgent
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
    from core_perception.cnn_model import (
        CIL_NvidiaCNN,
        ConditionalSteeringCNN,
        NvidiaCNN,
        NvidiaCNNV2,
        classify_checkpoint_state_dict,
        unwrap_state_dict,
    )
except Exception:
    CIL_NvidiaCNN = None
    ConditionalSteeringCNN = None
    NvidiaCNN = None
    NvidiaCNNV2 = None
    classify_checkpoint_state_dict = None
    unwrap_state_dict = None

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

try:
    from core_control.gtnet_supervisor import GTNetSupervisor, GTNetSupervisorConfig
except Exception as exc:
    logging.warning("Failed to import GTNetSupervisor: %s", exc)
    GTNetSupervisor = None
    GTNetSupervisorConfig = None

try:
    from core_control.pure_pursuit import PurePursuitController
except Exception as exc:
    logging.warning("Failed to import PurePursuitController: %s", exc)
    PurePursuitController = None

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


DETECTOR_DISPLAY_CLASSES = (
    "vehicle",
    "two_wheeler",
    "traffic_light_red",
    "traffic_sign",
    "pedestrian",
    "traffic_light_green",
    "stop_line",
)


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


def _control_without_autopilot_brake(control: Any) -> Any:
    """Preserve autopilot steering/throttle but remove any stop/brake command."""
    if control is None or carla is None:
        return control

    sanitized = carla.VehicleControl()
    sanitized.throttle = float(clamp(float(getattr(control, "throttle", 0.0)), 0.0, 1.0))
    sanitized.steer = float(clamp(float(getattr(control, "steer", 0.0)), -1.0, 1.0))
    sanitized.brake = 0.0
    sanitized.hand_brake = False
    sanitized.reverse = bool(getattr(control, "reverse", False))
    sanitized.manual_gear_shift = bool(getattr(control, "manual_gear_shift", False))
    sanitized.gear = int(getattr(control, "gear", 0))
    return sanitized


def _configure_navigation_agent_ignore_stop_rules(nav_agent: Any) -> tuple[bool, bool]:
    """Disable planner-managed stop rules so autopilot cannot decide braking."""
    if nav_agent is None:
        return False, False

    def set_ignore(method_name: str, attr_name: str) -> bool:
        configured = False
        ignore_fn = getattr(nav_agent, method_name, None)
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

        if not configured and hasattr(nav_agent, attr_name):
            try:
                setattr(nav_agent, attr_name, True)
                configured = True
            except Exception:
                configured = False
        return configured

    lights_configured = set_ignore("ignore_traffic_lights", "_ignore_traffic_lights")
    signs_configured = set_ignore("ignore_stop_signs", "_ignore_stop_signs")
    return lights_configured, signs_configured


def decode_carla_depth_to_meters(image) -> Any:
    """Decode CARLA depth camera BGRA buffer into meters."""
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((image.height, image.width, 4)).astype(np.float32)
    normalized = (
        bgra[:, :, 2] + bgra[:, :, 1] * 256.0 + bgra[:, :, 0] * 65536.0
    ) / 16777215.0
    return normalized * 1000.0


def build_supervisor_config() -> Dict[str, Any]:
    """Default TrafficSupervisor configuration (shared by all agents)."""
    return {
        "confidence_threshold": 0.5,
        "temporal_filter_frames": 3,
        "red_light_distance_threshold": 30.0,
        "red_stopline_trigger_distance_m": 7.0,
        "red_stopline_approach_start_distance_m": 18.0,
        "red_stopline_approach_min_brake": 0.08,
        "red_stopline_approach_floor_brake_near": 0.35,
        "red_stopline_approach_max_brake": 0.95,
        "red_stopline_vehicle_max_decel_mps2": 8.0,
        "red_hard_stop_min_brake": 1.0,
        "red_hard_stop_hold_seconds": 1.5,
        "green_immunity_frames": 10,
        "green_immunity_red_override_stopline_m": 18.0,
        "stop_line_crawl_start_distance_m": 18.0,
        "stop_line_crawl_end_distance_m": 7.0,
        "stop_line_crawl_max_brake": 0.45,
        "stop_line_crawl_red_min_brake": 0.08,
        "stop_line_crawl_red_max_brake": 0.50,
        "stop_line_crawl_target_speed_kmh": 12.0,
        "stop_line_crawl_preview_brake": 0.03,
        "stop_line_crawl_release_distance_m": 0.3,
        "stop_line_tracking_max_missing_seconds": 3.5,
        "obstacle_distance_threshold": 5.0,
        "max_stopped_time": 30.0,
    }


def to_supervisor_detections(detections: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Convert YOLO detection dicts into TrafficSupervisor input format."""
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


    danger_polygon = supervisor_debug.get("danger_polygon")
    if danger_polygon is None or len(danger_polygon) < 3:

        obstacle_roi = debug_info.get("obstacle_danger_roi", {})
        danger_polygon = obstacle_roi.get("polygon", [])
        if not danger_polygon or len(danger_polygon) < 3:
            return False


    try:
        points = np.array(danger_polygon, dtype=np.int32).reshape((-1, 1, 2))


        corridor_color = (0, 255, 255)


        overlay = frame_bgr.copy()
        cv2.fillPoly(overlay, [points], (30, 200, 255))
        cv2.addWeighted(overlay, 0.25, frame_bgr, 0.75, 0.0, frame_bgr)


        cv2.polylines(frame_bgr, [points], True, corridor_color, 3)


        center_color = (255, 255, 255)

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

def _draw_red_light_zone_rois(
    frame_bgr: Any,
    supervisor_debug: Dict[str, Any],
    *,
    visible: bool = True,
) -> bool:
    """Draw static red-light ROI bands used by TrafficSupervisor."""
    if not visible:
        return False
    if np is None or cv2 is None:
        return False
    if frame_bgr is None or getattr(frame_bgr, "shape", None) is None:
        return False

    frame_h, frame_w = frame_bgr.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return False


    urban_x1 = int(round(0.35 * frame_w))
    urban_x2 = int(round(0.65 * frame_w))
    rural_x1 = int(round(0.65 * frame_w))
    rural_x2 = int(round(0.95 * frame_w))
    locked_zone = str(supervisor_debug.get("locked_zone") or "")
    roi_y1 = 0
    roi_y2 = int(round(0.60 * frame_h))
    roi_y2 = max(1, min(frame_h - 1, roi_y2))

    try:
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (urban_x1, roi_y1), (urban_x2, roi_y2), (60, 140, 255), -1)
        cv2.rectangle(overlay, (rural_x1, roi_y1), (rural_x2, roi_y2), (60, 255, 120), -1)
        cv2.addWeighted(overlay, 0.08, frame_bgr, 0.92, 0.0, frame_bgr)

        urban_color = (0, 165, 255) if locked_zone == "urban" else (100, 130, 160)
        rural_color = (0, 220, 0) if locked_zone == "rural_right" else (100, 130, 160)
        cv2.rectangle(frame_bgr, (urban_x1, roi_y1), (urban_x2, roi_y2), urban_color, 2)
        cv2.rectangle(frame_bgr, (rural_x1, roi_y1), (rural_x2, roi_y2), rural_color, 2)

        cv2.putText(
            frame_bgr,
            "TL ROI urban [0.35W-0.65W, top60%H]",
            (urban_x1 + 4, max(18, int(0.06 * frame_h))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            urban_color,
            1,
        )
        cv2.putText(
            frame_bgr,
            "TL ROI rural_right [0.65W-0.95W, top60%H]",
            (rural_x1 + 4, max(36, int(0.11 * frame_h))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            rural_color,
            1,
        )
    except Exception as exc:
        logging.warning("Failed to draw red-light ROI zones: %s", exc)
        return False

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


    setter(destination_location)


BALANCED_ROUTE_COMMANDS: tuple[int, int, int] = (1, 2, 3)
BALANCED_ROUTE_CANDIDATE_LIMIT = 12
BALANCED_ROUTE_MIN_DISTANCE_M = 80.0


def summarize_reference_route_commands(route_plan: Any) -> Dict[int, int]:
    counts = {command: 0 for command in BALANCED_ROUTE_COMMANDS}
    if not route_plan:
        return counts

    for item in route_plan:
        if not isinstance(item, dict):
            continue
        command = int(item.get("command", 0))
        if command in counts:
            counts[command] += 1
    return counts


def score_reference_route_balance(
    route_command_counts: Dict[int, int],
    historical_command_counts: Dict[int, int],
    distance_m: float,
) -> float:
    max_history = max((int(historical_command_counts.get(cmd, 0)) for cmd in BALANCED_ROUTE_COMMANDS), default=0)
    denom = max(1.0, float(max_history))

    score = 0.0
    present_types = 0
    total_turn_events = 0
    for command in BALANCED_ROUTE_COMMANDS:
        count = max(0, int(route_command_counts.get(command, 0)))
        if count <= 0:
            continue
        present_types += 1
        total_turn_events += count
        balance_weight = 1.0 + 2.0 * (float(max_history) - float(historical_command_counts.get(command, 0))) / denom
        score += balance_weight * (1.25 + 0.75 * min(count, 2))

    if total_turn_events <= 0:
        score -= 1.25
    else:
        score += 0.35 * min(total_turn_events, 4)
        score += 0.30 * present_types

    score += 0.08 * math.log1p(max(0.0, float(distance_m)))
    score += random.random() * 0.05
    return float(score)


def apply_random_weather(world) -> str:

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

        "clearnight": getattr(carla.WeatherParameters, "ClearNight", carla.WeatherParameters.ClearNoon),
        "cloudynight": getattr(carla.WeatherParameters, "CloudyNight", carla.WeatherParameters.CloudyNoon),
        "wetnight": getattr(carla.WeatherParameters, "WetNight", carla.WeatherParameters.WetNoon),
        "wetcloudynight": getattr(carla.WeatherParameters, "WetCloudyNight", carla.WeatherParameters.WetCloudyNoon),
        "softrainnight": getattr(carla.WeatherParameters, "SoftRainNight", carla.WeatherParameters.SoftRainNoon),
        "midrainnight": getattr(carla.WeatherParameters, "MidRainyNight", carla.WeatherParameters.MidRainyNoon),
        "hardrainnight": getattr(carla.WeatherParameters, "HardRainNight", carla.WeatherParameters.HardRainNoon),
        "duststorm": getattr(carla.WeatherParameters, "DustStorm", carla.WeatherParameters.ClearNoon),

        "foggynoon": carla.WeatherParameters(
            cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=10.0, sun_azimuth_angle=0.0, sun_altitude_angle=45.0,
            fog_density=80.0, fog_distance=5.0, fog_falloff=0.2, wetness=0.0
        ),
        "foggynight": carla.WeatherParameters(
            cloudiness=65.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=6.0, sun_azimuth_angle=0.0, sun_altitude_angle=-18.0,
            fog_density=45.0, fog_distance=35.0, fog_falloff=0.6, wetness=0.0
        )
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

    def is_steering_candidate(path: Path) -> bool:
        name = path.name.lower()
        return "waypoint" not in name and "cil" not in name

    preferred = newest("cnn_steering TL=*.pth")
    if preferred:
        return preferred[0]

    steering_models = newest("cnn_steering*.pth")
    steering_models = [path for path in steering_models if is_steering_candidate(path)]
    if steering_models:
        return steering_models[0]

    any_pth = [path for path in newest("*.pth") if is_steering_candidate(path)]
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
        return (project_root / "models" / "waypoint_predictor.pth").resolve()

    def newest(pattern: str) -> list[Path]:
        return sorted(
            models_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def is_waypoint_candidate(path: Path) -> bool:
        name = path.name.lower()
        return "waypoint" in name or "cil" in name

    preferred = newest("waypoint_predictor*.pth")
    preferred = [path for path in preferred if is_waypoint_candidate(path)]
    if preferred:
        return preferred[0]

    preferred = newest("waypoint*.pth")
    preferred = [path for path in preferred if is_waypoint_candidate(path)]
    if preferred:
        return preferred[0]

    preferred = newest("*waypoint*.pth")
    preferred = [path for path in preferred if is_waypoint_candidate(path)]
    if preferred:
        return preferred[0]

    preferred = newest("cil*.pth")
    preferred = [path for path in preferred if is_waypoint_candidate(path)]
    if preferred:
        return preferred[0]

    preferred = newest("*cil*.pth")
    preferred = [path for path in preferred if is_waypoint_candidate(path)]
    if preferred:
        return preferred[0]

    any_pth = [path for path in newest("*.pth") if is_waypoint_candidate(path)]
    if any_pth:
        return any_pth[0]

    return (models_dir / "waypoint_predictor.pth").resolve()


def resolve_yolo_model_path(model_path: str) -> Path:
    project_root = Path(__file__).resolve().parent
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


@dataclass
class RunConfig:
    env_config_path: str
    host: str
    port: int
    tm_port: int
    timeout: float
    sync: bool
    fixed_delta: float
    no_rendering: bool
    map_name: str
    vehicle_filter: str
    spawn_point: int
    destination_point: int
    ticks: int
    tick_interval: float
    dry_run: bool
    seed: Optional[int]
    model_path: str
    cil_model_path: str
    yolo_model_path: str
    model_device: str
    target_speed_kmh: float
    max_throttle: float
    max_brake: float
    steer_smoothing: float
    camera_width: int
    camera_height: int
    camera_fov: float
    eval_online: bool
    lock_spectator_on_spawn: bool
    spectator_reapply_each_tick: bool
    spectator_follow_distance: float
    spectator_height: float
    spectator_pitch: float
    collect_data: bool
    collect_data_dir: str
    save_every_n: int
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
    weather_preset: str
    fps_log_interval_ticks: int
    cil_enable_hud: bool
    cil_enable_route_map: bool
    cil_opencv_route_map: bool
    cil_world_waypoint_debug: bool
    cil_enable_telemetry_csv: bool
    cil_profile_tick_timing: bool
    cil_profile_log_interval_ticks: int
    recovery_interval_frames: int
    recovery_duration_frames: int
    recovery_steer_offset: float
    autopilot_backend: str
    nav_agent_type: str
    yolo_backend: str
    yolo_nav_agent_type: str
    yolo_disable_autopilot_red_light: bool
    yolo_inference_every_n_ticks: int
    yolo_visualize: bool
    yolo_draw_overlay: bool
    yolo_show_red_light_rois: bool
    yolo_inference_imgsz: int
    yolo_tracker_config: str
    yolo_secondary_tracker_config: str
    gtnet_enabled: bool
    gtnet_model_path: str
    gtnet_inference_every_n_ticks: int
    gtnet_draw_debug: bool
    gtnet_history_frames: int
    gtnet_expected_dt: float
    gtnet_adjacency_mode: str
    gtnet_fixed_adjacency_radius_m: float
    gtnet_max_actor_distance_m: float
    cil_command_prep_time_s: float
    cil_command_trigger_min_m: float
    cil_command_trigger_max_m: float
    cil_use_pure_pursuit: bool
    cil_use_carla_waypoints: bool
    cil_lane_constrain_blended: bool
    cil_lane_constrain_strength: float
    traffic_light_red_time: float
    traffic_light_green_time: float
    traffic_light_yellow_time: float

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

    def had_collision_at(self, frame_id: int) -> bool:
        _ = frame_id
        return False


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

    def had_collision_at(self, frame_id: int) -> bool:
        if self._manager is None:
            return False
        return bool(self._manager.had_collision_at(frame_id))

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
            seed=self.config.seed,
            traffic_light_red_time=self.config.traffic_light_red_time,
            traffic_light_green_time=self.config.traffic_light_green_time,
            traffic_light_yellow_time=self.config.traffic_light_yellow_time,
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

    def _resolve_frame_context(self, step_idx: int) -> tuple[int, float]:
        world = self.session.world if self.session is not None else None
        if world is None:
            return int(step_idx), float(step_idx) * float(self.config.fixed_delta)

        snapshot = world.get_snapshot()
        return int(snapshot.frame), float(snapshot.timestamp.elapsed_seconds)

    def _had_collision_at(self, frame_id: int) -> bool:
        if self.session is None:
            return False
        return bool(self.session.had_collision_at(frame_id))

    def _is_vehicle_at_junction(self, vehicle: Any) -> bool:
        world = self.session.world if self.session is not None else None
        if world is None or vehicle is None:
            return False
        try:
            waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
        except Exception:
            return False
        return bool(getattr(waypoint, "is_junction", False))


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
        self._command_nav_agent = None
        self._spawn_points = []
        self._reference_route_plan: list[Dict[str, Any]] = []
        self._route_destination_index: Optional[int] = None
        self._route_destination_location = None
        self._fixed_destination_consumed = False
        self._route_balance_command_counts: Dict[int, int] = {
            command: 0 for command in BALANCED_ROUTE_COMMANDS
        }
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
        self._route_destination_index = None
        self._route_destination_location = None
        self._fixed_destination_consumed = False
        self._route_balance_command_counts = {
            command: 0 for command in BALANCED_ROUTE_COMMANDS
        }
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

        if self.config.autopilot_backend == "tm":
            self._nav_agent = None
            self._tm_fallback_mode = True
            self._init_tm_command_planner(world, vehicle)
            self._enable_tm_native_autopilot(vehicle)
            logging.info(
                "Autopilot backend initialized: tm native control with shadow route planner for collector commands."
            )
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
        except Exception:
            self._nav_agent = None
            self._tm_fallback_mode = True
            self._init_tm_command_planner(world, vehicle)
            self._enable_tm_native_autopilot(vehicle)
            logging.info(
                "Using TM autopilot fallback with shadow route planner for collector commands. Reason: navigation agent unavailable."
            )
            return

        self._tm_fallback_mode = False
        self._set_new_destination(vehicle)
        logging.info("Navigation agent initialized: %s", self.config.nav_agent_type)

    def _init_tm_command_planner(self, world, vehicle) -> None:
        self._command_nav_agent = None
        self._reference_route_plan = []
        self._route_destination_index = None
        self._route_destination_location = None

        try:
            ensure_navigation_agent_imports()
            if BasicAgent is None:
                raise RuntimeError("BasicAgent missing")
            self._command_nav_agent = BasicAgent(vehicle, target_speed=max(10.0, self.config.target_speed_kmh))
        except Exception as exc:
            logging.warning("TM command shadow planner unavailable; collector command will remain 0. Reason: %s", exc)
            return

        try:
            self._set_new_destination(vehicle)
            logging.info("TM command shadow planner initialized for collector route labels.")
        except Exception as exc:
            self._command_nav_agent = None
            self._reference_route_plan = []
            logging.warning("Failed to initialize TM command route labels; collector command will remain 0. Reason: %s", exc)

    def _enable_tm_native_autopilot(self, vehicle) -> None:
        try:
            vehicle.set_autopilot(True, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(True)

        self._configure_tm_traffic_lights(vehicle)

        tm = getattr(self.session, "traffic_manager", None) if self.session is not None else None
        if tm is None:
            logging.warning("TrafficManager unavailable; native TM autopilot speed tuning skipped.")
            return

        try:
            tm.auto_lane_change(vehicle, False)
        except Exception:
            pass

        try:
            tm.distance_to_leading_vehicle(vehicle, 3.0)
        except Exception:
            pass

        self._configure_tm_target_speed(vehicle, log=True)

    def _configure_tm_target_speed(self, vehicle, log: bool = False) -> None:
        tm = getattr(self.session, "traffic_manager", None) if self.session is not None else None
        if tm is None:
            if log:
                logging.warning("TrafficManager unavailable; native TM autopilot speed tuning skipped.")
            return

        try:
            speed_limit_kmh = float(vehicle.get_speed_limit())
        except Exception:
            speed_limit_kmh = 0.0

        target_speed_kmh = max(1.0, float(self.config.target_speed_kmh))
        if speed_limit_kmh > 1.0:
            speed_delta_pct = 100.0 * (1.0 - (target_speed_kmh / speed_limit_kmh))
            speed_delta_pct = clamp(speed_delta_pct, -75.0, 95.0)
            try:
                tm.vehicle_percentage_speed_difference(vehicle, float(speed_delta_pct))
                if log:
                    logging.info(
                        "Configured TM native autopilot target speed %.1f km/h against limit %.1f km/h (speed_delta_pct=%.1f).",
                        target_speed_kmh,
                        speed_limit_kmh,
                        float(speed_delta_pct),
                    )
            except Exception as exc:
                logging.warning("Failed to tune TM native autopilot speed: %s", exc)
        elif log:
            logging.info("TM native autopilot enabled without speed-limit tuning (unknown speed limit).")

    def _configure_nav_agent_traffic_lights(self) -> None:
        if self._nav_agent is None:
            return

        lights_configured, signs_configured = _configure_navigation_agent_ignore_stop_rules(self._nav_agent)
        if lights_configured or signs_configured:
            logging.info(
                "Autopilot planner stop rules disabled (traffic_lights=%s, stop_signs=%s); brake commands are suppressed.",
                lights_configured,
                signs_configured,
            )
        else:
            logging.warning("Could not disable planner stop rules for autopilot; brake commands will still be suppressed.")

    def _configure_tm_traffic_lights(self, vehicle) -> None:
        if self.session is None:
            return

        tm = getattr(self.session, "traffic_manager", None)
        if tm is None:
            logging.warning("TrafficManager unavailable; cannot disable TM stop rules for autopilot.")
            return

        try:
            tm.ignore_lights_percentage(vehicle, 100.0)
            logging.info("Autopilot TM configured to ignore all traffic lights (100%%).")
        except Exception as exc:
            logging.warning("Failed to disable TM traffic-light stops for autopilot: %s", exc)
        ignore_signs_fn = getattr(tm, "ignore_signs_percentage", None)
        if callable(ignore_signs_fn):
            try:
                ignore_signs_fn(vehicle, 100.0)
                logging.info("Autopilot TM configured to ignore all stop signs (100%%).")
            except Exception as exc:
                logging.warning("Failed to disable TM stop-sign stops for autopilot: %s", exc)

    def _set_new_destination(self, vehicle) -> None:
        nav_agent = self._nav_agent if self._nav_agent is not None else self._command_nav_agent
        if not self._spawn_points or nav_agent is None:
            return
        current_loc = vehicle.get_location()
        world = self.session.world if self.session is not None else None
        world_map = world.get_map() if world is not None else None
        current_idx = self._nearest_spawn_index(current_loc)

        route_plan: list[Dict[str, Any]] = []
        route_command_counts = {command: 0 for command in BALANCED_ROUTE_COMMANDS}
        if self.config.collect_data:
            dest_idx, destination, route_plan, route_command_counts = self._choose_balanced_destination(
                current_loc=current_loc,
                current_idx=current_idx,
                world_map=world_map,
            )
        elif self.config.destination_point >= 0 and not self._fixed_destination_consumed:
            dest_idx = self.config.destination_point % len(self._spawn_points)
            destination = self._spawn_points[dest_idx].location
            if world_map is not None and current_loc is not None:
                route_plan = build_global_reference_route(
                    world_map=world_map,
                    start_location=current_loc,
                    destination_location=destination,
                )
                route_command_counts = summarize_reference_route_commands(route_plan)
        else:
            candidate_indices = [
                idx for idx in range(len(self._spawn_points))
                if current_idx is None or idx != current_idx
            ]
            if not candidate_indices:
                candidate_indices = list(range(len(self._spawn_points)))
            dest_idx = int(random.choice(candidate_indices))
            destination = self._spawn_points[dest_idx].location

        current_loc = vehicle.get_location()
        self._route_destination_index = int(dest_idx)
        self._route_destination_location = destination
        set_navigation_destination(nav_agent, current_loc, destination)
        if route_plan:
            self._reference_route_plan = list(route_plan)
        else:
            self._cache_reference_route_plan(force=True)
            route_plan = list(self._reference_route_plan)
            route_command_counts = summarize_reference_route_commands(route_plan)
        self._accumulate_route_balance(route_command_counts)
        if self.config.collect_data:
            logging.info(
                (
                    "Autopilot collect-data roaming destination set: spawn=%d "
                    "cmd_mix(L/R/S)=%d/%d/%d hist=%d/%d/%d"
                ),
                int(dest_idx),
                int(route_command_counts.get(1, 0)),
                int(route_command_counts.get(2, 0)),
                int(route_command_counts.get(3, 0)),
                int(self._route_balance_command_counts.get(1, 0)),
                int(self._route_balance_command_counts.get(2, 0)),
                int(self._route_balance_command_counts.get(3, 0)),
            )
        else:
            logging.info(
                "Autopilot destination set: spawn=%d cmd_mix(L/R/S)=%d/%d/%d hist=%d/%d/%d",
                int(dest_idx),
                int(route_command_counts.get(1, 0)),
                int(route_command_counts.get(2, 0)),
                int(route_command_counts.get(3, 0)),
                int(self._route_balance_command_counts.get(1, 0)),
                int(self._route_balance_command_counts.get(2, 0)),
                int(self._route_balance_command_counts.get(3, 0)),
            )
        self._command_oracle.reset()

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

    def _choose_balanced_destination(self, current_loc, current_idx: Optional[int], world_map: Any):
        if not self._spawn_points:
            raise RuntimeError("No spawn points available for autopilot destination selection.")

        candidate_indices: list[int] = []
        min_distance_m = float(BALANCED_ROUTE_MIN_DISTANCE_M)
        for idx, transform in enumerate(self._spawn_points):
            if current_idx is not None and idx == current_idx:
                continue
            if self._route_destination_index is not None and idx == self._route_destination_index:
                continue
            if current_loc is not None and current_loc.distance(transform.location) < min_distance_m:
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            for idx, _transform in enumerate(self._spawn_points):
                if current_idx is not None and idx == current_idx:
                    continue
                if self._route_destination_index is not None and idx == self._route_destination_index:
                    continue
                candidate_indices.append(idx)

        if not candidate_indices:
            fallback_idx = int((current_idx or 0) % len(self._spawn_points))
            fallback_destination = self._spawn_points[fallback_idx].location
            return fallback_idx, fallback_destination, [], {command: 0 for command in BALANCED_ROUTE_COMMANDS}

        sample_count = min(len(candidate_indices), BALANCED_ROUTE_CANDIDATE_LIMIT)
        if sample_count < len(candidate_indices):
            candidate_indices = random.sample(candidate_indices, sample_count)

        best_idx = int(candidate_indices[0])
        best_destination = self._spawn_points[best_idx].location
        best_route_plan: list[Dict[str, Any]] = []
        best_route_command_counts = {command: 0 for command in BALANCED_ROUTE_COMMANDS}
        best_score = float("-inf")

        for idx in candidate_indices:
            destination = self._spawn_points[idx].location
            route_plan = []
            if world_map is not None and current_loc is not None:
                route_plan = build_global_reference_route(
                    world_map=world_map,
                    start_location=current_loc,
                    destination_location=destination,
                )
            route_command_counts = summarize_reference_route_commands(route_plan)
            distance_m = (
                float(current_loc.distance(destination))
                if current_loc is not None and destination is not None
                else 0.0
            )
            score = score_reference_route_balance(
                route_command_counts=route_command_counts,
                historical_command_counts=self._route_balance_command_counts,
                distance_m=distance_m,
            )
            if route_plan:
                score += 0.15
            if score > best_score:
                best_idx = int(idx)
                best_destination = destination
                best_route_plan = list(route_plan)
                best_route_command_counts = dict(route_command_counts)
                best_score = float(score)

        return best_idx, best_destination, best_route_plan, best_route_command_counts

    def _accumulate_route_balance(self, route_command_counts: Dict[int, int]) -> None:
        for command in BALANCED_ROUTE_COMMANDS:
            count = max(0, int(route_command_counts.get(command, 0)))
            self._route_balance_command_counts[command] = (
                int(self._route_balance_command_counts.get(command, 0)) + min(count, 2)
            )

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
        nav_agent = self._nav_agent if self._nav_agent is not None else self._command_nav_agent
        if nav_agent is None or not hasattr(nav_agent, "get_local_planner"):
            return None
        try:
            return nav_agent.get_local_planner()
        except Exception:
            return None

    def _get_reference_route_plan(self):
        return list(self._reference_route_plan)

    def _cache_reference_route_plan(self, force: bool = False) -> int:
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
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(pitch=-8.0))),
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
        recovery_delta = 0.0
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
            nav_control = _control_without_autopilot_brake(nav_control)
            vehicle.apply_control(nav_control)
        elif vehicle is not None and self._tm_fallback_mode:
            frame_id_for_control = step_idx
            if world is not None:
                frame_id_for_control = int(world.get_snapshot().frame)
            recovery_delta = self._recovery_offset(frame_id_for_control)
            if frame_id_for_control % 30 == 0:
                self._configure_tm_target_speed(vehicle, log=False)
            if self._command_nav_agent is not None:
                try:
                    if hasattr(self._command_nav_agent, "done") and self._command_nav_agent.done():
                        self._set_new_destination(vehicle)
                    self._command_nav_agent.run_step()
                except Exception:
                    pass
            control = vehicle.get_control()
            tm_brake = float(getattr(control, "brake", 0.0))
            tm_hand_brake = bool(getattr(control, "hand_brake", False))
            if abs(recovery_delta) > 1e-6:
                control.steer = float(clamp(control.steer + recovery_delta, -1.0, 1.0))
            if abs(recovery_delta) > 1e-6 or tm_brake > 1e-6 or tm_hand_brake:
                control = _control_without_autopilot_brake(control)
                vehicle.apply_control(control)

        if self._collector is not None and vehicle is not None:
            velocity = vehicle.get_velocity()
            speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            control = vehicle.get_control()
            transform = vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            frame_id, timestamp = self._resolve_frame_context(step_idx)
            command = self._extract_current_command(speed_kmh=speed_kmh)
            self._collector.add_vehicle_state(
                frame_id=frame_id,
                timestamp=timestamp,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                x=location.x,
                y=location.y,
                z=location.z,
                has_crash=self._had_collision_at(frame_id),
                is_recovering=abs(recovery_delta) > 1e-6,
                is_junction=self._is_vehicle_at_junction(vehicle),
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
        self._command_nav_agent = None
        self._reference_route_plan = []
        self._route_destination_index = None
        self._route_destination_location = None
        self._fixed_destination_consumed = False
        self._route_balance_command_counts = {
            command: 0 for command in BALANCED_ROUTE_COMMANDS
        }
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
    STEERING_SPEED_NORM_KMH = 120.0

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._enabled = False
        self._model = None
        self._model_kind = "unknown"
        self._device = None
        self._camera = None
        self._depth_camera = None
        self._latest_rgb = None
        self._latest_depth_m = None
        self._frame_lock = threading.Lock()
        self._frame_history = deque(maxlen=3)  # FIX: prevent AttributeError in callback
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

        self._yolo_detector = None
        self._traffic_supervisor = None
        self._last_supervisor_ts: Optional[float] = None
        self._yolo_window_name = "Lane Follow + YOLO Detection"
        self._yolo_enabled = False

        self._speed_controller = SpeedPIDController(
            target_speed_kmh=config.target_speed_kmh,
            max_throttle=config.max_throttle,
            max_brake=config.max_brake,
        )

        self._use_pure_pursuit = bool(config.cil_use_pure_pursuit)
        if self._use_pure_pursuit and PurePursuitController is not None and np is not None:
            dt = max(float(config.fixed_delta), 1e-3)
            self._pure_pursuit = PurePursuitController(dt=dt)
        else:
            self._pure_pursuit = None

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
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(pitch=-8.0))),
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
            self._traffic_supervisor = None
            return

        if YoloDetector is None:
            logging.warning("YoloDetector not available. Install ultralytics to enable YOLO detection.")
            self._traffic_supervisor = None
            return

        model_path = resolve_yolo_model_path(self.config.yolo_model_path)
        if not model_path.exists():
            logging.warning("YOLO model file not found: %s. YOLO detection disabled.", model_path)
            self._traffic_supervisor = None
            return

        self._yolo_detector = YoloDetector(
            str(model_path),
            display_classes=DETECTOR_DISPLAY_CLASSES,
            camera_fov_deg=self.config.camera_fov,
            obstacle_base_distance_m=8.0,
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
        )
        if TrafficSupervisor is None:
            self._traffic_supervisor = None
            logging.warning("TrafficSupervisor unavailable. lane_follow will use detector-only emergency fallback.")
        else:
            try:
                self._traffic_supervisor = TrafficSupervisor(build_supervisor_config())
                self._last_supervisor_ts = None
                logging.info("TrafficSupervisor integrated into lane_follow YOLO loop.")
            except Exception as exc:
                self._traffic_supervisor = None
                logging.warning("Failed to initialize TrafficSupervisor for lane_follow: %s", exc)
        self._yolo_enabled = True
        logging.info("YOLO detection integrated with lane_follow. Model: %s", model_path)

    def _load_model(self):
        if torch is None:
            raise RuntimeError("PyTorch is required for lane_follow agent.")
        if cv2 is None:
            raise RuntimeError("opencv-python is required for lane_follow agent.")
        if np is None:
            raise RuntimeError("numpy is required for lane_follow agent.")
        if NvidiaCNN is None or unwrap_state_dict is None or classify_checkpoint_state_dict is None:
            raise RuntimeError("Cannot import lane-follow model definitions from core_perception.cnn_model.")

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

        checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        state_dict = unwrap_state_dict(checkpoint)
        model_kind = classify_checkpoint_state_dict(state_dict)

        if model_kind == "conditional_steering":
            if ConditionalSteeringCNN is None:
                raise RuntimeError("ConditionalSteeringCNN is unavailable for this checkpoint.")
            model = ConditionalSteeringCNN().to(self._device)
        elif model_kind == "steering_v2":
            logging.info("Detected V2 steering architecture for %s", model_path)
            model = NvidiaCNNV2().to(self._device)
        elif model_kind == "steering":
            model = NvidiaCNN().to(self._device)
        else:
            raise RuntimeError(
                f"Incompatible lane-follow checkpoint '{model_path.name}' detected as '{model_kind}'. "
                "Use a steering checkpoint such as cnn_steering*.pth."
            )

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._model_kind = model_kind
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
            self._frame_history.append(rgb)

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

    def _predict_steering(self, rgb_frame, speed_kmh: float = 0.0, command: int = 0) -> float:
        height = rgb_frame.shape[0]
        cropped = rgb_frame[int(height * 0.45) :, :, :]
        resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)


        yuv_image = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        tensor = torch.from_numpy(yuv_image).permute(2, 0, 1).float().div_(255.0)
        tensor.sub_(0.5).div_(0.5)
        tensor.unsqueeze_(0)
        tensor = tensor.to(self._device, non_blocking=True)
        with torch.inference_mode():
            if self._model_kind == "conditional_steering":
                speed_norm = clamp(float(speed_kmh) / self.STEERING_SPEED_NORM_KMH, 0.0, 1.0)
                speed_tensor = torch.tensor([speed_norm], dtype=torch.float32, device=self._device)
                command_tensor = torch.tensor([max(0, min(3, int(command)))], dtype=torch.long, device=self._device)
                steering = self._model(tensor, speed_tensor, command_tensor).item()
            else:
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


        detections, detector_emergency = self._yolo_detector.detect_and_evaluate(
            frame_bgr,
            distance_threshold=None,
            depth_map_m=depth_map_m,
            vehicle_steer=current_steer,
            speed_kmh=speed_kmh,
        )
        debug_info = {}
        if hasattr(self._yolo_detector, "get_last_debug_info"):
            debug_info = self._yolo_detector.get_last_debug_info() or {}

        supervisor_brake = 0.0
        sup_debug: Dict[str, Any] = {}
        if self._traffic_supervisor is not None:
            now_ts = time.time()
            if self._last_supervisor_ts is None:
                dt = self.config.fixed_delta if self.config.sync else (1.0 / 30.0)
            elif self.config.sync:
                dt = self.config.fixed_delta
            else:
                dt = max(1e-3, now_ts - float(self._last_supervisor_ts))
            self._last_supervisor_ts = now_ts

            try:
                sup_dets = to_supervisor_detections(detections)
                supervisor_brake = float(
                    self._traffic_supervisor.compute(
                        detections=sup_dets,
                        current_speed=0.0 if speed_kmh is None else (float(speed_kmh) / 3.6),
                        image_shape=frame_bgr.shape,
                        distance_threshold=None,
                        vehicle_steer=0.0 if current_steer is None else float(current_steer),
                        dt=dt,
                    )
                )
                supervisor_brake = clamp(supervisor_brake, 0.0, 1.0)
                sup_debug = self._traffic_supervisor.get_debug_info()
            except Exception as exc:
                supervisor_brake = 0.0
                sup_debug = {}
                logging.warning("Lane-follow TrafficSupervisor compute failed: %s", exc)

        debug_info["supervisor_brake"] = float(supervisor_brake)
        supervisor_target_type = "none"
        if sup_debug:
            target_type = str(sup_debug.get("selected_target_type", "none"))
            supervisor_target_type = target_type
            debug_info["decision_reason"] = target_type
            debug_info["supervisor_state"] = str(sup_debug.get("state", "n/a"))
            debug_info["locked_zone"] = sup_debug.get("locked_zone")
            debug_info["green_immunity_counter"] = int(sup_debug.get("green_immunity_counter", 0))
            debug_info["red_hard_stop_active"] = bool(sup_debug.get("red_hard_stop_active", False))
            debug_info["red_hard_stop_latch_s"] = float(sup_debug.get("red_hard_stop_latch_s", 0.0))
            debug_info["red_light_active"] = bool(
                supervisor_brake > 0.0 and target_type in ("red_light", "traffic_light_red", "stop_line")
            )
            debug_info["obstacle_emergency"] = bool(supervisor_brake > 0.0 and target_type == "obstacle")
            debug_info["obstacle_reason"] = str(sup_debug.get("obstacle_reason", ""))

            polygon = sup_debug.get("danger_polygon")
            polygon_list = []
            if isinstance(polygon, np.ndarray):
                polygon_list = [[int(p[0]), int(p[1])] for p in polygon.tolist() if isinstance(p, (list, tuple)) and len(p) >= 2]
            elif isinstance(polygon, list):
                polygon_list = [[int(p[0]), int(p[1])] for p in polygon if isinstance(p, (list, tuple)) and len(p) >= 2]
            if polygon_list:
                obstacle_roi = debug_info.get("obstacle_danger_roi", {}) or {}
                obstacle_roi["polygon"] = polygon_list
                debug_info["obstacle_danger_roi"] = obstacle_roi


        hard_supervisor_emergency = (
            supervisor_brake >= 0.60 and supervisor_target_type == "obstacle"
        )
        if self._traffic_supervisor is not None:
            is_emergency = bool(hard_supervisor_emergency)
        else:
            is_emergency = bool(detector_emergency)

        annotated_frame = frame_bgr.copy()
        _draw_red_light_zone_rois(
            annotated_frame,
            sup_debug,
            visible=bool(self.config.yolo_show_red_light_rois),
        )
        _draw_yellow_danger_corridor(annotated_frame, debug_info, sup_debug)

        for det in detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class_name']
            conf = det['confidence']
            distance = det['distance']
            in_danger_roi = bool(det.get("in_danger_roi", False))
            danger_match = bool(det.get("danger_match", False))

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


        is_emergency = False
        supervisor_brake = 0.0
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
            supervisor_brake = float(yolo_debug_info.get("supervisor_brake", 0.0))

        self._write_video_frame(frame)
        if self._stop_requested:
            logging.info("Video duration target reached, stopping agent loop.")
            return

        steering_raw = self._predict_steering(frame, speed_kmh=speed_kmh, command=0)
        alpha = clamp(self.config.steer_smoothing, 0.0, 0.99)
        steering = alpha * self._last_steer + (1.0 - alpha) * steering_raw
        self._last_steer = steering

        throttle, brake = self._longitudinal_control(speed_kmh)
        supervisor_reason = str(yolo_debug_info.get("decision_reason", "none")).strip().lower()
        red_hard_stop_active = bool(yolo_debug_info.get("red_hard_stop_active", False))
        hold_hand_brake = bool(
            red_hard_stop_active
            and supervisor_brake >= 0.99
            and supervisor_reason in ("stop_line", "red_light", "traffic_light_red")
            and float(speed_kmh) <= 2.0
        )


        if is_emergency or supervisor_brake > 0.0:
            throttle = 0.0
            emergency_floor = self.config.max_brake if is_emergency else 0.0
            brake = max(brake, supervisor_brake, emergency_floor)
            if hold_hand_brake:
                brake = max(brake, 1.0)

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(clamp(steering, -1.0, 1.0)),
            brake=float(brake),
            hand_brake=bool(hold_hand_brake),
            reverse=False,
        )
        self.session.ego_vehicle.apply_control(control)
        transform = self.session.ego_vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        yaw_deg = float(rotation.yaw)

        if self._collector is not None:
            frame_id, timestamp = self._resolve_frame_context(step_idx)
            self._collector.add_vehicle_state(
                frame_id=frame_id,
                timestamp=timestamp,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                x=location.x,
                y=location.y,
                z=location.z,
                has_crash=self._had_collision_at(frame_id),
                is_recovering=False,
                is_junction=self._is_vehicle_at_junction(self.session.ego_vehicle),
                command=0,
                pitch=rotation.pitch,
                roll=rotation.roll,
                yaw=rotation.yaw,
            )

        emergency_reason = (
            yolo_debug_info.get("decision_reason", "none")
            if (is_emergency or supervisor_brake > 0.0)
            else "none"
        )
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

        if self._yolo_enabled:
            try:
                cv2.destroyWindow(self._yolo_window_name)
            except Exception:
                pass
            self._yolo_enabled = False
            logging.info("Closed YOLO detection window.")
        self._yolo_detector = None
        self._traffic_supervisor = None
        self._last_supervisor_ts = None

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
        self._frame_history = deque(maxlen=3)
        self._frame_lock = threading.Lock()
        self._waiting_frame_logged = False
        self._cil_yolo_enabled = False
        self._yolo_detector = None
        self._traffic_supervisor = None
        self._last_supervisor_ts: Optional[float] = None
        self._last_yolo_detection_step: Optional[int] = None
        self._cached_yolo_emergency = False
        self._cached_yolo_debug_info: Dict[str, Any] = {}
        self._cached_yolo_annotated_frame = None
        self._gtnet_supervisor = None
        self._last_gtnet_debug_info: Dict[str, Any] = {}
        self._yolo_window_name = "CARLA CIL + YOLO"
        self._collector: Optional[DataCollector] = None
        self._data_cameras = []
        self._video_writer = None
        self._video_output: Optional[Path] = None
        self._video_frames_written = 0
        self._video_max_frames = 0
        self._stop_requested = False
        self._visualizer = None
        self._route_map = None
        self._opencv_yolo_visible = bool(config.yolo_visualize)
        self._opencv_route_visible = bool(config.cil_opencv_route_map)
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

        self._hud_ema_fps: Optional[float] = None
        self._hud_last_tick_time: Optional[float] = None
        self._route_overlay_bounds: Optional[tuple[float, float, float, float]] = None
        self._spectator_follow_transform = None
        self._spectator_follow_log_tick = 0



        self._command_delay_buffer: deque = deque()
        self._command_delay_n_frames: int = max(1, round(0.3 / max(1e-3, float(config.fixed_delta))))

        self._speed_controller = SpeedPIDController(
            target_speed_kmh=config.target_speed_kmh,
            max_throttle=config.max_throttle,
            max_brake=config.max_brake,
        )


        self._use_pure_pursuit = bool(config.cil_use_pure_pursuit)
        if self._use_pure_pursuit and PurePursuitController is not None and np is not None:
            dt = max(float(config.fixed_delta), 1e-3)
            self._pure_pursuit = PurePursuitController(dt=dt)
        else:
            self._pure_pursuit = None
        self._wobble_noise = 0.0

        # FIX: initialize eval_online metric accumulators
        self._metric_total_frames: int = 0
        self._metric_cnn_inference_count: int = 0
        self._metric_cnn_latency_sum: float = 0.0
        self._metric_min_ttc: float = float("inf")
        self._metric_initial_route_distance: float = 0.0

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
        with self._frame_lock:
            self._latest_rgb = None
            self._frame_history.clear()
        self._camera = self._spawn_camera(world, vehicle)
        self._camera.listen(self._on_camera_frame)
        if self._cil_yolo_enabled:
            self._init_cil_yolo_fusion()
        self._init_gtnet_supervisor()
        self._init_navigation_agent(world, vehicle)
        self._command_oracle.reset()
        self._command_delay_buffer.clear()
        self._wobble_noise = 0.0
        self._spectator_follow_transform = None
        self._spectator_follow_log_tick = 0

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


        if RouteMapVisualizer is not None and self._opencv_route_visible:
            self._route_map = RouteMapVisualizer(
                window_name="CIL Route Map",
                canvas_size=620,
            )
            logging.info("RouteMapVisualizer OpenCV window enabled (runtime: press 'r' to toggle).")

        self._enabled = True
        logging.info(
            "CIL agent is ready (hud=%s, route_map_world=%s, route_map_opencv=%s, telemetry_csv=%s).",
            self.config.cil_enable_hud,
            self.config.cil_enable_route_map,
            self._opencv_route_visible,
            self.config.cil_enable_telemetry_csv,
        )
        if self._cil_yolo_enabled:
            logging.info(
                "cil_yolo OpenCV: press 'y' to toggle YOLO window, 'r' to toggle route map window "
                "(initial YOLO=%s, route_cv=%s).",
                self._opencv_yolo_visible,
                self._opencv_route_visible,
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
                "steer_source",
                "uncertainty",
                "ai_confused",
                "steer",
                "throttle",
                "brake",
                "gtnet_brake",
                "gtnet_reason",
            ]
        )
        self._telemetry_fp.flush()

    def _init_cil_yolo_fusion(self) -> None:
        if YoloDetector is None:
            logging.warning("YoloDetector unavailable; cil_yolo will run as plain CIL.")
            self._yolo_detector = None
            return

        model_path = resolve_yolo_model_path(self.config.yolo_model_path)
        if not model_path.exists():
            logging.warning("YOLO model file not found: %s. cil_yolo will run as plain CIL.", model_path)
            self._yolo_detector = None
            return

        detector_imgsz = int(self.config.yolo_inference_imgsz)
        detector_imgsz_arg = detector_imgsz if detector_imgsz > 0 else None
        load_t0 = time.perf_counter()
        self._yolo_detector = YoloDetector(
            str(model_path),
            display_classes=DETECTOR_DISPLAY_CLASSES,
            inference_imgsz=detector_imgsz_arg,
            camera_fov_deg=self.config.camera_fov,
            obstacle_base_distance_m=8.0,
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
            tracker_config=self.config.yolo_tracker_config,
        )
        logging.info(
            "cil_yolo detector initialized in %.1f ms.",
            (time.perf_counter() - load_t0) * 1000.0,
        )

        warmup_fn = getattr(self._yolo_detector, "warmup", None)
        if callable(warmup_fn):
            warmup_t0 = time.perf_counter()
            warmup_fn(self.config.camera_width, self.config.camera_height)
            logging.info("cil_yolo detector warmup completed in %.1f ms.", (time.perf_counter() - warmup_t0) * 1000.0)

        if TrafficSupervisor is None:
            self._traffic_supervisor = None
            logging.warning("TrafficSupervisor unavailable; cil_yolo will use detector-only fallback.")
        else:
            try:
                self._traffic_supervisor = TrafficSupervisor(build_supervisor_config())
                self._last_supervisor_ts = None
                logging.info("TrafficSupervisor integrated into CIL+YOLO control loop.")
            except Exception as exc:
                self._traffic_supervisor = None
                logging.warning("Failed to initialize TrafficSupervisor for cil_yolo: %s", exc)

        logging.info(
            "CIL+YOLO fusion enabled. Detector=%s every_n_ticks=%d opencv_yolo=%s draw_overlay=%s",
            model_path,
            int(self.config.yolo_inference_every_n_ticks),
            bool(self._opencv_yolo_visible),
            bool(self.config.yolo_draw_overlay),
        )

    def _init_gtnet_supervisor(self) -> None:
        self._gtnet_supervisor = None
        self._last_gtnet_debug_info = {}
        if not bool(self.config.gtnet_enabled):
            logging.info("GTNet supervisor disabled in config; skipping checkpoint load.")
            return
        if GTNetSupervisor is None or GTNetSupervisorConfig is None:
            logging.warning("GTNet supervisor unavailable; CIL/CIL+YOLO will run without trajectory prediction.")
            return
        if torch is None or np is None:
            logging.warning("GTNet supervisor requires torch and numpy; disabled.")
            return

        model_path = Path(self.config.gtnet_model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        if not model_path.exists():
            logging.warning("GTNet model file not found: %s. GTNet supervisor disabled.", model_path)
            return

        try:
            cfg = GTNetSupervisorConfig(
                model_path=str(model_path),
                enabled=True,
                inference_every_n_ticks=max(1, int(self.config.gtnet_inference_every_n_ticks)),
                history_frames=max(1, int(self.config.gtnet_history_frames)),
                expected_dt=float(self.config.gtnet_expected_dt),
                fixed_delta=float(self.config.fixed_delta),
                max_actor_distance_m=float(self.config.gtnet_max_actor_distance_m),
                adjacency_mode=str(self.config.gtnet_adjacency_mode),
                fixed_adjacency_radius_m=float(self.config.gtnet_fixed_adjacency_radius_m),
                draw_debug=bool(self.config.gtnet_draw_debug),
            )
            load_t0 = time.perf_counter()
            loaded_supervisor = GTNetSupervisor(cfg, device=self.config.model_device)
            logging.info(
                "GTNet supervisor checkpoint initialized in %.1f ms.",
                (time.perf_counter() - load_t0) * 1000.0,
            )
        except Exception as exc:
            self._gtnet_supervisor = None
            logging.warning("Failed to initialize GTNet supervisor: %s", exc)
            return

        self._gtnet_supervisor = loaded_supervisor
        logging.info(
            "GTNet supervisor ACTIVE: model=%s enabled=true "
            "(will override braking when trajectory conflicts are detected)",
            model_path,
        )

    def _annotate_cil_yolo_frame(
        self,
        frame_bgr: Any,
        detections: list,
        debug_info: Dict[str, Any],
        sup_debug: Dict[str, Any],
        speed_kmh: float,
        control: Any,
        fps: float,
    ) -> Any:
        annotated = frame_bgr.copy()
        self._draw_cil_yolo_fps_overlay(annotated, fps)
        if not self.config.yolo_draw_overlay:
            return annotated

        _draw_red_light_zone_rois(
            annotated,
            sup_debug,
            visible=bool(self.config.yolo_show_red_light_rois),
        )
        _draw_yellow_danger_corridor(annotated, debug_info, sup_debug)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.get("box", (0, 0, 0, 0))]
            class_name = str(det.get("class_name", "unknown"))
            confidence = float(det.get("confidence", 0.0))
            distance = float(det.get("distance", float("inf")))
            danger_match = bool(det.get("danger_match", False))
            in_danger_roi = bool(det.get("in_danger_roi", False))
            if class_name == "traffic_light_red" or danger_match:
                color = (0, 0, 255)
            elif class_name == "traffic_light_green":
                color = (0, 255, 0)
            elif in_danger_roi or distance < 10.0:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{class_name} {confidence:.2f} ({distance:.1f}m)",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        supervisor_brake = float(debug_info.get("supervisor_brake", 0.0))
        state = str(debug_info.get("supervisor_state", "n/a"))
        reason = str(debug_info.get("decision_reason", "none"))
        status_color = (0, int(round(255.0 * (1.0 - supervisor_brake))), int(round(255.0 * supervisor_brake)))
        cv2.putText(
            annotated,
            f"CIL+YOLO {state} brake={supervisor_brake:.2f} reason={reason}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
        )
        cv2.putText(
            annotated,
            f"speed={speed_kmh:.1f} km/h steer={float(control.steer):+.3f} thr={float(control.throttle):.2f} brk={float(control.brake):.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (40, 220, 255),
            2,
        )
        return annotated

    @staticmethod
    def _draw_cil_yolo_fps_overlay(frame_bgr: Any, fps: float) -> None:
        if cv2 is None or frame_bgr is None or getattr(frame_bgr, "shape", None) is None:
            return

        text = f"FPS: {float(fps):.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.72
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size
        frame_h, frame_w = frame_bgr.shape[:2]
        pad = 8
        x1 = max(0, frame_w - text_w - 2 * pad - 10)
        y1 = 10
        x2 = min(frame_w - 1, x1 + text_w + 2 * pad)
        y2 = min(frame_h - 1, y1 + text_h + baseline + 2 * pad)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(
            frame_bgr,
            text,
            (x1 + pad, y2 - pad - baseline),
            font,
            scale,
            (40, 220, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _run_cil_yolo_fusion(
        self,
        frame_rgb: Any,
        step_idx: int,
        current_steer: float,
        speed_kmh: float,
        control: Any,
        fps: float,
    ) -> tuple[bool, Dict[str, Any], Any]:
        if self._yolo_detector is None:
            return False, {}, None

        detect_every_n = max(1, int(self.config.yolo_inference_every_n_ticks))
        should_run_detector = (
            self._last_yolo_detection_step is None
            or detect_every_n <= 1
            or (step_idx - int(self._last_yolo_detection_step)) >= detect_every_n
        )
        if not should_run_detector:
            cached_debug_info = dict(self._cached_yolo_debug_info)
            cached_debug_info["detector_cache_hit"] = True
            cached_debug_info["detector_cache_age_ticks"] = (
                0 if self._last_yolo_detection_step is None else step_idx - int(self._last_yolo_detection_step)
            )
            annotated = None
            if self._opencv_yolo_visible:
                annotated = self._cached_yolo_annotated_frame
                if annotated is not None:
                    annotated = annotated.copy()
                    self._draw_cil_yolo_fps_overlay(annotated, fps)
            return (
                bool(self._cached_yolo_emergency),
                cached_debug_info,
                annotated,
            )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections, detector_emergency = self._yolo_detector.detect_and_evaluate(
            frame_bgr,
            distance_threshold=None,
            depth_map_m=None,
            vehicle_steer=current_steer,
            speed_kmh=speed_kmh,
        )
        debug_info = {}
        if hasattr(self._yolo_detector, "get_last_debug_info"):
            debug_info = self._yolo_detector.get_last_debug_info() or {}
        debug_info = dict(debug_info)
        debug_info["detector_cache_hit"] = False
        debug_info["detector_cache_age_ticks"] = 0

        supervisor_brake = 0.0
        sup_debug: Dict[str, Any] = {}
        supervisor_target_type = "none"
        if self._traffic_supervisor is not None:
            now_ts = time.time()
            if self._last_supervisor_ts is None:
                dt = self.config.fixed_delta if self.config.sync else (1.0 / 30.0)
            elif self.config.sync:
                dt = self.config.fixed_delta
            else:
                dt = max(1e-3, now_ts - float(self._last_supervisor_ts))
            self._last_supervisor_ts = now_ts
            try:
                supervisor_brake = float(
                    self._traffic_supervisor.compute(
                        detections=to_supervisor_detections(detections),
                        current_speed=float(speed_kmh) / 3.6,
                        image_shape=frame_bgr.shape,
                        distance_threshold=None,
                        vehicle_steer=float(current_steer),
                        dt=dt,
                    )
                )
                supervisor_brake = clamp(supervisor_brake, 0.0, 1.0)
                sup_debug = self._traffic_supervisor.get_debug_info()
                supervisor_target_type = str(sup_debug.get("selected_target_type", "none"))
            except Exception as exc:
                logging.warning("CIL+YOLO TrafficSupervisor compute failed: %s", exc)
                supervisor_brake = 0.0
                sup_debug = {}

        hard_supervisor_emergency = supervisor_brake >= 0.60 and supervisor_target_type == "obstacle"
        is_emergency = bool(hard_supervisor_emergency or (detector_emergency and self._traffic_supervisor is None))

        debug_info["supervisor_brake"] = float(supervisor_brake)
        debug_info["decision_reason"] = supervisor_target_type
        if sup_debug:
            debug_info["supervisor_state"] = str(sup_debug.get("state", "n/a"))
            debug_info["red_hard_stop_active"] = bool(sup_debug.get("red_hard_stop_active", False))
            debug_info["locked_zone"] = sup_debug.get("locked_zone")
            debug_info["green_immunity_counter"] = int(sup_debug.get("green_immunity_counter", 0))
            debug_info["obstacle_reason"] = str(sup_debug.get("obstacle_reason", ""))

        annotated = None
        if self._opencv_yolo_visible:
            annotated = self._annotate_cil_yolo_frame(
                frame_bgr,
                detections,
                debug_info,
                sup_debug,
                speed_kmh,
                control,
                fps,
            )

        self._last_yolo_detection_step = int(step_idx)
        self._cached_yolo_emergency = bool(is_emergency)
        self._cached_yolo_debug_info = dict(debug_info)
        self._cached_yolo_annotated_frame = annotated
        return bool(is_emergency), debug_info, annotated

    def _ensure_route_map_visualizer(self) -> None:
        if self._route_map is not None or RouteMapVisualizer is None:
            return
        self._route_map = RouteMapVisualizer(
            window_name="CIL Route Map",
            canvas_size=620,
        )
        logging.info("RouteMapVisualizer created (OpenCV route map window).")

    def _destroy_yolo_opencv_window(self) -> None:
        if cv2 is None or not self._cil_yolo_enabled:
            return
        try:
            cv2.destroyWindow(self._yolo_window_name)
        except Exception:
            pass

    def _destroy_route_opencv_window(self) -> None:
        if self._route_map is None:
            return
        self._route_map.close()
        self._route_map = None

    def _process_cil_opencv_hotkeys(self, key: int) -> None:
        if key < 0 or not self._cil_yolo_enabled:
            return
        ch = key & 0xFF
        if ch in (ord("y"), ord("Y")):
            self._opencv_yolo_visible = not self._opencv_yolo_visible
            if not self._opencv_yolo_visible:
                self._destroy_yolo_opencv_window()
            logging.info(
                "cil_yolo YOLO OpenCV window %s (press 'y' to toggle).",
                "ON" if self._opencv_yolo_visible else "OFF",
            )
        if ch in (ord("r"), ord("R")):
            self._opencv_route_visible = not self._opencv_route_visible
            if not self._opencv_route_visible:
                self._destroy_route_opencv_window()
            else:
                self._ensure_route_map_visualizer()
            logging.info(
                "cil_yolo route map OpenCV window %s (press 'r' to toggle).",
                "ON" if self._opencv_route_visible else "OFF",
            )

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

    def _spectator_transform_for_vehicle(self, vehicle_transform) -> Any:
        forward = vehicle_transform.get_forward_vector()
        loc = vehicle_transform.location
        follow_dist = float(self.config.spectator_follow_distance)
        height = float(self.config.spectator_height)
        camera_loc = carla.Location(
            x=float(loc.x) - float(forward.x) * follow_dist,
            y=float(loc.y) - float(forward.y) * follow_dist,
            z=float(loc.z) + height,
        )
        look_at = carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z) + 1.2)
        dx = float(look_at.x - camera_loc.x)
        dy = float(look_at.y - camera_loc.y)
        dz = float(look_at.z - camera_loc.z)
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(dz, max(1e-3, math.hypot(dx, dy))))
        return carla.Transform(
            camera_loc,
            carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
        )

    def _apply_spawn_locked_spectator(self, world, spawn_transform) -> None:
        if carla is None or not self.config.lock_spectator_on_spawn:
            return

        try:
            spectator_tf = self._spectator_transform_for_vehicle(spawn_transform)
            world.get_spectator().set_transform(spectator_tf)
            manager = getattr(self.session, "_manager", None) if self.session is not None else None
            remember = getattr(manager, "remember_spectator_transform", None)
            if callable(remember):
                remember(spectator_tf)
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
            # FIX: diagnostic logging for route map navigation debugging
            cmd_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            junction_count = 0
            for item in reference_route:
                cmd = int(item.get("command", 0))
                cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
                if bool(item.get("is_junction", False)):
                    junction_count += 1
            logging.info(
                "CIL cached fixed route plan with %d points from %s. "
                "Commands: FOLLOW=%d LEFT=%d RIGHT=%d STRAIGHT=%d | Junctions=%d",
                len(reference_route),
                route_source,
                cmd_counts[0], cmd_counts[1], cmd_counts[2], cmd_counts[3],
                junction_count,
            )
            if cmd_counts[1] == 0 and cmd_counts[2] == 0 and cmd_counts[3] == 0:
                logging.warning(
                    "⚠️ ROUTE MAP BUG: All commands are FOLLOW(0)! The route appears to be a straight "
                    "line from S to D with no turns. Possible causes: (1) destination_point is too close "
                    "or on the same road as spawn_point, (2) GlobalRoutePlanner could not find junctions. "
                    "Try setting destination_point to a spawn index that requires turning."
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
        if self._nav_agent is None:
            return None
        for attr_name in ("get_local_planner", "_local_planner"):
            if hasattr(self._nav_agent, attr_name):
                attr = getattr(self._nav_agent, attr_name)
                if callable(attr):
                    try:
                        return attr()
                    except Exception:
                        pass
                else:
                    return attr
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

        planner = self._get_local_planner()

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

    def _collect_route_draw_locations(self, fallback_route_locations: list[Any]) -> list[Any]:
        """Return the full S->D route for visualization, falling back to planner-local points."""
        draw_locations: list[Any] = []

        if self._reference_route_plan:
            draw_locations = self._route_planner.collect_reference_route_locations(
                self._reference_route_plan,
                max_points=4096,
                anchor_location=None,
            )

        if not draw_locations:
            draw_locations = list(fallback_route_locations or [])

        if draw_locations:
            if (
                self._route_start_location is not None
                and self._xy_distance(draw_locations[0], self._route_start_location) > 0.75
            ):
                draw_locations.insert(0, self._route_start_location)
            if (
                self._route_destination_location is not None
                and self._xy_distance(draw_locations[-1], self._route_destination_location) > 0.75
            ):
                draw_locations.append(self._route_destination_location)
        elif self._route_start_location is not None and self._route_destination_location is not None:
            draw_locations = [self._route_start_location, self._route_destination_location]

        return draw_locations

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
            v = clamp(1.0 - (pt_xy[1] - min_y) / span_y, 0.0, 1.0)
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
        if CIL_NvidiaCNN is None or unwrap_state_dict is None or classify_checkpoint_state_dict is None:
            raise RuntimeError("Cannot import CIL waypoint model definitions from core_perception.cnn_model.")

        model_path = resolve_cil_model_path(self.config.cil_model_path)
        if self.config.cil_model_path.lower() == "auto":
            logging.info("Auto-selected CIL model path: %s", model_path)
        if not model_path.exists():
            models_dir = Path(__file__).resolve().parent / "models"
            existing = ", ".join(str(p.name) for p in sorted(models_dir.glob("*.pth")))
            if not existing:
                existing = "no .pth file found in models/"
            raise RuntimeError(f"CIL model file not found: {model_path}. Available: {existing}")
        if model_path.suffix.lower() not in {".pth", ".pt"}:
            raise RuntimeError(
                f"CIL model path must be a PyTorch checkpoint (.pth/.pt), got '{model_path.suffix}': {model_path}. "
                "TensorRT/ONNX detector files belong in --yolo-model-path."
            )

        device_name = self.config.model_device.lower()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            device_name = "cpu"

        self._device = torch.device(device_name)

        load_t0 = time.perf_counter()
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        state_dict = unwrap_state_dict(checkpoint)
        checkpoint_load_ms = (time.perf_counter() - load_t0) * 1000.0
        model_kind = classify_checkpoint_state_dict(state_dict)
        if model_kind != "waypoint":
            raise RuntimeError(
                f"Incompatible CIL checkpoint '{model_path.name}' detected as '{model_kind}'. "
                "Train or provide a waypoint predictor checkpoint such as models/waypoint_predictor.pth."
            )

        uses_single_film_checkpoint = (
            "film.embedding.weight" in state_dict
            and "film_s4.embedding.weight" not in state_dict
        )
        if uses_single_film_checkpoint:
            upgraded_state_dict = {}
            for key, value in state_dict.items():
                clean_key = str(key)
                if clean_key.startswith("film."):




                    upgraded_state_dict[f"film_s4.{clean_key[len('film.'):]}"] = value
                else:
                    upgraded_state_dict[clean_key] = value
            state_dict = upgraded_state_dict

        model_t0 = time.perf_counter()
        model = CIL_NvidiaCNN(pretrained_backbone=False).to(self._device)
        if uses_single_film_checkpoint:
            load_result = model.load_state_dict(state_dict, strict=False)



            allowed_missing = {k for k in load_result.missing_keys if k.startswith("film_s3.")}
            real_missing = set(load_result.missing_keys) - allowed_missing
            unexpected = set(load_result.unexpected_keys)
            if unexpected or real_missing:
                raise RuntimeError(
                    f"Cannot adapt old single-FiLM CIL checkpoint '{model_path.name}' to current model. "
                    f"Missing={sorted(real_missing)} Unexpected={sorted(unexpected)}"
                )
            logging.info(
                "Loaded old single-FiLM CIL checkpoint '%s': film.* -> film_s4.* remapped; "
                "film_s3 uses identity init (gamma=1, beta=0 passthrough).",
                model_path.name,
            )
        else:
            model.load_state_dict(state_dict, strict=True)
        model.eval()
        logging.info(
            "Loaded CIL model from %s on %s (checkpoint=%.1f ms, model_init=%.1f ms)",
            model_path,
            self._device,
            checkpoint_load_ms,
            (time.perf_counter() - model_t0) * 1000.0,
        )
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
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(pitch=-8.0))),
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
            self._frame_history.append(rgb)  # FIX: populate temporal buffer

    def _read_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_rgb
            self._latest_rgb = None
        return frame

    def _read_latest_triplet(self):
        with self._frame_lock:
            if not self._frame_history:
                return None
            frames = list(self._frame_history)
        if len(frames) >= 3:
            return frames[-3:]
        last = frames[-1]
        return [last] * (3 - len(frames)) + frames

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

    def _calculate_dynamic_speed(
        self,
        waypoints_2d: Any,
        max_speed: float = 30.0,
        min_speed: float = 12.0,
    ) -> float:
        if np is None:
            return float(min_speed)
        if waypoints_2d is None or len(waypoints_2d) < 3:
            return float(min_speed)


        n = len(waypoints_2d)
        max_angle_deg = 0.0
        for i in range(n - 2):
            v1 = waypoints_2d[i + 1] - waypoints_2d[i]
            v2 = waypoints_2d[i + 2] - waypoints_2d[i + 1]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-4 or n2 < 1e-4:
                continue
            cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_theta)))
            max_angle_deg = max(max_angle_deg, angle)

        if max_angle_deg > 25.0:
            target_speed = float(min_speed)
        else:
            speed_drop = (max_angle_deg / 25.0) * (float(max_speed) - float(min_speed))
            target_speed = float(max_speed) - float(speed_drop)

        return float(np.clip(target_speed, float(min_speed), float(max_speed)))

    def _predict_cil_waypoints(
        self,
        rgb_frame,
        speed_kmh: float,
        command: int,
    ) -> tuple[Any, float]:
        if np is None or torch is None or cv2 is None:
            return (np.zeros((5, 2), dtype=np.float32) if np is not None else None, 0.0)

        if isinstance(rgb_frame, (list, tuple)):
            frames = list(rgb_frame)
        else:
            frames = [rgb_frame]

        if len(frames) == 0:
            return (np.zeros((5, 2), dtype=np.float32), 0.0)
        if len(frames) < 3:
            last = frames[-1]
            frames = [last] * (3 - len(frames)) + frames
        if len(frames) > 3:
            frames = frames[-3:]

        yuv_frames = []
        for frame in frames:
            height = frame.shape[0]
            cropped = frame[int(height * 0.45) :, :, :]
            resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)
            yuv_frames.append(cv2.cvtColor(resized, cv2.COLOR_RGB2YUV))

        stacked = np.concatenate(yuv_frames, axis=-1)
        image_tensor = torch.from_numpy(stacked).permute(2, 0, 1).float().div_(255.0)
        image_tensor.sub_(0.5).div_(0.5)
        image_tensor.unsqueeze_(0)

        command_idx = max(0, min(3, int(command)))
        speed_norm = clamp(float(speed_kmh) / self.CIL_MAX_SPEED_KMH, 0.0, 1.0)
        command_tensor = torch.tensor([command_idx], dtype=torch.long)
        speed_tensor = torch.tensor([speed_norm], dtype=torch.float32)

        image_tensor = image_tensor.to(self._device, non_blocking=True)
        command_tensor = command_tensor.to(self._device, non_blocking=True)
        speed_tensor = speed_tensor.to(self._device, non_blocking=True)

        with torch.inference_mode():
            predictions = self._model(image_tensor, command_tensor, speed_tensor)


        if torch.is_tensor(predictions):
            pred_tensor = predictions.detach().squeeze(0).cpu().float().numpy()
            if pred_tensor.shape[0] >= 15:
                wp_array = pred_tensor[:10].reshape(5, 2)
                mean_uncertainty = float(np.mean(pred_tensor[10:15]))
            else:
                logging.error(
                    "Shape Model Output bị sai: %s (Kỳ vọng ít nhất 15). Dùng Zeros.",
                    pred_tensor.shape,
                )
                wp_array = np.zeros((5, 2), dtype=np.float32)
                mean_uncertainty = 0.0
        else:
            logging.error("Model không trả về Tensor! Dùng Zeros.")
            wp_array = np.zeros((5, 2), dtype=np.float32)
            mean_uncertainty = 0.0

        return wp_array, mean_uncertainty

    def _constrain_waypoints_to_lane(
        self,
        pred_waypoints: Any,
        vehicle: Any,
        command: int,
    ) -> Any:
        """Smoothly blend predicted ego-waypoints toward valid CARLA lane waypoints.

        Strategy:
        1) Project model waypoints to driving lanes in map space (same direction).
        2) Build a smooth transition curve with 4 control points in ego-frame:
           - C0: current ego position (0, 0)
           - C1: inertia point ahead of ego heading
           - C2: lane-projected waypoint A2
           - C3: lane-projected waypoint A4
        3) Blend model waypoints W_i with smooth curve S_i using progressive weights.
        """
        if carla is None or np is None or vehicle is None:
            return pred_waypoints
        if self.session is None or self.session.world is None:
            return pred_waypoints

        try:
            waypoints = np.asarray(pred_waypoints, dtype=np.float32)
            if waypoints.ndim != 2 or waypoints.shape[1] != 2:
                return pred_waypoints
            if waypoints.shape[0] < 1:
                return waypoints

            world_map = self.session.world.get_map()
            transform = vehicle.get_transform()
            ego_loc = transform.location
            yaw_rad = math.radians(float(transform.rotation.yaw))
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)

            ego_wp = world_map.get_waypoint(
                ego_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if ego_wp is None:
                return pred_waypoints

            ego_lane_id = int(getattr(ego_wp, "lane_id", 0))
            lane_projected = np.array(waypoints, copy=True)
            prev_valid_wp = ego_wp
            valid_lane_points = 0

            for i in range(lane_projected.shape[0]):
                ex = float(waypoints[i, 0])
                ey = float(waypoints[i, 1])

                wx = float(ego_loc.x) + cos_yaw * ex - sin_yaw * ey
                wy = float(ego_loc.y) + sin_yaw * ex + cos_yaw * ey
                world_loc = carla.Location(x=wx, y=wy, z=float(ego_loc.z))

                map_wp = world_map.get_waypoint(
                    world_loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if map_wp is None:
                    continue

                map_lane_id = int(getattr(map_wp, "lane_id", 0))
                same_direction = ego_lane_id == 0 or map_lane_id == 0 or (map_lane_id * ego_lane_id) > 0
                if not same_direction:
                    next_wps = prev_valid_wp.next(max(3.0, ex * 0.3))
                    if not next_wps:
                        continue
                    map_wp = next_wps[0]

                snapped_loc = map_wp.transform.location
                dx = float(snapped_loc.x - ego_loc.x)
                dy = float(snapped_loc.y - ego_loc.y)
                lane_projected[i, 0] = cos_yaw * dx + sin_yaw * dy
                lane_projected[i, 1] = -sin_yaw * dx + cos_yaw * dy
                prev_valid_wp = map_wp
                valid_lane_points += 1

            if valid_lane_points < 2:
                if valid_lane_points >= 1:
                    return lane_projected.astype(np.float32)
                return waypoints


            n_points = int(waypoints.shape[0])
            idx_mid = min(2, n_points - 1)
            idx_far = min(4, n_points - 1)


            forward_hint = max(2.0, float(waypoints[1, 0]) if n_points > 1 else 2.0)
            inertia_x = float(np.clip(forward_hint, 2.0, 8.0))


            a2_idx = idx_mid

            if int(command) in (1, 2):
                inertia_x = max(2.0, inertia_x * 0.8)


            a2 = lane_projected[a2_idx]
            a4 = lane_projected[idx_far]


            c0 = np.array([0.0, 0.0], dtype=np.float32)
            c1 = np.array([inertia_x, 0.0], dtype=np.float32)
            c2 = np.array([float(a2[0]), float(a2[1])], dtype=np.float32)
            c3 = np.array([float(a4[0]), float(a4[1])], dtype=np.float32)

            def _bezier_point(t: float):
                omt = 1.0 - t
                return (
                    (omt ** 3) * c0
                    + 3.0 * (omt ** 2) * t * c1
                    + 3.0 * omt * (t ** 2) * c2
                    + (t ** 3) * c3
                )

            ts = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
            smooth_curve = np.stack([_bezier_point(float(t)) for t in ts], axis=0).astype(np.float32)



            alphas = np.linspace(0.20, 0.90, n_points, dtype=np.float32)
            if int(command) in (1, 2, 3):
                alphas = np.clip(alphas + 0.01, 0.0, 0.92)


            strength = float(getattr(self.config, "cil_lane_constrain_strength", 1.0))
            alphas = alphas * strength

            blended = (1.0 - alphas[:, None]) * waypoints + alphas[:, None] * smooth_curve


            blended[:, 0] = np.maximum.accumulate(np.maximum(blended[:, 0], 0.2))
            return blended.astype(np.float32)
        except Exception as exc:
            logging.debug("CIL lane constraint skipped: %s", exc)
            return pred_waypoints

    def _stabilize_cil_steering(
        self,
        steering_raw: float,
        speed_kmh: float,
        command: int,
        command_phase: str,
    ) -> float:





        speed_ms = float(speed_kmh) / 3.6
        is_turning = int(command) in (1, 2)


        if speed_ms < 3.0:
            alpha = 0.70
        elif speed_ms < 6.0:
            alpha = 0.55
        elif speed_ms < 9.0:
            alpha = 0.45
        else:
            alpha = 0.35


        if is_turning:
            alpha = min(alpha + 0.10, 0.75)

        smoothed = alpha * float(steering_raw) + (1.0 - alpha) * self._last_steer


        max_change = 0.10 if is_turning else 0.06
        final = clamp(smoothed, self._last_steer - max_change, self._last_steer + max_change)
        self._last_steer = clamp(float(final), -1.0, 1.0)
        return float(self._last_steer)

    def _route_locations_to_ego_waypoints(
        self,
        vehicle: Any,
        route_locations: list[Any],
        max_points: int = 5,
    ) -> Optional[Any]:
        if np is None or vehicle is None or not route_locations:
            return None

        transform = vehicle.get_transform() if vehicle is not None else None
        if transform is None:
            return None

        yaw_rad = math.radians(float(transform.rotation.yaw))
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        origin = transform.location

        ego_points: list[tuple[float, float]] = []
        for loc in route_locations:
            if loc is None:
                continue
            dx = float(loc.x - origin.x)
            dy = float(loc.y - origin.y)
            ego_x = cos_yaw * dx + sin_yaw * dy
            ego_y = -sin_yaw * dx + cos_yaw * dy
            if ego_x <= 0.0:
                continue
            ego_points.append((ego_x, ego_y))
            if len(ego_points) >= max_points:
                break

        if len(ego_points) < 2:
            return None
        return np.asarray(ego_points, dtype=np.float32)

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


        if show_route_map and self._route_destination_location is not None:
            d_loc = self._route_destination_location
            d_point = carla.Location(
                x=float(d_loc.x), y=float(d_loc.y), z=float(d_loc.z) + 2.0,
            )
            debug.draw_point(d_point, size=0.20, color=carla.Color(r=255, g=50, b=50), life_time=life_time, persistent_lines=False)
            debug.draw_string(d_point, "  DEST", False, carla.Color(r=255, g=50, b=50), life_time, False)

    def _update_spectator_follow(self) -> None:
        """Lock the spectator camera to follow behind the ego vehicle each tick."""
        if not bool(self.config.spectator_reapply_each_tick):
            return
        if self.session is None or self.session.world is None or self.session.ego_vehicle is None:
            return
        if carla is None:
            return

        vehicle = self.session.ego_vehicle
        transform = vehicle.get_transform()
        target_tf = self._spectator_transform_for_vehicle(transform)
        try:
            self.session.world.get_spectator().set_transform(target_tf)
            self._spectator_follow_transform = target_tf
            self._spectator_follow_log_tick += 1
            if self._spectator_follow_log_tick % 100 == 0:
                loc = transform.location
                spec_loc = target_tf.location
                logging.info(
                    "CIL spectator follow ego id=%s ego=(%.1f, %.1f, %.1f) spec=(%.1f, %.1f, %.1f) yaw=%.1f pitch=%.1f",
                    getattr(vehicle, "id", "?"),
                    float(loc.x),
                    float(loc.y),
                    float(loc.z),
                    float(spec_loc.x),
                    float(spec_loc.y),
                    float(spec_loc.z),
                    float(target_tf.rotation.yaw),
                    float(target_tf.rotation.pitch),
                )
            manager = getattr(self.session, "_manager", None)
            remember = getattr(manager, "remember_spectator_transform", None)
            if callable(remember):
                remember(target_tf)
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

        # FIX: increment eval_online metric counter
        self._metric_total_frames += 1

        speed_kmh = self._current_speed_kmh()
        self._last_speed_kmh = speed_kmh
        hud_fps = self._update_hud_fps()
        stage_times["read"] = time.perf_counter() - read_t0

        self._write_video_frame(frame)
        if self._stop_requested:
            logging.info("CIL video duration target reached, stopping agent loop.")
            return

        destination_distance_m = self._distance_to_destination(vehicle.get_location())
        # FIX: capture initial route distance for eval_online metrics
        if self._metric_initial_route_distance <= 0.0 and destination_distance_m is not None:
            self._metric_initial_route_distance = float(destination_distance_m)
        if destination_distance_m is not None and destination_distance_m <= self._arrival_distance_m:
            self._request_stop_at_destination("distance_threshold", destination_distance_m)
            return

        nav_t0 = time.perf_counter()

        if self._nav_agent is not None:
            if not self._reference_route_plan:
                self._cache_reference_route_plan()
            try:
                if self._nav_agent.done():
                    if destination_distance_m is None:
                        self._request_stop_at_destination("planner_done", destination_distance_m)
                        return
                    if destination_distance_m <= self._arrival_distance_m:
                        self._request_stop_at_destination("planner_done", destination_distance_m)
                        return
                    if (
                        self._route_destination_location is not None
                        and step_idx - int(self._last_replan_tick) >= 30
                    ):
                        try:
                            self._route_start_location = vehicle.get_location()
                            set_navigation_destination(
                                self._nav_agent,
                                self._route_start_location,
                                self._route_destination_location,
                            )
                            self._last_replan_tick = int(step_idx)
                            self._cache_reference_route_plan(force=True)
                            self._command_oracle.reset()
                            logging.warning(
                                "CIL ignored early planner_done at %.2fm from D; reissued destination and refreshed reference route.",
                                float(destination_distance_m),
                            )
                        except Exception as exc:
                            self._last_replan_tick = int(step_idx)
                            logging.debug("CIL failed to reissue destination after early planner_done: %s", exc)
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
        route_draw_locations = self._collect_route_draw_locations(route_locations)

        command, command_debug = self._update_distance_based_command(
            speed_kmh=speed_kmh,
            step_idx=step_idx,
        )




        self._command_delay_buffer.append(command)
        if len(self._command_delay_buffer) > self._command_delay_n_frames:
            command = self._command_delay_buffer.popleft()
        else:
            command = self._command_delay_buffer[0]

        stage_times["nav"] = time.perf_counter() - nav_t0

        model_t0 = time.perf_counter()
        frame_triplet = self._read_latest_triplet() or [frame, frame, frame]
        _cnn_t0 = time.perf_counter()
        model_waypoints, mean_uncertainty = self._predict_cil_waypoints(frame_triplet, speed_kmh, command)
        self._metric_cnn_latency_sum += time.perf_counter() - _cnn_t0
        self._metric_cnn_inference_count += 1
        stage_times["model"] = time.perf_counter() - model_t0

        control_t0 = time.perf_counter()

        if bool(self.config.cil_use_carla_waypoints):
            carla_waypoints = self._route_locations_to_ego_waypoints(
                vehicle,
                route_locations,
                max_points=15,
            )
            if carla_waypoints is None or (np is not None and carla_waypoints.shape[0] < 2):
                pred_waypoints = model_waypoints
            else:
                pred_waypoints = carla_waypoints
        elif bool(self.config.cil_lane_constrain_blended):
            pred_waypoints = self._constrain_waypoints_to_lane(model_waypoints, vehicle, command)
        else:
            pred_waypoints = model_waypoints


        ai_is_confused = float(mean_uncertainty) > 1e9

        if ai_is_confused:
            logging.warning(
                "[VETO] AI mất tự tin (Uncertainty=%.2f) -> PHANH KHẨN CẤP!",
                float(mean_uncertainty),
            )
            adaptive_target_kmh = 0.0
            throttle, brake = 0.0, 1.0
            steering_source = self._last_steer
        else:
            adaptive_target_kmh = self._calculate_dynamic_speed(
                pred_waypoints,
                max_speed=30.0,
                min_speed=12.0,
            )
            self._speed_controller.set_target_speed(adaptive_target_kmh)
            throttle, brake = self._speed_controller.compute(speed_kmh)

            if self._pure_pursuit is not None:
                steering_source = self._pure_pursuit.compute_steering(pred_waypoints, speed_kmh)
            else:
                steering_source = 0.0
                logging.error("PurePursuitController chưa được khởi tạo!")

        if bool(self.config.cil_use_carla_waypoints):
            raw_noise = random.gauss(0.0, 0.09)
            self._wobble_noise = 0.88 * self._wobble_noise + 0.12 * raw_noise
            steering = float(steering_source) + self._wobble_noise
        else:
            steering = self._stabilize_cil_steering(
                steering_raw=steering_source,
                speed_kmh=speed_kmh,
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
        yolo_debug_info: Dict[str, Any] = {}
        yolo_emergency = False
        annotated_yolo_frame = None
        if self._cil_yolo_enabled and self._yolo_detector is not None:
            yolo_emergency, yolo_debug_info, annotated_yolo_frame = self._run_cil_yolo_fusion(
                frame,
                step_idx=step_idx,
                current_steer=float(control.steer),
                speed_kmh=float(speed_kmh),
                control=control,
                fps=hud_fps,
            )
            supervisor_brake = float(yolo_debug_info.get("supervisor_brake", 0.0))
            red_hard_stop_active = bool(yolo_debug_info.get("red_hard_stop_active", False))
            supervisor_reason = str(yolo_debug_info.get("decision_reason", "none")).strip().lower()
            hold_hand_brake = bool(
                red_hard_stop_active
                and supervisor_brake >= 0.99
                and supervisor_reason in ("stop_line", "red_light", "traffic_light_red")
                and float(speed_kmh) <= 2.0
            )
            if yolo_emergency or supervisor_brake > 0.0:
                emergency_floor = 0.6 if yolo_emergency else 0.0
                control.throttle = 0.0
                control.brake = float(clamp(max(float(control.brake), supervisor_brake, emergency_floor), 0.0, 1.0))
                control.hand_brake = bool(hold_hand_brake)
        gtnet_debug_info: Dict[str, Any] = {}
        if self._gtnet_supervisor is not None and self.session is not None and self.session.world is not None:
            try:
                gtnet_debug_info = self._gtnet_supervisor.update(
                    world=self.session.world,
                    ego_vehicle=vehicle,
                    step_idx=int(step_idx),
                    speed_kmh=float(speed_kmh),
                    vehicle_steer=float(control.steer),
                )
                self._last_gtnet_debug_info = dict(gtnet_debug_info)
                gtnet_brake = float(gtnet_debug_info.get("brake", 0.0))
                gtnet_throttle_floor = float(gtnet_debug_info.get("throttle_floor", 0.0))
                if gtnet_brake > float(control.brake):
                    control.throttle = 0.0
                    control.brake = float(clamp(gtnet_brake, 0.0, 1.0))
                    control.hand_brake = False
                elif (gtnet_throttle_floor > 0.0
                      and float(control.brake) == 0.0
                      and float(control.throttle) < gtnet_throttle_floor):
                    control.throttle = float(clamp(gtnet_throttle_floor, 0.0, 1.0))
            except Exception as exc:
                logging.warning("GTNet supervisor update failed: %s", exc)
                gtnet_debug_info = {}
        vehicle.apply_control(control)
        vehicle_location = vehicle.get_location()
        rotation = vehicle.get_transform().rotation
        self._update_route_history(vehicle_location)



        if self.config.cil_world_waypoint_debug and step_idx % 3 == 0 and np is not None:
            _yaw_r = math.radians(float(rotation.yaw))
            _cos_y, _sin_y = math.cos(_yaw_r), math.sin(_yaw_r)
            _z_draw = float(vehicle_location.z) + 0.5
            _high_unc = float(mean_uncertainty) > 50.0
            _wp_color = carla.Color(255, 0, 0) if _high_unc else carla.Color(0, 200, 255)
            _line_color = carla.Color(255, 200, 0)
            _prev_w = None
            for _wi in range(model_waypoints.shape[0]):
                _ex = float(model_waypoints[_wi, 0])
                _ey = float(model_waypoints[_wi, 1])
                _wx = float(vehicle_location.x) + _cos_y * _ex - _sin_y * _ey
                _wy = float(vehicle_location.y) + _sin_y * _ex + _cos_y * _ey
                _wloc = carla.Location(x=_wx, y=_wy, z=_z_draw)
                self.session.world.debug.draw_point(_wloc, size=0.15, color=_wp_color, life_time=0.15)
                if _prev_w is not None:
                    self.session.world.debug.draw_line(_prev_w, _wloc, thickness=0.06, color=_line_color, life_time=0.15)
                _prev_w = _wloc


        destination_distance_m = self._distance_to_destination(vehicle_location)
        if destination_distance_m is not None and destination_distance_m <= self._arrival_distance_m:
            self._request_stop_at_destination("distance_threshold", destination_distance_m)

        if self._collector is not None:
            frame_id, timestamp = self._resolve_frame_context(step_idx)
            self._collector.add_vehicle_state(
                frame_id=frame_id,
                timestamp=timestamp,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z,
                has_crash=self._had_collision_at(frame_id),
                is_recovering=False,
                is_junction=self._is_vehicle_at_junction(vehicle),
                command=command,
                pitch=rotation.pitch,
                roll=rotation.roll,
                yaw=rotation.yaw,
            )

        stage_times["control"] = time.perf_counter() - control_t0

        viz_t0 = time.perf_counter()

        self._update_spectator_follow()
        self._draw_hud_on_screen(
            step_idx, speed_kmh, adaptive_target_kmh,
            control.steer, control.throttle, control.brake,
            command, hud_fps, destination_distance_m, route_draw_locations,
        )


        opencv_key = -1
        if cv2 is not None and self._enabled:
            if (
                self._cil_yolo_enabled
                and self._yolo_detector is not None
                and self._opencv_yolo_visible
                and annotated_yolo_frame is not None
            ):
                cv2.imshow(self._yolo_window_name, annotated_yolo_frame)
            route_due = self._route_map is not None and self._opencv_route_visible and step_idx % 3 == 0
            if route_due:
                vehicle_location = vehicle.get_location()
                heading_yaw = float(vehicle.get_transform().rotation.yaw)
                self._route_map.show(
                    route_points=route_draw_locations,
                    current_location=vehicle_location,
                    start_location=self._route_start_location,
                    destination_location=self._route_destination_location,
                    heading_yaw_deg=heading_yaw,
                    trajectory_points=self._route_history_xy,
                    command=command,
                    invoke_wait_key=False,
                )
            need_opencv_wait = self._cil_yolo_enabled or (
                self._route_map is not None and self._opencv_route_visible
            )
            if need_opencv_wait:
                opencv_key = int(cv2.waitKey(1) & 0xFF)
            if self._cil_yolo_enabled:
                self._process_cil_opencv_hotkeys(opencv_key)

        stage_times["viz"] = time.perf_counter() - viz_t0

        telemetry_t0 = time.perf_counter()

        if step_idx % 20 == 0:
            logging.info(
                "cil tick=%d speed=%.1f km/h target=%.1f cmd=%d phase=%s next=%d src=%s reset=%s s_from_start=%.1f d_turn=%.1f d_junc=%.1f trigger=%.1f pass_delta=%.2f rise=%d best_turn=%.2f steer=%.3f source=%.3f unc=%.3f veto=%s throttle=%.2f brake=%.2f yolo=%s yolo_brake=%.2f yolo_reason=%s gtnet=%s gtnet_brake=%.2f gtnet_reason=%s",
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
                float(command_debug.get("passed_turn_delta_m", 0.0)),
                int(command_debug.get("turn_distance_rising_frames", 0)),
                float(command_debug.get("best_armed_distance_to_turn_m", float("inf"))),
                control.steer,
                steering_source,
                float(mean_uncertainty),
                str(ai_is_confused),
                control.throttle,
                control.brake,
                str(bool(self._cil_yolo_enabled and self._yolo_detector is not None)),
                float(yolo_debug_info.get("supervisor_brake", 0.0)),
                str(yolo_debug_info.get("decision_reason", "none")),
                str(bool(self._gtnet_supervisor is not None)),
                float(gtnet_debug_info.get("brake", 0.0)),
                str(gtnet_debug_info.get("reason", "none")),
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
                    f"{float(steering_source):.4f}",
                    f"{float(mean_uncertainty):.4f}",
                    str(ai_is_confused),
                    f"{float(control.steer):.4f}",
                    f"{float(control.throttle):.4f}",
                    f"{float(control.brake):.4f}",
                    f"{float(gtnet_debug_info.get('brake', 0.0)):.4f}",
                    str(gtnet_debug_info.get("reason", "none")),
                ]
            )
            if step_idx % 20 == 0 and self._telemetry_fp is not None:
                self._telemetry_fp.flush()

        stage_times["telemetry"] = time.perf_counter() - telemetry_t0
        stage_times["total"] = time.perf_counter() - tick_t0
        self._accumulate_tick_timing(stage_times, step_idx)

    def teardown(self) -> None:
        if self.config.eval_online:




            _out = sys.stderr
            print("\n" + "="*60, file=_out)
            print(" 📊 CIL AGENT ONLINE EVALUATION SUMMARY", file=_out)
            print("="*60, file=_out)
            print(f" Total Frames Ran:       {self._metric_total_frames}", file=_out)

            avg_latency_ms = 0.0
            if self._metric_cnn_inference_count > 0:
                avg_latency_ms = (self._metric_cnn_latency_sum / self._metric_cnn_inference_count) * 1000.0
            print(f" Avg CNN Latency:        {avg_latency_ms:.2f} ms", file=_out)

            if self._hud_ema_fps is not None:
                print(f" System FPS (EMA):       {self._hud_ema_fps:.2f} FPS", file=_out)

            if self._metric_min_ttc != float('inf'):
                print(f" Minimum TTC observed:   {self._metric_min_ttc:.2f} s", file=_out)
            else:
                print(" Minimum TTC observed:   N/A (No vehicles ahead)", file=_out)

            route_completion = 0.0
            try:
                if self._route_start_location and self._route_destination_location:
                    current_loc = self._vehicle_location()
                    if current_loc:
                        dist_remaining = math.hypot(
                            current_loc.x - self._route_destination_location.x,
                            current_loc.y - self._route_destination_location.y
                        )
                        if self._metric_initial_route_distance > 0:
                            route_completion = max(0.0, 100.0 * (1.0 - dist_remaining / self._metric_initial_route_distance))
                            route_completion = min(100.0, route_completion)
            except Exception:
                pass

            print(f" Route Completion:       {route_completion:.1f}%", file=_out)
            print("="*60 + "\n", file=_out)
            _out.flush()


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
        self._yolo_detector = None
        self._traffic_supervisor = None
        self._last_supervisor_ts = None
        self._last_yolo_detection_step = None
        self._cached_yolo_emergency = False
        self._cached_yolo_debug_info = {}
        self._cached_yolo_annotated_frame = None
        self._gtnet_supervisor = None
        self._last_gtnet_debug_info = {}
        if self._cil_yolo_enabled and cv2 is not None:
            try:
                cv2.destroyWindow(self._yolo_window_name)
            except Exception:
                pass
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
        self._command_delay_buffer.clear()
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


class CILYoloAgent(CILAgent):
    name = "cil_yolo"

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._cil_yolo_enabled = True


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
        self._secondary_detector = None
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
        self._control_backend_name = "planner"
        self._last_detection_step: Optional[int] = None
        self._cached_detections: list[dict[str, Any]] = []
        self._cached_detector_emergency = False
        self._cached_debug_info: Dict[str, Any] = {}
        self._last_detection_runtime_ms: Optional[float] = None
        self._tracking_metrics_dir: Optional[Path] = None
        self._tracking_predictions_path: Optional[Path] = None
        self._secondary_tracking_predictions_path: Optional[Path] = None
        self._tracking_ground_truth_path: Optional[Path] = None
        self._tracking_seqinfo_path: Optional[Path] = None
        self._tracking_metadata_path: Optional[Path] = None
        self._tracking_model_path: Optional[Path] = None
        self._tracking_seq_name: str = ""
        self._tracking_gt_logs: list[str] = []
        self._tracking_gt_frame_id = 0
        self._tracking_image_size: tuple[int, int] = (0, 0)
        self._tracking_initial_spawn_index: Optional[int] = None
        self._tracking_initial_transform: Optional[Dict[str, Any]] = None

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
        self._control_backend_name = "planner"
        self._last_detection_step = None
        self._cached_detections = []
        self._cached_detector_emergency = False
        self._cached_debug_info = {}
        self._last_detection_runtime_ms = None
        self._tracking_gt_logs = []
        self._tracking_gt_frame_id = 0
        self._tracking_image_size = (0, 0)
        self._tracking_metrics_dir = None
        self._tracking_predictions_path = None
        self._secondary_tracking_predictions_path = None
        self._tracking_ground_truth_path = None
        self._tracking_seqinfo_path = None
        self._tracking_metadata_path = None
        self._tracking_model_path = None
        self._tracking_seq_name = ""
        self._tracking_initial_spawn_index = None
        self._tracking_initial_transform = None
        self._secondary_detector = None
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
        detector_imgsz: Optional[int] = None
        if int(self.config.yolo_inference_imgsz) > 0:
            detector_imgsz = int(self.config.yolo_inference_imgsz)
        elif model_path.suffix.lower() != ".engine":
            detector_imgsz = 640

        detector_display_classes = list(DETECTOR_DISPLAY_CLASSES)

        self._detector = YoloDetector(
            str(model_path),
            display_classes=detector_display_classes,
            inference_imgsz=detector_imgsz,
            camera_fov_deg=self.config.camera_fov,
            obstacle_base_distance_m=8.0,
            camera_mount_x_m=1.5,
            camera_mount_y_m=0.0,
            camera_mount_z_m=2.2,
            camera_pitch_deg=-8.0,
            tracker_config=self.config.yolo_tracker_config,
            enable_tracking_metrics_logging=True,
        )
        secondary_tracker = str(self.config.yolo_secondary_tracker_config or "").strip()
        if secondary_tracker and secondary_tracker != str(self.config.yolo_tracker_config):
            self._secondary_detector = YoloDetector(
                str(model_path),
                display_classes=detector_display_classes,
                inference_imgsz=detector_imgsz,
                camera_fov_deg=self.config.camera_fov,
                obstacle_base_distance_m=8.0,
                camera_mount_x_m=1.5,
                camera_mount_y_m=0.0,
                camera_mount_z_m=2.2,
                camera_pitch_deg=-8.0,
                tracker_config=secondary_tracker,
                enable_tracking_metrics_logging=True,
            )
            logging.info(
                "Secondary YOLO tracker enabled for same-sequence metrics (%s), display classes: %s",
                secondary_tracker,
                ", ".join(detector_display_classes),
            )
        else:
            self._secondary_detector = None
        self._init_tracking_metrics_workspace(model_path)
        self._use_depth_camera = bool(getattr(self._detector, "uses_depth_input", False))
        warmup_fn = getattr(self._detector, "warmup", None)
        if callable(warmup_fn):
            warmup_t0 = time.perf_counter()
            warmup_fn(self.config.camera_width, self.config.camera_height)
            logging.info(
                "YOLO detector warmup completed in %.1f ms.",
                (time.perf_counter() - warmup_t0) * 1000.0,
            )
        if self._secondary_detector is not None:
            secondary_warmup_fn = getattr(self._secondary_detector, "warmup", None)
            if callable(secondary_warmup_fn):
                warmup_t0 = time.perf_counter()
                secondary_warmup_fn(self.config.camera_width, self.config.camera_height)
                logging.info(
                    "Secondary YOLO detector warmup completed in %.1f ms.",
                    (time.perf_counter() - warmup_t0) * 1000.0,
                )
        logging.info(
            "YOLO runtime config: every_n_ticks=%d visualize=%s draw_overlay=%s imgsz=%s tracker=%s "
            "classes_for_tracking_metrics=%s",
            int(self.config.yolo_inference_every_n_ticks),
            bool(self.config.yolo_visualize),
            bool(self.config.yolo_draw_overlay),
            detector_imgsz if detector_imgsz is not None else "engine-default",
            self.config.yolo_tracker_config,
            "[" + ", ".join(detector_display_classes) + "]",
        )
        if TrafficSupervisor is None:
            logging.warning("TrafficSupervisor unavailable. yolo_detect will run without supervisor brake fusion.")
            self._traffic_supervisor = None
        else:
            try:
                self._traffic_supervisor = TrafficSupervisor(build_supervisor_config())
                logging.info("TrafficSupervisor integrated into yolo_detect control loop.")
            except Exception as exc:
                self._traffic_supervisor = None
                logging.warning("Failed to initialize TrafficSupervisor: %s", exc)
        self._camera = self._spawn_camera(world, vehicle)
        self._camera.listen(self._on_camera_frame)
        needs_depth_camera = bool(self._use_depth_camera or self._tracking_metrics_dir is not None)
        if needs_depth_camera:
            try:
                self._depth_camera = self._spawn_depth_camera(world, vehicle)
                self._depth_camera.listen(self._on_depth_frame)
            except Exception as exc:
                self._depth_camera = None
                logging.warning(
                    "Depth camera unavailable for yolo_detect/tracking metrics, fallback to projection-only GT. Reason: %s",
                    exc,
                )
        else:
            self._depth_camera = None
            logging.info("YOLO detector does not consume depth_map_m; skipping depth camera for higher FPS.")
        self._init_navigation_agent(world, vehicle)
        self._capture_tracking_run_context(vehicle)
        self._enabled = True
        if self._nav_agent is not None:
            logging.info(
                "YOLO detection enabled with %s planner autopilot. Model: %s",
                self.config.yolo_nav_agent_type,
                model_path,
            )
        else:
            logging.info(
                "YOLO detection enabled with %s. Model: %s",
                self._control_backend_name,
                model_path,
            )

    def _init_navigation_agent(self, world, vehicle) -> None:
        self._spawn_points = world.get_map().get_spawn_points()
        yolo_backend = str(self.config.yolo_backend).lower()
        if yolo_backend == "tm":
            self._nav_agent = None
            self._enable_tm_autopilot(vehicle)
            self._tm_fallback_mode = True
            self._control_backend_name = "TM native autopilot"
            logging.info("YOLO backend initialized: TM native autopilot (explicit).")
            return

        if not self._spawn_points:
            logging.warning("No spawn points found for YOLO route planner, using TM autopilot fallback.")
            self._enable_tm_autopilot(vehicle)
            self._tm_fallback_mode = True
            self._control_backend_name = "TM autopilot fallback"
            return

        ensure_navigation_agent_imports()
        nav_type = self.config.yolo_nav_agent_type.lower()
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
            self._control_backend_name = f"{nav_type} planner autopilot"
        except Exception as exc:
            self._nav_agent = None
            self._enable_tm_autopilot(vehicle)
            self._tm_fallback_mode = True
            self._control_backend_name = "TM autopilot fallback"
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

    def _disable_tm_autopilot(self, vehicle) -> None:
        try:
            vehicle.set_autopilot(False, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(False)
        self._tm_autopilot_enabled = False

    def _set_tm_manual_brake_mode(self, vehicle: Any, enabled: bool, step_idx: int) -> None:
        if vehicle is None or not self._tm_fallback_mode:
            return
        if enabled:
            if self._tm_autopilot_enabled:
                self._disable_tm_autopilot(vehicle)
                logging.info(
                    "YOLO TM autopilot temporarily disabled at tick %d so supervisor brake owns vehicle control.",
                    step_idx,
                )
            return

        if not self._tm_autopilot_enabled:
            self._enable_tm_autopilot(vehicle)
            logging.info(
                "YOLO TM autopilot re-enabled at tick %d after supervisor brake released.",
                step_idx,
            )

    def _configure_nav_agent_traffic_lights(self) -> None:
        if self._nav_agent is None:
            return

        lights_configured, signs_configured = _configure_navigation_agent_ignore_stop_rules(self._nav_agent)
        if lights_configured or signs_configured:
            logging.info(
                "YOLO planner autopilot stop rules disabled (traffic_lights=%s, stop_signs=%s); only YOLO/supervisor may brake.",
                lights_configured,
                signs_configured,
            )
        else:
            logging.warning("Could not disable planner stop rules for YOLO autopilot; brake commands will still be suppressed.")

    def _configure_tm_traffic_lights(self, vehicle) -> None:
        if self.session is None:
            return

        tm = getattr(self.session, "traffic_manager", None)
        if tm is None:
            logging.warning("TrafficManager unavailable; cannot disable TM stop rules for YOLO autopilot.")
            return

        try:
            tm.ignore_lights_percentage(vehicle, 100.0)
            logging.info("YOLO TM fallback configured to ignore all traffic lights (100%%).")
        except Exception as exc:
            logging.warning("Failed to configure TM ignore_lights_percentage for YOLO fallback: %s", exc)
        ignore_signs_fn = getattr(tm, "ignore_signs_percentage", None)
        if callable(ignore_signs_fn):
            try:
                ignore_signs_fn(vehicle, 100.0)
                logging.info("YOLO TM fallback configured to ignore all stop signs (100%%).")
            except Exception as exc:
                logging.warning("Failed to configure TM ignore_signs_percentage for YOLO fallback: %s", exc)

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

    def _init_tracking_metrics_workspace(self, model_path: Path) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_stem = Path(model_path).stem
        tracker_stem = Path(str(self.config.yolo_tracker_config)).stem or "tracker"
        self._tracking_seq_name = f"carla_{self.name}_{tracker_stem}_{timestamp}"
        metrics_root = (Path(__file__).resolve().parent / "outputs" / "tracking_metrics" / self._tracking_seq_name).resolve()
        metrics_root.mkdir(parents=True, exist_ok=True)
        self._tracking_metrics_dir = metrics_root
        self._tracking_predictions_path = metrics_root / f"{model_stem}_{tracker_stem}_tracker_predictions.txt"
        secondary_tracker = str(self.config.yolo_secondary_tracker_config or "").strip()
        if secondary_tracker:
            secondary_tracker_stem = Path(secondary_tracker).stem or "secondary_tracker"
            self._secondary_tracking_predictions_path = (
                metrics_root / f"{model_stem}_{secondary_tracker_stem}_tracker_predictions.txt"
            )
        else:
            self._secondary_tracking_predictions_path = None
        self._tracking_ground_truth_path = metrics_root / "ground_truth.txt"
        self._tracking_seqinfo_path = metrics_root / "seqinfo.ini"
        self._tracking_metadata_path = metrics_root / "run_metadata.json"
        self._tracking_model_path = Path(model_path).resolve()
        self._tracking_gt_logs = []
        self._tracking_gt_frame_id = 0
        logging.info("Tracking metrics workspace: %s", metrics_root)

    @staticmethod
    def _transform_to_metadata(transform: Any) -> Optional[Dict[str, Dict[str, float]]]:
        if transform is None:
            return None
        try:
            location = transform.location
            rotation = transform.rotation
            return {
                "location": {
                    "x": round(float(location.x), 4),
                    "y": round(float(location.y), 4),
                    "z": round(float(location.z), 4),
                },
                "rotation": {
                    "pitch": round(float(rotation.pitch), 4),
                    "yaw": round(float(rotation.yaw), 4),
                    "roll": round(float(rotation.roll), 4),
                },
            }
        except Exception:
            return None

    def _capture_tracking_run_context(self, vehicle: Any) -> None:
        if vehicle is None:
            return
        try:
            transform = vehicle.get_transform()
        except Exception:
            transform = None
        self._tracking_initial_transform = self._transform_to_metadata(transform)
        if transform is not None:
            self._tracking_initial_spawn_index = self._nearest_spawn_index(transform.location)

    def _build_tracking_run_metadata(self) -> Dict[str, Any]:
        resolved_model = self._tracking_model_path
        if resolved_model is None:
            try:
                resolved_model = resolve_yolo_model_path(self.config.yolo_model_path)
            except Exception:
                resolved_model = None
        actual_route_destination = self._route_destination_index
        if actual_route_destination is None and str(self.config.yolo_backend).strip().lower() == "tm":
            actual_route_destination = "not_applicable_tm_backend"

        metadata: Dict[str, Any] = {
            "metadata_version": 1,
            "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seq_name": self._tracking_seq_name,
            "agent": self.name,
            "env_config_path": str(self.config.env_config_path),
            "map_name": str(self.config.map_name),
            "sync": bool(self.config.sync),
            "fixed_delta": float(self.config.fixed_delta),
            "no_rendering": bool(self.config.no_rendering),
            "seed": self.config.seed,
            "ticks": int(self.config.ticks),
            "tick_interval": float(self.config.tick_interval),
            "weather_preset": str(self.config.weather_preset),
            "vehicle_filter": str(self.config.vehicle_filter),
            "configured_spawn_point": int(self.config.spawn_point),
            "actual_initial_spawn_point": self._tracking_initial_spawn_index,
            "initial_transform": self._tracking_initial_transform,
            "configured_destination_point": int(self.config.destination_point),
            "actual_route_destination_point": actual_route_destination,
            "target_speed_kmh": float(self.config.target_speed_kmh),
            "tm_port": int(self.config.tm_port),
            "npc_vehicle_count": int(self.config.npc_vehicle_count),
            "npc_bike_count": int(self.config.npc_bike_count),
            "npc_motorbike_count": int(self.config.npc_motorbike_count),
            "npc_pedestrian_count": int(self.config.npc_pedestrian_count),
            "npc_enable_autopilot": bool(self.config.npc_enable_autopilot),
            "camera_width": int(self.config.camera_width),
            "camera_height": int(self.config.camera_height),
            "camera_fov": float(self.config.camera_fov),
            "camera_mount": {
                "x_m": 1.5,
                "y_m": 0.0,
                "z_m": 2.2,
                "pitch_deg": -8.0,
            },
            "yolo_backend": str(self.config.yolo_backend),
            "yolo_nav_agent_type": str(self.config.yolo_nav_agent_type),
            "yolo_tracker_config": str(self.config.yolo_tracker_config),
            "yolo_secondary_tracker_config": str(self.config.yolo_secondary_tracker_config or ""),
            "yolo_model_path": str(self.config.yolo_model_path),
            "resolved_yolo_model_path": str(resolved_model) if resolved_model is not None else "",
            "yolo_inference_imgsz": int(self.config.yolo_inference_imgsz),
            "yolo_inference_every_n_ticks": int(self.config.yolo_inference_every_n_ticks),
            "detector_uses_depth_input": bool(self._use_depth_camera),
            "gt_occlusion_filter": {
                "enabled": bool(self._depth_camera is not None),
                "method": "projected_3d_bbox_visible_area_plus_depth_map",
                "min_visible_area_ratio": 0.35,
                "min_depth_visible_ratio": 0.20,
                "max_gt_distance_m": 50.0,
                "min_gt_bbox_dim_px": 10,
                "min_gt_bbox_area_px": 400,
            },
            "tracking_metrics_dir": str(self._tracking_metrics_dir) if self._tracking_metrics_dir is not None else "",
            "predictions_path": str(self._tracking_predictions_path) if self._tracking_predictions_path is not None else "",
            "secondary_predictions_path": (
                str(self._secondary_tracking_predictions_path)
                if self._secondary_tracking_predictions_path is not None
                else ""
            ),
            "ground_truth_path": str(self._tracking_ground_truth_path) if self._tracking_ground_truth_path is not None else "",
            "seqinfo_path": str(self._tracking_seqinfo_path) if self._tracking_seqinfo_path is not None else "",
        }
        return metadata

    def _camera_intrinsics_matrix(self, image_width: int, image_height: int) -> Any:
        if np is None:
            return None
        fov_rad = math.radians(float(self.config.camera_fov))
        fx = (float(image_width) / 2.0) / max(math.tan(fov_rad / 2.0), 1e-6)
        fy = fx
        cx = float(image_width) / 2.0
        cy = float(image_height) / 2.0
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _project_world_location_to_image(
        self,
        world_location: Any,
        world_to_camera: Any,
        intrinsics: Any,
    ) -> Optional[tuple[float, float, float]]:
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

    @staticmethod
    def _infer_gt_class_name(actor: Any) -> Optional[str]:
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

    @staticmethod
    def _projected_depth_visibility_ratio(
        projected_points: list[tuple[float, float, float]],
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

    def _log_ground_truth_frame(
        self,
        frame_shape: tuple[int, int, int],
        ego_vehicle: Any,
        depth_map_m: Any = None,
    ) -> None:
        if (
            np is None
            or self.session is None
            or self.session.world is None
            or self._camera is None
            or ego_vehicle is None
        ):
            return

        image_h = int(frame_shape[0])
        image_w = int(frame_shape[1])
        if image_h <= 0 or image_w <= 0:
            return
        self._tracking_image_size = (image_w, image_h)

        intrinsics = self._camera_intrinsics_matrix(image_w, image_h)
        if intrinsics is None:
            return

        try:
            world_to_camera = np.array(
                self._camera.get_transform().get_inverse_matrix(),
                dtype=np.float64,
            )
        except Exception:
            return

        self._tracking_gt_frame_id += 1
        frame_id = int(self._tracking_gt_frame_id)

        try:
            actors = self.session.world.get_actors()
        except Exception:
            return






        MAX_GT_DISTANCE_M = 40.0


        MIN_GT_BBOX_DIM = 10


        MIN_GT_BBOX_AREA = 600




        MIN_VISIBLE_AREA_RATIO = 0.50



        MAX_GT_BBOX_ASPECT_RATIO = 6.0


        try:
            ego_location = ego_vehicle.get_location()
        except Exception:
            ego_location = None

        for actor in actors:
            try:
                actor_id = int(actor.id)
            except Exception:
                continue
            if actor_id == int(ego_vehicle.id):
                continue

            class_name = self._infer_gt_class_name(actor)
            if class_name is None:
                continue


            if ego_location is not None:
                try:
                    actor_location = actor.get_location()
                    distance = ego_location.distance(actor_location)
                    if distance > MAX_GT_DISTANCE_M:
                        continue
                except Exception:
                    pass

            bbox_3d = getattr(actor, "bounding_box", None)
            if bbox_3d is None:
                continue

            try:
                bbox_vertices = bbox_3d.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            if not bbox_vertices:
                continue

            projected_points: list[tuple[float, float, float]] = []
            for vertex in bbox_vertices:
                projected = self._project_world_location_to_image(
                    world_location=vertex,
                    world_to_camera=world_to_camera,
                    intrinsics=intrinsics,
                )
                if projected is None:
                    continue
                projected_points.append(projected)

            try:
                center_projected = self._project_world_location_to_image(
                    world_location=actor.get_location(),
                    world_to_camera=world_to_camera,
                    intrinsics=intrinsics,
                )
                if center_projected is not None:
                    projected_points.append(center_projected)
            except Exception:
                pass

            if len(projected_points) < 2:
                continue

            xs = [p[0] for p in projected_points]
            ys = [p[1] for p in projected_points]
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


            if width < MIN_GT_BBOX_DIM or height < MIN_GT_BBOX_DIM:
                continue

            if width * height < MIN_GT_BBOX_AREA:
                continue

            aspect = max(width, height) / max(min(width, height), 1.0)
            if aspect > MAX_GT_BBOX_ASPECT_RATIO:
                continue

            cx = x1 + width / 2.0
            cy = y1 + height / 2.0
            if cx < 0.0 or cx >= float(image_w) or cy < 0.0 or cy >= float(image_h):
                continue

            if raw_area > 1.0:
                visible_area_ratio = (width * height) / raw_area
                if visible_area_ratio < MIN_VISIBLE_AREA_RATIO:
                    continue

            depth_visibility = self._projected_depth_visibility_ratio(
                projected_points=projected_points,
                depth_map_m=depth_map_m,
                image_w=image_w,
                image_h=image_h,
            )
            if depth_visibility is not None and depth_visibility < 0.20:
                continue




            gt_line = (
                f"{frame_id},{actor_id},"
                f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1,{class_name}"
            )
            self._tracking_gt_logs.append(gt_line)

    def _save_tracking_metrics_outputs(self) -> None:
        if self._detector is not None and self._tracking_predictions_path is not None:
            try:
                self._detector.save_tracking_metrics_log(str(self._tracking_predictions_path))
            except Exception as exc:
                logging.warning("Failed to save tracker predictions log: %s", exc)
        if self._secondary_detector is not None and self._secondary_tracking_predictions_path is not None:
            try:
                self._secondary_detector.save_tracking_metrics_log(str(self._secondary_tracking_predictions_path))
            except Exception as exc:
                logging.warning("Failed to save secondary tracker predictions log: %s", exc)

        if self._tracking_ground_truth_path is not None:
            try:
                self._tracking_ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
                with self._tracking_ground_truth_path.open("w", encoding="utf-8") as file_obj:
                    for line in self._tracking_gt_logs:
                        file_obj.write(f"{line}\n")
            except Exception as exc:
                logging.warning("Failed to save GT tracking log: %s", exc)

        seq_len = int(self._tracking_gt_frame_id)
        image_w, image_h = self._tracking_image_size
        if self._tracking_seqinfo_path is not None and seq_len > 0 and image_w > 0 and image_h > 0:
            try:
                fps = int(round(1.0 / max(1e-3, float(self.config.fixed_delta)))) if self.config.sync else int(round(self.config.video_fps))
                fps = max(1, fps)
                with self._tracking_seqinfo_path.open("w", encoding="utf-8") as file_obj:
                    file_obj.write("[Sequence]\n")
                    file_obj.write(f"name={self._tracking_seq_name}\n")
                    file_obj.write(f"imDir=img1\n")
                    file_obj.write(f"frameRate={fps}\n")
                    file_obj.write(f"seqLength={seq_len}\n")
                    file_obj.write(f"imWidth={image_w}\n")
                    file_obj.write(f"imHeight={image_h}\n")
                    file_obj.write("imExt=.jpg\n")
            except Exception as exc:
                logging.warning("Failed to save seqinfo.ini for tracking metrics: %s", exc)

        if self._tracking_metadata_path is not None:
            try:
                self._tracking_metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata = self._build_tracking_run_metadata()
                with self._tracking_metadata_path.open("w", encoding="utf-8") as file_obj:
                    json.dump(metadata, file_obj, indent=2, sort_keys=True)
                    file_obj.write("\n")
            except Exception as exc:
                logging.warning("Failed to save tracking run metadata: %s", exc)

        if self._tracking_metrics_dir is not None:
            logging.info(
                "[Metrics] Tracking logs exported: predictions=%s | ground_truth=%s | seqinfo=%s | metadata=%s",
                self._tracking_predictions_path,
                self._tracking_ground_truth_path,
                self._tracking_seqinfo_path,
                self._tracking_metadata_path,
            )

    def run_step(self, step_idx: int) -> None:
        """
        Main detection & control loop for YOLO agent.

        Workflow:
        1. Read frame + depth from camera
        2. Run YOLO detection
        3. Build danger_polygon from TrafficSupervisor
        4. Compute supervisor/advisory state
        5. Pass through control from nav_agent or TM autopilot
        6. Draw annotations including YELLOW CORRIDOR
        7. Display & log
        """
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("YOLO agent waiting for CARLA runtime.")
            return




        frame_bgr, depth_map_m = self._read_latest_frame()
        if frame_bgr is None:
            if not self._waiting_frame_logged:
                logging.info("YOLO waiting for first camera frame...")
                self._waiting_frame_logged = True
            return




        vehicle = self.session.ego_vehicle if self.session is not None else None
        current_steer = None
        speed_kmh = None
        display_steer = 0.0
        display_throttle = 0.0
        display_brake = 0.0

        if vehicle is not None:
            try:

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




        detect_every_n = max(1, int(self.config.yolo_inference_every_n_ticks))
        should_run_detector = (
            self._last_detection_step is None
            or detect_every_n <= 1
            or (step_idx - int(self._last_detection_step)) >= detect_every_n
        )
        if should_run_detector:
            detector_t0 = time.perf_counter()
            detections, detector_emergency = self._detector.detect_and_evaluate(
                frame_bgr,
                distance_threshold=None,
                depth_map_m=depth_map_m,
                vehicle_steer=current_steer,
                speed_kmh=speed_kmh,
            )
            if self._secondary_detector is not None:
                try:
                    self._secondary_detector.detect_and_evaluate(
                        frame_bgr,
                        distance_threshold=None,
                        depth_map_m=depth_map_m,
                        vehicle_steer=current_steer,
                        speed_kmh=speed_kmh,
                    )
                except Exception as exc:
                    logging.warning("Secondary YOLO tracker failed on tick %d: %s", step_idx, exc)
            self._log_ground_truth_frame(frame_bgr.shape, vehicle, depth_map_m=depth_map_m)
            self._last_detection_runtime_ms = (time.perf_counter() - detector_t0) * 1000.0
            debug_info = {}
            if hasattr(self._detector, "get_last_debug_info"):
                debug_info = self._detector.get_last_debug_info() or {}
            debug_info = dict(debug_info)
            debug_info["detector_cache_hit"] = False
            debug_info["detector_cache_age_ticks"] = 0
            debug_info["detector_last_run_ms"] = self._last_detection_runtime_ms
            self._cached_detections = [dict(det) for det in detections]
            self._cached_detector_emergency = bool(detector_emergency)
            self._cached_debug_info = dict(debug_info)
            self._last_detection_step = int(step_idx)
        else:
            detections = [dict(det) for det in self._cached_detections]
            detector_emergency = bool(self._cached_detector_emergency)
            debug_info = dict(self._cached_debug_info)
            debug_info["detector_cache_hit"] = True
            debug_info["detector_cache_age_ticks"] = (
                step_idx - int(self._last_detection_step) if self._last_detection_step is not None else 0
            )
            debug_info["detector_last_run_ms"] = self._last_detection_runtime_ms




        supervisor_brake = 0.0
        supervisor_state = "n/a"
        supervisor_reason = "n/a"
        hard_supervisor_emergency = False
        red_hard_stop_active = False
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


                sup_dets = to_supervisor_detections(detections)
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
                red_hard_stop_active = bool(sup_debug.get("red_hard_stop_active", False))
                hard_supervisor_emergency = (
                    supervisor_brake >= 0.60 and supervisor_reason == "obstacle"
                )

            except Exception as exc:
                sup_debug = {}
                self._last_supervisor_debug_info = {}
                supervisor_brake = 0.0
                logging.warning("TrafficSupervisor compute failed: %s", exc)

        is_emergency = bool(detector_emergency or hard_supervisor_emergency)




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
            autopilot_brake = float(getattr(nav_control, "brake", 0.0))
            final_control = _control_without_autopilot_brake(nav_control)
            hold_hand_brake = bool(
                red_hard_stop_active
                and supervisor_brake >= 0.99
                and str(supervisor_reason).strip().lower() in ("stop_line", "red_light", "traffic_light_red")
                and speed_kmh is not None
                and float(speed_kmh) <= 2.0
            )


            if is_emergency or supervisor_brake > 0.0:
                emergency_floor = 0.6 if is_emergency else 0.0
                final_control = _control_without_autopilot_brake(nav_control)
                final_control.throttle = 0.0
                final_control.brake = float(
                    clamp(
                        max(float(supervisor_brake), float(emergency_floor)),
                        0.0,
                        1.0,
                    )
                )
                if hold_hand_brake:
                    final_control.brake = 1.0
                final_control.hand_brake = bool(hold_hand_brake)
                logging.debug(
                    "[TICK %d] Supervisor override planner control: emergency=%s supervisor_brake=%.2f planner_brake_suppressed=%.2f final_brake=%.2f",
                    step_idx,
                    bool(is_emergency),
                    float(supervisor_brake),
                    autopilot_brake,
                    float(final_control.brake),
                )
            elif autopilot_brake > 1e-6 or bool(getattr(nav_control, "hand_brake", False)):
                logging.debug(
                    "[TICK %d] Suppressed planner autopilot brake=%.2f hand_brake=%s.",
                    step_idx,
                    autopilot_brake,
                    bool(getattr(nav_control, "hand_brake", False)),
                )

            vehicle.apply_control(final_control)
            display_steer = float(final_control.steer)
            display_throttle = float(final_control.throttle)
            display_brake = float(final_control.brake)
        elif vehicle is not None and self._tm_fallback_mode:
            current_control = vehicle.get_control()
            autopilot_brake = float(getattr(current_control, "brake", 0.0))
            final_control = _control_without_autopilot_brake(current_control)
            supervisor_manual_override = bool(is_emergency or supervisor_brake > 0.0)
            self._set_tm_manual_brake_mode(vehicle, supervisor_manual_override, step_idx)
            hold_hand_brake = bool(
                red_hard_stop_active
                and supervisor_brake >= 0.99
                and str(supervisor_reason).strip().lower() in ("stop_line", "red_light", "traffic_light_red")
                and speed_kmh is not None
                and float(speed_kmh) <= 2.0
            )
            if supervisor_manual_override:
                emergency_floor = 0.6 if is_emergency else 0.0
                final_control.throttle = 0.0
                final_control.brake = float(
                    clamp(
                        max(float(supervisor_brake), float(emergency_floor)),
                        0.0,
                        1.0,
                    )
                )
                if hold_hand_brake:
                    final_control.brake = 1.0
                final_control.hand_brake = bool(hold_hand_brake)
                logging.debug(
                    "[TICK %d] Supervisor override TM fallback: emergency=%s supervisor_brake=%.2f tm_brake_suppressed=%.2f final_brake=%.2f",
                    step_idx,
                    bool(is_emergency),
                    float(supervisor_brake),
                    autopilot_brake,
                    float(final_control.brake),
                )
            elif autopilot_brake > 1e-6 or bool(getattr(current_control, "hand_brake", False)):
                logging.debug(
                    "[TICK %d] Suppressed TM autopilot brake=%.2f hand_brake=%s.",
                    step_idx,
                    autopilot_brake,
                    bool(getattr(current_control, "hand_brake", False)),
                )

            vehicle.apply_control(final_control)
            display_steer = float(final_control.steer)
            display_throttle = float(final_control.throttle)
            display_brake = float(final_control.brake)




        hud_fps = self._update_hud_fps()
        steer_angle_deg = self._steer_to_angle_deg(vehicle, display_steer)
        speed_text = f"{float(speed_kmh):.1f} km/h" if speed_kmh is not None else "n/a"
        steer_text = f"STEER={display_steer:+.3f}"
        if steer_angle_deg is not None:
            steer_text = f"{steer_text} ({steer_angle_deg:+.1f} deg)"
        yellow_drew = False
        annotated_frame = None

        if self.config.yolo_visualize:
            annotated_frame = frame_bgr.copy()
            if self.config.yolo_draw_overlay:
                _draw_red_light_zone_rois(
                    annotated_frame,
                    sup_debug,
                    visible=bool(self.config.yolo_show_red_light_rois),
                )

                yellow_drew = _draw_yellow_danger_corridor(
                    annotated_frame,
                    debug_info,
                    sup_debug,
                )

                if not yellow_drew:
                    logging.debug("[TICK %d] Yellow corridor not drawn (no polygon available)", step_idx)
                else:
                    logging.debug("[TICK %d] Yellow corridor drawn successfully", step_idx)

                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    class_name = det["class_name"]
                    confidence = det["confidence"]
                    distance = det["distance"]
                    in_danger_roi = bool(det.get("in_danger_roi", False))
                    danger_match = bool(det.get("danger_match", False))

                    label = f"{class_name} {confidence:.2f} ({distance:.1f}m)"

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
                    else:
                        color = (0, 255, 0)

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
                    cv2.circle(annotated_frame, (int((bx1 + bx2) / 2), by2), 5, color, -1)

                brake_level = float(clamp(display_brake, 0.0, 1.0))
                if self._traffic_supervisor is not None:
                    supervisor_state_upper = str(sup_debug.get("state", "cruising")).upper()
                    status_text = f"SUPERVISOR {supervisor_state_upper} {brake_level:.2f}"
                    if supervisor_reason not in ("n/a", "none", ""):
                        status_text = f"{status_text} ({supervisor_reason})"
                    status_color = (
                        0,
                        int(round(255.0 * (1.0 - brake_level))),
                        int(round(255.0 * brake_level)),
                    )
                elif is_emergency:
                    status_text = f"EMERGENCY BRAKE {brake_level:.2f}"
                    status_color = (0, 0, 255)
                else:
                    status_text = f"NORMAL {brake_level:.2f}"
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

            cv2.imshow(self._window_name, annotated_frame)
            cv2.waitKey(1)




        if step_idx % 20 == 0:
            logging.info(
                "yolo_detect tick=%d fps=%.1f speed=%s steer=%.3f angle=%s throttle=%.2f brake=%.2f "
                "detections=%d emergency=%s supervisor_brake=%.2f state=%s reason=%s yellow_polygon=%s "
                "detector_cache=%s cache_age=%s detector_ms=%s",
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
                "drawn" if yellow_drew else ("raw" if self.config.yolo_visualize else "disabled"),
                bool(debug_info.get("detector_cache_hit", False)),
                debug_info.get("detector_cache_age_ticks", 0),
                f"{float(debug_info['detector_last_run_ms']):.1f}ms"
                if debug_info.get("detector_last_run_ms") is not None
                else "n/a",
            )

    def teardown(self) -> None:
        self._save_tracking_metrics_outputs()
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
    CILYoloAgent.name: CILYoloAgent,
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

    with path.open("r", encoding="utf-8-sig") as fp:
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
        help="Choose lane_follow/cil/cil_yolo/autopilot/yolo_detect agent mode.",
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
        "--yolo-imgsz",
        type=int,
        default=None,
        help="Override YOLO/RT-DETR inference image size. For static TensorRT engines leave unset or match export size.",
    )
    parser.add_argument(
        "--yolo-every-n-ticks",
        type=int,
        default=None,
        help="Run YOLO detection every N ticks and reuse cached detections in between.",
    )
    parser.add_argument(
        "--yolo-tracker",
        default=None,
        help="Ultralytics tracker config for yolo_detect, e.g. botsort.yaml or bytetrack.yaml.",
    )
    parser.add_argument(
        "--yolo-secondary-tracker",
        default=None,
        help=(
            "Optional second Ultralytics tracker config to run on the same frames for fair same-sequence "
            "metrics, e.g. bytetrack.yaml while --yolo-tracker is botsort.yaml."
        ),
    )
    parser.add_argument(
        "--gtnet-model-path",
        default=None,
        help="Path to GTNet trajectory checkpoint used as an extra CIL/CIL+YOLO safety supervisor.",
    )
    parser.add_argument(
        "--enable-gtnet",
        dest="gtnet_enabled",
        action="store_true",
        default=None,
        help="Enable GTNet trajectory supervisor.",
    )
    parser.add_argument(
        "--disable-gtnet",
        dest="gtnet_enabled",
        action="store_false",
        help="Disable GTNet trajectory supervisor.",
    )
    parser.add_argument(
        "--gtnet-every-n-ticks",
        type=int,
        default=None,
        help="Run GTNet trajectory inference every N ticks and reuse cached risk in between.",
    )
    parser.add_argument(
        "--gtnet-history-frames",
        type=int,
        default=None,
        help="Number of history frames expected by GTNet.",
    )
    parser.add_argument(
        "--gtnet-expected-dt",
        type=float,
        default=None,
        help="Seconds between sampled GTNet history frames.",
    )
    parser.add_argument(
        "--gtnet-adjacency-mode",
        choices=["fixed", "adaptive", "checkpoint"],
        default=None,
        help="Runtime adjacency used by GTNet. Use fixed for checkpoints trained on fixed-radius datasets.",
    )
    parser.add_argument(
        "--gtnet-fixed-adj-radius",
        type=float,
        default=None,
        help="Fixed graph radius in metres when --gtnet-adjacency-mode=fixed.",
    )
    parser.add_argument(
        "--gtnet-max-actor-distance",
        type=float,
        default=None,
        help="Maximum live actor distance considered by GTNet.",
    )
    parser.add_argument(
        "--gtnet-draw-debug",
        dest="gtnet_draw_debug",
        action="store_true",
        default=None,
        help="Draw GTNet threat trajectories in the CARLA world debug view.",
    )
    parser.add_argument(
        "--no-gtnet-draw-debug",
        dest="gtnet_draw_debug",
        action="store_false",
        help="Disable GTNet world debug trajectory drawing.",
    )
    parser.add_argument(
        "--yolo-visualize",
        dest="yolo_visualize",
        action="store_true",
        default=None,
        help="Show YOLO camera window during yolo_detect.",
    )
    parser.add_argument(
        "--no-yolo-visualize",
        dest="yolo_visualize",
        action="store_false",
        help="Disable YOLO camera window for higher FPS.",
    )
    parser.add_argument(
        "--yolo-draw-overlay",
        dest="yolo_draw_overlay",
        action="store_true",
        default=None,
        help="Draw YOLO boxes/HUD overlays when visualization is enabled.",
    )
    parser.add_argument(
        "--no-yolo-draw-overlay",
        dest="yolo_draw_overlay",
        action="store_false",
        help="Show raw YOLO camera feed without overlay drawing.",
    )
    parser.add_argument(
        "--eval-online",
        action="store_true",
        default=False,
        help="Print detailed online metrics summary at teardown.",
    )
    parser.add_argument(
        "--yolo-show-red-light-rois",
        dest="yolo_show_red_light_rois",
        action="store_true",
        default=None,
        help="Draw the two red-light ROI zones on the YOLO debug overlay.",
    )
    parser.add_argument(
        "--no-yolo-show-red-light-rois",
        dest="yolo_show_red_light_rois",
        action="store_false",
        help="Hide the two red-light ROI zones while keeping red-light stop logic enabled.",
    )
    parser.add_argument(
        "--opencv-route-map",
        dest="opencv_route_map",
        action="store_true",
        default=None,
        help="Enable the separate OpenCV 'CIL Route Map' window (runtime: press 'r' when using cil_yolo).",
    )
    parser.add_argument(
        "--no-opencv-route-map",
        dest="opencv_route_map",
        action="store_false",
        help="Disable the OpenCV route map window at startup.",
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
    parser.add_argument(
        "--cil-lane-constrain-blended",
        dest="cil_lane_constrain_blended",
        action="store_true",
        default=None,
        help="Enable smooth lane-constrain trajectory blending for CIL waypoints.",
    )
    parser.add_argument(
        "--no-cil-lane-constrain-blended",
        dest="cil_lane_constrain_blended",
        action="store_false",
        help="Disable smooth lane-constrain blending and use raw model waypoints.",
    )
    parser.add_argument(
        "--cil-lane-constrain-strength",
        type=float,
        default=None,
        help="Intervention strength of the lane constraint blending (0.0 to 1.0, default: 1.0).",
    )
    parser.add_argument(
        "--cil-use-carla-waypoints",
        dest="cil_use_carla_waypoints",
        action="store_true",
        default=None,
        help="Use CARLA route waypoints for CIL pure-pursuit control.",
    )
    parser.add_argument(
        "--no-cil-use-carla-waypoints",
        dest="cil_use_carla_waypoints",
        action="store_false",
        help="Use CIL model-predicted waypoints for pure-pursuit control.",
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
        "--autopilot-backend",
        choices=["planner", "tm"],
        default=None,
        help="Autopilot control backend for agent=autopilot. 'planner' uses BasicAgent/BehaviorAgent, 'tm' uses CARLA Traffic Manager autopilot.",
    )
    parser.add_argument(
        "--yolo-disable-autopilot-red-light",
        dest="yolo_disable_autopilot_red_light",
        action="store_true",
        default=None,
        help="Deprecated compatibility flag; yolo_detect now always suppresses autopilot stop/brake handling.",
    )
    parser.add_argument(
        "--no-yolo-disable-autopilot-red-light",
        dest="yolo_disable_autopilot_red_light",
        action="store_false",
        default=None,
        help="Deprecated compatibility flag; autopilot stop/brake handling stays suppressed.",
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

    if collect_data and args.agent == "autopilot":
        if destination_point_cfg >= 0:
            logging.info(
                "Collect-data autopilot ignores configured destination_point=%d and uses roaming destinations instead.",
                destination_point_cfg,
            )
        destination_point_cfg = -1
        npc_vehicle_count = 0
        npc_bike_count = 0
        npc_motorbike_count = 0
        npc_pedestrian_count = 0
        npc_enable_autopilot = False
        logging.info(
            "Collect-data autopilot forcing zero traffic_spawn counts (vehicles=0 bikes=0 motorbikes=0 pedestrians=0)."
        )

    is_cil_route_agent = args.agent in {"cil", "cil_yolo"}

    if is_cil_route_agent and spawn_point_cfg < 0:
        logging.warning(
            "CIL/CIL+YOLO route map requires deterministic S point. vehicle.spawn_point < 0, forcing spawn_point=0."
        )
        spawn_point_cfg = 0

    if is_cil_route_agent and destination_point_cfg < 0:
        destination_point_cfg = spawn_point_cfg + 1
        logging.warning(
            "CIL/CIL+YOLO route map requires deterministic D point. vehicle.destination_point < 0, "
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

    cil_use_pure_pursuit = _to_bool(
        _cfg_get(env_cfg, "cil", "use_pure_pursuit", False),
        False,
    )
    cil_use_carla_waypoints = _to_bool(
        args.cil_use_carla_waypoints,
        _to_bool(
            _cfg_get(
                env_cfg,
                "cil",
                "use_carla_waypoints",
                _cfg_get(env_cfg, "cil", "waypoint_carla", False),
            ),
            False,
        ),
    )
    cil_lane_constrain_blended = _to_bool(
        args.cil_lane_constrain_blended,
        _to_bool(_cfg_get(env_cfg, "cil", "lane_constrain_blended", False), False),
    )
    cil_lane_constrain_strength = float(
        pick(
            args.cil_lane_constrain_strength,
            "cil",
            "lane_constrain_strength",
            1.0,
        )
    )

    nav_agent_type = str(pick(args.nav_agent_type, "cil", "nav_agent_type", "basic")).lower()
    if nav_agent_type not in {"basic", "behavior"}:
        logging.warning("Unsupported nav_agent_type=%s. Falling back to 'basic'.", nav_agent_type)
        nav_agent_type = "basic"
    autopilot_backend = str(pick(args.autopilot_backend, "autopilot", "backend", "planner")).lower()
    if autopilot_backend not in {"planner", "tm"}:
        logging.warning("Unsupported autopilot_backend=%s. Falling back to 'planner'.", autopilot_backend)
        autopilot_backend = "planner"
    yolo_backend = str(_cfg_get(env_cfg, "yolo", "backend", "planner")).lower()
    if yolo_backend not in {"planner", "tm"}:
        logging.warning("Unsupported yolo.backend=%s. Falling back to 'planner'.", yolo_backend)
        yolo_backend = "planner"
    yolo_nav_agent_type = str(_cfg_get(env_cfg, "yolo", "nav_agent_type", nav_agent_type)).lower()
    if yolo_nav_agent_type not in {"basic", "behavior"}:
        logging.warning(
            "Unsupported yolo.nav_agent_type=%s. Falling back to '%s'.",
            yolo_nav_agent_type,
            nav_agent_type,
        )
        yolo_nav_agent_type = nav_agent_type

    yolo_disable_autopilot_red_light = _to_bool(
        args.yolo_disable_autopilot_red_light,
        _to_bool(_cfg_get(env_cfg, "yolo", "disable_autopilot_red_light", False), False),
    )
    yolo_inference_every_n_ticks = max(
        1,
        int(
            pick(
                args.yolo_every_n_ticks,
                "yolo",
                "inference_every_n_ticks",
                1,
            )
        ),
    )
    yolo_visualize = _to_bool(
        args.yolo_visualize,
        _to_bool(
            _cfg_get(
                env_cfg,
                "yolo",
                "debug_window",
                _cfg_get(env_cfg, "yolo", "visualize", True),
            ),
            True,
        ),
    )
    yolo_draw_overlay = _to_bool(
        args.yolo_draw_overlay,
        _to_bool(
            _cfg_get(
                env_cfg,
                "yolo",
                "debug_overlay",
                _cfg_get(env_cfg, "yolo", "draw_overlay", True),
            ),
            True,
        ),
    )
    if not yolo_visualize:
        yolo_draw_overlay = False
    yolo_show_red_light_rois = _to_bool(
        args.yolo_show_red_light_rois,
        _to_bool(_cfg_get(env_cfg, "yolo", "show_red_light_rois", True), True),
    )
    yolo_inference_imgsz = max(
        0,
        int(
            pick(
                args.yolo_imgsz,
                "yolo",
                "inference_imgsz",
                0,
            )
        ),
    )
    def normalize_yolo_tracker_config(value: Any, default: str = "") -> str:
        tracker_config = str(value if value is not None else default).strip()
        if not tracker_config:
            return ""
        if tracker_config.lower() in {"botsort", "bot-sort", "bot_sort"}:
            return "botsort.yaml"
        if tracker_config.lower() in {"bytetrack", "byte-track", "byte_track"}:
            return "bytetrack.yaml"
        return tracker_config

    yolo_tracker_config = normalize_yolo_tracker_config(
        pick(args.yolo_tracker, "yolo", "tracker", "botsort.yaml"),
        default="botsort.yaml",
    )
    if not yolo_tracker_config:
        yolo_tracker_config = "botsort.yaml"
    yolo_secondary_tracker_config = normalize_yolo_tracker_config(
        pick(args.yolo_secondary_tracker, "yolo", "secondary_tracker", ""),
    )
    if yolo_secondary_tracker_config == yolo_tracker_config:
        yolo_secondary_tracker_config = ""

    gtnet_enabled = _to_bool(
        args.gtnet_enabled,
        _to_bool(_cfg_get(env_cfg, "gtnet", "enabled", False), False),
    )
    gtnet_model_path = str(
        pick(args.gtnet_model_path, "gtnet", "model_path", "models/ablation_111_best.pt")
    )
    gtnet_inference_every_n_ticks = max(
        1,
        int(
            pick(
                args.gtnet_every_n_ticks,
                "gtnet",
                "inference_every_n_ticks",
                3,
            )
        ),
    )
    gtnet_history_frames = max(
        1,
        int(pick(args.gtnet_history_frames, "gtnet", "history_frames", 40)),
    )
    gtnet_expected_dt = max(
        1e-3,
        float(pick(args.gtnet_expected_dt, "gtnet", "expected_dt", 0.1)),
    )
    gtnet_adjacency_mode = str(
        pick(args.gtnet_adjacency_mode, "gtnet", "adjacency_mode", "fixed")
    ).strip().lower().replace("-", "_")
    if gtnet_adjacency_mode not in {"fixed", "adaptive", "checkpoint"}:
        logging.warning(
            "Unsupported gtnet.adjacency_mode=%s. Falling back to 'fixed'.",
            gtnet_adjacency_mode,
        )
        gtnet_adjacency_mode = "fixed"
    gtnet_fixed_adjacency_radius_m = max(
        1.0,
        float(pick(args.gtnet_fixed_adj_radius, "gtnet", "fixed_adjacency_radius_m", 100.0)),
    )
    gtnet_max_actor_distance_m = max(
        1.0,
        float(pick(args.gtnet_max_actor_distance, "gtnet", "max_actor_distance_m", 100.0)),
    )
    gtnet_draw_debug = _to_bool(
        args.gtnet_draw_debug,
        _to_bool(_cfg_get(env_cfg, "gtnet", "draw_debug", False), False),
    )

    weather_preset = str(_cfg_get(env_cfg, "weather", "preset", "ClearNoon"))
    traffic_light_red_time = float(_cfg_get(env_cfg, "traffic_lights", "red_time", 10.0))
    traffic_light_green_time = float(_cfg_get(env_cfg, "traffic_lights", "green_time", 15.0))
    traffic_light_yellow_time = float(_cfg_get(env_cfg, "traffic_lights", "yellow_time", 3.0))
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
    cil_opencv_route_map = _to_bool(
        pick(args.opencv_route_map, "runtime", "cil_opencv_route_map", True),
        True,
    )
    cil_world_waypoint_debug = _to_bool(
        _cfg_get(env_cfg, "runtime", "cil_world_waypoint_debug", True),
        True,
    )
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
        eval_online=args.eval_online or _to_bool(
            _cfg_get(env_cfg, "runtime", "eval_online", False), False
        ),
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
        cil_opencv_route_map=cil_opencv_route_map,
        cil_world_waypoint_debug=cil_world_waypoint_debug,
        cil_enable_telemetry_csv=cil_enable_telemetry_csv,
        cil_profile_tick_timing=cil_profile_tick_timing,
        cil_profile_log_interval_ticks=cil_profile_log_interval_ticks,
        recovery_interval_frames=recovery_interval_frames,
        recovery_duration_frames=recovery_duration_frames,
        recovery_steer_offset=recovery_steer_offset,
        autopilot_backend=autopilot_backend,
        nav_agent_type=nav_agent_type,
        yolo_backend=yolo_backend,
        yolo_nav_agent_type=yolo_nav_agent_type,
        yolo_disable_autopilot_red_light=bool(yolo_disable_autopilot_red_light),
        yolo_inference_every_n_ticks=int(yolo_inference_every_n_ticks),
        yolo_visualize=bool(yolo_visualize),
        yolo_draw_overlay=bool(yolo_draw_overlay),
        yolo_show_red_light_rois=bool(yolo_show_red_light_rois),
        yolo_inference_imgsz=int(yolo_inference_imgsz),
        yolo_tracker_config=yolo_tracker_config,
        yolo_secondary_tracker_config=yolo_secondary_tracker_config,
        gtnet_enabled=bool(gtnet_enabled),
        gtnet_model_path=str(gtnet_model_path),
        gtnet_inference_every_n_ticks=int(gtnet_inference_every_n_ticks),
        gtnet_draw_debug=bool(gtnet_draw_debug),
        gtnet_history_frames=int(gtnet_history_frames),
        gtnet_expected_dt=float(gtnet_expected_dt),
        gtnet_adjacency_mode=str(gtnet_adjacency_mode),
        gtnet_fixed_adjacency_radius_m=float(gtnet_fixed_adjacency_radius_m),
        gtnet_max_actor_distance_m=float(gtnet_max_actor_distance_m),
        cil_command_prep_time_s=cil_command_prep_time_s,
        cil_command_trigger_min_m=cil_command_trigger_min_m,
        cil_command_trigger_max_m=cil_command_trigger_max_m,
        cil_use_pure_pursuit=cil_use_pure_pursuit,
        cil_use_carla_waypoints=cil_use_carla_waypoints,
        cil_lane_constrain_blended=cil_lane_constrain_blended,
        cil_lane_constrain_strength=cil_lane_constrain_strength,
        traffic_light_red_time=traffic_light_red_time,
        traffic_light_green_time=traffic_light_green_time,
        traffic_light_yellow_time=traffic_light_yellow_time,
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
