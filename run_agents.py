from __future__ import annotations

import argparse
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
    from core_perception.cnn_model import NvidiaCNN, NvidiaCNNV2
except Exception:
    NvidiaCNN = None
    NvidiaCNNV2 = None

try:
    from core_perception.yolo_detector import YoloDetector
except Exception as exc:
    logging.warning("Failed to import YoloDetector: %s", exc)
    YoloDetector = None

from core_control.carla_manager import CarlaManager, SpectatorConfig
from core_control.collect_data import DataCollector


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


def map_road_option_to_command(road_option) -> int:
    if road_option is None:
        return 0

    option_name = getattr(road_option, "name", str(road_option)).lower()
    if "left" in option_name:
        return 1
    if "right" in option_name:
        return 2
    if "straight" in option_name:
        return 3
    return 0


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
    ticks: int  # Max simulation steps to run
    tick_interval: float  # Sleep interval between ticks in dry-run mode
    dry_run: bool  # Run without connecting to CARLA
    seed: Optional[int]  # Random seed for reproducibility
    model_path: str  # Path to model weights (.pth, .pt)
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
        destination = random.choice(self._spawn_points).location
        current_loc = vehicle.get_location()
        try:
            self._nav_agent.set_destination(current_loc, destination)
        except TypeError:
            self._nav_agent.set_destination(destination)

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
        # YOLO integration
        self._yolo_detector = None
        self._yolo_window_name = "Lane Follow + YOLO Detection"
        self._yolo_enabled = False

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        if vehicle is None or session.world is None:
            logging.info("No CARLA vehicle/world available; lane_follow runs as noop.")
            return

        self._model = self._load_model()
        self._camera = self._spawn_camera(session.world, vehicle)
        self._camera.listen(self._on_camera_frame)

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

        self._yolo_detector = YoloDetector(str(model_path))
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
        error = self.config.target_speed_kmh - speed_kmh
        if error >= 0.0:
            throttle = clamp(0.20 + 0.02 * error, 0.05, self.config.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp((-error) / 20.0, 0.0, self.config.max_brake)
        return throttle, brake

    def _run_yolo_detection(self, frame, step_idx: int) -> tuple[bool, Dict[str, Any]]:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # distance_threshold applies to dynamic obstacles (pedestrian/vehicle/two_wheeler).
        detections, is_emergency = self._yolo_detector.detect_and_evaluate(frame_bgr, distance_threshold=5.0)
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
        obstacle_polygon = obstacle_roi.get("polygon", [])
        if np is not None and len(obstacle_polygon) >= 3:
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
            roi_zone = det.get('roi_zone')
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
            if roi_zone is not None:
                label = f"{label} [{roi_zone}]"
            if in_danger_roi:
                label = f"{label} [path]"
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

        cv2.imshow(self._yolo_window_name, annotated_frame)
        cv2.waitKey(1)

        if step_idx % 20 == 0 and detections:
            logging.info(
                "YOLO detections: %d objects | Emergency: %s | Reason: %s",
                len(detections),
                is_emergency,
                debug_info.get("decision_reason", "n/a"),
            )

        return is_emergency, debug_info

    def run_step(self, step_idx: int) -> None:
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("Lane-follow agent waiting for CARLA runtime.")
            return

        frame = self._read_latest_frame()
        if frame is None:
            if not self._waiting_frame_logged:
                logging.info("Waiting for first camera frame...")
                self._waiting_frame_logged = True
            return

        # Run YOLO detection and display if enabled
        is_emergency = False
        yolo_debug_info: Dict[str, Any] = {}
        if self._yolo_enabled and self._yolo_detector is not None:
            is_emergency, yolo_debug_info = self._run_yolo_detection(frame, step_idx)

        self._write_video_frame(frame)
        if self._stop_requested:
            logging.info("Video duration target reached, stopping agent loop.")
            return

        steering_raw = self._predict_steering(frame)
        alpha = clamp(self.config.steer_smoothing, 0.0, 0.99)
        steering = alpha * self._last_steer + (1.0 - alpha) * steering_raw
        self._last_steer = steering

        speed_kmh = self._current_speed_kmh()
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
            )

        if step_idx % 20 == 0:
            emergency_reason = (
                yolo_debug_info.get("decision_reason", "n/a") if is_emergency else "none"
            )
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
            logging.info("Destroyed lane-follow camera.")
        for sensor in self._data_cameras:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self._data_cameras = []
        # Clean up YOLO window
        if self._yolo_enabled:
            cv2.destroyWindow(self._yolo_window_name)
            self._yolo_enabled = False
            logging.info("Closed YOLO detection window.")

    def should_stop(self) -> bool:
        return self._stop_requested


class YoloDetectAgent(BaseAgent):
    name = "yolo_detect"

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._enabled = False
        self._camera = None
        self._latest_rgb = None
        self._frame_lock = threading.Lock()
        self._waiting_frame_logged = False
        self._detector = None
        self._window_name = "CARLA YOLO Detections"
        self._tm_autopilot_enabled = False

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

        self._detector = YoloDetector(str(model_path))
        self._camera = self._spawn_camera(world, vehicle)
        self._camera.listen(self._on_camera_frame)
        try:
            vehicle.set_autopilot(True, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(True)
        self._tm_autopilot_enabled = True
        self._enabled = True
        logging.info("YOLO detection enabled with model: %s", model_path)

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

    def run_step(self, step_idx: int) -> None:
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("YOLO agent waiting for CARLA runtime.")
            return

        frame = self._read_latest_frame()
        if frame is None:
            if not self._waiting_frame_logged:
                logging.info("YOLO waiting for first camera frame...")
                self._waiting_frame_logged = True
            return

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detections, is_emergency = self._detector.detect_and_evaluate(
            frame_bgr,
            distance_threshold=5.0,
        )
        debug_info = {}
        if hasattr(self._detector, "get_last_debug_info"):
            debug_info = self._detector.get_last_debug_info() or {}

        # Emergency override on top of CARLA autopilot.
        if is_emergency and self.session is not None and self.session.ego_vehicle is not None:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0
            control.hand_brake = False
            self.session.ego_vehicle.apply_control(control)
            logging.warning(
                "[TICK %d] EMERGENCY BRAKE! Reason: %s",
                step_idx,
                debug_info.get("decision_reason", "dangerous object nearby"),
            )

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
        obstacle_polygon = obstacle_roi.get("polygon", [])
        if np is not None and len(obstacle_polygon) >= 3:
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
            class_name = det["class_name"]
            confidence = det["confidence"]
            distance = det["distance"]
            roi_zone = det.get("roi_zone")
            in_danger_roi = bool(det.get("in_danger_roi", False))
            danger_match = bool(det.get("danger_match", False))
            label = f"{class_name} {confidence:.2f} ({distance:.1f}m)"
            if roi_zone is not None:
                label = f"{label} [{roi_zone}]"
            if in_danger_roi:
                label = f"{label} [path]"
            if danger_match:
                label = f"{label} [BRAKE]"

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

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow(self._window_name, annotated_frame)
        cv2.waitKey(1)

        if step_idx % 20 == 0:
            logging.info(
                "yolo_detect tick=%d detections=%d | Emergency: %s | Reason: %s",
                step_idx,
                len(detections),
                is_emergency,
                debug_info.get("decision_reason", "n/a"),
            )

    def teardown(self) -> None:
        if self._tm_autopilot_enabled and self.session is not None and self.session.ego_vehicle is not None:
            try:
                self.session.ego_vehicle.set_autopilot(False, self.config.tm_port)
            except TypeError:
                self.session.ego_vehicle.set_autopilot(False)
            self._tm_autopilot_enabled = False

        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
            self._camera = None
            logging.info("Destroyed YOLO camera.")

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
    YoloDetectAgent.name: YoloDetectAgent,
    NoopAgent.name: NoopAgent,
}


def load_env_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
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
        help="Choose lane_follow/autopilot/yolo_detect agent mode.",
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
        "--yolo-model-path",
        default="best.pt",
        help="Path to YOLO .pt model used by yolo_detect agent.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for lane_follow.",
    )
    parser.add_argument("--target-speed-kmh", type=float, default=30.0)
    parser.add_argument("--max-throttle", type=float, default=0.2)
    parser.add_argument("--max-brake", type=float, default=0.60)
    parser.add_argument(
        "--steer-smoothing",
        type=float,
        default=0.70,
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
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    env_cfg = load_env_config(args.config)

    lock_from_yaml = bool(_cfg_get(env_cfg, "spectator", "lock_on_spawn", True))
    lock_spectator = (
        args.lock_spectator_on_spawn
        if args.lock_spectator_on_spawn is not None
        else lock_from_yaml
    )

    sync = bool(_cfg_get(env_cfg, "carla", "sync", args.sync))
    fixed_delta = float(_cfg_get(env_cfg, "carla", "fixed_delta", args.fixed_delta))
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

    npc_vehicle_count = int(_cfg_get(env_cfg, "traffic_spawn", "vehicle_count", args.npc_vehicle_count))
    npc_bike_count = int(_cfg_get(env_cfg, "traffic_spawn", "bike_count", args.npc_bike_count))
    npc_motorbike_count = int(_cfg_get(env_cfg, "traffic_spawn", "motorbike_count", args.npc_motorbike_count))
    npc_pedestrian_count = int(_cfg_get(env_cfg, "traffic_spawn", "pedestrian_count", args.npc_pedestrian_count))
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

    return RunConfig(
        env_config_path=args.config,
        host=_cfg_get(env_cfg, "carla", "host", args.host),
        port=int(_cfg_get(env_cfg, "carla", "port", args.port)),
        tm_port=int(_cfg_get(env_cfg, "carla", "tm_port", args.tm_port)),
        timeout=float(_cfg_get(env_cfg, "carla", "timeout", args.timeout)),
        sync=sync,
        fixed_delta=fixed_delta,
        no_rendering=bool(_cfg_get(env_cfg, "carla", "no_rendering", args.no_rendering)),
        map_name=args.map if args.map != "Town03" else _cfg_get(env_cfg, "carla", "map", args.map),
        vehicle_filter=_cfg_get(env_cfg, "vehicle", "filter", args.vehicle_filter),
        spawn_point=args.spawn_point if args.spawn_point != -1 else int(_cfg_get(env_cfg, "vehicle", "spawn_point", args.spawn_point)),
        ticks=ticks,
        tick_interval=args.tick_interval,
        dry_run=args.dry_run,
        seed=args.seed,
        model_path=args.model_path,
        yolo_model_path=args.yolo_model_path,
        model_device=args.device,
        target_speed_kmh=args.target_speed_kmh,
        max_throttle=args.max_throttle,
        max_brake=args.max_brake,
        steer_smoothing=args.steer_smoothing,
        camera_width=int(_cfg_get(env_cfg, "camera", "width", args.camera_width)),
        camera_height=int(_cfg_get(env_cfg, "camera", "height", args.camera_height)),
        camera_fov=float(_cfg_get(env_cfg, "camera", "fov", args.camera_fov)),
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
