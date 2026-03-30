from __future__ import annotations

import argparse
import logging
import math
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
    from core_perception.cil_cnn_model import NvidiaCNN, NvidiaCNNV2, CIL_NvidiaCNN
except Exception:
    NvidiaCNN = None
    NvidiaCNNV2 = None
    CIL_NvidiaCNN = None

try:
    from core_perception.spatial_math import SpatialMath
except Exception:
    SpatialMath = None

try:
    from core_perception.object_tracker import SimpleTracker
except Exception:
    SimpleTracker = None

try:
    from core_control.adas_logic import ADASModule
except Exception:
    ADASModule = None

try:
    from core_control.traffic_supervisor import TrafficSupervisor, TrafficState
except Exception:
    TrafficSupervisor = None
    TrafficState = None

try:
    from core_perception.yolo_detector import YoloDetector
except Exception:
    YoloDetector = None

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner as _GRP
except ImportError:
    logging.error(
        "Failed to import GlobalRoutePlanner; route planning disabled.",
        exc_info=True,
    )
    _GRP = None



from core_control.carla_manager import CarlaManager, SpectatorConfig
from core_control.collect_data import DataCollector


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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

def apply_random_weather(world) -> str:
    # 40% clear day, 20% sunset, 20% night, 20% rain.
    roll = random.random()
    if roll < 0.40:
        preset_name = "clear_day"
        weather = carla.WeatherParameters(
            cloudiness=8.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=8.0, fog_density=0.0, wetness=0.0,
            sun_azimuth_angle=30.0, sun_altitude_angle=70.0,
        )
    elif roll < 0.60:
        preset_name = "sunset"
        weather = carla.WeatherParameters(
            cloudiness=30.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=10.0, fog_density=2.0, wetness=0.0,
            sun_azimuth_angle=15.0, sun_altitude_angle=8.0,
        )
    elif roll < 0.80:
        preset_name = "night"
        weather = carla.WeatherParameters(
            cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
            wind_intensity=6.0, fog_density=3.0, wetness=0.0,
            sun_azimuth_angle=0.0, sun_altitude_angle=-35.0,
        )
    else:
        preset_name = "rain"
        weather = carla.WeatherParameters(
            cloudiness=85.0, precipitation=70.0, precipitation_deposits=60.0,
            wind_intensity=35.0, fog_density=12.0, wetness=75.0,
            sun_azimuth_angle=25.0, sun_altitude_angle=45.0,
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

@dataclass
class RunConfig:
    env_config_path: str  # ─É╞░ß╗¥ng dß║½n tß╗¢i file cß║Ñu h├¼nh m├┤i tr╞░ß╗¥ng (.env, .json, .yaml...)
    host: str  # ─Éß╗ïa chß╗ë IP cß╗ºa m├íy chß╗º CARLA (th╞░ß╗¥ng l├á '127.0.0.1' hoß║╖c 'localhost')
    port: int  # Cß╗òng kß║┐t nß╗æi ch├¡nh cß╗ºa CARLA (mß║╖c ─æß╗ïnh l├á 2000)
    tm_port: int  # Cß╗òng kß║┐t nß╗æi cho Traffic Manager ─æß╗â ─æiß╗üu phß╗æi giao th├┤ng (th╞░ß╗¥ng l├á 8000)
    timeout: float  # Thß╗¥i gian chß╗¥ tß╗æi ─æa (gi├óy) ─æß╗â kß║┐t nß╗æi vß╗¢i server CARLA
    sync: bool  # Bß║¡t/tß║»t chß║┐ ─æß╗Ö ─æß╗ông bß╗Ö (Synchronous mode) ─æß╗â kh├┤ng bß╗ï rß╗¢t frame
    fixed_delta: float  # B╞░ß╗¢c thß╗¥i gian cß╗æ ─æß╗ïnh cho mß╗ùi frame (v├¡ dß╗Ñ: 0.05s t╞░╞íng ─æ╞░╞íng 20 FPS)
    no_rendering: bool  # Tß║»t ─æß╗ô hß╗ìa hiß╗ân thß╗ï cß╗ºa server ─æß╗â t─âng tß╗æc ─æß╗Ö giß║ú lß║¡p
    map_name: str  # T├¬n bß║ún ─æß╗ô cß║ºn nß║íp (v├¡ dß╗Ñ: 'Town01', 'Town04')
    vehicle_filter: str  # Loß║íi xe tß╗▒ l├íi muß╗æn sinh ra (v├¡ dß╗Ñ: 'vehicle.tesla.model3')
    spawn_point: int  # Vß╗ï tr├¡ (index) sinh ra xe tr├¬n bß║ún ─æß╗ô
    ticks: int  # Tß╗òng sß╗æ b╞░ß╗¢c (frames) tß╗æi ─æa m├á giß║ú lß║¡p sß║╜ chß║íy
    tick_interval: float  # Khoß║úng thß╗¥i gian nghß╗ë giß╗»a c├íc tick (nß║┐u muß╗æn chß║íy theo thß╗¥i gian thß╗▒c)
    dry_run: bool  # Cß╗¥ chß║íy thß╗¡ nghiß╗çm (kiß╗âm tra logic code m├á kh├┤ng l╞░u data hay chß║íy model)
    seed: Optional[int]  # Hß║ít giß╗æng random ─æß╗â cß╗æ ─æß╗ïnh kß║┐t quß║ú giß╗»a c├íc lß║ºn chß║íy (dß╗à debug)
    model_path: str  # ─É╞░ß╗¥ng dß║½n tß╗¢i file weights cß╗ºa m├┤ h├¼nh PyTorch (.pth, .pt)
    model_device: str  # Thiß║┐t bß╗ï d├╣ng ─æß╗â suy luß║¡n (v├¡ dß╗Ñ: 'cuda' cho GPU hoß║╖c 'cpu')
    target_speed_kmh: float  # Tß╗æc ─æß╗Ö mß╗Ñc ti├¬u m├á xe tß╗▒ l├íi cß║ºn duy tr├¼ (km/h)
    max_throttle: float  # Mß╗⌐c ─æß║íp ga tß╗æi ─æa cho ph├⌐p (tß╗½ 0.0 ─æß║┐n 1.0)
    max_brake: float  # Mß╗⌐c ─æß║íp phanh tß╗æi ─æa cho ph├⌐p (tß╗½ 0.0 ─æß║┐n 1.0)
    steer_smoothing: float  # Hß╗ç sß╗æ l├ám m╞░ß╗út v├┤ l─âng, tr├ính xe bß╗ï ─æ├ính l├íi giß║¡t cß╗Ñc
    camera_width: int  # Chiß╗üu rß╗Öng ß║únh thu thß║¡p tß╗½ camera RGB (pixel)
    camera_height: int  # Chiß╗üu cao ß║únh thu thß║¡p tß╗½ camera RGB (pixel)
    camera_fov: float  # G├│c nh├¼n (Field of View) cß╗ºa camera (t├¡nh bß║▒ng ─æß╗Ö)
    lock_spectator_on_spawn: bool  # Tß╗▒ ─æß╗Öng dß╗ïch chuyß╗ân camera tß╗òng (spectator) ─æß║┐n xe ngay khi spawn
    spectator_reapply_each_tick: bool  # Li├¬n tß╗Ñc kh├│a g├│c nh├¼n spectator theo ─æu├┤i xe mß╗ùi frame
    spectator_follow_distance: float  # Khoß║úng c├ích theo ─æu├┤i cß╗ºa spectator so vß╗¢i xe
    spectator_height: float  # ─Éß╗Ö cao cß╗ºa spectator so vß╗¢i n├│c xe
    spectator_pitch: float  # G├│c ch├║i xuß╗æng cß╗ºa camera spectator (t├¡nh bß║▒ng ─æß╗Ö)
    collect_data: bool  # C├┤ng tß║»c bß║¡t/tß║»t viß╗çc thu thß║¡p v├á l╞░u dataset
    collect_data_dir: str  # Th╞░ mß╗Ñc ─æ├¡ch ─æß╗â l╞░u ß║únh v├á log file cho qu├í tr├¼nh training
    save_every_n: int  # Chß╗ë l╞░u ß║únh mß╗ùi N tick (gi├║p khung cß║únh thay ─æß╗òi giß╗»a c├íc ß║únh)
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
    # Phase 2 CIL fields
    yolo_model_path: str = "best.pt"
    destination_spawn_point: int = -1
    random_weather: bool = True
    weather_preset: str = "ClearNoon"
    recovery_interval_frames: int = 100
    recovery_duration_frames: int = 10
    recovery_steer_offset: float = 0.3
    red_light_max_distance: float = 15.0
    roi_width_left: float = 2.0
    roi_width_right: float = 4.0
    green_immunity_frames: int = 50
    

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
        self._data_cameras = []  # Thay thế cho _data_camera cũ
        self._collector: Optional[DataCollector] = None
        self._recovery_start_frame = -1
        self._recovery_direction = 1.0

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        vehicle = session.ego_vehicle
        if vehicle is None or session.world is None:
            logging.info("No ego vehicle in this session; autopilot setup skipped.")
            return
        try:
            vehicle.set_autopilot(True, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(True)
        self._start_video_recording(session.world, vehicle)
        self._start_data_collection(session.world, vehicle)
        logging.info("Autopilot enabled for ego vehicle.")

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

        # Nạp DataCollector chuẩn Phương án C2 (Queue ngầm)
        self._collector = DataCollector(
            output_dir=self.config.collect_data_dir,
            enabled=True,
            use_multi_camera=True,  # Kích hoạt chế độ 3 camera
            save_every_n=self.config.save_every_n,
        )
        self._collector.start()

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick") and self.config.sync:
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        # Cấu hình 3 góc nhìn
        camera_setups = [
            ("center", carla.Transform(carla.Location(x=1.5, y=0.0, z=2.2), carla.Rotation(pitch=-8.0))),
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(yaw=-25.0, pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(yaw=25.0, pitch=-8.0))),
        ]
        
        self._data_cameras = []
        for side, transform in camera_setups:
            sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            sensor.listen(self._collector.make_sensor_callback(side))
            self._data_cameras.append(sensor)

        logging.info("Autopilot data collector started with 3 synchronized cameras.")

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
    def _recovery_offset(self, frame_id: int) -> float:
        """Kích hoạt lạng lách xe ngẫu nhiên để thu thập dữ liệu phục hồi."""
        if not self.config.collect_data:
            return 0.0
        interval = max(1, self.config.recovery_interval_frames)
        duration = max(1, self.config.recovery_duration_frames)
        if frame_id % interval == 0:
            self._recovery_start_frame = frame_id
            self._recovery_direction = random.choice([-1.0, 1.0])

        if self._recovery_start_frame >= 0 and frame_id < self._recovery_start_frame + duration:
            return self._recovery_direction * self.config.recovery_steer_offset
        return 0.0

    def _extract_current_command(self) -> int:
        """Suy luận lệnh điều hướng thô (heuristic) khi ở ngã tư."""
        vehicle = self.session.ego_vehicle if self.session is not None else None
        world = self.session.world if self.session is not None else None

        if vehicle is not None and world is not None:
            m = world.get_map()
            waypoint = m.get_waypoint(vehicle.get_location(), project_to_road=True)
            if waypoint is None:
                return 0
            if waypoint.is_junction:
                steer = float(vehicle.get_control().steer)
                if steer < -0.10:
                    return 1  # Rẽ trái
                if steer > 0.10:
                    return 2  # Rẽ phải
                return 3      # Đi thẳng
            return 0
        return 0
    def run_step(self, step_idx: int) -> None:
        frame = self._read_latest_video_frame()
        world = self.session.world if self.session is not None else None
        vehicle = self.session.ego_vehicle if self.session is not None else None

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

        if vehicle is not None:
            # 1. Áp dụng lạng lách phục hồi (Recovery Steer)
            frame_id_for_control = step_idx
            if world is not None:
                frame_id_for_control = int(world.get_snapshot().frame)
            
            recovery_delta = self._recovery_offset(frame_id_for_control)
            if abs(recovery_delta) > 1e-6:
                control = vehicle.get_control()
                control.steer = float(clamp(control.steer + recovery_delta, -1.0, 1.0))
                vehicle.apply_control(control)

            # 2. Ghi nhật ký (Data Logging)
            if self._collector is not None:
                velocity = vehicle.get_velocity()
                speed_kmh = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                control = vehicle.get_control()
                command = self._extract_current_command()
                
                # Gọi hàm add() của Phương án C2 (Không cần nhét frame ảnh, Background thread tự kéo từ Queue)
                self._collector.add(
                    tick=frame_id_for_control,
                    steer=control.steer,
                    throttle=control.throttle,
                    brake=control.brake,
                    speed_kmh=speed_kmh,
                    command=command
                )

        if step_idx % 20 == 0:
            logging.info("Autopilot tick %d", step_idx)

    def teardown(self) -> None:
        if self.session is None:
            return
        vehicle = self.session.ego_vehicle
        if vehicle is None:
            return
        try:
            vehicle.set_autopilot(False, self.config.tm_port)
        except TypeError:
            vehicle.set_autopilot(False)
        if self._collector is not None:
            self._collector.close()
            self._collector = None
            
        # Dọn dẹp mảng 3 camera
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
        self._video_writer = None
        self._video_output: Optional[Path] = None
        self._video_frames_written = 0
        self._video_max_frames = 0
        self._stop_requested = False

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
        self._init_video_writer()

        self._enabled = True
        logging.info("Lane-follow agent is ready.")

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

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(clamp(steering, -1.0, 1.0)),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )
        self.session.ego_vehicle.apply_control(control)

        if self._collector is not None:
            self._collector.add(
                tick=step_idx,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                command=0,
                image_center=frame,
            )

        if step_idx % 20 == 0:
            logging.info(
                "lane_follow tick=%d speed=%.1f km/h steer=%.3f throttle=%.2f brake=%.2f",
                step_idx,
                speed_kmh,
                control.steer,
                control.throttle,
                control.brake,
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

    def should_stop(self) -> bool:
        return self._stop_requested


class NoopAgent(BaseAgent):
    name = "noop"

    def run_step(self, step_idx: int) -> None:
        if step_idx % 50 == 0:
            logging.info("Noop agent alive at tick %d", step_idx)


# ---------------------------------------------------------------------------
# CILLaneFollowAgent – Phase 2
# ---------------------------------------------------------------------------

class CILLaneFollowAgent(LaneFollowAgent):
    """
    Agent tự lái Phase 2: CIL + GlobalRoutePlanner + IPM + ADAS.

    Luồng xử lý mỗi bước (run_step)
    ---------------------------------
    1. Nhận frame RGB từ camera.
    2. Lấy lệnh điều hướng từ ``GlobalRoutePlanner`` (Follow/Left/Right/Straight).
    3. Đo vận tốc hiện tại của xe.
    4. Chạy YOLO để phát hiện vật thể (nếu model YOLO có sẵn).
    5. Dùng IPM chuyển pixel → mét, cập nhật tracker.
    6. Tính TTC và lệnh ga/phanh (ADAS / ACC).
    7. Đưa ảnh + vận tốc + lệnh GPS vào CIL CNN → lấy góc lái.
    8. Áp dụng điều khiển (steer từ CIL, throttle/brake từ ADAS).
    9. Thu thập dữ liệu kèm cột ``command`` cho Phase 2 training.

    Sự khác biệt so với ``LaneFollowAgent``
    ----------------------------------------
    * Nạp ``CIL_NvidiaCNN`` thay vì ``NvidiaCNNV2``.
    * Dự đoán steering với 3 đầu vào: ảnh, vận tốc chuẩn hoá, lệnh GPS.
    * Tích hợp ``GlobalRoutePlanner`` để theo dõi route và trích xuất lệnh.
    * Tích hợp ``SpatialMath`` + ``SimpleTracker`` + ``ADASModule``
      cho ADAS với TTC chính xác (mét).
    """

    name = "cil_lane_follow"

    # CIL command mapping từ CARLA RoadOption
    _ROAD_OPTION_TO_CMD = {
        "RoadOption.LANEFOLLOW": 0,
        "RoadOption.LEFT": 1,
        "RoadOption.RIGHT": 2,
        "RoadOption.STRAIGHT": 3,
        "RoadOption.CHANGELANELEFT": 1,
        "RoadOption.CHANGELANERIGHT": 2,
        "RoadOption.VOID": 0,
    }
    # Tốc độ tối đa để chuẩn hoá (km/h → [0, 1])
    MAX_SPEED_KMH = 120.0
    _SPEED_NORM_FACTOR = 1.0 / MAX_SPEED_KMH
    # Camera mount params (khớp với CARLA camera transform trong LaneFollowAgent)
    _CAM_HEIGHT_M = 2.2    # z offset của camera (mét)
    _CAM_PITCH_DEG = 8.0   # |pitch| của camera (CARLA pitch=-8 → looking down 8°)
    # Bán kính tìm waypoint gần nhất trên route (mét)
    WAYPOINT_SEARCH_RADIUS = 10.0

    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self._spatial_math = None
        self._tracker = None
        self._adas = None
        self._yolo = None
        self._route = []          # list of (waypoint, RoadOption)
        self._current_cmd = 0     # lệnh điều hướng hiện tại (0–3)
        self._route_idx = 0       # chỉ số waypoint hiện tại trên route
        # 🌟 [CẬP NHẬT] Hybrid navigation state: EMA + dynamic look-ahead + gating + hysteresis
        self._speed_ema_mps = 0.0
        self._speed_ema_initialized = False
        self._speed_ema_alpha = 0.30
        self._lookahead_time_s = 1.2
        self._lookahead_min_m = 8.0
        self._lookahead_max_m = 35.0
        self._intersection_gate_min_m = 3.0
        self._intersection_gate_max_m = 22.0
        self._command_latch = 0
        self._command_latch_frames = 0
        self._command_latch_max_frames = 12

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, session: BaseSession) -> None:
        super().setup(session)
        if session.ego_vehicle is None or session.world is None:
            return

        # --- BẮT ĐẦU CẤY GHÉP ĐA CAMERA ---
        if self.config.collect_data:
            if self._collector is not None:
                self._collector.close()  # Đóng collector 1 camera mặc định của lớp cha
            self._start_multi_camera_data_collection(session.world, session.ego_vehicle)
        # --- KẾT THÚC CẤY GHÉP ---

        self._spatial_math = self._init_spatial_math()
        self._tracker = SimpleTracker() if SimpleTracker is not None else None
        self._adas = ADASModule() if ADASModule is not None else None
        self._traffic_supervisor = TrafficSupervisor(self.config) if TrafficSupervisor is not None else None
        self._yolo = self._init_yolo()
        self._route = self._plan_route(session)
        self._route_idx = 0
        logging.info("CIL agent ready. Route length: %d waypoints.", len(self._route))

    def _start_multi_camera_data_collection(self, world, vehicle):
        """Khởi tạo 3 camera phục vụ thu thập dữ liệu Closed-loop."""
        self._collector = DataCollector(
            output_dir=self.config.collect_data_dir,
            enabled=True,
            use_multi_camera=True,
            save_every_n=self.config.save_every_n,
        )
        self._collector.start()

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.camera_width))
        camera_bp.set_attribute("image_size_y", str(self.config.camera_height))
        camera_bp.set_attribute("fov", str(self.config.camera_fov))
        if camera_bp.has_attribute("sensor_tick") and self.config.sync:
            camera_bp.set_attribute("sensor_tick", str(self.config.fixed_delta))

        camera_setups = [
            ("center", carla.Transform(carla.Location(x=1.5, y=0.0, z=2.2), carla.Rotation(pitch=-8.0))),
            ("left", carla.Transform(carla.Location(x=1.5, y=-0.35, z=2.2), carla.Rotation(yaw=-25.0, pitch=-8.0))),
            ("right", carla.Transform(carla.Location(x=1.5, y=0.35, z=2.2), carla.Rotation(yaw=25.0, pitch=-8.0))),
        ]
        
        self._data_cameras = []
        for side, transform in camera_setups:
            sensor = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            sensor.listen(self._collector.make_sensor_callback(side))
            self._data_cameras.append(sensor)
        logging.info("CILLaneFollowAgent: Multi-camera closed-loop collection started.")

    def _init_spatial_math(self):
        if SpatialMath is None:
            logging.warning("SpatialMath module not available.")
            return None
        return SpatialMath(
            image_width=self.config.camera_width,
            image_height=self.config.camera_height,
            fov_deg=self.config.camera_fov,
            cam_height=self._CAM_HEIGHT_M,
            cam_pitch_deg=self._CAM_PITCH_DEG,
        )

    def _init_yolo(self):
        if YoloDetector is None:
            return None
        model_path = self.config.yolo_model_path
        try:
            yolo = YoloDetector(model_path=model_path)
            logging.info("YOLO detector loaded from %s.", model_path)
            return yolo
        except Exception as exc:
            logging.warning("Could not load YOLO detector (%s): %s", model_path, exc)
            return None

    def _plan_route(self, session: BaseSession):
        """
        Lên kế hoạch route từ vị trí spawn hiện tại đến điểm đích.

        Điểm đích được lấy từ ``config.destination_spawn_point``:
        * ≥ 0 → spawn point theo index đó.
        * -1 → chọn ngẫu nhiên một spawn point khác.
        """
        if _GRP is None or carla is None:
            logging.warning("GlobalRoutePlanner unavailable; no route planned.")
            return []

        world = session.world
        vehicle = session.ego_vehicle
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()

        if len(spawn_points) < 2:
            logging.warning("Not enough spawn points to plan a route.")
            return []

        start_loc = vehicle.get_location()
        dest_idx = self.config.destination_spawn_point
        if dest_idx < 0 or dest_idx >= len(spawn_points):
            dest_idx = random.randint(0, len(spawn_points) - 1)
        dest_loc = spawn_points[dest_idx].location

        try:
            grp = _GRP(carla_map, sampling_resolution=2.0)
            route = grp.trace_route(start_loc, dest_loc)
            logging.info(
                "Route planned: %d waypoints → spawn[%d] at (%.1f, %.1f)",
                len(route), dest_idx, dest_loc.x, dest_loc.y,
            )
            return route
        except Exception as exc:
            logging.warning("Route planning failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Model loading (override)
    # ------------------------------------------------------------------

    def _load_model(self):
        if torch is None:
            raise RuntimeError("PyTorch is required for cil_lane_follow agent.")
        if CIL_NvidiaCNN is None:
            raise RuntimeError("Cannot import CIL_NvidiaCNN from core_perception.cnn_model.")

        model_path = resolve_model_path(self.config.model_path)
        if self.config.model_path.lower() == "auto":
            logging.info("Auto-selected model path: %s", model_path)
        if not model_path.exists():
            raise RuntimeError(f"CIL model file not found: {model_path}")

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
            state_dict = checkpoint.get(
                "state_dict",
                checkpoint.get("model_state_dict", checkpoint)
            )
        if not isinstance(state_dict, dict):
            raise RuntimeError("Unsupported checkpoint format.")

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        model = CIL_NvidiaCNN().to(self._device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        logging.info("Loaded CIL model from %s on %s", model_path, self._device)
        return model

    # ------------------------------------------------------------------
    # Navigation command extraction
    # ------------------------------------------------------------------

    def _get_navigation_command(self) -> int:
        """
        Lấy lệnh điều hướng tại vị trí hiện tại từ route đã lên kế hoạch.

        Tìm waypoint gần nhất *phía trước* xe trên route và trả về
        ``RoadOption`` tương ứng dưới dạng chỉ số nguyên (0–3).

        Returns
        -------
        int
            0 = Follow Lane, 1 = Left, 2 = Right, 3 = Straight.
        """
        # 🌟 [CẬP NHẬT] Hybrid Dynamic Look-ahead + Intersection-aware Gating + Hysteresis
        if not self._route or self.session is None:
            self._command_latch = 0
            self._command_latch_frames = 0
            return 0

        vehicle = self.session.ego_vehicle
        if vehicle is None:
            self._command_latch = 0
            self._command_latch_frames = 0
            return 0

        ego_loc = vehicle.get_location()
        route_len = len(self._route)
        if route_len == 0:
            self._command_latch = 0
            self._command_latch_frames = 0
            return 0

        # 1) Speed EMA filtering
        velocity = vehicle.get_velocity()
        speed_mps = math.sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )
        if not self._speed_ema_initialized:
            self._speed_ema_mps = speed_mps
            self._speed_ema_initialized = True
        else:
            alpha = clamp(self._speed_ema_alpha, 0.0, 1.0)
            self._speed_ema_mps = alpha * speed_mps + (1.0 - alpha) * self._speed_ema_mps

        # 2) Dynamic look-ahead distance: L(v) = clip(Lmin + v*T, Lmin, Lmax)
        lookahead_m = clamp(
            self._lookahead_min_m + self._speed_ema_mps * self._lookahead_time_s,
            self._lookahead_min_m,
            self._lookahead_max_m,
        )

        # 3) Update route index by nearest waypoint around current progress
        search_start = max(0, self._route_idx)
        search_end = min(search_start + 80, route_len)
        best_idx = search_start
        best_dist = float("inf")
        for i in range(search_start, search_end):
            wp, _ = self._route[i]
            wp_loc = wp.transform.location
            dx = wp_loc.x - ego_loc.x
            dy = wp_loc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist < self.WAYPOINT_SEARCH_RADIUS:
            self._route_idx = min(best_idx + 1, route_len - 1)

        # Find index at dynamic look-ahead distance
        lookahead_idx = self._route_idx
        traveled = 0.0
        while lookahead_idx < route_len - 1 and traveled < lookahead_m:
            wp_curr, _ = self._route[lookahead_idx]
            wp_next, _ = self._route[lookahead_idx + 1]
            seg = wp_curr.transform.location.distance(wp_next.transform.location)
            traveled += seg
            lookahead_idx += 1

        # 4) Intersection-aware gating: only emit maneuver commands near intersections
        candidate_cmd = 0
        candidate_dist = 0.0
        cumulative = 0.0
        for i in range(self._route_idx, lookahead_idx + 1):
            wp, road_option = self._route[i]
            cmd_i = self._ROAD_OPTION_TO_CMD.get(str(road_option), 0)
            if i > self._route_idx:
                wp_prev, _ = self._route[i - 1]
                cumulative += wp_prev.transform.location.distance(wp.transform.location)

            if cmd_i == 0:
                continue

            is_intersection = bool(
                getattr(wp, "is_intersection", False) or getattr(wp, "is_junction", False)
            )
            if not is_intersection:
                for j in range(max(self._route_idx, i - 2), min(route_len, i + 3)):
                    wp_n, _ = self._route[j]
                    if bool(
                        getattr(wp_n, "is_intersection", False)
                        or getattr(wp_n, "is_junction", False)
                    ):
                        is_intersection = True
                        break

            if is_intersection:
                candidate_cmd = cmd_i
                candidate_dist = cumulative
                break

        gated_cmd = 0
        if candidate_cmd in (1, 2, 3):
            if self._intersection_gate_min_m <= candidate_dist <= self._intersection_gate_max_m:
                gated_cmd = candidate_cmd

        # 5) Hysteresis/state latch to prevent command hunting
        if self._command_latch_frames > 0 and self._command_latch in (1, 2, 3):
            self._command_latch_frames -= 1
            return self._command_latch

        if gated_cmd in (1, 2, 3):
            self._command_latch = gated_cmd
            self._command_latch_frames = self._command_latch_max_frames
            return gated_cmd

        self._command_latch = 0
        self._command_latch_frames = 0
        return 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_steering_cil(self, rgb_frame, speed_kmh: float, command: int) -> float:
        """
        Dự đoán góc lái với CIL CNN (ảnh + vận tốc + lệnh).

        Parameters
        ----------
        rgb_frame : np.ndarray
            Frame RGB gốc từ camera (H × W × 3).
        speed_kmh : float
            Tốc độ hiện tại (km/h).
        command : int
            Lệnh điều hướng (0–3).

        Returns
        -------
        float
            Góc lái trong [-1, 1].
        """
        height = rgb_frame.shape[0]
        cropped = rgb_frame[int(height * 0.45):, :, :]
        resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)

        yuv_image = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        tensor = torch.from_numpy(yuv_image).permute(2, 0, 1).float().div_(255.0)
        tensor.sub_(0.5).div_(0.5)
        tensor.unsqueeze_(0)
        tensor = tensor.to(self._device, non_blocking=True)

        speed_norm = clamp(speed_kmh * self._SPEED_NORM_FACTOR, 0.0, 1.0)
        speed_t = torch.tensor([[speed_norm]], dtype=torch.float32, device=self._device)
        cmd_t = torch.tensor([command], dtype=torch.long, device=self._device)

        with torch.inference_mode():
            steering = self._model(tensor, speed_t, cmd_t).item()
        return clamp(steering, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def run_step(self, step_idx: int) -> None:
        if not self._enabled:
            if step_idx % 50 == 0:
                logging.info("CIL agent waiting for CARLA runtime.")
            return

        frame = self._read_latest_frame()
        if frame is None:
            if not self._waiting_frame_logged:
                logging.info("CIL agent: waiting for first camera frame...")
                self._waiting_frame_logged = True
            return

        self._write_video_frame(frame)
        if self._stop_requested:
            return

        # 1. Navigation command từ route planner
        command = self._get_navigation_command()
        self._current_cmd = command

        # 2. Tốc độ hiện tại
        speed_kmh = self._current_speed_kmh()
        speed_mps = speed_kmh / 3.6

        # 3. YOLO detection + IPM + Tracker + ADAS
        throttle, brake, tracks = self._run_survival_pipeline(frame, speed_mps)

        # 🌟 [CẬP NHẬT] Đánh giá Luật lệ giao thông (Traffic Supervisor)
        if self._traffic_supervisor is not None and TrafficState is not None:
            traffic_state, traffic_debug = self._traffic_supervisor.evaluate(tracks)
            if traffic_state == TrafficState.MUST_STOP:
                throttle = 0.0
                brake = max(brake, self.config.max_brake) 
                if step_idx % 20 == 0:
                    logging.info("TrafficSupervisor: ĐÈN ĐỎ TRONG ROI - KÍCH HOẠT PHANH!")

        # 4. CIL steering
        steering_raw = self._predict_steering_cil(frame, speed_kmh, command)
        alpha = clamp(self.config.steer_smoothing, 0.0, 0.99)
        steering = alpha * self._last_steer + (1.0 - alpha) * steering_raw
        self._last_steer = steering

        # 5. Áp dụng điều khiển
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(clamp(steering, -1.0, 1.0)),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )
        self.session.ego_vehicle.apply_control(control)

        # 6. Lưu dữ liệu (kèm command cho Phase 2 dataset)
        if self._collector is not None and self.config.collect_data:
            frame_id_for_control = step_idx
            if self.session.world is not None:
                frame_id_for_control = int(self.session.world.get_snapshot().frame)
                
            self._collector.add(
                tick=frame_id_for_control,
                steer=control.steer,
                throttle=control.throttle,
                brake=control.brake,
                speed_kmh=speed_kmh,
                command=command,
            )

        if step_idx % 20 == 0:
            logging.info(
                "cil_lane_follow tick=%d cmd=%d speed=%.1f km/h "
                "steer=%.3f throttle=%.2f brake=%.2f",
                step_idx, command, speed_kmh,
                control.steer, control.throttle, control.brake,
            )
    

    def _run_survival_pipeline(self, rgb_frame, ego_speed_mps: float):
        """
        Luồng sinh tồn: YOLO → SpatialMath → Tracker → ADAS → (throttle, brake).

        Trả về điều khiển dọc (ga/phanh) từ module ADAS.
        Nếu YOLO hoặc ADAS không có sẵn, dùng điều khiển tốc độ cơ bản.

        Parameters
        ----------
        rgb_frame : np.ndarray
            Frame RGB gốc.
        ego_speed_mps : float
            Tốc độ xe (m/s).

        Returns
        -------
        throttle, brake : float
        """
        target_speed_mps = self.config.target_speed_kmh / 3.6

        if self._adas is None:
            # Fallback: điều khiển tốc độ đơn giản
            throttle, brake = self._longitudinal_control(ego_speed_mps * 3.6)
            return throttle, brake, []

        # YOLO detection
        detections = []
        if self._yolo is not None:
            try:
                _, raw_dets = self._yolo.detect(rgb_frame)
                detections = raw_dets
            except Exception as exc:
                logging.debug("YOLO detect failed: %s", exc)

        # Tracker update
        tracks = []
        if self._tracker is not None:
            tracks = self._tracker.update(detections)
            if self._spatial_math is not None:
                self._tracker.update_world_positions(self._spatial_math, tracks)

        # ADAS: tính TTC và ACC
        throttle, brake, min_ttc = self._adas.compute_control(
            tracks, ego_speed_mps, target_speed_mps
        )
        if min_ttc < float('inf'):
            logging.debug("ADAS TTC=%.2fs → throttle=%.2f brake=%.2f", min_ttc, throttle, brake)
        return throttle, brake, tracks

    def teardown(self) -> None:
        super().teardown()
        if hasattr(self, '_data_cameras'):
            for sensor in self._data_cameras:
                try:
                    sensor.stop()
                    sensor.destroy()
                except RuntimeError:
                    pass
            self._data_cameras = []


AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    AutopilotAgent.name: AutopilotAgent,
    LaneFollowAgent.name: LaneFollowAgent,
    CILLaneFollowAgent.name: CILLaneFollowAgent,
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
        default="cil_lane_follow",
        help="Choose cil_lane_follow (Phase 2 CIL+GPS+ADAS), lane_follow, or autopilot.",
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
    parser.add_argument("--collect-data-dir", default="data/raw_driving")
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=50,
        help="Only save a frame every N ticks (default: 50 for diverse scenes).",
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
    # Phase 2 CIL arguments
    parser.add_argument(
        "--yolo-model-path",
        default="best.pt",
        help="Path to YOLO model weights for CIL agent (Phase 2).",
    )
    parser.add_argument(
        "--destination-spawn-point",
        type=int,
        default=-1,
        help="Destination spawn-point index for CIL route planning (-1 = random).",
    )
    parser.add_argument(
        "--no-random-weather", 
        action="store_true", 
        help="Disable random weather preset selection at start.")
    parser.add_argument(
        "--recovery-interval-frames", 
        type=int, 
        default=100, 
        help="Apply recovery steering disturbance every N frames.")
    parser.add_argument(
        "--recovery-duration-frames", 
        type=int, 
        default=10, 
        help="Number of frames to keep disturbance active.")
    parser.add_argument(
        "--recovery-steer-offset", 
        type=float, 
        default=0.3, 
        help="Steering offset added during disturbance window.")
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
        yolo_model_path=args.yolo_model_path,
        destination_spawn_point=args.destination_spawn_point,
        random_weather=not args.no_random_weather,
        weather_preset=str(_cfg_get(env_cfg, "weather", "preset", "ClearNoon")),
        recovery_interval_frames=max(1, args.recovery_interval_frames),
        recovery_duration_frames=max(1, args.recovery_duration_frames),
        recovery_steer_offset=abs(args.recovery_steer_offset),
        red_light_max_distance=float(_cfg_get(env_cfg, "traffic_rules", "red_light_max_distance", 15.0)),
        roi_width_left=float(_cfg_get(env_cfg, "traffic_rules", "roi_width_left", 2.0)),
        roi_width_right=float(_cfg_get(env_cfg, "traffic_rules", "roi_width_right", 4.0)),
        green_immunity_frames=int(_cfg_get(env_cfg, "traffic_rules", "green_immunity_frames", 50)),
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
