"""
collect_recovery_data.py — Intentional Drift Recovery Dataset Collector v4
===========================================================================
v4: Tái cấu trúc hoàn toàn phần ghi data.

THAY ĐỔI CHÍNH SO VỚI v3:
  [BREAKING] Bỏ RecoveryCSVWriter + tự ghi ảnh thủ công.
             Thay bằng DataCollector từ collect_data.py.

  [OUTPUT]   CSV schema giờ HOÀN TOÀN GIỐNG driving_log.csv của collect_data.py:
               - Có đủ temporal stacking (img_id_tm06, img_id_tm03)
               - Có waypoint ground truth (wp_1_x..wp_5_y)
               - Có 3 camera (center, left, right)
               - recovery_flag = 1 cho mọi frame được ghi
               - is_recovering = True → DataCollector tự dense-sample

  [REMOVED]  Các cột riêng của v3 (noise_magnitude, phase, cycle_id,
             post_collision, correction_valid) — không cần thiết cho training.

  [SENSORS]  Thêm 2 camera left/right để tương thích với DataCollector.
             DataCollector yêu cầu đủ 3 camera mới ghi frame.

  [COMPAT]   Toàn bộ logic Phase (CRUISING/DRIFTING/RECOVERING/REVERSING)
             và các fix v3 (collision sensor, STUCK detection, balanced noise)
             được GIỮ NGUYÊN.

Sử dụng:
  python scripts/collect_recovery_data.py --ticks 20000 --out-dir data/recovery
"""

from __future__ import annotations

import argparse
import enum
import logging
import math
import queue
import random
import sys
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "Cannot import carla. Set PYTHONPATH to <CARLA_ROOT>/PythonAPI/carla"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required.") from exc

try:
    import yaml
except ImportError:
    yaml = None

from core_control.carla_manager import CarlaManager, SpectatorConfig
from collect_data import DataCollector


# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS — giữ nguyên từ v3
# ═══════════════════════════════════════════════════════════════════════════

CRUISE_DURATION_S: float      = 5.0
NOISE_DURATION_S: float       = 1.5
NOISE_INTENSITY: float        = 0.30
DRIFT_THROTTLE: float         = 0.45
RECOVERY_DURATION_S: float    = 3.0
MIN_SPEED_TO_DRIFT_KMH: float = 8.0
MIN_RECORD_SPEED_KMH: float   = 2.0

STUCK_SPEED_THRESHOLD_KMH: float = 1.0
STUCK_MAX_TICKS: int             = 40
RECOVERY_WARMUP_TICKS: int       = 5
REVERSE_TICKS: int               = 30
REVERSE_THROTTLE: float          = 0.4
POST_COLLISION_FLAG_TICKS: int   = 30

# Camera config
CAMERA_WIDTH: int   = 640
CAMERA_HEIGHT: int  = 360
CAMERA_FOV: float   = 90.0
SAVE_WIDTH: int     = 400
SAVE_HEIGHT: int    = 200
JPEG_QUALITY: int   = 95
SAVE_EVERY_N_FRAMES: int = 1

# Camera offsets — phải khớp với collect_data.py
CAM_X: float     =  1.5
CAM_Z: float     =  2.2
CAM_PITCH: float = -8.0
CAM_LEFT_Y: float  = -1.2
CAM_RIGHT_Y: float =  1.2

TARGET_SPEED_KMH: float = 35.0
WARMUP_TICKS: int       = 150


# ═══════════════════════════════════════════════════════════════════════════
# PHASE ENUM
# ═══════════════════════════════════════════════════════════════════════════

class Phase(enum.Enum):
    CRUISING   = "cruising"
    DRIFTING   = "drifting"
    RECOVERING = "recovering"
    REVERSING  = "reversing"


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_config(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data if isinstance(data, dict) else {}


def _cfg(config: dict, section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    return value.get(key, default) if isinstance(value, dict) else default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_speed_kmh(ego: Any) -> float:
    v = ego.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _drain_queue(q: "queue.Queue[Any]") -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _respawn_to_nearest_waypoint(
    ego: Any, world: Any, target_speed_kmh: float, tm: Any, tm_port: int
) -> None:
    """Teleport fallback khi stuck > STUCK_MAX_TICKS."""
    try:
        current_loc = ego.get_transform().location
        carla_map   = world.get_map()
        wp = carla_map.get_waypoint(
            current_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            logging.warning("Respawn: cannot find nearby waypoint, skipping.")
            return

        safe_tf          = wp.transform
        safe_tf.location.z += 0.5
        ego.set_transform(safe_tf)

        zero_vec = carla.Vector3D(0.0, 0.0, 0.0)
        ego.set_target_velocity(zero_vec)
        ego.set_target_angular_velocity(zero_vec)

        reset_ctrl                    = carla.VehicleControl()
        reset_ctrl.throttle           = 0.0
        reset_ctrl.brake              = 0.0
        reset_ctrl.hand_brake         = False
        reset_ctrl.manual_gear_shift  = False
        reset_ctrl.gear               = 1
        ego.apply_control(reset_ctrl)

        logging.warning(
            "Respawn (fallback): teleported to (%.1f, %.1f) road_id=%d",
            safe_tf.location.x, safe_tf.location.y, wp.road_id,
        )
    except Exception as exc:
        logging.error("Respawn failed: %s", exc)


def _enable_autopilot(
    ego: Any, tm: Any, tm_port: int, target_speed_kmh: float
) -> None:
    ctrl                     = carla.VehicleControl()
    ctrl.hand_brake          = False
    ctrl.manual_gear_shift   = False
    ctrl.gear                = 1
    ctrl.brake               = 0.0
    ctrl.throttle            = 0.0
    ego.apply_control(ctrl)

    actual_tm_port = tm.get_port() if tm is not None else tm_port
    try:
        ego.set_autopilot(True, actual_tm_port)
    except TypeError:
        ego.set_autopilot(True)

    if tm is None:
        return

    try:
        tm.auto_lane_change(ego, True)
        tm.distance_to_leading_vehicle(ego, 3.0)
        tm.ignore_lights_percentage(ego, 100.0)
        tm.ignore_signs_percentage(ego, 100.0)
    except Exception:
        pass

    if target_speed_kmh > 0:
        try:
            speed_limit = float(ego.get_speed_limit())
            if speed_limit <= 0.1:
                speed_limit = 30.0
            pct = 100.0 * (speed_limit - target_speed_kmh) / speed_limit
            pct = max(-150.0, min(100.0, pct))
            tm.vehicle_percentage_speed_difference(ego, float(pct))
        except Exception:
            pass


def _disable_autopilot(ego: Any, tm_port: int) -> None:
    try:
        ego.set_autopilot(False, tm_port)
    except TypeError:
        ego.set_autopilot(False)


def _get_route_command(ego: Any, world: Any) -> int:
    """
    Trả về route command đơn giản.
    DataCollector sẽ override thông qua _resolve_sample_command
    dựa trên trajectory thực tế khi ở junction.
    """
    try:
        carla_map = world.get_map()
        loc = ego.get_transform().location
        wp  = carla_map.get_waypoint(loc, project_to_road=True)
        if wp is not None and wp.is_junction:
            return 3  # junction — DataCollector infer thực
    except Exception:
        pass
    return 0  # lane follow


def _is_junction(ego: Any, world: Any) -> bool:
    try:
        carla_map = world.get_map()
        loc = ego.get_transform().location
        wp  = carla_map.get_waypoint(loc, project_to_road=True)
        return wp is not None and wp.is_junction
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# ARG PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect recovery dataset via Intentional Drift in CARLA (v4)."
    )
    p.add_argument("--config",            type=Path,  default=Path("configs/carla_env.yaml"))
    p.add_argument("--host",                          default=None)
    p.add_argument("--port",              type=int,   default=None)
    p.add_argument("--tm-port",           type=int,   default=None)
    p.add_argument("--map", dest="map_name",          default=None)
    p.add_argument("--spawn-point",       type=int,   default=-1)
    p.add_argument("--ticks",             type=int,   default=15000)
    p.add_argument("--out-dir",                       default="data/recovery")
    p.add_argument("--target-speed-kmh",  type=float, default=None)
    p.add_argument("--noise-intensity",   type=float, default=None)
    p.add_argument("--noise-duration",    type=float, default=None)
    p.add_argument("--recovery-duration", type=float, default=None)
    p.add_argument("--cruise-duration",   type=float, default=None)
    p.add_argument("--drift-throttle",    type=float, default=None)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--npc-vehicle-count", type=int,   default=0)
    p.add_argument("--log-every",         type=int,   default=200)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(args.seed)

    noise_intensity     = args.noise_intensity    or NOISE_INTENSITY
    noise_duration_s    = args.noise_duration     or NOISE_DURATION_S
    recovery_duration_s = args.recovery_duration  or RECOVERY_DURATION_S
    cruise_duration_s   = args.cruise_duration    or CRUISE_DURATION_S
    drift_throttle      = args.drift_throttle     or DRIFT_THROTTLE

    config      = _load_config(args.config)
    host        = args.host    or _cfg(config, "carla", "host",        "127.0.0.1")
    port        = args.port    or int(_cfg(config, "carla", "port",    2000))
    tm_port     = args.tm_port or int(_cfg(config, "carla", "tm_port", 8000))
    timeout     = float(_cfg(config, "carla", "timeout",               60.0))
    fixed_delta = float(_cfg(config, "carla", "fixed_delta",           0.033333))
    map_name    = args.map_name or _cfg(config, "carla", "map",        "Town03")
    vehicle_filter = _cfg(config, "vehicle", "filter", "vehicle.tesla.model3")
    target_speed   = args.target_speed_kmh or TARGET_SPEED_KMH

    cruise_ticks   = max(1, int(cruise_duration_s   / fixed_delta))
    noise_ticks    = max(1, int(noise_duration_s    / fixed_delta))
    recovery_ticks = max(1, int(recovery_duration_s / fixed_delta))

    logging.info("=" * 65)
    logging.info("  INTENTIONAL DRIFT v4 — Recovery Dataset Collector")
    logging.info("  Output schema: IDENTICAL to collect_data.py (driving_log.csv)")
    logging.info("=" * 65)
    logging.info("  Map                : %s", map_name)
    logging.info("  Target speed       : %.0f km/h", target_speed)
    logging.info("  Noise intensity    : %.3f", noise_intensity)
    logging.info("  Drift throttle     : %.2f", drift_throttle)
    logging.info("  Min drift speed    : %.1f km/h", MIN_SPEED_TO_DRIFT_KMH)
    logging.info("  Min record speed   : %.1f km/h", MIN_RECORD_SPEED_KMH)
    logging.info("  Cruise duration    : %.2fs (%d ticks)", cruise_duration_s,   cruise_ticks)
    logging.info("  Noise duration     : %.2fs (%d ticks)", noise_duration_s,    noise_ticks)
    logging.info("  Recovery duration  : %.2fs (%d ticks)", recovery_duration_s, recovery_ticks)
    logging.info("  Recovery warmup    : %d ticks", RECOVERY_WARMUP_TICKS)
    logging.info("  Stuck max ticks    : %d", STUCK_MAX_TICKS)
    logging.info("  Reverse ticks      : %d", REVERSE_TICKS)
    logging.info("  Total ticks        : %d", args.ticks)
    logging.info("=" * 65)

    # ── DataCollector — dùng chung với collect_data.py ───────────────────
    # save_every_n=1: ghi mọi frame khi RECOVERING
    # DataCollector tự dense-sample khi is_recovering=True
    collector = DataCollector(
        output_dir=args.out_dir,
        enabled=True,
        jpeg_quality=JPEG_QUALITY,
        save_every_n=SAVE_EVERY_N_FRAMES,
        flush_every_n=50,
        resize_width=SAVE_WIDTH,
        resize_height=SAVE_HEIGHT,
    )

    manager = CarlaManager(
        host=str(host), port=int(port), tm_port=int(tm_port),
        timeout=timeout, map_name=str(map_name), sync=True,
        fixed_delta=fixed_delta, no_rendering=False,
        vehicle_filter=str(vehicle_filter), spawn_point=args.spawn_point,
        spectator_cfg=SpectatorConfig(lock_on_spawn=True),
        npc_vehicle_count=args.npc_vehicle_count,
        npc_bike_count=0, npc_motorbike_count=0, npc_pedestrian_count=0,
        npc_enable_autopilot=True, seed=args.seed,
    )

    collision_queue: "queue.Queue[Any]" = queue.Queue(maxsize=10)

    cam_center:       Optional[Any] = None
    cam_left:         Optional[Any] = None
    cam_right:        Optional[Any] = None
    collision_sensor: Optional[Any] = None

    saved_count        = 0
    cycle_count        = 0
    skipped_low_speed  = 0
    collision_count    = 0

    try:
        manager.start()
        world = manager.world
        ego   = manager.ego_vehicle
        tm    = manager.tm
        assert world is not None and ego is not None

        if tm is not None:
            try:
                tm.set_synchronous_mode(True)
                logging.info("Traffic Manager synchronous mode: ON")
            except Exception as exc:
                logging.warning("Could not set TM sync mode: %s", exc)

        bp_lib = world.get_blueprint_library()

        def _make_cam_bp() -> Any:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
            bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
            bp.set_attribute("fov",          str(CAMERA_FOV))
            if bp.has_attribute("sensor_tick"):
                bp.set_attribute("sensor_tick", str(fixed_delta))
            return bp

        # ── 3 cameras — tương thích DataCollector ────────────────────────
        cam_center = world.spawn_actor(
            _make_cam_bp(),
            carla.Transform(
                carla.Location(x=CAM_X, y=0.0,        z=CAM_Z),
                carla.Rotation(pitch=CAM_PITCH),
            ),
            attach_to=ego,
        )
        cam_left = world.spawn_actor(
            _make_cam_bp(),
            carla.Transform(
                carla.Location(x=CAM_X, y=CAM_LEFT_Y, z=CAM_Z),
                carla.Rotation(pitch=CAM_PITCH),
            ),
            attach_to=ego,
        )
        cam_right = world.spawn_actor(
            _make_cam_bp(),
            carla.Transform(
                carla.Location(x=CAM_X, y=CAM_RIGHT_Y, z=CAM_Z),
                carla.Rotation(pitch=CAM_PITCH),
            ),
            attach_to=ego,
        )

        # DataCollector callback: resize + push vào sensor_queue nội bộ
        cam_center.listen(collector.make_sensor_callback("center"))
        cam_left.listen(collector.make_sensor_callback("left"))
        cam_right.listen(collector.make_sensor_callback("right"))

        # ── Collision sensor ──────────────────────────────────────────────
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        collision_sensor.listen(
            lambda evt: collision_queue.put_nowait(evt)
            if not collision_queue.full() else None
        )
        logging.info("Collision sensor attached.")

        # ── Start DataCollector (tạo dirs + CSV) ─────────────────────────
        collector.start()
        logging.info(
            "DataCollector started → %s/driving_log.csv",
            Path(args.out_dir).resolve(),
        )

        # ── Kickstart ─────────────────────────────────────────────────────
        kick_ctrl = carla.VehicleControl(
            throttle=1.0, steer=0.0, brake=0.0,
            hand_brake=False, manual_gear_shift=False, gear=1,
        )
        ego.apply_control(kick_ctrl)
        for _ in range(10):
            world.tick()
        logging.info("Kickstart done. Speed=%.1f km/h", _get_speed_kmh(ego))

        _enable_autopilot(ego, tm, tm_port, target_speed)
        for _ in range(max(1, WARMUP_TICKS - 10)):
            world.tick()
        logging.info(
            "Warmup done (%d ticks). Speed=%.1f km/h", WARMUP_TICKS, _get_speed_kmh(ego)
        )
        _drain_queue(collision_queue)

        # ══════════════════════════════════════════════════════════════════
        # GAME LOOP
        # ══════════════════════════════════════════════════════════════════
        phase                  = Phase.CRUISING
        phase_tick_counter     = 0
        noise_direction        = 1.0
        recovery_frame_counter = 0
        current_noise_dir      = 0
        last_auto_steer        = 0.0
        stuck_ticks            = 0
        post_collision_counter = 0
        noise_history: List[float] = []

        for tick_idx in range(1, args.ticks + 1):
            world.tick()
            phase_tick_counter += 1

            ego_transform = ego.get_transform()
            speed_kmh     = _get_speed_kmh(ego)
            rot           = ego_transform.rotation
            loc           = ego_transform.location

            # ── Collision detection ───────────────────────────────────────
            collision_this_tick = False
            while not collision_queue.empty():
                try:
                    collision_queue.get_nowait()
                    collision_this_tick = True
                except queue.Empty:
                    break

            if collision_this_tick and phase != Phase.REVERSING:
                collision_count += 1
                post_collision_counter = POST_COLLISION_FLAG_TICKS
                logging.warning(
                    "tick=%d: collision #%d (phase=%s, speed=%.1f) → REVERSING",
                    tick_idx, collision_count, phase.value, speed_kmh,
                )
                if phase in (Phase.CRUISING, Phase.RECOVERING):
                    _disable_autopilot(ego, tm_port)
                phase              = Phase.REVERSING
                phase_tick_counter = 0
                stuck_ticks        = 0

            if post_collision_counter > 0:
                post_collision_counter -= 1

            # ── Stuck detection ───────────────────────────────────────────
            if speed_kmh < STUCK_SPEED_THRESHOLD_KMH:
                stuck_ticks += 1
            else:
                stuck_ticks = 0

            if stuck_ticks >= STUCK_MAX_TICKS:
                logging.warning(
                    "tick=%d: stuck %d ticks (speed=%.1f, phase=%s) → teleport fallback",
                    tick_idx, stuck_ticks, speed_kmh, phase.value,
                )
                if phase in (Phase.DRIFTING, Phase.REVERSING):
                    _enable_autopilot(ego, tm, tm_port, target_speed)
                _respawn_to_nearest_waypoint(ego, world, target_speed, tm, tm_port)
                _enable_autopilot(ego, tm, tm_port, target_speed)
                post_collision_counter = POST_COLLISION_FLAG_TICKS
                for _ in range(30):
                    world.tick()
                phase              = Phase.CRUISING
                phase_tick_counter = 0
                stuck_ticks        = 0
                _drain_queue(collision_queue)
                continue

            # ══════════════════════════════════════════════════════════════
            # PHASE: CRUISING — không ghi
            # ══════════════════════════════════════════════════════════════
            if phase == Phase.CRUISING:
                last_auto_steer = float(ego.get_control().steer)

                if phase_tick_counter >= cruise_ticks:
                    if speed_kmh < MIN_SPEED_TO_DRIFT_KMH:
                        skipped_low_speed += 1
                        phase_tick_counter = max(0, cruise_ticks - 30)
                        continue

                    left_so_far  = sum(1 for d in noise_history if d < 0)
                    right_so_far = len(noise_history) - left_so_far
                    if left_so_far < right_so_far:
                        noise_direction = -1.0
                    elif right_so_far < left_so_far:
                        noise_direction =  1.0
                    else:
                        noise_direction = -1.0 if random.random() < 0.5 else 1.0
                    noise_history.append(noise_direction)

                    _disable_autopilot(ego, tm_port)
                    phase              = Phase.DRIFTING
                    phase_tick_counter = 0
                    current_noise_dir  = int(noise_direction)
                    cycle_count       += 1
                    logging.debug(
                        "Cycle #%d: → DRIFTING (%s) speed=%.1f",
                        cycle_count,
                        "LEFT" if noise_direction < 0 else "RIGHT",
                        speed_kmh,
                    )

            # ══════════════════════════════════════════════════════════════
            # PHASE: DRIFTING — không ghi, bơm noise steering
            # ══════════════════════════════════════════════════════════════
            elif phase == Phase.DRIFTING:
                noisy_control          = carla.VehicleControl()
                noisy_control.throttle = float(drift_throttle)
                noisy_control.brake    = 0.0
                noisy_control.steer    = _clamp(
                    last_auto_steer + noise_direction * noise_intensity, -1.0, 1.0,
                )
                noisy_control.hand_brake = False
                noisy_control.reverse    = False
                ego.apply_control(noisy_control)

                if phase_tick_counter >= noise_ticks:
                    _enable_autopilot(ego, tm, tm_port, target_speed)
                    phase                  = Phase.RECOVERING
                    phase_tick_counter     = 0
                    recovery_frame_counter = 0
                    logging.debug("Cycle #%d: → RECOVERING", cycle_count)

            # ══════════════════════════════════════════════════════════════
            # PHASE: RECOVERING — GHI qua DataCollector
            # ══════════════════════════════════════════════════════════════
            elif phase == Phase.RECOVERING:
                auto_control           = ego.get_control()
                recovery_frame_counter += 1

                # Bỏ qua N tick đầu (TM ổn định) + frame khi xe không di chuyển
                should_record = (
                    recovery_frame_counter > RECOVERY_WARMUP_TICKS
                    and speed_kmh >= MIN_RECORD_SPEED_KMH
                )

                if should_record:
                    # Timestamp từ world snapshot (đồng bộ với camera)
                    ts        = float(world.get_snapshot().timestamp.elapsed_seconds)
                    route_cmd = _get_route_command(ego, world)
                    junction  = _is_junction(ego, world)

                    # add_vehicle_state: đẩy state vào DataCollector
                    # DataCollector tự ghép với ảnh camera qua sensor_queue
                    # is_recovering=True → CSV sẽ có recovery_flag=1
                    # DataCollector cũng dense-sample khi is_recovering=True
                    collector.add_vehicle_state(
                        frame_id=tick_idx,
                        timestamp=ts,
                        steer=float(auto_control.steer),
                        throttle=float(auto_control.throttle),
                        brake=float(auto_control.brake),
                        speed_kmh=speed_kmh,
                        x=float(loc.x),
                        y=float(loc.y),
                        z=float(loc.z),
                        has_crash=False,      # không ghi crash frames
                        is_recovering=True,   # → recovery_flag=1 trong CSV
                        is_junction=junction,
                        command=route_cmd,
                        pitch=float(rot.pitch),
                        roll=float(rot.roll),
                        yaw=float(rot.yaw),
                    )
                    saved_count += 1

                if phase_tick_counter >= recovery_ticks:
                    phase              = Phase.CRUISING
                    phase_tick_counter = 0
                    logging.debug(
                        "Cycle #%d: → CRUISING (fed ~%d state frames)",
                        cycle_count, saved_count,
                    )

            # ══════════════════════════════════════════════════════════════
            # PHASE: REVERSING — không ghi, lùi xe ra khỏi vật cản
            # ══════════════════════════════════════════════════════════════
            elif phase == Phase.REVERSING:
                reverse_ctrl                      = carla.VehicleControl()
                reverse_ctrl.throttle             = REVERSE_THROTTLE
                reverse_ctrl.brake                = 0.0
                reverse_ctrl.steer                = 0.0
                reverse_ctrl.reverse              = True
                reverse_ctrl.hand_brake           = False
                reverse_ctrl.manual_gear_shift    = False
                ego.apply_control(reverse_ctrl)

                if phase_tick_counter >= REVERSE_TICKS:
                    _enable_autopilot(ego, tm, tm_port, target_speed)
                    phase              = Phase.CRUISING
                    phase_tick_counter = 0
                    stuck_ticks        = 0
                    _drain_queue(collision_queue)
                    logging.info(
                        "tick=%d: REVERSING done → CRUISING (total collisions: %d)",
                        tick_idx, collision_count,
                    )

            # ── Periodic logging ──────────────────────────────────────────
            if tick_idx % args.log_every == 0:
                left_pct = (
                    100.0 * sum(1 for d in noise_history if d < 0) / len(noise_history)
                    if noise_history else 0.0
                )
                logging.info(
                    "tick=%d/%d | phase=%-10s | cycle=#%d | "
                    "speed=%.1f km/h | collisions=%d | noise L/R=%.0f%%/%.0f%%",
                    tick_idx, args.ticks, phase.value, cycle_count,
                    speed_kmh, collision_count, left_pct, 100.0 - left_pct,
                )

        # ── Flush + finalize ──────────────────────────────────────────────
        # collector.close() xử lý nốt các target frames còn trong queue
        # và tính waypoint GT cho chúng
        collector.close()

        logging.info("=" * 65)
        logging.info("  COLLECTION COMPLETE")
        logging.info("  Total ticks      : %d", args.ticks)
        logging.info("  Drift cycles     : %d", cycle_count)
        logging.info("  State frames fed : %d (actual CSV rows có thể ít hơn do WP filter)", saved_count)
        logging.info("  Skipped (slow)   : %d", skipped_low_speed)
        logging.info("  Collisions caught: %d", collision_count)
        logging.info("  CSV              : %s", Path(args.out_dir).resolve() / "driving_log.csv")
        logging.info("  Images center    : %s", Path(args.out_dir).resolve() / "images_center")
        logging.info("  Images left      : %s", Path(args.out_dir).resolve() / "images_left")
        logging.info("  Images right     : %s", Path(args.out_dir).resolve() / "images_right")
        logging.info("=" * 65)
        return 0

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        collector.close()
        return 1

    finally:
        for sensor in (cam_center, cam_left, cam_right, collision_sensor):
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except RuntimeError:
                    pass
        manager.cleanup()
        logging.info(
            "Cleanup done. %d state frames in %d cycles. %d collisions handled.",
            saved_count, cycle_count, collision_count,
        )


if __name__ == "__main__":
    raise SystemExit(main())