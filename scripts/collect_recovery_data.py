"""
collect_recovery_data.py — Intentional Drift Recovery Dataset Collector v3
==========================================================================
Changes v3 (based on dataset quality report):

  [CRITICAL] Collision sensor + reverse maneuver thay vì teleport tức thì.
             Phát hiện va chạm ngay lập tức → lùi xe ~30 ticks → CRUISING.
             Teleport chỉ còn là fallback khi stuck > STUCK_MAX_TICKS.

  [CRITICAL] MIN_RECORD_SPEED_KMH guard: chỉ ghi frame khi speed >= 2.0 km/h.
             Loại bỏ 59.7% frames xe đứng yên húc tường khỏi dataset.

  [CRITICAL] STUCK_MAX_TICKS giảm từ 150 → 40 ticks để phản ứng nhanh hơn.

  [MAJOR]   RECOVERY_WARMUP_TICKS = 5: bỏ qua N tick đầu của RECOVERING để
             TM kịp nhận ra vị trí và bắt đầu correction thực sự.

  [MAJOR]   Thêm cột CSV: noise_magnitude, phase, cycle_id,
             post_collision, correction_valid.
             recovery_flag giữ lại để tương thích ngược (luôn = 1).

  [MAJOR]   Balanced noise direction: thay vì random() < 0.5 thuần túy,
             dùng cumulative counter để đảm bảo left/right ~50/50.

  [MINOR]   Phase.REVERSING mới: trạng thái riêng để reverse ra khỏi chướng ngại.

Sử dụng:
  python scripts/collect_recovery_data.py --ticks 20000 --out-dir data/recovery
"""

from __future__ import annotations

import argparse
import csv
import enum
import logging
import math
import os
import queue
import random
import sys
import time
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
    import cv2
    import numpy as np
except ImportError as exc:
    raise RuntimeError("opencv-python and numpy are required.") from exc

try:
    import yaml
except ImportError:
    yaml = None

from core_control.carla_manager import CarlaManager, SpectatorConfig


# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Số giây xe chạy bình thường (autopilot ON) trước khi drift tiếp.
CRUISE_DURATION_S: float = 5.0

# Thời gian bơm nhiễu steering. Quá ngắn → chưa lệch. Quá dài → văng.
NOISE_DURATION_S: float = 1.5

# Cường độ noise cộng vào steering (0.15–0.50).
NOISE_INTENSITY: float = 0.30

# Throttle thủ công trong pha drift (giữ xe di chuyển khi autopilot OFF).
DRIFT_THROTTLE: float = 0.45

# Thời gian ghi dữ liệu recovery (autopilot tự đánh lái gắt).
RECOVERY_DURATION_S: float = 3.0

# Tốc độ tối thiểu (km/h) để bắt đầu drift. Nếu thấp hơn → chờ thêm.
MIN_SPEED_TO_DRIFT_KMH: float = 8.0

# [FIX v3 - CRITICAL] Tốc độ tối thiểu để GHI frame trong RECOVERING.
# Loại bỏ frames xe đứng yên húc tường (vốn chiếm 59.7% dataset cũ).
MIN_RECORD_SPEED_KMH: float = 2.0

# [FIX v3 - CRITICAL] Giảm từ 150 → 40 để phát hiện stuck nhanh hơn.
# 30 FPS * ~1.3s = 40 ticks. Collision sensor sẽ bắt phần lớn va chạm
# trước khi stuck counter đạt ngưỡng này.
STUCK_SPEED_THRESHOLD_KMH: float = 1.0
STUCK_MAX_TICKS: int = 40

# [NEW v3 - MAJOR] Bỏ qua N tick đầu của RECOVERING để TM ổn định.
# Tránh ghi các frame TM đang "khởi động" (steer 0→0.15→0.45...).
RECOVERY_WARMUP_TICKS: int = 5

# [NEW v3 - CRITICAL] Số tick lùi xe khi phát hiện va chạm qua collision sensor.
# Thay thế teleport tức thì: xe lùi ra khỏi vật cản trước khi tiếp tục.
REVERSE_TICKS: int = 30
REVERSE_THROTTLE: float = 0.4  # throttle cho reverse (gear = reverse)

# [NEW v3] Số tick sau va chạm để đánh dấu post_collision = 1 trong CSV.
POST_COLLISION_FLAG_TICKS: int = 30

# Camera & Output
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 360
CAMERA_FOV: float = 90.0
JPEG_QUALITY: int = 95
SAVE_WIDTH: int = 400
SAVE_HEIGHT: int = 200
SAVE_EVERY_N_FRAMES: int = 1
TARGET_SPEED_KMH: float = 35.0

# Số tick warmup để TM ổn định tốc độ trước khi bắt đầu loop.
WARMUP_TICKS: int = 150


# ═══════════════════════════════════════════════════════════════════════════
# PHASE ENUM
# ═══════════════════════════════════════════════════════════════════════════

class Phase(enum.Enum):
    CRUISING   = "cruising"
    DRIFTING   = "drifting"
    RECOVERING = "recovering"
    REVERSING  = "reversing"   # [NEW v3] lùi xe ra khỏi vật cản


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


def _image_to_rgb(image: Any) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((image.height, image.width, 4))
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_speed_kmh(ego: Any) -> float:
    v = ego.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _drain_queue(q: queue.Queue) -> None:
    """Xả sạch queue không cần xử lý."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _respawn_to_nearest_waypoint(
    ego: Any, world: Any, target_speed_kmh: float, tm: Any, tm_port: int
) -> None:
    """
    [FALLBACK] Teleport xe về waypoint hợp lệ gần nhất.
    Chỉ gọi khi collision sensor + reverse đều thất bại (stuck > STUCK_MAX_TICKS).
    """
    try:
        current_loc = ego.get_transform().location
        carla_map = world.get_map()
        wp = carla_map.get_waypoint(
            current_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            logging.warning("Respawn: cannot find nearby waypoint, skipping.")
            return

        safe_tf = wp.transform
        safe_tf.location.z += 0.5
        ego.set_transform(safe_tf)

        zero_vec = carla.Vector3D(0.0, 0.0, 0.0)
        ego.set_target_velocity(zero_vec)
        ego.set_target_angular_velocity(zero_vec)

        reset_ctrl = carla.VehicleControl()
        reset_ctrl.throttle = 0.0
        reset_ctrl.brake = 0.0
        reset_ctrl.hand_brake = False
        reset_ctrl.manual_gear_shift = False
        reset_ctrl.gear = 1
        ego.apply_control(reset_ctrl)

        logging.warning(
            "Respawn (fallback): teleported to (%.1f, %.1f) road_id=%d",
            safe_tf.location.x, safe_tf.location.y, wp.road_id,
        )
    except Exception as exc:
        logging.error("Respawn failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════════

# [v3] Thêm: noise_magnitude, phase, cycle_id, post_collision, correction_valid
# recovery_flag giữ lại cho tương thích ngược (luôn = 1 vì chỉ ghi khi RECOVERING).
CSV_FIELDNAMES = [
    "frame_id", "timestamp", "image_filename",
    "steering", "throttle", "brake", "speed_kmh",
    "noise_direction", "noise_magnitude",
    "x", "y", "z", "yaw",
    "recovery_flag",       # deprecated: luôn = 1, giữ cho compat
    "phase",               # [v3] giá trị Phase enum khi frame được ghi
    "cycle_id",            # [v3] số thứ tự cycle (drift→recovery) hiện tại
    "post_collision",      # [v3] 1 nếu trong POST_COLLISION_FLAG_TICKS sau va chạm
    "correction_valid",    # [v3] 1 nếu steering đúng chiều recovery (ngược noise_direction)
]


class RecoveryCSVWriter:
    def __init__(self, csv_path: Path) -> None:
        self._path = csv_path
        append = csv_path.exists()
        if append:
            self._validate_schema()
        self._file = csv_path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDNAMES)
        if not append:
            self._writer.writeheader()
            self._file.flush()
        self._rows_since_flush = 0
        self.total_rows = 0

    def _validate_schema(self) -> None:
        with self._path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader.fieldnames or [])
        if existing != CSV_FIELDNAMES:
            raise RuntimeError(
                f"CSV schema mismatch.\n"
                f"  Expected : {CSV_FIELDNAMES}\n"
                f"  Got      : {existing}\n"
                "Delete old CSV or use a fresh --out-dir."
            )

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._rows_since_flush += 1
        self.total_rows += 1
        if self._rows_since_flush >= 50:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()


# ═══════════════════════════════════════════════════════════════════════════
# ARG PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect recovery dataset via Intentional Drift in CARLA (v3)."
    )
    p.add_argument("--config",              type=Path,  default=Path("configs/carla_env.yaml"))
    p.add_argument("--host",                            default=None)
    p.add_argument("--port",                type=int,   default=None)
    p.add_argument("--tm-port",             type=int,   default=None)
    p.add_argument("--map",   dest="map_name",          default=None)
    p.add_argument("--spawn-point",         type=int,   default=-1)
    p.add_argument("--ticks",               type=int,   default=15000)
    p.add_argument("--out-dir",                         default="data/recovery")
    p.add_argument("--target-speed-kmh",    type=float, default=None)
    p.add_argument("--noise-intensity",     type=float, default=None)
    p.add_argument("--noise-duration",      type=float, default=None)
    p.add_argument("--recovery-duration",   type=float, default=None)
    p.add_argument("--cruise-duration",     type=float, default=None)
    p.add_argument("--drift-throttle",      type=float, default=None)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--npc-vehicle-count",   type=int,   default=0)
    p.add_argument("--log-every",           type=int,   default=200)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# AUTOPILOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _enable_autopilot(ego: Any, tm: Any, tm_port: int, target_speed_kmh: float) -> None:
    """Bật autopilot + cấu hình TM. Gọi mỗi lần chuyển sang CRUISING/RECOVERING."""

    # Ép gear=1 (số D) để TM có thể đạp ga tiến lên.
    ctrl = carla.VehicleControl()
    ctrl.hand_brake = False
    ctrl.manual_gear_shift = False
    ctrl.gear = 1
    ctrl.brake = 0.0
    ctrl.throttle = 0.0
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
            logging.debug(
                "TM speed config: limit=%.0f target=%.0f pct=%.1f%%",
                speed_limit, target_speed_kmh, pct,
            )
        except Exception:
            pass


def _disable_autopilot(ego: Any, tm_port: int) -> None:
    """Tắt autopilot — chuyển sang manual control."""
    try:
        ego.set_autopilot(False, tm_port)
    except TypeError:
        ego.set_autopilot(False)


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

    noise_intensity    = args.noise_intensity    or NOISE_INTENSITY
    noise_duration_s   = args.noise_duration     or NOISE_DURATION_S
    recovery_duration_s = args.recovery_duration or RECOVERY_DURATION_S
    cruise_duration_s  = args.cruise_duration    or CRUISE_DURATION_S
    drift_throttle     = args.drift_throttle     or DRIFT_THROTTLE

    config      = _load_config(args.config)
    host        = args.host     or _cfg(config, "carla", "host",        "127.0.0.1")
    port        = args.port     or int(_cfg(config, "carla", "port",    2000))
    tm_port     = args.tm_port  or int(_cfg(config, "carla", "tm_port", 8000))
    timeout     = float(_cfg(config, "carla", "timeout",       60.0))
    fixed_delta = float(_cfg(config, "carla", "fixed_delta",   0.033333))
    map_name    = args.map_name or _cfg(config, "carla", "map",         "Town03")
    vehicle_filter = _cfg(config, "vehicle", "filter", "vehicle.tesla.model3")
    target_speed   = args.target_speed_kmh or TARGET_SPEED_KMH

    cruise_ticks   = max(1, int(cruise_duration_s   / fixed_delta))
    noise_ticks    = max(1, int(noise_duration_s    / fixed_delta))
    recovery_ticks = max(1, int(recovery_duration_s / fixed_delta))

    logging.info("=" * 65)
    logging.info("  INTENTIONAL DRIFT v3 — Recovery Dataset Collector")
    logging.info("=" * 65)
    logging.info("  Map                : %s", map_name)
    logging.info("  Target speed       : %.0f km/h", target_speed)
    logging.info("  Noise intensity    : %.3f", noise_intensity)
    logging.info("  Drift throttle     : %.2f", drift_throttle)
    logging.info("  Min drift speed    : %.1f km/h", MIN_SPEED_TO_DRIFT_KMH)
    logging.info("  Min record speed   : %.1f km/h [v3]", MIN_RECORD_SPEED_KMH)
    logging.info("  Cruise duration    : %.2fs (%d ticks)", cruise_duration_s,   cruise_ticks)
    logging.info("  Noise duration     : %.2fs (%d ticks)", noise_duration_s,    noise_ticks)
    logging.info("  Recovery duration  : %.2fs (%d ticks)", recovery_duration_s, recovery_ticks)
    logging.info("  Recovery warmup    : %d ticks [v3]", RECOVERY_WARMUP_TICKS)
    logging.info("  Stuck max ticks    : %d [v3]", STUCK_MAX_TICKS)
    logging.info("  Reverse ticks      : %d [v3]", REVERSE_TICKS)
    logging.info("  Total ticks        : %d", args.ticks)
    logging.info("  Warmup ticks       : %d", WARMUP_TICKS)
    logging.info("=" * 65)

    out_dir    = Path(args.out_dir).resolve()
    images_dir = out_dir / "images_center"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = out_dir / "recovery_log.csv"
    csv_writer = RecoveryCSVWriter(csv_path)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    existing_max = 0
    for p in images_dir.glob("*.jpg"):
        if p.stem.isdigit():
            existing_max = max(existing_max, int(p.stem))
    next_img_idx = existing_max + 1

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

    frame_queue:     queue.Queue[Any] = queue.Queue(maxsize=30)
    collision_queue: queue.Queue[Any] = queue.Queue(maxsize=10)
    camera_sensor:    Optional[Any] = None
    collision_sensor: Optional[Any] = None
    saved_count       = 0
    cycle_count       = 0
    skipped_low_speed = 0
    collision_count   = 0   # tổng số va chạm được phát hiện

    try:
        manager.start()
        world = manager.world
        ego   = manager.ego_vehicle
        tm    = manager.tm
        assert world is not None and ego is not None

        # TM synchronous mode
        if tm is not None:
            try:
                tm.set_synchronous_mode(True)
                logging.info("Traffic Manager synchronous mode: ON")
            except Exception as exc:
                logging.warning("Could not set TM sync mode: %s", exc)

        bp_lib = world.get_blueprint_library()

        # ── Camera sensor ─────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        cam_bp.set_attribute("fov",          str(CAMERA_FOV))
        if cam_bp.has_attribute("sensor_tick"):
            cam_bp.set_attribute("sensor_tick", str(fixed_delta))
        cam_transform  = carla.Transform(
            carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0),
        )
        camera_sensor  = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        camera_sensor.listen(
            lambda img: collision_queue.put_nowait(img)   # re-use pattern
            if False else                                  # (camera goes to frame_queue)
            frame_queue.put_nowait(img)
            if not frame_queue.full() else None
        )
        # Correct listener:
        camera_sensor.stop()
        camera_sensor.listen(
            lambda img: frame_queue.put_nowait(img) if not frame_queue.full() else None
        )

        # ── [NEW v3] Collision sensor ──────────────────────────────────────
        # Phát hiện va chạm ngay lập tức → trigger REVERSING thay vì chờ stuck.
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        collision_sensor.listen(
            lambda evt: collision_queue.put_nowait(evt) if not collision_queue.full() else None
        )
        logging.info("Collision sensor attached to ego vehicle.")

        # ── Kickstart ─────────────────────────────────────────────────────
        kick_ctrl = carla.VehicleControl(
            throttle=1.0, steer=0.0, brake=0.0,
            hand_brake=False, manual_gear_shift=False, gear=1,
        )
        ego.apply_control(kick_ctrl)
        for _ in range(10):
            world.tick()
        kick_speed = _get_speed_kmh(ego)
        logging.info("Kickstart done (10 ticks). Speed=%.1f km/h", kick_speed)

        _enable_autopilot(ego, tm, tm_port, target_speed)

        for _ in range(max(1, WARMUP_TICKS - 10)):
            world.tick()
        warmup_speed = _get_speed_kmh(ego)
        logging.info("Warmup done (%d ticks). Speed=%.1f km/h", WARMUP_TICKS, warmup_speed)
        _drain_queue(frame_queue)
        _drain_queue(collision_queue)

        # ══════════════════════════════════════════════════════════════════
        # GAME LOOP
        # ══════════════════════════════════════════════════════════════════
        phase               = Phase.CRUISING
        phase_tick_counter  = 0
        noise_direction     = 1.0
        recovery_frame_counter = 0
        current_noise_dir   = 0
        last_auto_steer     = 0.0
        stuck_ticks         = 0
        post_collision_counter = 0  # [v3] đếm ngược từ POST_COLLISION_FLAG_TICKS

        # [v3] Lịch sử noise_direction để cân bằng left/right
        noise_history: List[float] = []

        for tick_idx in range(1, args.ticks + 1):
            world.tick()
            phase_tick_counter += 1

            ego_transform = ego.get_transform()
            speed_kmh     = _get_speed_kmh(ego)

            # ── [NEW v3] COLLISION DETECTION ──────────────────────────────
            # Xử lý trước stuck detection để phản ứng ngay khi va chạm.
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
                    "tick=%d: collision #%d detected (phase=%s, speed=%.1f) → REVERSING",
                    tick_idx, collision_count, phase.value, speed_kmh,
                )
                # Đảm bảo autopilot OFF trước khi manual reverse
                if phase in (Phase.CRUISING, Phase.RECOVERING):
                    _disable_autopilot(ego, tm_port)
                # phase DRIFTING: autopilot đã off sẵn
                phase              = Phase.REVERSING
                phase_tick_counter = 0
                stuck_ticks        = 0
                _drain_queue(frame_queue)

            # Đếm ngược post_collision flag
            if post_collision_counter > 0:
                post_collision_counter -= 1

            # ── STUCK DETECTION ───────────────────────────────────────────
            # [v3] STUCK_MAX_TICKS = 40 (giảm từ 150).
            # Collision sensor đã bắt phần lớn va chạm → stuck chỉ còn là
            # fallback cho trường hợp reverse vẫn không thoát được.
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
                _drain_queue(frame_queue)
                _drain_queue(collision_queue)
                continue

            # ══════════════════════════════════════════════════════════════
            # PHASE: CRUISING
            # ══════════════════════════════════════════════════════════════
            if phase == Phase.CRUISING:
                _drain_queue(frame_queue)
                last_auto_steer = float(ego.get_control().steer)

                if phase_tick_counter >= cruise_ticks:
                    if speed_kmh < MIN_SPEED_TO_DRIFT_KMH:
                        skipped_low_speed += 1
                        phase_tick_counter = max(0, cruise_ticks - 30)
                        continue

                    # [v3] BALANCED noise direction:
                    # Đếm left/right trong lịch sử để đảm bảo ~50/50.
                    left_so_far  = sum(1 for d in noise_history if d < 0)
                    right_so_far = len(noise_history) - left_so_far
                    if left_so_far < right_so_far:
                        noise_direction = -1.0          # cần thêm left
                    elif right_so_far < left_so_far:
                        noise_direction =  1.0          # cần thêm right
                    else:
                        noise_direction = -1.0 if random.random() < 0.5 else 1.0
                    noise_history.append(noise_direction)

                    _disable_autopilot(ego, tm_port)
                    phase              = Phase.DRIFTING
                    phase_tick_counter = 0
                    current_noise_dir  = int(noise_direction)
                    cycle_count       += 1
                    logging.debug(
                        "Cycle #%d: → DRIFTING (%s) speed=%.1f | L=%d R=%d",
                        cycle_count,
                        "LEFT" if noise_direction < 0 else "RIGHT",
                        speed_kmh, left_so_far, right_so_far,
                    )

            # ══════════════════════════════════════════════════════════════
            # PHASE: DRIFTING
            # ══════════════════════════════════════════════════════════════
            elif phase == Phase.DRIFTING:
                _drain_queue(frame_queue)

                noisy_control         = carla.VehicleControl()
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
                    _drain_queue(frame_queue)
                    logging.debug("Cycle #%d: → RECOVERING", cycle_count)

            # ══════════════════════════════════════════════════════════════
            # PHASE: RECOVERING
            # ══════════════════════════════════════════════════════════════
            elif phase == Phase.RECOVERING:
                auto_control = ego.get_control()

                latest_image = None
                while not frame_queue.empty():
                    try:
                        latest_image = frame_queue.get_nowait()
                    except queue.Empty:
                        break

                recovery_frame_counter += 1

                # [FIX v3 - MAJOR] Bỏ qua N tick đầu để TM ổn định.
                # [FIX v3 - CRITICAL] Chỉ ghi khi xe đang thực sự di chuyển.
                should_record = (
                    latest_image is not None
                    and recovery_frame_counter > RECOVERY_WARMUP_TICKS
                    and speed_kmh >= MIN_RECORD_SPEED_KMH
                    and recovery_frame_counter % SAVE_EVERY_N_FRAMES == 0
                )

                if should_record:
                    rgb = _image_to_rgb(latest_image)
                    rgb_resized = cv2.resize(
                        rgb, (SAVE_WIDTH, SAVE_HEIGHT), interpolation=cv2.INTER_AREA,
                    )
                    bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)

                    img_id       = f"{next_img_idx:08d}"
                    img_filename = f"{img_id}.jpg"
                    cv2.imwrite(str(images_dir / img_filename), bgr, encode_params)
                    next_img_idx += 1

                    loc = ego_transform.location
                    rot = ego_transform.rotation

                    # [v3] correction_valid: steer đúng chiều recovery?
                    # noise_direction=-1 (đẩy trái) → cần steer phải (+)
                    # noise_direction=+1 (đẩy phải) → cần steer trái (-)
                    # Hợp lệ khi sign(steer) ngược với noise_direction.
                    steer_val = float(auto_control.steer)
                    correction_valid = (
                        1 if (steer_val * current_noise_dir < -0.05) else 0
                    )

                    csv_writer.write({
                        "frame_id":         int(latest_image.frame),
                        "timestamp":        f"{latest_image.timestamp:.6f}",
                        "image_filename":   img_filename,
                        "steering":         round(steer_val, 5),
                        "throttle":         round(float(auto_control.throttle), 4),
                        "brake":            round(float(auto_control.brake), 4),
                        "speed_kmh":        round(speed_kmh, 2),
                        "noise_direction":  current_noise_dir,
                        "noise_magnitude":  round(noise_intensity, 4),   # [v3]
                        "x":               round(float(loc.x), 4),
                        "y":               round(float(loc.y), 4),
                        "z":               round(float(loc.z), 4),
                        "yaw":             round(float(rot.yaw), 4),
                        "recovery_flag":   1,
                        "phase":           Phase.RECOVERING.value,        # [v3]
                        "cycle_id":        cycle_count,                   # [v3]
                        "post_collision":  1 if post_collision_counter > 0 else 0,  # [v3]
                        "correction_valid": correction_valid,             # [v3]
                    })
                    saved_count += 1

                if phase_tick_counter >= recovery_ticks:
                    phase              = Phase.CRUISING
                    phase_tick_counter = 0
                    logging.debug(
                        "Cycle #%d: → CRUISING (saved %d total)", cycle_count, saved_count,
                    )

            # ══════════════════════════════════════════════════════════════
            # PHASE: REVERSING  [NEW v3]
            # ══════════════════════════════════════════════════════════════
            elif phase == Phase.REVERSING:
                _drain_queue(frame_queue)

                # Lùi xe ra khỏi vật cản: throttle với gear reverse.
                reverse_ctrl              = carla.VehicleControl()
                reverse_ctrl.throttle     = REVERSE_THROTTLE
                reverse_ctrl.brake        = 0.0
                reverse_ctrl.steer        = 0.0
                reverse_ctrl.reverse      = True
                reverse_ctrl.hand_brake   = False
                reverse_ctrl.manual_gear_shift = False
                ego.apply_control(reverse_ctrl)

                if phase_tick_counter >= REVERSE_TICKS:
                    # Kết thúc reverse → bật autopilot → về CRUISING
                    _enable_autopilot(ego, tm, tm_port, target_speed)
                    phase              = Phase.CRUISING
                    phase_tick_counter = 0
                    stuck_ticks        = 0
                    _drain_queue(frame_queue)
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
                    "tick=%d/%d | phase=%-10s | cycle=#%d | saved=%d | "
                    "speed=%.1f km/h | collisions=%d | noise L/R=%.0f%%/%.0f%%",
                    tick_idx, args.ticks, phase.value, cycle_count,
                    saved_count, speed_kmh, collision_count,
                    left_pct, 100.0 - left_pct,
                )

        logging.info("=" * 65)
        logging.info("  COLLECTION COMPLETE")
        logging.info("  Total ticks      : %d", args.ticks)
        logging.info("  Drift cycles     : %d", cycle_count)
        logging.info("  Recovery samples : %d", saved_count)
        logging.info("  Skipped (slow)   : %d", skipped_low_speed)
        logging.info("  Collisions caught: %d", collision_count)
        if noise_history:
            left_final  = sum(1 for d in noise_history if d < 0)
            right_final = len(noise_history) - left_final
            logging.info(
                "  Noise direction  : LEFT=%d (%.1f%%)  RIGHT=%d (%.1f%%)",
                left_final,  100.0 * left_final  / len(noise_history),
                right_final, 100.0 * right_final / len(noise_history),
            )
        logging.info("  CSV path         : %s", csv_path)
        logging.info("  Images dir       : %s", images_dir)
        logging.info("=" * 65)
        return 0

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 1
    finally:
        csv_writer.close()
        if camera_sensor is not None:
            try:
                camera_sensor.stop()
                camera_sensor.destroy()
            except RuntimeError:
                pass
        if collision_sensor is not None:
            try:
                collision_sensor.stop()
                collision_sensor.destroy()
            except RuntimeError:
                pass
        manager.cleanup()
        logging.info(
            "Cleanup done. %d frames in %d cycles. %d collisions handled.",
            saved_count, cycle_count, collision_count,
        )


if __name__ == "__main__":
    raise SystemExit(main())