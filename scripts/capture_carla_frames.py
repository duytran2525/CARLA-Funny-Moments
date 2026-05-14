from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Cannot import carla. Set PYTHONPATH to <CARLA_ROOT>/PythonAPI/carla before running this script."
    ) from exc


DEFAULT_CONFIG = Path("configs/carla_env.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/carla_real_frames")


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data if isinstance(data, dict) else {}


def cfg_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def image_to_rgb(image: Any) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def drain(frame_queue: queue.Queue) -> None:
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            return


def parse_spawn_indices(raw: str | None, start: int, count: int) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [start + offset * 3 for offset in range(max(1, count))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real CARLA RGB camera frames for CBAM visualization.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--map", dest="map_name", default=None)
    parser.add_argument("--spawn-indices", default=None, help="Comma-separated spawn indices. Default: spawn, spawn+3, spawn+6.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--warmup-ticks", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    host = str(args.host or cfg_get(config, "carla", "host", "127.0.0.1"))
    port = int(args.port or cfg_get(config, "carla", "port", 2000))
    map_name = str(args.map_name or cfg_get(config, "carla", "map", "Town06"))
    fixed_delta = float(cfg_get(config, "carla", "fixed_delta", 0.03))
    width = int(cfg_get(config, "camera", "width", 640))
    height = int(cfg_get(config, "camera", "height", 360))
    fov = float(cfg_get(config, "camera", "fov", 90.0))
    vehicle_filter = str(cfg_get(config, "vehicle", "filter", "vehicle.tesla.model3"))
    configured_spawn = int(cfg_get(config, "vehicle", "spawn_point", 0))
    weather_preset = str(cfg_get(config, "weather", "preset", "ClearNoon"))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(host, port)
    client.set_timeout(float(args.timeout))

    world = client.get_world()
    target_map = f"Carla/Maps/{map_name}"
    if not world.get_map().name.endswith(map_name):
        world = client.load_world(map_name)

    original_settings = world.get_settings()
    actors: list[Any] = []
    camera = None
    vehicle = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        if hasattr(carla.WeatherParameters, weather_preset):
            world.set_weather(getattr(carla.WeatherParameters, weather_preset))

        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter(vehicle_filter)
        if not vehicle_blueprints:
            vehicle_blueprints = blueprint_library.filter("vehicle.*")
        if not vehicle_blueprints:
            raise RuntimeError("No vehicle blueprint available.")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in CARLA map.")

        spawn_indices = parse_spawn_indices(args.spawn_indices, configured_spawn, args.count)
        spawn_indices = [idx % len(spawn_points) for idx in spawn_indices]
        vehicle = world.try_spawn_actor(vehicle_blueprints[0], spawn_points[spawn_indices[0]])
        if vehicle is None:
            for transform in spawn_points:
                vehicle = world.try_spawn_actor(vehicle_blueprints[0], transform)
                if vehicle is not None:
                    break
        if vehicle is None:
            raise RuntimeError("Could not spawn ego vehicle.")
        actors.append(vehicle)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", str(fov))
        if camera_bp.has_attribute("sensor_tick"):
            camera_bp.set_attribute("sensor_tick", str(fixed_delta))

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.2), carla.Rotation(pitch=-8.0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors.append(camera)
        frame_queue: queue.Queue[Any] = queue.Queue()
        camera.listen(frame_queue.put)

        written: list[Path] = []
        for idx in spawn_indices:
            transform = spawn_points[idx]
            vehicle.set_transform(transform)
            vehicle.set_target_velocity(carla.Vector3D())
            vehicle.set_target_angular_velocity(carla.Vector3D())
            drain(frame_queue)
            for _ in range(max(1, int(args.warmup_ticks))):
                world.tick()
            image = frame_queue.get(timeout=float(args.timeout))
            rgb = image_to_rgb(image)
            path = output_dir / f"{map_name}_spawn_{idx:03d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            written.append(path)

        for path in written:
            print(path)
    finally:
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
        for actor in reversed(actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
