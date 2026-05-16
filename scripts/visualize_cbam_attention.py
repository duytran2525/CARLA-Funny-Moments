from __future__ import annotations

import argparse
import csv
import queue
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_perception.cnn_model import CIL_NvidiaCNN, unwrap_state_dict


MODEL_INPUT_SIZE = (200, 66)
DEFAULT_MODEL_PATH = Path("models/waypoint_predictor_h5.pth")
DEFAULT_OUTPUT_DIR = Path("outputs/cbam_attention")
DEFAULT_CARLA_CONFIG = Path("configs/carla_env.yaml")
COMMAND_LABELS = {
    0: "LANE_FOLLOW",
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
}


def clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def normalize_map(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if lower is None:
        lower = float(np.percentile(values, 2.0))
    if upper is None:
        upper = float(np.percentile(values, 98.0))
    if upper <= lower + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return clamp01((values - lower) / (upper - lower))


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data if isinstance(data, dict) else {}


def cfg_get(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    section_value = config.get(section, {})
    if isinstance(section_value, dict):
        return section_value.get(key, default)
    return default


def carla_image_to_rgb(image: Any) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def drain_queue(frame_queue: queue.Queue[Any]) -> None:
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            return


def parse_spawn_indices(raw: str | None, start: int, count: int) -> list[int]:
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [start + offset * 3 for offset in range(max(1, count))]


def command_label(command: int) -> str:
    return COMMAND_LABELS.get(int(command), f"CMD_{int(command)}")


def command_suffix(command: int) -> str:
    return f"cmd{int(command)}_{command_label(command).lower()}"


def make_synthetic_scene(width: int, height: int, variant: str) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * 0.46)

    for y in range(height):
        t = y / max(1, height - 1)
        if y < horizon:
            sky = np.array([120, 175, 225], dtype=np.float32) * (1.0 - t) + np.array([225, 235, 240], dtype=np.float32) * t
            image[y, :, :] = sky.astype(np.uint8)
        else:
            road_t = (y - horizon) / max(1, height - horizon)
            road = np.array([55, 58, 60], dtype=np.float32) * (1.0 - road_t) + np.array([82, 82, 80], dtype=np.float32) * road_t
            image[y, :, :] = road.astype(np.uint8)

    cv2.rectangle(image, (0, horizon), (width, height), (72, 83, 70), thickness=-1)
    cv2.fillPoly(
        image,
        [np.array([(0, horizon), (int(width * 0.22), height), (0, height)], dtype=np.int32)],
        (82, 105, 78),
    )
    cv2.fillPoly(
        image,
        [np.array([(width, horizon), (int(width * 0.78), height), (width, height)], dtype=np.int32)],
        (82, 105, 78),
    )

    if variant == "left_curve":
        lane_center_top = int(width * 0.54)
        lane_center_bottom = int(width * 0.48)
        curve = -0.24
    elif variant == "right_curve":
        lane_center_top = int(width * 0.47)
        lane_center_bottom = int(width * 0.52)
        curve = 0.22
    else:
        lane_center_top = int(width * 0.50)
        lane_center_bottom = int(width * 0.50)
        curve = 0.0

    ys = np.linspace(horizon + 3, height - 1, 160)
    progress = (ys - horizon) / max(1, height - horizon)
    center = lane_center_top + (lane_center_bottom - lane_center_top) * progress + curve * width * (progress - 0.5) ** 2
    lane_half_width = width * (0.05 + 0.18 * progress)
    left_line = np.stack([center - lane_half_width, ys], axis=1).astype(np.int32)
    right_line = np.stack([center + lane_half_width, ys], axis=1).astype(np.int32)
    mid_line = np.stack([center, ys], axis=1).astype(np.int32)

    cv2.polylines(image, [left_line], False, (235, 235, 220), thickness=7, lineType=cv2.LINE_AA)
    cv2.polylines(image, [right_line], False, (235, 235, 220), thickness=7, lineType=cv2.LINE_AA)
    for idx in range(0, len(mid_line) - 14, 24):
        cv2.line(image, tuple(mid_line[idx]), tuple(mid_line[idx + 12]), (220, 210, 80), 5, cv2.LINE_AA)

    if variant == "right_curve":
        cv2.rectangle(image, (int(width * 0.66), int(height * 0.55)), (int(width * 0.77), int(height * 0.70)), (70, 74, 82), -1)
        cv2.rectangle(image, (int(width * 0.68), int(height * 0.58)), (int(width * 0.75), int(height * 0.66)), (130, 160, 190), -1)
    elif variant == "left_curve":
        cv2.rectangle(image, (int(width * 0.19), int(height * 0.52)), (int(width * 0.30), int(height * 0.67)), (115, 90, 65), -1)
        cv2.rectangle(image, (int(width * 0.21), int(height * 0.55)), (int(width * 0.28), int(height * 0.63)), (170, 190, 205), -1)
    else:
        shadow = np.array([(0, int(height * 0.66)), (width, int(height * 0.56)), (width, int(height * 0.69)), (0, int(height * 0.79))], dtype=np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [shadow], (35, 38, 40))
        image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)

    return image


def capture_carla_temporal_samples(args: argparse.Namespace) -> list[tuple[str, list[np.ndarray]]]:
    try:
        import carla
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Cannot import carla. Set PYTHONPATH to <CARLA_ROOT>/PythonAPI/carla before using --carla-live."
        ) from exc

    config = load_config(args.carla_config)
    host = str(args.carla_host or cfg_get(config, "carla", "host", "127.0.0.1"))
    port = int(args.carla_port or cfg_get(config, "carla", "port", 2000))
    map_name = str(args.carla_map or cfg_get(config, "carla", "map", "Town06"))
    timeout = float(args.carla_timeout or cfg_get(config, "carla", "timeout", 60.0))
    fixed_delta = float(cfg_get(config, "carla", "fixed_delta", 0.03))
    width = int(cfg_get(config, "camera", "width", 640))
    height = int(cfg_get(config, "camera", "height", 360))
    fov = float(cfg_get(config, "camera", "fov", 90.0))
    vehicle_filter = str(cfg_get(config, "vehicle", "filter", "vehicle.tesla.model3"))
    configured_spawn = int(cfg_get(config, "vehicle", "spawn_point", 0))
    weather_preset = str(cfg_get(config, "weather", "preset", "ClearNoon"))
    spawn_indices = parse_spawn_indices(args.carla_spawn_indices, configured_spawn, args.carla_count)

    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    if args.carla_reload_world and not world.get_map().name.endswith(map_name):
        world = client.load_world(map_name)
    map_name = str(world.get_map().name).replace("\\", "/").split("/")[-1]

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
        vehicle_blueprints = blueprint_library.filter(vehicle_filter) or blueprint_library.filter("vehicle.*")
        if not vehicle_blueprints:
            raise RuntimeError("No vehicle blueprint available in CARLA.")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in current CARLA map.")

        spawn_indices = [idx % len(spawn_points) for idx in spawn_indices]
        vehicle = world.try_spawn_actor(vehicle_blueprints[0], spawn_points[spawn_indices[0]])
        if vehicle is None:
            for transform in spawn_points:
                vehicle = world.try_spawn_actor(vehicle_blueprints[0], transform)
                if vehicle is not None:
                    break
        if vehicle is None:
            raise RuntimeError("Could not spawn ego vehicle for CARLA capture.")
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

        samples: list[tuple[str, list[np.ndarray]]] = []
        raw_dir = args.output_dir / "carla_inputs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        frame_count = max(3, int(args.carla_temporal_frames))
        frame_skip = max(1, int(args.carla_frame_skip))

        for spawn_idx in spawn_indices:
            vehicle.set_transform(spawn_points[spawn_idx])
            vehicle.set_target_velocity(carla.Vector3D())
            vehicle.set_target_angular_velocity(carla.Vector3D())
            drain_queue(frame_queue)
            for _ in range(max(1, int(args.carla_warmup_ticks))):
                world.tick()

            frames: list[np.ndarray] = []
            for frame_idx in range(frame_count):
                for _ in range(frame_skip):
                    world.tick()
                image = frame_queue.get(timeout=timeout)
                rgb = carla_image_to_rgb(image)
                frames.append(rgb)
                save_rgb(raw_dir / f"{map_name}_spawn_{spawn_idx:03d}_frame_{frame_idx:02d}.png", rgb)

            samples.append((f"{map_name}_spawn_{spawn_idx:03d}", frames[-3:]))
        return samples
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


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def make_temporal_frames(rgb: np.ndarray) -> list[np.ndarray]:
    height, width = rgb.shape[:2]
    shifts = [-4, 0, 4]
    frames = []
    for shift in shifts:
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
        shifted = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames.append(shifted)
    return frames


def preprocess_frames(frames: Iterable[np.ndarray], device: torch.device) -> torch.Tensor:
    yuv_frames = []
    for frame in frames:
        height = frame.shape[0]
        cropped = frame[int(height * 0.45) :, :, :]
        resized = cv2.resize(cropped, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        yuv_frames.append(cv2.cvtColor(resized, cv2.COLOR_RGB2YUV))
    stacked = np.concatenate(yuv_frames, axis=-1)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).float().div_(255.0)
    tensor.sub_(0.5).div_(0.5)
    return tensor.unsqueeze(0).to(device)


def load_model(model_path: Path, device: torch.device) -> CIL_NvidiaCNN:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = unwrap_state_dict(checkpoint)
    model = CIL_NvidiaCNN(pretrained_backbone=False).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def extract_cbam_maps(
    model: CIL_NvidiaCNN,
    image_tensor: torch.Tensor,
    command: int,
    speed_kmh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del command, speed_kmh
    with torch.inference_mode():
        p2 = model.stem(image_tensor)
        p2 = model.adapter(p2)
        s1 = model.backbone_stage1(p2)
        s2 = model.backbone_stage2(s1)
        s3 = model.backbone_stage3(s2)
        before = model.backbone_stage4(s3)
        channel_attention = model.cbam.channel_attn(before)
        after_channel = before * channel_attention
        spatial_attention = model.cbam.spatial_attn(after_channel)
        after = after_channel * spatial_attention

    before_map = before.detach().abs().mean(dim=1)[0].cpu().numpy()
    after_map = after.detach().abs().mean(dim=1)[0].cpu().numpy()
    spatial_map = spatial_attention.detach()[0, 0].cpu().numpy()
    channel_weights = channel_attention.detach()[0, :, 0, 0].cpu().numpy()
    return before_map, after_map, spatial_map, channel_weights


def full_frame_heatmap(
    heatmap: np.ndarray,
    original_shape: tuple[int, int, int],
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    height, width = original_shape[:2]
    crop_y = int(height * 0.45)
    crop_h = height - crop_y
    resized = cv2.resize(heatmap.astype(np.float32), (width, crop_h), interpolation=interpolation)
    full = np.zeros((height, width), dtype=np.float32)
    full[crop_y:, :] = resized
    return clamp01(full)


def draw_feature_grid(rgb: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    output = rgb.copy()
    height, width = output.shape[:2]
    crop_y = int(height * 0.45)
    grid_h, grid_w = grid_shape
    for idx in range(1, grid_w):
        x = int(round(idx * width / max(1, grid_w)))
        cv2.line(output, (x, crop_y), (x, height - 1), (255, 255, 255), 1, cv2.LINE_AA)
    for idx in range(1, grid_h):
        y = crop_y + int(round(idx * (height - crop_y) / max(1, grid_h)))
        cv2.line(output, (0, y), (width - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, crop_y), (width - 1, height - 1), (255, 255, 255), 1)
    return output


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, title: str, alpha: float = 0.54) -> np.ndarray:
    heatmap_u8 = np.uint8(clamp01(heatmap) * 255.0)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0.0)
    cv2.rectangle(overlay, (12, 12), (440, 54), (0, 0, 0), thickness=-1)
    cv2.putText(overlay, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def annotate_original(rgb: np.ndarray, title: str) -> np.ndarray:
    output = rgb.copy()
    cv2.rectangle(output, (12, 12), (360, 54), (0, 0, 0), thickness=-1)
    cv2.putText(output, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def build_comparison(original: np.ndarray, before: np.ndarray, after: np.ndarray, spatial: np.ndarray) -> np.ndarray:
    top = np.concatenate([original, before], axis=1)
    bottom = np.concatenate([spatial, after], axis=1)
    return np.concatenate([top, bottom], axis=0)


def annotate_command_banner(rgb: np.ndarray, command: int, speed_kmh: float) -> np.ndarray:
    output = rgb.copy()
    text = f"Command {int(command)}: {command_label(command)} | speed={float(speed_kmh):.1f} km/h"
    cv2.rectangle(output, (12, output.shape[0] - 52), (560, output.shape[0] - 12), (0, 0, 0), thickness=-1)
    cv2.putText(output, text, (24, output.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def create_visualization(
    model: CIL_NvidiaCNN,
    rgb: np.ndarray,
    name: str,
    output_dir: Path,
    device: torch.device,
    command: int,
    speed_kmh: float,
    temporal_frames: Iterable[np.ndarray] | None = None,
) -> dict[str, Path]:
    command = max(0, min(3, int(command)))
    frames = list(temporal_frames) if temporal_frames is not None else make_temporal_frames(rgb)
    if len(frames) < 3:
        frames = [frames[-1] if frames else rgb] * (3 - len(frames)) + frames
    frames = frames[-3:]
    image_tensor = preprocess_frames(frames, device)
    before_raw, after_raw, spatial_raw, channel_weights = extract_cbam_maps(model, image_tensor, command, speed_kmh)

    lower = float(np.percentile(np.concatenate([before_raw.ravel(), after_raw.ravel()]), 2.0))
    upper = float(np.percentile(np.concatenate([before_raw.ravel(), after_raw.ravel()]), 98.0))
    before_map = full_frame_heatmap(normalize_map(before_raw, lower, upper), rgb.shape)
    after_map = full_frame_heatmap(normalize_map(after_raw, lower, upper), rgb.shape)
    spatial_map = full_frame_heatmap(normalize_map(spatial_raw, 0.0, 1.0), rgb.shape, interpolation=cv2.INTER_NEAREST)

    original_img = annotate_original(rgb, f"Original input | Command {command}: {command_label(command)}")
    before_img = overlay_heatmap(rgb, before_map, "Before CBAM: raw Stage-4 activation")
    after_img = overlay_heatmap(rgb, after_map, "After CBAM: refined activation")
    spatial_img = overlay_heatmap(rgb, spatial_map, "CBAM spatial attention mask")
    spatial_grid_img = draw_feature_grid(spatial_img, spatial_raw.shape)

    comparison = annotate_command_banner(
        build_comparison(original_img, before_img, spatial_img, after_img),
        command,
        speed_kmh,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{name}_{command_suffix(command)}"
    paths = {
        "original": output_dir / f"{name}_00_original.png",
        "before": output_dir / f"{name}_01_before_cbam.png",
        "spatial": output_dir / f"{name}_02_cbam_spatial_mask.png",
        "after": output_dir / f"{name}_03_after_cbam.png",
        "compare": output_dir / f"{name}_04_compare.png",
        "channel": output_dir / f"{name}_05_channel_weights.png",
        "grid": output_dir / f"{name}_06_cbam_spatial_grid.png",
    }
    save_rgb(paths["original"], original_img)
    save_rgb(paths["before"], before_img)
    save_rgb(paths["spatial"], spatial_img)
    save_rgb(paths["after"], after_img)
    save_rgb(paths["compare"], comparison)
    save_channel_weights(paths["channel"], channel_weights)
    save_rgb(paths["grid"], spatial_grid_img)
    return paths


def save_channel_weights(path: Path, weights: np.ndarray) -> None:
    width, height = 880, 260
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    values = normalize_map(weights, float(weights.min()), float(weights.max()))
    margin = 46
    bar_width = max(1, (width - 2 * margin) // len(values))
    for idx, value in enumerate(values):
        x0 = margin + idx * bar_width
        x1 = min(width - margin, x0 + bar_width)
        y1 = height - 44
        y0 = int(y1 - value * 170)
        color = (int(40 + 200 * value), int(90 + 80 * value), int(210 - 120 * value))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.putText(canvas, "CBAM channel attention weights (440 channels)", (28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "low", (28, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, "high", (width - 78, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize pre/post CBAM attention maps for the current WaypointPredictor.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--images", type=Path, nargs="*", default=None, help="Optional RGB road images. If omitted, synthetic road scenes are generated.")
    parser.add_argument("--temporal-images", type=Path, nargs=3, default=None, help="Three ordered RGB frames for one temporal CIL sample: old middle current.")
    parser.add_argument("--carla-live", action="store_true", help="Capture real RGB frames from a running CARLA server and visualize CBAM on them.")
    parser.add_argument("--carla-config", type=Path, default=DEFAULT_CARLA_CONFIG)
    parser.add_argument("--carla-host", default=None)
    parser.add_argument("--carla-port", type=int, default=None)
    parser.add_argument("--carla-map", default=None)
    parser.add_argument("--carla-reload-world", action="store_true", help="Load the configured map before capture if the current CARLA map differs.")
    parser.add_argument("--carla-spawn-indices", default=None, help="Comma-separated spawn indices. Default: config spawn, spawn+3, ...")
    parser.add_argument("--carla-count", type=int, default=1, help="Number of CARLA spawn locations to capture.")
    parser.add_argument("--carla-temporal-frames", type=int, default=3, help="Number of sensor frames to capture per sample.")
    parser.add_argument("--carla-frame-skip", type=int, default=2, help="World ticks between captured temporal frames.")
    parser.add_argument("--carla-warmup-ticks", type=int, default=8)
    parser.add_argument("--carla-timeout", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--command", type=int, default=0)
    parser.add_argument("--commands", type=int, nargs="*", default=None, help="Optional per-sample commands. Values cycle if fewer than samples.")
    parser.add_argument("--speed-kmh", type=float, default=30.0)
    args = parser.parse_args()
    selected_sources = sum(bool(value) for value in (args.carla_live, args.images, args.temporal_images))
    if selected_sources > 1:
        parser.error("--carla-live, --images, and --temporal-images are mutually exclusive.")
    return args


def command_for_sample(args: argparse.Namespace, index: int) -> int:
    commands = args.commands or [args.command]
    if not commands:
        return int(args.command)
    return max(0, min(3, int(commands[index % len(commands)])))


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["sample", "command", "command_label", "speed_kmh", "compare_path", "grid_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(device_name)

    model = load_model(args.model, device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.carla_live:
        samples = [
            (name, frames[-1], frames)
            for name, frames in capture_carla_temporal_samples(args)
        ]
    elif args.temporal_images:
        frames = [load_rgb(path) for path in args.temporal_images]
        samples = [(args.temporal_images[-1].stem, frames[-1], frames)]
    elif args.images:
        samples = [(path.stem, load_rgb(path), None) for path in args.images]
    else:
        sample_dir = output_dir / "synthetic_inputs"
        samples = []
        for variant in ("straight_shadow", "left_curve", "right_curve"):
            rgb = make_synthetic_scene(640, 360, variant)
            raw_path = sample_dir / f"{variant}.png"
            save_rgb(raw_path, rgb)
            samples.append((variant, rgb, None))

    written: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    for sample_idx, (name, rgb, temporal_frames) in enumerate(samples):
        command = command_for_sample(args, sample_idx)
        paths = create_visualization(
            model,
            rgb,
            name,
            output_dir,
            device,
            command,
            args.speed_kmh,
            temporal_frames=temporal_frames,
        )
        written.extend(paths.values())
        manifest_rows.append(
            {
                "sample": name,
                "command": int(command),
                "command_label": command_label(command),
                "speed_kmh": f"{float(args.speed_kmh):.3f}",
                "compare_path": str(paths["compare"]),
                "grid_path": str(paths["grid"]),
            }
        )

    if manifest_rows:
        written.append(write_manifest(output_dir, manifest_rows))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
