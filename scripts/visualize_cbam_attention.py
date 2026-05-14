from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_perception.cnn_model import CIL_NvidiaCNN, unwrap_state_dict


MODEL_INPUT_SIZE = (200, 66)
DEFAULT_MODEL_PATH = Path("models/waypoint_predictor_h5.pth")
DEFAULT_OUTPUT_DIR = Path("outputs/cbam_attention")


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


def full_frame_heatmap(heatmap: np.ndarray, original_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = original_shape[:2]
    crop_y = int(height * 0.45)
    crop_h = height - crop_y
    resized = cv2.resize(heatmap.astype(np.float32), (width, crop_h), interpolation=cv2.INTER_CUBIC)
    full = np.zeros((height, width), dtype=np.float32)
    full[crop_y:, :] = resized
    return clamp01(full)


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


def create_visualization(
    model: CIL_NvidiaCNN,
    rgb: np.ndarray,
    name: str,
    output_dir: Path,
    device: torch.device,
    command: int,
    speed_kmh: float,
) -> dict[str, Path]:
    image_tensor = preprocess_frames(make_temporal_frames(rgb), device)
    before_raw, after_raw, spatial_raw, channel_weights = extract_cbam_maps(model, image_tensor, command, speed_kmh)

    lower = float(np.percentile(np.concatenate([before_raw.ravel(), after_raw.ravel()]), 2.0))
    upper = float(np.percentile(np.concatenate([before_raw.ravel(), after_raw.ravel()]), 98.0))
    before_map = full_frame_heatmap(normalize_map(before_raw, lower, upper), rgb.shape)
    after_map = full_frame_heatmap(normalize_map(after_raw, lower, upper), rgb.shape)
    spatial_map = full_frame_heatmap(normalize_map(spatial_raw, 0.0, 1.0), rgb.shape)

    original_img = annotate_original(rgb, "Original input")
    before_img = overlay_heatmap(rgb, before_map, "Before CBAM: raw Stage-4 activation")
    after_img = overlay_heatmap(rgb, after_map, "After CBAM: refined activation")
    spatial_img = overlay_heatmap(rgb, spatial_map, "CBAM spatial attention mask")

    comparison = build_comparison(original_img, before_img, spatial_img, after_img)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "original": output_dir / f"{name}_00_original.png",
        "before": output_dir / f"{name}_01_before_cbam.png",
        "spatial": output_dir / f"{name}_02_cbam_spatial_mask.png",
        "after": output_dir / f"{name}_03_after_cbam.png",
        "compare": output_dir / f"{name}_04_compare.png",
        "channel": output_dir / f"{name}_05_channel_weights.png",
    }
    save_rgb(paths["original"], original_img)
    save_rgb(paths["before"], before_img)
    save_rgb(paths["spatial"], spatial_img)
    save_rgb(paths["after"], after_img)
    save_rgb(paths["compare"], comparison)
    save_channel_weights(paths["channel"], channel_weights)
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
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--command", type=int, default=0)
    parser.add_argument("--speed-kmh", type=float, default=30.0)
    return parser.parse_args()


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

    if args.images:
        samples = [(path.stem, load_rgb(path)) for path in args.images]
    else:
        sample_dir = output_dir / "synthetic_inputs"
        samples = []
        for variant in ("straight_shadow", "left_curve", "right_curve"):
            rgb = make_synthetic_scene(640, 360, variant)
            raw_path = sample_dir / f"{variant}.png"
            save_rgb(raw_path, rgb)
            samples.append((variant, rgb))

    written: list[Path] = []
    for name, rgb in samples:
        paths = create_visualization(model, rgb, name, output_dir, device, args.command, args.speed_kmh)
        written.extend(paths.values())

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
