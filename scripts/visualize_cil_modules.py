from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_cbam_attention import (  # noqa: E402
    COMMAND_LABELS,
    DEFAULT_CARLA_CONFIG,
    clamp01,
    capture_carla_temporal_samples,
    command_label,
    full_frame_heatmap,
    load_model,
    load_rgb,
    make_temporal_frames,
    normalize_map,
    overlay_heatmap,
    preprocess_frames,
    save_rgb,
)


DEFAULT_MODEL_PATH = Path("models/waypoint_predictor_h5.pth")
DEFAULT_TEMPORAL_DIR = Path("outputs/cbam_attention/carla_inputs")
DEFAULT_OUTPUT_DIR = Path("outputs/cil_module_visualizations")
CIL_MAX_SPEED_KMH = 120.0


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def speed_to_model_input(speed_kmh: float) -> float:
    return float(np.clip(float(speed_kmh) / CIL_MAX_SPEED_KMH, 0.0, 1.0))


def load_temporal_samples(temporal_dir: Path, max_samples: int) -> list[tuple[str, list[np.ndarray]]]:
    pattern = re.compile(r"(.+)_frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(temporal_dir.glob("*")):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        groups.setdefault(match.group(1), []).append((int(match.group(2)), path))

    samples: list[tuple[str, list[np.ndarray]]] = []
    for name, indexed_paths in sorted(groups.items()):
        ordered = [path for _idx, path in sorted(indexed_paths)]
        if len(ordered) < 3:
            continue
        frames = [load_rgb(path) for path in ordered[-3:]]
        samples.append((safe_name(name), frames))
        if len(samples) >= max(1, int(max_samples)):
            break
    return samples


def load_image_samples(paths: list[Path], max_samples: int) -> list[tuple[str, list[np.ndarray]]]:
    samples: list[tuple[str, list[np.ndarray]]] = []
    for path in paths[: max(1, int(max_samples))]:
        rgb = load_rgb(path)
        samples.append((safe_name(path.stem), make_temporal_frames(rgb)))
    return samples


def activation_map(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().abs().mean(dim=1)[0].float().cpu().numpy()


def extract_internals(
    model: Any,
    image_tensor: torch.Tensor,
    command: int,
    speed_kmh: float,
) -> dict[str, Any]:
    command = int(np.clip(int(command), 0, 3))
    device = image_tensor.device
    command_tensor = torch.tensor([command], dtype=torch.long, device=device)
    speed_value = speed_to_model_input(speed_kmh)
    speed_tensor = torch.tensor([speed_value], dtype=torch.float32, device=device)

    with torch.inference_mode():
        p2_stem = model.stem(image_tensor)
        p2 = model.adapter(p2_stem)
        s1 = model.backbone_stage1(p2)
        s2 = model.backbone_stage2(s1)
        s3 = model.backbone_stage3(s2)
        s4_raw = model.backbone_stage4(s3)

        channel_attention = model.cbam.channel_attn(s4_raw)
        s4_channel = s4_raw * channel_attention
        spatial_attention = model.cbam.spatial_attn(s4_channel)
        s4_cbam = s4_channel * spatial_attention

        commands = torch.arange(4, dtype=torch.long, device=device)
        speeds = torch.full((4,), speed_value, dtype=torch.float32, device=device)
        film_condition = torch.cat(
            [model.film.embedding(commands), speeds.view(-1, 1)],
            dim=1,
        )
        gamma_beta_all = model.film.mlp(film_condition)
        gamma_all, beta_all = torch.split(gamma_beta_all, model.film.channels, dim=1)

        gamma_current = gamma_all[command : command + 1].view(1, model.film.channels, 1, 1)
        beta_current = beta_all[command : command + 1].view(1, model.film.channels, 1, 1)
        s4_film = s4_cbam * gamma_current + beta_current

        f3, f4, f5 = model.fpn(s2, s3, s4_film)
        fused = (
            f3
            + F.interpolate(f4, size=f3.shape[-2:], mode="nearest")
            + F.interpolate(f5, size=f3.shape[-2:], mode="nearest")
        )
        features = model.shared_features(fused)
        wp_out = model.waypoint_head(features)
        coords = torch.tanh(wp_out[:, :10])
        coords_view = coords.view(-1, 5, 2)
        x_scaled = coords_view[..., 0] * model.scaling.x_scale
        y_scaled = coords_view[..., 1] * model.scaling.y_scale
        waypoints = torch.stack([x_scaled, y_scaled], dim=-1)[0]
        sigma = F.softplus(wp_out[:, 10:15]) * model.scaling.sigma_scale + model.scaling.sigma_eps
        speed_pred = model.speed_head(features)

    return {
        "p2_stem": p2_stem.detach(),
        "s4_raw": s4_raw.detach(),
        "s4_cbam": s4_cbam.detach(),
        "spatial_attention": spatial_attention.detach(),
        "channel_attention": channel_attention.detach(),
        "gamma": gamma_all.detach().float().cpu().numpy(),
        "beta": beta_all.detach().float().cpu().numpy(),
        "fpn": (f3.detach(), f4.detach(), f5.detach()),
        "fused": fused.detach(),
        "waypoints": waypoints.detach().float().cpu().numpy(),
        "sigma": sigma.detach()[0].float().cpu().numpy(),
        "speed_pred": float(speed_pred.detach()[0, 0].float().cpu().item()),
    }


def add_title(rgb: np.ndarray, title: str, subtitle: str | None = None) -> np.ndarray:
    output = rgb.copy()
    height, width = output.shape[:2]
    box_h = 58 if subtitle else 44
    cv2.rectangle(output, (10, 10), (min(width - 10, 620), 10 + box_h), (0, 0, 0), -1)
    cv2.putText(output, title, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(output, subtitle, (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1, cv2.LINE_AA)
    return output


def make_motion_heatmap(frames: list[np.ndarray]) -> np.ndarray:
    f1, f2, f3 = [frame.astype(np.float32) for frame in frames[-3:]]
    diff_12 = np.mean(np.abs(f2 - f1), axis=2)
    diff_23 = np.mean(np.abs(f3 - f2), axis=2)
    motion = 0.45 * diff_12 + 0.55 * diff_23
    motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=5.0)
    return normalize_map(motion)


def visualize_physics(frames: list[np.ndarray]) -> np.ndarray:
    rgb = frames[-1]
    motion = make_motion_heatmap(frames)
    return overlay_heatmap(rgb, motion, "PhysicsAwareStem: frame-diff motion", alpha=0.58)


def visualize_cbam(rgb: np.ndarray, internals: dict[str, Any]) -> np.ndarray:
    before_raw = activation_map(internals["s4_raw"])
    after_raw = activation_map(internals["s4_cbam"])
    spatial_raw = internals["spatial_attention"].detach()[0, 0].float().cpu().numpy()

    lower = float(np.percentile(np.concatenate([before_raw.ravel(), after_raw.ravel()]), 2.0))
    upper = float(np.percentile(np.concatenate([before_raw.ravel(), after_raw.ravel()]), 98.0))
    before_map = full_frame_heatmap(normalize_map(before_raw, lower, upper), rgb.shape)
    after_map = full_frame_heatmap(normalize_map(after_raw, lower, upper), rgb.shape)
    spatial_map = full_frame_heatmap(normalize_map(spatial_raw, 0.0, 1.0), rgb.shape, interpolation=cv2.INTER_NEAREST)

    before = overlay_heatmap(rgb, before_map, "Before CBAM")
    spatial = overlay_heatmap(rgb, spatial_map, "CBAM spatial mask")
    after = overlay_heatmap(rgb, after_map, "After CBAM")
    return np.concatenate([resize_panel(before), resize_panel(spatial), resize_panel(after)], axis=1)


def symmetric_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(values)))
    if max_abs < 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    return clamp01(0.5 + 0.5 * values / max_abs)


def heat_strip(values: np.ndarray, width: int, height: int, label: str, center: float = 0.0) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32) - float(center)
    normalized = symmetric_normalize(rows)
    strip = np.uint8(normalized * 255.0)
    strip = cv2.resize(strip, (width, height), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(strip, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    for idx, command in enumerate(range(4)):
        y = int((idx + 0.62) * height / 4)
        cv2.putText(colored, COMMAND_LABELS[command], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(colored, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return colored


def visualize_film(gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    width, height = 980, 620
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "FiLM conditioning: gamma / beta by command", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)

    gamma_mean = gamma.mean(axis=1)
    beta_mean = beta.mean(axis=1)
    gamma_std = gamma.std(axis=1)
    beta_std = beta.std(axis=1)
    metrics = [
        ("gamma mean", gamma_mean, (70, 130, 230)),
        ("beta mean", beta_mean, (50, 170, 80)),
        ("gamma std", gamma_std, (230, 140, 50)),
        ("beta std", beta_std, (170, 80, 190)),
    ]
    chart_x, chart_y, chart_w, chart_h = 70, 78, 840, 210
    cv2.rectangle(canvas, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (230, 230, 230), 1)
    all_values = np.concatenate([values for _name, values, _color in metrics])
    y_min = min(0.0, float(all_values.min()))
    y_max = max(1.0, float(all_values.max()))
    if y_max <= y_min + 1e-6:
        y_max = y_min + 1.0

    def y_from_value(value: float) -> int:
        ratio = (float(value) - y_min) / (y_max - y_min)
        return int(chart_y + chart_h - ratio * chart_h)

    zero_y = y_from_value(0.0)
    cv2.line(canvas, (chart_x, zero_y), (chart_x + chart_w, zero_y), (140, 140, 140), 1, cv2.LINE_AA)
    group_w = chart_w // 4
    bar_w = 24
    for cmd in range(4):
        base_x = chart_x + cmd * group_w + 35
        cv2.putText(canvas, COMMAND_LABELS[cmd], (base_x - 12, chart_y + chart_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
        for metric_idx, (_name, values, color) in enumerate(metrics):
            x0 = base_x + metric_idx * (bar_w + 4)
            y_val = y_from_value(float(values[cmd]))
            y0, y1 = min(y_val, zero_y), max(y_val, zero_y)
            cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y1), color, -1)
            cv2.putText(canvas, f"{float(values[cmd]):.2f}", (x0 - 8, max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (50, 50, 50), 1, cv2.LINE_AA)

    legend_x = 70
    for idx, (name, _values, color) in enumerate(metrics):
        x = legend_x + idx * 190
        cv2.rectangle(canvas, (x, 308), (x + 20, 328), color, -1)
        cv2.putText(canvas, name, (x + 28, 326), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, cv2.LINE_AA)

    gamma_strip = heat_strip(gamma, 900, 92, "gamma channel heatmap (center=1)", center=1.0)
    beta_strip = heat_strip(beta, 900, 92, "beta channel heatmap (center=0)", center=0.0)
    canvas[350:442, 40:940] = gamma_strip
    canvas[478:570, 40:940] = beta_strip
    return canvas


def visualize_fpn(rgb: np.ndarray, internals: dict[str, Any]) -> np.ndarray:
    f3, f4, f5 = internals["fpn"]
    labels = [
        "FPN P3: detail scale",
        "FPN P4: mid semantic scale",
        "FPN P5: global semantic scale",
    ]
    raw_maps = [activation_map(f3), activation_map(f4), activation_map(f5)]
    panels = []
    for raw_map, label in zip(raw_maps, labels):
        full = full_frame_heatmap(normalize_map(raw_map), rgb.shape)
        panels.append(resize_panel(overlay_heatmap(rgb, full, label)))
    return np.concatenate(panels, axis=1)


def visualize_sigma(sigma: np.ndarray, command: int, speed_pred: float) -> np.ndarray:
    width, height = 760, 420
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, "Bayesian waypoint uncertainty head", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"command={command_label(command)} | speed_head={float(speed_pred) * CIL_MAX_SPEED_KMH:.1f} km/h",
        (28, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (75, 75, 75),
        1,
        cv2.LINE_AA,
    )
    left, top, chart_w, chart_h = 78, 112, 610, 235
    max_sigma = max(1.0, float(np.max(sigma)) * 1.25)
    cv2.rectangle(canvas, (left, top), (left + chart_w, top + chart_h), (220, 220, 220), 1)
    for tick in range(5):
        value = max_sigma * tick / 4
        y = int(top + chart_h - (value / max_sigma) * chart_h)
        cv2.line(canvas, (left, y), (left + chart_w, y), (230, 230, 230), 1)
        cv2.putText(canvas, f"{value:.1f}", (24, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    bar_w = 72
    gap = (chart_w - 5 * bar_w) // 6
    for idx, value in enumerate(sigma):
        x0 = left + gap + idx * (bar_w + gap)
        y0 = int(top + chart_h - (float(value) / max_sigma) * chart_h)
        color_ratio = float(value) / max_sigma
        color = (int(60 + 180 * color_ratio), int(165 - 80 * color_ratio), int(220 - 180 * color_ratio))
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, top + chart_h), color, -1)
        cv2.putText(canvas, f"WP{idx + 1}", (x0 + 15, top + chart_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{float(value):.2f}m", (x0 - 1, max(94, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"mean sigma={float(np.mean(sigma)):.2f}m", (28, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def waypoint_to_pixel(waypoint: np.ndarray, image_shape: tuple[int, int, int], max_forward: float) -> tuple[int, int]:
    height, width = image_shape[:2]
    forward = float(max(0.0, waypoint[0]))
    lateral = float(waypoint[1])
    progress = float(np.clip(forward / max(max_forward, 1.0), 0.0, 1.0))
    bottom_y = height - 8
    horizon_y = int(height * 0.46)
    image_y = bottom_y - int((progress ** 0.62) * (bottom_y - horizon_y))
    lateral_scale = width * (0.035 + 0.105 * (1.0 - progress))
    image_x = int(width * 0.5 + lateral * lateral_scale)
    return int(np.clip(image_x, 0, width - 1)), int(np.clip(image_y, 0, height - 1))


def visualize_waypoints(rgb: np.ndarray, waypoints: np.ndarray, sigma: np.ndarray, command: int) -> np.ndarray:
    output = add_title(
        rgb,
        "Waypoints + uncertainty circles",
        "ego-frame points projected onto camera image for presentation",
    )
    forward_values = np.maximum(waypoints[:, 0], 0.0)
    max_forward = max(18.0, float(np.max(forward_values)) + 5.0)
    points = [waypoint_to_pixel(wp, output.shape, max_forward) for wp in waypoints]

    for idx in range(1, len(points)):
        cv2.line(output, points[idx - 1], points[idx], (255, 210, 40), 3, cv2.LINE_AA)

    max_sigma = max(1.0, float(np.max(sigma)))
    for idx, (point, wp_sigma) in enumerate(zip(points, sigma)):
        radius = int(np.clip(8 + 34 * float(wp_sigma) / max_sigma, 8, 54))
        cv2.circle(output, point, radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(output, point, 5, (255, 210, 40), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{idx + 1}: {float(wp_sigma):.1f}m",
            (point[0] + 8, max(18, point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"command={command_label(command)}",
        (22, output.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def resize_panel(rgb: np.ndarray, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def make_contact_sheet(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    panel_w, panel_h = 640, 360
    cells = []
    for title, panel in panels:
        resized = resize_panel(panel, (panel_w, panel_h))
        cv2.rectangle(resized, (0, 0), (panel_w - 1, 30), (0, 0, 0), -1)
        cv2.putText(resized, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(resized)
    while len(cells) < 6:
        cells.append(np.full((panel_h, panel_w, 3), 245, dtype=np.uint8))
    top = np.concatenate(cells[:3], axis=1)
    bottom = np.concatenate(cells[3:6], axis=1)
    return np.concatenate([top, bottom], axis=0)


def create_visualizations(
    model: Any,
    name: str,
    frames: list[np.ndarray],
    output_dir: Path,
    device: torch.device,
    command: int,
    speed_kmh: float,
) -> dict[str, Path]:
    rgb = frames[-1]
    image_tensor = preprocess_frames(frames[-3:], device)
    internals = extract_internals(model, image_tensor, command, speed_kmh)

    physics = visualize_physics(frames)
    cbam = visualize_cbam(rgb, internals)
    film = visualize_film(internals["gamma"], internals["beta"])
    fpn = visualize_fpn(rgb, internals)
    sigma = visualize_sigma(internals["sigma"], command, internals["speed_pred"])
    waypoints = visualize_waypoints(rgb, internals["waypoints"], internals["sigma"], command)
    contact = make_contact_sheet(
        [
            ("PhysicsAwareStem motion", physics),
            ("CBAM attention", cbam),
            ("FiLM gamma/beta", film),
            ("FPN 3-scale maps", fpn),
            ("Bayesian sigma", sigma),
            ("Waypoints + uncertainty", waypoints),
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{safe_name(name)}_cmd{int(command)}_{COMMAND_LABELS[int(command)].lower()}"
    paths = {
        "physics": Path(f"{prefix}_01_physics_motion.png"),
        "cbam": Path(f"{prefix}_02_cbam_attention.png"),
        "film": Path(f"{prefix}_03_film_gamma_beta.png"),
        "fpn": Path(f"{prefix}_04_fpn_scales.png"),
        "bayesian": Path(f"{prefix}_05_bayesian_sigma.png"),
        "waypoints": Path(f"{prefix}_06_waypoints_uncertainty.png"),
        "contact": Path(f"{prefix}_00_all_modules_contact_sheet.png"),
    }
    save_rgb(paths["physics"], physics)
    save_rgb(paths["cbam"], cbam)
    save_rgb(paths["film"], film)
    save_rgb(paths["fpn"], fpn)
    save_rgb(paths["bayesian"], sigma)
    save_rgb(paths["waypoints"], waypoints)
    save_rgb(paths["contact"], contact)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PhysicsAwareStem, CBAM, FiLM, FPN, Bayesian sigma, and waypoints for the CIL model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--temporal-dir",
        type=Path,
        default=DEFAULT_TEMPORAL_DIR,
        help=(
            "Disk: folder with <stem>_frame_<n>.png (same stem, ≥3 frames). "
            "With --carla-live: if this path ends with 'carla_inputs', PNGs are written here; "
            "otherwise they are written under <this-dir>/carla_inputs/. "
            f"Default: {DEFAULT_TEMPORAL_DIR}"
        ),
    )
    parser.add_argument("--images", type=Path, nargs="*", default=None, help="Fallback single RGB images; temporal frames are made by shifting each image.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--command", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--speed-kmh", type=float, default=30.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="Only export PhysicsAwareStem motion heatmaps from 3-frame sequences; no model checkpoint required.",
    )
    carla = parser.add_argument_group(
        "CARLA live capture",
        "Use --carla-live to grab 3-frame sequences from a running CARLA server, then run all module visualizations. "
        "Requires PYTHONPATH to include <CARLA>/PythonAPI/carla. Mutually exclusive with --images.",
    )
    carla.add_argument(
        "--carla-live",
        action="store_true",
        help="Capture RGB triplets from CARLA (writes under <capture-root>/carla_inputs/; see --temporal-dir).",
    )
    carla.add_argument("--carla-config", type=Path, default=DEFAULT_CARLA_CONFIG)
    carla.add_argument("--carla-host", default=None)
    carla.add_argument("--carla-port", type=int, default=None)
    carla.add_argument("--carla-map", default=None)
    carla.add_argument("--carla-reload-world", action="store_true")
    carla.add_argument("--carla-spawn-indices", default=None)
    carla.add_argument("--carla-count", type=int, default=1)
    carla.add_argument("--carla-temporal-frames", type=int, default=3)
    carla.add_argument("--carla-frame-skip", type=int, default=2)
    carla.add_argument("--carla-warmup-ticks", type=int, default=8)
    carla.add_argument("--carla-timeout", type=float, default=None)
    args = parser.parse_args()
    if args.carla_live and args.images:
        parser.error("--carla-live cannot be used together with --images.")
    return args


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sample",
                "command",
                "physics",
                "cbam",
                "film",
                "fpn",
                "bayesian",
                "waypoints",
                "contact",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _carla_capture_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """capture_carla_temporal_samples writes to output_dir / 'carla_inputs'."""
    td = args.temporal_dir
    if td.name == "carla_inputs":
        output_dir = td.parent
    else:
        output_dir = td
    return argparse.Namespace(
        output_dir=output_dir,
        carla_config=args.carla_config,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        carla_map=args.carla_map,
        carla_reload_world=args.carla_reload_world,
        carla_spawn_indices=args.carla_spawn_indices,
        carla_count=args.carla_count,
        carla_temporal_frames=args.carla_temporal_frames,
        carla_frame_skip=args.carla_frame_skip,
        carla_warmup_ticks=args.carla_warmup_ticks,
        carla_timeout=args.carla_timeout,
    )


def main() -> None:
    args = parse_args()
    if args.carla_live:
        samples = list(capture_carla_temporal_samples(_carla_capture_namespace(args)))
        if not samples:
            raise RuntimeError("CARLA live capture returned no samples (check spawn indices and server).")
    elif args.images:
        samples = load_image_samples(args.images, args.max_samples)
    else:
        samples = load_temporal_samples(args.temporal_dir, args.max_samples)
    samples = samples[: max(1, int(args.max_samples))]
    if not samples:
        raise RuntimeError(
            f"No usable samples under {args.temporal_dir.resolve()}. "
            "Either the folder is empty, or no files match the temporal naming pattern. "
            "Expected per-sequence files like: <name>_frame_0.png, <name>_frame_1.png, <name>_frame_2.png "
            "(at least 3 frames per <name>; extension .png/.jpg/.jpeg). "
            "Fix: (1) put such images in that folder, or (2) pass --temporal-dir PATH_TO_FRAMES, or "
            "(3) use still images: --images path/to/screenshot.png "
            "(the script builds a synthetic 3-frame stack from each image), or "
            "(4) capture from CARLA in one step: scripts/visualize_cil_modules.py --carla-live, or "
            "scripts/visualize_cbam_attention.py --carla-live then this script with matching --temporal-dir."
        )

    if args.physics_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        rows: list[dict[str, str]] = []
        for name, frames in samples:
            physics = visualize_physics(frames)
            stem = args.output_dir / f"{safe_name(name)}_physics_only_motion"
            path = Path(f"{stem}.png")
            save_rgb(path, physics)
            written.append(path)
            rows.append({"sample": name, "command": "-", "physics": str(path)})
        manifest = args.output_dir / "manifest_physics_only.csv"
        with manifest.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["sample", "command", "physics"])
            writer.writeheader()
            writer.writerows(rows)
        written.append(manifest)
        for path in written:
            print(path)
        return

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(device_name)

    model = load_model(args.model, device)
    rows: list[dict[str, str]] = []
    written: list[Path] = []
    for name, frames in samples:
        paths = create_visualizations(
            model=model,
            name=name,
            frames=frames,
            output_dir=args.output_dir,
            device=device,
            command=args.command,
            speed_kmh=args.speed_kmh,
        )
        written.extend(paths.values())
        rows.append(
            {
                "sample": name,
                "command": str(int(args.command)),
                **{key: str(path) for key, path in paths.items()},
            }
        )

    written.append(write_manifest(args.output_dir, rows))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
