#!/usr/bin/env python3
"""
Offline & Online Evaluation Script for CIL Waypoint Model
Calculates: ADE, FDE, FPS, Latency, Route Complete, TTC

Usage:
  # Full offline evaluation (requires H5 dataset):
  set H5_PATH=C:/path/to/carla_images_drive.h5
  python scripts/evaluate_cnn_metrics.py

  # Benchmark-only mode (no dataset needed, measures FPS/Latency):
  python scripts/evaluate_cnn_metrics.py --benchmark-only

  # Show online metrics from last CARLA run:
  python scripts/evaluate_cnn_metrics.py --online-log outputs/telemetry.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.cnn_model import WaypointPredictor


# ============================================================================
# MODEL DISCOVERY
# ============================================================================
def find_best_checkpoint(models_dir: Path) -> Path:
    """Auto-detect the best waypoint model checkpoint."""
    # Priority order
    candidates = [
        "waypoint_predictor.pth",
        "waypoint_predictor_h5.pth",
    ]
    for name in candidates:
        p = models_dir / name
        if p.exists():
            return p

    # Fallback: any file with 'waypoint' or 'cil' in name
    for pattern in ["*waypoint*.pth", "*cil*.pth"]:
        matches = sorted(models_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]

    return models_dir / "waypoint_predictor.pth"  # will fail later with clear message


def load_model(device: torch.device) -> torch.nn.Module:
    """Load WaypointPredictor model with auto-detected checkpoint."""
    models_dir = PROJECT_ROOT / "models"
    checkpoint_path = find_best_checkpoint(models_dir)

    if not checkpoint_path.exists():
        print(f"[ERROR] Model checkpoint not found at: {checkpoint_path}")
        print(f"   Available .pth files in {models_dir}:")
        for f in sorted(models_dir.glob("*.pth")):
            print(f"     - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = WaypointPredictor().to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        sys.exit(1)

    model.eval()
    return model


# ============================================================================
# BENCHMARK: FPS & LATENCY (no dataset needed)
# ============================================================================
def benchmark_inference(model: torch.nn.Module, device: torch.device,
                        num_warmup: int = 20, num_iterations: int = 200) -> dict:
    """
    Measure model inference FPS and Latency using synthetic inputs.
    Returns dict with fps, latency_ms, latency_std_ms.
    """
    print(f"\n{'-'*60}")
    print(f" [BENCH]  INFERENCE BENCHMARK (device={device})")
    print(f"{'-'*60}")

    # Create synthetic inputs matching real pipeline dimensions
    # Input: [B, 9, 66, 200] (3 temporal YUV frames, cropped & resized)
    dummy_image = torch.randn(1, 9, 66, 200, device=device)
    dummy_command = torch.tensor([0], dtype=torch.long, device=device)
    dummy_speed = torch.tensor([0.5], dtype=torch.float32, device=device)

    # Warmup (let GPU kernels compile / cache)
    print(f"  Warming up ({num_warmup} iterations)...")
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy_image, dummy_command, dummy_speed)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_iterations} iterations)...")
    latencies = []

    with torch.inference_mode():
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_image, dummy_command, dummy_speed)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies_np = np.array(latencies)

    # Remove outliers (top/bottom 5%) for more stable measurement
    p5, p95 = np.percentile(latencies_np, [5, 95])
    filtered = latencies_np[(latencies_np >= p5) & (latencies_np <= p95)]

    mean_latency = float(np.mean(filtered))
    std_latency = float(np.std(filtered))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    # Also measure batch throughput
    batch_sizes = [1, 4, 16, 32]
    batch_fps = {}
    for bs in batch_sizes:
        try:
            b_img = torch.randn(bs, 9, 66, 200, device=device)
            b_cmd = torch.zeros(bs, dtype=torch.long, device=device)
            b_spd = torch.full((bs,), 0.5, dtype=torch.float32, device=device)

            # Warmup
            with torch.inference_mode():
                for _ in range(5):
                    _ = model(b_img, b_cmd, b_spd)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(50):
                    _ = model(b_img, b_cmd, b_spd)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_samples = 50 * bs
            batch_fps[bs] = total_samples / (t1 - t0)
            del b_img, b_cmd, b_spd
        except RuntimeError:
            batch_fps[bs] = 0.0  # OOM

    return {
        "fps": fps,
        "latency_ms": mean_latency,
        "latency_std_ms": std_latency,
        "batch_fps": batch_fps,
        "num_iterations": len(filtered),
    }


# ============================================================================
# OFFLINE: ADE & FDE (requires H5 dataset)
# ============================================================================
def evaluate_ade_fde(model: torch.nn.Module, device: torch.device,
                     h5_path: str, csv_root_hint: str | None) -> dict | None:
    """
    Calculate ADE (Average Displacement Error) and FDE (Final Displacement Error).
    Returns dict or None if dataset is unavailable.
    """
    try:
        import yaml
        import h5py
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
        from scripts.kaggle_train_h5 import WaypointCarlaDatasetH5, find_csv_root
    except ImportError as e:
        print(f"[WARN] Cannot run ADE/FDE evaluation: {e}")
        return None

    if not os.path.exists(h5_path):
        print(f"[WARN] H5 dataset not found at: {h5_path}")
        print("   Skipping ADE/FDE evaluation.")
        print("   To enable: set H5_PATH=C:/path/to/carla_images_drive.h5")
        return None

    # Load config
    config_path = PROJECT_ROOT / "configs" / "train_params.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"\n{'-'*60}")
    print(f" [ACCURACY] OFFLINE ADE/FDE EVALUATION")
    print(f"{'-'*60}")

    try:
        csv_root = find_csv_root(h5_path, csv_root_hint)
        with h5py.File(h5_path, "r") as f:
            towns = [k for k in f.keys() if isinstance(f[k], h5py.Group)]

        transform = transforms.Compose([transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)])
        val_dataset = WaypointCarlaDatasetH5(
            h5_path=h5_path,
            csv_root=csv_root,
            towns=towns,
            transform=transform,
            is_training=False,
            train_ratio=float(config.get("train_split", 0.75)),
            geometric_offset=float(config.get("geometric_offset", 0.35)),
            include_side_cameras=False,
        )

        def collate_fn(batch):
            imgs, wps, cmds, recs, speeds = zip(*batch)
            return (
                torch.stack(imgs),
                torch.stack(wps).float(),
                torch.stack(cmds).long(),
                torch.tensor(recs, dtype=torch.float32),
                torch.tensor(speeds, dtype=torch.float32),
            )

        val_loader = DataLoader(
            val_dataset, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fn
        )
        print(f"  Validation Dataset size: {len(val_dataset)} samples")
    except Exception as e:
        print(f"[ERROR] Error preparing dataset: {e}")
        return None

    # Evaluation Loop
    print("  Evaluating...")
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    inference_times = []

    with torch.inference_mode():
        for i, (imgs, wps, cmds, _, speeds) in enumerate(val_loader):
            imgs = imgs.to(device, non_blocking=True)
            cmds = cmds.to(device, non_blocking=True)
            wps = wps.to(device, non_blocking=True)
            speeds = speeds.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                out = model(imgs, cmds, speeds)
                pred_wp = out[:, :10].view(-1, 5, 2)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            inference_times.append((t1 - t0) * 1000.0)  # ms

            # wps and pred_wp shape: [batch_size, 5, 2]
            distances = torch.norm(pred_wp - wps, dim=-1)
            batch_ade = distances.mean(dim=1).sum().item()
            batch_fde = distances[:, 4].sum().item()

            total_ade += batch_ade
            total_fde += batch_fde
            total_samples += imgs.size(0)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i+1}/{len(val_loader)} batches...")

    if total_samples == 0:
        return None

    mean_ade = total_ade / total_samples
    mean_fde = total_fde / total_samples
    avg_batch_latency = float(np.mean(inference_times)) if inference_times else 0.0

    return {
        "ade_m": mean_ade,
        "fde_m": mean_fde,
        "total_samples": total_samples,
        "avg_batch_latency_ms": avg_batch_latency,
    }


# ============================================================================
# RESULTS DISPLAY
# ============================================================================
def print_results(benchmark: dict, ade_fde: dict | None) -> None:
    """Print comprehensive metrics report."""
    print("\n")
    print("=" * 60)
    print(" [TARGET] COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)

    # -- Section 1: Inference Performance --
    print("\n+---------------------------------------------------------+")
    print("|  [STATS] INFERENCE PERFORMANCE                              |")
    print("+---------------------------------------------------------+")
    print(f"|  FPS (batch=1):         {benchmark['fps']:>8.1f} FPS               |")
    print(f"|  Latency (batch=1):     {benchmark['latency_ms']:>8.2f} +- {benchmark['latency_std_ms']:.2f} ms        |")
    print("|                                                         |")
    print("|  Batch Throughput:                                      |")
    for bs, fps in benchmark.get("batch_fps", {}).items():
        if fps > 0:
            per_sample = 1000.0 / fps if fps > 0 else 0
            print(f"|    batch={bs:<3d}  ->  {fps:>8.1f} FPS  ({per_sample:.2f} ms/sample)   |")
    print("+---------------------------------------------------------+")

    # -- Section 2: Trajectory Accuracy --
    print("\n+---------------------------------------------------------+")
    print("|  [ACCURACY] TRAJECTORY ACCURACY (Offline)                      |")
    print("+---------------------------------------------------------+")
    if ade_fde is not None:
        ade = ade_fde["ade_m"]
        fde = ade_fde["fde_m"]
        samples = ade_fde["total_samples"]
        print(f"|  ADE (Avg Displacement Error):  {ade:>8.4f} m             |")
        print(f"|  FDE (Final Displacement Error): {fde:>8.4f} m             |")
        print(f"|  Evaluated on: {samples:>6d} validation samples             |")

        # Performance rating
        if ade < 1.0 and fde < 1.5:
            rating = "[EXCELLENT] EXCELLENT"
        elif ade < 2.0 and fde < 3.0:
            rating = "[GOOD] GOOD"
        elif ade < 3.0 and fde < 5.0:
            rating = "[WARN] FAIR"
        else:
            rating = "[ERROR] NEEDS IMPROVEMENT"
        print(f"|  Rating: {rating:<48s}|")
    else:
        print("|  [WARN] No H5 dataset available for offline evaluation.   |")
        print("|  Set H5_PATH env var to enable ADE/FDE metrics.       |")
    print("+---------------------------------------------------------+")

    # -- Section 3: Online Metrics (from CARLA) --
    print("\n+---------------------------------------------------------+")
    print("|  [ONLINE] ONLINE METRICS (from CARLA simulation)             |")
    print("+---------------------------------------------------------+")
    print("|  Route Completion:  Measured at runtime in CILAgent    |")
    print("|  Min TTC:           Measured at runtime in CILAgent    |")
    print("|                                                         |")
    print("|  These metrics are printed automatically when the CIL  |")
    print("|  agent finishes a run (see teardown() summary).        |")
    print("|                                                         |")
    print("|  To collect:                                            |")
    print("|    python run_agents.py --config configs/cil_light.yaml |")
    print("|    -> Look for '[STATS] CIL AGENT ONLINE EVALUATION SUMMARY' |")
    print("+---------------------------------------------------------+")

    # -- Summary Table --
    print("\n+--------------------------------------------------+")
    print("|           METRICS SUMMARY TABLE                  |")
    print("+-----------------------+--------------------------+")
    print("| Metric                | Value                    |")
    print("+-----------------------+--------------------------+")
    if ade_fde:
        print(f"| ADE                   | {ade_fde['ade_m']:>8.4f} m              |")
        print(f"| FDE                   | {ade_fde['fde_m']:>8.4f} m              |")
    else:
        print("| ADE                   | N/A (no dataset)         |")
        print("| FDE                   | N/A (no dataset)         |")
    print(f"| FPS (batch=1)         | {benchmark['fps']:>8.1f} FPS            |")
    print(f"| Latency (batch=1)     | {benchmark['latency_ms']:>8.2f} ms             |")
    print("| Route Completion      | See CARLA online run     |")
    print("| TTC (Time-To-Collision| See CARLA online run     |")
    print("+-----------------------+--------------------------+")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="CIL Waypoint Model Evaluation")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Only run FPS/Latency benchmark (no dataset needed)")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (default)")
    args = parser.parse_args()

    print("=" * 60)
    print(" CIL WAYPOINT MODEL - COMPREHENSIVE METRIC EVALUATION")
    print("=" * 60)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    model = load_model(device)

    # Always run benchmark
    benchmark = benchmark_inference(model, device)

    # Run ADE/FDE if dataset is available (and not --benchmark-only)
    ade_fde = None
    if not args.benchmark_only:
        h5_path = os.environ.get("H5_PATH",
                                  "/kaggle/input/datasets/yudtrann/dataset-carlav3/carla_images_drive.h5")
        try:
            import yaml
            config_path = PROJECT_ROOT / "configs" / "train_params.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            csv_root_hint = os.environ.get("CSV_ROOT", config.get("data_root", None))
        except Exception:
            csv_root_hint = None

        ade_fde = evaluate_ade_fde(model, device, h5_path, csv_root_hint)

    # Print results
    print_results(benchmark, ade_fde)


if __name__ == "__main__":
    main()
