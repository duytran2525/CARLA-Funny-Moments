from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_trajectory import (
    WindowBuildConfig,
    build_multi_agent_samples,
    read_raw_frames,
    sample_to_torch_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ego-centric multi-agent trajectory .pt samples from raw CARLA CSV logs. "
            "At 10 FPS, the default 20 history + 30 future frames creates 5-second samples."
        )
    )
    parser.add_argument("--raw-csv", required=True, help="Raw CSV from scripts/collect_multi_agent_raw.py.")
    parser.add_argument(
        "--out-dir",
        default="data/multi_agent/processed",
        help="Directory where sample_XXXXXX.pt and manifest.csv will be written.",
    )
    parser.add_argument("--history-frames", type=int, default=40)
    parser.add_argument("--future-frames", type=int, default=60)
    # BUG FIX #4: stride default changed from 2 → 1 to match ps1 and avoid
    # inconsistent sample counts when running directly vs via build_all_datasets.ps1.
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride in frames.")
    parser.add_argument("--dt", type=float, default=0.1, help="Expected sampling dt in seconds.")
    parser.add_argument("--max-dt-error", type=float, default=0.03)
    # BUG FIX #3: default changed from 40.0 → 100.0 to match CARLA collector's
    # VISIBILITY_RADIUS_METERS=100.0. The old 40.0 default silently discarded ~60%
    # of NPC data whenever the script was invoked directly (not via ps1).
    parser.add_argument("--adjacency-radius-m", type=float, default=100.0)
    # BUG FIX #5: default changed from 1 → 2 to match ps1 (--min-agents 2).
    # Allowing single-agent frames produces degenerate "interaction" samples with
    # no social context; ps1 was already filtering them out.
    parser.add_argument("--min-agents", type=int, default=2)
    parser.add_argument(
        "--max-step-m",
        type=float,
        default=6.0,
        help="Max per-agent displacement (metres) between consecutive frames; agents that exceed this are filtered out.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.5,
        help="Minimum fraction of valid history frames an agent must retain after teleportation filtering.",
    )
    # BUG FIX #6: --allow-missing now uses BooleanOptionalAction with default=True
    # so that running the script directly matches ps1 behaviour (which always passes
    # --allow-missing). The old default=False silently dropped any anchor that
    # disappeared for even one frame, discarding large fractions of CARLA data.
    # Use --no-allow-missing to restore the strict behaviour.
    parser.add_argument(
        "--allow-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep anchor actors even if missing in some history/future frames; "
            "masks mark valid steps (default: True). Use --no-allow-missing to "
            "require fully-complete tracks."
        ),
    )
    parser.add_argument(
        "--adaptive-radius",
        action="store_true",
        help="Enable adaptive radius mode: compute per-agent radius from velocity magnitude.",
    )
    parser.add_argument(
        "--radius-base",
        type=float,
        default=40.0,
        help="Base radius in meters for adaptive radius mode (default: 40.0).",
    )
    parser.add_argument(
        "--radius-alpha",
        type=float,
        default=1.0,
        help="Velocity scaling factor for adaptive radius mode (default: 1.0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of samples to write, useful for smoke tests.",
    )
    return parser.parse_args()


def _write_manifest(rows: List[Dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_file",
        "anchor_frame",
        "anchor_timestamp",
        "town",
        "run_id",
        "num_agents",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    if torch is None:
        raise RuntimeError("torch is required to write .pt samples. Install requirements.txt first.")

    args = parse_args()
    raw_csv = Path(args.raw_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = WindowBuildConfig(
        history_frames=max(1, args.history_frames),
        future_frames=max(1, args.future_frames),
        stride=max(1, args.stride),
        adjacency_radius_m=args.adjacency_radius_m,
        require_complete_tracks=not args.allow_missing,
        min_agents=max(1, args.min_agents),
        expected_dt=args.dt,
        max_dt_error=args.max_dt_error,
        max_step_m=args.max_step_m,
        min_valid_ratio=args.min_valid_ratio,
        adaptive_radius_enabled=bool(args.adaptive_radius),
        radius_base=args.radius_base,
        radius_alpha=args.radius_alpha,
    )

    frames = read_raw_frames(raw_csv)
    samples = build_multi_agent_samples(frames, config)
    if args.limit > 0:
        samples = samples[: args.limit]

    manifest_rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_name = f"sample_{index:06d}.pt"
        sample_path = out_dir / sample_name
        torch.save(sample_to_torch_payload(sample, raw_source=raw_csv), sample_path)
        manifest_rows.append(
            {
                "sample_file": sample_name,
                "anchor_frame": int(sample["anchor_frame"]),
                "anchor_timestamp": f"{float(sample['anchor_timestamp']):.6f}",
                "town": str(sample.get("town", "")),
                "run_id": str(sample.get("run_id", "")),
                "num_agents": int(sample["x"].shape[0]),
            }
        )

    _write_manifest(manifest_rows, out_dir / "manifest.csv")
    summary = {
        "raw_csv": str(raw_csv),
        "out_dir": str(out_dir),
        "frames_read": len(frames),
        "samples_written": len(manifest_rows),
        "config": {
            "history_frames": config.history_frames,
            "future_frames": config.future_frames,
            "stride": config.stride,
            "adjacency_radius_m": config.adjacency_radius_m,
            "require_complete_tracks": config.require_complete_tracks,
            "min_agents": config.min_agents,
            "expected_dt": config.expected_dt,
            "max_dt_error": config.max_dt_error,
            "max_step_m": config.max_step_m,
            "min_valid_ratio": config.min_valid_ratio,
            "adaptive_radius_enabled": config.adaptive_radius_enabled,
            "radius_base": config.radius_base,
            "radius_alpha": config.radius_alpha,
        },
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Read frames: {len(frames)}")
    print(f"[OK] Wrote samples: {len(manifest_rows)}")
    print(f"[OK] Manifest: {out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
