#!/usr/bin/env python3
"""
migrate_dataset.py — Dataset Migration Script
==============================================

Converts old datasets (fixed radius) to new format (adaptive radius).
Recomputes adjacency matrix with adaptive radius while preserving all metadata.

Usage:
    python scripts/migrate_dataset.py --input-dir data/multi_agent/old --output-dir data/multi_agent/new
    python scripts/migrate_dataset.py --input-dir data/multi_agent/old --output-dir data/multi_agent/new --radius-base 25.0 --radius-alpha 0.6

Requirements: 10.9, 10.10
"""

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

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_trajectory import build_adaptive_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate old datasets (fixed radius) to new format (adaptive radius). "
            "Recomputes adjacency matrix with adaptive radius while preserving all metadata."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing old dataset with manifest.csv and .pt samples.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where migrated samples will be written.",
    )
    parser.add_argument(
        "--radius-base",
        type=float,
        default=20.0,
        help="Base radius in meters for adaptive radius mode (default: 20.0).",
    )
    parser.add_argument(
        "--radius-alpha",
        type=float,
        default=0.5,
        help="Velocity scaling factor for adaptive radius mode (default: 0.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of samples to migrate, useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print migration plan without writing files.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load manifest.csv and return list of sample metadata."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(dict(row))
    
    return rows


def write_manifest(rows: List[Dict[str, Any]], manifest_path: Path) -> None:
    """Write manifest.csv with sample metadata."""
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


def extract_global_positions_and_velocities(sample: dict) -> tuple[Any, Any]:
    """Extract global positions and velocities from sample for adjacency computation.
    
    Args:
        sample: Dictionary containing sample data with keys:
            - 'x': History features [N, T, 6] with (local_x, local_y, local_vx, local_vy, heading_x, heading_y)
            - 'x_mask': Valid history frames [N, T]
            - 'ego_pose': Ego vehicle pose at anchor frame [3] with (x, y, yaw)
    
    Returns:
        Tuple of (global_positions, velocities):
            - global_positions: [N, 2] in global (world) coordinates as float64
            - velocities: [N, 2] in m/s as float32
    """
    if np is None:
        raise RuntimeError("numpy is required for extract_global_positions_and_velocities.")
    
    # Extract data from sample
    x = sample["x"]  # [N, T, 6]
    x_mask = sample["x_mask"]  # [N, T]
    ego_pose = sample["ego_pose"]  # [3]: (ego_x, ego_y, ego_yaw)
    
    n_agents = x.shape[0]
    
    # Extract ego pose at anchor frame
    ego_x = float(ego_pose[0])
    ego_y = float(ego_pose[1])
    ego_yaw_deg = float(ego_pose[2])
    
    # Initialize arrays
    global_positions = np.zeros((n_agents, 2), dtype=np.float64)
    velocities = np.zeros((n_agents, 2), dtype=np.float32)
    
    # For each agent, extract position and velocity from the last valid history frame
    for agent_idx in range(n_agents):
        # Find last valid history frame for this agent
        valid_frames = np.where(x_mask[agent_idx])[0]
        
        if len(valid_frames) == 0:
            # No valid frames for this agent - use zero position and velocity
            continue
        
        # Use last valid frame
        last_frame_idx = valid_frames[-1]
        
        # Extract local coordinates and velocities from history features
        # x[agent_idx, last_frame_idx] = [local_x, local_y, local_vx, local_vy, heading_x, heading_y]
        local_x = float(x[agent_idx, last_frame_idx, 0])
        local_y = float(x[agent_idx, last_frame_idx, 1])
        local_vx = float(x[agent_idx, last_frame_idx, 2])
        local_vy = float(x[agent_idx, last_frame_idx, 3])
        
        # Transform local position back to global coordinates
        # Local frame: +Y forward (ego heading), +X right
        # Inverse transformation: rotate by -ego_yaw and translate by ego position
        yaw_rad = np.radians(ego_yaw_deg)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        # Inverse rotation: local to global
        # local_x = -sin_yaw * dx + cos_yaw * dy
        # local_y = cos_yaw * dx + sin_yaw * dy
        # Solving for dx, dy:
        # dx = cos_yaw * local_y - sin_yaw * local_x
        # dy = sin_yaw * local_y + cos_yaw * local_x
        dx = cos_yaw * local_y - sin_yaw * local_x
        dy = sin_yaw * local_y + cos_yaw * local_x
        
        global_x = ego_x + dx
        global_y = ego_y + dy
        
        global_positions[agent_idx] = [global_x, global_y]
        
        # Transform local velocity back to global coordinates
        # Velocities are relative to ego, so we need to add ego velocity and rotate
        # First, rotate local velocity to global frame
        dvx = cos_yaw * local_vy - sin_yaw * local_vx
        dvy = sin_yaw * local_vy + cos_yaw * local_vx
        
        # Add ego velocity to get absolute velocity
        # Note: ego velocity is already in global frame
        # But we stored relative velocity, so we need to add ego velocity
        # Actually, the stored velocity is already relative to ego, so we just rotate it
        velocities[agent_idx] = [dvx, dvy]
    
    return global_positions, velocities


def migrate_sample(
    sample_path: Path,
    output_path: Path,
    radius_base: float,
    radius_alpha: float,
) -> Dict[str, Any]:
    """Migrate a single sample from fixed radius to adaptive radius.
    
    Args:
        sample_path: Path to input .pt sample file
        output_path: Path to output .pt sample file
        radius_base: Base radius in meters for adaptive radius
        radius_alpha: Velocity scaling factor for adaptive radius
    
    Returns:
        Dictionary with migration metadata:
            - 'sample_file': Output filename
            - 'anchor_frame': Anchor frame index
            - 'anchor_timestamp': Anchor timestamp
            - 'town': CARLA town name
            - 'run_id': Data collection run identifier
            - 'num_agents': Number of agents in sample
    """
    if torch is None:
        raise RuntimeError("torch is required to migrate samples.")
    if np is None:
        raise RuntimeError("numpy is required to migrate samples.")
    
    # Load old sample
    sample = torch.load(sample_path, weights_only=False)
    
    # Convert tensors to numpy for processing
    x = sample["x"].numpy()  # [N, T, 6]
    y = sample["y"].numpy()  # [N, T, 2]
    x_mask = sample["x_mask"].numpy()  # [N, T]
    y_mask = sample["y_mask"].numpy()  # [N, T]
    actor_ids = sample["actor_ids"].numpy()  # [N]
    ego_pose = sample["ego_pose"].numpy()  # [3]
    
    # Extract global positions and velocities for adjacency computation
    global_positions, velocities = extract_global_positions_and_velocities(sample)
    
    # Recompute adjacency matrix with adaptive radius
    adjacency = build_adaptive_adjacency(
        global_positions,
        velocities,
        r_base=radius_base,
        alpha=radius_alpha,
    )
    
    # Create new sample with updated adjacency matrix
    # Preserve all other data and metadata
    migrated_sample = {
        "x": torch.as_tensor(x, dtype=torch.float32),
        "y": torch.as_tensor(y, dtype=torch.float32),
        "adj": torch.as_tensor(adjacency, dtype=torch.float32),
        "x_mask": torch.as_tensor(x_mask, dtype=torch.bool),
        "y_mask": torch.as_tensor(y_mask, dtype=torch.bool),
        "actor_ids": torch.as_tensor(actor_ids, dtype=torch.int64),
        "ego_pose": torch.as_tensor(ego_pose, dtype=torch.float32),
        "anchor_frame": sample["anchor_frame"],
        "anchor_timestamp": sample["anchor_timestamp"],
        "town": sample["town"],
        "run_id": sample["run_id"],
    }
    
    # Add raw_source if present in original sample
    if "raw_source" in sample:
        migrated_sample["raw_source"] = sample["raw_source"]
    
    # Write migrated sample
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated_sample, output_path)
    
    # Return metadata for manifest
    return {
        "sample_file": output_path.name,
        "anchor_frame": int(sample["anchor_frame"]),
        "anchor_timestamp": f"{float(sample['anchor_timestamp']):.6f}",
        "town": str(sample["town"]),
        "run_id": str(sample["run_id"]),
        "num_agents": int(x.shape[0]),
    }


def main() -> int:
    if torch is None:
        raise RuntimeError("torch is required to migrate samples. Install requirements.txt first.")
    if np is None:
        raise RuntimeError("numpy is required to migrate samples. Install requirements.txt first.")
    
    args = parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1
    
    manifest_path = input_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 1
    
    # Load manifest
    print(f"[INFO] Loading manifest from {manifest_path}")
    manifest_rows = load_manifest(manifest_path)
    
    if int(args.limit) > 0:
        manifest_rows = manifest_rows[: int(args.limit)]
    
    print(f"[INFO] Found {len(manifest_rows)} samples to migrate")
    
    if args.dry_run:
        print("[INFO] Dry run mode - no files will be written")
        print(f"[INFO] Would migrate {len(manifest_rows)} samples from {input_dir} to {output_dir}")
        print(f"[INFO] Adaptive radius parameters: r_base={args.radius_base}, alpha={args.radius_alpha}")
        return 0
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Migrate samples
    migrated_rows: List[Dict[str, Any]] = []
    failed_samples: List[str] = []
    
    for idx, row in enumerate(manifest_rows):
        sample_file = row["sample_file"]
        sample_path = input_dir / sample_file
        
        if not sample_path.exists():
            print(f"[WARNING] Sample not found: {sample_path}")
            failed_samples.append(sample_file)
            continue
        
        # Use same filename in output directory
        output_path = output_dir / sample_file
        
        try:
            migrated_row = migrate_sample(
                sample_path,
                output_path,
                radius_base=args.radius_base,
                radius_alpha=args.radius_alpha,
            )
            migrated_rows.append(migrated_row)
            
            if (idx + 1) % 100 == 0:
                print(f"[INFO] Migrated {idx + 1}/{len(manifest_rows)} samples")
        
        except Exception as exc:
            print(f"[ERROR] Failed to migrate {sample_file}: {exc}")
            failed_samples.append(sample_file)
            continue
    
    # Write new manifest
    output_manifest_path = output_dir / "manifest.csv"
    write_manifest(migrated_rows, output_manifest_path)
    
    # Write migration summary
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_samples": len(manifest_rows),
        "migrated_samples": len(migrated_rows),
        "failed_samples": len(failed_samples),
        "failed_sample_files": failed_samples,
        "adaptive_radius_config": {
            "radius_base": args.radius_base,
            "radius_alpha": args.radius_alpha,
        },
    }
    
    summary_path = output_dir / "migration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    
    # Print summary
    print(f"\n[OK] Migration complete!")
    print(f"[OK] Migrated samples: {len(migrated_rows)}")
    print(f"[OK] Failed samples: {len(failed_samples)}")
    print(f"[OK] Output directory: {output_dir}")
    print(f"[OK] Manifest: {output_manifest_path}")
    print(f"[OK] Summary: {summary_path}")
    
    if failed_samples:
        print(f"\n[WARNING] {len(failed_samples)} samples failed to migrate:")
        for sample_file in failed_samples[:10]:  # Show first 10
            print(f"  - {sample_file}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more (see migration_summary.json)")
    
    return 0 if len(failed_samples) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
