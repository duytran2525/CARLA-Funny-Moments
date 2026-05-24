from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


class MultiAgentTrajectoryDataset(Dataset):
    """Load processed multi-agent .pt samples produced by build_multi_agent_dataset.py."""

    def __init__(
        self,
        root_dir: str | Path,
        manifest_path: str | Path | None = None,
        sample_files: Sequence[str | Path] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else self.root_dir / "manifest.csv"

        if sample_files is not None:
            # Normalize path separators immediately when sample_files is provided
            self.sample_paths = []
            for path in sample_files:
                # Convert to string, replace backslashes, then convert to Path
                clean_path = str(path).replace('\\', '/')
                p = Path(clean_path)
                self.sample_paths.append(p if p.is_absolute() else self.root_dir / p)
        else:
            self.sample_paths = self._read_manifest(self.manifest_path)

        # Final normalization: resolve and ensure forward slashes
        self.sample_paths = [Path(str(path).replace('\\', '/')) for path in self.sample_paths]
        if not self.sample_paths:
            raise RuntimeError(f"No multi-agent .pt samples found under: {self.root_dir}")

    @staticmethod
    def _read_manifest(path: Path) -> List[Path]:
        if not path.exists():
            raise FileNotFoundError(f"Multi-agent manifest not found: {path}")
        rows: List[Path] = []
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if "sample_file" not in (reader.fieldnames or []):
                raise RuntimeError(f"Manifest missing 'sample_file' column: {path}")
            for row in reader:
                sample_file = str(row.get("sample_file") or "").strip()
                if sample_file:
                    # Normalize path separators for cross-platform compatibility
                    sample_file = sample_file.replace('\\', '/')
                    rows.append(path.parent / sample_file)
        return rows

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int | float]:
        # Force string conversion and normalize path separators for cross-platform compatibility
        clean_path_str = str(self.sample_paths[int(index)]).replace('\\', '/')
        sample_path = Path(clean_path_str)
        sample = torch.load(sample_path, map_location="cpu", weights_only=True)
        
        # Load and detect format
        x = torch.as_tensor(sample["x"], dtype=torch.float32)
        x = self._ensure_6d_features(x)
        
        return {
            "x": x,
            "y": torch.as_tensor(sample["y"], dtype=torch.float32),
            "adj": torch.as_tensor(sample["adj"], dtype=torch.float32),
            "x_mask": torch.as_tensor(sample["x_mask"], dtype=torch.bool),
            "y_mask": torch.as_tensor(sample["y_mask"], dtype=torch.bool),
            "actor_ids": torch.as_tensor(sample["actor_ids"], dtype=torch.long),
            "ego_pose": torch.as_tensor(sample["ego_pose"], dtype=torch.float32),
            "anchor_frame": int(sample.get("anchor_frame", -1)),
            "anchor_timestamp": float(sample.get("anchor_timestamp", 0.0)),
            "town": str(sample.get("town", "")),
            "run_id": str(sample.get("run_id", "")),
            "sample_path": str(sample_path),
        }
    
    @staticmethod
    def _ensure_6d_features(x: torch.Tensor) -> torch.Tensor:
        """
        Ensure features are 6-dimensional for backward compatibility.
        
        Old format (fixed radius): [num_agents, history_steps, 4]
            Features: (local_x, local_y, heading_x, heading_y)
        
        New format (adaptive radius): [num_agents, history_steps, 6]
            Features: (local_x, local_y, local_vx, local_vy, heading_x, heading_y)
        
        If old format is detected (4D features), pad with zero velocity components
        to create 6D features: (local_x, local_y, 0.0, 0.0, heading_x, heading_y)
        
        Args:
            x: Input features tensor [num_agents, history_steps, feature_dim]
        
        Returns:
            Features tensor with 6D features [num_agents, history_steps, 6]
        """
        if x.shape[-1] == 6:
            # New format with velocity features - return as-is
            return x
        elif x.shape[-1] == 4:
            # Old format without velocity features - pad with zeros
            num_agents, history_steps, _ = x.shape
            # Create 6D tensor: (local_x, local_y, local_vx=0, local_vy=0, heading_x, heading_y)
            x_6d = torch.zeros((num_agents, history_steps, 6), dtype=x.dtype)
            x_6d[:, :, 0:2] = x[:, :, 0:2]  # local_x, local_y
            # x_6d[:, :, 2:4] already zeros (local_vx=0, local_vy=0)
            x_6d[:, :, 4:6] = x[:, :, 2:4]  # heading_x, heading_y
            return x_6d
        else:
            raise ValueError(
                f"Unexpected feature dimension: {x.shape[-1]}. "
                f"Expected 4 (old format) or 6 (new format)."
            )


def split_sample_paths(
    sample_paths: Sequence[Path],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[List[Path], List[Path]]:
    """Split sample paths into train / val sets.

    BUG FIX (data leakage): the previous implementation did a plain random
    shuffle over *all* windows, then split by index.  Because adjacent windows
    share up to (history_steps - 1 + future_steps - 1) frames when stride = 1,
    they are nearly identical sequences.  A random 80/20 split therefore places
    highly correlated windows on both sides of the boundary, causing ~95%
    feature overlap between train and val → inflated val metrics, unreliable
    early-stopping.

    Fix: group windows by their *run* (the parent directory, which corresponds
    to one continuous CARLA recording).  Shuffle runs (not individual windows),
    then assign whole runs to train or val.  Windows within a run are ordered
    by sample index so temporal ordering is preserved inside each run.

    If a path doesn't sit inside a recognisable run directory (flat layout),
    we fall back to a deterministic time-ordered split: the first train_ratio
    fraction goes to train, the rest to val — no shuffling.
    """
    import random
    from collections import defaultdict

    paths = [Path(p) for p in sample_paths]
    if not paths:
        return [], []

    # Group by parent directory (= run directory in the per-town subdir layout
    # produced by build_all_datasets.ps1: .../Town01/sample_000123.pt).
    run_groups: dict[Path, List[Path]] = defaultdict(list)
    for p in paths:
        run_groups[p.parent].append(p)

    # Sort windows within each run by filename to preserve temporal order.
    for run in run_groups:
        run_groups[run].sort(key=lambda p: p.name)

    runs = sorted(run_groups.keys())  # deterministic order before shuffle

    if len(runs) == 1:
        # Single run — time-ordered split (no shuffle) to avoid leakage.
        ordered = run_groups[runs[0]]
        if len(ordered) <= 1:
            return ordered, []
        split_idx = min(max(1, int(round(train_ratio * len(ordered)))), len(ordered) - 1)
        return ordered[:split_idx], ordered[split_idx:]

    # Multiple runs — shuffle at the run level.
    rng = random.Random(int(seed))
    rng.shuffle(runs)

    split_idx = min(max(1, int(round(train_ratio * len(runs)))), len(runs) - 1)
    train_runs, val_runs = runs[:split_idx], runs[split_idx:]

    train_paths = [p for r in train_runs for p in run_groups[r]]
    val_paths   = [p for r in val_runs   for p in run_groups[r]]
    return train_paths, val_paths


def _pad_tensor(target: torch.Tensor, source: torch.Tensor, slices: tuple[slice, ...]) -> None:
    target[slices] = source


def collate_multi_agent_trajectory(batch: Iterable[Dict[str, object]]) -> Dict[str, object]:
    samples = list(batch)
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(samples)
    max_agents = max(int(torch.as_tensor(sample["x"]).shape[0]) for sample in samples)
    history_len = int(torch.as_tensor(samples[0]["x"]).shape[1])
    input_dim = int(torch.as_tensor(samples[0]["x"]).shape[2])
    future_len = int(torch.as_tensor(samples[0]["y"]).shape[1])

    x = torch.zeros((batch_size, max_agents, history_len, input_dim), dtype=torch.float32)
    y = torch.zeros((batch_size, max_agents, future_len, 2), dtype=torch.float32)
    adj = torch.zeros((batch_size, max_agents, max_agents), dtype=torch.float32)
    x_mask = torch.zeros((batch_size, max_agents, history_len), dtype=torch.bool)
    y_mask = torch.zeros((batch_size, max_agents, future_len), dtype=torch.bool)
    agent_mask = torch.zeros((batch_size, max_agents), dtype=torch.bool)
    actor_ids = torch.full((batch_size, max_agents), -1, dtype=torch.long)
    ego_pose = torch.zeros((batch_size, 3), dtype=torch.float32)

    anchor_frames: List[int] = []
    anchor_timestamps: List[float] = []
    towns: List[str] = []
    run_ids: List[str] = []
    sample_paths: List[str] = []

    for batch_idx, sample in enumerate(samples):
        sample_x = torch.as_tensor(sample["x"], dtype=torch.float32)
        sample_y = torch.as_tensor(sample["y"], dtype=torch.float32)
        sample_adj = torch.as_tensor(sample["adj"], dtype=torch.float32)
        sample_x_mask = torch.as_tensor(sample["x_mask"], dtype=torch.bool)
        sample_y_mask = torch.as_tensor(sample["y_mask"], dtype=torch.bool)
        sample_actor_ids = torch.as_tensor(sample["actor_ids"], dtype=torch.long)
        n_agents = int(sample_x.shape[0])

        _pad_tensor(x, sample_x, (slice(batch_idx, batch_idx + 1), slice(0, n_agents), slice(None), slice(None)))
        _pad_tensor(y, sample_y, (slice(batch_idx, batch_idx + 1), slice(0, n_agents), slice(None), slice(None)))
        adj[batch_idx, :n_agents, :n_agents] = sample_adj
        x_mask[batch_idx, :n_agents] = sample_x_mask
        y_mask[batch_idx, :n_agents] = sample_y_mask
        agent_mask[batch_idx, :n_agents] = True
        actor_ids[batch_idx, :n_agents] = sample_actor_ids
        ego_pose[batch_idx] = torch.as_tensor(sample["ego_pose"], dtype=torch.float32)

        anchor_frames.append(int(sample.get("anchor_frame", -1)))
        anchor_timestamps.append(float(sample.get("anchor_timestamp", 0.0)))
        towns.append(str(sample.get("town", "")))
        run_ids.append(str(sample.get("run_id", "")))
        sample_paths.append(str(sample.get("sample_path", "")))

    return {
        "x": x,
        "y": y,
        "adj": adj,
        "x_mask": x_mask,
        "y_mask": y_mask,
        "agent_mask": agent_mask,
        "actor_ids": actor_ids,
        "ego_pose": ego_pose,
        "anchor_frames": anchor_frames,
        "anchor_timestamps": anchor_timestamps,
        "towns": towns,
        "run_ids": run_ids,
        "sample_paths": sample_paths,
    }
