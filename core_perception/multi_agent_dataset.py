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
            self.sample_paths = [
                path if Path(path).is_absolute() else self.root_dir / Path(path)
                for path in sample_files
            ]
        else:
            self.sample_paths = self._read_manifest(self.manifest_path)

        self.sample_paths = [Path(path).resolve() for path in self.sample_paths]
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
                    rows.append(path.parent / sample_file)
        return rows

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int | float]:
        sample_path = self.sample_paths[int(index)]
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        return {
            "x": torch.as_tensor(sample["x"], dtype=torch.float32),
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


def split_sample_paths(
    sample_paths: Sequence[Path],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[List[Path], List[Path]]:
    import random

    paths = [Path(path) for path in sample_paths]
    rng = random.Random(int(seed))
    rng.shuffle(paths)
    if len(paths) <= 1:
        return paths, []
    split_idx = int(round(float(train_ratio) * len(paths)))
    split_idx = min(max(1, split_idx), len(paths) - 1)
    return paths[:split_idx], paths[split_idx:]


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

