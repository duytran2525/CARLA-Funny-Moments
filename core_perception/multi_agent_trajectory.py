from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


RAW_FIELDNAMES = [
    "run_id",
    "town",
    "frame",
    "timestamp",
    "ego_id",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_vx",
    "ego_vy",
    "ego_yaw",
    "actor_id",
    "actor_type",
    "actor_x",
    "actor_y",
    "actor_z",
    "actor_vx",
    "actor_vy",
    "actor_yaw",
    "distance_m",
]


@dataclass(frozen=True)
class EgoState:
    actor_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    yaw: float


@dataclass(frozen=True)
class ActorState:
    actor_id: int
    actor_type: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    yaw: float
    distance_m: float


@dataclass
class FrameData:
    frame: int
    timestamp: float
    run_id: str
    town: str
    ego: EgoState
    actors: Dict[int, ActorState]


@dataclass(frozen=True)
class WindowBuildConfig:
    history_frames: int = 20
    future_frames: int = 30
    stride: int = 2
    adjacency_radius_m: float = 15.0
    require_complete_tracks: bool = True
    min_agents: int = 1
    expected_dt: float = 0.1
    max_dt_error: float = 0.03


def write_raw_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()


def append_raw_rows(csv_path: Path, rows: Iterable[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    append = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RAW_FIELDNAMES})


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def read_raw_frames(csv_path: str | Path) -> List[FrameData]:
    path = Path(csv_path)
    frames: Dict[int, FrameData] = {}
    if not path.exists():
        raise FileNotFoundError(f"Raw multi-agent CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [key for key in RAW_FIELDNAMES if key not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"Raw CSV is missing required columns: {missing}")

        for row in reader:
            frame_id = _parse_int(row.get("frame"), default=-1)
            if frame_id < 0:
                continue
            ego = EgoState(
                actor_id=_parse_int(row.get("ego_id"), default=-1),
                x=_parse_float(row.get("ego_x")),
                y=_parse_float(row.get("ego_y")),
                z=_parse_float(row.get("ego_z")),
                vx=_parse_float(row.get("ego_vx")),
                vy=_parse_float(row.get("ego_vy")),
                yaw=_parse_float(row.get("ego_yaw")),
            )
            frame = frames.get(frame_id)
            if frame is None:
                frame = FrameData(
                    frame=frame_id,
                    timestamp=_parse_float(row.get("timestamp")),
                    run_id=str(row.get("run_id") or ""),
                    town=str(row.get("town") or ""),
                    ego=ego,
                    actors={},
                )
                frames[frame_id] = frame

            actor_id = _parse_int(row.get("actor_id"), default=-1)
            if actor_id < 0 or actor_id == ego.actor_id:
                continue
            frame.actors[actor_id] = ActorState(
                actor_id=actor_id,
                actor_type=str(row.get("actor_type") or "vehicle"),
                x=_parse_float(row.get("actor_x")),
                y=_parse_float(row.get("actor_y")),
                z=_parse_float(row.get("actor_z")),
                vx=_parse_float(row.get("actor_vx")),
                vy=_parse_float(row.get("actor_vy")),
                yaw=_parse_float(row.get("actor_yaw")),
                distance_m=_parse_float(row.get("distance_m")),
            )

    return [frames[key] for key in sorted(frames)]


def normalize_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def rotate_global_to_ego_forward_y(dx: float, dy: float, ego_yaw_deg: float) -> tuple[float, float]:
    """Return ego-frame coordinates with +Y forward and +X to ego right."""
    yaw_rad = math.radians(float(ego_yaw_deg))
    sin_yaw = math.sin(yaw_rad)
    cos_yaw = math.cos(yaw_rad)
    local_x = -sin_yaw * float(dx) + cos_yaw * float(dy)
    local_y = cos_yaw * float(dx) + sin_yaw * float(dy)
    return float(local_x), float(local_y)


def actor_feature_in_anchor_frame(actor: ActorState, anchor_ego: EgoState) -> tuple[float, float, float, float, float, float]:
    rel_x = float(actor.x) - float(anchor_ego.x)
    rel_y = float(actor.y) - float(anchor_ego.y)
    local_x, local_y = rotate_global_to_ego_forward_y(rel_x, rel_y, anchor_ego.yaw)

    rel_vx = float(actor.vx) - float(anchor_ego.vx)
    rel_vy = float(actor.vy) - float(anchor_ego.vy)
    local_vx, local_vy = rotate_global_to_ego_forward_y(rel_vx, rel_vy, anchor_ego.yaw)

    rel_yaw = normalize_angle_rad(math.radians(float(actor.yaw) - float(anchor_ego.yaw)))
    return (
        local_x,
        local_y,
        local_vx,
        local_vy,
        math.cos(rel_yaw),
        math.sin(rel_yaw),
    )


def actor_position_in_anchor_frame(actor: ActorState, anchor_ego: EgoState) -> tuple[float, float]:
    rel_x = float(actor.x) - float(anchor_ego.x)
    rel_y = float(actor.y) - float(anchor_ego.y)
    return rotate_global_to_ego_forward_y(rel_x, rel_y, anchor_ego.yaw)


def _regular_time_window(frames: Sequence[FrameData], expected_dt: float, max_dt_error: float) -> bool:
    if len(frames) <= 1:
        return True
    for prev, current in zip(frames, frames[1:]):
        dt = float(current.timestamp) - float(prev.timestamp)
        if abs(dt - float(expected_dt)) > float(max_dt_error):
            return False
    return True


def _select_actor_ids(
    history: Sequence[FrameData],
    future: Sequence[FrameData],
    anchor: FrameData,
    require_complete_tracks: bool,
) -> List[int]:
    actor_ids = sorted(anchor.actors)
    if not require_complete_tracks:
        return actor_ids

    required_frames = list(history) + list(future)
    complete_ids = []
    for actor_id in actor_ids:
        if all(actor_id in frame.actors for frame in required_frames):
            complete_ids.append(actor_id)
    return complete_ids


def _build_adjacency(anchor_positions: Any, adjacency_radius_m: float) -> Any:
    n_agents = int(anchor_positions.shape[0])
    adjacency = np.eye(n_agents, dtype=np.float32)
    radius = float(adjacency_radius_m)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dx = float(anchor_positions[i, 0] - anchor_positions[j, 0])
            dy = float(anchor_positions[i, 1] - anchor_positions[j, 1])
            if math.hypot(dx, dy) <= radius:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    return adjacency


def build_window_sample(
    history: Sequence[FrameData],
    future: Sequence[FrameData],
    anchor: FrameData,
    actor_ids: Sequence[int],
    config: WindowBuildConfig,
) -> dict:
    if np is None:
        raise RuntimeError("numpy is required to build multi-agent trajectory samples.")

    n_agents = len(actor_ids)
    history_len = len(history)
    future_len = len(future)
    x = np.zeros((n_agents, history_len, 6), dtype=np.float32)
    y = np.zeros((n_agents, future_len, 2), dtype=np.float32)
    x_mask = np.zeros((n_agents, history_len), dtype=np.bool_)
    y_mask = np.zeros((n_agents, future_len), dtype=np.bool_)
    anchor_positions = np.zeros((n_agents, 2), dtype=np.float32)

    for agent_idx, actor_id in enumerate(actor_ids):
        anchor_actor = anchor.actors.get(actor_id)
        if anchor_actor is not None:
            anchor_positions[agent_idx] = np.asarray(
                actor_position_in_anchor_frame(anchor_actor, anchor.ego),
                dtype=np.float32,
            )

        for hist_idx, frame in enumerate(history):
            actor = frame.actors.get(actor_id)
            if actor is None:
                continue
            x[agent_idx, hist_idx] = np.asarray(
                actor_feature_in_anchor_frame(actor, anchor.ego),
                dtype=np.float32,
            )
            x_mask[agent_idx, hist_idx] = True

        for fut_idx, frame in enumerate(future):
            actor = frame.actors.get(actor_id)
            if actor is None:
                continue
            y[agent_idx, fut_idx] = np.asarray(
                actor_position_in_anchor_frame(actor, anchor.ego),
                dtype=np.float32,
            )
            y_mask[agent_idx, fut_idx] = True

    return {
        "x": x,
        "y": y,
        "adj": _build_adjacency(anchor_positions, config.adjacency_radius_m),
        "x_mask": x_mask,
        "y_mask": y_mask,
        "actor_ids": np.asarray(list(actor_ids), dtype=np.int64),
        "anchor_frame": int(anchor.frame),
        "anchor_timestamp": float(anchor.timestamp),
        "town": str(anchor.town),
        "run_id": str(anchor.run_id),
        "ego_pose": np.asarray([anchor.ego.x, anchor.ego.y, anchor.ego.yaw], dtype=np.float32),
    }


def build_multi_agent_samples(
    frames: Sequence[FrameData],
    config: WindowBuildConfig,
) -> List[dict]:
    if np is None:
        raise RuntimeError("numpy is required to build multi-agent trajectory samples.")
    if config.history_frames <= 0 or config.future_frames <= 0:
        raise ValueError("history_frames and future_frames must be positive.")
    if config.stride <= 0:
        raise ValueError("stride must be positive.")

    sorted_frames = sorted(frames, key=lambda item: item.frame)
    samples: List[dict] = []
    first_anchor_idx = int(config.history_frames) - 1
    last_anchor_idx = len(sorted_frames) - int(config.future_frames) - 1
    if last_anchor_idx < first_anchor_idx:
        return samples

    for anchor_idx in range(first_anchor_idx, last_anchor_idx + 1, int(config.stride)):
        history = sorted_frames[anchor_idx - int(config.history_frames) + 1 : anchor_idx + 1]
        future = sorted_frames[anchor_idx + 1 : anchor_idx + int(config.future_frames) + 1]
        window = list(history) + list(future)
        if not _regular_time_window(window, config.expected_dt, config.max_dt_error):
            continue

        anchor = sorted_frames[anchor_idx]
        actor_ids = _select_actor_ids(
            history=history,
            future=future,
            anchor=anchor,
            require_complete_tracks=bool(config.require_complete_tracks),
        )
        if len(actor_ids) < int(config.min_agents):
            continue

        samples.append(
            build_window_sample(
                history=history,
                future=future,
                anchor=anchor,
                actor_ids=actor_ids,
                config=config,
            )
        )
    return samples


def sample_to_torch_payload(sample: dict, raw_source: str | Path = "") -> dict:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required to write .pt multi-agent samples.") from exc

    return {
        "x": torch.as_tensor(sample["x"], dtype=torch.float32),
        "y": torch.as_tensor(sample["y"], dtype=torch.float32),
        "adj": torch.as_tensor(sample["adj"], dtype=torch.float32),
        "x_mask": torch.as_tensor(sample["x_mask"], dtype=torch.bool),
        "y_mask": torch.as_tensor(sample["y_mask"], dtype=torch.bool),
        "actor_ids": torch.as_tensor(sample["actor_ids"], dtype=torch.long),
        "anchor_frame": int(sample["anchor_frame"]),
        "anchor_timestamp": float(sample["anchor_timestamp"]),
        "town": str(sample.get("town", "")),
        "run_id": str(sample.get("run_id", "")),
        "ego_pose": torch.as_tensor(sample["ego_pose"], dtype=torch.float32),
        "raw_source": str(raw_source),
    }

