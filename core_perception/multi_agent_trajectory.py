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
    history_frames: int = 40
    future_frames: int = 60
    stride: int = 2
    adjacency_radius_m: float = 40.0
    require_complete_tracks: bool = True
    min_agents: int = 1
    expected_dt: float = 0.1
    max_dt_error: float = 0.03
    max_step_m: float = 6.0
    min_valid_ratio: float = 0.5
    adaptive_radius_enabled: bool = False
    radius_base: float = 40.0
    radius_alpha: float = 1.0


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

    # Rotate the actor's global heading unit vector through the SAME transform
    # used for positions and velocities, ensuring geometric consistency.
    actor_yaw_rad = math.radians(float(actor.yaw))
    heading_x, heading_y = rotate_global_to_ego_forward_y(
        math.cos(actor_yaw_rad), math.sin(actor_yaw_rad), anchor_ego.yaw
    )
    return (
        local_x,
        local_y,
        local_vx,
        local_vy,
        heading_x,
        heading_y,
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


def compute_adaptive_radius(
    velocities: Any,
    r_base: float = 40.0,
    alpha: float = 1.0,
) -> Any:
    """Compute per-agent interaction radius based on velocity magnitude.
    
    Formula: r(i) = r_base + alpha * ||v_i||
    
    Args:
        velocities: Agent velocities in m/s with shape (num_agents, 2) containing (vx, vy)
        r_base: Base radius in meters (default 40.0)
        alpha: Velocity scaling factor (default 1.0)
    
    Returns:
        Per-agent radius in meters as numpy array with shape (num_agents,)
    
    Requirements: 3.1, 3.2, 3.10
    """
    if np is None:
        raise RuntimeError("numpy is required for compute_adaptive_radius.")
    
    # Extract velocity magnitude from input velocities array
    # velocities shape: (num_agents, 2) where columns are (vx, vy)
    speed = np.linalg.norm(velocities, axis=-1)  # shape: (num_agents,)
    
    # Apply formula: r(i) = r_base + alpha * ||v_i||
    radius = float(r_base) + float(alpha) * speed
    
    return radius


def build_adaptive_adjacency(
    positions: Any,
    velocities: Any,
    r_base: float = 40.0,
    alpha: float = 1.0,
) -> Any:
    """Build adjacency matrix with adaptive radius based on agent velocities.
    
    Algorithm:
    1. Compute per-agent radius: r[i] = r_base + alpha * ||v[i]||
    2. Compute pairwise distances: d[i,j] = ||pos[i] - pos[j]||
    3. Connect if within radius: adj[i,j] = 1 if d[i,j] <= min(r[i], r[j]) else 0
    4. Ensure symmetry: adj[i,j] = adj[j,i]
    
    Args:
        positions: Agent positions in global (world) coordinates with shape (num_agents, 2)
                  stored as float64 for precise distance computation
        velocities: Agent velocities in m/s with shape (num_agents, 2) containing (vx, vy)
        r_base: Base radius in meters (default 40.0)
        alpha: Velocity scaling factor (default 1.0)
    
    Returns:
        Symmetric adjacency matrix with shape (num_agents, num_agents) as float32
    
    Requirements: 3.5, 3.6, 3.7
    """
    if np is None:
        raise RuntimeError("numpy is required for build_adaptive_adjacency.")
    
    n_agents = int(positions.shape[0])
    
    # Initialize adjacency matrix with self-connections (diagonal = 1)
    adjacency = np.eye(n_agents, dtype=np.float32)
    
    # Compute per-agent radius using adaptive radius function
    radii = compute_adaptive_radius(velocities, r_base=r_base, alpha=alpha)
    
    # Compute pairwise distances and build adjacency matrix
    # Use float64 precision for distance computation to avoid numerical errors
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            # Compute pairwise distance in global coordinates using float64 precision
            dx = float(positions[i, 0]) - float(positions[j, 0])
            dy = float(positions[i, 1]) - float(positions[j, 1])
            distance = math.hypot(dx, dy)
            
            # Connect agents i and j if distance(i,j) <= min(r(i), r(j))
            # Using min ensures symmetric adjacency matrix
            threshold = min(float(radii[i]), float(radii[j]))
            
            if distance <= threshold:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    
    return adjacency


def _build_adjacency(positions: Any, adjacency_radius_m: float) -> Any:
    """Build a symmetric adjacency matrix from pairwise 2-D distances.

    *positions* should be **global (world) coordinates** stored as float64 so
    that distance comparisons near the radius boundary are not affected by
    float32 quantisation of the ego-centric transform.
    """
    n_agents = int(positions.shape[0])
    adjacency = np.eye(n_agents, dtype=np.float32)
    radius = float(adjacency_radius_m)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dx = float(positions[i, 0]) - float(positions[j, 0])
            dy = float(positions[i, 1]) - float(positions[j, 1])
            if math.hypot(dx, dy) <= radius:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    return adjacency


def _filter_teleportation(
    x: Any,
    x_mask: Any,
    y: Any,
    y_mask: Any,
    max_step_m: float,
) -> None:
    """Invalidate agent frames where consecutive-frame displacement exceeds *max_step_m*.

    CARLA may recycle actor IDs: a destroyed NPC's ID can be reused by a newly
    spawned vehicle.  When this happens inside a sliding window the pipeline
    silently joins two unrelated trajectories, producing 40-50 m jumps in a
    single 0.1 s timestep.  This function detects those jumps and zeros out
    everything from the first violation onward so downstream code never sees
    physically impossible motion.
    """
    n_agents = int(x.shape[0])
    history_len = int(x.shape[1])
    future_len = int(y.shape[1])
    threshold = float(max_step_m)

    for agent_idx in range(n_agents):
        # --- history X ---
        for t in range(1, history_len):
            if not (x_mask[agent_idx, t] and x_mask[agent_idx, t - 1]):
                continue
            dx = float(x[agent_idx, t, 0] - x[agent_idx, t - 1, 0])
            dy = float(x[agent_idx, t, 1] - x[agent_idx, t - 1, 1])
            if math.hypot(dx, dy) > threshold:
                x_mask[agent_idx, t:] = False
                x[agent_idx, t:] = 0.0
                break

        # --- future Y ---
        # Check continuity between last valid history position and first future position,
        # then between consecutive future positions.
        last_hist_pos = None
        for t in range(history_len - 1, -1, -1):
            if x_mask[agent_idx, t]:
                last_hist_pos = (float(x[agent_idx, t, 0]), float(x[agent_idx, t, 1]))
                break

        for t in range(future_len):
            if not y_mask[agent_idx, t]:
                continue
            cur_pos = (float(y[agent_idx, t, 0]), float(y[agent_idx, t, 1]))
            if last_hist_pos is not None:
                dx = cur_pos[0] - last_hist_pos[0]
                dy = cur_pos[1] - last_hist_pos[1]
                if math.hypot(dx, dy) > threshold:
                    y_mask[agent_idx, t:] = False
                    y[agent_idx, t:] = 0.0
                    break
            last_hist_pos = cur_pos


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
    # Global (world) positions kept in float64 for precise adjacency distance
    # computation.  The ego-centric anchor_positions above are only used if
    # needed downstream; adjacency is computed from global_positions.
    global_positions = np.zeros((n_agents, 2), dtype=np.float64)
    # Store velocities for adaptive radius computation
    velocities = np.zeros((n_agents, 2), dtype=np.float32)

    for agent_idx, actor_id in enumerate(actor_ids):
        anchor_actor = anchor.actors.get(actor_id)
        if anchor_actor is not None:
            anchor_positions[agent_idx] = np.asarray(
                actor_position_in_anchor_frame(anchor_actor, anchor.ego),
                dtype=np.float32,
            )
            global_positions[agent_idx] = np.asarray(
                [anchor_actor.x, anchor_actor.y], dtype=np.float64,
            )
            # Extract velocity from anchor frame for adaptive radius
            velocities[agent_idx] = np.asarray(
                [anchor_actor.vx, anchor_actor.vy], dtype=np.float32,
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

    # --- Teleportation filter: reject physically impossible jumps ----------
    _filter_teleportation(x, x_mask, y, y_mask, max_step_m=float(config.max_step_m))

    # --- Drop agents that lost too many frames after filtering -------------
    min_valid_history = max(1, int(history_len * float(config.min_valid_ratio)))
    keep_mask = np.array(
        [int(x_mask[i].sum()) >= min_valid_history for i in range(n_agents)],
        dtype=np.bool_,
    )
    if int(keep_mask.sum()) < n_agents:
        keep_indices = np.where(keep_mask)[0]
        if len(keep_indices) == 0:
            # All agents filtered out – return empty sample that caller can skip
            return {
                "x": np.zeros((0, history_len, 6), dtype=np.float32),
                "y": np.zeros((0, future_len, 2), dtype=np.float32),
                "adj": np.zeros((0, 0), dtype=np.float32),
                "x_mask": np.zeros((0, history_len), dtype=np.bool_),
                "y_mask": np.zeros((0, future_len), dtype=np.bool_),
                "actor_ids": np.array([], dtype=np.int64),
                "anchor_frame": int(anchor.frame),
                "anchor_timestamp": float(anchor.timestamp),
                "town": str(anchor.town),
                "run_id": str(anchor.run_id),
                "ego_pose": np.asarray([anchor.ego.x, anchor.ego.y, anchor.ego.yaw], dtype=np.float32),
            }
        x = x[keep_indices]
        y = y[keep_indices]
        x_mask = x_mask[keep_indices]
        y_mask = y_mask[keep_indices]
        anchor_positions = anchor_positions[keep_indices]
        global_positions = global_positions[keep_indices]
        velocities = velocities[keep_indices]
        actor_ids = [actor_ids[i] for i in keep_indices]
        n_agents = len(actor_ids)

    # --- Build adjacency matrix: fixed or adaptive radius ------------------
    if config.adaptive_radius_enabled:
        # Use adaptive radius based on velocity magnitude
        adjacency = build_adaptive_adjacency(
            global_positions,
            velocities,
            r_base=float(config.radius_base),
            alpha=float(config.radius_alpha),
        )
    else:
        # Use fixed radius (legacy mode)
        adjacency = _build_adjacency(global_positions, config.adjacency_radius_m)

    return {
        "x": x,
        "y": y,
        "adj": adjacency,
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

        sample = build_window_sample(
            history=history,
            future=future,
            anchor=anchor,
            actor_ids=actor_ids,
            config=config,
        )
        # Skip samples where teleportation filtering removed all agents.
        if int(sample["x"].shape[0]) < int(config.min_agents):
            continue
        samples.append(sample)
    return samples


def compute_multimodal_metrics(
    pred: Any,
    target: Any,
    y_mask: Any,
    agent_mask: Any,
) -> dict:
    """Compute minADE, minFDE, MissRate for multimodal predictions.
    
    This function evaluates multimodal trajectory predictions by selecting the best
    mode per agent and computing standard trajectory prediction metrics.
    
    Args:
        pred: Predicted trajectories with shape [B, N, K, T, 2] for multimodal (K>1)
              or [B, N, T, 2] for unimodal (K=1), where:
              - B: batch size
              - N: max number of agents
              - K: number of trajectory modes
              - T: number of future timesteps
              - 2: (x, y) coordinates
        target: Ground truth trajectories with shape [B, N, T, 2]
        y_mask: Valid timestep mask with shape [B, N, T] (True if observed, False if missing)
        agent_mask: Valid agent mask with shape [B, N] (True if valid agent, False if padding)
    
    Returns:
        Dictionary containing:
        - 'minADE': Minimum Average Displacement Error across K modes (float)
        - 'minFDE': Minimum Final Displacement Error across K modes (float)
        - 'MissRate': Fraction of predictions where minFDE > 2.0 meters (float)
        - 'mode_0_ADE', 'mode_1_ADE', ...: Per-mode ADE for analysis (float)
        - 'mode_0_FDE', 'mode_1_FDE', ...: Per-mode FDE for analysis (float)
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for compute_multimodal_metrics.") from exc
    
    # Convert inputs to torch tensors if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.as_tensor(target, dtype=torch.float32)
    if not isinstance(y_mask, torch.Tensor):
        y_mask = torch.as_tensor(y_mask, dtype=torch.bool)
    if not isinstance(agent_mask, torch.Tensor):
        agent_mask = torch.as_tensor(agent_mask, dtype=torch.bool)
    
    # Handle both unimodal (K=1) and multimodal (K>1) predictions
    # Unimodal: [B, N, T, 2] → add mode dimension → [B, N, 1, T, 2]
    # Multimodal: [B, N, K, T, 2] → keep as is
    if pred.dim() == 4:
        # Unimodal case: add mode dimension
        pred = pred.unsqueeze(2)  # [B, N, T, 2] → [B, N, 1, T, 2]
    
    batch_size, max_agents, num_modes, future_steps, _ = pred.shape
    
    # Expand target to match pred shape for broadcasting: [B, N, T, 2] → [B, N, 1, T, 2]
    target_expanded = target.unsqueeze(2)  # [B, N, 1, T, 2]
    
    # Expand y_mask for broadcasting: [B, N, T] → [B, N, 1, T]
    y_mask_expanded = y_mask.unsqueeze(2)  # [B, N, 1, T]
    
    # Compute displacement errors: [B, N, K, T]
    # Euclidean distance between predicted and ground truth positions at each timestep
    displacements = torch.norm(pred - target_expanded, dim=-1)  # [B, N, K, T]
    
    # Apply y_mask: set invalid timesteps to 0 (they won't contribute to metrics)
    # Shape: [B, N, K, T]
    masked_displacements = displacements * y_mask_expanded.float()
    
    # Count valid timesteps per agent per mode: [B, N, K]
    valid_timesteps = y_mask_expanded.float().sum(dim=-1)  # [B, N, K]
    # Avoid division by zero: replace 0 with 1 (will be masked out by agent_mask later)
    valid_timesteps = torch.clamp(valid_timesteps, min=1.0)
    
    # Compute ADE per agent per mode: average displacement over valid timesteps
    # ADE[b, n, k] = sum_t(displacement[b, n, k, t] * mask[b, n, t]) / count_valid[b, n]
    # Shape: [B, N, K]
    ade_per_mode = masked_displacements.sum(dim=-1) / valid_timesteps  # [B, N, K]
    
    # Compute FDE per agent per mode: displacement at final valid timestep
    # For each agent, find the last valid timestep and extract displacement there
    # Shape: [B, N, K]
    fde_per_mode = torch.zeros_like(ade_per_mode)  # [B, N, K]
    
    # Find last valid timestep for each agent using vectorized operations
    # Create a mask of valid timesteps and find the last one per agent
    # y_mask: [B, N, T]
    # We need to find the index of the last True value in each [B, N] sequence
    
    # Create indices tensor: [T]
    timestep_indices = torch.arange(future_steps, device=y_mask.device)  # [T]
    
    # Expand to match y_mask shape: [B, N, T]
    timestep_indices_expanded = timestep_indices.view(1, 1, -1).expand(batch_size, max_agents, -1)
    
    # Set invalid timesteps to -1
    masked_indices = torch.where(y_mask, timestep_indices_expanded, torch.tensor(-1, device=y_mask.device))
    
    # Find the maximum index (last valid timestep) for each agent: [B, N]
    last_valid_timesteps, _ = torch.max(masked_indices, dim=2)  # [B, N]
    
    # Clamp to valid range [0, future_steps-1] to avoid index errors
    last_valid_timesteps = torch.clamp(last_valid_timesteps, min=0, max=future_steps-1)
    
    # Extract FDE at last valid timestep for all modes
    # Use advanced indexing to gather the displacements at the last valid timestep
    # displacements: [B, N, K, T]
    # We want to select displacements[b, n, k, last_valid_timesteps[b, n]] for all b, n, k
    
    batch_indices = torch.arange(batch_size, device=displacements.device).view(-1, 1, 1).expand(batch_size, max_agents, num_modes)
    agent_indices = torch.arange(max_agents, device=displacements.device).view(1, -1, 1).expand(batch_size, max_agents, num_modes)
    mode_indices = torch.arange(num_modes, device=displacements.device).view(1, 1, -1).expand(batch_size, max_agents, num_modes)
    time_indices = last_valid_timesteps.unsqueeze(2).expand(batch_size, max_agents, num_modes)
    
    fde_per_mode = displacements[batch_indices, agent_indices, mode_indices, time_indices]  # [B, N, K]
    
    # Select best mode per agent (minimum ADE and minimum FDE)
    # minADE: [B, N]
    minADE_per_agent, best_mode_ade = torch.min(ade_per_mode, dim=2)  # [B, N]
    
    # minFDE: [B, N]
    minFDE_per_agent, best_mode_fde = torch.min(fde_per_mode, dim=2)  # [B, N]
    
    # Apply agent_mask: only consider valid agents
    # Expand agent_mask for broadcasting: [B, N] → [B, N]
    valid_agents = agent_mask.float()  # [B, N]
    
    # Compute MissRate: fraction where minFDE > 2.0 meters
    # Shape: [B, N]
    miss_threshold = 2.0
    miss_per_agent = (minFDE_per_agent > miss_threshold).float()  # [B, N]
    
    # Aggregate metrics across all valid agents
    num_valid_agents = valid_agents.sum()
    
    if num_valid_agents == 0:
        # No valid agents: return zero metrics
        metrics = {
            'minADE': 0.0,
            'minFDE': 0.0,
            'MissRate': 0.0,
        }
        # Add per-mode metrics
        for k in range(num_modes):
            metrics[f'mode_{k}_ADE'] = 0.0
            metrics[f'mode_{k}_FDE'] = 0.0
        return metrics
    
    # Compute final metrics
    minADE = (minADE_per_agent * valid_agents).sum() / num_valid_agents
    minFDE = (minFDE_per_agent * valid_agents).sum() / num_valid_agents
    MissRate = (miss_per_agent * valid_agents).sum() / num_valid_agents
    
    metrics = {
        'minADE': float(minADE.item()),
        'minFDE': float(minFDE.item()),
        'MissRate': float(MissRate.item()),
    }
    
    # Compute per-mode metrics for analysis
    # For each mode k, compute average ADE and FDE across all valid agents
    for k in range(num_modes):
        mode_ade = (ade_per_mode[:, :, k] * valid_agents).sum() / num_valid_agents
        mode_fde = (fde_per_mode[:, :, k] * valid_agents).sum() / num_valid_agents
        
        metrics[f'mode_{k}_ADE'] = float(mode_ade.item())
        metrics[f'mode_{k}_FDE'] = float(mode_fde.item())
    
    return metrics


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

