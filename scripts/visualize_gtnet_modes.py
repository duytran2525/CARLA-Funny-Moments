from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_dataset import (  # noqa: E402
    MultiAgentTrajectoryDataset,
    collate_multi_agent_trajectory,
)
from core_perception.multi_agent_model import (  # noqa: E402
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
)
from core_perception.multi_agent_trajectory import build_adaptive_adjacency  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every multimodal GTNet trajectory mode for one processed sample."
    )
    parser.add_argument("--checkpoint", default="models/ablation_111_best.pt")
    parser.add_argument("--data-dir", default="data/multi_agent/processed")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--agent-index",
        type=int,
        default=-1,
        help="Agent index inside the sample. -1 selects the agent with the largest valid future displacement.",
    )
    parser.add_argument("--agent-id", type=int, default=-1, help="Optional CARLA actor id to plot.")
    parser.add_argument("--output", default="outputs/gtnet_modes_sample.png")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--adjacency-mode",
        choices=["sample", "fixed", "adaptive"],
        default="sample",
        help="sample uses the adjacency saved in the dataset; fixed/adaptive recompute it for comparison.",
    )
    parser.add_argument("--fixed-radius-m", type=float, default=100.0)
    parser.add_argument("--radius-base", type=float, default=None)
    parser.add_argument("--radius-alpha", type=float, default=None)
    parser.add_argument("--plot-frame", choices=["ego", "world"], default="ego")
    parser.add_argument("--metrics-json", default="", help="Optional path for per-mode metrics JSON.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> tuple[MultiAgentTrajectoryPredictor, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    if "model_config" not in checkpoint:
        raise RuntimeError(f"Checkpoint missing model_config: {path}")

    valid_config_keys = {field.name for field in fields(MultiAgentModelConfig)}
    raw_config = dict(checkpoint["model_config"])
    model_config = MultiAgentModelConfig(
        **{key: value for key, value in raw_config.items() if key in valid_config_keys}
    )
    model = MultiAgentTrajectoryPredictor(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("x", "y", "adj", "x_mask", "y_mask", "agent_mask", "actor_ids", "ego_pose"):
        value = moved.get(key)
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
    return moved


def last_valid_history_values(x: torch.Tensor, x_mask: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    valid_counts = x_mask.long().sum(dim=-1).clamp_min(1)
    indices = (valid_counts - 1).view(x.shape[0], x.shape[1], 1, 1)
    indices = indices.expand(-1, -1, 1, stop - start)
    return torch.gather(x[..., start:stop], dim=2, index=indices).squeeze(2)


def build_fixed_adjacency(positions: torch.Tensor, agent_mask: torch.Tensor, radius_m: float) -> torch.Tensor:
    batch_size, max_agents, _ = positions.shape
    adj = torch.eye(max_agents, device=positions.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    distances = torch.cdist(positions.float(), positions.float(), p=2)
    connected = distances <= float(radius_m)
    valid = agent_mask.bool()
    adj = torch.where(connected & valid.unsqueeze(1) & valid.unsqueeze(2), torch.ones_like(adj), torch.zeros_like(adj))
    eye = torch.eye(max_agents, device=positions.device, dtype=torch.float32).unsqueeze(0)
    return torch.maximum(adj, eye * valid.unsqueeze(1).float())


def build_runtime_adjacency(
    batch: dict[str, Any],
    mode: str,
    fixed_radius_m: float,
    radius_base: float,
    radius_alpha: float,
) -> torch.Tensor:
    if mode == "sample":
        return batch["adj"]

    positions = last_valid_history_values(batch["x"], batch["x_mask"], 0, 2)
    agent_mask = batch["agent_mask"]
    if mode == "fixed":
        return build_fixed_adjacency(positions, agent_mask, fixed_radius_m)

    velocities = last_valid_history_values(batch["x"], batch["x_mask"], 2, 4)
    adjs = []
    for sample_idx in range(positions.shape[0]):
        n_valid = int(agent_mask[sample_idx].sum().item())
        pos_np = positions[sample_idx, :n_valid].detach().cpu().numpy()
        vel_np = velocities[sample_idx, :n_valid].detach().cpu().numpy()
        adj_np = build_adaptive_adjacency(pos_np, vel_np, r_base=radius_base, alpha=radius_alpha)
        padded = torch.zeros(
            (positions.shape[1], positions.shape[1]),
            dtype=torch.float32,
            device=positions.device,
        )
        if n_valid > 0:
            padded[:n_valid, :n_valid] = torch.as_tensor(adj_np, dtype=torch.float32, device=positions.device)
        adjs.append(padded)
    return torch.stack(adjs, dim=0)


def choose_agent(batch: dict[str, Any], agent_index: int, agent_id: int) -> int:
    actor_ids = batch["actor_ids"][0].detach().cpu()
    agent_mask = batch["agent_mask"][0].detach().cpu().bool()
    valid_indices = torch.nonzero(agent_mask, as_tuple=False).flatten().tolist()
    if not valid_indices:
        raise RuntimeError("Sample has no valid agents.")

    if agent_id >= 0:
        matches = torch.nonzero(actor_ids == int(agent_id), as_tuple=False).flatten().tolist()
        matches = [idx for idx in matches if idx in valid_indices]
        if not matches:
            raise RuntimeError(f"actor id {agent_id} is not present in this sample.")
        return int(matches[0])

    if agent_index >= 0:
        if agent_index not in valid_indices:
            raise RuntimeError(f"agent index {agent_index} is not valid for this sample.")
        return int(agent_index)

    y = batch["y"][0].detach().cpu()
    y_mask = batch["y_mask"][0].detach().cpu().bool()
    x = batch["x"][0].detach().cpu()
    x_mask = batch["x_mask"][0].detach().cpu().bool()

    best_idx = valid_indices[0]
    best_disp = -1.0
    for idx in valid_indices:
        valid_future = torch.nonzero(y_mask[idx], as_tuple=False).flatten()
        valid_history = torch.nonzero(x_mask[idx], as_tuple=False).flatten()
        if len(valid_future) == 0 or len(valid_history) == 0:
            continue
        first_pos = x[idx, int(valid_history[-1]), :2]
        last_pos = y[idx, int(valid_future[-1]), :2]
        disp = float(torch.linalg.norm(last_pos - first_pos).item())
        if disp > best_disp:
            best_disp = disp
            best_idx = int(idx)
    return int(best_idx)


def compute_per_mode_metrics(
    pred_agent: torch.Tensor,
    target_agent: torch.Tensor,
    y_mask_agent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    valid = y_mask_agent.bool()
    if not bool(valid.any()):
        raise RuntimeError("Selected agent has no valid future frames.")

    displacement = torch.linalg.norm(pred_agent[:, valid, :] - target_agent[valid, :].unsqueeze(0), dim=-1)
    ade = displacement.mean(dim=-1)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    last_idx = int(valid_indices[-1].item())
    fde = torch.linalg.norm(pred_agent[:, last_idx, :] - target_agent[last_idx, :].unsqueeze(0), dim=-1)
    best_mode = int(torch.argmin(ade).item())
    return ade.detach().cpu(), fde.detach().cpu(), best_mode


def ego_to_world(points: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    right = points[..., 0]
    forward = points[..., 1]
    ego_x = float(ego_pose[0].item())
    ego_y = float(ego_pose[1].item())
    yaw = math.radians(float(ego_pose[2].item()))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = ego_x + cos_yaw * forward - sin_yaw * right
    world_y = ego_y + sin_yaw * forward + cos_yaw * right
    return torch.stack([world_x, world_y], dim=-1)


def plot_modes(
    *,
    batch: dict[str, Any],
    pred: torch.Tensor,
    agent_idx: int,
    ade: torch.Tensor,
    fde: torch.Tensor,
    best_mode: int,
    output_path: Path,
    plot_frame: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization. Install matplotlib first.") from exc

    history = batch["x"][0, agent_idx, :, :2].detach().cpu()
    history_mask = batch["x_mask"][0, agent_idx].detach().cpu().bool()
    future = batch["y"][0, agent_idx].detach().cpu()
    future_mask = batch["y_mask"][0, agent_idx].detach().cpu().bool()
    pred_agent = pred[0, agent_idx].detach().cpu()

    if plot_frame == "world":
        ego_pose = batch["ego_pose"][0].detach().cpu()
        history = ego_to_world(history, ego_pose)
        future = ego_to_world(future, ego_pose)
        pred_agent = ego_to_world(pred_agent, ego_pose)
        xlabel = "world x (m)"
        ylabel = "world y (m)"
    else:
        xlabel = "right (m)"
        ylabel = "forward (m)"

    actor_id = int(batch["actor_ids"][0, agent_idx].detach().cpu().item())
    sample_path = str(batch["sample_paths"][0]) if batch.get("sample_paths") else ""
    town = str(batch["towns"][0]) if batch.get("towns") else ""
    run_id = str(batch["run_ids"][0]) if batch.get("run_ids") else ""

    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=140)
    colors = plt.cm.tab10.colors

    if bool(history_mask.any()):
        hist_xy = history[history_mask]
        ax.plot(hist_xy[:, 0], hist_xy[:, 1], color="black", linewidth=2.5, marker="o", markersize=3.5, label="history")
        ax.scatter(hist_xy[-1, 0], hist_xy[-1, 1], color="black", s=50, zorder=5, label="anchor")

    if bool(future_mask.any()):
        gt_xy = future[future_mask]
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#1b9e77", linewidth=3.0, marker="o", markersize=3.0, label="ground truth")

    for mode_idx in range(pred_agent.shape[0]):
        mode_xy = pred_agent[mode_idx]
        is_best = mode_idx == int(best_mode)
        label = f"mode {mode_idx} ADE={float(ade[mode_idx]):.2f} FDE={float(fde[mode_idx]):.2f}"
        ax.plot(
            mode_xy[:, 0],
            mode_xy[:, 1],
            color=colors[mode_idx % len(colors)],
            linewidth=3.0 if is_best else 1.6,
            alpha=0.95 if is_best else 0.58,
            linestyle="-" if is_best else "--",
            label=label + (" best" if is_best else ""),
        )

    ax.set_title(f"GTNet modes | {town} {run_id} | actor_id={actor_id} | sample={Path(sample_path).name}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.7)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    data_dir = (PROJECT_ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    dataset = MultiAgentTrajectoryDataset(data_dir)
    if int(args.sample_index) < 0 or int(args.sample_index) >= len(dataset):
        raise RuntimeError(f"--sample-index must be in [0, {len(dataset) - 1}], got {args.sample_index}.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_multi_agent_trajectory,
        sampler=[int(args.sample_index)],
    )
    raw_batch = next(iter(loader))
    batch = move_batch_to_device(raw_batch, device)

    radius_base = float(args.radius_base) if args.radius_base is not None else float(model.config.radius_base)
    radius_alpha = float(args.radius_alpha) if args.radius_alpha is not None else float(model.config.radius_alpha)
    adj = build_runtime_adjacency(
        batch=batch,
        mode=str(args.adjacency_mode),
        fixed_radius_m=float(args.fixed_radius_m),
        radius_base=radius_base,
        radius_alpha=radius_alpha,
    )

    with torch.inference_mode():
        pred = model(
            batch["x"],
            adj,
            x_mask=batch["x_mask"],
            agent_mask=batch["agent_mask"],
        )
    if pred.dim() == 4:
        pred = pred.unsqueeze(2)

    agent_idx = choose_agent(batch, agent_index=int(args.agent_index), agent_id=int(args.agent_id))
    ade, fde, best_mode = compute_per_mode_metrics(
        pred_agent=pred[0, agent_idx],
        target_agent=batch["y"][0, agent_idx],
        y_mask_agent=batch["y_mask"][0, agent_idx],
    )

    plot_modes(
        batch=raw_batch,
        pred=pred.detach().cpu(),
        agent_idx=agent_idx,
        ade=ade,
        fde=fde,
        best_mode=best_mode,
        output_path=output_path,
        plot_frame=str(args.plot_frame),
    )

    actor_id = int(raw_batch["actor_ids"][0, agent_idx].item())
    metrics = {
        "checkpoint": str(checkpoint_path),
        "sample_path": str(raw_batch["sample_paths"][0]),
        "town": str(raw_batch["towns"][0]),
        "run_id": str(raw_batch["run_ids"][0]),
        "agent_index": int(agent_idx),
        "actor_id": int(actor_id),
        "adjacency_mode": str(args.adjacency_mode),
        "num_modes": int(pred.shape[2]),
        "best_mode": int(best_mode),
        "best_mode_ADE": float(ade[best_mode].item()),
        "best_mode_FDE": float(fde[best_mode].item()),
        "mode_0_ADE": float(ade[0].item()),
        "mode_0_FDE": float(fde[0].item()),
        "per_mode_ADE": [float(v) for v in ade.tolist()],
        "per_mode_FDE": [float(v) for v in fde.tolist()],
        "checkpoint_val_ADE": checkpoint.get("val_ade"),
        "checkpoint_val_FDE": checkpoint.get("val_fde"),
        "output": str(output_path),
    }

    if args.metrics_json:
        metrics_path = (PROJECT_ROOT / args.metrics_json).resolve() if not Path(args.metrics_json).is_absolute() else Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
