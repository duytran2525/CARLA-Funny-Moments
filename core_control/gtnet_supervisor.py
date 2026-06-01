from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # pyrefly: ignore [missing-import]
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    # pyrefly: ignore [missing-import]
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    # pyrefly: ignore [missing-import]
    import carla
except ImportError:  # pragma: no cover
    carla = None

from core_perception.multi_agent_model import (
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
)
from core_perception.multi_agent_trajectory import (
    ActorState,
    EgoState,
    FrameData,
    actor_feature_in_anchor_frame,
    build_adaptive_adjacency,
)


@dataclass(frozen=True)
class GTNetSupervisorConfig:
    model_path: str
    enabled: bool = False
    inference_every_n_ticks: int = 3
    history_frames: int = 20
    expected_dt: float = 0.1
    fixed_delta: float = 0.033333
    max_agents: int = 32
    max_actor_distance_m: float = 100.0
    min_history_ratio: float = 0.55
    adjacency_mode: str = "fixed"
    fixed_adjacency_radius_m: float = 100.0
    danger_forward_min_m: float = 1.0
    danger_forward_max_m: float = 35.0
    danger_time_horizon_s: float = 3.0
    hard_brake_time_s: float = 1.2
    hard_brake_forward_m: float = 8.0
    caution_brake: float = 0.35
    emergency_brake: float = 0.75
    corridor_base_half_width_m: float = 1.25
    # BUG FIX: reduced from 0.035 → 0.025 → 0.018 and max from 3.0 → 2.5 → 2.2.
    # At 35m forward with 0.018: half_width = 1.25 + 0.63 = 1.88m (total 3.76m).
    # A CARLA lane is ~3.5-3.7m, so corridor stays within ego's own lane.
    corridor_growth_per_m: float = 0.018
    corridor_curve_gain: float = 0.55
    corridor_max_half_width_m: float = 2.2
    # BUG FIX: mode consensus — require at least this many modes (out of K) to
    # predict a corridor violation before treating as a real threat.
    # With K=5 modes, setting min_threat_modes=2 filters out noisy single-mode
    # false positives where 1 of 5 modes drifts into the ego corridor.
    min_threat_modes: int = 2
    # Receding-vehicle filter: a leading vehicle is considered to be "pulling
    # away" (and therefore safe) when its predicted forward position at the
    # END of the danger horizon exceeds its current position by at least
    # min_receding_gap_m.  The filter is bypassed for vehicles that are
    # currently closer than receding_min_initial_fwd_m, because a close
    # vehicle can still be dangerous even if it momentarily appears to be
    # moving away (e.g., oscillating, about to cut back in).
    min_receding_gap_m: float = 2.0
    receding_min_initial_fwd_m: float = 5.0
    draw_debug: bool = False


class GTNetSupervisor:
    """Runtime safety wrapper for a trained GTNet multi-agent trajectory model."""

    def __init__(
        self,
        config: GTNetSupervisorConfig,
        device: str = "auto",
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for GTNetSupervisor.")
        if np is None:
            raise RuntimeError("numpy is required for GTNetSupervisor.")

        self.config = config
        self._device = self._resolve_device(device)
        self._model, self._model_config, self._checkpoint_meta = self._load_model(config.model_path)
        self._adjacency_mode = self._resolve_adjacency_mode(config.adjacency_mode)
        self._history_stride_ticks = max(
            1,
            int(round(float(config.expected_dt) / max(float(config.fixed_delta), 1e-3))),
        )
        self._raw_history: deque[FrameData] = deque(
            maxlen=max(
                int(config.history_frames),
                int(config.history_frames) * self._history_stride_ticks + 2,
            )
        )
        self._last_inference_step: Optional[int] = None
        self._cached_result = self._empty_result("not_started")

        # ── BUG FIX: Detect dt / future_steps mismatches ──────────────────────
        # The model was trained with a specific future_steps (e.g. 60 at 20fps
        # = 3s horizon).  If runtime expected_dt doesn't match the training dt,
        # the supervisor will misinterpret prediction timestamps and corridor
        # time-to-collision calculations.
        ckpt_future_steps = int(self._model_config.future_steps)
        runtime_dt = float(config.expected_dt)
        carla_dt = float(config.fixed_delta)

        if abs(runtime_dt - carla_dt) > 1e-6 and self._history_stride_ticks > 1:
            logging.warning(
                "GTNet expected_dt=%.4f != fixed_delta=%.4f → history is sub-sampled "
                "(stride=%d ticks). If the model was trained at %.0f fps, "
                "set expected_dt=%.4f to feed every CARLA tick without sub-sampling.",
                runtime_dt, carla_dt, self._history_stride_ticks,
                1.0 / carla_dt, carla_dt,
            )

        logging.info(
            "GTNet supervisor loaded: model=%s device=%s history=%d stride_ticks=%d "
            "future_steps=%d expected_dt=%.4f carla_dt=%.4f "
            "GAT=%s multimodal=%s modes=%d checkpoint_adaptive_radius=%s "
            "runtime_adjacency=%s fixed_radius=%.1fm val_ADE=%s val_FDE=%s",
            config.model_path,
            self._device,
            int(config.history_frames),
            int(self._history_stride_ticks),
            ckpt_future_steps,
            runtime_dt,
            carla_dt,
            bool(self._model_config.enable_gat),
            bool(self._model_config.enable_multimodal),
            int(self._model_config.num_modes),
            bool(self._model_config.enable_adaptive_radius),
            self._adjacency_mode,
            float(self.config.fixed_adjacency_radius_m),
            self._checkpoint_meta.get("val_ade"),
            self._checkpoint_meta.get("val_fde"),
        )

    @staticmethod
    def _resolve_device(device: str) -> "torch.device":
        requested = str(device or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested for GTNet but unavailable. Falling back to CPU.")
            requested = "cpu"
        return torch.device(requested)

    def _load_model(self, model_path: str) -> tuple[MultiAgentTrajectoryPredictor, MultiAgentModelConfig, Dict[str, Any]]:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GTNet checkpoint not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self._device)
        except Exception as exc:
            logging.debug("weights_only GTNet load failed, retrying regular torch.load: %s", exc)
            checkpoint = torch.load(path, map_location=self._device)

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(f"GTNet checkpoint has unsupported format: {path}")
        if "model_config" not in checkpoint:
            raise RuntimeError(f"GTNet checkpoint missing model_config: {path}")

        valid_config_keys = set(MultiAgentModelConfig.__dataclass_fields__)
        model_config = MultiAgentModelConfig(
            **{
                key: value
                for key, value in dict(checkpoint["model_config"]).items()
                if key in valid_config_keys
            }
        )
        model = MultiAgentTrajectoryPredictor(model_config).to(self._device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        meta = {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_ade": checkpoint.get("val_ade"),
            "val_fde": checkpoint.get("val_fde"),
        }
        return model, model_config, meta

    def _resolve_adjacency_mode(self, mode: str) -> str:
        normalized = str(mode or "fixed").strip().lower().replace("-", "_")
        if normalized in {"checkpoint", "from_checkpoint", "model"}:
            return "adaptive" if bool(self._model_config.enable_adaptive_radius) else "fixed"
        if normalized in {"fixed", "adaptive"}:
            return normalized
        logging.warning(
            "Unsupported GTNet adjacency_mode=%r. Falling back to fixed radius %.1fm.",
            mode,
            float(self.config.fixed_adjacency_radius_m),
        )
        return "fixed"

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        return {
            "enabled": True,
            "ready": False,
            "threat": False,
            "brake": 0.0,
            "reason": reason,
            "num_agents": 0,
            "cache_hit": False,
            "latency_ms": 0.0,
        }

    @staticmethod
    def _actor_kind(actor: Any) -> Optional[str]:
        type_id = str(getattr(actor, "type_id", "")).lower()
        if type_id.startswith("vehicle."):
            return "vehicle"
        if type_id.startswith("walker.pedestrian"):
            return "pedestrian"
        return None

    @staticmethod
    def _velocity_xy(actor: Any) -> tuple[float, float]:
        try:
            vel = actor.get_velocity()
            return float(vel.x), float(vel.y)
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _yaw_deg(actor: Any) -> float:
        try:
            return float(actor.get_transform().rotation.yaw)
        except Exception:
            return 0.0

    def _capture_frame(self, world: Any, ego_vehicle: Any, step_idx: int) -> FrameData:
        ego_loc = ego_vehicle.get_location()
        ego_vel = ego_vehicle.get_velocity()
        ego_tf = ego_vehicle.get_transform()
        ego = EgoState(
            actor_id=int(ego_vehicle.id),
            x=float(ego_loc.x),
            y=float(ego_loc.y),
            z=float(ego_loc.z),
            vx=float(ego_vel.x),
            vy=float(ego_vel.y),
            yaw=float(ego_tf.rotation.yaw),
        )

        actors: Dict[int, ActorState] = {}
        try:
            world_actors = world.get_actors()
            actor_iter = list(world_actors.filter("vehicle.*")) + list(world_actors.filter("walker.pedestrian.*"))
        except Exception:
            actor_iter = []

        max_dist = float(self.config.max_actor_distance_m)
        for actor in actor_iter:
            try:
                actor_id = int(actor.id)
            except Exception:
                continue
            if actor_id == int(ego_vehicle.id):
                continue
            kind = self._actor_kind(actor)
            if kind is None:
                continue
            try:
                loc = actor.get_location()
                distance_m = float(ego_loc.distance(loc))
            except Exception:
                continue
            if not math.isfinite(distance_m) or distance_m > max_dist:
                continue

            vx, vy = self._velocity_xy(actor)
            actors[actor_id] = ActorState(
                actor_id=actor_id,
                actor_type=kind,
                x=float(loc.x),
                y=float(loc.y),
                z=float(loc.z),
                vx=float(vx),
                vy=float(vy),
                yaw=float(self._yaw_deg(actor)),
                distance_m=distance_m,
            )

        try:
            timestamp = float(world.get_snapshot().timestamp.elapsed_seconds)
            frame_id = int(world.get_snapshot().frame)
        except Exception:
            timestamp = time.time()
            frame_id = int(step_idx)

        town = ""
        try:
            town = str(world.get_map().name)
        except Exception:
            pass

        return FrameData(
            frame=frame_id,
            timestamp=timestamp,
            run_id="live",
            town=town,
            ego=ego,
            actors=actors,
        )

    def _sample_history(self) -> list[FrameData]:
        frames = list(self._raw_history)
        needed = int(self.config.history_frames)
        if len(frames) < needed:
            return []

        expected_dt = max(1e-3, float(self.config.expected_dt))
        latest_ts = float(frames[-1].timestamp)
        earliest_target = latest_ts - float(needed - 1) * expected_dt
        tolerance = max(expected_dt * 0.55, float(self.config.fixed_delta) * 1.1)
        if float(frames[0].timestamp) > earliest_target + tolerance:
            return []

        targets = [earliest_target + i * expected_dt for i in range(needed)]
        sampled: list[FrameData] = []
        cursor = 0
        for target_ts in targets:
            while (
                cursor + 1 < len(frames)
                and abs(float(frames[cursor + 1].timestamp) - target_ts)
                <= abs(float(frames[cursor].timestamp) - target_ts)
            ):
                cursor += 1
            sampled.append(frames[cursor])

        if len({frame.frame for frame in sampled}) < needed:
            return []
        return sampled

    def _select_actor_ids(self, history: list[FrameData]) -> list[int]:
        if not history:
            return []
        anchor = history[-1]
        min_valid = max(1, int(round(float(self.config.min_history_ratio) * len(history))))
        actor_ids = []
        for actor_id, actor in anchor.actors.items():
            valid_count = sum(1 for frame in history if actor_id in frame.actors)
            if valid_count >= min_valid:
                actor_ids.append((actor_id, float(actor.distance_m)))
        actor_ids.sort(key=lambda item: item[1])
        return [actor_id for actor_id, _distance in actor_ids[: int(self.config.max_agents)]]

    def _build_tensors(self, history: list[FrameData], actor_ids: list[int]) -> Optional[Dict[str, Any]]:
        if np is None:
            return None
        if not history or not actor_ids:
            return None

        anchor = history[-1]
        n_agents = len(actor_ids)
        history_len = len(history)
        x = np.zeros((n_agents, history_len, int(self._model_config.input_dim)), dtype=np.float32)
        x_mask = np.zeros((n_agents, history_len), dtype=bool)
        global_positions = np.zeros((n_agents, 2), dtype=np.float64)
        velocities = np.zeros((n_agents, 2), dtype=np.float32)

        for agent_idx, actor_id in enumerate(actor_ids):
            anchor_actor = anchor.actors.get(actor_id)
            if anchor_actor is None:
                continue
            global_positions[agent_idx] = np.asarray([anchor_actor.x, anchor_actor.y], dtype=np.float64)
            velocities[agent_idx] = np.asarray([anchor_actor.vx, anchor_actor.vy], dtype=np.float32)

            for hist_idx, frame in enumerate(history):
                actor = frame.actors.get(actor_id)
                if actor is None:
                    continue
                feat = np.asarray(actor_feature_in_anchor_frame(actor, anchor.ego), dtype=np.float32)
                x[agent_idx, hist_idx, : min(feat.shape[0], x.shape[-1])] = feat[: x.shape[-1]]
                x_mask[agent_idx, hist_idx] = True

        if self._adjacency_mode == "adaptive":
            adj = build_adaptive_adjacency(
                global_positions,
                velocities,
                r_base=float(self._model_config.radius_base),
                alpha=float(self._model_config.radius_alpha),
            )
        else:
            adj = self._build_fixed_adjacency(
                global_positions,
                radius_m=float(self.config.fixed_adjacency_radius_m),
            )

        agent_mask = x_mask.any(axis=-1)
        return {
            "x": torch.as_tensor(x, dtype=torch.float32, device=self._device).unsqueeze(0),
            "adj": torch.as_tensor(adj, dtype=torch.float32, device=self._device).unsqueeze(0),
            "x_mask": torch.as_tensor(x_mask, dtype=torch.bool, device=self._device).unsqueeze(0),
            "agent_mask": torch.as_tensor(agent_mask, dtype=torch.bool, device=self._device).unsqueeze(0),
        }

    @staticmethod
    def _build_fixed_adjacency(positions: Any, radius_m: float) -> Any:
        n_agents = int(positions.shape[0])
        adj = np.eye(n_agents, dtype=np.float32)
        radius = float(radius_m)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dx = float(positions[i, 0] - positions[j, 0])
                dy = float(positions[i, 1] - positions[j, 1])
                if math.hypot(dx, dy) <= radius:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    @staticmethod
    def _steer_to_curvature(vehicle_steer: float) -> float:
        steer = max(-1.0, min(1.0, float(vehicle_steer)))
        wheel_angle = steer * math.radians(35.0)
        curvature = math.tan(wheel_angle) / 2.85
        return float(max(-0.45, min(0.45, curvature)))

    def _corridor_half_width(self, forward_m: float, vehicle_steer: float) -> float:
        base = float(self.config.corridor_base_half_width_m)
        growth = float(self.config.corridor_growth_per_m) * max(0.0, float(forward_m))
        curve = abs(self._steer_to_curvature(vehicle_steer))
        curve_bonus = float(self.config.corridor_curve_gain) * curve * max(0.0, float(forward_m))
        return float(min(float(self.config.corridor_max_half_width_m), base + growth + curve_bonus))

    def _corridor_center_right(self, forward_m: float, vehicle_steer: float) -> float:
        curvature = self._steer_to_curvature(vehicle_steer)
        return 0.5 * curvature * float(forward_m) * float(forward_m)

    @staticmethod
    def _mode_is_approaching(
        pred_np: "np.ndarray",
        agent_idx: int,
        mode_idx: int,
        max_t_idx: int,
        vehicle_steer: float,
        corridor_center_fn: "Any",
    ) -> bool:
        """Check if an agent's predicted trajectory is moving TOWARD the ego corridor.

        Returns True if the lateral distance to corridor center is decreasing
        over the prediction horizon. This filters out vehicles in adjacent lanes
        that maintain a constant offset (parallel traffic).
        """
        # Sample lateral distances at the start (25%) and end (75%) of the horizon
        t_early = max(0, max_t_idx // 4)
        t_late = min(max_t_idx - 1, max_t_idx * 3 // 4)
        if t_early >= t_late:
            return True  # Too few steps to judge; assume approaching

        right_early = float(pred_np[agent_idx, mode_idx, t_early, 0])
        fwd_early = float(pred_np[agent_idx, mode_idx, t_early, 1])
        right_late = float(pred_np[agent_idx, mode_idx, t_late, 0])
        fwd_late = float(pred_np[agent_idx, mode_idx, t_late, 1])

        center_early = corridor_center_fn(max(fwd_early, 0.1), vehicle_steer)
        center_late = corridor_center_fn(max(fwd_late, 0.1), vehicle_steer)

        lat_dist_early = abs(right_early - center_early)
        lat_dist_late = abs(right_late - center_late)

        # Agent is approaching if lateral distance decreased by at least 0.3m
        return lat_dist_late < lat_dist_early - 0.3

    def _assess_predictions(
        self,
        pred: "torch.Tensor",
        actor_ids: list[int],
        vehicle_steer: float,
        history: Optional[list] = None,
    ) -> tuple[Dict[str, Any], np.ndarray]:
        pred_np = pred.detach().squeeze(0).float().cpu().numpy()
        if pred_np.ndim == 4:
            # [N, K, T, 2]
            pass
        elif pred_np.ndim == 3:
            pred_np = pred_np[:, None, :, :]
        else:
            return self._empty_result("bad_prediction_shape"), pred_np

        future_dt = float(self.config.expected_dt)
        max_t_idx = min(
            pred_np.shape[2],
            max(1, int(math.ceil(float(self.config.danger_time_horizon_s) / max(future_dt, 1e-3)))),
        )
        num_modes = int(pred_np.shape[1])
        min_threat_modes = min(int(self.config.min_threat_modes), num_modes)

        # ── BUG FIX: Mode consensus + approach velocity filter ────────────────
        # Old code triggered braking if ANY single mode at ANY timestep placed an
        # agent inside the corridor.  With K=5 multimodal predictions, noisy modes
        # frequently "drift" adjacent-lane vehicles or sidewalk pedestrians into
        # the ego corridor, causing false braking.
        #
        # Fix:
        #  1) For each agent, count how many modes predict a corridor violation.
        #  2) Only treat as threat if >= min_threat_modes modes agree.
        #  3) Verify the agent is actually APPROACHING the corridor (lateral
        #     distance decreasing), not just passing parallel.

        best: Optional[Dict[str, Any]] = None
        rear_warnings: list[Dict[str, Any]] = []

        # Build a lookup of current position, speed, lateral error, heading,
        # and towards-corridor velocity per actor from the last observed frame
        # (ground-truth ego-centric state).
        current_fwd_by_id: Dict[int, float] = {}
        current_speed_by_id: Dict[int, float] = {}
        current_lat_err_by_id: Dict[int, float] = {}
        current_half_width_by_id: Dict[int, float] = {}
        current_v_towards_by_id: Dict[int, float] = {}
        current_heading_y_by_id: Dict[int, float] = {}
        current_right_by_id: Dict[int, float] = {}

        if history:
            anchor = history[-1]
            for aid, actor_state in anchor.actors.items():
                # actor_feature_in_anchor_frame returns:
                #   [0] local_x (lateral/right), [1] local_y (forward),
                #   [2] rel_vx (lateral vel), [3] rel_vy (forward vel),
                #   [4] heading_x, [5] heading_y (heading unit vector in ego frame)
                feat = actor_feature_in_anchor_frame(actor_state, anchor.ego)
                if len(feat) >= 6:
                    fwd_m = float(feat[1])
                    right_m = float(feat[0])
                    lat_vx = float(feat[2])
                    heading_y = float(feat[5])

                    current_fwd_by_id[aid] = fwd_m
                    current_right_by_id[aid] = right_m
                    current_speed_by_id[aid] = float(math.hypot(actor_state.vx, actor_state.vy))
                    current_heading_y_by_id[aid] = heading_y

                    center_right = self._corridor_center_right(fwd_m, vehicle_steer)
                    half_width = self._corridor_half_width(fwd_m, vehicle_steer)
                    current_lat_err_by_id[aid] = abs(right_m - center_right)
                    current_half_width_by_id[aid] = half_width

                    # Determine speed component heading TOWARDS the corridor center
                    if right_m > center_right:
                        # Actor is on the right, moving left (-lat_vx) is moving towards center
                        v_towards = -lat_vx
                    else:
                        # Actor is on the left, moving right (+lat_vx) is moving towards center
                        v_towards = lat_vx
                    current_v_towards_by_id[aid] = v_towards

        for agent_idx, actor_id in enumerate(actor_ids):
            # ── BUG FIX: Euclidean distance pre-filter ─────────────────────────
            # The corridor scan only checks forward_m ∈ [1, 35], but an actor
            # at 70m Euclidean distance can still have forward_m=30 if it's
            # mostly ahead.  Its predicted trajectory is unreliable at that
            # range, so skip actors whose ACTUAL current distance exceeds the
            # danger zone.
            if actor_id in current_fwd_by_id and actor_id in current_right_by_id:
                _cur_fwd = current_fwd_by_id[actor_id]
                _cur_right = current_right_by_id[actor_id]
                _actual_dist = math.hypot(_cur_fwd, _cur_right)
                if _actual_dist > float(self.config.danger_forward_max_m):
                    continue  # Actor is beyond danger range — skip entirely

            # ── BUG FIX: Far same-direction same-lane vehicle filter ───────────
            # A vehicle 20-35m ahead on the SAME lane, driving the SAME direction
            # at similar speed, will have its entire predicted trajectory inside
            # the corridor (because it IS in the same lane).  This is normal
            # car-following, NOT a collision threat.  Without this filter, the
            # supervisor triggers caution_brake continuously for any leading
            # vehicle on the same lane.
            #
            # Conditions for filtering:
            #  1. heading_y > 0.8 (within ~±37° of ego heading — same direction)
            #  2. current forward distance > 15m (far enough to not be urgent)
            #  3. relative forward velocity >= -1.0 m/s (not rapidly closing in;
            #     rel_vy >= 0 means actor moves forward relative to ego,
            #     rel_vy slightly negative means closing slowly — still safe
            #     at 15m+)
            if (actor_id in current_heading_y_by_id
                    and actor_id in current_fwd_by_id
                    and current_fwd_by_id[actor_id] > 15.0
                    and current_heading_y_by_id[actor_id] > 0.8):
                # Check relative forward velocity from ground-truth features
                if history:
                    _anchor = history[-1]
                    _actor_st = _anchor.actors.get(actor_id)
                    if _actor_st is not None:
                        _feat = actor_feature_in_anchor_frame(_actor_st, _anchor.ego)
                        if len(_feat) >= 4:
                            _rel_vy = float(_feat[3])  # relative forward velocity
                            if _rel_vy >= -1.0:
                                continue  # Same-lane leader, not closing fast — safe

            # Stationary/Slow Adjacent Vehicle Filter & Parallel Motion Filter:
            # 1. If the vehicle is currently stationary or moving extremely slowly (< 1.5 m/s)
            #    AND its current position is OUTSIDE our corridor, ignore it.
            # 2. If the vehicle is currently OUTSIDE our corridor AND its actual lateral
            #    velocity directed towards our corridor is very small (< 0.35 m/s), meaning
            #    it is driving parallel or moving away, then any predicted path that abruptly
            #    cuts into our corridor is noise. We ignore it to prevent false-alarm stops.
            if actor_id in current_speed_by_id and actor_id in current_lat_err_by_id:
                cur_speed = current_speed_by_id[actor_id]
                cur_lat_err = current_lat_err_by_id[actor_id]
                cur_half_w = current_half_width_by_id[actor_id]
                
                # Stationary adjacent filter
                if cur_speed < 1.5 and cur_lat_err > cur_half_w:
                    continue
                    
                # Parallel / Away motion filter
                if cur_lat_err > cur_half_w and actor_id in current_v_towards_by_id:
                    v_towards = current_v_towards_by_id[actor_id]
                    if v_towards < 0.35:
                        continue

            # Stationary Vehicle Prediction Override:
            # If the vehicle is currently stationary or moving extremely slowly (< 0.8 m/s),
            # it cannot physically swerve or cut lanes abruptly, nor can it move backwards.
            # However, because of relative velocity coordinate discrepancies when the ego vehicle
            # is moving, the GTNet model routinely predicts that stationary vehicles will
            # swerve 90 degrees left or run backwards in reverse.
            # We override all modes and future timesteps of the prediction to stay exactly at the
            # vehicle's current observed position, ensuring highly robust obstacle avoidance
            # while preventing chaotic trajectory hallucinations.
            if actor_id in current_speed_by_id and current_speed_by_id[actor_id] < 0.8:
                cur_right = current_right_by_id.get(actor_id, 0.0)
                cur_fwd = current_fwd_by_id.get(actor_id, 0.0)
                pred_np[agent_idx, :, :, 0] = cur_right
                pred_np[agent_idx, :, :, 1] = cur_fwd

            # Co-directional traffic filter (heading alignment):
            # When ego is turning/merging, the corridor shifts and widens via
            # curve bonus.  Vehicles in adjacent lanes driving the SAME DIRECTION
            # get swallowed into the expanded corridor and falsely trigger braking.
            # Fix: if the agent is heading roughly the same direction as ego
            # (heading_y > 0.7, i.e. within ~±45°) AND would be OUTSIDE the
            # STRAIGHT corridor (calculated with steer=0, no curve expansion),
            # then it is co-directional adjacent-lane traffic — not a threat.
            if actor_id in current_heading_y_by_id and actor_id in current_right_by_id:
                heading_y = current_heading_y_by_id[actor_id]
                if heading_y > 0.7:
                    right_m = current_right_by_id[actor_id]
                    fwd_m = current_fwd_by_id.get(actor_id, 0.0)
                    # Corridor with steer=0: center_right=0, no curve bonus
                    straight_half_w = self._corridor_half_width(fwd_m, 0.0)
                    straight_lat_err = abs(right_m)
                    if straight_lat_err > straight_half_w:
                        continue  # Same-direction traffic outside straight corridor

            # Phase 0: Classify agent direction — behind or in front of ego.
            # Use the ground-truth current forward position from history (last frame),
            # NOT the t=0 of the model prediction (which is the first future step).
            # This is much more reliable for determining front vs behind.
            if actor_id in current_fwd_by_id:
                current_fwd = current_fwd_by_id[actor_id]
            else:
                # Fallback: median of t=0 prediction across modes.
                # Log at DEBUG level so repeated fallbacks are traceable —
                # this usually means the actor appeared mid-sequence or the
                # history anchor frame is missing the actor (e.g., occluded).
                initial_forwards = [
                    float(pred_np[agent_idx, m, 0, 1]) for m in range(num_modes)
                ]
                initial_forwards.sort()
                current_fwd = initial_forwards[num_modes // 2]
                logging.debug(
                    "GTNet Phase0 fallback for actor_id=%d (not in history anchor): "
                    "using pred t=0 median forward=%.2fm. "
                    "If this repeats every inference, check history sampling.",
                    actor_id,
                    current_fwd,
                )
            median_initial_fwd = current_fwd
            is_from_behind = current_fwd < 0.0

            # Phase 1: Per-mode scan — find earliest corridor violation per mode.
            mode_threats: list[Optional[Dict[str, Any]]] = [None] * num_modes
            for mode_idx in range(num_modes):
                for t_idx in range(max_t_idx):
                    right_m = float(pred_np[agent_idx, mode_idx, t_idx, 0])
                    forward_m = float(pred_np[agent_idx, mode_idx, t_idx, 1])
                    if forward_m < float(self.config.danger_forward_min_m):
                        continue
                    if forward_m > float(self.config.danger_forward_max_m):
                        continue
                    center_right = self._corridor_center_right(forward_m, vehicle_steer)
                    half_width = self._corridor_half_width(forward_m, vehicle_steer)
                    lateral_error = abs(right_m - center_right)
                    if lateral_error > half_width:
                        continue

                    ttc_s = float(t_idx + 1) * future_dt
                    severity = (half_width - lateral_error) + max(0.0, 20.0 - forward_m) * 0.08
                    candidate = {
                        "actor_id": int(actor_id),
                        "mode": int(mode_idx),
                        "time_s": ttc_s,
                        "forward_m": forward_m,
                        "right_m": right_m,
                        "corridor_half_width_m": half_width,
                        "lateral_error_m": lateral_error,
                        "severity": float(severity),
                    }
                    # Keep earliest violation for this mode
                    if mode_threats[mode_idx] is None:
                        mode_threats[mode_idx] = candidate
                    break  # first violation per mode is enough

            # Phase 2: Mode consensus — count how many modes flagged a violation.
            threatening_modes = [m for m in mode_threats if m is not None]
            if len(threatening_modes) < min_threat_modes:
                continue  # Not enough modes agree — skip this agent

            # Phase 2b: Behind-ego agents → record as rear warning, skip braking.
            if is_from_behind:
                earliest = min(threatening_modes, key=lambda m: float(m["time_s"]))
                earliest["initial_fwd"] = median_initial_fwd
                rear_warnings.append(earliest)
                continue  # Do NOT affect front-threat braking decision

            # Phase 3: Approach velocity filter — reject agents moving parallel
            # OR agents that are RECEDING (pulling away forward from ego).
            #
            # KEY FIX: A vehicle driving straight AHEAD of ego, faster than ego,
            # will have waypoints at increasing forward_m (5m → 10m → 15m…)
            # all inside the narrow corridor.  Without this filter, GTNet would
            # brake every time any leading vehicle's trajectory is visualised.
            #
            # "Receding" = median last predicted forward position across modes is
            # substantially LARGER than the current observed position by at least
            # min_receding_gap_m, meaning the gap is growing → vehicle is faster
            # than ego → NOT a threat.
            #
            # Safety exception: if the vehicle is currently CLOSER than
            # receding_min_initial_fwd_m (default 5m), the receding filter is
            # DISABLED regardless — a close vehicle can still be dangerous even
            # if it momentarily seems to be pulling away (e.g., oscillation,
            # brake-then-cut, or short horizon noise).
            receding_gap_m = float(self.config.min_receding_gap_m)
            receding_min_fwd_m = float(self.config.receding_min_initial_fwd_m)
            last_t = max_t_idx - 1
            last_forwards = [
                float(pred_np[agent_idx, m, last_t, 1]) for m in range(num_modes)
            ]
            last_forwards.sort()
            median_last_fwd = last_forwards[num_modes // 2]
            is_receding = (
                median_last_fwd > median_initial_fwd + receding_gap_m
                and median_initial_fwd >= receding_min_fwd_m  # close-range safety exception
            )
            if is_receding:
                continue  # Vehicle ahead is pulling away — safe, no threat

            # ── BUG FIX: Relative-velocity receding filter ─────────────────────
            # The prediction-based receding filter above can miss vehicles that
            # are driving at a similar speed to ego (gap barely grows in 3s).
            # Use ground-truth relative forward velocity: if rel_vy >= 0
            # (actor is moving forward in ego frame = same speed or faster)
            # AND actor is far enough (> 10m), it's safe car-following.
            if (actor_id in current_fwd_by_id
                    and median_initial_fwd > 10.0
                    and history):
                _anchor_rv = history[-1]
                _actor_rv = _anchor_rv.actors.get(actor_id)
                if _actor_rv is not None:
                    _feat_rv = actor_feature_in_anchor_frame(_actor_rv, _anchor_rv.ego)
                    if len(_feat_rv) >= 4:
                        _rel_fwd_vel = float(_feat_rv[3])  # relative forward velocity
                        if _rel_fwd_vel >= -0.5:  # not closing faster than 0.5 m/s
                            continue  # Vehicle ahead at stable/growing gap — safe

            # EXCEPTION: Skip approach check when agent is already at close
            # range (forward_m < 3m) or deep inside corridor (lat_error < 0.5m).
            first_threat = min(threatening_modes, key=lambda m: float(m["time_s"]))
            already_close = (
                float(first_threat["forward_m"]) < 3.0
                or float(first_threat["lateral_error_m"]) < 0.5
            )
            if not already_close and num_modes >= 2:
                any_approaching = any(
                    self._mode_is_approaching(
                        pred_np, agent_idx,
                        int(m["mode"]),
                        max_t_idx, vehicle_steer,
                        self._corridor_center_right,
                    )
                    for m in threatening_modes
                )
                if not any_approaching:
                    continue  # All modes show parallel motion — not a real threat

            # Phase 4: Pick the earliest violation across agreeing modes.
            for candidate in threatening_modes:
                if best is None:
                    best = candidate
                else:
                    old_key = (float(best["time_s"]), -float(best["severity"]))
                    new_key = (float(candidate["time_s"]), -float(candidate["severity"]))
                    if new_key < old_key:
                        best = candidate

        # ── Build result dict ─────────────────────────────────────────────────
        # rear_threat_info: expose rear agent data for debug + throttle hints.
        # throttle_floor: when a rear vehicle is predicted to enter the
        # corridor, suggest the caller NOT reduce throttle below this value
        # (i.e. maintain speed instead of braking) to keep safe separation.
        has_rear_threat = len(rear_warnings) > 0
        rear_info: Dict[str, Any] = {}
        throttle_floor = 0.0
        if has_rear_threat:
            closest_rear = min(rear_warnings, key=lambda w: float(w["time_s"]))
            rear_ttc = float(closest_rear["time_s"])
            # Gentle throttle floor: faster approaching → higher floor
            if rear_ttc < 1.0:
                throttle_floor = 0.10  # Very close — maintain speed firmly
            elif rear_ttc < 2.0:
                throttle_floor = 0.05  # Nearby — light speed maintenance
            rear_info = {
                "rear_threat": True,
                "rear_actor_id": int(closest_rear["actor_id"]),
                "rear_time_s": rear_ttc,
                "rear_forward_m": float(closest_rear["forward_m"]),
                "rear_count": len(rear_warnings),
            }

        if best is None:
            return {
                "enabled": True,
                "ready": True,
                "threat": False,
                "brake": 0.0,
                "throttle_floor": throttle_floor,
                "reason": "clear",
                "num_agents": len(actor_ids),
                "cache_hit": False,
                "latency_ms": 0.0,
                **rear_info,
            }, pred_np

        # BUG FIX: Changed from OR to AND with time guard.
        # Old logic: emergency brake if TTC <= 1.2s OR forward <= 13m.
        # This caused emergency braking for actors at 12m predicted forward
        # even when TTC was 2.5s (plenty of time).  New logic: emergency
        # brake only when the threat is BOTH close AND imminent.
        hard = (
            float(best["time_s"]) <= float(self.config.hard_brake_time_s)
            and float(best["forward_m"]) <= float(self.config.hard_brake_forward_m)
        )
        brake = float(self.config.emergency_brake if hard else self.config.caution_brake)
        return {
            "enabled": True,
            "ready": True,
            "threat": True,
            "brake": brake,
            "throttle_floor": 0.0,  # Front threat → brake, no throttle floor
            "reason": "predicted_path_conflict",
            "num_agents": len(actor_ids),
            "cache_hit": False,
            "latency_ms": 0.0,
            **best,
            **rear_info,
        }, pred_np

    def update(
        self,
        world: Any,
        ego_vehicle: Any,
        step_idx: int,
        speed_kmh: float,
        vehicle_steer: float,
    ) -> Dict[str, Any]:
        if world is None or ego_vehicle is None:
            return self._empty_result("missing_world_or_vehicle")

        self._raw_history.append(self._capture_frame(world, ego_vehicle, step_idx))

        every_n = max(1, int(self.config.inference_every_n_ticks))
        if (
            self._last_inference_step is not None
            and (int(step_idx) - int(self._last_inference_step)) < every_n
        ):
            cached = dict(self._cached_result)
            cached["cache_hit"] = True
            return cached

        sampled_history = self._sample_history()
        if len(sampled_history) < int(self.config.history_frames):
            self._cached_result = self._empty_result(
                f"warming_history {len(self._raw_history)}/{int(self.config.history_frames) * int(self._history_stride_ticks)}"
            )
            return dict(self._cached_result)

        actor_ids = self._select_actor_ids(sampled_history)
        if not actor_ids:
            self._cached_result = self._empty_result("no_tracked_agents")
            self._cached_result["ready"] = True
            return dict(self._cached_result)

        payload = self._build_tensors(sampled_history, actor_ids)
        if payload is None:
            self._cached_result = self._empty_result("build_failed")
            return dict(self._cached_result)

        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            pred = self._model(
                payload["x"],
                payload["adj"],
                x_mask=payload["x_mask"],
                agent_mask=payload["agent_mask"],
            )
        latency_ms = (time.perf_counter() - infer_t0) * 1000.0

        result, pred_np = self._assess_predictions(pred, actor_ids, vehicle_steer=float(vehicle_steer), history=sampled_history)
        result["latency_ms"] = float(latency_ms)
        result["speed_kmh"] = float(speed_kmh)

        if bool(self.config.draw_debug):
            self._draw_debug(world, ego_vehicle, pred_np, actor_ids, result)

        self._last_inference_step = int(step_idx)
        self._cached_result = dict(result)
        return result

    def _draw_debug(
        self,
        world: Any,
        ego_vehicle: Any,
        pred_np: "np.ndarray",
        actor_ids: list[int],
        result: Dict[str, Any],
    ) -> None:
        if carla is None or world is None or ego_vehicle is None:
            return

        try:
            debug = world.debug
            transform = ego_vehicle.get_transform()
            origin = transform.location
            yaw_rad = math.radians(float(transform.rotation.yaw))
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            points = pred_np
            life_time = max(
                0.05,
                float(self.config.inference_every_n_ticks) * float(self.config.fixed_delta) * 1.2,
            )

            def _draw_trajectory(
                actor_id: int, mode_idx: int, color: "Any", max_steps: int = 30,
            ) -> None:
                try:
                    actor_idx = actor_ids.index(actor_id)
                except ValueError:
                    return
                traj = points[actor_idx, mode_idx, : min(points.shape[2], max_steps)]
                prev_loc = None
                for right_m, forward_m in traj:
                    wx = float(origin.x) + cos_yaw * float(forward_m) - sin_yaw * float(right_m)
                    wy = float(origin.y) + sin_yaw * float(forward_m) + cos_yaw * float(right_m)
                    loc = carla.Location(x=wx, y=wy, z=float(origin.z) + 0.65)
                    debug.draw_point(
                        loc, size=0.08, color=color,
                        life_time=life_time, persistent_lines=False,
                    )
                    if prev_loc is not None:
                        debug.draw_line(
                            prev_loc, loc, thickness=0.04, color=color,
                            life_time=life_time, persistent_lines=False,
                        )
                    prev_loc = loc

            # Determine active threat IDs to avoid drawing over them
            threat_actor_id = int(result.get("actor_id", -1)) if bool(result.get("threat", False)) else -1
            rear_actor_id = int(result.get("rear_actor_id", -1)) if bool(result.get("rear_threat", False)) else -1

            # 1. Draw neutral trajectory (gray) for all tracked actors (using mode 0 as best guess)
            # to confirm GTNet model inference is successfully running and predicting.
            for actor_id in actor_ids:
                if actor_id == threat_actor_id or actor_id == rear_actor_id:
                    continue
                # Draw mode 0 in high-visibility Cyan for other tracked agents
                _draw_trajectory(
                    actor_id, 0,
                    carla.Color(r=0, g=240, b=240),
                    max_steps=25,  # Draw a slightly shorter horizon to keep visual clean
                )

            # 2. Draw front threat trajectory (orange) - HIGHLIGHTED
            if threat_actor_id != -1:
                threat_mode = int(result.get("mode", 0))
                _draw_trajectory(
                    threat_actor_id, threat_mode,
                    carla.Color(r=255, g=120, b=0),
                    max_steps=30,
                )

            # 3. Draw rear threat trajectory (blue) - HIGHLIGHTED
            if rear_actor_id != -1:
                _draw_trajectory(
                    rear_actor_id, 0,
                    carla.Color(r=0, g=120, b=255),
                    max_steps=30,
                )

        except Exception as exc:
            logging.debug("GTNet debug draw skipped: %s", exc)

