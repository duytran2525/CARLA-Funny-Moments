import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from core_perception.multi_agent_dataset import MultiAgentTrajectoryDataset, collate_multi_agent_trajectory
from core_perception.multi_agent_model import (
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
    masked_smooth_l1_loss,
)
from core_perception.multi_agent_trajectory import (
    ActorState,
    EgoState,
    FrameData,
    RAW_FIELDNAMES,
    WindowBuildConfig,
    build_multi_agent_samples,
    read_raw_frames,
    rotate_global_to_ego_forward_y,
    actor_feature_in_anchor_frame,
    sample_to_torch_payload,
)


def _actor(actor_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0) -> ActorState:
    return ActorState(
        actor_id=actor_id,
        actor_type="vehicle.test",
        x=x,
        y=y,
        z=0.0,
        vx=vx,
        vy=vy,
        yaw=yaw,
        distance_m=float(np.hypot(x, y)),
    )


def _frame(index: int, actors: dict[int, ActorState]) -> FrameData:
    return FrameData(
        frame=index,
        timestamp=0.1 * index,
        run_id="test_run",
        town="TownUnit",
        ego=EgoState(
            actor_id=1,
            x=0.0,
            y=0.0,
            z=0.0,
            vx=1.0,
            vy=0.0,
            yaw=0.0,
        ),
        actors=actors,
    )


class MultiAgentTrajectoryTests(unittest.TestCase):

    # ----- rotation function tests -----

    def test_rotation_forward_is_positive_y(self) -> None:
        """When ego faces +X (yaw=0), a point directly ahead (dx=1,dy=0) maps to local +Y."""
        lx, ly = rotate_global_to_ego_forward_y(1.0, 0.0, 0.0)
        self.assertAlmostEqual(lx, 0.0, places=6)
        self.assertAlmostEqual(ly, 1.0, places=6)

    def test_rotation_preserves_distance(self) -> None:
        """Rotation (reflection) must preserve vector norms."""
        for ego_yaw in range(0, 360, 30):
            for dx, dy in [(1, 0), (0, 1), (3, 4), (-2, 7)]:
                lx, ly = rotate_global_to_ego_forward_y(float(dx), float(dy), float(ego_yaw))
                orig = math.hypot(dx, dy)
                rotated = math.hypot(lx, ly)
                self.assertAlmostEqual(orig, rotated, places=6,
                                       msg=f"yaw={ego_yaw} ({dx},{dy})")

    # ----- heading / feature consistency tests -----

    def test_heading_norm_is_unit_vector(self) -> None:
        """features[4]² + features[5]² must always equal 1.0."""
        for ego_yaw in range(0, 360, 30):
            for actor_yaw in range(0, 360, 30):
                ego = EgoState(1, 0, 0, 0, 0, 0, float(ego_yaw))
                actor = _actor(10, x=10.0, y=5.0, vx=1.0, vy=1.0, yaw=float(actor_yaw))
                f = actor_feature_in_anchor_frame(actor, ego)
                ss = f[4] ** 2 + f[5] ** 2
                self.assertAlmostEqual(ss, 1.0, places=5,
                                       msg=f"ego={ego_yaw} actor={actor_yaw} cos²+sin²={ss}")

    def test_heading_matches_velocity_direction(self) -> None:
        """Heading direction must match velocity direction for straight-moving vehicles."""
        for ego_yaw in range(0, 360, 30):
            for actor_yaw in range(0, 360, 30):
                speed = 10.0
                actor_vx = speed * math.cos(math.radians(actor_yaw))
                actor_vy = speed * math.sin(math.radians(actor_yaw))

                ego = EgoState(1, 100.0, 50.0, 0, 0.0, 0.0, float(ego_yaw))
                actor = ActorState(10, "v", 110.0, 60.0, 0, actor_vx, actor_vy,
                                   float(actor_yaw), 14.14)

                f = actor_feature_in_anchor_frame(actor, ego)
                local_vx, local_vy = f[2], f[3]
                heading_x, heading_y = f[4], f[5]

                v_speed = math.hypot(local_vx, local_vy)
                if v_speed < 0.01:
                    continue

                heading_angle = math.atan2(heading_y, heading_x)
                vel_angle = math.atan2(local_vy, local_vx)
                diff = abs(heading_angle - vel_angle)
                if diff > math.pi:
                    diff = 2 * math.pi - diff

                self.assertAlmostEqual(diff, 0.0, places=3,
                                       msg=f"ego={ego_yaw} actor={actor_yaw} "
                                           f"heading_angle={math.degrees(heading_angle):.1f}° "
                                           f"vel_angle={math.degrees(vel_angle):.1f}°")

    # ----- adjacency matrix tests -----

    def test_adjacency_correct_at_40m_radius(self) -> None:
        """Agents within 40m of each other must have edge=1 in adjacency."""
        frames = []
        for frame_idx in range(6):
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=float(frame_idx), y=0.0, vx=2.0, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx), y=35.0, vx=2.0, vy=0.0, yaw=0.0),
                        30: _actor(30, x=float(frame_idx), y=50.0, vx=2.0, vy=0.0, yaw=0.0),
                    },
                )
            )

        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=40.0,
                expected_dt=0.1,
                max_dt_error=1e-6,
            ),
        )
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        adj = sample["adj"]
        # Actor 10 and 20 are 35m apart → edge should exist
        self.assertEqual(adj[0, 1], 1.0, "10↔20 at 35m should have edge")
        self.assertEqual(adj[1, 0], 1.0, "20↔10 at 35m should have edge")
        # Actor 20 and 30 are 15m apart → edge should exist
        self.assertEqual(adj[1, 2], 1.0, "20↔30 at 15m should have edge")

    # ----- teleportation filter tests -----

    def test_teleportation_filter_rejects_impossible_jumps(self) -> None:
        """An agent with a 50m jump between consecutive frames should be filtered."""
        frames = []
        for frame_idx in range(6):
            x_pos = float(frame_idx)
            if frame_idx == 3:
                x_pos = 50.0  # teleportation: jump from ~2 to 50
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=x_pos, y=0.0, vx=1.0, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx), y=5.0, vx=1.0, vy=0.0, yaw=0.0),
                    },
                )
            )

        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=40.0,
                expected_dt=0.1,
                max_dt_error=1e-6,
                max_step_m=6.0,
                min_valid_ratio=0.5,
            ),
        )

        # Samples that include the teleportation frame should either:
        # - have agent 10 filtered out (if jump is within history), or
        # - have agent 10's data zeroed from the jump onward
        for sample in samples:
            x = sample["x"]
            x_mask = sample["x_mask"]
            # For any agent, consecutive valid positions should never jump > 6m
            for agent_idx in range(x.shape[0]):
                for t in range(1, x.shape[1]):
                    if x_mask[agent_idx, t] and x_mask[agent_idx, t - 1]:
                        dx = x[agent_idx, t, 0] - x[agent_idx, t - 1, 0]
                        dy = x[agent_idx, t, 1] - x[agent_idx, t - 1, 1]
                        jump = math.hypot(dx, dy)
                        self.assertLessEqual(
                            jump, 6.0,
                            f"agent={agent_idx} t={t} jump={jump:.1f}m exceeds threshold"
                        )

    # ----- sample building tests -----

    def test_build_samples_shapes_masks_and_adjacency(self) -> None:
        frames = []
        for frame_idx in range(6):
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=float(frame_idx), y=0.0, vx=2.0, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx), y=5.0, vx=2.0, vy=0.0, yaw=90.0),
                    },
                )
            )

        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=6.0,
                expected_dt=0.1,
                max_dt_error=1e-6,
            ),
        )

        self.assertEqual(len(samples), 3)
        sample = samples[0]
        self.assertEqual(tuple(sample["x"].shape), (2, 2, 6))
        self.assertEqual(tuple(sample["y"].shape), (2, 2, 2))
        self.assertEqual(tuple(sample["adj"].shape), (2, 2))
        self.assertTrue(sample["x_mask"].all())
        self.assertTrue(sample["y_mask"].all())
        self.assertTrue(np.allclose(sample["adj"], np.ones((2, 2), dtype=np.float32)))

        # Actor 10 at anchor frame 1: position (1, 0) global, ego at (0,0) yaw=0
        # After rotation: local_x = -sin(0)*1 + cos(0)*0 = 0
        #                 local_y =  cos(0)*1 + sin(0)*0 = 1
        self.assertAlmostEqual(float(sample["x"][0, 1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(sample["x"][0, 1, 1]), 1.0, places=6)
        # Relative velocity is actor vx 2 - ego vx 1 = 1 in global X
        # Rotated: local_vy = cos(0)*1 + sin(0)*0 = 1
        self.assertAlmostEqual(float(sample["x"][0, 1, 3]), 1.0, places=6)
        self.assertAlmostEqual(float(sample["y"][0, 0, 1]), 2.0, places=6)

    def test_read_raw_frames_ignores_sentinel_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "raw.csv"
            rows = [
                {
                    "run_id": "run",
                    "town": "Town03",
                    "frame": "1",
                    "timestamp": "0.1",
                    "ego_id": "1",
                    "ego_x": "0",
                    "ego_y": "0",
                    "ego_z": "0",
                    "ego_vx": "0",
                    "ego_vy": "0",
                    "ego_yaw": "0",
                    "actor_id": "-1",
                    "actor_type": "__none__",
                },
                {
                    "run_id": "run",
                    "town": "Town03",
                    "frame": "2",
                    "timestamp": "0.2",
                    "ego_id": "1",
                    "ego_x": "0",
                    "ego_y": "0",
                    "ego_z": "0",
                    "ego_vx": "0",
                    "ego_vy": "0",
                    "ego_yaw": "0",
                    "actor_id": "7",
                    "actor_type": "vehicle.test",
                    "actor_x": "1",
                    "actor_y": "0",
                    "actor_z": "0",
                    "actor_vx": "0",
                    "actor_vy": "0",
                    "actor_yaw": "0",
                    "distance_m": "1",
                },
            ]
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                import csv

                writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            frames = read_raw_frames(path)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].actors, {})
            self.assertEqual(sorted(frames[1].actors), [7])

    def test_dataset_collate_and_model_forward(self) -> None:
        frames = []
        for frame_idx in range(6):
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=float(frame_idx), y=0.0, vx=2.0, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx), y=5.0, vx=2.0, vy=0.0, yaw=90.0),
                    },
                )
            )
        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(history_frames=2, future_frames=2, stride=1, adjacency_radius_m=6.0),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = root / "manifest.csv"
            for index, sample in enumerate(samples[:2]):
                torch.save(sample_to_torch_payload(sample), root / f"sample_{index:06d}.pt")
            manifest.write_text(
                "sample_file,anchor_frame,anchor_timestamp,town,run_id,num_agents\n"
                "sample_000000.pt,1,0.1,TownUnit,test,2\n"
                "sample_000001.pt,2,0.2,TownUnit,test,2\n",
                encoding="utf-8",
            )

            dataset = MultiAgentTrajectoryDataset(root)
            batch = collate_multi_agent_trajectory([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["x"].shape), (2, 2, 2, 6))
            self.assertEqual(tuple(batch["y"].shape), (2, 2, 2, 2))

            model = MultiAgentTrajectoryPredictor(
                MultiAgentModelConfig(input_dim=6, hidden_dim=16, graph_layers=1, future_steps=2)
            )
            pred = model(
                x=batch["x"],
                adj=batch["adj"],
                x_mask=batch["x_mask"],
                agent_mask=batch["agent_mask"],
            )
            self.assertEqual(tuple(pred.shape), (2, 2, 2, 2))
            loss = masked_smooth_l1_loss(pred, batch["y"], batch["y_mask"], batch["agent_mask"])
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
