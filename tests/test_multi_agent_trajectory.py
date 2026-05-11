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
    def test_rotation_uses_forward_positive_y(self) -> None:
        self.assertAlmostEqual(rotate_global_to_ego_forward_y(1.0, 0.0, 0.0)[0], 0.0)
        self.assertAlmostEqual(rotate_global_to_ego_forward_y(1.0, 0.0, 0.0)[1], 1.0)
        self.assertAlmostEqual(rotate_global_to_ego_forward_y(0.0, 1.0, 90.0)[0], 0.0, places=6)
        self.assertAlmostEqual(rotate_global_to_ego_forward_y(0.0, 1.0, 90.0)[1], 1.0, places=6)

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

        # Actor 10 at anchor frame 1 is 1 m in front of ego when yaw=0.
        self.assertAlmostEqual(float(sample["x"][0, 1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(sample["x"][0, 1, 1]), 1.0, places=6)
        # Relative velocity is actor vx 2 - ego vx 1, rotated to local +Y.
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
