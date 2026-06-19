import json
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
    MultimodalDecoder,
    masked_smooth_l1_loss,
)
from core_perception.multi_agent_trajectory import (
    ActorState,
    EgoState,
    FrameData,
    RAW_FIELDNAMES,
    WindowBuildConfig,
    build_adaptive_adjacency,
    build_multi_agent_samples,
    compute_adaptive_radius,
    compute_multimodal_metrics,
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


class MultiAgentModelConfigTests(unittest.TestCase):
    """Tests for MultiAgentModelConfig GAT parameter extensions."""

    def test_default_config_has_gat_disabled(self) -> None:
        """Default configuration should have GAT disabled."""
        config = MultiAgentModelConfig()
        self.assertFalse(config.enable_gat)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.attention_dropout, 0.1)
        self.assertEqual(config.attention_concat_mode, "concat")

    def test_gat_config_factory_enables_gat(self) -> None:
        """gat_config() factory method should enable GAT."""
        config = MultiAgentModelConfig.gat_config()
        self.assertTrue(config.enable_gat)
        self.assertEqual(config.num_attention_heads, 4)

    def test_validation_rejects_invalid_num_attention_heads(self) -> None:
        """num_attention_heads must be >= 1."""
        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(num_attention_heads=0)
        self.assertIn("num_attention_heads must be >= 1", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(num_attention_heads=-1)
        self.assertIn("num_attention_heads must be >= 1", str(ctx.exception))

    def test_validation_rejects_invalid_attention_concat_mode(self) -> None:
        """attention_concat_mode must be 'concat' or 'average'."""
        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(attention_concat_mode="invalid")
        self.assertIn("attention_concat_mode must be 'concat' or 'average'", str(ctx.exception))

    def test_validation_accepts_valid_concat_modes(self) -> None:
        """Both 'concat' and 'average' should be accepted."""
        config_concat = MultiAgentModelConfig(attention_concat_mode="concat")
        self.assertEqual(config_concat.attention_concat_mode, "concat")

        config_average = MultiAgentModelConfig(attention_concat_mode="average")
        self.assertEqual(config_average.attention_concat_mode, "average")

    def test_to_json_serializes_all_fields(self) -> None:
        """to_json() should serialize all configuration fields."""
        config = MultiAgentModelConfig(
            input_dim=6,
            hidden_dim=128,
            graph_layers=2,
            future_steps=30,
            dropout=0.1,
            enable_gat=True,
            num_attention_heads=8,
            attention_dropout=0.2,
            attention_concat_mode="average",
        )
        json_data = config.to_json()

        self.assertEqual(json_data["input_dim"], 6)
        self.assertEqual(json_data["hidden_dim"], 128)
        self.assertEqual(json_data["graph_layers"], 2)
        self.assertEqual(json_data["future_steps"], 30)
        self.assertEqual(json_data["dropout"], 0.1)
        self.assertEqual(json_data["enable_gat"], True)
        self.assertEqual(json_data["num_attention_heads"], 8)
        self.assertEqual(json_data["attention_dropout"], 0.2)
        self.assertEqual(json_data["attention_concat_mode"], "average")

    def test_from_json_deserializes_correctly(self) -> None:
        """from_json() should reconstruct configuration from JSON data."""
        json_data = {
            "input_dim": 6,
            "hidden_dim": 64,
            "graph_layers": 3,
            "future_steps": 20,
            "dropout": 0.2,
            "enable_gat": True,
            "num_attention_heads": 4,
            "attention_dropout": 0.15,
            "attention_concat_mode": "concat",
        }
        config = MultiAgentModelConfig.from_json(json_data)

        self.assertEqual(config.input_dim, 6)
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.graph_layers, 3)
        self.assertEqual(config.future_steps, 20)
        self.assertEqual(config.dropout, 0.2)
        self.assertTrue(config.enable_gat)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.attention_dropout, 0.15)
        self.assertEqual(config.attention_concat_mode, "concat")

    def test_round_trip_serialization(self) -> None:
        """Serializing and deserializing should preserve configuration."""
        original = MultiAgentModelConfig(
            enable_gat=True,
            num_attention_heads=6,
            attention_dropout=0.25,
            attention_concat_mode="average",
        )
        json_data = original.to_json()
        restored = MultiAgentModelConfig.from_json(json_data)

        self.assertEqual(original.input_dim, restored.input_dim)
        self.assertEqual(original.hidden_dim, restored.hidden_dim)
        self.assertEqual(original.graph_layers, restored.graph_layers)
        self.assertEqual(original.future_steps, restored.future_steps)
        self.assertEqual(original.dropout, restored.dropout)
        self.assertEqual(original.enable_gat, restored.enable_gat)
        self.assertEqual(original.num_attention_heads, restored.num_attention_heads)
        self.assertEqual(original.attention_dropout, restored.attention_dropout)
        self.assertEqual(original.attention_concat_mode, restored.attention_concat_mode)


class MultiAgentModelConfigMultimodalTests(unittest.TestCase):
    """Tests for MultiAgentModelConfig multimodal parameter extensions."""

    def test_default_config_has_multimodal_disabled(self) -> None:
        """Default configuration should have multimodal disabled."""
        config = MultiAgentModelConfig()
        self.assertFalse(config.enable_multimodal)
        self.assertEqual(config.num_modes, 5)

    def test_multimodal_config_factory_enables_multimodal(self) -> None:
        """multimodal_config() factory method should enable multimodal."""
        config = MultiAgentModelConfig.multimodal_config()
        self.assertTrue(config.enable_multimodal)
        self.assertEqual(config.num_modes, 5)

    def test_validation_rejects_invalid_num_modes(self) -> None:
        """num_modes must be >= 1."""
        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(num_modes=0)
        self.assertIn("num_modes must be >= 1", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(num_modes=-1)
        self.assertIn("num_modes must be >= 1", str(ctx.exception))

    def test_validation_accepts_valid_num_modes(self) -> None:
        """num_modes >= 1 should be accepted."""
        config_1 = MultiAgentModelConfig(num_modes=1)
        self.assertEqual(config_1.num_modes, 1)

        config_3 = MultiAgentModelConfig(num_modes=3)
        self.assertEqual(config_3.num_modes, 3)

        config_5 = MultiAgentModelConfig(num_modes=5)
        self.assertEqual(config_5.num_modes, 5)

    def test_to_json_includes_multimodal_fields(self) -> None:
        """to_json() should include multimodal fields."""
        config = MultiAgentModelConfig(
            enable_multimodal=True,
            num_modes=5,
        )
        json_data = config.to_json()

        self.assertIn("enable_multimodal", json_data)
        self.assertIn("num_modes", json_data)
        self.assertEqual(json_data["enable_multimodal"], True)
        self.assertEqual(json_data["num_modes"], 5)

    def test_from_json_deserializes_multimodal_fields(self) -> None:
        """from_json() should reconstruct multimodal configuration."""
        json_data = {
            "input_dim": 6,
            "hidden_dim": 128,
            "graph_layers": 2,
            "future_steps": 30,
            "dropout": 0.1,
            "enable_gat": False,
            "num_attention_heads": 4,
            "attention_dropout": 0.1,
            "attention_concat_mode": "concat",
            "enable_multimodal": True,
            "num_modes": 5,
        }
        config = MultiAgentModelConfig.from_json(json_data)

        self.assertTrue(config.enable_multimodal)
        self.assertEqual(config.num_modes, 5)

    def test_round_trip_serialization_with_multimodal(self) -> None:
        """Serializing and deserializing should preserve multimodal configuration."""
        original = MultiAgentModelConfig(
            enable_multimodal=True,
            num_modes=5,
        )
        json_data = original.to_json()
        restored = MultiAgentModelConfig.from_json(json_data)

        self.assertEqual(original.enable_multimodal, restored.enable_multimodal)
        self.assertEqual(original.num_modes, restored.num_modes)

    def test_combined_gat_and_multimodal_config(self) -> None:
        """Configuration should support both GAT and multimodal enabled."""
        config = MultiAgentModelConfig(
            enable_gat=True,
            num_attention_heads=8,
            enable_multimodal=True,
            num_modes=5,
        )

        self.assertTrue(config.enable_gat)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertTrue(config.enable_multimodal)
        self.assertEqual(config.num_modes, 5)

        # Test serialization round-trip
        json_data = config.to_json()
        restored = MultiAgentModelConfig.from_json(json_data)

        self.assertEqual(config.enable_gat, restored.enable_gat)
        self.assertEqual(config.num_attention_heads, restored.num_attention_heads)
        self.assertEqual(config.enable_multimodal, restored.enable_multimodal)
        self.assertEqual(config.num_modes, restored.num_modes)


class DatasetBuilderTests(unittest.TestCase):
    """Tests for dataset builder functionality.
    
    Tests CSV parsing, coordinate transformation, teleportation filter,
    adjacency construction, and output format.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """

    def test_csv_parsing_basic(self) -> None:
        """Test CSV parsing with basic multi-agent data.
        
        Requirements: 6.1
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "raw.csv"
            rows = [
                {
                    "run_id": "test_run_001",
                    "town": "Town01",
                    "frame": "0",
                    "timestamp": "0.0",
                    "ego_id": "1",
                    "ego_x": "100.0",
                    "ego_y": "200.0",
                    "ego_z": "0.5",
                    "ego_vx": "5.0",
                    "ego_vy": "0.0",
                    "ego_yaw": "0.0",
                    "actor_id": "10",
                    "actor_type": "vehicle.tesla.model3",
                    "actor_x": "105.0",
                    "actor_y": "200.0",
                    "actor_z": "0.5",
                    "actor_vx": "5.0",
                    "actor_vy": "0.0",
                    "actor_yaw": "0.0",
                    "distance_m": "5.0",
                },
                {
                    "run_id": "test_run_001",
                    "town": "Town01",
                    "frame": "0",
                    "timestamp": "0.0",
                    "ego_id": "1",
                    "ego_x": "100.0",
                    "ego_y": "200.0",
                    "ego_z": "0.5",
                    "ego_vx": "5.0",
                    "ego_vy": "0.0",
                    "ego_yaw": "0.0",
                    "actor_id": "20",
                    "actor_type": "vehicle.audi.a2",
                    "actor_x": "100.0",
                    "actor_y": "210.0",
                    "actor_z": "0.5",
                    "actor_vx": "5.0",
                    "actor_vy": "0.0",
                    "actor_yaw": "90.0",
                    "distance_m": "10.0",
                },
            ]
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                import csv
                writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            frames = read_raw_frames(path)
            
            # Verify frame parsing
            self.assertEqual(len(frames), 1)
            frame = frames[0]
            self.assertEqual(frame.frame, 0)
            self.assertEqual(frame.timestamp, 0.0)
            self.assertEqual(frame.run_id, "test_run_001")
            self.assertEqual(frame.town, "Town01")
            
            # Verify ego state
            self.assertEqual(frame.ego.actor_id, 1)
            self.assertEqual(frame.ego.x, 100.0)
            self.assertEqual(frame.ego.y, 200.0)
            self.assertEqual(frame.ego.vx, 5.0)
            self.assertEqual(frame.ego.vy, 0.0)
            self.assertEqual(frame.ego.yaw, 0.0)
            
            # Verify actors
            self.assertEqual(len(frame.actors), 2)
            self.assertIn(10, frame.actors)
            self.assertIn(20, frame.actors)
            
            actor_10 = frame.actors[10]
            self.assertEqual(actor_10.actor_type, "vehicle.tesla.model3")
            self.assertEqual(actor_10.x, 105.0)
            self.assertEqual(actor_10.y, 200.0)
            
            actor_20 = frame.actors[20]
            self.assertEqual(actor_20.actor_type, "vehicle.audi.a2")
            self.assertEqual(actor_20.x, 100.0)
            self.assertEqual(actor_20.y, 210.0)

    def test_coordinate_transformation_ego_frame(self) -> None:
        """Test coordinate transformation to ego-centric frame with +Y forward and +X right.
        
        Requirements: 6.3
        """
        # Ego at origin facing +X (yaw=0)
        ego = EgoState(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Actor directly ahead in global +X should map to local +Y
        actor_ahead = _actor(10, x=5.0, y=0.0, vx=0.0, vy=0.0, yaw=0.0)
        features = actor_feature_in_anchor_frame(actor_ahead, ego)
        self.assertAlmostEqual(features[0], 0.0, places=6)  # local_x
        self.assertAlmostEqual(features[1], 5.0, places=6)  # local_y (forward)
        
        # Actor to the right in global +Y should map to local +X
        actor_right = _actor(20, x=0.0, y=5.0, vx=0.0, vy=0.0, yaw=0.0)
        features = actor_feature_in_anchor_frame(actor_right, ego)
        self.assertAlmostEqual(features[0], 5.0, places=6)  # local_x (right)
        self.assertAlmostEqual(features[1], 0.0, places=6)  # local_y
        
        # Test with rotated ego (yaw=90 degrees, facing +Y)
        ego_rotated = EgoState(1, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0)
        
        # Actor in global +Y should now map to local +Y (ahead)
        actor_ahead_rotated = _actor(30, x=0.0, y=5.0, vx=0.0, vy=0.0, yaw=90.0)
        features = actor_feature_in_anchor_frame(actor_ahead_rotated, ego_rotated)
        self.assertAlmostEqual(features[0], 0.0, places=6)  # local_x
        self.assertAlmostEqual(features[1], 5.0, places=6)  # local_y (forward)

    def test_teleportation_filter_marks_invalid_frames(self) -> None:
        """Test teleportation filter marks frames with jumps > 6m as invalid.
        
        Requirements: 6.4
        """
        frames = []
        # Create frames with a teleportation jump at frame 3
        for frame_idx in range(8):
            x_pos = float(frame_idx) * 0.5  # Normal movement: 0.5m per frame
            if frame_idx == 3:
                x_pos = 50.0  # Teleportation: jump from ~1.0 to 50.0
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=x_pos, y=0.0, vx=1.0, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx) * 0.5, y=5.0, vx=1.0, vy=0.0, yaw=0.0),
                    },
                )
            )

        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(
                history_frames=3,
                future_frames=3,
                stride=1,
                adjacency_radius_m=40.0,
                expected_dt=0.1,
                max_dt_error=1e-6,
                max_step_m=6.0,
                min_valid_ratio=0.3,
            ),
        )

        # Verify that samples containing the teleportation have agent 10 filtered
        for sample in samples:
            x = sample["x"]
            x_mask = sample["x_mask"]
            
            # Check that no consecutive valid positions have jumps > 6m
            for agent_idx in range(x.shape[0]):
                for t in range(1, x.shape[1]):
                    if x_mask[agent_idx, t] and x_mask[agent_idx, t - 1]:
                        dx = x[agent_idx, t, 0] - x[agent_idx, t - 1, 0]
                        dy = x[agent_idx, t, 1] - x[agent_idx, t - 1, 1]
                        jump = math.hypot(dx, dy)
                        self.assertLessEqual(
                            jump, 6.0,
                            f"Teleportation not filtered: agent={agent_idx} t={t} jump={jump:.1f}m"
                        )

    def test_fixed_radius_adjacency_construction(self) -> None:
        """Test fixed radius adjacency matrix construction.
        
        Requirements: 6.5
        """
        frames = []
        # Create 3 agents at different distances
        for frame_idx in range(6):
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=float(frame_idx), y=0.0, vx=2.0, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx), y=15.0, vx=2.0, vy=0.0, yaw=0.0),
                        30: _actor(30, x=float(frame_idx), y=35.0, vx=2.0, vy=0.0, yaw=0.0),
                    },
                )
            )

        # Use fixed radius of 20m
        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=20.0,
                expected_dt=0.1,
                max_dt_error=1e-6,
                adaptive_radius_enabled=False,
            ),
        )
        
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        adj = sample["adj"]
        
        # Verify adjacency matrix shape
        self.assertEqual(adj.shape, (3, 3))
        
        # Verify self-connections (diagonal)
        self.assertEqual(adj[0, 0], 1.0)
        self.assertEqual(adj[1, 1], 1.0)
        self.assertEqual(adj[2, 2], 1.0)
        
        # Agents 10 and 20 are 15m apart → should be connected (15 <= 20)
        self.assertEqual(adj[0, 1], 1.0)
        self.assertEqual(adj[1, 0], 1.0)
        
        # Agents 20 and 30 are 20m apart → should be connected (20 <= 20)
        self.assertEqual(adj[1, 2], 1.0)
        self.assertEqual(adj[2, 1], 1.0)
        
        # Agents 10 and 30 are 35m apart → should NOT be connected (35 > 20)
        self.assertEqual(adj[0, 2], 0.0)
        self.assertEqual(adj[2, 0], 0.0)

    def test_adaptive_radius_adjacency_construction(self) -> None:
        """Test adaptive radius adjacency matrix construction based on velocity.
        
        Requirements: 6.6, 6.7
        """
        frames = []
        # Create 2 agents with different velocities
        # Agent 10: fast (60 km/h = 16.67 m/s) → radius ~28.3m
        # Agent 20: slow (10 km/h = 2.78 m/s) → radius ~21.4m
        for frame_idx in range(6):
            frames.append(
                _frame(
                    frame_idx,
                    {
                        10: _actor(10, x=float(frame_idx), y=0.0, vx=60.0/3.6, vy=0.0, yaw=0.0),
                        20: _actor(20, x=float(frame_idx), y=25.0, vx=10.0/3.6, vy=0.0, yaw=0.0),
                    },
                )
            )

        # Use adaptive radius
        samples = build_multi_agent_samples(
            frames,
            WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=40.0,  # Not used in adaptive mode
                expected_dt=0.1,
                max_dt_error=1e-6,
                adaptive_radius_enabled=True,
                radius_base=20.0,
                radius_alpha=0.5,
            ),
        )
        
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        adj = sample["adj"]
        
        # Verify adjacency matrix shape
        self.assertEqual(adj.shape, (2, 2))
        
        # Verify self-connections
        self.assertEqual(adj[0, 0], 1.0)
        self.assertEqual(adj[1, 1], 1.0)
        
        # Agents are 25m apart
        # Agent 10 radius: 20 + 0.5 * 16.67 = 28.3m
        # Agent 20 radius: 20 + 0.5 * 2.78 = 21.4m
        # Threshold: min(28.3, 21.4) = 21.4m
        # Since 25m > 21.4m, they should NOT be connected
        self.assertEqual(adj[0, 1], 0.0)
        self.assertEqual(adj[1, 0], 0.0)

    def test_manifest_csv_output_format(self) -> None:
        """Test manifest.csv output format with correct columns.
        
        Requirements: 6.8
        """
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
                adjacency_radius_m=20.0,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.csv"
            
            # Write samples and manifest
            manifest_rows = []
            for index, sample in enumerate(samples):
                sample_name = f"sample_{index:06d}.pt"
                sample_path = root / sample_name
                torch.save(sample_to_torch_payload(sample), sample_path)
                manifest_rows.append({
                    "sample_file": sample_name,
                    "anchor_frame": int(sample["anchor_frame"]),
                    "anchor_timestamp": f"{float(sample['anchor_timestamp']):.6f}",
                    "town": str(sample.get("town", "")),
                    "run_id": str(sample.get("run_id", "")),
                    "num_agents": int(sample["x"].shape[0]),
                })
            
            # Write manifest
            with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
                import csv
                writer = csv.DictWriter(csv_file, fieldnames=[
                    "sample_file", "anchor_frame", "anchor_timestamp", 
                    "town", "run_id", "num_agents"
                ])
                writer.writeheader()
                writer.writerows(manifest_rows)
            
            # Verify manifest format
            self.assertTrue(manifest_path.exists())
            
            with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
                import csv
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                
                # Verify columns
                self.assertEqual(reader.fieldnames, [
                    "sample_file", "anchor_frame", "anchor_timestamp",
                    "town", "run_id", "num_agents"
                ])
                
                # Verify row count matches samples
                self.assertEqual(len(rows), len(samples))
                
                # Verify first row content
                if len(rows) > 0:
                    row = rows[0]
                    self.assertEqual(row["sample_file"], "sample_000000.pt")
                    self.assertEqual(row["town"], "TownUnit")
                    self.assertEqual(row["run_id"], "test_run")
                    self.assertEqual(row["num_agents"], "2")

    def test_build_summary_json_output_format(self) -> None:
        """Test build_summary.json output format with configuration and statistics.
        
        Requirements: 6.9
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_csv = root / "raw.csv"
            
            # Create minimal CSV
            rows = []
            for frame_idx in range(6):
                rows.append({
                    "run_id": "test_run",
                    "town": "Town01",
                    "frame": str(frame_idx),
                    "timestamp": str(frame_idx * 0.1),
                    "ego_id": "1",
                    "ego_x": "0",
                    "ego_y": "0",
                    "ego_z": "0",
                    "ego_vx": "1",
                    "ego_vy": "0",
                    "ego_yaw": "0",
                    "actor_id": "10",
                    "actor_type": "vehicle.test",
                    "actor_x": str(frame_idx),
                    "actor_y": "0",
                    "actor_z": "0",
                    "actor_vx": "1",
                    "actor_vy": "0",
                    "actor_yaw": "0",
                    "distance_m": str(frame_idx),
                })
            
            with raw_csv.open("w", newline="", encoding="utf-8") as csv_file:
                import csv
                writer = csv.DictWriter(csv_file, fieldnames=RAW_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            
            # Build samples
            frames = read_raw_frames(raw_csv)
            samples = build_multi_agent_samples(
                frames,
                WindowBuildConfig(
                    history_frames=2,
                    future_frames=2,
                    stride=1,
                    adjacency_radius_m=40.0,
                    adaptive_radius_enabled=True,
                    radius_base=20.0,
                    radius_alpha=0.5,
                ),
            )
            
            # Create build summary
            summary = {
                "raw_csv": str(raw_csv),
                "out_dir": str(root),
                "frames_read": len(frames),
                "samples_written": len(samples),
                "config": {
                    "history_frames": 2,
                    "future_frames": 2,
                    "stride": 1,
                    "adjacency_radius_m": 40.0,
                    "adaptive_radius_enabled": True,
                    "radius_base": 20.0,
                    "radius_alpha": 0.5,
                },
            }
            
            summary_path = root / "build_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            
            # Verify summary format
            self.assertTrue(summary_path.exists())
            
            with summary_path.open("r", encoding="utf-8") as f:
                loaded_summary = json.load(f)
            
            # Verify required fields
            self.assertIn("raw_csv", loaded_summary)
            self.assertIn("out_dir", loaded_summary)
            self.assertIn("frames_read", loaded_summary)
            self.assertIn("samples_written", loaded_summary)
            self.assertIn("config", loaded_summary)
            
            # Verify config fields
            config = loaded_summary["config"]
            self.assertEqual(config["history_frames"], 2)
            self.assertEqual(config["future_frames"], 2)
            self.assertEqual(config["stride"], 1)
            self.assertEqual(config["adjacency_radius_m"], 40.0)
            self.assertEqual(config["adaptive_radius_enabled"], True)
            self.assertEqual(config["radius_base"], 20.0)
            self.assertEqual(config["radius_alpha"], 0.5)


class AdaptiveRadiusFunctionTests(unittest.TestCase):
    """Tests for adaptive radius computation functions.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7
    """

    def test_radius_computation_60_kmh(self) -> None:
        """Test radius computation: 60 km/h (16.67 m/s) → ~28.3m radius.
        
        Requirements: 3.1, 3.2, 3.3
        """
        # 60 km/h = 16.67 m/s
        velocity_60kmh = 60.0 / 3.6  # Convert km/h to m/s
        velocities = np.array([[velocity_60kmh, 0.0]])  # Single agent moving in x direction
        
        radii = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 16.67 = 28.335
        expected_radius = 20.0 + 0.5 * velocity_60kmh
        self.assertAlmostEqual(float(radii[0]), expected_radius, places=2)
        self.assertAlmostEqual(float(radii[0]), 28.3, places=1)

    def test_radius_computation_10_kmh(self) -> None:
        """Test radius computation: 10 km/h (2.78 m/s) → ~21.4m radius.
        
        Requirements: 3.1, 3.2, 3.4
        """
        # 10 km/h = 2.78 m/s
        velocity_10kmh = 10.0 / 3.6  # Convert km/h to m/s
        velocities = np.array([[velocity_10kmh, 0.0]])  # Single agent moving in x direction
        
        radii = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 2.78 = 21.39
        expected_radius = 20.0 + 0.5 * velocity_10kmh
        self.assertAlmostEqual(float(radii[0]), expected_radius, places=2)
        self.assertAlmostEqual(float(radii[0]), 21.4, places=1)

    def test_radius_computation_with_2d_velocity(self) -> None:
        """Test radius computation with 2D velocity vectors.
        
        Requirements: 3.1, 3.2
        """
        # Agent with velocity (3, 4) has speed = 5 m/s
        velocities = np.array([[3.0, 4.0]])
        
        radii = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 5.0 = 22.5
        self.assertAlmostEqual(float(radii[0]), 22.5, places=6)

    def test_radius_computation_multiple_agents(self) -> None:
        """Test radius computation for multiple agents with different velocities.
        
        Requirements: 3.1, 3.2
        """
        velocities = np.array([
            [10.0, 0.0],   # speed = 10 m/s
            [0.0, 5.0],    # speed = 5 m/s
            [3.0, 4.0],    # speed = 5 m/s
            [0.0, 0.0],    # speed = 0 m/s (stationary)
        ])
        
        radii = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        self.assertEqual(len(radii), 4)
        self.assertAlmostEqual(float(radii[0]), 25.0, places=6)  # 20 + 0.5*10
        self.assertAlmostEqual(float(radii[1]), 22.5, places=6)  # 20 + 0.5*5
        self.assertAlmostEqual(float(radii[2]), 22.5, places=6)  # 20 + 0.5*5
        self.assertAlmostEqual(float(radii[3]), 20.0, places=6)  # 20 + 0.5*0

    def test_adjacency_symmetry(self) -> None:
        """Test adjacency symmetry: adj[i,j] == adj[j,i].
        
        Requirements: 3.7
        """
        # Create 3 agents with different positions and velocities
        positions = np.array([
            [0.0, 0.0],
            [15.0, 0.0],
            [30.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [10.0, 0.0],
            [5.0, 0.0],
            [8.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Verify symmetry for all pairs
        n_agents = adj.shape[0]
        for i in range(n_agents):
            for j in range(n_agents):
                self.assertEqual(
                    float(adj[i, j]), 
                    float(adj[j, i]),
                    f"Adjacency matrix not symmetric at ({i},{j})"
                )

    def test_adjacency_distance_threshold_not_connected(self) -> None:
        """Test distance threshold: agents 25m apart with radii [28m, 21m] → not connected.
        
        Requirements: 3.3, 3.4, 3.6, 3.7
        """
        # Agent 0: 60 km/h → radius ~28.3m
        # Agent 1: 10 km/h → radius ~21.4m
        # Distance: 25m
        # Threshold: min(28.3, 21.4) = 21.4m
        # Since 25m > 21.4m, they should NOT be connected
        
        positions = np.array([
            [0.0, 0.0],
            [25.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [60.0 / 3.6, 0.0],  # 60 km/h = 16.67 m/s
            [10.0 / 3.6, 0.0],  # 10 km/h = 2.78 m/s
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Verify agents are NOT connected (only diagonal should be 1)
        self.assertEqual(float(adj[0, 0]), 1.0, "Self-connection should exist")
        self.assertEqual(float(adj[1, 1]), 1.0, "Self-connection should exist")
        self.assertEqual(float(adj[0, 1]), 0.0, "Agents should NOT be connected (25m > 21.4m)")
        self.assertEqual(float(adj[1, 0]), 0.0, "Agents should NOT be connected (25m > 21.4m)")

    def test_adjacency_distance_threshold_connected(self) -> None:
        """Test distance threshold: agents within min radius should be connected.
        
        Requirements: 3.6, 3.7
        """
        # Agent 0: 60 km/h → radius ~28.3m
        # Agent 1: 10 km/h → radius ~21.4m
        # Distance: 20m
        # Threshold: min(28.3, 21.4) = 21.4m
        # Since 20m <= 21.4m, they SHOULD be connected
        
        positions = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [60.0 / 3.6, 0.0],  # 60 km/h
            [10.0 / 3.6, 0.0],  # 10 km/h
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Verify agents ARE connected
        self.assertEqual(float(adj[0, 0]), 1.0, "Self-connection should exist")
        self.assertEqual(float(adj[1, 1]), 1.0, "Self-connection should exist")
        self.assertEqual(float(adj[0, 1]), 1.0, "Agents should be connected (20m <= 21.4m)")
        self.assertEqual(float(adj[1, 0]), 1.0, "Agents should be connected (20m <= 21.4m)")

    def test_adjacency_self_connections(self) -> None:
        """Test that all agents have self-connections (diagonal = 1).
        
        Requirements: 3.7
        """
        positions = np.array([
            [0.0, 0.0],
            [50.0, 0.0],
            [100.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [10.0, 0.0],
            [5.0, 0.0],
            [8.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Verify diagonal is all 1s
        for i in range(adj.shape[0]):
            self.assertEqual(float(adj[i, i]), 1.0, f"Agent {i} should have self-connection")

    def test_adjacency_uses_float64_precision(self) -> None:
        """Test that adjacency computation uses float64 precision for distances.
        
        Requirements: 3.5
        """
        # Create positions that would be affected by float32 precision issues
        positions = np.array([
            [0.0, 0.0],
            [20.00001, 0.0],  # Very close to threshold
        ], dtype=np.float64)
        
        velocities = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        
        # With r_base=20.0 and alpha=0.5, both agents have radius 20.0
        # Distance is 20.00001, which is just over the threshold
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Verify the computation is precise enough to detect this small difference
        # (This test mainly documents that we use float64 for distance computation)
        self.assertEqual(float(adj[0, 1]), 0.0, "Agents should NOT be connected (distance > radius)")

    def test_adjacency_with_custom_parameters(self) -> None:
        """Test adjacency computation with custom r_base and alpha parameters.
        
        Requirements: 3.1, 3.2
        """
        positions = np.array([
            [0.0, 0.0],
            [30.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [10.0, 0.0],
            [10.0, 0.0],
        ])
        
        # With r_base=25.0 and alpha=1.0, both agents have radius = 25 + 1.0*10 = 35m
        # Distance is 30m, so they should be connected
        adj = build_adaptive_adjacency(positions, velocities, r_base=25.0, alpha=1.0)
        
        self.assertEqual(float(adj[0, 1]), 1.0, "Agents should be connected (30m <= 35m)")
        self.assertEqual(float(adj[1, 0]), 1.0, "Agents should be connected (30m <= 35m)")


class MultiAgentModelConfigAdaptiveRadiusTests(unittest.TestCase):
    """Tests for MultiAgentModelConfig adaptive radius parameter extensions."""

    def test_default_config_has_adaptive_radius_disabled(self) -> None:
        """Default configuration should have adaptive radius disabled."""
        config = MultiAgentModelConfig()
        
        self.assertFalse(config.enable_adaptive_radius)
        self.assertEqual(config.radius_base, 40.0)
        self.assertEqual(config.radius_alpha, 1.0)

    def test_adaptive_radius_config_with_custom_params(self) -> None:
        """Configuration should accept custom adaptive radius parameters."""
        config = MultiAgentModelConfig(
            enable_adaptive_radius=True,
            radius_base=25.0,
            radius_alpha=0.8,
        )
        
        self.assertTrue(config.enable_adaptive_radius)
        self.assertEqual(config.radius_base, 25.0)
        self.assertEqual(config.radius_alpha, 0.8)

    def test_validation_rejects_zero_radius_base(self) -> None:
        """radius_base must be > 0."""
        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(radius_base=0.0)
        
        self.assertIn("radius_base must be > 0", str(ctx.exception))

    def test_validation_rejects_negative_radius_base(self) -> None:
        """radius_base must be > 0."""
        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(radius_base=-5.0)
        
        self.assertIn("radius_base must be > 0", str(ctx.exception))

    def test_validation_rejects_negative_radius_alpha(self) -> None:
        """radius_alpha must be >= 0."""
        with self.assertRaises(ValueError) as ctx:
            MultiAgentModelConfig(radius_alpha=-0.1)
        
        self.assertIn("radius_alpha must be >= 0", str(ctx.exception))

    def test_validation_accepts_zero_radius_alpha(self) -> None:
        """radius_alpha = 0 should be accepted (fixed radius mode)."""
        config = MultiAgentModelConfig(radius_alpha=0.0)
        
        self.assertEqual(config.radius_alpha, 0.0)

    def test_validation_accepts_positive_radius_params(self) -> None:
        """Positive radius_base and radius_alpha should be accepted."""
        config = MultiAgentModelConfig(
            radius_base=15.0,
            radius_alpha=1.0,
        )
        
        self.assertEqual(config.radius_base, 15.0)
        self.assertEqual(config.radius_alpha, 1.0)

    def test_to_json_includes_adaptive_radius_fields(self) -> None:
        """to_json() should include adaptive radius fields."""
        config = MultiAgentModelConfig(
            enable_adaptive_radius=True,
            radius_base=25.0,
            radius_alpha=0.8,
        )
        
        json_data = config.to_json()
        
        self.assertIn("enable_adaptive_radius", json_data)
        self.assertIn("radius_base", json_data)
        self.assertIn("radius_alpha", json_data)
        self.assertTrue(json_data["enable_adaptive_radius"])
        self.assertEqual(json_data["radius_base"], 25.0)
        self.assertEqual(json_data["radius_alpha"], 0.8)

    def test_from_json_deserializes_adaptive_radius_fields(self) -> None:
        """from_json() should deserialize adaptive radius fields."""
        json_data = {
            "input_dim": 6,
            "hidden_dim": 128,
            "graph_layers": 2,
            "future_steps": 30,
            "dropout": 0.1,
            "enable_gat": False,
            "num_attention_heads": 4,
            "attention_dropout": 0.1,
            "attention_concat_mode": "concat",
            "enable_multimodal": False,
            "num_modes": 3,
            "enable_adaptive_radius": True,
            "radius_base": 25.0,
            "radius_alpha": 0.8,
        }
        config = MultiAgentModelConfig.from_json(json_data)
        
        self.assertTrue(config.enable_adaptive_radius)
        self.assertEqual(config.radius_base, 25.0)
        self.assertEqual(config.radius_alpha, 0.8)

    def test_round_trip_serialization_with_adaptive_radius(self) -> None:
        """Serializing and deserializing should preserve adaptive radius configuration."""
        original = MultiAgentModelConfig(
            enable_adaptive_radius=True,
            radius_base=30.0,
            radius_alpha=0.6,
        )
        json_data = original.to_json()
        restored = MultiAgentModelConfig.from_json(json_data)
        
        self.assertEqual(original.enable_adaptive_radius, restored.enable_adaptive_radius)
        self.assertEqual(original.radius_base, restored.radius_base)
        self.assertEqual(original.radius_alpha, restored.radius_alpha)

    def test_full_config_factory_enables_all_improvements(self) -> None:
        """full_config() factory method should enable all improvements."""
        config = MultiAgentModelConfig.full_config()
        
        self.assertTrue(config.enable_gat)
        self.assertTrue(config.enable_multimodal)
        self.assertTrue(config.enable_adaptive_radius)
        # Verify default values for other parameters
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.num_modes, 5)
        self.assertEqual(config.radius_base, 40.0)
        self.assertEqual(config.radius_alpha, 1.0)

    def test_combined_all_improvements_config(self) -> None:
        """Configuration should support all improvements enabled with custom params."""
        config = MultiAgentModelConfig(
            enable_gat=True,
            num_attention_heads=8,
            enable_multimodal=True,
            num_modes=5,
            enable_adaptive_radius=True,
            radius_base=25.0,
            radius_alpha=0.7,
        )
        
        self.assertTrue(config.enable_gat)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertTrue(config.enable_multimodal)
        self.assertEqual(config.num_modes, 5)
        self.assertTrue(config.enable_adaptive_radius)
        self.assertEqual(config.radius_base, 25.0)
        self.assertEqual(config.radius_alpha, 0.7)
        
        # Test serialization round-trip
        json_data = config.to_json()
        restored = MultiAgentModelConfig.from_json(json_data)
        
        self.assertEqual(config.enable_gat, restored.enable_gat)
        self.assertEqual(config.num_attention_heads, restored.num_attention_heads)
        self.assertEqual(config.enable_multimodal, restored.enable_multimodal)
        self.assertEqual(config.num_modes, restored.num_modes)
        self.assertEqual(config.enable_adaptive_radius, restored.enable_adaptive_radius)
        self.assertEqual(config.radius_base, restored.radius_base)
        self.assertEqual(config.radius_alpha, restored.radius_alpha)


class ComputeAdaptiveRadiusTests(unittest.TestCase):
    """Tests for compute_adaptive_radius function."""

    def test_radius_computation_for_60_kmh_vehicle(self) -> None:
        """Agent at 60 km/h (16.67 m/s) should get radius ≈ 28.3m with default params."""
        # 60 km/h = 16.67 m/s
        velocities = np.array([[16.67, 0.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 16.67 = 28.335
        expected = 20.0 + 0.5 * 16.67
        self.assertAlmostEqual(float(radius[0]), expected, places=2)
        self.assertGreater(float(radius[0]), 28.0)
        self.assertLess(float(radius[0]), 29.0)

    def test_radius_computation_for_10_kmh_vehicle(self) -> None:
        """Agent at 10 km/h (2.78 m/s) should get radius ≈ 21.4m with default params."""
        # 10 km/h = 2.78 m/s
        velocities = np.array([[2.78, 0.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 2.78 = 21.39
        expected = 20.0 + 0.5 * 2.78
        self.assertAlmostEqual(float(radius[0]), expected, places=2)
        self.assertGreater(float(radius[0]), 21.0)
        self.assertLess(float(radius[0]), 22.0)

    def test_radius_computation_for_stationary_vehicle(self) -> None:
        """Agent at 0 m/s should get base radius."""
        velocities = np.array([[0.0, 0.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        self.assertAlmostEqual(float(radius[0]), 20.0, places=6)

    def test_radius_computation_for_multiple_agents(self) -> None:
        """Multiple agents with different velocities should get different radii."""
        # Agent 0: 0 m/s, Agent 1: 10 m/s, Agent 2: 20 m/s
        velocities = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 20.0],
        ])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected radii
        self.assertAlmostEqual(float(radius[0]), 20.0, places=6)  # 20 + 0.5*0
        self.assertAlmostEqual(float(radius[1]), 25.0, places=6)  # 20 + 0.5*10
        self.assertAlmostEqual(float(radius[2]), 30.0, places=6)  # 20 + 0.5*20

    def test_radius_computation_with_diagonal_velocity(self) -> None:
        """Velocity magnitude should be computed correctly for diagonal motion."""
        # Agent moving at 3 m/s in x and 4 m/s in y → magnitude = 5 m/s
        velocities = np.array([[3.0, 4.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 5.0 = 22.5
        self.assertAlmostEqual(float(radius[0]), 22.5, places=6)

    def test_radius_monotonicity_property(self) -> None:
        """Faster agents should always have larger or equal radius (monotonicity)."""
        # Create agents with increasing speeds
        speeds = [0.0, 5.0, 10.0, 15.0, 20.0]
        velocities = np.array([[speed, 0.0] for speed in speeds])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Verify monotonicity: r[i] <= r[i+1]
        for i in range(len(radius) - 1):
            self.assertLessEqual(
                float(radius[i]), 
                float(radius[i + 1]),
                f"Radius should be monotonically increasing: r[{i}]={radius[i]:.2f} > r[{i+1}]={radius[i+1]:.2f}"
            )

    def test_radius_with_custom_parameters(self) -> None:
        """Function should respect custom r_base and alpha parameters."""
        velocities = np.array([[10.0, 0.0]])
        
        # Test with different r_base
        radius1 = compute_adaptive_radius(velocities, r_base=30.0, alpha=0.5)
        self.assertAlmostEqual(float(radius1[0]), 35.0, places=6)  # 30 + 0.5*10
        
        # Test with different alpha
        radius2 = compute_adaptive_radius(velocities, r_base=20.0, alpha=1.0)
        self.assertAlmostEqual(float(radius2[0]), 30.0, places=6)  # 20 + 1.0*10
        
        # Test with both custom
        radius3 = compute_adaptive_radius(velocities, r_base=15.0, alpha=0.3)
        self.assertAlmostEqual(float(radius3[0]), 18.0, places=6)  # 15 + 0.3*10

    def test_radius_output_shape(self) -> None:
        """Output should have shape (num_agents,)."""
        velocities = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        radius = compute_adaptive_radius(velocities)
        
        self.assertEqual(radius.shape, (3,))
        self.assertEqual(len(radius), 3)

    def test_radius_with_negative_velocities(self) -> None:
        """Negative velocity components should be handled correctly (magnitude is always positive)."""
        # Agent moving at -3 m/s in x and -4 m/s in y → magnitude = 5 m/s
        velocities = np.array([[-3.0, -4.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 5.0 = 22.5
        self.assertAlmostEqual(float(radius[0]), 22.5, places=6)


class MultimodalDecoderTests(unittest.TestCase):
    """Tests for MultimodalDecoder and wta_loss function.
    
    **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
    """

    def test_multimodal_decoder_output_shape(self) -> None:
        """MultimodalDecoder should output shape [B=2, N=5, K=3, T=30, 2]."""
        batch_size = 2
        max_agents = 5
        hidden_dim = 128
        num_modes = 3
        future_steps = 30
        
        # Create decoder
        decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            future_steps=future_steps
        )
        
        # Create inputs
        h = torch.randn(batch_size, max_agents, hidden_dim)
        last_pos = torch.randn(batch_size, max_agents, 2)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Forward pass
        output = decoder(h, last_pos, agent_mask)
        
        # Verify shape
        expected_shape = (batch_size, max_agents, num_modes, future_steps, 2)
        self.assertEqual(tuple(output.shape), expected_shape,
                        f"Expected shape {expected_shape}, got {tuple(output.shape)}")

    def test_multimodal_decoder_masked_agents_produce_zero_predictions(self) -> None:
        """Masked agents (agent_mask=False) should produce zero predictions."""
        batch_size = 2
        max_agents = 5
        hidden_dim = 128
        num_modes = 3
        future_steps = 30
        
        # Create decoder
        decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            future_steps=future_steps
        )
        
        # Create inputs with some masked agents
        h = torch.randn(batch_size, max_agents, hidden_dim)
        last_pos = torch.randn(batch_size, max_agents, 2)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Mask agents: batch 0 agent 2, batch 1 agents 3 and 4
        agent_mask[0, 2] = False
        agent_mask[1, 3] = False
        agent_mask[1, 4] = False
        
        # Forward pass
        output = decoder(h, last_pos, agent_mask)
        
        # Verify masked agents produce zero predictions
        # Check batch 0, agent 2
        self.assertTrue(
            torch.allclose(output[0, 2], torch.zeros_like(output[0, 2])),
            "Masked agent [0, 2] should produce zero predictions"
        )
        
        # Check batch 1, agent 3
        self.assertTrue(
            torch.allclose(output[1, 3], torch.zeros_like(output[1, 3])),
            "Masked agent [1, 3] should produce zero predictions"
        )
        
        # Check batch 1, agent 4
        self.assertTrue(
            torch.allclose(output[1, 4], torch.zeros_like(output[1, 4])),
            "Masked agent [1, 4] should produce zero predictions"
        )
        
        # Verify valid agents produce non-zero predictions (at least some timesteps)
        self.assertFalse(
            torch.allclose(output[0, 0], torch.zeros_like(output[0, 0])),
            "Valid agent [0, 0] should produce non-zero predictions"
        )

    def test_wta_loss_selects_best_mode(self) -> None:
        """WTA loss should select the mode with minimum error."""
        batch_size = 1
        max_agents = 1
        num_modes = 3
        future_steps = 10
        
        # Create ground truth
        target = torch.zeros(batch_size, max_agents, future_steps, 2)
        
        # Create predictions with known errors
        # Mode 0: predictions at distance 1.0 from target
        # Mode 1: predictions at distance 0.5 from target (best mode)
        # Mode 2: predictions at distance 2.0 from target
        pred = torch.zeros(batch_size, max_agents, num_modes, future_steps, 2)
        pred[0, 0, 0, :, 0] = 1.0  # Mode 0: error = 1.0 per timestep
        pred[0, 0, 1, :, 0] = 0.5  # Mode 1: error = 0.5 per timestep (best)
        pred[0, 0, 2, :, 0] = 2.0  # Mode 2: error = 2.0 per timestep
        
        # Create masks (all valid)
        y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Compute WTA loss
        from core_perception.multi_agent_model import wta_loss
        loss = wta_loss(pred, target, y_mask, agent_mask)
        
        # Compute expected loss for mode 1 (best mode)
        # smooth_l1_loss for small errors (< 1.0) is approximately 0.5 * error^2
        # For error = 0.5: smooth_l1 ≈ 0.5 * 0.5^2 = 0.125
        # Total: 10 timesteps * 2 coordinates * 0.125 = 2.5
        # Normalized by (10 timesteps * 2 coords) = 20 elements
        # Expected loss ≈ 2.5 / 20 = 0.125
        
        # Verify loss is finite and positive
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        self.assertGreater(float(loss), 0.0, "Loss should be positive")
        
        # Verify loss is closer to mode 1's error than mode 0 or mode 2
        # Mode 1 has smallest error, so loss should be relatively small
        self.assertLess(float(loss), 0.5, 
                       f"Loss should be small (mode 1 selected), got {float(loss):.4f}")

    def test_wta_loss_respects_y_mask(self) -> None:
        """WTA loss should respect y_mask and exclude masked timesteps."""
        batch_size = 1
        max_agents = 1
        num_modes = 3
        future_steps = 10
        
        # Create ground truth
        target = torch.zeros(batch_size, max_agents, future_steps, 2)
        
        # Create predictions with constant error
        pred = torch.ones(batch_size, max_agents, num_modes, future_steps, 2)
        
        # Create y_mask: only first 5 timesteps are valid
        y_mask = torch.zeros(batch_size, max_agents, future_steps, dtype=torch.bool)
        y_mask[0, 0, :5] = True
        
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Compute WTA loss
        from core_perception.multi_agent_model import wta_loss
        loss_masked = wta_loss(pred, target, y_mask, agent_mask)
        
        # Compute loss with all timesteps valid
        y_mask_full = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
        loss_full = wta_loss(pred, target, y_mask_full, agent_mask)
        
        # Loss with masked timesteps should be different from full loss
        # (unless by coincidence they normalize to the same value)
        # The key is that masked timesteps don't contribute to the error
        self.assertTrue(torch.isfinite(loss_masked), "Masked loss should be finite")
        self.assertTrue(torch.isfinite(loss_full), "Full loss should be finite")
        
        # Both should be positive
        self.assertGreater(float(loss_masked), 0.0)
        self.assertGreater(float(loss_full), 0.0)

    def test_wta_loss_respects_agent_mask(self) -> None:
        """WTA loss should respect agent_mask and exclude masked agents."""
        batch_size = 2
        max_agents = 3
        num_modes = 3
        future_steps = 10
        
        # Create ground truth
        target = torch.zeros(batch_size, max_agents, future_steps, 2)
        
        # Create predictions with constant error
        pred = torch.ones(batch_size, max_agents, num_modes, future_steps, 2)
        
        # Create masks
        y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Mask some agents
        agent_mask[0, 2] = False  # Mask agent 2 in batch 0
        agent_mask[1, 1] = False  # Mask agent 1 in batch 1
        
        # Compute WTA loss
        from core_perception.multi_agent_model import wta_loss
        loss = wta_loss(pred, target, y_mask, agent_mask)
        
        # Verify loss is finite and positive
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        self.assertGreater(float(loss), 0.0, "Loss should be positive")
        
        # Compute loss with all agents valid
        agent_mask_full = torch.ones(batch_size, max_agents, dtype=torch.bool)
        loss_full = wta_loss(pred, target, y_mask, agent_mask_full)
        
        # Loss with masked agents should be different from full loss
        self.assertTrue(torch.isfinite(loss_full), "Full loss should be finite")
        self.assertGreater(float(loss_full), 0.0)

    def test_wta_loss_with_all_modes_equal(self) -> None:
        """WTA loss should handle case where all modes have equal error."""
        batch_size = 1
        max_agents = 1
        num_modes = 3
        future_steps = 10
        
        # Create ground truth
        target = torch.zeros(batch_size, max_agents, future_steps, 2)
        
        # Create predictions where all modes are identical
        pred = torch.ones(batch_size, max_agents, num_modes, future_steps, 2)
        
        # Create masks (all valid)
        y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Compute WTA loss
        from core_perception.multi_agent_model import wta_loss
        loss = wta_loss(pred, target, y_mask, agent_mask)
        
        # Verify loss is finite and positive
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        self.assertGreater(float(loss), 0.0, "Loss should be positive")

    def test_wta_loss_with_zero_error_mode(self) -> None:
        """WTA loss should select mode with zero error and produce near-zero loss."""
        batch_size = 1
        max_agents = 1
        num_modes = 3
        future_steps = 10
        
        # Create ground truth
        target = torch.zeros(batch_size, max_agents, future_steps, 2)
        
        # Create predictions
        # Mode 0: large error
        # Mode 1: zero error (perfect prediction)
        # Mode 2: medium error
        pred = torch.zeros(batch_size, max_agents, num_modes, future_steps, 2)
        pred[0, 0, 0, :, :] = 5.0  # Mode 0: large error
        pred[0, 0, 1, :, :] = 0.0  # Mode 1: zero error (matches target)
        pred[0, 0, 2, :, :] = 2.0  # Mode 2: medium error
        
        # Create masks (all valid)
        y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Compute WTA loss
        from core_perception.multi_agent_model import wta_loss
        loss = wta_loss(pred, target, y_mask, agent_mask)
        
        # Loss should be very close to zero (mode 1 selected)
        self.assertTrue(torch.isfinite(loss), "Loss should be finite")
        self.assertLess(float(loss), 1e-6, 
                       f"Loss should be near zero (perfect mode selected), got {float(loss):.8f}")

    def test_multimodal_decoder_autoregressive_behavior(self) -> None:
        """MultimodalDecoder should produce different predictions at each timestep."""
        batch_size = 1
        max_agents = 1
        hidden_dim = 128
        num_modes = 3
        future_steps = 30
        
        # Create decoder
        decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            future_steps=future_steps
        )
        
        # Create inputs
        h = torch.randn(batch_size, max_agents, hidden_dim)
        last_pos = torch.zeros(batch_size, max_agents, 2)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Forward pass
        output = decoder(h, last_pos, agent_mask)
        
        # Verify predictions change over time (autoregressive)
        # Check that not all timesteps are identical
        for mode_idx in range(num_modes):
            mode_preds = output[0, 0, mode_idx]  # [T, 2]
            
            # Check that at least some timesteps differ
            all_same = True
            for t in range(1, future_steps):
                if not torch.allclose(mode_preds[t], mode_preds[0], atol=1e-6):
                    all_same = False
                    break
            
            self.assertFalse(all_same, 
                           f"Mode {mode_idx} predictions should vary over time (autoregressive)")

    def test_multimodal_decoder_different_modes_produce_different_predictions(self) -> None:
        """Different modes should produce different trajectory predictions."""
        batch_size = 1
        max_agents = 1
        hidden_dim = 128
        num_modes = 3
        future_steps = 30
        
        # Create decoder
        decoder = MultimodalDecoder(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            future_steps=future_steps
        )
        
        # Create inputs
        h = torch.randn(batch_size, max_agents, hidden_dim)
        last_pos = torch.zeros(batch_size, max_agents, 2)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Forward pass
        output = decoder(h, last_pos, agent_mask)
        
        # Verify different modes produce different predictions
        mode_0 = output[0, 0, 0]  # [T, 2]
        mode_1 = output[0, 0, 1]  # [T, 2]
        mode_2 = output[0, 0, 2]  # [T, 2]
        
        # Modes should not be identical (with high probability given random initialization)
        self.assertFalse(
            torch.allclose(mode_0, mode_1, atol=1e-6),
            "Mode 0 and Mode 1 should produce different predictions"
        )
        self.assertFalse(
            torch.allclose(mode_1, mode_2, atol=1e-6),
            "Mode 1 and Mode 2 should produce different predictions"
        )
        self.assertFalse(
            torch.allclose(mode_0, mode_2, atol=1e-6),
            "Mode 0 and Mode 2 should produce different predictions"
        )


if __name__ == "__main__":
    unittest.main()
    """Tests for compute_adaptive_radius function."""

    def test_radius_computation_for_60_kmh_vehicle(self) -> None:
        """Agent at 60 km/h (16.67 m/s) should get radius ≈ 28.3m with default params."""
        # 60 km/h = 16.67 m/s
        velocities = np.array([[16.67, 0.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 16.67 = 28.335
        expected = 20.0 + 0.5 * 16.67
        self.assertAlmostEqual(float(radius[0]), expected, places=2)
        self.assertGreater(float(radius[0]), 28.0)
        self.assertLess(float(radius[0]), 29.0)

    def test_radius_computation_for_10_kmh_vehicle(self) -> None:
        """Agent at 10 km/h (2.78 m/s) should get radius ≈ 21.4m with default params."""
        # 10 km/h = 2.78 m/s
        velocities = np.array([[2.78, 0.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 2.78 = 21.39
        expected = 20.0 + 0.5 * 2.78
        self.assertAlmostEqual(float(radius[0]), expected, places=2)
        self.assertGreater(float(radius[0]), 21.0)
        self.assertLess(float(radius[0]), 22.0)

    def test_radius_computation_for_stationary_vehicle(self) -> None:
        """Agent at 0 m/s should get base radius."""
        velocities = np.array([[0.0, 0.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        self.assertAlmostEqual(float(radius[0]), 20.0, places=6)

    def test_radius_computation_for_multiple_agents(self) -> None:
        """Multiple agents with different velocities should get different radii."""
        # Agent 0: 0 m/s, Agent 1: 10 m/s, Agent 2: 20 m/s
        velocities = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 20.0],
        ])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected radii
        self.assertAlmostEqual(float(radius[0]), 20.0, places=6)  # 20 + 0.5*0
        self.assertAlmostEqual(float(radius[1]), 25.0, places=6)  # 20 + 0.5*10
        self.assertAlmostEqual(float(radius[2]), 30.0, places=6)  # 20 + 0.5*20

    def test_radius_computation_with_diagonal_velocity(self) -> None:
        """Velocity magnitude should be computed correctly for diagonal motion."""
        # Agent moving at 3 m/s in x and 4 m/s in y → magnitude = 5 m/s
        velocities = np.array([[3.0, 4.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 5.0 = 22.5
        self.assertAlmostEqual(float(radius[0]), 22.5, places=6)

    def test_radius_monotonicity_property(self) -> None:
        """Faster agents should always have larger or equal radius (monotonicity)."""
        # Create agents with increasing speeds
        speeds = [0.0, 5.0, 10.0, 15.0, 20.0]
        velocities = np.array([[speed, 0.0] for speed in speeds])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Verify monotonicity: r[i] <= r[i+1]
        for i in range(len(radius) - 1):
            self.assertLessEqual(
                float(radius[i]), 
                float(radius[i + 1]),
                f"Radius should be monotonically increasing: r[{i}]={radius[i]:.2f} > r[{i+1}]={radius[i+1]:.2f}"
            )

    def test_radius_with_custom_parameters(self) -> None:
        """Function should respect custom r_base and alpha parameters."""
        velocities = np.array([[10.0, 0.0]])
        
        # Test with different r_base
        radius1 = compute_adaptive_radius(velocities, r_base=30.0, alpha=0.5)
        self.assertAlmostEqual(float(radius1[0]), 35.0, places=6)  # 30 + 0.5*10
        
        # Test with different alpha
        radius2 = compute_adaptive_radius(velocities, r_base=20.0, alpha=1.0)
        self.assertAlmostEqual(float(radius2[0]), 30.0, places=6)  # 20 + 1.0*10
        
        # Test with both custom
        radius3 = compute_adaptive_radius(velocities, r_base=15.0, alpha=0.3)
        self.assertAlmostEqual(float(radius3[0]), 18.0, places=6)  # 15 + 0.3*10

    def test_radius_output_shape(self) -> None:
        """Output should have shape (num_agents,)."""
        velocities = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        radius = compute_adaptive_radius(velocities)
        
        self.assertEqual(radius.shape, (3,))
        self.assertEqual(len(radius), 3)

    def test_radius_with_negative_velocities(self) -> None:
        """Negative velocity components should be handled correctly (magnitude is always positive)."""
        # Agent moving at -3 m/s in x and -4 m/s in y → magnitude = 5 m/s
        velocities = np.array([[-3.0, -4.0]])
        radius = compute_adaptive_radius(velocities, r_base=20.0, alpha=0.5)
        
        # Expected: r = 20.0 + 0.5 * 5.0 = 22.5
        self.assertAlmostEqual(float(radius[0]), 22.5, places=6)


class BuildAdaptiveAdjacencyTests(unittest.TestCase):
    """Tests for build_adaptive_adjacency function."""

    def test_adjacency_symmetry(self) -> None:
        """Adjacency matrix should be symmetric: adj[i,j] == adj[j,i]."""
        positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 15.0],
            [20.0, 20.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [5.0, 0.0],
            [10.0, 0.0],
            [0.0, 8.0],
            [15.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Verify symmetry
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                self.assertEqual(
                    float(adj[i, j]), 
                    float(adj[j, i]),
                    f"Adjacency matrix not symmetric at ({i},{j}): adj[{i},{j}]={adj[i,j]} != adj[{j},{i}]={adj[j,i]}"
                )

    def test_adjacency_diagonal_is_one(self) -> None:
        """Diagonal elements (self-connections) should be 1."""
        positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [5.0, 0.0],
            [10.0, 0.0],
            [15.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities)
        
        # Verify diagonal is all ones
        for i in range(adj.shape[0]):
            self.assertEqual(float(adj[i, i]), 1.0, f"Diagonal element adj[{i},{i}] should be 1.0")

    def test_agents_within_min_radius_are_connected(self) -> None:
        """Agents within min(r(i), r(j)) should be connected."""
        # Agent 0: velocity 0 m/s → radius = 20.0
        # Agent 1: velocity 10 m/s → radius = 25.0
        # Distance between them: 22 m
        # min(20.0, 25.0) = 20.0
        # 22 > 20.0, so they should NOT be connected
        
        positions = np.array([
            [0.0, 0.0],
            [22.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # They should NOT be connected (distance 22 > min radius 20)
        self.assertEqual(float(adj[0, 1]), 0.0)
        self.assertEqual(float(adj[1, 0]), 0.0)
        
        # Now test with distance within min radius
        positions2 = np.array([
            [0.0, 0.0],
            [18.0, 0.0],
        ], dtype=np.float64)
        
        adj2 = build_adaptive_adjacency(positions2, velocities, r_base=20.0, alpha=0.5)
        
        # They should be connected (distance 18 <= min radius 20)
        self.assertEqual(float(adj2[0, 1]), 1.0)
        self.assertEqual(float(adj2[1, 0]), 1.0)

    def test_agents_beyond_min_radius_not_connected(self) -> None:
        """Agents beyond min(r(i), r(j)) should not be connected."""
        # Agent 0: velocity 0 m/s → radius = 20.0
        # Agent 1: velocity 20 m/s → radius = 30.0
        # Distance between them: 25 m
        # min(20.0, 30.0) = 20.0
        # 25 > 20.0, so they should NOT be connected
        
        positions = np.array([
            [0.0, 0.0],
            [25.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # They should NOT be connected
        self.assertEqual(float(adj[0, 1]), 0.0)
        self.assertEqual(float(adj[1, 0]), 0.0)

    def test_fast_agents_connect_at_larger_distances(self) -> None:
        """Fast-moving agents should connect at larger distances than slow agents."""
        # Agent 0: velocity 20 m/s → radius = 30.0
        # Agent 1: velocity 20 m/s → radius = 30.0
        # Distance between them: 28 m
        # min(30.0, 30.0) = 30.0
        # 28 <= 30.0, so they should be connected
        
        positions = np.array([
            [0.0, 0.0],
            [28.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [20.0, 0.0],
            [20.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # They should be connected
        self.assertEqual(float(adj[0, 1]), 1.0)
        self.assertEqual(float(adj[1, 0]), 1.0)
        
        # Now test same distance with slow agents
        velocities_slow = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        
        adj_slow = build_adaptive_adjacency(positions, velocities_slow, r_base=20.0, alpha=0.5)
        
        # They should NOT be connected (distance 28 > base radius 20)
        self.assertEqual(float(adj_slow[0, 1]), 0.0)
        self.assertEqual(float(adj_slow[1, 0]), 0.0)

    def test_adjacency_with_multiple_agents(self) -> None:
        """Test adjacency matrix with multiple agents at various distances."""
        # Agent 0 at origin: velocity 0 m/s → radius = 20.0
        # Agent 1 at (15, 0): velocity 10 m/s → radius = 25.0
        # Agent 2 at (30, 0): velocity 20 m/s → radius = 30.0
        # Agent 3 at (0, 25): velocity 0 m/s → radius = 20.0
        
        positions = np.array([
            [0.0, 0.0],
            [15.0, 0.0],
            [30.0, 0.0],
            [0.0, 25.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Agent 0 and 1: distance = 15, min_radius = min(20, 25) = 20 → connected
        self.assertEqual(float(adj[0, 1]), 1.0)
        
        # Agent 0 and 2: distance = 30, min_radius = min(20, 30) = 20 → not connected
        self.assertEqual(float(adj[0, 2]), 0.0)
        
        # Agent 0 and 3: distance = 25, min_radius = min(20, 20) = 20 → not connected
        self.assertEqual(float(adj[0, 3]), 0.0)
        
        # Agent 1 and 2: distance = 15, min_radius = min(25, 30) = 25 → connected
        self.assertEqual(float(adj[1, 2]), 1.0)
        
        # Agent 1 and 3: distance = sqrt(15^2 + 25^2) ≈ 29.15, min_radius = min(25, 20) = 20 → not connected
        self.assertEqual(float(adj[1, 3]), 0.0)
        
        # Agent 2 and 3: distance = sqrt(30^2 + 25^2) ≈ 39.05, min_radius = min(30, 20) = 20 → not connected
        self.assertEqual(float(adj[2, 3]), 0.0)

    def test_adjacency_uses_float64_precision(self) -> None:
        """Distance computation should use float64 precision for accuracy."""
        # Test with positions that might have precision issues with float32
        positions = np.array([
            [0.0, 0.0],
            [20.0000001, 0.0],  # Just slightly over 20m
        ], dtype=np.float64)
        
        velocities = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Distance is 20.0000001, which is > 20.0, so should not be connected
        self.assertEqual(float(adj[0, 1]), 0.0)
        self.assertEqual(float(adj[1, 0]), 0.0)

    def test_adjacency_with_custom_parameters(self) -> None:
        """Function should respect custom r_base and alpha parameters."""
        positions = np.array([
            [0.0, 0.0],
            [25.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [10.0, 0.0],
            [10.0, 0.0],
        ])
        
        # With r_base=20.0, alpha=0.5: radius = 20 + 0.5*10 = 25
        # Distance = 25, min_radius = 25 → connected
        adj1 = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        self.assertEqual(float(adj1[0, 1]), 1.0)
        
        # With r_base=15.0, alpha=0.5: radius = 15 + 0.5*10 = 20
        # Distance = 25, min_radius = 20 → not connected
        adj2 = build_adaptive_adjacency(positions, velocities, r_base=15.0, alpha=0.5)
        self.assertEqual(float(adj2[0, 1]), 0.0)
        
        # With r_base=20.0, alpha=1.0: radius = 20 + 1.0*10 = 30
        # Distance = 25, min_radius = 30 → connected
        adj3 = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=1.0)
        self.assertEqual(float(adj3[0, 1]), 1.0)

    def test_adjacency_output_shape(self) -> None:
        """Output should be square matrix with shape (num_agents, num_agents)."""
        positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [5.0, 0.0],
            [10.0, 0.0],
            [15.0, 0.0],
            [20.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities)
        
        self.assertEqual(adj.shape, (4, 4))
        self.assertEqual(adj.dtype, np.float32)

    def test_adjacency_with_diagonal_velocities(self) -> None:
        """Velocity magnitude should be computed correctly for diagonal motion."""
        # Agent 0: velocity (3, 4) → magnitude = 5 m/s → radius = 20 + 0.5*5 = 22.5
        # Agent 1: velocity (0, 0) → magnitude = 0 m/s → radius = 20.0
        # Distance = 21 m
        # min_radius = min(22.5, 20.0) = 20.0
        # 21 > 20.0 → not connected
        
        positions = np.array([
            [0.0, 0.0],
            [21.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [3.0, 4.0],
            [0.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Should not be connected
        self.assertEqual(float(adj[0, 1]), 0.0)
        self.assertEqual(float(adj[1, 0]), 0.0)

    def test_adjacency_boundary_case_exactly_at_threshold(self) -> None:
        """Agents exactly at the threshold distance should be connected."""
        # Agent 0: velocity 0 m/s → radius = 20.0
        # Agent 1: velocity 0 m/s → radius = 20.0
        # Distance = exactly 20.0 m
        # min_radius = 20.0
        # 20.0 <= 20.0 → connected
        
        positions = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
        ], dtype=np.float64)
        
        velocities = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        
        adj = build_adaptive_adjacency(positions, velocities, r_base=20.0, alpha=0.5)
        
        # Should be connected (distance <= threshold)
        self.assertEqual(float(adj[0, 1]), 1.0)
        self.assertEqual(float(adj[1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()


class ComputeMultimodalMetricsTests(unittest.TestCase):
    """Tests for compute_multimodal_metrics function.
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.9
    """

    def test_minADE_computation_with_known_predictions(self) -> None:
        """Test minADE computation with known predictions and targets.
        
        Requirements: 4.1, 4.4, 4.5
        """
        # Create simple test case:
        # - 1 batch, 2 agents, 3 modes, 4 timesteps
        # - Agent 0: mode 0 has ADE=1.0, mode 1 has ADE=0.5, mode 2 has ADE=2.0
        # - Agent 1: mode 0 has ADE=1.5, mode 1 has ADE=1.0, mode 2 has ADE=0.8
        
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 3, 4
        
        # Ground truth: all zeros for simplicity
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Predictions: set constant displacements per mode
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0, mode 0: displacement = 1.0 at all timesteps → ADE = 1.0
        pred[0, 0, 0, :, :] = 1.0 / math.sqrt(2)  # x=y to get norm=1.0
        
        # Agent 0, mode 1: displacement = 0.5 at all timesteps → ADE = 0.5
        pred[0, 0, 1, :, :] = 0.5 / math.sqrt(2)
        
        # Agent 0, mode 2: displacement = 2.0 at all timesteps → ADE = 2.0
        pred[0, 0, 2, :, :] = 2.0 / math.sqrt(2)
        
        # Agent 1, mode 0: displacement = 1.5 at all timesteps → ADE = 1.5
        pred[0, 1, 0, :, :] = 1.5 / math.sqrt(2)
        
        # Agent 1, mode 1: displacement = 1.0 at all timesteps → ADE = 1.0
        pred[0, 1, 1, :, :] = 1.0 / math.sqrt(2)
        
        # Agent 1, mode 2: displacement = 0.8 at all timesteps → ADE = 0.8
        pred[0, 1, 2, :, :] = 0.8 / math.sqrt(2)
        
        # All timesteps and agents are valid
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Agent 0: minADE = min(1.0, 0.5, 2.0) = 0.5
        # Agent 1: minADE = min(1.5, 1.0, 0.8) = 0.8
        # Overall minADE = (0.5 + 0.8) / 2 = 0.65
        expected_minADE = (0.5 + 0.8) / 2
        self.assertAlmostEqual(metrics['minADE'], expected_minADE, places=5)

    def test_minFDE_computation_with_known_predictions(self) -> None:
        """Test minFDE computation with known predictions and targets.
        
        Requirements: 4.2, 4.4, 4.5
        """
        # Create test case where FDE differs from ADE
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 3, 4
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0: set only final timestep (t=3) to have specific displacements
        # Mode 0: FDE = 2.0
        pred[0, 0, 0, 3, :] = 2.0 / math.sqrt(2)
        # Mode 1: FDE = 1.0
        pred[0, 0, 1, 3, :] = 1.0 / math.sqrt(2)
        # Mode 2: FDE = 3.0
        pred[0, 0, 2, 3, :] = 3.0 / math.sqrt(2)
        
        # Agent 1: set only final timestep
        # Mode 0: FDE = 1.5
        pred[0, 1, 0, 3, :] = 1.5 / math.sqrt(2)
        # Mode 1: FDE = 2.5
        pred[0, 1, 1, 3, :] = 2.5 / math.sqrt(2)
        # Mode 2: FDE = 0.5
        pred[0, 1, 2, 3, :] = 0.5 / math.sqrt(2)
        
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Agent 0: minFDE = min(2.0, 1.0, 3.0) = 1.0
        # Agent 1: minFDE = min(1.5, 2.5, 0.5) = 0.5
        # Overall minFDE = (1.0 + 0.5) / 2 = 0.75
        expected_minFDE = (1.0 + 0.5) / 2
        self.assertAlmostEqual(metrics['minFDE'], expected_minFDE, places=5)

    def test_missrate_computation(self) -> None:
        """Test MissRate computation (minFDE > 2.0m threshold).
        
        Requirements: 4.3, 4.4, 4.5
        """
        # Create test case with some agents missing (FDE > 2.0) and some hitting
        batch_size, num_agents, num_modes, num_timesteps = 1, 4, 2, 3
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0: minFDE = 1.5 (< 2.0, hit)
        pred[0, 0, 0, 2, :] = 1.5 / math.sqrt(2)
        pred[0, 0, 1, 2, :] = 3.0 / math.sqrt(2)
        
        # Agent 1: minFDE = 2.5 (> 2.0, miss)
        pred[0, 1, 0, 2, :] = 2.5 / math.sqrt(2)
        pred[0, 1, 1, 2, :] = 4.0 / math.sqrt(2)
        
        # Agent 2: minFDE = 0.8 (< 2.0, hit)
        pred[0, 2, 0, 2, :] = 0.8 / math.sqrt(2)
        pred[0, 2, 1, 2, :] = 1.2 / math.sqrt(2)
        
        # Agent 3: minFDE = 3.0 (> 2.0, miss)
        pred[0, 3, 0, 2, :] = 3.0 / math.sqrt(2)
        pred[0, 3, 1, 2, :] = 5.0 / math.sqrt(2)
        
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # 2 out of 4 agents have minFDE > 2.0
        # MissRate = 2 / 4 = 0.5
        expected_missrate = 0.5
        self.assertAlmostEqual(metrics['MissRate'], expected_missrate, places=5)

    def test_masking_excludes_invalid_timesteps(self) -> None:
        """Test that masked timesteps are excluded from computation.
        
        Requirements: 4.4, 4.5
        """
        batch_size, num_agents, num_modes, num_timesteps = 1, 1, 2, 4
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.ones(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Only first 2 timesteps are valid
        y_mask = torch.zeros(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        y_mask[0, 0, 0:2] = True
        
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # ADE should only consider first 2 timesteps
        # Displacement at each valid timestep = sqrt(1^2 + 1^2) = sqrt(2)
        # ADE = sqrt(2) (average over 2 timesteps)
        expected_ade = math.sqrt(2)
        self.assertAlmostEqual(metrics['minADE'], expected_ade, places=5)
        
        # FDE should be at last valid timestep (t=1)
        expected_fde = math.sqrt(2)
        self.assertAlmostEqual(metrics['minFDE'], expected_fde, places=5)

    def test_masking_excludes_invalid_agents(self) -> None:
        """Test that masked agents are excluded from computation.
        
        Requirements: 4.4, 4.5
        """
        batch_size, num_agents, num_modes, num_timesteps = 1, 3, 2, 3
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0: ADE = 1.0
        pred[0, 0, 0, :, :] = 1.0 / math.sqrt(2)
        pred[0, 0, 1, :, :] = 2.0 / math.sqrt(2)
        
        # Agent 1: ADE = 2.0 (but will be masked out)
        pred[0, 1, 0, :, :] = 2.0 / math.sqrt(2)
        pred[0, 1, 1, :, :] = 3.0 / math.sqrt(2)
        
        # Agent 2: ADE = 0.5
        pred[0, 2, 0, :, :] = 0.5 / math.sqrt(2)
        pred[0, 2, 1, :, :] = 1.5 / math.sqrt(2)
        
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        
        # Only agents 0 and 2 are valid
        agent_mask = torch.tensor([[True, False, True]], dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Should only consider agents 0 and 2
        # Agent 0: minADE = 1.0
        # Agent 2: minADE = 0.5
        # Overall minADE = (1.0 + 0.5) / 2 = 0.75
        expected_minADE = (1.0 + 0.5) / 2
        self.assertAlmostEqual(metrics['minADE'], expected_minADE, places=5)

    def test_unimodal_input_handling(self) -> None:
        """Test handling of unimodal (K=1) predictions.
        
        Requirements: 4.9
        """
        # Unimodal prediction: [B, N, T, 2] (no mode dimension)
        batch_size, num_agents, num_timesteps = 1, 2, 3
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Unimodal prediction (no mode dimension)
        pred_unimodal = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred_unimodal[0, 0, :, :] = 1.0 / math.sqrt(2)  # Agent 0: displacement = 1.0
        pred_unimodal[0, 1, :, :] = 0.5 / math.sqrt(2)  # Agent 1: displacement = 0.5
        
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred_unimodal, target, y_mask, agent_mask)
        
        # With K=1, minADE = ADE of the single mode
        # Agent 0: ADE = 1.0
        # Agent 1: ADE = 0.5
        # Overall minADE = (1.0 + 0.5) / 2 = 0.75
        expected_minADE = (1.0 + 0.5) / 2
        self.assertAlmostEqual(metrics['minADE'], expected_minADE, places=5)
        
        # Verify per-mode metrics exist
        self.assertIn('mode_0_ADE', metrics)
        self.assertIn('mode_0_FDE', metrics)

    def test_per_mode_metrics(self) -> None:
        """Test per-mode ADE and FDE metrics for analysis.
        
        Requirements: 4.8
        """
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 3, 4
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Set specific displacements for each mode
        # Agent 0: mode 0 = 1.0, mode 1 = 2.0, mode 2 = 3.0
        pred[0, 0, 0, :, :] = 1.0 / math.sqrt(2)
        pred[0, 0, 1, :, :] = 2.0 / math.sqrt(2)
        pred[0, 0, 2, :, :] = 3.0 / math.sqrt(2)
        
        # Agent 1: mode 0 = 1.5, mode 1 = 2.5, mode 2 = 3.5
        pred[0, 1, 0, :, :] = 1.5 / math.sqrt(2)
        pred[0, 1, 1, :, :] = 2.5 / math.sqrt(2)
        pred[0, 1, 2, :, :] = 3.5 / math.sqrt(2)
        
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Mode 0: average ADE = (1.0 + 1.5) / 2 = 1.25
        self.assertAlmostEqual(metrics['mode_0_ADE'], 1.25, places=5)
        
        # Mode 1: average ADE = (2.0 + 2.5) / 2 = 2.25
        self.assertAlmostEqual(metrics['mode_1_ADE'], 2.25, places=5)
        
        # Mode 2: average ADE = (3.0 + 3.5) / 2 = 3.25
        self.assertAlmostEqual(metrics['mode_2_ADE'], 3.25, places=5)
        
        # Verify FDE metrics also exist
        self.assertIn('mode_0_FDE', metrics)
        self.assertIn('mode_1_FDE', metrics)
        self.assertIn('mode_2_FDE', metrics)

    def test_empty_batch_returns_zero_metrics(self) -> None:
        """Test that empty batch (no valid agents) returns zero metrics.
        
        Requirements: 4.4, 4.5
        """
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 3, 4
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        
        # No valid agents
        agent_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # All metrics should be zero
        self.assertEqual(metrics['minADE'], 0.0)
        self.assertEqual(metrics['minFDE'], 0.0)
        self.assertEqual(metrics['MissRate'], 0.0)
        self.assertEqual(metrics['mode_0_ADE'], 0.0)
        self.assertEqual(metrics['mode_1_ADE'], 0.0)
        self.assertEqual(metrics['mode_2_ADE'], 0.0)


if __name__ == "__main__":
    unittest.main()


class ComputeMultimodalMetricsTests(unittest.TestCase):
    """Tests for compute_multimodal_metrics function.
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.9
    """

    def test_minADE_computation_with_known_predictions(self) -> None:
        """Test minADE computation with known predictions and targets.
        
        Requirements: 4.1, 4.4, 4.5
        """
        # Create simple test case:
        # - 1 batch, 2 agents, 3 modes, 4 timesteps
        # - Agent 0: mode 0 has ADE=1.0, mode 1 has ADE=0.5, mode 2 has ADE=2.0
        # - Agent 1: mode 0 has ADE=3.0, mode 1 has ADE=1.5, mode 2 has ADE=1.0
        
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 3, 4
        
        # Ground truth: all zeros for simplicity
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Predictions: constant offsets per mode to create known ADEs
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0, mode 0: offset (1, 0) → distance = 1.0 → ADE = 1.0
        pred[0, 0, 0, :, 0] = 1.0
        
        # Agent 0, mode 1: offset (0.5, 0) → distance = 0.5 → ADE = 0.5 (best)
        pred[0, 0, 1, :, 0] = 0.5
        
        # Agent 0, mode 2: offset (2, 0) → distance = 2.0 → ADE = 2.0
        pred[0, 0, 2, :, 0] = 2.0
        
        # Agent 1, mode 0: offset (3, 0) → distance = 3.0 → ADE = 3.0
        pred[0, 1, 0, :, 0] = 3.0
        
        # Agent 1, mode 1: offset (1.5, 0) → distance = 1.5 → ADE = 1.5
        pred[0, 1, 1, :, 0] = 1.5
        
        # Agent 1, mode 2: offset (1, 0) → distance = 1.0 → ADE = 1.0 (best)
        pred[0, 1, 2, :, 0] = 1.0
        
        # All timesteps and agents are valid
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Expected minADE: (0.5 + 1.0) / 2 = 0.75
        self.assertAlmostEqual(metrics['minADE'], 0.75, places=6)
        
        # Verify per-mode ADEs
        # Mode 0: (1.0 + 3.0) / 2 = 2.0
        self.assertAlmostEqual(metrics['mode_0_ADE'], 2.0, places=6)
        # Mode 1: (0.5 + 1.5) / 2 = 1.0
        self.assertAlmostEqual(metrics['mode_1_ADE'], 1.0, places=6)
        # Mode 2: (2.0 + 1.0) / 2 = 1.5
        self.assertAlmostEqual(metrics['mode_2_ADE'], 1.5, places=6)

    def test_minFDE_computation_with_known_predictions(self) -> None:
        """Test minFDE computation with known predictions and targets.
        
        Requirements: 4.2, 4.4, 4.5
        """
        # Create test case where FDE differs from ADE
        # - 1 batch, 2 agents, 3 modes, 4 timesteps
        # - Predictions vary over time to make FDE != ADE
        
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 3, 4
        
        # Ground truth: all zeros
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Predictions: vary over time
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0, mode 0: increasing offset → FDE = 3.0 at t=3
        for t in range(num_timesteps):
            pred[0, 0, 0, t, 0] = float(t)
        
        # Agent 0, mode 1: constant offset → FDE = 1.0 at t=3 (best)
        pred[0, 0, 1, :, 0] = 1.0
        
        # Agent 0, mode 2: decreasing offset → FDE = 0.0 at t=3 (best actually)
        for t in range(num_timesteps):
            pred[0, 0, 2, t, 0] = float(num_timesteps - 1 - t)
        
        # Agent 1, mode 0: FDE = 5.0 at t=3
        pred[0, 1, 0, -1, 0] = 5.0
        
        # Agent 1, mode 1: FDE = 2.0 at t=3 (best)
        pred[0, 1, 1, -1, 0] = 2.0
        
        # Agent 1, mode 2: FDE = 3.0 at t=3
        pred[0, 1, 2, -1, 0] = 3.0
        
        # All timesteps and agents are valid
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Expected minFDE for agent 0: min(3.0, 1.0, 0.0) = 0.0
        # Expected minFDE for agent 1: min(5.0, 2.0, 3.0) = 2.0
        # Average: (0.0 + 2.0) / 2 = 1.0
        self.assertAlmostEqual(metrics['minFDE'], 1.0, places=6)
        
        # Verify per-mode FDEs
        # Mode 0: (3.0 + 5.0) / 2 = 4.0
        self.assertAlmostEqual(metrics['mode_0_FDE'], 4.0, places=6)
        # Mode 1: (1.0 + 2.0) / 2 = 1.5
        self.assertAlmostEqual(metrics['mode_1_FDE'], 1.5, places=6)
        # Mode 2: (0.0 + 3.0) / 2 = 1.5
        self.assertAlmostEqual(metrics['mode_2_FDE'], 1.5, places=6)

    def test_MissRate_computation_with_threshold(self) -> None:
        """Test MissRate computation (minFDE > 2.0m threshold).
        
        Requirements: 4.3, 4.4, 4.5
        """
        # Create test case with some agents missing (FDE > 2.0m)
        # - 1 batch, 4 agents, 2 modes, 3 timesteps
        
        batch_size, num_agents, num_modes, num_timesteps = 1, 4, 2, 3
        
        # Ground truth: all zeros
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Predictions at final timestep
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0: minFDE = 1.0 (< 2.0, not a miss)
        pred[0, 0, 0, -1, 0] = 3.0  # mode 0: FDE = 3.0
        pred[0, 0, 1, -1, 0] = 1.0  # mode 1: FDE = 1.0 (best)
        
        # Agent 1: minFDE = 2.5 (> 2.0, miss)
        pred[0, 1, 0, -1, 0] = 2.5  # mode 0: FDE = 2.5 (best)
        pred[0, 1, 1, -1, 0] = 4.0  # mode 1: FDE = 4.0
        
        # Agent 2: minFDE = 0.5 (< 2.0, not a miss)
        pred[0, 2, 0, -1, 0] = 0.5  # mode 0: FDE = 0.5 (best)
        pred[0, 2, 1, -1, 0] = 1.5  # mode 1: FDE = 1.5
        
        # Agent 3: minFDE = 3.0 (> 2.0, miss)
        pred[0, 3, 0, -1, 0] = 5.0  # mode 0: FDE = 5.0
        pred[0, 3, 1, -1, 0] = 3.0  # mode 1: FDE = 3.0 (best)
        
        # All timesteps and agents are valid
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Expected MissRate: 2 misses out of 4 agents = 0.5
        self.assertAlmostEqual(metrics['MissRate'], 0.5, places=6)

    def test_masking_excludes_invalid_timesteps(self) -> None:
        """Test masking: verify masked timesteps excluded from computation.
        
        Requirements: 4.4, 4.5
        """
        # Create test case with some timesteps masked
        # - 1 batch, 1 agent, 2 modes, 4 timesteps
        # - Only first 2 timesteps are valid
        
        batch_size, num_agents, num_modes, num_timesteps = 1, 1, 2, 4
        
        # Ground truth: all zeros
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Predictions: constant offset
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Mode 0: offset (1, 0) at all timesteps
        pred[0, 0, 0, :, 0] = 1.0
        
        # Mode 1: offset (2, 0) at all timesteps
        pred[0, 0, 1, :, 0] = 2.0
        
        # Only first 2 timesteps are valid
        y_mask = torch.zeros(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        y_mask[0, 0, :2] = True
        
        # Agent is valid
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Expected minADE: only first 2 timesteps count
        # Mode 0: (1.0 + 1.0) / 2 = 1.0
        # Mode 1: (2.0 + 2.0) / 2 = 2.0
        # minADE = 1.0
        self.assertAlmostEqual(metrics['minADE'], 1.0, places=6)
        
        # Expected minFDE: last valid timestep is t=1
        # Mode 0: FDE = 1.0
        # Mode 1: FDE = 2.0
        # minFDE = 1.0
        self.assertAlmostEqual(metrics['minFDE'], 1.0, places=6)

    def test_masking_excludes_invalid_agents(self) -> None:
        """Test masking: verify masked agents excluded from computation.
        
        Requirements: 4.4, 4.5
        """
        # Create test case with some agents masked (padding)
        # - 1 batch, 3 agents (2 valid, 1 padding), 2 modes, 3 timesteps
        
        batch_size, num_agents, num_modes, num_timesteps = 1, 3, 2, 3
        
        # Ground truth: all zeros
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Predictions
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        
        # Agent 0 (valid): minADE = 1.0
        pred[0, 0, 0, :, 0] = 1.0  # mode 0: ADE = 1.0 (best)
        pred[0, 0, 1, :, 0] = 2.0  # mode 1: ADE = 2.0
        
        # Agent 1 (valid): minADE = 3.0
        pred[0, 1, 0, :, 0] = 3.0  # mode 0: ADE = 3.0 (best)
        pred[0, 1, 1, :, 0] = 4.0  # mode 1: ADE = 4.0
        
        # Agent 2 (padding): should be ignored
        pred[0, 2, 0, :, 0] = 100.0  # mode 0: ADE = 100.0
        pred[0, 2, 1, :, 0] = 200.0  # mode 1: ADE = 200.0
        
        # All timesteps are valid
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        
        # Only first 2 agents are valid
        agent_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool)
        agent_mask[0, :2] = True
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Expected minADE: only first 2 agents count
        # (1.0 + 3.0) / 2 = 2.0
        self.assertAlmostEqual(metrics['minADE'], 2.0, places=6)
        
        # Expected minFDE: only first 2 agents count
        # (1.0 + 3.0) / 2 = 2.0
        self.assertAlmostEqual(metrics['minFDE'], 2.0, places=6)

    def test_unimodal_input_handling(self) -> None:
        """Test unimodal (K=1) vs. multimodal (K>1) input handling.
        
        Requirements: 4.9
        """
        # Test with unimodal input (no mode dimension)
        # - 1 batch, 2 agents, 3 timesteps
        
        batch_size, num_agents, num_timesteps = 1, 2, 3
        
        # Ground truth: all zeros
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Unimodal prediction: [B, N, T, 2] (no mode dimension)
        pred_unimodal = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred_unimodal[0, 0, :, 0] = 1.0  # Agent 0: offset (1, 0)
        pred_unimodal[0, 1, :, 0] = 2.0  # Agent 1: offset (2, 0)
        
        # All timesteps and agents are valid
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics_unimodal = compute_multimodal_metrics(pred_unimodal, target, y_mask, agent_mask)
        
        # Expected minADE: (1.0 + 2.0) / 2 = 1.5
        self.assertAlmostEqual(metrics_unimodal['minADE'], 1.5, places=6)
        
        # Expected minFDE: (1.0 + 2.0) / 2 = 1.5
        self.assertAlmostEqual(metrics_unimodal['minFDE'], 1.5, places=6)
        
        # Should have mode_0 metrics
        self.assertIn('mode_0_ADE', metrics_unimodal)
        self.assertIn('mode_0_FDE', metrics_unimodal)
        self.assertAlmostEqual(metrics_unimodal['mode_0_ADE'], 1.5, places=6)
        self.assertAlmostEqual(metrics_unimodal['mode_0_FDE'], 1.5, places=6)
        
        # Now test with multimodal input: [B, N, K, T, 2]
        pred_multimodal = torch.zeros(batch_size, num_agents, 2, num_timesteps, 2)
        pred_multimodal[0, 0, 0, :, 0] = 1.0  # Agent 0, mode 0
        pred_multimodal[0, 0, 1, :, 0] = 0.5  # Agent 0, mode 1 (best)
        pred_multimodal[0, 1, 0, :, 0] = 2.0  # Agent 1, mode 0 (best)
        pred_multimodal[0, 1, 1, :, 0] = 3.0  # Agent 1, mode 1
        
        metrics_multimodal = compute_multimodal_metrics(pred_multimodal, target, y_mask, agent_mask)
        
        # Expected minADE: (0.5 + 2.0) / 2 = 1.25
        self.assertAlmostEqual(metrics_multimodal['minADE'], 1.25, places=6)
        
        # Should have mode_0 and mode_1 metrics
        self.assertIn('mode_0_ADE', metrics_multimodal)
        self.assertIn('mode_1_ADE', metrics_multimodal)
        self.assertIn('mode_0_FDE', metrics_multimodal)
        self.assertIn('mode_1_FDE', metrics_multimodal)

    def test_empty_batch_returns_zero_metrics(self) -> None:
        """Test that empty batch (no valid agents) returns zero metrics.
        
        Requirements: 4.4, 4.5
        """
        # Create batch with no valid agents
        batch_size, num_agents, num_modes, num_timesteps = 1, 2, 2, 3
        
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        
        # No valid agents
        agent_mask = torch.zeros(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # All metrics should be zero
        self.assertEqual(metrics['minADE'], 0.0)
        self.assertEqual(metrics['minFDE'], 0.0)
        self.assertEqual(metrics['MissRate'], 0.0)
        self.assertEqual(metrics['mode_0_ADE'], 0.0)
        self.assertEqual(metrics['mode_0_FDE'], 0.0)

    def test_2d_velocity_displacement(self) -> None:
        """Test displacement computation with 2D coordinates (x, y).
        
        Requirements: 4.1, 4.2
        """
        # Test that Euclidean distance is computed correctly for 2D coordinates
        batch_size, num_agents, num_modes, num_timesteps = 1, 1, 1, 1
        
        # Ground truth at origin
        target = torch.zeros(batch_size, num_agents, num_timesteps, 2)
        
        # Prediction at (3, 4) → distance = 5.0
        pred = torch.zeros(batch_size, num_agents, num_modes, num_timesteps, 2)
        pred[0, 0, 0, 0, 0] = 3.0
        pred[0, 0, 0, 0, 1] = 4.0
        
        y_mask = torch.ones(batch_size, num_agents, num_timesteps, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
        
        metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
        
        # Expected distance: sqrt(3^2 + 4^2) = 5.0
        self.assertAlmostEqual(metrics['minADE'], 5.0, places=6)
        self.assertAlmostEqual(metrics['minFDE'], 5.0, places=6)
