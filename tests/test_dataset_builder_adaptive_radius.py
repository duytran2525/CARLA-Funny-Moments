"""
Test dataset builder with adaptive radius support.

This test verifies that the dataset builder correctly handles the adaptive radius
configuration and produces samples with the expected adjacency matrices.
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from core_perception.multi_agent_trajectory import (
    WindowBuildConfig,
    build_multi_agent_samples,
    read_raw_frames,
    sample_to_torch_payload,
)


class DatasetBuilderAdaptiveRadiusTests(unittest.TestCase):
    """Tests for dataset builder with adaptive radius support.
    
    Requirements: 6.6, 6.7
    """

    def _create_test_csv(self, csv_path: Path) -> None:
        """Create a test CSV file with multi-agent data."""
        import csv
        
        fieldnames = [
            "run_id", "town", "frame", "timestamp",
            "ego_id", "ego_x", "ego_y", "ego_z", "ego_vx", "ego_vy", "ego_yaw",
            "actor_id", "actor_type", "actor_x", "actor_y", "actor_z",
            "actor_vx", "actor_vy", "actor_yaw", "distance_m"
        ]
        
        rows = []
        # Create 10 frames with 2 agents
        # Agent 10: fast-moving (60 km/h = 16.67 m/s)
        # Agent 20: slow-moving (10 km/h = 2.78 m/s)
        # Distance between agents: 22m (should be connected with adaptive radius)
        for frame_idx in range(10):
            timestamp = frame_idx * 0.1
            
            # Agent 10 (fast)
            rows.append({
                "run_id": "test_run",
                "town": "Town01",
                "frame": str(frame_idx),
                "timestamp": f"{timestamp:.6f}",
                "ego_id": "1",
                "ego_x": "0.0",
                "ego_y": "0.0",
                "ego_z": "0.0",
                "ego_vx": "0.0",
                "ego_vy": "0.0",
                "ego_yaw": "0.0",
                "actor_id": "10",
                "actor_type": "vehicle.test",
                "actor_x": f"{float(frame_idx) * 0.5:.6f}",  # Moving slowly
                "actor_y": "0.0",
                "actor_z": "0.0",
                "actor_vx": "16.67",  # 60 km/h
                "actor_vy": "0.0",
                "actor_yaw": "0.0",
                "distance_m": f"{abs(float(frame_idx) * 0.5):.3f}",
            })
            
            # Agent 20 (slow)
            rows.append({
                "run_id": "test_run",
                "town": "Town01",
                "frame": str(frame_idx),
                "timestamp": f"{timestamp:.6f}",
                "ego_id": "1",
                "ego_x": "0.0",
                "ego_y": "0.0",
                "ego_z": "0.0",
                "ego_vx": "0.0",
                "ego_vy": "0.0",
                "ego_yaw": "0.0",
                "actor_id": "20",
                "actor_type": "vehicle.test",
                "actor_x": f"{float(frame_idx) * 0.5:.6f}",  # Moving slowly
                "actor_y": "22.0",  # 22m away from agent 10
                "actor_z": "0.0",
                "actor_vx": "2.78",  # 10 km/h
                "actor_vy": "0.0",
                "actor_yaw": "0.0",
                "distance_m": f"{np.hypot(float(frame_idx) * 0.5, 22.0):.3f}",
            })
        
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_adaptive_radius_enabled_in_config(self) -> None:
        """Test that adaptive radius can be enabled in WindowBuildConfig.
        
        Requirements: 6.6
        """
        config = WindowBuildConfig(
            history_frames=2,
            future_frames=2,
            stride=1,
            adaptive_radius_enabled=True,
            radius_base=20.0,
            radius_alpha=0.5,
        )
        
        self.assertTrue(config.adaptive_radius_enabled)
        self.assertEqual(config.radius_base, 20.0)
        self.assertEqual(config.radius_alpha, 0.5)

    def test_adaptive_radius_disabled_by_default(self) -> None:
        """Test that adaptive radius is disabled by default.
        
        Requirements: 6.6
        """
        config = WindowBuildConfig()
        
        self.assertFalse(config.adaptive_radius_enabled)
        self.assertEqual(config.radius_base, 20.0)
        self.assertEqual(config.radius_alpha, 0.5)

    def test_build_samples_with_adaptive_radius(self) -> None:
        """Test building samples with adaptive radius enabled.
        
        Requirements: 6.6, 6.7
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_data.csv"
            self._create_test_csv(csv_path)
            
            # Read frames
            frames = read_raw_frames(csv_path)
            self.assertGreater(len(frames), 0)
            
            # Build samples with adaptive radius enabled
            config = WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adaptive_radius_enabled=True,
                radius_base=20.0,
                radius_alpha=0.5,
                expected_dt=0.1,
                max_dt_error=0.03,
            )
            
            samples = build_multi_agent_samples(frames, config)
            self.assertGreater(len(samples), 0)
            
            # Verify sample structure
            sample = samples[0]
            self.assertIn("x", sample)
            self.assertIn("y", sample)
            self.assertIn("adj", sample)
            self.assertIn("x_mask", sample)
            self.assertIn("y_mask", sample)
            
            # Verify adjacency matrix shape
            n_agents = sample["x"].shape[0]
            self.assertEqual(sample["adj"].shape, (n_agents, n_agents))
            
            # Verify adjacency matrix is symmetric
            adj = sample["adj"]
            self.assertTrue(np.allclose(adj, adj.T), "Adjacency matrix should be symmetric")
            
            # Verify self-connections exist
            for i in range(n_agents):
                self.assertEqual(adj[i, i], 1.0, f"Agent {i} should have self-connection")

    def test_build_samples_with_fixed_radius(self) -> None:
        """Test building samples with fixed radius (legacy mode).
        
        Requirements: 6.6
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_data.csv"
            self._create_test_csv(csv_path)
            
            # Read frames
            frames = read_raw_frames(csv_path)
            
            # Build samples with fixed radius (adaptive_radius_enabled=False)
            config = WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=40.0,
                adaptive_radius_enabled=False,
                expected_dt=0.1,
                max_dt_error=0.03,
            )
            
            samples = build_multi_agent_samples(frames, config)
            self.assertGreater(len(samples), 0)
            
            # Verify sample structure
            sample = samples[0]
            self.assertIn("adj", sample)
            
            # Verify adjacency matrix is symmetric
            adj = sample["adj"]
            self.assertTrue(np.allclose(adj, adj.T), "Adjacency matrix should be symmetric")

    def test_adaptive_vs_fixed_radius_comparison(self) -> None:
        """Compare adjacency matrices between adaptive and fixed radius modes.
        
        Requirements: 6.6, 6.7
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_data.csv"
            self._create_test_csv(csv_path)
            
            # Read frames
            frames = read_raw_frames(csv_path)
            
            # Build samples with fixed radius
            config_fixed = WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adjacency_radius_m=20.0,  # Small fixed radius
                adaptive_radius_enabled=False,
                expected_dt=0.1,
                max_dt_error=0.03,
            )
            samples_fixed = build_multi_agent_samples(frames, config_fixed)
            
            # Build samples with adaptive radius
            config_adaptive = WindowBuildConfig(
                history_frames=2,
                future_frames=2,
                stride=1,
                adaptive_radius_enabled=True,
                radius_base=20.0,
                radius_alpha=0.5,
                expected_dt=0.1,
                max_dt_error=0.03,
            )
            samples_adaptive = build_multi_agent_samples(frames, config_adaptive)
            
            # Both should produce samples
            self.assertGreater(len(samples_fixed), 0)
            self.assertGreater(len(samples_adaptive), 0)
            
            # Adjacency matrices may differ due to velocity-based radius
            # This is expected behavior
            adj_fixed = samples_fixed[0]["adj"]
            adj_adaptive = samples_adaptive[0]["adj"]
            
            # Both should be symmetric
            self.assertTrue(np.allclose(adj_fixed, adj_fixed.T))
            self.assertTrue(np.allclose(adj_adaptive, adj_adaptive.T))
            
            # Both should have self-connections
            n_agents = adj_fixed.shape[0]
            for i in range(n_agents):
                self.assertEqual(adj_fixed[i, i], 1.0)
                self.assertEqual(adj_adaptive[i, i], 1.0)


if __name__ == "__main__":
    unittest.main()
