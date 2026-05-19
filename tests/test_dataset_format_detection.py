"""
Tests for dataset loader format detection (backward compatibility).

Tests verify that the dataset loader can handle both:
- Old format: 4D features (local_x, local_y, heading_x, heading_y) without velocity
- New format: 6D features (local_x, local_y, local_vx, local_vy, heading_x, heading_y) with velocity
"""

import tempfile
from pathlib import Path

import pytest
import torch

from core_perception.multi_agent_dataset import MultiAgentTrajectoryDataset


class TestFormatDetection:
    """Test dataset loader format detection for backward compatibility."""

    def test_new_format_6d_features(self, tmp_path: Path) -> None:
        """Test that new format (6D features with velocity) is loaded correctly."""
        # Create a sample with 6D features (new format)
        sample = {
            "x": torch.randn(3, 20, 6),  # 3 agents, 20 history steps, 6 features
            "y": torch.randn(3, 30, 2),  # 3 agents, 30 future steps, 2 coords
            "adj": torch.ones(3, 3),
            "x_mask": torch.ones(3, 20, dtype=torch.bool),
            "y_mask": torch.ones(3, 30, dtype=torch.bool),
            "actor_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "ego_pose": torch.tensor([0.0, 0.0, 0.0]),
            "anchor_frame": 100,
            "anchor_timestamp": 10.0,
            "town": "Town01",
            "run_id": "test_run",
        }
        
        # Save sample
        sample_path = tmp_path / "sample_000000.pt"
        torch.save(sample, sample_path)
        
        # Create manifest
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "sample_file,anchor_frame,anchor_timestamp,town,run_id,num_agents\n"
            "sample_000000.pt,100,10.0,Town01,test_run,3\n"
        )
        
        # Load dataset
        dataset = MultiAgentTrajectoryDataset(tmp_path)
        loaded = dataset[0]
        
        # Verify features are still 6D
        assert loaded["x"].shape == (3, 20, 6)
        # Verify data is preserved
        torch.testing.assert_close(loaded["x"], sample["x"])

    def test_old_format_4d_features(self, tmp_path: Path) -> None:
        """Test that old format (4D features without velocity) is converted to 6D."""
        # Create a sample with 4D features (old format)
        x_4d = torch.randn(3, 20, 4)  # 3 agents, 20 history steps, 4 features
        sample = {
            "x": x_4d,
            "y": torch.randn(3, 30, 2),
            "adj": torch.ones(3, 3),
            "x_mask": torch.ones(3, 20, dtype=torch.bool),
            "y_mask": torch.ones(3, 30, dtype=torch.bool),
            "actor_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "ego_pose": torch.tensor([0.0, 0.0, 0.0]),
            "anchor_frame": 100,
            "anchor_timestamp": 10.0,
            "town": "Town01",
            "run_id": "test_run",
        }
        
        # Save sample
        sample_path = tmp_path / "sample_000000.pt"
        torch.save(sample, sample_path)
        
        # Create manifest
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "sample_file,anchor_frame,anchor_timestamp,town,run_id,num_agents\n"
            "sample_000000.pt,100,10.0,Town01,test_run,3\n"
        )
        
        # Load dataset
        dataset = MultiAgentTrajectoryDataset(tmp_path)
        loaded = dataset[0]
        
        # Verify features are converted to 6D
        assert loaded["x"].shape == (3, 20, 6)
        
        # Verify conversion is correct:
        # (local_x, local_y, local_vx=0, local_vy=0, heading_x, heading_y)
        torch.testing.assert_close(loaded["x"][:, :, 0:2], x_4d[:, :, 0:2])  # local_x, local_y preserved
        torch.testing.assert_close(loaded["x"][:, :, 2:4], torch.zeros(3, 20, 2))  # velocity padded with zeros
        torch.testing.assert_close(loaded["x"][:, :, 4:6], x_4d[:, :, 2:4])  # heading_x, heading_y preserved

    def test_invalid_feature_dimension(self, tmp_path: Path) -> None:
        """Test that invalid feature dimensions raise an error."""
        # Create a sample with invalid 5D features
        sample = {
            "x": torch.randn(3, 20, 5),  # Invalid: neither 4 nor 6
            "y": torch.randn(3, 30, 2),
            "adj": torch.ones(3, 3),
            "x_mask": torch.ones(3, 20, dtype=torch.bool),
            "y_mask": torch.ones(3, 30, dtype=torch.bool),
            "actor_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "ego_pose": torch.tensor([0.0, 0.0, 0.0]),
            "anchor_frame": 100,
            "anchor_timestamp": 10.0,
            "town": "Town01",
            "run_id": "test_run",
        }
        
        # Save sample
        sample_path = tmp_path / "sample_000000.pt"
        torch.save(sample, sample_path)
        
        # Create manifest
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "sample_file,anchor_frame,anchor_timestamp,town,run_id,num_agents\n"
            "sample_000000.pt,100,10.0,Town01,test_run,3\n"
        )
        
        # Load dataset
        dataset = MultiAgentTrajectoryDataset(tmp_path)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Unexpected feature dimension: 5"):
            _ = dataset[0]

    def test_mixed_format_dataset(self, tmp_path: Path) -> None:
        """Test that dataset can handle mix of old and new format samples."""
        # Create old format sample
        sample_old = {
            "x": torch.randn(2, 20, 4),  # Old format: 4D
            "y": torch.randn(2, 30, 2),
            "adj": torch.ones(2, 2),
            "x_mask": torch.ones(2, 20, dtype=torch.bool),
            "y_mask": torch.ones(2, 30, dtype=torch.bool),
            "actor_ids": torch.tensor([1, 2], dtype=torch.long),
            "ego_pose": torch.tensor([0.0, 0.0, 0.0]),
            "anchor_frame": 100,
            "anchor_timestamp": 10.0,
            "town": "Town01",
            "run_id": "test_run",
        }
        
        # Create new format sample
        sample_new = {
            "x": torch.randn(3, 20, 6),  # New format: 6D
            "y": torch.randn(3, 30, 2),
            "adj": torch.ones(3, 3),
            "x_mask": torch.ones(3, 20, dtype=torch.bool),
            "y_mask": torch.ones(3, 30, dtype=torch.bool),
            "actor_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "ego_pose": torch.tensor([0.0, 0.0, 0.0]),
            "anchor_frame": 110,
            "anchor_timestamp": 11.0,
            "town": "Town01",
            "run_id": "test_run",
        }
        
        # Save samples
        torch.save(sample_old, tmp_path / "sample_000000.pt")
        torch.save(sample_new, tmp_path / "sample_000001.pt")
        
        # Create manifest
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "sample_file,anchor_frame,anchor_timestamp,town,run_id,num_agents\n"
            "sample_000000.pt,100,10.0,Town01,test_run,2\n"
            "sample_000001.pt,110,11.0,Town01,test_run,3\n"
        )
        
        # Load dataset
        dataset = MultiAgentTrajectoryDataset(tmp_path)
        assert len(dataset) == 2
        
        # Load both samples
        loaded_old = dataset[0]
        loaded_new = dataset[1]
        
        # Both should have 6D features
        assert loaded_old["x"].shape == (2, 20, 6)
        assert loaded_new["x"].shape == (3, 20, 6)
        
        # Old format should have zero velocities
        torch.testing.assert_close(loaded_old["x"][:, :, 2:4], torch.zeros(2, 20, 2))
        
        # New format should preserve original velocities
        torch.testing.assert_close(loaded_new["x"], sample_new["x"])

    def test_ensure_6d_features_static_method(self) -> None:
        """Test the _ensure_6d_features static method directly."""
        # Test 6D input (new format)
        x_6d = torch.randn(5, 10, 6)
        result = MultiAgentTrajectoryDataset._ensure_6d_features(x_6d)
        assert result.shape == (5, 10, 6)
        torch.testing.assert_close(result, x_6d)
        
        # Test 4D input (old format)
        x_4d = torch.randn(5, 10, 4)
        result = MultiAgentTrajectoryDataset._ensure_6d_features(x_4d)
        assert result.shape == (5, 10, 6)
        # Check structure: (local_x, local_y, 0, 0, heading_x, heading_y)
        torch.testing.assert_close(result[:, :, 0:2], x_4d[:, :, 0:2])
        torch.testing.assert_close(result[:, :, 2:4], torch.zeros(5, 10, 2))
        torch.testing.assert_close(result[:, :, 4:6], x_4d[:, :, 2:4])
        
        # Test invalid input
        x_invalid = torch.randn(5, 10, 3)
        with pytest.raises(ValueError, match="Unexpected feature dimension: 3"):
            MultiAgentTrajectoryDataset._ensure_6d_features(x_invalid)
