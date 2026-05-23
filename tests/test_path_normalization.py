"""
Test path normalization in MultiAgentTrajectoryDataset.

This test ensures that the dataset correctly handles both Windows backslash
and Unix forward slash path separators, making it cross-platform compatible.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_dataset import MultiAgentTrajectoryDataset


@pytest.fixture
def temp_dataset_dir(tmp_path: Path):
    """Create a temporary dataset with sample files and manifest."""
    # Create sample data
    sample_data = {
        "x": torch.randn(5, 20, 6),  # 5 agents, 20 history frames, 6 features
        "y": torch.randn(5, 30, 2),  # 5 agents, 30 future frames, 2 coords
        "adj": torch.ones(5, 5),
        "x_mask": torch.ones(5, 20, dtype=torch.bool),
        "y_mask": torch.ones(5, 30, dtype=torch.bool),
        "actor_ids": torch.arange(5, dtype=torch.long),
        "ego_pose": torch.tensor([0.0, 0.0, 0.0]),
        "anchor_frame": 20,
        "anchor_timestamp": 2.0,
        "town": "TestTown",
        "run_id": "test_run_001",
    }
    
    # Create subdirectory for town
    town_dir = tmp_path / "TestTown"
    town_dir.mkdir()
    
    # Save sample files
    sample_files = []
    for i in range(3):
        sample_file = town_dir / f"sample_{i:06d}.pt"
        torch.save(sample_data, sample_file)
        sample_files.append(sample_file)
    
    return tmp_path, town_dir, sample_files


def test_manifest_with_backslashes(temp_dataset_dir):
    """Test that dataset handles manifest with Windows backslashes."""
    tmp_path, town_dir, sample_files = temp_dataset_dir
    
    # Create manifest with backslashes (Windows style)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_file", "anchor_frame", "town"])
        writer.writeheader()
        for i, sample_file in enumerate(sample_files):
            # Use backslash separator (Windows style)
            relative_path = f"TestTown\\sample_{i:06d}.pt"
            writer.writerow({
                "sample_file": relative_path,
                "anchor_frame": 20,
                "town": "TestTown",
            })
    
    # Load dataset - should work despite backslashes
    dataset = MultiAgentTrajectoryDataset(root_dir=tmp_path, manifest_path=manifest_path)
    
    assert len(dataset) == 3
    
    # Load first sample - should not raise FileNotFoundError
    sample = dataset[0]
    assert sample["x"].shape == (5, 20, 6)
    assert sample["town"] == "TestTown"


def test_manifest_with_forward_slashes(temp_dataset_dir):
    """Test that dataset handles manifest with Unix forward slashes."""
    tmp_path, town_dir, sample_files = temp_dataset_dir
    
    # Create manifest with forward slashes (Unix style)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_file", "anchor_frame", "town"])
        writer.writeheader()
        for i, sample_file in enumerate(sample_files):
            # Use forward slash separator (Unix style)
            relative_path = f"TestTown/sample_{i:06d}.pt"
            writer.writerow({
                "sample_file": relative_path,
                "anchor_frame": 20,
                "town": "TestTown",
            })
    
    # Load dataset
    dataset = MultiAgentTrajectoryDataset(root_dir=tmp_path, manifest_path=manifest_path)
    
    assert len(dataset) == 3
    
    # Load first sample
    sample = dataset[0]
    assert sample["x"].shape == (5, 20, 6)
    assert sample["town"] == "TestTown"


def test_sample_files_with_backslashes(temp_dataset_dir):
    """Test that dataset handles sample_files parameter with backslashes."""
    tmp_path, town_dir, sample_files = temp_dataset_dir
    
    # Create sample_files list with backslash paths
    sample_files_with_backslash = [
        f"TestTown\\sample_{i:06d}.pt" for i in range(3)
    ]
    
    # Load dataset with sample_files parameter
    dataset = MultiAgentTrajectoryDataset(
        root_dir=tmp_path,
        sample_files=sample_files_with_backslash
    )
    
    assert len(dataset) == 3
    
    # Load first sample - should not raise FileNotFoundError
    sample = dataset[0]
    assert sample["x"].shape == (5, 20, 6)
    assert sample["town"] == "TestTown"


def test_sample_files_with_forward_slashes(temp_dataset_dir):
    """Test that dataset handles sample_files parameter with forward slashes."""
    tmp_path, town_dir, sample_files = temp_dataset_dir
    
    # Create sample_files list with forward slash paths
    sample_files_with_forward_slash = [
        f"TestTown/sample_{i:06d}.pt" for i in range(3)
    ]
    
    # Load dataset with sample_files parameter
    dataset = MultiAgentTrajectoryDataset(
        root_dir=tmp_path,
        sample_files=sample_files_with_forward_slash
    )
    
    assert len(dataset) == 3
    
    # Load first sample
    sample = dataset[0]
    assert sample["x"].shape == (5, 20, 6)
    assert sample["town"] == "TestTown"


def test_mixed_path_separators(temp_dataset_dir):
    """Test that dataset handles mixed path separators."""
    tmp_path, town_dir, sample_files = temp_dataset_dir
    
    # Create manifest with mixed separators
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_file", "anchor_frame", "town"])
        writer.writeheader()
        # Mix backslash and forward slash
        writer.writerow({
            "sample_file": "TestTown\\sample_000000.pt",
            "anchor_frame": 20,
            "town": "TestTown",
        })
        writer.writerow({
            "sample_file": "TestTown/sample_000001.pt",
            "anchor_frame": 21,
            "town": "TestTown",
        })
        writer.writerow({
            "sample_file": "TestTown\\sample_000002.pt",
            "anchor_frame": 22,
            "town": "TestTown",
        })
    
    # Load dataset - should handle mixed separators
    dataset = MultiAgentTrajectoryDataset(root_dir=tmp_path, manifest_path=manifest_path)
    
    assert len(dataset) == 3
    
    # Load all samples - none should raise FileNotFoundError
    for i in range(3):
        sample = dataset[i]
        assert sample["x"].shape == (5, 20, 6)
        assert sample["town"] == "TestTown"


def test_path_normalization_in_getitem(temp_dataset_dir):
    """Test that __getitem__ normalizes paths correctly."""
    tmp_path, town_dir, sample_files = temp_dataset_dir
    
    # Create dataset
    dataset = MultiAgentTrajectoryDataset(
        root_dir=tmp_path,
        sample_files=[f"TestTown/sample_{i:06d}.pt" for i in range(3)]
    )
    
    # Manually inject a path with backslash to test __getitem__ normalization
    # This simulates what could happen if Path.resolve() returns a Windows path
    dataset.sample_paths[0] = Path(str(dataset.sample_paths[0]).replace("/", "\\"))
    
    # Load sample - should still work due to normalization in __getitem__
    sample = dataset[0]
    assert sample["x"].shape == (5, 20, 6)
    assert sample["town"] == "TestTown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
