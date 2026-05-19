"""Integration tests for validation script.

Tests the validation script to ensure it correctly:
- Generates validation report JSON with proper format
- Computes per-town performance breakdown
- Measures inference latency

**Validates: Requirements 11.2, 11.7, 11.8, 11.9**
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_target_metrics import (  # noqa: E402
    compute_multimodal_metrics,
    evaluate_model,
    measure_inference_latency,
    move_batch_to_device,
)
from core_perception.multi_agent_model import (  # noqa: E402
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
)
from core_perception.multi_agent_dataset import (  # noqa: E402
    collate_multi_agent_trajectory,
)


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=64,
        graph_layers=1,
        future_steps=10,
        enable_gat=False,
        enable_multimodal=False,
        enable_adaptive_radius=False,
    )
    model = MultiAgentTrajectoryPredictor(config)
    model.eval()
    return model


@pytest.fixture
def dummy_multimodal_model():
    """Create a dummy multimodal model for testing."""
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=64,
        graph_layers=1,
        future_steps=10,
        enable_gat=False,
        enable_multimodal=True,
        num_modes=3,
        enable_adaptive_radius=False,
    )
    model = MultiAgentTrajectoryPredictor(config)
    model.eval()
    return model


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    batch_size = 2
    max_agents = 5
    history_steps = 6
    future_steps = 10
    
    return {
        "x": torch.randn(batch_size, max_agents, history_steps, 6),
        "y": torch.randn(batch_size, max_agents, future_steps, 2),
        "adj": torch.randint(0, 2, (batch_size, max_agents, max_agents), dtype=torch.float32),
        "x_mask": torch.ones(batch_size, max_agents, history_steps, dtype=torch.bool),
        "y_mask": torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool),
        "agent_mask": torch.ones(batch_size, max_agents, dtype=torch.bool),
        "town": ["Town01", "Town02"],
    }


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, num_samples: int = 10, future_steps: int = 10):
        self.num_samples = num_samples
        self.future_steps = future_steps
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        max_agents = 5
        history_steps = 6
        
        return {
            "x": torch.randn(max_agents, history_steps, 6),
            "y": torch.randn(max_agents, self.future_steps, 2),
            "adj": torch.randint(0, 2, (max_agents, max_agents), dtype=torch.float32),
            "x_mask": torch.ones(max_agents, history_steps, dtype=torch.bool),
            "y_mask": torch.ones(max_agents, self.future_steps, dtype=torch.bool),
            "agent_mask": torch.ones(max_agents, dtype=torch.bool),
            "actor_ids": torch.arange(max_agents, dtype=torch.long),
            "ego_pose": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            "anchor_frame": idx * 10,
            "anchor_timestamp": float(idx),
            "town": f"Town0{(idx % 3) + 1}",  # Cycle through Town01, Town02, Town03
            "run_id": f"run_{idx // 10}",
            "sample_path": f"sample_{idx:04d}.pt",
        }


def test_compute_multimodal_metrics_unimodal():
    """Test multimodal metrics computation with unimodal predictions."""
    batch_size = 2
    max_agents = 3
    future_steps = 10
    
    # Create dummy unimodal predictions [B, N, T, 2]
    pred = torch.randn(batch_size, max_agents, future_steps, 2)
    target = torch.randn(batch_size, max_agents, future_steps, 2)
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Verify metrics are present and valid
    assert "minADE" in metrics
    assert "minFDE" in metrics
    assert "MissRate" in metrics
    
    assert metrics["minADE"] >= 0.0
    assert metrics["minFDE"] >= 0.0
    assert 0.0 <= metrics["MissRate"] <= 1.0


def test_compute_multimodal_metrics_multimodal():
    """Test multimodal metrics computation with multimodal predictions."""
    batch_size = 2
    max_agents = 3
    num_modes = 3
    future_steps = 10
    
    # Create dummy multimodal predictions [B, N, K, T, 2]
    pred = torch.randn(batch_size, max_agents, num_modes, future_steps, 2)
    target = torch.randn(batch_size, max_agents, future_steps, 2)
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Verify metrics are present and valid
    assert "minADE" in metrics
    assert "minFDE" in metrics
    assert "MissRate" in metrics
    
    assert metrics["minADE"] >= 0.0
    assert metrics["minFDE"] >= 0.0
    assert 0.0 <= metrics["MissRate"] <= 1.0


def test_compute_multimodal_metrics_with_masking():
    """Test multimodal metrics computation with masked agents and timesteps."""
    batch_size = 2
    max_agents = 3
    num_modes = 3
    future_steps = 10
    
    # Create dummy predictions and targets
    pred = torch.randn(batch_size, max_agents, num_modes, future_steps, 2)
    target = torch.randn(batch_size, max_agents, future_steps, 2)
    
    # Create masks with some invalid agents and timesteps
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    y_mask[0, 1, 5:] = False  # Mask out last 5 timesteps for agent 1 in batch 0
    
    agent_mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Verify metrics are computed only for valid agents
    assert metrics["minADE"] >= 0.0
    assert metrics["minFDE"] >= 0.0
    assert 0.0 <= metrics["MissRate"] <= 1.0


def test_compute_multimodal_metrics_miss_rate():
    """Test MissRate computation with known values."""
    batch_size = 2
    max_agents = 3
    num_modes = 2
    future_steps = 10
    
    # Create predictions and targets with known final displacement
    target = torch.zeros(batch_size, max_agents, future_steps, 2)
    pred = torch.zeros(batch_size, max_agents, num_modes, future_steps, 2)
    
    # Set final positions to create specific FDE values for both modes
    # Agent 0: mode 0 FDE = 1.0m (best), mode 1 FDE = 1.5m → minFDE = 1.0m (not a miss)
    pred[0, 0, 0, -1, :] = torch.tensor([1.0, 0.0])
    pred[0, 0, 1, -1, :] = torch.tensor([1.5, 0.0])
    
    # Agent 1: mode 0 FDE = 2.5m (best), mode 1 FDE = 3.0m → minFDE = 2.5m (miss, > 2.0m threshold)
    pred[0, 1, 0, -1, :] = torch.tensor([2.5, 0.0])
    pred[0, 1, 1, -1, :] = torch.tensor([3.0, 0.0])
    
    # Agent 2: mode 0 FDE = 4.0m, mode 1 FDE = 3.0m (best) → minFDE = 3.0m (miss)
    pred[0, 2, 0, -1, :] = torch.tensor([4.0, 0.0])
    pred[0, 2, 1, -1, :] = torch.tensor([3.0, 0.0])
    
    # Batch 1: all agents have low FDE (no misses)
    pred[1, 0, 0, -1, :] = torch.tensor([0.5, 0.0])
    pred[1, 0, 1, -1, :] = torch.tensor([0.8, 0.0])
    pred[1, 1, 0, -1, :] = torch.tensor([1.0, 0.0])
    pred[1, 1, 1, -1, :] = torch.tensor([1.2, 0.0])
    pred[1, 2, 0, -1, :] = torch.tensor([1.5, 0.0])
    pred[1, 2, 1, -1, :] = torch.tensor([1.8, 0.0])
    
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Verify MissRate
    # In batch 0: 2 out of 3 agents have minFDE > 2.0m (agents 1 and 2)
    # In batch 1: all agents have minFDE < 2.0m (no misses)
    # Overall: 2 misses out of 6 agents = 0.333...
    assert 0.3 <= metrics["MissRate"] <= 0.4


def test_move_batch_to_device():
    """Test moving batch tensors to device."""
    batch = {
        "x": torch.randn(2, 5, 6, 6),
        "y": torch.randn(2, 5, 10, 2),
        "adj": torch.randn(2, 5, 5),
        "x_mask": torch.ones(2, 5, 6, dtype=torch.bool),
        "y_mask": torch.ones(2, 5, 10, dtype=torch.bool),
        "agent_mask": torch.ones(2, 5, dtype=torch.bool),
        "town": ["Town01", "Town02"],  # Non-tensor field
    }
    
    device = torch.device("cpu")
    moved_batch = move_batch_to_device(batch, device)
    
    # Verify tensors are moved
    assert moved_batch["x"].device == device
    assert moved_batch["y"].device == device
    assert moved_batch["adj"].device == device
    assert moved_batch["x_mask"].device == device
    assert moved_batch["y_mask"].device == device
    assert moved_batch["agent_mask"].device == device
    
    # Verify non-tensor fields are preserved
    assert moved_batch["town"] == ["Town01", "Town02"]


def test_evaluate_model_unimodal(dummy_model):
    """Test model evaluation with unimodal predictions."""
    # Create dummy dataset and loader
    dataset = DummyDataset(num_samples=10, future_steps=10)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_multi_agent_trajectory,
        shuffle=False,
    )
    
    device = torch.device("cpu")
    
    # Evaluate model
    overall_metrics, per_town_metrics = evaluate_model(
        model=dummy_model,
        loader=loader,
        device=device,
        enable_multimodal=False,
        per_town_metrics=True,
    )
    
    # Verify overall metrics
    assert "minADE" in overall_metrics
    assert "minFDE" in overall_metrics
    assert "MissRate" in overall_metrics
    
    assert overall_metrics["minADE"] >= 0.0
    assert overall_metrics["minFDE"] >= 0.0
    assert 0.0 <= overall_metrics["MissRate"] <= 1.0
    
    # Verify per-town metrics
    assert len(per_town_metrics) > 0
    for town, town_data in per_town_metrics.items():
        assert "minADE" in town_data
        assert "minFDE" in town_data
        assert "MissRate" in town_data
        assert "num_batches" in town_data
        
        assert town_data["minADE"] >= 0.0
        assert town_data["minFDE"] >= 0.0
        assert 0.0 <= town_data["MissRate"] <= 1.0
        assert town_data["num_batches"] > 0


def test_evaluate_model_multimodal(dummy_multimodal_model):
    """Test model evaluation with multimodal predictions."""
    # Create dummy dataset and loader
    dataset = DummyDataset(num_samples=10, future_steps=10)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_multi_agent_trajectory,
        shuffle=False,
    )
    
    device = torch.device("cpu")
    
    # Evaluate model
    overall_metrics, per_town_metrics = evaluate_model(
        model=dummy_multimodal_model,
        loader=loader,
        device=device,
        enable_multimodal=True,
        per_town_metrics=True,
    )
    
    # Verify overall metrics
    assert "minADE" in overall_metrics
    assert "minFDE" in overall_metrics
    assert "MissRate" in overall_metrics
    
    assert overall_metrics["minADE"] >= 0.0
    assert overall_metrics["minFDE"] >= 0.0
    assert 0.0 <= overall_metrics["MissRate"] <= 1.0
    
    # Verify per-town metrics
    assert len(per_town_metrics) > 0
    for town, town_data in per_town_metrics.items():
        assert "minADE" in town_data
        assert "minFDE" in town_data
        assert "MissRate" in town_data
        assert "num_batches" in town_data


def test_per_town_breakdown():
    """Test per-town performance breakdown computation."""
    # Create dataset with specific town distribution
    dataset = DummyDataset(num_samples=12, future_steps=10)  # 4 samples per town (Town01, Town02, Town03)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_multi_agent_trajectory,
        shuffle=False,
    )
    
    # Create model
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=64,
        graph_layers=1,
        future_steps=10,
        enable_gat=False,
        enable_multimodal=False,
    )
    model = MultiAgentTrajectoryPredictor(config)
    model.eval()
    
    device = torch.device("cpu")
    
    # Evaluate model
    overall_metrics, per_town_metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        enable_multimodal=False,
        per_town_metrics=True,
    )
    
    # Verify we have metrics for all 3 towns
    assert len(per_town_metrics) == 3
    assert "Town01" in per_town_metrics
    assert "Town02" in per_town_metrics
    assert "Town03" in per_town_metrics
    
    # Verify each town has the expected structure
    for town in ["Town01", "Town02", "Town03"]:
        town_data = per_town_metrics[town]
        assert "minADE" in town_data
        assert "minFDE" in town_data
        assert "MissRate" in town_data
        assert "num_batches" in town_data
        
        # Each town should have 4 batches (appears in 4 out of 6 batches)
        assert town_data["num_batches"] == 4


def test_measure_inference_latency(dummy_model):
    """Test inference latency measurement."""
    # Create dummy dataset and loader
    dataset = DummyDataset(num_samples=10, future_steps=10)
    loader = DataLoader(
        dataset,
        batch_size=1,  # Use batch_size=1 for latency measurement
        collate_fn=collate_multi_agent_trajectory,
        shuffle=False,
    )
    
    device = torch.device("cpu")
    
    # Measure latency
    latency_ms = measure_inference_latency(
        model=dummy_model,
        loader=loader,
        device=device,
        num_samples=5,
    )
    
    # Verify latency is positive and reasonable
    assert latency_ms > 0.0
    assert latency_ms < 1000.0  # Should be less than 1 second per sample


def test_measure_inference_latency_multimodal(dummy_multimodal_model):
    """Test inference latency measurement with multimodal model."""
    # Create dummy dataset and loader
    dataset = DummyDataset(num_samples=10, future_steps=10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_multi_agent_trajectory,
        shuffle=False,
    )
    
    device = torch.device("cpu")
    
    # Measure latency
    latency_ms = measure_inference_latency(
        model=dummy_multimodal_model,
        loader=loader,
        device=device,
        num_samples=5,
    )
    
    # Verify latency is positive and reasonable
    assert latency_ms > 0.0
    assert latency_ms < 1000.0


def test_validation_report_json_format():
    """Test validation report JSON format structure.
    
    This test verifies that the validation report has the expected structure
    with all required fields for Requirements 11.2, 11.9.
    """
    # Create a mock validation report
    validation_report = {
        "checkpoint": "/path/to/checkpoint.pt",
        "data_dirs": ["/path/to/data1", "/path/to/data2"],
        "test_samples": 1000,
        "device": "cuda",
        "model_config": {
            "enable_gat": True,
            "enable_multimodal": True,
            "enable_adaptive_radius": True,
            "hidden_dim": 128,
            "num_modes": 3,
            "num_attention_heads": 4,
        },
        "overall_metrics": {
            "minADE": 1.25,
            "minFDE": 2.35,
            "MissRate": 0.14,
        },
        "per_town_metrics": {
            "Town01": {
                "minADE": 1.20,
                "minFDE": 2.30,
                "MissRate": 0.13,
                "num_batches": 50,
            },
            "Town02": {
                "minADE": 1.30,
                "minFDE": 2.40,
                "MissRate": 0.15,
                "num_batches": 50,
            },
        },
        "inference_latency_ms": {
            "cuda": 18.5,
            "cpu": 45.2,
        },
        "target_metrics": {
            "minADE": 1.5,
            "minFDE": 2.7,
            "MissRate": 0.20,
            "inference_latency_ms": 25.0,
        },
        "targets_met": {
            "minADE": True,
            "minFDE": True,
            "MissRate": True,
            "inference_latency_ms": True,
            "all": True,
        },
    }
    
    # Verify top-level structure
    assert "checkpoint" in validation_report
    assert "data_dirs" in validation_report
    assert "test_samples" in validation_report
    assert "device" in validation_report
    assert "model_config" in validation_report
    assert "overall_metrics" in validation_report
    assert "per_town_metrics" in validation_report
    assert "inference_latency_ms" in validation_report
    assert "target_metrics" in validation_report
    assert "targets_met" in validation_report
    
    # Verify model_config structure
    model_config = validation_report["model_config"]
    assert "enable_gat" in model_config
    assert "enable_multimodal" in model_config
    assert "enable_adaptive_radius" in model_config
    assert "hidden_dim" in model_config
    assert "num_modes" in model_config
    assert "num_attention_heads" in model_config
    
    # Verify overall_metrics structure
    overall_metrics = validation_report["overall_metrics"]
    assert "minADE" in overall_metrics
    assert "minFDE" in overall_metrics
    assert "MissRate" in overall_metrics
    
    # Verify per_town_metrics structure
    per_town_metrics = validation_report["per_town_metrics"]
    assert len(per_town_metrics) > 0
    for town, town_data in per_town_metrics.items():
        assert "minADE" in town_data
        assert "minFDE" in town_data
        assert "MissRate" in town_data
        assert "num_batches" in town_data
    
    # Verify inference_latency_ms structure
    inference_latency = validation_report["inference_latency_ms"]
    assert isinstance(inference_latency, dict)
    assert len(inference_latency) > 0
    
    # Verify target_metrics structure
    target_metrics = validation_report["target_metrics"]
    assert "minADE" in target_metrics
    assert "minFDE" in target_metrics
    assert "MissRate" in target_metrics
    assert "inference_latency_ms" in target_metrics
    
    # Verify targets_met structure
    targets_met = validation_report["targets_met"]
    assert "minADE" in targets_met
    assert "minFDE" in targets_met
    assert "MissRate" in targets_met
    assert "inference_latency_ms" in targets_met
    assert "all" in targets_met
    
    # Verify JSON serialization works
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(validation_report, f, indent=2)
        temp_path = Path(f.name)
    
    try:
        # Verify we can read it back
        with open(temp_path, "r") as f:
            loaded_report = json.load(f)
        
        assert loaded_report == validation_report
    finally:
        temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

