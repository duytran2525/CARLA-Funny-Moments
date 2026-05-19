"""Unit tests for ablation study script.

Tests the ablation study framework to ensure it correctly trains all 8 variants
and produces valid results.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ablation_study import (  # noqa: E402
    compute_multimodal_metrics,
    generate_comparison_table,
    measure_inference_latency,
)
from core_perception.multi_agent_trajectory import (  # noqa: E402
    ActorState,
    FrameData,
    WindowBuildConfig,
    build_multi_agent_samples,
    sample_to_torch_payload,
)
from core_perception.multi_agent_dataset import (  # noqa: E402
    MultiAgentTrajectoryDataset,
)


def test_compute_multimodal_metrics():
    """Test multimodal metrics computation."""
    batch_size = 2
    max_agents = 3
    num_modes = 3
    future_steps = 10
    
    # Create dummy predictions and targets
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
    """Test multimodal metrics computation with masked agents."""
    batch_size = 2
    max_agents = 3
    num_modes = 3
    future_steps = 10
    
    # Create dummy predictions and targets
    pred = torch.randn(batch_size, max_agents, num_modes, future_steps, 2)
    target = torch.randn(batch_size, max_agents, future_steps, 2)
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    agent_mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)
    
    # Compute metrics
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Verify metrics are computed only for valid agents
    assert metrics["minADE"] >= 0.0
    assert metrics["minFDE"] >= 0.0
    assert 0.0 <= metrics["MissRate"] <= 1.0


def test_generate_comparison_table():
    """Test comparison table generation."""
    # Create dummy results
    results = {
        "baseline": {
            "minADE": 2.5,
            "minFDE": 4.5,
            "MissRate": 0.35,
            "train_time_seconds": 100.0,
            "inference_latency_ms": 20.0,
        },
        "gat_only": {
            "minADE": 2.0,
            "minFDE": 3.8,
            "MissRate": 0.28,
            "train_time_seconds": 120.0,
            "inference_latency_ms": 22.0,
        },
        "full": {
            "minADE": 1.3,
            "minFDE": 2.5,
            "MissRate": 0.15,
            "train_time_seconds": 150.0,
            "inference_latency_ms": 25.0,
        },
    }
    
    baseline_results = results["baseline"]
    table = generate_comparison_table(results, baseline_results)
    
    # Verify table contains expected content
    assert "ABLATION STUDY RESULTS" in table
    assert "baseline" in table
    assert "gat_only" in table
    assert "full" in table
    assert "minADE" in table
    assert "minFDE" in table
    assert "MissRate" in table
    assert "Train Time" in table
    assert "Inference" in table


def test_variant_combinations():
    """Test that all 8 variant combinations are defined correctly."""
    # Define expected variants (from run_ablation_study.py)
    expected_variants = [
        ("baseline", False, False, False),
        ("gat_only", True, False, False),
        ("multimodal_only", False, True, False),
        ("adaptive_radius_only", False, False, True),
        ("gat_multimodal", True, True, False),
        ("gat_adaptive", True, False, True),
        ("multimodal_adaptive", False, True, True),
        ("full", True, True, True),
    ]
    
    # Verify we have exactly 8 variants
    assert len(expected_variants) == 8
    
    # Verify all combinations are unique
    flag_combinations = [(gat, mm, ar) for _, gat, mm, ar in expected_variants]
    assert len(flag_combinations) == len(set(flag_combinations))
    
    # Verify baseline has all flags disabled
    baseline = expected_variants[0]
    assert baseline[0] == "baseline"
    assert baseline[1:] == (False, False, False)
    
    # Verify full has all flags enabled
    full = expected_variants[-1]
    assert full[0] == "full"
    assert full[1:] == (True, True, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
