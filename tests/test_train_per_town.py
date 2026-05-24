"""Tests for per-town training script."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_model import (
    MultiAgentModelConfig,
    MultiAgentTrajectoryPredictor,
)


def test_train_per_town_script_exists():
    """Test that the per-town training script exists."""
    script_path = PROJECT_ROOT / "scripts" / "train_per_town.py"
    assert script_path.exists(), f"Training script not found at {script_path}"


def test_train_per_town_imports():
    """Test that the training script can be imported without errors."""
    import scripts.train_per_town as train_module
    
    # Check that key functions exist
    assert hasattr(train_module, "parse_args")
    assert hasattr(train_module, "set_seed")
    assert hasattr(train_module, "resolve_device")
    assert hasattr(train_module, "filter_samples_by_town")
    assert hasattr(train_module, "compute_multimodal_metrics")
    assert hasattr(train_module, "run_epoch")
    assert hasattr(train_module, "main")


def test_filter_samples_by_town_no_filter():
    """Test that filter_samples_by_town returns all samples when no filter is specified."""
    from scripts.train_per_town import filter_samples_by_town
    
    sample_paths = [Path("sample1.pt"), Path("sample2.pt"), Path("sample3.pt")]
    filtered = filter_samples_by_town(sample_paths, None)
    
    assert filtered == sample_paths


def test_compute_multimodal_metrics_shape():
    """Test that compute_multimodal_metrics returns correct metric keys."""
    from scripts.train_per_town import compute_multimodal_metrics
    
    batch_size = 2
    max_agents = 5
    num_modes = 3
    future_steps = 30
    
    # Create dummy tensors
    pred = torch.randn(batch_size, max_agents, num_modes, future_steps, 2)
    target = torch.randn(batch_size, max_agents, future_steps, 2)
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
    
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Check that required metrics are present
    assert "minADE" in metrics
    assert "minFDE" in metrics
    
    # Check per-mode metrics
    for mode_idx in range(num_modes):
        assert f"mode_{mode_idx}_ADE" in metrics
        assert f"mode_{mode_idx}_FDE" in metrics
    
    # Check that metrics are floats
    assert isinstance(metrics["minADE"], float)
    assert isinstance(metrics["minFDE"], float)


def test_compute_multimodal_metrics_with_masking():
    """Test that compute_multimodal_metrics respects masking."""
    from scripts.train_per_town import compute_multimodal_metrics
    
    batch_size = 2
    max_agents = 3
    num_modes = 3
    future_steps = 10
    
    # Create dummy tensors
    pred = torch.randn(batch_size, max_agents, num_modes, future_steps, 2)
    target = torch.randn(batch_size, max_agents, future_steps, 2)
    y_mask = torch.ones(batch_size, max_agents, future_steps, dtype=torch.bool)
    agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
    
    # Mask out some agents
    agent_mask[0, 2] = False
    agent_mask[1, 1] = False
    
    # Mask out some timesteps
    y_mask[0, 0, 5:] = False
    
    metrics = compute_multimodal_metrics(pred, target, y_mask, agent_mask)
    
    # Metrics should be computed without errors
    assert metrics["minADE"] >= 0.0
    assert metrics["minFDE"] >= 0.0


def test_model_config_serialization():
    """Test that model config can be serialized and deserialized."""
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=128,
        graph_layers=2,
        future_steps=30,
        dropout=0.1,
        enable_gat=True,
        num_attention_heads=4,
        enable_multimodal=True,
        num_modes=3,
        enable_adaptive_radius=True,
    )
    
    # Serialize to JSON
    config_dict = config.to_json()
    
    # Check that all fields are present
    assert config_dict["input_dim"] == 6
    assert config_dict["hidden_dim"] == 128
    assert config_dict["enable_gat"] is True
    assert config_dict["enable_multimodal"] is True
    assert config_dict["enable_adaptive_radius"] is True
    
    # Deserialize from JSON
    config_restored = MultiAgentModelConfig.from_json(config_dict)
    
    # Check that restored config matches original
    assert config_restored.input_dim == config.input_dim
    assert config_restored.hidden_dim == config.hidden_dim
    assert config_restored.enable_gat == config.enable_gat
    assert config_restored.enable_multimodal == config.enable_multimodal
    assert config_restored.enable_adaptive_radius == config.enable_adaptive_radius


def test_resolve_device():
    """Test device resolution logic."""
    from scripts.train_per_town import resolve_device
    
    # Test auto mode
    device = resolve_device("auto")
    assert device.type in ["cuda", "cpu"]
    
    # Test cpu mode
    device = resolve_device("cpu")
    assert device.type == "cpu"
    
    # Test cuda mode (should raise if unavailable)
    if torch.cuda.is_available():
        device = resolve_device("cuda")
        assert device.type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_set_seed():
    """Test that set_seed sets random seeds correctly."""
    from scripts.train_per_town import set_seed
    
    # Set seed and generate random numbers
    set_seed(42)
    rand1 = torch.rand(5)
    
    # Set same seed again
    set_seed(42)
    rand2 = torch.rand(5)
    
    # Should get same random numbers
    assert torch.allclose(rand1, rand2)


def test_checkpoint_save_load():
    """Test that checkpoint saving and loading preserves all training state."""
    from scripts.train_per_town import load_checkpoint
    
    # Create a simple model and optimizer
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=32,
        graph_layers=1,
        future_steps=10,
        dropout=0.1,
    )
    model = MultiAgentTrajectoryPredictor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # Create checkpoint data
    epoch = 5
    best_val_minADE = 1.234
    metrics_history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.5},
        {"epoch": 2, "train_loss": 1.8, "val_loss": 2.3},
        {"epoch": 3, "train_loss": 1.6, "val_loss": 2.1},
        {"epoch": 4, "train_loss": 1.4, "val_loss": 1.9},
        {"epoch": 5, "train_loss": 1.2, "val_loss": 1.7},
    ]
    train_config = {
        "data_dirs": ["/path/to/data"],
        "town_filter": ["Town01"],
        "hyperparameters": {"epochs": 30, "batch_size": 16},
    }
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_minADE": best_val_minADE,
        "model_config": config.to_json(),
        "train_config": train_config,
        "metrics_history": metrics_history,
    }
    
    # Save checkpoint to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = Path(tmp.name)
        torch.save(checkpoint, checkpoint_path)
    
    try:
        # Create new model and optimizer to load into
        new_model = MultiAgentTrajectoryPredictor(config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            new_optimizer, mode="min", factor=0.5, patience=3
        )
        
        # Load checkpoint
        device = torch.device("cpu")
        start_epoch, loaded_best_val, loaded_metrics_history = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
        )
        
        # Verify that training state was restored correctly
        assert start_epoch == epoch + 1, f"Expected start_epoch={epoch + 1}, got {start_epoch}"
        assert abs(loaded_best_val - best_val_minADE) < 1e-6, f"Expected best_val={best_val_minADE}, got {loaded_best_val}"
        assert loaded_metrics_history == metrics_history, "Metrics history mismatch"
        
        # Verify model state was loaded
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters mismatch"
        
        # Verify optimizer state was loaded
        assert len(optimizer.state_dict()["state"]) == len(new_optimizer.state_dict()["state"])
        
    finally:
        # Clean up temporary file
        checkpoint_path.unlink()


def test_checkpoint_contains_required_fields():
    """Test that saved checkpoints contain all required fields."""
    # Create a simple model
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=32,
        graph_layers=1,
        future_steps=10,
        dropout=0.1,
        enable_multimodal=True,
        num_modes=3,
    )
    model = MultiAgentTrajectoryPredictor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 10,
        "best_val_minADE": 1.5,
        "model_config": config.to_json(),
        "train_config": {"data_dirs": ["/path/to/data"]},
        "metrics_history": [{"epoch": 1, "train_loss": 2.0}],
    }
    
    # Verify all required fields are present
    required_fields = [
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_val_minADE",
        "model_config",
        "train_config",
        "metrics_history",
    ]
    
    for field in required_fields:
        assert field in checkpoint, f"Required field '{field}' missing from checkpoint"
    
    # Verify model_config can be deserialized
    restored_config = MultiAgentModelConfig.from_json(checkpoint["model_config"])
    assert restored_config.input_dim == config.input_dim
    assert restored_config.hidden_dim == config.hidden_dim
    assert restored_config.enable_multimodal == config.enable_multimodal


def test_load_checkpoint_missing_file():
    """Test that load_checkpoint handles missing checkpoint file gracefully."""
    from scripts.train_per_town import load_checkpoint
    
    config = MultiAgentModelConfig(input_dim=6, hidden_dim=32, graph_layers=1, future_steps=10)
    model = MultiAgentTrajectoryPredictor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    device = torch.device("cpu")
    
    # Try to load from non-existent file
    non_existent_path = Path("/tmp/non_existent_checkpoint.pt")
    
    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            checkpoint_path=non_existent_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )


def test_resume_training_integration():
    """Test that --resume argument works correctly in an end-to-end scenario."""
    from scripts.train_per_town import load_checkpoint
    
    # Create a model with multimodal enabled
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=32,
        graph_layers=1,
        future_steps=10,
        dropout=0.1,
        enable_multimodal=True,
        num_modes=3,
    )
    model = MultiAgentTrajectoryPredictor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    # Simulate training for a few epochs
    initial_epoch = 7
    initial_best_val = 1.456
    initial_metrics_history = [
        {"epoch": 1, "train_loss": 3.0, "val_loss": 3.2, "train_minADE": 2.8, "val_minADE": 3.0},
        {"epoch": 2, "train_loss": 2.5, "val_loss": 2.7, "train_minADE": 2.3, "val_minADE": 2.5},
        {"epoch": 3, "train_loss": 2.0, "val_loss": 2.2, "train_minADE": 1.9, "val_minADE": 2.1},
        {"epoch": 4, "train_loss": 1.8, "val_loss": 2.0, "train_minADE": 1.7, "val_minADE": 1.9},
        {"epoch": 5, "train_loss": 1.6, "val_loss": 1.8, "train_minADE": 1.5, "val_minADE": 1.7},
        {"epoch": 6, "train_loss": 1.5, "val_loss": 1.6, "train_minADE": 1.4, "val_minADE": 1.5},
        {"epoch": 7, "train_loss": 1.4, "val_loss": 1.5, "train_minADE": 1.3, "val_minADE": 1.456},
    ]
    
    train_config = {
        "data_dirs": ["/path/to/data"],
        "town_filter": ["Town01"],
        "town_name": "Town01",
        "train_samples": 1000,
        "val_samples": 200,
        "device": "cpu",
        "model_config": config.to_json(),
        "hyperparameters": {
            "epochs": 30,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "early_stopping_patience": 8,
        },
    }
    
    # Take a step with optimizer to create state
    dummy_loss = sum(p.sum() for p in model.parameters())
    dummy_loss.backward()
    optimizer.step()
    
    # Step the scheduler
    scheduler.step(initial_best_val)
    
    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": initial_epoch,
        "best_val_minADE": initial_best_val,
        "model_config": config.to_json(),
        "train_config": train_config,
        "metrics_history": initial_metrics_history,
        "val_loss": 1.5,
        "train_loss": 1.4,
        "val_metrics": {"minADE": 1.456, "minFDE": 2.3},
        "train_metrics": {"minADE": 1.3, "minFDE": 2.1},
    }
    
    # Save checkpoint to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = Path(tmp.name)
        torch.save(checkpoint, checkpoint_path)
    
    try:
        # Create fresh model, optimizer, and scheduler
        new_model = MultiAgentTrajectoryPredictor(config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3, weight_decay=1e-4)
        new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            new_optimizer, mode="min", factor=0.5, patience=3
        )
        
        # Load checkpoint (simulating --resume)
        device = torch.device("cpu")
        start_epoch, loaded_best_val, loaded_metrics_history = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
        )
        
        # Verify training should resume from next epoch
        assert start_epoch == initial_epoch + 1, f"Expected start_epoch={initial_epoch + 1}, got {start_epoch}"
        
        # Verify best validation metric was restored
        assert abs(loaded_best_val - initial_best_val) < 1e-6, (
            f"Expected best_val={initial_best_val}, got {loaded_best_val}"
        )
        
        # Verify metrics history was restored
        assert len(loaded_metrics_history) == len(initial_metrics_history), (
            f"Expected {len(initial_metrics_history)} history entries, got {len(loaded_metrics_history)}"
        )
        assert loaded_metrics_history == initial_metrics_history, "Metrics history mismatch"
        
        # Verify model parameters were restored
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters were not restored correctly"
        
        # Verify optimizer state was restored (check that state exists)
        assert len(new_optimizer.state_dict()["state"]) > 0, "Optimizer state was not restored"
        
        # Verify scheduler state was restored
        assert new_scheduler.state_dict()["best"] == scheduler.state_dict()["best"], (
            "Scheduler state was not restored correctly"
        )
        
        print(f"✓ Resume training test passed: will resume from epoch {start_epoch}")
        print(f"✓ Best validation minADE: {loaded_best_val:.4f}")
        print(f"✓ Metrics history: {len(loaded_metrics_history)} epochs")
        
    finally:
        # Clean up temporary file
        checkpoint_path.unlink()


def test_checkpoint_backward_compatibility():
    """Test that checkpoints without certain fields can still be loaded."""
    from scripts.train_per_town import load_checkpoint
    
    config = MultiAgentModelConfig(
        input_dim=6,
        hidden_dim=32,
        graph_layers=1,
        future_steps=10,
        dropout=0.1,
    )
    model = MultiAgentTrajectoryPredictor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # Create a legacy checkpoint without best_val_minADE (only val_loss)
    legacy_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 5,
        "val_loss": 2.5,  # Legacy field
        "model_config": config.to_json(),
        "train_config": {"data_dirs": ["/path/to/data"]},
        # Note: metrics_history is missing (should default to empty list)
    }
    
    # Save legacy checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = Path(tmp.name)
        torch.save(legacy_checkpoint, checkpoint_path)
    
    try:
        # Create new model and optimizer
        new_model = MultiAgentTrajectoryPredictor(config)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            new_optimizer, mode="min", factor=0.5, patience=3
        )
        
        # Load legacy checkpoint
        device = torch.device("cpu")
        start_epoch, loaded_best_val, loaded_metrics_history = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
        )
        
        # Verify that defaults were used for missing fields
        assert start_epoch == 6, f"Expected start_epoch=6, got {start_epoch}"
        assert loaded_best_val == 2.5, f"Expected best_val=2.5 (from val_loss), got {loaded_best_val}"
        assert loaded_metrics_history == [], f"Expected empty metrics_history, got {loaded_metrics_history}"
        
        print("✓ Backward compatibility test passed")
        
    finally:
        # Clean up
        checkpoint_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

