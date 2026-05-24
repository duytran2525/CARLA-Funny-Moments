# Per-Town Training Script

## Overview

The `train_per_town.py` script provides a flexible training pipeline for multi-agent trajectory prediction models with support for per-town training, multimodal prediction, and Graph Attention Networks (GAT).

## Features

- **Per-Town Training**: Train separate models for specific CARLA towns or combinations of towns
- **Town Filtering**: Filter training data by town name(s) using `--town-filter`
- **Organized Output**: Automatically creates town-specific subdirectories in `models/multi_agent/{town}/`
- **Multimodal Prediction**: Support for K-mode trajectory prediction with Winner-Takes-All (WTA) loss
- **Graph Attention Networks**: Optional GAT layers for learnable neighbor importance
- **Adaptive Radius**: Support for velocity-based adaptive interaction radius
- **Comprehensive Metrics**: Logs both best-trajectory metrics (minADE, minFDE) and per-mode metrics
- **Early Stopping**: Configurable patience (default: 8 epochs)
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor=0.5 and patience=3
- **Gradient Clipping**: Configurable max norm (default: 1.0)
- **Kaggle Optimization**: Designed for efficient training in resource-constrained environments

## Usage

### Basic Usage

Train on a specific town with baseline configuration:

```bash
python scripts/train_per_town.py \
    --data-dir data/multi_agent/processed \
    --town-filter Town01 \
    --out-dir models/multi_agent \
    --epochs 30 \
    --batch-size 16 \
    --device auto
```

### Advanced Usage

Train with all improvements enabled:

```bash
python scripts/train_per_town.py \
    --data-dir data/multi_agent/processed \
    --town-filter Town01 \
    --out-dir models/multi_agent \
    --enable-gat \
    --enable-multimodal \
    --enable-adaptive-radius \
    --num-attention-heads 4 \
    --num-modes 3 \
    --epochs 30 \
    --batch-size 16 \
    --device cuda
```

### Multiple Towns

Train on multiple towns:

```bash
python scripts/train_per_town.py \
    --data-dir data/multi_agent/processed \
    --town-filter Town01 Town02 Town03 \
    --out-dir models/multi_agent \
    --enable-gat \
    --enable-multimodal \
    --epochs 30 \
    --device auto
```

### All Towns

Train on all available towns (no filter):

```bash
python scripts/train_per_town.py \
    --data-dir data/multi_agent/processed \
    --out-dir models/multi_agent \
    --enable-gat \
    --enable-multimodal \
    --epochs 30 \
    --device auto
```

## Command-Line Arguments

### Required Arguments

- `--data-dir`: One or more processed dataset directories containing `manifest.csv` and `.pt` samples

### Optional Arguments

#### Data and Output
- `--town-filter`: Train on specific towns only (e.g., `Town01 Town02`). If not specified, trains on all towns
- `--out-dir`: Base checkpoint/output directory (default: `models/multi_agent`). Town-specific subdirs will be created
- `--limit-samples`: Optional smoke-test sample cap (default: 0, no limit)

#### Training Hyperparameters
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay for AdamW optimizer (default: 1e-4)
- `--train-ratio`: Train/validation split ratio (default: 0.8)
- `--grad-clip`: Gradient clipping max norm (default: 1.0)
- `--early-stopping-patience`: Early stopping patience in epochs (default: 8)

#### Model Architecture
- `--hidden-dim`: Hidden dimension size (default: 128)
- `--graph-layers`: Number of graph interaction layers (default: 2)
- `--dropout`: Dropout probability (default: 0.1)

#### Model Features
- `--enable-gat`: Enable Graph Attention Networks
- `--enable-multimodal`: Enable multimodal prediction
- `--enable-adaptive-radius`: Enable adaptive interaction radius
- `--num-modes`: Number of trajectory modes for multimodal prediction (default: 3)
- `--num-attention-heads`: Number of attention heads for GAT (default: 4)

#### System
- `--device`: Device to use: `auto` (detect), `cpu`, or `cuda` (default: auto)
- `--num-workers`: Number of DataLoader workers (default: 0)
- `--seed`: Random seed (default: 42)
- `--log-every`: Log metrics every N batches (default: 20)

## Output Structure

The script creates the following output structure:

```
models/multi_agent/
└── {town_name}/
    ├── train_config.json       # Training configuration and hyperparameters
    ├── metrics_history.json    # Per-epoch metrics history
    ├── best.pt                 # Best checkpoint (lowest val_minADE or val_loss)
    └── last.pt                 # Last checkpoint
```

Where `{town_name}` is:
- The town name if `--town-filter` specifies a single town (e.g., `Town01`)
- Concatenated town names if multiple towns are specified (e.g., `Town01_Town02`)
- `all_towns` if no town filter is specified

## Checkpoint Format

Each checkpoint (`.pt` file) contains:

```python
{
    "model_state_dict": OrderedDict,      # Model parameters
    "model_config": dict,                 # Serialized MultiAgentModelConfig
    "optimizer_state_dict": dict,         # Optimizer state
    "scheduler_state_dict": dict,         # LR scheduler state
    "epoch": int,                         # Current epoch
    "val_loss": float,                    # Validation loss
    "train_loss": float,                  # Training loss
    "val_metrics": dict,                  # Validation metrics
    "train_metrics": dict,                # Training metrics
}
```

## Training Configuration

The `train_config.json` file contains:

```json
{
    "data_dirs": ["path/to/data"],
    "town_filter": ["Town01"],
    "town_name": "Town01",
    "train_samples": 8000,
    "val_samples": 2000,
    "device": "cuda",
    "model_config": {
        "input_dim": 6,
        "hidden_dim": 128,
        "graph_layers": 2,
        "future_steps": 30,
        "dropout": 0.1,
        "enable_gat": true,
        "num_attention_heads": 4,
        "enable_multimodal": true,
        "num_modes": 3,
        "enable_adaptive_radius": true,
        "radius_base": 20.0,
        "radius_alpha": 0.5
    },
    "hyperparameters": {
        "epochs": 30,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "early_stopping_patience": 8
    }
}
```

## Metrics

### Unimodal Metrics (baseline)
- `train_ADE`: Training Average Displacement Error
- `train_FDE`: Training Final Displacement Error
- `val_ADE`: Validation Average Displacement Error
- `val_FDE`: Validation Final Displacement Error

### Multimodal Metrics (with `--enable-multimodal`)
- `train_minADE`: Training minimum ADE across K modes
- `train_minFDE`: Training minimum FDE across K modes
- `val_minADE`: Validation minimum ADE across K modes
- `val_minFDE`: Validation minimum FDE across K modes
- `train_mode_0_ADE`, `train_mode_1_ADE`, etc.: Per-mode training ADE
- `train_mode_0_FDE`, `train_mode_1_FDE`, etc.: Per-mode training FDE
- `val_mode_0_ADE`, `val_mode_1_ADE`, etc.: Per-mode validation ADE
- `val_mode_0_FDE`, `val_mode_1_FDE`, etc.: Per-mode validation FDE

## Loss Functions

The script automatically selects the appropriate loss function:

- **Unimodal** (baseline): `masked_smooth_l1_loss`
- **Multimodal** (`--enable-multimodal`): `wta_loss` (Winner-Takes-All)

## Early Stopping

Early stopping is based on:
- **Multimodal**: `val_minADE` (lower is better)
- **Unimodal**: `val_loss` (lower is better)

The best checkpoint is saved when the metric improves, and training stops if no improvement is seen for `--early-stopping-patience` epochs.

## Learning Rate Scheduling

The script uses `ReduceLROnPlateau` scheduler:
- **Mode**: min (reduce when metric stops decreasing)
- **Factor**: 0.5 (multiply LR by 0.5 when reducing)
- **Patience**: 3 epochs (wait 3 epochs before reducing)

## Examples

See `example_train_per_town.sh` (Linux/Mac) or `example_train_per_town.ps1` (Windows) for comprehensive usage examples.

## Requirements Validation

This script implements the following requirements from the GTNet Improvements specification:

- **Requirement 7.1**: Accept `--data-dir` argument supporting multiple dataset directories ✓
- **Requirement 7.2**: Support `--town-filter` argument to train on specific towns only ✓
- **Requirement 7.3**: Save checkpoints to `models/multi_agent/{town}/` directory structure ✓
- **Requirement 7.4**: Log training configuration to `train_config.json` including data sources and hyperparameters ✓
- **Requirement 7.5**: Implement early stopping with configurable patience (default: 8 epochs) ✓
- **Requirement 7.6**: Use ReduceLROnPlateau scheduler with factor=0.5 and patience=3 ✓
- **Requirement 7.7**: Save both best checkpoint (lowest val_minADE) and last checkpoint ✓
- **Requirement 7.8**: Support gradient clipping with configurable max_norm (default: 1.0) ✓
- **Requirement 7.9**: Log metrics every N batches (configurable via `--log-every`) ✓
- **Requirement 7.10**: Support `--device` argument with options: auto, cpu, cuda ✓
- **Requirement 2.10**: Log both best-trajectory metrics (minADE, minFDE) and per-mode metrics ✓
- **Requirement 4.6**: Log metrics per epoch including train/val minADE, minFDE ✓
- **Requirement 4.7**: Save best checkpoint based on val_minADE (lower is better) ✓
- **Requirement 4.10**: Write evaluation metrics to JSON file for post-training analysis ✓

## Testing

Run the test suite:

```bash
python -m pytest tests/test_train_per_town.py -v
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` (try 8 or 4)
- Reduce `--hidden-dim` (try 64)
- Use `--device cpu` for testing

### NaN Loss
- Reduce `--learning-rate` (try 5e-4 or 1e-4)
- Check input data for NaN values
- Increase `--grad-clip` (try 2.0 or 5.0)

### Poor Convergence
- Increase `--epochs` (try 50 or 100)
- Adjust `--learning-rate` (try 5e-4 or 2e-3)
- Increase `--early-stopping-patience` (try 10 or 15)

### No Samples Found
- Check that `--data-dir` points to a valid processed dataset directory
- Verify that `--town-filter` matches town names in the dataset
- Check that `.pt` sample files contain the `town` metadata field

## See Also

- `train_multi_agent_trajectory.py`: Original baseline training script
- `build_multi_agent_dataset.py`: Dataset building pipeline
- `core_perception/multi_agent_model.py`: Model architecture and loss functions
- `core_perception/multi_agent_dataset.py`: Dataset loading and collation
