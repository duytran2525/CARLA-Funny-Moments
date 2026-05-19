# Ablation Study Script

This script performs a comprehensive ablation study to measure the individual contribution of each GTNet improvement:
- **GAT (Graph Attention Networks)**: Learnable attention weights for neighbor importance
- **Multimodal Prediction**: K=3 alternative trajectories with Winner-Takes-All loss
- **Adaptive Radius**: Velocity-based interaction radius

## Overview

The ablation study trains **8 model variants** representing all combinations of the 3 binary feature flags:

| Variant | GAT | Multimodal | Adaptive Radius |
|---------|-----|------------|-----------------|
| baseline | ❌ | ❌ | ❌ |
| gat_only | ✅ | ❌ | ❌ |
| multimodal_only | ❌ | ✅ | ❌ |
| adaptive_radius_only | ❌ | ❌ | ✅ |
| gat_multimodal | ✅ | ✅ | ❌ |
| gat_adaptive | ✅ | ❌ | ✅ |
| multimodal_adaptive | ❌ | ✅ | ✅ |
| full | ✅ | ✅ | ✅ |

All variants use:
- **Identical training data** (same train/val split with fixed seed)
- **Identical hyperparameters** (learning rate, batch size, etc.)
- **Identical random seeds** for reproducibility
- **Same validation set** for fair comparison

## Usage

### Basic Usage

```bash
python scripts/run_ablation_study.py \
  --data-dir data/processed/multi_agent \
  --out-dir ablation_results \
  --epochs 30 \
  --batch-size 16 \
  --seed 42
```

### Quick Smoke Test

For rapid testing with reduced epochs and samples:

```bash
python scripts/run_ablation_study.py \
  --data-dir data/processed/multi_agent \
  --out-dir ablation_results \
  --quick-ablation
```

This automatically sets:
- `--epochs 5` (reduced from default 30)
- `--limit-samples 200` (if not already specified)

### Multiple Data Directories

Combine datasets from multiple sources:

```bash
python scripts/run_ablation_study.py \
  --data-dir data/processed/town01 data/processed/town02 \
  --out-dir ablation_results
```

## Command-Line Arguments

### Required Arguments

- `--data-dir`: One or more processed dataset directories containing `manifest.csv` and `.pt` samples

### Optional Arguments

**Output:**
- `--out-dir`: Output directory for results and checkpoints (default: `ablation_results`)

**Training:**
- `--epochs`: Training epochs per variant (default: 30)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--weight-decay`: Weight decay for AdamW (default: 1e-4)
- `--train-ratio`: Train/val split ratio (default: 0.8)
- `--grad-clip`: Gradient clipping max norm (default: 1.0)
- `--early-stopping-patience`: Early stopping patience in epochs (default: 8)

**Model:**
- `--hidden-dim`: Hidden dimension (default: 128)
- `--graph-layers`: Number of graph layers (default: 2)
- `--dropout`: Dropout probability (default: 0.1)

**System:**
- `--device`: Device to use: `auto`, `cpu`, or `cuda` (default: `auto`)
- `--num-workers`: DataLoader workers (default: 0)
- `--seed`: Random seed for reproducibility (default: 42)

**Testing:**
- `--quick-ablation`: Enable quick smoke test mode
- `--limit-samples`: Limit number of samples (0 = no limit)
- `--log-every`: Log metrics every N batches (default: 20)

## Output Files

The script generates the following files in the output directory:

### 1. `ablation_results.json`

Complete results for all 8 variants in JSON format:

```json
{
  "baseline": {
    "variant_name": "baseline",
    "enable_gat": false,
    "enable_multimodal": false,
    "enable_adaptive_radius": false,
    "minADE": 2.5,
    "minFDE": 4.5,
    "MissRate": 0.35,
    "train_time_seconds": 120.5,
    "inference_latency_ms": 18.2
  },
  "full": {
    "variant_name": "full",
    "enable_gat": true,
    "enable_multimodal": true,
    "enable_adaptive_radius": true,
    "minADE": 1.3,
    "minFDE": 2.5,
    "MissRate": 0.15,
    "train_time_seconds": 180.3,
    "inference_latency_ms": 24.7
  }
}
```

### 2. `comparison_table.txt`

Human-readable comparison table showing improvements over baseline:

```
====================================================================================================
ABLATION STUDY RESULTS
====================================================================================================
Variant                      minADE     Δ ADE     minFDE     Δ FDE   MissRate    Δ Miss
----------------------------------------------------------------------------------------------------
baseline                     2.5000         -     4.5000         -     0.3500         -
gat_only                     2.0000   -0.5000     3.8000   -0.7000     0.2800   -0.0700
multimodal_only              1.8000   -0.7000     3.2000   -1.3000     0.2200   -0.1300
adaptive_radius_only         2.3000   -0.2000     4.2000   -0.3000     0.3200   -0.0300
gat_multimodal               1.5000   -1.0000     2.8000   -1.7000     0.1800   -0.1700
gat_adaptive                 1.9000   -0.6000     3.5000   -1.0000     0.2500   -0.1000
multimodal_adaptive          1.6000   -0.9000     2.9000   -1.6000     0.1900   -0.1600
full                         1.3000   -1.2000     2.5000   -2.0000     0.1500   -0.2000
====================================================================================================
```

## Metrics Explained

### minADE (Minimum Average Displacement Error)
- Average Euclidean distance between predicted and ground truth trajectories
- For multimodal prediction, selects the best mode per agent
- **Lower is better**
- Target: < 1.5m (baseline: ~2.5m)

### minFDE (Minimum Final Displacement Error)
- Euclidean distance at the final timestep (3 seconds ahead)
- For multimodal prediction, selects the best mode per agent
- **Lower is better**
- Target: < 2.7m (baseline: ~4.5m)

### MissRate
- Fraction of predictions where minFDE > 2.0 meters
- Measures catastrophic failures
- **Lower is better**
- Target: < 0.20 (baseline: ~0.35)

### Train Time
- Total training time in seconds for the variant
- Includes all epochs until convergence or early stopping

### Inference Latency
- Average inference time in milliseconds per sample
- Measured on 100 samples after warm-up
- Target: < 25ms

## Interpreting Results

### Delta Metrics (Δ)
- **Negative values** indicate improvement over baseline
- **Positive values** indicate degradation
- Example: `Δ ADE = -0.5` means 0.5m improvement in ADE

### Individual Contributions
Compare single-improvement variants to baseline:
- `gat_only` vs `baseline`: GAT contribution
- `multimodal_only` vs `baseline`: Multimodal contribution
- `adaptive_radius_only` vs `baseline`: Adaptive radius contribution

### Interaction Effects
Compare combined variants to sum of individual contributions:
- If `full` improvement > sum of individual improvements: **positive synergy**
- If `full` improvement < sum of individual improvements: **negative interaction**

## Example Output

```
Ablation Study Configuration:
  Data directories: 1
    - D:\AI\CARLA-Funny-Moments\data\processed\multi_agent
  Output directory: D:\AI\CARLA-Funny-Moments\ablation_results
  Device: cuda
  Epochs: 30
  Batch size: 16
  Random seed: 42
  Quick ablation: False
  Total samples: 5000
  Train samples: 4000
  Val samples: 1000

Training 8 variants...

================================================================================
Training variant: baseline
  GAT: False, Multimodal: False, Adaptive Radius: False
================================================================================
  Epoch 005: val_metric=2.5234 [BEST]
  Epoch 010: val_metric=2.4891 [BEST]
  ...
  Early stopping at epoch 25

Variant baseline Results:
  minADE: 2.5000
  minFDE: 4.5000
  MissRate: 0.3500
  Train Time: 120.5s
  Inference Latency: 18.23ms

[... similar output for other 7 variants ...]

[OK] Results saved to: D:\AI\CARLA-Funny-Moments\ablation_results\ablation_results.json
[OK] Comparison table saved to: D:\AI\CARLA-Funny-Moments\ablation_results\comparison_table.txt

Target Metrics Validation:
  minADE: 1.3000 (target: < 1.5)
  minFDE: 2.5000 (target: < 2.7)
  MissRate: 0.1500 (target: < 0.20)

✓ Full model meets all target metrics!

[OK] Ablation study complete!
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python scripts/run_ablation_study.py --data-dir ... --batch-size 8
```

### Training Takes Too Long
Use quick ablation mode for testing:
```bash
python scripts/run_ablation_study.py --data-dir ... --quick-ablation
```

### Poor Convergence
- Increase epochs: `--epochs 50`
- Adjust learning rate: `--learning-rate 5e-4`
- Check data quality and quantity

### Inconsistent Results
- Ensure fixed seed: `--seed 42`
- Verify identical data splits across variants
- Check for data augmentation or randomness in preprocessing

## Requirements Validation

This script validates the following requirements:
- **8.1**: Support configuration flags (enable_gat, enable_multimodal, enable_adaptive_radius)
- **8.2**: Train 8 model variants (all combinations of 3 binary flags)
- **8.3**: Use identical training data, hyperparameters, and random seeds
- **8.4**: Evaluate each variant on the same validation set
- **8.5**: Log results to ablation_results.json
- **8.6**: Compute delta metrics showing improvement over baseline
- **8.7**: Generate comparison table
- **8.8**: Measure inference latency (ms per sample)
- **8.10**: Support --quick-ablation flag for smoke testing

## See Also

- `train_per_town.py`: Per-town training script
- `build_multi_agent_dataset.py`: Dataset preprocessing
- `README_train_per_town.md`: Training documentation
