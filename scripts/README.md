# Scripts Directory

This directory contains utility scripts for data collection, dataset building, training, and validation.

## Data Collection Scripts

### `collect_multi_agent_data.py`
Collect raw multi-agent trajectory data from CARLA simulator.

```bash
python collect_multi_agent_data.py --town Town01 --num-vehicles 50 --duration 300
```

## Dataset Building Scripts

### `build_multi_agent_dataset.py`
Build processed `.pt` samples from raw CSV logs.

```bash
python scripts/build_multi_agent_dataset.py \
    --raw-csv data/multi_agent/raw/Town01_20240101_120000.csv \
    --out-dir data/multi_agent/processed/Town01 \
    --history-frames 20 \
    --future-frames 30 \
    --stride 1 \
    --adjacency-radius-m 100.0 \
    --min-agents 2 \
    --allow-missing
```

**Key Parameters:**
- `--history-frames`: Number of past frames (default: 20)
- `--future-frames`: Number of future frames to predict (default: 30)
- `--stride`: Sliding window stride (default: 1)
- `--adjacency-radius-m`: Radius for agent connectivity (default: 100.0)
- `--min-agents`: Minimum agents per sample (default: 2)
- `--allow-missing`: Allow agents with missing frames (default: True)
- `--adaptive-radius`: Enable velocity-based adaptive radius
- `--radius-base`: Base radius for adaptive mode (default: 20.0)
- `--radius-alpha`: Velocity scaling factor (default: 0.5)

### `fix_manifest_paths.py` ⭐ NEW
Fix manifest.csv to use forward slashes for cross-platform compatibility.

```bash
python scripts/fix_manifest_paths.py \
    --manifest data/multi_agent/processed/manifest.csv \
    --backup
```

**Use Cases:**
- Fix existing manifests created on Windows
- Prepare datasets for Linux/Kaggle deployment
- Ensure cross-platform compatibility

**Options:**
- `--manifest`: Path to manifest.csv (default: data/multi_agent/processed/manifest.csv)
- `--backup`: Create backup before modifying (recommended)

### `test_dataset_loading.py` ⭐ NEW
Verify that dataset loads correctly.

```bash
python scripts/test_dataset_loading.py \
    --dataset-dir data/multi_agent/processed \
    --num-samples 5
```

**Use Cases:**
- Verify dataset after building
- Test dataset after fixing paths
- Quick sanity check before training

**Options:**
- `--dataset-dir`: Path to processed dataset directory
- `--num-samples`: Number of samples to test (default: 5)

## Training Scripts

### `train_multi_agent_trajectory.py`
Train GTNet model on multi-agent trajectory data.

```bash
python scripts/train_multi_agent_trajectory.py \
    --dataset-dir data/multi_agent/processed \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

### `train_per_town.py`
Train separate models for each town.

```bash
python scripts/train_per_town.py \
    --dataset-dir data/multi_agent/processed \
    --output-dir models/per_town \
    --epochs 50
```

### `kaggle_train_gtnet.py`
Training script optimized for Kaggle environment.

```bash
python kaggle_train_gtnet.py \
    --mode ablation \
    --data-dirs /kaggle/input/dataset/multi_agent/processed \
    --epochs 100
```

## Validation Scripts

### `validate_target_metrics.py`
Validate dataset against target metrics.

```bash
python scripts/validate_target_metrics.py \
    --dataset-dir data/multi_agent/processed
```

**Checks:**
- Sample count per town
- Agent count distribution
- Feature dimensions
- Temporal coverage

### `run_ablation_study.py`
Run ablation study to evaluate model components.

```bash
python scripts/run_ablation_study.py \
    --dataset-dir data/multi_agent/processed \
    --output-dir results/ablation
```

## Migration Scripts

### `migrate_dataset.py`
Migrate dataset from old format to new format.

```bash
python scripts/migrate_dataset.py \
    --old-dir data/old_format \
    --new-dir data/new_format
```

## PowerShell Scripts

### `build_all_datasets.ps1`
Build datasets for all towns in one go.

```powershell
.\build_all_datasets.ps1
```

**Features:**
- Processes all CSV files in `data/multi_agent/raw/`
- Creates per-town subdirectories
- Generates global manifest.csv
- Shows progress and summary

### `collect_all_towns.ps1`
Collect data from all CARLA towns.

```powershell
.\collect_all_towns.ps1
```

### `collect_all_towns_adaptive.ps1`
Collect data with adaptive radius enabled.

```powershell
.\collect_all_towns_adaptive.ps1
```

## Common Workflows

### 1. Collect and Build Dataset
```bash
# Step 1: Collect raw data
python collect_multi_agent_data.py --town Town01 --duration 300

# Step 2: Build processed dataset
python scripts/build_multi_agent_dataset.py \
    --raw-csv data/multi_agent/raw/Town01_*.csv \
    --out-dir data/multi_agent/processed/Town01

# Step 3: Fix paths for cross-platform compatibility
python scripts/fix_manifest_paths.py --manifest data/multi_agent/processed/manifest.csv --backup

# Step 4: Verify dataset
python scripts/test_dataset_loading.py --dataset-dir data/multi_agent/processed
```

### 2. Build All Towns at Once
```powershell
# Collect all towns
.\collect_all_towns.ps1

# Build all datasets
.\build_all_datasets.ps1

# Fix paths (if needed for Linux/Kaggle)
python scripts/fix_manifest_paths.py --manifest data/multi_agent/processed/manifest.csv --backup

# Validate
python scripts/validate_target_metrics.py --dataset-dir data/multi_agent/processed
```

### 3. Prepare for Kaggle
```bash
# Fix paths for Linux compatibility
python scripts/fix_manifest_paths.py \
    --manifest data/multi_agent/processed/manifest.csv \
    --backup

# Test loading
python scripts/test_dataset_loading.py \
    --dataset-dir data/multi_agent/processed \
    --num-samples 10

# Upload to Kaggle and train
# (Upload dataset, then run kaggle_train_gtnet.py on Kaggle)
```

## Troubleshooting

### Path Separator Issues
If you see `FileNotFoundError` with backslashes in paths on Linux/Kaggle:

```bash
# Fix the manifest
python scripts/fix_manifest_paths.py --manifest <path-to-manifest> --backup

# Verify fix
python scripts/test_dataset_loading.py --dataset-dir <dataset-dir>
```

See `docs/PATH_SEPARATOR_FIX.md` for detailed information.

### Dataset Validation Failures
```bash
# Check dataset structure
python scripts/validate_target_metrics.py --dataset-dir data/multi_agent/processed

# Test loading samples
python scripts/test_dataset_loading.py --dataset-dir data/multi_agent/processed --num-samples 10
```

### Missing Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# For CARLA
pip install carla==0.9.13
```

## Script Dependencies

Most scripts require:
- Python 3.10+
- PyTorch
- NumPy
- Pandas (for CSV processing)

CARLA collection scripts additionally require:
- CARLA 0.9.13 simulator
- carla Python package

## Documentation

For more information, see:
- `docs/PATH_SEPARATOR_FIX.md` - Path separator fix details
- `docs/KAGGLE_TRAINING_GUIDE.md` - Kaggle training guide
- `docs/MULTI_AGENT_COLLECTION_UPDATE.md` - Data collection guide
- `GTNet_README.md` - Model architecture and training
- `README.md` - Project overview

## Recent Additions (2026-05-21)

### Path Separator Fix
Two new utility scripts added to handle cross-platform path compatibility:

1. **`fix_manifest_paths.py`**: Converts Windows backslashes to forward slashes in manifest files
2. **`test_dataset_loading.py`**: Verifies dataset can be loaded correctly

These scripts ensure datasets created on Windows work seamlessly on Linux/Kaggle environments.

See `QUICK_FIX_SUMMARY.md` for quick reference or `docs/PATH_SEPARATOR_FIX.md` for detailed documentation.
