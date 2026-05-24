# GTNet Optimized Training Script for Kaggle

## Model Size Comparison

| Config | hidden_dim | graph_layers | Params | Size | T4 VRAM (~batch 32) |
|--------|-----------|-------------|--------|------|---------------------|
| Current (BUG) | 128 | 2 | 423K | 1.6MB | ~2GB |
| **Medium (recommended)** | **256** | **3** | **1.73M** | **6.6MB** | **~4GB** |
| Large | 384 | 4 | 4.03M | 15.4MB | ~8GB |
| XLarge | 512 | 4 | 7.14M | 27.2MB | ~12GB |

## Kaggle Notebook Cells

### Cell 1: Setup & Verify

```python
import sys
from pathlib import Path

CODE_ROOT = Path("/kaggle/input/datasets/trasuaolong/d134567")
TRAIN_SCRIPT = Path("/kaggle/input/datasets/trasuaolong/d78907/kaggle_train_gtnet.py")
DATA_DIR = Path("/kaggle/input/datasets/trasuaolong/d12456/multi_agent_20fps/processed_adaptive")

for p in [CODE_ROOT, TRAIN_SCRIPT.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

print("CODE_ROOT exists:", CODE_ROOT.exists())
print("TRAIN_SCRIPT exists:", TRAIN_SCRIPT.exists())
print("DATA_DIR exists:", DATA_DIR.exists())
print("manifest exists:", (DATA_DIR / "manifest.csv").exists())

from core_perception.multi_agent_dataset import MultiAgentTrajectoryDataset
from core_perception.multi_agent_model import MultiAgentModelConfig, MultiAgentTrajectoryPredictor

# Verify dataset
ds = MultiAgentTrajectoryDataset(DATA_DIR)
sample = ds[0]
print(f"Dataset size: {len(ds)} samples")
print(f"x shape: {sample['x'].shape} (agents, history={sample['x'].shape[1]}, features={sample['x'].shape[2]})")
print(f"y shape: {sample['y'].shape} (agents, future={sample['y'].shape[1]}, xy=2)")

# Verify model size
cfg = MultiAgentModelConfig(
    hidden_dim=256, graph_layers=3, future_steps=sample['y'].shape[1],
    num_modes=5, enable_gat=True, enable_multimodal=True, enable_adaptive_radius=True,
)
model = MultiAgentTrajectoryPredictor(cfg)
n_params = sum(p.numel() for p in model.parameters())
size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
print(f"\nModel: hidden_dim={cfg.hidden_dim} graph_layers={cfg.graph_layers} future_steps={cfg.future_steps}")
print(f"Params: {n_params:,} ({size_mb:.1f} MB)")
print("Import core_perception OK")
```

### Cell 2: Train — Medium Model (256 dim, 1.73M params) — RECOMMENDED

```bash
!PYTHONPATH=/kaggle/input/datasets/trasuaolong/d134567:$PYTHONPATH \
python /kaggle/input/datasets/trasuaolong/d78907/kaggle_train_gtnet.py \
  --data-dir /kaggle/input/datasets/trasuaolong/d12456/multi_agent_20fps/processed_adaptive \
  --out-dir /kaggle/working/gtnet_256dim \
  --mode full \
  --epochs 80 \
  --batch-size 64 \
  --accum-steps 1 \
  --hidden-dim 256 \
  --graph-layers 3 \
  --num-modes 5 \
  --num-attention-heads 4 \
  --dropout 0.1 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --lr-patience 6 \
  --early-stopping-patience 15 \
  --grad-clip 1.0 \
  --gat-lr-scale 0.1 \
  --gat-clip-scale 0.5 \
  --num-workers 2 \
  --cosine-lr \
  --seed 42 \
  --log-every 50 \
  --enable-gat \
  --enable-multimodal \
  --enable-adaptive-radius
```

### Cell 3 (Optional): Train — Large Model (384 dim, 4.0M params) — if T4 has enough VRAM

```bash
!PYTHONPATH=/kaggle/input/datasets/trasuaolong/d134567:$PYTHONPATH \
python /kaggle/input/datasets/trasuaolong/d78907/kaggle_train_gtnet.py \
  --data-dir /kaggle/input/datasets/trasuaolong/d12456/multi_agent_20fps/processed_adaptive \
  --out-dir /kaggle/working/gtnet_384dim \
  --mode full \
  --epochs 80 \
  --batch-size 32 \
  --accum-steps 2 \
  --hidden-dim 384 \
  --graph-layers 4 \
  --num-modes 5 \
  --num-attention-heads 4 \
  --dropout 0.15 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --lr-patience 6 \
  --early-stopping-patience 15 \
  --grad-clip 0.8 \
  --gat-lr-scale 0.1 \
  --gat-clip-scale 0.5 \
  --num-workers 2 \
  --cosine-lr \
  --seed 42 \
  --log-every 50 \
  --enable-gat \
  --enable-multimodal \
  --enable-adaptive-radius
```

### Cell 4: Copy best model

```python
import shutil
from pathlib import Path

# Copy from the medium model (or large model if you ran Cell 3)
src = Path("/kaggle/working/gtnet_256dim/GTNet_Full_best.pt")  # or gtnet_384dim
if not src.exists():
    # Try alternate naming
    candidates = list(Path("/kaggle/working/gtnet_256dim").glob("*best*.pt"))
    if candidates:
        src = candidates[0]
    else:
        print("No best checkpoint found!")
        print("Available files:", list(Path("/kaggle/working/gtnet_256dim").rglob("*.pt")))

if src.exists():
    import torch
    ckpt = torch.load(src, map_location="cpu")
    cfg = ckpt.get("model_config", {})
    print(f"Best model: {src.name}")
    print(f"  hidden_dim: {cfg.get('hidden_dim')}")
    print(f"  graph_layers: {cfg.get('graph_layers')}")
    print(f"  future_steps: {cfg.get('future_steps')}")
    print(f"  val_ade: {ckpt.get('val_ade', 'N/A')}")
    print(f"  val_fde: {ckpt.get('val_fde', 'N/A')}")
    print(f"  epoch: {ckpt.get('epoch', 'N/A')}")
    
    # Copy to /kaggle/working for download
    dst = Path("/kaggle/working/gtnet_full_best.pt")
    shutil.copy2(src, dst)
    print(f"\nCopied to: {dst}")
    print(f"Size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
```

## After Training: Update configs/carla_env.yaml

Replace the downloaded model file in `models/gtnet_full_best.pt` and the config stays the same
since we already fixed it to use `expected_dt: 0.05` and `history_frames: 40`.
