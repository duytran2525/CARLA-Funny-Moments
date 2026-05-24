# Example usage of train_per_town.py script
# This script demonstrates how to train per-town models with various configurations

# Example 1: Train on Town01 only with baseline configuration
Write-Host "Example 1: Baseline configuration on Town01" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 `
    --out-dir models/multi_agent `
    --epochs 30 `
    --batch-size 16 `
    --learning-rate 1e-3 `
    --device auto `
    --log-every 20

# Example 2: Train on Town01 with GAT enabled
Write-Host "Example 2: GAT enabled on Town01" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 `
    --out-dir models/multi_agent `
    --enable-gat `
    --num-attention-heads 4 `
    --epochs 30 `
    --batch-size 16 `
    --device auto

# Example 3: Train on Town01 with multimodal prediction
Write-Host "Example 3: Multimodal prediction on Town01" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 `
    --out-dir models/multi_agent `
    --enable-multimodal `
    --num-modes 3 `
    --epochs 30 `
    --batch-size 16 `
    --device auto

# Example 4: Train on Town01 with all improvements enabled
Write-Host "Example 4: All improvements enabled on Town01" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 `
    --out-dir models/multi_agent `
    --enable-gat `
    --enable-multimodal `
    --enable-adaptive-radius `
    --num-attention-heads 4 `
    --num-modes 3 `
    --epochs 30 `
    --batch-size 16 `
    --device auto

# Example 5: Train on multiple towns (Town01 and Town02)
Write-Host "Example 5: Multiple towns (Town01 and Town02)" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 Town02 `
    --out-dir models/multi_agent `
    --enable-gat `
    --enable-multimodal `
    --epochs 30 `
    --batch-size 16 `
    --device auto

# Example 6: Train on all towns (no town filter)
Write-Host "Example 6: All towns" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --out-dir models/multi_agent `
    --enable-gat `
    --enable-multimodal `
    --enable-adaptive-radius `
    --epochs 30 `
    --batch-size 16 `
    --device auto

# Example 7: Smoke test with limited samples
Write-Host "Example 7: Smoke test with limited samples" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 `
    --out-dir models/multi_agent `
    --limit-samples 100 `
    --epochs 2 `
    --batch-size 8 `
    --device cpu

# Example 8: Training with custom hyperparameters
Write-Host "Example 8: Custom hyperparameters" -ForegroundColor Green
python scripts/train_per_town.py `
    --data-dir data/multi_agent/processed `
    --town-filter Town01 `
    --out-dir models/multi_agent `
    --enable-gat `
    --enable-multimodal `
    --epochs 50 `
    --batch-size 32 `
    --learning-rate 5e-4 `
    --weight-decay 1e-5 `
    --grad-clip 2.0 `
    --early-stopping-patience 10 `
    --hidden-dim 256 `
    --graph-layers 3 `
    --device cuda
