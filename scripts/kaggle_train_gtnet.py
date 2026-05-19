"""
Kaggle Training Script for GTNet Multi-Agent Trajectory Prediction
===================================================================

Script tối ưu cho môi trường Kaggle với GPU T4/P100.
Hỗ trợ training với tất cả 5 cải tiến GTNet.

Usage trong Kaggle Notebook:
    !python kaggle_train_gtnet.py --epochs 50 --batch-size 32
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GTNet on Kaggle"
    )
    
    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/kaggle/input/gtnet-dataset/processed",
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--town-filter",
        type=str,
        nargs="+",
        default=None,
        help="Train on specific towns only (e.g., Town01 Town02)",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Early stopping patience",
    )
    
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--graph-layers", type=int, default=2, help="Graph layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    
    # Model features (5 cải tiến GTNet)
    parser.add_argument(
        "--enable-gat",
        action="store_true",
        help="Enable Graph Attention Networks",
    )
    parser.add_argument(
        "--enable-multimodal",
        action="store_true",
        help="Enable multimodal prediction (K=3 modes)",
    )
    parser.add_argument(
        "--enable-adaptive-radius",
        action="store_true",
        help="Enable adaptive interaction radius",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=3,
        help="Number of trajectory modes (K)",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=4,
        help="Number of attention heads for GAT",
    )
    
    # System
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=20, help="Log every N batches")
    
    # Output
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/kaggle/working/models",
        help="Output directory for checkpoints",
    )
    
    return parser.parse_args()


def setup_kaggle_environment():
    """Setup Kaggle environment and check GPU."""
    logging.info("=" * 70)
    logging.info("Kaggle Environment Setup")
    logging.info("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        logging.warning("No GPU available! Training will be slow.")
    
    # Check input data
    if KAGGLE_INPUT.exists():
        logging.info(f"Kaggle input directory: {KAGGLE_INPUT}")
        datasets = list(KAGGLE_INPUT.glob("*"))
        logging.info(f"Available datasets: {len(datasets)}")
        for ds in datasets:
            logging.info(f"  - {ds.name}")
    else:
        logging.warning("Kaggle input directory not found!")
    
    logging.info("=" * 70)
    logging.info("")


def load_dataset(data_dir, town_filter=None):
    """Load GTNet dataset."""
    from core_perception.multi_agent_dataset import (
        MultiAgentTrajectoryDataset,
        collate_multi_agent_trajectory,
    )
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    logging.info(f"Loading dataset from: {data_path}")
    
    dataset = MultiAgentTrajectoryDataset(
        data_dir=data_path,
        town_filter=town_filter,
    )
    
    logging.info(f"Total samples: {len(dataset)}")
    
    return dataset, collate_multi_agent_trajectory


def create_model(args):
    """Create GTNet model with specified configuration."""
    from core_perception.multi_agent_model import (
        MultiAgentModelConfig,
        MultiAgentTrajectoryPredictor,
    )
    
    config = MultiAgentModelConfig(
        input_dim=6,  # (local_x, local_y, local_vx, local_vy, heading_x, heading_y)
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        future_steps=30,  # 3 seconds at 10 FPS
        dropout=args.dropout,
        enable_gat=args.enable_gat,
        num_attention_heads=args.num_attention_heads,
        enable_multimodal=args.enable_multimodal,
        num_modes=args.num_modes,
        enable_adaptive_radius=args.enable_adaptive_radius,
    )
    
    logging.info("Model Configuration:")
    logging.info(f"  Hidden dim: {config.hidden_dim}")
    logging.info(f"  Graph layers: {config.graph_layers}")
    logging.info(f"  GAT enabled: {config.enable_gat}")
    if config.enable_gat:
        logging.info(f"  Attention heads: {config.num_attention_heads}")
    logging.info(f"  Multimodal enabled: {config.enable_multimodal}")
    if config.enable_multimodal:
        logging.info(f"  Number of modes: {config.num_modes}")
    logging.info(f"  Adaptive radius: {config.enable_adaptive_radius}")
    
    model = MultiAgentTrajectoryPredictor(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model, config


def train_epoch(model, dataloader, optimizer, device, args, epoch):
    """Train for one epoch."""
    from core_perception.multi_agent_trajectory import (
        wta_loss,
        masked_smooth_l1_loss,
        compute_multimodal_metrics,
    )
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        adj = batch["adj"].to(device)
        x_mask = batch["x_mask"].to(device)
        y_mask = batch["y_mask"].to(device)
        agent_mask = batch["agent_mask"].to(device)
        
        # Forward
        optimizer.zero_grad()
        pred = model(x, adj, x_mask, agent_mask)
        
        # Loss
        if args.enable_multimodal:
            loss = wta_loss(pred, y, y_mask, agent_mask)
        else:
            loss = masked_smooth_l1_loss(pred, y, y_mask, agent_mask)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Stats
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Log
        if (batch_idx + 1) % args.log_every == 0:
            avg_loss = total_loss / total_samples
            logging.info(
                f"Epoch {epoch:03d} | Batch {batch_idx+1:04d}/{len(dataloader):04d} | "
                f"Loss: {avg_loss:.4f}"
            )
    
    avg_loss = total_loss / total_samples
    return avg_loss


def validate(model, dataloader, device, args):
    """Validate model."""
    from core_perception.multi_agent_trajectory import (
        wta_loss,
        masked_smooth_l1_loss,
        compute_multimodal_metrics,
    )
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            adj = batch["adj"].to(device)
            x_mask = batch["x_mask"].to(device)
            y_mask = batch["y_mask"].to(device)
            agent_mask = batch["agent_mask"].to(device)
            
            # Forward
            pred = model(x, adj, x_mask, agent_mask)
            
            # Loss
            if args.enable_multimodal:
                loss = wta_loss(pred, y, y_mask, agent_mask)
            else:
                loss = masked_smooth_l1_loss(pred, y, y_mask, agent_mask)
            
            # Stats
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Metrics
            metrics = compute_multimodal_metrics(pred, y, y_mask, agent_mask)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_loss = total_loss / total_samples
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_loss, avg_metrics


def save_checkpoint(model, config, optimizer, scheduler, epoch, metrics, out_dir, is_best=False):
    """Save checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config.to_json(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    
    # Save last checkpoint
    last_path = out_dir / "last.pt"
    torch.save(checkpoint, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = out_dir / "best.pt"
        torch.save(checkpoint, best_path)
        logging.info(f"✓ Saved best checkpoint: {best_path}")


def main():
    args = parse_args()
    
    # Setup
    setup_kaggle_environment()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info("")
    
    # Load dataset
    dataset, collate_fn = load_dataset(args.data_dir, args.town_filter)
    
    # Split train/val
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")
    logging.info("")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Create model
    model, config = create_model(args)
    model = model.to(device)
    logging.info("")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
    )
    
    # Training loop
    logging.info("=" * 70)
    logging.info("Starting Training")
    logging.info("=" * 70)
    
    best_val_metric = float("inf")
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        logging.info("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args, epoch)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, args)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log
        logging.info(f"\nEpoch {epoch} Summary:")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}")
        
        if args.enable_multimodal:
            logging.info(f"  Val minADE: {val_metrics['minADE']:.4f}")
            logging.info(f"  Val minFDE: {val_metrics['minFDE']:.4f}")
            logging.info(f"  Val MissRate: {val_metrics['MissRate']:.4f}")
            val_metric = val_metrics["minADE"]
        else:
            logging.info(f"  Val ADE: {val_metrics['ADE']:.4f}")
            logging.info(f"  Val FDE: {val_metrics['FDE']:.4f}")
            val_metric = val_metrics["ADE"]
        
        # Save checkpoint
        is_best = val_metric < best_val_metric
        if is_best:
            best_val_metric = val_metric
            patience_counter = 0
            logging.info(f"  ✓ New best model! (metric: {val_metric:.4f})")
        else:
            patience_counter += 1
            logging.info(f"  Patience: {patience_counter}/{args.early_stopping_patience}")
        
        save_checkpoint(
            model, config, optimizer, scheduler, epoch,
            {"train_loss": train_loss, "val_loss": val_loss, **val_metrics},
            args.out_dir, is_best
        )
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logging.info(f"\nEarly stopping at epoch {epoch}")
            break
    
    logging.info("")
    logging.info("=" * 70)
    logging.info("Training Complete!")
    logging.info(f"Best validation metric: {best_val_metric:.4f}")
    logging.info(f"Checkpoints saved to: {args.out_dir}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
