"""
Quick test to verify dataset loading works correctly with fixed paths.

Usage:
    python scripts/test_dataset_loading.py --dataset-dir data/multi_agent/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_perception.multi_agent_dataset import MultiAgentTrajectoryDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test dataset loading with fixed paths.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/multi_agent/processed",
        help="Path to the processed dataset directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test loading (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    
    print("=" * 80)
    print("Test Dataset Loading")
    print("=" * 80)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Test samples: {args.num_samples}")
    print("=" * 80)
    print()
    
    # Load dataset
    try:
        dataset = MultiAgentTrajectoryDataset(root_dir=dataset_dir)
        print(f"[OK] Dataset loaded successfully")
        print(f"[OK] Total samples: {len(dataset)}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return 1
    
    # Test loading first N samples
    num_to_test = min(args.num_samples, len(dataset))
    print(f"Testing first {num_to_test} samples...")
    print()
    
    for i in range(num_to_test):
        try:
            sample = dataset[i]
            print(f"[{i+1}/{num_to_test}] Sample {i}:")
            print(f"  Path: {sample['sample_path']}")
            print(f"  Town: {sample['town']}")
            print(f"  Agents: {sample['x'].shape[0]}")
            print(f"  History: {sample['x'].shape[1]} frames")
            print(f"  Features: {sample['x'].shape[2]}D")
            print(f"  Future: {sample['y'].shape[1]} frames")
            print(f"  ✓ Loaded successfully")
            print()
        except Exception as e:
            print(f"[{i+1}/{num_to_test}] Sample {i}: ✗ FAILED")
            print(f"  Error: {e}")
            print()
            return 1
    
    print("=" * 80)
    print(f"[SUCCESS] All {num_to_test} samples loaded successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
