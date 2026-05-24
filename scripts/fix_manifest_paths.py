"""
Fix manifest.csv to use forward slashes for cross-platform compatibility.

This script reads the global manifest.csv and replaces all backslashes
with forward slashes in the sample_file column, ensuring compatibility
with Linux/Kaggle environments.

Usage:
    python scripts/fix_manifest_paths.py --manifest data/multi_agent/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix manifest.csv paths to use forward slashes for cross-platform compatibility."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/multi_agent/processed/manifest.csv",
        help="Path to the manifest.csv file to fix (default: data/multi_agent/processed/manifest.csv)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original manifest before modifying",
    )
    return parser.parse_args()


def fix_manifest_paths(manifest_path: Path, create_backup: bool = False) -> int:
    """
    Fix all backslashes in sample_file column to forward slashes.
    
    Args:
        manifest_path: Path to the manifest.csv file
        create_backup: Whether to create a backup before modifying
    
    Returns:
        0 on success, 1 on error
    """
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        return 1
    
    # Create backup if requested
    if create_backup:
        backup_path = manifest_path.with_suffix(".csv.bak")
        shutil.copy2(manifest_path, backup_path)
        print(f"[OK] Created backup: {backup_path}")
    
    # Read all rows
    rows = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "sample_file" not in fieldnames:
            print(f"ERROR: Manifest missing 'sample_file' column: {manifest_path}")
            return 1
        
        for row in reader:
            # Replace backslashes with forward slashes
            if "sample_file" in row and row["sample_file"]:
                row["sample_file"] = row["sample_file"].replace("\\", "/")
            rows.append(row)
    
    # Write back
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[OK] Fixed {len(rows)} rows in {manifest_path}")
    print(f"[OK] All backslashes replaced with forward slashes")
    return 0


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    
    print("=" * 80)
    print("Fix Manifest Paths for Cross-Platform Compatibility")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Backup:   {args.backup}")
    print("=" * 80)
    print()
    
    return fix_manifest_paths(manifest_path, args.backup)


if __name__ == "__main__":
    raise SystemExit(main())
