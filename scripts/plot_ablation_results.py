#!/usr/bin/env python3
"""
Plot Ablation Study Results from ablation_results.json.
Generates professional publication-quality figures for evaluation metrics:
- minADE & minFDE Comparison
- Miss Rate Comparison
- Efficiency Trade-offs (Training Time vs Inference Latency)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path just in case
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Professional Academic Palette
COLORS = {
    'primary': '#2E86AB',       # Blue
    'secondary': '#A23B72',     # Purple
    'accent': '#F18F01',        # Orange
    'success': '#06A77D',       # Green
    'danger': '#C73E1D',        # Red
    'dark': '#1B2021',          # Dark Gray
    'light': '#F4F4F8',         # Light Gray
    'highlight': '#FFF9E6',     # Warm Light Yellow for best variant
    'border': '#BDC3C7',
}

# Set consistent styling
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
})

# Logical ordering for variants with their binary codes and descriptive names
VARIANT_ORDER = [
    ("baseline", "000", "Baseline"),
    ("adaptive_radius_only", "001", "Radius Only"),
    ("multimodal_only", "010", "Multi Only"),
    ("multimodal_adaptive", "011", "Multi+Radius"),
    ("gat_only", "100", "GAT Only"),
    ("gat_adaptive", "101", "GAT+Radius"),
    ("gat_multimodal", "110", "GAT+Multi"),
    ("full", "111", "GTNet Full"),
]

def load_results(json_path: Path) -> dict:
    """Load and parse ablation_results.json or ablation_summary.json.
    
    Supports both:
    1. Dict structure mapping variant names to metric dicts.
    2. List structure containing dicts with "code", "variant", and "best_val_ade"/"best_val_fde".
    """
    if not json_path.exists():
        print(f"Error: Results file not found at {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    normalized = {}
    
    # Map code to standard keys
    code_to_key = {
        "000": "baseline",
        "001": "adaptive_radius_only",
        "010": "multimodal_only",
        "011": "multimodal_adaptive",
        "100": "gat_only",
        "101": "gat_adaptive",
        "110": "gat_multimodal",
        "111": "full"
    }
    
    if isinstance(data, list):
        for item in data:
            code = str(item.get("code", ""))
            # If code is missing, try to map from variant name
            if not code and "variant" in item:
                var_name = str(item["variant"]).lower()
                for k, c, l in VARIANT_ORDER:
                    if k == var_name or l.lower() == var_name or var_name.startswith(k):
                        code = c
                        break
            
            key = code_to_key.get(code)
            if not key:
                # Try using run_tag directly
                run_tag = str(item.get("run_tag", "")).lower()
                for c, sk in code_to_key.items():
                    if c in run_tag:
                        key = sk
                        break
            
            if key:
                normalized[key] = {
                    "minADE": item.get("minADE") or item.get("best_val_ade") or item.get("ade") or 0.0,
                    "minFDE": item.get("minFDE") or item.get("best_val_fde") or item.get("fde") or 0.0,
                    "MissRate": item.get("MissRate") or item.get("best_val_miss_rate") or item.get("miss_rate") or 0.0,
                    "train_time_seconds": item.get("train_time_seconds") or item.get("train_time") or 0.0,
                    "inference_latency_ms": item.get("inference_latency_ms") or item.get("inference_latency") or item.get("latency") or 0.0,
                }
    elif isinstance(data, dict):
        for k, v in data.items():
            std_key = k if k in code_to_key.values() else code_to_key.get(str(k))
            if not std_key:
                for c, sk in code_to_key.items():
                    if c in str(k) or sk in str(k).lower():
                        std_key = sk
                        break
            
            if not std_key:
                std_key = k
                
            if isinstance(v, dict):
                normalized[std_key] = {
                    "minADE": v.get("minADE") or v.get("best_val_ade") or v.get("ade") or 0.0,
                    "minFDE": v.get("minFDE") or v.get("best_val_fde") or v.get("fde") or 0.0,
                    "MissRate": v.get("MissRate") or v.get("best_val_miss_rate") or v.get("miss_rate") or 0.0,
                    "train_time_seconds": v.get("train_time_seconds") or v.get("train_time") or 0.0,
                    "inference_latency_ms": v.get("inference_latency_ms") or v.get("inference_latency") or v.get("latency") or 0.0,
                }
                
    return normalized

def plot_accuracy_metrics(results: dict, out_dir: Path):
    """Plot minADE and minFDE comparisons."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    codes = []
    labels = []
    ade_vals = []
    fde_vals = []
    
    for key, code, label in VARIANT_ORDER:
        if key in results:
            codes.append(code)
            labels.append(f"{label}\n({code})")
            ade_vals.append(results[key]['minADE'])
            fde_vals.append(results[key]['minFDE'])
    
    if not ade_vals:
        print("Warning: No matching variants found in JSON results.")
        plt.close(fig)
        return

    x = np.arange(len(labels))
    width = 0.35
    
    # Draw bars
    bars_ade = ax.bar(x - width/2, ade_vals, width, label='minADE (m)',
                      color=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=1)
    bars_fde = ax.bar(x + width/2, fde_vals, width, label='minFDE (m)',
                      color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=1)
    
    # Add values on top of bars
    for i, bar in enumerate(bars_ade):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height < 1 else height + 0.05, f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, color=COLORS['dark'], fontweight='bold')
                
    for i, bar in enumerate(bars_fde):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height < 1 else height + 0.05, f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, color=COLORS['dark'], fontweight='bold')
    
    # Highlight full variant improvement percentage
    if 'baseline' in results and 'full' in results:
        base_fde = results['baseline']['minFDE']
        full_fde = results['full']['minFDE']
        if base_fde > 0:
            impr_fde = (base_fde - full_fde) / base_fde * 100
            
            base_ade = results['baseline']['minADE']
            full_ade = results['full']['minADE']
            impr_ade = (base_ade - full_ade) / base_ade * 100
            
            highlight_text = f"Full Model Improvement:\n minADE: ↓{impr_ade:.1f}%\n minFDE: ↓{impr_fde:.1f}%"
            ax.text(0.68, 0.85, highlight_text, transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], 
                             edgecolor=COLORS['accent'], linewidth=1.5),
                    fontsize=10, fontweight='bold', color=COLORS['dark'])
    
    # Styling
    ax.set_xlabel('Model Variants (Code)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Study: Accuracy Metrics (minADE & minFDE)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylim(0, max(fde_vals) * 1.25 if fde_vals else 1.0)
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_accuracy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved accuracy plot to: {plot_path}")

def plot_miss_rate(results: dict, out_dir: Path):
    """Plot MissRate comparisons."""
    labels = []
    miss_rates = []
    bar_colors = []
    
    for key, code, label in VARIANT_ORDER:
        if key in results:
            mr = results[key].get('MissRate', 0.0)
            labels.append(f"{label}\n({code})")
            miss_rates.append(mr)
            if key == 'baseline':
                bar_colors.append(COLORS['danger'])
            elif key == 'full':
                bar_colors.append(COLORS['success'])
            else:
                bar_colors.append(COLORS['primary'])
                
    if not miss_rates or all(mr == 0.0 for mr in miss_rates):
        print("Skipping miss rate plot (no non-zero MissRate values found).")
        return
        
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    x = np.arange(len(labels))
    bars = ax.bar(x, miss_rates, width=0.55, color=bar_colors, edgecolor=COLORS['dark'], linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['dark'])
                
    ax.set_xlabel('Model Variants (Code)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Miss Rate (FDE > 2.0m)', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Study: Catastrophic Miss Rate Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(miss_rates) * 1.25 if miss_rates else 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_miss_rate_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved miss rate plot to: {plot_path}")

def plot_efficiency_tradeoff(results: dict, out_dir: Path):
    """Plot training time vs inference latency trade-off."""
    labels = []
    latencies = []
    train_times = []
    
    for key, code, label in VARIANT_ORDER:
        if key in results:
            lat = results[key].get('inference_latency_ms', 0.0)
            tt = results[key].get('train_time_seconds', 0.0)
            if lat > 0.0 or tt > 0.0:
                labels.append(f"{label}\n({code})")
                latencies.append(lat)
                train_times.append(tt)
            
    if not latencies or not train_times or all(l == 0.0 for l in latencies) or all(t == 0.0 for t in train_times):
        print("Skipping efficiency trade-off plot (no latency or training time measurements found).")
        return
        
    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
    x = np.arange(len(labels))
    width = 0.35
    
    color_lat = COLORS['secondary']
    color_time = COLORS['primary']
    
    bars_lat = ax1.bar(x - width/2, latencies, width, label='Inference Latency (ms)',
                       color=color_lat, edgecolor=COLORS['dark'], linewidth=1)
    ax1.set_xlabel('Model Variants (Code)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inference Latency (ms)', fontsize=11, fontweight='bold', color=color_lat)
    ax1.tick_params(axis='y', labelcolor=color_lat)
    
    ax2 = ax1.twinx()
    bars_time = ax2.bar(x + width/2, train_times, width, label='Training Time (s)',
                        color=color_time, edgecolor=COLORS['dark'], linewidth=1, alpha=0.85)
    ax2.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold', color=color_time)
    ax2.tick_params(axis='y', labelcolor=color_time)
    
    for bar in bars_lat:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}ms',
                 ha='center', va='bottom', fontsize=8, color=color_lat, fontweight='bold')
                 
    for bar in bars_time:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}s',
                 ha='center', va='bottom', fontsize=8, color=color_time, fontweight='bold')
                 
    ax1.set_title('Ablation Study: Efficiency & Complexity Trade-off', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    
    lines, labels_leg = ax1.get_legend_handles_labels()
    lines2, labels_leg2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_leg + labels_leg2, loc='upper left', frameon=True, fancybox=True)
    
    ax1.set_ylim(0, max(latencies) * 1.3 if latencies else 1.0)
    ax2.set_ylim(0, max(train_times) * 1.3 if train_times else 1.0)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_efficiency_tradeoff.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved efficiency plot to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Ablation Study metrics from JSON.")
    parser.add_argument('--results-json', default='ablation_results/ablation_results.json',
                        help='Path to ablation_results.json file.')
    parser.add_argument('--out-dir', default='ablation_plots',
                        help='Directory to save output figures.')
    args = parser.parse_args()
    
    json_path = Path(args.results_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading ablation results from: {json_path}")
    results = load_results(json_path)
    
    print("Generating figures...")
    plot_accuracy_metrics(results, out_dir)
    plot_miss_rate(results, out_dir)
    plot_efficiency_tradeoff(results, out_dir)
    print(f"\n[OK] All plots saved successfully in: {out_dir}/")

if __name__ == '__main__':
    main()
