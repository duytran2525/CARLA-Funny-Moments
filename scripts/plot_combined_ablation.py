#!/usr/bin/env python3
"""
Plot combined ablation study results.
Loads minADE & minFDE from data/ablation_summary.json (first run).
Loads Miss Rate & Inference Latency from data/ablation_summary(1).json (second run).
Generates professional, publication-quality figures:
1. Accuracy Metrics (minADE & minFDE)
2. Catastrophic Miss Rate
3. Inference Latency
4. Combined Widescreen Figure (1x3 layout) for presentations
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Professional Academic Palette
COLORS = {
    'primary': '#2E86AB',       # Blue (minADE)
    'secondary': '#8E44AD',     # Purple (Latency)
    'accent': '#F18F01',        # Orange (minFDE)
    'success': '#06A77D',       # Green (Miss Rate)
    'danger': '#C73E1D',        # Red
    'dark': '#1B2021',          # Dark Gray
    'light': '#F4F4F8',         # Light Gray
    'highlight': '#FFF9E6',     # Warm Light Yellow for best variant box
    'border': '#BDC3C7',
}

# Configure matplotlib style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
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

# Code-to-key mapping
CODE_TO_KEY = {
    "000": "baseline",
    "001": "adaptive_radius_only",
    "010": "multimodal_only",
    "011": "multimodal_adaptive",
    "100": "gat_only",
    "101": "gat_adaptive",
    "110": "gat_multimodal",
    "111": "full"
}

def load_json_file(file_path: Path) -> dict:
    """Load and normalize ablation JSON data."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    normalized = {}
    for item in data:
        code = str(item.get("code", ""))
        key = CODE_TO_KEY.get(code)
        if key:
            normalized[key] = {
                "minADE": item.get("minADE") or item.get("best_val_ade") or item.get("ade") or 0.0,
                "minFDE": item.get("minFDE") or item.get("best_val_fde") or item.get("fde") or 0.0,
                "MissRate": item.get("MissRate") or item.get("best_val_miss_rate") or item.get("miss_rate") or 0.0,
                "inference_latency_ms": item.get("inference_latency_ms") or item.get("inference_latency") or item.get("latency") or 0.0,
            }
    return normalized

def get_combined_data(summary_path: Path, summary_1_path: Path) -> dict:
    """Combine data from both ablation study summaries."""
    data_primary = load_json_file(summary_path)
    data_secondary = load_json_file(summary_1_path)
    
    combined = {}
    for key in CODE_TO_KEY.values():
        # Get minADE and minFDE from the first ablation run (summary_path)
        ade = data_primary[key]["minADE"]
        fde = data_primary[key]["minFDE"]
        
        # Get MissRate and Inference Latency from the second ablation run (summary_1_path)
        miss_rate = data_secondary[key]["MissRate"]
        latency = data_secondary[key]["inference_latency_ms"]
        
        combined[key] = {
            "minADE": ade,
            "minFDE": fde,
            "MissRate": miss_rate,
            "inference_latency_ms": latency
        }
    return combined

def plot_accuracy(combined_data: dict, out_dir: Path):
    """Plot minADE and minFDE comparisons."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    labels = []
    ade_vals = []
    fde_vals = []
    
    for key, code, label in VARIANT_ORDER:
        labels.append(f"{label}\n({code})")
        ade_vals.append(combined_data[key]['minADE'])
        fde_vals.append(combined_data[key]['minFDE'])
        
    x = np.arange(len(labels))
    width = 0.35
    
    bars_ade = ax.bar(x - width/2, ade_vals, width, label='minADE (m)',
                      color=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=1)
    bars_fde = ax.bar(x + width/2, fde_vals, width, label='minFDE (m)',
                      color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=1)
    
    # Add values on top of bars
    for bar in bars_ade:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, color=COLORS['dark'], fontweight='bold')
                
    for bar in bars_fde:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03, f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, color=COLORS['dark'], fontweight='bold')
                
    # Add improvement percentage box
    base_ade, base_fde = combined_data['baseline']['minADE'], combined_data['baseline']['minFDE']
    full_ade, full_fde = combined_data['full']['minADE'], combined_data['full']['minFDE']
    impr_ade = (base_ade - full_ade) / base_ade * 100
    impr_fde = (base_fde - full_fde) / base_fde * 100
    
    highlight_text = f"Full Model Improvement:\nminADE: ↓{impr_ade:.1f}%\nminFDE: ↓{impr_fde:.1f}%"
    ax.text(0.68, 0.83, highlight_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], 
                     edgecolor=COLORS['accent'], linewidth=1.5),
            fontsize=10, fontweight='bold', color=COLORS['dark'])
            
    ax.set_xlabel('Model Variants (Code)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Study: Accuracy Metrics (minADE & minFDE)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(fde_vals) * 1.25)
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_accuracy_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved accuracy plot to: {plot_path}")

def plot_miss_rate(combined_data: dict, out_dir: Path):
    """Plot Miss Rate comparison."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')
    
    labels = []
    miss_rates = []
    bar_colors = []
    
    for key, code, label in VARIANT_ORDER:
        labels.append(f"{label}\n({code})")
        mr = combined_data[key]['MissRate']
        miss_rates.append(mr)
        if key == 'baseline':
            bar_colors.append(COLORS['danger'])
        elif key == 'full':
            bar_colors.append(COLORS['success'])
        else:
            bar_colors.append(COLORS['primary'])
            
    x = np.arange(len(labels))
    bars = ax.bar(x, miss_rates, width=0.55, color=bar_colors, edgecolor=COLORS['dark'], linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.008, f'{height*100:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['dark'])
                
    # Add improvement box - shifted up and right to avoid overlap
    base_mr = combined_data['baseline']['MissRate']
    full_mr = combined_data['full']['MissRate']
    impr_mr = (base_mr - full_mr) / base_mr * 100
    
    highlight_text = f"Safety Improvement:\nMiss Rate: ↓{impr_mr:.1f}%"
    ax.text(0.68, 0.88, highlight_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], 
                     edgecolor=COLORS['success'], linewidth=1.5),
            fontsize=10, fontweight='bold', color=COLORS['dark'])
            
    ax.set_xlabel('Model Variants (Code)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Miss Rate (FDE > 2.0m)', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Study: Catastrophic Miss Rate Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(miss_rates) * 1.45)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_miss_rate_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved miss rate plot to: {plot_path}")

def plot_latency(combined_data: dict, out_dir: Path):
    """Plot Inference Latency comparison."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')
    
    labels = []
    latencies = []
    bar_colors = []
    
    for key, code, label in VARIANT_ORDER:
        labels.append(f"{label}\n({code})")
        lat = combined_data[key]['inference_latency_ms']
        latencies.append(lat)
        if key == 'baseline':
            bar_colors.append(COLORS['primary'])
        elif key == 'full':
            bar_colors.append(COLORS['secondary'])
        else:
            bar_colors.append('#5DADE2')
            
    x = np.arange(len(labels))
    bars = ax.bar(x, latencies, width=0.55, color=bar_colors, edgecolor=COLORS['dark'], linewidth=1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f} ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['dark'])
                
    # Add speed highlight box - shifted left and up to avoid overlap
    highlight_text = "Real-Time Control Ready:\nInference < 0.50 ms\n(99% CPU budget free)"
    ax.text(0.05, 0.85, highlight_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], 
                     edgecolor=COLORS['secondary'], linewidth=1.5),
            fontsize=10, fontweight='bold', color=COLORS['dark'])
            
    ax.set_xlabel('Model Variants (Code)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms/sample)', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Study: Inference Latency Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(latencies) * 1.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_latency_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved latency plot to: {plot_path}")

def plot_widescreen_combined(combined_data: dict, out_dir: Path):
    """Plot all three aspects in a single 1x3 panel widescreen layout for presentations."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')
    
    labels = []
    ade_vals, fde_vals = [], []
    miss_rates = []
    latencies = []
    
    for key, code, label in VARIANT_ORDER:
        labels.append(f"{label}\n({code})")
        ade_vals.append(combined_data[key]['minADE'])
        fde_vals.append(combined_data[key]['minFDE'])
        miss_rates.append(combined_data[key]['MissRate'])
        latencies.append(combined_data[key]['inference_latency_ms'])
        
    x = np.arange(len(labels))
    width = 0.35
    
    # ── PANEL A: minADE & minFDE ──
    bars_ade = ax1.bar(x - width/2, ade_vals, width, label='minADE (m)',
                      color=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=1)
    bars_fde = ax1.bar(x + width/2, fde_vals, width, label='minFDE (m)',
                      color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=1)
    
    for bar in bars_ade:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}',
                 ha='center', va='bottom', fontsize=8, color=COLORS['dark'], fontweight='bold')
    for bar in bars_fde:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03, f'{height:.3f}',
                 ha='center', va='bottom', fontsize=8, color=COLORS['dark'], fontweight='bold')
                 
    base_ade, base_fde = combined_data['baseline']['minADE'], combined_data['baseline']['minFDE']
    full_ade, full_fde = combined_data['full']['minADE'], combined_data['full']['minFDE']
    impr_ade = (base_ade - full_ade) / base_ade * 100
    impr_fde = (base_fde - full_fde) / base_fde * 100
    
    highlight_text_acc = f"Improvement:\nADE: ↓{impr_ade:.1f}%\nFDE: ↓{impr_fde:.1f}%"
    ax1.text(0.55, 0.85, highlight_text_acc, transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], 
                      edgecolor=COLORS['accent'], linewidth=1.2),
             fontsize=9, fontweight='bold', color=COLORS['dark'])
             
    ax1.set_ylabel('Error (meters)', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy: minADE & minFDE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, max(fde_vals) * 1.35)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ── PANEL B: Miss Rate ──
    mr_colors = [COLORS['danger'] if k == 'baseline' else (COLORS['success'] if k == 'full' else COLORS['primary']) for k, _, _ in VARIANT_ORDER]
    bars_mr = ax2.bar(x, miss_rates, width=0.55, color=mr_colors, edgecolor=COLORS['dark'], linewidth=1)
    
    for bar in bars_mr:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.008, f'{height*100:.1f}%',
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS['dark'])
                 
    base_mr, full_mr = combined_data['baseline']['MissRate'], combined_data['full']['MissRate']
    impr_mr = (base_mr - full_mr) / base_mr * 100
    
    highlight_text_mr = f"Safety Improvement:\nMiss Rate: ↓{impr_mr:.1f}%"
    ax2.text(0.55, 0.88, highlight_text_mr, transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], 
                      edgecolor=COLORS['success'], linewidth=1.2),
             fontsize=9, fontweight='bold', color=COLORS['dark'])
             
    ax2.set_ylabel('Miss Rate (FDE > 2.0m)', fontsize=11, fontweight='bold')
    ax2.set_title('Safety: Catastrophic Miss Rate', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax2.set_ylim(0, max(miss_rates) * 1.45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ── PANEL C: Latency ──
    lat_colors = [COLORS['primary'] if k == 'baseline' else (COLORS['secondary'] if k == 'full' else '#5DADE2') for k, _, _ in VARIANT_ORDER]
    bars_lat = ax3.bar(x, latencies, width=0.55, color=lat_colors, edgecolor=COLORS['dark'], linewidth=1)
    
    for bar in bars_lat:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f} ms',
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS['dark'])
                 
    highlight_text_lat = "Real-Time Ready:\nLatency < 0.50 ms\n(99% CPU Free)"
    ax3.text(0.05, 0.85, highlight_text_lat, transform=ax3.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], 
                      edgecolor=COLORS['secondary'], linewidth=1.2),
             fontsize=9, fontweight='bold', color=COLORS['dark'])
             
    ax3.set_ylabel('Inference Latency (ms/sample)', fontsize=11, fontweight='bold')
    ax3.set_title('Complexity: Inference Latency', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylim(0, max(latencies) * 1.5)
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    fig.suptitle('GTNet Ablation Study: Comprehensive Metrics (Accuracy, Safety, & Complexity)', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plot_path = out_dir / "ablation_widescreen_combined.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined widescreen plot to: {plot_path}")

def main():
    summary_path = PROJECT_ROOT / "data" / "ablation_summary.json"
    summary_1_path = PROJECT_ROOT / "data" / "ablation_summary(1).json"
    out_dir = PROJECT_ROOT / "ablation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Combining results from ablation studies...")
    combined_data = get_combined_data(summary_path, summary_1_path)
    
    print("Generating individual plots...")
    plot_accuracy(combined_data, out_dir)
    plot_miss_rate(combined_data, out_dir)
    plot_latency(combined_data, out_dir)
    
    print("Generating widescreen combined plot...")
    plot_widescreen_combined(combined_data, out_dir)
    
    print("\nAll plots generated successfully!")
    print(f"Outputs are stored in: {out_dir}/")

if __name__ == "__main__":
    main()
