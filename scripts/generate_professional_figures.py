"""
Generate professional-looking figures for GTNet presentation.
Uses academic paper style instead of AI-generated look.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import matplotlib.patheffects as pe
import numpy as np
import os

# Output directory
OUT = 'presentation_figures'
os.makedirs(OUT, exist_ok=True)

# Professional color scheme (based on academic papers)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'danger': '#C73E1D',       # Red
    'dark': '#1B2021',         # Dark gray
    'light': '#F4F4F8',        # Light gray
    'white': '#FFFFFF',
}

# Set global matplotlib style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'text.usetex': False,  # Set to True if you have LaTeX installed
})


def save_figure(fig, name, dpi=300):
    """Save figure with high quality."""
    fig.savefig(f'{OUT}/{name}.png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'✓ Saved {name}.png')


# ============================================================================
# SLIDE 2: Professional Pipeline Diagram
# ============================================================================
def generate_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    modules = [
        ('Sensors\n(Camera/LiDAR)', 1.5, COLORS['light']),
        ('Perception', 3.5, COLORS['primary']),
        ('Prediction\n(GTNet)', 5.5, COLORS['accent']),
        ('Planning', 7.5, COLORS['primary']),
        ('Control', 9.5, COLORS['light']),
    ]
    
    box_width = 1.6
    box_height = 1.2
    y_center = 1.5
    
    for i, (label, x, color) in enumerate(modules):
        # Draw box
        is_gtnet = i == 2
        lw = 3 if is_gtnet else 1.5
        ec = COLORS['danger'] if is_gtnet else COLORS['dark']
        
        rect = FancyBboxPatch(
            (x - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle='round,pad=0.1',
            facecolor=color,
            edgecolor=ec,
            linewidth=lw,
            zorder=3
        )
        ax.add_patch(rect)
        
        # Add text
        text_color = 'white' if color != COLORS['light'] else COLORS['dark']
        ax.text(x, y_center, label, ha='center', va='center',
                fontsize=11, color=text_color, fontweight='bold' if is_gtnet else 'normal',
                zorder=4)
        
        # Draw arrow to next module
        if i < len(modules) - 1:
            arrow = FancyArrowPatch(
                (x + box_width/2 + 0.05, y_center),
                (modules[i+1][1] - box_width/2 - 0.05, y_center),
                arrowstyle='->,head_width=0.4,head_length=0.4',
                color=COLORS['dark'],
                linewidth=2,
                zorder=2
            )
            ax.add_patch(arrow)
    
    ax.text(6, 0.3, 'GTNet: Multi-Agent Trajectory Prediction Module',
            ha='center', fontsize=10, style='italic', color=COLORS['dark'])
    
    save_figure(fig, 'slide02_pipeline_professional')


# ============================================================================
# SLIDE 5: GTNet Baseline Architecture (Clean Academic Style)
# ============================================================================
def generate_baseline_architecture():
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    modules = [
        ('GRU\nEncoder', 2, COLORS['primary'], 'History\n→ h_i'),
        ('Graph\nAggregation', 5, COLORS['success'], 'Mean\n→ z_i'),
        ('GRU\nDecoder', 8, COLORS['secondary'], 'Output\nK=1'),
    ]
    
    box_width = 2
    box_height = 1.5
    y_center = 2
    
    # Input
    ax.text(0.5, y_center, 'Input\n(x, y)', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
            edgecolor=COLORS['dark'], linewidth=1.5))
    
    for i, (label, x, color, desc) in enumerate(modules):
        # Main box
        rect = FancyBboxPatch(
            (x - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle='round,pad=0.1',
            facecolor=color,
            edgecolor=COLORS['dark'],
            linewidth=2,
            zorder=3
        )
        ax.add_patch(rect)
        
        ax.text(x, y_center + 0.2, label, ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=4)
        ax.text(x, y_center - 0.3, desc, ha='center', va='center',
                fontsize=8, color='white', style='italic', zorder=4)
        
        # Arrow from previous
        if i == 0:
            start_x = 1.2
        else:
            start_x = modules[i-1][1] + box_width/2 + 0.1
        
        arrow = FancyArrowPatch(
            (start_x, y_center),
            (x - box_width/2 - 0.1, y_center),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color=COLORS['dark'],
            linewidth=2,
            zorder=2
        )
        ax.add_patch(arrow)
    
    # Loss
    ax.text(10.5, y_center, 'Smooth L1\nLoss', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor=COLORS['danger'], 
            edgecolor=COLORS['dark'], linewidth=1.5), color='white', fontweight='bold')
    
    arrow = FancyArrowPatch(
        (9, y_center),
        (9.8, y_center),
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color=COLORS['dark'],
        linewidth=2,
        zorder=2
    )
    ax.add_patch(arrow)
    
    save_figure(fig, 'slide05_baseline_professional')


# ============================================================================
# SLIDE 9: ADE & FDE Metrics (Academic Style)
# ============================================================================
def generate_metrics_illustration():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Generate trajectories
    t = np.linspace(0, 1, 15)
    gt_x = 1 + 8 * t
    gt_y = 2 + 2 * np.sin(np.pi * t * 0.8)
    
    pred_x = 1 + 8 * t
    pred_y = 2.3 + 1.5 * np.sin(np.pi * t * 0.9) + 0.2
    
    # Plot trajectories
    ax.plot(gt_x, gt_y, '-', color=COLORS['success'], linewidth=3, 
            label='Ground Truth', zorder=3)
    ax.plot(pred_x, pred_y, '--', color=COLORS['danger'], linewidth=3, 
            label='Prediction', zorder=3)
    
    # Start point
    ax.plot(gt_x[0], gt_y[0], 'o', color=COLORS['dark'], markersize=12, zorder=4)
    
    # ADE arrows (every 3rd point)
    for i in range(3, len(t)-1, 3):
        ax.plot([gt_x[i], pred_x[i]], [gt_y[i], pred_y[i]], 
                color=COLORS['accent'], linewidth=1.5, alpha=0.7, zorder=2)
        ax.plot([gt_x[i], pred_x[i]], [gt_y[i], pred_y[i]], 
                'o', color=COLORS['accent'], markersize=4, zorder=2)
    
    # FDE arrow
    ax.annotate('', xy=(gt_x[-1], gt_y[-1]), xytext=(pred_x[-1], pred_y[-1]),
                arrowprops=dict(arrowstyle='<->', color=COLORS['danger'], 
                               linewidth=2.5, shrinkA=0, shrinkB=0))
    ax.text((gt_x[-1] + pred_x[-1])/2 + 0.2, (gt_y[-1] + pred_y[-1])/2,
            'FDE', fontsize=11, color=COLORS['danger'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['danger']))
    
    # End points
    ax.plot(gt_x[-1], gt_y[-1], 'D', color=COLORS['success'], markersize=10, zorder=4)
    ax.plot(pred_x[-1], pred_y[-1], 'D', color=COLORS['danger'], markersize=10, zorder=4)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Annotations
    ax.text(5, 0.5, 'ADE: Average Displacement Error (orange lines)',
            ha='center', fontsize=10, color=COLORS['accent'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['accent']))
    ax.text(5, 4.5, 'FDE: Final Displacement Error (red arrow)',
            ha='center', fontsize=10, color=COLORS['danger'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['danger']))
    
    save_figure(fig, 'slide09_metrics_professional')


# ============================================================================
# SLIDE 20: Results Comparison (Publication-Quality Bar Chart)
# ============================================================================
def generate_results_comparison():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    models = ['Baseline\n(000)', 'Multi_only\n(010)', 'GAT+Multi\n(110)', 'GTNet Full\n(111)']
    ade = np.array([1.84, 1.21, 1.05, 0.89])
    fde = np.array([4.12, 2.35, 1.98, 1.72])
    
    x = np.arange(len(models))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, ade, width, label='minADE (m)', 
                   color=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=1.5)
    bars2 = ax.bar(x + width/2, fde, width, label='minFDE (m)', 
                   color=COLORS['accent'], edgecolor=COLORS['dark'], linewidth=1.5)
    
    # Add value labels on bars
    for i, (a, f) in enumerate(zip(ade, fde)):
        ax.text(x[i] - width/2, a + 0.1, f'{a:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=COLORS['primary'])
        ax.text(x[i] + width/2, f + 0.1, f'{f:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=COLORS['accent'])
    
    # Improvement percentages
    baseline_ade, baseline_fde = ade[0], fde[0]
    for i in range(1, len(models)):
        pct_ade = (baseline_ade - ade[i]) / baseline_ade * 100
        pct_fde = (baseline_fde - fde[i]) / baseline_fde * 100
        
        ax.text(x[i] - width/2, ade[i]/2, f'↓{pct_ade:.0f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
        ax.text(x[i] + width/2, fde[i]/2, f'↓{pct_fde:.0f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    
    # Styling
    ax.set_xlabel('Model Variant', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error (meters)', fontsize=11, fontweight='bold')
    ax.set_title('GTNet Performance Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 5.0)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Highlight best result
    ax.text(3, 4.5, 'GTNet Full: 52% FDE improvement', ha='center', fontsize=11,
            color=COLORS['success'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['success'], linewidth=2))
    
    save_figure(fig, 'slide20_results_professional')


# ============================================================================
# SLIDE 1: Bird's Eye View (Improved)
# ============================================================================
def generate_bev_improved():
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Bird's Eye View: Multi-Agent Graph Interaction", 
                 fontsize=14, fontweight='bold', color=COLORS['dark'], pad=10)
    
    # Road
    road_h = Rectangle((-6, -1), 12, 2, color='#95a5a6', alpha=0.3, zorder=0)
    road_v = Rectangle((-1, -6), 2, 12, color='#95a5a6', alpha=0.3, zorder=0)
    ax.add_patch(road_h)
    ax.add_patch(road_v)
    
    # Lane markings
    for x in np.linspace(-5, -1.5, 4):
        ax.plot([x, x+0.4], [0, 0], '--', color='white', lw=1.5, alpha=0.8)
    for x in np.linspace(1.5, 5, 4):
        ax.plot([x, x+0.4], [0, 0], '--', color='white', lw=1.5, alpha=0.8)
    for y in np.linspace(-5, -1.5, 4):
        ax.plot([0, 0], [y, y+0.4], '--', color='white', lw=1.5, alpha=0.8)
    for y in np.linspace(1.5, 5, 4):
        ax.plot([0, 0], [y, y+0.4], '--', color='white', lw=1.5, alpha=0.8)
    
    # Ego vehicle
    ego = FancyBboxPatch((-0.4, -0.25), 0.8, 0.5, boxstyle='round,pad=0.05',
                         facecolor=COLORS['primary'], edgecolor=COLORS['dark'], 
                         linewidth=2, zorder=5)
    ax.add_patch(ego)
    ax.text(0, 0, 'Ego', ha='center', va='center', fontsize=10, 
            color='white', fontweight='bold', zorder=6)
    
    # Other agents
    agents = [
        (3, 0.2, COLORS['danger'], 'A1'),
        (-2.8, 0.1, COLORS['accent'], 'A2'),
        (0.15, 3, COLORS['success'], 'A3'),
        (0.1, -3.2, COLORS['secondary'], 'A4'),
    ]
    
    for x, y, color, label in agents:
        agent = FancyBboxPatch((x-0.35, y-0.2), 0.7, 0.4, boxstyle='round,pad=0.05',
                               facecolor=color, edgecolor=COLORS['dark'], 
                               linewidth=1.5, zorder=4)
        ax.add_patch(agent)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, 
                color='white', fontweight='bold', zorder=5)
        
        # Graph edge to ego
        ax.plot([0, x], [0, y], '-', color=color, lw=2, alpha=0.5, zorder=2)
        
        # History trajectory
        hist_x = np.linspace(x-1, x, 8) + np.random.uniform(-0.05, 0.05, 8)
        hist_y = np.linspace(y-0.5, y, 8) + np.random.uniform(-0.05, 0.05, 8)
        ax.plot(hist_x, hist_y, '-', color=color, lw=2, alpha=0.7, zorder=3)
        
        # Future predictions (3 modes)
        for k in range(3):
            pred_x = [x + (k*0.08)*np.cos(0.2*j + k*0.6) for j in range(1, 6)]
            pred_y = [y + 0.5*j + k*0.08 for j in range(1, 6)]
            ax.plot(pred_x, pred_y, '--', color=color, lw=1.5, alpha=0.4, zorder=3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='History'),
        Line2D([0], [0], color='gray', lw=1.5, ls='--', label='Prediction (K=3)'),
        Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Graph edge'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
              frameon=True, fancybox=True, shadow=True)
    
    save_figure(fig, 'slide01_bev_professional')


# ============================================================================
# SLIDE 3: Comparison Table (Professional)
# ============================================================================
def generate_comparison_table():
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='white')
    ax.axis('off')
    ax.set_title("Comparison of Interaction Modeling Approaches", 
                 fontsize=13, fontweight='bold', color=COLORS['dark'], pad=10)
    
    # Table data
    headers = ['Criterion', 'Single-Agent', 'Social Force', 'GTNet']
    rows = [
        ['Interaction Modeling', '✗ None', '✓ Physics', '✓ Graph Learning'],
        ['Learn from Data', '✗ No', '✗ No', '✓ End-to-end'],
        ['Flexibility', '✗ Poor', '✗ Rigid', '✓ Flexible'],
        ['Complex Interactions', '✗ Ignored', '△ Limited', '✓ Natural'],
    ]
    
    # Colors
    col_widths = [2.5, 2.2, 2.2, 2.2]
    row_height = 0.7
    start_y = 3.5
    
    # Headers
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        x = sum(col_widths[:i]) + 0.2
        rect = Rectangle((x, start_y), width, row_height, 
                        facecolor=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + width/2, start_y + row_height/2, header, 
               ha='center', va='center', fontsize=11, 
               color='white', fontweight='bold')
    
    # Rows
    for r, row in enumerate(rows):
        y = start_y - (r+1) * row_height
        for c, (cell, width) in enumerate(zip(row, col_widths)):
            x = sum(col_widths[:c]) + 0.2
            
            # Background color
            if c == 0:
                bg = COLORS['light']
            elif '✗' in cell:
                bg = '#fde8e8'
            elif '△' in cell:
                bg = '#fff4e6'
            else:
                bg = '#e8f5e9'
            
            rect = Rectangle((x, y), width, row_height, 
                           facecolor=bg, edgecolor='#bdc3c7', linewidth=1)
            ax.add_patch(rect)
            
            # Text color
            if '✓' in cell:
                color = COLORS['success']
            elif '✗' in cell:
                color = COLORS['danger']
            else:
                color = COLORS['dark']
            
            ax.text(x + width/2, y + row_height/2, cell, 
                   ha='center', va='center', fontsize=10, color=color,
                   fontweight='bold' if c > 0 else 'normal')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    
    save_figure(fig, 'slide03_comparison_professional')


# ============================================================================
# SLIDE 4: Problem Formulation (Improved)
# ============================================================================
def generate_problem_formulation():
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("Multi-Agent Trajectory Prediction Problem", 
                 fontsize=13, fontweight='bold', color=COLORS['dark'], pad=10)
    
    # Timeline
    ax.axvline(x=5.5, color=COLORS['dark'], lw=2, linestyle='--', alpha=0.7, zorder=1)
    ax.text(5.5, 4.7, 't = 0 (present)', ha='center', fontsize=10, 
            color=COLORS['dark'], fontweight='bold')
    
    # History box
    hist_box = FancyBboxPatch((0.5, 0.8), 4.5, 3.2, boxstyle='round,pad=0.15',
                              facecolor='#e3f2fd', edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(hist_box)
    ax.text(2.75, 4.3, 'History (N_obs = 20 steps ≈ 2s)', ha='center', fontsize=11,
            color=COLORS['primary'], fontweight='bold')
    
    # History trajectories
    for i in range(4):
        t = np.linspace(0, 1, 20)
        x = 1 + 3.5 * t
        y = 1.5 + 0.5*i + 0.2*np.sin(3*t + i)
        ax.plot(x, y, '-', color=COLORS['primary'], lw=2.5, alpha=0.7, zorder=3)
        ax.plot(x[-1], y[-1], 'o', color=COLORS['primary'], ms=7, zorder=4)
    
    # Future box
    future_box = FancyBboxPatch((6, 0.8), 5.5, 3.2, boxstyle='round,pad=0.15',
                                facecolor='#fff3e0', edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(future_box)
    ax.text(8.75, 4.3, 'Future (N_pred = 30 steps ≈ 3s, K=3 modes)', ha='center', fontsize=11,
            color=COLORS['accent'], fontweight='bold')
    
    # Future predictions (multimodal)
    colors_k = [COLORS['danger'], COLORS['success'], COLORS['secondary']]
    labels_k = ['Mode 1', 'Mode 2', 'Mode 3']
    
    for k, (color, label) in enumerate(zip(colors_k, labels_k)):
        t = np.linspace(0, 1, 30)
        x = 6.2 + 5 * t
        angle = [-0.4, 0, 0.4][k]
        y = 2.5 + k*0.4 + angle * t
        ax.plot(x, y, '--', color=color, lw=2.5, label=label, zorder=3)
        ax.plot(x[-1], y[-1], 's', color=color, ms=8, zorder=4)
    
    ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Metrics
    ax.text(8.75, 0.3, 'minADE = min_k Σ||ŷ_k - y||₂/T', ha='center', fontsize=9,
            color=COLORS['dark'], bbox=dict(boxstyle='round', facecolor='white', 
            edgecolor=COLORS['primary'], linewidth=1.5))
    
    save_figure(fig, 'slide04_problem_professional')


# ============================================================================
# SLIDE 7: Mean Regression Problem (Improved)
# ============================================================================
def generate_mean_regression():
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Mean Regression Problem at Intersection", 
                 fontsize=13, fontweight='bold', color=COLORS['dark'], pad=10)
    
    # Roads
    road_h = Rectangle((-5, -0.8), 10, 1.6, color='#95a5a6', alpha=0.3, zorder=0)
    road_v = Rectangle((-0.8, -5), 1.6, 10, color='#95a5a6', alpha=0.3, zorder=0)
    ax.add_patch(road_h)
    ax.add_patch(road_v)
    
    # History
    hist_y = np.linspace(-4, -1, 10)
    ax.plot(np.zeros(10), hist_y, '-', color=COLORS['primary'], lw=4, 
            label='History', zorder=3)
    ax.plot(0, -1, 'o', color=COLORS['primary'], ms=12, zorder=4)
    
    # Ground truth options
    t = np.linspace(0, 1, 20)
    
    # Left turn
    xl = -t * 3
    yl = 1 + t * 0.5
    ax.plot(xl, yl, '-', color=COLORS['success'], lw=3, 
            label='Turn Left (GT)', alpha=0.8, zorder=3)
    ax.text(-2.5, 1.8, 'Left', fontsize=10, color=COLORS['success'], fontweight='bold')
    
    # Straight
    xs = np.zeros(20)
    ys = np.linspace(1, 4, 20)
    ax.plot(xs, ys, '-', color=COLORS['primary'], lw=3, 
            label='Straight (GT)', alpha=0.8, zorder=3)
    ax.text(0.5, 3.5, 'Straight', fontsize=10, color=COLORS['primary'], fontweight='bold')
    
    # Right turn
    xr = t * 3
    yr = 1 + t * 0.5
    ax.plot(xr, yr, '-', color=COLORS['secondary'], lw=3, 
            label='Turn Right (GT)', alpha=0.8, zorder=3)
    ax.text(2.5, 1.8, 'Right', fontsize=10, color=COLORS['secondary'], fontweight='bold')
    
    # Baseline prediction (mean = straight with slight offset)
    xb = np.linspace(0, 0.2, 20)
    yb = np.linspace(1, 4, 20)
    ax.plot(xb, yb, '--', color=COLORS['danger'], lw=4, 
            label='Baseline Prediction ❌', zorder=4)
    
    ax.annotate('Baseline\n(Average = Wrong!)', xy=(0.3, 2.5), fontsize=10,
                color=COLORS['danger'], fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor=COLORS['danger'], linewidth=2))
    
    ax.legend(loc='lower left', fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    ax.text(0, -4.7, 'K=1 prediction averages all modes → unrealistic trajectory',
            ha='center', fontsize=10, color=COLORS['danger'], style='italic',
            bbox=dict(boxstyle='round', facecolor='#fde8e8', 
                     edgecolor=COLORS['danger'], linewidth=1.5))
    
    save_figure(fig, 'slide07_mean_regression_professional')


# ============================================================================
# SLIDE 11: Three Improvements Overview
# ============================================================================
def generate_three_improvements():
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title("Three Key Improvements to GTNet", 
                 fontsize=13, fontweight='bold', color=COLORS['dark'], pad=10)
    
    improvements = [
        (2, COLORS['primary'], 'Improvement 1\nGAT', 
         'Graph Attention\nNetwork', 'Solves:\nBlind aggregation'),
        (6, COLORS['success'], 'Improvement 2\nWTA Loss', 
         'Multimodal +\nWinner-Takes-All', 'Solves:\nMode collapse'),
        (10, COLORS['secondary'], 'Improvement 3\nAdaptive R', 
         'Adaptive\nRadius', 'Solves:\nPoor graph quality'),
    ]
    
    for x, color, title, method, problem in improvements:
        # Main card
        card = FancyBboxPatch((x-1.5, 0.5), 3, 4, boxstyle='round,pad=0.15',
                             facecolor='white', edgecolor=color, linewidth=3)
        ax.add_patch(card)
        
        # Title box
        title_box = FancyBboxPatch((x-1.4, 3.2), 2.8, 1.2, boxstyle='round,pad=0.1',
                                   facecolor=color)
        ax.add_patch(title_box)
        ax.text(x, 3.8, title, ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')
        
        # Method
        ax.text(x, 2.3, method, ha='center', va='center', fontsize=10,
                color=COLORS['dark'], fontweight='bold')
        
        # Arrow
        arrow = FancyArrowPatch((x, 2.0), (x, 1.5),
                               arrowstyle='->,head_width=0.3,head_length=0.3',
                               color=color, linewidth=2)
        ax.add_patch(arrow)
        
        # Problem solved
        ax.text(x, 1.0, problem, ha='center', va='center', fontsize=9,
                color=COLORS['dark'], style='italic')
    
    ax.text(6, 0.1, 'Three independent improvements → Ablation Study measures individual contributions',
            ha='center', fontsize=10, color=COLORS['dark'], style='italic')
    
    save_figure(fig, 'slide11_improvements_professional')


# ============================================================================
# SLIDE 17: Ablation Study Design
# ============================================================================
def generate_ablation_design():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')
    ax.axis('off')
    ax.set_title("Ablation Study: 2³ = 8 Experimental Variants", 
                 fontsize=13, fontweight='bold', color=COLORS['dark'], pad=10)
    
    variants = [
        ('000', 'Baseline', 0, 0, 0, COLORS['danger']),
        ('001', 'Radius_only', 0, 0, 1, COLORS['accent']),
        ('010', 'Multi_only', 0, 1, 0, '#f1c40f'),
        ('011', 'Multi+Radius', 0, 1, 1, '#2ecc71'),
        ('100', 'GAT_only', 1, 0, 0, '#3498db'),
        ('101', 'GAT+Radius', 1, 0, 1, '#9b59b6'),
        ('110', 'GAT+Multi', 1, 1, 0, '#1abc9c'),
        ('111', 'GTNet_Full', 1, 1, 1, COLORS['success']),
    ]
    
    headers = ['Code', 'Variant Name', 'GAT', 'WTA Loss', 'Adaptive R']
    col_x = [0.5, 2.0, 5.0, 6.5, 8.0]
    col_widths = [1.2, 2.5, 1.2, 1.2, 1.2]
    row_height = 0.55
    start_y = 5.0
    
    # Headers
    for i, (header, x, w) in enumerate(zip(headers, col_x, col_widths)):
        rect = Rectangle((x, start_y), w, row_height,
                        facecolor=COLORS['primary'], edgecolor=COLORS['dark'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, start_y + row_height/2, header,
               ha='center', va='center', fontsize=10,
               color='white', fontweight='bold')
    
    # Rows
    for r, (code, name, gat, wta, rad, color) in enumerate(variants):
        y = start_y - (r+1) * row_height
        
        # Background
        if code in ('000', '111'):
            bg = '#f0f8ff' if code == '111' else '#fff5f5'
        else:
            bg = 'white'
        
        bg_rect = Rectangle((0.5, y), 8.7, row_height,
                           facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(bg_rect)
        
        # Code
        ax.text(col_x[0] + col_widths[0]/2, y + row_height/2, code,
               ha='center', va='center', fontsize=9,
               color=COLORS['dark'], fontweight='bold', family='monospace')
        
        # Name
        ax.text(col_x[1] + col_widths[1]/2, y + row_height/2, name,
               ha='center', va='center', fontsize=9,
               color=COLORS['dark'], fontweight='bold' if code in ('000', '111') else 'normal')
        
        # Checkmarks
        for ci, val in enumerate([gat, wta, rad]):
            x_pos = col_x[2+ci] + col_widths[2+ci]/2
            symbol = '✓' if val else '✗'
            sym_color = COLORS['success'] if val else COLORS['danger']
            ax.text(x_pos, y + row_height/2, symbol,
                   ha='center', va='center', fontsize=14,
                   color=sym_color, fontweight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    save_figure(fig, 'slide17_ablation_professional')


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating professional figures for GTNet presentation...")
    print("=" * 60)
    
    # Original 4 figures
    generate_pipeline()
    generate_baseline_architecture()
    generate_metrics_illustration()
    generate_results_comparison()
    
    # New figures
    generate_bev_improved()
    generate_comparison_table()
    generate_problem_formulation()
    generate_mean_regression()
    generate_three_improvements()
    generate_ablation_design()
    
    print("=" * 60)
    print(f"✓ All 10 figures generated successfully in '{OUT}/' directory")
    print("\nGenerated figures:")
    print("  1. slide01_bev_professional.png")
    print("  2. slide02_pipeline_professional.png")
    print("  3. slide03_comparison_professional.png")
    print("  4. slide04_problem_professional.png")
    print("  5. slide05_baseline_professional.png")
    print("  7. slide07_mean_regression_professional.png")
    print("  9. slide09_metrics_professional.png")
    print(" 11. slide11_improvements_professional.png")
    print(" 17. slide17_ablation_professional.png")
    print(" 20. slide20_results_professional.png")
    print("\nThese figures use academic paper style and are publication-quality.")
    print("You can now replace the AI-generated images in your presentation.")
