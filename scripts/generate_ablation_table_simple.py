"""
Generate simplified ablation study table with white background.
No minADE/minFDE columns - only showing which components are enabled.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import os

# Output directory
OUT = 'presentation_figures'
os.makedirs(OUT, exist_ok=True)

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',
    'success': '#06A77D',
    'danger': '#C73E1D',
    'dark': '#1B2021',
    'light': '#F4F4F8',
    'white': '#FFFFFF',
    'border': '#BDC3C7',
}

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
})


def generate_ablation_table_simple():
    """Generate simplified ablation study table."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.axis('off')
    ax.set_title("Ablation Study: 2³ = 8 Experimental Variants", 
                 fontsize=14, fontweight='bold', color=COLORS['dark'], pad=15)
    
    # Table data
    variants = [
        ('000', 'Baseline', 0, 0, 0),
        ('001', 'Radius_only', 0, 0, 1),
        ('010', 'Multi_only', 0, 1, 0),
        ('011', 'Multi+Radius', 0, 1, 1),
        ('100', 'GAT_only', 1, 0, 0),
        ('101', 'GAT+Radius', 1, 0, 1),
        ('110', 'GAT+Multi', 1, 1, 0),
        ('111', 'GTNet_Full', 1, 1, 1),
    ]
    
    # Headers
    headers = ['Code', 'Variant Name', 'GAT', 'WTA Loss', 'Adaptive Radius']
    col_x = [0.5, 2.2, 5.5, 6.8, 8.1]
    col_widths = [1.4, 2.8, 1.0, 1.0, 1.2]
    row_height = 0.6
    start_y = 5.0
    
    # Draw header row
    for i, (header, x, w) in enumerate(zip(headers, col_x, col_widths)):
        rect = Rectangle((x, start_y), w, row_height,
                        facecolor=COLORS['primary'], 
                        edgecolor=COLORS['dark'], 
                        linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, start_y + row_height/2, header,
               ha='center', va='center', fontsize=11,
               color='white', fontweight='bold')
    
    # Draw data rows
    for r, (code, name, gat, wta, rad) in enumerate(variants):
        y = start_y - (r+1) * row_height
        
        # Determine background color
        if code == '111':  # GTNet Full - highlight
            bg = '#FFF9E6'
            edge_color = '#F39C12'
            edge_width = 2.5
        elif code == '000':  # Baseline
            bg = '#FFF5F5'
            edge_color = COLORS['danger']
            edge_width = 1.5
        else:
            bg = 'white'
            edge_color = COLORS['border']
            edge_width = 1
        
        # Background rectangle for entire row
        bg_rect = Rectangle((0.5, y), sum(col_widths), row_height,
                           facecolor=bg, 
                           edgecolor=edge_color, 
                           linewidth=edge_width,
                           zorder=1)
        ax.add_patch(bg_rect)
        
        # Code column
        ax.text(col_x[0] + col_widths[0]/2, y + row_height/2, code,
               ha='center', va='center', fontsize=10,
               color=COLORS['dark'], fontweight='bold', 
               family='monospace', zorder=2)
        
        # Name column
        name_weight = 'bold' if code in ('000', '111') else 'normal'
        ax.text(col_x[1] + col_widths[1]/2, y + row_height/2, name,
               ha='center', va='center', fontsize=10,
               color=COLORS['dark'], fontweight=name_weight, zorder=2)
        
        # Component columns (GAT, WTA, Adaptive R)
        for ci, val in enumerate([gat, wta, rad]):
            x_pos = col_x[2+ci] + col_widths[2+ci]/2
            
            # Draw cell background
            cell_bg = '#E8F5E9' if val else '#FFEBEE'
            cell_rect = Rectangle((col_x[2+ci] + 0.05, y + 0.05), 
                                  col_widths[2+ci] - 0.1, row_height - 0.1,
                                  facecolor=cell_bg, 
                                  edgecolor=COLORS['border'],
                                  linewidth=0.5,
                                  zorder=1)
            ax.add_patch(cell_rect)
            
            # Draw checkmark or X
            symbol = '✓' if val else '✗'
            sym_color = COLORS['success'] if val else COLORS['danger']
            ax.text(x_pos, y + row_height/2, symbol,
                   ha='center', va='center', fontsize=16,
                   color=sym_color, fontweight='bold', zorder=2)
    
    # Add note at bottom
    ax.text(5.5, 0.3, 
            'Each variant tests a different combination of improvements',
            ha='center', fontsize=10, color=COLORS['dark'], 
            style='italic')
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    
    # Save
    fig.savefig(f'{OUT}/slide17_ablation_simple.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('✓ Saved slide17_ablation_simple.png')


if __name__ == '__main__':
    print("Generating simplified ablation study table...")
    print("=" * 60)
    generate_ablation_table_simple()
    print("=" * 60)
    print("✓ Table generated successfully!")
    print(f"  Output: {OUT}/slide17_ablation_simple.png")
    print("\nFeatures:")
    print("  - White background")
    print("  - No minADE/minFDE columns")
    print("  - Clean, professional design")
    print("  - GTNet Full highlighted in yellow")
    print("  - Baseline highlighted in light red")
