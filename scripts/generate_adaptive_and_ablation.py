"""
Generate Adaptive Radius and Ablation Study images.
Based on the SVG specifications provided.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os

# Output directory
OUT = 'presentation_figures'
os.makedirs(OUT, exist_ok=True)

# Colors
COLORS = {
    'primary': '#1a5fa8',
    'blue': '#2d7dd2',
    'green': '#2ecc71',
    'green2': '#27ae60',
    'gray': '#95a5a6',
    'road': '#b0b8c4',
    'dark': '#1a2a3a',
    'orange': '#e67e22',
    'light_blue': '#3498db',
    'white': '#FFFFFF',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
})


# ============================================================================
# IMAGE 1: Adaptive Radius (Fixed - no overlap)
# ============================================================================
def generate_adaptive_radius():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle('Adaptive Radius: r_i = r_base + α × v_i', 
                 fontsize=16, fontweight='bold', color=COLORS['dark'], y=0.98)
    
    # ========== LEFT PANEL: High Speed ==========
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.set_aspect('equal')
    ax_left.axis('off')
    
    # Title
    ax_left.text(5, 9.2, 'Tốc độ cao (cao tốc)', ha='center', fontsize=13, 
                fontweight='bold', color=COLORS['primary'])
    ax_left.text(5, 8.7, 'r_i = lớn → bán kính rộng', ha='center', fontsize=11, 
                color=COLORS['primary'])
    
    # Road
    road_left = Rectangle((0.5, 4), 9, 2, facecolor=COLORS['road'], 
                          edgecolor='none', zorder=1)
    ax_left.add_patch(road_left)
    
    # Lane markings
    for x in np.linspace(1, 8.5, 8):
        ax_left.plot([x, x+0.3], [5, 5], '--', color='white', lw=1.5, alpha=0.8)
    
    # Large radius circle
    circle_left = Circle((5, 5), 3.1, fill=True, facecolor='#dbeeff', 
                        alpha=0.5, edgecolor=COLORS['blue'], 
                        linewidth=2.5, linestyle='--', zorder=2)
    ax_left.add_patch(circle_left)
    
    # r_i label (positioned above to avoid overlap)
    ax_left.plot([5, 5], [8.1, 8.3], '--', color=COLORS['blue'], lw=1.5)
    label_box = FancyBboxPatch((3.8, 8.25), 2.4, 0.5, boxstyle='round,pad=0.05',
                               facecolor='white', edgecolor=COLORS['blue'], 
                               linewidth=1.5, zorder=10)
    ax_left.add_patch(label_box)
    ax_left.text(5, 8.5, 'r_i = 60m', ha='center', va='center', fontsize=12, 
                fontweight='bold', color=COLORS['blue'], zorder=11)
    
    # Ego vehicle
    ego_left = FancyBboxPatch((4.7, 4.7), 0.6, 0.6, boxstyle='round,pad=0.05',
                             facecolor=COLORS['dark'], edgecolor='none', zorder=5)
    ax_left.add_patch(ego_left)
    ax_left.text(5, 5, 'i', ha='center', va='center', fontsize=11, 
                color='white', fontweight='bold', zorder=6)
    
    # Neighbor agents (green squares) - scattered around
    agents_left = [
        (2.2, 4.8), (2.8, 5.0), (2.3, 5.4), (2.9, 5.5),
        (3.6, 4.8), (3.7, 5.4), (2.5, 6.0)
    ]
    for i, (x, y) in enumerate(agents_left):
        color = COLORS['green'] if i % 2 == 0 else COLORS['green2']
        agent = FancyBboxPatch((x, y), 0.5, 0.4, boxstyle='round,pad=0.03',
                              facecolor=color, edgecolor='none', zorder=4)
        ax_left.add_patch(agent)
        # Connection line
        ax_left.plot([5, x+0.25], [5, y+0.2], '-', color=COLORS['blue'], 
                    lw=1, alpha=0.5, zorder=3)
    
    # Direction arrow
    arrow_left = FancyArrowPatch((5.6, 5), (7.5, 5), 
                                arrowstyle='->,head_width=0.3,head_length=0.3',
                                color=COLORS['blue'], linewidth=2.5, zorder=4)
    ax_left.add_patch(arrow_left)
    
    # Caption
    ax_left.text(5, 1.5, '⚡ Tốc độ cao', ha='center', fontsize=13, 
                fontweight='bold', color=COLORS['orange'])
    ax_left.text(5, 1.0, 'Nhiều tác nhân trong vùng tương tác', 
                ha='center', fontsize=10, color='#555')
    
    # ========== RIGHT PANEL: Low Speed ==========
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)
    ax_right.set_aspect('equal')
    ax_right.axis('off')
    
    # Title
    ax_right.text(5, 9.2, 'Tốc độ thấp (khu dân cư)', ha='center', fontsize=13, 
                 fontweight='bold', color=COLORS['primary'])
    ax_right.text(5, 8.7, 'r_i = nhỏ → bán kính hẹp', ha='center', fontsize=11, 
                 color=COLORS['primary'])
    
    # Road
    road_right = Rectangle((0.5, 4), 9, 2, facecolor=COLORS['road'], 
                           edgecolor='none', zorder=1)
    ax_right.add_patch(road_right)
    
    # Lane markings
    for x in np.linspace(1, 8.5, 8):
        ax_right.plot([x, x+0.3], [5, 5], '--', color='white', lw=1.5, alpha=0.8)
    
    # Small radius circle
    circle_right = Circle((5, 5), 1.4, fill=True, facecolor='#dbeeff', 
                         alpha=0.5, edgecolor=COLORS['blue'], 
                         linewidth=2.5, linestyle='--', zorder=2)
    ax_right.add_patch(circle_right)
    
    # r_i label (positioned to the right side, above)
    ax_right.plot([6.4, 7.2], [5, 7.5], '--', color=COLORS['blue'], lw=1.5)
    label_box_r = FancyBboxPatch((6.5, 7.5), 2.2, 0.5, boxstyle='round,pad=0.05',
                                 facecolor='white', edgecolor=COLORS['blue'], 
                                 linewidth=1.5, zorder=10)
    ax_right.add_patch(label_box_r)
    ax_right.text(7.6, 7.75, 'r_i = 45m', ha='center', va='center', fontsize=12, 
                 fontweight='bold', color=COLORS['blue'], zorder=11)
    
    # Ego vehicle
    ego_right = FancyBboxPatch((4.7, 4.7), 0.6, 0.6, boxstyle='round,pad=0.05',
                              facecolor=COLORS['dark'], edgecolor='none', zorder=5)
    ax_right.add_patch(ego_right)
    ax_right.text(5, 5, 'i', ha='center', va='center', fontsize=11, 
                 color='white', fontweight='bold', zorder=6)
    
    # In-range agents (2 green)
    agents_in = [(4.0, 5.0), (5.5, 4.9)]
    for x, y in agents_in:
        agent = FancyBboxPatch((x, y), 0.5, 0.4, boxstyle='round,pad=0.03',
                              facecolor=COLORS['green'], edgecolor='none', zorder=4)
        ax_right.add_patch(agent)
        ax_right.plot([5, x+0.25], [5, y+0.2], '-', color=COLORS['blue'], 
                     lw=1, alpha=0.6, zorder=3)
    
    # Out-of-range agents (gray)
    agents_out = [(1.5, 4.9), (1.8, 5.5), (8.0, 4.9), (8.5, 5.4)]
    for x, y in agents_out:
        agent = FancyBboxPatch((x, y), 0.5, 0.4, boxstyle='round,pad=0.03',
                              facecolor=COLORS['gray'], edgecolor='none', 
                              alpha=0.4, zorder=4)
        ax_right.add_patch(agent)
    
    # Direction arrow
    arrow_right = FancyArrowPatch((5.6, 5), (7.5, 5), 
                                 arrowstyle='->,head_width=0.3,head_length=0.3',
                                 color=COLORS['blue'], linewidth=2.5, zorder=4)
    ax_right.add_patch(arrow_right)
    
    # Caption
    ax_right.text(5, 1.5, '🐢 Tốc độ thấp', ha='center', fontsize=13, 
                 fontweight='bold', color=COLORS['light_blue'])
    ax_right.text(5, 1.0, 'Ít tác nhân, chỉ láng giềng gần', 
                 ha='center', fontsize=10, color='#555')
    
    # Legend (bottom center of figure)
    fig.text(0.25, 0.02, '■', fontsize=14, color=COLORS['green'])
    fig.text(0.27, 0.02, 'Trong vùng tương tác', fontsize=9)
    
    fig.text(0.45, 0.02, '■', fontsize=14, color=COLORS['gray'], alpha=0.5)
    fig.text(0.47, 0.02, 'Ngoài vùng (bị bỏ qua)', fontsize=9, color='#888')
    
    fig.text(0.68, 0.02, '■', fontsize=14, color=COLORS['dark'])
    fig.text(0.70, 0.02, 'Xe ego (i)', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(f'{OUT}/slide14_adaptive_radius_fixed.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('✓ Saved slide14_adaptive_radius_fixed.png')


# ============================================================================
# IMAGE 2: Ablation Study Table (White background, no ADE/FDE)
# ============================================================================
def generate_ablation_table():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    
    # Title
    ax.text(5.5, 6.7, 'Ablation Study — 8 Biến thể (2³ Combinations)', 
            ha='center', fontsize=16, fontweight='bold', color=COLORS['dark'])
    
    # Table data
    variants = [
        ('Baseline  (000)', False, False, False, False),
        ('GAT_only  (100)', True, False, False, False),
        ('Multi_only (010)', False, True, False, False),
        ('Radius_only (001)', False, False, True, False),
        ('GAT+Multi  (110)', True, True, False, False),
        ('GAT+Radius (101)', True, False, True, False),
        ('Multi+Radius (011)', False, True, True, False),
        ('GTNet Full (111)', True, True, True, True),
    ]
    
    # Column positions
    col_x = [0.5, 3.8, 6.0, 8.2]
    col_w = [3.0, 2.0, 2.0, 2.0]
    row_h = 0.6
    header_h = 0.7
    start_y = 5.8
    
    # Header
    header = Rectangle((0.5, start_y), 10, header_h, 
                       facecolor=COLORS['dark'], edgecolor='none')
    ax.add_patch(header)
    
    headers = ['Biến thể', 'GAT', 'WTA', 'Adaptive Radius']
    for i, (h, x, w) in enumerate(zip(headers, col_x, col_w)):
        ax.text(x + w/2, start_y + header_h/2, h, ha='center', va='center',
               fontsize=12, fontweight='bold', color='#38bdf8')
    
    # Rows
    for r, (name, gat, wta, rad, is_best) in enumerate(variants):
        y = start_y - (r+1) * row_h
        
        # Row background
        if is_best:
            bg_color = '#FFF9E6'
            edge_color = '#f5a623'
            edge_width = 2.5
        elif r % 2 == 0:
            bg_color = '#f7f9fc'
            edge_color = '#d0d8e4'
            edge_width = 0.5
        else:
            bg_color = '#eef2f7'
            edge_color = '#d0d8e4'
            edge_width = 0.5
        
        row_rect = Rectangle((0.5, y), 10, row_h, facecolor=bg_color, 
                            edgecolor=edge_color, linewidth=edge_width)
        ax.add_patch(row_rect)
        
        # Name
        name_color = '#f5a623' if is_best else '#2c3e50'
        name_weight = 'bold' if is_best else 'normal'
        ax.text(col_x[0] + col_w[0]/2, y + row_h/2, name, 
               ha='center', va='center', fontsize=11, 
               color=name_color, fontweight=name_weight)
        
        # Checkmarks/crosses
        for ci, val in enumerate([gat, wta, rad]):
            x_pos = col_x[ci+1] + col_w[ci+1]/2
            
            # Cell background
            cell_bg = '#E8F5E9' if val else '#FFEBEE'
            cell_rect = Rectangle((col_x[ci+1] + 0.05, y + 0.05), 
                                 col_w[ci+1] - 0.1, row_h - 0.1,
                                 facecolor=cell_bg, edgecolor='#d0d8e4', 
                                 linewidth=0.5)
            ax.add_patch(cell_rect)
            
            # Draw symbol using lines instead of Unicode
            if val:
                # Draw checkmark (✓) using lines
                check_x = [x_pos - 0.15, x_pos - 0.05, x_pos + 0.2]
                check_y = [y + row_h/2, y + row_h/2 - 0.15, y + row_h/2 + 0.2]
                ax.plot(check_x, check_y, '-', color='#27ae60', linewidth=3, 
                       solid_capstyle='round', solid_joinstyle='round')
            else:
                # Draw X using two lines
                offset = 0.15
                # Line 1: top-left to bottom-right
                ax.plot([x_pos - offset, x_pos + offset], 
                       [y + row_h/2 + offset, y + row_h/2 - offset],
                       '-', color='#e74c3c', linewidth=3, 
                       solid_capstyle='round')
                # Line 2: bottom-left to top-right
                ax.plot([x_pos - offset, x_pos + offset], 
                       [y + row_h/2 - offset, y + row_h/2 + offset],
                       '-', color='#e74c3c', linewidth=3, 
                       solid_capstyle='round')
    
    # GTNet Full highlight border
    highlight_y = start_y - 8 * row_h
    highlight_rect = Rectangle((0.52, highlight_y + 0.02), 9.96, row_h - 0.04,
                              fill=False, edgecolor='#f5a623', 
                              linewidth=3, linestyle='-')
    ax.add_patch(highlight_rect)
    
    plt.tight_layout()
    fig.savefig(f'{OUT}/slide17_ablation_table_clean.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('✓ Saved slide17_ablation_table_clean.png')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Generating Adaptive Radius and Ablation Study images...")
    print("=" * 70)
    
    generate_adaptive_radius()
    generate_ablation_table()
    
    print("=" * 70)
    print("✓ All images generated successfully!")
    print(f"  Output directory: {OUT}/")
    print("\nGenerated files:")
    print("  1. slide14_adaptive_radius_fixed.png - Fixed overlap issue")
    print("  2. slide17_ablation_table_clean.png - White bg, no ADE/FDE columns")
    print("=" * 70)
