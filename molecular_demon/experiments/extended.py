"""
Extended Ultra-Visual Figures: Molecular Computing Future
Visualizing the path from hardware to ambient molecular networks

Author: Kundai Sachikonye
Date: 2025-11-21
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Wedge, Rectangle, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns

# Styling
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 9,
})

COLORS = {
    'hardware': '#1f77b4',
    'molecular': '#2ca02c',
    'biological': '#9467bd',
    'planetary': '#d62728',
    'cosmic': '#ff7f0e',
}

PLANCK_TIME = 5.39116e-44

# ============================================================================
# FIGURE 9: Molecular Network Scaling (Room → Universe)
# ============================================================================

def create_molecular_scaling_roadmap(save_path='figure_molecular_scaling.png'):
    """
    Visualize scaling from current hardware to cosmic molecular network.
    """

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Define scales
    scales = [
        {
            'name': 'Hardware\n(Current)',
            'molecules': 1.95e3,
            'precision': 2e-66,
            'cost': 1000,
            'year': 2024,
            'color': COLORS['hardware']
        },
        {
            'name': 'Room\n(Near-term)',
            'molecules': 4.9e23,
            'precision': 1e-87,
            'cost': 1000,
            'year': 2027,
            'color': COLORS['molecular']
        },
        {
            'name': 'Body\n(Mid-term)',
            'molecules': 7e27,
            'precision': 1e-105,
            'cost': 10000,
            'year': 2035,
            'color': COLORS['biological']
        },
        {
            'name': 'Building\n(Long-term)',
            'molecules': 1e30,
            'precision': 1e-120,
            'cost': 100000,
            'year': 2045,
            'color': COLORS['planetary']
        },
        {
            'name': 'City\n(Far-term)',
            'molecules': 1e35,
            'precision': 1e-140,
            'cost': 1e6,
            'year': 2060,
            'color': COLORS['planetary']
        },
        {
            'name': 'Earth\n(Ultimate)',
            'molecules': 1e44,
            'precision': 1e-150,
            'cost': 1e9,
            'year': 2080,
            'color': COLORS['cosmic']
        }
    ]

    # ========================================================================
    # Panel A: Molecule Count Scaling (3D)
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, :], projection='3d')

    x = [s['year'] for s in scales]
    y = [np.log10(s['molecules']) for s in scales]
    z = [-np.log10(s['precision']) for s in scales]
    colors_3d = [s['color'] for s in scales]

    # Plot trajectory
    ax_a.plot(x, y, z, linewidth=3, color='gray', alpha=0.5, zorder=1)

    # Plot points
    for i, scale in enumerate(scales):
        ax_a.scatter(x[i], y[i], z[i], s=300, c=[colors_3d[i]],
                    edgecolors='white', linewidth=3, zorder=10)

        # Labels
        ax_a.text(x[i], y[i], z[i]+5, scale['name'],
                 fontsize=8, ha='center', fontweight='bold')

    ax_a.set_xlabel('\nYear', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('\nlog₁₀(Molecules)', fontsize=11, fontweight='bold')
    ax_a.set_zlabel('\nlog₁₀(1/Precision)', fontsize=11, fontweight='bold')
    ax_a.set_title('Molecular Network Scaling: Hardware → Cosmic',
                  fontsize=13, fontweight='bold', pad=20)
    ax_a.view_init(elev=20, azim=45)
    ax_a.grid(True, alpha=0.3)

    # ========================================================================
    # Panel B: Precision Roadmap
    # ========================================================================
    ax_b = fig.add_subplot(gs[1, 0])

    years = [s['year'] for s in scales]
    precisions = [s['precision'] for s in scales]
    colors_road = [s['color'] for s in scales]

    # Plot trajectory
    ax_b.plot(years, precisions, linewidth=3, color='gray', alpha=0.5, zorder=1)

    # Plot milestones
    for i, scale in enumerate(scales):
        ax_b.scatter(years[i], precisions[i], s=200, c=[colors_road[i]],
                    edgecolors='white', linewidth=2, zorder=10)

        # Annotations
        if i % 2 == 0:
            ax_b.annotate(scale['name'], xy=(years[i], precisions[i]),
                         xytext=(years[i], precisions[i]*1e10),
                         fontsize=8, ha='center', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', lw=1.5, color=colors_road[i]))

    # Planck time
    ax_b.axhline(PLANCK_TIME, color='red', linestyle='--', linewidth=2,
                label='Planck Time', zorder=0)

    ax_b.set_yscale('log')
    ax_b.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Time Precision (s)', fontsize=11, fontweight='bold')
    ax_b.set_title('Precision Roadmap: 2024-2080', fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.legend()

    # ========================================================================
    # Panel C: Cost vs Precision
    # ========================================================================
    ax_c = fig.add_subplot(gs[1, 1])

    costs = [s['cost'] for s in scales]

    # Bubble chart
    for i, scale in enumerate(scales):
        # Bubble size proportional to molecules
        size = np.log10(scale['molecules']) * 30

        ax_c.scatter(costs[i], precisions[i], s=size, c=[colors_road[i]],
                    alpha=0.6, edgecolors='white', linewidth=2)

        # Labels
        ax_c.text(costs[i], precisions[i], scale['name'],
                 fontsize=7, ha='center', va='center', fontweight='bold')

    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xlabel('Cost (USD)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Precision (s)', fontsize=11, fontweight='bold')
    ax_c.set_title('Cost-Precision Tradeoff', fontweight='bold')
    ax_c.grid(True, alpha=0.3)

    # Add note
    ax_c.text(0.95, 0.05, 'Bubble size ∝ log(molecules)',
             transform=ax_c.transAxes, ha='right', va='bottom',
             fontsize=8, style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========================================================================
    # Panel D: Orders Below Planck
    # ========================================================================
    ax_d = fig.add_subplot(gs[2, :])

    orders_below = [-np.log10(s['precision'] / PLANCK_TIME) for s in scales]
    names = [s['name'].replace('\n', ' ') for s in scales]

    # Bar chart with gradient
    bars = ax_d.bar(range(len(scales)), orders_below,
                    color=colors_road, edgecolor='white', linewidth=2)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, orders_below)):
        ax_d.text(i, val + 2, f'{val:.0f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_d.set_xticks(range(len(scales)))
    ax_d.set_xticklabels(names, fontsize=10)
    ax_d.set_ylabel('Orders Below Planck Time', fontsize=11, fontweight='bold')
    ax_d.set_title('Trans-Planckian Depth Evolution', fontweight='bold')
    ax_d.grid(True, alpha=0.3, axis='y')

    # Highlight regions
    ax_d.axhspan(0, 20, alpha=0.1, color='red', label='Planck regime')
    ax_d.axhspan(20, 50, alpha=0.1, color='yellow', label='Deep trans-Planckian')
    ax_d.axhspan(50, 200, alpha=0.1, color='green', label='Ultra trans-Planckian')
    ax_d.legend(loc='upper left')

    plt.suptitle('Molecular Computing Roadmap: From Hardware to Cosmic Scale',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# FIGURE 10: Chord Diagram - Molecular Cross-Connections
# ============================================================================

def create_chord_diagram(save_path='figure_chord_diagram.png'):
    """
    Chord diagram showing cross-connections between molecular types.
    """

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Define molecular types and their connections
    molecules = ['N₂', 'O₂', 'H₂O', 'CO₂', 'Ar', 'CPU', 'RAM', 'Screen']
    n_molecules = len(molecules)

    # Connection matrix (symmetric)
    connections = np.array([
        [0,    1000, 800,  200,  100,  50,   30,   20],   # N₂
        [1000, 0,    900,  300,  150,  60,   40,   25],   # O₂
        [800,  900,  0,    500,  200,  100,  80,   50],   # H₂O
        [200,  300,  500,  0,    80,   40,   30,   15],   # CO₂
        [100,  150,  200,  80,   0,    20,   15,   10],   # Ar
        [50,   60,   100,  40,   20,   0,    500,  300],  # CPU
        [30,   40,   80,   30,   15,   500,  0,    200],  # RAM
        [20,   25,   50,   15,   10,   300,  200,  0],    # Screen
    ])

    # Normalize
    connections = connections / connections.max() * 100

    # Colors for each molecule type
    colors_mol = [
        '#FF6B6B',  # N₂
        '#4ECDC4',  # O₂
        '#45B7D1',  # H₂O
        '#FFA07A',  # CO₂
        '#98D8C8',  # Ar
        '#1f77b4',  # CPU
        '#ff7f0e',  # RAM
        '#2ca02c',  # Screen
    ]

    # Draw outer circle
    radius = 1.0
    center = (0, 0)

    # Calculate positions for each molecule
    angles = np.linspace(0, 2*np.pi, n_molecules, endpoint=False)

    # Draw arcs for each molecule
    arc_width = 2*np.pi / n_molecules * 0.8
    for i, (angle, mol, color) in enumerate(zip(angles, molecules, colors_mol)):
        # Arc
        wedge = Wedge(center, radius, np.degrees(angle),
                     np.degrees(angle + arc_width),
                     width=0.1, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)

        # Label
        label_angle = angle + arc_width / 2
        label_radius = radius + 0.15
        x = label_radius * np.cos(label_angle)
        y = label_radius * np.sin(label_angle)
        ax.text(x, y, mol, ha='center', va='center',
               fontsize=12, fontweight='bold', color=color)

    # Draw connections (chords)
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            if connections[i, j] > 10:  # Only show significant connections
                # Start and end angles
                angle_i = angles[i] + arc_width / 2
                angle_j = angles[j] + arc_width / 2

                # Start and end points
                x_i = radius * 0.9 * np.cos(angle_i)
                y_i = radius * 0.9 * np.sin(angle_i)
                x_j = radius * 0.9 * np.cos(angle_j)
                y_j = radius * 0.9 * np.sin(angle_j)

                # Control point (for curve)
                control_x = 0
                control_y = 0

                # Create Bezier curve
                t = np.linspace(0, 1, 100)
                x_curve = (1-t)**2 * x_i + 2*(1-t)*t * control_x + t**2 * x_j
                y_curve = (1-t)**2 * y_i + 2*(1-t)*t * control_y + t**2 * y_j

                # Line width proportional to connection strength
                linewidth = connections[i, j] / 10

                # Color blend
                alpha = connections[i, j] / 100

                ax.plot(x_curve, y_curve, linewidth=linewidth,
                       color=colors_mol[i], alpha=alpha)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('Molecular Cross-Connections: Harmonic Network Topology',
                fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.text(0, -1.4, 'Line thickness ∝ connection strength | Color = source molecule',
           ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# FIGURE 11: Treemap - Enhancement Hierarchy
# ============================================================================

def create_treemap(hardware_data, save_path='figure_treemap.png'):
    """
    Treemap showing hierarchical breakdown of enhancement factors.
    """

    import squarify

    fig, ax = plt.subplots(figsize=(14, 10))

    enhancements = hardware_data['enhancement_factors']

    # Define hierarchy
    # Level 1: Total
    # Level 2: Network, BMD, Reflectance
    # Level 3: Hardware sources, BMD depths, Reflection iterations

    labels = [
        'Network\n{:.2e}×'.format(enhancements['network']),
        'BMD\n{:.2e}×'.format(enhancements['bmd']),
        'Reflectance\n{:.2e}×'.format(enhancements['reflectance']),
        'Screen LED\n3 osc',
        'CPU Clock\n3 osc',
        'RAM Refresh\n2 osc',
        'USB Polling\n2 osc',
        'Network\n3 osc',
    ]

    sizes = [
        np.log10(enhancements['network']),
        np.log10(enhancements['bmd']),
        np.log10(enhancements['reflectance']),
        3, 3, 2, 2, 3
    ]

    colors_tree = [
        COLORS['hardware'],
        COLORS['molecular'],
        COLORS['biological'],
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'
    ]

    # Create treemap
    squarify.plot(sizes=sizes, label=labels, color=colors_tree,
                 alpha=0.7, text_kwargs={'fontsize':10, 'fontweight':'bold'},
                 edgecolor='white', linewidth=3, ax=ax)

    ax.set_title('Enhancement Hierarchy: Treemap Visualization',
                fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# FIGURE 12: Parallel Coordinates - Parameter Space
# ============================================================================

def create_parallel_coordinates(save_path='figure_parallel_coordinates.png'):
    """
    Parallel coordinates plot showing multi-dimensional parameter space.
    """

    from pandas.plotting import parallel_coordinates
    import pandas as pd

    fig, ax = plt.subplots(figsize=(14, 8))

    # Define configurations
    configs = [
        {
            'Name': 'Hardware',
            'Molecules': np.log10(1.95e3),
            'BMD_Depth': 10,
            'Reflections': 10,
            'Precision': -np.log10(2e-66),
            'Cost': np.log10(1000),
            'Year': 2024
        },
        {
            'Name': 'Room',
            'Molecules': np.log10(4.9e23),
            'BMD_Depth': 15,
            'Reflections': 15,
            'Precision': -np.log10(1e-87),
            'Cost': np.log10(1000),
            'Year': 2027
        },
        {
            'Name': 'Body',
            'Molecules': np.log10(7e27),
            'BMD_Depth': 20,
            'Reflections': 20,
            'Precision': -np.log10(1e-105),
            'Cost': np.log10(10000),
            'Year': 2035
        },
        {
            'Name': 'Building',
            'Molecules': np.log10(1e30),
            'BMD_Depth': 25,
            'Reflections': 25,
            'Precision': -np.log10(1e-120),
            'Cost': np.log10(100000),
            'Year': 2045
        },
        {
            'Name': 'Earth',
            'Molecules': np.log10(1e44),
            'BMD_Depth': 30,
            'Reflections': 30,
            'Precision': -np.log10(1e-150),
            'Cost': np.log10(1e9),
            'Year': 2080
        }
    ]

    df = pd.DataFrame(configs)

    # Create parallel coordinates
    parallel_coordinates(df, 'Name', colormap='viridis',
                        linewidth=3, alpha=0.7, ax=ax)

    ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax.set_title('Multi-Dimensional Parameter Space: Configuration Comparison',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=10)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# FIGURE 13: Radar Chart - Method Comparison
# ============================================================================

def create_radar_chart(save_path='figure_radar_chart.png'):
    """
    Radar chart comparing different methods across multiple metrics.
    """

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Define metrics
    categories = ['Precision', 'Cost', 'Accessibility', 'Speed', 'Reliability', 'Scalability']
    n_categories = len(categories)

    # Define methods (normalized 0-10)
    methods = {
        'Mechanical Clock': [1, 10, 10, 5, 7, 3],
        'Atomic Clock': [7, 2, 3, 6, 8, 4],
        'Nuclear Clock': [8, 1, 1, 5, 6, 2],
        'Categorical (This Work)': [10, 9, 8, 10, 9, 10],
    }

    # Angles for each axis
    angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Plot each method
    colors_radar = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c']

    for (method, values), color in zip(methods.items(), colors_radar):
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_title('Method Comparison: Multi-Metric Radar Chart\n',
                fontsize=13, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# FIGURE 14: Waterfall Chart - Cumulative Enhancement
# ============================================================================

def create_waterfall_chart(hardware_data, save_path='figure_waterfall.png'):
    """
    Waterfall chart showing cumulative enhancement through cascade.
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    enhancements = hardware_data['enhancement_factors']

    # Define stages
    stages = ['Base', 'Network', 'BMD', 'Reflectance', 'Total']
    values = [
        1,
        enhancements['network'],
        enhancements['bmd'],
        enhancements['reflectance'],
        enhancements['total']
    ]

    # Calculate cumulative and incremental
    cumulative = [1]
    incremental = [1]

    for i in range(1, len(values)):
        cumulative.append(cumulative[-1] * values[i])
        incremental.append(values[i])

    # Log scale
    log_cumulative = [np.log10(c) for c in cumulative]
    log_incremental = [np.log10(i) for i in incremental]

    # Colors
    colors_water = [COLORS['hardware'], COLORS['hardware'],
                   COLORS['molecular'], COLORS['biological'], COLORS['cosmic']]

    # Plot bars
    for i in range(len(stages)):
        if i == 0:
            # Base bar
            ax.bar(i, log_cumulative[i], color=colors_water[i],
                  edgecolor='white', linewidth=2)
        else:
            # Incremental bar
            bottom = log_cumulative[i-1]
            height = log_cumulative[i] - log_cumulative[i-1]
            ax.bar(i, height, bottom=bottom, color=colors_water[i],
                  edgecolor='white', linewidth=2)

            # Connector line
            ax.plot([i-0.4, i-0.4], [log_cumulative[i-1], log_cumulative[i]],
                   color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels
    for i, (stage, val) in enumerate(zip(stages, cumulative)):
        ax.text(i, log_cumulative[i] + 0.5, f'{val:.2e}×',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax.set_ylabel('log₁₀(Cumulative Enhancement)', fontsize=11, fontweight='bold')
    ax.set_title('Waterfall Chart: Cumulative Enhancement Flow',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# FIGURE 15: Sunburst Diagram - Nested Hierarchy
# ============================================================================

def create_sunburst_diagram(hardware_data, save_path='figure_sunburst.png'):
    """
    Sunburst diagram showing nested molecular hierarchy.
    """

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Define hierarchy
    # Level 1: Total (center)
    # Level 2: Network, BMD, Reflectance
    # Level 3: Hardware sources, BMD depths, Reflections

    enhancements = hardware_data['enhancement_factors']
    hw_details = hardware_data['hardware_details']

    # Calculate sizes
    total = enhancements['total']
    network = enhancements['network']
    bmd = enhancements['bmd']
    reflectance = enhancements['reflectance']

    # Normalize for angles
    level2_total = network + bmd + reflectance
    network_angle = 360 * (network / level2_total)
    bmd_angle = 360 * (bmd / level2_total)
    ref_angle = 360 * (reflectance / level2_total)

    # Level 1: Center circle (Total)
    center_circle = Circle((0, 0), 0.3, facecolor=COLORS['cosmic'],
                          edgecolor='white', linewidth=3)
    ax.add_patch(center_circle)
    ax.text(0, 0, f'Total\n{total:.2e}×', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    # Level 2: Network, BMD, Reflectance
    wedges_level2 = [
        {'start': 0, 'extent': network_angle, 'color': COLORS['hardware'],
         'label': f'Network\n{network:.2e}×'},
        {'start': network_angle, 'extent': bmd_angle, 'color': COLORS['molecular'],
         'label': f'BMD\n{bmd:.2e}×'},
        {'start': network_angle + bmd_angle, 'extent': ref_angle, 'color': COLORS['biological'],
         'label': f'Reflectance\n{reflectance:.2e}×'},
    ]

    for wedge_data in wedges_level2:
        wedge = Wedge((0, 0), 0.6, wedge_data['start'],
                     wedge_data['start'] + wedge_data['extent'],
                     width=0.3, facecolor=wedge_data['color'], alpha=0.7,
                     edgecolor='white', linewidth=3)
        ax.add_patch(wedge)

        # Label
        angle_mid = wedge_data['start'] + wedge_data['extent'] / 2
        label_radius = 0.45
        x = label_radius * np.cos(np.radians(angle_mid))
        y = label_radius * np.sin(np.radians(angle_mid))
        ax.text(x, y, wedge_data['label'], ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

    # Level 3: Hardware sources (subdivide Network wedge)
    hw_total = sum(hw_details.values())
    hw_colors = {
        'screen_led': '#FF6B6B',
        'cpu_clock': '#4ECDC4',
        'ram_refresh': '#45B7D1',
        'usb_polling': '#FFA07A',
        'network': '#98D8C8',
    }

    current_angle = 0
    for hw_type, count in hw_details.items():
        hw_angle = network_angle * (count / hw_total)

        wedge = Wedge((0, 0), 0.9, current_angle, current_angle + hw_angle,
                     width=0.3, facecolor=hw_colors[hw_type], alpha=0.7,
                     edgecolor='white', linewidth=2)
        ax.add_patch(wedge)

        # Label
        angle_mid = current_angle + hw_angle / 2
        label_radius = 0.75
        x = label_radius * np.cos(np.radians(angle_mid))
        y = label_radius * np.sin(np.radians(angle_mid))
        ax.text(x, y, f'{hw_type.replace("_", " ").title()}\n{count}',
               ha='center', va='center', fontsize=7, fontweight='bold')

        current_angle += hw_angle

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title('Sunburst Diagram: Nested Enhancement Hierarchy',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_extended_visual_figures():
    """
    Generate all extended visual figures.
    """

    print("="*70)
    print("GENERATING EXTENDED VISUAL FIGURES")
    print("="*70)
    print()

    # Load data
    print("Loading data...")
    with open('hardware_trans_planckian_20251120_234553.json', 'r') as f:
        hardware_data = json.load(f)

    print("✓ Data loaded")
    print()

    # Generate figures
    figures = [
        ("Molecular Scaling Roadmap", lambda: create_molecular_scaling_roadmap()),
        ("Chord Diagram", lambda: create_chord_diagram()),
        ("Treemap", lambda: create_treemap(hardware_data)),
        ("Parallel Coordinates", lambda: create_parallel_coordinates()),
        ("Radar Chart", lambda: create_radar_chart()),
        ("Waterfall Chart", lambda: create_waterfall_chart(hardware_data)),
        ("Sunburst Diagram", lambda: create_sunburst_diagram(hardware_data)),
    ]

    for name, func in figures:
        print(f"Creating {name}...")
        try:
            func()
            print(f"✓ {name} complete")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("="*70)
    print("ALL EXTENDED FIGURES GENERATED")
    print("="*70)


if __name__ == "__main__":
    create_extended_visual_figures()
