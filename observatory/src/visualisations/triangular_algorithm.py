#!/usr/bin/env python3
"""
Triangular Amplification Algorithm Visualization
================================================

Visualizes triangular teleportation and categorical amplification results.

Creates publication-quality multi-panel figure showing:
- Panel A: FTL ratio vs distance across all experiments
- Panel B: Amplification factors per wavelength band
- Panel C: Multi-band parallel validation
- Panel D: Reconstruction error analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime

def load_results():
    """Load triangular teleportation results"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_file = os.path.join(project_root, 'results',
                                'triangular_teleportation_20251115_052027.json')

    with open(results_file, 'r') as f:
        return json.load(f)

def create_publication_figure(results):
    """Create 4-panel publication figure"""

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })

    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.06)

    # Extract data
    experiments = results['results']
    distances = [exp['distance_m'] for exp in experiments]
    molecules = [exp['molecule'] for exp in experiments]

    # Panel A: FTL Ratio vs Distance (Multi-Band)
    ax1 = fig.add_subplot(gs[0, 0])

    band_colors = {'blue': '#4472C4', 'green': '#70AD47', 'red': '#C55A11'}
    band_markers = {'blue': 'o', 'green': 's', 'red': '^'}

    for band_name in ['blue', 'green', 'red']:
        ftl_ratios = []
        dist_vals = []
        for exp in experiments:
            for tri in exp['triangles']:
                if tri['band'] == band_name:
                    ftl_ratios.append(tri['ftl_ratio'])
                    dist_vals.append(exp['distance_m'])

        ax1.plot(dist_vals, ftl_ratios, marker=band_markers[band_name],
                linestyle='-', linewidth=2, markersize=8,
                color=band_colors[band_name], alpha=0.7,
                label=f'{band_name.capitalize()} ({int(experiments[0]["triangles"][[t["band"] for t in experiments[0]["triangles"]].index(band_name)]["wavelength_nm"])} nm)',
                markeredgecolor='black', markeredgewidth=1)

    # FTL threshold line
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5,
                label='FTL Threshold (ratio = 1)', zorder=1, alpha=0.7)

    ax1.set_xlabel('Distance [meters]', fontweight='bold')
    ax1.set_ylabel('FTL Ratio [v_eff / c]', fontweight='bold')
    ax1.set_title('(A) FTL Ratio vs Distance Across Wavelength Bands',
                  fontweight='bold', loc='left')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')

    # Panel B: Amplification Factors by Band
    ax2 = fig.add_subplot(gs[0, 1])

    x_positions = np.arange(len(experiments))
    width = 0.25

    for i, band_name in enumerate(['blue', 'green', 'red']):
        amplifications = []
        for exp in experiments:
            for tri in exp['triangles']:
                if tri['band'] == band_name:
                    amplifications.append(tri['amplification'])
                    break

        offset = (i - 1) * width
        bars = ax2.bar(x_positions + offset, amplifications, width,
                      label=band_name.capitalize(), color=band_colors[band_name],
                      alpha=0.7, edgecolor='black', linewidth=1.2)

        # Add value labels
        for j, (bar, amp) in enumerate(zip(bars, amplifications)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{amp:.2f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    # Reference line at 1.0 (no amplification)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Experiment', fontweight='bold')
    ax2.set_ylabel('Amplification Factor', fontweight='bold')
    ax2.set_title('(B) Triangular Amplification by Wavelength Band',
                  fontweight='bold', loc='left')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f"{mol}\n{d} m" for mol, d in zip(molecules, distances)],
                        fontsize=7)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel C: Multi-Band Parallel Validation
    ax3 = fig.add_subplot(gs[1, 0])

    # For each experiment, show the three bands
    for i, exp in enumerate(experiments):
        ftl_ratios = [tri['ftl_ratio'] for tri in exp['triangles']]
        bands = [tri['band'] for tri in exp['triangles']]

        # Plot as grouped bars
        for j, (band, ftl) in enumerate(zip(bands, ftl_ratios)):
            color = band_colors[band]
            ax3.bar(i * 4 + j, ftl, color=color, alpha=0.7,
                   edgecolor='black', linewidth=1.2, width=0.8)

    # FTL threshold
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='FTL Threshold', alpha=0.7, zorder=1)

    # X-axis labels
    tick_positions = [i * 4 + 1 for i in range(len(experiments))]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([f"{mol}\n{d} m" for mol, d in zip(molecules, distances)],
                        fontsize=8)

    ax3.set_ylabel('FTL Ratio', fontweight='bold')
    ax3.set_title('(C) Multi-Band Parallel Categorical Validation',
                  fontweight='bold', loc='left')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add legend for bands
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=band_colors[b], edgecolor='black',
                            label=b.capitalize(), alpha=0.7)
                      for b in ['blue', 'green', 'red']]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                     linewidth=2, label='FTL Threshold'))
    ax3.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    # Panel D: Reconstruction Error Analysis
    ax4 = fig.add_subplot(gs[1, 1])

    # Scatter plot of error vs distance for each band
    for band_name in ['blue', 'green', 'red']:
        errors = []
        dist_vals = []
        for exp in experiments:
            for tri in exp['triangles']:
                if tri['band'] == band_name:
                    errors.append(tri['reconstruction_error'])
                    dist_vals.append(exp['distance_m'])

        ax4.scatter(dist_vals, errors, s=120, marker=band_markers[band_name],
                   color=band_colors[band_name], alpha=0.7,
                   edgecolors='black', linewidth=1.5,
                   label=band_name.capitalize(), zorder=3)

        # Fit trend line
        log_dist = np.log10(dist_vals)
        coeffs = np.polyfit(log_dist, errors, 1)
        dist_fit = np.logspace(np.log10(min(distances)), np.log10(max(distances)), 50)
        error_fit = coeffs[0] * np.log10(dist_fit) + coeffs[1]
        ax4.plot(dist_fit, error_fit, '--', color=band_colors[band_name],
                linewidth=1.5, alpha=0.5, zorder=2)

    # Error tolerance threshold
    ax4.axhline(y=5.0, color='orange', linestyle=':', linewidth=2,
               label='Error Tolerance (5.0)', alpha=0.7)

    ax4.set_xlabel('Distance [meters]', fontweight='bold')
    ax4.set_ylabel('Reconstruction Error [categorical units]', fontweight='bold')
    ax4.set_title('(D) Reconstruction Error vs Distance',
                  fontweight='bold', loc='left')
    ax4.set_xscale('log')
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--', which='both')

    # Add statistics text box
    total_triangles = results['summary']['total_triangles']
    successful_triangles = results['summary']['successful_triangles']
    textstr = (f'Total triangles: {total_triangles}\n'
               f'Successful: {successful_triangles}\n'
               f'Success rate: {successful_triangles/total_triangles*100:.1f}%\n'
               f'Bands: RGB (3×)')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5)
    ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Overall title
    fig.suptitle('Triangular Amplification: Multi-Band Parallel Categorical Prediction',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig

def save_figure(fig):
    """Save figure to results directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'triangular_amplification_{timestamp}.png'
    filepath = os.path.join(results_dir, filename)

    fig.savefig(filepath, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {filepath}")

    plt.close(fig)
    return filepath

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" Triangular Amplification Visualization")
    print("="*70)

    # Load results
    print("\nLoading results...")
    results = load_results()

    # Create figure
    print("Creating publication figure...")
    fig = create_publication_figure(results)

    # Save
    print("Saving figure...")
    filepath = save_figure(fig)

    # Summary stats
    summary = results['summary']
    print("\n" + "="*70)
    print(" Visualization Complete")
    print("="*70)
    print(f"\nTotal experiments: {summary['total_experiments']}")
    print(f"Total triangles: {summary['total_triangles']}")
    print(f"Successful triangles: {summary['successful_triangles']}")
    print(f"Success rate: {summary['successful_triangles']/summary['total_triangles']*100:.1f}%")
    print(f"\nFigure: {filepath}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
