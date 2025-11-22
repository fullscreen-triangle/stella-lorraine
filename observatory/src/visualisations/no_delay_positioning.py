#!/usr/bin/env python3
"""
Zero-Delay Positioning Visualization
=====================================

Visualizes light field equivalence and categorical transmission results.

Creates publication-quality multi-panel figure showing:
- Panel A: FTL ratio vs distance scaling
- Panel B: Multi-band validation success rates
- Panel C: Transmission time analysis
- Panel D: Light field equivalence verification
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
    """Load zero-delay positioning results"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_file = os.path.join(project_root, 'results',
                                'zero_delay_positioning_20251115_044427.json')

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
    ftl_ratios = [exp['ftl_ratio'] for exp in experiments]
    transmission_times = [exp['transmission_time_ns'] for exp in experiments]
    capture_times = [exp['max_capture_time_ns'] for exp in experiments]

    # Panel A: FTL Ratio vs Distance Scaling
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(distances, ftl_ratios, 'o-', linewidth=3, markersize=12,
            color='#2E86AB', alpha=0.7, markeredgecolor='black',
            markeredgewidth=2, label='Zero-Delay FTL Ratio')

    # FTL threshold
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5,
               label='FTL Threshold (ratio = 1)', alpha=0.7)

    # Fill region above FTL threshold
    ax1.fill_between(distances, 1.0, ftl_ratios,
                     where=np.array(ftl_ratios) >= 1.0,
                     alpha=0.2, color='green', label='FTL Region')

    ax1.set_xlabel('Distance [meters]', fontweight='bold')
    ax1.set_ylabel('FTL Ratio [v_eff / c]', fontweight='bold')
    ax1.set_title('(A) FTL Ratio Scaling with Distance',
                  fontweight='bold', loc='left')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')

    # Add annotations for FTL achievements
    for i, (d, ftl) in enumerate(zip(distances, ftl_ratios)):
        if ftl >= 1.0:
            ax1.annotate(f'{molecules[i]}\n{ftl:.1f}× c',
                        xy=(d, ftl), xytext=(10, 10),
                        textcoords='offset points', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->',
                                       connectionstyle='arc3,rad=0'))

    # Panel B: Multi-Band Validation Success
    ax2 = fig.add_subplot(gs[0, 1])

    # Extract per-band validation data
    bands_matched = [exp['per_band_validation']['bands_matched'] for exp in experiments]
    bands_ftl = [exp['per_band_validation']['bands_ftl'] for exp in experiments]
    num_bands = experiments[0]['per_band_validation']['num_bands']

    x_positions = np.arange(len(experiments))
    width = 0.35

    bars1 = ax2.bar(x_positions - width/2, bands_matched, width,
                   label='Bands Matched', color='#70AD47', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    bars2 = ax2.bar(x_positions + width/2, bands_ftl, width,
                   label='Bands FTL', color='#FFC000', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    # Reference line for total bands
    ax2.axhline(y=num_bands, color='gray', linestyle=':', linewidth=2,
               label=f'Total Bands ({num_bands})', alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax2.set_xlabel('Experiment', fontweight='bold')
    ax2.set_ylabel('Number of Bands', fontweight='bold')
    ax2.set_title('(B) Multi-Band Validation Success Rates',
                  fontweight='bold', loc='left')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f"{mol}\n{d} m" for mol, d in zip(molecules, distances)],
                        fontsize=7)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0, num_bands + 0.5)

    # Panel C: Transmission Time Analysis
    ax3 = fig.add_subplot(gs[1, 0])

    # Calculate light travel times
    c = 299792458  # m/s
    light_travel_times = [d / c * 1e9 for d in distances]  # Convert to ns

    x_positions = np.arange(len(experiments))
    width = 0.25

    bars1 = ax3.bar(x_positions - width, transmission_times, width,
                   label='Transmission Time', color='#5B9BD5', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    bars2 = ax3.bar(x_positions, light_travel_times, width,
                   label='Light Travel Time', color='#ED7D31', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    bars3 = ax3.bar(x_positions + width, capture_times, width,
                   label='Capture Time', color='#70AD47', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    ax3.set_xlabel('Experiment', fontweight='bold')
    ax3.set_ylabel('Time [nanoseconds]', fontweight='bold')
    ax3.set_title('(C) Time Components: Transmission vs Light Travel',
                  fontweight='bold', loc='left')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([f"{mol}\n{d} m" for mol, d in zip(molecules, distances)],
                        fontsize=7)
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel D: Light Field Equivalence Verification
    ax4 = fig.add_subplot(gs[1, 1])

    # Create metric comparison
    metrics = ['Light Field\nEquivalence', 'All Bands\nMatched',
               'FTL\nAchieved', 'Overall\nSuccess']

    success_data = []
    for exp in experiments:
        equiv = 1 if exp['per_band_validation']['equivalence'] else 0
        matched = exp['per_band_validation']['bands_matched'] / num_bands
        ftl_achieved = exp['ftl_ratio'] >= 1.0
        success = 1 if exp['success'] else 0
        success_data.append([equiv, matched, 1 if ftl_achieved else 0, success])

    # Create heatmap
    success_array = np.array(success_data).T
    im = ax4.imshow(success_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax4.set_xticks(np.arange(len(experiments)))
    ax4.set_yticks(np.arange(len(metrics)))
    ax4.set_xticklabels([f"{mol}\n{d} m" for mol, d in zip(molecules, distances)],
                        fontsize=7)
    ax4.set_yticklabels(metrics, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Success Metric', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(experiments)):
            text = ax4.text(j, i, f'{success_array[i, j]:.2f}',
                          ha="center", va="center", color="black",
                          fontsize=8, fontweight='bold')

    ax4.set_title('(D) Light Field Equivalence Validation Matrix',
                  fontweight='bold', loc='left')

    # Add summary statistics box
    summary = results['summary']
    textstr = (f'Summary:\n'
               f'• Total experiments: {summary["total_experiments"]}\n'
               f'• Successful: {summary["successful_experiments"]}\n'
               f'• Success rate: {summary["success_rate"]:.1%}\n'
               f'• Total bands: {summary["total_wavelength_bands"]}\n'
               f'• FTL bands: {summary["ftl_validated_bands"]}\n'
               f'• Band FTL rate: {summary["per_band_ftl_rate"]:.1%}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5)
    ax4.text(1.45, 0.5, textstr, transform=ax4.transAxes, fontsize=8,
            verticalalignment='center', bbox=props)

    # Overall title
    fig.suptitle('Zero-Delay Positioning: Light Field Equivalence via Categorical Transmission',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig

def save_figure(fig):
    """Save figure to results directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'zero_delay_positioning_{timestamp}.png'
    filepath = os.path.join(results_dir, filename)

    fig.savefig(filepath, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {filepath}")

    plt.close(fig)
    return filepath

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" Zero-Delay Positioning Visualization")
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
    print(f"Successful experiments: {summary['successful_experiments']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total wavelength bands: {summary['total_wavelength_bands']}")
    print(f"FTL validated bands: {summary['ftl_validated_bands']}")
    print(f"Per-band FTL rate: {summary['per_band_ftl_rate']:.1%}")
    print(f"\nFigure: {filepath}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
