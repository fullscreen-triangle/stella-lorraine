#!/usr/bin/env python3
"""
Phase-Lock Network Categorical Completion Visualization
=======================================================

Visualizes categorical FTL experiments comparing two approaches:
- V1: Exact state prediction
- V2: Trajectory prediction

Creates publication-quality multi-panel figure showing:
- Panel A: FTL ratio comparison between V1 and V2
- Panel B: Prediction accuracy improvement (V1 vs V2)
- Panel C: Speedup progression with distance
- Panel D: Combined success metrics
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
    """Load both categorical FTL results"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    results_v1_file = os.path.join(project_root, 'results',
                                   'categorical_ftl_results_20251114_200608.json')
    results_v2_file = os.path.join(project_root, 'results',
                                   'categorical_ftl_v2_20251115_030010.json')

    with open(results_v1_file, 'r') as f:
        results_v1 = json.load(f)

    with open(results_v2_file, 'r') as f:
        results_v2 = json.load(f)

    return results_v1, results_v2

def create_publication_figure(results_v1, results_v2):
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

    # Extract V1 data
    v1_experiments = results_v1['results']
    v1_distances = [exp['equivalent_distance_m'] for exp in v1_experiments]
    v1_ftl_ratios = [exp['ftl_ratio'] for exp in v1_experiments]
    v1_confidence = [exp['confidence'] for exp in v1_experiments]
    v1_names = [exp['experiment_name'] for exp in v1_experiments]

    # Extract V2 data
    v2_experiments = results_v2['results']
    v2_distances = [exp['molecule_1'] + '→' + exp['molecule_2'] for exp in v2_experiments]
    v2_ftl_ratios = [exp['ftl_ratio'] for exp in v2_experiments]
    v2_direction_acc = [exp['direction_accuracy'] for exp in v2_experiments]
    v2_magnitude_acc = [exp['magnitude_accuracy'] for exp in v2_experiments]
    v2_dist_numeric = [1, 10, 100, 1000, 10000]  # From experiment names

    # Panel A: FTL Ratio Comparison (V1 vs V2)
    ax1 = fig.add_subplot(gs[0, 0])

    # Handle different lengths - plot separately on same axis
    width = 0.35

    # Plot V1 bars (4 experiments: 1m, 10m, 100m, 1km)
    x_v1 = np.arange(len(v1_ftl_ratios))
    bars1 = ax1.bar(x_v1 - width/2, v1_ftl_ratios, width,
                   label='V1: Exact State', color='#5B9BD5', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    # Plot V2 bars (5 experiments: 1m, 10m, 100m, 1km, 10km)
    # Only plot first 4 for comparison, show 5th separately
    x_v2_compare = np.arange(4)
    bars2 = ax1.bar(x_v2_compare + width/2, v2_ftl_ratios[:4], width,
                   label='V2: Trajectory (1m-1km)', color='#ED7D31', alpha=0.7,
                   edgecolor='black', linewidth=1.2)

    # Plot V2's 5th experiment (10km) separately
    if len(v2_ftl_ratios) > 4:
        bars3 = ax1.bar([4 + width/2], [v2_ftl_ratios[4]], width,
                       label='V2: Trajectory (10km)', color='#FFA500', alpha=0.7,
                       edgecolor='black', linewidth=1.2)

    # FTL threshold
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5,
               label='FTL Threshold', alpha=0.7, zorder=1)

    ax1.set_xlabel('Distance [meters]', fontweight='bold')
    ax1.set_ylabel('FTL Ratio [v_eff / c]', fontweight='bold')
    ax1.set_title('(A) FTL Ratio: Exact State vs Trajectory Prediction',
                  fontweight='bold', loc='left')

    # Set x-ticks for all distances
    all_distances = v1_distances + ([10000] if len(v2_ftl_ratios) > 4 else [])
    ax1.set_xticks(range(len(all_distances)))
    ax1.set_xticklabels([f'{d} m' if d < 1000 else f'{d//1000} km' for d in all_distances], fontsize=8)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Highlight V2 FTL achievement (if any)
    if any(exp['ftl_achieved'] for exp in v2_experiments):
        ftl_indices = [i for i, exp in enumerate(v2_experiments) if exp['ftl_achieved']]
        for ftl_idx in ftl_indices:
            ax1.plot(ftl_idx + width/2, v2_ftl_ratios[ftl_idx], '*',
                    markersize=25, color='gold', markeredgecolor='black',
                    markeredgewidth=2, zorder=5)
        # Only add label once
        ax1.plot([], [], '*', markersize=20, color='gold', markeredgecolor='black',
                markeredgewidth=2, label='FTL Achieved!')
        ax1.legend(loc='upper left', framealpha=0.9, fontsize=8)

    # Panel B: Prediction Accuracy (V1 vs V2)
    ax2 = fig.add_subplot(gs[0, 1])

    # V1 uses confidence, V2 uses direction accuracy
    x_v1 = np.arange(len(v1_confidence))
    x_v2 = np.arange(len(v2_direction_acc))

    ax2.plot(v1_distances, v1_confidence, 'o-', linewidth=2.5, markersize=10,
            color='#5B9BD5', alpha=0.7, markeredgecolor='black',
            markeredgewidth=1.5, label='V1: Confidence')

    ax2.plot(v2_dist_numeric, v2_direction_acc, 's-', linewidth=2.5, markersize=10,
            color='#ED7D31', alpha=0.7, markeredgecolor='black',
            markeredgewidth=1.5, label='V2: Direction Accuracy')

    ax2.plot(v2_dist_numeric, v2_magnitude_acc, '^-', linewidth=2.5, markersize=10,
            color='#70AD47', alpha=0.7, markeredgecolor='black',
            markeredgewidth=1.5, label='V2: Magnitude Accuracy')

    ax2.set_xlabel('Distance [meters]', fontweight='bold')
    ax2.set_ylabel('Accuracy / Confidence', fontweight='bold')
    ax2.set_title('(B) Prediction Accuracy Comparison',
                  fontweight='bold', loc='left')
    ax2.set_xscale('log')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.05)

    # Panel C: Speedup Progression with Distance
    ax3 = fig.add_subplot(gs[1, 0])

    # Calculate effective velocity
    c = 299792458  # m/s
    v1_v_eff = [exp['v_effective'] for exp in v1_experiments]
    v2_v_eff = [exp['ftl_ratio'] * c for exp in v2_experiments]

    ax3.plot(v1_distances, v1_v_eff, 'o-', linewidth=2.5, markersize=10,
            color='#5B9BD5', alpha=0.7, markeredgecolor='black',
            markeredgewidth=1.5, label='V1: Exact State')

    ax3.plot(v2_dist_numeric, v2_v_eff, 's-', linewidth=2.5, markersize=10,
            color='#ED7D31', alpha=0.7, markeredgecolor='black',
            markeredgewidth=1.5, label='V2: Trajectory')

    # Speed of light reference
    ax3.axhline(y=c, color='red', linestyle='--', linewidth=2.5,
               label='Speed of Light', alpha=0.7)

    ax3.set_xlabel('Distance [meters]', fontweight='bold')
    ax3.set_ylabel('Effective Velocity [m/s]', fontweight='bold')
    ax3.set_title('(C) Effective Velocity Scaling with Distance',
                  fontweight='bold', loc='left')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', which='both')

    # Panel D: Combined Success Metrics
    ax4 = fig.add_subplot(gs[1, 1])

    # Create grouped metrics
    metrics = ['FTL\nAchieved', 'Avg FTL\nRatio', 'Avg\nAccuracy', 'Success\nRate']

    v1_ftl_count = sum(1 for exp in v1_experiments if exp['ftl_achieved'])
    v1_avg_ftl = np.mean([exp['ftl_ratio'] for exp in v1_experiments])
    v1_avg_acc = np.mean(v1_confidence)
    v1_success_rate = v1_ftl_count / len(v1_experiments)

    v2_ftl_count = sum(1 for exp in v2_experiments if exp['ftl_achieved'])
    v2_avg_ftl = np.mean([exp['ftl_ratio'] for exp in v2_experiments])
    v2_avg_acc = np.mean(v2_direction_acc)
    v2_success_rate = results_v2['summary']['success_rate']

    v1_values = [v1_ftl_count, v1_avg_ftl, v1_avg_acc, v1_success_rate]
    v2_values = [v2_ftl_count, v2_avg_ftl, v2_avg_acc, v2_success_rate]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax4.bar(x - width/2, v1_values, width, label='V1: Exact State',
                   color='#5B9BD5', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax4.bar(x + width/2, v2_values, width, label='V2: Trajectory',
                   color='#ED7D31', alpha=0.7, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax4.set_ylabel('Value', fontweight='bold')
    ax4.set_title('(D) Combined Performance Metrics',
                  fontweight='bold', loc='left')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add improvement statistics
    improvement_text = (
        f'Improvement V1 → V2:\n'
        f'• FTL success: {v1_ftl_count} → {v2_ftl_count}\n'
        f'• Avg FTL ratio: {v1_avg_ftl:.3f} → {v2_avg_ftl:.3f}\n'
        f'• Accuracy: {v1_avg_acc:.3f} → {v2_avg_acc:.3f}\n'
        f'• Success rate: {v1_success_rate:.1%} → {v2_success_rate:.1%}'
    )
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5)
    ax4.text(0.95, 0.95, improvement_text, transform=ax4.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    # Overall title
    fig.suptitle('Categorical State Prediction: Exact State vs Trajectory Approach',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig

def save_figure(fig):
    """Save figure to results directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'phase_lock_network_completion_{timestamp}.png'
    filepath = os.path.join(results_dir, filename)

    fig.savefig(filepath, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {filepath}")

    plt.close(fig)
    return filepath

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" Phase-Lock Network Categorical Completion Visualization")
    print("="*70)

    # Load results
    print("\nLoading results...")
    results_v1, results_v2 = load_results()

    # Create figure
    print("Creating publication figure...")
    fig = create_publication_figure(results_v1, results_v2)

    # Save
    print("Saving figure...")
    filepath = save_figure(fig)

    # Summary stats
    print("\n" + "="*70)
    print(" Visualization Complete")
    print("="*70)
    print("\nV1 (Exact State):")
    print(f"  Total experiments: {results_v1['summary']['total_experiments']}")
    print(f"  FTL achieved: {results_v1['summary']['ftl_achieved_count']}")
    print("\nV2 (Trajectory):")
    print(f"  Total experiments: {results_v2['summary']['total_experiments']}")
    print(f"  FTL achieved: {results_v2['summary']['ftl_achieved_count']}")
    print(f"  Success rate: {results_v2['summary']['success_rate']:.1%}")
    print(f"\nFigure: {filepath}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
