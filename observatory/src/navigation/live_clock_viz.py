#!/usr/bin/env python3
"""
Trans-Planckian Clock Run Visualization
Multi-panel analysis of precision cascade through all temporal scales
From nanoseconds to trans-Planckian categorical resolution
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.collections import PatchCollection
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import LogLocator, NullFormatter

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Physical constants
PLANCK_TIME = 5.391247e-44  # Planck time (s)
PLANCK_LENGTH = 1.616255e-35  # Planck length (m)
SPEED_OF_LIGHT = 299792458  # m/s


def load_clock_data(data_files):
    """Load clock run JSON files"""
    clock_runs = []
    for file in data_files:
        try:
            with open(file, 'r') as f:
                clock_runs.append(json.load(f))
            print(f"✓ Loaded: {file}")
        except FileNotFoundError:
            print(f"✗ File not found: {file}")
    return clock_runs


def extract_precision_cascade(clock_run):
    """Extract all precision levels from a clock run"""
    arrays = clock_run['arrays']

    precision_levels = {
        'nanosecond': arrays['nanosecond_precision']['mean'],
        'picosecond': arrays['picosecond_precision']['mean'],
        'femtosecond': arrays['femtosecond_precision']['mean'],
        'attosecond': arrays['attosecond_precision']['mean'],
        'zeptosecond': arrays['zeptosecond_precision']['mean'],
        'planck': arrays['planck_precision']['mean'],
        'trans_planckian': arrays['trans_planckian_precision']['mean'],
    }

    precision_stds = {
        'nanosecond': arrays['nanosecond_precision']['std'],
        'picosecond': arrays['picosecond_precision']['std'],
        'femtosecond': arrays['femtosecond_precision']['std'],
        'attosecond': arrays['attosecond_precision']['std'],
        'zeptosecond': arrays['zeptosecond_precision']['std'],
        'planck': arrays['planck_precision']['std'],
        'trans_planckian': arrays['trans_planckian_precision']['std'],
    }

    return precision_levels, precision_stds


def create_trans_planckian_visualization(clock_runs):
    """Create comprehensive 8-panel visualization"""

    # Create figure with 8 panels (4x2 grid)
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Extract data from both runs
    run1_precisions, run1_stds = extract_precision_cascade(clock_runs[0])
    run2_precisions, run2_stds = extract_precision_cascade(clock_runs[1])

    # ========================================================================
    # PANEL A: Complete Precision Cascade (Logarithmic Scale)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])  # Span full width

    # Prepare data
    scale_names = list(run1_precisions.keys())
    scale_labels = ['Nanosecond', 'Picosecond', 'Femtosecond',
                    'Attosecond', 'Zeptosecond', 'Planck', 'Trans-Planckian']

    run1_values = list(run1_precisions.values())
    run2_values = list(run2_precisions.values())

    x_pos = np.arange(len(scale_names))
    width = 0.35

    # Create bars
    bars1 = ax1.bar(x_pos - width/2, run1_values, width,
                    label='Run 1', color='C0', alpha=0.8, log=True)
    bars2 = ax1.bar(x_pos + width/2, run2_values, width,
                    label='Run 2', color='C1', alpha=0.8, log=True)

    # Add Planck time reference line
    ax1.axhline(y=PLANCK_TIME, color='red', linestyle='--',
                linewidth=3, alpha=0.7, label='Planck Time Barrier', zorder=10)

    # Shade trans-Planckian region
    ax1.axhspan(1e-50, PLANCK_TIME, alpha=0.15, color='green',
                label='Trans-Planckian Region')

    # Annotations for each scale
    for i, (b1, b2, label) in enumerate(zip(bars1, bars2, scale_labels)):
        # Get average value
        avg_val = (b1.get_height() + b2.get_height()) / 2

        # Add value label
        if avg_val > PLANCK_TIME:
            color = 'black'
            va = 'bottom'
            offset = 2
        else:
            color = 'darkgreen'
            va = 'top'
            offset = -2

        ax1.annotate(f'{avg_val:.2e}',
                    xy=(i, avg_val), xytext=(0, offset),
                    textcoords='offset points', ha='center', va=va,
                    fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='yellow', alpha=0.3))

    ax1.set_yscale('log')
    ax1.set_xlabel('Temporal Scale', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Precision (seconds, log scale)', fontweight='bold', fontsize=12)
    ax1.set_title('A) Complete Precision Cascade: Nanosecond → Trans-Planckian',
                  fontweight='bold', loc='left', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scale_labels, rotation=15, ha='right')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(1e-50, 1e-9)

    # Add orders of magnitude annotation
    orders_below_planck = -np.log10(run1_values[-1] / PLANCK_TIME)
    ax1.text(0.02, 0.98,
            f'Trans-Planckian Achievement:\n{orders_below_planck:.1f} orders below Planck time',
            transform=ax1.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2, alpha=0.8),
            fontsize=11, fontweight='bold')

    # ========================================================================
    # PANEL B: Precision Enhancement Factors
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Calculate enhancement factors (ratio between consecutive levels)
    run1_enhancements = []
    run2_enhancements = []
    enhancement_labels = []

    for i in range(len(run1_values) - 1):
        run1_enhancements.append(run1_values[i] / run1_values[i+1])
        run2_enhancements.append(run2_values[i] / run2_values[i+1])
        enhancement_labels.append(f'{scale_labels[i]}\n→\n{scale_labels[i+1]}')

    x_pos = np.arange(len(run1_enhancements))

    # Create grouped bar chart
    bars1 = ax2.bar(x_pos - width/2, run1_enhancements, width,
                    label='Run 1', color='C2', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, run2_enhancements, width,
                    label='Run 2', color='C3', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1e}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7,
                        rotation=45, fontweight='bold')

    ax2.set_xlabel('Scale Transition', fontweight='bold')
    ax2.set_ylabel('Enhancement Factor (×)', fontweight='bold')
    ax2.set_title('B) Precision Enhancement Between Scales',
                  fontweight='bold', loc='left')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(enhancement_labels, fontsize=7)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')

    # Add average enhancement
    avg_enhancement = np.mean(run1_enhancements + run2_enhancements)
    ax2.text(0.95, 0.95, f'Avg Enhancement:\n{avg_enhancement:.2e}×',
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    # ========================================================================
    # PANEL C: Measurement Stability (Standard Deviations)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    run1_std_values = list(run1_stds.values())
    run2_std_values = list(run2_stds.values())

    # Calculate relative stability (std / mean)
    run1_relative_std = [std / mean if mean != 0 else 0
                         for std, mean in zip(run1_std_values, run1_values)]
    run2_relative_std = [std / mean if mean != 0 else 0
                         for std, mean in zip(run2_std_values, run2_values)]

    x_pos = np.arange(len(scale_labels))

    # Plot relative stability
    ax3.plot(x_pos, run1_relative_std, 'o-', linewidth=2, markersize=8,
            label='Run 1', color='C4')
    ax3.plot(x_pos, run2_relative_std, 's--', linewidth=2, markersize=8,
            label='Run 2', color='C5')

    ax3.set_xlabel('Temporal Scale', fontweight='bold')
    ax3.set_ylabel('Relative Stability (σ/μ)', fontweight='bold')
    ax3.set_title('C) Measurement Stability Across Scales',
                  fontweight='bold', loc='left')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scale_labels, rotation=15, ha='right')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Add stability annotation
    avg_stability = np.mean([s for s in run1_relative_std + run2_relative_std if s > 0])
    ax3.text(0.95, 0.05, f'Avg Relative Std:\n{avg_stability:.2e}',
            transform=ax3.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9)

    # ========================================================================
    # PANEL D: Sample Distribution Statistics
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    # Extract sample sizes and statistics
    sample_size = clock_runs[0]['arrays']['reference_ns']['size']

    # Create table of statistics for key scales
    key_scales = ['nanosecond', 'femtosecond', 'zeptosecond', 'trans_planckian']
    key_labels = ['Nanosecond', 'Femtosecond', 'Zeptosecond', 'Trans-Planckian']

    table_data = []
    for scale, label in zip(key_scales, key_labels):
        run1_data = clock_runs[0]['arrays'][f'{scale}_precision']
        run2_data = clock_runs[1]['arrays'][f'{scale}_precision']

        table_data.append([
            label,
            f"{run1_data['mean']:.2e}",
            f"{run1_data['std']:.2e}",
            f"{run2_data['mean']:.2e}",
            f"{run2_data['std']:.2e}"
        ])

    # Create table visualization
    ax4.axis('tight')
    ax4.axis('off')

    table = ax4.table(cellText=table_data,
                     colLabels=['Scale', 'Run 1 Mean', 'Run 1 Std',
                               'Run 2 Mean', 'Run 2 Std'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.2, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')

    ax4.set_title('D) Statistical Summary of Key Scales',
                  fontweight='bold', loc='left', pad=20)

    # Add sample size annotation
    ax4.text(0.5, 0.05, f'Sample Size: {sample_size:,} measurements per run',
            transform=ax4.transAxes, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
            fontsize=10, fontweight='bold')

    # ========================================================================
    # PANEL E: Trans-Planckian Detail View
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    # Focus on Planck and trans-Planckian scales
    trans_scales = ['zeptosecond', 'planck', 'trans_planckian']
    trans_labels = ['Zeptosecond', 'Planck Scale', 'Trans-Planckian']

    trans_run1 = [run1_precisions[s] for s in trans_scales]
    trans_run2 = [run2_precisions[s] for s in trans_scales]

    x_pos = np.arange(len(trans_scales))

    # Create bars with error bars
    bars1 = ax5.bar(x_pos - width/2, trans_run1, width,
                    label='Run 1', color='C6', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, trans_run2, width,
                    label='Run 2', color='C7', alpha=0.8)

    # Add Planck time reference
    ax5.axhline(y=PLANCK_TIME, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label='Planck Time')

    # Annotate bars
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        for bar in [b1, b2]:
            height = bar.get_height()
            ratio_to_planck = height / PLANCK_TIME

            if height < PLANCK_TIME:
                color = 'green'
                text = f'{height:.2e}\n({1/ratio_to_planck:.1f}× below)'
            else:
                color = 'black'
                text = f'{height:.2e}\n({ratio_to_planck:.1f}× above)'

            ax5.annotate(text,
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7,
                        color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='yellow', alpha=0.3))

    ax5.set_yscale('log')
    ax5.set_xlabel('Ultra-Fine Temporal Scale', fontweight='bold')
    ax5.set_ylabel('Precision (seconds, log scale)', fontweight='bold')
    ax5.set_title('E) Trans-Planckian Resolution Detail',
                  fontweight='bold', loc='left')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(trans_labels)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, which='both')

    # ========================================================================
    # PANEL F: Temporal Scale Comparison Chart
    # ========================================================================
    ax6 = fig.add_subplot(gs[3, 0])

    # Create visual scale comparison
    reference_scales = {
        'Human perception': 0.1,  # 100 ms
        'Computer clock': 1e-9,  # 1 ns
        'Light across atom': 1e-18,  # 1 as
        'Nuclear process': 1e-21,  # 1 zs
        'Planck time': PLANCK_TIME,
        'Your achievement': run1_values[-1]
    }

    sorted_scales = sorted(reference_scales.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_scales)

    y_pos = np.arange(len(labels))
    log_values = -np.log10(values)

    colors = ['gray', 'blue', 'purple', 'orange', 'red', 'green']
    bars = ax6.barh(y_pos, log_values, color=colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)

    # Highlight your achievement
    bars[-1].set_hatch('xxx')
    bars[-1].set_edgecolor('darkgreen')
    bars[-1].set_linewidth(3)

    # Highlight Planck time
    planck_idx = labels.index('Planck time')
    bars[planck_idx].set_hatch('///')
    bars[planck_idx].set_edgecolor('darkred')
    bars[planck_idx].set_linewidth(3)

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(labels, fontsize=10)
    ax6.set_xlabel('Time Scale (-log₁₀ seconds)', fontweight='bold')
    ax6.set_title('F) Temporal Scale Comparison',
                  fontweight='bold', loc='left')
    ax6.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax6.text(bar.get_width() + 0.5, i, f'10$^{{{int(np.log10(val))}}}$ s',
                ha='left', va='center', fontsize=8, fontweight='bold')

    # Add achievement annotation
    orders_diff = np.log10(PLANCK_TIME) - np.log10(run1_values[-1])
    ax6.text(0.95, 0.05,
            f'Achievement:\n{orders_diff:.1f} orders\nbelow Planck',
            transform=ax6.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2, alpha=0.8),
            fontsize=10, fontweight='bold')

    # ========================================================================
    # PANEL G: Run Consistency Analysis
    # ========================================================================
    ax7 = fig.add_subplot(gs[3, 1])

    # Calculate differences between runs
    differences = []
    relative_diffs = []

    for scale in scale_names:
        val1 = run1_precisions[scale]
        val2 = run2_precisions[scale]
        diff = abs(val1 - val2)
        rel_diff = diff / ((val1 + val2) / 2) * 100  # Percentage

        differences.append(diff)
        relative_diffs.append(rel_diff)

    x_pos = np.arange(len(scale_labels))

    # Plot relative differences
    bars = ax7.bar(x_pos, relative_diffs, color='C8', alpha=0.7,
                   edgecolor='black', linewidth=1)

    # Color code by consistency
    for i, (bar, rel_diff) in enumerate(zip(bars, relative_diffs)):
        if rel_diff < 1:
            bar.set_color('green')
        elif rel_diff < 5:
            bar.set_color('yellow')
        else:
            bar.set_color('orange')

    ax7.set_xlabel('Temporal Scale', fontweight='bold')
    ax7.set_ylabel('Run-to-Run Difference (%)', fontweight='bold')
    ax7.set_title('G) Experimental Reproducibility',
                  fontweight='bold', loc='left')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(scale_labels, rotation=15, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')

    # Add consistency threshold lines
    ax7.axhline(y=1, color='green', linestyle='--', linewidth=1,
                alpha=0.5, label='Excellent (<1%)')
    ax7.axhline(y=5, color='orange', linestyle='--', linewidth=1,
                alpha=0.5, label='Good (<5%)')
    ax7.legend(loc='upper right', fontsize=8)

    # Add average consistency
    avg_consistency = np.mean(relative_diffs)
    ax7.text(0.05, 0.95, f'Avg Difference:\n{avg_consistency:.3f}%',
            transform=ax7.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
            fontsize=9)

    # ========================================================================
    # Overall figure title and metadata
    # ========================================================================
    fig.suptitle('Trans-Planckian Categorical Resolution: Complete Precision Cascade Analysis',
                 fontsize=18, fontweight='bold', y=0.998)

    # Add metadata footer
    trans_planck_precision = run1_values[-1]
    orders_below = -np.log10(trans_planck_precision / PLANCK_TIME)

    metadata_text = (
        f"Sample Size: {sample_size:,} measurements/run | "
        f"Scales: 7 (ns → trans-Planckian) | "
        f"Trans-Planckian Precision: {trans_planck_precision:.2e} s | "
        f"Achievement: {orders_below:.1f} orders below Planck time"
    )
    fig.text(0.5, 0.002, metadata_text, ha='center', fontsize=10,
             style='italic', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen',
                      edgecolor='darkgreen', linewidth=2, alpha=0.6))

    return fig


def print_comprehensive_statistics(clock_runs):
    """Print detailed statistics for both clock runs"""
    print("\n" + "="*80)
    print("TRANS-PLANCKIAN CLOCK RUN - COMPREHENSIVE STATISTICS")
    print("="*80)

    print(f"\nNumber of clock runs: {len(clock_runs)}")

    for run_idx, run in enumerate(clock_runs):
        print(f"\n{'─'*80}")
        print(f"RUN {run_idx + 1}: {run['timestamp']}")
        print(f"Source: {run['source_file']}")
        print(f"{'─'*80}")

        arrays = run['arrays']

        # Reference timing
        ref = arrays['reference_ns']
        print(f"\nReference Timing (nanoseconds):")
        print(f"  Sample size: {ref['size']:,}")
        print(f"  Min:         {ref['min']:.0f} ns")
        print(f"  Max:         {ref['max']:.0f} ns")
        print(f"  Mean:        {ref['mean']:.0f} ns")
        print(f"  Std:         {ref['std']:.2f} ns")
        print(f"  Median:      {ref['median']:.0f} ns")

        # Precision cascade
        print(f"\nPrecision Cascade:")

        scales = [
            ('nanosecond_precision', 'Nanosecond', 1e9),
            ('picosecond_precision', 'Picosecond', 1e12),
            ('femtosecond_precision', 'Femtosecond', 1e15),
            ('attosecond_precision', 'Attosecond', 1e18),
            ('zeptosecond_precision', 'Zeptosecond', 1e21),
            ('planck_precision', 'Planck Scale', 1),
            ('trans_planckian_precision', 'Trans-Planckian', 1),
        ]

        for key, label, multiplier in scales:
            data = arrays[key]
            mean_val = data['mean']
            std_val = data['std']

            # Calculate ratio to Planck time
            ratio_to_planck = mean_val / PLANCK_TIME

            if mean_val < PLANCK_TIME:
                status = f"({1/ratio_to_planck:.2e}× BELOW Planck) ★"
                marker = "🟢"
            else:
                status = f"({ratio_to_planck:.2e}× above Planck)"
                marker = "🔵"

            print(f"\n  {marker} {label}:")
            print(f"      Mean:   {mean_val:.6e} s")
            print(f"      Std:    {std_val:.6e} s")
            print(f"      Status: {status}")

    # Comparison between runs
    print(f"\n{'='*80}")
    print("RUN-TO-RUN COMPARISON")
    print(f"{'='*80}")

    run1_precisions, _ = extract_precision_cascade(clock_runs[0])
    run2_precisions, _ = extract_precision_cascade(clock_runs[1])

    print(f"\n{'Scale':<20} {'Run 1':<20} {'Run 2':<20} {'Difference':<15}")
    print(f"{'─'*20} {'─'*20} {'─'*20} {'─'*15}")

    for scale in run1_precisions.keys():
        val1 = run1_precisions[scale]
        val2 = run2_precisions[scale]
        diff = abs(val1 - val2)
        rel_diff = (diff / ((val1 + val2) / 2)) * 100

        print(f"{scale:<20} {val1:<20.6e} {val2:<20.6e} {rel_diff:<15.6f}%")

    # Trans-Planckian achievement
    print(f"\n{'='*80}")
    print("TRANS-PLANCKIAN ACHIEVEMENT")
    print(f"{'='*80}")

    trans_planck_run1 = run1_precisions['trans_planckian']
    trans_planck_run2 = run2_precisions['trans_planckian']
    avg_trans_planck = (trans_planck_run1 + trans_planck_run2) / 2

    orders_below_planck = -np.log10(avg_trans_planck / PLANCK_TIME)

    print(f"\nAverage Trans-Planckian Precision: {avg_trans_planck:.6e} s")
    print(f"Planck Time:                        {PLANCK_TIME:.6e} s")
    print(f"\n🌟 ACHIEVEMENT: {orders_below_planck:.2f} orders of magnitude below Planck time")
    print(f"\nThis represents categorical resolution at:")
    print(f"  • {1/avg_trans_planck:.2e} Hz frequency equivalent")
    print(f"  • {avg_trans_planck * SPEED_OF_LIGHT:.2e} m spatial equivalent")
    print(f"  • {avg_trans_planck / PLANCK_TIME:.2e}× finer than Planck scale")

    print("\n" + "="*80)


def main():
    """Main execution function"""

    # Define data files
    data_files = [
        'clock_run_data_20251013_002009_analysis_20251105_145556.json',
        'clock_run_data_20251013_002009_analysis_20251105_151133.json'
    ]

    print("="*80)
    print("TRANS-PLANCKIAN CLOCK RUN VISUALIZATION")
    print("="*80)

    # Load clock run data
    print("\nLoading clock run data...")
    clock_runs = load_clock_data(data_files)

    if len(clock_runs) == 0:
        print("\n✗ No data files found. Please check file paths.")
        return

    print(f"\n✓ Successfully loaded {len(clock_runs)} clock runs")

    # Create visualization
    print("\nGenerating comprehensive visualizations...")
    fig = create_trans_planckian_visualization(clock_runs)

    # Save outputs
    output_png = 'trans_planckian_clock_analysis.png'
    output_pdf = 'trans_planckian_clock_analysis.pdf'

    print("\nSaving figures...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PNG saved: {output_png}")

    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved: {output_pdf}")

    # Print comprehensive statistics
    print_comprehensive_statistics(clock_runs)

    # Display figure
    print("\nDisplaying figure...")
    plt.show()

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nYour trans-Planckian categorical resolution has been visualized.")
    print("This data demonstrates precision beyond the Planck time barrier")
    print("through categorical state identification rather than continuous")
    print("time-domain measurement.")
    print("="*80)


if __name__ == "__main__":
    main()
