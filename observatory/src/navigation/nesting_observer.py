#!/usr/bin/env python3
"""
Recursive Observer Nesting Visualization
Multi-panel analysis of observer cascade and transcendent measurement paths
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path

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
SPEED_OF_LIGHT = 299792458  # m/s


def load_experimental_data(data_files):
    """Load all experimental JSON files"""
    experiments = []
    for file in data_files:
        try:
            with open(file, 'r') as f:
                experiments.append(json.load(f))
            print(f"✓ Loaded: {file}")
        except FileNotFoundError:
            print(f"✗ File not found: {file}")
    return experiments


def create_recursive_observer_visualization(experiments):
    """Create comprehensive 6-panel visualization"""

    # Create figure with 6 panels
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ========================================================================
    # PANEL A: Precision Cascade Through Recursion Levels
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    for idx, exp in enumerate(experiments):
        levels = exp['recursion_results']['levels']
        precisions = exp['recursion_results']['precision_cascade_s']

        ax1.semilogy(levels, precisions, 'o-', linewidth=2, markersize=8,
                    label=f"Run {idx+1}", alpha=0.8)

        # Annotate precision values
        for level, prec in zip(levels, precisions):
            ax1.annotate(f'{prec:.1e}',
                        xy=(level, prec), xytext=(5, 5),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='yellow', alpha=0.3))

    # Add Planck time reference
    ax1.axhline(y=PLANCK_TIME, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label='Planck Time')
    ax1.text(max(levels)*0.95, PLANCK_TIME*1.5, 'Planck Barrier',
            ha='right', va='bottom', color='red', fontweight='bold', fontsize=9)

    ax1.set_xlabel('Recursion Level', fontweight='bold')
    ax1.set_ylabel('Precision (seconds, log scale)', fontweight='bold')
    ax1.set_title('A) Precision Cascade Through Observer Recursion',
                  fontweight='bold', loc='left')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Calculate enhancement per level
    if len(experiments) > 0:
        precisions = experiments[0]['recursion_results']['precision_cascade_s']
        if len(precisions) > 1:
            enhancement = precisions[0] / precisions[1]
            ax1.text(0.05, 0.95, f'Enhancement per level:\n{enhancement:.2e}×',
                    transform=ax1.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                    fontsize=9)

    # ========================================================================
    # PANEL B: Active Observers vs Observation Paths
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    for idx, exp in enumerate(experiments):
        levels = exp['recursion_results']['levels']
        observers = exp['recursion_results']['active_observers']
        paths = exp['recursion_results']['observation_paths']

        # Dual axis plot
        color1 = f'C{idx}'
        ax2.plot(levels, observers, 'o-', color=color1, linewidth=2,
                markersize=8, label=f'Observers (Run {idx+1})')

        # Add path count on secondary axis
        ax2_twin = ax2.twinx() if idx == 0 else ax2_twin
        color2 = f'C{idx+2}'
        ax2_twin.plot(levels, paths, 's--', color=color2, linewidth=2,
                     markersize=8, label=f'Paths (Run {idx+1})', alpha=0.7)

    ax2.set_xlabel('Recursion Level', fontweight='bold')
    ax2.set_ylabel('Active Observers', fontweight='bold', color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    ax2_twin.set_ylabel('Observation Paths', fontweight='bold', color='C2')
    ax2_twin.tick_params(axis='y', labelcolor='C2')
    ax2.set_title('B) Observer Cascade & Path Multiplication',
                  fontweight='bold', loc='left')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add combinatorial growth annotation
    if len(experiments) > 0:
        observers = experiments[0]['recursion_results']['active_observers']
        if len(observers) > 1:
            growth_rate = observers[-1] / observers[0]
            ax2.text(0.5, 0.5, f'Observer Growth:\n{growth_rate:.0f}× per level',
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10, fontweight='bold')

    # ========================================================================
    # PANEL C: Transcendent Observation Paths
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Extract transcendent data
    total_paths = [exp['transcendent_results']['observation_paths'] for exp in experiments]
    resolved_freqs = [exp['transcendent_results']['resolved_frequencies'] for exp in experiments]

    x_pos = np.arange(len(experiments))
    width = 0.35

    # Create grouped bar chart
    bars1 = ax3.bar(x_pos - width/2, np.array(total_paths)/1000, width,
                    label='Observation Paths (×10³)', color='C0', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, resolved_freqs, width,
                    label='Resolved Frequencies', color='C1', alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Experimental Run', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('C) Transcendent Observation Paths & Frequency Resolution',
                  fontweight='bold', loc='left')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add efficiency metric
    avg_paths = np.mean(total_paths)
    avg_freqs = np.mean(resolved_freqs)
    efficiency = avg_freqs / (avg_paths / 1000)
    ax3.text(0.95, 0.95,
            f'Avg Paths: {avg_paths:.0f}\nAvg Freqs: {avg_freqs:.0f}\nEfficiency: {efficiency:.2f}',
            transform=ax3.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9)

    # ========================================================================
    # PANEL D: Frequency Resolution Analysis
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Extract frequency resolution data
    freq_resolutions = [exp['transcendent_results']['frequency_resolution_Hz'] for exp in experiments]
    base_frequencies = [exp['configuration']['base_frequency_Hz'] for exp in experiments]

    # Calculate relative resolution
    relative_resolutions = [fr / bf for fr, bf in zip(freq_resolutions, base_frequencies)]

    # Create comparison plot
    x_pos = np.arange(len(experiments))

    # Absolute resolution (THz)
    ax4.bar(x_pos, np.array(freq_resolutions)/1e12, width=0.6,
            color='C3', alpha=0.7, label='Frequency Resolution')

    # Add relative resolution as text
    for i, (abs_res, rel_res) in enumerate(zip(freq_resolutions, relative_resolutions)):
        ax4.text(i, abs_res/1e12,
                f'{abs_res/1e12:.1f} THz\n({rel_res*100:.2f}%)',
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax4.set_xlabel('Experimental Run', fontweight='bold')
    ax4.set_ylabel('Frequency Resolution (THz)', fontweight='bold')
    ax4.set_title('D) Frequency Resolution Capability',
                  fontweight='bold', loc='left')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])
    ax4.grid(True, alpha=0.3, axis='y')

    # Add base frequency reference
    avg_base_freq = np.mean(base_frequencies) / 1e12
    ax4.axhline(y=avg_base_freq, color='red', linestyle='--',
                linewidth=1.5, alpha=0.5, label=f'Base Freq: {avg_base_freq:.0f} THz')
    ax4.legend(loc='upper right')

    # ========================================================================
    # PANEL E: Ultimate Precision & FFT Performance
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 0])

    # Extract ultimate precision and FFT timing
    ultimate_precisions = [exp['transcendent_results']['ultimate_precision_s'] for exp in experiments]
    fft_times = [exp['transcendent_results']['fft_time_us'] for exp in experiments]

    # Create dual-axis plot
    x_pos = np.arange(len(experiments))

    color1 = 'C4'
    ax5.bar(x_pos - width/2, np.array(ultimate_precisions)*1e15, width,
            color=color1, alpha=0.7, label='Ultimate Precision (fs)')
    ax5.set_ylabel('Ultimate Precision (fs)', fontweight='bold', color=color1)
    ax5.tick_params(axis='y', labelcolor=color1)

    # FFT timing on secondary axis
    ax5_twin = ax5.twinx()
    color2 = 'C5'
    ax5_twin.bar(x_pos + width/2, fft_times, width,
                color=color2, alpha=0.7, label='FFT Time (μs)')
    ax5_twin.set_ylabel('FFT Computation Time (μs)', fontweight='bold', color=color2)
    ax5_twin.tick_params(axis='y', labelcolor=color2)

    ax5.set_xlabel('Experimental Run', fontweight='bold')
    ax5.set_title('E) Ultimate Precision & Computational Performance',
                  fontweight='bold', loc='left')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])
    ax5.grid(True, alpha=0.3, axis='y')

    # Add statistics
    avg_precision = np.mean(ultimate_precisions) * 1e15
    avg_fft = np.mean(fft_times)
    ax5.text(0.5, 0.95,
            f'Avg Precision: {avg_precision:.2f} fs\nAvg FFT Time: {avg_fft:.1f} μs',
            transform=ax5.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
            fontsize=9)

    # ========================================================================
    # PANEL F: Planck Analysis & System Configuration
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 1])

    # Create table-like visualization of key parameters
    ax6.axis('off')

    # Collect configuration data
    config = experiments[0]['configuration']
    planck = experiments[0]['planck_analysis']

    # Create summary table
    table_data = [
        ['Parameter', 'Value', 'Unit'],
        ['─'*20, '─'*15, '─'*10],
        ['Molecules', f"{config['n_molecules']}", ''],
        ['Base Frequency', f"{config['base_frequency_Hz']/1e12:.0f}", 'THz'],
        ['Coherence Time', f"{config['coherence_time_fs']:.0f}", 'fs'],
        ['Chamber Size', f"{config['chamber_size_mm']:.1f}", 'mm'],
        ['', '', ''],
        ['Cascade Precision', f"{planck['precision_s']:.1e}", 's'],
        ['Planck Time', f"{planck['planck_time_s']:.1e}", 's'],
        ['Ratio to Planck', f"{planck['ratio']:.2e}", '×'],
        ['Status', planck['status'], ''],
        ['', '', ''],
        ['Avg Obs. Paths', f"{np.mean([e['transcendent_results']['observation_paths'] for e in experiments]):.0f}", ''],
        ['Avg Frequencies', f"{np.mean([e['transcendent_results']['resolved_frequencies'] for e in experiments]):.0f}", ''],
    ]

    # Draw table
    y_start = 0.95
    y_step = 0.065

    for i, row in enumerate(table_data):
        y_pos = y_start - i * y_step

        # Header row styling
        if i == 0:
            weight = 'bold'
            color = 'darkblue'
            size = 11
        elif '─' in row[0]:
            # Separator
            ax6.plot([0.05, 0.95], [y_pos, y_pos], 'k-', linewidth=1, alpha=0.3)
            continue
        else:
            weight = 'normal'
            color = 'black'
            size = 10

        # Draw cells
        ax6.text(0.05, y_pos, row[0], ha='left', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax6.transAxes)
        ax6.text(0.55, y_pos, row[1], ha='right', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax6.transAxes, family='monospace')
        ax6.text(0.60, y_pos, row[2], ha='left', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax6.transAxes)

    # Add title
    ax6.text(0.5, 0.98, 'F) System Configuration & Planck Analysis',
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=ax6.transAxes)

    # Add status indicator
    status_color = 'green' if planck['status'] == 'Above Planck' else 'red'
    status_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.08,
                               boxstyle="round,pad=0.01",
                               facecolor=status_color, alpha=0.2,
                               edgecolor=status_color, linewidth=2,
                               transform=ax6.transAxes)
    ax6.add_patch(status_box)
    ax6.text(0.5, 0.06, f"Status: {planck['status']}",
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=status_color, transform=ax6.transAxes)

    # ========================================================================
    # Overall figure title and metadata
    # ========================================================================
    fig.suptitle('Recursive Observer Nesting: Transcendent Measurement Paths',
                 fontsize=16, fontweight='bold', y=0.995)

    # Add metadata footer
    metadata_text = (
        f"Recursive Levels: {max([max(e['recursion_results']['levels']) for e in experiments])} | "
        f"Molecules: {config['n_molecules']} | "
        f"Runs: {len(experiments)} | "
        f"Max Precision: {min([e['recursion_results']['precision_cascade_s'][-1] for e in experiments]):.1e} s"
    )
    fig.text(0.5, 0.005, metadata_text, ha='center', fontsize=9,
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return fig


def print_summary_statistics(experiments):
    """Print comprehensive summary statistics"""
    print("\n" + "="*70)
    print("RECURSIVE OBSERVER NESTING - SUMMARY STATISTICS")
    print("="*70)

    print(f"\nNumber of experimental runs: {len(experiments)}")

    # Configuration
    config = experiments[0]['configuration']
    print(f"\nConfiguration:")
    print(f"  Molecules: {config['n_molecules']}")
    print(f"  Base Frequency: {config['base_frequency_Hz']/1e12:.0f} THz")
    print(f"  Coherence Time: {config['coherence_time_fs']:.0f} fs")
    print(f"  Chamber Size: {config['chamber_size_mm']:.1f} mm")

    # Recursion results
    print(f"\nRecursion Cascade:")
    for exp_idx, exp in enumerate(experiments):
        print(f"  Run {exp_idx+1}:")
        for level, prec, obs, paths in zip(
            exp['recursion_results']['levels'],
            exp['recursion_results']['precision_cascade_s'],
            exp['recursion_results']['active_observers'],
            exp['recursion_results']['observation_paths']
        ):
            print(f"    Level {level}: Precision={prec:.2e} s, "
                  f"Observers={obs}, Paths={paths}")

    # Transcendent results
    print(f"\nTranscendent Measurement:")
    obs_paths = [e['transcendent_results']['observation_paths'] for e in experiments]
    res_freqs = [e['transcendent_results']['resolved_frequencies'] for e in experiments]
    freq_res = [e['transcendent_results']['frequency_resolution_Hz'] for e in experiments]
    ult_prec = [e['transcendent_results']['ultimate_precision_s'] for e in experiments]
    fft_time = [e['transcendent_results']['fft_time_us'] for e in experiments]

    print(f"  Observation Paths:")
    print(f"    Mean: {np.mean(obs_paths):.0f}")
    print(f"    Std:  {np.std(obs_paths):.2f}")

    print(f"  Resolved Frequencies:")
    print(f"    Mean: {np.mean(res_freqs):.1f}")
    print(f"    Std:  {np.std(res_freqs):.2f}")

    print(f"  Frequency Resolution:")
    print(f"    Mean: {np.mean(freq_res)/1e12:.2f} THz")
    print(f"    Std:  {np.std(freq_res)/1e12:.3f} THz")

    print(f"  Ultimate Precision:")
    print(f"    Mean: {np.mean(ult_prec):.3e} s")
    print(f"    Std:  {np.std(ult_prec):.3e} s")

    print(f"  FFT Computation Time:")
    print(f"    Mean: {np.mean(fft_time):.2f} μs")
    print(f"    Std:  {np.std(fft_time):.2f} μs")

    # Planck analysis
    planck = experiments[0]['planck_analysis']
    print(f"\nPlanck Analysis:")
    print(f"  Cascade Precision: {planck['precision_s']:.2e} s")
    print(f"  Planck Time: {planck['planck_time_s']:.2e} s")
    print(f"  Ratio: {planck['ratio']:.2e}× above Planck")
    print(f"  Status: {planck['status']}")

    print("\n" + "="*70)


def main():
    """Main execution function"""

    # Define data files
    data_files = [
        'recursive_observers_20251105_115928.json',
        'recursive_observers_20251105_120727.json'
    ]

    print("="*70)
    print("RECURSIVE OBSERVER NESTING VISUALIZATION")
    print("="*70)

    # Load experimental data
    print("\nLoading experimental data...")
    experiments = load_experimental_data(data_files)

    if len(experiments) == 0:
        print("\n✗ No data files found. Please check file paths.")
        return

    print(f"\n✓ Successfully loaded {len(experiments)} experimental runs")

    # Create visualization
    print("\nGenerating visualizations...")
    fig = create_recursive_observer_visualization(experiments)

    # Save outputs
    output_png = 'recursive_observers_analysis.png'
    output_pdf = 'recursive_observers_analysis.pdf'

    print("\nSaving figures...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PNG saved: {output_png}")

    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved: {output_pdf}")

    # Print summary statistics
    print_summary_statistics(experiments)

    # Display figure
    print("\nDisplaying figure...")
    plt.show()

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
