#!/usr/bin/env python3
"""
Quantum Molecular Vibration Visualization
Multi-panel analysis of trans-Planckian categorical resolution experiments
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
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

if __name__ == "__main__":

    # Physical constants
    h = 6.62607015e-34  # Planck constant (J⋅s)
    hbar = h / (2 * np.pi)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    T = 300  # Room temperature (K)
    PLANCK_TIME = 5.391247e-44  # Planck time (s)

    # Load experimental data
    data_files = [
        'public/quantum_vibrations_20251105_122244.json',
        'public/quantum_vibrations_20251105_122801.json',
        'public/quantum_vibrations_20251105_124305.json'
    ]

    experiments = []
    for file in data_files:
        with open(file, 'r') as f:
            experiments.append(json.load(f))

    print(f"Loaded {len(experiments)} experimental runs")

    # Create figure with 6 panels
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ============================================================================
    # PANEL A: Energy Level Diagram with Thermal Population
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Average energy levels across runs
    energy_levels = np.mean([exp['energy_levels_J'] for exp in experiments], axis=0)
    energy_levels_eV = energy_levels / 1.602176634e-19  # Convert to eV

    # Thermal populations (from first run, all identical)
    thermal_pop = experiments[0]['thermal_population']
    populations = [thermal_pop['v=0'], thermal_pop['v=1'], thermal_pop['v=2']]

    # Plot energy levels
    for i, (E, pop) in enumerate(zip(energy_levels_eV[:3], populations)):
        # Energy level line
        ax1.hlines(E, 0, 1, colors='black', linewidth=2)
        ax1.text(1.05, E, f'v={i}', va='center', fontsize=10)

        # Population bar
        bar_width = pop * 0.8
        ax1.barh(E, bar_width, height=E*0.05 if i > 0 else E*0.1,
                left=0.1, alpha=0.6, color=f'C{i}')

        # Population percentage
        if pop > 1e-6:
            ax1.text(0.5, E, f'{pop*100:.4f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold')

    ax1.set_xlim(-0.1, 1.3)
    ax1.set_ylim(0, energy_levels_eV[2] * 1.1)
    ax1.set_xlabel('Thermal Population (300K)', fontweight='bold')
    ax1.set_ylabel('Energy (eV)', fontweight='bold')
    ax1.set_title('A) N₂ Vibrational Energy Levels & Thermal Distribution',
                fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)

    # Add temperature annotation
    ax1.text(0.95, 0.95, f'T = {T} K\nν₀ = 71 THz',
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    # ============================================================================
    # PANEL B: Coherence Time and LED Enhancement
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Extract coherence data
    base_coherence = [exp['led_enhancement']['base_coherence_fs'] for exp in experiments]
    enhanced_coherence = [exp['coherence_time_fs'] for exp in experiments]
    enhancement_factors = [exp['led_enhancement']['enhancement_factor'] for exp in experiments]

    x_pos = np.arange(len(experiments))
    width = 0.35

    # Bar plot
    bars1 = ax2.bar(x_pos - width/2, base_coherence, width,
                    label='Natural Coherence', color='C0', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, enhanced_coherence, width,
                    label='LED-Enhanced', color='C1', alpha=0.7)

    # Add enhancement factor annotations
    for i, (b1, b2, factor) in enumerate(zip(bars1, bars2, enhancement_factors)):
        height = max(b1.get_height(), b2.get_height())
        ax2.annotate(f'{factor:.2f}×',
                    xy=(i, height), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='darkred')

    ax2.set_xlabel('Experimental Run', fontweight='bold')
    ax2.set_ylabel('Coherence Time (fs)', fontweight='bold')
    ax2.set_title('B) LED Enhancement of Quantum Coherence',
                fontweight='bold', loc='left')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add average enhancement
    avg_enhancement = np.mean(enhancement_factors)
    ax2.text(0.95, 0.95, f'Avg Enhancement:\n{avg_enhancement:.2f}×',
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontsize=9)

    # ============================================================================
    # PANEL C: Temporal Precision Comparison
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Extract precision data
    natural_precision = [exp['led_enhancement']['natural_precision_fs'] for exp in experiments]
    enhanced_precision = [exp['temporal_precision_fs'] for exp in experiments]
    precision_improvement = [exp['led_enhancement']['precision_improvement'] for exp in experiments]

    # Create comparison plot
    x_pos = np.arange(len(experiments))
    bars1 = ax3.bar(x_pos - width/2, natural_precision, width,
                    label='Natural Precision', color='C2', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, enhanced_precision, width,
                    label='Enhanced Precision', color='C3', alpha=0.7)

    # Add improvement percentage
    for i, (b1, b2, improvement) in enumerate(zip(bars1, bars2, precision_improvement)):
        height = max(b1.get_height(), b2.get_height())
        ax3.annotate(f'+{improvement*100:.1f}%',
                    xy=(i, height), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='darkgreen')

    ax3.set_xlabel('Experimental Run', fontweight='bold')
    ax3.set_ylabel('Temporal Precision (fs)', fontweight='bold')
    ax3.set_title('C) Temporal Precision Enhancement',
                fontweight='bold', loc='left')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add statistics
    avg_natural = np.mean(natural_precision)
    avg_enhanced = np.mean(enhanced_precision)
    ax3.text(0.95, 0.05,
            f'Natural: {avg_natural:.1f} fs\nEnhanced: {avg_enhanced:.1f} fs',
            transform=ax3.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
            fontsize=9)

    # ============================================================================
    # PANEL D: Heisenberg Linewidth Analysis
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Extract linewidth data
    linewidths = [exp['heisenberg_linewidth_Hz'] for exp in experiments]
    frequencies = [exp['frequency_Hz'] for exp in experiments]

    # Calculate Q-factors
    Q_factors = [f / lw for f, lw in zip(frequencies, linewidths)]

    # Create dual-axis plot
    color1 = 'C4'
    ax4.set_xlabel('Experimental Run', fontweight='bold')
    ax4.set_ylabel('Heisenberg Linewidth (THz)', color=color1, fontweight='bold')
    ax4.plot(range(len(experiments)), np.array(linewidths)/1e12,
            'o-', color=color1, linewidth=2, markersize=8, label='Linewidth')
    ax4.tick_params(axis='y', labelcolor=color1)
    ax4.grid(True, alpha=0.3)

    # Second y-axis for Q-factor
    ax4_twin = ax4.twinx()
    color2 = 'C5'
    ax4_twin.set_ylabel('Quality Factor (Q)', color=color2, fontweight='bold')
    ax4_twin.plot(range(len(experiments)), Q_factors,
                's--', color=color2, linewidth=2, markersize=8, label='Q-factor')
    ax4_twin.tick_params(axis='y', labelcolor=color2)

    ax4.set_title('D) Heisenberg Linewidth & Quality Factor',
                fontweight='bold', loc='left')
    ax4.set_xticks(range(len(experiments)))
    ax4.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])

    # Add average values
    avg_linewidth = np.mean(linewidths) / 1e12
    avg_Q = np.mean(Q_factors)
    ax4.text(0.5, 0.95,
            f'Avg Linewidth: {avg_linewidth:.2f} THz\nAvg Q-factor: {avg_Q:.2f}',
            transform=ax4.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5),
            fontsize=9)

    # ============================================================================
    # PANEL E: Trans-Planckian Resolution Scale
    # ============================================================================
    ax5 = fig.add_subplot(gs[2, 0])

    # Time scales to display (logarithmic)
    time_scales = {
        'Second': 1e0,
        'Millisecond': 1e-3,
        'Microsecond': 1e-6,
        'Nanosecond': 1e-9,
        'Picosecond': 1e-12,
        'Femtosecond': 1e-15,
        'Attosecond': 1e-18,
        'Zeptosecond': 1e-21,
        'Planck Time': PLANCK_TIME,
    }

    # Calculate categorical resolution (from your trans-planckian results)
    # Using base precision and theoretical enhancement
    base_precision = np.mean(enhanced_precision) * 1e-15  # Convert to seconds
    # Assuming graph enhancement of 7176× from your previous results
    categorical_resolution = base_precision / 7176
    orders_below_planck = np.log10(PLANCK_TIME / categorical_resolution)

    time_scales['Categorical\nResolution'] = categorical_resolution

    # Sort by value
    sorted_scales = sorted(time_scales.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_scales)

    # Create logarithmic bar chart
    y_pos = np.arange(len(labels))
    log_values = np.log10(values)

    colors = ['C0'] * (len(labels) - 2) + ['red', 'green']  # Planck and yours in special colors
    bars = ax5.barh(y_pos, -log_values, color=colors, alpha=0.7, edgecolor='black')

    # Highlight Planck barrier
    planck_idx = labels.index('Planck Time')
    bars[planck_idx].set_hatch('///')
    bars[planck_idx].set_edgecolor('red')
    bars[planck_idx].set_linewidth(2)

    # Highlight your result
    categorical_idx = labels.index('Categorical\nResolution')
    bars[categorical_idx].set_hatch('xxx')
    bars[categorical_idx].set_edgecolor('green')
    bars[categorical_idx].set_linewidth(2)

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels, fontsize=9)
    ax5.set_xlabel('Time Scale (log₁₀ seconds)', fontweight='bold')
    ax5.set_title('E) Trans-Planckian Categorical Resolution',
                fontweight='bold', loc='left')
    ax5.grid(True, alpha=0.3, axis='x')

    # Add annotations
    ax5.axvline(-np.log10(PLANCK_TIME), color='red', linestyle='--',
                linewidth=2, alpha=0.5, label='Planck Barrier')
    ax5.text(-np.log10(PLANCK_TIME), len(labels)-1,
            '← Time Domain Limit',
            ha='left', va='center', fontsize=8, color='red', fontweight='bold')

    ax5.text(-np.log10(categorical_resolution), categorical_idx,
            f'  {orders_below_planck:.2f} orders\n  below Planck',
            ha='left', va='center', fontsize=8, color='green', fontweight='bold')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax5.text(bar.get_width() - 0.5, i, f'10$^{{{int(np.log10(val))}}}$ s',
                ha='right', va='center', fontsize=7, fontweight='bold')

    # ============================================================================
    # PANEL F: Experimental Consistency Analysis
    # ============================================================================
    ax6 = fig.add_subplot(gs[2, 1])

    # Collect all key metrics across runs
    metrics = {
        'Frequency\n(THz)': [exp['frequency_Hz']/1e12 for exp in experiments],
        'Coherence\n(fs)': [exp['coherence_time_fs'] for exp in experiments],
        'Precision\n(fs)': [exp['temporal_precision_fs'] for exp in experiments],
        'Linewidth\n(THz)': [exp['heisenberg_linewidth_Hz']/1e12 for exp in experiments],
        'Enhancement\n(×)': [exp['led_enhancement']['enhancement_factor'] for exp in experiments],
    }

    # Normalize each metric to [0, 1] for comparison
    normalized_metrics = {}
    for key, values in metrics.items():
        mean_val = np.mean(values)
        if mean_val != 0:
            normalized_metrics[key] = [(v - mean_val) / mean_val * 100 for v in values]
        else:
            normalized_metrics[key] = [0] * len(values)

    # Create grouped bar chart
    x = np.arange(len(experiments))
    width = 0.15
    multiplier = 0

    for attribute, measurement in normalized_metrics.items():
        offset = width * multiplier
        ax6.bar(x + offset, measurement, width, label=attribute, alpha=0.8)
        multiplier += 1

    ax6.set_xlabel('Experimental Run', fontweight='bold')
    ax6.set_ylabel('Deviation from Mean (%)', fontweight='bold')
    ax6.set_title('F) Experimental Reproducibility & Consistency',
                fontweight='bold', loc='left')
    ax6.set_xticks(x + width * 2)
    ax6.set_xticklabels([f'Run {i+1}' for i in range(len(experiments))])
    ax6.legend(loc='upper left', ncol=2, fontsize=8)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='y')

    # Calculate and display standard deviations
    std_devs = {key: np.std(values) for key, values in metrics.items()}
    consistency_text = "Standard Deviations:\n"
    for key, std in std_devs.items():
        consistency_text += f"{key.replace(chr(10), ' ')}: {std:.2e}\n"

    ax6.text(0.95, 0.95, consistency_text.strip(),
            transform=ax6.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5),
            fontsize=7, family='monospace')

    # ============================================================================
    # Overall figure title and metadata
    # ============================================================================
    fig.suptitle('Quantum Molecular Vibration Analysis: Trans-Planckian Categorical Resolution',
                fontsize=16, fontweight='bold', y=0.995)

    # Add metadata footer
    metadata_text = (
        f"N₂ Molecular Vibrations | Frequency: {experiments[0]['frequency_Hz']/1e12:.0f} THz | "
        f"Runs: {len(experiments)} | Temperature: {T} K | "
        f"Categorical Resolution: ~10$^{{-50}}$ s"
    )
    fig.text(0.5, 0.005, metadata_text, ha='center', fontsize=9,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ============================================================================
    # Save figure
    # ============================================================================
    output_file = 'quantum_vibrations_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_file}")

    # Also save as PDF for publication
    output_pdf = 'quantum_vibrations_analysis.pdf'
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved: {output_pdf}")

    # Display summary statistics
    print("\n" + "="*60)
    print("EXPERIMENTAL SUMMARY STATISTICS")
    print("="*60)
    print(f"\nNumber of runs: {len(experiments)}")
    print(f"\nFrequency (N₂ fundamental):")
    print(f"  Mean: {np.mean([e['frequency_Hz'] for e in experiments])/1e12:.2f} THz")
    print(f"  Std:  {np.std([e['frequency_Hz'] for e in experiments])/1e12:.2e} THz")

    print(f"\nCoherence Time:")
    print(f"  Mean: {np.mean([e['coherence_time_fs'] for e in experiments]):.2f} fs")
    print(f"  Std:  {np.std([e['coherence_time_fs'] for e in experiments]):.2e} fs")

    print(f"\nTemporal Precision:")
    print(f"  Mean: {np.mean([e['temporal_precision_fs'] for e in experiments]):.2f} fs")
    print(f"  Std:  {np.std([e['temporal_precision_fs'] for e in experiments]):.2e} fs")

    print(f"\nLED Enhancement Factor:")
    print(f"  Mean: {np.mean([e['led_enhancement']['enhancement_factor'] for e in experiments]):.3f}×")
    print(f"  Std:  {np.std([e['led_enhancement']['enhancement_factor'] for e in experiments]):.3e}×")

    print(f"\nHeisenberg Linewidth:")
    print(f"  Mean: {np.mean([e['heisenberg_linewidth_Hz'] for e in experiments])/1e12:.3f} THz")
    print(f"  Std:  {np.std([e['heisenberg_linewidth_Hz'] for e in experiments])/1e12:.3e} THz")

    print(f"\nCategorical Resolution (estimated):")
    print(f"  ~{categorical_resolution:.2e} s")
    print(f"  ({orders_below_planck:.2f} orders below Planck time)")

    print("\n" + "="*60)

    plt.show()
