"""
Visualization Script 1: Folding Dynamics Overview

Creates 4 charts showing the overall folding process:
1. Stability & Variance Evolution over Cycles
2. Cavity Frequency Modulation Timeline
3. Bond Formation Timeline
4. Phase-Locked Bond Accumulation
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

if __name__ == "__main__":
    # Load validation results
    results_file = Path(__file__).parent / 'cycle_by_cycle_validation.json'
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract test 4 (complete folding) and test 5 (pathway) data
    folding_data = results['test_4']
    pathway_data = results['test_5']

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Protein Folding Dynamics: Phase-Locked GroEL-Mediated Folding',
                fontsize=16, fontweight='bold')

    # ============================================================================
    # CHART 1: Stability & Variance Evolution
    # ============================================================================
    ax1 = plt.subplot(2, 2, 1)

    cycle_history = folding_data['cycle_history']
    cycles = [h['cycle'] for h in cycle_history]
    stabilities = [h['final_stability'] for h in cycle_history]
    variances = [h['final_variance'] for h in cycle_history]
    mean_stabilities = [h['mean_stability'] for h in cycle_history]

    # Plot stability (left y-axis)
    color_stability = '#2E7D32'  # Green
    ax1.plot(cycles, stabilities, 'o-', color=color_stability, linewidth=2.5,
            markersize=8, label='Final Stability')
    ax1.plot(cycles, mean_stabilities, 's--', color=color_stability, alpha=0.5,
            linewidth=1.5, markersize=6, label='Mean Stability')
    ax1.set_xlabel('ATP Cycle Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Network Stability', fontsize=12, fontweight='bold', color=color_stability)
    ax1.tick_params(axis='y', labelcolor=color_stability)
    ax1.grid(True, alpha=0.3)

    # Highlight best cycle
    best_cycle = folding_data['best_cycle']
    best_stability = folding_data['best_stability']
    ax1.axvline(x=best_cycle, color='gold', linestyle='--', linewidth=2, alpha=0.7)
    ax1.scatter([best_cycle], [best_stability], s=300, c='gold', marker='*',
            edgecolors='black', linewidths=2, zorder=5, label='Best Cycle')

    # Plot variance (right y-axis)
    ax1_twin = ax1.twinx()
    color_variance = '#C62828'  # Red
    ax1_twin.plot(cycles, variances, '^-', color=color_variance, linewidth=2.5,
                markersize=8, label='Variance')
    ax1_twin.set_ylabel('Phase Coherence Variance', fontsize=12, fontweight='bold',
                        color=color_variance)
    ax1_twin.tick_params(axis='y', labelcolor=color_variance)

    # Add threshold line
    ax1.axhline(y=0.7, color='green', linestyle=':', linewidth=2, alpha=0.5,
            label='Success Threshold (0.7)')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

    ax1.set_title('A. Network Stability & Variance Evolution', fontsize=13, fontweight='bold')

    # ============================================================================
    # CHART 2: Cavity Frequency Modulation
    # ============================================================================
    ax2 = plt.subplot(2, 2, 2)

    # Extract cavity frequencies
    cavity_freqs = [h['cavity_frequency_range'][0] for h in cycle_history]

    # Convert to THz for readability
    cavity_freqs_thz = np.array(cavity_freqs) / 1e12

    # Color code by ATP cycle phase (different harmonics)
    colors = plt.cm.viridis(np.linspace(0, 1, len(cycles)))

    ax2.scatter(cycles, cavity_freqs_thz, c=colors, s=200, alpha=0.7,
            edgecolors='black', linewidths=1.5)
    ax2.plot(cycles, cavity_freqs_thz, 'k--', alpha=0.3, linewidth=1)

    # Add O2 master clock reference line
    o2_clock_thz = 10.0  # 10^13 Hz = 10 THz
    ax2.axhline(y=o2_clock_thz, color='red', linestyle='--', linewidth=2,
            label='O₂ Master Clock (10 THz)')

    # Add harmonic lines
    for harmonic in [0.5, 1.0, 2.0, 3.0, 5.0]:
        freq = o2_clock_thz * harmonic
        ax2.axhline(y=freq, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax2.text(0.5, freq + 0.5, f'{harmonic}× O₂', fontsize=8, alpha=0.5)

    ax2.set_xlabel('ATP Cycle Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cavity Frequency (THz)', fontsize=12, fontweight='bold')
    ax2.set_title('B. GroEL Cavity Frequency Scanning', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # ============================================================================
    # CHART 3: Bond Formation Timeline
    # ============================================================================
    ax3 = plt.subplot(2, 2, 3)

    formation_events = pathway_data['formation_events']

    # Group bonds by cycle
    cycle_bonds = {}
    for event in formation_events:
        cycle = event['cycle']
        bond = event['bond']
        coherence = event['coherence']

        if cycle not in cycle_bonds:
            cycle_bonds[cycle] = []
        cycle_bonds[cycle].append((bond, coherence))

    # Plot timeline
    for i, event in enumerate(formation_events):
        cycle = event['cycle']
        coherence = event['coherence']

        # Color by coherence
        color = plt.cm.RdYlGn(coherence)

        # Horizontal bar for each bond
        ax3.barh(i, cycle, left=0, height=0.8, color=color,
                edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add bond label
        ax3.text(-1, i, event['bond'], ha='right', va='center', fontsize=9,
                fontweight='bold')

        # Add coherence value
        ax3.text(cycle + 0.5, i, f"{coherence:.2f}", va='center', fontsize=8)

    # Add critical cycle markers
    critical_cycles = pathway_data['critical_cycles']
    for cc in critical_cycles:
        ax3.axvline(x=cc, color='purple', linestyle='--', linewidth=2, alpha=0.5)

    ax3.set_xlabel('Formation Cycle', fontsize=12, fontweight='bold')
    ax3.set_ylabel('H-Bond', fontsize=12, fontweight='bold')
    ax3.set_title('C. H-Bond Formation Timeline (Color = Phase Coherence)',
                fontsize=13, fontweight='bold')
    ax3.set_yticks(range(len(formation_events)))
    ax3.set_yticklabels([e['bond'] for e in formation_events])
    ax3.grid(True, axis='x', alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                            norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, orientation='vertical', pad=0.02)
    cbar.set_label('Phase Coherence', fontsize=10)

    # ============================================================================
    # CHART 4: Phase-Locked Bond Accumulation
    # ============================================================================
    ax4 = plt.subplot(2, 2, 4)

    # Count cumulative bonds phase-locked per cycle
    all_cycles = sorted(set([e['cycle'] for e in formation_events]))
    cumulative_bonds = []
    current_count = 0

    for cycle in range(1, max(all_cycles) + 1):
        bonds_this_cycle = sum(1 for e in formation_events if e['cycle'] == cycle)
        current_count += bonds_this_cycle
        cumulative_bonds.append(current_count)

    plot_cycles = range(1, max(all_cycles) + 1)

    # Step plot
    ax4.step(plot_cycles, cumulative_bonds, where='post', linewidth=3,
            color='#1976D2', label='Cumulative Bonds')
    ax4.fill_between(plot_cycles, 0, cumulative_bonds, step='post',
                    alpha=0.3, color='#1976D2')

    # Mark formation events
    for event in formation_events:
        cycle = event['cycle']
        count = sum(1 for e in formation_events if e['cycle'] <= cycle)
        ax4.plot(cycle, count, 'ro', markersize=10, zorder=5)

    # Total bonds line
    total_bonds = pathway_data['total_bonds']
    ax4.axhline(y=total_bonds, color='green', linestyle='--', linewidth=2,
            label=f'Total Bonds ({total_bonds})')

    # Folding nucleus indicator
    nucleus = pathway_data['folding_nucleus']
    nucleus_cycle = nucleus['cycle']
    ax4.axvline(x=nucleus_cycle, color='gold', linestyle='--', linewidth=2,
            alpha=0.7, label=f'Folding Nucleus (cycle {nucleus_cycle})')

    ax4.set_xlabel('ATP Cycle Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Phase-Locked Bonds', fontsize=12, fontweight='bold')
    ax4.set_title('D. Cumulative Bond Phase-Locking', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, total_bonds + 1)

    # ============================================================================
    # Save figure
    # ============================================================================
    plt.tight_layout()
    output_file = Path(__file__).parent / 'results' / 'folding_dynamics_panel.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.show()
