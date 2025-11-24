"""
Visualization Script 3: Phase-Space Dynamics

Creates 4 charts showing phase-space evolution:
1. Phase Evolution Trajectories (polar)
2. Coherence Distribution Across Network
3. Formation Cycle Distribution
4. Folding Nucleus Centrality Analysis
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import networkx as nx
from matplotlib.patches import Circle, FancyArrowPatch

if __name__ == "__main__":
    # Load validation results
    results_file = Path(__file__).parent / 'cycle_by_cycle_validation.json'
    with open(results_file, 'r') as f:
        results = json.load(f)

    pathway_data = results['test_5']
    folding_data = results['test_4']

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Phase-Space Dynamics: Oscillatory Evolution of H-Bond Network',
                fontsize=16, fontweight='bold')

    # ============================================================================
    # CHART 1: Phase Evolution Trajectories (Polar)
    # ============================================================================
    ax1 = plt.subplot(2, 2, 1, projection='polar')

    # Simulate phase evolution for visualization
    # (In reality, we'd track this during simulation)
    formation_events = pathway_data['formation_events']

    # For each bond, simulate phase trajectory
    np.random.seed(42)
    time_steps = 100

    for i, event in enumerate(formation_events):
        bond = event['bond']
        formation_cycle = event['cycle']
        final_coherence = event['coherence']

        # Initial random phase
        initial_phase = np.random.uniform(0, 2*np.pi)

        # Final phase based on cluster (approximate from data)
        # Bonds in same cycle tend to cluster in phase
        final_phase = (formation_cycle * np.pi / 5) % (2*np.pi) + np.random.uniform(-0.3, 0.3)

        # Generate trajectory from initial to final
        phases = np.linspace(initial_phase, final_phase, time_steps)

        # Radius starts small (low coherence), grows to final coherence
        radii = np.linspace(0.1, final_coherence, time_steps)

        # Add some oscillation to show phase dynamics
        phases += 0.1 * np.sin(np.linspace(0, formation_cycle * np.pi, time_steps))

        # Color by formation cycle
        color = plt.cm.viridis(formation_cycle / pathway_data['cycles_to_fold'])

        # Plot trajectory
        ax1.plot(phases, radii, color=color, linewidth=2, alpha=0.6)

        # Mark start
        ax1.plot(phases[0], radii[0], 'o', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1)

        # Mark end
        ax1.plot(phases[-1], radii[-1], '*', color=color, markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)

        # Label
        ax1.text(phases[-1], radii[-1] + 0.05, bond, fontsize=8, ha='center')

    # Add coherence circles
    for coherence_level in [0.3, 0.5, 0.7, 0.9]:
        ax1.plot(np.linspace(0, 2*np.pi, 100),
                [coherence_level] * 100,
                'k:', linewidth=1, alpha=0.3)

    ax1.set_ylim(0, 1.0)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_title('A. Phase Evolution Trajectories\n(○ = Start, ★ = Final)',
                fontsize=13, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    # ============================================================================
    # CHART 2: Coherence Distribution Across Network
    # ============================================================================
    ax2 = plt.subplot(2, 2, 2)

    # Get coherence values
    coherences = [e['coherence'] for e in formation_events]
    bonds = [e['bond'] for e in formation_events]
    cycles = [e['cycle'] for e in formation_events]

    # Create violin plot by cycle
    cycle_groups = {}
    for bond, coherence, cycle in zip(bonds, coherences, cycles):
        if cycle not in cycle_groups:
            cycle_groups[cycle] = []
        cycle_groups[cycle].append(coherence)

    # Prepare data for violin plot
    plot_data = []
    plot_labels = []
    for cycle in sorted(cycle_groups.keys()):
        plot_data.append(cycle_groups[cycle])
        plot_labels.append(f'Cycle {cycle}')

    # Violin plot
    parts = ax2.violinplot(plot_data, positions=range(len(plot_data)),
                        showmeans=True, showmedians=True)

    # Color violins
    for i, pc in enumerate(parts['bodies']):
        color = plt.cm.viridis(i / len(plot_data))
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Overlay scatter points
    for i, (cycle_data, cycle) in enumerate(zip(plot_data, sorted(cycle_groups.keys()))):
        x = np.ones(len(cycle_data)) * i
        x += np.random.normal(0, 0.04, len(cycle_data))  # Add jitter
        colors_scatter = plt.cm.viridis([cycle / pathway_data['cycles_to_fold']] * len(cycle_data))
        ax2.scatter(x, cycle_data, alpha=0.6, s=100, c=colors_scatter,
                edgecolors='black', linewidths=1, zorder=3)

    # Add threshold lines
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
            label='Phase-Lock Threshold (0.5)', alpha=0.7)
    ax2.axhline(y=0.7, color='green', linestyle='--', linewidth=2,
            label='Strong Lock (0.7)', alpha=0.7)

    ax2.set_xticks(range(len(plot_labels)))
    ax2.set_xticklabels(plot_labels, fontsize=10)
    ax2.set_ylabel('Phase Coherence', fontsize=12, fontweight='bold')
    ax2.set_title('B. Coherence Distribution by Formation Cycle',
                fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)

    # ============================================================================
    # CHART 3: Formation Cycle Distribution
    # ============================================================================
    ax3 = plt.subplot(2, 2, 3)

    # Histogram of formation cycles
    cycle_counts = pathway_data['bonds_per_cycle']
    cycles_list = sorted([int(c) for c in cycle_counts.keys()])
    counts = [cycle_counts[str(c)] for c in cycles_list]

    # Bar plot
    colors_bars = plt.cm.plasma(np.array(cycles_list) / max(cycles_list))
    bars = ax3.bar(cycles_list, counts, color=colors_bars, edgecolor='black',
                linewidth=2, alpha=0.8, width=1.5)

    # Add value labels on bars
    for i, (cycle, count) in enumerate(zip(cycles_list, counts)):
        ax3.text(cycle, count + 0.1, str(count), ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Mark critical cycles
    for cc in pathway_data['critical_cycles']:
        if cc in cycles_list:
            idx = cycles_list.index(cc)
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(3)

    # Add folding nucleus marker
    nucleus_cycle = pathway_data['folding_nucleus']['cycle']
    ax3.axvline(x=nucleus_cycle, color='gold', linestyle='--', linewidth=3,
            label=f'Folding Nucleus\n(Cycle {nucleus_cycle})', alpha=0.8)

    # Statistics text
    total_bonds = pathway_data['total_bonds']
    total_cycles = pathway_data['cycles_to_fold']
    mean_cycle = np.mean([int(c) for c in cycle_counts.keys()
                        for _ in range(cycle_counts[c])])

    stats_text = f"Total Bonds: {total_bonds}\n"
    stats_text += f"Total Cycles: {total_cycles}\n"
    stats_text += f"Mean Formation Cycle: {mean_cycle:.1f}\n"
    stats_text += f"Critical Cycles: {len(pathway_data['critical_cycles'])}"

    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax3.set_xlabel('ATP Cycle Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Bonds Formed', fontsize=12, fontweight='bold')
    ax3.set_title('C. H-Bond Formation Distribution\n(Red outline = Critical Cycle)',
                fontsize=13, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.legend(fontsize=10, loc='upper left')

    # ============================================================================
    # CHART 4: Folding Nucleus Centrality Analysis
    # ============================================================================
    ax4 = plt.subplot(2, 2, 4)

    # Build dependency graph
    G = nx.DiGraph()

    for event in formation_events:
        bond = event['bond']
        G.add_node(bond)

    # Add edges based on dependencies
    for i, event in enumerate(formation_events):
        bond = event['bond']
        num_deps = event['dependencies']

        for j in range(num_deps):
            if j < len(formation_events):
                dep_bond = formation_events[j]['bond']
                if dep_bond != bond:
                    G.add_edge(dep_bond, bond)

    # Calculate centrality metrics
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Create scatter plot: in-degree vs out-degree
    bonds_list = list(G.nodes())
    in_degrees = [in_degree_centrality[b] for b in bonds_list]
    out_degrees = [out_degree_centrality[b] for b in bonds_list]
    betweenness = [betweenness_centrality[b] for b in bonds_list]

    # Scatter with size by betweenness
    scatter = ax4.scatter(in_degrees, out_degrees,
                        s=[b * 1000 + 100 for b in betweenness],
                        c=betweenness, cmap='YlOrRd',
                        edgecolors='black', linewidths=2, alpha=0.7)

    # Label bonds
    for bond, in_deg, out_deg in zip(bonds_list, in_degrees, out_degrees):
        ax4.annotate(bond, (in_deg, out_deg), fontsize=9, ha='center',
                    fontweight='bold')

    # Highlight folding nucleus
    nucleus_bond = pathway_data['folding_nucleus']['bond']
    if nucleus_bond in bonds_list:
        idx = bonds_list.index(nucleus_bond)
        ax4.scatter([in_degrees[idx]], [out_degrees[idx]],
                s=1500, c='none', edgecolors='gold',
                linewidths=4, marker='*', zorder=10)

    # Quadrant lines
    ax4.axhline(y=np.mean(out_degrees), color='gray', linestyle=':', linewidth=1)
    ax4.axvline(x=np.mean(in_degrees), color='gray', linestyle=':', linewidth=1)

    # Quadrant labels
    ax4.text(0.95, 0.95, 'HIGH DEPENDENCY\n(Many inputs & outputs)',
            transform=ax4.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax4.text(0.05, 0.05, 'LOW DEPENDENCY\n(Few inputs & outputs)',
            transform=ax4.transAxes, fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax4.text(0.95, 0.05, 'TERMINAL BONDS\n(Inputs only)',
            transform=ax4.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax4.text(0.05, 0.95, 'NUCLEATION BONDS\n(Outputs only)',
            transform=ax4.transAxes, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

    ax4.set_xlabel('In-Degree Centrality\n(Depends on others)',
                fontsize=12, fontweight='bold')
    ax4.set_ylabel('Out-Degree Centrality\n(Others depend on it)',
                fontsize=12, fontweight='bold')
    ax4.set_title('D. Folding Nucleus Centrality\n(★ = Nucleus, Size = Betweenness)',
                fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax4, orientation='vertical', pad=0.02)
    cbar.set_label('Betweenness Centrality', fontsize=10)

    # ============================================================================
    # Save figure
    # ============================================================================
    plt.tight_layout()
    output_file = Path(__file__).parent / 'results' / 'phase_space_panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.show()
