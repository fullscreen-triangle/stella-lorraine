"""
Visualization Script 2: Network Structure & Dependencies

Creates 4 charts showing the protein network structure:
1. H-Bond Dependency Graph (network diagram)
2. Phase Coherence Clusters (polar plot)
3. GroEL Cavity Volume Modulation
4. Bond-Cycle Coupling Strength Heatmap
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import networkx as nx
from matplotlib.patches import Circle, Wedge

if __name__ == "__main__":

    # Load validation results
    results_file = Path(__file__).parent / 'cycle_by_cycle_validation.json'
    with open(results_file, 'r') as f:
        results = json.load(f)

    pathway_data = results['test_5']
    folding_data = results['test_4']

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Protein Network Structure: Phase-Locked H-Bond Dependencies',
                fontsize=16, fontweight='bold')

    # ============================================================================
    # CHART 1: H-Bond Dependency Graph
    # ============================================================================
    ax1 = plt.subplot(2, 2, 1)

    # Build dependency graph
    G = nx.DiGraph()

    formation_events = pathway_data['formation_events']

    # Add nodes (bonds)
    for event in formation_events:
        bond = event['bond']
        cycle = event['cycle']
        coherence = event['coherence']
        G.add_node(bond, cycle=cycle, coherence=coherence)

    # Add edges (dependencies)
    for i, event in enumerate(formation_events):
        bond = event['bond']
        num_deps = event['dependencies']

        # Dependencies are on earlier bonds
        for j in range(num_deps):
            if j < len(formation_events):
                dep_bond = formation_events[j]['bond']
                if dep_bond != bond:
                    G.add_edge(dep_bond, bond)

    # Layout - hierarchical by formation cycle
    pos = {}
    cycle_groups = {}

    for node in G.nodes():
        cycle = G.nodes[node]['cycle']
        if cycle not in cycle_groups:
            cycle_groups[cycle] = []
        cycle_groups[cycle].append(node)

    # Position nodes in layers by cycle
    x_offset = 0
    for cycle in sorted(cycle_groups.keys()):
        nodes_in_cycle = cycle_groups[cycle]
        y_positions = np.linspace(-1, 1, len(nodes_in_cycle) + 2)[1:-1]

        for i, node in enumerate(nodes_in_cycle):
            pos[node] = (x_offset, y_positions[i] if len(nodes_in_cycle) > 1 else 0)

        x_offset += 1.5

    # Node colors by coherence
    node_colors = [G.nodes[node]['coherence'] for node in G.nodes()]

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='RdYlGn',
                        node_size=1000, vmin=0, vmax=1, ax=ax1,
                        edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                        arrowsize=20, width=2, ax=ax1,
                        connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax1)

    # Highlight folding nucleus
    nucleus_bond = pathway_data['folding_nucleus']['bond']
    if nucleus_bond in pos:
        nucleus_pos = pos[nucleus_bond]
        circle = Circle(nucleus_pos, 0.15, fill=False, edgecolor='gold',
                    linewidth=4, zorder=10)
        ax1.add_patch(circle)

    ax1.set_title('A. H-Bond Dependency Graph\n(Color = Coherence, Gold = Nucleus)',
                fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Add cycle labels
    for cycle, x in zip(sorted(cycle_groups.keys()),
                    np.arange(0, len(cycle_groups) * 1.5, 1.5)):
        ax1.text(x, -1.5, f'Cycle {cycle}', ha='center', fontsize=10,
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================================
    # CHART 2: Phase Coherence Clusters (Polar Plot)
    # ============================================================================
    ax2 = plt.subplot(2, 2, 2, projection='polar')

    final_network = folding_data['final_network_state']
    clusters = final_network['clusters']

    # Plot each cluster
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for i, cluster in enumerate(clusters):
        size = cluster['size']
        center_phase = cluster['center_phase']
        coherence = cluster['coherence']
        coupling = cluster['coupling']

        # Radial position based on coherence
        r = coherence

        # Angular width based on cluster size
        theta_width = 2 * np.pi / 20 * size  # Proportional to size

        # Plot wedge for cluster
        theta_start = center_phase - theta_width / 2
        theta_end = center_phase + theta_width / 2

        # Create wedge
        angles = np.linspace(theta_start, theta_end, 50)
        radii = np.ones(50) * r
        ax2.plot(angles, radii, color=cluster_colors[i], linewidth=8, alpha=0.7,
                label=f'Cluster {i+1} (n={size})')
        ax2.scatter([center_phase], [r], s=200, c=[cluster_colors[i]],
                edgecolors='black', linewidths=2, zorder=5, marker='o')

    # Add coherence circles
    for coherence_level in [0.5, 0.7, 0.9]:
        circle = Circle((0, 0), coherence_level, transform=ax2.transData._b,
                    fill=False, edgecolor='gray', linestyle=':', linewidth=1)

    ax2.set_ylim(0, 1)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('B. Phase-Coherence Clusters\n(Radius = Coherence, Angle = Phase)',
                fontsize=13, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=9)
    ax2.grid(True)

    # ============================================================================
    # CHART 3: GroEL Cavity Volume Modulation
    # ============================================================================
    ax3 = plt.subplot(2, 2, 3)

    # ATP cycle phases
    cycle_phases = np.linspace(0, 2*np.pi, 100)
    phase_names = ['ATP\nBound', 'Transition\nState', 'ADP+Pi', 'ADP\nRelease']
    phase_boundaries = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]

    # Volume modulation (from GroEL model)
    baseline_volume = 85000.0  # Angstrom^3

    def volume_modulation(phase):
        """Calculate cavity volume based on ATP cycle phase."""
        if phase < np.pi/2:  # ATP bound
            return baseline_volume * 0.9
        elif phase < np.pi:  # Transition
            return baseline_volume * 0.85
        elif phase < 3*np.pi/2:  # ADP+Pi
            return baseline_volume * 1.1
        else:  # ADP release
            return baseline_volume * 1.0

    volumes = [volume_modulation(phase) for phase in cycle_phases]

    # Plot volume trace
    ax3.plot(cycle_phases / np.pi, volumes, linewidth=3, color='#1976D2')
    ax3.fill_between(cycle_phases / np.pi, baseline_volume * 0.8, volumes,
                    alpha=0.3, color='#1976D2')

    # Add baseline
    ax3.axhline(y=baseline_volume, color='black', linestyle='--', linewidth=2,
            label=f'Baseline ({baseline_volume:.0f} ų)')

    # Shade ATP cycle phases
    colors_phase = ['#FFCDD2', '#FFF9C4', '#C8E6C9', '#B3E5FC']
    for i, (start, end) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
        ax3.axvspan(start / np.pi, end / np.pi, alpha=0.2, color=colors_phase[i])
        mid = (start + end) / 2 / np.pi
        ax3.text(mid, baseline_volume * 1.12, phase_names[i],
                ha='center', fontsize=9, fontweight='bold')

    ax3.set_xlabel('ATP Cycle Phase (× π radians)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cavity Volume (ų)', fontsize=12, fontweight='bold')
    ax3.set_title('C. GroEL Cavity Volume During ATP Cycle', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim(0, 2)

    # ============================================================================
    # CHART 4: Bond-Cycle Coupling Strength Heatmap
    # ============================================================================
    ax4 = plt.subplot(2, 2, 4)

    # Create matrix: bonds × cycles showing when bonds phase-lock
    bonds = [e['bond'] for e in formation_events]
    all_cycles_list = list(range(1, pathway_data['cycles_to_fold'] + 1))

    # Build coupling matrix
    coupling_matrix = np.zeros((len(bonds), len(all_cycles_list)))

    for i, event in enumerate(formation_events):
        bond = event['bond']
        formation_cycle = event['cycle']
        coherence = event['coherence']

        # Show coherence at formation cycle and beyond
        for j, cycle in enumerate(all_cycles_list):
            if cycle == formation_cycle:
                coupling_matrix[i, j] = coherence
            elif cycle > formation_cycle:
                # Bonds stay phase-locked after formation (with some decay)
                coupling_matrix[i, j] = coherence * 0.8

    # Plot heatmap
    im = ax4.imshow(coupling_matrix, aspect='auto', cmap='YlOrRd',
                vmin=0, vmax=1, interpolation='nearest')

    # Set ticks
    ax4.set_yticks(range(len(bonds)))
    ax4.set_yticklabels(bonds, fontsize=9)
    ax4.set_xticks(range(0, len(all_cycles_list), 5))
    ax4.set_xticklabels(range(1, len(all_cycles_list) + 1, 5))

    # Mark formation events
    for i, event in enumerate(formation_events):
        cycle_idx = event['cycle'] - 1
        ax4.plot(cycle_idx, i, 'g*', markersize=15, markeredgecolor='black',
                markeredgewidth=1)

    # Mark critical cycles
    for cc in pathway_data['critical_cycles']:
        ax4.axvline(x=cc - 1, color='blue', linestyle='--', linewidth=2, alpha=0.5)

    ax4.set_xlabel('ATP Cycle Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('H-Bond', fontsize=12, fontweight='bold')
    ax4.set_title('D. Bond Phase-Locking Across Cycles\n(★ = Formation Event)',
                fontsize=13, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax4, orientation='vertical', pad=0.02)
    cbar.set_label('Phase Coherence', fontsize=10)

    # ============================================================================
    # Save figure
    # ============================================================================
    plt.tight_layout()
    output_file = Path(__file__).parent / 'results' / 'network_structure_panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    plt.show()
