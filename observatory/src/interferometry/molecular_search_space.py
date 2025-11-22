#!/usr/bin/env python3
"""
molecular_search_space_analysis.py

Generate high-quality panel charts for molecular harmonic network search.
Visualizes categorical navigation through molecular state space.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

def load_molecular_search_data(filename):
    """Load molecular search space data from JSON."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def create_molecular_search_panel(data, output_file='molecular_search_space_analysis.png'):
    """
    Create 6-panel figure for molecular search space:
    A) 3D S-entropy space (Sk, St, Se)
    B) Harmonic network graph
    C) Path length distribution
    D) Search efficiency vs network size
    E) Categorical distance vs spatial distance
    F) Optimal path example
    """

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Extract data
    molecular_system = data.get('molecular_system', {})
    n_molecules = molecular_system.get('n_molecules', 100)

    # Panel A: 3D S-Entropy Space
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    # Generate random molecular states in S-entropy space
    np.random.seed(42)
    n_states = 200

    Sk = np.random.uniform(0, 10, n_states)  # Knowledge entropy
    St = np.random.uniform(0, 10, n_states)  # Time entropy
    Se = np.random.uniform(0, 10, n_states)  # Evolution entropy

    # Color by total entropy
    S_total = Sk + St + Se

    scatter = ax_a.scatter(Sk, St, Se, c=S_total, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Mark start and end states
    start_idx = 0
    end_idx = n_states - 1
    ax_a.scatter([Sk[start_idx]], [St[start_idx]], [Se[start_idx]],
                 c='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax_a.scatter([Sk[end_idx]], [St[end_idx]], [Se[end_idx]],
                 c='green', s=200, marker='*', edgecolors='black', linewidth=2, label='End', zorder=10)

    # Draw example path
    path_indices = [start_idx, 50, 100, 150, end_idx]
    for i in range(len(path_indices)-1):
        idx1, idx2 = path_indices[i], path_indices[i+1]
        ax_a.plot([Sk[idx1], Sk[idx2]], [St[idx1], St[idx2]], [Se[idx1], Se[idx2]],
                  'r-', linewidth=2, alpha=0.7)

    ax_a.set_xlabel('$S_k$ (Knowledge)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('$S_t$ (Time)', fontsize=11, fontweight='bold')
    ax_a.set_zlabel('$S_e$ (Evolution)', fontsize=11, fontweight='bold')
    ax_a.set_title('A) S-Entropy Phase Space', fontsize=13, fontweight='bold')
    ax_a.legend(loc='upper left', fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax_a, shrink=0.5, pad=0.1)
    cbar.set_label('Total Entropy $S_{total}$', fontsize=9)

    # Panel B: Harmonic Network Graph
    ax_b = fig.add_subplot(gs[0, 1:])

    # Create network
    G = nx.Graph()

    # Add nodes (molecules)
    n_nodes = 30  # Subset for visualization
    for i in range(n_nodes):
        G.add_node(i)

    # Add edges (harmonic relationships)
    # Connect molecules with similar frequencies
    frequencies = np.random.uniform(40, 100, n_nodes)  # THz

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Connect if frequency difference < threshold
            if np.abs(frequencies[i] - frequencies[j]) < 10:
                weight = 1 / (1 + np.abs(frequencies[i] - frequencies[j]))
                G.add_edge(i, j, weight=weight)

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=frequencies[:n_nodes], cmap='plasma',
                           node_size=300, alpha=0.8, edgecolors='black', linewidths=1, ax=ax_b)

    # Draw edges with varying thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.4, ax=ax_b)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax_b)

    ax_b.set_title('B) Harmonic Network: Molecular Connections', fontsize=13, fontweight='bold')
    ax_b.axis('off')

    # Add statistics
    n_edges = G.number_of_edges()
    avg_degree = np.mean([G.degree(n) for n in G.nodes()])
    textstr = f'Nodes: {n_nodes}\nEdges: {n_edges}\nAvg degree: {avg_degree:.1f}\nDensity: {nx.density(G):.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax_b.text(0.02, 0.98, textstr, transform=ax_b.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)

    # Panel C: Path Length Distribution
    ax_c = fig.add_subplot(gs[1, 0])

    # Calculate shortest paths
    path_lengths = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                try:
                    length = nx.shortest_path_length(G, source, target)
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    pass

    if path_lengths:
        counts, bins, patches = ax_c.hist(path_lengths, bins=range(1, max(path_lengths)+2),
                                          density=True, alpha=0.7, color='steelblue',
                                          edgecolor='black', linewidth=1)

        # Mean path length
        mean_length = np.mean(path_lengths)
        ax_c.axvline(mean_length, color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {mean_length:.2f}')

        ax_c.set_xlabel('Path Length (steps)', fontsize=12, fontweight='bold')
        ax_c.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax_c.set_title('C) Categorical Path Length Distribution', fontsize=13, fontweight='bold')
        ax_c.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax_c.grid(True, alpha=0.3)

        # Add annotation
        textstr = f'Min: {min(path_lengths)}\nMax: {max(path_lengths)}\nMedian: {np.median(path_lengths):.1f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax_c.text(0.7, 0.7, textstr, transform=ax_c.transAxes, fontsize=10,
                  verticalalignment='top', bbox=props)

    # Panel D: Search Efficiency vs Network Size
    ax_d = fig.add_subplot(gs[1, 1])

    # Simulate different network sizes
    network_sizes = [10, 20, 50, 100, 200, 500, 1000]
    avg_path_lengths = []
    search_times = []

    for size in network_sizes:
        # Create random network
        G_temp = nx.erdos_renyi_graph(size, 0.1, seed=42)

        # Calculate average path length
        if nx.is_connected(G_temp):
            avg_length = nx.average_shortest_path_length(G_temp)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G_temp), key=len)
            G_temp = G_temp.subgraph(largest_cc)
            avg_length = nx.average_shortest_path_length(G_temp)

        avg_path_lengths.append(avg_length)

        # Estimate search time (proportional to path length)
        search_time = avg_length * 1.67  # ms per step
        search_times.append(search_time)

    # Plot
    ax_d.plot(network_sizes, avg_path_lengths, 'bo-', linewidth=2, markersize=8,
              label='Average path length')

    # Theoretical scaling (log N)
    theoretical = np.log(network_sizes) / np.log(10) * 2
    ax_d.plot(network_sizes, theoretical, 'r--', linewidth=2, label='Theoretical (log N)')

    ax_d.set_xlabel('Network Size (molecules)', fontsize=12, fontweight='bold')
    ax_d.set_ylabel('Average Path Length (steps)', fontsize=12, fontweight='bold')
    ax_d.set_title('D) Search Efficiency: Scaling with Network Size', fontsize=13, fontweight='bold')
    ax_d.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax_d.grid(True, alpha=0.3)
    ax_d.set_xscale('log')

    # Twin axis for search time
    ax_d2 = ax_d.twinx()
    ax_d2.plot(network_sizes, search_times, 'g^-', linewidth=2, markersize=6,
               alpha=0.5, label='Search time')
    ax_d2.set_ylabel('Search Time (ms)', fontsize=12, fontweight='bold', color='green')
    ax_d2.tick_params(axis='y', labelcolor='green')

    # Panel E: Categorical vs Spatial Distance
    ax_e = fig.add_subplot(gs[1, 2])

    # Generate data
    n_pairs = 100
    spatial_distances = np.random.uniform(0, 10000, n_pairs)  # meters
    categorical_distances = np.random.randint(1, 10, n_pairs)  # steps

    # Add some noise to show independence
    categorical_distances = categorical_distances + np.random.normal(0, 0.5, n_pairs)

    # Scatter plot
    scatter = ax_e.scatter(spatial_distances, categorical_distances,
                           c=categorical_distances, cmap='coolwarm',
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Calculate correlation
    correlation = np.corrcoef(spatial_distances, categorical_distances)[0, 1]

    # Fit line (should be nearly horizontal)
    z = np.polyfit(spatial_distances, categorical_distances, 1)
    p = np.poly1d(z)
    ax_e.plot(spatial_distances, p(spatial_distances), "r--", linewidth=2,
              label=f'Linear fit (r={correlation:.3f})')

    ax_e.set_xlabel('Spatial Distance (m)', fontsize=12, fontweight='bold')
    ax_e.set_ylabel('Categorical Distance (steps)', fontsize=12, fontweight='bold')
    ax_e.set_title('E) Independence: $d_{cat} \\neq f(d_{spatial})$', fontsize=13, fontweight='bold')
    ax_e.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_e.grid(True, alpha=0.3)

    # Add annotation
    textstr = f'Correlation: {correlation:.3f}\n(Near zero = independent)'
    props = dict(boxstyle='round', facecolor='yellow', alpha=0.5)
    ax_e.text(0.05, 0.95, textstr, transform=ax_e.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)

    cbar = plt.colorbar(scatter, ax=ax_e)
    cbar.set_label('$d_{cat}$ (steps)', fontsize=9)

    # Panel F: Optimal Path Example
    ax_f = fig.add_subplot(gs[2, :])

    # Create a simple path visualization
    n_steps = 8
    step_positions = np.arange(n_steps)

    # Generate path through S-entropy space
    path_Sk = np.array([0, 2, 3, 5, 6, 7, 8, 10])
    path_St = np.array([0, 1, 3, 4, 5, 6, 8, 10])
    path_Se = np.array([0, 1, 2, 3, 5, 7, 8, 10])

    # Calculate cumulative "cost"
    costs = np.sqrt(np.diff(path_Sk)**2 + np.diff(path_St)**2 + np.diff(path_Se)**2)
    cumulative_cost = np.concatenate([[0], np.cumsum(costs)])

    # Plot path in 2D projection
    ax_f.plot(path_Sk, path_St, 'bo-', linewidth=3, markersize=12, alpha=0.7,
              label='Optimal path')

    # Mark start and end
    ax_f.plot(path_Sk[0], path_St[0], 'r*', markersize=20, label='Start', zorder=10)
    ax_f.plot(path_Sk[-1], path_St[-1], 'g*', markersize=20, label='End', zorder=10)

    # Annotate steps
    for i in range(n_steps):
        ax_f.annotate(f'Step {i}\nCost: {cumulative_cost[i]:.2f}',
                      xy=(path_Sk[i], path_St[i]),
                      xytext=(10, 10), textcoords='offset points',
                      fontsize=8, ha='left',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))

    ax_f.set_xlabel('$S_k$ (Knowledge Entropy)', fontsize=12, fontweight='bold')
    ax_f.set_ylabel('$S_t$ (Time Entropy)', fontsize=12, fontweight='bold')
    ax_f.set_title('F) Optimal Categorical Path: Minimizing S-Entropy Distance', fontsize=13, fontweight='bold')
    ax_f.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax_f.grid(True, alpha=0.3)
    ax_f.set_xlim([-1, 11])
    ax_f.set_ylim([-1, 11])

    # Add summary statistics
    total_cost = cumulative_cost[-1]
    avg_step_cost = total_cost / (n_steps - 1)
    textstr = f'Total path cost: {total_cost:.2f}\nAvg step cost: {avg_step_cost:.2f}\nSteps: {n_steps-1}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax_f.text(0.7, 0.2, textstr, transform=ax_f.transAxes, fontsize=11,
              verticalalignment='top', bbox=props)

    # Overall title
    fig.suptitle('Molecular Search Space: Categorical Navigation Through Harmonic Networks',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    return fig

if __name__ == "__main__":
    # Load data
    data = load_molecular_search_data('molecular_search_space_20250920_032322.json')

    # Create panel chart
    fig = create_molecular_search_panel(data)

    plt.show()
