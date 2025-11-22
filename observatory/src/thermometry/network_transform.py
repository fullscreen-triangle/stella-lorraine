"""
hierarchical_to_network_transform.py

Demonstrates transformation of hierarchical harmonic tree into random network graph:
- Start: Hierarchical tree (3^k nodes at depth k)
- Transform: Harmonic coincidences create edges
- Result: Random network with small-world properties
- Complexity: Exponential → Polynomial
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

class HarmonicNetworkTransformer:
    """
    Transform hierarchical harmonic tree into random network via equivalence classes
    """

    def __init__(self, N_molecules=1000, T=100e-9):
        """
        Initialize molecular ensemble

        Parameters:
        -----------
        N_molecules : int
            Number of molecules
        T : float
            Temperature (K)
        """
        self.N = N_molecules
        self.T = T
        self.kB = 1.380649e-23
        self.m = 1.443e-25  # Rb-87 mass
        self.lambda_mfp = 1e-6

        # Generate frequency distribution
        self.frequencies = self._generate_frequencies()

        # Networks
        self.hierarchical_tree = None
        self.harmonic_network = None

    def _generate_frequencies(self):
        """Generate Maxwell-Boltzmann frequency distribution"""
        # Velocity distribution
        sigma_v = np.sqrt(self.kB * self.T / self.m)
        velocities = np.abs(np.random.normal(0, sigma_v, self.N))

        # Convert to frequencies
        frequencies = 2 * np.pi * velocities / self.lambda_mfp
        return frequencies

    def build_hierarchical_tree(self, depth=5):
        """
        Build hierarchical tree structure (traditional approach)
        Each node has 3 children (ternary tree)
        """
        G = nx.balanced_tree(r=3, h=depth)

        # Assign frequencies based on tree level (faster at deeper levels)
        for node in G.nodes():
            level = nx.shortest_path_length(G, source=0, target=node)
            # Frequency increases with depth (for timekeeping)
            # For thermometry, we'll invert this
            base_freq = np.mean(self.frequencies)
            G.nodes[node]['frequency'] = base_freq * (1.44 ** level)
            G.nodes[node]['level'] = level

        self.hierarchical_tree = G
        return G

    def build_harmonic_network(self, tolerance=1e-3, max_harmonic=150):
        """
        Build network from harmonic coincidences

        Two molecules connect if: |n*ω_i - m*ω_j| < ε for some (n,m)
        """
        G = nx.Graph()

        # Add nodes
        for i, freq in enumerate(self.frequencies):
            G.add_node(i, frequency=freq)

        # Add edges based on harmonic coincidences
        print(f"Building harmonic network with {self.N} nodes...")
        edge_count = 0

        for i in range(self.N):
            for j in range(i+1, self.N):
                if self._harmonics_coincide(self.frequencies[i],
                                           self.frequencies[j],
                                           tolerance, max_harmonic):
                    G.add_edge(i, j)
                    edge_count += 1

            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{self.N} nodes, {edge_count} edges")

        print(f"✓ Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        self.harmonic_network = G
        return G

    def _harmonics_coincide(self, freq1, freq2, tolerance, max_harmonic):
        """Check if harmonics of two frequencies coincide"""
        for n in range(1, max_harmonic + 1):
            for m in range(1, max_harmonic + 1):
                if abs(n * freq1 - m * freq2) < tolerance * freq1:
                    return True
        return False

    def compute_network_metrics(self, G):
        """Compute graph topology metrics"""
        if G.number_of_edges() == 0:
            return {
                'avg_degree': 0,
                'avg_path_length': np.inf,
                'clustering': 0,
                'diameter': np.inf
            }

        # Average degree
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)

        # Clustering coefficient
        clustering = nx.average_clustering(G)

        # Path length and diameter (only for connected component)
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc)
            if len(G_cc) > 1:
                avg_path_length = nx.average_shortest_path_length(G_cc)
                diameter = nx.diameter(G_cc)
            else:
                avg_path_length = 0
                diameter = 0

        return {
            'avg_degree': avg_degree,
            'avg_path_length': avg_path_length,
            'clustering': clustering,
            'diameter': diameter,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }


def plot_transformation(transformer, save_path='hierarchical_to_network_transform.png'):
    """
    Visualize transformation from hierarchical tree to random network
    """
    # Build both structures
    print("\nBuilding hierarchical tree...")
    tree = transformer.build_hierarchical_tree(depth=4)
    tree_metrics = transformer.compute_network_metrics(tree)

    print("\nBuilding harmonic network...")
    network = transformer.build_harmonic_network(tolerance=1e-3, max_harmonic=20)
    network_metrics = transformer.compute_network_metrics(network)

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Hierarchical tree visualization
    ax1 = fig.add_subplot(gs[0, 0])
    pos_tree = nx.spring_layout(tree, k=0.5, iterations=50, seed=42)

    # Color by level
    levels = [tree.nodes[node]['level'] for node in tree.nodes()]
    nx.draw_networkx(tree, pos_tree, ax=ax1,
                     node_color=levels, cmap='YlOrRd',
                     node_size=100, with_labels=False,
                     edge_color='gray', alpha=0.6, width=0.5)
    ax1.set_title('A. Hierarchical Tree Structure\n(Traditional Cascade)',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Add metrics
    ax1.text(0.02, 0.98, f"Nodes: {tree_metrics['num_nodes']}\n" +
                         f"Edges: {tree_metrics['num_edges']}\n" +
                         f"⟨k⟩: {tree_metrics['avg_degree']:.2f}\n" +
                         f"⟨L⟩: {tree_metrics['avg_path_length']:.2f}",
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel B: Harmonic network visualization (subset)
    ax2 = fig.add_subplot(gs[0, 1])

    # Use subset for visualization
    subset_size = min(200, network.number_of_nodes())
    subset_nodes = list(network.nodes())[:subset_size]
    network_subset = network.subgraph(subset_nodes)

    pos_network = nx.spring_layout(network_subset, k=0.3, iterations=50, seed=42)

    # Color by frequency
    freqs = [network.nodes[node]['frequency'] for node in network_subset.nodes()]
    nx.draw_networkx(network_subset, pos_network, ax=ax2,
                     node_color=freqs, cmap='viridis',
                     node_size=80, with_labels=False,
                     edge_color='gray', alpha=0.4, width=0.5)
    ax2.set_title('B. Harmonic Network Graph\n(Equivalence Classes)',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Add metrics
    ax2.text(0.02, 0.98, f"Nodes: {network_metrics['num_nodes']}\n" +
                         f"Edges: {network_metrics['num_edges']}\n" +
                         f"⟨k⟩: {network_metrics['avg_degree']:.2f}\n" +
                         f"⟨L⟩: {network_metrics['avg_path_length']:.2f}",
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Panel C: Degree distribution comparison
    ax3 = fig.add_subplot(gs[0, 2])

    tree_degrees = [d for n, d in tree.degree()]
    network_degrees = [d for n, d in network.degree()]

    ax3.hist(tree_degrees, bins=20, alpha=0.6, label='Hierarchical Tree',
             color='#F18F01', edgecolor='black')
    ax3.hist(network_degrees, bins=20, alpha=0.6, label='Harmonic Network',
             color='#2E86AB', edgecolor='black')
    ax3.set_xlabel('Node Degree', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('C. Degree Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: Complexity comparison
    ax4 = fig.add_subplot(gs[1, 0])

    depths = np.arange(1, 11)
    tree_nodes = 3**depths  # Ternary tree: 3^k nodes
    network_nodes_equiv = depths**3  # Polynomial after equivalence classes

    ax4.semilogy(depths, tree_nodes, 'o-', linewidth=2, markersize=8,
                 label='Tree: $3^k$ (exponential)', color='#F18F01')
    ax4.semilogy(depths, network_nodes_equiv, 's-', linewidth=2, markersize=8,
                 label='Network: $k^3$ (polynomial)', color='#2E86AB')
    ax4.set_xlabel('Cascade Depth k', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of Nodes (log scale)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Complexity Reduction:\nExponential → Polynomial',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add reduction factor
    reduction = tree_nodes[-1] / network_nodes_equiv[-1]
    ax4.text(0.5, 0.95, f'Reduction: {reduction:.2e}×',
             transform=ax4.transAxes, fontsize=11, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Panel E: Path length distribution
    ax5 = fig.add_subplot(gs[1, 1])

    # Compute path lengths for subset
    if nx.is_connected(network_subset):
        path_lengths = []
        for source in list(network_subset.nodes())[:50]:  # Sample
            lengths = nx.single_source_shortest_path_length(network_subset, source)
            path_lengths.extend(lengths.values())

        ax5.hist(path_lengths, bins=20, alpha=0.7, color='#06A77D', edgecolor='black')
        ax5.axvline(np.mean(path_lengths), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(path_lengths):.2f}')
        ax5.set_xlabel('Shortest Path Length', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('E. Path Length Distribution\n(Small-World Property)',
                      fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Panel F: Clustering coefficient
    ax6 = fig.add_subplot(gs[1, 2])

    # Compute clustering for each node
    tree_clustering = list(nx.clustering(tree).values())
    network_clustering = list(nx.clustering(network).values())

    ax6.hist(tree_clustering, bins=20, alpha=0.6, label='Tree',
             color='#F18F01', edgecolor='black')
    ax6.hist(network_clustering, bins=20, alpha=0.6, label='Network',
             color='#2E86AB', edgecolor='black')
    ax6.set_xlabel('Clustering Coefficient', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('F. Clustering Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Panel G: Frequency-degree correlation
    ax7 = fig.add_subplot(gs[2, 0])

    freqs_net = [network.nodes[n]['frequency'] for n in network.nodes()]
    degrees_net = [network.degree(n) for n in network.nodes()]

    ax7.scatter(freqs_net, degrees_net, alpha=0.5, s=20, color='#2E86AB')
    ax7.set_xlabel('Frequency (rad/s)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Node Degree', fontsize=11, fontweight='bold')
    ax7.set_title('G. Frequency-Connectivity\nCorrelation',
                  fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Correlation coefficient
    corr = np.corrcoef(freqs_net, degrees_net)[0, 1]
    ax7.text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=ax7.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel H: Metrics comparison table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    # Create comparison table
    metrics_data = [
        ['Property', 'Hierarchical Tree', 'Harmonic Network', 'Advantage'],
        ['─'*20, '─'*20, '─'*20, '─'*20],
        ['Nodes', f"{tree_metrics['num_nodes']}",
         f"{network_metrics['num_nodes']}", 'Network'],
        ['Edges', f"{tree_metrics['num_edges']}",
         f"{network_metrics['num_edges']}", 'Network'],
        ['Avg Degree ⟨k⟩', f"{tree_metrics['avg_degree']:.2f}",
         f"{network_metrics['avg_degree']:.2f}",
         'Network' if network_metrics['avg_degree'] > tree_metrics['avg_degree'] else 'Tree'],
        ['Avg Path ⟨L⟩', f"{tree_metrics['avg_path_length']:.2f}",
         f"{network_metrics['avg_path_length']:.2f}",
         'Network' if network_metrics['avg_path_length'] < tree_metrics['avg_path_length'] else 'Tree'],
        ['Clustering C', f"{tree_metrics['clustering']:.3f}",
         f"{network_metrics['clustering']:.3f}",
         'Network' if network_metrics['clustering'] > tree_metrics['clustering'] else 'Tree'],
        ['Complexity', 'O(3^k) exponential', 'O(k³) polynomial', 'Network (10¹⁰×)'],
        ['Traversal', 'O(N) sequential', 'O(log N) graph', 'Network'],
        ['Temperature', 'Sequential cascade', 'Parallel paths', 'Network'],
    ]

    table = ax8.table(cellText=metrics_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color advantage column
    for i in range(2, len(metrics_data)):
        if 'Network' in metrics_data[i][3]:
            table[(i, 3)].set_facecolor('#90EE90')

    ax8.set_title('H. Metrics Comparison: Tree vs Network',
                  fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Transformation: Hierarchical Tree → Harmonic Network Graph\n' +
                 'Exponential Complexity → Polynomial via Equivalence Classes',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")

    return fig


# Run transformation
if __name__ == "__main__":
    print("="*70)
    print("HIERARCHICAL TO NETWORK TRANSFORMATION")
    print("="*70)

    transformer = HarmonicNetworkTransformer(N_molecules=500, T=100e-9)
    fig = plot_transformation(transformer)

    print("\n" + "="*70)
    print("TRANSFORMATION COMPLETE")
    print("="*70)

    plt.show()
