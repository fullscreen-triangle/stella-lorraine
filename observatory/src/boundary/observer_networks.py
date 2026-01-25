# Script 2: Observer Networks and Categorical Partitions
# File: observer_networks.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch
import seaborn as sns

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Single Observer Categorical Partition
    ax1 = fig.add_subplot(gs[0, 0])

    # Create a visual representation of categorical space
    np.random.seed(42)
    n_categories = 100
    categories_x = np.random.rand(n_categories)
    categories_y = np.random.rand(n_categories)

    # Observer position
    observer_x, observer_y = 0.5, 0.5

    # Determine observed vs unobserved (based on distance)
    distances = np.sqrt((categories_x - observer_x)**2 + (categories_y - observer_y)**2)
    observation_threshold = 0.3
    observed_mask = distances < observation_threshold

    # Plot
    ax1.scatter(categories_x[observed_mask], categories_y[observed_mask],
            c='blue', s=50, alpha=0.7, label='Observed', edgecolors='darkblue', linewidth=0.5)
    ax1.scatter(categories_x[~observed_mask], categories_y[~observed_mask],
            c='lightcoral', s=50, alpha=0.5, label='Unobserved', edgecolors='red', linewidth=0.5)

    # Observer
    ax1.scatter(observer_x, observer_y, c='gold', s=300, marker='*',
            edgecolors='black', linewidth=2, label='Observer', zorder=5)

    # Observation horizon
    circle = Circle((observer_x, observer_y), observation_threshold,
                fill=False, edgecolor='black', linewidth=2, linestyle='--', label='Observation Horizon')
    ax1.add_patch(circle)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Categorical Dimension 1', fontweight='bold')
    ax1.set_ylabel('Categorical Dimension 2', fontweight='bold')
    ax1.set_title('A. Single Observer:\nCategorical Partition', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax1.set_aspect('equal')

    # Add text annotations
    n_observed = np.sum(observed_mask)
    n_unobserved = np.sum(~observed_mask)
    ax1.text(0.05, 0.95, f'Observed: {n_observed}\nUnobserved: {n_unobserved}\nRatio: {n_unobserved/n_observed:.2f}:1',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 2: Multiple Observers
    ax2 = fig.add_subplot(gs[0, 1])

    # Three observers
    observers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
    colors_obs = ['gold', 'cyan', 'lime']
    observation_thresholds = [0.25, 0.25, 0.25]

    # Regenerate categories
    categories_x = np.random.rand(n_categories)
    categories_y = np.random.rand(n_categories)

    # Determine which observer(s) observe each category
    observed_by = [[] for _ in range(n_categories)]
    for i, (ox, oy) in enumerate(observers):
        distances = np.sqrt((categories_x - ox)**2 + (categories_y - oy)**2)
        for j, dist in enumerate(distances):
            if dist < observation_thresholds[i]:
                observed_by[j].append(i)

    # Plot categories colored by number of observers
    for j in range(n_categories):
        n_obs = len(observed_by[j])
        if n_obs == 0:
            color = 'lightgray'
            alpha = 0.3
            size = 30
        elif n_obs == 1:
            color = colors_obs[observed_by[j][0]]
            alpha = 0.6
            size = 50
        else:
            color = 'purple'
            alpha = 0.8
            size = 70

        ax2.scatter(categories_x[j], categories_y[j], c=color, s=size, alpha=alpha,
                edgecolors='black', linewidth=0.5)

    # Plot observers
    for i, (ox, oy) in enumerate(observers):
        ax2.scatter(ox, oy, c=colors_obs[i], s=300, marker='*',
                edgecolors='black', linewidth=2, zorder=5)
        circle = Circle((ox, oy), observation_thresholds[i],
                    fill=False, edgecolor=colors_obs[i], linewidth=2, linestyle='--')
        ax2.add_patch(circle)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Categorical Dimension 1', fontweight='bold')
    ax2.set_ylabel('Categorical Dimension 2', fontweight='bold')
    ax2.set_title('B. Multiple Observers:\nOverlapping Horizons', fontweight='bold', pad=20)
    ax2.set_aspect('equal')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', alpha=0.3, label='Unobserved by all'),
        Patch(facecolor='gold', alpha=0.6, label='Observed by one'),
        Patch(facecolor='purple', alpha=0.8, label='Observed by multiple')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=True,
            fancybox=True, shadow=True, fontsize=8)

    # Panel 3: Observer Network Graph
    ax3 = fig.add_subplot(gs[0, 2])

    # Create network of observers and shared observations
    G = nx.Graph()

    # Add observer nodes
    for i in range(3):
        G.add_node(f'O{i+1}', node_type='observer')

    # Add shared observation edges (weighted by number of shared categories)
    shared_counts = np.zeros((3, 3))
    for j in range(n_categories):
        obs_list = observed_by[j]
        for i1 in obs_list:
            for i2 in obs_list:
                if i1 < i2:
                    shared_counts[i1, i2] += 1

    for i1 in range(3):
        for i2 in range(i1+1, 3):
            if shared_counts[i1, i2] > 0:
                G.add_edge(f'O{i1+1}', f'O{i2+1}', weight=shared_counts[i1, i2])

    # Draw network
    pos = nx.spring_layout(G, seed=42)
    node_colors = [colors_obs[int(node[1])-1] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                        edgecolors='black', linewidths=2, ax=ax3)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax3)

    # Draw edges with width proportional to shared observations
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    widths = [5 * w / max_weight for w in weights]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, ax=ax3)

    # Edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.0f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax3)

    ax3.set_title('C. Observer Network:\nShared Observations', fontweight='bold', pad=20)
    ax3.axis('off')

    # Panel 4: Categorical Growth with Multiple Observers
    ax4 = fig.add_subplot(gs[1, 0])

    n_observers_range = np.arange(1, 11)
    categorical_complexity = []

    for n_obs in n_observers_range:
        # Each observer partitions space into observed/unobserved
        # Total partitions grow as 2^n_obs (simplified)
        complexity = 2**n_obs
        categorical_complexity.append(complexity)

    ax4.semilogy(n_observers_range, categorical_complexity, 'o-',
                linewidth=3, markersize=10, color='darkred')
    ax4.fill_between(n_observers_range, 1, categorical_complexity, alpha=0.3, color='darkred')

    ax4.set_xlabel('Number of Observers', fontweight='bold')
    ax4.set_ylabel('Categorical Partitions', fontweight='bold')
    ax4.set_title('D. Exponential Growth:\nMultiple Observers', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Add annotation
    ax4.annotate('2^n growth', xy=(5, 2**5), xytext=(7, 2**3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                fontsize=11, fontweight='bold')

    # Panel 5: Measurement Sequence and Categorical Actualization
    ax5 = fig.add_subplot(gs[1, 1])

    # Simulate measurement sequence
    n_measurements = 30
    C_total = 1000  # Total potential categories
    actualized = [0]
    potential = [C_total]

    np.random.seed(42)
    for m in range(n_measurements):
        # Each measurement actualizes some categories
        # Rate decreases as more are actualized (harder to find new ones)
        remaining = potential[-1]
        new_actualized = int(np.random.exponential(scale=remaining/100))
        new_actualized = min(new_actualized, remaining)

        actualized.append(actualized[-1] + new_actualized)
        potential.append(potential[-1] - new_actualized)

    measurements = np.arange(n_measurements + 1)
    ax5.plot(measurements, actualized, 'o-', linewidth=2, markersize=6,
            label='Actualized', color='steelblue')
    ax5.plot(measurements, potential, 's-', linewidth=2, markersize=6,
            label='Potential', color='coral')

    ax5.set_xlabel('Measurement Number', fontweight='bold')
    ax5.set_ylabel('Category Count', fontweight='bold')
    ax5.set_title('E. Measurement Sequence:\nActualization Dynamics', fontweight='bold', pad=20)
    ax5.legend(loc='right', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Add ratio line
    ax5_twin = ax5.twinx()
    ratio = np.array(potential[1:]) / (np.array(actualized[1:]) + 1)  # Avoid division by zero
    ax5_twin.plot(measurements[1:], ratio, 'g--', linewidth=2, alpha=0.7, label='Ratio')
    ax5_twin.set_ylabel('Potential/Actualized Ratio', fontweight='bold', color='green')
    ax5_twin.tick_params(axis='y', labelcolor='green')

    # Panel 6: Observer Embedding Paradox
    ax6 = fig.add_subplot(gs[1, 2])

    # Visualize the recursion: observer trying to observe themselves
    levels = ['System', 'Observer\nin System', 'Observer\nObserving\nObserver', 'Observer\nObserving\nObserver\nObserving...']
    sizes = [100, 80, 60, 40]
    colors_levels = plt.cm.Reds(np.linspace(0.3, 0.9, len(levels)))

    y_pos = np.arange(len(levels))
    bars = ax6.barh(y_pos, sizes, color=colors_levels, edgecolor='black', linewidth=2)

    # Add infinity symbol at the end
    ax6.text(45, 3, '∞', fontsize=40, ha='center', va='center', fontweight='bold')

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(levels, fontsize=9)
    ax6.set_xlabel('Categorical Complexity', fontweight='bold')
    ax6.set_title('F. Observer Embedding:\nInfinite Regress', fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3, axis='x', linestyle='--')

    # Add arrows showing recursion
    for i in range(len(levels) - 1):
        ax6.annotate('', xy=(sizes[i+1] - 5, i + 1), xytext=(sizes[i] - 5, i),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))

    plt.suptitle('Observer Networks and Categorical Partitions',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('observer_networks_panel.png', dpi=300, bbox_inches='tight')
    plt.savefig('observer_networks_panel.pdf', dpi=300, bbox_inches='tight')
    print("Saved: observer_networks_panel.png and .pdf")
    plt.show()
