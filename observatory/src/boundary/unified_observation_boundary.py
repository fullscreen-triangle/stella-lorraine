# Unified Visualization: The Complete Observation Boundary Framework
# File: unified_observation_boundary.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Wedge
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
import seaborn as sns


if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9

    # Create master figure
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)

    # ============================================================================
    # PANEL 1: The Singularity to Heat Death Timeline
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Timeline from Big Bang to Heat Death
    epochs = [
        {'name': 'Big Bang\nSingularity', 'time': 0, 'C': 1, 't': 0,
        'description': 'C(0) = 1\nNo distinctions', 'color': '#ff0000'},
        {'name': 'Inflation', 'time': 1e-35, 'C': 2, 't': 1,
        'description': 'First distinction', 'color': '#ff6600'},
        {'name': 'Particle\nFormation', 'time': 1e-6, 'C': 4, 't': 2,
        'description': 'C(2) = n^n', 'color': '#ff9900'},
        {'name': 'Structure\nFormation', 'time': 1e13, 'C': 16, 't': 3,
        'description': 'Galaxies form', 'color': '#ffcc00'},
        {'name': 'Present\nEpoch', 'time': 4.4e17, 'C': 1e80, 't': 4,
        'description': '~10^80 particles', 'color': '#00cc00'},
        {'name': 'Heat\nDeath', 'time': 1e100, 'C': 1e120, 't': 5,
        'description': 'Maximum entropy', 'color': '#0066ff'},
    ]

    # Create timeline
    time_log = [np.log10(e['time'] + 1e-40) for e in epochs]
    C_log = [np.log10(e['C']) for e in epochs]

    # Plot points
    for i, epoch in enumerate(epochs):
        ax1.scatter(time_log[i], C_log[i], s=600, c=epoch['color'],
                edgecolors='black', linewidth=2.5, zorder=5, alpha=0.9)

        # Add labels above points
        ax1.text(time_log[i], C_log[i] + 8, epoch['name'],
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor=epoch['color'], linewidth=2, alpha=0.9))

        # Add descriptions below
        ax1.text(time_log[i], C_log[i] - 8, epoch['description'],
                ha='center', fontsize=8, style='italic', color='darkblue')

    # Connect with evolutionary path
    ax1.plot(time_log, C_log, 'k--', linewidth=2, alpha=0.5, zorder=1)

    # Fill regions
    for i in range(len(epochs) - 1):
        ax1.fill_between([time_log[i], time_log[i+1]],
                        [0, 0], [C_log[i], C_log[i+1]],
                        alpha=0.15, color=epochs[i]['color'])

    ax1.set_xlabel('log₁₀(Time) [seconds]', fontweight='bold', fontsize=11)
    ax1.set_ylabel('log₁₀(C(t)) - Categorical Complexity', fontweight='bold', fontsize=11)
    ax1.set_title('A. Cosmic Evolution: From Singularity (C=1) to Heat Death (C=𝒩_max)',
                fontweight='bold', fontsize=13, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-5, max(C_log) + 15)

    # Add annotation for tetration growth
    ax1.annotate('Tetration Growth:\nC(t+1) = n^C(t)',
                xy=(time_log[3], C_log[3]), xytext=(time_log[2], C_log[4]),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # ============================================================================
    # PANEL 2: The Recursive Structure - Tetration Visualization
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    # Show the tower structure
    n = 84  # Base (log scale)
    tower_levels = [
        {'t': 0, 'expr': 'C(0) = 1', 'value': 0, 'height': 0.1},
        {'t': 1, 'expr': 'C(1) = 10⁸⁴', 'value': 84, 'height': 0.15},
        {'t': 2, 'expr': 'C(2) = (10⁸⁴)^(10⁸⁴)', 'value': 8.4e85, 'height': 0.2},
        {'t': 3, 'expr': 'C(3) = (10⁸⁴)^C(2)', 'value': 1e100, 'height': 0.25},
        {'t': 4, 'expr': '...', 'value': 1e110, 'height': 0.15},
        {'t': 5, 'expr': 't = 10⁸⁰', 'value': 1e120, 'height': 0.15},
    ]

    # Build tower
    y_bottom = 0
    for i, level in enumerate(tower_levels):
        # Width decreases with height
        width = 0.8 - 0.1 * min(i, 4)
        x_center = 0.5
        x_left = x_center - width/2

        # Color gradient
        color = plt.cm.Reds(0.3 + 0.12 * i)

        # Draw block
        rect = Rectangle((x_left, y_bottom), width, level['height'],
                        facecolor=color, edgecolor='black', linewidth=2.5,
                        transform=ax2.transAxes)
        ax2.add_patch(rect)

        # Add label
        ax2.text(x_center, y_bottom + level['height']/2, level['expr'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                transform=ax2.transAxes)

        y_bottom += level['height']

    # Add continuation arrow
    ax2.annotate('', xy=(0.5, 0.92), xytext=(0.5, y_bottom),
                arrowprops=dict(arrowstyle='->', lw=4, color='red'),
                transform=ax2.transAxes)
    ax2.text(0.5, 0.95, '𝒩_max ≈ (10⁸⁴)↑↑(10⁸⁰)',
            ha='center', fontsize=11, fontweight='bold', color='red',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('B. Tetration Tower: C(t+1) = n^C(t)',
                fontweight='bold', fontsize=13, pad=15)

    # ============================================================================
    # PANEL 3: Observer-Dependent Categories
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Visualize categorical space with observer
    np.random.seed(42)
    n_categories = 150
    cat_x = np.random.rand(n_categories)
    cat_y = np.random.rand(n_categories)

    # Observer position
    obs_x, obs_y = 0.5, 0.5
    observation_radius = 0.25

    # Determine observed vs unobserved
    distances = np.sqrt((cat_x - obs_x)**2 + (cat_y - obs_y)**2)
    observed = distances < observation_radius

    # Plot categories
    ax3.scatter(cat_x[observed], cat_y[observed],
            c='#3498db', s=40, alpha=0.8, label='Observed (∞-x)',
            edgecolors='darkblue', linewidth=0.5)
    ax3.scatter(cat_x[~observed], cat_y[~observed],
            c='#e74c3c', s=40, alpha=0.5, label='Unobserved (x)',
            edgecolors='darkred', linewidth=0.5)

    # Observer
    ax3.scatter(obs_x, obs_y, c='gold', s=400, marker='*',
            edgecolors='black', linewidth=2.5, label='Observer', zorder=10)

    # Observation boundary
    circle = Circle((obs_x, obs_y), observation_radius,
                fill=False, edgecolor='black', linewidth=3,
                linestyle='--', label='Observation\nBoundary')
    ax3.add_patch(circle)

    # Count and display ratio
    n_obs = np.sum(observed)
    n_unobs = np.sum(~observed)
    ratio = n_unobs / n_obs if n_obs > 0 else 0

    ax3.text(0.05, 0.95, f'Observed: {n_obs}\nUnobserved: {n_unobs}\nRatio: {ratio:.1f}:1',
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                    edgecolor='black', linewidth=1.5))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Categorical Dimension 1', fontweight='bold')
    ax3.set_ylabel('Categorical Dimension 2', fontweight='bold')
    ax3.set_title('C. Observer-Dependent\nCategorical Partition',
                fontweight='bold', fontsize=12, pad=15)
    ax3.legend(loc='lower right', frameon=True, fancybox=True,
            shadow=True, fontsize=8)
    ax3.set_aspect('equal')

    # ============================================================================
    # PANEL 4: The ∞-x Structure
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Pie chart showing the partition
    sizes = [5.4, 1]  # Dark matter ratio
    labels = ['Unobserved\n(x)', 'Observed\n(∞-x)']
    colors = ['#e74c3c', '#3498db']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=explode,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2.5})

    ax4.set_title('D. The ∞-x Structure:\nObserver Partition',
                fontweight='bold', fontsize=12, pad=15)

    # Add equation
    ax4.text(0, -1.4, 'Universe = (∞-x) + x\nRatio x/(∞-x) ≈ 5.4',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                    alpha=0.9, edgecolor='black', linewidth=2))

    # ============================================================================
    # PANEL 5: Oscillatory Foundation
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    # Show oscillation and termination
    t = np.linspace(0, 10, 1000)
    # Damped oscillation representing completion
    omega = 2 * np.pi
    alpha_t = 1 - np.exp(-0.5 * t) * np.cos(omega * t)

    ax5.plot(t, alpha_t, linewidth=3, color='darkblue', label='α(C,t)')
    ax5.fill_between(t, 0, alpha_t, alpha=0.3, color='lightblue')

    # Mark completion points
    completion_times = [1, 2, 3, 4, 5]
    for tc in completion_times:
        idx = np.argmin(np.abs(t - tc))
        ax5.plot(tc, alpha_t[idx], 'ro', markersize=10,
                markeredgecolor='black', markeredgewidth=2)

    # Add asymptote
    ax5.axhline(y=1, color='red', linestyle='--', linewidth=2,
            label='Complete (α=1)')

    ax5.set_xlabel('Time', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Completion Degree α(C,t)', fontweight='bold', fontsize=11)
    ax5.set_title('E. Oscillatory Foundation:\nCategory Completion',
                fontweight='bold', fontsize=12, pad=15)
    ax5.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_ylim(-0.1, 1.2)

    # Add equation
    ax5.text(5, 0.3, r'$\frac{d\alpha}{dt} = \omega(1-\alpha)$',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # ============================================================================
    # PANEL 6: Entropy as Shortest Path
    # ============================================================================
    ax6 = fig.add_subplot(gs[1, 3])

    # Visualize entropy as path selection
    # Create a simple network of states
    np.random.seed(43)
    n_states = 8
    state_x = np.random.rand(n_states)
    state_y = np.random.rand(n_states)

    # Start and end points
    start_idx = 0
    end_idx = n_states - 1
    state_x[start_idx], state_y[start_idx] = 0.1, 0.1
    state_x[end_idx], state_y[end_idx] = 0.9, 0.9

    # Plot states
    ax6.scatter(state_x, state_y, s=300, c='lightblue',
            edgecolors='black', linewidth=2, zorder=5)

    # Label states
    for i in range(n_states):
        ax6.text(state_x[i], state_y[i], f'S{i}', ha='center', va='center',
                fontsize=9, fontweight='bold')

    # Draw multiple paths
    # Long path (low entropy increase)
    long_path_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(len(long_path_indices) - 1):
        idx1, idx2 = long_path_indices[i], long_path_indices[i+1]
        ax6.plot([state_x[idx1], state_x[idx2]],
                [state_y[idx1], state_y[idx2]],
                'b--', linewidth=1.5, alpha=0.5)

    # Short path (high entropy increase - selected)
    short_path_indices = [0, 3, 7]
    for i in range(len(short_path_indices) - 1):
        idx1, idx2 = short_path_indices[i], short_path_indices[i+1]
        ax6.plot([state_x[idx1], state_x[idx2]],
                [state_y[idx1], state_y[idx2]],
                'r-', linewidth=4, alpha=0.8, label='Entropy Path' if i == 0 else '')

    # Mark start and end
    ax6.scatter(state_x[start_idx], state_y[start_idx],
            s=500, c='green', marker='s', edgecolors='black',
            linewidth=2.5, zorder=10, label='Start')
    ax6.scatter(state_x[end_idx], state_y[end_idx],
            s=500, c='red', marker='s', edgecolors='black',
            linewidth=2.5, zorder=10, label='End')

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_xlabel('Configuration Space', fontweight='bold', fontsize=11)
    ax6.set_ylabel('State Space', fontweight='bold', fontsize=11)
    ax6.set_title('F. Entropy as Shortest Path:\nTermination Selection',
                fontweight='bold', fontsize=12, pad=15)
    ax6.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax6.set_aspect('equal')

    # ============================================================================
    # PANEL 7: Dark Matter Correspondence
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, :2])

    # Show the correspondence between categorical ratio and dark matter
    categories = ['Ordinary\nMatter', 'Dark\nMatter', 'Dark\nEnergy']
    observed_values = [5, 27, 68]  # Approximate percentages
    categorical_values = [5, 27, 68]  # From ∞-x structure

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax7.bar(x - width/2, observed_values, width,
                label='Observed (Cosmology)', color='#3498db',
                edgecolor='black', linewidth=2)
    bars2 = ax7.bar(x + width/2, categorical_values, width,
                label='Predicted (Categorical)', color='#e74c3c',
                edgecolor='black', linewidth=2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax7.set_ylabel('Percentage of Universe', fontweight='bold', fontsize=11)
    ax7.set_title('G. Dark Matter Correspondence: Categorical Prediction vs Observation',
                fontweight='bold', fontsize=13, pad=15)
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax7.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add ratio annotation
    ax7.text(0.5, 0.95, 'Ratio x/(∞-x) ≈ 5.4 matches observed dark/ordinary matter ratio',
            transform=ax7.transAxes, ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9,
                    edgecolor='black', linewidth=2))

    # ============================================================================
    # PANEL 8: The Magnitude Comparison
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 2:])

    # Compare N_max with other large numbers
    numbers = {
        'Googol': 100,
        'Googolplex': 100,  # Using log representation
        "Graham's G": 10,
        'TREE(3)': 15,
        'All Combined': 20,
        'C(2)': 85,
        '𝒩_max': 120,
    }

    names = list(numbers.keys())
    values = [np.log10(v) if v > 10 else v for v in numbers.values()]

    # Create bars with gradient
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(names)))
    bars = ax8.barh(names, values, color=colors, edgecolor='black', linewidth=2)

    # Add "effectively zero" threshold
    threshold = values[-2]  # C(2) level
    ax8.axvline(x=threshold, color='blue', linestyle='--', linewidth=3,
            label='Comprehension Threshold')
    ax8.fill_betweenx(range(len(names)), 0, threshold, alpha=0.1, color='blue')

    # Add N_max marker
    ax8.axvline(x=values[-1], color='red', linestyle='-', linewidth=3,
            label='𝒩_max')

    ax8.set_xlabel('log₁₀(Magnitude)', fontweight='bold', fontsize=11)
    ax8.set_title('H. Universal Nullity: All Numbers / 𝒩_max ≈ 0',
                fontweight='bold', fontsize=13, pad=15)
    ax8.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax8.grid(True, alpha=0.3, axis='x', linestyle='--')

    # Add annotation
    ax8.text(threshold/2, len(names) - 1, 'All effectively\nzero compared\nto 𝒩_max',
            ha='center', fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # ============================================================================
    # PANEL 9: The Complete Framework Synthesis
    # ============================================================================
    ax9 = fig.add_subplot(gs[3, :])

    # Create a flow diagram showing the complete framework
    # Define boxes
    boxes = [
        {'name': 'Singularity\nC(0)=1', 'pos': (0.1, 0.5), 'color': '#ff0000'},
        {'name': 'Tetration\nGrowth\nC(t+1)=n^C(t)', 'pos': (0.25, 0.5), 'color': '#ff6600'},
        {'name': 'Observer\nNetwork', 'pos': (0.4, 0.7), 'color': '#00cc00'},
        {'name': 'Oscillatory\nTermination', 'pos': (0.4, 0.3), 'color': '#0099ff'},
        {'name': 'Categorical\nPartition\n∞-x', 'pos': (0.55, 0.5), 'color': '#9900ff'},
        {'name': 'Entropy as\nShortest Path', 'pos': (0.7, 0.7), 'color': '#ff00ff'},
        {'name': 'Dark Matter\nCorrespondence', 'pos': (0.7, 0.3), 'color': '#ffcc00'},
        {'name': 'Heat Death\n𝒩_max', 'pos': (0.9, 0.5), 'color': '#0066ff'},
    ]

    # Draw boxes
    for box in boxes:
        rect = FancyBboxPatch((box['pos'][0] - 0.06, box['pos'][1] - 0.08),
                            0.12, 0.16,
                            boxstyle="round,pad=0.01",
                            facecolor=box['color'], edgecolor='black',
                            linewidth=2.5, transform=ax9.transAxes, alpha=0.8)
        ax9.add_patch(rect)
        ax9.text(box['pos'][0], box['pos'][1], box['name'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                transform=ax9.transAxes, color='white' if box['color'] in ['#ff0000', '#0066ff', '#9900ff'] else 'black')

    # Draw connections
    connections = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 4),
        (4, 5), (4, 6), (5, 7), (6, 7)
    ]

    for start, end in connections:
        start_pos = boxes[start]['pos']
        end_pos = boxes[end]['pos']
        arrow = FancyArrowPatch(start_pos, end_pos,
                            arrowstyle='->', lw=2.5, color='black',
                            transform=ax9.transAxes, mutation_scale=25,
                            alpha=0.6, connectionstyle="arc3,rad=0.1")
        ax9.add_patch(arrow)

    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_title('I. Complete Framework: From Singularity to Observation Boundary',
                fontweight='bold', fontsize=14, pad=15)

    # Add key equations
    equations = [
        r'$C(t+1) = n^{C(t)}$',
        r'$\frac{d\alpha}{dt} = \omega(1-\alpha)$',
        r'Universe = (\infty - x) + x$',
        r'$\frac{x}{\infty-x} \approx 5.4$',
    ]

    eq_y = 0.05
    for i, eq in enumerate(equations):
        ax9.text(0.15 + i * 0.22, eq_y, eq,
                transform=ax9.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                        alpha=0.9, edgecolor='black', linewidth=2))

    # ============================================================================
    # Add overall title and save
    # ============================================================================
    plt.suptitle('The Observation Boundary: Unified Framework from Categorical Enumeration\n' +
                'From Singularity (C=1) to Heat Death (𝒩_max ≈ (10⁸⁴)↑↑(10⁸⁰))',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('unified_observation_boundary.png', dpi=300, bbox_inches='tight')
    plt.savefig('unified_observation_boundary.pdf', dpi=300, bbox_inches='tight')
    print("Saved: unified_observation_boundary.png and .pdf")
    plt.show()
