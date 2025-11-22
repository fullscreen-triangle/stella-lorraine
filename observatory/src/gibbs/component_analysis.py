"""
Comparative Component Analysis
4-panel cross-component comparison visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)

def create_comparative_analysis(data):
    """Create 4-panel comparative analysis visualization"""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Cross-Component Comparative Analysis\n' +
                 f'Simulation Module: {data["module"]}',
                 fontsize=16, fontweight='bold')

    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    components = data['components_tested']

    # ============================================================
    # PANEL 1: Item Count Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Extract item counts for successful components
    item_data = []
    for comp in components:
        if comp['status'] == 'success' and 'tests' in comp:
            if 'item_count' in comp['tests']:
                item_data.append({
                    'name': comp['component'],
                    'count': comp['tests']['item_count']
                })

    if item_data:
        names = [d['name'] for d in item_data]
        counts = [d['count'] for d in item_data]

        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))

        bars = ax1.bar(names, counts, color=colors,
                      edgecolor='black', linewidth=2)

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax1.set_ylabel('Available Items', fontweight='bold')
        ax1.set_title('Component API Surface Area\n(Available Items per Component)',
                     fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, axis='y', alpha=0.3)

        # Add mean line
        mean_count = np.mean(counts)
        ax1.axhline(mean_count, color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_count:.1f}')
        ax1.legend()

    # ============================================================
    # PANEL 2: Molecular Properties Radar Chart
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')

    # Get molecular data
    molecule_comp = next((c for c in components if c['component'] == 'Molecule'), None)

    if molecule_comp and molecule_comp['status'] == 'success':
        tests = molecule_comp['tests']

        # Normalize metrics for radar chart
        metrics = {
            'Frequency\n(norm)': tests['vibrational_frequency_Hz'] / 1e14,
            'Period\n(norm)': tests['vibrational_period_fs'] / 20,
            'Precision\n(norm)': tests['clock_precision_fs'] / 20,
            'Q-factor\n(norm)': tests['Q_factor'] / 50000,
            'Energy\nLevels': len(tests['energy_levels']) / 15,
            'Ensemble\nSize': tests['ensemble_size'] / 150
        }

        # Setup radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Number of variables
        N = len(categories)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        # Plot
        ax2.plot(angles, values, 'o-', linewidth=2, color='#3498db', label='Normalized Value')
        ax2.fill(angles, values, alpha=0.25, color='#3498db')

        # Fix axis to go in the right order
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title('Molecular Clock Properties\n(Normalized Radar Chart)',
                     fontweight='bold', pad=20)
        ax2.grid(True)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # ============================================================
    # PANEL 3: Component Complexity Matrix
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Create complexity matrix
    complexity_metrics = []
    component_labels = []

    for comp in components:
        if comp['status'] == 'success' and 'tests' in comp:
            component_labels.append(comp['component'])

            # Calculate complexity score
            tests = comp['tests']

            # Different metrics for different components
            if 'item_count' in tests:
                complexity = tests['item_count']
            elif 'energy_levels' in tests:
                complexity = len(tests['energy_levels']) * 2
            else:
                complexity = len(tests)

            complexity_metrics.append(complexity)

    if complexity_metrics:
        # Create matrix (for visualization, we'll create a correlation-like matrix)
        n_comp = len(component_labels)
        matrix = np.zeros((n_comp, n_comp))

        for i in range(n_comp):
            for j in range(n_comp):
                # Similarity based on complexity difference
                diff = abs(complexity_metrics[i] - complexity_metrics[j])
                max_diff = max(complexity_metrics)
                similarity = 1 - (diff / max_diff)
                matrix[i, j] = similarity

        # Plot heatmap
        im = ax3.imshow(matrix, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Complexity Similarity', fontweight='bold')

        # Set ticks
        ax3.set_xticks(np.arange(n_comp))
        ax3.set_yticks(np.arange(n_comp))
        ax3.set_xticklabels(component_labels, rotation=45, ha='right')
        ax3.set_yticklabels(component_labels)

        # Add values in cells
        for i in range(n_comp):
            for j in range(n_comp):
                text = ax3.text(j, i, f'{matrix[i, j]:.2f}',
                              ha="center", va="center",
                              color="black" if matrix[i, j] > 0.5 else "white",
                              fontsize=9, fontweight='bold')

        ax3.set_title('Component Complexity Similarity Matrix\n' +
                     '(1.0 = identical complexity, 0.0 = maximally different)',
                     fontweight='bold')

    # ============================================================
    # PANEL 4: Success Rate & Testing Depth
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate metrics
    total_components = len(components)
    successful = sum(1 for c in components if c['status'] == 'success')
    failed = sum(1 for c in components if c['status'] == 'failed')

    # Testing depth (average number of tests per component)
    test_depths = []
    for comp in components:
        if comp['status'] == 'success' and 'tests' in comp:
            test_depths.append(len(comp['tests']))

    avg_test_depth = np.mean(test_depths) if test_depths else 0

    # Create grouped bar chart
    categories = ['Success\nRate (%)', 'Failure\nRate (%)', 'Avg Test\nDepth']
    values = [
        (successful / total_components) * 100,
        (failed / total_components) * 100,
        avg_test_depth
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    bars = ax4.bar(categories, values, color=colors,
                   edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax4.set_ylabel('Value', fontweight='bold')
    ax4.set_title('Testing Quality Metrics', fontweight='bold', fontsize=14)
    ax4.grid(True, axis='y', alpha=0.3)

    # Add reference lines
    ax4.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax4.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Add summary text
    summary_text = f"Total Components: {total_components}\n"
    summary_text += f"Successful: {successful}\n"
    summary_text += f"Failed: {failed}\n"
    summary_text += f"Avg Tests/Component: {avg_test_depth:.1f}"

    ax4.text(0.98, 0.98, summary_text,
            transform=ax4.transAxes, ha='right', va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    return fig

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    with open('simulation_data.json', 'r') as f:
        data = json.load(f)

    fig = create_comparative_analysis(data)

    output_file = f"comparative_analysis_{data['timestamp']}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    plt.show()
