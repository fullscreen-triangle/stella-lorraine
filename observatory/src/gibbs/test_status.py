"""
Component Status Dashboard
4-panel system architecture and testing status visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

def create_component_status_dashboard(data):
    """Create 4-panel component status visualization"""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Simulation Component Status Dashboard\n' +
                 f'Module: {data["module"]} | Timestamp: {data["timestamp"]}',
                 fontsize=16, fontweight='bold')

    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ============================================================
    # PANEL 1: Component Status Overview
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    components = data['components_tested']
    component_names = [c['component'] for c in components]
    statuses = [c['status'] for c in components]

    # Color mapping
    status_colors = {'success': '#2ecc71', 'failed': '#e74c3c'}
    colors = [status_colors[s] for s in statuses]

    # Horizontal bar chart
    y_pos = np.arange(len(component_names))
    bars = ax1.barh(y_pos, [1]*len(component_names), color=colors,
                    edgecolor='black', linewidth=2, height=0.7)

    # Add status labels
    for i, (bar, status) in enumerate(zip(bars, statuses)):
        ax1.text(0.5, bar.get_y() + bar.get_height()/2,
                status.upper(),
                ha='center', va='center', fontweight='bold',
                fontsize=12, color='white')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(component_names, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_xticks([])
    ax1.set_title('Component Testing Status', fontweight='bold', fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # Add legend
    success_patch = mpatches.Patch(color='#2ecc71', label='Success')
    failed_patch = mpatches.Patch(color='#e74c3c', label='Failed')
    ax1.legend(handles=[success_patch, failed_patch], loc='lower right')

    # ============================================================
    # PANEL 2: Test Coverage Analysis
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Count tests per component
    test_counts = []
    for comp in components:
        if comp['status'] == 'success' and 'tests' in comp:
            # Count number of test parameters
            test_counts.append(len(comp['tests']))
        else:
            test_counts.append(0)

    # Pie chart of test distribution
    explode = [0.05 if count == 0 else 0 for count in test_counts]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(component_names)))

    wedges, texts, autotexts = ax2.pie(test_counts, labels=component_names,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors_pie, explode=explode,
                                        textprops={'fontweight': 'bold'})

    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)

    ax2.set_title(f'Test Coverage Distribution\nTotal Tests: {sum(test_counts)}',
                  fontweight='bold', fontsize=14)

    # ============================================================
    # PANEL 3: Component Dependency Graph
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Component Architecture\n(Hierarchical Dependencies)',
                  fontweight='bold', fontsize=14)

    # Define component positions (hierarchical layout)
    positions = {
        'Molecule': (5, 9),
        'GasChamber': (2.5, 7),
        'Observer': (5, 7),
        'Wave': (7.5, 7),
        'Alignment': (2.5, 5),
        'Propagation': (5, 5),
        'Transcendent': (7.5, 5)
    }

    # Draw components as boxes
    for comp in components:
        name = comp['component']
        if name in positions:
            x, y = positions[name]
            color = status_colors[comp['status']]

            # Draw fancy box
            box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='black',
                                linewidth=2, alpha=0.7)
            ax3.add_patch(box)

            # Add label
            ax3.text(x, y, name, ha='center', va='center',
                    fontweight='bold', fontsize=9, color='white')

    # Draw dependency arrows
    dependencies = [
        ('Molecule', 'GasChamber'),
        ('Molecule', 'Observer'),
        ('Molecule', 'Wave'),
        ('Observer', 'Alignment'),
        ('Wave', 'Propagation'),
        ('Observer', 'Transcendent'),
        ('Propagation', 'Transcendent')
    ]

    for source, target in dependencies:
        if source in positions and target in positions:
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            ax3.annotate('', xy=(x2, y2+0.3), xytext=(x1, y1-0.3),
                        arrowprops=dict(arrowstyle='->', lw=2,
                                      color='gray', alpha=0.6))

    # ============================================================
    # PANEL 4: Error Analysis
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Collect error information
    success_count = sum(1 for c in components if c['status'] == 'success')
    failed_count = sum(1 for c in components if c['status'] == 'failed')

    # Create stacked bar chart
    categories = ['Total\nComponents', 'Successful\nTests', 'Failed\nTests']
    values = [len(components), success_count, failed_count]
    colors_bar = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax4.bar(categories, values, color=colors_bar,
                   edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}',
                ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Testing Summary Statistics', fontweight='bold', fontsize=14)
    ax4.grid(True, axis='y', alpha=0.3)

    # Add success rate annotation
    success_rate = (success_count / len(components)) * 100
    ax4.text(0.5, 0.95, f'Success Rate: {success_rate:.1f}%',
            transform=ax4.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add failed component details
    failed_comps = [c for c in components if c['status'] == 'failed']
    if failed_comps:
        error_text = "Failed Components:\n"
        for comp in failed_comps:
            error_text += f"• {comp['component']}: {comp.get('error', 'Unknown error')}\n"

        ax4.text(0.02, 0.02, error_text,
                transform=ax4.transAxes, ha='left', va='bottom',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8))

    plt.tight_layout()
    return fig

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Load data (save your JSON as 'simulation_data.json')
    with open('simulation_data.json', 'r') as f:
        data = json.load(f)

    # Create visualization
    fig = create_component_status_dashboard(data)

    # Save
    output_file = f"component_status_dashboard_{data['timestamp']}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    plt.show()
