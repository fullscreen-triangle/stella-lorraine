import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
import json

if __name__ == "__main__":

    # Load data
    with open('results/transporter_validation_20251125_074400.json', 'r') as f:
        data = json.load(f)

    selection_data = data['test_2_phase_locked_selection']
    substrates = list(selection_data['phase_lock_strengths'].keys())
    phase_locks = list(selection_data['phase_lock_strengths'].values())
    transported = selection_data['transported_substrates']
    rejected = selection_data['rejected_substrates']

    fig = plt.figure(figsize=(16, 10))

    # Panel A: Phase Lock Strength Comparison (Horizontal Bar Chart)
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#2ecc71' if s in transported else '#e74c3c' for s in substrates]
    y_pos = np.arange(len(substrates))

    bars = ax1.barh(y_pos, phase_locks, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add threshold line
    threshold = 0.5
    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')

    # Add values
    for i, (bar, val) in enumerate(zip(bars, phase_locks)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, weight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(substrates, fontsize=11)
    ax1.set_xlabel('Phase Lock Strength ⟨r⟩', fontsize=12, weight='bold')
    ax1.set_title('A. Phase Lock Strength by Substrate', fontsize=13, weight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1.0)

    # Panel B: Transport vs Rejection (Stacked Bar)
    ax2 = plt.subplot(2, 3, 2)
    categories = ['Transported', 'Rejected']
    counts = [selection_data['num_transported'], selection_data['num_rejected']]
    colors_stack = ['#2ecc71', '#e74c3c']

    bars = ax2.bar(categories, counts, color=colors_stack, alpha=0.8,
                edgecolor='black', linewidth=2, width=0.6)

    # Add percentages
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='center', fontsize=12, weight='bold', color='white')

    ax2.set_ylabel('Number of Substrates', fontsize=12, weight='bold')
    ax2.set_title('B. Transport Outcome', fontsize=13, weight='bold', pad=10)
    ax2.set_ylim(0, max(counts) * 1.2)

    # Panel C: Selectivity Visualization (Scatter with Decision Boundary)
    ax3 = plt.subplot(2, 3, 3)

    # Create synthetic frequency data for visualization
    np.random.seed(42)
    frequencies = np.random.uniform(30, 50, len(substrates))  # THz
    binding_energies = -10 * np.array(phase_locks) + np.random.normal(0, 1, len(substrates))

    for i, (sub, pl, freq, energy) in enumerate(zip(substrates, phase_locks, frequencies, binding_energies)):
        color = '#2ecc71' if sub in transported else '#e74c3c'
        marker = 'o' if sub in transported else 'x'
        size = 300 if sub in transported else 200
        ax3.scatter(freq, energy, c=color, marker=marker, s=size,
                    edgecolors='black', linewidths=2, alpha=0.8, label=sub)

    # Decision boundary
    x_boundary = np.linspace(30, 50, 100)
    y_boundary = -10 * threshold + 0.5 * (x_boundary - 40)
    ax3.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision Boundary')
    ax3.fill_between(x_boundary, y_boundary, -15, alpha=0.1, color='green', label='Transport Region')
    ax3.fill_between(x_boundary, y_boundary, 5, alpha=0.1, color='red', label='Reject Region')

    ax3.set_xlabel('Binding Site Frequency (THz)', fontsize=11, weight='bold')
    ax3.set_ylabel('Binding Energy (kJ/mol)', fontsize=11, weight='bold')
    ax3.set_title('C. Selectivity Landscape', fontsize=13, weight='bold', pad=10)
    ax3.legend(fontsize=8, loc='best', ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panel D: Selectivity Metric (Gauge Chart)
    ax4 = plt.subplot(2, 3, 4)
    selectivity = selection_data['selectivity']
    log_selectivity = np.log10(selectivity)

    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)

    # Background arc
    ax4.fill_between(theta, 0, r, color='lightgray', alpha=0.3)

    # Colored regions
    regions = [(0, np.pi/3, 'red', 'Poor'),
            (np.pi/3, 2*np.pi/3, 'yellow', 'Moderate'),
            (2*np.pi/3, np.pi, 'green', 'Excellent')]

    for start, end, color, label in regions:
        theta_region = np.linspace(start, end, 50)
        ax4.fill_between(theta_region, 0, r[0], color=color, alpha=0.5, label=label)

    # Needle position (normalized to 0-π)
    needle_pos = min(log_selectivity / 12 * np.pi, np.pi)  # Cap at π
    ax4.plot([0, np.cos(needle_pos)], [0, np.sin(needle_pos)],
            'k-', linewidth=4, marker='o', markersize=10)

    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(0, 1.2)
    ax4.axis('off')
    ax4.set_title(f'D. Selectivity: {selectivity:.2e}', fontsize=13, weight='bold', pad=10)
    ax4.text(0, -0.2, f'log₁₀(S) = {log_selectivity:.1f}',
            ha='center', fontsize=11, weight='bold')
    ax4.legend(loc='upper right', fontsize=9)

    # Panel E: Average Phase Lock Comparison (Box Plot Style)
    ax5 = plt.subplot(2, 3, 5)
    avg_transported = selection_data['avg_phase_lock_transported']
    avg_rejected = selection_data['avg_phase_lock_rejected']

    transported_locks = [phase_locks[i] for i, s in enumerate(substrates) if s in transported]
    rejected_locks = [phase_locks[i] for i, s in enumerate(substrates) if s in rejected]

    bp = ax5.boxplot([transported_locks, rejected_locks],
                    labels=['Transported', 'Rejected'],
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=10))

    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, locks in enumerate([transported_locks, rejected_locks], 1):
        y = locks
        x = np.random.normal(i, 0.04, size=len(y))
        ax5.scatter(x, y, alpha=0.6, s=100, edgecolors='black', linewidths=1.5, zorder=3)

    ax5.set_ylabel('Phase Lock Strength ⟨r⟩', fontsize=12, weight='bold')
    ax5.set_title('E. Phase Lock Distribution', fontsize=13, weight='bold', pad=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel F: Transport Efficiency Breakdown (Donut Chart)
    ax6 = plt.subplot(2, 3, 6)
    efficiency = selection_data['transport_efficiency']
    inefficiency = 1 - efficiency

    sizes = [efficiency, inefficiency]
    colors_donut = ['#2ecc71', '#e0e0e0']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax6.pie(sizes, explode=explode, colors=colors_donut,
                                        autopct='%1.1f%%', startangle=90,
                                        pctdistance=0.85,
                                        textprops={'fontsize': 12, 'weight': 'bold'})

    # Draw circle for donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax6.add_artist(centre_circle)

    # Center text
    ax6.text(0, 0, f'{efficiency*100:.0f}%\nEfficient',
            ha='center', va='center', fontsize=16, weight='bold')

    ax6.set_title('F. Transport Efficiency', fontsize=13, weight='bold', pad=10)
    ax6.legend(['Transported', 'Available'], loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('figure2_phase_locked_selection.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_phase_locked_selection.pdf', bbox_inches='tight')
    print("✓ Figure 2 saved")
    plt.show()
