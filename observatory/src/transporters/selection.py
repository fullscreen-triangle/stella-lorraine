import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, FancyBboxPatch
import matplotlib.patches as mpatches

if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 10))

    # Data from console output
    substrates = {
        'Doxorubicin': {'MW': 543.5, 'f0': 3.50e13, 'charge': 1, 'phase_lock': 0.100, 'transported': False},
        'Verapamil': {'MW': 454.6, 'f0': 3.80e13, 'charge': 1, 'phase_lock': 0.910, 'transported': True},
        'Glucose': {'MW': 180.2, 'f0': 2.50e13, 'charge': 0, 'phase_lock': 0.228, 'transported': False},
        'Rhodamine_123': {'MW': 380.8, 'f0': 3.70e13, 'charge': 1, 'phase_lock': 0.250, 'transported': False},
        'Metformin': {'MW': 129.2, 'f0': 2.80e13, 'charge': 2, 'phase_lock': 0.037, 'transported': False}
    }

    efficiency = 0.20
    selectivity = 9.10e9

    # Panel A: Molecular Properties vs Phase Lock
    ax1 = plt.subplot(2, 3, 1)
    names = list(substrates.keys())
    mw = [substrates[s]['MW'] for s in names]
    phase_locks = [substrates[s]['phase_lock'] for s in names]
    transported = [substrates[s]['transported'] for s in names]

    colors = ['#2ecc71' if t else '#e74c3c' for t in transported]
    sizes = [substrates[s]['charge'] * 200 + 100 for s in names]

    for i, (m, pl, c, sz, name) in enumerate(zip(mw, phase_locks, colors, sizes, names)):
        ax1.scatter(m, pl, s=sz, c=c, alpha=0.7, edgecolors='black', linewidths=2)
        ax1.annotate(name, (m, pl), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, weight='bold')

    # Decision boundary
    ax1.axhline(0.5, color='black', linestyle='--', linewidth=2,
            label='Selection Threshold')
    ax1.fill_between([0, 600], 0.5, 1.0, alpha=0.1, color='green',
                    label='Transport Region')
    ax1.fill_between([0, 600], 0, 0.5, alpha=0.1, color='red',
                    label='Reject Region')

    ax1.set_xlabel('Molecular Weight (Da)', fontsize=11, weight='bold')
    ax1.set_ylabel('Phase Lock Strength ⟨r⟩', fontsize=11, weight='bold')
    ax1.set_title('A. Molecular Weight vs Phase Lock', fontsize=13, weight='bold', pad=10)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 600)
    ax1.set_ylim(0, 1.0)

    # Panel B: Frequency Space Distribution
    ax2 = plt.subplot(2, 3, 2)
    frequencies = [substrates[s]['f0']/1e13 for s in names]

    # Histogram with KDE overlay
    ax2.hist(frequencies, bins=10, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=2, label='Frequency Distribution')

    # Mark each substrate
    for i, (f, pl, name, t) in enumerate(zip(frequencies, phase_locks, names, transported)):
        color = '#2ecc71' if t else '#e74c3c'
        marker = 'o' if t else 'x'
        ax2.plot(f, 0.5 + pl, marker, markersize=12, color=color,
                markeredgecolor='black', markeredgewidth=2)
        ax2.text(f, 0.5 + pl + 0.1, name, fontsize=7, rotation=45, ha='left')

    ax2.set_xlabel('Natural Frequency (×10¹³ Hz)', fontsize=11, weight='bold')
    ax2.set_ylabel('Count / Phase Lock', fontsize=11, weight='bold')
    ax2.set_title('B. Frequency Space Distribution', fontsize=13, weight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x')

    # Panel C: Charge State Analysis
    ax3 = plt.subplot(2, 3, 3)
    charges = [substrates[s]['charge'] for s in names]
    unique_charges = sorted(set(charges))

    charge_data = {c: {'phase_locks': [], 'transported': []} for c in unique_charges}
    for name in names:
        c = substrates[name]['charge']
        charge_data[c]['phase_locks'].append(substrates[name]['phase_lock'])
        charge_data[c]['transported'].append(substrates[name]['transported'])

    # Box plot
    positions = unique_charges
    bp = ax3.boxplot([charge_data[c]['phase_locks'] for c in unique_charges],
                    positions=positions, widths=0.3, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red'))

    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)

    # Overlay scatter
    for c in unique_charges:
        y = charge_data[c]['phase_locks']
        t = charge_data[c]['transported']
        x = np.random.normal(c, 0.05, size=len(y))
        colors_scatter = ['#2ecc71' if transported else '#e74c3c' for transported in t]
        ax3.scatter(x, y, c=colors_scatter, s=100, alpha=0.8,
                edgecolors='black', linewidths=1.5, zorder=3)

    ax3.set_xlabel('Charge State', fontsize=11, weight='bold')
    ax3.set_ylabel('Phase Lock Strength ⟨r⟩', fontsize=11, weight='bold')
    ax3.set_title('C. Charge-Dependent Phase Locking', fontsize=13, weight='bold', pad=10)
    ax3.set_xticks(unique_charges)
    ax3.set_xticklabels([f'+{c}' if c > 0 else str(c) for c in unique_charges])
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Selection Decision Tree
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    # Root node
    root = FancyBboxPatch((3, 8), 4, 1, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor='lightblue', linewidth=2)
    ax4.add_patch(root)
    ax4.text(5, 8.5, 'Substrate Binding', ha='center', va='center',
            fontsize=10, weight='bold')

    # Decision node
    decision = FancyBboxPatch((3, 5.5), 4, 1, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax4.add_patch(decision)
    ax4.text(5, 6, 'Phase Lock > 0.5?', ha='center', va='center',
            fontsize=10, weight='bold')

    # Outcome nodes
    transport = FancyBboxPatch((0.5, 3), 3, 1, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax4.add_patch(transport)
    ax4.text(2, 3.5, 'TRANSPORT\n(Verapamil)', ha='center', va='center',
            fontsize=9, weight='bold')

    reject = FancyBboxPatch((6.5, 3), 3, 1, boxstyle="round,pad=0.1",
                        edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax4.add_patch(reject)
    ax4.text(8, 3.5, 'REJECT\n(4 substrates)', ha='center', va='center',
            fontsize=9, weight='bold')

    # Arrows
    ax4.annotate('', xy=(5, 8), xytext=(5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax4.annotate('', xy=(2, 4), xytext=(4, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax4.annotate('', xy=(8, 4), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax4.text(3.5, 4.7, 'YES', fontsize=9, weight='bold', color='green')
    ax4.text(6.5, 4.7, 'NO', fontsize=9, weight='bold', color='red')

    # Statistics
    stats_text = f"Efficiency: {efficiency*100:.0f}%\n"
    stats_text += f"Selectivity: {selectivity:.2e}"
    ax4.text(5, 1, stats_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax4.set_title('D. Selection Decision Tree', fontsize=13, weight='bold', pad=10)

    # Panel E: Phase Lock Ranking
    ax5 = plt.subplot(2, 3, 5)
    sorted_indices = np.argsort(phase_locks)[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_locks = [phase_locks[i] for i in sorted_indices]
    sorted_transported = [transported[i] for i in sorted_indices]

    colors_rank = ['#2ecc71' if t else '#e74c3c' for t in sorted_transported]
    y_pos = np.arange(len(sorted_names))

    bars = ax5.barh(y_pos, sorted_locks, color=colors_rank, alpha=0.8,
                edgecolor='black', linewidth=2)

    # Add values and symbols
    for i, (bar, val, t) in enumerate(zip(bars, sorted_locks, sorted_transported)):
        symbol = '✓' if t else '✗'
        ax5.text(val + 0.02, i, f'{val:.3f} {symbol}', va='center',
                fontsize=10, weight='bold')

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(sorted_names, fontsize=10)
    ax5.set_xlabel('Phase Lock Strength ⟨r⟩', fontsize=11, weight='bold')
    ax5.set_title('E. Substrate Ranking', fontsize=13, weight='bold', pad=10)
    ax5.axvline(0.5, color='black', linestyle='--', linewidth=2)
    ax5.set_xlim(0, 1.0)
    ax5.grid(True, alpha=0.3, axis='x')

    # Panel F: Selectivity Metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Create metrics table
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Total Substrates', '5', ''],
        ['Transported', '1', '✓'],
        ['Rejected', '4', '✗'],
        ['Efficiency', f'{efficiency*100:.1f}%', ''],
        ['Selectivity', f'{selectivity:.2e}', '✓✓'],
        ['Avg Phase Lock (Transport)', '0.910', '✓'],
        ['Avg Phase Lock (Reject)', '0.154', '✗'],
    ]

    table = ax6.table(cellText=metrics_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(metrics_data)):
        for j in range(3):
            cell = table[(i, j)]
            if j == 2:  # Status column
                if metrics_data[i][2] == '✓' or metrics_data[i][2] == '✓✓':
                    cell.set_facecolor('#d5f4e6')
                    cell.set_text_props(weight='bold', color='green')
                elif metrics_data[i][2] == '✗':
                    cell.set_facecolor('#f4d5d5')
                    cell.set_text_props(weight='bold', color='red')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

    ax6.set_title('F. Selection Metrics Summary', fontsize=13, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figure6_phase_locked_selection_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure6_phase_locked_selection_detailed.pdf', bbox_inches='tight')
    print("✓ Figure 6 saved")
    plt.show()
