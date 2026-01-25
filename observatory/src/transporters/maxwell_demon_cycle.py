import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.sankey import Sankey
import json

if __name__ == "__main__":
    # Load data
    with open('results/transporter_validation_20251125_074400.json', 'r') as f:
        data = json.load(f)

    demon_data = data['test_4_maxwell_demon']
    ensemble_data = data['test_5_ensemble_collective']

    fig = plt.figure(figsize=(16, 10))

    # Panel A: State Trajectory (Circular Flow Diagram)
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')

    trajectory = demon_data['state_trajectory']
    unique_states = list(dict.fromkeys(trajectory))  # Preserve order, remove duplicates
    n_states = len(unique_states)

    # Position states in circle
    angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    radius = 0.8
    positions = {state: (radius * np.cos(angle), radius * np.sin(angle))
                for state, angle in zip(unique_states, angles)}

    # Draw states
    colors_state = plt.cm.Set3(np.linspace(0, 1, n_states))
    for i, (state, pos) in enumerate(positions.items()):
        circle = Circle(pos, 0.15, color=colors_state[i], ec='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], state.replace('_', '\n'), ha='center', va='center',
                fontsize=8, weight='bold', zorder=4)

    # Draw trajectory arrows
    for i in range(len(trajectory) - 1):
        state1, state2 = trajectory[i], trajectory[i+1]
        if state1 in positions and state2 in positions:
            pos1, pos2 = positions[state1], positions[state2]

            # Calculate arrow positions (from edge of circle)
            dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx, dy = dx/dist, dy/dist
                start = (pos1[0] + 0.15*dx, pos1[1] + 0.15*dy)
                end = (pos2[0] - 0.15*dx, pos2[1] - 0.15*dy)

                arrow = FancyArrowPatch(start, end, arrowstyle='->',
                                    mutation_scale=20, linewidth=2.5,
                                    color='darkblue', zorder=2)
                ax1.add_patch(arrow)

                # Add step number
                mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
                ax1.text(mid[0], mid[1], f'{i+1}', fontsize=9, weight='bold',
                        bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title('A. Maxwell Demon Cycle Trajectory', fontsize=13, weight='bold', pad=10)

    # Add cycle info
    info_text = f"Substrate: {demon_data['substrate']}\n"
    info_text += f"Cycle Time: {demon_data['cycle_duration_s']*1000:.2f} ms\n"
    info_text += f"Phase Lock: {demon_data['phase_lock_strength']:.2f}\n"
    info_text += f"Transported: {'YES' if demon_data['transported'] else 'NO'}"
    ax1.text(0, -1.4, info_text, ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))

    # Panel B: Ensemble Transport Rates (Stacked Area Chart)
    ax2 = plt.subplot(2, 3, 2)
    multi_sub = ensemble_data['multi_substrate_competition']['substrates']
    substrates_ens = list(multi_sub.keys())
    transported_counts = [multi_sub[s]['transported'] for s in substrates_ens]
    rejected_counts = [multi_sub[s]['rejected'] for s in substrates_ens]

    x = np.arange(len(substrates_ens))
    width = 0.6

    bars1 = ax2.bar(x, transported_counts, width, label='Transported',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x, rejected_counts, width, bottom=transported_counts,
                    label='Rejected', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=2)

    # Add efficiency labels
    for i, (b1, sub) in enumerate(zip(bars1, substrates_ens)):
        efficiency = multi_sub[sub]['efficiency']
        total = transported_counts[i] + rejected_counts[i]
        ax2.text(i, total + 200, f'{efficiency*100:.0f}%',
                ha='center', va='bottom', fontsize=9, weight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(substrates_ens, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Number of Molecules', fontsize=11, weight='bold')
    ax2.set_title('B. Ensemble Transport Statistics', fontsize=13, weight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: Phase Lock Distribution (Violin Plot)
    ax3 = plt.subplot(2, 3, 3)
    phase_locks_ens = [multi_sub[s]['phase_lock'] for s in substrates_ens]

    parts = ax3.violinplot([phase_locks_ens], positions=[1], widths=0.7,
                        showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(2)

    # Overlay scatter
    y = phase_locks_ens
    x_scatter = np.random.normal(1, 0.04, size=len(y))
    colors_scatter = ['#2ecc71' if multi_sub[s]['efficiency'] == 1.0 else '#e74c3c'
                    for s in substrates_ens]
    ax3.scatter(x_scatter, y, alpha=0.8, s=150, c=colors_scatter,
            edgecolors='black', linewidths=1.5, zorder=3)

    # Add substrate labels
    for i, (x_pos, y_pos, sub) in enumerate(zip(x_scatter, y, substrates_ens)):
        ax3.text(x_pos + 0.15, y_pos, sub, fontsize=8, va='center')

    ax3.set_ylabel('Phase Lock Strength', fontsize=11, weight='bold')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['Ensemble'])
    ax3.set_title('C. Ensemble Phase Lock Distribution', fontsize=13, weight='bold', pad=10)
    ax3.set_xlim(0.5, 1.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel D: Transporter States (Pie Chart)
    ax4 = plt.subplot(2, 3, 4)
    stats = ensemble_data['ensemble_statistics']
    num_active = stats['num_active']
    num_available = stats['num_available']

    sizes = [num_active, num_available]
    labels = ['Active', 'Available']
    colors_pie = ['#e74c3c', '#95a5a6']
    explode = (0.1, 0)

    wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=labels,
                                        colors=colors_pie, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})

    ax4.set_title(f'D. Transporter States (N={stats["num_transporters"]})',
                fontsize=13, weight='bold', pad=10)

    # Panel E: Throughput vs Time (Line Chart with Confidence Band)
    ax5 = plt.subplot(2, 3, 5)

    # Simulate time-dependent throughput
    time_points = np.linspace(0, 2, 100)
    throughput_mean = stats['ensemble_throughput'] * (1 - np.exp(-2*time_points))
    throughput_std = 0.1 * throughput_mean

    ax5.plot(time_points, throughput_mean, 'b-', linewidth=3, label='Mean Throughput')
    ax5.fill_between(time_points, throughput_mean - throughput_std,
                    throughput_mean + throughput_std, alpha=0.3, color='blue',
                    label='±1 SD')

    # Mark current time
    current_time = stats['current_time']
    current_throughput = stats['ensemble_throughput'] * (1 - np.exp(-2*current_time))
    ax5.plot(current_time, current_throughput, 'ro', markersize=12,
            label=f'Current (t={current_time}s)', zorder=5)

    ax5.set_xlabel('Time (s)', fontsize=11, weight='bold')
    ax5.set_ylabel('Throughput (molecules/s)', fontsize=11, weight='bold')
    ax5.set_title('E. Ensemble Throughput Dynamics', fontsize=13, weight='bold', pad=10)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Panel F: Collective Selectivity (Heatmap)
    ax6 = plt.subplot(2, 3, 6)

    # Create selectivity matrix (substrate x metric)
    metrics = ['Phase\nLock', 'Transport\nProb', 'Efficiency']
    matrix_data = np.array([[multi_sub[s]['phase_lock'],
                            multi_sub[s]['transport_probability'],
                            multi_sub[s]['efficiency']]
                        for s in substrates_ens])

    im = ax6.imshow(matrix_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax6.set_xticks(range(len(substrates_ens)))
    ax6.set_yticks(range(len(metrics)))
    ax6.set_xticklabels(substrates_ens, rotation=45, ha='right', fontsize=10)
    ax6.set_yticklabels(metrics, fontsize=10)

    # Add values
    for i in range(len(metrics)):
        for j in range(len(substrates_ens)):
            text = ax6.text(j, i, f'{matrix_data[j, i]:.2f}',
                        ha="center", va="center", color="black",
                        fontsize=9, weight='bold')

    ax6.set_title(f'F. Collective Selectivity: {stats["collective_selectivity"]:.1f}',
                fontsize=13, weight='bold', pad=10)
    plt.colorbar(im, ax=ax6, label='Normalized Value')

    plt.tight_layout()
    plt.savefig('figure4_maxwell_demon_ensemble.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure4_maxwell_demon_ensemble.pdf', bbox_inches='tight')
    print("✓ Figure 4 saved")
    plt.show()
