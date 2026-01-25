import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.sankey import Sankey
import matplotlib.patches as mpatches

if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 10))

    # Data from console output
    states = {
        'OPEN_OUTSIDE': {'volume': 5000, 'freq': 3.80e13, 'S': [0.10, 0.00, 1.00], 'energy': 0.0, 'ATP': True},
        'OCCLUDED': {'volume': 3000, 'freq': 4.50e13, 'S': [0.90, 0.25, 0.50], 'energy': 15.0, 'ATP': True},
        'OPEN_INSIDE': {'volume': 4500, 'freq': 3.20e13, 'S': [0.20, 0.50, 0.30], 'energy': -10.0, 'ATP': False},
        'RESETTING': {'volume': 4000, 'freq': 3.50e13, 'S': [0.05, 0.75, 0.80], 'energy': 5.0, 'ATP': False}
    }

    transitions = {
        ('OPEN_OUTSIDE', 'OCCLUDED'): {'empty': 2.96e3, 'bound': 1.44e5, 'enhancement': 48.5},
        ('OCCLUDED', 'OPEN_INSIDE'): {'empty': 1.87e15, 'bound': 1.87e15, 'enhancement': 1.0},
        ('OPEN_INSIDE', 'RESETTING'): {'empty': 2.96e3, 'bound': 2.96e3, 'enhancement': 1.0},
        ('RESETTING', 'OPEN_OUTSIDE'): {'empty': 6.96e6, 'bound': 6.96e6, 'enhancement': 1.0}
    }

    # Panel A: Transition Rate Comparison (Log Scale)
    ax1 = plt.subplot(2, 3, 1)
    transition_names = [f"{s1.split('_')[0]}\n→\n{s2.split('_')[0]}"
                        for s1, s2 in transitions.keys()]
    empty_rates = [transitions[t]['empty'] for t in transitions.keys()]
    bound_rates = [transitions[t]['bound'] for t in transitions.keys()]

    x = np.arange(len(transition_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, empty_rates, width, label='Empty',
                    color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x + width/2, bound_rates, width, label='Substrate-Bound',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)

    ax1.set_ylabel('Transition Rate (s⁻¹, log scale)', fontsize=11, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(transition_names, fontsize=9)
    ax1.set_yscale('log')
    ax1.set_ylim(1e2, 1e16)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_title('A. Transition Rate Enhancement', fontsize=13, weight='bold', pad=10)
    ax1.grid(True, alpha=0.3, which='both', axis='y')

    # Add enhancement factors
    for i, (bars, trans) in enumerate(zip([bars1, bars2], transitions.keys())):
        enhancement = transitions[trans]['enhancement']
        if enhancement > 1.5:
            ax1.text(i, max(empty_rates[i], bound_rates[i]) * 2,
                    f'{enhancement:.1f}×', ha='center', va='bottom',
                    fontsize=9, weight='bold', color='red')

    # Panel B: Energy Landscape with Barriers
    ax2 = plt.subplot(2, 3, 2)
    state_order = ['OPEN_OUTSIDE', 'OCCLUDED', 'OPEN_INSIDE', 'RESETTING', 'OPEN_OUTSIDE']
    energies = [states[s]['energy'] for s in state_order[:-1]] + [states[state_order[0]]['energy']]
    x_pos = np.arange(len(energies))

    # Calculate barrier heights (simplified)
    barriers = []
    for i in range(len(energies)-1):
        barrier = max(energies[i], energies[i+1]) + 20  # Add activation energy
        barriers.append(barrier)

    # Plot energy levels
    ax2.plot(x_pos, energies, 'o-', linewidth=3, markersize=12,
            color='#3498db', label='Ground State')

    # Plot barriers
    for i in range(len(barriers)):
        x_barrier = x_pos[i] + 0.5
        ax2.plot([x_pos[i], x_barrier, x_pos[i+1]],
                [energies[i], barriers[i], energies[i+1]],
                '--', linewidth=2, color='#e74c3c', alpha=0.6)
        ax2.plot(x_barrier, barriers[i], '^', markersize=10, color='#e74c3c')

    # Fill regions
    for i in range(len(energies)-1):
        ax2.fill_between([x_pos[i], x_pos[i+1]],
                        [energies[i], energies[i+1]],
                        min(energies) - 5, alpha=0.1, color='blue')

    ax2.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xticks(x_pos[:-1])
    ax2.set_xticklabels([s.replace('_', '\n') for s in state_order[:-1]],
                        fontsize=9, rotation=0)
    ax2.set_ylabel('Free Energy (kJ/mol)', fontsize=11, weight='bold')
    ax2.set_title('B. Free Energy Landscape', fontsize=13, weight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-0.5, len(energies)-0.5)

    # Panel C: Cavity Volume & Frequency Correlation
    ax3 = plt.subplot(2, 3, 3)
    volumes = [states[s]['volume'] for s in states.keys()]
    frequencies = [states[s]['freq']/1e13 for s in states.keys()]
    energies_scatter = [states[s]['energy'] for s in states.keys()]
    atp_status = [states[s]['ATP'] for s in states.keys()]

    # Scatter plot
    colors_scatter = ['#e74c3c' if atp else '#2ecc71' for atp in atp_status]
    sizes = [300 + abs(e)*20 for e in energies_scatter]

    for i, (v, f, s, c, name) in enumerate(zip(volumes, frequencies, sizes,
                                                colors_scatter, states.keys())):
        ax3.scatter(v, f, s=s, c=c, alpha=0.7, edgecolors='black', linewidths=2)
        ax3.annotate(name.replace('_', '\n'), (v, f),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.8))

    # Trend line
    z = np.polyfit(volumes, frequencies, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(volumes), max(volumes), 100)
    ax3.plot(x_trend, p(x_trend), '--', linewidth=2, color='gray', alpha=0.5)

    ax3.set_xlabel('Cavity Volume (Ų)', fontsize=11, weight='bold')
    ax3.set_ylabel('Binding Frequency (×10¹³ Hz)', fontsize=11, weight='bold')
    ax3.set_title('C. Volume-Frequency Relationship', fontsize=13, weight='bold', pad=10)
    ax3.grid(True, alpha=0.3)

    # Legend
    atp_patch = mpatches.Patch(color='#e74c3c', label='ATP Bound', alpha=0.7)
    adp_patch = mpatches.Patch(color='#2ecc71', label='ATP Free', alpha=0.7)
    ax3.legend(handles=[atp_patch, adp_patch], fontsize=10, loc='best')

    # Panel D: S-Space Distance Matrix
    ax4 = plt.subplot(2, 3, 4)
    state_names = list(states.keys())
    n = len(state_names)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            s1 = np.array(states[state_names[i]]['S'])
            s2 = np.array(states[state_names[j]]['S'])
            distance_matrix[i, j] = np.sqrt(np.sum((s1 - s2)**2))

    im = ax4.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(n))
    ax4.set_yticks(range(n))
    ax4.set_xticklabels([s.replace('_', '\n') for s in state_names],
                        fontsize=9, rotation=45, ha='right')
    ax4.set_yticklabels([s.replace('_', '\n') for s in state_names], fontsize=9)

    # Add values
    for i in range(n):
        for j in range(n):
            text = ax4.text(j, i, f'{distance_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black" if distance_matrix[i, j] < 1 else "white",
                        fontsize=9, weight='bold')

    ax4.set_title('D. S-Space Distance Matrix', fontsize=13, weight='bold', pad=10)
    plt.colorbar(im, ax=ax4, label='Categorical Distance')

    # Panel E: Enhancement Factor Analysis
    ax5 = plt.subplot(2, 3, 5)
    enhancements = [transitions[t]['enhancement'] for t in transitions.keys()]
    colors_enh = ['#e74c3c' if e > 10 else '#95a5a6' for e in enhancements]

    bars = ax5.barh(range(len(transition_names)), enhancements,
                    color=colors_enh, alpha=0.8, edgecolor='black', linewidth=2)

    # Threshold line
    ax5.axvline(10, color='black', linestyle='--', linewidth=2,
            label='High Enhancement (>10×)')

    # Add values
    for i, (bar, val) in enumerate(zip(bars, enhancements)):
        ax5.text(val + 1, i, f'{val:.1f}×', va='center', fontsize=10, weight='bold')

    ax5.set_yticks(range(len(transition_names)))
    ax5.set_yticklabels(transition_names, fontsize=9)
    ax5.set_xlabel('Enhancement Factor', fontsize=11, weight='bold')
    ax5.set_title('E. Substrate Binding Enhancement', fontsize=13, weight='bold', pad=10)
    ax5.legend(fontsize=10)
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3, axis='x')

    # Panel F: Cycle Trajectory Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Summary statistics
    summary_text = "CONFORMATIONAL CYCLE SUMMARY\n" + "="*50 + "\n\n"
    summary_text += f"Total States: {len(states)}\n"
    summary_text += f"Trajectory Points: 20\n"
    summary_text += f"S-Space Distance: 14.73\n\n"

    summary_text += "STATE PROPERTIES:\n" + "-"*50 + "\n"
    for name, props in states.items():
        summary_text += f"\n{name}:\n"
        summary_text += f"  Volume: {props['volume']:.0f} Ų\n"
        summary_text += f"  Frequency: {props['freq']:.2e} Hz\n"
        summary_text += f"  Energy: {props['energy']:+.1f} kJ/mol\n"
        summary_text += f"  ATP: {'Bound' if props['ATP'] else 'Free'}\n"

    summary_text += "\n" + "="*50 + "\n"
    summary_text += "KEY FINDINGS:\n"
    summary_text += "• Substrate binding enhances rate 48.5×\n"
    summary_text += "• Occluded→Inside transition fastest (1.87e15 s⁻¹)\n"
    summary_text += "• Energy minimum at OPEN_INSIDE (-10 kJ/mol)\n"
    summary_text += "• Maximum barrier at OCCLUDED (+15 kJ/mol)\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('figure5_conformational_transitions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure5_conformational_transitions.pdf', bbox_inches='tight')
    print("✓ Figure 5 saved")
    plt.show()
