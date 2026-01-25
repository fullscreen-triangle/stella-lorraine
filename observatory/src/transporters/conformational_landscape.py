import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import json


if __name__ == "__main__":
    # Load data
    with open('results/transporter_validation_20251125_074400.json', 'r') as f:
        data = json.load(f)

    states = data['test_1_conformational_landscape']['states']

    fig = plt.figure(figsize=(16, 10))

    # Panel A: Free Energy Landscape (2D contour)
    ax1 = plt.subplot(2, 3, 1)
    state_names = list(states.keys())
    energies = [states[s]['free_energy'] for s in state_names]
    volumes = [states[s]['cavity_volume'] for s in state_names]
    frequencies = [states[s]['binding_site_frequency']/1e12 for s in state_names]

    # Create energy surface
    x = np.linspace(min(volumes)-500, max(volumes)+500, 100)
    y = np.linspace(min(frequencies)-5, max(frequencies)+5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i, state in enumerate(state_names):
        v, f, e = volumes[i], frequencies[i], energies[i]
        Z += e * np.exp(-((X-v)**2/500**2 + (Y-f)**2/5**2))

    contour = ax1.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax1.scatter(volumes, frequencies, c=energies, s=300, cmap='RdYlBu_r',
                edgecolors='black', linewidths=2, zorder=5)

    for i, name in enumerate(state_names):
        ax1.annotate(name.replace('_', '\n'), (volumes[i], frequencies[i]),
                    fontsize=8, ha='center', va='center', weight='bold')

    ax1.set_xlabel('Cavity Volume (Å³)', fontsize=11, weight='bold')
    ax1.set_ylabel('Binding Frequency (THz)', fontsize=11, weight='bold')
    ax1.set_title('A. Free Energy Landscape', fontsize=13, weight='bold', pad=10)
    plt.colorbar(contour, ax=ax1, label='Free Energy (kJ/mol)')

    # Panel B: 3D S-Space Trajectory
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    s_coords = np.array([[states[s]['s_coordinates']['S_k'],
                        states[s]['s_coordinates']['S_t'],
                        states[s]['s_coordinates']['S_e']] for s in state_names])

    # Plot trajectory
    for i in range(len(state_names)):
        next_i = (i + 1) % len(state_names)
        ax2.plot([s_coords[i,0], s_coords[next_i,0]],
                [s_coords[i,1], s_coords[next_i,1]],
                [s_coords[i,2], s_coords[next_i,2]],
                'b-', linewidth=2, alpha=0.6)

    # Plot states
    colors = plt.cm.viridis(np.linspace(0, 1, len(state_names)))
    for i, (name, color) in enumerate(zip(state_names, colors)):
        ax2.scatter(*s_coords[i], s=400, c=[color], edgecolors='black',
                    linewidths=2, depthshade=True)
        ax2.text(*s_coords[i], f'  {name}', fontsize=8)

    ax2.set_xlabel('Sₖ (Knowledge)', fontsize=10, weight='bold')
    ax2.set_ylabel('Sₜ (Temporal)', fontsize=10, weight='bold')
    ax2.set_zlabel('Sₑ (Evolution)', fontsize=10, weight='bold')
    ax2.set_title('B. S-Space Trajectory', fontsize=13, weight='bold', pad=10)
    ax2.view_init(elev=20, azim=45)

    # Panel C: State Properties Radar Chart
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    categories = ['Volume\n(norm)', 'Frequency\n(norm)', 'Distance\n(norm)', 'Energy\n(norm)']
    N = len(categories)

    # Normalize properties
    norm_volumes = np.array(volumes) / max(volumes)
    norm_freqs = np.array(frequencies) / max(frequencies)
    norm_distances = np.array([states[s]['transmembrane_distance'] for s in state_names])
    norm_distances = norm_distances / max(norm_distances)
    norm_energies = (np.array(energies) - min(energies)) / (max(energies) - min(energies))

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, (name, color) in enumerate(zip(state_names, colors)):
        values = [norm_volumes[i], norm_freqs[i], norm_distances[i], norm_energies[i]]
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax3.fill(angles, values, alpha=0.15, color=color)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_title('C. State Properties (Normalized)', fontsize=13, weight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax3.grid(True)

    # Panel D: ATP Binding States (Pie Chart)
    ax4 = plt.subplot(2, 3, 4)
    atp_bound = sum([states[s]['atp_bound'] for s in state_names])
    atp_free = len(state_names) - atp_bound
    colors_pie = ['#ff6b6b', '#4ecdc4']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax4.pie([atp_bound, atp_free],
                                        explode=explode,
                                        labels=['ATP Bound', 'ATP Free'],
                                        colors=colors_pie,
                                        autopct='%1.0f%%',
                                        startangle=90,
                                        textprops={'fontsize': 11, 'weight': 'bold'})

    ax4.set_title('D. ATP Binding Distribution', fontsize=13, weight='bold', pad=10)

    # Panel E: Frequency Modulation Range
    ax5 = plt.subplot(2, 3, 5)
    freq_mod = data['test_1_conformational_landscape']['frequency_modulation_hz'] / 1e12
    freq_range = [min(frequencies), max(frequencies)]
    freq_center = np.mean(freq_range)

    ax5.barh(['Binding Site\nFrequencies'], [freq_range[1] - freq_range[0]],
            left=freq_range[0], height=0.4, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=2)
    ax5.axvline(freq_center, color='red', linestyle='--', linewidth=2,
                label=f'Center: {freq_center:.1f} THz')
    ax5.axvline(freq_center - freq_mod/2, color='orange', linestyle=':', linewidth=2)
    ax5.axvline(freq_center + freq_mod/2, color='orange', linestyle=':', linewidth=2,
                label=f'Modulation: ±{freq_mod/2:.1f} THz')

    for freq, name in zip(frequencies, state_names):
        ax5.plot(freq, 0, 'o', markersize=12, color='darkblue', zorder=5)
        ax5.text(freq, 0.25, name.replace('_', '\n'), fontsize=8, ha='center', rotation=0)

    ax5.set_xlabel('Frequency (THz)', fontsize=11, weight='bold')
    ax5.set_title('E. Frequency Modulation Range', fontsize=13, weight='bold', pad=10)
    ax5.legend(fontsize=9, loc='upper right')
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_yticks([])

    # Panel F: S-Distance Matrix (Heatmap)
    ax6 = plt.subplot(2, 3, 6)
    n_states = len(state_names)
    distance_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            dist = np.sqrt(sum((s_coords[i] - s_coords[j])**2))
            distance_matrix[i, j] = dist

    im = ax6.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(n_states))
    ax6.set_yticks(range(n_states))
    ax6.set_xticklabels([s.replace('_', '\n') for s in state_names], fontsize=9, rotation=45, ha='right')
    ax6.set_yticklabels([s.replace('_', '\n') for s in state_names], fontsize=9)

    # Add distance values
    for i in range(n_states):
        for j in range(n_states):
            text = ax6.text(j, i, f'{distance_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8, weight='bold')

    ax6.set_title('F. S-Space Distance Matrix', fontsize=13, weight='bold', pad=10)
    plt.colorbar(im, ax=ax6, label='Categorical Distance')

    plt.tight_layout()
    plt.savefig('figure1_conformational_landscape.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_conformational_landscape.pdf', bbox_inches='tight')
    print("✓ Figure 1 saved")
    plt.show()
