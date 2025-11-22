"""
Molecular Quantum Properties Visualization
4-panel analysis of molecular clock characteristics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_simulation_data(json_file):
    """Load simulation data from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_molecular_quantum_viz(data):
    """Create 4-panel visualization of molecular quantum properties"""

    # Extract molecular data
    molecule_data = next(c for c in data['components_tested']
                         if c['component'] == 'Molecule')
    tests = molecule_data['tests']

    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Molecular Quantum Clock Properties\n' +
                 f'Timestamp: {data["timestamp"]}',
                 fontsize=16, fontweight='bold')

    # ============================================================
    # PANEL 1: Energy Level Ladder
    # ============================================================
    ax1 = axes[0, 0]
    energy_levels = np.array(tests['energy_levels'])
    n_levels = np.arange(len(energy_levels))

    # Convert to more readable units (zeptojoules, zJ = 10^-21 J)
    energy_zJ = energy_levels * 1e21

    # Plot energy ladder
    ax1.barh(n_levels, energy_zJ, height=0.7,
             color=plt.cm.viridis(n_levels / len(n_levels)),
             edgecolor='black', linewidth=1.5)

    # Add energy spacing annotations
    for i in range(1, len(energy_levels)):
        delta_E = energy_zJ[i] - energy_zJ[i-1]
        ax1.annotate(f'ΔE = {delta_E:.2f} zJ',
                    xy=(energy_zJ[i], i),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=8, color='red')

    ax1.set_xlabel('Energy (zeptojoules, zJ)', fontweight='bold')
    ax1.set_ylabel('Quantum Level n', fontweight='bold')
    ax1.set_title('Vibrational Energy Level Ladder\n' +
                  f'ℏω = {energy_zJ[1] - energy_zJ[0]:.2f} zJ',
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(n_levels)

    # ============================================================
    # PANEL 2: Clock Precision Analysis
    # ============================================================
    ax2 = axes[0, 1]

    freq_Hz = tests['vibrational_frequency_Hz']
    period_fs = tests['vibrational_period_fs']
    precision_fs = tests['clock_precision_fs']
    Q_factor = tests['Q_factor']

    # Create precision metrics
    metrics = {
        'Frequency\n(THz)': freq_Hz / 1e12,
        'Period\n(fs)': period_fs,
        'Precision\n(fs)': precision_fs,
        'Q-factor\n(×10³)': Q_factor / 1e3
    }

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax2.bar(range(len(metrics)), metrics.values(),
                   color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for i, (bar, (key, val)) in enumerate(zip(bars, metrics.items())):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metrics.keys(), fontweight='bold')
    ax2.set_ylabel('Value (various units)', fontweight='bold')
    ax2.set_title('Molecular Clock Precision Metrics\n' +
                  f'Trans-Planckian Resolution: {precision_fs:.2f} fs',
                  fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_yscale('log')

    # ============================================================
    # PANEL 3: Frequency Distribution (Ensemble)
    # ============================================================
    ax3 = axes[1, 0]

    ensemble_size = tests['ensemble_size']
    freq_std = tests['ensemble_freq_std']

    # Generate synthetic ensemble data (since we have mean and std)
    np.random.seed(42)  # Reproducibility
    ensemble_freqs = np.random.normal(freq_Hz, freq_std, ensemble_size)

    # Convert to THz for readability
    ensemble_freqs_THz = ensemble_freqs / 1e12
    mean_freq_THz = freq_Hz / 1e12
    std_freq_THz = freq_std / 1e12

    # Histogram
    counts, bins, patches = ax3.hist(ensemble_freqs_THz, bins=30,
                                     color='skyblue', edgecolor='black',
                                     alpha=0.7, density=True)

    # Overlay Gaussian fit
    x_fit = np.linspace(ensemble_freqs_THz.min(), ensemble_freqs_THz.max(), 200)
    gaussian = (1 / (std_freq_THz * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x_fit - mean_freq_THz) / std_freq_THz)**2)
    ax3.plot(x_fit, gaussian, 'r-', linewidth=3, label='Gaussian Fit')

    # Add mean line
    ax3.axvline(mean_freq_THz, color='green', linestyle='--',
                linewidth=2, label=f'Mean: {mean_freq_THz:.2f} THz')

    # Add ±1σ lines
    ax3.axvline(mean_freq_THz - std_freq_THz, color='orange',
                linestyle=':', linewidth=2, label=f'±1σ: {std_freq_THz:.2f} THz')
    ax3.axvline(mean_freq_THz + std_freq_THz, color='orange',
                linestyle=':', linewidth=2)

    ax3.set_xlabel('Vibrational Frequency (THz)', fontweight='bold')
    ax3.set_ylabel('Probability Density', fontweight='bold')
    ax3.set_title(f'Ensemble Frequency Distribution (N={ensemble_size})\n' +
                  f'Relative Precision: {(freq_std/freq_Hz)*100:.4f}%',
                  fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # ============================================================
    # PANEL 4: Quantum State Occupancy (Boltzmann)
    # ============================================================
    ax4 = axes[1, 1]

    # Calculate Boltzmann occupancy at room temperature
    k_B = 1.380649e-23  # J/K
    T = 300  # K (room temperature)

    # Partition function
    Z = np.sum(np.exp(-energy_levels / (k_B * T)))

    # Occupancy probabilities
    occupancy = np.exp(-energy_levels / (k_B * T)) / Z

    # Plot occupancy
    ax4.bar(n_levels, occupancy, color=plt.cm.plasma(occupancy / occupancy.max()),
            edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for i, (n, p) in enumerate(zip(n_levels, occupancy)):
        if p > 0.01:  # Only label significant occupancies
            ax4.text(n, p, f'{p*100:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_xlabel('Quantum Level n', fontweight='bold')
    ax4.set_ylabel('Occupancy Probability', fontweight='bold')
    ax4.set_title(f'Boltzmann State Occupancy (T={T} K)\n' +
                  f'Ground State: {occupancy[0]*100:.2f}%',
                  fontweight='bold')
    ax4.set_xticks(n_levels)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Add thermal energy reference
    thermal_energy = k_B * T
    ax4.axhline(thermal_energy / energy_levels.max(),
                color='red', linestyle='--', linewidth=2,
                label=f'k_B T = {thermal_energy*1e21:.2f} zJ')
    ax4.legend()

    plt.tight_layout()
    return fig

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Load data
    data = load_simulation_data('simulation_data.json')

    # Create visualization
    fig = create_molecular_quantum_viz(data)

    # Save
    output_file = f"molecular_quantum_viz_{data['timestamp']}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")

    plt.show()
