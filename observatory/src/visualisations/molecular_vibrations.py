import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

def create_figure7_quantum_coherence():
    """
    Figure 7: Quantum Vibrational Coherence Measurements
    Shows coherence time, linewidth, and temporal precision
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Load all quantum vibration data
    data_files = [
        'public/quantum_vibrations_20251105_122244.json',
        'public/quantum_vibrations_20251105_122801.json',
        'public/quantum_vibrations_20251105_124305.json',
        'public/quantum_vibrations_20251105_151729.json'
    ]

    runs = []
    for file in data_files:
        with open(file, 'r') as f:
            runs.append(json.load(f))

    # Extract common parameters
    frequency_THz = runs[0]['frequency_Hz'] / 1e12
    coherence_fs = [run['coherence_time_fs'] for run in runs]
    linewidth_GHz = [run['heisenberg_linewidth_Hz'] / 1e9 for run in runs]
    precision_ps = [run['temporal_precision_fs'] / 1000 for run in runs]

    run_labels = ['Run 1\n12:22:44', 'Run 2\n12:28:01',
                  'Run 3\n12:43:05', 'Run 4\n15:17:29']

    # Panel A: Coherence Time Reproducibility
    ax1 = fig.add_subplot(gs[0, :2])

    x_pos = np.arange(len(run_labels))
    bars = ax1.bar(x_pos, coherence_fs, color='#2E86AB', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add mean line
    mean_coherence = np.mean(coherence_fs)
    ax1.axhline(y=mean_coherence, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_coherence:.1f} fs')

    ax1.set_ylabel('Coherence Time (fs)', fontweight='bold')
    ax1.set_xlabel('Experimental Run', fontweight='bold')
    ax1.set_title('(A) Quantum Coherence Time - Reproducibility',
                 fontweight='bold', loc='left')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(run_labels)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, coherence_fs)):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 2,
                f'{val:.0f}', ha='center', fontweight='bold', fontsize=9)

    # Panel B: Heisenberg Linewidth
    ax2 = fig.add_subplot(gs[0, 2])

    bars2 = ax2.barh(run_labels, linewidth_GHz, color='#A23B72', alpha=0.8,
                     edgecolor='black', linewidth=1.5)

    mean_linewidth = np.mean(linewidth_GHz)
    ax2.axvline(x=mean_linewidth, color='red', linestyle='--', linewidth=2)

    ax2.set_xlabel('Linewidth (GHz)', fontweight='bold')
    ax2.set_title('(B) Heisenberg Linewidth', fontweight='bold', loc='left')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, linewidth_GHz)):
        ax2.text(val + 5, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}', va='center', fontweight='bold', fontsize=8)

    # Panel C: Temporal Precision
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.bar(x_pos, precision_ps, color='#FFD700', alpha=0.8,
           edgecolor='black', linewidth=1.5)

    mean_precision = np.mean(precision_ps)
    ax3.axhline(y=mean_precision, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_precision:.1f} ps')

    ax3.set_ylabel('Temporal Precision (ps)', fontweight='bold')
    ax3.set_xlabel('Run', fontweight='bold')
    ax3.set_title('(C) Temporal Precision', fontweight='bold', loc='left')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['1', '2', '3', '4'])
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    # Panel D: Coherence-Linewidth Relationship
    ax4 = fig.add_subplot(gs[1, 1])

    # Theoretical relationship: Δν·Δt ≈ 1/(2π)
    theoretical_product = 1 / (2 * np.pi)  # ≈ 0.159
    measured_products = [c * 1e-15 * l * 1e9 for c, l in zip(coherence_fs, linewidth_GHz)]

    ax4.scatter(coherence_fs, linewidth_GHz, s=200, color='#2E86AB',
               marker='o', edgecolor='black', linewidth=2, zorder=3)

    # Theoretical curve
    coherence_range = np.linspace(min(coherence_fs)*0.9, max(coherence_fs)*1.1, 100)
    theoretical_linewidth = theoretical_product / (coherence_range * 1e-15) / 1e9
    ax4.plot(coherence_range, theoretical_linewidth, 'r--', linewidth=2,
            label='Heisenberg Limit', alpha=0.7)

    ax4.set_xlabel('Coherence Time (fs)', fontweight='bold')
    ax4.set_ylabel('Linewidth (GHz)', fontweight='bold')
    ax4.set_title('(D) Heisenberg Uncertainty Validation', fontweight='bold', loc='left')
    ax4.legend(loc='upper right')
    ax4.grid(alpha=0.3)

    # Panel E: Energy Level Structure
    ax5 = fig.add_subplot(gs[1, 2])

    # Get energy levels from first run (all identical)
    energy_levels = runs[0]['energy_levels_J']
    quantum_numbers = list(range(len(energy_levels)))

    # Convert to meV for readability
    energy_meV = [e * 6.242e15 for e in energy_levels[:10]]  # First 10 levels

    ax5.plot(quantum_numbers[:10], energy_meV, marker='o', markersize=8,
            linewidth=2, color='#FF4500', markeredgecolor='black',
            markeredgewidth=1.5)

    ax5.set_xlabel('Quantum Number (n)', fontweight='bold')
    ax5.set_ylabel('Energy (meV)', fontweight='bold')
    ax5.set_title('(E) Vibrational Energy Levels', fontweight='bold', loc='left')
    ax5.grid(alpha=0.3)
    ax5.set_xticks(quantum_numbers[:10])

    # Panel F: Frequency Domain
    ax6 = fig.add_subplot(gs[2, :2])

    # Simulate frequency spectrum with Lorentzian lineshape
    freq_center = frequency_THz
    freq_range = np.linspace(freq_center - 1, freq_center + 1, 1000)

    # Lorentzian for each run
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (lw, color, label) in enumerate(zip(linewidth_GHz, colors, run_labels)):
        gamma = lw / 2000  # Half-width in THz
        lorentzian = gamma**2 / ((freq_range - freq_center)**2 + gamma**2)
        ax6.plot(freq_range, lorentzian, linewidth=2, color=color,
                label=label, alpha=0.7)

    ax6.set_xlabel('Frequency (THz)', fontweight='bold')
    ax6.set_ylabel('Intensity (normalized)', fontweight='bold')
    ax6.set_title('(F) Spectral Lineshape - Frequency Domain',
                 fontweight='bold', loc='left')
    ax6.legend(loc='upper right', ncol=2)
    ax6.grid(alpha=0.3)
    ax6.set_xlim(freq_center - 0.8, freq_center + 0.8)

    # Panel G: Time Domain Coherence
    ax7 = fig.add_subplot(gs[2, 2])

    # Simulate coherence decay
    time_fs = np.linspace(0, 1000, 1000)

    for i, (tau, color, label) in enumerate(zip(coherence_fs, colors, ['R1', 'R2', 'R3', 'R4'])):
        coherence = np.exp(-time_fs / tau)
        ax7.plot(time_fs, coherence, linewidth=2, color=color,
                label=label, alpha=0.7)

    ax7.axhline(y=1/np.e, color='gray', linestyle='--', linewidth=1,
               label='1/e decay', alpha=0.5)

    ax7.set_xlabel('Time (fs)', fontweight='bold')
    ax7.set_ylabel('Coherence', fontweight='bold')
    ax7.set_title('(G) Temporal Coherence Decay', fontweight='bold', loc='left')
    ax7.legend(loc='upper right', ncol=2, fontsize=8)
    ax7.grid(alpha=0.3)
    ax7.set_xlim(0, 800)

    plt.suptitle('Figure 7: Quantum Vibrational Coherence at 71 THz',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig('Figure7_Quantum_Coherence.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure7_Quantum_Coherence.pdf', dpi=300, bbox_inches='tight')
    print("✓ Figure 7 saved: Quantum Coherence")
    return fig


def create_figure8_energy_quantization():
    """
    Figure 8: Molecular Energy Level Quantization
    Shows quantum state structure and spacing
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Load data
    with open('public/quantum_vibrations_20251105_122244.json', 'r') as f:
        data = json.load(f)

    energy_levels = data['energy_levels_J']
    frequency_Hz = data['frequency_Hz']

    # Panel A: Energy Level Diagram
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot first 15 levels
    n_levels = 15
    quantum_numbers = list(range(n_levels))
    energies_meV = [e * 6.242e15 for e in energy_levels[:n_levels]]

    for i, (n, E) in enumerate(zip(quantum_numbers, energies_meV)):
        ax1.hlines(E, 0, 1, colors='blue', linewidth=2)
        ax1.text(1.05, E, f'n={n}', va='center', fontsize=8)

        # Add transitions
        if i > 0:
            ax1.arrow(0.5, energies_meV[i-1], 0, E - energies_meV[i-1] - 0.5,
                     head_width=0.05, head_length=0.3, fc='red', ec='red',
                     alpha=0.3, linewidth=0.5)

    ax1.set_ylabel('Energy (meV)', fontweight='bold')
    ax1.set_title('(A) Quantized Energy Levels', fontweight='bold', loc='left')
    ax1.set_xlim(-0.1, 1.3)
    ax1.set_xticks([])
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: Energy Spacing
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate energy differences
    energy_spacing = [energies_meV[i+1] - energies_meV[i]
                      for i in range(len(energies_meV)-1)]
    transitions = [f'{i}→{i+1}' for i in range(len(energy_spacing))]

    ax2.plot(range(len(energy_spacing)), energy_spacing, marker='o',
            markersize=8, linewidth=2, color='#2E86AB',
            markeredgecolor='black', markeredgewidth=1.5)

    # Expected spacing (ℏω for harmonic oscillator)
    h = 6.626e-34
    expected_spacing = h * frequency_Hz * 6.242e15  # meV
    ax2.axhline(y=expected_spacing, color='red', linestyle='--', linewidth=2,
               label=f'ℏω = {expected_spacing:.2f} meV')

    ax2.set_xlabel('Transition', fontweight='bold')
    ax2.set_ylabel('Energy Spacing (meV)', fontweight='bold')
    ax2.set_title('(B) Energy Level Spacing', fontweight='bold', loc='left')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    # Panel C: Quantum Number vs Energy
    ax3 = fig.add_subplot(gs[0, 2])

    ax3.scatter(quantum_numbers, energies_meV, s=150, color='#A23B72',
               marker='o', edgecolor='black', linewidth=2, zorder=3)

    # Fit line
    z = np.polyfit(quantum_numbers, energies_meV, 1)
    p = np.poly1d(z)
    ax3.plot(quantum_numbers, p(quantum_numbers), 'r--', linewidth=2,
            label=f'Linear fit: {z[0]:.2f}n + {z[1]:.2f}', alpha=0.7)

    ax3.set_xlabel('Quantum Number (n)', fontweight='bold')
    ax3.set_ylabel('Energy (meV)', fontweight='bold')
    ax3.set_title('(C) Energy vs Quantum Number', fontweight='bold', loc='left')
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.3)

    # Panel D: Population Distribution (Boltzmann)
    ax4 = fig.add_subplot(gs[1, 0])

    # Calculate Boltzmann distribution at room temperature
    kB = 1.381e-23  # J/K
    T = 300  # K

    populations = []
    for E in energy_levels[:n_levels]:
        pop = np.exp(-E / (kB * T))
        populations.append(pop)

    # Normalize
    populations = np.array(populations) / sum(populations)

    ax4.bar(quantum_numbers, populations, color='#FFD700', alpha=0.8,
           edgecolor='black', linewidth=1.5)

    ax4.set_xlabel('Quantum Number (n)', fontweight='bold')
    ax4.set_ylabel('Population (normalized)', fontweight='bold')
    ax4.set_title('(D) Boltzmann Distribution (T=300K)', fontweight='bold', loc='left')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_yscale('log')

    # Panel E: Cumulative Energy
    ax5 = fig.add_subplot(gs[1, 1])

    cumulative_energy = np.cumsum(energies_meV)

    ax5.plot(quantum_numbers, cumulative_energy, marker='s', markersize=8,
            linewidth=2.5, color='#FF4500', markeredgecolor='black',
            markeredgewidth=1.5)

    ax5.fill_between(quantum_numbers, cumulative_energy, alpha=0.3,
                    color='#FF4500')

    ax5.set_xlabel('Quantum Number (n)', fontweight='bold')
    ax5.set_ylabel('Cumulative Energy (meV)', fontweight='bold')
    ax5.set_title('(E) Cumulative Energy Distribution', fontweight='bold', loc='left')
    ax5.grid(alpha=0.3)

    # Panel F: Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = f"""
    QUANTUM VIBRATION PARAMETERS

    Fundamental Frequency:
    • ν₀ = {frequency_Hz/1e12:.1f} THz
    • λ = {3e8/frequency_Hz*1e6:.2f} μm (IR)
    • ℏω = {expected_spacing:.2f} meV

    Coherence Properties:
    • τ_coh = {data['coherence_time_fs']:.0f} fs
    • Δν = {data['heisenberg_linewidth_Hz']/1e9:.1f} GHz
    • Δν·τ = {data['heisenberg_linewidth_Hz']*data['coherence_time_fs']*1e-15:.3f}

    Energy Level Structure:
    • Ground state: {energies_meV[0]:.2f} meV
    • First excited: {energies_meV[1]:.2f} meV
    • Spacing: {energy_spacing[0]:.2f} meV
    • Levels measured: {n_levels}

    Quantum Regime:
    • kT (300K) = 25.9 meV
    • ℏω/kT = {expected_spacing/25.9:.2f}
    • Quantum effects: Significant
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))

    plt.suptitle('Figure 8: Molecular Energy Level Quantization at 71 THz',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('Figure8_Energy_Quantization.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure8_Energy_Quantization.pdf', dpi=300, bbox_inches='tight')
    print("✓ Figure 8 saved: Energy Quantization")
    return fig


def create_figure9_quantum_classical_bridge():
    """
    Figure 9: Quantum-Classical Bridge
    Connects quantum vibrations to pattern transfer mechanism
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # Load data
    with open('public/quantum_vibrations_20251105_122244.json', 'r') as f:
        qdata = json.load(f)

    # Panel A: Multi-Scale Time Domains
    ax1 = fig.add_subplot(gs[0, :])

    # Time scales
    time_scales = {
        'Quantum coherence': qdata['coherence_time_fs'] * 1e-15,
        'Vibrational period': 1/qdata['frequency_Hz'],
        'Pattern transfer (H₂O)': 1.17e-8,
        'Pattern transfer (CH₄)': 2.53e-9,
        'Light travel (1 AU)': 500,
        'Positioning (10 ly)': 3.51 * 365.25 * 24 * 3600
    }

    labels = list(time_scales.keys())
    times_log = [np.log10(t) for t in time_scales.values()]

    colors_scale = ['#8B00FF', '#FFD700', '#2E86AB', '#A23B72', '#FF4500', '#2ca02c']

    bars = ax1.barh(labels, times_log, color=colors_scale, alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Time (log₁₀ seconds)', fontweight='bold')
    ax1.set_title('(A) Multi-Scale Temporal Hierarchy', fontweight='bold', loc='left')
    ax1.grid(axis='x', alpha=0.3)

    # Add actual time labels
    for i, (bar, label, time) in enumerate(zip(bars, labels, time_scales.values())):
        if time < 1e-12:
            time_str = f'{time*1e15:.0f} fs'
        elif time < 1e-9:
            time_str = f'{time*1e12:.1f} ps'
        elif time < 1e-6:
            time_str = f'{time*1e9:.1f} ns'
        elif time < 1:
            time_str = f'{time*1e3:.1f} ms'
        elif time < 3600:
            time_str = f'{time:.1f} s'
        elif time < 86400:
            time_str = f'{time/3600:.1f} hr'
        else:
            time_str = f'{time/(365.25*24*3600):.1f} yr'

        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
                time_str, va='center', fontweight='bold', fontsize=9)

    # Panel B: Frequency-Energy Relationship
    ax2 = fig.add_subplot(gs[1, 0])

    # Multiple frequency domains
    frequencies_Hz = np.logspace(6, 15, 100)  # MHz to PHz
    energies_eV = frequencies_Hz * 4.136e-15  # eV

    ax2.loglog(frequencies_Hz, energies_eV, linewidth=3, color='blue', alpha=0.7)

    # Mark key points
    key_points = {
        'LED system': (16.1e6, 16.1e6 * 4.136e-15),
        'Molecular vibration': (qdata['frequency_Hz'], qdata['frequency_Hz'] * 4.136e-15)
    }

    for label, (freq, energy) in key_points.items():
        ax2.scatter(freq, energy, s=200, marker='*', edgecolor='black',
                   linewidth=2, zorder=5)
        ax2.annotate(label, xy=(freq, energy), xytext=(freq*3, energy*0.3),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    ax2.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax2.set_ylabel('Energy (eV)', fontweight='bold')
    ax2.set_title('(B) Frequency-Energy Relationship', fontweight='bold', loc='left')
    ax2.grid(alpha=0.3, which='both')

    # Panel C: Coherence Length vs Velocity
    ax3 = fig.add_subplot(gs[1, 1])

    # Coherence length = c * coherence_time
    c = 3e8  # m/s
    coherence_length_nm = c * qdata['coherence_time_fs'] * 1e-15 * 1e9

    # For different velocities
    velocities_c = [1.0, 2.846, 8.103, 23.08, 65.71]
    coherence_lengths = [coherence_length_nm * v for v in velocities_c]

    bars = ax3.bar(range(len(velocities_c)), coherence_lengths,
                   color=['gray', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax3.set_ylabel('Coherence Length (nm)', fontweight='bold')
    ax3.set_xlabel('Velocity Configuration', fontweight='bold')
    ax3.set_title('(C) Coherence Length Enhancement', fontweight='bold', loc='left')
    ax3.set_xticks(range(len(velocities_c)))
    ax3.set_xticklabels([f'{v}c' for v in velocities_c])
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, coherence_lengths):
        ax3.text(bar.get_x() + bar.get_width()/2., val + max(coherence_lengths)*0.02,
                f'{val:.0f}', ha='center', fontweight='bold', fontsize=9)

    plt.suptitle('Figure 9: Quantum-Classical Bridge - Multi-Scale Integration',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('Figure9_Quantum_Classical_Bridge.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure9_Quantum_Classical_Bridge.pdf', dpi=300, bbox_inches='tight')
    print("✓ Figure 9 saved: Quantum-Classical Bridge")
    return fig


# Add to main function
def main_quantum_figures():
    """Generate quantum vibration figures"""
    print("="*70)
    print("GENERATING QUANTUM VIBRATION FIGURES")
    print("="*70)
    print()

    try:
        print("Creating Figure 7: Quantum Coherence...")
        create_figure7_quantum_coherence()

        print("Creating Figure 8: Energy Quantization...")
        create_figure8_energy_quantization()

        print("Creating Figure 9: Quantum-Classical Bridge...")
        create_figure9_quantum_classical_bridge()

        print()
        print("="*70)
        print("QUANTUM FIGURES GENERATED SUCCESSFULLY")
        print("="*70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_quantum_figures()
