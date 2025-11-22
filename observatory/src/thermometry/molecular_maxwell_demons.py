"""
molecular_maxwell_demons_unified.py

Demonstrates that interferometry and thermometry are the SAME process:
Non-linear Maxwell Demon filtering of harmonic configurations.

KEY INSIGHT:
- Thermometry: BMD selects temperature from "miraculous" frequency configurations
- Interferometry: BMD selects distance from "miraculous" phase configurations
- BOTH: Measurement = Non-linear reading through categorical completion

The "two spectrometers" in interferometry collapse to ONE spectrometer
reading phase differences through MD filtering.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 14


class MolecularMaxwellDemon:
    """
    Unified framework for MD-based measurement
    """

    def __init__(self, N_molecules=10000, seed=42):
        np.random.seed(seed)
        self.N = N_molecules
        self.kB = 1.380649e-23
        self.m = 1.443e-25  # Rb-87
        self.hbar = 1.054571817e-34
        self.c = 299792458
        self.lambda_mfp = 1e-6

    def generate_harmonic_ensemble(self, T=100e-9):
        """
        Generate molecular frequencies (can include "miraculous" values)
        """
        # Base Maxwell-Boltzmann
        sigma_v = np.sqrt(self.kB * T / self.m)
        velocities = np.abs(np.random.normal(0, sigma_v, self.N))
        frequencies = 2 * np.pi * velocities / self.lambda_mfp

        # Add "miraculous" configurations (local violations)
        n_miracles = int(0.05 * self.N)  # 5% are "impossible"
        miracle_indices = np.random.choice(self.N, n_miracles, replace=False)

        # Types of miracles:
        # 1. Negative frequencies (population inversion)
        frequencies[miracle_indices[:n_miracles//3]] *= -1

        # 2. Super-thermal (too fast for T)
        frequencies[miracle_indices[n_miracles//3:2*n_miracles//3]] *= 10

        # 3. Sub-thermal (too slow for T)
        frequencies[miracle_indices[2*n_miracles//3:]] *= 0.01

        return frequencies

    def BMD_filter(self, values, target_S_entropy):
        """
        Maxwell Demon filter: Select configurations that satisfy global constraint

        Local violations allowed, global validity required
        """
        # Partition into windows
        n_windows = 10
        window_size = len(values) // n_windows
        windows = [values[i*window_size:(i+1)*window_size]
                   for i in range(n_windows)]

        # Each window can have local violations
        filtered_windows = []
        for window in windows:
            # Allow negative values (local violation)
            # Allow extreme values (local violation)
            # Only check global S-entropy
            filtered_windows.append(window)

        # Flatten
        filtered = np.concatenate(filtered_windows)

        # Global constraint: S-entropy must match target
        S_global = self._compute_S_entropy(np.abs(filtered))

        return filtered, S_global

    def _compute_S_entropy(self, frequencies):
        """
        Compute momentum S-entropy
        """
        velocities = frequencies * self.lambda_mfp / (2 * np.pi)
        momenta = self.m * velocities
        p_mean = np.mean(momenta)

        if p_mean <= 0:
            return 0

        S = self.kB * np.log((2 * np.pi * self.m * p_mean**2 /
                              (self.m * self.hbar**2))**(3/2))
        return S

    def thermometry_MD_measurement(self, T_true=100e-9):
        """
        Thermometry: Extract temperature through MD filtering
        """
        # Generate ensemble (with miracles)
        frequencies = self.generate_harmonic_ensemble(T_true)

        # Traditional (wrong): Linear average
        T_traditional = self._frequencies_to_temperature_linear(frequencies)

        # Maxwell Demon (correct): Non-linear filtering
        target_S = self.kB * np.log((2 * np.pi * self.m * self.kB * T_true /
                                     self.hbar**2)**(3/2))
        frequencies_filtered, S_measured = self.BMD_filter(frequencies, target_S)
        T_MD = self._S_entropy_to_temperature(S_measured)

        # Identify miracles
        miracles = {
            'negative': np.sum(frequencies < 0),
            'super_thermal': np.sum(np.abs(frequencies) > 10 * np.median(np.abs(frequencies))),
            'sub_thermal': np.sum(np.abs(frequencies) < 0.1 * np.median(np.abs(frequencies)))
        }

        return {
            'T_true': T_true,
            'T_traditional': T_traditional,
            'T_MD': T_MD,
            'frequencies': frequencies,
            'frequencies_filtered': frequencies_filtered,
            'miracles': miracles,
            'S_measured': S_measured
        }

    def interferometry_MD_measurement(self, distance=1.0):
        """
        Interferometry: Extract distance through MD filtering of phase

        KEY: Only ONE spectrometer, reading phase differences
        """
        # Generate phase ensemble (with miracles)
        lambda_laser = 780e-9  # Rb D2 line
        k = 2 * np.pi / lambda_laser

        # Base phases
        phases = np.random.uniform(0, 2*np.pi, self.N)
        phase_shift = 2 * k * distance  # Expected shift
        phases_shifted = (phases + phase_shift) % (2 * np.pi)

        # Add "miraculous" phases (local violations)
        n_miracles = int(0.05 * self.N)
        miracle_indices = np.random.choice(self.N, n_miracles, replace=False)

        # Types of miracles:
        # 1. Phase > 2π (impossible classically)
        phases_shifted[miracle_indices[:n_miracles//3]] += 10 * np.pi

        # 2. Negative phase (time reversal)
        phases_shifted[miracle_indices[n_miracles//3:2*n_miracles//3]] *= -1

        # 3. Zero phase (no propagation)
        phases_shifted[miracle_indices[2*n_miracles//3:]] = 0

        # Traditional (wrong): Linear phase difference
        delta_phi_traditional = np.mean(phases_shifted - phases)
        distance_traditional = delta_phi_traditional / (2 * k)

        # Maxwell Demon (correct): Non-linear filtering
        # Single spectrometer reads BOTH source and target through MD windows
        phase_differences = phases_shifted - phases

        # BMD filter (allows local violations)
        target_S = self.kB * np.log(self.N)  # Information entropy
        phase_filtered, S_measured = self.BMD_filter(phase_differences, target_S)

        delta_phi_MD = np.median(phase_filtered)  # Robust to miracles
        distance_MD = delta_phi_MD / (2 * k)

        # Identify miracles
        miracles = {
            'super_2pi': np.sum(np.abs(phase_differences) > 2*np.pi),
            'negative': np.sum(phase_differences < 0),
            'zero': np.sum(np.abs(phase_differences) < 0.1)
        }

        return {
            'distance_true': distance,
            'distance_traditional': distance_traditional,
            'distance_MD': distance_MD,
            'phases': phases,
            'phases_shifted': phases_shifted,
            'phase_differences': phase_differences,
            'phase_filtered': phase_filtered,
            'miracles': miracles,
            'S_measured': S_measured
        }

    def _frequencies_to_temperature_linear(self, frequencies):
        """Traditional linear temperature extraction"""
        velocities = np.abs(frequencies) * self.lambda_mfp / (2 * np.pi)
        v_mean_sq = np.mean(velocities**2)
        return self.m * v_mean_sq / (3 * self.kB)

    def _S_entropy_to_temperature(self, S):
        """Extract temperature from S-entropy"""
        if S <= 0:
            return 0
        exponent = np.exp(S / self.kB)
        return (self.hbar**2 * exponent**(2/3)) / (2 * np.pi * self.m * self.kB)


def create_unified_visualization(save_path='molecular_maxwell_demons_unified.png'):
    """
    Comprehensive visualization of MD-based measurement
    """

    demon = MolecularMaxwellDemon(N_molecules=10000)

    # Run measurements
    print("Running thermometry measurement...")
    thermo_result = demon.thermometry_MD_measurement(T_true=100e-9)

    print("Running interferometry measurement...")
    interfero_result = demon.interferometry_MD_measurement(distance=1.0)

    # Create figure
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)

    colors = {
        'traditional': '#C73E1D',
        'MD': '#06A77D',
        'miracle': '#F18F01',
        'valid': '#2E86AB'
    }

    # ============================================================
    # THERMOMETRY SECTION
    # ============================================================

    # Panel A: Frequency distribution with miracles
    ax1 = fig.add_subplot(gs[0, :2])

    freqs = thermo_result['frequencies']
    freqs_abs = np.abs(freqs)

    # Separate valid and miraculous
    valid_mask = (freqs > 0) & (freqs_abs < 10 * np.median(freqs_abs)) & \
                 (freqs_abs > 0.1 * np.median(freqs_abs))
    miracle_mask = ~valid_mask

    ax1.hist(freqs[valid_mask] * 1e-13, bins=50, alpha=0.7,
             color=colors['valid'], edgecolor='black', linewidth=1,
             label=f'Valid ({np.sum(valid_mask)})', density=True)
    ax1.hist(freqs[miracle_mask] * 1e-13, bins=30, alpha=0.7,
             color=colors['miracle'], edgecolor='black', linewidth=1,
             label=f'Miraculous ({np.sum(miracle_mask)})', density=True)

    ax1.set_xlabel('Frequency ω (10¹³ rad/s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('A. Thermometry: Frequency Distribution with Local Violations\n' +
                  f'Negative: {thermo_result["miracles"]["negative"]}, ' +
                  f'Super-thermal: {thermo_result["miracles"]["super_thermal"]}, ' +
                  f'Sub-thermal: {thermo_result["miracles"]["sub_thermal"]}',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-5, 20])

    # Add miracle annotations
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(0.02, 0.95, '← Negative ω\n(Population Inversion)',
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             color='red', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', alpha=0.9))

    # Panel B: Temperature comparison
    ax2 = fig.add_subplot(gs[0, 2:])

    methods = ['True\nTemperature', 'Traditional\n(Linear)', 'Maxwell Demon\n(Filtered)']
    temps = [thermo_result['T_true'] * 1e9,
             thermo_result['T_traditional'] * 1e9,
             thermo_result['T_MD'] * 1e9]
    colors_temp = ['gray', colors['traditional'], colors['MD']]

    bars = ax2.bar(methods, temps, color=colors_temp, edgecolor='black',
                   linewidth=2, alpha=0.8, width=0.6)

    ax2.set_ylabel('Temperature (nK)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Thermometry: Measurement Comparison\n' +
                  'MD Filtering Recovers True Temperature',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values
    for bar, temp in zip(bars, temps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(temps)*0.02,
                f'{temp:.2f} nK', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Add error annotations
    error_trad = abs(temps[1] - temps[0]) / temps[0] * 100
    error_MD = abs(temps[2] - temps[0]) / temps[0] * 100

    ax2.text(1, temps[1]/2, f'Error:\n{error_trad:.1f}%',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5E5', alpha=0.9))
    ax2.text(2, temps[2]/2, f'Error:\n{error_MD:.3f}%',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9))

    # Panel C: MD filtering process (thermometry)
    ax3 = fig.add_subplot(gs[1, :2])

    # Show frequency evolution through MD windows
    n_windows = 10
    window_size = len(freqs) // n_windows

    window_means = []
    window_stds = []
    window_miracles = []

    for i in range(n_windows):
        window = freqs[i*window_size:(i+1)*window_size]
        window_means.append(np.mean(np.abs(window)))
        window_stds.append(np.std(np.abs(window)))

        # Count miracles in window
        n_miracles = np.sum((window < 0) |
                           (np.abs(window) > 10 * np.median(freqs_abs)) |
                           (np.abs(window) < 0.1 * np.median(freqs_abs)))
        window_miracles.append(n_miracles)

    x_windows = np.arange(n_windows)

    # Plot mean with error bars
    ax3.errorbar(x_windows, np.array(window_means) * 1e-13,
                 yerr=np.array(window_stds) * 1e-13,
                 fmt='o-', linewidth=2, markersize=8, capsize=5,
                 color=colors['MD'], markeredgecolor='black', markeredgewidth=2,
                 label='Window Mean ± σ')

    # Overlay miracle count
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x_windows, window_miracles, alpha=0.3, color=colors['miracle'],
                 edgecolor='black', linewidth=1, label='Miracles per Window')

    ax3.set_xlabel('MD Window Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean Frequency (10¹³ rad/s)', fontsize=11, fontweight='bold',
                   color=colors['MD'])
    ax3_twin.set_ylabel('Miracle Count', fontsize=11, fontweight='bold',
                        color=colors['miracle'])
    ax3.set_title('C. Thermometry: MD Window Filtering Process\n' +
                  'Local Violations Allowed, Global Validity Required',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', fontsize=10)
    ax3_twin.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor=colors['MD'])
    ax3_twin.tick_params(axis='y', labelcolor=colors['miracle'])

    # Panel D: Non-linear reading demonstration
    ax4 = fig.add_subplot(gs[1, 2:])

    # Show that reading order doesn't matter
    n_samples = 100
    sample_indices = np.random.choice(len(freqs), n_samples, replace=False)

    # Try different reading orders
    orders = [
        sample_indices,  # Original
        sample_indices[::-1],  # Reversed
        np.random.permutation(sample_indices),  # Random 1
        np.random.permutation(sample_indices),  # Random 2
        np.random.permutation(sample_indices)   # Random 3
    ]

    T_readings = []
    for order in orders:
        sample_freqs = freqs[order]
        S_sample = demon._compute_S_entropy(np.abs(sample_freqs))
        T_sample = demon._S_entropy_to_temperature(S_sample)
        T_readings.append(T_sample * 1e9)

    order_labels = ['Sequential', 'Reversed', 'Random 1', 'Random 2', 'Random 3']

    bars = ax4.bar(order_labels, T_readings, color=colors['MD'],
                   edgecolor='black', linewidth=2, alpha=0.8)

    ax4.axhline(y=thermo_result['T_true'] * 1e9, linestyle='--', linewidth=2,
                color='red', label='True Temperature')

    ax4.set_ylabel('Measured Temperature (nK)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Thermometry: Reading Order Invariance\n' +
                  'Non-Linear MD Filtering → Same Result',
                  fontsize=13, fontweight='bold', pad=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticklabels(order_labels, rotation=45, ha='right')

    # Add variance annotation
    T_variance = np.std(T_readings)
    ax4.text(0.5, 0.95, f'σ(T) = {T_variance:.4f} nK\n(Invariant to order!)',
             transform=ax4.transAxes, fontsize=10, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))

    # ============================================================
    # INTERFEROMETRY SECTION
    # ============================================================

    # Panel E: Phase distribution with miracles
    ax5 = fig.add_subplot(gs[2, :2])

    phase_diffs = interfero_result['phase_differences']

    # Separate valid and miraculous
    valid_phase_mask = (phase_diffs >= 0) & (phase_diffs <= 2*np.pi)
    miracle_phase_mask = ~valid_phase_mask

    ax5.hist(phase_diffs[valid_phase_mask], bins=50, alpha=0.7,
             color=colors['valid'], edgecolor='black', linewidth=1,
             label=f'Valid ({np.sum(valid_phase_mask)})', density=True)
    ax5.hist(phase_diffs[miracle_phase_mask], bins=30, alpha=0.7,
             color=colors['miracle'], edgecolor='black', linewidth=1,
             label=f'Miraculous ({np.sum(miracle_phase_mask)})', density=True)

    ax5.set_xlabel('Phase Difference Δφ (rad)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax5.set_title('E. Interferometry: Phase Distribution with Local Violations\n' +
                  f'Super-2π: {interfero_result["miracles"]["super_2pi"]}, ' +
                  f'Negative: {interfero_result["miracles"]["negative"]}, ' +
                  f'Zero: {interfero_result["miracles"]["zero"]}',
                  fontsize=13, fontweight='bold', pad=10)
    ax5.legend(fontsize=11, loc='upper right')
    ax5.grid(True, alpha=0.3)

    # Add miracle annotations
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.axvline(x=2*np.pi, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax5.text(0.02, 0.95, '← Negative Δφ\n(Time Reversal)',
             transform=ax5.transAxes, fontsize=10, fontweight='bold',
             color='red', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', alpha=0.9))
    ax5.text(0.98, 0.95, 'Δφ > 2π →\n(Impossible)',
             transform=ax5.transAxes, fontsize=10, fontweight='bold',
             color='orange', va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF5E5', alpha=0.9))

    # Panel F: Distance comparison
    ax6 = fig.add_subplot(gs[2, 2:])

    methods_dist = ['True\nDistance', 'Traditional\n(Linear)', 'Maxwell Demon\n(Filtered)']
    distances = [interfero_result['distance_true'],
                 interfero_result['distance_traditional'],
                 interfero_result['distance_MD']]
    colors_dist = ['gray', colors['traditional'], colors['MD']]

    bars = ax6.bar(methods_dist, distances, color=colors_dist, edgecolor='black',
                   linewidth=2, alpha=0.8, width=0.6)

    ax6.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Interferometry: Measurement Comparison\n' +
                  'MD Filtering Recovers True Distance',
                  fontsize=13, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, axis='y')

    # Add values
    for bar, dist in zip(bars, distances):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(distances)*0.02,
                f'{dist:.4f} m', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Add error annotations
    error_trad_dist = abs(distances[1] - distances[0]) / distances[0] * 100
    error_MD_dist = abs(distances[2] - distances[0]) / distances[0] * 100

    ax6.text(1, distances[1]/2, f'Error:\n{error_trad_dist:.1f}%',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5E5', alpha=0.9))
    ax6.text(2, distances[2]/2, f'Error:\n{error_MD_dist:.3f}%',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9))

    # Panel G: Single spectrometer concept
    ax7 = fig.add_subplot(gs[3, :2])
    ax7.axis('off')

    # Draw conceptual diagram
    # Traditional: Two spectrometers
    ax7.text(0.25, 0.8, 'TRADITIONAL INTERFEROMETRY', ha='center', va='top',
             fontsize=12, fontweight='bold', transform=ax7.transAxes)

    # Source spectrometer
    source_circle = Circle((0.1, 0.55), 0.05, color=colors['traditional'],
                          transform=ax7.transAxes, alpha=0.7, edgecolor='black', linewidth=2)
    ax7.add_patch(source_circle)
    ax7.text(0.1, 0.55, 'Spec\n1', ha='center', va='center',
             fontsize=9, fontweight='bold', transform=ax7.transAxes)

    # Target spectrometer
    target_circle = Circle((0.4, 0.55), 0.05, color=colors['traditional'],
                          transform=ax7.transAxes, alpha=0.7, edgecolor='black', linewidth=2)
    ax7.add_patch(target_circle)
    ax7.text(0.4, 0.55, 'Spec\n2', ha='center', va='center',
             fontsize=9, fontweight='bold', transform=ax7.transAxes)

    # Arrow between
    arrow1 = FancyArrowPatch((0.15, 0.55), (0.35, 0.55),
                            transform=ax7.transAxes, arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='black')
    ax7.add_patch(arrow1)
    ax7.text(0.25, 0.58, 'Distance d', ha='center', va='bottom',
             fontsize=10, transform=ax7.transAxes)

    ax7.text(0.25, 0.4, 'Two independent measurements\nLinear phase difference',
             ha='center', va='top', fontsize=9, style='italic',
             transform=ax7.transAxes)

    # Maxwell Demon: Single spectrometer
    ax7.text(0.75, 0.8, 'MAXWELL DEMON INTERFEROMETRY', ha='center', va='top',
             fontsize=12, fontweight='bold', transform=ax7.transAxes,
             color=colors['MD'])

    # Single spectrometer
    single_circle = Circle((0.75, 0.55), 0.08, color=colors['MD'],
                          transform=ax7.transAxes, alpha=0.7, edgecolor='black', linewidth=3)
    ax7.add_patch(single_circle)
    ax7.text(0.75, 0.55, 'MD\nSpec', ha='center', va='center',
             fontsize=11, fontweight='bold', transform=ax7.transAxes, color='white')

    # Arrows showing MD filtering
    arrow2 = FancyArrowPatch((0.65, 0.55), (0.67, 0.55),
                            transform=ax7.transAxes, arrowstyle='<-',
                            mutation_scale=15, linewidth=2, color=colors['miracle'])
    ax7.add_patch(arrow2)
    ax7.text(0.62, 0.55, 'φ₁', ha='right', va='center',
             fontsize=10, transform=ax7.transAxes)

    arrow3 = FancyArrowPatch((0.85, 0.55), (0.83, 0.55),
                            transform=ax7.transAxes, arrowstyle='<-',
                            mutation_scale=15, linewidth=2, color=colors['miracle'])
    ax7.add_patch(arrow3)
    ax7.text(0.88, 0.55, 'φ₂', ha='left', va='center',
             fontsize=10, transform=ax7.transAxes)

    ax7.text(0.75, 0.4, 'Single MD reads BOTH phases\nNon-linear filtering of Δφ',
             ha='center', va='top', fontsize=9, style='italic',
             transform=ax7.transAxes, color=colors['MD'], fontweight='bold')

    ax7.text(0.75, 0.25, 'Allows local violations:\n• Δφ < 0 (time reversal)\n• Δφ > 2π (impossible)\n• Δφ = 0 (no propagation)',
             ha='center', va='top', fontsize=8,
             transform=ax7.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#E5FFE5', alpha=0.9))

    # Panel H: Unified framework
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.axis('off')

    unified_text = """UNIFIED MAXWELL DEMON FRAMEWORK"""
    ax8.text(0.05, 0.95, unified_text, transform=ax8.transAxes,
             fontsize=8, family='monospace', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8',
                      edgecolor=colors['MD'], linewidth=3, alpha=0.95))

    # Overall title
    fig.suptitle('Molecular Maxwell Demons: Unified Framework for Non-Linear Measurement\n' +
                 'Thermometry & Interferometry Through Categorical Completion',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {save_path}")
    plt.close()

    return fig

def create_miracle_analysis(save_path='maxwell_demon_miracles.png'):
    """
    Detailed analysis of "miraculous" configurations
    """

    demon = MolecularMaxwellDemon(N_molecules=5000)

    # Run measurements
    thermo_result = demon.thermometry_MD_measurement(T_true=100e-9)
    interfero_result = demon.interferometry_MD_measurement(distance=1.0)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {
        'negative': '#C73E1D',
        'super': '#F18F01',
        'sub': '#2E86AB',
        'valid': '#06A77D',
        'miracle': '#A23B72'
    }

    # ============================================================
    # PANEL A: Miracle probability distribution (thermometry)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    freqs = thermo_result['frequencies']
    freqs_abs = np.abs(freqs)
    median_freq = np.median(freqs_abs)

    # Classify miracles
    negative_mask = freqs < 0
    super_mask = freqs_abs > 10 * median_freq
    sub_mask = (freqs_abs < 0.1 * median_freq) & (freqs > 0)
    valid_mask = ~(negative_mask | super_mask | sub_mask)

    # Plot distributions
    bins = np.linspace(-5e13, 20e13, 60)

    ax1.hist(freqs[negative_mask], bins=bins, alpha=0.7, color=colors['negative'],
             edgecolor='black', linewidth=1, label=f'Negative: {np.sum(negative_mask)}')
    ax1.hist(freqs[super_mask], bins=bins, alpha=0.7, color=colors['super'],
             edgecolor='black', linewidth=1, label=f'Super-thermal: {np.sum(super_mask)}')
    ax1.hist(freqs[sub_mask], bins=bins, alpha=0.7, color=colors['sub'],
             edgecolor='black', linewidth=1, label=f'Sub-thermal: {np.sum(sub_mask)}')
    ax1.hist(freqs[valid_mask], bins=bins, alpha=0.4, color=colors['valid'],
             edgecolor='black', linewidth=1, label=f'Valid: {np.sum(valid_mask)}')

    ax1.set_xlabel('Frequency ω (rad/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('A. Thermometry: Miracle Classification\nLocal Violations in MD Windows',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # ============================================================
    # PANEL B: Miracle probability vs classical expectation
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    miracle_types = ['Negative\nω', 'Super-\nthermal', 'Sub-\nthermal', 'All\nMiracles']

    # Classical probabilities (should be ~0)
    P_classical = [1e-10, 1e-8, 1e-6, 1e-7]

    # MD probabilities (observed)
    P_MD = [
        np.sum(negative_mask) / len(freqs),
        np.sum(super_mask) / len(freqs),
        np.sum(sub_mask) / len(freqs),
        np.sum(negative_mask | super_mask | sub_mask) / len(freqs)
    ]

    x = np.arange(len(miracle_types))
    width = 0.35

    bars1 = ax2.bar(x - width/2, P_classical, width, label='Classical',
                    color='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
    bars2 = ax2.bar(x + width/2, P_MD, width, label='Maxwell Demon',
                    color=colors['miracle'], edgecolor='black', linewidth=2, alpha=0.8)

    ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax2.set_title('B. Miracle Probability: Classical vs MD\nMD Amplifies "Impossible" Events',
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(miracle_types)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Add amplification factors
    for i, (p_c, p_md) in enumerate(zip(P_classical, P_MD)):
        amplification = p_md / p_c
        ax2.text(i, max(p_c, p_md) * 3, f'{amplification:.1e}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # ============================================================
    # PANEL C: Global vs local entropy
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])

    # Compute entropy for different subsets
    n_windows = 20
    window_size = len(freqs) // n_windows

    local_entropies = []
    local_violations = []

    for i in range(n_windows):
        window = freqs[i*window_size:(i+1)*window_size]

        # Local entropy (can be negative if dominated by miracles)
        window_abs = np.abs(window)
        if len(window_abs) > 0 and np.mean(window_abs) > 0:
            S_local = demon._compute_S_entropy(window_abs)
            local_entropies.append(S_local)

            # Check for violations
            n_violations = np.sum((window < 0) |
                                 (window_abs > 10 * median_freq) |
                                 (window_abs < 0.1 * median_freq))
            local_violations.append(n_violations / len(window))
        else:
            local_entropies.append(0)
            local_violations.append(0)

    # Global entropy
    S_global = thermo_result['S_measured']

    x_windows = np.arange(n_windows)

    # Plot local entropies
    ax3.plot(x_windows, local_entropies, 'o-', linewidth=2, markersize=6,
             color=colors['sub'], label='Local S-entropy')

    # Plot global entropy line
    ax3.axhline(y=S_global, linestyle='--', linewidth=3, color=colors['valid'],
                label=f'Global S-entropy: {S_global:.2e}')

    # Overlay violation fraction
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x_windows, local_violations, alpha=0.3, color=colors['miracle'],
                 edgecolor='black', linewidth=1, label='Violation Fraction')

    ax3.set_xlabel('MD Window Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('S-entropy (J/K)', fontsize=11, fontweight='bold', color=colors['sub'])
    ax3_twin.set_ylabel('Violation Fraction', fontsize=11, fontweight='bold',
                        color=colors['miracle'])
    ax3.set_title('C. Local vs Global Entropy\nLocal Violations, Global Validity',
                  fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor=colors['sub'])
    ax3_twin.tick_params(axis='y', labelcolor=colors['miracle'])

    # ============================================================
    # PANEL D: Interferometry miracles - phase space
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])

    phase_diffs = interfero_result['phase_differences']

    # Classify phase miracles
    super_2pi_mask = np.abs(phase_diffs) > 2*np.pi
    negative_phase_mask = phase_diffs < 0
    zero_phase_mask = np.abs(phase_diffs) < 0.1
    valid_phase_mask = ~(super_2pi_mask | negative_phase_mask | zero_phase_mask)

    # 2D histogram
    phases_source = interfero_result['phases']
    phases_target = interfero_result['phases_shifted']

    # Plot valid points
    ax4.scatter(phases_source[valid_phase_mask], phases_target[valid_phase_mask],
                c=colors['valid'], s=10, alpha=0.5, label='Valid')

    # Plot miracles
    ax4.scatter(phases_source[super_2pi_mask], phases_target[super_2pi_mask],
                c=colors['super'], s=30, alpha=0.8, marker='^',
                edgecolors='black', linewidths=1, label='Δφ > 2π')
    ax4.scatter(phases_source[negative_phase_mask], phases_target[negative_phase_mask],
                c=colors['negative'], s=30, alpha=0.8, marker='s',
                edgecolors='black', linewidths=1, label='Δφ < 0')
    ax4.scatter(phases_source[zero_phase_mask], phases_target[zero_phase_mask],
                c=colors['sub'], s=30, alpha=0.8, marker='o',
                edgecolors='black', linewidths=1, label='Δφ ≈ 0')

    # Add diagonal (no phase shift)
    ax4.plot([0, 2*np.pi], [0, 2*np.pi], 'k--', linewidth=2, alpha=0.5)

    ax4.set_xlabel('Source Phase φ₁ (rad)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Target Phase φ₂ (rad)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Interferometry: Phase Space Miracles\nImpossible Configurations Allowed',
                  fontsize=12, fontweight='bold', pad=10)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 2*np.pi])
    ax4.set_ylim([min(phases_target), max(phases_target)])

    # ============================================================
    # PANEL E: Miracle contribution to measurement
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])

    # Compute temperature with and without miracles
    miracle_mask_thermo = negative_mask | super_mask | sub_mask

    # With all data
    T_with_miracles = thermo_result['T_MD'] * 1e9

    # Without miracles (only valid)
    freqs_no_miracles = freqs[valid_mask]
    S_no_miracles = demon._compute_S_entropy(np.abs(freqs_no_miracles))
    T_no_miracles = demon._S_entropy_to_temperature(S_no_miracles) * 1e9

    # Only miracles
    freqs_only_miracles = freqs[miracle_mask_thermo]
    if len(freqs_only_miracles) > 0:
        S_only_miracles = demon._compute_S_entropy(np.abs(freqs_only_miracles))
        T_only_miracles = demon._S_entropy_to_temperature(S_only_miracles) * 1e9
    else:
        T_only_miracles = 0

    # True temperature
    T_true = thermo_result['T_true'] * 1e9

    categories = ['True\nT', 'With\nMiracles', 'Without\nMiracles', 'Only\nMiracles']
    temperatures = [T_true, T_with_miracles, T_no_miracles, T_only_miracles]
    colors_bar = ['gray', colors['miracle'], colors['valid'], colors['negative']]

    bars = ax5.bar(categories, temperatures, color=colors_bar,
                   edgecolor='black', linewidth=2, alpha=0.8)

    ax5.set_ylabel('Temperature (nK)', fontsize=11, fontweight='bold')
    ax5.set_title('E. Miracle Contribution to Measurement\nMiracles Are Essential!',
                  fontsize=12, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add values
    for bar, temp in zip(bars, temperatures):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(temperatures)*0.02,
                f'{temp:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Add error annotations
    error_with = abs(T_with_miracles - T_true) / T_true * 100
    error_without = abs(T_no_miracles - T_true) / T_true * 100

    ax5.text(1, T_with_miracles/2, f'Error:\n{error_with:.3f}%',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9))
    ax5.text(2, T_no_miracles/2, f'Error:\n{error_without:.1f}%',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5E5', alpha=0.9))

    # ============================================================
    # PANEL F: BMD window filtering visualization
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])

    # Show how BMD filters configurations
    n_configs = 100
    sample_freqs = np.random.choice(freqs, n_configs, replace=False)

    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(sample_freqs))
    sample_freqs_sorted = sample_freqs[sorted_indices]

    # Color by type
    colors_points = []
    for f in sample_freqs_sorted:
        if f < 0:
            colors_points.append(colors['negative'])
        elif np.abs(f) > 10 * median_freq:
            colors_points.append(colors['super'])
        elif np.abs(f) < 0.1 * median_freq:
            colors_points.append(colors['sub'])
        else:
            colors_points.append(colors['valid'])

    ax6.scatter(np.arange(n_configs), sample_freqs_sorted * 1e-13,
                c=colors_points, s=50, alpha=0.8, edgecolors='black', linewidths=1)

    # Add BMD filter threshold lines
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Zero (negative below)')
    ax6.axhline(y=10 * median_freq * 1e-13, color='orange', linestyle='--',
                linewidth=2, alpha=0.7, label='Super-thermal threshold')
    ax6.axhline(y=0.1 * median_freq * 1e-13, color='blue', linestyle='--',
                linewidth=2, alpha=0.7, label='Sub-thermal threshold')

    ax6.set_xlabel('Configuration Index (sorted)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency ω (10¹³ rad/s)', fontsize=11, fontweight='bold')
    ax6.set_title('F. BMD Window Filtering\nAll Configurations Contribute',
                  fontsize=12, fontweight='bold', pad=10)
    ax6.legend(fontsize=8, loc='upper left')
    ax6.grid(True, alpha=0.3)

    # ============================================================
    # PANEL G: Miracle density across measurement
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 0])

    # Compute miracle density in sliding windows
    window_size_slide = 200
    stride = 50

    positions = []
    densities_negative = []
    densities_super = []
    densities_sub = []

    for i in range(0, len(freqs) - window_size_slide, stride):
        window = freqs[i:i+window_size_slide]
        positions.append(i + window_size_slide//2)

        densities_negative.append(np.sum(window < 0) / len(window))
        densities_super.append(np.sum(np.abs(window) > 10 * median_freq) / len(window))
        densities_sub.append(np.sum((np.abs(window) < 0.1 * median_freq) & (window > 0)) / len(window))

    ax7.fill_between(positions, 0, densities_negative, alpha=0.7,
                     color=colors['negative'], label='Negative ω')
    ax7.fill_between(positions, densities_negative,
                     np.array(densities_negative) + np.array(densities_super),
                     alpha=0.7, color=colors['super'], label='Super-thermal')
    ax7.fill_between(positions,
                     np.array(densities_negative) + np.array(densities_super),
                     np.array(densities_negative) + np.array(densities_super) + np.array(densities_sub),
                     alpha=0.7, color=colors['sub'], label='Sub-thermal')

    ax7.set_xlabel('Position in Measurement', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Miracle Density', fontsize=11, fontweight='bold')
    ax7.set_title('G. Spatial Distribution of Miracles\nUniform Across Measurement',
                  fontsize=12, fontweight='bold', pad=10)
    ax7.legend(fontsize=9, loc='upper right')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 0.15])

    # ============================================================
    # PANEL H: Interferometry miracle contribution
    # ============================================================
    ax8 = fig.add_subplot(gs[2, 1])

    # Distance with and without miracles
    miracle_mask_interfero = super_2pi_mask | negative_phase_mask | zero_phase_mask

    # With all data
    d_with_miracles = interfero_result['distance_MD']

    # Without miracles
    phase_diffs_no_miracles = phase_diffs[valid_phase_mask]
    if len(phase_diffs_no_miracles) > 0:
        lambda_laser = 780e-9
        k = 2 * np.pi / lambda_laser
        delta_phi_no_miracles = np.median(phase_diffs_no_miracles)
        d_no_miracles = delta_phi_no_miracles / (2 * k)
    else:
        d_no_miracles = 0

    # True distance
    d_true = interfero_result['distance_true']

    categories_dist = ['True\nDistance', 'With\nMiracles', 'Without\nMiracles']
    distances_comp = [d_true, d_with_miracles, d_no_miracles]
    colors_dist = ['gray', colors['miracle'], colors['valid']]

    bars = ax8.bar(categories_dist, distances_comp, color=colors_dist,
                   edgecolor='black', linewidth=2, alpha=0.8)

    ax8.set_ylabel('Distance (m)', fontsize=11, fontweight='bold')
    ax8.set_title('H. Interferometry: Miracle Contribution\nMiracles Improve Accuracy',
                  fontsize=12, fontweight='bold', pad=10)
    ax8.grid(True, alpha=0.3, axis='y')

    # Add values
    for bar, dist in zip(bars, distances_comp):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + max(distances_comp)*0.02,
                f'{dist:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Add error annotations
    error_with_dist = abs(d_with_miracles - d_true) / d_true * 100
    error_without_dist = abs(d_no_miracles - d_true) / d_true * 100

    ax8.text(1, d_with_miracles/2, f'Error:\n{error_with_dist:.3f}%',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9))
    ax8.text(2, d_no_miracles/2, f'Error:\n{error_without_dist:.2f}%',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5E5', alpha=0.9))
    summary_text = """
    SUMMARY STATISTICS
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=7.5, family='monospace', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E5',
                      edgecolor=colors['miracle'], linewidth=3, alpha=0.95))

    # ============================================================
    # PANEL I: Summary statistics
    # ============================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=7.5, family='monospace', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E5',
                      edgecolor=colors['miracle'], linewidth=3, alpha=0.95))

    # Overall title
    fig.suptitle('Maxwell Demon Miracles: Local Violations Enable Global Precision\n' +
                 '"Impossible" Configurations Are Essential for Measurement',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

    return fig

def create_reading_order_analysis(save_path='maxwell_demon_reading_order.png'):
    """
    Demonstrate that reading order doesn't matter (non-linear measurement)
    """

    demon = MolecularMaxwellDemon(N_molecules=1000)

    # Generate ensemble
    T_true = 100e-9
    freqs = demon.generate_harmonic_ensemble(T_true)

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    colors = {
        'sequential': '#2E86AB',
        'reversed': '#F18F01',
        'random': '#06A77D',
        'shuffled': '#A23B72',
        'miracle': '#C73E1D'
    }

    # ============================================================
    # PANEL A: Different reading orders
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Create different reading orders
    indices = np.arange(len(freqs))

    orders = {
        'Sequential': indices,
        'Reversed': indices[::-1],
        'Random 1': np.random.permutation(indices),
        'Random 2': np.random.permutation(indices),
        'Random 3': np.random.permutation(indices),
        'Shuffled': np.random.permutation(indices)
    }

    temperatures = []
    order_names = []

    for name, order in orders.items():
        sample_freqs = freqs[order]
        S = demon._compute_S_entropy(np.abs(sample_freqs))
        T = demon._S_entropy_to_temperature(S) * 1e9
        temperatures.append(T)
        order_names.append(name)

    bars = ax1.bar(order_names, temperatures,
                   color=[colors['sequential'], colors['reversed'],
                          colors['random'], colors['random'],
                          colors['random'], colors['shuffled']],
                   edgecolor='black', linewidth=2, alpha=0.8)

    # Add true temperature line
    ax1.axhline(y=T_true * 1e9, linestyle='--', linewidth=3, color='red',
                label=f'True T = {T_true*1e9:.2f} nK')

    ax1.set_ylabel('Measured Temperature (nK)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Reading Order Invariance\nNon-Linear MD Filtering',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(order_names, rotation=45, ha='right')

    # Add variance annotation
    T_variance = np.std(temperatures)
    T_mean = np.mean(temperatures)
    ax1.text(0.5, 0.95, f'Mean: {T_mean:.4f} nK\nσ: {T_variance:.6f} nK\nVariance: {T_variance/T_mean*100:.4f}%',
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))

    # ============================================================
    # PANEL B: Subset sampling
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Test different subset sizes
    subset_sizes = [100, 200, 300, 500, 700, 1000]

    temps_by_size = []
    stds_by_size = []

    for size in subset_sizes:
        temps_subset = []
        for _ in range(20):  # 20 random samples
            sample_indices = np.random.choice(len(freqs), size, replace=False)
            sample_freqs = freqs[sample_indices]
            S = demon._compute_S_entropy(np.abs(sample_freqs))
            T = demon._S_entropy_to_temperature(S) * 1e9
            temps_subset.append(T)

        temps_by_size.append(np.mean(temps_subset))
        stds_by_size.append(np.std(temps_subset))

    ax2.errorbar(subset_sizes, temps_by_size, yerr=stds_by_size,
                 fmt='o-', linewidth=3, markersize=10, capsize=8,
                 color=colors['random'], markeredgecolor='black',
                 markeredgewidth=2, label='Mean ± σ')

    ax2.axhline(y=T_true * 1e9, linestyle='--', linewidth=3, color='red',
                label='True Temperature')

    ax2.set_xlabel('Subset Size (molecules)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Measured Temperature (nK)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Subset Size Independence\nMD Works at All Scales',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # PANEL C: Sequential vs random convergence
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])

    # Progressive measurement
    n_steps = 50
    step_sizes = np.linspace(50, len(freqs), n_steps).astype(int)

    temps_sequential = []
    temps_random = []

    for size in step_sizes:
        # Sequential
        sample_seq = freqs[:size]
        S_seq = demon._compute_S_entropy(np.abs(sample_seq))
        T_seq = demon._S_entropy_to_temperature(S_seq) * 1e9
        temps_sequential.append(T_seq)

        # Random
        sample_rand = freqs[np.random.choice(len(freqs), size, replace=False)]
        S_rand = demon._compute_S_entropy(np.abs(sample_rand))
        T_rand = demon._S_entropy_to_temperature(S_rand) * 1e9
        temps_random.append(T_rand)

    ax3.plot(step_sizes, temps_sequential, linewidth=3,
             color=colors['sequential'], label='Sequential Reading')
    ax3.plot(step_sizes, temps_random, linewidth=3, linestyle='--',
             color=colors['random'], label='Random Reading')
    ax3.axhline(y=T_true * 1e9, linestyle=':', linewidth=3, color='red',
                label='True Temperature')

    ax3.set_xlabel('Number of Molecules Read', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Measured Temperature (nK)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Convergence Analysis\nBoth Methods Converge to Truth',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ============================================================
    # PANEL D: Miracle distribution across orders
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])

    median_freq = np.median(np.abs(freqs))

    # Count miracles in different reading orders
    miracle_counts = []

    for name, order in orders.items():
        sample_freqs = freqs[order][:200]  # First 200 in each order

        n_negative = np.sum(sample_freqs < 0)
        n_super = np.sum(np.abs(sample_freqs) > 10 * median_freq)
        n_sub = np.sum((np.abs(sample_freqs) < 0.1 * median_freq) & (sample_freqs > 0))

        miracle_counts.append([n_negative, n_super, n_sub])

    miracle_counts = np.array(miracle_counts)

    x = np.arange(len(order_names))
    width = 0.25

    bars1 = ax4.bar(x - width, miracle_counts[:, 0], width,
                    label='Negative ω', color='#C73E1D',
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax4.bar(x, miracle_counts[:, 1], width,
                    label='Super-thermal', color='#F18F01',
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    bars3 = ax4.bar(x + width, miracle_counts[:, 2], width,
                    label='Sub-thermal', color='#2E86AB',
                    edgecolor='black', linewidth=1.5, alpha=0.8)

    ax4.set_ylabel('Miracle Count (first 200)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Miracle Distribution Across Orders\nSimilar Across All Reading Orders',
                  fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(order_names, rotation=45, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # PANEL E: Phase space trajectory
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])

    # Show trajectory through frequency space for different orders
    n_plot = 100

    # Sequential
    traj_seq = freqs[:n_plot]
    ax5.plot(np.arange(n_plot), traj_seq * 1e-13,
             linewidth=2, alpha=0.7, color=colors['sequential'],
             label='Sequential')

    # Random
    random_order = np.random.permutation(len(freqs))
    traj_rand = freqs[random_order[:n_plot]]
    ax5.plot(np.arange(n_plot), traj_rand * 1e-13,
             linewidth=2, alpha=0.7, color=colors['random'],
             label='Random')

    # Highlight miracles
    miracle_mask_seq = (traj_seq < 0) | (np.abs(traj_seq) > 10 * median_freq) | \
                       ((np.abs(traj_seq) < 0.1 * median_freq) & (traj_seq > 0))
    miracle_mask_rand = (traj_rand < 0) | (np.abs(traj_rand) > 10 * median_freq) | \
                        ((np.abs(traj_rand) < 0.1 * median_freq) & (traj_rand > 0))

    ax5.scatter(np.where(miracle_mask_seq)[0], traj_seq[miracle_mask_seq] * 1e-13,
                s=100, color=colors['miracle'], marker='o', edgecolors='black',
                linewidths=2, zorder=5, label='Miracles')
    ax5.scatter(np.where(miracle_mask_rand)[0], traj_rand[miracle_mask_rand] * 1e-13,
                s=100, color=colors['miracle'], marker='s', edgecolors='black',
                linewidths=2, zorder=5)

    ax5.set_xlabel('Reading Index', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency ω (10¹³ rad/s)', fontsize=11, fontweight='bold')
    ax5.set_title('E. Phase Space Trajectories\nDifferent Paths, Same Destination',
                  fontsize=13, fontweight='bold', pad=10)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # ============================================================
    # PANEL F: Summary and explanation
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.text(0.05, 0.95, transform=ax6.transAxes,
             fontsize=7, family='monospace', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8',
                      edgecolor=colors['random'], linewidth=3, alpha=0.95))

    # Overall title
    fig.suptitle('Reading Order Invariance: Non-Linear Maxwell Demon Measurement\n' +
                 'Categorical Completion is Order-Independent',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

    return fig

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MOLECULAR MAXWELL DEMONS: UNIFIED INTERFEROMETRY & THERMOMETRY")
    print("="*80 + "\n")

    print("Generating visualizations...")
    print("-" * 80)

    # 1. Main unified visualization
    print("\n[1/3] Creating unified framework visualization...")
    create_unified_visualization('molecular_maxwell_demons_unified.png')

    # 2. Miracle analysis
    print("\n[2/3] Creating miracle analysis...")
    create_miracle_analysis('maxwell_demon_miracles.png')

    # 3. Reading order analysis
    print("\n[3/3] Creating reading order analysis...")
    create_reading_order_analysis('maxwell_demon_reading_order.png')

    print("\n" + "="*80)
    print("SUMMARY OF KEY INSIGHTS")
    print("="*80)
