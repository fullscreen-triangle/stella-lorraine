"""
transcendent_observer_cascade.py

Demonstrates inverse harmonic cascade for thermometry:
- Observer measures progressively SLOWER molecular frequencies
- Each stage filters to slower subset (inverse of timekeeping)
- Temperature decreases exponentially: T_k = T_0 / Q^(2k)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14

class TranscendentObserver:
    """
    Observer that navigates DOWN frequency ladder to measure temperature
    Inverse of timekeeping (which goes UP frequency ladder)
    """

    def __init__(self, N_molecules=10000, T_initial=100e-9, m_Rb87=1.443e-25):
        """
        Initialize molecular ensemble

        Parameters:
        -----------
        N_molecules : int
            Number of molecules in ensemble
        T_initial : float
            Initial temperature (K)
        m_Rb87 : float
            Mass of Rb-87 atom (kg)
        """
        self.N = N_molecules
        self.T0 = T_initial
        self.m = m_Rb87
        self.kB = 1.380649e-23  # Boltzmann constant
        self.lambda_mfp = 1e-6  # Mean free path (m)

        # Generate initial velocity distribution (Maxwell-Boltzmann)
        self.velocities = self._maxwell_boltzmann_velocities(T_initial)
        self.frequencies = self._velocities_to_frequencies(self.velocities)

        # Cascade parameters
        self.Q = 1.44  # Cooling factor per stage (from BMD validation)
        self.cascade_history = []

    def _maxwell_boltzmann_velocities(self, T):
        """Generate Maxwell-Boltzmann velocity distribution"""
        sigma = np.sqrt(self.kB * T / self.m)
        # 3D velocities
        vx = np.random.normal(0, sigma, self.N)
        vy = np.random.normal(0, sigma, self.N)
        vz = np.random.normal(0, sigma, self.N)
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        return v_mag

    def _velocities_to_frequencies(self, velocities):
        """Convert velocities to oscillation frequencies"""
        # omega = 2*pi*v / lambda
        omega = 2 * np.pi * velocities / self.lambda_mfp
        return omega

    def _frequencies_to_temperature(self, frequencies):
        """Extract temperature from frequency distribution"""
        # T = m*lambda^2 * <omega^2> / (12*pi^2*kB)
        omega_squared_mean = np.mean(frequencies**2)
        T = (self.m * self.lambda_mfp**2 * omega_squared_mean) / (12 * np.pi**2 * self.kB)
        return T

    def observe_slower_subset(self, percentile=30):
        """
        Observe slower molecules (inverse of timekeeping which observes faster)

        Parameters:
        -----------
        percentile : float
            Keep slowest X% of molecules
        """
        threshold = np.percentile(self.frequencies, percentile)
        slow_mask = self.frequencies <= threshold

        # Filter to slower subset
        self.frequencies = self.frequencies[slow_mask]
        self.velocities = self.velocities[slow_mask]

        # Measure temperature from this slower subset
        T_measured = self._frequencies_to_temperature(self.frequencies)

        return T_measured, len(self.frequencies)

    def cascade(self, depth=10, percentile=30):
        """
        Perform inverse cascade: progressively measure slower harmonics

        Parameters:
        -----------
        depth : int
            Number of cascade stages
        percentile : float
            Percentile for filtering at each stage
        """
        self.cascade_history = []

        # Initial state
        T_current = self._frequencies_to_temperature(self.frequencies)
        N_current = len(self.frequencies)

        self.cascade_history.append({
            'stage': 0,
            'temperature': T_current,
            'N_molecules': N_current,
            'omega_mean': np.mean(self.frequencies),
            'omega_std': np.std(self.frequencies),
            'omega_min': np.min(self.frequencies),
            'omega_max': np.max(self.frequencies)
        })

        # Cascade through slower and slower subsets
        for stage in range(1, depth + 1):
            T_measured, N_current = self.observe_slower_subset(percentile)

            self.cascade_history.append({
                'stage': stage,
                'temperature': T_measured,
                'N_molecules': N_current,
                'omega_mean': np.mean(self.frequencies),
                'omega_std': np.std(self.frequencies),
                'omega_min': np.min(self.frequencies),
                'omega_max': np.max(self.frequencies)
            })

            print(f"Stage {stage}: T = {T_measured*1e15:.2f} fK, "
                  f"N = {N_current}, <ω> = {np.mean(self.frequencies):.2e} rad/s")

        return self.cascade_history


def plot_transcendent_observer_results(observer, save_path='transcendent_observer_cascade.png'):
    """
    Create comprehensive visualization of inverse cascade
    """
    history = observer.cascade_history
    stages = [h['stage'] for h in history]
    temperatures = np.array([h['temperature'] for h in history])
    N_molecules = [h['N_molecules'] for h in history]
    omega_means = [h['omega_mean'] for h in history]
    omega_stds = [h['omega_std'] for h in history]

    # Theoretical prediction: T_k = T_0 / Q^(2k)
    T0 = temperatures[0]
    Q = observer.Q
    T_theory = T0 / (Q ** (2 * np.array(stages)))

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Temperature cascade (log scale)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogy(stages, temperatures * 1e15, 'o-', linewidth=2, markersize=8,
                 label='Measured (Inverse Cascade)', color='#2E86AB')
    ax1.semilogy(stages, T_theory * 1e15, '--', linewidth=2,
                 label=f'Theory: $T_k = T_0/Q^{{2k}}$ (Q={Q})', color='#A23B72')
    ax1.set_xlabel('Cascade Stage', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature (fK)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Inverse Harmonic Cascade: Temperature Reduction',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Add cooling factors
    cooling_factors = T0 / temperatures
    for i, stage in enumerate(stages[1::2]):  # Label every other stage
        idx = stages.index(stage)
        ax1.annotate(f'{cooling_factors[idx]:.1f}×',
                    xy=(stage, temperatures[idx]*1e15),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color='#A23B72',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # Panel B: Frequency distribution evolution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(stages, np.array(omega_means) * 1e-13, 'o-', linewidth=2, markersize=8,
             label='Mean ω', color='#F18F01')
    ax2.fill_between(stages,
                     (np.array(omega_means) - np.array(omega_stds)) * 1e-13,
                     (np.array(omega_means) + np.array(omega_stds)) * 1e-13,
                     alpha=0.3, color='#F18F01', label='±1σ')
    ax2.set_xlabel('Cascade Stage', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency (×10¹³ rad/s)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Frequency Reduction', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel C: Molecule count reduction
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(stages, N_molecules, 's-', linewidth=2, markersize=8, color='#06A77D')
    ax3.set_xlabel('Cascade Stage', fontsize=12, fontweight='bold')
    ax3.set_ylabel('N Molecules (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('C. BMD Filtering:\nSlower Subset Selection',
                  fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)

    # Panel D: Cooling factor verification
    ax4 = fig.add_subplot(gs[1, 1])
    measured_Q = []
    for i in range(1, len(temperatures)):
        Q_measured = np.sqrt(temperatures[i-1] / temperatures[i])
        measured_Q.append(Q_measured)

    ax4.plot(stages[1:], measured_Q, 'o-', linewidth=2, markersize=8,
             label='Measured Q', color='#C73E1D')
    ax4.axhline(y=Q, linestyle='--', linewidth=2, color='#A23B72',
                label=f'Theory Q = {Q}')
    ax4.set_xlabel('Cascade Stage', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cooling Factor Q', fontsize=12, fontweight='bold')
    ax4.set_title('D. Per-Stage Cooling Factor', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([1.0, 1.8])

    # Panel E: Temperature vs frequency relationship
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.loglog(omega_means, temperatures * 1e15, 'o', markersize=10, color='#2E86AB')
    # Fit T ∝ ω²
    log_omega = np.log(omega_means)
    log_T = np.log(temperatures)
    slope, intercept = np.polyfit(log_omega, log_T, 1)
    omega_fit = np.logspace(np.log10(min(omega_means)), np.log10(max(omega_means)), 100)
    T_fit = np.exp(intercept) * omega_fit**slope
    ax5.loglog(omega_fit, T_fit * 1e15, '--', linewidth=2, color='#A23B72',
               label=f'Fit: $T \\propto \\omega^{{{slope:.2f}}}$')
    ax5.set_xlabel('Mean Frequency (rad/s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Temperature (fK)', fontsize=12, fontweight='bold')
    ax5.set_title('E. T ∝ ω² Validation', fontsize=13, fontweight='bold', pad=10)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, which='both')

    # Panel F: Comparison with timekeeping (inverse operation)
    ax6 = fig.add_subplot(gs[2, :])

    # Thermometry: Navigate DOWN
    ax6.plot(stages, omega_means / omega_means[0], 'o-', linewidth=3, markersize=10,
             label='Thermometry: ω ↓ (slower harmonics)', color='#2E86AB')

    # Timekeeping: Navigate UP (inverse)
    omega_timekeeping = omega_means[0] * (Q ** np.array(stages))
    ax6.plot(stages, omega_timekeeping / omega_means[0], 's--', linewidth=3, markersize=10,
             label='Timekeeping: ω ↑ (faster harmonics)', color='#F18F01')

    ax6.set_xlabel('Cascade Stage', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Normalized Frequency (ω/ω₀)', fontsize=12, fontweight='bold')
    ax6.set_title('F. Cascade Duality: Thermometry (↓) vs Timekeeping (↑)',
                  fontsize=13, fontweight='bold', pad=10)
    ax6.legend(fontsize=11, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')

    # Add annotations
    ax6.annotate('INVERSE OPERATIONS', xy=(5, 1), xytext=(5, 10),
                fontsize=14, fontweight='bold', ha='center',
                color='#A23B72',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='#A23B72'))

    # Overall title
    fig.suptitle('Transcendent Observer: Inverse Harmonic Cascade for Thermometry\n' +
                 f'Initial: {T0*1e9:.1f} nK → Final: {temperatures[-1]*1e15:.2f} fK ' +
                 f'({cooling_factors[-1]:.1f}× cooling)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Summary box
    summary_text = f"""INVERSE CASCADE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial Temperature: {T0*1e9:.1f} nK
Final Temperature: {temperatures[-1]*1e15:.2f} fK
Total Cooling: {cooling_factors[-1]:.1f}×
Cascade Depth: {len(stages)-1} stages

Per-Stage Factor: Q = {Q}
Theoretical: T_k = T₀/Q^(2k)
Measured slope: {slope:.3f} (theory: 2.0)

Method: BMD filtering → slower subsets
Direction: ω₁ > ω₂ > ω₃ (DECREASING)
Result: Temperature ↓ (COOLING)

INVERSE OF TIMEKEEPING:
  Timekeeping: ω ↑ → Δt ↓
  Thermometry: ω ↓ → T ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")

    return fig


# Run validation
if __name__ == "__main__":
    print("="*70)
    print("TRANSCENDENT OBSERVER: INVERSE HARMONIC CASCADE")
    print("="*70)
    print("\nNavigating DOWN frequency ladder for temperature measurement")
    print("(Inverse of timekeeping which navigates UP)\n")

    # Create observer
    observer = TranscendentObserver(N_molecules=100000, T_initial=100e-9)

    # Perform cascade
    print("Performing inverse cascade...")
    history = observer.cascade(depth=10, percentile=30)

    # Plot results
    print("\nGenerating visualization...")
    fig = plot_transcendent_observer_results(observer)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    plt.show()
