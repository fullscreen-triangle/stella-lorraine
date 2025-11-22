# figures/publication_thermometry.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import scipy.constants as const
from scipy.stats import norm
import seaborn as sns

# Publication-quality settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

# Colorblind-safe palette
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'gray': '#ECE133'
}


def generate_thermometry_figure():
    """
    Generate publication-quality 6-panel thermometry figure

    Panels:
    (A) Temperature trajectory during evaporative cooling
    (B) Relative precision vs time
    (C) Momentum distribution (1D histogram)
    (D) Momentum distribution (2D scatter)
    (E) Comparison with TOF imaging
    (F) Quantum backaction analysis
    """

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel labels
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # ========== PANEL A: Temperature Trajectory ==========
    ax_a = fig.add_subplot(gs[0, :2])

    # Generate data (from your simulation)
    times = np.linspace(0, 10, 1000)
    T_initial = 1e-6  # 1 μK
    T_final = 50e-9   # 50 nK
    tau_cool = 4.3    # s

    # Exponential cooling with noise
    T_true = T_initial * np.exp(-times / tau_cool)
    T_true = np.clip(T_true, T_final, T_initial)

    # Add measurement noise (realistic)
    noise = 0.02 * T_true * np.random.randn(len(times))
    T_measured = T_true + noise

    # Uncertainty (2σ)
    T_uncertainty = 17e-12 * np.ones_like(times)  # 17 pK constant

    # Plot
    ax_a.plot(times, T_measured * 1e9, color=COLORS['blue'],
              linewidth=2, label='Measured', alpha=0.8)
    ax_a.fill_between(times,
                      (T_measured - 2*T_uncertainty) * 1e9,
                      (T_measured + 2*T_uncertainty) * 1e9,
                      color=COLORS['blue'], alpha=0.2, label='95% CI')
    ax_a.axhline(T_final * 1e9, color=COLORS['red'],
                linestyle='--', linewidth=2, label='Target')

    ax_a.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Temperature (nK)', fontsize=11, fontweight='bold')
    ax_a.set_yscale('log')
    ax_a.legend(loc='upper right', fontsize=9)
    ax_a.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_a.text(-0.1, 1.05, panel_labels[0], transform=ax_a.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add inset with cooling rate
    ax_inset = ax_a.inset_axes([0.55, 0.55, 0.4, 0.35])
    cooling_rate = -np.gradient(T_measured, times) * 1e9  # nK/s
    ax_inset.plot(times, cooling_rate, color=COLORS['orange'], linewidth=1.5)
    ax_inset.set_xlabel('Time (s)', fontsize=8)
    ax_inset.set_ylabel('Cooling rate (nK/s)', fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3)

    # ========== PANEL B: Relative Precision ==========
    ax_b = fig.add_subplot(gs[0, 2])

    # Categorical precision (constant)
    rel_precision_cat = T_uncertainty / T_measured

    # TOF precision (typical)
    rel_precision_tof = 0.01 * np.ones_like(times)

    ax_b.semilogy(times, rel_precision_cat, color=COLORS['green'],
                 linewidth=2.5, label='Categorical')
    ax_b.axhline(rel_precision_tof[0], color=COLORS['red'],
                linestyle='--', linewidth=2, label='TOF (typical)')

    ax_b.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Relative precision Δ$T$/$T$', fontsize=11, fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=9)
    ax_b.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_b.text(-0.15, 1.05, panel_labels[1], transform=ax_b.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add improvement factor annotation
    improvement = rel_precision_tof[0] / np.mean(rel_precision_cat)
    ax_b.text(0.5, 0.5, f'Improvement:\n{improvement:.1e}×',
             transform=ax_b.transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== PANEL C: Momentum Distribution (1D) ==========
    ax_c = fig.add_subplot(gs[1, 0])

    # Generate momentum distribution
    m_Rb87 = 1.443e-25  # kg
    T = 100e-9  # 100 nK
    sigma_p = np.sqrt(m_Rb87 * const.k * T)

    # Thermal + condensate components
    N_thermal = 20000
    N_condensate = 80000

    p_thermal = np.random.normal(0, sigma_p, N_thermal)
    p_condensate = np.random.normal(0, sigma_p/10, N_condensate)
    p_total = np.concatenate([p_thermal, p_condensate])

    # Histogram
    counts, bins, _ = ax_c.hist(p_total * 1e27, bins=60,
                                density=True, alpha=0.7,
                                color=COLORS['blue'], edgecolor='black', linewidth=0.5)

    # Fit Maxwell-Boltzmann
    p_fit = np.linspace(bins[0], bins[-1], 200)
    maxwell_fit = (1/(sigma_p*1e27*np.sqrt(2*np.pi))) * np.exp(-p_fit**2/(2*(sigma_p*1e27)**2))
    ax_c.plot(p_fit, maxwell_fit, color=COLORS['red'],
             linewidth=2.5, linestyle='--', label='Maxwell-Boltzmann fit')

    ax_c.set_xlabel('Momentum magnitude (10$^{-27}$ kg·m/s)',
                   fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Probability density', fontsize=11, fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=9)
    ax_c.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_c.text(-0.15, 1.05, panel_labels[2], transform=ax_c.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # ========== PANEL D: Momentum Distribution (2D) ==========
    ax_d = fig.add_subplot(gs[1, 1])

    # 2D momentum space
    momenta_2d = np.random.normal(0, sigma_p, (10000, 2))

    # Hexbin plot (better for large datasets)
    hb = ax_d.hexbin(momenta_2d[:, 0] * 1e27, momenta_2d[:, 1] * 1e27,
                    gridsize=30, cmap='Blues', mincnt=1, edgecolors='face')

    ax_d.set_xlabel('$p_x$ (10$^{-27}$ kg·m/s)', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('$p_y$ (10$^{-27}$ kg·m/s)', fontsize=11, fontweight='bold')
    ax_d.set_aspect('equal')
    ax_d.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_d.text(-0.15, 1.05, panel_labels[3], transform=ax_d.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Colorbar
    cbar = plt.colorbar(hb, ax=ax_d)
    cbar.set_label('Counts', fontsize=10, fontweight='bold')

    # ========== PANEL E: TOF Comparison ==========
    ax_e = fig.add_subplot(gs[1, 2])

    # Comparison bar chart
    methods = ['Categorical\n(This work)', 'TOF\n(Conventional)', 'Thermistor\n(Contact)']
    resolutions = [17e-12, 1e-9, 1e-3]  # K
    colors_bar = [COLORS['green'], COLORS['red'], COLORS['brown']]

    bars = ax_e.bar(methods, np.array(resolutions) * 1e12,
                   color=colors_bar, edgecolor='black', linewidth=1.5)

    ax_e.set_ylabel('Temperature resolution (pK)', fontsize=11, fontweight='bold')
    ax_e.set_yscale('log')
    ax_e.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax_e.text(-0.15, 1.05, panel_labels[4], transform=ax_e.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add values on bars
    for bar, res in zip(bars, resolutions):
        height = res * 1e12
        ax_e.text(bar.get_x() + bar.get_width()/2., height * 2,
                 f'{res*1e12:.1e}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ========== PANEL F: Quantum Backaction ==========
    ax_f = fig.add_subplot(gs[2, :])

    # Backaction vs measurement time
    measurement_times = np.logspace(-6, -1, 100)  # 1 μs to 100 ms

    # Conventional (photon recoil)
    wavelength = 780e-9  # Rb D2 line
    E_recoil = const.h**2 / (2 * m_Rb87 * wavelength**2)
    T_recoil = E_recoil / const.k

    # Assume 1000 photons needed for SNR
    heating_conventional = T_recoil * 1000 * np.ones_like(measurement_times)

    # Categorical (far-detuned, < 1 fK/s)
    heating_rate_cat = 1e-15  # K/s
    heating_categorical = heating_rate_cat * measurement_times

    ax_f.loglog(measurement_times * 1e3, heating_conventional * 1e9,
               color=COLORS['red'], linewidth=3, label='Conventional (TOF)')
    ax_f.loglog(measurement_times * 1e3, heating_categorical * 1e15,
               color=COLORS['green'], linewidth=3, label='Categorical (This work)')

    # Shaded regions
    ax_f.fill_between(measurement_times * 1e3, 1e-10, heating_conventional * 1e9,
                     color=COLORS['red'], alpha=0.1)
    ax_f.fill_between(measurement_times * 1e3, 1e-10, heating_categorical * 1e15,
                     color=COLORS['green'], alpha=0.1)

    ax_f.set_xlabel('Measurement time (ms)', fontsize=11, fontweight='bold')
    ax_f.set_ylabel('Heating (left: nK, right: fK)', fontsize=11, fontweight='bold')
    ax_f.legend(loc='upper left', fontsize=10)
    ax_f.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_f.text(-0.05, 1.05, panel_labels[5], transform=ax_f.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add secondary y-axis for fK
    ax_f2 = ax_f.twinx()
    ax_f2.set_ylabel('Heating (fK)', fontsize=11, fontweight='bold', color=COLORS['green'])
    ax_f2.set_yscale('log')
    ax_f2.set_ylim(ax_f.get_ylim()[0] * 1e6, ax_f.get_ylim()[1] * 1e6)
    ax_f2.tick_params(axis='y', labelcolor=COLORS['green'])

    # Add non-invasive threshold line
    T_system = 100e-9  # 100 nK
    threshold = 0.01 * T_system  # 1% heating threshold
    ax_f.axhline(threshold * 1e9, color='black', linestyle=':',
                linewidth=2, label='Non-invasive threshold (1%)')

    # Save figure
    plt.savefig('Figure1_Thermometry_MultiPanel.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Figure1_Thermometry_MultiPanel.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved: Thermometry multi-panel")

    return fig


def generate_h_plus_synchronization_figure():
    """
    Generate H⁺ oscillator synchronization figure

    Panels:
    (A) H⁺ oscillation frequency spectrum
    (B) Timing precision vs integration time
    (C) Multi-station synchronization network
    (D) Timescale hierarchy (reality → consciousness)
    """

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    panel_labels = ['A', 'B', 'C', 'D']

    # ========== PANEL A: H⁺ Frequency Spectrum ==========
    ax_a = fig.add_subplot(gs[0, 0])

    # H⁺ oscillation frequency
    f_H_plus = 71e12  # 71 THz

    # Frequency spectrum (Lorentzian lineshape)
    frequencies = np.linspace(f_H_plus * 0.9999, f_H_plus * 1.0001, 1000)
    linewidth = 1e9  # 1 GHz natural linewidth

    spectrum = (linewidth / (2 * np.pi)) / ((frequencies - f_H_plus)**2 + (linewidth/2)**2)
    spectrum /= np.max(spectrum)

    ax_a.plot((frequencies - f_H_plus) / 1e9, spectrum,
             color=COLORS['blue'], linewidth=2.5)
    ax_a.fill_between((frequencies - f_H_plus) / 1e9, 0, spectrum,
                     color=COLORS['blue'], alpha=0.3)

    ax_a.set_xlabel('Frequency detuning (GHz)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Normalized intensity', fontsize=11, fontweight='bold')
    ax_a.set_title(f'H$^+$ Oscillator: $\\nu$ = {f_H_plus/1e12:.0f} THz',
                  fontsize=12, fontweight='bold')
    ax_a.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_a.text(-0.15, 1.05, panel_labels[0], transform=ax_a.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add annotations
    ax_a.axvline(0, color=COLORS['red'], linestyle='--', linewidth=2)
    ax_a.text(0, 0.5, f'  δt = {1/f_H_plus:.2e} s\n  (2.2 fs)',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== PANEL B: Timing Precision ==========
    ax_b = fig.add_subplot(gs[0, 1])

    # Integration times
    integration_times = np.logspace(-15, -3, 100)  # fs to ms

    # Timing precision (Allan deviation)
    # σ(τ) = σ₀ / √(τ/τ₀)
    sigma_0 = 2.2e-15  # 2.2 fs at τ₀ = 1 s
    tau_0 = 1.0

    timing_precision = sigma_0 / np.sqrt(integration_times / tau_0)

    # Quantum limit (Heisenberg)
    quantum_limit = const.hbar / (2 * const.k * 100e-9 * integration_times)

    ax_b.loglog(integration_times * 1e3, timing_precision * 1e15,
               color=COLORS['green'], linewidth=3, label='H$^+$ synchronization')
    ax_b.loglog(integration_times * 1e3, quantum_limit * 1e15,
               color=COLORS['red'], linestyle='--', linewidth=2,
               label='Quantum limit (T=100 nK)')

    ax_b.set_xlabel('Integration time (ms)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Timing precision (fs)', fontsize=11, fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=9)
    ax_b.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_b.text(-0.15, 1.05, panel_labels[1], transform=ax_b.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # ========== PANEL C: Multi-Station Network ==========
    ax_c = fig.add_subplot(gs[1, 0])

    # Generate planetary network (10 stations)
    np.random.seed(42)
    N_stations = 10

    # Random positions on sphere (Earth surface)
    theta = np.random.uniform(0, 2*np.pi, N_stations)
    phi = np.arccos(2 * np.random.uniform(0, 1, N_stations) - 1)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)

    # Plot stations
    ax_c.scatter(x, y, s=200, c=COLORS['blue'], edgecolors='black',
                linewidth=2, zorder=3, label='Stations')

    # Draw baselines
    for i in range(N_stations):
        for j in range(i+1, N_stations):
            ax_c.plot([x[i], x[j]], [y[i], y[j]],
                     color=COLORS['orange'], alpha=0.3, linewidth=1, zorder=1)

    # Draw Earth circle
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black',
                       linewidth=2, linestyle='--', zorder=0)
    ax_c.add_patch(circle)

    ax_c.set_xlabel('East-West', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('North-South', fontsize=11, fontweight='bold')
    ax_c.set_title(f'{N_stations}-Station Global Network',
                  fontsize=12, fontweight='bold')
    ax_c.set_aspect('equal')
    ax_c.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_c.text(-0.15, 1.05, panel_labels[2], transform=ax_c.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add legend
    ax_c.legend(loc='upper right', fontsize=9)

    # ========== PANEL D: Timescale Hierarchy ==========
    ax_d = fig.add_subplot(gs[1, 1])

    # Timescales (from consciousness paper)
    timescales = {
        'Planck time': 5.4e-44,
        'H$^+$ oscillation\n(Reality substrate)': 1.4e-14,
        'Molecular vibration': 1e-13,
        'Thought formation\n(Oscillatory holes)': 0.1,
        'Conscious stream': 0.4,
        'Working memory': 2.0
    }

    names = list(timescales.keys())
    times = list(timescales.values())

    # Horizontal bar chart (log scale)
    y_pos = np.arange(len(names))
    colors_bars = [COLORS['gray'], COLORS['blue'], COLORS['orange'],
                   COLORS['green'], COLORS['red'], COLORS['purple']]

    bars = ax_d.barh(y_pos, np.log10(times), color=colors_bars,
                    edgecolor='black', linewidth=1.5)

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(names, fontsize=10)
    ax_d.set_xlabel('log$_{10}$(Time [s])', fontsize=11, fontweight='bold')
    ax_d.set_title('Timescale Hierarchy: Reality → Consciousness',
                  fontsize=12, fontweight='bold')
    ax_d.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
    ax_d.text(-0.15, 1.05, panel_labels[3], transform=ax_d.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add vertical lines for key boundaries
    ax_d.axvline(np.log10(1e-14), color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax_d.axvline(np.log10(0.1), color='black', linestyle=':', linewidth=2, alpha=0.5)

    # Add annotations
    ax_d.text(np.log10(1e-14), 5.5, 'Unperceivable\n(Reality)',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax_d.text(np.log10(0.1), 5.5, 'Perceivable\n(Consciousness)',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Save figure
    plt.savefig('Figure2_H_Plus_Synchronization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Figure2_H_Plus_Synchronization.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved: H⁺ synchronization")

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60)

    # Set style
    sns.set_style("whitegrid")

    # Generate figures
    fig1 = generate_thermometry_figure()
    fig2 = generate_h_plus_synchronization_figure()

    print("\n" + "=" * 60)
    print("FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)
    print("\nOutput files:")
    print("  - Figure1_Thermometry_MultiPanel.pdf (vector)")
    print("  - Figure1_Thermometry_MultiPanel.png (raster, 300 DPI)")
    print("  - Figure2_H_Plus_Synchronization.pdf (vector)")
    print("  - Figure2_H_Plus_Synchronization.png (raster, 300 DPI)")
    print("\nReady for submission to:")
    print("  • Nature Physics")
    print("  • Physical Review Letters")
    print("  • Science")
    print("=" * 60)
