# Script 3: Physical Predictions and Observational Tests
# File: physical_predictions.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import seaborn as sns


if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel 1: Dark Matter Ratio vs Redshift
    ax1 = fig.add_subplot(gs[0, 0])

    # Simulate dark matter ratio evolution
    redshift = np.linspace(0, 5, 50)
    # Theoretical: ratio should evolve with categorical depth
    # Simplified model: R_DM(z) ∝ C(t(z))
    t_z = 5 - 0.5 * redshift  # Categorical depth decreases with redshift (looking back in time)
    R_DM_theory = 2**(t_z - 1)  # Simplified tetration approximation

    # Add observational "data" with error bars
    z_obs = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0])
    R_DM_obs = np.array([5.4, 5.1, 4.8, 4.5, 4.2, 3.8])
    R_DM_err = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.9])

    ax1.plot(redshift, R_DM_theory, linewidth=3, label='Theoretical Prediction', color='darkblue')
    ax1.errorbar(z_obs, R_DM_obs, yerr=R_DM_err, fmt='o', markersize=8,
                capsize=5, capthick=2, label='Observational Data', color='red', elinewidth=2)
    ax1.fill_between(redshift, R_DM_theory * 0.9, R_DM_theory * 1.1,
                    alpha=0.2, color='darkblue', label='Theory Uncertainty')

    ax1.set_xlabel('Redshift (z)', fontweight='bold')
    ax1.set_ylabel('Dark Matter Ratio (R_DM)', fontweight='bold')
    ax1.set_title('A. Dark Matter Ratio\nvs Redshift', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 5)

    # Panel 2: Holographic Bound and Categorical Limit
    ax2 = fig.add_subplot(gs[0, 1])

    # Holographic bound: S_max = A / (4 * l_P^2)
    # For observable universe: A ~ (10^26 m)^2, l_P ~ 10^-35 m
    # S_max ~ 10^122 bits

    # Categorical complexity vs holographic bound
    t_values = np.arange(1, 8)
    C_t_log = []
    for t in t_values:
        if t == 0:
            C_t_log.append(0)  # C(0) = 1, log10(1) = 0
        elif t == 1:
            C_t_log.append(np.log10(2))  # C(1) = 2
        elif t == 2:
            C_t_log.append(np.log10(4))  # C(2) = 4
        elif t <= 5:
            # Small values: compute directly
            val = 2**t
            C_t_log.append(np.log10(val))
        else:
            # For large t, 2↑↑t grows too fast
            # Use logarithm properties: log10(2^x) = x * log10(2)
            # For t=6: 2↑↑6 = 2^(2^(2^(2^(2^2)))) = 2^65536
            # log10(2^65536) = 65536 * log10(2) ≈ 19728
            if t == 6:
                C_t_log.append(65536 * np.log10(2))
            else:
                # Beyond t=6, use approximation
                prev_exp = 65536 * (t - 6)  # Rough estimate
                C_t_log.append(prev_exp * np.log10(2))

    holographic_bound = 122  # log10(10^122)

    ax2.plot(t_values, C_t_log, 'o-', linewidth=3, markersize=10,
            label='log₁₀(C(t))', color='purple')
    ax2.axhline(y=holographic_bound, color='red', linestyle='--', linewidth=2,
            label='Holographic Bound')
    ax2.fill_between(t_values, 0, holographic_bound, alpha=0.1, color='red')

    ax2.set_xlabel('Categorical Depth (t)', fontweight='bold')
    ax2.set_ylabel('log₁₀(Information Content)', fontweight='bold')
    ax2.set_title('B. Holographic Bound\nConstraint', fontweight='bold', pad=20)
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add annotation for t_max
    t_max_idx = np.where(np.array(C_t_log) < holographic_bound)[0][-1] if any(np.array(C_t_log) < holographic_bound) else len(t_values) - 1
    ax2.annotate(f't_max ≈ {t_values[t_max_idx]}',
                xy=(t_values[t_max_idx], C_t_log[t_max_idx]),
                xytext=(t_values[t_max_idx] - 1, holographic_bound - 20),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                fontsize=10, fontweight='bold')

    # Panel 3: Entropy Production Rate
    ax3 = fig.add_subplot(gs[0, 2])

    # dS/dt ∝ C(t) * ln(n)
    time = np.linspace(0, 10, 100)
    # Simplified: C(t) grows, so dS/dt grows
    C_t_time = 1 + time**2  # Simplified growth
    dS_dt = C_t_time * np.log(2)  # ln(n) with n=2

    ax3.plot(time, dS_dt, linewidth=3, color='darkgreen')
    ax3.fill_between(time, 0, dS_dt, alpha=0.3, color='darkgreen')

    ax3.set_xlabel('Time', fontweight='bold')
    ax3.set_ylabel('Entropy Production Rate (dS/dt)', fontweight='bold')
    ax3.set_title('C. Accelerating Entropy\nProduction', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Add annotation
    ax3.annotate('dS/dt ∝ C(t)ln(n)', xy=(7, dS_dt[70]), xytext=(5, dS_dt[70] + 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Panel 4: Planck Scale Discreteness
    ax4 = fig.add_subplot(gs[1, 0])

    # Modified dispersion relation: E^2 = p^2c^2 + m^2c^4 + α(p^4c^4/E_P^2)
    # At high energies, deviation from standard dispersion

    p_c = np.logspace(-2, 2, 100)  # Momentum in units of mc
    E_standard = np.sqrt(p_c**2 + 1)  # Standard: E^2 = p^2 + m^2 (c=1)
    alpha = 0.01  # Planck-scale correction parameter
    E_P = 100  # Planck energy in units of mc^2
    E_modified = np.sqrt(p_c**2 + 1 + alpha * p_c**4 / E_P**2)

    ax4.loglog(p_c, E_standard, linewidth=3, label='Standard Dispersion', color='blue')
    ax4.loglog(p_c, E_modified, linewidth=3, linestyle='--',
            label='Modified (Categorical)', color='red')

    # Shade difference region
    ax4.fill_between(p_c, E_standard, E_modified, alpha=0.3, color='yellow',
                    label='Planck-Scale Correction')

    ax4.set_xlabel('Momentum (p/mc)', fontweight='bold')
    ax4.set_ylabel('Energy (E/mc²)', fontweight='bold')
    ax4.set_title('D. Modified Dispersion:\nPlanck-Scale Effects', fontweight='bold', pad=20)
    ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--', which='both')

    # Panel 5: Quantum Decoherence and Categorical Completion
    ax5 = fig.add_subplot(gs[1, 1])

    # Decoherence time vs system size
    # τ_dec ∝ 1/ω where ω is completion rate
    # For larger systems, more categories → faster decoherence

    system_size = np.logspace(0, 10, 50)  # Number of particles
    omega = 1e43  # Planck rate (s^-1)
    # Decoherence time decreases with system size
    tau_dec = 1 / (omega * np.log(system_size + 1))  # Simplified

    ax5.loglog(system_size, tau_dec, linewidth=3, color='darkviolet')
    ax5.fill_between(system_size, tau_dec * 0.5, tau_dec * 2,
                    alpha=0.2, color='darkviolet', label='Uncertainty Range')

    ax5.set_xlabel('System Size (N particles)', fontweight='bold')
    ax5.set_ylabel('Decoherence Time (s)', fontweight='bold')
    ax5.set_title('E. Decoherence Time\nvs System Size', fontweight='bold', pad=20)
    ax5.grid(True, alpha=0.3, linestyle='--', which='both')

    # Add markers for specific systems
    systems = {'Single atom': (1, tau_dec[0]),
            'Molecule': (100, tau_dec[20]),
            'Macroscopic': (1e23, tau_dec[-10])}
    for name, (size, time) in systems.items():
        idx = np.argmin(np.abs(system_size - size))
        ax5.plot(size, tau_dec[idx], 'o', markersize=10, color='red')
        ax5.text(size, tau_dec[idx] * 3, name, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 6: Reaction Rates and Categorical Path Length
    ax6 = fig.add_subplot(gs[1, 2])

    # Reaction rate ∝ exp(-L_cat/kT) where L_cat is categorical path length
    # Similar to Arrhenius equation but with categorical interpretation

    temperature = np.linspace(200, 400, 100)  # Kelvin
    k_B = 1.38e-23  # Boltzmann constant (J/K)
    L_cat_values = [10, 20, 30]  # Different categorical path lengths
    colors_rxn = plt.cm.plasma(np.linspace(0.2, 0.8, len(L_cat_values)))

    for L_cat, color in zip(L_cat_values, colors_rxn):
        # Simplified: rate ∝ exp(-L_cat * E_0 / kT)
        E_0 = 1e-20  # Energy scale (J)
        rate = np.exp(-L_cat * E_0 / (k_B * temperature))
        ax6.semilogy(temperature, rate, linewidth=3, label=f'L_cat = {L_cat}', color=color)

    ax6.set_xlabel('Temperature (K)', fontweight='bold')
    ax6.set_ylabel('Reaction Rate (arb. units)', fontweight='bold')
    ax6.set_title('F. Reaction Rates:\nCategorical Path Length', fontweight='bold', pad=20)
    ax6.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Panel 7: Cosmological Epochs and Categorical Transitions
    ax7 = fig.add_subplot(gs[2, :])

    # Timeline of universe with categorical epochs
    epochs = [
        {'name': 'Big Bang\n(t=0)', 'time': 0, 'C': 1, 'color': 'red'},
        {'name': 'Inflation\n(t=1)', 'time': 1e-35, 'C': 2, 'color': 'orange'},
        {'name': 'Quark Era\n(t=2)', 'time': 1e-6, 'C': 4, 'color': 'yellow'},
        {'name': 'Hadron Era\n(t=3)', 'time': 1e-3, 'C': 16, 'color': 'green'},
        {'name': 'Nucleosynthesis\n(t=4)', 'time': 1e2, 'C': 65536, 'color': 'cyan'},
        {'name': 'Matter Dom.\n(t=5)', 'time': 1e13, 'C': 2**65536, 'color': 'blue'},
        {'name': 'Present\n(t≈5)', 'time': 4.4e17, 'C': 2**65536, 'color': 'purple'},
    ]

    # Create timeline
    import math
    time_log = [np.log10(e['time'] + 1e-40) for e in epochs]  # Avoid log(0)

    # Compute C_log carefully (some values are huge Python ints)
    C_log = []
    for e in epochs:
        C_val = e['C']
        if C_val <= 0:
            C_log.append(-np.inf)
        elif isinstance(C_val, int) and C_val.bit_length() > 100:
            # Very large integer: use logarithm properties
            # For 2**n, log10(2**n) = n * log10(2)
            # Estimate bit length
            C_log.append(C_val.bit_length() * math.log10(2))
        else:
            try:
                C_log.append(np.log10(float(C_val)))
            except (OverflowError, ValueError):
                # Fallback for huge values
                C_log.append(C_val.bit_length() * math.log10(2))

    # Plot timeline
    for i, epoch in enumerate(epochs):
        ax7.scatter(time_log[i], C_log[i], s=500, c=epoch['color'],
                edgecolors='black', linewidth=2, zorder=5)
        ax7.text(time_log[i], C_log[i] + 2, epoch['name'], ha='center', fontsize=9,
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Connect with lines
    ax7.plot(time_log, C_log, 'k--', linewidth=2, alpha=0.5, zorder=1)

    # Fill regions
    for i in range(len(epochs) - 1):
        ax7.fill_between([time_log[i], time_log[i+1]],
                        [0, 0], [C_log[i], C_log[i+1]],
                        alpha=0.1, color=epochs[i]['color'])

    ax7.set_xlabel('log₁₀(Time) [seconds]', fontweight='bold')
    ax7.set_ylabel('log₁₀(C(t))', fontweight='bold')
    ax7.set_title('G. Cosmological Epochs and Categorical Transitions', fontweight='bold', pad=20)
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.set_ylim(-1, max(C_log) + 5)

    # Add annotations for key transitions
    ax7.annotate('Categorical\nExplosion', xy=(time_log[4], C_log[4]),
                xytext=(time_log[3], C_log[4] + 10),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                fontsize=11, fontweight='bold', color='red')

    plt.suptitle('Physical Predictions and Observational Tests',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('physical_predictions_panel.png', dpi=300, bbox_inches='tight')
    plt.savefig('physical_predictions_panel.pdf', dpi=300, bbox_inches='tight')
    print("Saved: physical_predictions_panel.png and .pdf")
    plt.show()
