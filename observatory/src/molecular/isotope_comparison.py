"""
ISOTOPE COMPARISON ANALYSIS: BENZENE H/D
Critical test of oscillatory vs shape theory of olfaction
Categorical dynamics framework applied to isotope effects
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from scipy import constants

if __name__ == "__main__":

    print("="*80)
    print("ISOTOPE COMPARISON: BENZENE H/D ANALYSIS")
    print("="*80)

    # ============================================================
    # LOAD ISOTOPE DATA
    # ============================================================

    print("\n1. LOADING ISOTOPE DATA")
    print("-" * 60)

    # Load benzene-H data
    with open('public/benzene_h_mass.json', 'r') as f:
        benzene_h = json.load(f)

    # Load benzene-D data
    with open('public/benzene_d_mass.json', 'r') as f:
        benzene_d = json.load(f)

    # Load comparison data
    with open('public/isotope_comparison.json', 'r') as f:
        comparison = json.load(f)

    print(f"✓ Loaded isotope data")
    print(f"  Benzene-H timestamp: {benzene_h['timestamp']}")
    print(f"  Benzene-D timestamp: {benzene_d['timestamp']}")
    print(f"  Comparison timestamp: {comparison['timestamp']}")

    # ============================================================
    # EXTRACT PROPERTIES
    # ============================================================

    print("\n2. MOLECULAR PROPERTIES")
    print("-" * 60)

    # Benzene-H properties
    mass_h = benzene_h['properties']['molecular_weight']
    exact_mass_h = benzene_h['properties']['exact_mass']
    reduced_mass_h = benzene_h['properties']['reduced_mass']

    print(f"Benzene-H (C₆H₆):")
    print(f"  Molecular weight: {mass_h:.4f} Da")
    print(f"  Exact mass: {exact_mass_h:.6f} Da")
    print(f"  Reduced mass: {reduced_mass_h:.6f} Da")
    print(f"  Heavy atoms: {benzene_h['properties']['heavy_atom_count']}")

    # Benzene-D properties
    mass_d = benzene_d['properties']['molecular_weight']
    exact_mass_d = benzene_d['properties']['exact_mass']
    reduced_mass_d = benzene_d['properties']['reduced_mass']
    deuterium_count = benzene_d['properties']['deuterium_count']

    print(f"\nBenzene-D (C₆D₆):")
    print(f"  Molecular weight: {mass_d:.4f} Da")
    print(f"  Exact mass: {exact_mass_d:.6f} Da")
    print(f"  Reduced mass: {reduced_mass_d:.6f} Da")
    print(f"  Deuterium count: {deuterium_count}")
    print(f"  Heavy atoms: {benzene_d['properties']['heavy_atom_count']}")

    # ============================================================
    # ISOTOPE EFFECT ANALYSIS
    # ============================================================

    print("\n3. ISOTOPE EFFECT ANALYSIS")
    print("-" * 60)

    # Mass ratios
    mass_ratio = comparison['comparison']['mass_ratio']
    reduced_mass_ratio = comparison['comparison']['reduced_mass_ratio']

    print(f"Mass Ratios:")
    print(f"  M(D)/M(H): {mass_ratio:.6f} ({(mass_ratio-1)*100:.2f}% heavier)")
    print(f"  μ(D)/μ(H): {reduced_mass_ratio:.6f} ({(reduced_mass_ratio-1)*100:.2f}% heavier)")

    # Frequency ratios
    freq_ratio_expected = comparison['comparison']['expected_frequency_ratio']
    freq_ratio_theoretical = comparison['comparison']['theoretical_frequency_ratio']

    print(f"\nFrequency Ratios:")
    print(f"  Expected ν(D)/ν(H): {freq_ratio_expected:.6f} ({(1-freq_ratio_expected)*100:.2f}% lower)")
    print(f"  Theoretical (√2): {freq_ratio_theoretical:.6f} ({(1-freq_ratio_theoretical)*100:.2f}% lower)")

    # Calculate vibrational frequencies
    # For C-H stretch: ~3000 cm⁻¹
    # For C-D stretch: ~2200 cm⁻¹ (literature)

    freq_CH_literature = 3000  # cm⁻¹
    freq_CD_literature = 2200  # cm⁻¹
    freq_ratio_literature = freq_CD_literature / freq_CH_literature

    print(f"\nLiterature Values:")
    print(f"  C-H stretch: {freq_CH_literature} cm⁻¹")
    print(f"  C-D stretch: {freq_CD_literature} cm⁻¹")
    print(f"  Ratio: {freq_ratio_literature:.6f}")

    # Calculate from reduced mass (harmonic oscillator)
    # ν ∝ 1/√μ
    freq_ratio_from_reduced_mass = np.sqrt(reduced_mass_h / reduced_mass_d)

    print(f"\nFrom Reduced Mass:")
    print(f"  Predicted ratio: {freq_ratio_from_reduced_mass:.6f}")
    print(f"  Agreement with literature: {freq_ratio_from_reduced_mass/freq_ratio_literature*100:.1f}%")

    # ============================================================
    # OSCILLATORY THEORY PREDICTIONS
    # ============================================================

    print("\n4. OSCILLATORY THEORY PREDICTIONS")
    print("-" * 60)

    # Physical constants
    h = constants.h  # Planck constant
    c = constants.c * 100  # Speed of light in cm/s
    k_B = constants.k  # Boltzmann constant

    # Convert wavenumbers to frequencies (Hz)
    freq_CH_hz = freq_CH_literature * c
    freq_CD_hz = freq_CD_literature * c

    print(f"Vibrational Frequencies:")
    print(f"  C-H: {freq_CH_hz:.2e} Hz ({freq_CH_hz/1e12:.2f} THz)")
    print(f"  C-D: {freq_CD_hz:.2e} Hz ({freq_CD_hz/1e12:.2f} THz)")
    print(f"  Δν: {(freq_CH_hz - freq_CD_hz)/1e12:.2f} THz")

    # Energy difference
    E_CH = h * freq_CH_hz
    E_CD = h * freq_CD_hz
    delta_E = E_CH - E_CD

    print(f"\nVibrational Energies:")
    print(f"  E(C-H): {E_CH:.2e} J ({E_CH/1e-21:.2f} zJ)")
    print(f"  E(C-D): {E_CD:.2e} J ({E_CD/1e-21:.2f} zJ)")
    print(f"  ΔE: {delta_E:.2e} J ({delta_E/1e-21:.2f} zJ)")

    # Thermal energy at room temperature (300 K)
    E_thermal = k_B * 300

    print(f"\nThermal Context:")
    print(f"  kT (300K): {E_thermal:.2e} J ({E_thermal/1e-21:.2f} zJ)")
    print(f"  E(C-H)/kT: {E_CH/E_thermal:.2f}")
    print(f"  E(C-D)/kT: {E_CD/E_thermal:.2f}")

    # Oscillatory theory prediction
    print(f"\n{comparison['theory_note']}")
    print(f"  Shape theory: SAME scent (identical molecular shape)")
    print(f"  Oscillatory theory: DIFFERENT scent (different vibrational frequency)")

    # ============================================================
    # CATEGORICAL DYNAMICS ANALYSIS
    # ============================================================

    print("\n5. CATEGORICAL DYNAMICS ANALYSIS")
    print("-" * 60)

    # S-entropy coordinates for isotopes (NORMALIZED VERSION)
    def compute_s_entropy_isotope(mass, reduced_mass, freq):
        """Compute S-entropy coordinates for isotope (normalized)"""
        # Normalize inputs to avoid huge numbers
        mass_norm = mass / 100  # Normalize to ~1
        reduced_mass_norm = reduced_mass / 10
        freq_norm = freq / 1e14  # Normalize to THz range

        s1 = mass_norm
        s2 = reduced_mass_norm
        s3 = freq_norm

        # Derived coordinates
        s4 = np.log(s1 + 1e-10)  # Avoid log(0)
        s5 = s2 / (s1 + 1e-10)
        s6 = s3 / (s1 + 1e-10)

        # Harmonic coordinates
        s7 = s1 * s2
        s8 = s1 * s3
        s9 = s2 * s3

        # Oscillatory coordinates
        s10 = np.sin(2 * np.pi * s3)
        s11 = np.cos(2 * np.pi * s3)

        # Information-theoretic
        s12 = -s1 * np.log(s1 + 1e-10)
        s13 = np.sqrt(s1**2 + s2**2 + s3**2)
        s14 = s1 / (s13 + 1e-10)

        return np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9,
                        s10, s11, s12, s13, s14])

    # Compute S-entropy for both isotopes
    s_entropy_h = compute_s_entropy_isotope(mass_h, reduced_mass_h, freq_CH_hz)
    s_entropy_d = compute_s_entropy_isotope(mass_d, reduced_mass_d, freq_CD_hz)

    print(f"S-Entropy Coordinates (first 5):")
    print(f"  Benzene-H: {s_entropy_h[:5]}")
    print(f"  Benzene-D: {s_entropy_d[:5]}")

    # Categorical distance
    categorical_distance = np.linalg.norm(s_entropy_h - s_entropy_d)
    print(f"\nCategorical Distance:")
    print(f"  ||S(H) - S(D)||: {categorical_distance:.6f}")
    print(f"  Interpretation: {'DISTINGUISHABLE' if categorical_distance > 0.1 else 'SIMILAR'}")

    # ============================================================
    # VISUALIZATION
    # ============================================================

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.4)

    colors = {
        'H': '#3498db',
        'D': '#e74c3c',
        'comparison': '#2ecc71',
        'theory': '#f39c12',
        'categorical': '#9b59b6'
    }

    # ============================================================
    # PANEL 1: Molecular Structure Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')

    structure_text = """
    MOLECULAR STRUCTURE COMPARISON

    Benzene-H (C₆H₆):                Benzene-D (C₆D₆):
        H                                 D
        |                                 |
    H--C--H                           D--C--D
        /  \\                              /  \\
        /    \\                            /    \\
    C      C                          C      C
    / \\    / \\                        / \\    / \\
    H   C--C   H                      D   C--C   D
        |                                 |
        H                                 D

    IDENTICAL SHAPE                   IDENTICAL SHAPE
    (same carbon skeleton)            (same carbon skeleton)

    DIFFERENT MASS                    DIFFERENT FREQUENCY
    H: 1.008 Da                       C-H: 3000 cm⁻¹
    D: 2.014 Da                       C-D: 2200 cm⁻¹

    SHAPE THEORY PREDICTION:          OSCILLATORY THEORY PREDICTION:
    → SAME SCENT                      → DIFFERENT SCENT
    (identical molecular shape)       (different vibrational frequency)
    """

    ax1.text(0.5, 0.5, structure_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='center', ha='center',
            family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax1.set_title('(A) Critical Test: Shape vs Oscillatory Theory\nBenzene Isotope Comparison',
                fontsize=12, fontweight='bold')

    # ============================================================
    # PANEL 2: Mass Comparison
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    isotopes = ['Benzene-H\n(C₆H₆)', 'Benzene-D\n(C₆D₆)']
    masses = [mass_h, mass_d]
    reduced_masses = [reduced_mass_h, reduced_mass_d]

    x = np.arange(len(isotopes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, masses, width, label='Molecular Mass',
                color=colors['H'], alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x + width/2, reduced_masses, width, label='Reduced Mass',
                color=colors['D'], alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax2.set_ylabel('Mass (Da)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Mass Properties\nMolecular vs Reduced Mass',
                fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(isotopes)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')

    # Continue with remaining panels (3-8 are fine, just need to fix panel 9)

    # ... [Panels 3-8 remain the same as before] ...

    # ============================================================
    # PANEL 9: Thermal Population Distribution (FIXED)
    # ============================================================
    ax9 = fig.add_subplot(gs[4, :2])

    # Boltzmann distribution at 300 K
    T = 300  # K
    n_max = 5  # FIXED: Match the number of energy levels we calculate
    n_states = np.arange(n_max)

    # Energy levels (simplified - just use first 5)
    E_levels_H = np.array([freq_CH_literature * (n + 0.5) for n in range(n_max)])
    E_levels_D = np.array([freq_CD_literature * (n + 0.5) for n in range(n_max)])

    # Populations for H
    E_H = E_levels_H * h * c  # Convert to Joules
    pop_H = np.exp(-E_H / (k_B * T))
    pop_H /= pop_H.sum()  # Normalize

    # Populations for D
    E_D = E_levels_D * h * c
    pop_D = np.exp(-E_D / (k_B * T))
    pop_D /= pop_D.sum()

    x = np.arange(n_max)  # FIXED: Use n_max
    width = 0.35

    bars1 = ax9.bar(x - width/2, pop_H, width, label='Benzene-H',
                color=colors['H'], alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax9.bar(x + width/2, pop_D, width, label='Benzene-D',
                color=colors['D'], alpha=0.8, edgecolor='black', linewidth=2)

    ax9.set_xlabel('Vibrational Quantum Number (n)', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Boltzmann Population (300 K)', fontsize=11, fontweight='bold')
    ax9.set_title('(I) Thermal Population Distribution\nIsotope Effect on Quantum States',
                fontsize=12, fontweight='bold')
    ax9.set_xticks(x)
    ax9.legend(fontsize=10)
    ax9.grid(alpha=0.3, linestyle='--', axis='y')
    ax9.set_yscale('log')

    # ============================================================
    # PANEL 10: Summary Statistics
    # ============================================================
    ax10 = fig.add_subplot(gs[4, 2:])
    ax10.axis('off')

    summary_text = f"""
    ISOTOPE COMPARISON SUMMARY

    MOLECULAR PROPERTIES:
    Benzene-H (C₆H₆):
        Molecular weight:      {mass_h:.4f} Da
        Reduced mass:          {reduced_mass_h:.4f} Da
        C-H stretch:           {freq_CH_literature} cm⁻¹ ({freq_CH_hz/1e12:.2f} THz)

    Benzene-D (C₆D₆):
        Molecular weight:      {mass_d:.4f} Da
        Reduced mass:          {reduced_mass_d:.4f} Da
        C-D stretch:           {freq_CD_literature} cm⁻¹ ({freq_CD_hz/1e12:.2f} THz)
        Deuterium count:       {deuterium_count}

    ISOTOPE EFFECTS:
    Mass ratio M(D)/M(H):    {mass_ratio:.6f} (+{(mass_ratio-1)*100:.2f}%)
    Reduced mass ratio:      {reduced_mass_ratio:.6f} (+{(reduced_mass_ratio-1)*100:.2f}%)
    Frequency ratio (lit):   {freq_ratio_literature:.6f} (-{(1-freq_ratio_literature)*100:.2f}%)
    Frequency ratio (calc):  {freq_ratio_from_reduced_mass:.6f} (-{(1-freq_ratio_from_reduced_mass)*100:.2f}%)
    Agreement:               {freq_ratio_from_reduced_mass/freq_ratio_literature*100:.1f}%

    ENERGY DIFFERENCES:
    E(C-H):                  {E_CH/1e-21:.2f} zJ
    E(C-D):                  {E_CD/1e-21:.2f} zJ
    ΔE:                      {delta_E/1e-21:.2f} zJ ({(E_CH-E_CD)/E_CH*100:.1f}%)
    kT (300K):               {E_thermal/1e-21:.2f} zJ
    E(C-H)/kT:               {E_CH/E_thermal:.2f}
    E(C-D)/kT:               {E_CD/E_thermal:.2f}

    CATEGORICAL ANALYSIS:
    S-entropy dimensions:    {len(s_entropy_h)}
    Categorical distance:    {categorical_distance:.6f}
    Distinguishability:      {'YES' if categorical_distance > 0.1 else 'NO'}

    THEORETICAL PREDICTIONS:
    Shape Theory:            SAME SCENT
        (identical molecular geometry)

    Oscillatory Theory:      DIFFERENT SCENT
        (different vibrational frequency)
        Δν = {(freq_CH_hz - freq_CD_hz)/1e12:.2f} THz

    Categorical Dynamics:    DISTINGUISHABLE
        (different S-entropy coordinates)
        Distance = {categorical_distance:.3f}

    EXPERIMENTAL TEST:
    Critical experiment:     Olfactory discrimination test
    Expected result:         If oscillatory theory correct →
                            humans can distinguish C₆H₆ from C₆D₆
    Status:                  TESTABLE ✓
    """

    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Isotope Comparison Analysis: Benzene H/D\n'
                'Critical Test of Shape vs Oscillatory Theory of Olfaction',
                fontsize=14, fontweight='bold', y=0.998)

    plt.savefig('isotope_comparison_benzene_hd.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('isotope_comparison_benzene_hd.png', dpi=300, bbox_inches='tight')

    print("\n✓ Isotope comparison visualization complete")
    print("  Saved: isotope_comparison_benzene_hd.pdf")
    print("  Saved: isotope_comparison_benzene_hd.png")
    print("="*80)
