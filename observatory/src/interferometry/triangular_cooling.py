"""
Experimental Results: Triangular Cooling Validation
Analysis of experimental data showing triangular cooling depletion
UPDATED WITH ACTUAL EXPERIMENTAL DATA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.gridspec as gridspec
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'standard': '#1976D2',
    'triangular': '#D32F2F',
    'theory': '#388E3C',
    'highlight': '#F57C00'
}

if __name__ == "__main__":
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ============================================================================
    # EXPERIMENTAL DATA (from your validation)
    # ============================================================================
    T_initial = 100e-9  # 100 nK
    N = 10  # reflections
    Q = 0.7  # cooling factor
    epsilon = 0.1  # energy extraction rate

    # Your experimental results
    T_standard_exp = 2824752.49e-15  # fK
    T_triangular_exp = 18990970.22e-15  # fK
    cooling_standard_exp = 35.4
    cooling_triangular_exp = 5.27
    amplification_exp = 0.149

    # Molecule 1 depletion
    T_mol1_initial = 100.00e-9  # nK
    T_mol1_after_5 = 59.05e-9  # nK
    T_mol1_after_10 = 34.87e-9  # nK
    cooling_mol1 = T_mol1_initial / T_mol1_after_10  # 2.87×

    # Cascade depth scaling data
    cascade_depths = np.array([1, 2, 5, 10, 15, 20])
    T_standard_depths = np.array([70000000.00, 49000000.00, 16807000.00,
                                2824752.49, 474756.15, 79792.27]) * 1e-15  # fK
    T_triangular_depths = np.array([59767047.78, 48704764.92, 32550504.61,
                                    18990970.22, 11209764.21, 6619175.92]) * 1e-15  # fK
    amplifications_depths = np.array([1.171, 1.006, 0.516, 0.149, 0.042, 0.012])

    # Energy extraction rate sensitivity
    epsilons = np.array([0.05, 0.10, 0.15, 0.20])
    T_triangular_eps = np.array([30899098.42, 18990970.22, 11351733.16, 6576950.75]) * 1e-15  # fK
    amplifications_eps = T_standard_exp / T_triangular_eps

    # FTL comparison
    ftl_amplification_per_stage = 2.847
    cooling_amplification_per_stage = 0.827  # Your measured value

    # ============================================================================
    # PANEL A: TEMPERATURE EVOLUTION
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Generate standard cascade
    def standard_cascade(T0, N, Q):
        T_history = [T0]
        T = T0
        for n in range(N):
            T *= Q
            T_history.append(T)
        return np.array(T_history)

    # Generate triangular cascade (with depletion)
    def triangular_cascade(T0, N, Q, epsilon):
        T_history = [T0]
        T_ref = [T0]  # Reference temperatures

        for n in range(1, N + 1):
            T = T_ref[-1] * Q

            # Self-referencing with depletion
            for ref_idx in range(len(T_ref)):
                # Extract energy from reference
                energy_extracted = epsilon * T_ref[ref_idx]
                T_ref[ref_idx] -= energy_extracted

                # Current molecule affected by depleted reference
                T *= (1 + energy_extracted / T)

            T_ref.append(T)
            T_history.append(T)

        return np.array(T_history)

    # Generate data
    reflections = np.arange(N + 1)
    T_standard = standard_cascade(T_initial, N, Q)
    T_triangular = triangular_cascade(T_initial, N, Q, epsilon)

    # Plot
    ax1.semilogy(reflections, T_standard * 1e15, 'o-', linewidth=3, markersize=10,
                color=colors['standard'], label='Standard cascade',
                markerfacecolor='white', markeredgewidth=2)
    ax1.semilogy(reflections, T_triangular * 1e15, 's-', linewidth=3, markersize=10,
                color=colors['triangular'], label='Triangular cascade',
                markerfacecolor='white', markeredgewidth=2)

    # Mark experimental endpoints
    ax1.scatter([N], [T_standard_exp * 1e15], s=400, marker='*',
            color=colors['standard'], edgecolors='black', linewidths=2,
            zorder=5, label='Standard (experimental)')
    ax1.scatter([N], [T_triangular_exp * 1e15], s=400, marker='*',
            color=colors['triangular'], edgecolors='black', linewidths=2,
            zorder=5, label='Triangular (experimental)')

    # Annotations
    ax1.annotate('', xy=(N, T_triangular_exp * 1e15),
                xytext=(N, T_standard_exp * 1e15),
                arrowprops=dict(arrowstyle='<->', lw=3, color=colors['highlight']))
    ax1.text(N - 0.5, np.sqrt(T_standard_exp * T_triangular_exp) * 1e15,
            f'{amplification_exp:.3f}×\n(6.7× WORSE)',
            ha='right', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_xlabel('Number of Reflections', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Temperature (fK)', fontsize=13, fontweight='bold')
    ax1.set_title('A. Temperature Evolution: Standard vs Triangular Cascade',
                fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim([-0.5, N + 0.5])

    # Add experimental parameters
    params_text = (f"Initial T: {T_initial*1e9:.0f} nK\n"
                f"Reflections: {N}\n"
                f"Q factor: {Q}\n"
                f"ε: {epsilon}")
    ax1.text(0.02, 0.98, params_text, transform=ax1.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================================================
    # PANEL B: COOLING FACTOR COMPARISON
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    methods = ['Standard\nCascade', 'Triangular\nCascade']
    cooling_factors = [cooling_standard_exp, cooling_triangular_exp]
    bar_colors = [colors['standard'], colors['triangular']]

    bars = ax2.bar(methods, cooling_factors, color=bar_colors, alpha=0.7,
                edgecolor='black', linewidth=2)

    # Add value labels
    for bar, cf in zip(bars, cooling_factors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{cf:.2f}×',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add performance comparison
    ax2.annotate('', xy=(1, cooling_triangular_exp), xytext=(0, cooling_standard_exp),
                arrowprops=dict(arrowstyle='<->', lw=2, color=colors['highlight']))
    ax2.text(0.5, (cooling_standard_exp + cooling_triangular_exp)/2,
            f'{amplification_exp:.3f}×',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax2.set_ylabel('Total Cooling Factor', fontsize=12, fontweight='bold')
    ax2.set_title('B. Cooling Factor Comparison',
                fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, cooling_standard_exp * 1.2])

    # ============================================================================
    # PANEL C: MOLECULE 1 ENERGY DEPLETION
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Experimental data for Molecule 1
    observations_mol1 = np.array([0, 5, 10])
    T_mol1_history = np.array([T_mol1_initial, T_mol1_after_5, T_mol1_after_10])

    # Interpolate for smooth curve
    observations_interp = np.linspace(0, 10, 100)
    T_mol1_interp = T_mol1_initial * (1 - epsilon)**observations_interp

    # Plot
    ax3.plot(observations_interp, T_mol1_interp * 1e9, '-', linewidth=2,
            color=colors['triangular'], alpha=0.5, label='Theory')
    ax3.plot(observations_mol1, T_mol1_history * 1e9, 'o', markersize=12,
            color=colors['triangular'], markerfacecolor='white',
            markeredgewidth=3, label='Experimental', zorder=5)

    # Highlight key points
    ax3.scatter([0], [T_mol1_initial * 1e9], s=300, color='green',
            marker='*', edgecolors='black', linewidths=2, zorder=6)
    ax3.text(0, T_mol1_initial * 1e9 * 1.1, 'Initial\n100.00 nK',
            ha='center', fontsize=10, fontweight='bold')

    ax3.scatter([10], [T_mol1_after_10 * 1e9], s=300, color='red',
            marker='*', edgecolors='black', linewidths=2, zorder=6)
    ax3.text(10, T_mol1_after_10 * 1e9 * 0.85, 'Depleted\n34.87 nK',
            ha='center', fontsize=10, fontweight='bold')

    # Add depletion annotation
    ax3.text(0.5, 0.95, f'Molecule 1 depletion: {cooling_mol1:.2f}×',
            transform=ax3.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax3.set_xlabel('Observation Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Molecule 1 Temperature (nK)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Molecule 1 Energy Depletion (Experimental)',
                fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-0.5, 10.5])

    # ============================================================================
    # PANEL D: ENERGY EXTRACTION PER OBSERVATION
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate energy extracted at each observation (from Molecule 1 data)
    energy_extracted_0_5 = T_mol1_initial - T_mol1_after_5
    energy_extracted_5_10 = T_mol1_after_5 - T_mol1_after_10

    # Plot bars
    ax4.bar([2.5], [energy_extracted_0_5 * 1e9], width=4,
        color=colors['highlight'], alpha=0.7, edgecolor='black',
        linewidth=2, label='Obs 1-5')
    ax4.bar([7.5], [energy_extracted_5_10 * 1e9], width=4,
        color=colors['triangular'], alpha=0.7, edgecolor='black',
        linewidth=2, label='Obs 6-10')

    # Add value labels
    ax4.text(2.5, energy_extracted_0_5 * 1e9 * 1.05,
            f'{energy_extracted_0_5 * 1e9:.2f} nK',
            ha='center', fontsize=11, fontweight='bold')
    ax4.text(7.5, energy_extracted_5_10 * 1e9 * 1.05,
            f'{energy_extracted_5_10 * 1e9:.2f} nK',
            ha='center', fontsize=11, fontweight='bold')

    # Theoretical decay curve
    obs_theory = np.arange(1, 11)
    theory_extraction = epsilon * T_mol1_initial * (1 - epsilon)**(obs_theory - 1)
    ax4.plot(obs_theory, theory_extraction * 1e9, 'k--', linewidth=2,
            label='Theory: ε×T₀×(1-ε)^n', alpha=0.7)

    ax4.set_xlabel('Observation Period', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Energy Extracted (nK)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Energy Extraction per Observation',
                fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim([0, 10])

    # Add annotation
    ax4.text(0.5, 0.95, 'Extraction decreases\n(reference depletes)',
            transform=ax4.transAxes, ha='center', va='top',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # ============================================================================
    # PANEL E: CASCADE DEPTH SCALING (EXPERIMENTAL DATA)
    # ============================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    # Plot experimental data
    ax5.semilogy(cascade_depths, amplifications_depths, 'o-', linewidth=3,
                markersize=10, color=colors['triangular'],
                markerfacecolor='white', markeredgewidth=2,
                label='Experimental')

    # Highlight N=10 point
    ax5.scatter([10], [amplification_exp], s=400, marker='*',
            color='red', edgecolors='black', linewidths=2, zorder=5,
            label='N=10 (main experiment)')

    # Reference line at 1.0 (equal performance)
    ax5.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='Equal performance')

    # Fill region below 1.0
    ax5.fill_between(cascade_depths, amplifications_depths, 1.0,
                    where=(amplifications_depths < 1.0),
                    alpha=0.3, color=colors['triangular'])

    # Annotate key points
    for depth, amp in zip([1, 2, 10, 20], [1.171, 1.006, 0.149, 0.012]):
        if depth in cascade_depths:
            idx = np.where(cascade_depths == depth)[0][0]
            ax5.text(depth, amp * 1.3, f'{amp:.3f}×',
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax5.set_xlabel('Cascade Depth (N)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Triangular / Standard', fontsize=12, fontweight='bold')
    ax5.set_title('E. Cascade Depth Scaling (Experimental)',
                fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3, which='both')
    ax5.set_xlim([0, 21])

    # Add annotation
    ax5.text(0.5, 0.05, 'Deeper cascade → More depletion → Worse performance',
            transform=ax5.transAxes, ha='center', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # ============================================================================
    # PANEL F: ENERGY EXTRACTION RATE SENSITIVITY (EXPERIMENTAL DATA)
    # ============================================================================
    ax6 = fig.add_subplot(gs[2, 0])

    # Plot experimental data
    ax6.plot(epsilons, amplifications_eps, 'o-', linewidth=3, markersize=10,
            color=colors['highlight'], markerfacecolor='white',
            markeredgewidth=2)

    # Mark experimental point (ε=0.1)
    ax6.scatter([epsilon], [amplification_exp], s=300, marker='*',
            color='red', edgecolors='black', linewidths=2, zorder=5,
            label=f'Experimental (ε={epsilon})')

    # Reference line at 1.0
    ax6.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='Equal performance')

    # Annotate data points
    for eps, amp in zip(epsilons, amplifications_eps):
        ax6.text(eps, amp * 1.15, f'{amp:.3f}×',
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax6.set_xlabel('Energy Extraction Rate (ε)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Triangular / Standard', fontsize=12, fontweight='bold')
    ax6.set_title('F. Energy Extraction Rate Sensitivity',
                fontsize=14, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0.04, 0.21])

    # Add annotation
    ax6.text(0.5, 0.05, 'Higher ε → More depletion → Worse performance',
            transform=ax6.transAxes, ha='center', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # ============================================================================
    # PANEL G: COMPARISON WITH FTL
    # ============================================================================
    ax7 = fig.add_subplot(gs[2, 1])

    categories = ['FTL\n(Kinematic)', 'Cooling\n(Thermodynamic)']
    amplifications_compare = [ftl_amplification_per_stage, cooling_amplification_per_stage]
    bar_colors_compare = [colors['theory'], colors['triangular']]

    bars = ax7.bar(categories, amplifications_compare, color=bar_colors_compare,
                alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, amp in zip(bars, amplifications_compare):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{amp:.3f}×',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Reference line at 1.0
    ax7.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5,
            label='No amplification')

    # Add symbols
    ax7.text(0, ftl_amplification_per_stage + 0.3, '✓', ha='center',
            fontsize=40, color='green')
    ax7.text(1, cooling_amplification_per_stage - 0.15, '✗', ha='center',
            fontsize=40, color='red')

    # Add ratio annotation
    ratio = cooling_amplification_per_stage / ftl_amplification_per_stage
    ax7.text(0.5, 1.5, f'Ratio: {ratio:.3f}',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax7.set_ylabel('Amplification per Stage', fontsize=12, fontweight='bold')
    ax7.set_title('G. Comparison with FTL Triangular Amplification',
                fontsize=14, fontweight='bold', pad=20)
    ax7.legend(loc='upper right', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_ylim([0, ftl_amplification_per_stage * 1.3])

    # ============================================================================
    # PANEL H: SUMMARY TABLE
    # ============================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    # Create summary table
    summary_data = [
        ['Parameter', 'Value'],
        ['─' * 30, '─' * 25],
        ['Initial temperature', f'{T_initial*1e9:.0f} nK'],
        ['Cascade depth', f'{N}'],
        ['Cooling factor (Q)', f'{Q}'],
        ['Extraction rate (ε)', f'{epsilon}'],
        ['', ''],
        ['STANDARD CASCADE', ''],
        ['Final temperature', f'{T_standard_exp*1e15:.2f} fK'],
        ['Total cooling', f'{cooling_standard_exp:.2f}×'],
        ['', ''],
        ['TRIANGULAR CASCADE', ''],
        ['Final temperature', f'{T_triangular_exp*1e15:.2f} fK'],
        ['Total cooling', f'{cooling_triangular_exp:.2f}×'],
        ['', ''],
        ['MOLECULE 1 DEPLETION', ''],
        ['Initial', f'{T_mol1_initial*1e9:.2f} nK'],
        ['After 5 obs', f'{T_mol1_after_5*1e9:.2f} nK'],
        ['After 10 obs', f'{T_mol1_after_10*1e9:.2f} nK'],
        ['Total depletion', f'{cooling_mol1:.2f}×'],
        ['', ''],
        ['COMPARISON', ''],
        ['Triangular/Standard', f'{amplification_exp:.3f}'],
        ['Performance', f'{(1-amplification_exp)*100:.1f}% WORSE'],
        ['', ''],
        ['FTL amplification', f'{ftl_amplification_per_stage:.3f}×'],
        ['Cooling amplification', f'{cooling_amplification_per_stage:.3f}×'],
        ['Ratio', f'{cooling_amplification_per_stage/ftl_amplification_per_stage:.3f}'],
        ['', ''],
        ['CONCLUSION', ''],
        ['Triangular cooling', 'FAILS ✗'],
        ['Reason', 'Energy depletion'],
        ['Mechanism', 'Finite energy'],
    ]

    # Format table
    table_text = '\n'.join([f"{row[0]:<30} {row[1]:>25}" for row in summary_data])

    ax8.text(0.5, 0.95, table_text,
            transform=ax8.transAxes,
            ha='center', va='top',
            fontsize=8.5, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax8.set_title('H. Experimental Summary',
                fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # OVERALL TITLE AND ANNOTATIONS
    # ============================================================================
    fig.suptitle('Experimental Validation: Triangular Cooling Depletion',
                fontsize=18, fontweight='bold', y=0.995)

    # Add key results box
    results_text = (
        f"✗ Triangular cascade: {amplification_exp:.3f}× of standard ({(1-amplification_exp)*100:.1f}% WORSE)\n"
        f"✗ Reason: Energy depletion of reference molecule (Molecule 1: {cooling_mol1:.2f}× depleted)\n"
        f"✗ Scaling: Deeper cascade → worse performance (N=20: {amplifications_depths[-1]:.3f}×)\n"
        f"✓ Categorical observation is passive (no backaction) but reveals physical depletion"
    )

    fig.text(0.5, 0.002, results_text,
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, pad=10))

    plt.tight_layout(rect=[0, 0.025, 1, 0.99])
    plt.savefig('experimental_triangular_cooling_validation.png', dpi=300, bbox_inches='tight')
    print("✓ Figure saved: experimental_triangular_cooling_validation.png")
    print(f"\n{'='*70}")
    print("KEY EXPERIMENTAL RESULTS")
    print(f"{'='*70}")
    print(f"\nTEST 1: Triangular Amplification Factor")
    print(f"  Standard cascade:   {cooling_standard_exp:.2f}× cooling")
    print(f"  Triangular cascade: {cooling_triangular_exp:.2f}× cooling")
    print(f"  Performance ratio:  {amplification_exp:.3f} (triangular is {(1-amplification_exp)*100:.1f}% worse)")
    print(f"\nTEST 2: Molecule 1 Depletion")
    print(f"  Initial:      {T_mol1_initial*1e9:.2f} nK")
    print(f"  After 5 obs:  {T_mol1_after_5*1e9:.2f} nK")
    print(f"  After 10 obs: {T_mol1_after_10*1e9:.2f} nK")
    print(f"  Depletion:    {cooling_mol1:.2f}× from repeated observations")
    print(f"\nTEST 3: Cascade Depth Scaling")
    for depth, amp in zip(cascade_depths, amplifications_depths):
        print(f"  N={depth:2d}: Amplification = {amp:.3f}×")
    print(f"\nTEST 4: Energy Extraction Rate Sensitivity")
    for eps, amp in zip(epsilons, amplifications_eps):
        print(f"  ε={eps:.2f}: Amplification = {amp:.3f}×")
    print(f"\nTEST 5: FTL Comparison")
    print(f"  FTL amplification per stage:     {ftl_amplification_per_stage:.3f}×")
    print(f"  Cooling amplification per stage: {cooling_amplification_per_stage:.3f}×")
    print(f"  Ratio:                           {cooling_amplification_per_stage/ftl_amplification_per_stage:.3f}")
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("Triangular amplification FAILS for thermodynamic operations")
    print("due to energy conservation and finite energy constraints.")
    print("\nMechanism: Reference molecule depletes with repeated observations")
    print("Result: Later molecules extract less energy → reduced total cooling")
    print("Contrast: FTL works because position advances (not depletes)")
    print(f"{'='*70}")
    plt.close()
