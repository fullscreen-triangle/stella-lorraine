"""
Theoretical Insight: Kinematic vs Thermodynamic Asymmetry in Triangular Amplification
Demonstrates why triangular amplification works for FTL but fails for cooling
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib import patches
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'kinematic': '#388E3C',
    'thermodynamic': '#D32F2F',
    'neutral': '#757575',
    'highlight': '#F57C00'
}

if __name__ == "__main__":

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ============================================================================
    # PANEL A: FTL - KINEMATIC OPERATION (WORKS)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    # Time axis
    time_points = [0, 1, 2, 3]
    y_positions = [0.3, 0.5, 0.7]

    # Draw timeline
    ax1.plot([0.1, 0.9], [0.5, 0.5], 'k-', linewidth=2, alpha=0.3)

    # Projectile 1 trajectory (advances)
    x1_positions = [0.2, 0.4, 0.6, 0.8]
    for i, (t, x) in enumerate(zip(time_points, x1_positions)):
        circle = Circle((x, 0.5), 0.04, color=colors['kinematic'],
                    alpha=0.3 + i*0.2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(x, 0.5, '1', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)

        # Time label
        ax1.text(x, 0.35, f't={t}', ha='center', fontsize=8, style='italic')

        # Position label
        if i == 0:
            ax1.text(x, 0.65, f'x₀', ha='center', fontsize=9, color=colors['kinematic'])
        elif i == len(time_points) - 1:
            ax1.text(x, 0.65, f'x₀+Δx', ha='center', fontsize=9,
                    color=colors['kinematic'], fontweight='bold')

    # Projectile 3 observes Projectile 1
    obs_x = 0.5
    arrow = FancyArrowPatch((obs_x, 0.75), (0.6, 0.54),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=3, color=colors['highlight'],
                        linestyle='--')
    ax1.add_patch(arrow)

    circle3 = Circle((obs_x, 0.8), 0.05, color=colors['highlight'],
                    alpha=0.8, zorder=3, edgecolor='black', linewidth=2)
    ax1.add_patch(circle3)
    ax1.text(obs_x, 0.8, '3', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)

    # Annotation
    ax1.text(obs_x, 0.9, 'Observes ADVANCED\nposition', ha='center',
            fontsize=10, fontweight='bold', color=colors['highlight'],
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Key properties
    props_text = (
        "KINEMATIC PROPERTIES:\n"
        "• Position NOT conserved\n"
        "• Position NOT finite\n"
        "• Observation does NOT deplete\n"
        "• Reference ADVANCES\n"
        "• Result: Amplification ✓"
    )
    ax1.text(0.05, 0.15, props_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor=colors['kinematic'], alpha=0.3),
            verticalalignment='top', family='monospace')

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('A. FTL: Kinematic Operation (Position Advances)',
                fontsize=14, fontweight='bold', pad=20, color=colors['kinematic'])

    # ============================================================================
    # PANEL B: COOLING - THERMODYNAMIC OPERATION (FAILS)
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Time axis
    ax2.plot([0.1, 0.9], [0.5, 0.5], 'k-', linewidth=2, alpha=0.3)

    # Molecule 1 energy depletion
    E_positions = [0.2, 0.4, 0.6, 0.8]
    E_values = [1.0, 0.7, 0.5, 0.35]  # Depleting energy

    for i, (x, E) in enumerate(zip(E_positions, E_values)):
        # Circle size represents energy
        circle = Circle((x, 0.5), 0.03 + E*0.03, color=colors['thermodynamic'],
                    alpha=0.3 + i*0.2, zorder=3)
        ax2.add_patch(circle)
        ax2.text(x, 0.5, '1', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)

        # Time label
        ax2.text(x, 0.35, f't={i}', ha='center', fontsize=8, style='italic')

        # Energy label
        if i == 0:
            ax2.text(x, 0.68, f'E₀', ha='center', fontsize=9, color=colors['thermodynamic'])
        elif i == len(E_positions) - 1:
            ax2.text(x, 0.68, f'E₀-ΔE', ha='center', fontsize=9,
                    color=colors['thermodynamic'], fontweight='bold')

    # Molecule 3 observes Molecule 1
    obs_x = 0.5
    arrow = FancyArrowPatch((obs_x, 0.75), (0.6, 0.56),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=3, color=colors['highlight'],
                        linestyle='--')
    ax2.add_patch(arrow)

    circle3 = Circle((obs_x, 0.8), 0.05, color=colors['highlight'],
                    alpha=0.8, zorder=3, edgecolor='black', linewidth=2)
    ax2.add_patch(circle3)
    ax2.text(obs_x, 0.8, '3', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=4)

    # Annotation
    ax2.text(obs_x, 0.9, 'Observes DEPLETED\nenergy', ha='center',
            fontsize=10, fontweight='bold', color=colors['highlight'],
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Key properties
    props_text = (
        "THERMODYNAMIC PROPERTIES:\n"
        "• Energy IS conserved\n"
        "• Energy IS finite\n"
        "• Extraction DOES deplete\n"
        "• Reference DEPLETES\n"
        "• Result: Depletion ✗"
    )
    ax2.text(0.05, 0.15, props_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor=colors['thermodynamic'], alpha=0.3),
            verticalalignment='top', family='monospace')

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('B. Cooling: Thermodynamic Operation (Energy Depletes)',
                fontsize=14, fontweight='bold', pad=20, color=colors['thermodynamic'])

    # ============================================================================
    # PANEL C: ENERGY CONSERVATION CONSTRAINT
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, :])

    # System energy over time
    time = np.linspace(0, 10, 100)

    # Total energy (conserved)
    E_total = np.ones_like(time) * 100

    # Molecule 1 energy (depleting)
    E_mol1 = 100 * np.exp(-0.3 * time)

    # Other molecules energy (increasing)
    E_others = E_total - E_mol1

    # Plot
    ax3.plot(time, E_total, linewidth=4, color='black',
            label='Total Energy (conserved)', linestyle='-', alpha=0.8)
    ax3.plot(time, E_mol1, linewidth=3, color=colors['thermodynamic'],
            label='Molecule 1 (depleting)', linestyle='--')
    ax3.plot(time, E_others, linewidth=3, color=colors['kinematic'],
            label='Other Molecules (increasing)', linestyle='-.')

    # Fill between
    ax3.fill_between(time, 0, E_mol1, alpha=0.3, color=colors['thermodynamic'])
    ax3.fill_between(time, E_mol1, E_total, alpha=0.3, color=colors['kinematic'])

    # Attempt to reheat (marked region)
    reheat_time = 7
    ax3.axvline(x=reheat_time, color=colors['highlight'], linestyle=':',
            linewidth=3, alpha=0.7)
    ax3.text(reheat_time, 105, 'Attempt to\nreheat Molecule 1',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Arrow showing energy must come from others
    arrow = FancyArrowPatch((reheat_time + 0.5, 80), (reheat_time + 0.5, 30),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=colors['highlight'])
    ax3.add_patch(arrow)
    ax3.text(reheat_time + 1.5, 55, 'Energy must\ncome from here',
            ha='left', fontsize=10, style='italic',
            color=colors['highlight'])

    ax3.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Energy', fontsize=13, fontweight='bold')
    ax3.set_title('C. Energy Conservation: Cannot Restore Without External Input',
                fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='center right', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 10])
    ax3.set_ylim([0, 110])

    # Add key insight
    ax3.text(0.5, 0.05,
            'KEY INSIGHT: Even if Molecule 1 is reheated, it MUST be cooler than original state\n'
            '(otherwise no energy was extracted → contradiction)',
            transform=ax3.transAxes, ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # ============================================================================
    # PANEL D: CATEGORICAL IRREVERSIBILITY
    # ============================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    # State diagram
    states = [
        {'name': 'C₀', 'pos': (0.2, 0.7), 'label': 'Initial\nE = E₀', 'color': colors['kinematic']},
        {'name': 'C₁', 'pos': (0.5, 0.7), 'label': 'Depleted\nE = E₀-ΔE', 'color': colors['thermodynamic']},
        {'name': 'C₂', 'pos': (0.8, 0.7), 'label': 'Reheated\nE = E₀-ΔE+δE', 'color': colors['neutral']},
    ]

    # Draw states
    for state in states:
        x, y = state['pos']
        circle = Circle((x, y), 0.08, color=state['color'], alpha=0.6,
                    edgecolor='black', linewidth=2, zorder=3)
        ax4.add_patch(circle)
        ax4.text(x, y, state['name'], ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', zorder=4)
        ax4.text(x, y - 0.15, state['label'], ha='center', va='top',
                fontsize=9, style='italic')

    # Transitions
    # C₀ → C₁ (energy extraction)
    arrow1 = FancyArrowPatch((0.28, 0.7), (0.42, 0.7),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=3, color='black')
    ax4.add_patch(arrow1)
    ax4.text(0.35, 0.78, 'Extract\nΔE', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # C₁ → C₂ (attempt to reheat)
    arrow2 = FancyArrowPatch((0.58, 0.7), (0.72, 0.7),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=3, color='black')
    ax4.add_patch(arrow2)
    ax4.text(0.65, 0.78, 'Add\nδE', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # C₂ ≠ C₀ (cannot return)
    arrow3 = FancyArrowPatch((0.8, 0.62), (0.2, 0.62),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=3, color='red', linestyle='--')
    ax4.add_patch(arrow3)
    ax4.text(0.5, 0.55, '❌ IMPOSSIBLE', ha='center', fontsize=11,
            fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Completion markers
    ax4.text(0.2, 0.85, 'μ(C₀, t>0) = 0', ha='center', fontsize=9,
            style='italic', color='red')
    ax4.text(0.5, 0.85, 'μ(C₁, t>0) = 1', ha='center', fontsize=9,
            style='italic', color='green')

    # Explanation
    ax4.text(0.5, 0.35,
            'Categorical Irreversibility:\n'
            'C₀ → C₁ (completed, irreversible)\n'
            'C₁ → C₂ (new state, NOT C₀)\n'
            'C₂ ≠ C₀ (different configuration)\n\n'
            'Why? δE came from elsewhere in system\n'
            '→ Total system state changed\n'
            '→ Cannot return to original C₀',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            family='monospace')

    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_title('D. Categorical Irreversibility: Cannot Return to Original State',
                fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # PANEL E: MATHEMATICAL COMPARISON
    # ============================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Create comparison table
    comparison_data = [
        ['Property', 'FTL (Kinematic)', 'Cooling (Thermodynamic)'],
        ['─' * 20, '─' * 20, '─' * 25],
        ['Observable', 'Position x(t)', 'Energy E(t)'],
        ['Conservation', 'NOT conserved', 'Conserved (1st law)'],
        ['Finiteness', 'Unbounded', 'Finite (bounded)'],
        ['Depletion', 'NO depletion', 'YES depletion'],
        ['Reference', 'Advances', 'Depletes'],
        ['Observation', 'No energy cost', 'Energy extraction'],
        ['Reversibility', 'Reversible', 'Irreversible'],
        ['', '', ''],
        ['Triangular', 'AMPLIFICATION ✓', 'DEPLETION ✗'],
        ['Mechanism', 'See advanced state', 'See depleted state'],
        ['Result', 'Speed × A^N', 'Cooling / A^N'],
        ['Factor', 'A = 2.847', 'A = 6.7 (worse)'],
    ]

    # Format table with colors
    y_pos = 0.95
    for i, row in enumerate(comparison_data):
        if i == 0:  # Header
            text = f"{row[0]:<20} | {row[1]:<20} | {row[2]:<25}"
            ax5.text(0.05, y_pos, text, fontsize=10, fontweight='bold',
                    family='monospace', verticalalignment='top')
        elif '─' in row[0]:  # Separator
            ax5.plot([0.05, 0.95], [y_pos - 0.01, y_pos - 0.01], 'k-',
                    linewidth=1, alpha=0.5)
        elif row[0] == 'Triangular':  # Highlight result
            text = f"{row[0]:<20} | {row[1]:<20} | {row[2]:<25}"
            ax5.text(0.05, y_pos, text, fontsize=10, fontweight='bold',
                    family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            text = f"{row[0]:<20} | {row[1]:<20} | {row[2]:<25}"
            ax5.text(0.05, y_pos, text, fontsize=9,
                    family='monospace', verticalalignment='top')

        y_pos -= 0.06

    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.set_title('E. Mathematical Comparison: Kinematic vs Thermodynamic',
                fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # PANEL F: FUNDAMENTAL THEOREM
    # ============================================================================
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    # Theorem statement
    theorem_text = (
        "FUNDAMENTAL THEOREM: Triangular Amplification Asymmetry\n\n"

        "Statement:\n"
        "In a closed system with finite energy, triangular self-referencing amplifies kinematic\n"
        "operations but depletes thermodynamic operations due to conservation constraints.\n\n"

        "Proof:\n"
        "1. Kinematic (FTL):\n"
        "   • Observable: Position x(t) [not conserved, unbounded]\n"
        "   • Reference evolution: x₁(t+Δt) > x₁(t) [advances]\n"
        "   • Observation: No energy cost, no depletion\n"
        "   • Result: Later projectiles see ADVANCED state → Amplification ✓\n"
        "   • Formula: v_final = v₀ × A^N where A > 1\n\n"

        "2. Thermodynamic (Cooling):\n"
        "   • Observable: Energy E(t) [conserved, finite]\n"
        "   • Reference evolution: E₁(t+Δt) < E₁(t) [depletes]\n"
        "   • Observation: Energy extraction, depletion occurs\n"
        "   • Result: Later molecules see DEPLETED state → Depletion ✗\n"
        "   • Formula: T_final = T₀ / (A^N) where A > 1 (worse than standard)\n\n"

        "3. Irreversibility:\n"
        "   • Even reheating Molecule 1 cannot restore original state\n"
        "   • Reason: Energy must come from elsewhere (finite system)\n"
        "   • Categorical: C₀ → C₁ (completed), C₁ ≠ C₀ (irreversible)\n"
        "   • Conclusion: E₁(t') < E₁(0) for all t' > 0 (always depleted)\n\n"

        "QED: Triangular amplification succeeds for kinematic but fails for thermodynamic operations."
    )

    ax6.text(0.5, 0.95, theorem_text,
            transform=ax6.transAxes,
            ha='center', va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax6.set_title('F. Fundamental Theorem: Asymmetry of Triangular Amplification',
                fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # OVERALL TITLE AND ANNOTATIONS
    # ============================================================================
    fig.suptitle('Kinematic vs Thermodynamic Asymmetry: Why Triangular Amplification Works for FTL but Fails for Cooling',
                fontsize=16, fontweight='bold', y=0.995)

    # Add key insight box
    fig.text(0.5, 0.002,
            'KEY INSIGHT: Even reheating cannot restore original state (finite energy) → Otherwise no energy was extracted (contradiction)\n'
            'Triangular amplification: Kinematic ✓ (position advances) | Thermodynamic ✗ (energy depletes)',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7, pad=10))

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig('theoretical_kinematic_vs_thermodynamic_asymmetry.png', dpi=300, bbox_inches='tight')
    print("✓ Figure saved: theoretical_kinematic_vs_thermodynamic_asymmetry.png")
    plt.close()
