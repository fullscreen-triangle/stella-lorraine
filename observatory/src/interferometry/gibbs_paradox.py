"""
Figure 19: Gibbs Paradox Resolution via Categorical Irreversibility
Demonstrates how categorical completion resolves the 150-year-old Gibbs paradox
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Wedge
from matplotlib import patches
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'paradox': '#D32F2F',
    'resolution': '#388E3C',
    'entropy': '#F57C00',
    'oscillatory': '#7B1FA2'
}

if __name__ == "__main__":

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                        height_ratios=[1, 1, 1.2])

    # ============================================================================
    # PANEL A: TRADITIONAL PARADOX
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Similarity parameter (0 = identical, 1 = completely different)
    similarity = np.linspace(0, 1, 1000)

    # Traditional mixing entropy (discontinuous)
    S_mix_traditional = np.zeros_like(similarity)
    S_mix_traditional[similarity > 0.5] = 1.0  # Discontinuous jump

    ax1.plot(similarity, S_mix_traditional, linewidth=3,
            color=colors['paradox'], label='Traditional')

    # Mark discontinuity
    ax1.axvline(x=0.5, color=colors['paradox'], linestyle='--',
            alpha=0.5, linewidth=2)
    ax1.scatter([0.5], [0.5], s=500, color=colors['paradox'],
            marker='X', edgecolors='black', linewidths=3, zorder=5)

    # Add annotation
    ax1.annotate('DISCONTINUITY\n(Paradox)', xy=(0.5, 0.5),
                xytext=(0.7, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['paradox']),
                fontsize=11, fontweight='bold', color=colors['paradox'],
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax1.set_xlabel('Gas Similarity Parameter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mixing Entropy (k_B units)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Traditional Gibbs Paradox',
                fontsize=14, fontweight='bold', pad=20)

    # Add labels
    ax1.text(0.25, 0.1, 'Identical gases\nΔS = 0', ha='center',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax1.text(0.75, 0.9, 'Different gases\nΔS = k_B ln(2)', ha='center',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.2])

    # ============================================================================
    # PANEL B: CATEGORICAL IRREVERSIBILITY
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Draw mixing process
    # Initial state (separated)
    rect_left = Rectangle((0.1, 0.6), 0.15, 0.3,
                        facecolor='red', alpha=0.5, edgecolor='black', linewidth=2)
    rect_right = Rectangle((0.25, 0.6), 0.15, 0.3,
                        facecolor='blue', alpha=0.5, edgecolor='black', linewidth=2)
    ax2.add_patch(rect_left)
    ax2.add_patch(rect_right)
    ax2.text(0.175, 0.75, 'A', ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.325, 0.75, 'B', ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.25, 0.55, 'C_separated', ha='center', fontsize=10, style='italic')

    # Mixing arrow
    arrow1 = FancyArrowPatch((0.45, 0.75), (0.55, 0.75),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color=colors['resolution'])
    ax2.add_patch(arrow1)
    ax2.text(0.5, 0.8, 'MIXING', ha='center', fontsize=11, fontweight='bold',
            color=colors['resolution'])

    # Final state (mixed)
    rect_mixed = Rectangle((0.6, 0.6), 0.3, 0.3,
                        facecolor='purple', alpha=0.5, edgecolor='black', linewidth=2)
    ax2.add_patch(rect_mixed)
    # Add dots representing mixed molecules
    np.random.seed(42)
    for _ in range(20):
        x = np.random.uniform(0.6, 0.9)
        y = np.random.uniform(0.6, 0.9)
        color = 'red' if np.random.rand() > 0.5 else 'blue'
        ax2.scatter([x], [y], s=30, color=color, alpha=0.7)
    ax2.text(0.75, 0.55, 'C_mixed', ha='center', fontsize=10, style='italic')

    # Attempted separation (blocked)
    arrow2 = FancyArrowPatch((0.75, 0.5), (0.75, 0.4),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color=colors['paradox'], linestyle='--')
    ax2.add_patch(arrow2)

    # X mark (impossible)
    ax2.text(0.75, 0.35, '❌', fontsize=40, ha='center', color=colors['paradox'])
    ax2.text(0.75, 0.25, 'IMPOSSIBLE\n(Categorical irreversibility)',
            ha='center', fontsize=10, fontweight='bold', color=colors['paradox'],
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Back to separated (can't happen)
    rect_left2 = Rectangle((0.6, 0.05), 0.15, 0.15,
                        facecolor='red', alpha=0.3, edgecolor='gray',
                        linewidth=2, linestyle='--')
    rect_right2 = Rectangle((0.75, 0.05), 0.15, 0.15,
                            facecolor='blue', alpha=0.3, edgecolor='gray',
                            linewidth=2, linestyle='--')
    ax2.add_patch(rect_left2)
    ax2.add_patch(rect_right2)
    ax2.text(0.75, 0.01, 'C_separated (cannot return)', ha='center',
            fontsize=9, style='italic', color='gray')

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('B. Categorical Irreversibility',
                fontsize=14, fontweight='bold', pad=20)

    # Add explanation
    ax2.text(0.5, 0.95,
            'Once mixed (C_mixed completed), cannot return to C_separated',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # ============================================================================
    # PANEL C: OSCILLATORY ENTROPY
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Time evolution
    t = np.linspace(0, 10, 1000)

    # Oscillatory patterns (before observation)
    oscillation1 = np.sin(2*np.pi*t) + 0.5*np.sin(4*np.pi*t)
    oscillation2 = np.sin(2*np.pi*t + np.pi/4) + 0.5*np.sin(4*np.pi*t + np.pi/3)

    # Plot oscillations
    ax3.plot(t[:500], oscillation1[:500], linewidth=2,
            color=colors['oscillatory'], alpha=0.7, label='Gas A')
    ax3.plot(t[:500], oscillation2[:500], linewidth=2,
            color='orange', alpha=0.7, label='Gas B')

    # Observation/mixing event
    t_obs = 5
    ax3.axvline(x=t_obs, color='red', linestyle='--', linewidth=3, alpha=0.7)
    ax3.text(t_obs, 2.5, 'OBSERVATION\n(Mixing)', ha='center',
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # After observation (terminated)
    ax3.fill_between(t[500:], -3, 3, alpha=0.3, color='gray',
                    label='Terminated (mixed state)')

    # Termination probability
    ax3.text(7.5, -2, r'$\alpha$ = termination probability', ha='center',
            fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Oscillatory Amplitude', fontsize=12, fontweight='bold')
    ax3.set_title('C. Oscillatory Entropy Formulation',
                fontsize=14, fontweight='bold', pad=20)

    # Add entropy formula
    ax3.text(0.5, 0.95, r'$S = k_B \ln(\alpha)$',
            transform=ax3.transAxes, ha='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 10])
    ax3.set_ylim([-3, 3])

    # ============================================================================
    # PANEL D: RESOLUTION (SMOOTH ENTROPY)
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Similarity parameter
    similarity = np.linspace(0, 1, 1000)

    # Categorical resolution (smooth)
    S_mix_categorical = 1 - np.exp(-5*similarity)  # Smooth transition

    ax4.plot(similarity, S_mix_categorical, linewidth=3,
            color=colors['resolution'], label='Categorical resolution')

    # Compare with traditional (discontinuous)
    S_mix_traditional = np.zeros_like(similarity)
    S_mix_traditional[similarity > 0.5] = 1.0
    ax4.plot(similarity, S_mix_traditional, linewidth=2, linestyle='--',
            color=colors['paradox'], alpha=0.5, label='Traditional (paradox)')

    # Fill region
    ax4.fill_between(similarity, S_mix_categorical, alpha=0.3,
                    color=colors['resolution'])

    ax4.set_xlabel('Gas Similarity Parameter', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mixing Entropy (k_B units)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Resolution: Smooth Entropy via Categorical Completion',
                fontsize=14, fontweight='bold', pad=20)

    # Add annotation
    ax4.text(0.5, 0.5, 'NO DISCONTINUITY\n✓ Paradox resolved',
            ha='center', fontsize=12, fontweight='bold',
            color=colors['resolution'],
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1.2])

    # ============================================================================
    # PANEL E: MIXING-SEPARATION CYCLE (FULL WIDTH)
    # ============================================================================
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Draw cycle diagram
    cycle_states = [
        {'name': 'Separated', 'pos': (0.15, 0.5), 'cat': 'C_sep', 'color': 'blue'},
        {'name': 'Mixed', 'pos': (0.5, 0.7), 'cat': 'C_mix', 'color': 'purple'},
        {'name': 'Separated?', 'pos': (0.85, 0.5), 'cat': 'C_sep?', 'color': 'gray'},
    ]

    # Draw states
    for state in cycle_states:
        x, y = state['pos']
        circle = Circle((x, y), 0.08, color=state['color'], alpha=0.5,
                    edgecolor='black', linewidth=2, zorder=3)
        ax5.add_patch(circle)
        ax5.text(x, y, state['cat'], ha='center', va='center',
                fontsize=11, fontweight='bold', zorder=4)
        ax5.text(x, y - 0.12, state['name'], ha='center',
                fontsize=10, style='italic')

    # Forward process (mixing) - ALLOWED
    arrow_forward = FancyArrowPatch((0.23, 0.55), (0.42, 0.65),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=4, color=colors['resolution'])
    ax5.add_patch(arrow_forward)
    ax5.text(0.325, 0.62, 'MIXING\n(allowed)', ha='center',
            fontsize=11, fontweight='bold', color=colors['resolution'],
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Attempted reverse (separation) - FORBIDDEN
    arrow_reverse = FancyArrowPatch((0.58, 0.65), (0.77, 0.55),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=4, color=colors['paradox'],
                                linestyle='--')
    ax5.add_patch(arrow_reverse)
    ax5.text(0.675, 0.62, 'SEPARATION\n(forbidden)', ha='center',
            fontsize=11, fontweight='bold', color=colors['paradox'],
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # X mark on forbidden transition
    ax5.text(0.675, 0.5, '❌', fontsize=50, ha='center', color=colors['paradox'])

    # Entropy changes
    ax5.text(0.325, 0.45, 'ΔS > 0\n(entropy increases)', ha='center',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax5.text(0.675, 0.35, 'ΔS < 0?\n(impossible)', ha='center',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Explanation
    ax5.text(0.5, 0.15,
            'Categorical irreversibility: Once C_mix is completed, cannot return to C_sep\n'
            'This resolves Gibbs paradox: Full mixing-separation cycle ALWAYS increases entropy',
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Mathematical statement
    ax5.text(0.5, 0.05,
            r'$\oint dS > 0$ (cycle entropy always positive)',
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 0.8])
    ax5.set_title('E. Mixing-Separation Cycle: Categorical Irreversibility Ensures ΔS > 0',
                fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # OVERALL TITLE AND ANNOTATIONS
    # ============================================================================
    fig.suptitle('Resolution of Gibbs\' Paradox Through Categorical State Irreversibility',
                fontsize=16, fontweight='bold', y=0.99)

    # Add key insight box
    fig.text(0.5, 0.005,
            'KEY INSIGHT: Gibbs\' paradox (150-year-old problem) is resolved by categorical irreversibility.\n'
            'Physical configurations are distinguished by their position in an irreversible completion sequence.\n'
            'Once mixed (C_mixed completed), cannot return to separated (C_separated) state.\n'
            'Oscillatory entropy S = k_B ln(α) provides smooth transition, eliminating discontinuity.',
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5, pad=10))

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig('figure_19_gibbs_paradox_resolution.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 19 saved: figure_19_gibbs_paradox_resolution.png")
    plt.close()
