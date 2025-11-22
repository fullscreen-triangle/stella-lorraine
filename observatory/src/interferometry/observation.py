"""
Figure 16: Observation Creates Categories
Demonstrates how observation transforms continuous oscillations into discrete categorical states
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib import patches
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'reality': '#2E86AB',
    'observation': '#A23B72',
    'category': '#F18F01',
    'timeline': '#C73E1D'
}

if __name__ == "__main__":

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================================
    # PANEL A: CONTINUOUS OSCILLATIONS (REALITY)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    t = np.linspace(0, 4*np.pi, 1000)
    # Superposition of multiple frequencies
    psi = (np.sin(t) + 0.5*np.sin(2.3*t) + 0.3*np.sin(3.7*t) +
        0.2*np.sin(5.1*t) + 0.15*np.sin(7.3*t))

    ax1.plot(t, psi, linewidth=2, color=colors['reality'], alpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.fill_between(t, psi, alpha=0.2, color=colors['reality'])

    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude ψ(t)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Continuous Oscillations (Reality)',
                fontsize=14, fontweight='bold', pad=20)

    # Add mathematical expression
    ax1.text(0.5, 0.95, r'$\psi(t) = \sum_n A_n e^{i\omega_n t}$',
            transform=ax1.transAxes, fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center', va='top')

    # Add label
    ax1.text(0.5, 0.05, 'Reality: Always exists (continuous)',
            transform=ax1.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor=colors['reality'], alpha=0.3),
            ha='center', va='bottom', style='italic')

    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 4*np.pi])

    # ============================================================================
    # PANEL B: OBSERVATION EVENT
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Before observation
    t1 = np.linspace(0, 2*np.pi, 500)
    psi1 = (np.sin(t1) + 0.5*np.sin(2.3*t1) + 0.3*np.sin(3.7*t1))

    # Observation moment
    t_obs = 2*np.pi
    psi_obs = (np.sin(t_obs) + 0.5*np.sin(2.3*t_obs) + 0.3*np.sin(3.7*t_obs))

    # After observation (terminated)
    t2 = np.linspace(2*np.pi, 4*np.pi, 500)

    # Plot continuous part
    ax2.plot(t1, psi1, linewidth=2, color=colors['reality'], alpha=0.8,
            label='Before observation')
    ax2.fill_between(t1, psi1, alpha=0.2, color=colors['reality'])

    # Plot observation point
    ax2.scatter([t_obs], [psi_obs], s=300, color=colors['observation'],
            zorder=5, marker='*', edgecolors='black', linewidths=2,
            label='Observation')

    # Plot terminated region
    ax2.fill_between(t2, -2, 2, alpha=0.3, color='gray',
                    label='Terminated (no longer in reality)')
    ax2.plot(t2, np.zeros_like(t2), 'k--', alpha=0.5, linewidth=2)

    # Observation arrow
    arrow = FancyArrowPatch((t_obs, psi_obs + 1), (t_obs, psi_obs + 0.1),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3, color=colors['observation'])
    ax2.add_patch(arrow)
    ax2.text(t_obs, psi_obs + 1.3, 'OBSERVATION',
            ha='center', fontsize=12, fontweight='bold',
            color=colors['observation'])

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=t_obs, color=colors['observation'], linestyle=':',
            alpha=0.5, linewidth=2)

    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude ψ(t)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Observation Event',
                fontsize=14, fontweight='bold', pad=20)

    # Add label
    ax2.text(0.5, 0.05, 'Observation: Creates categorical completion (irreversible)',
            transform=ax2.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor=colors['observation'], alpha=0.3),
            ha='center', va='bottom', style='italic')

    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 4*np.pi])
    ax2.set_ylim([-2, 2.5])

    # ============================================================================
    # PANEL C: CATEGORICAL STATE
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Show discrete categorical state
    categories = ['C₁', 'C₂', 'C₃', 'C₄', 'C₅']
    n_cats = len(categories)

    # Create categorical space visualization
    for i, cat in enumerate(categories):
        y_pos = i

        # Draw category circle
        if i == 2:  # Highlight the observed category
            circle = Circle((0.5, y_pos), 0.15, color=colors['category'],
                        alpha=0.8, zorder=3)
            ax3.add_patch(circle)
            ax3.text(0.5, y_pos, cat, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white', zorder=4)

            # Add completion marker
            ax3.text(0.8, y_pos, 'μ(C₃, t) = 1', ha='left', va='center',
                    fontsize=11, style='italic',
                    bbox=dict(boxstyle='round', facecolor=colors['category'],
                            alpha=0.3))

            # Add label
            ax3.text(1.5, y_pos, '← Completed (terminated)', ha='left',
                    va='center', fontsize=10, color=colors['category'],
                    fontweight='bold')
        else:
            circle = Circle((0.5, y_pos), 0.15, color='lightgray',
                        alpha=0.5, zorder=2)
            ax3.add_patch(circle)
            ax3.text(0.5, y_pos, cat, ha='center', va='center',
                    fontsize=12, color='gray', zorder=3)

            if i < 2:
                ax3.text(0.8, y_pos, 'μ = 1', ha='left', va='center',
                        fontsize=9, style='italic', color='gray')
            else:
                ax3.text(0.8, y_pos, 'μ = 0', ha='left', va='center',
                        fontsize=9, style='italic', color='gray')

    # Add arrows showing irreversibility
    for i in range(n_cats - 1):
        if i <= 2:
            arrow = FancyArrowPatch((0.5, i + 0.2), (0.5, i + 0.8),
                                arrowstyle='->', mutation_scale=20,
                                linewidth=2, color='black', alpha=0.5)
            ax3.add_patch(arrow)

    ax3.set_xlim([0, 2.5])
    ax3.set_ylim([-0.5, n_cats - 0.5])
    ax3.set_aspect('equal')
    ax3.axis('off')

    ax3.set_title('C. Categorical State',
                fontsize=14, fontweight='bold', pad=20)

    # Add mathematical expression
    ax3.text(0.5, -0.3, r'Category: Terminated state (irreversible)',
            transform=ax3.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor=colors['category'], alpha=0.3),
            ha='center', va='top', style='italic')

    # Add irreversibility note
    ax3.text(1.25, n_cats - 0.5,
            'Irreversibility:\nμ(Cᵢ, t′) ≥ μ(Cᵢ, t)\nfor t′ > t',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # ============================================================================
    # PANEL D: MEASUREMENT HISTORY
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Timeline of categorical completions
    n_measurements = 8
    times = np.linspace(0, 10, n_measurements)
    categories_seq = [f'C₁', 'C₃', 'C₂', 'C₅', 'C₄', 'C₇', 'C₆', 'C₈']

    # Draw timeline
    ax4.plot([0, 10], [0.5, 0.5], 'k-', linewidth=3, alpha=0.5)

    # Draw measurement events
    for i, (t, cat) in enumerate(zip(times, categories_seq)):
        # Event marker
        ax4.scatter([t], [0.5], s=200, color=colors['timeline'],
                zorder=5, marker='o', edgecolors='black', linewidths=2)

        # Category label
        ax4.text(t, 0.7, cat, ha='center', va='bottom',
                fontsize=11, fontweight='bold')

        # Time label
        ax4.text(t, 0.3, f't_{i+1}', ha='center', va='top',
                fontsize=9, style='italic', color='gray')

        # Completion arrow
        if i < n_measurements - 1:
            arrow = FancyArrowPatch((t + 0.1, 0.5), (times[i+1] - 0.1, 0.5),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color=colors['timeline'],
                                alpha=0.5)
            ax4.add_patch(arrow)

    ax4.set_xlim([-0.5, 10.5])
    ax4.set_ylim([0, 1])
    ax4.axis('off')

    ax4.set_title('D. Measurement History',
                fontsize=14, fontweight='bold', pad=20)

    # Add label
    ax4.text(0.5, 0.05,
            'Measurement = Categorical navigation (discrete completion events)',
            transform=ax4.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor=colors['timeline'], alpha=0.3),
            ha='center', va='bottom', style='italic')

    # Add mathematical expression
    ax4.text(0.5, 0.95,
            r'History: $\mathcal{H} = \{(C_1, t_1), (C_2, t_2), \ldots, (C_N, t_N)\}$',
            transform=ax4.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center', va='top')

    # Add completion ordering
    ax4.text(5, 0.15,
            'Completion ordering: C₁ ≺ C₃ ≺ C₂ ≺ C₅ ≺ ...',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ============================================================================
    # OVERALL TITLE AND ANNOTATIONS
    # ============================================================================
    fig.suptitle('Observation Creates Categories: From Continuous Reality to Discrete Structure',
                fontsize=16, fontweight='bold', y=0.98)

    # Add key insight box
    fig.text(0.5, 0.01,
            'KEY INSIGHT: Observation is not passive measurement but active creation of categorical structure.\n'
            'Continuous oscillations terminate upon observation, creating discrete categorical states that cannot be re-occupied.',
            ha='center', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=10),
            wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('figure_16_observation_creates_categories.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 16 saved: figure_16_observation_creates_categories.png")
    plt.close()
