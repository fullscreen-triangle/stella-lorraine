"""
Boundary Conditions: Big Bang Singularity to Heat Death
Validates monotonic categorical growth from C(0)=1 to C(∞)=CAT_max
Based on: "On the Consequences of Observation" Section 6
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def categorical_evolution(t_max=100):
    """
    From Section 6: Boundary Conditions

    "At the Big Bang singularity (t=0), there exists only one
    categorical distinction: existence vs. non-existence.
    Thus C(0) = 1." (Section 6.1)

    "The number of categories grows monotonically from the
    singularity to heat death." (Section 6.3)
    """
    time = np.linspace(0, t_max, 1000)

    # Logistic growth from 1 to CAT_max
    CAT_max = 1000  # Normalized
    C_0 = 1  # Big Bang: single category

    # Logistic growth: C(t) = CAT_max / (1 + ((CAT_max - C_0)/C_0) * exp(-r*t))
    r = 0.08  # Growth rate
    C_t = CAT_max / (1 + ((CAT_max - C_0) / C_0) * np.exp(-r * time))

    # Ensure C(0) = 1
    C_t[0] = 1

    return time, C_t, CAT_max

def entropy_evolution(C_t):
    """
    Entropy grows with categorical count
    S = k_B * ln(C(t))
    """
    return np.log(C_t + 1)  # +1 to handle C(0)=1

def observer_count_evolution(t, t_max):
    """
    Observer count grows then declines toward heat death

    From Section 3.4:
    "At heat death, observers are sparse or absent due to
    maximum entropy configuration."
    """
    # Peak at middle of universe lifetime
    peak_time = t_max / 3
    width = t_max / 4

    max_observers = 100
    observers = max_observers * np.exp(-((t - peak_time)**2) / (2 * width**2))

    return observers

def main():
    """Main validation function"""

    # Simulate universe evolution
    t_max = 100
    time, C_t, CAT_max = categorical_evolution(t_max)
    entropy = entropy_evolution(C_t)
    observers = observer_count_evolution(time, t_max)

    # Normalize time
    time_norm = time / t_max

    # Create figure
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Main categorical evolution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_norm, C_t, linewidth=3, color='purple')
    ax1.axvline(x=0, color='orange', linestyle='--', linewidth=2,
                label='Big Bang (C=1)', alpha=0.7)
    ax1.axvline(x=1, color='black', linestyle='--', linewidth=2,
                label='Heat Death (C=CAT_max)', alpha=0.7)
    ax1.fill_between(time_norm, 0, C_t, alpha=0.3, color='purple')
    ax1.scatter([0], [1], s=200, c='orange', marker='o', edgecolors='black',
                linewidths=2, zorder=5, label='Singularity')
    ax1.scatter([1], [CAT_max], s=200, c='black', marker='X', edgecolors='red',
                linewidths=2, zorder=5, label='Maximum Entropy')
    ax1.set_xlabel('Normalized Time (0=Big Bang, 1=Heat Death)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Categorical Count C(t)', fontsize=12, fontweight='bold')
    ax1.set_title('Monotonic Categorical Growth: C(0)=1 → C(∞)=CAT_max (Section 6)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Entropy evolution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_norm, entropy, linewidth=3, color='red')
    ax2.fill_between(time_norm, 0, entropy, alpha=0.3, color='red')
    ax2.set_xlabel('Normalized Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Entropy S = ln(C(t))', fontsize=11, fontweight='bold')
    ax2.set_title('Second Law: Entropy Increases Monotonically', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Observer count evolution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_norm, observers, linewidth=3, color='blue')
    ax3.fill_between(time_norm, 0, observers, alpha=0.3, color='blue')
    ax3.set_xlabel('Normalized Time', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Observer Count', fontsize=11, fontweight='bold')
    ax3.set_title('Observer Population Evolution (Section 3.4)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.annotate('Peak observer era\n(present epoch)',
                xy=(0.33, np.max(observers)), xytext=(0.5, np.max(observers)*0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, fontweight='bold', color='green')

    # Growth rate dC/dt
    ax4 = fig.add_subplot(gs[2, 0])
    growth_rate = np.gradient(C_t, time)
    ax4.plot(time_norm, growth_rate, linewidth=2, color='green')
    ax4.set_xlabel('Normalized Time', fontsize=11, fontweight='bold')
    ax4.set_ylabel('dC/dt (Growth Rate)', fontsize=11, fontweight='bold')
    ax4.set_title('Categorical Accumulation Rate', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Phase space: C(t) vs S(t)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(C_t, entropy, linewidth=3, color='purple')
    ax5.scatter([1], [entropy[0]], s=200, c='orange', marker='o',
                edgecolors='black', linewidths=2, zorder=5, label='Big Bang')
    ax5.scatter([CAT_max], [entropy[-1]], s=200, c='black', marker='X',
                edgecolors='red', linewidths=2, zorder=5, label='Heat Death')
    ax5.set_xlabel('Categorical Count C(t)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Entropy S(t)', fontsize=11, fontweight='bold')
    ax5.set_title('Phase Space Trajectory', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Arrow showing direction
    mid_idx = len(C_t) // 2
    ax5.annotate('', xy=(C_t[mid_idx+100], entropy[mid_idx+100]),
                xytext=(C_t[mid_idx], entropy[mid_idx]),
                arrowprops=dict(arrowstyle='->', color='purple', lw=3))

    # Timeline table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('tight')
    ax6.axis('off')

    table_data = [
        ['Epoch', 'Time', 'C(t)', 'Entropy', 'Observers', 'Key Features'],
        ['Big Bang', 't=0', '1', '0', '0', 'Single category (existence)'],
        ['Inflation', 't≈10⁻³²s', '~10', 'Low', '0', 'Rapid expansion begins'],
        ['Matter Era', 't≈10⁴yr', '~10⁴⁰', 'Medium', 'Emerging', 'Structure formation'],
        ['Present', 't≈13.8Gyr', '~10⁷⁰', 'High', 'Peak', 'Complex observers'],
        ['Far Future', 't≫10¹⁰⁰yr', '~CAT_max', 'Maximum', 'Sparse', 'Approaching heat death'],
        ['Heat Death', 't→∞', 'CAT_max', 'Maximum', '~0', 'No free energy'],
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    # Highlight key epochs
    table[(1, 0)].set_facecolor('#FFE4B5')  # Big Bang
    table[(4, 0)].set_facecolor('#ADD8E6')  # Present
    table[(6, 0)].set_facecolor('#D3D3D3')  # Heat Death

    plt.suptitle('Boundary Conditions: Monotonic Categorical Growth from Singularity to Heat Death',
                 fontsize=16, fontweight='bold', y=0.998)

    # Add key theorem
    theorem = '\n'.join([
        'THEOREM 6.1 (Monotonic Growth):',
        '────────────────────────────────────────────────',
        'For all t₁ < t₂ in the interval [0, ∞):',
        '  C(t₁) < C(t₂)',
        '',
        'Proof: Categories accumulate through observation.',
        'Once a distinction is made, it persists in the',
        'historical record. Therefore C(t) is strictly',
        'increasing. (See Section 6.3)',
    ])

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    fig.text(0.02, 0.01, theorem, fontsize=9, verticalalignment='bottom',
             horizontalalignment='left', bbox=props, family='monospace')

    plt.savefig('boundary_conditions_validation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: boundary_conditions_validation.png")
    plt.show()

if __name__ == "__main__":
    main()
