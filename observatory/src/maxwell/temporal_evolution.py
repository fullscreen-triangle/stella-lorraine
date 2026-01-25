"""
figure_3_temporal_dynamics.py

Publication figure showing temporal evolution and face switching.
Tests 3 and 4 results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

COLORS = {
    'front': '#0173B2',
    'back': '#DE8F05',
    'separation': '#029E73',
}

def create_figure_3():
    """
    Figure 3: Temporal Dynamics

    Panel A: Synchronized evolution (Test 3)
    Panel B: Conjugate separation over time
    Panel C: Automatic face switching (Test 4)
    Panel D: Observable face timeline
    """

    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                          left=0.1, right=0.95, top=0.93, bottom=0.08)

    # Panel A: Evolution of S_k coordinates
    ax_a = fig.add_subplot(gs[0, :])
    plot_panel_a_evolution(ax_a)

    # Panel B: Separation over time
    ax_b = fig.add_subplot(gs[1, 0])
    plot_panel_b_separation(ax_b)

    # Panel C: Face switching
    ax_c = fig.add_subplot(gs[1, 1])
    plot_panel_c_switching(ax_c)

    # Overall title
    fig.suptitle('Temporal Dynamics: Evolution & Switching (Tests 3-4)',
                fontsize=12, fontweight='bold', y=0.98)

    return fig

def plot_panel_a_evolution(ax):
    """Panel A: Synchronized dual evolution from Test 3"""

    # Data from Test 3
    time_points = [0, 0.1, 0.3, 0.5]

    # Front face evolution
    front_sk = [1.342, 1.342, 1.342, 1.341]
    front_st = [0.000, 0.020, 0.040, 0.050]
    front_se = [0.000, 0.000, 0.000, 0.000]

    # Back face evolution (conjugate)
    back_sk = [-x for x in front_sk]
    back_st = [-x for x in front_st]
    back_se = [-x for x in front_se]

    # Plot S_k
    ax.plot(time_points, front_sk, 'o-', color=COLORS['front'],
           linewidth=2, markersize=6, label=r'Front $S_k$')
    ax.plot(time_points, back_sk, 's-', color=COLORS['back'],
           linewidth=2, markersize=6, label=r'Back $S_k$')

    # Plot S_t on secondary axis
    ax2 = ax.twinx()
    ax2.plot(time_points, front_st, '^--', color=COLORS['front'],
            linewidth=1.5, markersize=5, alpha=0.7, label=r'Front $S_t$')
    ax2.plot(time_points, back_st, 'v--', color=COLORS['back'],
            linewidth=1.5, markersize=5, alpha=0.7, label=r'Back $S_t$')

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel(r'$S_k$ Coordinate', fontsize=9, color='black')
    ax2.set_ylabel(r'$S_t$ Coordinate', fontsize=9, color='gray')
    ax.set_title('Synchronized Dual Evolution (Test 3)',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
             loc='upper left', fontsize=7, ncol=2, framealpha=0.9)

    # Panel label
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    # Add conjugate relationship annotation
    ax.annotate('', xy=(0.5, -1.341), xytext=(0.5, 1.341),
               arrowprops=dict(arrowstyle='<->', color=COLORS['separation'],
                             lw=2, ls='--'))
    ax.text(0.52, 0, 'Conjugate\nRelation', fontsize=7,
           color=COLORS['separation'], style='italic')

def plot_panel_b_separation(ax):
    """Panel B: Categorical separation over time"""

    # Data from Test 3
    time_points = [0, 0.1, 0.3, 0.5]
    separation = [2.685, 2.682, 2.682, 2.683]

    # Plot separation
    ax.plot(time_points, separation, 'o-', color=COLORS['separation'],
           linewidth=2.5, markersize=8)

    # Mean line
    mean_sep = np.mean(separation)
    ax.axhline(mean_sep, color='red', linestyle='--', linewidth=1.5,
              label=f'Mean: {mean_sep:.3f}')

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Categorical Separation', fontsize=9)
    ax.set_title('Conjugate Separation (Constant)',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)

    # Panel label
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    # Add note
    ax.text(0.5, 0.05, 'Separation remains constant during evolution',
           transform=ax.transAxes, ha='center', fontsize=7, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

def plot_panel_c_switching(ax):
    """Panel C: Automatic face switching from Test 4"""

    # Data from Test 4
    time = np.linspace(0, 1, 1000)
    frequency = 5.0  # Hz

    # Observable face (0 = front, 1 = back)
    # Switches at: 0.25, 0.45, 0.65, 0.90
    switch_times = [0.25, 0.45, 0.65, 0.90]

    # Create step function
    observable = np.zeros_like(time)
    current_face = 0
    for switch_time in switch_times:
        observable[time >= switch_time] = 1 - observable[time >= switch_time]

    # Plot
    ax.fill_between(time, 0, observable, where=(observable == 0),
                    color=COLORS['front'], alpha=0.3, label='Front observable')
    ax.fill_between(time, 0, observable, where=(observable == 1),
                    color=COLORS['back'], alpha=0.3, label='Back observable')
    ax.plot(time, observable, color='black', linewidth=2)

    # Mark switch points
    for st in switch_times:
        ax.axvline(st, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(st, 0.5, 'ro', markersize=8)

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Observable Face', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Front', 'Back'])
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f'Automatic Switching at {frequency} Hz (Test 4)',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # Panel label
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    # Add switch count
    ax.text(0.5, 0.95, f'Total switches: {len(switch_times)}',
           transform=ax.transAxes, ha='center', fontsize=7,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

if __name__ == '__main__':
    fig = create_figure_3()
    plt.savefig('figure_3_temporal_dynamics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_3_temporal_dynamics.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved: figure_3_temporal_dynamics.pdf/png")
    plt.show()
