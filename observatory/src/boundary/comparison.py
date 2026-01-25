"""
Comparison Chart: N_max vs All Other Numbers (Scientific/Symbolic)
Minimal text, logarithmic visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

def main():
    """Create comparison chart"""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                  left=0.1, right=0.9, top=0.93, bottom=0.07)

    colors = {
        'small': '#95A5A6',
        'large': '#3498DB',
        'huge': '#9B59B6',
        'nmax': '#E74C3C',
    }

    # ============================================================================
    # PANEL A: LOGARITHMIC SCALE
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, :])

    # Numbers to compare (log scale)
    numbers = [
        ('$10^2$', 2, colors['small']),
        ('$10^6$', 6, colors['small']),
        ('$10^{100}$', 100, colors['large']),
        ('$10^{10^{100}}$', 10**100, colors['huge']),
        ('$G$', 10**10**65, colors['huge']),
        ('$\mathrm{TREE}(3)$', 10**10**10**100, colors['huge']),
        ('$N_{\mathrm{max}}$', 10**10**10**10**80, colors['nmax']),
    ]

    # Use log-log-log scale (symbolic)
    y_positions = [0.5, 1, 1.5, 2, 2.5, 3, 4.5]

    for i, (label, value, color) in enumerate(numbers):
        y = y_positions[i]

        # Bar
        if i < len(numbers) - 1:
            bar = Rectangle((0, y-0.15), 0.3 + i*0.2, 0.3,
                          facecolor=color, edgecolor='black',
                          linewidth=1.5, alpha=0.7)
            ax_a.add_patch(bar)
        else:
            # N_max gets special treatment
            bar = Rectangle((0, y-0.2), 8, 0.4,
                          facecolor=color, edgecolor='black',
                          linewidth=3, alpha=0.8)
            ax_a.add_patch(bar)

            # Arrow indicating it goes beyond
            arrow = FancyArrowPatch((8, y), (9, y),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color=color)
            ax_a.add_patch(arrow)

        # Label
        ax_a.text(-0.3, y, label, fontsize=12, ha='right', va='center',
                 fontweight='bold' if i == len(numbers)-1 else 'normal')

    # Annotations
    ax_a.text(5, 4.7, r'ALL OTHERS $\approx 0$', fontsize=14, ha='center',
             fontweight='bold', color=colors['nmax'])

    ax_a.text(1, 3.2, r'$\frac{\mathrm{TREE}(3)}{N_{\mathrm{max}}} \rightarrow 0$',
             fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', linewidth=1.5, pad=0.4))

    ax_a.set_xlim(-1, 10)
    ax_a.set_ylim(0, 5)
    ax_a.axis('off')
    ax_a.set_title(r'$\mathbf{A.}$ Magnitude Comparison (Symbolic Scale)',
                  fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # PANEL B: GROWTH RATES
    # ============================================================================
    ax_b = fig.add_subplot(gs[1, 0])

    # Symbolic growth rate diagram
    x = np.linspace(0, 10, 1000)

    # Different growth rates
    y1 = x  # Linear
    y2 = x**2 / 10  # Polynomial
    y3 = 2**x / 100  # Exponential
    y4 = np.minimum(10**(x/2), 10)  # Super-exponential

    ax_b.plot(x, y1, 'k-', linewidth=1, alpha=0.3, label='Linear')
    ax_b.plot(x, y2, 'k-', linewidth=1, alpha=0.4, label='Polynomial')
    ax_b.plot(x, y3, 'k-', linewidth=1.5, alpha=0.5, label='Exponential')
    ax_b.plot(x, y4, color=colors['huge'], linewidth=2, alpha=0.7,
             label='Tetration')

    # N_max (off the chart)
    ax_b.arrow(8, 9, 1.5, 0, head_width=0.3, head_length=0.3,
              fc=colors['nmax'], ec='black', linewidth=2)
    ax_b.text(9.7, 9, r'$N_{\mathrm{max}}$', fontsize=12, ha='left',
             va='center', fontweight='bold', color=colors['nmax'])

    # Labels
    ax_b.text(9, 1, r'$n$', fontsize=11, ha='right', va='bottom')
    ax_b.text(9, 2.5, r'$n^2$', fontsize=11, ha='right', va='bottom')
    ax_b.text(7, 6, r'$2^n$', fontsize=11, ha='right', va='bottom')
    ax_b.text(5.5, 9.5, r'$n \uparrow\uparrow t$', fontsize=11, ha='center',
             va='bottom', color=colors['huge'])

    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.set_xlabel(r'$n$', fontsize=12, fontweight='bold')
    ax_b.set_ylabel(r'$f(n)$', fontsize=12, fontweight='bold')
    ax_b.set_title(r'$\mathbf{B.}$ Growth Rate Hierarchy',
                  fontsize=14, fontweight='bold')
    ax_b.grid(True, alpha=0.2)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # ============================================================================
    # PANEL C: RATIO TABLE
    # ============================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')

    ax_c.set_title(r'$\mathbf{C.}$ Ratios to $N_{\mathrm{max}}$',
                  fontsize=14, fontweight='bold')

    # Table
    ratios = [
        (r'$\displaystyle\frac{1}{N_{\mathrm{max}}}$', r'$\approx 0$'),
        (r'$\displaystyle\frac{10^{100}}{N_{\mathrm{max}}}$', r'$\approx 0$'),
        (r'$\displaystyle\frac{G}{N_{\mathrm{max}}}$', r'$\approx 0$'),
        (r'$\displaystyle\frac{\mathrm{TREE}(3)}{N_{\mathrm{max}}}$', r'$\approx 0$'),
    ]

    y = 8.5
    for i, (ratio, result) in enumerate(ratios):
        # Ratio
        box1 = FancyBboxPatch((0.5, y-i*1.8-0.4), 5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='black',
                             linewidth=1.5)
        ax_c.add_patch(box1)
        ax_c.text(3, y-i*1.8, ratio, fontsize=11, ha='center', va='center')

        # Result
        box2 = FancyBboxPatch((6.5, y-i*1.8-0.4), 2.5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['nmax'], edgecolor='black',
                             linewidth=2, alpha=0.3)
        ax_c.add_patch(box2)
        ax_c.text(7.75, y-i*1.8, result, fontsize=12, ha='center',
                 va='center', fontweight='bold')

    # Conclusion
    concl_box = FancyBboxPatch((1, 0.5), 8, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['nmax'], edgecolor='black',
                              linewidth=2.5, alpha=0.2)
    ax_c.add_patch(concl_box)
    ax_c.text(5, 1, r'$\forall n < N_{\mathrm{max}}: \displaystyle\frac{n}{N_{\mathrm{max}}} \rightarrow 0$',
             fontsize=12, ha='center', fontweight='bold')

    # Main title
    fig.suptitle(r'$N_{\mathrm{max}}$ Exceeds All Other Finite Numbers',
                fontsize=16, fontweight='bold')

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        filename = f'comparison_chart.{fmt}'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {filename}")

    plt.show()

if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print("COMPARISON CHART COMPLETE")
    print("="*70)
