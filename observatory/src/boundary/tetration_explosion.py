"""
Tetration Explosion: Visualizing Recursive Growth
Shows how C(t+1) = n^C(t) explodes beyond comprehension
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

def tetration_levels(base, max_level):
    """
    Calculate tetration values with overflow protection
    Returns log10(value) for plotting
    """
    levels = []
    current = 1  # C(0) = 1

    for level in range(max_level + 1):
        if level == 0:
            levels.append(0)  # log10(1) = 0
        else:
            # C(t+1) = base^C(t)
            if current > 100:  # Prevent overflow
                # Use logarithmic approximation
                log_value = current * np.log10(base)
                levels.append(log_value)
                current = log_value  # Continue in log space
            else:
                current = base ** current
                if current > 0 and np.isfinite(current):
                    levels.append(np.log10(current))
                else:
                    levels.append(levels[-1] * 2)  # Approximate doubling

    return levels

def main():
    """Main visualization function"""

    # Parameters
    base = 10  # Simplified from 10^84
    max_level = 6

    # Calculate tetration
    levels = tetration_levels(base, max_level)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Main explosion chart
    ax1 = fig.add_subplot(gs[0, :])

    x = np.arange(len(levels))
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(levels)))

    bars = ax1.bar(x, levels, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add explosion effect to last bar
    bars[-1].set_facecolor('red')
    bars[-1].set_edgecolor('darkred')
    bars[-1].set_linewidth(4)

    ax1.set_xlabel('Recursion Level t', fontsize=13, fontweight='bold')
    ax1.set_ylabel('log₁₀(C(t))', fontsize=13, fontweight='bold')
    ax1.set_title('Tetration Explosion: C(t+1) = n^C(t) Growth',
                  fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f't={i}' for i in x], fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, levels)):
        if i == 0:
            label = 'C(0) = 1'
        elif i == 1:
            label = f'C(1) = {base}'
        elif i == 2:
            label = f'C(2) = {base}^{base}'
        else:
            label = f'C({i}) ≈ 10^{val:.1e}'

        ax1.text(bar.get_x() + bar.get_width()/2, val + max(levels)*0.02,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold',
                rotation=0 if i < 3 else 45)

    # Growth rate comparison
    ax2 = fig.add_subplot(gs[1, 0])

    t_range = np.arange(1, 7)

    # Different growth rates
    import math
    linear = t_range
    exponential = 2 ** t_range
    factorial = np.array([math.factorial(t) for t in t_range])
    tetration = np.array([tetration_levels(2, t)[-1] for t in t_range])

    # Plot on log scale
    ax2.semilogy(t_range, linear, 'o-', label='Linear: t', linewidth=2, markersize=8)
    ax2.semilogy(t_range, exponential, 's-', label='Exponential: 2^t', linewidth=2, markersize=8)
    ax2.semilogy(t_range, factorial, '^-', label='Factorial: t!', linewidth=2, markersize=8)
    ax2.semilogy(t_range, 10**tetration, 'D-', label='Tetration: 2↑↑t',
                linewidth=3, markersize=10, color='red')

    ax2.set_xlabel('t', fontsize=11, fontweight='bold')
    ax2.set_ylabel('f(t) (log scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Growth Rate Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # Recursion diagram
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    # Draw recursion boxes
    boxes = [
        (1, 7, 'C(0) = 1'),
        (1, 5, f'C(1) = {base}^1 = {base}'),
        (1, 3, f'C(2) = {base}^{base}'),
        (1, 1, f'C(3) = {base}^({base}^{base})'),
    ]

    for i, (x, y, text) in enumerate(boxes):
        color = plt.cm.Reds(0.3 + 0.7 * i / len(boxes))
        box = FancyBboxPatch((x, y), 3, 1.5, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax3.add_patch(box)
        ax3.text(x + 1.5, y + 0.75, text, ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Draw arrow to next
        if i < len(boxes) - 1:
            arrow = FancyArrowPatch((x + 1.5, y), (x + 1.5, y - 0.4),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='darkred')
            ax3.add_patch(arrow)
            ax3.text(x + 2.5, y - 0.2, f'n^', fontsize=12,
                    fontweight='bold', color='darkred')

    # Add explosion indicator
    ax3.text(5.5, 5, '⚡ EXPLOSION ⚡', fontsize=20, fontweight='bold',
            color='red', ha='center', bbox=dict(boxstyle='round',
            facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=3))

    ax3.text(5.5, 3.5, 'Each level multiplies\nthe exponent by n',
            fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax3.set_title('Recursive Structure: Why Tetration Explodes',
                 fontsize=13, fontweight='bold', pad=20)

    # Numerical table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')

    # Calculate actual values for table
    table_data = [
        ['Level t', 'C(t) Formula', 'Approximate Value', 'Number of Digits', 'Can Write?'],
        ['0', '1', '1', '1', '✓ Yes'],
        ['1', '10^84', '10^84', '84', '✓ Yes'],
        ['2', '(10^84)^(10^84)', '10^(8.4×10^85)', '8.4×10^85', '✗ No (exceeds atoms)'],
        ['3', '(10^84)^(10^(8.4×10^85))', '10^(8.4×10^(8.4×10^85))', '8.4×10^(8.4×10^85)', '✗ No (incomprehensible)'],
        ['4', 'Tower continues...', 'Beyond notation', 'Beyond notation', '✗ No (beyond math)'],
        ['...', '...', '...', '...', '...'],
        ['10^80', 'N_max', '(10^84)↑↑(10^80)', '???', '✗ No (effectively ∞)'],
    ]

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.1, 0.25, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight key rows
    table[(2, 0)].set_facecolor('#FFE4B5')  # Level 2
    table[(3, 0)].set_facecolor('#FFB6C1')  # Level 3
    table[(7, 0)].set_facecolor('#FF6B6B')  # N_max

    for i in range(5):
        table[(7, i)].set_facecolor('#FF6B6B')
        table[(7, i)].set_text_props(weight='bold')

    plt.suptitle('Tetration Explosion: Why N_max Exceeds All Known Numbers',
                 fontsize=18, fontweight='bold', y=0.98)

    # Add theorem box
    theorem = '\n'.join([
        'THEOREM 5.1 (Categorical Recursion):',
        '─────────────────────────────────────────',
        'C(t+1) = n^C(t)  where n ≈ 10^84',
        '',
        'Solution: C(t) = n ↑↑ t  (tetration)',
        '',
        'At t = 10^80:',
        '  N_max = (10^84) ↑↑ (10^80)',
        '',
        'This exceeds:',
        '  • Graham\'s number: G ≪ N_max',
        '  • TREE(3): TREE(3) ≪ N_max',
        '  • All known numbers: Σ(all) ≈ 0',
    ])

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    fig.text(0.02, 0.01, theorem, fontsize=10, verticalalignment='bottom',
             horizontalalignment='left', bbox=props, family='monospace')

    plt.savefig('tetration_explosion.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tetration_explosion.png")
    plt.show()

if __name__ == "__main__":
    main()
