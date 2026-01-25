"""
The True Zero: Visualizing ⊙
Shows why x must be written as 1 but acts as 0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
import matplotlib.patches as mpatches

def main():
    """Visualize the true zero ⊙"""

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # THE PARADOX
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    ax1.text(5, 9, 'THE PARADOX: Why x Cannot Be A Number',
            fontsize=16, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Show the problem
    ax1.text(2, 7, 'If x is a number:', fontsize=13, ha='left', fontweight='bold')

    problems = [
        '1. x can be subdivided: x/2, x/3, x/4, ...',
        '2. Each subdivision is a category',
        '3. Therefore x is made of categories',
        '4. But we\'re counting categories!',
        '5. So x should be in (∞-x)',
        '6. CONTRADICTION ❌'
    ]

    y = 6
    for i, problem in enumerate(problems):
        color = 'black' if i < 5 else 'red'
        weight = 'normal' if i < 5 else 'bold'
        ax1.text(2.5, y - i*0.7, problem, fontsize=11, ha='left',
                color=color, fontweight=weight)

    # Show the solution
    ax1.text(8, 7, 'Solution:', fontsize=13, ha='right', fontweight='bold', color='green')

    ax1.text(8, 6.3, 'x = ⊙', fontsize=20, ha='right', fontweight='bold', color='green')
    ax1.text(8, 5.8, '(Observational Unit)', fontsize=10, ha='right', style='italic')

    solutions = [
        '• NOT a number',
        '• Indivisible',
        '• Contains 0 categories',
        '• Written as 1',
        '• Acts as 0',
        '• TRUE ZERO ✓'
    ]

    y = 5
    for i, sol in enumerate(solutions):
        ax1.text(8, y - i*0.5, sol, fontsize=11, ha='right', color='green')

    # THE TRUE ZERO
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    ax2.text(5, 9, 'The True Zero: ⊙', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Draw ⊙
    circle = Circle((5, 5), 2, facecolor='white', edgecolor='black', linewidth=3)
    ax2.add_patch(circle)
    ax2.text(5, 5, '⊙', fontsize=60, ha='center', va='center', fontweight='bold')

    # Properties
    props = [
        ('Written as', '1', 2),
        ('Acts as', '0', 8),
        ('Contains', '0 categories', 5),
    ]

    for text, value, x in props:
        ax2.text(x, 2, text, fontsize=10, ha='center', style='italic')
        ax2.text(x, 1.3, value, fontsize=14, ha='center', fontweight='bold', color='red')

    ax2.text(5, 0.3, 'Indivisible • Non-additive • Irreducible',
            fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # COMPARISON WITH REGULAR ZERO
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    ax3.text(5, 9, 'Regular Zero vs True Zero', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Table
    table_data = [
        ['Property', 'Regular 0', 'True Zero ⊙'],
        ['Exists?', 'No (nothing)', 'Yes (observer)'],
        ['Divisible?', 'Yes (0/2 = 0)', 'No (undefined)'],
        ['Additive?', 'Yes (0+0 = 0)', 'No (⊙+⊙ ≠ 2⊙)'],
        ['Categories?', '0', '0'],
        ['Representation', '0', '1'],
        ['Physical', 'Absence', 'Observer'],
    ]

    y = 7.5
    for i, row in enumerate(table_data):
        if i == 0:
            # Header
            ax3.text(2, y, row[0], fontsize=10, ha='center', fontweight='bold')
            ax3.text(5, y, row[1], fontsize=10, ha='center', fontweight='bold')
            ax3.text(8, y, row[2], fontsize=10, ha='center', fontweight='bold')
        else:
            ax3.text(2, y - i*0.8, row[0], fontsize=9, ha='center')
            ax3.text(5, y - i*0.8, row[1], fontsize=9, ha='center', color='blue')
            ax3.text(8, y - i*0.8, row[2], fontsize=9, ha='center', color='green')

    # THE ∞ - ⊙ STRUCTURE
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    ax4.text(5, 9, 'The ∞ - ⊙ Structure', fontsize=14, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Draw universe
    universe = Circle((5, 5), 3, facecolor='lightblue', edgecolor='black', linewidth=2, alpha=0.5)
    ax4.add_patch(universe)
    ax4.text(5, 7.5, '∞ (Total)', fontsize=12, ha='center', fontweight='bold')

    # Draw observer
    observer = Circle((5, 5), 0.5, facecolor='red', edgecolor='black', linewidth=2)
    ax4.add_patch(observer)
    ax4.text(5, 5, '⊙', fontsize=20, ha='center', va='center', fontweight='bold', color='white')

    # Label observable
    ax4.text(2, 5, '∞ - ⊙\n(Observable)', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax4.text(5, 2, 'Observer cannot observe\nthemselves completely',
            fontsize=9, ha='center', style='italic')

    # PHYSICAL INTERPRETATION
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.axis('off')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)

    ax5.text(5, 9, 'Physical Interpretation', fontsize=15, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Draw diagram
    # Observable matter
    obs_box = FancyBboxPatch((1, 5), 3, 2, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=2)
    ax5.add_patch(obs_box)
    ax5.text(2.5, 6, 'Observable\nMatter', fontsize=11, ha='center', fontweight='bold')
    ax5.text(2.5, 5.3, '∞ - ⊙', fontsize=10, ha='center', style='italic')

    # Observer
    obs_circle = Circle((5.5, 6), 0.4, facecolor='red', edgecolor='black', linewidth=2)
    ax5.add_patch(obs_circle)
    ax5.text(5.5, 6, '⊙', fontsize=16, ha='center', va='center',
            fontweight='bold', color='white')

    # Dark matter
    dark_box = FancyBboxPatch((6.5, 5), 3, 2, boxstyle="round,pad=0.1",
                             facecolor='gray', edgecolor='black', linewidth=2)
    ax5.add_patch(dark_box)
    ax5.text(8, 6, 'Dark\nMatter', fontsize=11, ha='center', fontweight='bold', color='white')
    ax5.text(8, 5.3, '≈ 5.4 × ⊙', fontsize=10, ha='center', style='italic', color='white')

    # Explanation
    explanation = [
        '• Observer (⊙) cannot observe themselves',
        '• Creates inaccessible region',
        '• Manifests as dark matter',
        '• Ratio ≈ 5.4 from observer network',
    ]

    y = 3.5
    for i, exp in enumerate(explanation):
        ax5.text(5, y - i*0.6, exp, fontsize=11, ha='center')

    # MATHEMATICAL CONSISTENCY
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)

    ax6.text(5, 9, 'Mathematical\nConsistency', fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Equations
    equations = [
        ('∞ - ⊙ + ⊙ = ∞', '✓', 'green'),
        ('(∞ - ⊙) / ⊙', 'undefined', 'blue'),
        ('⊙ < ∞', '✓', 'green'),
        ('⊙ ≠ 0', '✓', 'green'),
        ('⊙ / 2', 'undefined', 'blue'),
        ('⊙ + ⊙ ≠ 2⊙', '✓', 'green'),
    ]

    y = 7.5
    for i, (eq, result, color) in enumerate(equations):
        ax6.text(3, y - i*0.9, eq, fontsize=11, ha='left', family='monospace')
        ax6.text(7, y - i*0.9, result, fontsize=11, ha='right',
                fontweight='bold', color=color)

    ax6.text(5, 1, 'All paradoxes\nresolved ✓', fontsize=12, ha='center',
            fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle('The True Zero: Why x = ⊙ (Written as 1, Acts as 0)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Add key insight
    insight = '\n'.join([
        'KEY INSIGHT:',
        '═══════════════════════════════════════════════════════',
        'x cannot be a number because numbers are categories.',
        'If x were a number, it could be subdivided, creating',
        'more categories—contradiction.',
        '',
        'Instead, x = ⊙ (Observational Unit):',
        '  • Written as 1 (single unified entity)',
        '  • Acts as 0 (contains no categories)',
        '  • Indivisible (cannot be subdivided)',
        '  • Represents the observer\'s own existence',
        '',
        'This is the TRUE ZERO of categorical space.',
    ])

    props = dict(boxstyle='round', facecolor='yellow', alpha=0.9)
    fig.text(0.5, 0.01, insight, fontsize=10, verticalalignment='bottom',
             horizontalalignment='center', bbox=props, family='monospace',
             fontweight='bold')

    plt.savefig('true_zero_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: true_zero_visualization.png")
    plt.show()

if __name__ == "__main__":
    main()
