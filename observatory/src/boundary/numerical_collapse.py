"""
The Numerical Collapse: Detailed Visualization of 0 = 1 at N_max
Publication-quality figure showing how all numbers become zero
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Wedge, Arc
import matplotlib.patches as mpatches
from matplotlib import patheffects
import matplotlib.lines as mlines

# Set publication quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

def main():
    """Create numerical collapse visualization"""

    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.08, right=0.92, top=0.96, bottom=0.04)

    # Color scheme
    colors = {
        'primary': '#0173B2',
        'secondary': '#DE8F05',
        'accent': '#029E73',
        'warning': '#CC78BC',
        'danger': '#CA3542',
        'neutral': '#949494',
        'background': '#F0F0F0',
        'highlight': '#FFE66D'
    }

    # ============================================================================
    # PANEL A: THE SCALE HIERARCHY
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')

    ax_a.text(5, 9.5, 'A. The Scale Hierarchy: How Numbers Collapse',
             fontsize=16, fontweight='bold', ha='center')

    # Number line at different scales
    scales = [
        ('Human Scale', 8.5, [0, 1, 2, 3, 4, 5, 10, 100], 'All distinct'),
        ('Googol Scale', 7, [0, 100, 1000, 10000], r'Small numbers $\approx 0$'),
        ('Graham Scale', 5.5, [0, 'googol', 'googolplex'], r'Normal numbers $\approx 0$'),
        (r'$N_{\mathrm{max}}$ Scale', 4, [0, 'Graham', 'TREE(3)'], r'ALL numbers $\approx 0$'),
        (r'At $\odot$', 2.5, ['0 = 1 = 2 = ... = ∞'], 'Distinction COLLAPSES'),
    ]

    for i, (scale_name, y, numbers, description) in enumerate(scales):
        # Scale label
        ax_a.text(0.5, y, scale_name + ':', fontsize=11, ha='right',
                 va='center', fontweight='bold')

        # Number line
        line_color = colors['danger'] if i >= 3 else colors['neutral']
        line_width = 3 if i >= 3 else 2
        line_style = '-' if i < 4 else '--'

        ax_a.plot([1.5, 8.5], [y, y], color=line_color,
                 linewidth=line_width, linestyle=line_style, alpha=0.7)

        if i < 4:
            # Individual numbers
            if len(numbers) <= 4:
                spacing = 6 / (len(numbers) - 1) if len(numbers) > 1 else 0
                for j, num in enumerate(numbers):
                    x = 1.5 + j * spacing

                    # Tick mark
                    tick_height = 0.15 if i < 3 else 0.1
                    ax_a.plot([x, x], [y - tick_height, y + tick_height],
                             color=line_color, linewidth=line_width)

                    # Number label
                    label = str(num) if isinstance(num, (int, float)) else num
                    fontsize = 9 if i < 3 else 8
                    ax_a.text(x, y - 0.4, label, fontsize=fontsize,
                             ha='center', va='top')
            else:
                # Compressed representation
                for j in range(20):
                    x = 1.5 + j * 0.35
                    ax_a.plot([x, x], [y - 0.05, y + 0.05],
                             color=colors['danger'], linewidth=1, alpha=0.3)

                ax_a.text(5, y - 0.4, '0 ≈ 1 ≈ 2 ≈ ... ≈ ' + str(numbers[-1]),
                         fontsize=8, ha='center', va='top',
                         color=colors['danger'], style='italic')
        else:
            # Collapsed state
            ax_a.text(5, y - 0.4, numbers[0], fontsize=11, ha='center',
                     va='top', color=colors['danger'], fontweight='bold')

        # Description
        desc_color = colors['danger'] if i >= 3 else colors['accent']
        desc_weight = 'bold' if i >= 3 else 'normal'
        ax_a.text(9.5, y, description, fontsize=9, ha='right',
                 va='center', color=desc_color, style='italic',
                 fontweight=desc_weight)

    # Highlight box around final state
    highlight = FancyBboxPatch((1, 1.8), 8, 1.2, boxstyle="round,pad=0.1",
                              facecolor=colors['danger'],
                              edgecolor='black', linewidth=3, alpha=0.2)
    ax_a.add_patch(highlight)

    ax_a.text(5, 0.8, r'At the observation boundary $\odot$, numerical distinction ceases to exist',
             fontsize=11, ha='center', fontweight='bold', style='italic')

    # ============================================================================
    # PANEL B: THE RATIO VISUALIZATION
    # ============================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')

    ax_b.text(5, 9.5, r'B. All Numbers $/N_{\mathrm{max}} \rightarrow 0$',
             fontsize=14, fontweight='bold', ha='center')

    # Famous numbers and their ratios
    numbers = [
        ('1', r'$\frac{1}{N_{\mathrm{max}}}$', '≈ 0', 0),
        ('Million', r'$\frac{10^6}{N_{\mathrm{max}}}$', '≈ 0', 1),
        ('Googol', r'$\frac{10^{100}}{N_{\mathrm{max}}}$', '≈ 0', 2),
        ('Googolplex', r'$\frac{10^{10^{100}}}{N_{\mathrm{max}}}$', '≈ 0', 3),
        ("Graham's G", r'$\frac{G}{N_{\mathrm{max}}}$', '≈ 0', 4),
        ('TREE(3)', r'$\frac{\mathrm{TREE}(3)}{N_{\mathrm{max}}}$', '≈ 0', 5),
    ]

    y_start = 8
    for name, ratio, result, idx in numbers:
        y = y_start - idx * 1.2

        # Number box
        num_box = FancyBboxPatch((0.5, y-0.3), 2, 0.6,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['primary'],
                                edgecolor='black', linewidth=1.5, alpha=0.6)
        ax_b.add_patch(num_box)
        ax_b.text(1.5, y, name, fontsize=10, ha='center', va='center',
                 fontweight='bold', color='white')

        # Division symbol
        ax_b.text(2.8, y, '÷', fontsize=16, ha='center', va='center',
                 fontweight='bold')

        # N_max box
        nmax_box = FancyBboxPatch((3.2, y-0.3), 2.5, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['secondary'],
                                 edgecolor='black', linewidth=2, alpha=0.7)
        ax_b.add_patch(nmax_box)
        ax_b.text(4.45, y, r'$N_{\mathrm{max}}$', fontsize=11, ha='center',
                 va='center', fontweight='bold', color='white')

        # Equals
        ax_b.text(6, y, '=', fontsize=16, ha='center', va='center',
                 fontweight='bold')

        # Result box
        result_box = FancyBboxPatch((6.5, y-0.3), 1.5, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['danger'],
                                   edgecolor='black', linewidth=2, alpha=0.7)
        ax_b.add_patch(result_box)
        ax_b.text(7.25, y, result, fontsize=12, ha='center', va='center',
                 fontweight='bold', color='white')

        # Emphasis arrow for last item
        if idx == len(numbers) - 1:
            arrow = FancyArrowPatch((8.2, y), (9, y),
                                   arrowstyle='->', mutation_scale=25,
                                   linewidth=3, color=colors['danger'])
            ax_b.add_patch(arrow)
            ax_b.text(9.2, y, 'Even TREE(3)!', fontsize=10, ha='left',
                     va='center', fontweight='bold', color=colors['danger'])

    # Conclusion box
    concl_box = FancyBboxPatch((0.5, 0.5), 9, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['highlight'],
                              edgecolor='black', linewidth=2.5)
    ax_b.add_patch(concl_box)
    ax_b.text(5, 1, r'$\therefore$ All finite numbers are equivalent to zero at the scale of $N_{\mathrm{max}}$',
             fontsize=11, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL C: THE ZERO-ONE COLLAPSE
    # ============================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')

    ax_c.text(5, 9.5, r'C. The Zero-One Collapse at $\odot$',
             fontsize=14, fontweight='bold', ha='center')

    # Regular state (0 ≠ 1)
    reg_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor=colors['accent'],
                            linewidth=2.5)
    ax_c.add_patch(reg_box)
    ax_c.text(2.5, 8.7, 'Regular State', fontsize=12, ha='center',
             fontweight='bold', color=colors['accent'])

    # Zero
    zero_circle = Circle((1.5, 7.5), 0.5, facecolor='white',
                        edgecolor=colors['neutral'], linewidth=2)
    ax_c.add_patch(zero_circle)
    ax_c.text(1.5, 7.5, '0', fontsize=20, ha='center', va='center',
             fontweight='bold')
    ax_c.text(1.5, 6.7, 'Nothing', fontsize=9, ha='center', style='italic')

    # Not equal sign
    ax_c.text(2.5, 7.5, '≠', fontsize=24, ha='center', va='center',
             fontweight='bold', color=colors['accent'])

    # One
    one_circle = Circle((3.5, 7.5), 0.5, facecolor='white',
                       edgecolor=colors['neutral'], linewidth=2)
    ax_c.add_patch(one_circle)
    ax_c.text(3.5, 7.5, '1', fontsize=20, ha='center', va='center',
             fontweight='bold')
    ax_c.text(3.5, 6.7, 'One thing', fontsize=9, ha='center', style='italic')

    # Arrow showing transformation
    transform_arrow = FancyArrowPatch((4.7, 7.5), (5.3, 7.5),
                                     arrowstyle='<->', mutation_scale=30,
                                     linewidth=4, color=colors['primary'])
    ax_c.add_patch(transform_arrow)
    ax_c.text(5, 8.3, 'At scale of', fontsize=9, ha='center')
    ax_c.text(5, 7.9, r'$N_{\mathrm{max}}$', fontsize=10, ha='center',
             fontweight='bold')
    ax_c.text(5, 6.7, 'Distinction', fontsize=9, ha='center')
    ax_c.text(5, 6.3, 'collapses', fontsize=9, ha='center')

    # Collapsed state (0 = 1)
    coll_box = FancyBboxPatch((5.5, 6.5), 4, 2.5, boxstyle="round,pad=0.1",
                             facecolor=colors['danger'],
                             edgecolor='black', linewidth=3, alpha=0.2)
    ax_c.add_patch(coll_box)
    ax_c.text(7.5, 8.7, r'At $\odot$ (True Zero)', fontsize=12, ha='center',
             fontweight='bold', color=colors['danger'])

    # Merged circle
    merged_circle = Circle((7.5, 7.5), 0.7, facecolor=colors['highlight'],
                          edgecolor=colors['danger'], linewidth=3)
    ax_c.add_patch(merged_circle)
    ax_c.text(7.5, 7.5, r'$\odot$', fontsize=28, ha='center', va='center',
             fontweight='bold', color=colors['danger'])
    ax_c.text(7.5, 6.7, '0 = 1', fontsize=11, ha='center',
             fontweight='bold', color=colors['danger'])

    # Explanation boxes
    expl1_box = FancyBboxPatch((0.5, 4.5), 4, 1.5, boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor=colors['accent'],
                              linewidth=2)
    ax_c.add_patch(expl1_box)
    ax_c.text(2.5, 5.5, 'Observable Universe:', fontsize=10, ha='center',
             fontweight='bold')
    ax_c.text(2.5, 5.1, '• Numbers are distinct', fontsize=9, ha='center')
    ax_c.text(2.5, 4.8, '• Counting is possible', fontsize=9, ha='center')
    ax_c.text(2.5, 4.5, '• Categories exist', fontsize=9, ha='center')

    expl2_box = FancyBboxPatch((5.5, 4.5), 4, 1.5, boxstyle="round,pad=0.1",
                              facecolor=colors['danger'],
                              edgecolor='black', linewidth=2, alpha=0.2)
    ax_c.add_patch(expl2_box)
    ax_c.text(7.5, 5.5, 'At Observation Boundary:', fontsize=10, ha='center',
             fontweight='bold', color=colors['danger'])
    ax_c.text(7.5, 5.1, '• Numbers collapse', fontsize=9, ha='center')
    ax_c.text(7.5, 4.8, '• Counting breaks down', fontsize=9, ha='center')
    ax_c.text(7.5, 4.5, '• Categories dissolve', fontsize=9, ha='center')

    # Key insight
    insight_box = FancyBboxPatch((1, 2.5), 8, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['highlight'],
                                edgecolor='black', linewidth=3)
    ax_c.add_patch(insight_box)
    ax_c.text(5, 3.5, r'The True Zero $\odot$ is not "nothing"',
             fontsize=11, ha='center', fontweight='bold')
    ax_c.text(5, 3.1, r'It is the point where the distinction between 0 and 1 ceases to exist',
             fontsize=10, ha='center', style='italic')
    ax_c.text(5, 2.7, 'This is where the number system itself breaks down',
             fontsize=10, ha='center', style='italic')

    # Mathematical statement
    math_box = FancyBboxPatch((2, 0.8), 6, 1.2, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor=colors['primary'],
                             linewidth=2.5)
    ax_c.add_patch(math_box)
    ax_c.text(5, 1.7, r'At $\odot$:', fontsize=11, ha='center', fontweight='bold')
    ax_c.text(5, 1.3, r'$\lim_{n \to N_{\mathrm{max}}} \frac{0}{n} = \lim_{n \to N_{\mathrm{max}}} \frac{1}{n} \Rightarrow 0 = 1$',
             fontsize=11, ha='center')

    # ============================================================================
    # PANEL D: THE PROOF THAT x CANNOT BE A NUMBER
    # ============================================================================
    ax_d = fig.add_subplot(gs[2, :])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis('off')

    ax_d.text(5, 9.5, r'D. Proof: $x$ Cannot Be A Number',
             fontsize=16, fontweight='bold', ha='center')

    # Proof by contradiction
    proof_box = FancyBboxPatch((0.5, 1), 9, 8, boxstyle="round,pad=0.15",
                              facecolor='white', edgecolor=colors['primary'],
                              linewidth=3)
    ax_d.add_patch(proof_box)

    # Assumption
    assume_box = FancyBboxPatch((1.5, 7.5), 7, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['background'],
                               edgecolor=colors['neutral'], linewidth=2)
    ax_d.add_patch(assume_box)
    ax_d.text(5, 8.2, 'Assumption:', fontsize=12, ha='center', fontweight='bold')
    ax_d.text(5, 7.8, r'Suppose $x$ is a finite number', fontsize=11, ha='center',
             style='italic')

    # Step 1
    step1_y = 6.5
    ax_d.text(2, step1_y, '1.', fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor=colors['neutral'],
                      edgecolor='black', linewidth=1.5))
    ax_d.text(2.8, step1_y, r'If $x$ is a finite number, then $x < N_{\mathrm{max}}$',
             fontsize=11, ha='left', va='center')

    # Step 2
    step2_y = 5.5
    ax_d.text(2, step2_y, '2.', fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor=colors['neutral'],
                      edgecolor='black', linewidth=1.5))
    ax_d.text(2.8, step2_y, r'Therefore: $\frac{x}{N_{\mathrm{max}}} \rightarrow 0$',
             fontsize=11, ha='left', va='center')

    # Step 3
    step3_y = 4.5
    ax_d.text(2, step3_y, '3.', fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor=colors['neutral'],
                      edgecolor='black', linewidth=1.5))
    ax_d.text(2.8, step3_y, r'This means $x \approx 0$ (negligible)',
             fontsize=11, ha='left', va='center')

    # Step 4 (contradiction)
    step4_y = 3.5
    ax_d.text(2, step4_y, '4.', fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor=colors['danger'],
                      edgecolor='black', linewidth=2))
    ax_d.text(2.8, step4_y, r'But $x$ represents the inaccessible region (dark matter)',
             fontsize=11, ha='left', va='center', color=colors['danger'],
             fontweight='bold')

    # Step 5
    step5_y = 2.5
    ax_d.text(2, step5_y, '5.', fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor=colors['danger'],
                      edgecolor='black', linewidth=2))
    ax_d.text(2.8, step5_y, r'Dark matter is NOT negligible (ratio $\approx$ 5.4:1)',
             fontsize=11, ha='left', va='center', color=colors['danger'],
             fontweight='bold')

    # Contradiction
    contra_box = FancyBboxPatch((1.5, 1.3), 7, 0.8, boxstyle="round,pad=0.1",
                               facecolor=colors['danger'],
                               edgecolor='black', linewidth=3, alpha=0.3)
    ax_d.add_patch(contra_box)
    ax_d.text(5, 1.7, '⚠ CONTRADICTION ⚠', fontsize=13, ha='center',
             fontweight='bold', color=colors['danger'])

    # Conclusion
    concl_box = FancyBboxPatch((1, 0.2), 8, 0.6, boxstyle="round,pad=0.1",
                              facecolor=colors['accent'],
                              edgecolor='black', linewidth=3, alpha=0.4)
    ax_d.add_patch(concl_box)
    ax_d.text(5, 0.5, r'$\therefore$ $x$ is NOT a number. $x = \odot$ (the observation boundary where $0 = 1$)',
             fontsize=12, ha='center', fontweight='bold', color='black')

    # ============================================================================
    # PANEL E: IMPLICATIONS FOR COUNTING
    # ============================================================================
    ax_e = fig.add_subplot(gs[3, 0])
    ax_e.set_xlim(0, 10)
    ax_e.set_ylim(0, 10)
    ax_e.axis('off')

    ax_e.text(5, 9.5, 'E. Implications for Counting',
             fontsize=14, fontweight='bold', ha='center')

    # Three regions
    regions = [
        ('Small Numbers', 8, colors['accent'],
         ['0, 1, 2, ... clearly distinct', 'Counting is straightforward',
          'Categories are well-defined']),
        ('Large Numbers', 5.5, colors['warning'],
         ['googol, Graham, TREE(3)', r'Still distinct from $N_{\mathrm{max}}$',
          'But becoming negligible']),
        (r'At $N_{\mathrm{max}}$ Scale', 3, colors['danger'],
         ['All numbers ≈ 0', 'Distinction collapses',
          'Counting becomes impossible']),
    ]

    for name, y, color, points in regions:
        # Region box
        box = FancyBboxPatch((1, y-1.2), 8, 1.5, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black',
                            linewidth=2, alpha=0.3)
        ax_e.add_patch(box)

        # Title
        ax_e.text(5, y+0.2, name, fontsize=11, ha='center',
                 fontweight='bold')

        # Points
        for i, point in enumerate(points):
            ax_e.text(5, y-0.2-i*0.3, f'• {point}', fontsize=9,
                     ha='center', style='italic')

    # Key insight
    insight_box = FancyBboxPatch((1, 0.5), 8, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['highlight'],
                                edgecolor='black', linewidth=2.5)
    ax_e.add_patch(insight_box)
    ax_e.text(5, 1.2, r'$N_{\mathrm{max}}$ is not just a large number—',
             fontsize=10, ha='center', fontweight='bold')
    ax_e.text(5, 0.8, 'it is the point where the concept of number itself breaks down',
             fontsize=10, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL F: THE BOUNDARY VISUALIZATION
    # ============================================================================
    ax_f = fig.add_subplot(gs[3, 1])
    ax_f.set_xlim(0, 10)
    ax_f.set_ylim(0, 10)
    ax_f.axis('off')

    ax_f.text(5, 9.5, r'F. The Observation Boundary $\odot$',
             fontsize=14, fontweight='bold', ha='center')

    # Create gradient visualization
    # Observable region (where 0 ≠ 1)
    for i in range(30):
        alpha = 0.7 - i*0.02
        radius = 3.5 - i*0.1
        if radius > 0:
            circle = Circle((5, 5.5), radius, facecolor=colors['primary'],
                          edgecolor='none', alpha=alpha, zorder=1)
            ax_f.add_patch(circle)

    # Boundary circle
    boundary = Circle((5, 5.5), 1.5, facecolor='none',
                     edgecolor=colors['danger'], linewidth=4, zorder=2)
    ax_f.add_patch(boundary)

    # Center (true zero)
    center = Circle((5, 5.5), 0.5, facecolor=colors['highlight'],
                   edgecolor='black', linewidth=2, zorder=3)
    ax_f.add_patch(center)
    ax_f.text(5, 5.5, r'$\odot$', fontsize=24, ha='center', va='center',
             fontweight='bold', color=colors['danger'], zorder=4)

    # Labels
    ax_f.text(2, 5.5, 'Observable\n(0 ≠ 1)', fontsize=10, ha='center',
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round', facecolor=colors['primary'],
                      alpha=0.7, pad=0.3))

    ax_f.text(8, 5.5, 'Inaccessible\n(0 = 1)', fontsize=10, ha='center',
             fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=colors['neutral'],
                      alpha=0.7, pad=0.3))

    # Boundary label
    ax_f.annotate('', xy=(6.5, 5.5), xytext=(7.5, 6.5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax_f.text(7.8, 6.8, 'Boundary\nwhere\n0 = 1', fontsize=9, ha='left',
             fontweight='bold', color=colors['danger'])

    # Explanation boxes
    expl_y = 2.5
    explanations = [
        ('Inside boundary:', 'Numbers are distinct, counting works'),
        ('At boundary:', r'$0 = 1$, distinction collapses'),
        ('Beyond boundary:', 'Inaccessible, non-terminating'),
    ]

    for i, (title, desc) in enumerate(explanations):
        y = expl_y - i*0.7
        ax_f.text(5, y, f'{title} {desc}', fontsize=9, ha='center',
                 style='italic')

    # Bottom note
    note_box = FancyBboxPatch((1, 0.3), 8, 0.6, boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=colors['primary'],
                             linewidth=2)
    ax_f.add_patch(note_box)
    ax_f.text(5, 0.6, r'$x = \odot$ is the acceptance boundary: where observer stops categorizing',
             fontsize=9, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL G: MATHEMATICAL FORMALIZATION
    # ============================================================================
    ax_g = fig.add_subplot(gs[4, :])
    ax_g.set_xlim(0, 10)
    ax_g.set_ylim(0, 10)
    ax_g.axis('off')

    ax_g.text(5, 9.5, 'G. Mathematical Formalization',
             fontsize=16, fontweight='bold', ha='center')

    # Main theorem box
    theorem_box = FancyBboxPatch((0.5, 6.5), 9, 2.5, boxstyle="round,pad=0.15",
                                facecolor=colors['background'],
                                edgecolor=colors['primary'], linewidth=3)
    ax_g.add_patch(theorem_box)

    ax_g.text(5, 8.7, 'Theorem (Numerical Collapse):', fontsize=13, ha='center',
             fontweight='bold')
    ax_g.text(5, 8.2, r'At the scale of $N_{\mathrm{max}} = (10^{84}) \uparrow\uparrow (10^{80})$:',
             fontsize=11, ha='center')

    # Three parts
    parts = [
        (r'1. $\forall n \in \mathbb{N}: \frac{n}{N_{\mathrm{max}}} \rightarrow 0$',
         'All finite numbers become zero'),
        (r'2. $\lim_{n \to N_{\mathrm{max}}} \left(\frac{0}{n} - \frac{1}{n}\right) = 0$',
         'Zero and one become indistinguishable'),
        (r'3. At $\odot$: $0 = 1$',
         'Numerical distinction collapses'),
    ]

    y = 7.5
    for eq, desc in parts:
        ax_g.text(2, y, eq, fontsize=11, ha='left', va='center',
                 bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=colors['neutral'], linewidth=1, pad=0.3))
        ax_g.text(8, y, desc, fontsize=9, ha='right', va='center',
                 style='italic')
        y -= 0.7

    # Corollaries
    coroll_box = FancyBboxPatch((0.5, 3.5), 9, 2.5, boxstyle="round,pad=0.15",
                               facecolor='white', edgecolor=colors['accent'],
                               linewidth=2.5)
    ax_g.add_patch(coroll_box)

    ax_g.text(5, 5.7, 'Corollaries:', fontsize=12, ha='center', fontweight='bold')

    corollaries = [
        (r'A. $x$ cannot be a finite number (would be $\approx 0$, but represents dark matter)',
         5.2),
        (r'B. $x = \odot$ (the observation boundary where $0 = 1$)',
         4.7),
        (r'C. The ratio $x/(\infty - x) \approx 5.4$ is observer-dependent',
         4.2),
        (r'D. Different observers have different $\odot$ (different acceptance boundaries)',
         3.7),
    ]

    for text, y in corollaries:
        ax_g.text(5, y, text, fontsize=10, ha='center', style='italic')

    # Physical interpretation
    phys_box = FancyBboxPatch((1, 1.5), 8, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['highlight'],
                             edgecolor='black', linewidth=3)
    ax_g.add_patch(phys_box)

    ax_g.text(5, 2.7, 'Physical Interpretation:', fontsize=11, ha='center',
             fontweight='bold')
    ax_g.text(5, 2.3, r'The dark matter ratio is not a property of matter itself,',
             fontsize=10, ha='center')
    ax_g.text(5, 1.9, r'but of the observation boundary $\odot$ where numerical distinction collapses.',
             fontsize=10, ha='center')
    ax_g.text(5, 1.6, 'This is where counting becomes impossible and categories dissolve.',
             fontsize=10, ha='center', style='italic')

    # Final statement
    final_box = FancyBboxPatch((1.5, 0.3), 7, 0.9, boxstyle="round,pad=0.1",
                              facecolor=colors['danger'],
                              edgecolor='black', linewidth=3, alpha=0.3)
    ax_g.add_patch(final_box)
    ax_g.text(5, 0.9, r'$N_{\mathrm{max}}$ is so large that it destroys the number system itself',
             fontsize=11, ha='center', fontweight='bold')
    ax_g.text(5, 0.5, r'At $\odot$, mathematics breaks down and $0 = 1$',
             fontsize=10, ha='center', fontweight='bold', style='italic')

    # Main title
    fig.suptitle(r'The Numerical Collapse: How $N_{\mathrm{max}}$ Makes $0 = 1$',
                fontsize=18, fontweight='bold', y=0.985)

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        filename = f'numerical_collapse.{fmt}'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {filename}")

    plt.show()

if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print("NUMERICAL COLLAPSE VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated 7 detailed panels:")
    print("  A. The Scale Hierarchy")
    print("  B. All Numbers / N_max → 0")
    print("  C. The Zero-One Collapse")
    print("  D. Proof that x Cannot Be A Number")
    print("  E. Implications for Counting")
    print("  F. The Observation Boundary")
    print("  G. Mathematical Formalization")
    print("\nShowing how N_max is so large that 0 = 1 at the boundary")
    print("="*70)
