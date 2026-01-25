"""
Master Overview: The Complete N_max Framework
Publication-quality figure showing all key concepts
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Wedge
import matplotlib.patches as mpatches
from matplotlib import patheffects

# Set publication quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

def add_shadow(text_obj, offset=(2, -2), color='black', alpha=0.3):
    """Add shadow effect to text"""
    text_obj.set_path_effects([
        patheffects.withStroke(linewidth=3, foreground=color, alpha=alpha)
    ])

def main():
    """Create master overview figure"""

    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.08, right=0.92, top=0.96, bottom=0.04)

    # Color scheme (colorblind-friendly)
    colors = {
        'primary': '#0173B2',      # Blue
        'secondary': '#DE8F05',    # Orange
        'accent': '#029E73',       # Green
        'warning': '#CC78BC',      # Purple
        'danger': '#CA3542',       # Red
        'neutral': '#949494',      # Gray
        'background': '#F0F0F0',
        'highlight': '#FFE66D'
    }

    # ============================================================================
    # PANEL A: THE RECURSIVE FORMULA
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.axis('off')
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)

    # Title
    title = ax_a.text(5, 9, 'A. The Fundamental Recursion',
                     fontsize=16, fontweight='bold', ha='center')

    # Main equation box
    eq_box = FancyBboxPatch((1.5, 6), 7, 2.2, boxstyle="round,pad=0.15",
                           facecolor='white', edgecolor=colors['primary'],
                           linewidth=3, zorder=1)
    ax_a.add_patch(eq_box)

    # Equation
    ax_a.text(5, 7.5, r'$C(t+1) = n^{C(t)}$',
             fontsize=24, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=colors['highlight'],
                      alpha=0.3, pad=0.5))

    ax_a.text(5, 6.5, r'where $n \approx 10^{84}$ (entity-state pairs), $t \approx 10^{80}$ (observers)',
             fontsize=11, ha='center', style='italic')

    # Arrow to result
    arrow = FancyArrowPatch((5, 5.8), (5, 4.8),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color=colors['primary'])
    ax_a.add_patch(arrow)

    # Result box
    result_box = FancyBboxPatch((2, 3), 6, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['accent'], edgecolor='black',
                               linewidth=2, alpha=0.3)
    ax_a.add_patch(result_box)

    ax_a.text(5, 4, r'$N_{\mathrm{max}} = (10^{84}) \uparrow\uparrow (10^{80})$',
             fontsize=20, ha='center', fontweight='bold')
    ax_a.text(5, 3.4, 'The largest finite number',
             fontsize=12, ha='center', style='italic')

    # Properties
    props = [
        (r'$N_{\mathrm{max}} > $ Graham, TREE(3), all others', 1.5, 1.8),
        (r'All numbers $/N_{\mathrm{max}} \approx 0$', 5, 1.8),
        (r'Boundary between finite and infinite', 8.5, 1.8),
    ]

    for text, x, y in props:
        ax_a.text(x, y, text, fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=colors['neutral'], linewidth=1, pad=0.3))

    # ============================================================================
    # PANEL B: GROWTH RATE COMPARISON
    # ============================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')

    ax_b.text(5, 9.5, 'B. Growth Rate Comparison',
             fontsize=14, fontweight='bold', ha='center')

    # Logarithmic scale visualization
    functions = [
        ('Linear', 1, colors['neutral'], 0.3),
        ('Exponential', 2, colors['neutral'], 0.4),
        ('Factorial', 3, colors['neutral'], 0.5),
        ('Ackermann', 4, colors['secondary'], 0.6),
        ('Graham', 5, colors['warning'], 0.7),
        ('TREE(3)', 6, colors['danger'], 0.8),
        (r'$N_{\mathrm{max}}$', 7, colors['accent'], 1.0),
    ]

    y_base = 7.5
    for i, (name, level, color, alpha) in enumerate(functions):
        y = y_base - i * 1.0

        # Bar
        width = 0.5 + level * 1.0
        bar = Rectangle((1, y-0.3), width, 0.6,
                       facecolor=color, edgecolor='black',
                       linewidth=1.5, alpha=alpha)
        ax_b.add_patch(bar)

        # Label
        ax_b.text(0.5, y, name, fontsize=10, ha='right', va='center',
                 fontweight='bold' if i == len(functions)-1 else 'normal')

        # Emphasis for N_max
        if i == len(functions) - 1:
            ax_b.text(1 + width + 0.3, y, '← ALL OTHERS ≈ 0',
                     fontsize=11, ha='left', va='center',
                     fontweight='bold', color=colors['accent'])

    ax_b.text(5, 0.5, r'$N_{\mathrm{max}}$ exceeds all other numbers so extremely',
             fontsize=9, ha='center', style='italic')
    ax_b.text(5, 0.1, 'that they become effectively zero in comparison',
             fontsize=9, ha='center', style='italic')

    # ============================================================================
    # PANEL C: OSCILLATORY FOUNDATION
    # ============================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')

    ax_c.text(5, 9.5, 'C. Categories as Terminated Oscillations',
             fontsize=14, fontweight='bold', ha='center')

    # Draw continuous wave
    t = np.linspace(0, 4*np.pi, 1000)
    wave = np.sin(t)

    # Continuous part (95%)
    t_cont = t[:950]
    wave_cont = wave[:950]
    x_cont = t_cont / (4*np.pi) * 7 + 1.5
    y_cont = wave_cont * 1.2 + 6
    ax_c.plot(x_cont, y_cont, color=colors['primary'], linewidth=2.5,
             alpha=0.7, label='Continuous (95%)')

    # Terminated part (5%)
    t_term = t[950:]
    wave_term = wave[950:]
    x_term = t_term / (4*np.pi) * 7 + 1.5
    y_term = wave_term * 1.2 + 6
    ax_c.plot(x_term, y_term, color=colors['danger'], linewidth=3,
             label='Terminated (5%)')

    # Termination point
    ax_c.scatter([x_term[-1]], [y_term[-1]], s=300, c=colors['danger'],
               marker='X', edgecolors='black', linewidths=2, zorder=5)

    # Arrow to category
    arrow = FancyArrowPatch((x_term[-1], y_term[-1]-0.3), (5, 4.2),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color=colors['danger'])
    ax_c.add_patch(arrow)

    # Category box
    cat_box = FancyBboxPatch((3.5, 3), 3, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['highlight'],
                            edgecolor='black', linewidth=2)
    ax_c.add_patch(cat_box)
    ax_c.text(5, 3.5, 'CATEGORY', fontsize=13, ha='center',
             va='center', fontweight='bold')

    # Theorem
    theorem_box = FancyBboxPatch((1, 1), 8, 1.2, boxstyle="round,pad=0.1",
                                facecolor='white', edgecolor=colors['primary'],
                                linewidth=2)
    ax_c.add_patch(theorem_box)
    ax_c.text(5, 1.8, 'Theorem 7.1:', fontsize=10, ha='center', fontweight='bold')
    ax_c.text(5, 1.3, 'Category ⟺ Terminated Oscillation',
             fontsize=10, ha='center', style='italic')

    # ============================================================================
    # PANEL D: ENTROPY AS PATH SELECTION
    # ============================================================================
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis('off')

    ax_d.text(5, 9.5, 'D. Entropy as Path to Termination',
             fontsize=14, fontweight='bold', ha='center')

    # Initial state
    init_circle = Circle((2, 6), 0.4, facecolor=colors['primary'],
                        edgecolor='black', linewidth=2)
    ax_d.add_patch(init_circle)
    ax_d.text(2, 6, 'i', fontsize=12, ha='center', va='center',
             color='white', fontweight='bold')
    ax_d.text(2, 5, 'Initial', fontsize=9, ha='center')

    # Observable paths (entropy increasing)
    np.random.seed(42)
    for i in range(6):
        x_path = np.linspace(2, 8, 50)
        y_path = 6 + (i-2.5)*0.4 + 0.3*np.sin(x_path*2) + np.random.normal(0, 0.05, 50)
        ax_d.plot(x_path, y_path, color=colors['accent'],
                 linewidth=2, alpha=0.6)

        # Terminal point
        ax_d.scatter([8], [y_path[-1]], s=80, c=colors['accent'],
                   marker='o', edgecolors='black', linewidths=1)

    ax_d.text(5, 8, 'Observable Paths', fontsize=11, ha='center',
             fontweight='bold', color=colors['accent'])
    ax_d.text(5, 7.5, '(Entropy ↑, Terminated)', fontsize=9, ha='center',
             style='italic', color=colors['accent'])

    # Inaccessible paths (non-terminating)
    for i in range(3):
        x_path = np.linspace(2, 8, 50)
        y_path = 3 + i*0.5 + 0.2*np.cos(x_path*3)
        ax_d.plot(x_path, y_path, color=colors['danger'],
                 linewidth=2, alpha=0.5, linestyle='--')

    ax_d.text(5, 2, 'Inaccessible Paths (x)', fontsize=11, ha='center',
             fontweight='bold', color=colors['danger'])
    ax_d.text(5, 1.5, '(Entropy preserved, Non-terminated)', fontsize=9,
             ha='center', style='italic', color=colors['danger'])

    # Equation
    eq_box = FancyBboxPatch((3, 0.2), 4, 0.8, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor='black', linewidth=1.5)
    ax_d.add_patch(eq_box)
    ax_d.text(5, 0.6, r'$\frac{dS}{dt} = R_{\mathrm{cat}} \times P_{\mathrm{term}}$',
             fontsize=11, ha='center')

    # ============================================================================
    # PANEL E: THE ∞ - x STRUCTURE
    # ============================================================================
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.set_xlim(0, 10)
    ax_e.set_ylim(0, 10)
    ax_e.axis('off')

    ax_e.text(5, 9.5, r'E. The $\infty - x$ Structure',
             fontsize=14, fontweight='bold', ha='center')

    # Universe circle
    universe = Circle((5, 5.5), 3.5, facecolor=colors['background'],
                     edgecolor='black', linewidth=2.5, alpha=0.5)
    ax_e.add_patch(universe)
    ax_e.text(5, 9.2, r'$\infty = N_{\mathrm{max}}$', fontsize=12,
             ha='center', fontweight='bold')

    # Observable region
    obs_wedge = Wedge((5, 5.5), 3.5, 0, 200,
                     facecolor=colors['primary'],
                     edgecolor='black', linewidth=2, alpha=0.4)
    ax_e.add_patch(obs_wedge)
    ax_e.text(3, 5.5, 'Observable', fontsize=11, ha='center',
             fontweight='bold', color='white')
    ax_e.text(3, 5, r'$\infty - x$', fontsize=10, ha='center',
             style='italic', color='white')

    # Inaccessible region
    inac_wedge = Wedge((5, 5.5), 3.5, 200, 360,
                      facecolor=colors['danger'],
                      edgecolor='black', linewidth=2, alpha=0.4)
    ax_e.add_patch(inac_wedge)
    ax_e.text(7, 5.5, 'Inaccessible', fontsize=11, ha='center',
             fontweight='bold', color='white')
    ax_e.text(7, 5, r'$x = \odot$', fontsize=10, ha='center',
             style='italic', color='white')

    # Boundary line
    ax_e.plot([5, 5+3.5*np.cos(200*np.pi/180)],
             [5.5, 5.5+3.5*np.sin(200*np.pi/180)],
             'r-', linewidth=3)
    ax_e.text(3.5, 3.5, r'$\odot$', fontsize=16, ha='center',
             fontweight='bold', color=colors['danger'])

    # Ratio
    ratio_box = FancyBboxPatch((2.5, 1), 5, 0.8, boxstyle="round,pad=0.1",
                              facecolor=colors['highlight'],
                              edgecolor='black', linewidth=2)
    ax_e.add_patch(ratio_box)
    ax_e.text(5, 1.4, r'$\frac{x}{\infty - x} \approx 5.4$',
             fontsize=12, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL F: THE TRUE ZERO
    # ============================================================================
    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.set_xlim(0, 10)
    ax_f.set_ylim(0, 10)
    ax_f.axis('off')

    ax_f.text(5, 9.5, r'F. The True Zero: $0 = 1$ at $\odot$',
             fontsize=14, fontweight='bold', ha='center')

    # Regular zero
    circle1 = Circle((2.5, 6), 1.2, facecolor='white',
                    edgecolor=colors['neutral'], linewidth=2.5)
    ax_f.add_patch(circle1)
    ax_f.text(2.5, 6, '0', fontsize=32, ha='center', va='center',
             fontweight='bold', color=colors['neutral'])
    ax_f.text(2.5, 4.3, 'Regular Zero', fontsize=11, ha='center',
             fontweight='bold')
    ax_f.text(2.5, 3.8, '(nothing)', fontsize=9, ha='center',
             style='italic')
    ax_f.text(2.5, 3.3, r'$0 \neq 1$', fontsize=10, ha='center',
             color=colors['accent'])

    # Arrow
    arrow = FancyArrowPatch((3.7, 6), (6.3, 6),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color=colors['primary'])
    ax_f.add_patch(arrow)
    ax_f.text(5, 6.5, 'At scale of', fontsize=9, ha='center')
    ax_f.text(5, 6, r'$N_{\mathrm{max}}$', fontsize=10, ha='center',
             fontweight='bold')

    # True zero
    circle2 = Circle((7.5, 6), 1.2, facecolor=colors['highlight'],
                    edgecolor=colors['danger'], linewidth=3)
    ax_f.add_patch(circle2)
    ax_f.text(7.5, 6, r'$\odot$', fontsize=40, ha='center', va='center',
             fontweight='bold', color=colors['danger'])
    ax_f.text(7.5, 4.3, 'True Zero', fontsize=11, ha='center',
             fontweight='bold', color=colors['danger'])
    ax_f.text(7.5, 3.8, r'$(0 = 1)$', fontsize=10, ha='center',
             style='italic', color=colors['danger'])
    ax_f.text(7.5, 3.3, 'Distinction', fontsize=9, ha='center',
             color=colors['danger'])
    ax_f.text(7.5, 2.9, 'collapses', fontsize=9, ha='center',
             color=colors['danger'])

    # Explanation box
    expl_box = FancyBboxPatch((1, 0.5), 8, 1.8, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor=colors['primary'],
                             linewidth=2)
    ax_f.add_patch(expl_box)

    explanations = [
        r'At $\odot$, all finite numbers become equivalent to zero:',
        r'$1/N_{\mathrm{max}} \approx 0$, $\mathrm{googol}/N_{\mathrm{max}} \approx 0$, $\mathrm{Graham}/N_{\mathrm{max}} \approx 0$',
        r'Even the distinction between $0$ and $1$ collapses: $0 = 1$',
    ]

    y = 1.9
    for i, expl in enumerate(explanations):
        weight = 'bold' if i == 0 else 'normal'
        ax_f.text(5, y - i*0.4, expl, fontsize=9, ha='center',
                 fontweight=weight)

    # ============================================================================
    # PANEL G: PHYSICAL INTERPRETATION
    # ============================================================================
    ax_g = fig.add_subplot(gs[3, 1])
    ax_g.set_xlim(0, 10)
    ax_g.set_ylim(0, 10)
    ax_g.axis('off')

    ax_g.text(5, 9.5, 'G. Physical Interpretation',
             fontsize=14, fontweight='bold', ha='center')

    # Observable matter
    obs_rect = Rectangle((1, 6), 3.5, 2, facecolor=colors['primary'],
                        edgecolor='black', linewidth=2, alpha=0.6)
    ax_g.add_patch(obs_rect)
    ax_g.text(2.75, 7, 'Observable\nMatter', fontsize=11, ha='center',
             fontweight='bold', color='white')
    ax_g.text(2.75, 6.3, r'$\infty - x$', fontsize=10, ha='center',
             style='italic', color='white')

    # Observer
    obs_circle = Circle((4.75, 7), 0.5, facecolor=colors['secondary'],
                       edgecolor='black', linewidth=2)
    ax_g.add_patch(obs_circle)
    ax_g.text(4.75, 7, 'O', fontsize=14, ha='center', va='center',
             color='white', fontweight='bold')

    # Dark matter
    dark_rect = Rectangle((5.5, 6), 3.5, 2, facecolor=colors['neutral'],
                         edgecolor='black', linewidth=2, alpha=0.7)
    ax_g.add_patch(dark_rect)
    ax_g.text(7.25, 7, 'Dark\nMatter', fontsize=11, ha='center',
             fontweight='bold', color='white')
    ax_g.text(7.25, 6.3, r'$x$', fontsize=10, ha='center',
             style='italic', color='white')

    # Ratio
    ax_g.text(5, 4.5, 'Observed Ratio ≈ 5.4:1', fontsize=13, ha='center',
             fontweight='bold', color=colors['danger'])

    # Explanation
    explanations = [
        'Observable matter: Terminated oscillations (where 0 ≠ 1)',
        'Dark matter: Non-terminated oscillations (where 0 = 1)',
        'Ratio emerges from counting terminated vs non-terminated paths',
    ]

    y = 3.5
    for i, expl in enumerate(explanations):
        ax_g.text(5, y - i*0.6, f'• {expl}', fontsize=9, ha='center',
                 style='italic')

    # Connection box
    conn_box = FancyBboxPatch((1.5, 0.5), 7, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['accent'],
                             edgecolor='black', linewidth=2, alpha=0.3)
    ax_g.add_patch(conn_box)
    ax_g.text(5, 1, 'The dark matter ratio is not a property of matter,',
             fontsize=10, ha='center', fontweight='bold')
    ax_g.text(5, 0.6, 'but of observation itself',
             fontsize=10, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL H: WHY x CANNOT BE A NUMBER
    # ============================================================================
    ax_h = fig.add_subplot(gs[4, 0])
    ax_h.set_xlim(0, 10)
    ax_h.set_ylim(0, 10)
    ax_h.axis('off')

    ax_h.text(5, 9.5, r'H. Why $x$ Cannot Be A Number',
             fontsize=14, fontweight='bold', ha='center')

    # Proof steps
    steps = [
        ('1', 'Assume x is a number', 'black', 'normal'),
        ('2', r'Then $x / N_{\mathrm{max}} \rightarrow 0$', 'black', 'normal'),
        ('3', r'Therefore $x \approx 0$ (negligible)', 'black', 'normal'),
        ('4', 'But x is inaccessible (not negligible)', colors['danger'], 'bold'),
        ('5', 'CONTRADICTION ✗', colors['danger'], 'bold'),
        ('6', 'Therefore x is NOT a number', colors['accent'], 'bold'),
        ('7', r'$x = \odot$ (observation boundary)', colors['accent'], 'bold'),
    ]

    y = 8
    for i, (num, text, color, weight) in enumerate(steps):
        # Step number
        num_circle = Circle((1.5, y - i*0.9), 0.25,
                           facecolor=color if i >= 3 else colors['neutral'],
                           edgecolor='black', linewidth=1.5, alpha=0.7)
        ax_h.add_patch(num_circle)
        ax_h.text(1.5, y - i*0.9, num, fontsize=10, ha='center', va='center',
                 color='white', fontweight='bold')

        # Step text
        ax_h.text(2, y - i*0.9, text, fontsize=10, ha='left', va='center',
                 color=color, fontweight=weight)

    # Conclusion box
    concl_box = FancyBboxPatch((1, 1), 8, 1.2, boxstyle="round,pad=0.1",
                              facecolor=colors['highlight'],
                              edgecolor=colors['accent'], linewidth=3)
    ax_h.add_patch(concl_box)
    ax_h.text(5, 1.8, r'$x = \odot$ is the point where $0 = 1$',
             fontsize=12, ha='center', fontweight='bold')
    ax_h.text(5, 1.3, 'The observation boundary where distinction collapses',
             fontsize=10, ha='center', style='italic')

    # ============================================================================
    # PANEL I: THE ACCEPTANCE BOUNDARY
    # ============================================================================
    ax_i = fig.add_subplot(gs[4, 1])
    ax_i.set_xlim(0, 10)
    ax_i.set_ylim(0, 10)
    ax_i.axis('off')

    ax_i.text(5, 9.5, 'I. The Acceptance Boundary',
             fontsize=14, fontweight='bold', ha='center')

    # Universe (no boundary)
    universe = Circle((5, 6), 3, facecolor=colors['background'],
                     edgecolor=colors['neutral'], linewidth=2,
                     alpha=0.3, linestyle='--')
    ax_i.add_patch(universe)
    ax_i.text(5, 9.2, 'Universe', fontsize=11, ha='center',
             fontweight='bold', style='italic')
    ax_i.text(5, 8.7, '(No distinctions)', fontsize=9, ha='center',
             style='italic')

    # Observer 1
    obs1 = Circle((3.5, 6), 1.2, facecolor=colors['primary'],
                 edgecolor='black', linewidth=2, alpha=0.5)
    ax_i.add_patch(obs1)
    ax_i.text(3.5, 6, r'$O_1$', fontsize=14, ha='center', va='center',
             fontweight='bold', color='white')
    ax_i.text(3.5, 4.5, r'$x_1$', fontsize=11, ha='center',
             color=colors['primary'], fontweight='bold')

    # Observer 2
    obs2 = Circle((6.5, 6), 1.5, facecolor=colors['accent'],
                 edgecolor='black', linewidth=2, alpha=0.5)
    ax_i.add_patch(obs2)
    ax_i.text(6.5, 6, r'$O_2$', fontsize=14, ha='center', va='center',
             fontweight='bold', color='white')
    ax_i.text(6.5, 4.2, r'$x_2$', fontsize=11, ha='center',
             color=colors['accent'], fontweight='bold')

    # Explanation
    expl_box = FancyBboxPatch((1, 2.5), 8, 1.5, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor=colors['primary'],
                             linewidth=2)
    ax_i.add_patch(expl_box)

    ax_i.text(5, 3.6, 'Different observers have different acceptance boundaries',
             fontsize=10, ha='center', fontweight='bold')
    ax_i.text(5, 3.2, r'$x$ = where observer stops categorizing',
             fontsize=10, ha='center', style='italic')
    ax_i.text(5, 2.8, 'Universe has no x (makes no distinctions)',
             fontsize=10, ha='center', style='italic')

    # Key insight
    insight_box = FancyBboxPatch((1.5, 0.5), 7, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['accent'],
                                edgecolor='black', linewidth=2, alpha=0.3)
    ax_i.add_patch(insight_box)
    ax_i.text(5, 1.6, r'$x$ is not a property of the universe,',
             fontsize=11, ha='center', fontweight='bold')
    ax_i.text(5, 1.2, 'but of the observer',
             fontsize=11, ha='center', fontweight='bold')
    ax_i.text(5, 0.7, '(Different goals → Different categories → Different x)',
             fontsize=9, ha='center', style='italic')

    # ============================================================================
    # PANEL J: THE COMPLETE PICTURE
    # ============================================================================
    ax_j = fig.add_subplot(gs[5, :])
    ax_j.set_xlim(0, 10)
    ax_j.set_ylim(0, 10)
    ax_j.axis('off')

    ax_j.text(5, 9.5, 'J. The Complete Picture',
             fontsize=16, fontweight='bold', ha='center')

    # Flow diagram
    boxes = [
        ('Universe', 1, 7, colors['background']),
        ('Oscillations', 2.5, 7, colors['primary']),
        ('Termination', 4, 7, colors['secondary']),
        ('Categories', 5.5, 7, colors['accent']),
        ('Observation', 7, 7, colors['warning']),
        (r'$N_{\mathrm{max}}$', 8.5, 7, colors['danger']),
    ]

    for i, (label, x, y, color) in enumerate(boxes):
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                            boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black',
                            linewidth=2, alpha=0.6)
        ax_j.add_patch(box)
        ax_j.text(x, y, label, fontsize=10, ha='center', va='center',
                 fontweight='bold', color='white')

        # Arrow to next
        if i < len(boxes) - 1:
            arrow = FancyArrowPatch((x+0.6, y), (boxes[i+1][1]-0.6, y),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color='black')
            ax_j.add_patch(arrow)

    # Key equations
    eq_y = 5.5
    equations = [
        (r'$C(t+1) = n^{C(t)}$', 'Recursive formula'),
        (r'$N_{\mathrm{max}} = (10^{84}) \uparrow\uparrow (10^{80})$', 'Largest finite number'),
        (r'Observable $= \infty - x$', 'Observer perspective'),
        (r'$x = \odot$ where $0 = 1$', 'True zero'),
        (r'$x/(\infty - x) \approx 5.4$', 'Dark matter ratio'),
    ]

    for i, (eq, desc) in enumerate(equations):
        y = eq_y - i*0.8

        # Equation
        eq_box = FancyBboxPatch((1, y-0.25), 3.5, 0.5,
                               boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor=colors['primary'],
                               linewidth=1.5)
        ax_j.add_patch(eq_box)
        ax_j.text(2.75, y, eq, fontsize=10, ha='center', va='center')

        # Description
        ax_j.text(5, y, desc, fontsize=9, ha='left', va='center',
                 style='italic')

    # Summary box
    summary_box = FancyBboxPatch((0.5, 0.2), 9, 1.3,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['highlight'],
                                edgecolor='black', linewidth=3)
    ax_j.add_patch(summary_box)

    ax_j.text(5, 1.2, r'$N_{\mathrm{max}}$ is the largest finite number that can be counted in physical reality',
             fontsize=11, ha='center', fontweight='bold')
    ax_j.text(5, 0.8, r'It is so large that all other numbers become zero, and even $0 = 1$ at the observation boundary',
             fontsize=10, ha='center', style='italic')
    ax_j.text(5, 0.4, 'This explains the dark matter ratio as a property of observation, not matter',
             fontsize=10, ha='center', style='italic')

    # Main title
    fig.suptitle(r'The Complete Framework: From Oscillations to $N_{\mathrm{max}}$',
                fontsize=18, fontweight='bold', y=0.985)

    # Save in multiple formats
    for fmt in ['png', 'pdf', 'svg']:
        filename = f'master_overview.{fmt}'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {filename}")

    plt.show()

if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print("MASTER OVERVIEW COMPLETE")
    print("="*70)
    print("\nGenerated publication-quality figure with 10 panels:")
    print("  A. Fundamental Recursion")
    print("  B. Growth Rate Comparison")
    print("  C. Oscillatory Foundation")
    print("  D. Entropy as Path Selection")
    print("  E. The ∞ - x Structure")
    print("  F. The True Zero")
    print("  G. Physical Interpretation")
    print("  H. Why x Cannot Be A Number")
    print("  I. The Acceptance Boundary")
    print("  J. The Complete Picture")
    print("\nSaved in PNG, PDF, and SVG formats")
    print("="*70)
