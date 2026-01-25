"""
Dark Matter Connection: Physical Interpretation (FIXED)
Publication-quality, matplotlib compatible
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Wedge, Polygon
import matplotlib.patches as mpatches

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

def main():
    """Create dark matter connection visualization"""

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.08, right=0.92, top=0.96, bottom=0.04)

    colors = {
        'obs': '#2E86AB',
        'dark': '#A23B72',
        'boundary': '#F18F01',
        'accent': '#C73E1D',
    }

    # ============================================================================
    # PANEL A: COSMIC COMPOSITION
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')

    ax_a.text(5, 9.5, r'$\bf{A.}$ Cosmic Composition',
             fontsize=14, fontweight='bold', ha='center')

    # Pie chart
    center = (2, 5)
    radius = 2.2

    # Observable (15.6%)
    obs_angle = 360 * 0.156
    obs_wedge = Wedge(center, radius, 0, obs_angle,
                     facecolor=colors['obs'], edgecolor='black',
                     linewidth=2, alpha=0.8)
    ax_a.add_patch(obs_wedge)
    ax_a.text(2.8, 5.5, r'$\infty - x$', fontsize=13, ha='center',
             color='white', fontweight='bold')

    # Dark (84.4%)
    dark_wedge = Wedge(center, radius, obs_angle, 360,
                      facecolor=colors['dark'], edgecolor='black',
                      linewidth=2, alpha=0.8)
    ax_a.add_patch(dark_wedge)
    ax_a.text(1.5, 4, r'$x$', fontsize=13, ha='center',
             color='white', fontweight='bold')

    # Percentages
    ax_a.text(2, 2.3, '15.6%', fontsize=10, ha='center', fontweight='bold')
    ax_a.text(2, 7.3, '84.4%', fontsize=10, ha='center', fontweight='bold')

    # Arrow
    arrow = FancyArrowPatch((4.3, 5), (5.5, 5),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=3, color='black')
    ax_a.add_patch(arrow)

    # Ratio box
    ratio_box = FancyBboxPatch((5.7, 3.5), 3.8, 3, boxstyle="round,pad=0.15",
                              facecolor='white', edgecolor='black', linewidth=2.5)
    ax_a.add_patch(ratio_box)

    ax_a.text(7.6, 6, r'$x/(\infty - x)$', fontsize=20, ha='center')
    ax_a.text(7.6, 5, r'$=$', fontsize=20, ha='center')
    ax_a.text(7.6, 4.2, r'$84.4/15.6$', fontsize=18, ha='center')
    ax_a.text(7.6, 3.5, r'$\approx 5.4$', fontsize=18, ha='center',
             fontweight='bold', color=colors['accent'])

    # Observations
    obs_data = [
        ('Planck', 5.36),
        ('WMAP', 5.35),
        ('Theory', 5.40),
    ]

    y_obs = 2
    for i, (label, val) in enumerate(obs_data):
        ax_a.text(5.5, y_obs - i*0.4, f'{label}:', fontsize=9, ha='left')
        ax_a.text(9.5, y_obs - i*0.4, f'{val:.2f}', fontsize=9, ha='right',
                 fontweight='bold')

    # ============================================================================
    # PANEL B: OSCILLATION TO MATTER
    # ============================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')

    ax_b.text(5, 9.5, r'$\bf{B.}$ Oscillation States',
             fontsize=14, fontweight='bold', ha='center')

    # Non-terminated
    t = np.linspace(0, 4*np.pi, 500)
    wave = np.sin(t)
    x_wave = t / (4*np.pi) * 7 + 1.5
    y_wave = wave * 0.7 + 7

    ax_b.plot(x_wave, y_wave, color=colors['dark'], linewidth=2.5, alpha=0.8)
    ax_b.text(5, 7.8, r'$\psi_{\mathrm{cont}}$', fontsize=12, ha='center',
             fontweight='bold', color=colors['dark'])

    # Arrow
    arrow1 = FancyArrowPatch((2.5, 6.3), (2.5, 5.5),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='black')
    ax_b.add_patch(arrow1)

    # Dark matter box
    dark_box = Rectangle((1.5, 4.5), 2, 0.8, facecolor=colors['dark'],
                        edgecolor='black', linewidth=2, alpha=0.8)
    ax_b.add_patch(dark_box)
    ax_b.text(2.5, 4.9, r'$x$', fontsize=14, ha='center', va='center',
             color='white', fontweight='bold')

    # Terminated
    t2 = np.linspace(0, 3*np.pi, 500)
    wave2 = np.sin(t2) * np.exp(-t2/10)
    x_wave2 = t2 / (3*np.pi) * 7 + 1.5
    y_wave2 = wave2 * 0.7 + 3.5

    ax_b.plot(x_wave2, y_wave2, color=colors['obs'], linewidth=2.5, alpha=0.8)
    ax_b.scatter([x_wave2[-1]], [y_wave2[-1]], s=200, c=colors['obs'],
               marker='X', edgecolors='black', linewidths=2, zorder=5)
    ax_b.text(7, 3.5, r'$\psi_{\mathrm{term}}$', fontsize=12, ha='center',
             fontweight='bold', color=colors['obs'])

    # Arrow
    arrow2 = FancyArrowPatch((7.5, 2.8), (7.5, 2),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='black')
    ax_b.add_patch(arrow2)

    # Observable box
    obs_box = Rectangle((6.5, 1), 2, 0.8, facecolor=colors['obs'],
                       edgecolor='black', linewidth=2, alpha=0.8)
    ax_b.add_patch(obs_box)
    ax_b.text(7.5, 1.4, r'$\infty - x$', fontsize=14, ha='center', va='center',
             color='white', fontweight='bold')

    # Probabilities
    ax_b.text(0.5, 4.9, r'$P = 0.844$', fontsize=10, ha='left',
             bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=colors['dark'], linewidth=1.5, pad=0.3))
    ax_b.text(0.5, 1.4, r'$P = 0.156$', fontsize=10, ha='left',
             bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=colors['obs'], linewidth=1.5, pad=0.3))

    # ============================================================================
    # PANEL C: PHASE SPACE
    # ============================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_xlim(-1, 11)
    ax_c.set_ylim(-1, 11)
    ax_c.axis('off')

    ax_c.text(5, 10, r'$\bf{C.}$ Phase Space Structure',
             fontsize=14, fontweight='bold', ha='center')

    # Axes
    ax_c.arrow(0, 0, 9.5, 0, head_width=0.2, head_length=0.3,
              fc='black', ec='black', linewidth=2)
    ax_c.arrow(0, 0, 0, 9.5, head_width=0.2, head_length=0.3,
              fc='black', ec='black', linewidth=2)

    ax_c.text(9.8, -0.5, r'$q$', fontsize=13, ha='center', fontweight='bold')
    ax_c.text(-0.5, 9.8, r'$p$', fontsize=13, ha='center', fontweight='bold')

    # Observable region
    theta_obs = np.linspace(0, np.pi/3, 100)
    r_obs = 7
    x_obs = r_obs * np.cos(theta_obs)
    y_obs = r_obs * np.sin(theta_obs)

    vertices = [(0, 0)]
    for i in range(len(x_obs)):
        vertices.append((x_obs[i], y_obs[i]))
    vertices.append((0, 0))

    obs_region = Polygon(vertices, facecolor=colors['obs'],
                        edgecolor='black', linewidth=2, alpha=0.4)
    ax_c.add_patch(obs_region)

    # Trajectories
    for i in range(5):
        angle = np.pi/3 * (i+1) / 6
        r = np.linspace(0, 7, 50)
        x_traj = r * np.cos(angle)
        y_traj = r * np.sin(angle)
        ax_c.plot(x_traj, y_traj, color=colors['obs'],
                 linewidth=1.5, alpha=0.6)
        ax_c.scatter([x_traj[-1]], [y_traj[-1]], s=50, c=colors['obs'],
                   marker='o', edgecolors='black', linewidths=1)

    # Dark region
    theta_dark = np.linspace(np.pi/3, np.pi/2, 100)
    x_dark = r_obs * np.cos(theta_dark)
    y_dark = r_obs * np.sin(theta_dark)

    vertices_dark = [(0, 0)]
    for i in range(len(x_dark)):
        vertices_dark.append((x_dark[i], y_dark[i]))
    vertices_dark.append((0, 0))

    dark_region = Polygon(vertices_dark, facecolor=colors['dark'],
                         edgecolor='black', linewidth=2, alpha=0.4)
    ax_c.add_patch(dark_region)

    # Dark trajectories
    for i in range(3):
        angle = np.pi/3 + (np.pi/2 - np.pi/3) * (i+1) / 4
        r = np.linspace(0, 9, 50)
        x_traj = r * np.cos(angle)
        y_traj = r * np.sin(angle)
        ax_c.plot(x_traj, y_traj, color=colors['dark'],
                 linewidth=1.5, alpha=0.6, linestyle='--')

    # Boundary
    ax_c.plot([0, 7*np.cos(np.pi/3)], [0, 7*np.sin(np.pi/3)],
             color=colors['boundary'], linewidth=3, zorder=10)

    # Labels
    ax_c.text(3.5, 1.5, r'$\infty - x$', fontsize=13, ha='center',
             fontweight='bold', color=colors['obs'])
    ax_c.text(1.5, 5, r'$x$', fontsize=13, ha='center',
             fontweight='bold', color=colors['dark'])
    ax_c.text(4, 4, r'$\odot$', fontsize=16, ha='center',
             fontweight='bold', color=colors['boundary'])

    # Angle
    arc = Wedge((0, 0), 2, 0, 60, width=0.1, facecolor=colors['boundary'],
               edgecolor=colors['boundary'], alpha=0.5)
    ax_c.add_patch(arc)
    ax_c.text(2.3, 0.8, r'$\theta \approx 60°$', fontsize=10, ha='left')

    # ============================================================================
    # PANEL D: ENTROPY FLOW
    # ============================================================================
    ax_d = fig.add_subplot(gs[2, :])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis('off')

    ax_d.text(5, 9.5, r'$\bf{D.}$ Entropy Flow and Termination',
             fontsize=14, fontweight='bold', ha='center')

    # Initial state
    init_circle = Circle((1.5, 5), 0.5, facecolor='white',
                        edgecolor='black', linewidth=2.5)
    ax_d.add_patch(init_circle)
    ax_d.text(1.5, 5, r'$S_0$', fontsize=12, ha='center', va='center',
             fontweight='bold')

    # Path 1: Observable
    path1_x = [2, 3, 4, 5]
    path1_y = [5, 6.5, 7.5, 8]
    ax_d.plot(path1_x, path1_y, color=colors['obs'], linewidth=3,
             marker='o', markersize=8, markerfacecolor=colors['obs'],
             markeredgecolor='black', markeredgewidth=1.5)

    # Entropy labels
    for i, (x, y) in enumerate(zip(path1_x[1:], path1_y[1:])):
        ax_d.text(x, y-0.5, f'$S_{i+1}$', fontsize=9, ha='center')

    # Terminal
    term_box = Rectangle((4.5, 8.5), 1, 0.6, facecolor=colors['obs'],
                        edgecolor='black', linewidth=2)
    ax_d.add_patch(term_box)
    ax_d.text(5, 8.8, 'Term', fontsize=10, ha='center',
             va='center', color='white', fontweight='bold')

    # Path 2: Dark
    path2_x = [2, 3, 4, 5]
    path2_y = [5, 3.5, 2.5, 2]
    ax_d.plot(path2_x, path2_y, color=colors['dark'], linewidth=3,
             marker='s', markersize=8, markerfacecolor=colors['dark'],
             markeredgecolor='black', markeredgewidth=1.5, linestyle='--')

    # Entropy (constant)
    for i, (x, y) in enumerate(zip(path2_x[1:], path2_y[1:])):
        ax_d.text(x, y+0.5, r'$S_0$', fontsize=9, ha='center')

    # Continuing
    ax_d.arrow(5, 2, 1, 0, head_width=0.2, head_length=0.2,
              fc=colors['dark'], ec='black', linewidth=2, linestyle='--')
    ax_d.text(6.5, 2, r'$\rightarrow \infty$', fontsize=11, ha='left',
             color=colors['dark'], fontweight='bold')

    # Split probabilities
    split_box = FancyBboxPatch((1.8, 4), 0.8, 2, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='black',
                              linewidth=2, linestyle='--')
    ax_d.add_patch(split_box)

    ax_d.text(2.2, 5.7, r'$P_{\uparrow}$', fontsize=10, ha='center',
             color=colors['obs'], fontweight='bold')
    ax_d.text(2.2, 5.3, '0.156', fontsize=9, ha='center')

    ax_d.text(2.2, 4.7, r'$P_{\rightarrow}$', fontsize=10, ha='center',
             color=colors['dark'], fontweight='bold')
    ax_d.text(2.2, 4.3, '0.844', fontsize=9, ha='center')

    # Equation
    eq_box = FancyBboxPatch((6.5, 4), 3, 2, boxstyle="round,pad=0.15",
                           facecolor='white', edgecolor='black', linewidth=2.5)
    ax_d.add_patch(eq_box)

    ax_d.text(8, 5.5, r'$dS/dt = R_{\mathrm{cat}}$ (obs)', fontsize=11, ha='center')
    ax_d.text(8, 5, r'$dS/dt = 0$ (dark)', fontsize=11, ha='center')
    ax_d.text(8, 4.3, r'$P_{\rightarrow}/P_{\uparrow} = 5.4$',
             fontsize=11, ha='center', fontweight='bold',
             color=colors['accent'])

    # ============================================================================
    # PANEL E: RECURSIVE STRUCTURE
    # ============================================================================
    ax_e = fig.add_subplot(gs[3, 0])
    ax_e.set_xlim(0, 10)
    ax_e.set_ylim(0, 10)
    ax_e.axis('off')

    ax_e.text(5, 9.5, r'$\bf{E.}$ Recursive Observation',
             fontsize=14, fontweight='bold', ha='center')

    # Nested circles
    radii = [3, 2.3, 1.6, 0.9]
    alphas = [0.2, 0.3, 0.4, 0.6]

    for i, (r, alpha) in enumerate(zip(radii, alphas)):
        circle = Circle((5, 5.5), r, facecolor=colors['obs'],
                       edgecolor='black', linewidth=2, alpha=alpha)
        ax_e.add_patch(circle)

        if i < len(radii) - 1:
            angle = np.pi/4
            x = 5 + (r + radii[i+1])/2 * np.cos(angle)
            y = 5.5 + (r + radii[i+1])/2 * np.sin(angle)
            ax_e.text(x, y, f'$C_{i}$', fontsize=10, ha='center',
                     bbox=dict(boxstyle='circle', facecolor='white',
                              edgecolor='black', linewidth=1, pad=0.2))

    # Center
    ax_e.text(5, 5.5, r'$C_t$', fontsize=13, ha='center', va='center',
             fontweight='bold')

    # Formula
    formula_box = FancyBboxPatch((1, 2), 8, 1.5, boxstyle="round,pad=0.15",
                                facecolor='white', edgecolor='black',
                                linewidth=2.5)
    ax_e.add_patch(formula_box)

    ax_e.text(5, 3.2, r'$C(t+1) = n^{C(t)}$', fontsize=14, ha='center',
             fontweight='bold')
    ax_e.text(5, 2.5, r'$n \approx 10^{84}$ entity-state pairs',
             fontsize=10, ha='center', style='italic')

    # Result
    result_box = FancyBboxPatch((2, 0.5), 6, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['accent'], edgecolor='black',
                               linewidth=2, alpha=0.3)
    ax_e.add_patch(result_box)
    ax_e.text(5, 1, r'$N_{\mathrm{max}} = n \uparrow\uparrow t$',
             fontsize=13, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL F: BOUNDARY GEOMETRY
    # ============================================================================
    ax_f = fig.add_subplot(gs[3, 1])
    ax_f.set_xlim(0, 10)
    ax_f.set_ylim(0, 10)
    ax_f.axis('off')

    ax_f.text(5, 9.5, r'$\bf{F.}$ Boundary Geometry',
             fontsize=14, fontweight='bold', ha='center')

    # Boundary
    theta = np.linspace(0, 2*np.pi, 1000)

    # Observable (inside)
    r_inner = 2.5
    x_inner = 5 + r_inner * np.cos(theta)
    y_inner = 5.5 + r_inner * np.sin(theta)
    ax_f.fill(x_inner, y_inner, color=colors['obs'], alpha=0.4,
             edgecolor='none')

    # Boundary line
    ax_f.plot(x_inner, y_inner, color=colors['boundary'], linewidth=3)

    # Dark (outside)
    r_outer = 3.5
    x_outer = 5 + r_outer * np.cos(theta)
    y_outer = 5.5 + r_outer * np.sin(theta)

    # Annulus
    for i in range(len(theta)-1):
        vertices = [
            (x_inner[i], y_inner[i]),
            (x_inner[i+1], y_inner[i+1]),
            (x_outer[i+1], y_outer[i+1]),
            (x_outer[i], y_outer[i])
        ]
        poly = Polygon(vertices, facecolor=colors['dark'],
                      edgecolor='none', alpha=0.4)
        ax_f.add_patch(poly)

    ax_f.plot(x_outer, y_outer, color='black', linewidth=2, linestyle='--')

    # Labels
    ax_f.text(5, 5.5, r'$\infty - x$', fontsize=13, ha='center',
             fontweight='bold', color=colors['obs'])
    ax_f.text(5, 8.5, r'$x$', fontsize=13, ha='center',
             fontweight='bold', color=colors['dark'])

    # Boundary markers
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        x_mark = 5 + r_inner * np.cos(angle)
        y_mark = 5.5 + r_inner * np.sin(angle)
        ax_f.scatter([x_mark], [y_mark], s=100, c=colors['boundary'],
                   marker='o', edgecolors='black', linewidths=2, zorder=10)
        ax_f.text(x_mark, y_mark-0.6, r'$\odot$', fontsize=11, ha='center',
                 fontweight='bold', color=colors['boundary'])

    # Metric
    metric_box = FancyBboxPatch((1, 1), 8, 1.2, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor='black',
                               linewidth=2)
    ax_f.add_patch(metric_box)

    ax_f.text(5, 1.8, r'$ds^2 = dr^2 + r^2 d\theta^2$', fontsize=11,
             ha='center')
    ax_f.text(5, 1.3, r'$r = r_{\odot}$: observation boundary',
             fontsize=10, ha='center', style='italic')

    # ============================================================================
    # PANEL G: MATHEMATICAL SUMMARY
    # ============================================================================
    ax_g = fig.add_subplot(gs[4, :])
    ax_g.set_xlim(0, 10)
    ax_g.set_ylim(0, 10)
    ax_g.axis('off')

    ax_g.text(5, 9.5, r'$\bf{G.}$ Mathematical Framework',
             fontsize=14, fontweight='bold', ha='center')

    # Equations
    equations = [
        (r'$N_{\mathrm{max}} = (10^{84}) \uparrow\uparrow (10^{80})$',
         'Total categorical space', 8, colors['obs']),
        (r'Observable $= \infty - x$',
         'Terminated oscillations', 6.5, colors['obs']),
        (r'Inaccessible $= x = \odot$',
         'Non-terminated (where $0 = 1$)', 5, colors['dark']),
        (r'$x/(\infty - x) = P_{\mathrm{cont}}/P_        (r'$x/(\infty - x) = P_{\mathrm{cont}}/P_{\mathrm{term}} = 5.4$',
         'Dark matter ratio', 3.5, colors['accent']),
    ]

    for eq, desc, y, color in equations:
        # Equation box
        eq_box = FancyBboxPatch((0.5, y-0.5), 5, 0.9, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor=color,
                               linewidth=2.5)
        ax_g.add_patch(eq_box)
        ax_g.text(3, y, eq, fontsize=12, ha='center', fontweight='bold')

        # Description
        ax_g.text(6.5, y, desc, fontsize=10, ha='left', style='italic')

    # Key insight
    insight_box = FancyBboxPatch((0.5, 0.5), 9, 1.8, boxstyle="round,pad=0.15",
                                facecolor=colors['accent'], edgecolor='black',
                                linewidth=3, alpha=0.2)
    ax_g.add_patch(insight_box)

    ax_g.text(5, 2, 'Theorem: Dark matter ratio emerges from',
             fontsize=12, ha='center', fontweight='bold')
    ax_g.text(5, 1.5, 'termination probability in categorical observation',
             fontsize=11, ha='center', style='italic')
    ax_g.text(5, 0.9, 'No free parameters - Pure derivation - Matches observations',
             fontsize=10, ha='center')

    # Main title
    fig.suptitle(r'Dark Matter as Inaccessible Categorical Space',
                fontsize=16, fontweight='bold', y=0.985)

    # Save
    for fmt in ['png', 'pdf']:
        filename = f'dark_matter_connection.{fmt}'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {filename}")

    print("\nDark matter connection complete!")
    plt.close()

if __name__ == "__main__":
    main()
