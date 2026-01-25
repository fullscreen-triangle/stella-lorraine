"""
The Acceptance Boundary: Observer-Dependent Structure (Scientific/Symbolic)
Minimal text, geometric visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Wedge, Ellipse, Polygon, Arc
import matplotlib.patches as mpatches
from matplotlib import patheffects

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

def main():
    """Create acceptance boundary visualization"""

    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.08, right=0.92, top=0.96, bottom=0.04)

    colors = {
        'obs1': '#2E86AB',
        'obs2': '#A23B72',
        'obs3': '#F18F01',
        'boundary': '#C73E1D',
        'universe': '#E0E0E0',
    }

    # ============================================================================
    # PANEL A: UNIVERSE WITHOUT OBSERVER
    # ============================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')

    ax_a.text(5, 9.5, r'$\mathbf{A.}$ Universe (No Observer)',
             fontsize=13, fontweight='bold', ha='center')

    # Uniform field (no distinctions)
    circle = Circle((5, 5), 3.5, facecolor=colors['universe'],
                   edgecolor='black', linewidth=2, alpha=0.5,
                   linestyle='--')
    ax_a.add_patch(circle)

    # Random dots (no structure)
    np.random.seed(42)
    for _ in range(100):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 3.5)
        x = 5 + r * np.cos(angle)
        y = 5 + r * np.sin(angle)
        ax_a.scatter([x], [y], s=2, c='gray', alpha=0.3)

    ax_a.text(5, 5, r'$\mathcal{U}$', fontsize=20, ha='center', va='center',
             fontweight='bold', color='gray')

    # Properties
    props_box = FancyBboxPatch((1, 1), 8, 1.2, boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor='black',
                              linewidth=1.5)
    ax_a.add_patch(props_box)

    ax_a.text(5, 1.9, r'$x = \emptyset$ (no boundary)', fontsize=11,
             ha='center', fontweight='bold')
    ax_a.text(5, 1.4, r'No categories, no distinctions', fontsize=10,
             ha='center', style='italic')

    # ============================================================================
    # PANEL B: SINGLE OBSERVER
    # ============================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')

    ax_b.text(5, 9.5, r'$\mathbf{B.}$ Universe + Observer',
             fontsize=13, fontweight='bold', ha='center')

    # Universe
    circle_u = Circle((5, 5), 3.5, facecolor=colors['universe'],
                     edgecolor='black', linewidth=2, alpha=0.3,
                     linestyle='--')
    ax_b.add_patch(circle_u)

    # Observable region
    circle_obs = Circle((5, 5), 2.2, facecolor=colors['obs1'],
                       edgecolor='black', linewidth=2.5, alpha=0.5)
    ax_b.add_patch(circle_obs)

    # Observer
    obs = Circle((5, 5), 0.4, facecolor='white',
                edgecolor='black', linewidth=2)
    ax_b.add_patch(obs)
    ax_b.text(5, 5, r'$O$', fontsize=13, ha='center', va='center',
             fontweight='bold')

    # Boundary
    boundary = Circle((5, 5), 2.2, facecolor='none',
                     edgecolor=colors['boundary'], linewidth=3)
    ax_b.add_patch(boundary)

    # Labels
    ax_b.text(5, 6.5, r'$\infty - x$', fontsize=12, ha='center',
             fontweight='bold', color='white')
    ax_b.text(5, 8.2, r'$x$', fontsize=12, ha='center',
             fontweight='bold', color='gray')

    # Boundary marker
    ax_b.text(7.3, 5, r'$\odot$', fontsize=16, ha='center',
             fontweight='bold', color=colors['boundary'])

    # Properties
    props_box2 = FancyBboxPatch((1, 1), 8, 1.2, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor='black',
                               linewidth=1.5)
    ax_b.add_patch(props_box2)

    ax_b.text(5, 1.9, r'$x = \odot$ (acceptance boundary)', fontsize=11,
             ha='center', fontweight='bold')
    ax_b.text(5, 1.4, r'Categories emerge, $\infty - x$ accessible',
             fontsize=10, ha='center', style='italic')

    # ============================================================================
    # PANEL C: MULTIPLE OBSERVERS
    # ============================================================================
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')

    ax_c.text(5, 9.5, r'$\mathbf{C.}$ Multiple Observers, Different Boundaries',
             fontsize=14, fontweight='bold', ha='center')

    # Universe
    circle_u = Circle((5, 5), 3.8, facecolor=colors['universe'],
                     edgecolor='black', linewidth=2.5, alpha=0.3,
                     linestyle='--')
    ax_c.add_patch(circle_u)
    ax_c.text(5, 9, r'$\mathcal{U}$', fontsize=13, ha='center',
             fontweight='bold', color='gray')

    # Observer 1 (small boundary)
    obs1_pos = (2.5, 5)
    obs1_r = 1.5
    obs1_circle = Circle(obs1_pos, obs1_r, facecolor=colors['obs1'],
                        edgecolor='black', linewidth=2, alpha=0.4)
    ax_c.add_patch(obs1_circle)
    obs1 = Circle(obs1_pos, 0.3, facecolor='white',
                 edgecolor='black', linewidth=2)
    ax_c.add_patch(obs1)
    ax_c.text(obs1_pos[0], obs1_pos[1], r'$O_1$', fontsize=11,
             ha='center', va='center', fontweight='bold')
    ax_c.text(obs1_pos[0], obs1_pos[1]-2, r'$x_1$ (small)', fontsize=10,
             ha='center', fontweight='bold', color=colors['obs1'])

    # Observer 2 (medium boundary)
    obs2_pos = (5, 5)
    obs2_r = 2.5
    obs2_circle = Circle(obs2_pos, obs2_r, facecolor=colors['obs2'],
                        edgecolor='black', linewidth=2, alpha=0.3)
    ax_c.add_patch(obs2_circle)
    obs2 = Circle(obs2_pos, 0.3, facecolor='white',
                 edgecolor='black', linewidth=2)
    ax_c.add_patch(obs2)
    ax_c.text(obs2_pos[0], obs2_pos[1], r'$O_2$', fontsize=11,
             ha='center', va='center', fontweight='bold')
    ax_c.text(obs2_pos[0], obs2_pos[1]+3, r'$x_2$ (medium)', fontsize=10,
             ha='center', fontweight='bold', color=colors['obs2'])

    # Observer 3 (large boundary)
    obs3_pos = (7.5, 5)
    obs3_r = 3.3
    obs3_circle = Circle(obs3_pos, obs3_r, facecolor=colors['obs3'],
                        edgecolor='black', linewidth=2, alpha=0.25)
    ax_c.add_patch(obs3_circle)
    obs3 = Circle(obs3_pos, 0.3, facecolor='white',
                 edgecolor='black', linewidth=2)
    ax_c.add_patch(obs3)
    ax_c.text(obs3_pos[0], obs3_pos[1], r'$O_3$', fontsize=11,
             ha='center', va='center', fontweight='bold')
    ax_c.text(obs3_pos[0], obs3_pos[1]-2, r'$x_3$ (large)', fontsize=10,
             ha='center', fontweight='bold', color=colors['obs3'])

    # Key insight
    insight_box = FancyBboxPatch((1, 0.5), 8, 0.8, boxstyle="round,pad=0.1",
                                facecolor='white', edgecolor='black',
                                linewidth=2)
    ax_c.add_patch(insight_box)
    ax_c.text(5, 0.9, r'$x_i = f(\mathrm{goals}_i, \mathrm{capacity}_i, \mathrm{categories}_i)$',
             fontsize=11, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL D: BOUNDARY DYNAMICS
    # ============================================================================
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis('off')

    ax_d.text(5, 9.5, r'$\mathbf{D.}$ Boundary Evolution',
             fontsize=13, fontweight='bold', ha='center')

    # Time series of expanding boundary
    center = (5, 5)
    radii = [1, 1.8, 2.5, 3.2]
    alphas = [0.7, 0.5, 0.3, 0.15]
    times = [r'$t_0$', r'$t_1$', r'$t_2$', r'$t_3$']

    for i, (r, alpha, t) in enumerate(zip(radii, alphas, times)):
        circle = Circle(center, r, facecolor=colors['obs1'],
                       edgecolor='black', linewidth=1.5, alpha=alpha)
        ax_d.add_patch(circle)

        # Time label
        angle = np.pi/4 + i*np.pi/8
        x_label = center[0] + r * np.cos(angle)
        y_label = center[1] + r * np.sin(angle)
        ax_d.text(x_label, y_label, t, fontsize=10, ha='center',
                 bbox=dict(boxstyle='circle', facecolor='white',
                          edgecolor='black', linewidth=1, pad=0.15))

    # Observer
    obs = Circle(center, 0.35, facecolor='white',
                edgecolor='black', linewidth=2)
    ax_d.add_patch(obs)
    ax_d.text(center[0], center[1], r'$O$', fontsize=12,
             ha='center', va='center', fontweight='bold')

    # Arrow showing growth
    arrow = FancyArrowPatch((5, 1.5), (5, 0.8),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='black')
    ax_d.add_patch(arrow)
    ax_d.text(5, 0.5, r'$\frac{dx}{dt} > 0$', fontsize=11, ha='center',
             fontweight='bold')

    # ============================================================================
    # PANEL E: CATEGORY DEPTH
    # ============================================================================
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.set_xlim(0, 10)
    ax_e.set_ylim(0, 10)
    ax_e.axis('off')

    ax_e.text(5, 9.5, r'$\mathbf{E.}$ Categorical Depth',
             fontsize=13, fontweight='bold', ha='center')

    # Nested categories (tree structure)
    # Level 0 (root)
    root = Circle((5, 7.5), 0.4, facecolor=colors['obs1'],
                 edgecolor='black', linewidth=2)
    ax_e.add_patch(root)
    ax_e.text(5, 7.5, r'$C_0$', fontsize=10, ha='center', va='center',
             color='white', fontweight='bold')

    # Level 1
    level1_x = [3, 5, 7]
    for i, x in enumerate(level1_x):
        circle = Circle((x, 5.5), 0.35, facecolor=colors['obs1'],
                       edgecolor='black', linewidth=1.5, alpha=0.7)
        ax_e.add_patch(circle)
        ax_e.text(x, 5.5, r'$C_1^{' + str(i+1) + r'}$', fontsize=9,
                 ha='center', va='center', color='white', fontweight='bold')
        # Connection
        ax_e.plot([5, x], [7.1, 5.85], 'k-', linewidth=1.5, alpha=0.5)

    # Level 2
    level2_x = [2, 3, 4, 5, 6, 7, 8]
    for i, x in enumerate(level2_x):
        circle = Circle((x, 3.5), 0.3, facecolor=colors['obs1'],
                       edgecolor='black', linewidth=1, alpha=0.5)
        ax_e.add_patch(circle)
        # Connection to parent
        parent_x = level1_x[i // 3] if i < 6 else level1_x[2]
        ax_e.plot([parent_x, x], [5.15, 3.8], 'k-', linewidth=1, alpha=0.3)

    # Level 3 (many small dots)
    for i in range(20):
        x = 1 + i * 0.4
        circle = Circle((x, 1.5), 0.15, facecolor=colors['obs1'],
                       edgecolor='black', linewidth=0.5, alpha=0.3)
        ax_e.add_patch(circle)

    # Boundary marker
    ax_e.plot([0.5, 9.5], [0.8, 0.8], color=colors['boundary'],
             linewidth=3, linestyle='--')
    ax_e.text(9.7, 0.8, r'$\odot$', fontsize=14, ha='left', va='center',
             fontweight='bold', color=colors['boundary'])

    # Label
    ax_e.text(5, 0.3, r'Depth $= \log_n(x)$', fontsize=10, ha='center',
             fontweight='bold')

    # ============================================================================
    # PANEL F: INFORMATION CAPACITY
    # ============================================================================
    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.set_xlim(0, 10)
    ax_f.set_ylim(0, 10)
    ax_f.axis('off')

    ax_f.text(5, 9.5, r'$\mathbf{F.}$ Information Capacity',
             fontsize=13, fontweight='bold', ha='center')

    # Create capacity diagram
    # Observer capacity
    capacity_box = Rectangle((1, 6), 3, 2, facecolor=colors['obs1'],
                            edgecolor='black', linewidth=2, alpha=0.6)
    ax_f.add_patch(capacity_box)
    ax_f.text(2.5, 7, r'$I_{\mathrm{obs}}$', fontsize=12, ha='center',
             va='center', color='white', fontweight='bold')

    # Arrow
    arrow1 = FancyArrowPatch((4.2, 7), (5.8, 7),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='black')
    ax_f.add_patch(arrow1)

    # Boundary size
    boundary_box = Rectangle((6, 6), 3, 2, facecolor=colors['boundary'],
                            edgecolor='black', linewidth=2, alpha=0.4)
    ax_f.add_patch(boundary_box)
    ax_f.text(7.5, 7, r'$x$', fontsize=12, ha='center', va='center',
             fontweight='bold')

    # Relationship
    rel_box = FancyBboxPatch((2, 4), 6, 1.2, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='black',
                            linewidth=2)
    ax_f.add_patch(rel_box)
    ax_f.text(5, 4.6, r'$x \propto I_{\mathrm{obs}}$', fontsize=12,
             ha='center', fontweight='bold')

    # Examples
    examples = [
        (r'Human: $I \sim 10^{16}$ bits', r'$x_H$', 2.5),
        (r'Computer: $I \sim 10^{20}$ bits', r'$x_C$', 1.5),
        (r'Civilization: $I \sim 10^{30}$ bits', r'$x_{Civ}$', 0.5),
    ]

    for label, x_label, y in examples:
        ax_f.text(1, y, label, fontsize=9, ha='left')
        ax_f.text(8, y, x_label, fontsize=10, ha='left', fontweight='bold',
                 color=colors['boundary'])

    # ============================================================================
    # PANEL G: GOAL-DEPENDENT BOUNDARIES
    # ============================================================================
    ax_g = fig.add_subplot(gs[3, 1])
    ax_g.set_xlim(0, 10)
    ax_g.set_ylim(0, 10)
    ax_g.axis('off')

    ax_g.text(5, 9.5, r'$\mathbf{G.}$ Goal-Dependent Structure',
             fontsize=13, fontweight='bold', ha='center')

    # Same universe, different goals
    universe = Circle((5, 5.5), 3.5, facecolor=colors['universe'],
                     edgecolor='black', linewidth=2, alpha=0.3,
                     linestyle='--')
    ax_g.add_patch(universe)

    # Goal 1: Fine-grained (many small categories)
    wedge1 = Wedge((5, 5.5), 3.5, 0, 120, facecolor=colors['obs1'],
                  edgecolor='black', linewidth=2, alpha=0.4)
    ax_g.add_patch(wedge1)

    # Draw fine divisions
    for angle in np.linspace(0, 120, 13):
        rad = angle * np.pi / 180
        x_end = 5 + 3.5 * np.cos(rad)
        y_end = 5.5 + 3.5 * np.sin(rad)
        ax_g.plot([5, x_end], [5.5, y_end], 'k-', linewidth=0.5, alpha=0.3)

    ax_g.text(6.5, 7, r'$G_1$: fine', fontsize=10, ha='center',
             fontweight='bold', color=colors['obs1'])
    ax_g.text(6.5, 6.5, r'$x_1$ large', fontsize=9, ha='center',
             style='italic')

    # Goal 2: Coarse-grained (few large categories)
    wedge2 = Wedge((5, 5.5), 3.5, 120, 240, facecolor=colors['obs2'],
                  edgecolor='black', linewidth=2, alpha=0.4)
    ax_g.add_patch(wedge2)

    # Draw coarse divisions
    for angle in np.linspace(120, 240, 4):
        rad = angle * np.pi / 180
        x_end = 5 + 3.5 * np.cos(rad)
        y_end = 5.5 + 3.5 * np.sin(rad)
        ax_g.plot([5, x_end], [5.5, y_end], 'k-', linewidth=1.5, alpha=0.5)

    ax_g.text(2, 5.5, r'$G_2$: coarse', fontsize=10, ha='center',
             fontweight='bold', color=colors['obs2'])
    ax_g.text(2, 5, r'$x_2$ small', fontsize=9, ha='center',
             style='italic')

    # Goal 3: Intermediate
    wedge3 = Wedge((5, 5.5), 3.5, 240, 360, facecolor=colors['obs3'],
                  edgecolor='black', linewidth=2, alpha=0.4)
    ax_g.add_patch(wedge3)

    for angle in np.linspace(240, 360, 7):
        rad = angle * np.pi / 180
        x_end = 5 + 3.5 * np.cos(rad)
        y_end = 5.5 + 3.5 * np.sin(rad)
        ax_g.plot([5, x_end], [5.5, y_end], 'k-', linewidth=1, alpha=0.4)

    ax_g.text(6.5, 4, r'$G_3$: medium', fontsize=10, ha='center',
             fontweight='bold', color=colors['obs3'])
    ax_g.text(6.5, 3.5, r'$x_3$ medium', fontsize=9, ha='center',
             style='italic')

    # Central observer
    obs = Circle((5, 5.5), 0.4, facecolor='white',
                edgecolor='black', linewidth=2)
    ax_g.add_patch(obs)
    ax_g.text(5, 5.5, r'$O$', fontsize=12, ha='center', va='center',
             fontweight='bold')

    # Equation
    eq_box = FancyBboxPatch((1.5, 1), 7, 1, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor='black',
                           linewidth=2)
    ax_g.add_patch(eq_box)
    ax_g.text(5, 1.5, r'$x = x(G, I, C)$ where $G$ = goals',
             fontsize=11, ha='center', fontweight='bold')

    # ============================================================================
    # PANEL H: BOUNDARY INTERACTIONS
    # ============================================================================
    ax_h = fig.add_subplot(gs[4, :])
    ax_h.set_xlim(0, 10)
    ax_h.set_ylim(0, 10)
    ax_h.axis('off')

    ax_h.text(5, 9.5, r'$\mathbf{H.}$ Boundary Interactions',
             fontsize=14, fontweight='bold', ha='center')

    # Two observers with overlapping boundaries
    obs1_pos = (3, 5)
    obs1_r = 2
    obs2_pos = (7, 5)
    obs2_r = 2.5

    # Observer 1 region
    circle1 = Circle(obs1_pos, obs1_r, facecolor=colors['obs1'],
                    edgecolor='black', linewidth=2, alpha=0.4)
    ax_h.add_patch(circle1)

    # Observer 2 region
    circle2 = Circle(obs2_pos, obs2_r, facecolor=colors['obs2'],
                    edgecolor='black', linewidth=2, alpha=0.4)
    ax_h.add_patch(circle2)

    # Overlap region (shared categories)
    # Calculate intersection
    d = obs2_pos[0] - obs1_pos[0]
    if d < obs1_r + obs2_r:
        # Create overlap visualization
        theta1 = np.arccos((d**2 + obs1_r**2 - obs2_r**2) / (2*d*obs1_r))
        theta2 = np.arccos((d**2 + obs2_r**2 - obs1_r**2) / (2*d*obs2_r))

        # Highlight overlap
        overlap_x = np.linspace(obs1_pos[0], obs2_pos[0], 100)
        for x in overlap_x:
            if x < obs1_pos[0] + obs1_r and x > obs2_pos[0] - obs2_r:
                y_range = min(
                    np.sqrt(obs1_r**2 - (x - obs1_pos[0])**2),
                    np.sqrt(obs2_r**2 - (x - obs2_pos[0])**2)
                )
                ax_h.plot([x, x], [5-y_range, 5+y_range],
                         color='purple', linewidth=2, alpha=0.6)

    # Observers
    obs1 = Circle(obs1_pos, 0.35, facecolor='white',
                 edgecolor='black', linewidth=2)
    ax_h.add_patch(obs1)
    ax_h.text(obs1_pos[0], obs1_pos[1], r'$O_1$', fontsize=11,
             ha='center', va='center', fontweight='bold')

    obs2 = Circle(obs2_pos, 0.35, facecolor='white',
                 edgecolor='black', linewidth=2)
    ax_h.add_patch(obs2)
    ax_h.text(obs2_pos[0], obs2_pos[1], r'$O_2$', fontsize=11,
             ha='center', va='center', fontweight='bold')

    # Labels
    ax_h.text(2, 7, r'$x_1$', fontsize=12, ha='center', fontweight='bold',
             color=colors['obs1'])
    ax_h.text(8, 7.5, r'$x_2$', fontsize=12, ha='center', fontweight='bold',
             color=colors['obs2'])
    ax_h.text(5, 6.5, r'$x_1 \cap x_2$', fontsize=11, ha='center',
             fontweight='bold', color='purple')

    # Regions
    regions = [
        (r'$x_1 \setminus x_2$', 1.5, 5, colors['obs1']),
        (r'$x_1 \cap x_2$', 5, 5, 'purple'),
        (r'$x_2 \setminus x_1$', 8.5, 5, colors['obs2']),
    ]

    for label, x, y, color in regions:
        ax_h.text(x, y, label, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor=color, linewidth=2, pad=0.3))

    # Communication
    comm_arrow = FancyArrowPatch(obs1_pos, obs2_pos,
                                arrowstyle='<->', mutation_scale=25,
                                linewidth=2.5, color='black',
                                linestyle='--')
    ax_h.add_patch(comm_arrow)
    ax_h.text(5, 3.5, 'Shared categories', fontsize=10, ha='center',
             style='italic')

    # Equations
    eq_box = FancyBboxPatch((1, 1.5), 8, 1.5, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor='black',
                           linewidth=2)
    ax_h.add_patch(eq_box)

    ax_h.text(5, 2.7, r'$x_{\mathrm{shared}} = x_1 \cap x_2$', fontsize=11,
             ha='center', fontweight='bold')
    ax_h.text(5, 2.2, r'$x_{\mathrm{total}} = x_1 \cup x_2$', fontsize=11,
             ha='center', fontweight='bold')
    ax_h.text(5, 1.7, r'Communication requires $x_{\mathrm{shared}} \neq \emptyset$',
             fontsize=10, ha='center', style='italic')

    # ============================================================================
    # PANEL I: MATHEMATICAL FORMALIZATION
    # ============================================================================
    ax_i = fig.add_subplot(gs[5, :])
    ax_i.set_xlim(0, 10)
    ax_i.set_ylim(0, 10)
    ax_i.axis('off')

    ax_i.text(5, 9.5, r'$\mathbf{I.}$ Mathematical Framework',
             fontsize=14, fontweight='bold', ha='center')

    # Definition boxes
    definitions = [
        (r'Universe: $\mathcal{U}$ (no inherent structure)', 8.2, colors['universe']),
        (r'Observer: $O$ with goals $G$, capacity $I$', 7, colors['obs1']),
        (r'Categories: $C = \{c_1, c_2, \ldots, c_n\}$', 5.8, colors['obs1']),
        (r'Boundary: $\odot = \partial(\infty - x)$', 4.6, colors['boundary']),
        (r'Observable: $\infty - x = \{c \in C : O \text{ can access } c\}$', 3.4, colors['obs1']),
        (r'Inaccessible: $x = \mathcal{U} \setminus (\infty - x)$', 2.2, colors['obs2']),
    ]

    for text, y, color in definitions:
        box = FancyBboxPatch((0.5, y-0.4), 9, 0.8, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor=color,
                            linewidth=2)
        ax_i.add_patch(box)
        ax_i.text(5, y, text, fontsize=11, ha='center')

    # Key theorem
    theorem_box = FancyBboxPatch((0.5, 0.3), 9, 1.3, boxstyle="round,pad=0.15",
                                facecolor=colors['boundary'], edgecolor='black',
                                linewidth=3, alpha=0.2)
    ax_i.add_patch(theorem_box)

    ax_i.text(5, 1.4, r'Theorem: $x$ is observer-dependent:',
             fontsize=12, ha='center', fontweight='bold')
    ax_i.text(5, 0.9, r'$x = x(O) = f(G, I, C)$',
             fontsize=12, ha='center', fontweight='bold')
    ax_i.text(5, 0.5, r'Different observers $\Rightarrow$ different boundaries $\Rightarrow$ different ratios',
             fontsize=10, ha='center', style='italic')

    # Main title
    fig.suptitle(r'The Acceptance Boundary: Observer-Dependent Structure',
                fontsize=16, fontweight='bold', y=0.985)

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        filename = f'acceptance_boundary.{fmt}'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {filename}")

    plt.show()

if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print("ACCEPTANCE BOUNDARY (SCIENTIFIC STYLE)")
    print("="*70)
