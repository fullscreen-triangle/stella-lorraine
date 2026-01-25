"""
figure_5_circuit_complementarity.py

Publication figure showing the ammeter/voltmeter analogy for dual-membrane complementarity.
This makes the abstract concept concrete and familiar.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Colors
COLORS = {
    'ammeter': '#0173B2',    # Blue (like front face)
    'voltmeter': '#DE8F05',  # Orange (like back face)
    'circuit': '#949494',    # Gray
    'error': '#D55E00',      # Red
    'success': '#029E73',    # Green
}

def create_figure_5():
    """
    Figure 5: Circuit Complementarity - Ammeter/Voltmeter Analogy

    Panel A: Ammeter configuration (measures current, calculates voltage)
    Panel B: Voltmeter configuration (measures voltage, calculates current)
    Panel C: Invalid configuration (both in series - FAILS)
    Panel D: Mapping to dual-membrane complementarity
    """

    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.35,
                          left=0.08, right=0.95, top=0.95, bottom=0.05,
                          height_ratios=[1, 1, 1, 1.2])

    # Panel A: Ammeter configuration
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a_ammeter(ax_a)

    # Panel B: Voltmeter configuration
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b_voltmeter(ax_b)

    # Panel C: Invalid configuration
    ax_c = fig.add_subplot(gs[1, :])
    plot_panel_c_invalid(ax_c)

    # Panel D: Mapping table
    ax_d = fig.add_subplot(gs[2, :])
    plot_panel_d_mapping(ax_d)

    # Panel E: Dual-membrane analogy
    ax_e = fig.add_subplot(gs[3, :])
    plot_panel_e_dual_membrane(ax_e)

    # Add panel labels
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e],
                         ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    # Overall title
    fig.suptitle('Circuit Complementarity: The Ammeter/Voltmeter Constraint',
                fontsize=12, fontweight='bold', y=0.98)

    return fig

def draw_resistor(ax, x, y, width=0.3, height=0.15, vertical=True):
    """Draw a resistor symbol"""
    if vertical:
        # Vertical resistor (zigzag)
        zigzag_x = [x, x+width/3, x-width/3, x+width/3, x-width/3, x]
        zigzag_y = np.linspace(y-height/2, y+height/2, 6)
        ax.plot(zigzag_x, zigzag_y, 'k-', linewidth=2)
    else:
        # Horizontal resistor
        zigzag_y = [y, y+height/3, y-height/3, y+height/3, y-height/3, y]
        zigzag_x = np.linspace(x-width/2, x+width/2, 6)
        ax.plot(zigzag_x, zigzag_y, 'k-', linewidth=2)

def draw_ammeter(ax, x, y, radius=0.2):
    """Draw ammeter symbol"""
    circle = Circle((x, y), radius, facecolor='white',
                   edgecolor=COLORS['ammeter'], linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, 'A', ha='center', va='center',
           fontsize=10, fontweight='bold', color=COLORS['ammeter'])

def draw_voltmeter(ax, x, y, radius=0.2):
    """Draw voltmeter symbol"""
    circle = Circle((x, y), radius, facecolor='white',
                   edgecolor=COLORS['voltmeter'], linewidth=2.5)
    ax.add_patch(circle)
    ax.text(x, y, 'V', ha='center', va='center',
           fontsize=10, fontweight='bold', color=COLORS['voltmeter'])

def draw_battery(ax, x, y, height=0.3):
    """Draw battery symbol"""
    # Long line (positive)
    ax.plot([x-0.15, x+0.15], [y+height/4, y+height/4], 'k-', linewidth=2.5)
    # Short line (negative)
    ax.plot([x-0.1, x+0.1], [y-height/4, y-height/4], 'k-', linewidth=2)
    # Plus sign
    ax.text(x+0.25, y+height/4, '+', fontsize=10, fontweight='bold')
    # Minus sign
    ax.text(x+0.25, y-height/4, '−', fontsize=10, fontweight='bold')

def plot_panel_a_ammeter(ax):
    """Panel A: Ammeter configuration (measures current)"""
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(2, 4.7, 'Ammeter Configuration',
            ha='center', fontsize=10, fontweight='bold')
    ax.text(2, 4.4, '(Direct current measurement)',
            ha='center', fontsize=8, style='italic', color=COLORS['ammeter'])

    # Circuit elements
    # Battery
    draw_battery(ax, 0.5, 2.5)

    # Wires
    ax.plot([0.5, 0.5], [2.8, 4], 'k-', linewidth=2)  # Up from battery
    ax.plot([0.5, 3.5], [4, 4], 'k-', linewidth=2)    # Top wire
    ax.plot([3.5, 3.5], [4, 3.5], 'k-', linewidth=2)  # Down to resistor
    ax.plot([3.5, 3.5], [1.5, 1], 'k-', linewidth=2)  # Down from resistor
    ax.plot([3.5, 0.5], [1, 1], 'k-', linewidth=2)    # Bottom wire
    ax.plot([0.5, 0.5], [1, 2.2], 'k-', linewidth=2)  # Up to battery

    # Ammeter (in series)
    draw_ammeter(ax, 2, 4, radius=0.25)

    # Resistor
    draw_resistor(ax, 3.5, 2.5, width=0.3, height=0.8, vertical=True)
    ax.text(3.9, 2.5, 'R', fontsize=9, fontweight='bold')

    # Current arrow
    ax.annotate('', xy=(1, 4), xytext=(0.8, 4),
               arrowprops=dict(arrowstyle='->', color=COLORS['ammeter'],
                             lw=2.5))
    ax.text(0.9, 4.3, 'I', fontsize=10, fontweight='bold',
           color=COLORS['ammeter'])

    # Measurement box
    box_y = 0.3
    ax.text(2, box_y, '✓ Directly Measured: I (current)',
           ha='center', fontsize=8, fontweight='bold',
           color=COLORS['ammeter'],
           bbox=dict(boxstyle='round', facecolor=COLORS['ammeter'],
                    alpha=0.2, edgecolor=COLORS['ammeter']))

    # Calculation box
    ax.text(2, -0.2, '⊕ Calculated: V = I × R (voltage)',
           ha='center', fontsize=8, style='italic',
           color=COLORS['voltmeter'],
           bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.8, edgecolor=COLORS['voltmeter'], linestyle='--'))

    # Impedance note
    ax.text(2, -0.6, 'Ammeter: Low impedance (≈0 Ω)',
           ha='center', fontsize=7, style='italic', color='gray')

def plot_panel_b_voltmeter(ax):
    """Panel B: Voltmeter configuration (measures voltage)"""
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(2, 4.7, 'Voltmeter Configuration',
            ha='center', fontsize=10, fontweight='bold')
    ax.text(2, 4.4, '(Direct voltage measurement)',
            ha='center', fontsize=8, style='italic', color=COLORS['voltmeter'])

    # Circuit elements
    # Battery
    draw_battery(ax, 0.5, 2.5)

    # Main circuit wires
    ax.plot([0.5, 0.5], [2.8, 4], 'k-', linewidth=2)
    ax.plot([0.5, 3.5], [4, 4], 'k-', linewidth=2)
    ax.plot([3.5, 3.5], [4, 3.5], 'k-', linewidth=2)
    ax.plot([3.5, 3.5], [1.5, 1], 'k-', linewidth=2)
    ax.plot([3.5, 0.5], [1, 1], 'k-', linewidth=2)
    ax.plot([0.5, 0.5], [1, 2.2], 'k-', linewidth=2)

    # Resistor
    draw_resistor(ax, 3.5, 2.5, width=0.3, height=0.8, vertical=True)
    ax.text(3.9, 2.5, 'R', fontsize=9, fontweight='bold')

    # Voltmeter (in parallel) - separate branch
    ax.plot([3.5, 3], [3.5, 3.5], 'k-', linewidth=1.5, linestyle='--')
    ax.plot([3, 3], [3.5, 1.5], 'k-', linewidth=1.5, linestyle='--')
    ax.plot([3, 3.5], [1.5, 1.5], 'k-', linewidth=1.5, linestyle='--')
    draw_voltmeter(ax, 3, 2.5, radius=0.25)

    # Voltage markers
    ax.plot([3.3, 3.3], [3.3, 3.5], color=COLORS['voltmeter'],
           linewidth=2, marker='o', markersize=4)
    ax.plot([3.3, 3.3], [1.5, 1.7], color=COLORS['voltmeter'],
           linewidth=2, marker='o', markersize=4)
    ax.text(3.05, 3.8, '+', fontsize=9, fontweight='bold',
           color=COLORS['voltmeter'])
    ax.text(3.05, 1.2, '−', fontsize=9, fontweight='bold',
           color=COLORS['voltmeter'])

    # Measurement box
    box_y = 0.3
    ax.text(2, box_y, '✓ Directly Measured: V (voltage)',
           ha='center', fontsize=8, fontweight='bold',
           color=COLORS['voltmeter'],
           bbox=dict(boxstyle='round', facecolor=COLORS['voltmeter'],
                    alpha=0.2, edgecolor=COLORS['voltmeter']))

    # Calculation box
    ax.text(2, -0.2, '⊕ Calculated: I = V / R (current)',
           ha='center', fontsize=8, style='italic',
           color=COLORS['ammeter'],
           bbox=dict(boxstyle='round', facecolor='white',
                    alpha=0.8, edgecolor=COLORS['ammeter'], linestyle='--'))

    # Impedance note
    ax.text(2, -0.6, 'Voltmeter: High impedance (≈∞ Ω)',
           ha='center', fontsize=7, style='italic', color='gray')

def plot_panel_c_invalid(ax):
    """Panel C: Invalid configuration (both in series)"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5, 4.7, '✗ INVALID: Both in Series (Measurement Incompatibility)',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['error'])

    # Circuit elements
    # Battery
    draw_battery(ax, 1, 2.5)

    # Wires
    ax.plot([1, 1], [2.8, 4], 'k-', linewidth=2)
    ax.plot([1, 9], [4, 4], 'k-', linewidth=2)
    ax.plot([9, 9], [4, 3.5], 'k-', linewidth=2)
    ax.plot([9, 9], [1.5, 1], 'k-', linewidth=2)
    ax.plot([9, 1], [1, 1], 'k-', linewidth=2)
    ax.plot([1, 1], [1, 2.2], 'k-', linewidth=2)

    # Ammeter
    draw_ammeter(ax, 3, 4, radius=0.3)
    ax.text(3, 3.3, 'Low Z', ha='center', fontsize=7,
           style='italic', color=COLORS['ammeter'])

    # Voltmeter (incorrectly in series!)
    draw_voltmeter(ax, 6, 4, radius=0.3)
    ax.text(6, 3.3, 'High Z', ha='center', fontsize=7,
           style='italic', color=COLORS['voltmeter'])

    # Resistor
    draw_resistor(ax, 9, 2.5, width=0.3, height=0.8, vertical=True)
    ax.text(9.4, 2.5, 'R', fontsize=9, fontweight='bold')

    # Big X marks over the configuration
    ax.plot([2, 7], [4.5, 3.5], 'r-', linewidth=5, alpha=0.5)
    ax.plot([2, 7], [3.5, 4.5], 'r-', linewidth=5, alpha=0.5)

    # Error boxes
    error_y = 0.8

    # Ammeter wants low impedance
    ax.text(3, error_y, 'Ammeter wants:\nAll current to flow\n(Low Z ≈ 0)',
           ha='center', fontsize=7, color=COLORS['ammeter'],
           bbox=dict(boxstyle='round', facecolor=COLORS['ammeter'],
                    alpha=0.2, edgecolor=COLORS['ammeter']))

    # Conflict arrow
    ax.annotate('', xy=(5.2, error_y), xytext=(3.8, error_y),
               arrowprops=dict(arrowstyle='<->', color=COLORS['error'],
                             lw=3))
    ax.text(4.5, error_y+0.5, '⚠ CONFLICT', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['error'])

    # Voltmeter wants high impedance
    ax.text(7, error_y, 'Voltmeter wants:\nNo current to flow\n(High Z ≈ ∞)',
           ha='center', fontsize=7, color=COLORS['voltmeter'],
           bbox=dict(boxstyle='round', facecolor=COLORS['voltmeter'],
                    alpha=0.2, edgecolor=COLORS['voltmeter']))

    # Bottom explanation
    ax.text(5, 0.1, 'Cannot have both in series: Impedances are incompatible',
           ha='center', fontsize=8, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

def plot_panel_d_mapping(ax):
    """Panel D: Mapping table between circuits and dual-membrane"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, 'Mapping: Circuit ↔ Dual-Membrane',
            ha='center', fontsize=10, fontweight='bold')

    # Table headers
    header_y = 4.8
    ax.text(2.5, header_y, 'Electrical Circuit', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['circuit'])
    ax.text(7.5, header_y, 'Dual-Membrane', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['success'])

    # Separator line
    ax.plot([0.5, 9.5], [4.6, 4.6], 'k-', linewidth=1)

    # Mapping rows
    mappings = [
        ('Ammeter (measures I)', 'Front face (observable)', COLORS['ammeter']),
        ('Voltmeter (measures V)', 'Back face (hidden)', COLORS['voltmeter']),
        ("Ohm's law: V = IR", 'Conjugate: Back = T(Front)', COLORS['success']),
        ('Switch ammeter → voltmeter', 'Switch observable face', COLORS['success']),
        ('Cannot measure both', 'Complementarity', COLORS['error']),
        ('Low Z vs High Z', 'Categorical orthogonality', COLORS['circuit']),
    ]

    y_start = 4.2
    y_spacing = 0.65

    for i, (circuit, membrane, color) in enumerate(mappings):
        y = y_start - i * y_spacing

        # Circuit side
        ax.text(2.5, y, circuit, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor=color, alpha=0.7))

        # Arrow
        ax.annotate('', xy=(5.5, y), xytext=(4.5, y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Membrane side
        ax.text(7.5, y, membrane, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=color,
                        alpha=0.2, edgecolor=color))

    # Bottom note
    ax.text(5, 0.3, 'Same fundamental constraint: Measurement apparatus determines observable',
           ha='center', fontsize=8, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

def plot_panel_e_dual_membrane(ax):
    """Panel E: Dual-membrane with circuit analogy"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Dual-Membrane as Electrical Circuit',
            ha='center', fontsize=10, fontweight='bold')

    # Front face (Ammeter mode)
    front_x, front_y = 2, 3
    front_box = FancyBboxPatch((front_x-0.8, front_y-0.6), 1.6, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['ammeter'], alpha=0.2,
                              edgecolor=COLORS['ammeter'], linewidth=2.5)
    ax.add_patch(front_box)

    # Ammeter symbol in front
    draw_ammeter(ax, front_x, front_y+0.3, radius=0.15)
    ax.text(front_x, front_y-0.1, 'Front Face', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['ammeter'])
    ax.text(front_x, front_y-0.35, '(Observable)', ha='center', fontsize=7,
           style='italic', color=COLORS['ammeter'])

    # Measurement
    ax.text(front_x, front_y-0.9, '✓ Direct: S_k', ha='center', fontsize=7,
           fontweight='bold', color=COLORS['ammeter'])
    ax.text(front_x, front_y-1.15, '⊕ Derived: T(S_k)', ha='center', fontsize=7,
           style='italic', color=COLORS['voltmeter'])

    # Back face (Voltmeter mode)
    back_x, back_y = 8, 3
    back_box = FancyBboxPatch((back_x-0.8, back_y-0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS['voltmeter'], alpha=0.2,
                             edgecolor=COLORS['voltmeter'], linewidth=2.5)
    ax.add_patch(back_box)

    # Voltmeter symbol in back
    draw_voltmeter(ax, back_x, back_y+0.3, radius=0.15)
    ax.text(back_x, back_y-0.1, 'Back Face', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['voltmeter'])
    ax.text(back_x, back_y-0.35, '(Conjugate)', ha='center', fontsize=7,
           style='italic', color=COLORS['voltmeter'])

    # Measurement
    ax.text(back_x, back_y-0.9, '✓ Direct: T(S_k)', ha='center', fontsize=7,
           fontweight='bold', color=COLORS['voltmeter'])
    ax.text(back_x, back_y-1.15, '⊕ Derived: S_k', ha='center', fontsize=7,
           style='italic', color=COLORS['ammeter'])

    # Connection (conjugate transform)
    arrow = FancyArrowPatch((front_x+1, front_y), (back_x-1, back_y),
                           arrowstyle='<->', mutation_scale=25,
                           color=COLORS['success'], linewidth=3)
    ax.add_patch(arrow)
    ax.text(5, front_y+0.5, 'Conjugate Transform', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['success'])
    ax.text(5, front_y+0.15, '(Like V = IR)', ha='center', fontsize=7,
           style='italic', color=COLORS['success'])

    # Switch apparatus
    ax.text(5, front_y-0.3, '⇄ Switch Measurement Apparatus', ha='center',
           fontsize=8, style='italic', color='gray')

    # Complementarity note
    comp_y = 0.5
    ax.text(5, comp_y, '⚠ Cannot observe both simultaneously',
           ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3,
                    edgecolor=COLORS['error'], linewidth=2))
    ax.text(5, comp_y-0.5, '(Same as: Cannot have ammeter AND voltmeter in series)',
           ha='center', fontsize=7, style='italic', color='gray')

if __name__ == '__main__':
    fig = create_figure_5()
    plt.savefig('figure_5_circuit_complementarity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_5_circuit_complementarity.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 5 saved: figure_5_circuit_complementarity.pdf/png")
    plt.show()
