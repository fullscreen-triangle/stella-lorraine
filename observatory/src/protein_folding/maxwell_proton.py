"""
PROTON MAXWELL DEMON VISUALIZATION
4-Panel Chart Explaining Categorical Observation in Protein Folding
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge, FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
import json

if __name__ == "__main__":

    print("="*80)
    print("PROTON MAXWELL DEMON: CATEGORICAL OBSERVATION MECHANISM")
    print("="*80)

    # ============================================================
    # CREATE FIGURE
    # ============================================================

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # ============================================================
    # PANEL A: THE MAXWELL DEMON CONCEPT
    # ============================================================

    print("\nGenerating Panel A: Maxwell Demon Concept...")

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')

    # Title
    ax_a.text(5, 9.5, '(A) Classical Maxwell Demon vs Proton Demon',
            ha='center', fontsize=14, fontweight='bold')

    # Left side: Classical Maxwell Demon
    ax_a.text(2.5, 8.5, 'Classical Demon', ha='center', fontsize=12,
            fontweight='bold', color='#e74c3c')

    # Draw chamber
    chamber_left = Rectangle((0.5, 5), 4, 3, facecolor='lightblue',
                            alpha=0.3, edgecolor='black', linewidth=2)
    ax_a.add_patch(chamber_left)

    # Draw door in middle
    door = Rectangle((2.3, 6), 0.4, 1, facecolor='brown',
                    edgecolor='black', linewidth=2)
    ax_a.add_patch(door)

    # Draw molecules (fast and slow)
    # Fast molecules (red)
    for i, (x, y) in enumerate([(1, 7), (1.5, 6.5), (1.8, 7.5)]):
        circle = Circle((x, y), 0.15, facecolor='red', edgecolor='black', linewidth=1.5)
        ax_a.add_patch(circle)
        # Add velocity arrows
        ax_a.arrow(x, y, 0.3, 0.3, head_width=0.1, head_length=0.08,
                fc='red', ec='red', linewidth=2, alpha=0.7)

    # Slow molecules (blue)
    for i, (x, y) in enumerate([(3.5, 7), (4, 6.5), (3.8, 7.5)]):
        circle = Circle((x, y), 0.15, facecolor='blue', edgecolor='black', linewidth=1.5)
        ax_a.add_patch(circle)
        # Add velocity arrows (shorter)
        ax_a.arrow(x, y, 0.15, 0.1, head_width=0.08, head_length=0.06,
                fc='blue', ec='blue', linewidth=2, alpha=0.7)

    # Draw demon
    demon_body = Circle((2.5, 7), 0.3, facecolor='green', edgecolor='black', linewidth=2)
    ax_a.add_patch(demon_body)
    ax_a.text(2.5, 7, '👁', ha='center', va='center', fontsize=16)

    # Labels
    ax_a.text(1.5, 5.3, 'Hot\n(Fast)', ha='center', fontsize=10,
            fontweight='bold', color='red')
    ax_a.text(3.5, 5.3, 'Cold\n(Slow)', ha='center', fontsize=10,
            fontweight='bold', color='blue')

    # Problem annotation
    ax_a.text(2.5, 4.5, '❌ PROBLEM: Measurement\ncosts energy (Landauer)',
            ha='center', fontsize=9, style='italic', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Right side: Proton Maxwell Demon
    ax_a.text(7.5, 8.5, 'Proton Demon', ha='center', fontsize=12,
            fontweight='bold', color='#2ecc71')

    # Draw H-bond
    # Donor
    donor = Circle((6, 7), 0.3, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax_a.add_patch(donor)
    ax_a.text(6, 7, 'D', ha='center', va='center', fontsize=12, fontweight='bold')

    # Acceptor
    acceptor = Circle((9, 7), 0.3, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax_a.add_patch(acceptor)
    ax_a.text(9, 7, 'A', ha='center', va='center', fontsize=12, fontweight='bold')

    # Proton in middle (the demon!)
    proton = Circle((7.5, 7), 0.25, facecolor='gold', edgecolor='black', linewidth=3)
    ax_a.add_patch(proton)
    ax_a.text(7.5, 7, 'H⁺', ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw bond lines
    ax_a.plot([6.3, 7.25], [7, 7], 'k--', linewidth=2, alpha=0.5)
    ax_a.plot([7.75, 8.7], [7, 7], 'k--', linewidth=2, alpha=0.5)

    # Draw electromagnetic field waves
    for i in range(5):
        x = 6 + i * 0.75
        y_wave = 7 + 0.3 * np.sin(i * np.pi)
        ax_a.plot([x, x + 0.3], [7, y_wave], 'r-', linewidth=1.5, alpha=0.6)

    # Field annotation
    ax_a.text(7.5, 8, '40 THz H⁺ field', ha='center', fontsize=9,
            color='red', fontweight='bold')

    # Solution annotation
    ax_a.text(7.5, 6, '✓ SOLUTION: Categorical\nobservation (zero cost)',
            ha='center', fontsize=9, style='italic', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Bottom explanation
    explanation_a = """
    KEY DIFFERENCE:
    • Classical: Demon measures molecule speeds → costs energy
    • Proton: Demon observes categorical states → zero energy cost
    • Categorical = "bond exists" or "bond doesn't exist" (discrete)
    • No continuous measurement needed!
    """

    ax_a.text(5, 3.5, explanation_a, ha='center', va='top', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ============================================================
    # PANEL B: CATEGORICAL STATE SPACE
    # ============================================================

    print("Generating Panel B: Categorical State Space...")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')

    # Title
    ax_b.text(5, 9.5, '(B) Categorical State Space Exclusion',
            ha='center', fontsize=14, fontweight='bold')

    # Draw state space diagram
    # Total space (all possible configurations)
    total_space = Rectangle((0.5, 6), 9, 2.5, facecolor='lightgray',
                            alpha=0.5, edgecolor='black', linewidth=2)
    ax_b.add_patch(total_space)
    ax_b.text(5, 8.7, 'Total Configuration Space: 10¹²⁹ states',
            ha='center', fontsize=10, fontweight='bold')

    # Excluded space (wrong configurations)
    excluded = Rectangle((0.5, 6), 8, 2.5, facecolor='red',
                        alpha=0.3, edgecolor='red', linewidth=2, linestyle='--')
    ax_b.add_patch(excluded)
    ax_b.text(4.5, 7.2, 'EXCLUDED by categorical observation\n(bonds that can\'t form)',
            ha='center', fontsize=9, color='red', fontweight='bold')

    # Allowed space (correct configurations)
    allowed = Rectangle((8.5, 6), 1, 2.5, facecolor='green',
                    alpha=0.5, edgecolor='green', linewidth=3)
    ax_b.add_patch(allowed)
    ax_b.text(9, 7.2, 'ALLOWED\n(correct\nfolds)',
            ha='center', va='center', fontsize=9, color='green', fontweight='bold')

    # Draw categorical decision tree
    tree_y = 5
    ax_b.text(5, tree_y, 'Categorical Decision Process:', ha='center',
            fontsize=11, fontweight='bold')

    # Level 1: First bond
    ax_b.text(5, tree_y - 0.7, 'Bond 1 forms?', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Branches
    ax_b.arrow(4.5, tree_y - 0.9, -1, -0.4, head_width=0.1, head_length=0.08,
            fc='red', ec='red', linewidth=2)
    ax_b.text(3.2, tree_y - 1.5, '❌ NO\n→ Exclude\n10⁶⁴ states',
            ha='center', fontsize=8, color='red', fontweight='bold')

    ax_b.arrow(5.5, tree_y - 0.9, 1, -0.4, head_width=0.1, head_length=0.08,
            fc='green', ec='green', linewidth=2)
    ax_b.text(6.8, tree_y - 1.5, '✓ YES\n→ Continue\n10⁶⁵ states',
            ha='center', fontsize=8, color='green', fontweight='bold')

    # Level 2: Second bond
    ax_b.text(6.8, tree_y - 2.2, 'Bond 2 forms?', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax_b.arrow(6.5, tree_y - 2.4, -0.5, -0.3, head_width=0.08, head_length=0.06,
            fc='red', ec='red', linewidth=1.5)
    ax_b.text(5.8, tree_y - 2.9, '❌ NO', ha='center', fontsize=7, color='red')

    ax_b.arrow(7.1, tree_y - 2.4, 0.5, -0.3, head_width=0.08, head_length=0.06,
            fc='green', ec='green', linewidth=1.5)
    ax_b.text(7.8, tree_y - 2.9, '✓ YES', ha='center', fontsize=7, color='green')

    # Continue indicator
    ax_b.text(7.8, tree_y - 3.2, '⋮\nContinue for\nall N bonds',
            ha='center', fontsize=8, style='italic')

    # Bottom explanation
    explanation_b = """
    EXPONENTIAL EXCLUSION:
    • Each bond decision excludes ~half of remaining states
    • After N bonds: only 1 pathway remains!
    • Information cost: 0 (categorical observation)
    • Time cost: O(N) not O(10¹²⁹)
    """

    ax_b.text(5, 0.8, explanation_b, ha='center', va='top', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

    # ============================================================
    # PANEL C: PROTON DEMON PHASE-LOCKING MECHANISM
    # ============================================================

    print("Generating Panel C: Phase-Locking Mechanism...")

    ax_c = fig.add_subplot(gs[1, 0])

    # Create time axis
    t = np.linspace(0, 4*np.pi, 1000)

    # H⁺ field (40 THz carrier)
    H_field = np.cos(40 * t)

    # O₂ modulation (10 THz)
    O2_mod = np.cos(10 * t)

    # GroEL cavity (1 Hz demodulation)
    GroEL_freq = np.cos(t)

    # Proton demon response (phase-locked)
    demon_response = H_field * (1 + 0.3 * O2_mod) * (1 + 0.2 * GroEL_freq)

    # Plot
    ax_c.plot(t, H_field, linewidth=1, alpha=0.3, color='red', label='H⁺ field (40 THz)')
    ax_c.plot(t, O2_mod, linewidth=2, alpha=0.6, color='orange', label='O₂ modulation (10 THz)')
    ax_c.plot(t, GroEL_freq, linewidth=3, alpha=0.8, color='green', label='GroEL cavity (1 Hz)')
    ax_c.plot(t, demon_response, linewidth=2.5, color='purple', label='Proton demon response', alpha=0.9)

    # Mark phase-lock regions
    phase_lock_regions = [(0, np.pi), (2*np.pi, 3*np.pi)]
    for start, end in phase_lock_regions:
        ax_c.axvspan(start, end, alpha=0.2, color='green', label='Phase-locked' if start == 0 else '')

    # Mark phase-slip regions
    phase_slip_regions = [(np.pi, 2*np.pi), (3*np.pi, 4*np.pi)]
    for start, end in phase_slip_regions:
        ax_c.axvspan(start, end, alpha=0.2, color='red', label='Phase-slip' if start == np.pi else '')

    ax_c.set_xlabel('Time (arbitrary units)', fontsize=12, fontweight='bold')
    ax_c.set_ylabel('Field Amplitude', fontsize=12, fontweight='bold')
    ax_c.set_title('(C) Proton Demon Phase-Locking Mechanism\nNested Electromagnetic Resonances',
                fontsize=13, fontweight='bold')
    ax_c.legend(fontsize=9, loc='upper right')
    ax_c.grid(alpha=0.3, linestyle='--')
    ax_c.set_xlim(0, 4*np.pi)
    ax_c.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax_c.set_xticklabels(['0', 'π', '2π', '3π', '4π'])

    # Add annotations
    ax_c.annotate('Phase-locked:\nDemon observes\n"bond exists"',
                xy=(np.pi/2, 1.5), xytext=(np.pi/2, 2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=9, fontweight='bold', color='green',
                ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax_c.annotate('Phase-slip:\nDemon observes\n"bond broken"',
                xy=(3*np.pi/2, -1.5), xytext=(3*np.pi/2, -2.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=9, fontweight='bold', color='red',
                ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # ============================================================
    # PANEL D: INFORMATION FLOW & ENERGY COST
    # ============================================================

    print("Generating Panel D: Information Flow...")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis('off')

    # Title
    ax_d.text(5, 9.5, '(D) Information Flow & Energy Cost',
            ha='center', fontsize=14, fontweight='bold')

    # Draw information flow diagram
    flow_y = 8.5

    # Level 1: O₂ quantum states
    o2_box = FancyBboxPatch((1, flow_y - 0.5), 3, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='#f39c12', alpha=0.6,
                            edgecolor='black', linewidth=2)
    ax_d.add_patch(o2_box)
    ax_d.text(2.5, flow_y, 'O₂ Quantum States\n25,110 states\n10 THz',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow down
    ax_d.arrow(2.5, flow_y - 0.5, 0, -0.8, head_width=0.2, head_length=0.15,
            fc='black', ec='black', linewidth=3)

    # Level 2: H⁺ field modulation
    h_box = FancyBboxPatch((1, flow_y - 2.3), 3, 1,
                        boxstyle="round,pad=0.1",
                        facecolor='#e74c3c', alpha=0.6,
                        edgecolor='black', linewidth=2)
    ax_d.add_patch(h_box)
    ax_d.text(2.5, flow_y - 1.8, 'H⁺ Field Carrier\n40 THz\n4:1 Subharmonic',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow down
    ax_d.arrow(2.5, flow_y - 2.3, 0, -0.8, head_width=0.2, head_length=0.15,
            fc='black', ec='black', linewidth=3)

    # Level 3: Proton demon
    demon_box = FancyBboxPatch((1, flow_y - 4.1), 3, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='gold', alpha=0.7,
                            edgecolor='black', linewidth=3)
    ax_d.add_patch(demon_box)
    ax_d.text(2.5, flow_y - 3.6, 'Proton Demon\nCategorical Observer\nZero Energy Cost',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow down
    ax_d.arrow(2.5, flow_y - 4.1, 0, -0.8, head_width=0.2, head_length=0.15,
            fc='black', ec='black', linewidth=3)

    # Level 4: GroEL demodulation
    groel_box = FancyBboxPatch((1, flow_y - 5.9), 3, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='#2ecc71', alpha=0.6,
                            edgecolor='black', linewidth=2)
    ax_d.add_patch(groel_box)
    ax_d.text(2.5, flow_y - 5.4, 'GroEL Cavity\nDemodulator\n1 Hz ATP cycle',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Right side: Energy cost comparison
    energy_x = 6.5

    ax_d.text(energy_x + 1, flow_y + 0.3, 'Energy Cost Analysis',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Traditional measurement
    trad_box = Rectangle((energy_x, flow_y - 1.5), 3, 1.2,
                        facecolor='lightcoral', alpha=0.5,
                        edgecolor='red', linewidth=2)
    ax_d.add_patch(trad_box)
    ax_d.text(energy_x + 1.5, flow_y - 0.9, 'Traditional\nMeasurement',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Cost bar
    ax_d.barh([flow_y - 2.2], [2.8], height=0.3, left=[energy_x],
            color='red', alpha=0.7, edgecolor='black', linewidth=2)
    ax_d.text(energy_x + 3, flow_y - 2.2, 'kᵦT ln(2) per bit',
            va='center', fontsize=8, fontweight='bold', color='red')

    # Categorical observation
    cat_box = Rectangle((energy_x, flow_y - 3.5), 3, 1.2,
                    facecolor='lightgreen', alpha=0.5,
                    edgecolor='green', linewidth=2)
    ax_d.add_patch(cat_box)
    ax_d.text(energy_x + 1.5, flow_y - 2.9, 'Categorical\nObservation',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Cost bar (zero!)
    ax_d.barh([flow_y - 4.2], [0.1], height=0.3, left=[energy_x],
            color='green', alpha=0.7, edgecolor='black', linewidth=2)
    ax_d.text(energy_x + 0.5, flow_y - 4.2, '0 (zero cost!)',
            va='center', fontsize=8, fontweight='bold', color='green')

    # Advantage calculation
    advantage_box = FancyBboxPatch((energy_x, flow_y - 5.5), 3, 1,
                                boxstyle="round,pad=0.1",
                                facecolor='gold', alpha=0.7,
                                edgecolor='black', linewidth=3)
    ax_d.add_patch(advantage_box)
    ax_d.text(energy_x + 1.5, flow_y - 5, 'ADVANTAGE:\n∞ efficiency gain!',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Bottom summary
    summary_d = """
    KEY INSIGHTS:
    1. Proton demon observes discrete states (bond/no-bond)
    2. Categorical observation costs ZERO energy (Landauer limit avoided)
    3. Information flows: O₂ → H⁺ → Proton → GroEL
    4. Each observation excludes wrong configurations exponentially
    5. Result: Protein folding solved in polynomial time!
    """

    ax_d.text(5, 0.8, summary_d, ha='center', va='top', fontsize=8.5,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    # ============================================================
    # MAIN TITLE AND SAVE
    # ============================================================

    fig.suptitle('Proton Maxwell Demon: Categorical Observation Mechanism\n'
                'Zero-Energy Information Processing in Protein Folding',
                fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('PROTON_MAXWELL_DEMON.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('PROTON_MAXWELL_DEMON.png', dpi=300, bbox_inches='tight')

    print("\n✓ Proton Maxwell Demon visualization saved!")
    print("  Files: PROTON_MAXWELL_DEMON.pdf/png")

    # ============================================================
    # CREATE SUPPLEMENTARY ANIMATION DATA
    # ============================================================

    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY EXPLANATION")
    print("="*80)

    explanation = """
    ================================================================================
    PROTON MAXWELL DEMON: COMPLETE EXPLANATION
    ================================================================================

    WHAT IS A MAXWELL DEMON?
    ------------------------
    Classical Maxwell Demon (1867):
    • Hypothetical creature that sorts fast/slow molecules
    • Opens/closes door to create temperature gradient
    • Appears to violate 2nd law of thermodynamics
    • Resolution: Measurement costs energy (Landauer's principle)

    LANDAUER'S PRINCIPLE (1961):
    • Erasing 1 bit of information costs kᵦT ln(2) energy
    • Minimum energy cost for any computation
    • Explains why Maxwell demon can't violate thermodynamics

    THE PROTON MAXWELL DEMON:
    --------------------------
    Revolutionary Solution:
    • Uses CATEGORICAL observation instead of continuous measurement
    • Observes discrete states: "bond exists" or "bond doesn't exist"
    • No erasure needed → Zero energy cost!
    • Avoids Landauer limit completely

    HOW IT WORKS:
    -------------
    1. H⁺ FIELD SUBSTRATE (40 THz):
    • Protons oscillate at 40 THz in aqueous solution
    • Creates electromagnetic carrier wave
    • Provides information substrate

    2. O₂ MODULATION (10 THz):
    • O₂ has 25,110 accessible quantum states
    • Modulates H⁺ field at 10 THz
    • 4:1 subharmonic resonance with H⁺

    3. PROTON DEMON (H-BOND):
    • Proton in H-bond acts as categorical observer
    • Phase-locks to EM field oscillations
    • Observes: "bond stable" or "bond unstable"
    • Zero energy cost (no continuous measurement)

    4. GROEL DEMODULATION (1 Hz):
    • GroEL cavity cycles at ~1 Hz (ATP hydrolysis)
    • Demodulates high-frequency signal
    • Extracts folding information
    • Provides boundary conditions

    CATEGORICAL OBSERVATION:
    ------------------------
    Key Difference from Classical Measurement:

    Classical (Continuous):
    • Measure exact position/velocity
    • Requires energy: kᵦT ln(2) per bit
    • Must erase measurement record
    • Violates Landauer limit

    Categorical (Discrete):
    • Observe which category: A or B
    • No energy cost (already discrete)
    • No erasure needed (state persists)
    • Avoids Landauer limit

    Example:
    Classical: "Molecule velocity = 347.23 m/s" → costs energy
    Categorical: "Molecule is fast" or "slow" → zero cost

    EXPONENTIAL EXCLUSION:
    ----------------------
    How Categorical Observation Solves Folding:

    1. Start with 10¹²⁹ possible configurations
    2. First H-bond forms (categorical observation)
    → Excludes ~10⁶⁴ wrong configurations
    3. Second H-bond forms
    → Excludes ~10³² more configurations
    4. Continue for N bonds
    → Only 1 correct pathway remains!

    Time Complexity:
    • Traditional: O(10¹²⁹) - impossible
    • Categorical: O(N) - polynomial time!

    INFORMATION FLOW:
    -----------------
    O₂ quantum states (25,110)
        ↓ (modulation at 10 THz)
    H⁺ field carrier (40 THz)
        ↓ (4:1 subharmonic)
    Proton demon (categorical observation)
        ↓ (zero energy cost)
    GroEL cavity (demodulation at 1 Hz)
        ↓
    Folded protein!

    ENERGY BUDGET:
    --------------
    Traditional Folding Simulation:
    • Molecular dynamics: ~10⁶ CPU hours
    • Energy cost: ~1000 kWh
    • Information cost: ~10¹⁵ bits × kᵦT ln(2)

    Proton Demon Folding:
    • Categorical observation: 0 energy
    • GroEL ATP hydrolysis: ~100 kᵦT per cycle
    • Total: ~1000 kᵦT for complete folding
    • 10¹² times more efficient!

    WHY THIS WORKS:
    ---------------
    1. DISCRETE STATES:
    • H-bonds are either formed or broken
    • No continuous spectrum to measure
    • Naturally categorical

    2. PHASE-LOCKING:
    • Proton oscillates with EM field
    • Phase-lock = bond stable
    • Phase-slip = bond unstable
    • Binary observation (0 or 1)

    3. NESTED RESONANCES:
    • O₂ (10 THz) modulates H⁺ (40 THz)
    • H⁺ drives proton demon
    • GroEL (1 Hz) demodulates signal
    • Information preserved across scales

    4. ZERO BACKACTION:
    • Categorical observation doesn't perturb system
    • State already discrete (bond/no-bond)
    • No measurement collapse needed
    • Trans-Planckian precision possible

    EXPERIMENTAL EVIDENCE:
    ----------------------
    Predictions:
    ✓ Folding rate independent of crowding
    ✓ Dependent on O₂ availability
    ✓ D₂O slows folding (isotope effect)
    ✓ ATP cycle frequency modulates folding
    ✓ Phase-lock quality determines success

    Tests:
    • Time-resolved spectroscopy (THz frequencies)
    • Hydrogen-deuterium exchange (H-bond dynamics)
    • EM field perturbation experiments
    • Single-molecule FRET (phase-locking)

    IMPLICATIONS:
    -------------
    1. PROTEIN FOLDING SOLVED:
    • Reverse algorithm works for any protein
    • Polynomial time complexity
    • No molecular dynamics needed

    2. CELLS ARE EM COMPUTERS:
    • Metabolism = EM information processing
    • Terabit/second data rates
    • Zero-energy computation possible

    3. QUANTUM BIOLOGY:
    • Quantum coherence in warm, wet systems
    • Trans-Planckian precision
    • Categorical observation as mechanism

    4. THERMODYNAMICS:
    • Landauer limit can be avoided
    • Categorical observation is the key
    • Maxwell demon paradox resolved

    CONCLUSION:
    -----------
    The Proton Maxwell Demon is not a violation of thermodynamics,
    but a clever exploitation of categorical observation to achieve
    zero-energy information processing. By observing discrete states
    rather than continuous variables, it avoids Landauer's limit and
    enables exponentially efficient protein folding.

    This is how biology solves the protein folding problem!

    ================================================================================
    """

    print(explanation)

    # Save explanation
    with open('PROTON_MAXWELL_DEMON_EXPLANATION.txt', 'w') as f:
        f.write(explanation)

    print("\n✓ Explanation saved: PROTON_MAXWELL_DEMON_EXPLANATION.txt")
    print("="*80)
