#!/usr/bin/env python3
"""
Ion Journey Visualization - Template-Based Analysis Implementation
===================================================================

Shows a single ion's journey through each stage of the pipeline,
implementing the "mold" concept from Template-Based Analysis.

Each panel shows the ion at a different stage:
1. Injection/Sample
2. Chromatography (retention time)
3. Ionization (charge states)
4. MS1 (m/z space with S-entropy coordinates)
5. MS2 (fragmentation pattern) - if available
6. Partition Coordinates (quantum numbers n,l,m,s)
7. Thermodynamic State (We, Re, Oh)
8. Droplet Visualization (bijective encoding)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'dejavusans',
})

# Color palette
COLORS = {
    'ion': '#E63946',
    'mold': '#457B9D',
    'match': '#2A9D8F',
    'flow': '#264653',
    'highlight': '#F4A261',
    'background': '#F7F7F7',
}


@dataclass
class IonState:
    """Complete state of an ion at any pipeline stage."""
    mz: float
    intensity: float
    rt: float
    s_k: float = 0.0
    s_t: float = 0.0
    s_e: float = 0.0
    n: int = 0
    l: int = 0
    m: int = 0
    spin: float = 0.5
    velocity: float = 0.0
    radius: float = 0.0
    surface_tension: float = 0.0
    phase_coherence: float = 0.0
    categorical_state: int = 0


def load_pipeline_results(results_dir: Path) -> Dict:
    """Load all stage results from pipeline."""
    stages = {}
    stage_files = sorted(results_dir.glob('stages/*.json'))

    for stage_file in stage_files:
        with open(stage_file) as f:
            data = json.load(f)
            stages[data['stage_name']] = data

    return stages


def extract_sample_ion(stages: Dict) -> IonState:
    """Extract a representative ion from pipeline results."""
    # Get chromatography data for sample ion
    chrom = stages.get('02_chromatography', {}).get('data', {})
    sample_peaks = chrom.get('sample_peaks', [{}])

    if sample_peaks:
        peak = sample_peaks[0]
        ion_input = peak.get('input', {})
        s_entropy = peak.get('s_entropy', {})
        partition = peak.get('partition', {})

        ion = IonState(
            mz=ion_input.get('mz', 200.0),
            intensity=ion_input.get('intensity', 1000.0),
            rt=ion_input.get('retention_time', 5.0),
            s_k=s_entropy.get('S_k', 0.5),
            s_t=s_entropy.get('S_t', 0.1),
            s_e=s_entropy.get('S_e', 0.9),
            n=partition.get('n', 5),
            l=partition.get('l', 2),
            m=partition.get('m', 1),
            spin=partition.get('s', 0.5),
        )
    else:
        # Default ion
        ion = IonState(mz=200.0, intensity=1000.0, rt=5.0)

    # Get thermodynamic data
    thermo = stages.get('10_thermodynamics', {}).get('metrics', {})
    ion.velocity = 2.0  # m/s
    ion.radius = 3.0    # mm
    ion.surface_tension = 0.08  # N/m

    # Get visual validation data
    visual = stages.get('12_visual_validation', {}).get('data', {})
    droplet_summaries = visual.get('droplet_summaries', [{}])
    if droplet_summaries:
        ds = droplet_summaries[0]
        s_coords = ds.get('s_entropy_coords', {})
        d_params = ds.get('droplet_params', {})
        ion.s_k = s_coords.get('s_knowledge_mean', ion.s_k)
        ion.s_t = s_coords.get('s_time_mean', ion.s_t)
        ion.s_e = s_coords.get('s_entropy_mean', ion.s_e)
        ion.velocity = d_params.get('velocity_mean', ion.velocity)
        ion.radius = d_params.get('radius_mean', ion.radius)
        ion.surface_tension = d_params.get('surface_tension_mean', ion.surface_tension)
        ion.phase_coherence = d_params.get('phase_coherence_mean', 0.57)

        cat_states = ds.get('categorical_states', [1])
        ion.categorical_state = cat_states[0] if cat_states else 1

    return ion


def draw_stage_1_injection(ax: plt.Axes, ion: IonState) -> None:
    """Stage 1: Sample injection - ion as a point in the sample vial."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw sample vial
    vial = Rectangle((3, 1), 4, 6, fill=False, edgecolor=COLORS['flow'],
                     linewidth=1.5, linestyle='-')
    ax.add_patch(vial)

    # Draw liquid level
    liquid = Rectangle((3.1, 1.1), 3.8, 3, fill=True,
                       facecolor='#A8DADC', alpha=0.5, edgecolor='none')
    ax.add_patch(liquid)

    # Draw the ion as a bright dot in the liquid
    ion_circle = Circle((5, 2.5), 0.3, facecolor=COLORS['ion'],
                        edgecolor='white', linewidth=1)
    ax.add_patch(ion_circle)

    # Label
    ax.text(5, 8.5, 'Sample', ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 0.3, f'm/z = {ion.mz:.1f}', ha='center', fontsize=7)


def draw_stage_2_chromatography(ax: plt.Axes, ion: IonState) -> None:
    """Stage 2: Chromatography - ion at specific retention time."""
    # Draw chromatogram
    t = np.linspace(0, 15, 500)
    # Gaussian peaks
    chromatogram = (
        0.3 * np.exp(-((t - 3) / 0.5)**2) +
        0.5 * np.exp(-((t - 6) / 0.6)**2) +
        1.0 * np.exp(-((t - ion.rt) / 0.4)**2) +  # Our ion's peak
        0.4 * np.exp(-((t - 11) / 0.5)**2)
    )

    ax.fill_between(t, chromatogram, alpha=0.3, color=COLORS['mold'])
    ax.plot(t, chromatogram, color=COLORS['flow'], linewidth=1)

    # Highlight our ion's position
    ax.axvline(x=ion.rt, color=COLORS['ion'], linewidth=1.5, linestyle='--', alpha=0.7)
    ax.scatter([ion.rt], [1.0], s=100, c=COLORS['ion'], zorder=5, edgecolors='white')

    ax.set_xlabel('RT (min)')
    ax.set_ylabel('Signal')
    ax.set_title('Chromatography', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.3)


def draw_stage_3_ionization(ax: plt.Axes, ion: IonState) -> None:
    """Stage 3: Ionization - charge state distribution."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw ESI spray cone
    cone_x = [2, 5, 5, 2]
    cone_y = [5, 8, 2, 5]
    ax.fill(cone_x, cone_y, color='#A8DADC', alpha=0.3)
    ax.plot(cone_x[:2], cone_y[:2], color=COLORS['flow'], linewidth=1)
    ax.plot([cone_x[0], cone_x[3]], [cone_y[0], cone_y[3]], color=COLORS['flow'], linewidth=1)

    # Draw capillary
    cap = Rectangle((0.5, 4.5), 1.5, 1, fill=True, facecolor='gray', edgecolor='black')
    ax.add_patch(cap)

    # Draw droplets becoming ions
    for i, (x, y, size) in enumerate([(3, 5, 0.4), (4, 6, 0.25), (4.5, 4, 0.15),
                                       (5.5, 5.5, 0.1), (6, 5, 0.08)]):
        color = COLORS['ion'] if i == 2 else COLORS['mold']
        circle = Circle((x, y), size, facecolor=color, alpha=0.7)
        ax.add_patch(circle)

    # The ion with charge
    ion_pos = Circle((7, 5), 0.25, facecolor=COLORS['ion'], edgecolor='white', linewidth=1.5)
    ax.add_patch(ion_pos)
    ax.text(7.4, 5.3, '[M-H]$^-$', fontsize=7)

    ax.set_title('Ionization', fontsize=9, fontweight='bold')


def draw_stage_4_ms1(ax: plt.Axes, ion: IonState) -> None:
    """Stage 4: MS1 - mass spectrum with ion highlighted."""
    # Generate sample mass spectrum
    mz_base = np.array([100, 150, 180, ion.mz, 250, 300, 350])
    intensities = np.array([0.2, 0.35, 0.5, 1.0, 0.4, 0.25, 0.15])

    # Add isotopes around our ion
    isotopes = ion.mz + np.array([0, 1.003, 2.006])
    iso_int = np.array([1.0, 0.3, 0.1])

    # Plot spectrum
    for mz, i in zip(mz_base, intensities):
        color = COLORS['ion'] if abs(mz - ion.mz) < 1 else COLORS['flow']
        ax.bar(mz, i, width=2, color=color, alpha=0.7)

    # Isotope pattern
    for mz, i in zip(isotopes, iso_int):
        ax.bar(mz, i, width=1.5, color=COLORS['ion'], alpha=0.9)

    ax.set_xlabel('m/z')
    ax.set_ylabel('Rel. Int.')
    ax.set_title('MS1', fontsize=9, fontweight='bold')
    ax.set_xlim(50, 400)
    ax.set_ylim(0, 1.2)

    # Annotate our ion
    ax.annotate(f'{ion.mz:.1f}', xy=(ion.mz, 1.0), xytext=(ion.mz + 20, 1.1),
                fontsize=7, arrowprops=dict(arrowstyle='->', color=COLORS['ion']))


def draw_stage_5_sentropy(ax: plt.Axes, ion: IonState) -> None:
    """Stage 5: S-Entropy coordinates in 3D space."""
    # Create 3D effect with 2D projection
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Background grid representing S-entropy space
    for i in np.linspace(0.1, 0.9, 5):
        ax.axhline(y=i, color='gray', alpha=0.2, linewidth=0.5)
        ax.axvline(x=i, color='gray', alpha=0.2, linewidth=0.5)

    # S_e represented by color/size
    size = 200 + ion.s_e * 300

    # Plot ion position in (S_k, S_t) space
    ax.scatter([ion.s_k], [ion.s_t * 1000], s=size, c=COLORS['ion'],
               alpha=0.8, edgecolors='white', linewidths=1.5)

    # Add contours showing entropy landscape
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - ion.s_k)**2 + (Y - ion.s_t * 1000)**2) / 0.1)
    ax.contour(X, Y, Z, levels=5, colors=COLORS['mold'], alpha=0.3, linewidths=0.5)

    ax.set_xlabel(r'$S_k$')
    ax.set_ylabel(r'$S_t$')
    ax.set_title('S-Entropy', fontsize=9, fontweight='bold')


def draw_stage_6_partition(ax: plt.Axes, ion: IonState) -> None:
    """Stage 6: Partition coordinates (n, l, m, s) - orbital representation."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Draw orbital shells based on n
    for shell in range(1, min(ion.n + 1, 6)):  # Limit to 5 shells
        r = shell / max(ion.n, 1)
        alpha = min(0.3 + 0.1 * shell, 0.9)  # Cap alpha at 0.9
        circle = Circle((0, 0), r, fill=False, edgecolor=COLORS['mold'],
                        alpha=alpha, linewidth=0.5)
        ax.add_patch(circle)

    # Angular momentum visualization (l, m)
    theta = np.linspace(0, 2 * np.pi, 100)
    r_lobe = 0.8 * (1 + 0.5 * np.cos(ion.l * theta))
    x_lobe = r_lobe * np.cos(theta)
    y_lobe = r_lobe * np.sin(theta)
    ax.fill(x_lobe, y_lobe, color=COLORS['match'], alpha=0.3)
    ax.plot(x_lobe, y_lobe, color=COLORS['match'], linewidth=1)

    # Ion position within orbital
    angle = ion.m * np.pi / (ion.l + 1) if ion.l > 0 else 0
    r_ion = 0.6
    x_ion = r_ion * np.cos(angle)
    y_ion = r_ion * np.sin(angle)

    ion_marker = Circle((x_ion, y_ion), 0.12, facecolor=COLORS['ion'],
                        edgecolor='white', linewidth=1.5)
    ax.add_patch(ion_marker)

    # Spin indicator
    spin_arrow = '↑' if ion.spin > 0 else '↓'
    ax.text(x_ion + 0.15, y_ion + 0.15, spin_arrow, fontsize=10, color=COLORS['ion'])

    ax.axis('off')
    ax.set_title(f'Partition (n={ion.n}, l={ion.l})', fontsize=9, fontweight='bold')


def draw_stage_7_thermodynamics(ax: plt.Axes, ion: IonState) -> None:
    """Stage 7: Thermodynamic state in We-Re space."""
    # Calculate dimensionless numbers
    rho = 1000  # kg/m³
    mu = 0.001  # Pa·s
    r_m = ion.radius * 1e-3  # mm to m

    We = rho * ion.velocity**2 * r_m / ion.surface_tension
    Re = rho * ion.velocity * r_m / mu
    Oh = mu / np.sqrt(rho * ion.surface_tension * r_m)

    # Plot regime map
    We_range = np.logspace(-1, 3, 100)
    Re_range = np.logspace(-1, 5, 100)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Regime boundaries
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Fill regimes
    ax.fill_between([0.1, 1], [0.1, 0.1], [1, 1], alpha=0.1, color='blue', label='Stokes')
    ax.fill_between([1, 1000], [1, 1], [10000, 10000], alpha=0.1, color='green', label='Inertial')

    # Plot our ion
    ax.scatter([We], [Re], s=150, c=COLORS['ion'], edgecolors='white',
               linewidths=2, zorder=5)

    ax.set_xlabel('We')
    ax.set_ylabel('Re')
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, 10000)
    ax.set_title('Thermodynamics', fontsize=9, fontweight='bold')


def draw_stage_8_droplet(ax: plt.Axes, ion: IonState) -> None:
    """Stage 8: Droplet visualization - wave pattern encoding."""
    resolution = 128
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    # Wave pattern from droplet impact
    cx, cy = 0.5, 0.5
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Encode ion properties in wave
    wavelength = 0.1 + 0.05 * ion.s_k
    amplitude = ion.phase_coherence
    decay = 3.0 + ion.s_e

    wave = amplitude * np.cos(2 * np.pi * r / wavelength) * np.exp(-decay * r)

    # Add secondary impacts based on categorical state
    for i in range(min(3, ion.categorical_state)):
        angle = i * 2 * np.pi / 3
        cx2 = 0.5 + 0.25 * np.cos(angle)
        cy2 = 0.5 + 0.25 * np.sin(angle)
        r2 = np.sqrt((X - cx2)**2 + (Y - cy2)**2)
        wave += 0.3 * amplitude * np.cos(2 * np.pi * r2 / wavelength * 1.5) * np.exp(-decay * 1.5 * r2)

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('droplet',
                                              ['#2E4057', '#048A81', '#54C6EB', '#8EE3EF', '#F7F7F7'])

    ax.imshow(wave, cmap=cmap, aspect='equal', extent=[0, 1, 0, 1])
    ax.axis('off')
    ax.set_title('Droplet', fontsize=9, fontweight='bold')


def draw_flow_arrows(fig: plt.Figure, positions: List[Tuple[float, float, float, float]]) -> None:
    """Draw flow arrows between stages."""
    for i in range(len(positions) - 1):
        x1, y1, w1, h1 = positions[i]
        x2, y2, w2, h2 = positions[i + 1]

        # Arrow from right of panel i to left of panel i+1
        start_x = x1 + w1 + 0.01
        start_y = y1 + h1 / 2
        end_x = x2 - 0.01
        end_y = y2 + h2 / 2

        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->', mutation_scale=10,
            color=COLORS['flow'], linewidth=1.5,
            transform=fig.transFigure
        )
        fig.add_artist(arrow)


def create_ion_journey_panel(results_dir: Path, output_dir: Path) -> None:
    """Create the main ion journey visualization."""
    # Load data
    stages = load_pipeline_results(results_dir)
    ion = extract_sample_ion(stages)

    # Create figure with 2 rows, 4 columns
    fig = plt.figure(figsize=(12, 6))

    # Top row: physical stages
    ax1 = fig.add_subplot(2, 4, 1)
    draw_stage_1_injection(ax1, ion)

    ax2 = fig.add_subplot(2, 4, 2)
    draw_stage_2_chromatography(ax2, ion)

    ax3 = fig.add_subplot(2, 4, 3)
    draw_stage_3_ionization(ax3, ion)

    ax4 = fig.add_subplot(2, 4, 4)
    draw_stage_4_ms1(ax4, ion)

    # Bottom row: computational/encoding stages
    ax5 = fig.add_subplot(2, 4, 5)
    draw_stage_5_sentropy(ax5, ion)

    ax6 = fig.add_subplot(2, 4, 6)
    draw_stage_6_partition(ax6, ion)

    ax7 = fig.add_subplot(2, 4, 7)
    draw_stage_7_thermodynamics(ax7, ion)

    ax8 = fig.add_subplot(2, 4, 8)
    draw_stage_8_droplet(ax8, ion)

    # Add stage numbers
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], 1):
        ax.text(0.02, 0.98, f'{i}', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top',
                bbox=dict(boxstyle='circle', facecolor=COLORS['highlight'],
                         edgecolor='none', alpha=0.8))

    # Main title
    fig.suptitle('Ion Journey Through Pipeline Stages', fontsize=12, fontweight='bold', y=0.98)

    # Subtitle with ion info
    subtitle = f'm/z = {ion.mz:.2f}, RT = {ion.rt:.1f} min, Categorical State = {ion.categorical_state}'
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save
    fig.savefig(output_dir / 'ion_journey.png', dpi=300)
    fig.savefig(output_dir / 'ion_journey.pdf')
    plt.close(fig)
    print(f"Saved: ion_journey.png/pdf")


def create_compact_flow_diagram(output_dir: Path) -> None:
    """Create a compact flow diagram showing the template concept."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Flow stages
    stages = [
        (0.5, 'Sample'),
        (2, 'Chrom'),
        (3.5, 'Ion'),
        (5, 'MS1'),
        (6.5, 'S-Ent'),
        (8, 'Part'),
        (9.5, 'Drop'),
    ]

    # Draw stages
    for x, label in stages:
        # Mold (template) - gray circle
        mold = Circle((x, 2), 0.35, facecolor=COLORS['background'],
                      edgecolor=COLORS['mold'], linewidth=1.5)
        ax.add_patch(mold)

        # Ion passing through - if it matches
        ion = Circle((x, 2), 0.15, facecolor=COLORS['ion'], edgecolor='white', linewidth=1)
        ax.add_patch(ion)

        # Label below
        ax.text(x, 1.2, label, ha='center', fontsize=8, fontweight='bold')

        # Match indicator above
        ax.text(x, 2.6, '✓', ha='center', fontsize=10, color=COLORS['match'])

    # Flow arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.4
        x2 = stages[i + 1][0] - 0.4
        ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                   arrowprops=dict(arrowstyle='->', color=COLORS['flow'], lw=1.5))

    # Title
    ax.text(5, 0.5, 'Template-Based Analysis: Ion matches mold at each stage',
            ha='center', fontsize=10, style='italic')

    fig.savefig(output_dir / 'template_flow.png', dpi=300)
    fig.savefig(output_dir / 'template_flow.pdf')
    plt.close(fig)
    print(f"Saved: template_flow.png/pdf")


def main():
    """Generate ion journey visualization."""
    # Find results directory
    results_base = Path(__file__).parent.parent.parent.parent / 'pipeline_results'
    results_dirs = sorted(results_base.glob('H11_BD_A_neg_hilic_*'))

    if not results_dirs:
        print("No pipeline results found.")
        return

    results_dir = results_dirs[-1]
    print(f"Using results from: {results_dir}")

    # Output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating Ion Journey Visualization...")
    print("=" * 50)

    create_ion_journey_panel(results_dir, output_dir)
    create_compact_flow_diagram(output_dir)

    print("=" * 50)
    print(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
