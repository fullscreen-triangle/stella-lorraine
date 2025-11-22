#!/usr/bin/env python3
"""
Categorical-Spacetime Mapping Visualization
===========================================

Visualizes the unification of physical distance and categorical separation.

Creates publication-quality multi-panel figure showing:
- Panel A: Categorical distance vs Physical distance equivalence
- Panel B: Molecular transitions in categorical-physical space
- Panel C: Light travel time scaling
- Panel D: Coupling constant interpretation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime

def load_results():
    """Load categorical spacetime mapping results"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_file = os.path.join(project_root, 'results',
                                'categorical_spacetime_mapping_20251115_042238.json')

    with open(results_file, 'r') as f:
        return json.load(f)

def create_publication_figure(results):
    """Create 4-panel publication figure"""

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })

    # Create figure with GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.07)

    # Extract data
    alpha_c = results['coupling_constant']['alpha_c']
    mappings = results['mappings']

    delta_C_values = [m['delta_C'] for m in mappings]
    d_equiv_values = [m['d_equivalent'] for m in mappings]
    t_light_values = [m['t_light_ns'] for m in mappings]
    mol_pairs = [f"{m['molecule_1']}\n→\n{m['molecule_2']}" for m in mappings]

    # Panel A: Categorical-Physical Distance Equivalence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(delta_C_values, d_equiv_values, s=150, c='#2E86AB',
                alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)

    # Fit line
    delta_C_range = np.linspace(0, max(delta_C_values)*1.1, 100)
    d_fit = alpha_c * delta_C_range
    ax1.plot(delta_C_range, d_fit, '--', color='#A23B72', linewidth=2,
             label=f'$d = \\alpha_c \\cdot \\Delta C$\n$\\alpha_c = {alpha_c:.2f}$ m/cat.unit',
             zorder=2)

    ax1.set_xlabel('Categorical Distance $\\Delta C$ [categorical units]', fontweight='bold')
    ax1.set_ylabel('Physical Distance $d$ [meters]', fontweight='bold')
    ax1.set_title('(A) Categorical-Physical Distance Equivalence',
                  fontweight='bold', loc='left')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(delta_C_values)*1.1)
    ax1.set_ylim(0, max(d_equiv_values)*1.1)

    # Panel B: Molecular Transitions in Dual Space
    ax2 = fig.add_subplot(gs[0, 1])

    # Create molecule position plot
    mol_names = ['C', 'CCO', 'c1ccccc1', 'c1ccc(O)cc1', 'c1ccc2ccccc2c1']
    mol_categorical = []
    mol_physical = []

    for mol in mol_names:
        # Find this molecule in mappings
        for m in mappings:
            if m['molecule_1'] == mol:
                mol_categorical.append(np.linalg.norm(m['C1']))
                mol_physical.append(0)  # Source at origin
                break
            elif m['molecule_2'] == mol:
                mol_categorical.append(np.linalg.norm(m['C2']))
                mol_physical.append(m['d_equivalent'])
                break

    # Plot transitions
    for i, m in enumerate(mappings):
        C1_norm = np.linalg.norm(m['C1'])
        C2_norm = np.linalg.norm(m['C2'])

        # Arrow from source to target
        ax2.arrow(C1_norm, 0, C2_norm - C1_norm, m['d_equivalent'],
                 head_width=1.5, head_length=10, fc=f'C{i}', ec='black',
                 alpha=0.6, linewidth=1.5, length_includes_head=True,
                 label=f"{m['molecule_1']} → {m['molecule_2']}")

    ax2.set_xlabel('Categorical Position $||C||$ [categorical units]', fontweight='bold')
    ax2.set_ylabel('Physical Position $d$ [meters]', fontweight='bold')
    ax2.set_title('(B) Molecular Transitions in Categorical-Physical Space',
                  fontweight='bold', loc='left')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Panel C: Light Travel Time Scaling
    ax3 = fig.add_subplot(gs[1, 0])

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(mappings)))
    bars = ax3.bar(range(len(mappings)), t_light_values, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    ax3.set_xticks(range(len(mappings)))
    ax3.set_xticklabels([f"{m['molecule_1']}\n→\n{m['molecule_2']}"
                         for m in mappings], fontsize=8)
    ax3.set_ylabel('Light Travel Time [ns]', fontweight='bold')
    ax3.set_title('(C) Light Travel Time for Categorical Separations',
                  fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels on bars
    for i, (bar, t_val) in enumerate(zip(bars, t_light_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{t_val:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Panel D: Coupling Constant Interpretation
    ax4 = fig.add_subplot(gs[1, 1])

    # Create categorical vs physical scatter with error regions
    ax4.scatter(delta_C_values, d_equiv_values, s=200, c='#2E86AB',
                alpha=0.7, edgecolors='black', linewidth=2, zorder=3,
                label='Measured mappings')

    # Show ±1σ uncertainty band
    delta_C_fine = np.linspace(0, max(delta_C_values)*1.1, 200)
    d_center = alpha_c * delta_C_fine

    # Estimate uncertainty from data scatter
    residuals = [d_equiv_values[i] - alpha_c * delta_C_values[i]
                 for i in range(len(mappings))]
    sigma = np.std(residuals)

    ax4.fill_between(delta_C_fine, d_center - sigma, d_center + sigma,
                     alpha=0.2, color='#A23B72',
                     label=f'$\\pm 1\\sigma$ = {sigma:.2f} m')
    ax4.plot(delta_C_fine, d_center, '-', color='#A23B72', linewidth=2.5,
            label=f'Coupling $\\alpha_c = {alpha_c:.2f}$ m/cat.unit', zorder=2)

    ax4.set_xlabel('Categorical Distance $\\Delta C$ [categorical units]', fontweight='bold')
    ax4.set_ylabel('Physical Distance $d$ [meters]', fontweight='bold')
    ax4.set_title('(D) Exchange Rate: Categorical ↔ Physical Space',
                  fontweight='bold', loc='left')
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Add text box with interpretation
    textstr = (f'Coupling constant $\\alpha_c$:\n'
               f'• {alpha_c:.2f} meters per categorical unit\n'
               f'• Bidirectional mapping\n'
               f'• Universal across molecules\n'
               f'• $\\sigma$ = {sigma:.2f} m')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5)
    ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Overall title
    fig.suptitle('Categorical-Spacetime Mapping: Unification of Physical and Categorical Distance',
                 fontsize=14, fontweight='bold', y=0.98)

    return fig

def save_figure(fig):
    """Save figure to results directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'categorical_spacetime_mapping_{timestamp}.png'
    filepath = os.path.join(results_dir, filename)

    fig.savefig(filepath, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {filepath}")

    plt.close(fig)
    return filepath

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" Categorical-Spacetime Mapping Visualization")
    print("="*70)

    # Load results
    print("\nLoading results...")
    results = load_results()

    # Create figure
    print("Creating publication figure...")
    fig = create_publication_figure(results)

    # Save
    print("Saving figure...")
    filepath = save_figure(fig)

    print("\n" + "="*70)
    print(" Visualization Complete")
    print("="*70)
    print(f"\nCoupling constant: α_c = {results['coupling_constant']['alpha_c']:.2f} m/cat.unit")
    print(f"Total mappings: {len(results['mappings'])}")
    print(f"\nFigure: {filepath}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
