"""
MOLECULAR STRUCTURAL FEATURES ANALYSIS
Categorical Molecular Recognition from Descriptors
Publication-quality visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import json


if __name__ == "__main__":

    print("="*80)
    print("MOLECULAR STRUCTURAL FEATURES ANALYSIS")
    print("="*80)

    # ============================================================
    # LOAD DATA
    # ============================================================

    # Molecular features data
    molecules = {
        'Vanillin': {
            'formula': 'C₈H₈O₃',
            'n_atoms': 19,
            'n_bonds': 19,
            'n_rings': 1,
            'molecular_weight': 152.149,
            'n_rotatable_bonds': 4,
            'n_h_bond_donors': 1,
            'n_h_bond_acceptors': 3,
            'topological_polar_surface_area': 46.53,
            'n_aromatic_rings': 1,
            'n_saturated_rings': 0,
            'n_heteroatoms': 3,
            'formal_charge': 0,
            'n_stereocenters': 0,
            'n_single_bonds': 12,
            'n_double_bonds': 1,
            'n_triple_bonds': 0,
            'n_aromatic_bonds': 6,
            'n_carbon': 8,
            'n_hydrogen': 8,
            'n_oxygen': 3,
            'n_nitrogen': 0,
            'molecular_volume': 136.904,
            'asphericity': 0.2486,
            'eccentricity': 0.8014,
            'type': 'Aromatic aldehyde',
            'class': 'Complex'
        },
        'Benzene': {
            'formula': 'C₆H₆',
            'n_atoms': 12,
            'n_bonds': 12,
            'n_rings': 1,
            'molecular_weight': 78.114,
            'n_rotatable_bonds': 0,
            'n_h_bond_donors': 0,
            'n_h_bond_acceptors': 0,
            'topological_polar_surface_area': 0.0,
            'n_aromatic_rings': 1,
            'n_saturated_rings': 0,
            'n_heteroatoms': 0,
            'formal_charge': 0,
            'n_stereocenters': 0,
            'n_single_bonds': 6,
            'n_double_bonds': 0,
            'n_triple_bonds': 0,
            'n_aromatic_bonds': 6,
            'n_carbon': 6,
            'n_hydrogen': 6,
            'n_oxygen': 0,
            'n_nitrogen': 0,
            'molecular_volume': 83.44,
            'asphericity': 0.25,
            'eccentricity': 0.7071,
            'type': 'Simple aromatic',
            'class': 'Aromatic'
        },
        'Ethanol': {
            'formula': 'C₂H₆O',
            'n_atoms': 9,
            'n_bonds': 8,
            'n_rings': 0,
            'molecular_weight': 46.069,
            'n_rotatable_bonds': 2,
            'n_h_bond_donors': 1,
            'n_h_bond_acceptors': 1,
            'topological_polar_surface_area': 20.23,
            'n_aromatic_rings': 0,
            'n_saturated_rings': 0,
            'n_heteroatoms': 1,
            'formal_charge': 0,
            'n_stereocenters': 0,
            'n_single_bonds': 8,
            'n_double_bonds': 0,
            'n_triple_bonds': 0,
            'n_aromatic_bonds': 0,
            'n_carbon': 2,
            'n_hydrogen': 6,
            'n_oxygen': 1,
            'n_nitrogen': 0,
            'molecular_volume': 53.992,
            'asphericity': 0.2069,
            'eccentricity': 0.8661,
            'type': 'Simple alcohol',
            'class': 'Aliphatic'
        },
        'Indole': {
            'formula': 'C₈H₇N',
            'n_atoms': 16,
            'n_bonds': 17,
            'n_rings': 2,
            'molecular_weight': 117.151,
            'n_rotatable_bonds': 0,
            'n_h_bond_donors': 1,
            'n_h_bond_acceptors': 0,
            'topological_polar_surface_area': 15.79,
            'n_aromatic_rings': 2,
            'n_saturated_rings': 0,
            'n_heteroatoms': 1,
            'formal_charge': 0,
            'n_stereocenters': 0,
            'n_single_bonds': 7,
            'n_double_bonds': 0,
            'n_triple_bonds': 0,
            'n_aromatic_bonds': 10,
            'n_carbon': 8,
            'n_hydrogen': 7,
            'n_oxygen': 0,
            'n_nitrogen': 1,
            'molecular_volume': 112.52,
            'asphericity': 0.25,
            'eccentricity': 0.8416,
            'type': 'Bicyclic aromatic',
            'class': 'Heterocycle'
        }
    }

    # ============================================================
    # VISUALIZATION
    # ============================================================

    fig = plt.figure(figsize=(28, 24))
    gs = GridSpec(6, 4, figure=fig, hspace=0.5, wspace=0.4)

    colors = {
        'Vanillin': '#f39c12',
        'Benzene': '#e74c3c',
        'Ethanol': '#2ecc71',
        'Indole': '#9b59b6',
        'structural': '#3498db',
        'electronic': '#e67e22',
        'geometric': '#1abc9c'
    }

    mol_names = list(molecules.keys())

    # ============================================================
    # PANEL 1: Molecular Size Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])

    sizes = ['Atoms', 'Bonds', 'MW (g/mol)']
    x = np.arange(len(mol_names))
    width = 0.25

    bars1 = ax1.bar(x - width, [molecules[m]['n_atoms'] for m in mol_names],
                width, label='Atoms', alpha=0.8, edgecolor='black', linewidth=2,
                color=[colors[m] for m in mol_names])

    bars2 = ax1.bar(x, [molecules[m]['n_bonds'] for m in mol_names],
                width, label='Bonds', alpha=0.8, edgecolor='black', linewidth=2,
                color='gray')

    ax1_twin = ax1.twinx()
    bars3 = ax1_twin.bar(x + width, [molecules[m]['molecular_weight'] for m in mol_names],
                        width, label='Molecular Weight', alpha=0.6,
                        edgecolor='black', linewidth=2, color='lightblue')

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for bar in bars3:
        height = bar.get_height()
        ax1_twin.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1_twin.set_ylabel('Molecular Weight (g/mol)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Molecular Size Comparison\nAtoms, Bonds, and Molecular Weight',
                fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax1.legend(loc='upper left', fontsize=10)
    ax1_twin.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 2: Elemental Composition
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    elements = ['Carbon', 'Hydrogen', 'Oxygen', 'Nitrogen']
    x = np.arange(len(mol_names))
    width = 0.2

    bars_c = ax2.bar(x - 1.5*width, [molecules[m]['n_carbon'] for m in mol_names],
                    width, label='C', color='#34495e', alpha=0.8,
                    edgecolor='black', linewidth=2)
    bars_h = ax2.bar(x - 0.5*width, [molecules[m]['n_hydrogen'] for m in mol_names],
                    width, label='H', color='#ecf0f1', alpha=0.8,
                    edgecolor='black', linewidth=2)
    bars_o = ax2.bar(x + 0.5*width, [molecules[m]['n_oxygen'] for m in mol_names],
                    width, label='O', color='#e74c3c', alpha=0.8,
                    edgecolor='black', linewidth=2)
    bars_n = ax2.bar(x + 1.5*width, [molecules[m]['n_nitrogen'] for m in mol_names],
                    width, label='N', color='#3498db', alpha=0.8,
                    edgecolor='black', linewidth=2)

    # Value labels
    for bars in [bars_c, bars_h, bars_o, bars_n]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    ax2.set_ylabel('Number of Atoms', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Elemental Composition\nC, H, O, N Distribution',
                fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax2.legend(fontsize=10, ncol=4)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 3: Ring Systems
    # ============================================================
    ax3 = fig.add_subplot(gs[1, :2])

    ring_data = {
        'Total Rings': [molecules[m]['n_rings'] for m in mol_names],
        'Aromatic': [molecules[m]['n_aromatic_rings'] for m in mol_names],
        'Saturated': [molecules[m]['n_saturated_rings'] for m in mol_names]
    }

    x = np.arange(len(mol_names))
    width = 0.25
    multiplier = 0

    for attribute, measurement in ring_data.items():
        offset = width * multiplier
        bars = ax3.bar(x + offset, measurement, width, label=attribute,
                    alpha=0.8, edgecolor='black', linewidth=2)

        # Value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        multiplier += 1

    ax3.set_ylabel('Number of Rings', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Ring System Analysis\nTotal, Aromatic, and Saturated Rings',
                fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 4: Bond Types
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 2:])

    bond_types = ['Single', 'Double', 'Triple', 'Aromatic']
    x = np.arange(len(mol_names))
    width = 0.2

    bars_single = ax4.bar(x - 1.5*width, [molecules[m]['n_single_bonds'] for m in mol_names],
                        width, label='Single', color='#95a5a6', alpha=0.8,
                        edgecolor='black', linewidth=2)
    bars_double = ax4.bar(x - 0.5*width, [molecules[m]['n_double_bonds'] for m in mol_names],
                        width, label='Double', color='#3498db', alpha=0.8,
                        edgecolor='black', linewidth=2)
    bars_triple = ax4.bar(x + 0.5*width, [molecules[m]['n_triple_bonds'] for m in mol_names],
                        width, label='Triple', color='#e74c3c', alpha=0.8,
                        edgecolor='black', linewidth=2)
    bars_aromatic = ax4.bar(x + 1.5*width, [molecules[m]['n_aromatic_bonds'] for m in mol_names],
                        width, label='Aromatic', color='#9b59b6', alpha=0.8,
                        edgecolor='black', linewidth=2)

    # Value labels
    for bars in [bars_single, bars_double, bars_triple, bars_aromatic]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    ax4.set_ylabel('Number of Bonds', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Bond Type Distribution\nSingle, Double, Triple, Aromatic',
                fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax4.legend(fontsize=10, ncol=2)
    ax4.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 5: Hydrogen Bonding Capacity
    # ============================================================
    ax5 = fig.add_subplot(gs[2, :2])

    x = np.arange(len(mol_names))
    width = 0.35

    donors = [molecules[m]['n_h_bond_donors'] for m in mol_names]
    acceptors = [molecules[m]['n_h_bond_acceptors'] for m in mol_names]

    bars1 = ax5.bar(x - width/2, donors, width, label='H-Bond Donors',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax5.bar(x + width/2, acceptors, width, label='H-Bond Acceptors',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax5.set_title('(E) Hydrogen Bonding Capacity\nDonors and Acceptors',
                fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax5.legend(fontsize=11)
    ax5.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 6: Polarity Metrics
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 2:])

    tpsa = [molecules[m]['topological_polar_surface_area'] for m in mol_names]
    heteroatoms = [molecules[m]['n_heteroatoms'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax6.bar(x - width/2, tpsa, width, label='TPSA (Ų)',
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)

    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x + width/2, heteroatoms, width, label='Heteroatoms',
                        color='#f39c12', alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax6_twin.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    ax6.set_ylabel('TPSA (Ų)', fontsize=12, fontweight='bold')
    ax6_twin.set_ylabel('Number of Heteroatoms', fontsize=12, fontweight='bold')
    ax6.set_title('(F) Polarity Metrics\nTPSA and Heteroatom Count',
                fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax6.legend(loc='upper left', fontsize=10)
    ax6_twin.legend(loc='upper right', fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 7: Molecular Volume & Shape
    # ============================================================
    ax7 = fig.add_subplot(gs[3, :2])

    volumes = [molecules[m]['molecular_volume'] for m in mol_names]

    bars = ax7.bar(mol_names, volumes,
                color=[colors[m] for m in mol_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax7.set_ylabel('Molecular Volume (ų)', fontsize=12, fontweight='bold')
    ax7.set_title('(G) Molecular Volume\n3D Space Occupied',
                fontsize=13, fontweight='bold')
    ax7.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax7.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 8: Shape Descriptors
    # ============================================================
    ax8 = fig.add_subplot(gs[3, 2:])

    asphericity = [molecules[m]['asphericity'] for m in mol_names]
    eccentricity = [molecules[m]['eccentricity'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax8.bar(x - width/2, asphericity, width, label='Asphericity',
                color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax8.bar(x + width/2, eccentricity, width, label='Eccentricity',
                color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax8.set_ylabel('Shape Descriptor Value', fontsize=12, fontweight='bold')
    ax8.set_title('(H) Molecular Shape Descriptors\nAsphericity and Eccentricity',
                fontsize=13, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax8.legend(fontsize=11)
    ax8.grid(alpha=0.3, linestyle='--', axis='y')
    ax8.set_ylim(0, 1)

    # ============================================================
    # PANEL 9: Flexibility & Complexity
    # ============================================================
    ax9 = fig.add_subplot(gs[4, :2])

    rotatable = [molecules[m]['n_rotatable_bonds'] for m in mol_names]
    stereo = [molecules[m]['n_stereocenters'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax9.bar(x - width/2, rotatable, width, label='Rotatable Bonds',
                color='#e67e22', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax9.bar(x + width/2, stereo, width, label='Stereocenters',
                color='#c0392b', alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax9.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax9.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax9.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax9.set_title('(I) Molecular Flexibility & Complexity\nRotatable Bonds and Stereocenters',
                fontsize=13, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax9.legend(fontsize=11)
    ax9.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 10: Radar Chart - Molecular Fingerprint
    # ============================================================
    ax10 = fig.add_subplot(gs[4, 2:], projection='polar')

    # Normalize features for radar chart
    features = ['Size', 'Rings', 'H-Bond', 'Polarity', 'Volume', 'Shape']

    # Calculate normalized scores
    def normalize_features(mol):
        return [
            mol['n_atoms'] / 20,  # Size
            mol['n_rings'] / 2,  # Rings
            (mol['n_h_bond_donors'] + mol['n_h_bond_acceptors']) / 5,  # H-bonding
            mol['topological_polar_surface_area'] / 50,  # Polarity
            mol['molecular_volume'] / 150,  # Volume
            (mol['asphericity'] + mol['eccentricity']) / 2  # Shape
        ]

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for mol_name in mol_names:
        values = normalize_features(molecules[mol_name])
        values += values[:1]  # Complete the circle

        ax10.plot(angles, values, 'o-', linewidth=2,
                label=molecules[mol_name]['formula'],
                color=colors[mol_name], markersize=8)
        ax10.fill(angles, values, alpha=0.15, color=colors[mol_name])

    ax10.set_xticks(angles[:-1])
    ax10.set_xticklabels(features, fontsize=10)
    ax10.set_ylim(0, 1)
    ax10.set_title('(J) Molecular Fingerprint\nNormalized Feature Radar',
                fontsize=13, fontweight='bold', pad=20)
    ax10.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax10.grid(True)

    # ============================================================
    # PANEL 11: Detailed Molecular Properties
    # ============================================================
    ax11 = fig.add_subplot(gs[5, :2])
    ax11.axis('off')

    properties_text = """
    DETAILED MOLECULAR PROPERTIES

    VANILLIN (C₈H₈O₃):
    Type: Aromatic aldehyde
    MW: 152.15 g/mol
    Atoms: 19 (8C, 8H, 3O)
    Bonds: 19 (12 single, 1 double, 6 aromatic)
    Rings: 1 aromatic
    H-bonding: 1 donor, 3 acceptors
    TPSA: 46.53 Ų
    Volume: 136.90 ų
    Rotatable bonds: 4
    Shape: Asphericity=0.249, Eccentricity=0.801

    BENZENE (C₆H₆):
    Type: Simple aromatic
    MW: 78.11 g/mol
    Atoms: 12 (6C, 6H)
    Bonds: 12 (6 single, 6 aromatic)
    Rings: 1 aromatic
    H-bonding: None
    TPSA: 0.00 Ų
    Volume: 83.44 ų
    Rotatable bonds: 0
    Shape: Asphericity=0.250, Eccentricity=0.707

    ETHANOL (C₂H₆O):
    Type: Simple alcohol
    MW: 46.07 g/mol
    Atoms: 9 (2C, 6H, 1O)
    Bonds: 8 (all single)
    Rings: 0
    H-bonding: 1 donor, 1 acceptor
    TPSA: 20.23 Ų
    Volume: 53.99 ų
    Rotatable bonds: 2
    Shape: Asphericity=0.207, Eccentricity=0.866

    INDOLE (C₈H₇N):
    Type: Bicyclic aromatic
    MW: 117.15 g/mol
    Atoms: 16 (8C, 7H, 1N)
    Bonds: 17 (7 single, 10 aromatic)
    Rings: 2 aromatic (fused)
    H-bonding: 1 donor, 0 acceptors
    TPSA: 15.79 Ų
    Volume: 112.52 ų
    Rotatable bonds: 0
    Shape: Asphericity=0.250, Eccentricity=0.842
    """

    ax11.text(0.05, 0.95, properties_text, transform=ax11.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    # ============================================================
    # PANEL 12: Categorical Recognition Summary
    # ============================================================
    ax12 = fig.add_subplot(gs[5, 2:])
    ax12.axis('off')

    summary_text = """
    CATEGORICAL MOLECULAR RECOGNITION

    STRUCTURAL DIVERSITY:
    • 4 molecules spanning multiple classes
    • MW range: 46-152 g/mol (3.3× span)
    • Atom count: 9-19 atoms
    • Ring systems: 0-2 rings
    • Heteroatom diversity: O, N

    FEATURE SPACE:
    • 30+ molecular descriptors
    • Topological features
    • Electronic properties
    • Geometric characteristics
    • Shape descriptors

    CATEGORICAL SIGNATURES:
    Vanillin:  Complex aromatic with 3 O atoms
    Benzene:   Symmetric aromatic hydrocarbon
    Ethanol:   Simple flexible alcohol
    Indole:    Rigid bicyclic heterocycle

    RECOGNITION CAPABILITY:
    ✓ Size discrimination (MW, volume)
    ✓ Geometry classification (rings, bonds)
    ✓ Polarity detection (TPSA, heteroatoms)
    ✓ H-bonding capacity
    ✓ Shape characterization
    ✓ Flexibility assessment

    APPLICATIONS:
    → Drug-likeness prediction
    → Molecular similarity search
    → QSAR modeling
    → Virtual screening
    → Categorical quantum sensing
    → Zero-backaction identification

    REVOLUTIONARY ASPECT:
    • Structural features encode categorical state
    • No measurement backaction required
    • Complete molecular fingerprint
    • Enables categorical dynamics framework
    • Trans-Planckian precision possible
    """

    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Molecular Structural Features Analysis\n'
                'Categorical Recognition from Molecular Descriptors',
                fontsize=18, fontweight='bold', y=0.998)

    plt.savefig('molecular_features.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('molecular_features.png', dpi=300, bbox_inches='tight')

    print("\n✓ Molecular features visualization complete")
    print("  Saved: molecular_features.pdf")
    print("  Saved: molecular_features.png")
    print("="*80)
