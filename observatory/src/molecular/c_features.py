"""
MOLECULAR FEATURES VISUALIZATION - FIXED
Categorical molecular recognition from descriptors
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, Wedge
import json



if __name__ == "__main__":
    print("="*80)
    print("MOLECULAR STRUCTURAL FEATURES ANALYSIS")
    print("="*80)

    # ============================================================
    # MOLECULAR DATA
    # ============================================================

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

    mol_names = list(molecules.keys())

    colors = {
        'Vanillin': '#f39c12',
        'Benzene': '#e74c3c',
        'Ethanol': '#2ecc71',
        'Indole': '#9b59b6'
    }

    print(f"\n✓ Loaded {len(molecules)} molecules")
    for name in mol_names:
        print(f"  {name}: {molecules[name]['formula']}")

    # ============================================================
    # VISUALIZATION
    # ============================================================

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ============================================================
    # PANEL 1: Molecular Size Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])

    x = np.arange(len(mol_names))
    width = 0.25

    atoms = [molecules[m]['n_atoms'] for m in mol_names]
    bonds = [molecules[m]['n_bonds'] for m in mol_names]

    bars1 = ax1.bar(x - width/2, atoms, width, label='Atoms',
                color=[colors[m] for m in mol_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    bars2 = ax1.bar(x + width/2, bonds, width, label='Bonds',
                color='gray', alpha=0.6, edgecolor='black', linewidth=2)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Molecular Size Comparison\nAtoms and Bonds',
                fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 2: Molecular Weight
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2])

    mw = [molecules[m]['molecular_weight'] for m in mol_names]

    bars = ax2.bar(mol_names, mw,
                color=[colors[m] for m in mol_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax2.set_ylabel('MW (g/mol)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Molecular Weight',
                fontsize=12, fontweight='bold')
    ax2.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax2.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 3: Elemental Composition
    # ============================================================
    ax3 = fig.add_subplot(gs[1, :])

    x = np.arange(len(mol_names))
    width = 0.2

    carbon = [molecules[m]['n_carbon'] for m in mol_names]
    hydrogen = [molecules[m]['n_hydrogen'] for m in mol_names]
    oxygen = [molecules[m]['n_oxygen'] for m in mol_names]
    nitrogen = [molecules[m]['n_nitrogen'] for m in mol_names]

    bars_c = ax3.bar(x - 1.5*width, carbon, width, label='C',
                    color='#34495e', alpha=0.8, edgecolor='black', linewidth=2)
    bars_h = ax3.bar(x - 0.5*width, hydrogen, width, label='H',
                    color='#ecf0f1', alpha=0.8, edgecolor='black', linewidth=2)
    bars_o = ax3.bar(x + 0.5*width, oxygen, width, label='O',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
    bars_n = ax3.bar(x + 1.5*width, nitrogen, width, label='N',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bars in [bars_c, bars_h, bars_o, bars_n]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    ax3.set_ylabel('Number of Atoms', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Elemental Composition',
                fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([molecules[m]['formula'] for m in mol_names])
    ax3.legend(fontsize=11, ncol=4)
    ax3.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 4: Ring Systems
    # ============================================================
    ax4 = fig.add_subplot(gs[2, 0])

    rings = [molecules[m]['n_rings'] for m in mol_names]
    aromatic = [molecules[m]['n_aromatic_rings'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax4.bar(x - width/2, rings, width, label='Total Rings',
                color='gray', alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = ax4.bar(x + width/2, aromatic, width, label='Aromatic',
                color='purple', alpha=0.8, edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Ring Systems',
                fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 5: H-Bonding Capacity
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 1])

    donors = [molecules[m]['n_h_bond_donors'] for m in mol_names]
    acceptors = [molecules[m]['n_h_bond_acceptors'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax5.bar(x - width/2, donors, width, label='Donors',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax5.bar(x + width/2, acceptors, width, label='Acceptors',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

    ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('(E) H-Bonding',
                fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 6: Polarity (TPSA)
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 2])

    tpsa = [molecules[m]['topological_polar_surface_area'] for m in mol_names]

    bars = ax6.bar(mol_names, tpsa,
                color=[colors[m] for m in mol_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax6.set_ylabel('TPSA (Ų)', fontsize=11, fontweight='bold')
    ax6.set_title('(F) Polarity',
                fontsize=12, fontweight='bold')
    ax6.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax6.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 7: Molecular Volume
    # ============================================================
    ax7 = fig.add_subplot(gs[3, 0])

    volume = [molecules[m]['molecular_volume'] for m in mol_names]

    bars = ax7.bar(mol_names, volume,
                color=[colors[m] for m in mol_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax7.set_ylabel('Volume (ų)', fontsize=11, fontweight='bold')
    ax7.set_title('(G) Molecular Volume',
                fontsize=12, fontweight='bold')
    ax7.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax7.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 8: Shape Descriptors
    # ============================================================
    ax8 = fig.add_subplot(gs[3, 1])

    asphericity = [molecules[m]['asphericity'] for m in mol_names]
    eccentricity = [molecules[m]['eccentricity'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax8.bar(x - width/2, asphericity, width, label='Asphericity',
                color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax8.bar(x + width/2, eccentricity, width, label='Eccentricity',
                color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=2)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax8.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax8.set_title('(H) Shape Descriptors',
                fontsize=12, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax8.legend(fontsize=9)
    ax8.grid(alpha=0.3, linestyle='--', axis='y')
    ax8.set_ylim(0, 1)

    # ============================================================
    # PANEL 9: Flexibility
    # ============================================================
    ax9 = fig.add_subplot(gs[3, 2])

    rotatable = [molecules[m]['n_rotatable_bonds'] for m in mol_names]

    bars = ax9.bar(mol_names, rotatable,
                color=[colors[m] for m in mol_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax9.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax9.set_ylabel('Rotatable Bonds', fontsize=11, fontweight='bold')
    ax9.set_title('(I) Flexibility',
                fontsize=12, fontweight='bold')
    ax9.set_xticklabels([molecules[m]['formula'] for m in mol_names],
                        rotation=45, ha='right')
    ax9.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 10: Radar Chart
    # ============================================================
    ax10 = fig.add_subplot(gs[4, :2], projection='polar')

    features = ['Size', 'Rings', 'H-Bond', 'Polarity', 'Volume', 'Shape']

    def normalize_features(mol):
        return [
            mol['n_atoms'] / 20,
            mol['n_rings'] / 2,
            (mol['n_h_bond_donors'] + mol['n_h_bond_acceptors']) / 5,
            mol['topological_polar_surface_area'] / 50,
            mol['molecular_volume'] / 150,
            (mol['asphericity'] + mol['eccentricity']) / 2
        ]

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    for mol_name in mol_names:
        values = normalize_features(molecules[mol_name])
        values += values[:1]

        ax10.plot(angles, values, 'o-', linewidth=2,
                label=molecules[mol_name]['formula'],
                color=colors[mol_name], markersize=6)
        ax10.fill(angles, values, alpha=0.15, color=colors[mol_name])

    ax10.set_xticks(angles[:-1])
    ax10.set_xticklabels(features, fontsize=10)
    ax10.set_ylim(0, 1)
    ax10.set_title('(J) Molecular Fingerprint Radar\nNormalized Features',
                fontsize=13, fontweight='bold', pad=20)
    ax10.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax10.grid(True)

    # ============================================================
    # PANEL 11: Summary Table
    # ============================================================
    ax11 = fig.add_subplot(gs[4, 2])
    ax11.axis('off')

    summary_text = f"""
    MOLECULAR SUMMARY

    VANILLIN:
    Formula: C₈H₈O₃
    MW: 152.1 g/mol
    Type: Aromatic aldehyde
    Rings: 1 aromatic
    H-bond: 1D, 3A
    TPSA: 46.5 Ų

    BENZENE:
    Formula: C₆H₆
    MW: 78.1 g/mol
    Type: Simple aromatic
    Rings: 1 aromatic
    H-bond: None
    TPSA: 0.0 Ų

    ETHANOL:
    Formula: C₂H₆O
    MW: 46.1 g/mol
    Type: Simple alcohol
    Rings: 0
    H-bond: 1D, 1A
    TPSA: 20.2 Ų

    INDOLE:
    Formula: C₈H₇N
    MW: 117.2 g/mol
    Type: Bicyclic aromatic
    Rings: 2 aromatic
    H-bond: 1D, 0A
    TPSA: 15.8 Ų

    KEY FEATURES:
    ✓ 4 diverse molecules
    ✓ 30+ descriptors
    ✓ Categorical signatures
    ✓ Zero backaction ID
    """

    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
            fontsize=7.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    # Main title
    fig.suptitle('Molecular Structural Features Analysis\n'
                'Categorical Recognition from Descriptors',
                fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('molecular_features.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('molecular_features.png', dpi=300, bbox_inches='tight')

    print("\n✓ Molecular features visualization complete")
    print("  Saved: molecular_features.pdf")
    print("  Saved: molecular_features.png")
    print("="*80)
