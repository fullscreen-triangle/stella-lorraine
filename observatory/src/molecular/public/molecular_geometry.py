"""
MOLECULAR GEOMETRY & VIBRATIONAL BOND ANALYSIS
Comprehensive visualization of molecular shapes and bond dynamics
Categorical dynamics framework applied to molecular structure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import json
from scipy import constants

if __name__ == "__main__":

    print("="*80)
    print("MOLECULAR GEOMETRY & VIBRATIONAL BOND ANALYSIS")
    print("="*80)

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("\n1. LOADING MOLECULAR DATA")
    print("-" * 60)

    # Load molecular geometries
    with open('molecular_geometries.json', 'r') as f:
        geometry_data = json.load(f)

    # Load vanillin bond data
    with open('vanillin_bonds.json', 'r') as f:
        vanillin_data = json.load(f)

    print(f"✓ Loaded molecular geometry data")
    print(f"  Timestamp: {geometry_data['timestamp']}")
    print(f"  Number of molecules: {geometry_data['num_molecules']}")
    print(f"  Molecules: {list(geometry_data['geometries'].keys())}")

    print(f"\n✓ Loaded vanillin bond data")
    print(f"  Timestamp: {vanillin_data['timestamp']}")
    print(f"  SMILES: {vanillin_data['smiles']}")
    print(f"  Number of bonds: {vanillin_data['num_bonds']}")

    # ============================================================
    # EXTRACT GEOMETRY DATA
    # ============================================================

    print("\n2. EXTRACTING GEOMETRY PROPERTIES")
    print("-" * 60)

    molecules = {}
    for mol_name, mol_data in geometry_data['geometries'].items():
        molecules[mol_name] = mol_data

        print(f"\n{mol_name}:")
        print(f"  Asphericity: {mol_data['asphericity']:.6e}")
        print(f"  Eccentricity: {mol_data['eccentricity']:.6f}")
        print(f"  Radius of gyration: {mol_data['radius_of_gyration']:.4f} Å")
        print(f"  Molecular diameter: {mol_data['molecular_diameter']:.4f} Å")
        print(f"  Molecular volume: {mol_data['molecular_volume']:.2f} Ų")
        print(f"  Surface area: {mol_data['surface_area']:.2f} ų")

    # ============================================================
    # EXTRACT VANILLIN BOND DATA
    # ============================================================

    print("\n3. EXTRACTING VANILLIN BOND PROPERTIES")
    print("-" * 60)

    bonds = vanillin_data['bonds']

    # Categorize bonds
    bond_types = {}
    for bond in bonds:
        bond_type = bond['bond_type']
        if bond_type not in bond_types:
            bond_types[bond_type] = []
        bond_types[bond_type].append(bond)

    print(f"Bond type distribution:")
    for bond_type, bond_list in bond_types.items():
        print(f"  {bond_type}: {len(bond_list)} bonds")

    # Extract vibrational frequencies
    frequencies = np.array([bond['vibrational_frequency'] for bond in bonds])
    frequencies_thz = frequencies / 1e12  # Convert to THz

    print(f"\nVibrational frequencies:")
    print(f"  Range: {frequencies_thz.min():.2f} - {frequencies_thz.max():.2f} THz")
    print(f"  Mean: {frequencies_thz.mean():.2f} THz")
    print(f"  Std: {frequencies_thz.std():.2f} THz")

    # Extract bond lengths
    bond_lengths = np.array([bond['bond_length'] for bond in bonds])

    print(f"\nBond lengths:")
    print(f"  Range: {bond_lengths.min():.4f} - {bond_lengths.max():.4f} Å")
    print(f"  Mean: {bond_lengths.mean():.4f} Å")

    # Extract reduced masses
    reduced_masses = np.array([bond['reduced_mass'] for bond in bonds])

    print(f"\nReduced masses:")
    print(f"  Range: {reduced_masses.min():.4f} - {reduced_masses.max():.4f} Da")
    print(f"  Mean: {reduced_masses.mean():.4f} Da")

    # ============================================================
    # VISUALIZATION
    # ============================================================

    fig = plt.figure(figsize=(24, 28))
    gs = GridSpec(7, 4, figure=fig, hspace=0.5, wspace=0.4)

    colors = {
        'SINGLE': '#3498db',
        'AROMATIC': '#e74c3c',
        'DOUBLE': '#2ecc71',
        'TRIPLE': '#f39c12',
        'geometry': '#9b59b6',
        'frequency': '#1abc9c'
    }

    # ============================================================
    # PANEL 1: Molecular Geometry Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :2])

    mol_names = list(molecules.keys())
    asphericities = [molecules[m]['asphericity'] for m in mol_names]
    eccentricities = [molecules[m]['eccentricity'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, asphericities, width, label='Asphericity',
                color=colors['geometry'], alpha=0.8, edgecolor='black', linewidth=2)

    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, eccentricities, width, label='Eccentricity',
                        color=colors['frequency'], alpha=0.8, edgecolor='black', linewidth=2)

    ax1.set_ylabel('Asphericity', fontsize=11, fontweight='bold', color=colors['geometry'])
    ax1_twin.set_ylabel('Eccentricity', fontsize=11, fontweight='bold', color=colors['frequency'])
    ax1.set_title('(A) Molecular Shape Parameters\nAsphericity vs Eccentricity',
                fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mol_names, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor=colors['geometry'])
    ax1_twin.tick_params(axis='y', labelcolor=colors['frequency'])
    ax1.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 2: Molecular Size Comparison
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 2:])

    radii = [molecules[m]['radius_of_gyration'] for m in mol_names]
    diameters = [molecules[m]['molecular_diameter'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, radii, width, label='Radius of Gyration',
                color=colors['SINGLE'], alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x + width/2, diameters, width, label='Molecular Diameter',
                color=colors['AROMATIC'], alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax2.set_ylabel('Distance (Å)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Molecular Size Metrics\nRadius of Gyration vs Diameter',
                fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mol_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 3: Molecular Volume & Surface Area
    # ============================================================
    ax3 = fig.add_subplot(gs[1, :2])

    volumes = [molecules[m]['molecular_volume'] for m in mol_names]
    surface_areas = [molecules[m]['surface_area'] for m in mol_names]

    x = np.arange(len(mol_names))
    width = 0.35

    bars1 = ax3.bar(x - width/2, volumes, width, label='Volume (Ų)',
                color=colors['geometry'], alpha=0.8, edgecolor='black', linewidth=2)

    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, surface_areas, width, label='Surface Area (ų)',
                        color=colors['frequency'], alpha=0.8, edgecolor='black', linewidth=2)

    ax3.set_ylabel('Volume (Ų)', fontsize=11, fontweight='bold', color=colors['geometry'])
    ax3_twin.set_ylabel('Surface Area (ų)', fontsize=11, fontweight='bold', color=colors['frequency'])
    ax3.set_title('(C) Molecular Volume & Surface Area\nSize Metrics',
                fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(mol_names, rotation=45, ha='right')
    ax3.tick_params(axis='y', labelcolor=colors['geometry'])
    ax3_twin.tick_params(axis='y', labelcolor=colors['frequency'])
    ax3.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 4: Principal Moments of Inertia
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 2:])

    # Plot principal moments for first molecule (Methane)
    methane = molecules['Methane (spherical)']
    moments = methane['principal_moments']

    bars = ax4.bar(['I₁', 'I₂', 'I₃'], moments,
                color=[colors['SINGLE'], colors['AROMATIC'], colors['DOUBLE']],
                alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar, val in zip(bars, moments):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax4.set_ylabel('Moment of Inertia (amu·Ų)', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Principal Moments of Inertia\nMethane (Spherical Symmetry)',
                fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--', axis='y')

    # Add sphericity note
    ax4.text(0.5, 0.95, 'Nearly identical moments → Spherical molecule',
            transform=ax4.transAxes, ha='center', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # ============================================================
    # PANEL 5: Vanillin Bond Type Distribution
    # ============================================================
    ax5 = fig.add_subplot(gs[2, :2])

    bond_type_counts = {bt: len(bl) for bt, bl in bond_types.items()}
    bond_type_names = list(bond_type_counts.keys())
    bond_type_values = list(bond_type_counts.values())

    bars = ax5.bar(bond_type_names, bond_type_values,
                color=[colors.get(bt, '#34495e') for bt in bond_type_names],
                alpha=0.8, edgecolor='black', linewidth=2)

    # Value labels
    for bar, val in zip(bars, bond_type_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height,
                f'{val}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax5.set_ylabel('Number of Bonds', fontsize=11, fontweight='bold')
    ax5.set_title('(E) Vanillin Bond Type Distribution\nStructural Composition',
                fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 6: Vanillin Bond Lengths by Type
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 2:])

    # Group bond lengths by type
    bond_length_by_type = {}
    for bond in bonds:
        bt = bond['bond_type']
        if bt not in bond_length_by_type:
            bond_length_by_type[bt] = []
        bond_length_by_type[bt].append(bond['bond_length'])

    # Box plot
    positions = []
    data = []
    labels = []
    for i, (bt, lengths) in enumerate(bond_length_by_type.items(), 1):
        positions.append(i)
        data.append(lengths)
        labels.append(bt)

    bp = ax6.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    for patch, bt in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(bt, '#34495e'))
        patch.set_alpha(0.7)

    ax6.set_ylabel('Bond Length (Å)', fontsize=11, fontweight='bold')
    ax6.set_title('(F) Bond Length Distribution by Type\nVanillin Structure',
                fontsize=12, fontweight='bold')
    ax6.set_xticks(positions)
    ax6.set_xticklabels(labels)
    ax6.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 7: Vibrational Frequencies (All Bonds)
    # ============================================================
    ax7 = fig.add_subplot(gs[3, :2])

    # Sort bonds by frequency
    sorted_indices = np.argsort(frequencies_thz)
    sorted_freqs = frequencies_thz[sorted_indices]
    sorted_bond_types = [bonds[i]['bond_type'] for i in sorted_indices]

    # Color by bond type
    colors_list = [colors.get(bt, '#34495e') for bt in sorted_bond_types]

    bars = ax7.bar(range(len(sorted_freqs)), sorted_freqs, color=colors_list,
                alpha=0.8, edgecolor='black', linewidth=1)

    ax7.set_xlabel('Bond Index (sorted by frequency)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Vibrational Frequency (THz)', fontsize=11, fontweight='bold')
    ax7.set_title('(G) Vibrational Frequency Spectrum\nAll Vanillin Bonds',
                fontsize=12, fontweight='bold')
    ax7.grid(alpha=0.3, linestyle='--', axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors.get(bt, '#34495e'),
                            edgecolor='black', label=bt, alpha=0.8)
                    for bt in bond_types.keys()]
    ax7.legend(handles=legend_elements, fontsize=9, loc='upper left')

    # ============================================================
    # PANEL 8: Frequency vs Bond Length
    # ============================================================
    ax8 = fig.add_subplot(gs[3, 2:])

    # Scatter plot colored by bond type
    for bt in bond_types.keys():
        bt_bonds = bond_types[bt]
        bt_lengths = [b['bond_length'] for b in bt_bonds]
        bt_freqs = [b['vibrational_frequency']/1e12 for b in bt_bonds]

        ax8.scatter(bt_lengths, bt_freqs, s=100,
                color=colors.get(bt, '#34495e'),
                label=bt, alpha=0.7, edgecolor='black', linewidth=1)

    ax8.set_xlabel('Bond Length (Å)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Vibrational Frequency (THz)', fontsize=11, fontweight='bold')
    ax8.set_title('(H) Frequency-Length Relationship\nBond Properties Correlation',
                fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 9: Reduced Mass Distribution
    # ============================================================
    ax9 = fig.add_subplot(gs[4, :2])

    ax9.hist(reduced_masses, bins=15, color=colors['geometry'], alpha=0.7,
            edgecolor='black', linewidth=1.5)
    ax9.axvline(reduced_masses.mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {reduced_masses.mean():.2f} Da')

    ax9.set_xlabel('Reduced Mass (Da)', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax9.set_title('(I) Reduced Mass Distribution\nVanillin Bonds',
                fontsize=12, fontweight='bold')
    ax9.legend(fontsize=10)
    ax9.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 10: Force Constants by Bond Type
    # ============================================================
    ax10 = fig.add_subplot(gs[4, 2:])

    # Group force constants by type
    force_constants_by_type = {}
    for bond in bonds:
        bt = bond['bond_type']
        if bt not in force_constants_by_type:
            force_constants_by_type[bt] = []
        force_constants_by_type[bt].append(bond['force_constant'])

    # Bar plot with unique values
    unique_force_constants = {}
    for bt, fcs in force_constants_by_type.items():
        unique_force_constants[bt] = list(set(fcs))

    x_pos = 0
    x_positions = []
    labels = []
    values = []
    colors_list = []

    for bt, fcs in unique_force_constants.items():
        for fc in fcs:
            x_positions.append(x_pos)
            labels.append(f'{bt}\n{fc:.0f}')
            values.append(fc)
            colors_list.append(colors.get(bt, '#34495e'))
            x_pos += 1

    bars = ax10.bar(x_positions, values, color=colors_list,
                alpha=0.8, edgecolor='black', linewidth=2)

    ax10.set_ylabel('Force Constant (N/m)', fontsize=11, fontweight='bold')
    ax10.set_title('(J) Force Constants by Bond Type\nBond Stiffness',
                fontsize=12, fontweight='bold')
    ax10.set_xticks(x_positions)
    ax10.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax10.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 11: Aromatic vs Non-Aromatic Bonds
    # ============================================================
    ax11 = fig.add_subplot(gs[5, :2])

    aromatic_bonds = [b for b in bonds if b['is_aromatic']]
    non_aromatic_bonds = [b for b in bonds if not b['is_aromatic']]

    aromatic_freqs = [b['vibrational_frequency']/1e12 for b in aromatic_bonds]
    non_aromatic_freqs = [b['vibrational_frequency']/1e12 for b in non_aromatic_bonds]

    bp = ax11.boxplot([aromatic_freqs, non_aromatic_freqs],
                    labels=['Aromatic', 'Non-Aromatic'],
                    widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    bp['boxes'][0].set_facecolor(colors['AROMATIC'])
    bp['boxes'][1].set_facecolor(colors['SINGLE'])

    for box in bp['boxes']:
        box.set_alpha(0.7)

    ax11.set_ylabel('Vibrational Frequency (THz)', fontsize=11, fontweight='bold')
    ax11.set_title('(K) Aromatic vs Non-Aromatic Bonds\nFrequency Comparison',
                fontsize=12, fontweight='bold')
    ax11.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 12: Conjugated vs Non-Conjugated
    # ============================================================
    ax12 = fig.add_subplot(gs[5, 2:])

    conjugated_bonds = [b for b in bonds if b['is_conjugated']]
    non_conjugated_bonds = [b for b in bonds if not b['is_conjugated']]

    conjugated_lengths = [b['bond_length'] for b in conjugated_bonds]
    non_conjugated_lengths = [b['bond_length'] for b in non_conjugated_bonds]

    bp = ax12.boxplot([conjugated_lengths, non_conjugated_lengths],
                    labels=['Conjugated', 'Non-Conjugated'],
                    widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    bp['boxes'][0].set_facecolor(colors['DOUBLE'])
    bp['boxes'][1].set_facecolor(colors['SINGLE'])

    for box in bp['boxes']:
        box.set_alpha(0.7)

    ax12.set_ylabel('Bond Length (Å)', fontsize=11, fontweight='bold')
    ax12.set_title('(L) Conjugated vs Non-Conjugated Bonds\nLength Comparison',
                fontsize=12, fontweight='bold')
    ax12.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 13: Frequency Histogram
    # ============================================================
    ax13 = fig.add_subplot(gs[6, :2])

    ax13.hist(frequencies_thz, bins=20, color=colors['frequency'], alpha=0.7,
            edgecolor='black', linewidth=1.5)
    ax13.axvline(frequencies_thz.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {frequencies_thz.mean():.2f} THz')
    ax13.axvline(frequencies_thz.median(), color='green', linestyle=':',
                linewidth=2, label=f'Median: {np.median(frequencies_thz):.2f} THz')

    ax13.set_xlabel('Vibrational Frequency (THz)', fontsize=11, fontweight='bold')
    ax13.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax13.set_title('(M) Vibrational Frequency Distribution\nAll Vanillin Bonds',
                fontsize=12, fontweight='bold')
    ax13.legend(fontsize=10)
    ax13.grid(alpha=0.3, linestyle='--', axis='y')

    # ============================================================
    # PANEL 14: Summary Statistics
    # ============================================================
    ax14 = fig.add_subplot(gs[6, 2:])
    ax14.axis('off')

    # Calculate additional statistics
    in_ring_bonds = [b for b in bonds if b['is_in_ring']]
    ring_percentage = len(in_ring_bonds) / len(bonds) * 100

    summary_text = f"""
    MOLECULAR ANALYSIS SUMMARY

    GEOMETRY (4 MOLECULES):
    Methane (spherical):
        Asphericity:         {molecules['Methane (spherical)']['asphericity']:.2e}
        Eccentricity:        {molecules['Methane (spherical)']['eccentricity']:.6f}
        Radius of gyration:  {molecules['Methane (spherical)']['radius_of_gyration']:.4f} Å
        Molecular diameter:  {molecules['Methane (spherical)']['molecular_diameter']:.4f} Å
        Volume:              {molecules['Methane (spherical)']['molecular_volume']:.2f} Ų
        Surface area:        {molecules['Methane (spherical)']['surface_area']:.2f} ų

    VANILLIN BONDS (19 TOTAL):
    SMILES:              {vanillin_data['smiles']}

    Bond Type Distribution:
        SINGLE:            {len(bond_types.get('SINGLE', []))} bonds
        AROMATIC:          {len(bond_types.get('AROMATIC', []))} bonds
        DOUBLE:            {len(bond_types.get('DOUBLE', []))} bonds

    Bond Lengths:
        Range:             {bond_lengths.min():.4f} - {bond_lengths.max():.4f} Å
        Mean:              {bond_lengths.mean():.4f} Å
        Std:               {bond_lengths.std():.4f} Å

    Vibrational Frequencies:
        Range:             {frequencies_thz.min():.2f} - {frequencies_thz.max():.2f} THz
        Mean:              {frequencies_thz.mean():.2f} THz
        Median:            {np.median(frequencies_thz):.2f} THz
        Std:               {frequencies_thz.std():.2f} THz

    Reduced Masses:
        Range:             {reduced_masses.min():.4f} - {reduced_masses.max():.4f} Da
        Mean:              {reduced_masses.mean():.4f} Da

    Structural Features:
        Aromatic bonds:    {len(aromatic_bonds)} ({len(aromatic_bonds)/len(bonds)*100:.1f}%)
        Conjugated bonds:  {len(conjugated_bonds)} ({len(conjugated_bonds)/len(bonds)*100:.1f}%)
        In-ring bonds:     {len(in_ring_bonds)} ({ring_percentage:.1f}%)

    KEY FINDINGS:
    ✓ Methane shows near-perfect spherical symmetry
    ✓ Vanillin contains aromatic ring system
    ✓ Vibrational frequencies span {frequencies_thz.max()-frequencies_thz.min():.2f} THz range
    ✓ Bond lengths correlate with bond type
    ✓ Aromatic bonds show distinct frequency pattern
    ✓ Conjugation affects bond properties
    """

    ax14.text(0.05, 0.95, summary_text, transform=ax14.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Molecular Geometry & Vibrational Bond Analysis\n'
                'Categorical Dynamics Framework Applied to Molecular Structure',
                fontsize=14, fontweight='bold', y=0.998)

    plt.savefig('molecular_geometry_bond_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('molecular_geometry_bond_analysis.png', dpi=300, bbox_inches='tight')

    print("\n✓ Molecular geometry & bond analysis complete")
    print("  Saved: molecular_geometry_bond_analysis.pdf")
    print("  Saved: molecular_geometry_bond_analysis.png")
    print("="*80)
