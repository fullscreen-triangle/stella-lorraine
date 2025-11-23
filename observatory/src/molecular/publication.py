"""
MOLECULAR PREDICTION COMPARISON
Vanillin vs CH predictions with categorical framework
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import json

if __name__ == "__main__":



    print("="*80)
    print("MOLECULAR PREDICTION COMPARISON")
    print("="*80)

    # Load prediction data
    with open('public/vanillin_prediction_20251122_082500.json', 'r') as f:
        vanillin = json.load(f)

    with open('public/ch_prediction_20251122_082928.json', 'r') as f:
        ch = json.load(f)

    print(f"\n✓ Loaded predictions:")
    print(f"  Vanillin: {vanillin['timestamp']}")
    print(f"  CH: {ch['timestamp']}")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Molecular Prediction Comparison: Vanillin vs CH\n'
                'Categorical Framework Predictions',
                fontsize=14, fontweight='bold')

    # Add your specific comparison plots based on the prediction data structure
    # (Will need to see the actual structure of these JSON files to complete)

    plt.tight_layout()
    plt.savefig('molecular_prediction_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('molecular_prediction_comparison.png', dpi=300, bbox_inches='tight')

    print("\n✓ Prediction comparison complete")
    print("="*80)


    """
    3D MOLECULAR GEOMETRY VISUALIZATION
    Interactive 3D plots of principal axes and molecular shapes
    """


    print("="*80)
    print("3D MOLECULAR GEOMETRY VISUALIZATION")
    print("="*80)

    # Load data
    with open('public/molecular_geometries.json', 'r') as f:
        geometry_data = json.load(f)

    # Create 3D visualization
    fig = plt.figure(figsize=(20, 16))

    mol_names = list(geometry_data['geometries'].keys())
    n_mols = len(mol_names)

    for i, mol_name in enumerate(mol_names, 1):
        mol = geometry_data['geometries'][mol_name]

        ax = fig.add_subplot(2, 2, i, projection='3d')

        # Center of mass
        com = mol['center_of_mass']
        ax.scatter([com[0]], [com[1]], [com[2]], s=200, c='red',
                marker='o', label='Center of Mass', edgecolor='black', linewidth=2)

        # Principal axes
        axes = mol['principal_axes']
        moments = mol['principal_moments']

        colors_axes = ['blue', 'green', 'orange']
        for j, (axis, moment) in enumerate(zip(axes, moments)):
            # Scale axis by moment
            scale = np.sqrt(moment) * 0.5
            endpoint = [com[k] + axis[k] * scale for k in range(3)]

            ax.plot([com[0], endpoint[0]],
                [com[1], endpoint[1]],
                [com[2], endpoint[2]],
                color=colors_axes[j], linewidth=3, alpha=0.8,
                label=f'Axis {j+1} (I={moment:.2f})')

        # Set labels
        ax.set_xlabel('X (Å)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (Å)', fontsize=10, fontweight='bold')
        ax.set_title(f'{mol_name}\nAsphericity: {mol["asphericity"]:.2e}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

        # Equal aspect ratio
        max_range = max([
            max(abs(com[0]), abs(endpoint[0])),
            max(abs(com[1]), abs(endpoint[1])),
            max(abs(com[2]), abs(endpoint[2]))
        ]) * 1.5

        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    fig.suptitle('3D Principal Axes Visualization\nMolecular Geometry and Symmetry',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('molecular_geometry_3d.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('molecular_geometry_3d.png', dpi=300, bbox_inches='tight')

    print("\n✓ 3D visualization complete")
    print("="*80)
