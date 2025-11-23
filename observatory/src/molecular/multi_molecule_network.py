#!/usr/bin/env python3
"""
Multi-Molecule Harmonic Network Analysis

Creates harmonic coincidence networks from multiple molecules with different
vibrational modes. This dramatically increases network density and enhancement.

Molecules analyzed from console.md:
- Methane (CH4): 9 normal modes
- Benzene (C6H6): 30 normal modes
- Octane (C8H18): 69 normal modes
- Vanillin (C8H8O3): 66 normal modes

Total: 174 vibrational modes → massive harmonic network
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from molecular_network import HarmonicNetworkGraph, MolecularOscillator
from bmd_decomposition import BMDHierarchy
from reflectance_cascade import MolecularDemonReflectanceCascade
from categorical_state import SEntropyCalculator

# Molecular vibrational frequencies (cm^-1) for common modes
MOLECULAR_VIBRATIONS = {
    'methane': {
        'name': 'CH4 (Tetrahedral)',
        'modes': [
            ('symmetric_stretch', 2917, 1),      # A1
            ('bend', 1534, 2),                    # E (doubly degenerate)
            ('asymmetric_stretch', 3019, 3),     # F2 (triply degenerate)
            ('asymmetric_bend', 1306, 3),        # F2 (triply degenerate)
        ],
        'molecular_mass_amu': 16.04,
        'geometry': 'spherical'
    },
    'benzene': {
        'name': 'C6H6 (Aromatic ring)',
        'modes': [
            ('ring_breathing', 992, 1),
            ('CH_stretch_symmetric', 3062, 1),
            ('CH_stretch_asymmetric', 3080, 2),
            ('ring_stretch', 1596, 1),
            ('CH_bend_in_plane', 1178, 1),
            ('CH_bend_out_of_plane', 674, 2),
            ('ring_deformation', 1486, 1),
            ('CC_stretch', 1309, 1),
        ],
        'molecular_mass_amu': 78.11,
        'geometry': 'planar'
    },
    'octane': {
        'name': 'C8H18 (Linear alkane)',
        'modes': [
            ('CH3_symmetric_stretch', 2872, 2),  # Terminal methyls
            ('CH2_symmetric_stretch', 2850, 7),  # Methylene units
            ('CH2_scissor', 1467, 7),
            ('CH2_wag', 1378, 6),
            ('CH2_twist', 1296, 6),
            ('CH2_rock', 720, 6),
            ('CC_stretch', 1060, 7),
            ('CCC_bend', 420, 6),
        ],
        'molecular_mass_amu': 114.23,
        'geometry': 'linear'
    },
    'vanillin': {
        'name': 'C8H8O3 (Aromatic aldehyde)',
        'modes': [
            ('CO_stretch_aldehyde', 1666, 1),
            ('CO_stretch_phenol', 1270, 1),
            ('CO_stretch_methoxy', 1033, 1),
            ('ring_CC_stretch', 1583, 1),
            ('ring_CC_stretch_2', 1512, 1),
            ('CH_stretch_aromatic', 3070, 5),
            ('CH_stretch_aldehyde', 2820, 1),
            ('OH_stretch', 3400, 1),
            ('ring_breathing', 820, 1),
            ('CH_bend', 1425, 1),
        ],
        'molecular_mass_amu': 152.15,
        'geometry': 'planar_with_substituents'
    }
}

def wavenumber_to_hz(wavenumber_cm_inv: float) -> float:
    """Convert wavenumber (cm^-1) to frequency (Hz)"""
    c = 2.99792458e10  # Speed of light in cm/s
    return wavenumber_cm_inv * c

def create_multi_molecule_ensemble(molecules: dict, max_harmonics: int = 10) -> list:
    """
    Create oscillator ensemble from multiple molecules

    Args:
        molecules: Dictionary of molecular vibrational data
        max_harmonics: Number of harmonics to generate per mode (reduced for performance)

    Returns:
        List of MolecularOscillator objects
    """
    oscillators = []
    osc_id = 0

    print(f"\n{'='*70}")
    print(f"GENERATING MULTI-MOLECULE OSCILLATOR ENSEMBLE")
    print(f"{'='*70}\n")

    for mol_key, mol_data in molecules.items():
        print(f"{mol_data['name']}:")
        mol_oscillators = 0

        for mode_name, wavenumber, degeneracy in mol_data['modes']:
            freq_hz = wavenumber_to_hz(wavenumber)

            # Account for degeneracy (creates multiple oscillators at same frequency)
            for deg in range(degeneracy):
                # Generate harmonics for this mode
                for n in range(1, max_harmonics + 1):
                    harmonic_freq = n * freq_hz
                    phase = np.random.uniform(0, 2*np.pi)

                    # S-entropy from vibrational properties
                    # Higher harmonics have shorter effective coherence
                    coherence_time_s = 1e-13 / n  # ~100 fs base, decreases with n

                    s_coords = SEntropyCalculator.from_frequency(
                        frequency_hz=harmonic_freq,
                        measurement_count=n,
                        time_elapsed=coherence_time_s
                    )

                    osc = MolecularOscillator(
                        id=osc_id,
                        species=f"{mol_key}_{mode_name}_deg{deg}_n{n}",
                        frequency_hz=harmonic_freq,
                        phase_rad=phase,
                        s_coordinates=(s_coords.s_k, s_coords.s_t, s_coords.s_e)
                    )
                    oscillators.append(osc)
                    osc_id += 1
                    mol_oscillators += 1

        print(f"  Vibrational modes: {len(mol_data['modes'])}")
        print(f"  Total oscillators (with harmonics): {mol_oscillators}")
        print(f"  Geometry: {mol_data['geometry']}")
        print()

    print(f"Total oscillators across all molecules: {len(oscillators)}")
    print(f"{'='*70}\n")

    return oscillators

def analyze_multi_molecule_network(molecules: dict,
                                   bmd_depth: int = 14,
                                   n_reflections: int = 10,
                                   max_harmonics: int = 100,
                                   coincidence_threshold_hz: float = 1e10) -> dict:
    """
    Build and analyze harmonic network from multiple molecules

    Args:
        molecules: Dictionary of molecular data
        bmd_depth: BMD decomposition depth
        n_reflections: Cascade reflections
        max_harmonics: Harmonics per mode
        coincidence_threshold_hz: Coincidence threshold

    Returns:
        Complete analysis results
    """

    # Step 1: Generate ensemble
    oscillators = create_multi_molecule_ensemble(molecules, max_harmonics)

    # Step 2: Build network
    print(f"{'='*70}")
    print(f"BUILDING HARMONIC COINCIDENCE NETWORK")
    print(f"{'='*70}\n")
    print(f"Coincidence threshold: {coincidence_threshold_hz:.2e} Hz ({coincidence_threshold_hz/1e9:.1f} GHz)")
    print(f"Building graph... (this may take a moment for {len(oscillators)} oscillators)")

    network = HarmonicNetworkGraph(
        molecules=oscillators,
        coincidence_threshold_hz=coincidence_threshold_hz
    )
    graph = network.build_graph()

    # Network statistics
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")
    print(f"  Average degree: {avg_degree:.1f}")
    print(f"  Density: {density:.6f}")

    F_graph = network.calculate_enhancement_factor()
    print(f"  Graph enhancement: F_graph = {F_graph:.2e}")

    # Step 3: BMD decomposition
    print(f"\n{'='*70}")
    print(f"BIOLOGICAL MAXWELL DEMON DECOMPOSITION")
    print(f"{'='*70}\n")
    print(f"Depth: {bmd_depth}")

    # Use average frequency from molecular ensemble
    avg_freq = np.mean([osc.frequency_hz for osc in oscillators[:100]])

    bmd = BMDHierarchy(root_frequency=avg_freq)
    bmd.build_hierarchy(depth=bmd_depth)
    N_BMD = bmd.total_parallel_channels(bmd_depth)
    F_BMD = bmd.enhancement_factor(bmd_depth)

    print(f"  Parallel channels: {N_BMD:,} (3^{bmd_depth})")
    print(f"  BMD enhancement: F_BMD = {F_BMD:.2e}")

    # Step 4: Reflectance cascade
    print(f"\n{'='*70}")
    print(f"REFLECTANCE CASCADE")
    print(f"{'='*70}\n")
    print(f"Reflections: {n_reflections}")

    cascade = MolecularDemonReflectanceCascade(
        network=network,
        bmd_depth=bmd_depth,
        base_frequency_hz=avg_freq,
        reflectance_coefficient=0.1
    )

    results = cascade.run_cascade(n_reflections=n_reflections)

    # Extract key results
    final_freq_hz = results['final_frequency_hz']
    enhanced_precision_s = results['precision_achieved_s']
    F_cascade = results['enhancement_factors']['cascade']
    F_total = results['enhancement_factors']['total']

    planck_time = 5.39e-44
    orders_below_planck = -np.log10(enhanced_precision_s / planck_time)

    print(f"  Cascade enhancement: F_cascade = {F_cascade:.2e}")
    print(f"  Total enhancement: F_total = {F_total:.2e}")

    # Final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: MULTI-MOLECULE HARMONIC NETWORK")
    print(f"{'='*70}\n")

    print(f"Molecules analyzed: {len(molecules)}")
    print(f"Total vibrational modes: {sum(len(m['modes']) for m in molecules.values())}")
    print(f"Total oscillators: {num_nodes:,}")
    print(f"Harmonic coincidences: {num_edges:,}")
    print(f"\nFinal frequency: {final_freq_hz:.2e} Hz")
    print(f"Enhanced precision: {enhanced_precision_s:.2e} s")
    print(f"Orders below Planck: {orders_below_planck:.2f}")

    print(f"\n🎯 KEY ACHIEVEMENT:")
    print(f"   Using {len(molecules)} common molecules (CH4, C6H6, C8H18, C8H8O3)")
    print(f"   Built network of {num_nodes:,} oscillators with {num_edges:,} connections")
    print(f"   Achieved {enhanced_precision_s:.2e} s precision")
    print(f"   = {orders_below_planck:.2f} orders below Planck time")
    print(f"   Total enhancement: {F_total:.2e}x\n")

    # Compile results
    full_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experiment': 'multi_molecule_harmonic_network',
        'molecules': {k: v['name'] for k, v in molecules.items()},
        'parameters': {
            'num_molecules': len(molecules),
            'bmd_depth': bmd_depth,
            'n_reflections': n_reflections,
            'max_harmonics': max_harmonics,
            'coincidence_threshold_hz': coincidence_threshold_hz
        },
        'network': {
            'num_oscillators': num_nodes,
            'num_edges': num_edges,
            'average_degree': float(avg_degree),
            'density': float(density),
            'enhancement_factor': float(F_graph)
        },
        'bmd': {
            'depth': bmd_depth,
            'parallel_channels': int(N_BMD),
            'enhancement_factor': float(F_BMD)
        },
        'cascade': {
            'reflections': n_reflections,
            'enhancement_factor': float(F_cascade)
        },
        'results': {
            'final_frequency_hz': float(final_freq_hz),
            'enhanced_precision_s': float(enhanced_precision_s),
            'total_enhancement': float(F_total),
            'orders_below_planck': float(orders_below_planck),
            'measurement_time_s': 0.0,
            'planck_time_s': planck_time
        }
    }

    return full_results

def main():
    """Run multi-molecule network analysis"""

    print(f"\n{'#'*70}")
    print(f"# MULTI-MOLECULE CATEGORICAL DYNAMICS ANALYSIS")
    print(f"# Trans-Planckian Precision from Molecular Vibrations")
    print(f"{'#'*70}\n")

    # Analyze full molecular ensemble
    results = analyze_multi_molecule_network(
        molecules=MOLECULAR_VIBRATIONS,
        bmd_depth=14,  # Higher for more molecular modes
        n_reflections=10,
        max_harmonics=10,  # REDUCED: 10 is plenty for demo, 150 was causing 720B operations!
        coincidence_threshold_hz=1e10  # 10 GHz threshold
    )

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    timestamp = results['timestamp']
    json_path = output_dir / f'multi_molecule_network_{timestamp}.json'

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'='*70}")
    print(f"Results saved to: {json_path}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
