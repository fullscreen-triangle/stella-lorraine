"""
Reproduce Trans-Planckian Experimental Results

Target: Match 7.51×10^-50 s precision from experimental data
File: trans_planckian_20251011_085807.json

This script validates the complete Molecular Demon Reflectance Cascade
by reproducing the experimentally achieved trans-Planckian precision.
"""

import sys
import json
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import (
    MolecularOscillator,
    HarmonicNetworkGraph,
    CategoricalState,
    MolecularDemonReflectanceCascade
)

from physics import (
    MolecularOscillatorGenerator,
    HeisenbergBypass
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experimental_target():
    """Load experimental result to match"""
    data_path = Path(__file__).parent.parent / 'data' / 'experimental' / 'trans_planckian_20251011_085807.json'

    if not data_path.exists():
        logger.warning(f"Experimental data not found at {data_path}")
        logger.warning("Using hardcoded target values")
        return {
            'precision_achieved_s': 7.511154627619322e-50,
            'planck_analysis': {
                'planck_time_s': 5.39116e-44,
                'ratio_to_planck': 1.3932353385207122e-06,
                'orders_below_planck': 5.855975518473129
            },
            'network_analysis': {
                'total_nodes': 260000,
                'total_edges': 25794141,
                'avg_degree': 198.41646923076922,
                'graph_enhancement': 7175.993927991298
            }
        }

    with open(data_path) as f:
        return json.load(f)


def generate_molecular_ensemble(n_molecules: int = 260_000, species: str = 'N2'):
    """
    Generate molecular ensemble matching experimental conditions

    Args:
        n_molecules: Number of molecules (default matches experiment)
        species: Molecular species (default: N2)

    Returns:
        List of MolecularOscillator objects
    """
    logger.info(f"Generating {n_molecules:,} molecule ensemble ({species})...")

    generator = MolecularOscillatorGenerator(species=species, temperature_k=300.0)
    molecule_dicts = generator.generate_ensemble(n_molecules, seed=42)

    # Convert to MolecularOscillator objects
    molecules = []
    for mol_dict in molecule_dicts:
        molecules.append(MolecularOscillator(
            id=mol_dict['id'],
            species=mol_dict['species'],
            frequency_hz=mol_dict['frequency_hz'],
            phase_rad=mol_dict['phase_rad'],
            s_coordinates=mol_dict['s_coordinates']
        ))

    logger.info(f"  Mean frequency: {np.mean([m.frequency_hz for m in molecules]):.2e} Hz")
    logger.info(f"  Frequency spread: {np.std([m.frequency_hz for m in molecules]):.2e} Hz")

    return molecules


def build_harmonic_network(molecules, threshold_hz=1e6):
    """
    Build harmonic network graph

    Args:
        molecules: List of molecular oscillators
        threshold_hz: Coincidence threshold

    Returns:
        HarmonicNetworkGraph
    """
    logger.info("Building harmonic network graph...")
    logger.info(f"  Coincidence threshold: {threshold_hz:.2e} Hz")

    def progress_callback(current, total):
        if current % 100000 == 0:
            percent = 100 * current / total
            logger.info(f"  Progress: {current:,}/{total:,} pairs ({percent:.1f}%)")

    network = HarmonicNetworkGraph(
        molecules=molecules,
        coincidence_threshold_hz=threshold_hz,
        max_harmonics=150
    )

    graph = network.build_graph(progress_callback=progress_callback)

    stats = network.graph_statistics()
    logger.info(f"\nNetwork statistics:")
    logger.info(f"  Nodes: {stats['total_nodes']:,}")
    logger.info(f"  Edges: {stats['total_edges']:,}")
    logger.info(f"  Average degree: {stats['avg_degree']:.2f}")
    logger.info(f"  Density: {stats['density']:.6f}")
    logger.info(f"  Graph enhancement: {stats['graph_enhancement']:.2f}×")

    return network


def run_cascade(network, bmd_depth=10, n_reflections=10):
    """
    Execute Molecular Demon Reflectance Cascade

    Args:
        network: Harmonic network graph
        bmd_depth: BMD decomposition depth
        n_reflections: Number of cascade reflections

    Returns:
        Results dictionary
    """
    logger.info("\nInitializing Molecular Demon Reflectance Cascade...")

    cascade = MolecularDemonReflectanceCascade(
        network=network,
        bmd_depth=bmd_depth,
        base_frequency_hz=7.07e13,  # N2 fundamental
        reflectance_coefficient=0.1
    )

    logger.info("Executing cascade...")
    results = cascade.run_cascade(n_reflections=n_reflections)

    # Validate zero-time measurement
    logger.info("\nValidating measurement properties...")
    cascade.validate_zero_time()
    cascade.validate_spectrometer_dissolution()

    return results


def compare_with_experiment(results, experimental):
    """
    Compare achieved precision with experimental target

    Args:
        results: Computed results
        experimental: Experimental target data

    Returns:
        Comparison dictionary
    """
    logger.info("\n" + "="*70)
    logger.info("COMPARISON WITH EXPERIMENTAL DATA")
    logger.info("="*70)

    target = experimental['precision_achieved_s']
    achieved = results['precision_achieved_s']

    relative_error = abs(achieved - target) / target

    logger.info(f"\nPrecision:")
    logger.info(f"  Target (experimental): {target:.2e} s")
    logger.info(f"  Achieved (computed):   {achieved:.2e} s")
    logger.info(f"  Relative error:        {relative_error*100:.2f}%")

    logger.info(f"\nPlanck analysis:")
    logger.info(f"  Target orders below:   {experimental['planck_analysis']['orders_below_planck']:.2f}")
    logger.info(f"  Achieved orders below: {results['planck_analysis']['orders_below_planck']:.2f}")

    logger.info(f"\nNetwork statistics:")
    logger.info(f"  Target nodes:          {experimental['network_analysis']['total_nodes']:,}")
    logger.info(f"  Achieved nodes:        {results['network_analysis']['total_nodes']:,}")
    logger.info(f"  Target edges:          {experimental['network_analysis']['total_edges']:,}")
    logger.info(f"  Achieved edges:        {results['network_analysis']['total_edges']:,}")
    logger.info(f"  Target enhancement:    {experimental['network_analysis']['graph_enhancement']:.2f}×")
    logger.info(f"  Achieved enhancement:  {results['network_analysis']['graph_enhancement']:.2f}×")

    # Validation
    validation_passed = relative_error < 0.5  # Within 50%

    if validation_passed:
        logger.info("\n✓ VALIDATION PASSED")
        logger.info("  Achieved precision within acceptable range of experimental data")
    else:
        logger.warning("\n✗ VALIDATION FAILED")
        logger.warning(f"  Error {relative_error*100:.1f}% exceeds 50% threshold")

    logger.info("="*70)

    return {
        'target_precision_s': target,
        'achieved_precision_s': achieved,
        'relative_error': relative_error,
        'validation_passed': validation_passed
    }


def save_results(results, comparison):
    """Save results to file"""
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"validation_{results['timestamp']}.json"

    combined = {
        **results,
        'comparison': comparison
    }

    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


def main():
    """Main experimental reproduction"""
    print("\n" + "="*70)
    print("MOLECULAR DEMON REFLECTANCE CASCADE")
    print("Trans-Planckian Precision Validation")
    print("="*70)

    # Load experimental target
    logger.info("\nLoading experimental target data...")
    experimental = load_experimental_target()
    logger.info(f"Target precision: {experimental['precision_achieved_s']:.2e} s")

    # Verify Heisenberg bypass
    logger.info("\nVerifying Heisenberg bypass...")
    HeisenbergBypass.verify_orthogonality()
    HeisenbergBypass.zero_backaction_proof()

    # Generate molecular ensemble
    molecules = generate_molecular_ensemble(n_molecules=260_000, species='N2')

    # Build harmonic network
    network = build_harmonic_network(molecules, threshold_hz=1e6)

    # Run cascade
    results = run_cascade(
        network=network,
        bmd_depth=10,  # 3^10 = 59,049 parallel channels
        n_reflections=10
    )

    # Compare with experiment
    comparison = compare_with_experiment(results, experimental)

    # Save results
    save_results(results, comparison)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Precision achieved: {results['precision_achieved_s']:.2e} s")
    print(f"Orders below Planck: {results['planck_analysis']['orders_below_planck']:.2f}")
    print(f"Total enhancement: {results['enhancement_factors']['total']:.2e}×")
    print(f"Validation: {'PASSED ✓' if comparison['validation_passed'] else 'FAILED ✗'}")
    print("="*70)


if __name__ == "__main__":
    main()
