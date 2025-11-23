#!/usr/bin/env python3
"""
Molecular Vibration Resolution Enhancement via Categorical Dynamics

Applies harmonic network + BMD decomposition + reflectance cascade
to real molecular vibration data for trans-Planckian precision.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Imports are from the same directory
from molecular_network import HarmonicNetworkGraph, MolecularOscillator
from bmd_decomposition import BMDHierarchy
from reflectance_cascade import MolecularDemonReflectanceCascade
from categorical_state import SEntropyCalculator

def load_molecular_vibration_data(json_path: Path) -> dict:
    """Load molecular vibration experimental data"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_molecular_oscillator_ensemble(vibration_data: dict, num_harmonics: int = 15) -> list:
    """
    Create oscillator ensemble from molecular vibration data

    Uses:
    - Base frequency from quantum vibration
    - Multiple vibrational modes if available
    - Thermal population distribution
    """
    base_freq = vibration_data['frequency_Hz']
    coherence_time_s = vibration_data['coherence_time_fs'] * 1e-15

    oscillators = []
    osc_id = 0

    # Primary vibrational mode
    for n in range(1, num_harmonics + 1):
        freq = n * base_freq
        phase = np.random.uniform(0, 2*np.pi)

        # S-entropy from coherence properties
        s_coords = SEntropyCalculator.from_frequency(
            frequency_hz=freq,
            measurement_count=n,
            time_elapsed=coherence_time_s
        )

        osc = MolecularOscillator(
            id=osc_id,
            species=f"CO2_v{n}",
            frequency_hz=freq,
            phase_rad=phase,
            s_coordinates=(s_coords.s_k, s_coords.s_t, s_coords.s_e)
        )
        oscillators.append(osc)
        osc_id += 1

    # Add overtones and combination bands if energy levels available
    if 'energy_levels_J' in vibration_data:
        h = 6.62607015e-34  # Planck constant
        for i, energy_J in enumerate(vibration_data['energy_levels_J']):
            freq_overtone = energy_J / h
            phase = np.random.uniform(0, 2*np.pi)

            s_coords = SEntropyCalculator.from_frequency(
                frequency_hz=freq_overtone,
                measurement_count=i+1,
                time_elapsed=coherence_time_s
            )

            osc = MolecularOscillator(
                id=osc_id,
                species=f"CO2_E{i}",
                frequency_hz=freq_overtone,
                phase_rad=phase,
                s_coordinates=(s_coords.s_k, s_coords.s_t, s_coords.s_e)
            )
            oscillators.append(osc)
            osc_id += 1

    return oscillators

def calculate_molecular_resolution_enhancement(vibration_data: dict,
                                              bmd_depth: int = 12,
                                              n_reflections: int = 10,
                                              max_harmonics: int = 15,
                                              coincidence_threshold_hz: float = 1e9) -> dict:
    """
    Apply full categorical dynamics framework to molecular vibration data

    Args:
        vibration_data: Experimental molecular vibration results
        bmd_depth: Biological Maxwell Demon decomposition depth
        n_reflections: Number of cascade reflections
        max_harmonics: Maximum harmonic order
        coincidence_threshold_hz: Frequency coincidence threshold

    Returns:
        Enhanced resolution results with all metrics
    """

    print(f"\n{'='*70}")
    print(f"MOLECULAR VIBRATION RESOLUTION ENHANCEMENT")
    print(f"{'='*70}")

    # Original experimental precision
    base_freq = vibration_data['frequency_Hz']
    original_precision_s = vibration_data['temporal_precision_fs'] * 1e-15
    coherence_time_fs = vibration_data['coherence_time_fs']

    print(f"\nOriginal Experimental Results:")
    print(f"  Base frequency: {base_freq:.2e} Hz ({base_freq/1e12:.1f} THz)")
    print(f"  Temporal precision: {original_precision_s:.2e} s ({original_precision_s*1e15:.1f} fs)")
    print(f"  Coherence time: {coherence_time_fs:.1f} fs")
    print(f"  Heisenberg linewidth: {vibration_data['heisenberg_linewidth_Hz']:.2e} Hz")

    # Step 1: Generate molecular oscillator ensemble
    print(f"\nStep 1: Generating Molecular Oscillator Ensemble")
    print(f"  Harmonics per mode: {max_harmonics}")

    oscillators = create_molecular_oscillator_ensemble(vibration_data, max_harmonics)
    print(f"  Total oscillators: {len(oscillators)}")

    # Step 2: Build harmonic coincidence network
    print(f"\nStep 2: Building Harmonic Coincidence Network")
    print(f"  Coincidence threshold: {coincidence_threshold_hz:.2e} Hz")

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

    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Average degree: {avg_degree:.1f}")
    print(f"  Density: {density:.4f}")

    # Graph enhancement
    F_graph = network.calculate_enhancement_factor()
    print(f"  Graph enhancement: F_graph = {F_graph:.2e}")

    # Step 3: BMD decomposition
    print(f"\nStep 3: Biological Maxwell Demon Decomposition")
    print(f"  Decomposition depth: {bmd_depth}")

    bmd = BMDHierarchy(root_frequency=base_freq)
    bmd.build_hierarchy(depth=bmd_depth)
    N_BMD = bmd.total_parallel_channels(bmd_depth)
    F_BMD = bmd.enhancement_factor(bmd_depth)

    print(f"  Parallel channels: {N_BMD:,} (3^{bmd_depth})")
    print(f"  BMD enhancement: F_BMD = {F_BMD:.2e}")

    # Step 4: Reflectance cascade
    print(f"\nStep 4: Molecular Demon Reflectance Cascade")
    print(f"  Reflections: {n_reflections}")

    cascade = MolecularDemonReflectanceCascade(
        network=network,
        bmd_depth=bmd_depth,
        base_frequency_hz=base_freq,
        reflectance_coefficient=0.1
    )

    results = cascade.run_cascade(n_reflections=n_reflections)

    # Extract results
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
    print(f"ENHANCED RESOLUTION RESULTS")
    print(f"{'='*70}")
    print(f"\nFinal frequency: {final_freq_hz:.2e} Hz")
    print(f"Enhanced precision: {enhanced_precision_s:.2e} s")
    print(f"Orders below Planck time: {orders_below_planck:.2f}")

    # Improvement over original
    improvement_factor = original_precision_s / enhanced_precision_s
    print(f"\nImprovement over experimental:")
    print(f"  Original: {original_precision_s:.2e} s ({original_precision_s*1e15:.1f} fs)")
    print(f"  Enhanced: {enhanced_precision_s:.2e} s")
    print(f"  Improvement: {improvement_factor:.2e}x")

    # Comparison with limits
    print(f"\nComparison with fundamental limits:")
    print(f"  Heisenberg limit (experimental): {vibration_data['heisenberg_linewidth_Hz']:.2e} Hz")
    print(f"  Categorical resolution: {final_freq_hz:.2e} Hz")
    print(f"  Planck time: {planck_time:.2e} s")
    print(f"  This result: {enhanced_precision_s:.2e} s ({orders_below_planck:.2f} orders below Planck)")

    # Compile full results
    full_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experiment': 'molecular_vibration_categorical_enhancement',
        'input_data': vibration_data,
        'parameters': {
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
            'original_precision_s': float(original_precision_s),
            'original_precision_fs': float(original_precision_s * 1e15),
            'enhanced_frequency_hz': float(final_freq_hz),
            'enhanced_precision_s': float(enhanced_precision_s),
            'total_enhancement': float(F_total),
            'improvement_over_experimental': float(improvement_factor),
            'orders_below_planck': float(orders_below_planck),
            'measurement_time_s': 0.0,
            'zero_time_measurement': True
        },
        'comparison': {
            'experimental_heisenberg_hz': vibration_data['heisenberg_linewidth_Hz'],
            'categorical_resolution_hz': float(final_freq_hz),
            'planck_time_s': planck_time,
            'bypass_mechanism': 'categorical_state_access'
        }
    }

    return full_results

def main():
    """Run molecular vibration enhancement analysis"""

    # Path to experimental data - public folder is in the same directory as this script
    data_path = Path(__file__).parent / 'public' / 'quantum_vibrations_20251105_124305.json'

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please provide path to molecular vibration JSON file.")
        return

    # Load experimental data
    vibration_data = load_molecular_vibration_data(data_path)

    # Run enhancement analysis (reduced harmonics for performance)
    results = calculate_molecular_resolution_enhancement(
        vibration_data=vibration_data,
        bmd_depth=12,  # Deeper than hardware (more molecular modes)
        n_reflections=10,
        max_harmonics=15,  # REDUCED: 15 is sufficient, 150 was too slow
        coincidence_threshold_hz=1e9
    )

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    timestamp = results['timestamp']
    json_path = output_dir / f'molecular_enhancement_{timestamp}.json'

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {json_path}")
    print(f"{'='*70}")

    # Key takeaway
    print(f"\n🎯 KEY RESULT:")
    print(f"   Molecular vibration at {vibration_data['frequency_Hz']/1e12:.1f} THz")
    print(f"   Enhanced from {results['results']['original_precision_fs']:.1f} fs precision")
    print(f"   To {results['results']['enhanced_precision_s']:.2e} s precision")
    print(f"   = {results['results']['orders_below_planck']:.2f} orders below Planck time")
    print(f"   Improvement: {results['results']['improvement_over_experimental']:.2e}x\n")

if __name__ == '__main__':
    main()
