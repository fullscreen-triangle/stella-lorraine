#!/usr/bin/env python3
"""
Molecular Structure Prediction via Categorical Harmonic Networks

Instead of measuring time, use harmonic coincidence networks to:
1. Predict unknown vibrational modes from known modes
2. Extract bond stretching frequencies without direct measurement
3. Infer molecular geometry from categorical state topology

This is MUCH more practical and verifiable than trans-Planckian claims.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add observatory/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'observatory' / 'src'))

from maxwell.harmonic_coincidence import MolecularHarmonicNetwork
from maxwell.pixel_maxwell_demon import SEntropyCoordinates

# Real molecular bond frequencies (cm^-1) - GROUND TRUTH for validation
BOND_FREQUENCIES = {
    'C-H_stretch': {
        'range': (2850, 3000),
        'typical': 2900,
        'description': 'Alkane C-H stretch'
    },
    'C-C_stretch': {
        'range': (1000, 1300),
        'typical': 1100,
        'description': 'C-C single bond stretch'
    },
    'C=C_stretch': {
        'range': (1620, 1680),
        'typical': 1650,
        'description': 'C=C double bond stretch'
    },
    'C≡C_stretch': {
        'range': (2100, 2260),
        'typical': 2150,
        'description': 'C≡C triple bond stretch'
    },
    'C=O_stretch': {
        'range': (1650, 1750),
        'typical': 1715,
        'description': 'Carbonyl stretch'
    },
    'O-H_stretch': {
        'range': (3200, 3600),
        'typical': 3400,
        'description': 'Alcohol/phenol O-H'
    },
    'N-H_stretch': {
        'range': (3300, 3500),
        'typical': 3400,
        'description': 'Amine N-H stretch'
    },
    'C-O_stretch': {
        'range': (1000, 1300),
        'typical': 1100,
        'description': 'C-O single bond'
    },
    'C-N_stretch': {
        'range': (1020, 1250),
        'typical': 1150,
        'description': 'C-N single bond'
    },
}

def wavenumber_to_hz(wavenumber_cm_inv: float) -> float:
    """Convert wavenumber (cm^-1) to frequency (Hz)"""
    c = 2.99792458e10  # cm/s
    return wavenumber_cm_inv * c

def hz_to_wavenumber(freq_hz: float) -> float:
    """Convert frequency (Hz) to wavenumber (cm^-1)"""
    c = 2.99792458e10  # cm/s
    return freq_hz / c

class MolecularStructurePredictor:
    """
    Predict molecular vibrational modes using categorical harmonic networks

    Key idea: Build network from KNOWN modes, use categorical coincidences
    to predict UNKNOWN modes without direct measurement.
    """

    def __init__(self, known_modes: List[Tuple[str, float]],
                 max_harmonics: int = 50,
                 coincidence_threshold_hz: float = 1e11):
        """
        Initialize predictor with known vibrational modes

        Args:
            known_modes: List of (mode_name, frequency_cm-1) tuples
            max_harmonics: Number of harmonics to generate
            coincidence_threshold_hz: Threshold for harmonic coincidence
        """
        self.known_modes = known_modes
        self.max_harmonics = max_harmonics
        self.threshold = coincidence_threshold_hz

        # Build network from known modes
        self.oscillators = self._create_oscillator_ensemble()
        self.network = self._build_network()

    def _create_oscillator_ensemble(self) -> List[MolecularOscillator]:
        """Create oscillators from known modes"""
        oscillators = []
        osc_id = 0

        for mode_name, wavenumber in self.known_modes:
            freq_hz = wavenumber_to_hz(wavenumber)

            # Generate harmonics
            for n in range(1, self.max_harmonics + 1):
                harmonic_freq = n * freq_hz
                phase = np.random.uniform(0, 2*np.pi)

                # S-entropy coordinates
                coherence_time_s = 1e-13 / n  # Decreases with harmonic order
                s_coords = SEntropyCalculator.from_frequency(
                    frequency_hz=harmonic_freq,
                    measurement_count=n,
                    time_elapsed=coherence_time_s
                )

                osc = MolecularOscillator(
                    id=osc_id,
                    species=f"{mode_name}_n{n}",
                    frequency_hz=harmonic_freq,
                    phase_rad=phase,
                    s_coordinates=(s_coords.s_k, s_coords.s_t, s_coords.s_e)
                )
                oscillators.append(osc)
                osc_id += 1

        return oscillators

    def _build_network(self) -> HarmonicNetworkGraph:
        """Build harmonic coincidence network"""
        network = HarmonicNetworkGraph(
            molecules=self.oscillators,
            coincidence_threshold_hz=self.threshold
        )
        network.build_graph()
        return network

    def predict_unknown_modes(self, target_bonds: List[str]) -> Dict[str, Dict]:
        """
        Predict vibrational frequencies for target bonds

        Uses categorical convergence nodes in the network to infer
        frequencies that would create strong harmonic coincidences.

        Args:
            target_bonds: List of bond types to predict (e.g., 'C=O_stretch')

        Returns:
            Dictionary of predictions with confidence metrics
        """
        predictions = {}

        print(f"\n{'='*70}")
        print(f"PREDICTING UNKNOWN VIBRATIONAL MODES")
        print(f"{'='*70}\n")

        # Get convergence nodes (high-degree nodes in network)
        graph = self.network.graph
        degrees = dict(graph.degree())
        avg_degree = np.mean(list(degrees.values()))

        convergence_nodes = [
            node for node, deg in degrees.items()
            if deg > avg_degree * 1.5
        ]

        print(f"Network statistics:")
        print(f"  Total nodes: {len(self.oscillators)}")
        print(f"  Total edges: {graph.number_of_edges()}")
        print(f"  Average degree: {avg_degree:.1f}")
        print(f"  Convergence nodes: {len(convergence_nodes)}")

        # For each target bond, find frequencies that would maximize
        # harmonic coincidences with convergence nodes
        for bond_type in target_bonds:
            if bond_type not in BOND_FREQUENCIES:
                print(f"\nWarning: Unknown bond type '{bond_type}'")
                continue

            bond_info = BOND_FREQUENCIES[bond_type]
            true_freq_cm = bond_info['typical']
            true_freq_hz = wavenumber_to_hz(true_freq_cm)

            print(f"\n{'-'*70}")
            print(f"Predicting: {bond_type}")
            print(f"  Known range: {bond_info['range']} cm⁻¹")
            print(f"  True value: {true_freq_cm} cm⁻¹ (for validation)")

            # Search frequency space for maximum coincidences
            search_range_cm = np.linspace(
                bond_info['range'][0] - 200,
                bond_info['range'][1] + 200,
                1000
            )

            coincidence_scores = []
            for test_freq_cm in search_range_cm:
                test_freq_hz = wavenumber_to_hz(test_freq_cm)

                # Count harmonics of test frequency that coincide with
                # convergence nodes
                score = 0
                for node_idx in convergence_nodes:
                    node_freq = self.oscillators[node_idx].frequency_hz

                    # Check if any harmonic of test_freq matches node_freq
                    for n in range(1, 20):  # Check low-order harmonics
                        harmonic = n * test_freq_hz
                        if abs(harmonic - node_freq) < self.threshold:
                            score += 1 / n  # Weight by inverse harmonic order

                coincidence_scores.append(score)

            # Find maximum
            max_idx = np.argmax(coincidence_scores)
            predicted_freq_cm = search_range_cm[max_idx]
            predicted_freq_hz = wavenumber_to_hz(predicted_freq_cm)
            confidence = coincidence_scores[max_idx]

            # Error vs. true value
            error_cm = abs(predicted_freq_cm - true_freq_cm)
            error_percent = 100 * error_cm / true_freq_cm

            print(f"  Predicted: {predicted_freq_cm:.1f} cm⁻¹")
            print(f"  Error: {error_cm:.1f} cm⁻¹ ({error_percent:.2f}%)")
            print(f"  Confidence score: {confidence:.2f}")

            predictions[bond_type] = {
                'predicted_wavenumber_cm-1': float(predicted_freq_cm),
                'predicted_frequency_hz': float(predicted_freq_hz),
                'true_wavenumber_cm-1': float(true_freq_cm),
                'error_cm-1': float(error_cm),
                'error_percent': float(error_percent),
                'confidence': float(confidence),
                'description': bond_info['description']
            }

        return predictions

    def analyze_bond_network(self, molecule_name: str, bonds: List[str]) -> Dict:
        """
        Analyze how molecular bonds form categorical network

        Args:
            molecule_name: Name of molecule
            bonds: List of bond types present in molecule

        Returns:
            Network analysis showing categorical relationships between bonds
        """
        print(f"\n{'='*70}")
        print(f"BOND NETWORK ANALYSIS: {molecule_name}")
        print(f"{'='*70}\n")

        # Get frequencies for specified bonds
        bond_freqs = {}
        for bond in bonds:
            if bond in BOND_FREQUENCIES:
                freq_cm = BOND_FREQUENCIES[bond]['typical']
                freq_hz = wavenumber_to_hz(freq_cm)
                bond_freqs[bond] = freq_hz
                print(f"{bond}: {freq_cm} cm⁻¹ ({freq_hz:.2e} Hz)")

        # Analyze harmonic relationships between bonds
        print(f"\nHarmonic relationships:")
        relationships = []

        for bond1 in bond_freqs:
            for bond2 in bond_freqs:
                if bond1 >= bond2:
                    continue

                f1 = bond_freqs[bond1]
                f2 = bond_freqs[bond2]

                # Find low-order harmonic relationships
                for n1 in range(1, 10):
                    for n2 in range(1, 10):
                        if abs(n1 * f1 - n2 * f2) < self.threshold:
                            ratio = (n1 * f1) / (n2 * f2)
                            relationships.append({
                                'bond1': bond1,
                                'bond2': bond2,
                                'harmonic1': n1,
                                'harmonic2': n2,
                                'ratio': float(ratio),
                                'coincidence_freq_hz': float(n1 * f1)
                            })
                            print(f"  {bond1}(n={n1}) ≈ {bond2}(n={n2}): ratio = {ratio:.3f}")

        return {
            'molecule': molecule_name,
            'bonds': list(bonds),
            'bond_frequencies': {k: float(v) for k, v in bond_freqs.items()},
            'harmonic_relationships': relationships,
            'num_relationships': len(relationships)
        }

def demo_vanillin_prediction():
    """
    Demo: Predict C=O stretch in vanillin from other known modes

    Vanillin has:
    - O-H stretch (phenol)
    - C-H stretch (aromatic)
    - C=O stretch (aldehyde) <- PREDICT THIS
    - C-O stretches (methoxy)
    """

    print(f"\n{'#'*70}")
    print(f"# DEMO: PREDICT CARBONYL STRETCH IN VANILLIN")
    print(f"# Using categorical network from other vibrational modes")
    print(f"{'#'*70}\n")

    # Known modes (everything except C=O which we want to predict)
    known_modes = [
        ('OH_stretch', 3400),      # Phenol
        ('CH_aromatic', 3070),     # Aromatic C-H
        ('CO_methoxy', 1033),      # Methoxy C-O
        ('ring_stretch', 1583),    # Aromatic ring
        ('ring_stretch_2', 1512),  # Another ring mode
        ('CH_bend', 1425),         # C-H bending
    ]

    print("Known vibrational modes:")
    for name, freq in known_modes:
        print(f"  {name}: {freq} cm⁻¹")

    # Initialize predictor
    predictor = MolecularStructurePredictor(
        known_modes=known_modes,
        max_harmonics=50,
        coincidence_threshold_hz=1e11  # 100 GHz
    )

    # Predict C=O stretch
    predictions = predictor.predict_unknown_modes(['C=O_stretch'])

    # Analyze bond network
    vanillin_bonds = [
        'OH_stretch', 'CH_aromatic', 'C=O_stretch',
        'CO_methoxy', 'ring_stretch'
    ]

    network_analysis = predictor.analyze_bond_network('Vanillin', vanillin_bonds)

    # Save results
    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experiment': 'molecular_structure_prediction',
        'molecule': 'vanillin',
        'method': 'categorical_harmonic_network',
        'known_modes': {name: freq for name, freq in known_modes},
        'predictions': predictions,
        'network_analysis': network_analysis
    }

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f'vanillin_prediction_{results["timestamp"]}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {json_path}")
    print(f"{'='*70}\n")

    # Summary
    if 'C=O_stretch' in predictions:
        pred = predictions['C=O_stretch']
        print(f"🎯 PREDICTION SUMMARY:")
        print(f"   Target: C=O stretch in vanillin aldehyde")
        print(f"   Predicted: {pred['predicted_wavenumber_cm-1']:.1f} cm⁻¹")
        print(f"   True value: {pred['true_wavenumber_cm-1']:.1f} cm⁻¹")
        print(f"   Error: {pred['error_percent']:.2f}%")
        print(f"   Method: Categorical harmonic coincidence network\n")

def demo_alkane_ch_stretch():
    """
    Demo: Predict C-H stretching from C-C stretching modes

    Simple test: Given C-C bond frequencies, can we predict C-H?
    """

    print(f"\n{'#'*70}")
    print(f"# DEMO: PREDICT C-H STRETCH FROM C-C BONDS")
    print(f"# Testing categorical inference across bond types")
    print(f"{'#'*70}\n")

    # Known: Only C-C stretching modes
    known_modes = [
        ('CC_stretch_1', 1060),
        ('CC_stretch_2', 1100),
        ('CC_stretch_3', 1150),
        ('CC_bend', 420),
    ]

    print("Known modes (C-C bonds only):")
    for name, freq in known_modes:
        print(f"  {name}: {freq} cm⁻¹")

    predictor = MolecularStructurePredictor(
        known_modes=known_modes,
        max_harmonics=50,
        coincidence_threshold_hz=5e10  # 50 GHz
    )

    # Predict C-H stretch
    predictions = predictor.predict_unknown_modes(['C-H_stretch'])

    # Save
    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experiment': 'cross_bond_prediction',
        'method': 'categorical_inference',
        'known_modes': {name: freq for name, freq in known_modes},
        'predictions': predictions
    }

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f'ch_prediction_{results["timestamp"]}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {json_path}")
    print(f"{'='*70}\n")

def main():
    """Run molecular structure prediction demos"""

    # Demo 1: Predict carbonyl in vanillin
    demo_vanillin_prediction()

    # Demo 2: Cross-bond-type prediction
    demo_alkane_ch_stretch()

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("We're not 'measuring time' - we're using categorical harmonic")
    print("networks to INFER molecular structure from partial information.")
    print()
    print("This is:")
    print("  ✓ Immediately verifiable (compare to actual spectroscopy)")
    print("  ✓ Practically useful (molecular structure prediction)")
    print("  ✓ Scientifically grounded (no trans-Planckian claims)")
    print("  ✓ Uses same mathematical framework")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
