#!/usr/bin/env python3
"""
Molecular Structure Prediction via Categorical Harmonic Networks (SIMPLIFIED)

Predict unknown vibrational modes from known modes using harmonic coincidence networks.
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

# Real molecular bond frequencies (cm^-1) for validation
BOND_FREQUENCIES = {
    'C-H_stretch': (2850, 3000, 2900),
    'C-C_stretch': (1000, 1300, 1100),
    'C=C_stretch': (1620, 1680, 1650),
    'C=O_stretch': (1650, 1750, 1715),
    'O-H_stretch': (3200, 3600, 3400),
    'N-H_stretch': (3300, 3500, 3400),
}

def wavenumber_to_hz(wavenumber_cm_inv: float) -> float:
    """Convert wavenumber (cm^-1) to frequency (Hz)"""
    c = 2.99792458e10  # cm/s
    return wavenumber_cm_inv * c

def hz_to_wavenumber(freq_hz: float) -> float:
    """Convert frequency (Hz) to wavenumber (cm^-1)"""
    c = 2.99792458e10  # cm/s
    return freq_hz / c

def predict_bond_frequency(
    network: MolecularHarmonicNetwork,
    bond_type: str,
    threshold_hz: float = 1e11
) -> Dict:
    """
    Predict bond frequency from harmonic coincidences

    Returns prediction with validation against known values
    """
    if bond_type not in BOND_FREQUENCIES:
        return {'error': f'Unknown bond type: {bond_type}'}

    freq_min, freq_max, freq_true = BOND_FREQUENCIES[bond_type]

    print(f"\nPredicting: {bond_type}")
    print(f"  Known range: {freq_min}-{freq_max} cm⁻¹")
    print(f"  True value: {freq_true} cm⁻¹")

    # Search frequency space
    test_freqs_cm = np.linspace(freq_min - 200, freq_max + 200, 500)
    scores = []

    for test_freq_cm in test_freqs_cm:
        test_freq_hz = wavenumber_to_hz(test_freq_cm)

        # Count coincidences with network oscillators
        score = 0
        for osc in network.oscillators.values():
            # Check harmonics
            for n in range(1, 10):
                harmonic = n * test_freq_hz
                if abs(harmonic - osc.frequency_hz) < threshold_hz:
                    score += 1.0 / n

        scores.append(score)

    # Find maximum
    max_idx = np.argmax(scores)
    predicted_cm = test_freqs_cm[max_idx]

    error_cm = abs(predicted_cm - freq_true)
    error_percent = 100 * error_cm / freq_true

    print(f"  Predicted: {predicted_cm:.1f} cm⁻¹")
    print(f"  Error: {error_cm:.1f} cm⁻¹ ({error_percent:.2f}%)")

    return {
        'bond_type': bond_type,
        'predicted_cm': float(predicted_cm),
        'true_cm': float(freq_true),
        'error_cm': float(error_cm),
        'error_percent': float(error_percent),
        'max_score': float(max(scores))
    }

def main():
    """Run molecular structure prediction demo"""

    print("="*70)
    print("MOLECULAR STRUCTURE PREDICTION (Simplified)")
    print("="*70)

    # Known modes (from experimental data)
    known_modes = [
        ('C-H_stretch', 2900),
        ('C-C_stretch', 1100),
    ]

    print(f"\nKnown modes:")
    for name, freq_cm in known_modes:
        print(f"  {name}: {freq_cm} cm⁻¹")

    # Build network from known modes
    network = MolecularHarmonicNetwork(name="structure_prediction")

    print(f"\nBuilding harmonic network...")
    for mode_name, freq_cm in known_modes:
        freq_hz = wavenumber_to_hz(freq_cm)

        # Add fundamental and harmonics
        for n in range(1, 11):  # 10 harmonics
            network.add_oscillator(
                frequency=n * freq_hz,
                amplitude=1.0 / n,
                oscillator_id=f"{mode_name}_n{n}",
                metadata={'mode': mode_name, 'harmonic': n}
            )

    # Find coincidences
    network.find_coincidences(tolerance_hz=1e11)

    summary = network.get_summary()
    print(f"  Oscillators: {summary['num_oscillators']}")
    print(f"  Coincidences: {summary['num_coincidences']}")

    # Predict unknown modes
    print(f"\n{'='*70}")
    print("PREDICTING UNKNOWN MODES")
    print("="*70)

    targets = ['C=C_stretch', 'C=O_stretch', 'O-H_stretch']
    predictions = []

    for bond_type in targets:
        pred = predict_bond_frequency(network, bond_type)
        predictions.append(pred)

    # Summary
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print("="*70)

    for pred in predictions:
        if 'error' not in pred:
            print(f"{pred['bond_type']:15s}: {pred['predicted_cm']:7.1f} cm⁻¹ "
                  f"(error: {pred['error_percent']:5.2f}%)")

    # Save results
    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'known_modes': known_modes,
        'network_summary': summary,
        'predictions': predictions
    }

    output_dir = Path(__file__).parent.parent.parent / 'observatory' / 'results' / 'molecular_prediction'
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"structure_prediction_{results['timestamp']}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {json_path}")
    print("="*70)

if __name__ == '__main__':
    main()
