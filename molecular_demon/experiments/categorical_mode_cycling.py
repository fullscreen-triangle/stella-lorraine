#!/usr/bin/env python3
"""
Categorical Mode Cycling: Virtual Spectrometer Through Molecular Modes

Key insight: Each vibrational mode of a molecule is a CATEGORICAL STATE.
The virtual spectrometer cycles through these categories, materializing
only at specific categorical moments to access each mode.

This is the interferometry framework applied to molecular spectroscopy:
- Single molecule (not multiple oscillators)
- Each mode = categorical state
- Spectrometer cycles through modes (like planetary positions in interferometry)
- Source-target unification: molecule IS the spectrometer
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.categorical_state import SCategory, SEntropyCalculator

@dataclass
class VibrationalMode:
    """A single vibrational mode as a categorical state"""
    name: str
    frequency_cm: float  # Wavenumber
    frequency_hz: float
    symmetry: str  # e.g., 'A1', 'E', 'F2'
    intensity: float  # Relative intensity
    s_category: SCategory  # Categorical coordinates

    def __repr__(self):
        return f"{self.name}({self.frequency_cm:.0f} cm⁻¹, {self.symmetry})"

@dataclass
class CategoricalMoment:
    """A moment when virtual spectrometer materializes at a mode"""
    mode: VibrationalMode
    timestamp_categorical: int  # Categorical time (not chronological)
    spectrometer_state: str  # 'materialized' or 'potential'
    measured_intensity: float
    categorical_phase: float

class VirtualMolecularSpectrometer:
    """
    Virtual spectrometer that cycles through molecular vibrational modes

    Analogous to virtual interferometric stations in planetary interferometry,
    but cycling through modes of a SINGLE molecule rather than spatial positions.
    """

    def __init__(self, molecule_name: str):
        self.molecule_name = molecule_name
        self.modes: List[VibrationalMode] = []
        self.categorical_moments: List[CategoricalMoment] = []
        self.current_categorical_time = 0

    def add_mode(self, name: str, frequency_cm: float, symmetry: str,
                 intensity: float = 1.0):
        """Add a vibrational mode (categorical state) to the molecule"""

        # Convert to Hz
        c = 2.99792458e10  # cm/s
        frequency_hz = frequency_cm * c

        # Calculate S-entropy for this categorical state
        # Each mode has different categorical coordinates based on its properties
        s_category = SEntropyCalculator.from_frequency(
            frequency_hz=frequency_hz,
            measurement_count=len(self.modes) + 1,
            time_elapsed=1e-13  # Typical vibrational period
        )

        mode = VibrationalMode(
            name=name,
            frequency_cm=frequency_cm,
            frequency_hz=frequency_hz,
            symmetry=symmetry,
            intensity=intensity,
            s_category=s_category
        )

        self.modes.append(mode)
        print(f"Added mode: {mode}")

    def materialize_at_mode(self, mode: VibrationalMode) -> CategoricalMoment:
        """
        Materialize virtual spectrometer at specific vibrational mode

        The spectrometer does NOT exist between measurements.
        It only materializes when accessing a specific categorical state (mode).
        """

        # Categorical phase based on mode relationships
        phase = 2 * np.pi * self.current_categorical_time / len(self.modes)

        moment = CategoricalMoment(
            mode=mode,
            timestamp_categorical=self.current_categorical_time,
            spectrometer_state='materialized',
            measured_intensity=mode.intensity,
            categorical_phase=phase
        )

        self.categorical_moments.append(moment)
        self.current_categorical_time += 1

        return moment

    def dissolve_spectrometer(self):
        """
        Spectrometer returns to categorical potential
        Exists nowhere in phase space until next materialization
        """
        pass  # Spectrometer ceases to exist

    def cycle_through_modes(self) -> List[CategoricalMoment]:
        """
        Cycle virtual spectrometer through all molecular modes

        This is analogous to cycling through planetary positions in interferometry,
        but here we cycle through CATEGORICAL STATES (vibrational modes).
        """

        print(f"\n{'='*70}")
        print(f"CYCLING VIRTUAL SPECTROMETER THROUGH MODES")
        print(f"Molecule: {self.molecule_name}")
        print(f"{'='*70}\n")

        moments = []

        for mode in self.modes:
            print(f"Categorical moment {self.current_categorical_time}:")
            print(f"  Materializing at: {mode}")

            # Materialize spectrometer at this mode
            moment = self.materialize_at_mode(mode)
            print(f"  S-category: ({moment.mode.s_category.s_k:.3f}, "
                  f"{moment.mode.s_category.s_t:.3f}, {moment.mode.s_category.s_e:.3f})")
            print(f"  Categorical phase: {moment.categorical_phase:.3f} rad")
            print(f"  Intensity: {moment.measured_intensity:.3f}")

            # Dissolve spectrometer
            self.dissolve_spectrometer()
            print(f"  Dissolved\n")

            moments.append(moment)

        return moments

    def calculate_mode_mode_correlations(self) -> np.ndarray:
        """
        Calculate categorical correlations between modes

        Analogous to baseline correlations in interferometry.
        High correlation = modes are categorically close (similar S-entropy).
        """

        n_modes = len(self.modes)
        correlations = np.zeros((n_modes, n_modes))

        for i, mode_i in enumerate(self.modes):
            for j, mode_j in enumerate(self.modes):
                # Categorical distance in S-entropy space
                ds_k = mode_i.s_category.s_k - mode_j.s_category.s_k
                ds_t = mode_i.s_category.s_t - mode_j.s_category.s_t
                ds_e = mode_i.s_category.s_e - mode_j.s_category.s_e

                distance = np.sqrt(ds_k**2 + ds_t**2 + ds_e**2)

                # Correlation inversely proportional to distance
                correlations[i, j] = np.exp(-distance / 0.5)

        return correlations

    def predict_unknown_mode_from_cycling(self, known_mode_indices: List[int],
                                         target_symmetry: str) -> Dict:
        """
        Predict unknown mode by analyzing categorical cycling pattern

        Uses the fact that modes form categorical network through S-entropy space.
        By cycling through KNOWN modes, we can infer UNKNOWN modes that would
        complete the categorical pattern.

        Args:
            known_mode_indices: Indices of modes we've measured
            target_symmetry: Symmetry of mode we want to predict

        Returns:
            Prediction dictionary
        """

        print(f"\n{'='*70}")
        print(f"PREDICTING UNKNOWN MODE VIA CATEGORICAL CYCLING")
        print(f"{'='*70}\n")

        known_modes = [self.modes[i] for i in known_mode_indices]
        print(f"Known modes:")
        for mode in known_modes:
            print(f"  {mode}")

        # Calculate average S-entropy of known modes
        avg_s_k = np.mean([m.s_category.s_k for m in known_modes])
        avg_s_t = np.mean([m.s_category.s_t for m in known_modes])
        avg_s_e = np.mean([m.s_category.s_e for m in known_modes])

        print(f"\nAverage S-category of known modes:")
        print(f"  S_k = {avg_s_k:.3f}")
        print(f"  S_t = {avg_s_t:.3f}")
        print(f"  S_e = {avg_s_e:.3f}")

        # Search for frequency that would have S-entropy completing the pattern
        # This is where categorical cycling provides information:
        # The pattern of known modes constrains possible unknown modes

        # For demonstration, search frequency space
        c = 2.99792458e10
        test_frequencies_cm = np.linspace(500, 4000, 1000)

        scores = []
        for freq_cm in test_frequencies_cm:
            freq_hz = freq_cm * c

            # Calculate what S-entropy this frequency would have
            test_s = SEntropyCalculator.from_frequency(
                frequency_hz=freq_hz,
                measurement_count=len(known_modes) + 1,
                time_elapsed=1e-13
            )

            # Score based on categorical compatibility
            # High score = S-entropy that "fits" the pattern of known modes
            s_k_dev = abs(test_s.s_k - avg_s_k)
            s_t_dev = abs(test_s.s_t - avg_s_t)
            s_e_dev = abs(test_s.s_e - avg_s_e)

            deviation = np.sqrt(s_k_dev**2 + s_t_dev**2 + s_e_dev**2)
            score = np.exp(-deviation / 0.5)

            scores.append(score)

        # Find maximum
        best_idx = np.argmax(scores)
        predicted_freq_cm = test_frequencies_cm[best_idx]
        confidence = scores[best_idx]

        print(f"\nPrediction:")
        print(f"  Target symmetry: {target_symmetry}")
        print(f"  Predicted frequency: {predicted_freq_cm:.1f} cm⁻¹")
        print(f"  Confidence: {confidence:.3f}")

        return {
            'predicted_frequency_cm': float(predicted_freq_cm),
            'target_symmetry': target_symmetry,
            'confidence': float(confidence),
            'known_modes': [m.name for m in known_modes]
        }

def demo_water_molecule():
    """
    Demo: Water molecule (H2O) - 3 vibrational modes

    Modes:
    1. Symmetric stretch (ν₁): 3657 cm⁻¹ (A₁ symmetry)
    2. Bending (ν₂): 1595 cm⁻¹ (A₁ symmetry)
    3. Asymmetric stretch (ν₃): 3756 cm⁻¹ (B₁ symmetry)
    """

    print(f"\n{'#'*70}")
    print(f"# DEMO: WATER MOLECULE CATEGORICAL MODE CYCLING")
    print(f"{'#'*70}\n")

    # Create virtual spectrometer for water
    spectrometer = VirtualMolecularSpectrometer('H2O')

    # Add the 3 vibrational modes (categorical states)
    spectrometer.add_mode('symmetric_stretch', 3657, 'A1', intensity=1.0)
    spectrometer.add_mode('bending', 1595, 'A1', intensity=0.8)
    spectrometer.add_mode('asymmetric_stretch', 3756, 'B1', intensity=0.9)

    # Cycle through modes
    moments = spectrometer.cycle_through_modes()

    # Calculate mode-mode correlations
    correlations = spectrometer.calculate_mode_mode_correlations()

    print(f"{'='*70}")
    print(f"MODE-MODE CATEGORICAL CORRELATIONS")
    print(f"{'='*70}\n")
    print("Correlation matrix (higher = more categorically similar):")
    print(correlations)

    # Save results
    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experiment': 'categorical_mode_cycling',
        'molecule': 'H2O',
        'modes': [
            {
                'name': m.mode.name,
                'frequency_cm': m.mode.frequency_cm,
                'symmetry': m.mode.symmetry,
                's_category': {
                    's_k': m.mode.s_category.s_k,
                    's_t': m.mode.s_category.s_t,
                    's_e': m.mode.s_category.s_e
                },
                'categorical_time': m.timestamp_categorical,
                'phase': m.categorical_phase
            }
            for m in moments
        ],
        'correlations': correlations.tolist(),
        'method': 'virtual_spectrometer_cycling'
    }

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f'water_cycling_{results["timestamp"]}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {json_path}")

    return results

def demo_vanillin_prediction():
    """
    Demo: Predict C=O stretch in vanillin by cycling through known modes

    Known modes (cycle through these):
    - O-H stretch: 3400 cm⁻¹
    - C-H aromatic: 3070 cm⁻¹
    - Ring stretch: 1583 cm⁻¹

    Unknown (predict):
    - C=O stretch: ??? (true value: 1666 cm⁻¹)
    """

    print(f"\n{'#'*70}")
    print(f"# DEMO: PREDICT VANILLIN C=O VIA CATEGORICAL CYCLING")
    print(f"{'#'*70}\n")

    # Create virtual spectrometer
    spectrometer = VirtualMolecularSpectrometer('Vanillin')

    # Add known modes
    spectrometer.add_mode('OH_stretch', 3400, 'A', intensity=0.9)
    spectrometer.add_mode('CH_aromatic', 3070, 'A', intensity=1.0)
    spectrometer.add_mode('ring_stretch', 1583, 'A', intensity=0.8)
    spectrometer.add_mode('ring_stretch_2', 1512, 'A', intensity=0.7)

    # Add the true C=O mode (for validation)
    spectrometer.add_mode('CO_stretch_TRUE', 1666, 'A', intensity=1.0)

    # Cycle through known modes (indices 0-3)
    known_indices = [0, 1, 2, 3]
    moments = spectrometer.cycle_through_modes()

    # Predict unknown mode
    prediction = spectrometer.predict_unknown_mode_from_cycling(
        known_mode_indices=known_indices,
        target_symmetry='A'
    )

    # Compare to true value
    true_value = 1666
    error = abs(prediction['predicted_frequency_cm'] - true_value)
    error_percent = 100 * error / true_value

    print(f"\n{'='*70}")
    print(f"VALIDATION")
    print(f"{'='*70}")
    print(f"Predicted: {prediction['predicted_frequency_cm']:.1f} cm⁻¹")
    print(f"True value: {true_value} cm⁻¹")
    print(f"Error: {error:.1f} cm⁻¹ ({error_percent:.2f}%)")

    # Save
    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experiment': 'vanillin_prediction_via_cycling',
        'prediction': prediction,
        'validation': {
            'true_value_cm': true_value,
            'error_cm': float(error),
            'error_percent': float(error_percent)
        }
    }

    output_dir = Path(__file__).parent / 'results'
    json_path = output_dir / f'vanillin_cycling_{results["timestamp"]}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {json_path}")

    return results

def main():
    """Run categorical mode cycling demonstrations"""

    print(f"\n{'='*70}")
    print(f"CATEGORICAL MODE CYCLING")
    print(f"Virtual Spectrometer Through Molecular Vibrational Modes")
    print(f"{'='*70}")
    print(f"\nKey concept:")
    print(f"  - Each vibrational mode = categorical state")
    print(f"  - Virtual spectrometer cycles through modes")
    print(f"  - Materializes only at specific categorical moments")
    print(f"  - Source-target unification: molecule IS spectrometer")
    print(f"{'='*70}\n")

    # Demo 1: Simple molecule (water)
    water_results = demo_water_molecule()

    # Demo 2: Predict unknown mode (vanillin)
    vanillin_results = demo_vanillin_prediction()

    print(f"\n{'='*70}")
    print(f"KEY INSIGHTS")
    print(f"{'='*70}")
    print(f"1. Vibrational modes are categorical states")
    print(f"2. Virtual spectrometer cycles through categories")
    print(f"3. Each mode accessed at different categorical moment")
    print(f"4. Mode-mode correlations reveal categorical structure")
    print(f"5. Unknown modes predicted from cycling pattern")
    print(f"\nThis is the interferometry framework applied to molecules!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
