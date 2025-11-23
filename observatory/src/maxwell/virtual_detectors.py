"""
Virtual Detectors: Hypothesis Validation Through Multiple Modalities
====================================================================

Virtual detectors simulate what different physical instruments WOULD measure
given a hypothesis about molecular composition. They enable cross-validation
without actually running multiple physical experiments.

Key Concept: These don't replace physical detectors - they validate hypotheses
by checking consistency across multiple measurement modalities.

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
h = 6.62607015e-34  # Planck constant (J·s)
c = 299792458  # Speed of light (m/s)
N_A = 6.02214076e23  # Avogadro's number


class VirtualDetector(ABC):
    """
    Base class for virtual detectors

    Each virtual detector:
    1. Predicts signal given hypothesis
    2. Observes molecular demon states
    3. Checks consistency
    """

    def __init__(self, pixel_demon):
        self.pixel_demon = pixel_demon
        self.detector_type = self.__class__.__name__
        self.tolerance = 0.1  # 10% tolerance for consistency

    @abstractmethod
    def predict_signal(self, hypothesis) -> float:
        """Predict what signal this detector SHOULD see given hypothesis"""
        pass

    @abstractmethod
    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """Observe actual molecular demon states"""
        pass

    def is_consistent(self, expected: float, observed: float) -> bool:
        """Check if expected and observed are consistent"""
        if expected == 0:
            return abs(observed) < self.tolerance

        relative_error = abs(expected - observed) / abs(expected)
        return relative_error < self.tolerance

    def get_consistency_score(self, expected: float, observed: float) -> float:
        """Return continuous consistency score (0-1)"""
        if expected == 0:
            return 1.0 - min(abs(observed) / self.tolerance, 1.0)

        relative_error = abs(expected - observed) / abs(expected)
        return max(0.0, 1.0 - relative_error / self.tolerance)


class VirtualThermometer(VirtualDetector):
    """
    Virtual thermometer: Measures temperature from molecular motion
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.05  # 5% tolerance for temperature

    def predict_signal(self, hypothesis) -> float:
        """Predict temperature from hypothesis"""
        return hypothesis.expected_properties.get('temperature_k', 288.15)

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """
        Calculate temperature from molecular kinetic energy

        T = (2/3) * <E_kinetic> / k_B
        Encoded in S_t coordinate
        """
        if not molecular_demons:
            return 288.15  # Default

        # Average S_t across all demons (encodes temperature)
        s_t_values = [demon.s_state.S_t for demon in molecular_demons.values()]
        avg_s_t = np.mean(s_t_values)

        # Convert back to temperature (inverted from initialization)
        temperature_k = (avg_s_t ** 2) * 300.0

        return temperature_k


class VirtualBarometer(VirtualDetector):
    """
    Virtual barometer: Measures pressure from molecular collisions
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.08  # 8% tolerance

    def predict_signal(self, hypothesis) -> float:
        """Predict pressure from hypothesis"""
        return hypothesis.expected_properties.get('pressure_pa', 101325.0)

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """
        Calculate pressure from molecular number density

        P = n * k_B * T
        """
        if not molecular_demons:
            return 101325.0

        # Sum number density across all molecule types
        total_density = sum(
            demon.number_density for demon in molecular_demons.values()
        )

        # Get temperature from S_t
        s_t_values = [demon.s_state.S_t for demon in molecular_demons.values()]
        temperature_k = (np.mean(s_t_values) ** 2) * 300.0

        # Calculate pressure
        pressure_pa = total_density * k_B * temperature_k

        return pressure_pa


class VirtualHygrometer(VirtualDetector):
    """
    Virtual hygrometer: Measures humidity from H₂O content
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.15  # 15% tolerance (humidity is variable)

    def predict_signal(self, hypothesis) -> float:
        """Predict humidity fraction from hypothesis"""
        h2o_fraction = hypothesis.molecular_composition.get('H2O', 0.01)
        return hypothesis.expected_properties.get('humidity_fraction', h2o_fraction * 2.5)

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """
        Calculate humidity from H₂O partial pressure
        """
        if 'H2O' not in molecular_demons:
            return 0.0

        h2o_demon = molecular_demons['H2O']

        # H₂O number density
        n_h2o = h2o_demon.number_density

        # Temperature
        temperature_k = (h2o_demon.s_state.S_t ** 2) * 300.0

        # H₂O partial pressure
        p_h2o = n_h2o * k_B * temperature_k

        # Saturated vapor pressure (Clausius-Clapeyron)
        T_c = temperature_k - 273.15
        p_sat = 611 * np.exp(17.27 * T_c / (T_c + 237.3))

        # Relative humidity
        humidity_fraction = p_h2o / p_sat if p_sat > 0 else 0.0

        return np.clip(humidity_fraction, 0.0, 1.0)


class VirtualIRSpectrometer(VirtualDetector):
    """
    Virtual IR spectrometer: Detects vibrational modes
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.12  # 12% tolerance

    def predict_signal(self, hypothesis) -> float:
        """
        Predict IR absorption from molecular composition

        Returns total absorption (integrated over spectrum)
        """
        # IR-active molecules and their absorption coefficients
        ir_coefficients = {
            'H2O': 1.5,  # Strong IR absorber
            'CO2': 1.0,  # Moderate
            'N2': 0.0,   # IR-inactive (homonuclear)
            'O2': 0.0,   # IR-inactive (homonuclear)
            'Ar': 0.0    # Monoatomic
        }

        total_absorption = 0.0
        for molecule, fraction in hypothesis.molecular_composition.items():
            coeff = ir_coefficients.get(molecule, 0.0)
            total_absorption += fraction * coeff

        return total_absorption

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """
        Calculate IR absorption from vibrational modes
        """
        total_absorption = 0.0

        ir_coefficients = {
            'H2O': 1.5,
            'CO2': 1.0,
            'N2': 0.0,
            'O2': 0.0,
            'Ar': 0.0
        }

        # Sum absorption weighted by number density
        total_density = sum(d.number_density for d in molecular_demons.values())

        if total_density == 0:
            return 0.0

        for molecule, demon in molecular_demons.items():
            coeff = ir_coefficients.get(molecule, 0.0)
            fraction = demon.number_density / total_density
            total_absorption += fraction * coeff

        return total_absorption


class VirtualRamanSpectrometer(VirtualDetector):
    """
    Virtual Raman spectrometer: Detects polarizability changes
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.15

    def predict_signal(self, hypothesis) -> float:
        """
        Predict Raman signal from molecular composition
        """
        # Raman activity (different from IR!)
        raman_coefficients = {
            'N2': 1.0,   # Raman-active (homonuclear)
            'O2': 0.8,   # Raman-active
            'H2O': 0.5,  # Moderate
            'CO2': 0.3,  # Weak (symmetric stretch is Raman-active)
            'Ar': 0.0    # Monoatomic
        }

        total_signal = 0.0
        for molecule, fraction in hypothesis.molecular_composition.items():
            coeff = raman_coefficients.get(molecule, 0.0)
            total_signal += fraction * coeff

        return total_signal

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """Calculate Raman signal from molecular demons"""
        raman_coefficients = {
            'N2': 1.0,
            'O2': 0.8,
            'H2O': 0.5,
            'CO2': 0.3,
            'Ar': 0.0
        }

        total_density = sum(d.number_density for d in molecular_demons.values())
        if total_density == 0:
            return 0.0

        total_signal = 0.0
        for molecule, demon in molecular_demons.items():
            coeff = raman_coefficients.get(molecule, 0.0)
            fraction = demon.number_density / total_density
            total_signal += fraction * coeff

        return total_signal


class VirtualMassSpectrometer(VirtualDetector):
    """
    Virtual mass spectrometer: Detects molecular masses
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.05  # 5% tolerance (mass is precise)

    def predict_signal(self, hypothesis) -> float:
        """
        Predict weighted average mass from composition
        """
        molecular_masses = {
            'N2': 28.014,
            'O2': 31.998,
            'Ar': 39.948,
            'CO2': 44.009,
            'H2O': 18.015
        }

        weighted_mass = 0.0
        for molecule, fraction in hypothesis.molecular_composition.items():
            mass = molecular_masses.get(molecule, 0.0)
            weighted_mass += fraction * mass

        return weighted_mass

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """Calculate weighted average mass"""
        molecular_masses = {
            'N2': 28.014,
            'O2': 31.998,
            'Ar': 39.948,
            'CO2': 44.009,
            'H2O': 18.015
        }

        total_density = sum(d.number_density for d in molecular_demons.values())
        if total_density == 0:
            return 0.0

        weighted_mass = 0.0
        for molecule, demon in molecular_demons.items():
            mass = molecular_masses.get(molecule, 0.0)
            fraction = demon.number_density / total_density
            weighted_mass += fraction * mass

        return weighted_mass


class VirtualPhotodiode(VirtualDetector):
    """
    Virtual photodiode: Measures optical absorption
    """

    def __init__(self, pixel_demon, wavelength_nm: float = 550.0):
        super().__init__(pixel_demon)
        self.wavelength_nm = wavelength_nm
        self.tolerance = 0.10

    def predict_signal(self, hypothesis) -> float:
        """
        Predict optical absorption at specific wavelength
        """
        # Simplified: atmospheric molecules mostly transparent in visible
        # But H₂O, CO₂ have some absorption bands

        absorption_coefficients = {
            'H2O': 0.01,   # Weak visible absorption
            'CO2': 0.005,
            'N2': 0.0,
            'O2': 0.002,   # Very weak (blue sky!)
            'Ar': 0.0
        }

        total_absorption = 0.0
        for molecule, fraction in hypothesis.molecular_composition.items():
            coeff = absorption_coefficients.get(molecule, 0.0)
            total_absorption += fraction * coeff

        return total_absorption

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """Calculate optical absorption"""
        absorption_coefficients = {
            'H2O': 0.01,
            'CO2': 0.005,
            'N2': 0.0,
            'O2': 0.002,
            'Ar': 0.0
        }

        total_density = sum(d.number_density for d in molecular_demons.values())
        if total_density == 0:
            return 0.0

        total_absorption = 0.0
        for molecule, demon in molecular_demons.items():
            coeff = absorption_coefficients.get(molecule, 0.0)
            fraction = demon.number_density / total_density
            total_absorption += fraction * coeff

        return total_absorption


class VirtualInterferometer(VirtualDetector):
    """
    Virtual interferometer: Measures phase coherence
    """

    def __init__(self, pixel_demon):
        super().__init__(pixel_demon)
        self.tolerance = 0.20  # 20% tolerance (phase is sensitive)

    def predict_signal(self, hypothesis) -> float:
        """
        Predict phase coherence from molecular dynamics

        Returns coherence measure (0-1)
        """
        # Coherence related to molecular order
        # More uniform composition = higher coherence

        compositions = list(hypothesis.molecular_composition.values())

        # Calculate entropy of composition distribution
        compositions_arr = np.array(compositions)
        compositions_arr = compositions_arr / np.sum(compositions_arr)

        # Shannon entropy (normalized)
        entropy = -np.sum(compositions_arr * np.log(compositions_arr + 1e-10))
        max_entropy = np.log(len(compositions))

        # Coherence = 1 - normalized_entropy
        coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        return coherence

    def observe_molecular_demons(self, molecular_demons: Dict) -> float:
        """Calculate phase coherence from molecular demon states"""
        if not molecular_demons:
            return 0.0

        # Calculate from S_t (temporal) coordinates
        s_t_values = [demon.s_state.S_t for demon in molecular_demons.values()]

        # Phase coherence = how similar are the temporal states
        variance = np.var(s_t_values)
        coherence = np.exp(-variance)  # High variance = low coherence

        return coherence


class VirtualDetectorFactory:
    """Factory for creating virtual detectors"""

    _detector_classes = {
        'thermometer': VirtualThermometer,
        'barometer': VirtualBarometer,
        'hygrometer': VirtualHygrometer,
        'ir_spectrometer': VirtualIRSpectrometer,
        'raman_spectrometer': VirtualRamanSpectrometer,
        'mass_spectrometer': VirtualMassSpectrometer,
        'photodiode': VirtualPhotodiode,
        'interferometer': VirtualInterferometer
    }

    @classmethod
    def create(cls, detector_type: str, pixel_demon) -> VirtualDetector:
        """Create virtual detector of specified type"""
        detector_class = cls._detector_classes.get(detector_type)

        if detector_class is None:
            raise ValueError(f"Unknown detector type: {detector_type}")

        return detector_class(pixel_demon)

    @classmethod
    def get_available_detectors(cls) -> List[str]:
        """Get list of available detector types"""
        return list(cls._detector_classes.keys())

    @classmethod
    def create_full_suite(cls, pixel_demon) -> Dict[str, VirtualDetector]:
        """Create all available virtual detectors"""
        return {
            name: cls.create(name, pixel_demon)
            for name in cls._detector_classes.keys()
        }


class ConsilienceEngine:
    """
    Consilience Engine: Cross-validates hypotheses using multiple virtual detectors

    The term "consilience" (jumping together) refers to evidence from independent
    sources converging on the same conclusion.
    """

    def __init__(self, pixel_demon):
        self.pixel_demon = pixel_demon
        self.detector_suite = VirtualDetectorFactory.create_full_suite(pixel_demon)

    def validate_hypothesis(
        self,
        hypothesis,
        detector_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate hypothesis using multiple virtual detectors

        Returns detailed validation report
        """
        if detector_subset:
            detectors = {
                name: self.detector_suite[name]
                for name in detector_subset
                if name in self.detector_suite
            }
        else:
            detectors = self.detector_suite

        results = {
            'hypothesis_id': hypothesis.id,
            'hypothesis_description': hypothesis.description,
            'detector_results': {},
            'overall_consistency': 0.0,
            'consistent_detectors': 0,
            'total_detectors': len(detectors)
        }

        consistent_count = 0
        total_score = 0.0

        molecular_demons = self.pixel_demon.molecular_demons

        for name, detector in detectors.items():
            # Predict and observe
            expected = detector.predict_signal(hypothesis)
            observed = detector.observe_molecular_demons(molecular_demons)

            # Check consistency
            is_consistent = detector.is_consistent(expected, observed)
            consistency_score = detector.get_consistency_score(expected, observed)

            if is_consistent:
                consistent_count += 1

            total_score += consistency_score

            results['detector_results'][name] = {
                'expected': float(expected),
                'observed': float(observed),
                'is_consistent': bool(is_consistent),
                'consistency_score': float(consistency_score),
                'status': '✓ MATCH' if is_consistent else '✗ MISMATCH'
            }

        results['consistent_detectors'] = consistent_count
        results['overall_consistency'] = total_score / len(detectors) if detectors else 0.0

        return results

    def find_best_hypothesis(
        self,
        hypotheses: List,
        detector_subset: Optional[List[str]] = None
    ) -> Tuple[Any, Dict]:
        """
        Find hypothesis with highest consilience (cross-detector agreement)

        Returns (best_hypothesis, validation_report)
        """
        validation_reports = []

        for hypothesis in hypotheses:
            report = self.validate_hypothesis(hypothesis, detector_subset)
            validation_reports.append((hypothesis, report))

        # Sort by overall consistency
        validation_reports.sort(key=lambda x: x[1]['overall_consistency'], reverse=True)

        best_hypothesis, best_report = validation_reports[0]

        # Add ranking to report
        best_report['ranking'] = [
            {
                'hypothesis_id': h.id,
                'consistency': r['overall_consistency']
            }
            for h, r in validation_reports
        ]

        return best_hypothesis, best_report
