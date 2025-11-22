"""
Pixel Maxwell Demon: Spatial Observer with Molecular Demon Lattice
==================================================================

A Pixel Maxwell Demon is a categorical observer at a specific spatial location
that manages a lattice of Molecular Demons (one per molecule type) and uses
Virtual Detectors to validate hypotheses about what's at that location.

Key Concepts:
- Spatial location in physical space
- S-entropy state in categorical space
- Molecular demon lattice (O₂, N₂, H₂O, etc.)
- On-demand virtual detectors for hypothesis validation
- Consilience engine for disambiguation

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class SEntropyCoordinates:
    """
    Categorical state coordinates (orthogonal to physical space)

    S_k: Knowledge entropy (what is known about this state)
    S_t: Temporal entropy (temporal accessibility)
    S_e: Evolution entropy (thermodynamic/evolutionary state)
    """
    S_k: float
    S_t: float
    S_e: float

    def distance_to(self, other: 'SEntropyCoordinates') -> float:
        """Categorical distance (NOT Euclidean distance!)"""
        return np.sqrt(
            (self.S_k - other.S_k)**2 +
            (self.S_t - other.S_t)**2 +
            (self.S_e - other.S_e)**2
        )

    def to_dict(self) -> Dict[str, float]:
        return {'S_k': self.S_k, 'S_t': self.S_t, 'S_e': self.S_e}

    @classmethod
    def from_physical_state(
        cls,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        energy: Optional[float] = None
    ) -> 'SEntropyCoordinates':
        """Convert physical state to categorical coordinates"""
        # S_k: Spatial information (encoded in position)
        S_k = np.log(1 + np.linalg.norm(position))

        # S_t: Temporal information (encoded in velocity)
        if velocity is not None:
            S_t = np.linalg.norm(velocity) / 343.0  # Normalize to speed of sound
        else:
            S_t = 0.0

        # S_e: Energy state
        if energy is not None:
            S_e = np.log(1 + energy)
        else:
            S_e = 0.0

        return cls(S_k=S_k, S_t=S_t, S_e=S_e)


@dataclass
class MolecularDemon:
    """
    Molecular Maxwell Demon (BMD) - information catalyst

    Each molecule type (O₂, N₂, H₂O, etc.) has its own demon
    that filters information and reports state.
    """
    molecule_type: str  # 'O2', 'N2', 'H2O', 'CO2', etc.
    s_state: SEntropyCoordinates
    vibrational_modes: List[float]  # Frequencies in Hz
    number_density: float  # molecules/m³

    def input_filter(self, observations: List[Any]) -> List[Any]:
        """
        BMD input filter: Select which observations to process
        (Y↓ → Y↑ from Mizraji framework)
        """
        # Filter observations based on categorical proximity
        filtered = []
        for obs in observations:
            if hasattr(obs, 's_state'):
                distance = self.s_state.distance_to(obs.s_state)
                if distance < 1.0:  # Categorical threshold
                    filtered.append(obs)
        return filtered

    def output_filter(self, states: List[Any]) -> Dict[str, Any]:
        """
        BMD output filter: Select which information to report
        (Z↓ → Z↑ from Mizraji framework)
        """
        return {
            'molecule_type': self.molecule_type,
            's_state': self.s_state.to_dict(),
            'vibrational_modes': self.vibrational_modes,
            'number_density': self.number_density
        }


@dataclass
class Hypothesis:
    """
    A hypothesis about what's at a pixel location
    """
    id: str
    description: str
    molecular_composition: Dict[str, float]  # {molecule: concentration}
    expected_properties: Dict[str, Any]
    confidence: float = 0.0
    evidence: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'description': self.description,
            'molecular_composition': self.molecular_composition,
            'expected_properties': self.expected_properties,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


class PixelMaxwellDemon:
    """
    Pixel Maxwell Demon: Categorical observer at a spatial location

    Manages:
    - Spatial location (physical space)
    - S-entropy state (categorical space)
    - Molecular demon lattice (one per molecule type)
    - Virtual detectors (on-demand for hypothesis testing)
    - Consilience engine (cross-validation)
    """

    def __init__(
        self,
        position: np.ndarray,
        pixel_id: Optional[str] = None
    ):
        self.position = position  # Physical location [x, y, z]
        self.pixel_id = pixel_id or f"pixel_{id(self)}"

        # Categorical state
        self.s_state = SEntropyCoordinates.from_physical_state(position)

        # Molecular demon lattice
        self.molecular_demons: Dict[str, MolecularDemon] = {}

        # Hypothesis space
        self.hypotheses: List[Hypothesis] = []
        self.current_interpretation: Optional[Hypothesis] = None

        # Virtual detector cache (for efficiency)
        self.virtual_detector_cache: Dict[str, Any] = {}

        logger.debug(f"Created PixelMaxwellDemon at position {position}")

    def add_molecular_demon(self, demon: MolecularDemon):
        """Add a molecular demon to this pixel's lattice"""
        self.molecular_demons[demon.molecule_type] = demon
        logger.debug(f"Added {demon.molecule_type} demon to pixel {self.pixel_id}")

    def initialize_atmospheric_lattice(
        self,
        temperature_k: float = 288.15,
        pressure_pa: float = 101325.0,
        humidity_fraction: float = 0.5
    ):
        """
        Initialize molecular demon lattice for atmospheric composition
        (O₂, N₂, H₂O, CO₂, Ar)
        """
        # Standard atmospheric composition
        k_B = 1.380649e-23

        # Total number density
        n_total = pressure_pa / (k_B * temperature_k)

        # Dry air composition
        compositions = {
            'N2': 0.7808,
            'O2': 0.2095,
            'Ar': 0.0093,
            'CO2': 0.0004
        }

        # Water vapor (adjust for humidity)
        p_sat = 611 * np.exp(17.27 * (temperature_k - 273.15) /
                             ((temperature_k - 273.15) + 237.3))
        p_vapor = humidity_fraction * p_sat
        p_dry = pressure_pa - p_vapor

        # Create molecular demons
        for molecule, fraction in compositions.items():
            n_molecule = (p_dry / pressure_pa) * fraction * n_total

            # Vibrational modes (simplified)
            modes = self._get_vibrational_modes(molecule)

            demon = MolecularDemon(
                molecule_type=molecule,
                s_state=SEntropyCoordinates(
                    S_k=np.log(n_molecule) / np.log(n_total),
                    S_t=np.sqrt(temperature_k / 300.0),
                    S_e=fraction
                ),
                vibrational_modes=modes,
                number_density=n_molecule
            )

            self.add_molecular_demon(demon)

        # Add H2O demon
        n_h2o = p_vapor / (k_B * temperature_k)
        h2o_demon = MolecularDemon(
            molecule_type='H2O',
            s_state=SEntropyCoordinates(
                S_k=np.log(1 + n_h2o) / np.log(n_total),
                S_t=np.sqrt(temperature_k / 300.0),
                S_e=humidity_fraction
            ),
            vibrational_modes=self._get_vibrational_modes('H2O'),
            number_density=n_h2o
        )
        self.add_molecular_demon(h2o_demon)

        logger.info(f"Initialized atmospheric lattice with {len(self.molecular_demons)} demons")

    def _get_vibrational_modes(self, molecule: str) -> List[float]:
        """Get characteristic vibrational frequencies for molecule"""
        # Simplified vibrational modes (in Hz)
        modes_db = {
            'O2': [4.738e13],  # 1580 cm⁻¹
            'N2': [7.013e13],  # 2330 cm⁻¹
            'H2O': [1.121e14, 4.708e13, 1.126e14],  # 3756, 1595, 3657 cm⁻¹
            'CO2': [7.046e13, 3.996e13, 6.963e13],  # 2349, 1333, 2349 cm⁻¹
            'Ar': []  # Monoatomic, no vibrations
        }
        return modes_db.get(molecule, [])

    def generate_hypotheses(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Hypothesis]:
        """
        Generate hypotheses about what's at this pixel

        Context can include:
        - Physical measurements (temperature, pressure, etc.)
        - Neighboring pixel states
        - Prior knowledge about system
        """
        self.hypotheses = []

        # Hypothesis 1: Standard atmospheric composition
        h1 = Hypothesis(
            id='H1_standard_atmosphere',
            description='Standard atmospheric composition',
            molecular_composition={
                'N2': 0.7808,
                'O2': 0.2095,
                'Ar': 0.0093,
                'CO2': 0.0004,
                'H2O': 0.01  # Variable
            },
            expected_properties={
                'temperature_k': 288.15,
                'pressure_pa': 101325.0,
                'density_kg_m3': 1.225
            }
        )
        self.hypotheses.append(h1)

        # Hypothesis 2: High humidity atmosphere
        h2 = Hypothesis(
            id='H2_humid_atmosphere',
            description='High humidity atmospheric composition',
            molecular_composition={
                'N2': 0.75,
                'O2': 0.20,
                'Ar': 0.009,
                'CO2': 0.0004,
                'H2O': 0.04  # High humidity
            },
            expected_properties={
                'temperature_k': 293.15,
                'pressure_pa': 101325.0,
                'humidity_fraction': 0.8
            }
        )
        self.hypotheses.append(h2)

        # Hypothesis 3: Low pressure / high altitude
        h3 = Hypothesis(
            id='H3_low_pressure',
            description='Low pressure (high altitude) atmosphere',
            molecular_composition={
                'N2': 0.7808,
                'O2': 0.2095,
                'Ar': 0.0093,
                'CO2': 0.0004,
                'H2O': 0.005  # Low humidity
            },
            expected_properties={
                'temperature_k': 268.15,
                'pressure_pa': 70000.0,
                'density_kg_m3': 0.9
            }
        )
        self.hypotheses.append(h3)

        # Add context-specific hypotheses
        if context:
            if 'physical_measurement' in context:
                # Generate hypothesis matching physical measurement
                h_measured = self._hypothesis_from_measurement(
                    context['physical_measurement']
                )
                if h_measured:
                    self.hypotheses.append(h_measured)

        logger.info(f"Generated {len(self.hypotheses)} hypotheses for pixel {self.pixel_id}")
        return self.hypotheses

    def _hypothesis_from_measurement(
        self,
        measurement: Dict[str, float]
    ) -> Optional[Hypothesis]:
        """Generate hypothesis from physical measurement"""
        if 'temperature_k' not in measurement:
            return None

        return Hypothesis(
            id='H_measured',
            description='Hypothesis matching physical measurement',
            molecular_composition={
                'N2': 0.7808,
                'O2': 0.2095,
                'Ar': 0.0093,
                'CO2': 0.0004,
                'H2O': measurement.get('humidity_fraction', 0.5) * 0.04
            },
            expected_properties=measurement
        )

    def validate_hypothesis_with_virtual_detectors(
        self,
        hypothesis: Hypothesis,
        detector_types: List[str]
    ) -> Tuple[float, Dict[str, str]]:
        """
        Test hypothesis using virtual detectors

        Returns:
            (consistency_score, evidence_dict)
        """
        from .virtual_detectors import VirtualDetectorFactory

        consistent_count = 0
        total_count = len(detector_types)
        evidence = {}

        for detector_type in detector_types:
            # Get or create virtual detector
            detector = VirtualDetectorFactory.create(detector_type, self)

            # Predict what this detector SHOULD see given hypothesis
            expected = detector.predict_signal(hypothesis)

            # Check consistency with molecular demon states
            observed = detector.observe_molecular_demons(self.molecular_demons)

            # Compare
            is_consistent = detector.is_consistent(expected, observed)

            if is_consistent:
                consistent_count += 1
                evidence[detector_type] = f"✓ MATCH ({expected:.3f} ≈ {observed:.3f})"
            else:
                evidence[detector_type] = f"✗ MISMATCH ({expected:.3f} ≠ {observed:.3f})"

        consistency_score = consistent_count / total_count if total_count > 0 else 0.0

        return consistency_score, evidence

    def find_best_interpretation(
        self,
        detector_types: Optional[List[str]] = None
    ) -> Hypothesis:
        """
        Find most consistent hypothesis using consilience engine
        """
        if not self.hypotheses:
            self.generate_hypotheses()

        if detector_types is None:
            # Use standard atmospheric detectors
            detector_types = [
                'thermometer',
                'barometer',
                'hygrometer',
                'ir_spectrometer',
                'raman_spectrometer'
            ]

        # Test each hypothesis
        for hypothesis in self.hypotheses:
            score, evidence = self.validate_hypothesis_with_virtual_detectors(
                hypothesis,
                detector_types
            )
            hypothesis.confidence = score
            hypothesis.evidence = evidence

        # Find winner
        best = max(self.hypotheses, key=lambda h: h.confidence)
        self.current_interpretation = best

        logger.info(
            f"Best interpretation: {best.description} "
            f"(confidence: {best.confidence:.2%})"
        )

        return best

    def query_categorical_space(
        self,
        query_type: str,
        **kwargs
    ) -> Any:
        """
        Query categorical space for information

        Query types:
        - 'nearest_molecules': Find molecules categorically close
        - 'harmonic_coincidence': Find molecules with frequency resonance
        - 'information_density': Calculate local information density
        """
        if query_type == 'nearest_molecules':
            threshold = kwargs.get('threshold', 1.0)
            nearest = []
            for demon in self.molecular_demons.values():
                dist = self.s_state.distance_to(demon.s_state)
                if dist < threshold:
                    nearest.append((demon, dist))
            return sorted(nearest, key=lambda x: x[1])

        elif query_type == 'information_density':
            # Calculate OID (Oscillatory Information Density)
            total_info = 0.0
            for demon in self.molecular_demons.values():
                # Information from vibrational modes
                for freq in demon.vibrational_modes:
                    total_info += freq * demon.number_density
            return total_info

        else:
            logger.warning(f"Unknown query type: {query_type}")
            return None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of pixel demon state"""
        return {
            'pixel_id': self.pixel_id,
            'position': self.position.tolist(),
            's_state': self.s_state.to_dict(),
            'num_molecular_demons': len(self.molecular_demons),
            'molecular_types': list(self.molecular_demons.keys()),
            'num_hypotheses': len(self.hypotheses),
            'current_interpretation': (
                self.current_interpretation.to_dict()
                if self.current_interpretation else None
            )
        }


class PixelDemonGrid:
    """
    Grid of Pixel Maxwell Demons for imaging/sensing applications
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        physical_extent: Tuple[float, ...],
        name: str = "pixel_grid"
    ):
        self.shape = shape
        self.physical_extent = physical_extent
        self.name = name

        # Create grid of pixel demons
        self.demons = self._initialize_grid()

        logger.info(
            f"Created PixelDemonGrid '{name}' with shape {shape}, "
            f"physical extent {physical_extent}"
        )

    def _initialize_grid(self) -> np.ndarray:
        """Initialize grid of pixel demons"""
        if len(self.shape) == 2:
            # 2D grid
            ny, nx = self.shape
            ey, ex = self.physical_extent

            demons = np.empty(self.shape, dtype=object)

            for iy in range(ny):
                for ix in range(nx):
                    # Physical position
                    x = (ix / nx) * ex
                    y = (iy / ny) * ey
                    position = np.array([x, y, 0.0])

                    # Create demon
                    demons[iy, ix] = PixelMaxwellDemon(
                        position=position,
                        pixel_id=f"{self.name}_y{iy}_x{ix}"
                    )

            return demons

        elif len(self.shape) == 3:
            # 3D grid
            nz, ny, nx = self.shape
            ez, ey, ex = self.physical_extent

            demons = np.empty(self.shape, dtype=object)

            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        x = (ix / nx) * ex
                        y = (iy / ny) * ey
                        z = (iz / nz) * ez
                        position = np.array([x, y, z])

                        demons[iz, iy, ix] = PixelMaxwellDemon(
                            position=position,
                            pixel_id=f"{self.name}_z{iz}_y{iy}_x{ix}"
                        )

            return demons

        else:
            raise ValueError(f"Unsupported grid shape: {self.shape}")

    def initialize_all_atmospheric(
        self,
        temperature_k: float = 288.15,
        pressure_pa: float = 101325.0,
        humidity_fraction: float = 0.5
    ):
        """Initialize atmospheric lattice for all pixel demons"""
        flat_demons = self.demons.flatten()
        for demon in flat_demons:
            demon.initialize_atmospheric_lattice(
                temperature_k, pressure_pa, humidity_fraction
            )
        logger.info(f"Initialized atmospheric lattice for all {len(flat_demons)} demons")

    def interpret_all_pixels(
        self,
        detector_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Run consilience engine on all pixels

        Returns array of confidence scores
        """
        confidence_map = np.zeros(self.shape)

        flat_demons = self.demons.flatten()
        flat_confidence = confidence_map.flatten()

        for i, demon in enumerate(flat_demons):
            best = demon.find_best_interpretation(detector_types)
            flat_confidence[i] = best.confidence

        return confidence_map.reshape(self.shape)
