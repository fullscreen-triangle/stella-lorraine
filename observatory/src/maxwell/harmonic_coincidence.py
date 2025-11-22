"""
Harmonic Coincidence Network: O(1) Information Access via Frequency Resonance
=============================================================================

Build networks where nodes are oscillators (molecules, surfaces, lights) and
edges represent harmonic coincidences (integer frequency ratios).

Key insight: Information transfer is efficient when frequencies are in harmonic
relationship. Use gear ratios for O(1) navigation between scales.

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Oscillator:
    """
    An oscillator in the harmonic network

    Can represent:
    - Molecule (vibrational modes)
    - Surface (resonance frequency)
    - Light source (emission frequency)
    - Any periodic process
    """
    id: str
    frequency_hz: float
    amplitude: float
    phase: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Oscillator) and self.id == other.id


@dataclass
class HarmonicCoincidence:
    """
    A harmonic coincidence between two oscillators

    Coincidence exists when: f1/f2 ≈ n (integer ratio within tolerance)
    """
    osc1: Oscillator
    osc2: Oscillator
    ratio: float  # f1/f2
    nearest_integer: int
    deviation: float  # How far from integer
    coupling_strength: float  # Based on amplitude and proximity to integer

    def __repr__(self):
        return (f"Coincidence({self.osc1.id} <-> {self.osc2.id}, "
                f"ratio={self.ratio:.3f}, n={self.nearest_integer})")


class HarmonicCoincidenceNetwork:
    """
    Network of oscillators connected by harmonic coincidences

    Features:
    - Add oscillators dynamically
    - Find coincidences efficiently
    - Navigate using gear ratios (O(1) complexity)
    - Query by frequency, proximity, or metadata
    """

    def __init__(self, name: str = "harmonic_network"):
        self.name = name
        self.oscillators: Dict[str, Oscillator] = {}
        self.coincidences: List[HarmonicCoincidence] = []

        # Adjacency for fast lookup
        self.adjacency: Dict[str, List[HarmonicCoincidence]] = defaultdict(list)

        # Frequency index for fast queries
        self.frequency_sorted: List[Oscillator] = []

        logger.debug(f"Created HarmonicCoincidenceNetwork '{name}'")

    def add_oscillator(
        self,
        frequency: float,
        amplitude: float,
        phase: float = 0.0,
        oscillator_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Oscillator:
        """Add oscillator to network"""
        if oscillator_id is None:
            oscillator_id = f"osc_{len(self.oscillators)}"

        osc = Oscillator(
            id=oscillator_id,
            frequency_hz=frequency,
            amplitude=amplitude,
            phase=phase,
            metadata=metadata or {}
        )

        self.oscillators[oscillator_id] = osc
        self.frequency_sorted = None  # Invalidate cache

        return osc

    def find_coincidences(
        self,
        tolerance_hz: float = 1e9,
        min_coupling_strength: float = 0.01
    ):
        """
        Find all harmonic coincidences in network

        Args:
            tolerance_hz: Maximum frequency deviation for coincidence
            min_coupling_strength: Minimum coupling strength to keep
        """
        self.coincidences = []
        self.adjacency = defaultdict(list)

        oscillator_list = list(self.oscillators.values())
        n = len(oscillator_list)

        logger.info(f"Finding coincidences among {n} oscillators...")

        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                osc1 = oscillator_list[i]
                osc2 = oscillator_list[j]

                # Check if harmonic coincidence exists
                coincidence = self._check_coincidence(
                    osc1, osc2, tolerance_hz
                )

                if coincidence and coincidence.coupling_strength >= min_coupling_strength:
                    self.coincidences.append(coincidence)
                    self.adjacency[osc1.id].append(coincidence)
                    self.adjacency[osc2.id].append(coincidence)

        logger.info(f"Found {len(self.coincidences)} harmonic coincidences")

    def _check_coincidence(
        self,
        osc1: Oscillator,
        osc2: Oscillator,
        tolerance_hz: float
    ) -> Optional[HarmonicCoincidence]:
        """Check if two oscillators have harmonic coincidence"""
        f1 = osc1.frequency_hz
        f2 = osc2.frequency_hz

        if f1 == 0 or f2 == 0:
            return None

        # Frequency ratio
        ratio = f1 / f2

        # Nearest integer
        nearest_int = round(ratio)

        if nearest_int == 0:
            return None

        # Deviation from integer ratio
        ideal_f1 = nearest_int * f2
        deviation_hz = abs(f1 - ideal_f1)

        # Check if within tolerance
        if deviation_hz > tolerance_hz:
            return None

        # Coupling strength (stronger for closer to integer, higher amplitudes)
        # proximity_factor = exp(-deviation/tolerance)
        proximity_factor = np.exp(-deviation_hz / tolerance_hz)

        # amplitude_factor = sqrt(A1 * A2)
        amplitude_factor = np.sqrt(osc1.amplitude * osc2.amplitude)

        coupling_strength = proximity_factor * amplitude_factor

        return HarmonicCoincidence(
            osc1=osc1,
            osc2=osc2,
            ratio=ratio,
            nearest_integer=nearest_int,
            deviation=deviation_hz,
            coupling_strength=coupling_strength
        )

    def get_neighbors(
        self,
        oscillator_id: str
    ) -> List[Tuple[Oscillator, HarmonicCoincidence]]:
        """Get all oscillators harmonically coupled to given oscillator"""
        if oscillator_id not in self.oscillators:
            return []

        neighbors = []
        for coincidence in self.adjacency[oscillator_id]:
            # Get the other oscillator in the coincidence
            if coincidence.osc1.id == oscillator_id:
                other = coincidence.osc2
            else:
                other = coincidence.osc1

            neighbors.append((other, coincidence))

        return neighbors

    def gear_ratio_transform(
        self,
        from_oscillator_id: str,
        to_oscillator_id: str
    ) -> Optional[float]:
        """
        Calculate gear ratio between two oscillators

        Gear ratio R = f_to / f_from

        This allows O(1) transformation between frequency scales:
        If you know amplitude at f_from, amplitude at f_to = A_from * R
        """
        if from_oscillator_id not in self.oscillators:
            return None
        if to_oscillator_id not in self.oscillators:
            return None

        f_from = self.oscillators[from_oscillator_id].frequency_hz
        f_to = self.oscillators[to_oscillator_id].frequency_hz

        if f_from == 0:
            return None

        return f_to / f_from

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_hops: int = 5
    ) -> Optional[List[str]]:
        """
        Find path through harmonic coincidence network

        Uses BFS to find shortest path of coincidences
        """
        if from_id not in self.oscillators or to_id not in self.oscillators:
            return None

        if from_id == to_id:
            return [from_id]

        # BFS
        queue = [(from_id, [from_id])]
        visited = {from_id}

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            # Check neighbors
            for neighbor_osc, _ in self.get_neighbors(current_id):
                if neighbor_osc.id == to_id:
                    return path + [to_id]

                if neighbor_osc.id not in visited:
                    visited.add(neighbor_osc.id)
                    queue.append((neighbor_osc.id, path + [neighbor_osc.id]))

        return None  # No path found

    def query_by_frequency_range(
        self,
        f_min: float,
        f_max: float
    ) -> List[Oscillator]:
        """Find all oscillators in frequency range"""
        # Build sorted index if needed
        if self.frequency_sorted is None:
            self.frequency_sorted = sorted(
                self.oscillators.values(),
                key=lambda o: o.frequency_hz
            )

        # Binary search for range
        result = []
        for osc in self.frequency_sorted:
            if f_min <= osc.frequency_hz <= f_max:
                result.append(osc)
            elif osc.frequency_hz > f_max:
                break

        return result

    def query_by_metadata(
        self,
        key: str,
        value: Any
    ) -> List[Oscillator]:
        """Find oscillators with specific metadata"""
        result = []
        for osc in self.oscillators.values():
            if key in osc.metadata and osc.metadata[key] == value:
                result.append(osc)
        return result

    def calculate_information_density(
        self,
        frequency_hz: float,
        bandwidth_hz: float = 1e12
    ) -> float:
        """
        Calculate Oscillatory Information Density (OID) at frequency

        OID = Σ A_i * f_i for all oscillators in bandwidth
        """
        oscillators_in_band = self.query_by_frequency_range(
            frequency_hz - bandwidth_hz / 2,
            frequency_hz + bandwidth_hz / 2
        )

        oid = sum(osc.amplitude * osc.frequency_hz for osc in oscillators_in_band)

        return oid

    def find_resonance_clusters(
        self,
        min_cluster_size: int = 3
    ) -> List[List[Oscillator]]:
        """
        Find clusters of mutually resonant oscillators

        A resonance cluster is a set of oscillators all harmonically coupled
        """
        clusters = []
        visited = set()

        for osc_id in self.oscillators:
            if osc_id in visited:
                continue

            # BFS to find connected component
            cluster = []
            queue = [osc_id]
            cluster_visited = {osc_id}

            while queue:
                current_id = queue.pop(0)
                cluster.append(self.oscillators[current_id])
                visited.add(current_id)

                for neighbor_osc, _ in self.get_neighbors(current_id):
                    if neighbor_osc.id not in cluster_visited:
                        cluster_visited.add(neighbor_osc.id)
                        queue.append(neighbor_osc.id)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of network"""
        if not self.oscillators:
            return {
                'num_oscillators': 0,
                'num_coincidences': 0
            }

        frequencies = [osc.frequency_hz for osc in self.oscillators.values()]
        amplitudes = [osc.amplitude for osc in self.oscillators.values()]
        coupling_strengths = [c.coupling_strength for c in self.coincidences]

        # Calculate network connectivity
        degrees = [len(self.adjacency[osc_id]) for osc_id in self.oscillators]

        return {
            'name': self.name,
            'num_oscillators': len(self.oscillators),
            'num_coincidences': len(self.coincidences),
            'frequency_range_hz': (min(frequencies), max(frequencies)),
            'mean_amplitude': np.mean(amplitudes),
            'mean_coupling_strength': np.mean(coupling_strengths) if coupling_strengths else 0.0,
            'mean_degree': np.mean(degrees) if degrees else 0.0,
            'max_degree': max(degrees) if degrees else 0,
            'network_density': len(self.coincidences) / (len(self.oscillators) * (len(self.oscillators) - 1) / 2)
                               if len(self.oscillators) > 1 else 0.0
        }


class MolecularHarmonicNetwork(HarmonicCoincidenceNetwork):
    """
    Specialized harmonic network for molecules

    Adds molecule-specific features like vibrational mode analysis
    """

    def __init__(self, name: str = "molecular_network"):
        super().__init__(name)

    def add_molecule(
        self,
        molecule_type: str,
        vibrational_modes: List[float],
        number_density: float,
        position: Optional[np.ndarray] = None
    ) -> List[Oscillator]:
        """
        Add molecule with all its vibrational modes

        Each mode becomes an oscillator in the network
        """
        oscillators = []

        for i, freq in enumerate(vibrational_modes):
            osc_id = f"{molecule_type}_mode_{i}"

            # Amplitude based on number density
            amplitude = np.sqrt(number_density)

            metadata = {
                'molecule_type': molecule_type,
                'mode_index': i,
                'number_density': number_density
            }

            if position is not None:
                metadata['position'] = position.tolist()

            osc = self.add_oscillator(
                frequency=freq,
                amplitude=amplitude,
                oscillator_id=osc_id,
                metadata=metadata
            )

            oscillators.append(osc)

        logger.debug(
            f"Added molecule {molecule_type} with {len(vibrational_modes)} modes"
        )

        return oscillators

    def find_molecular_resonances(
        self,
        molecule1: str,
        molecule2: str
    ) -> List[HarmonicCoincidence]:
        """
        Find resonances between two molecule types
        """
        # Get oscillators for each molecule
        osc1_list = self.query_by_metadata('molecule_type', molecule1)
        osc2_list = self.query_by_metadata('molecule_type', molecule2)

        resonances = []

        for coincidence in self.coincidences:
            in_mol1 = coincidence.osc1 in osc1_list or coincidence.osc2 in osc1_list
            in_mol2 = coincidence.osc1 in osc2_list or coincidence.osc2 in osc2_list

            if in_mol1 and in_mol2:
                resonances.append(coincidence)

        return resonances


def build_atmospheric_harmonic_network(
    temperature_k: float = 288.15,
    pressure_pa: float = 101325.0,
    humidity_fraction: float = 0.5
) -> MolecularHarmonicNetwork:
    """
    Build harmonic coincidence network for atmospheric molecules
    """
    network = MolecularHarmonicNetwork(name="atmospheric_network")

    k_B = 1.380649e-23
    n_total = pressure_pa / (k_B * temperature_k)

    # Atmospheric composition with vibrational modes
    molecules = {
        'N2': {
            'fraction': 0.7808,
            'modes': [7.013e13]  # 2330 cm⁻¹
        },
        'O2': {
            'fraction': 0.2095,
            'modes': [4.738e13]  # 1580 cm⁻¹
        },
        'CO2': {
            'fraction': 0.0004,
            'modes': [7.046e13, 3.996e13, 6.963e13]  # 2349, 1333, 2349 cm⁻¹
        },
        'H2O': {
            'fraction': humidity_fraction * 0.04,
            'modes': [1.121e14, 4.708e13, 1.126e14]  # 3756, 1595, 3657 cm⁻¹
        }
    }

    # Add molecules to network
    for mol_type, props in molecules.items():
        n_molecule = props['fraction'] * n_total
        network.add_molecule(
            molecule_type=mol_type,
            vibrational_modes=props['modes'],
            number_density=n_molecule
        )

    # Find coincidences
    network.find_coincidences(tolerance_hz=1e12)

    logger.info(f"Built atmospheric network: {network.get_summary()}")

    return network
