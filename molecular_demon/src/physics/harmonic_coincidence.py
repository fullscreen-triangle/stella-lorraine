"""
Harmonic Coincidence Detection

Detects when harmonics of different molecules coincide in frequency:
n₁·ω₁ ≈ n₂·ω₂

These coincidences form the edges of the harmonic network graph,
enabling beat frequency precision enhancement and reflectance propagation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HarmonicCoincidence:
    """
    Record of a detected harmonic coincidence

    When the n₁-th harmonic of molecule i coincides with the n₂-th
    harmonic of molecule j, this creates a graph edge.
    """
    molecule_i: int
    molecule_j: int
    harmonic_order_i: int
    harmonic_order_j: int
    frequency_i_hz: float
    frequency_j_hz: float
    beat_frequency_hz: float  # |f_i - f_j|
    coincidence_quality: float  # 1 / |f_i - f_j|

    def __post_init__(self):
        """Validate coincidence"""
        if self.beat_frequency_hz < 0:
            raise ValueError("Beat frequency must be non-negative")


class HarmonicCoincidenceDetector:
    """
    Detect harmonic coincidences between molecular oscillators

    Algorithm:
    1. Generate harmonic series for each molecule
    2. Find pairs where harmonics match within threshold
    3. Calculate beat frequencies
    4. Rank by coincidence quality
    """

    def __init__(self, threshold_hz: float = 1e6, max_harmonics: int = 150):
        """
        Initialize detector

        Args:
            threshold_hz: Maximum frequency difference for coincidence (default 1 MHz)
            max_harmonics: Maximum harmonic order to check
        """
        self.threshold = threshold_hz
        self.max_harmonics = max_harmonics
        self.coincidences: List[HarmonicCoincidence] = []

    def detect_coincidence_pair(self,
                               freq_i: float,
                               freq_j: float) -> List[HarmonicCoincidence]:
        """
        Detect all harmonic coincidences between two molecules

        Args:
            freq_i: Fundamental frequency of molecule i (Hz)
            freq_j: Fundamental frequency of molecule j (Hz)

        Returns:
            List of detected coincidences
        """
        coincidences = []

        # Generate harmonic series
        harmonics_i = freq_i * np.arange(1, self.max_harmonics + 1)
        harmonics_j = freq_j * np.arange(1, self.max_harmonics + 1)

        # Check all pairs
        for n_i, f_i in enumerate(harmonics_i, start=1):
            for n_j, f_j in enumerate(harmonics_j, start=1):

                # Beat frequency
                beat_freq = abs(f_i - f_j)

                # Check if within threshold
                if beat_freq < self.threshold:
                    quality = 1.0 / (beat_freq + 1e-10)  # Higher = better match

                    coincidences.append(HarmonicCoincidence(
                        molecule_i=-1,  # Filled by caller
                        molecule_j=-1,  # Filled by caller
                        harmonic_order_i=n_i,
                        harmonic_order_j=n_j,
                        frequency_i_hz=f_i,
                        frequency_j_hz=f_j,
                        beat_frequency_hz=beat_freq,
                        coincidence_quality=quality
                    ))

        return coincidences

    def detect_all_coincidences(self,
                               molecules: List[Tuple[int, float]],
                               progress_callback=None) -> List[HarmonicCoincidence]:
        """
        Detect all harmonic coincidences in an ensemble

        Args:
            molecules: List of (id, frequency) tuples
            progress_callback: Optional function(current, total) for progress

        Returns:
            List of all detected coincidences
        """
        logger.info(f"Detecting harmonic coincidences for {len(molecules)} molecules...")
        logger.info(f"Threshold: {self.threshold:.2e} Hz")
        logger.info(f"Max harmonics: {self.max_harmonics}")

        n_molecules = len(molecules)
        total_pairs = n_molecules * (n_molecules - 1) // 2

        all_coincidences = []
        pair_count = 0

        for i, (id_i, freq_i) in enumerate(molecules):
            for j, (id_j, freq_j) in enumerate(molecules[i+1:], start=i+1):
                pair_count += 1

                if progress_callback and pair_count % 10000 == 0:
                    progress_callback(pair_count, total_pairs)

                # Detect coincidences for this pair
                coincidences = self.detect_coincidence_pair(freq_i, freq_j)

                # Fill in molecule IDs
                for coin in coincidences:
                    coin.molecule_i = id_i
                    coin.molecule_j = id_j

                all_coincidences.extend(coincidences)

        logger.info(f"Found {len(all_coincidences)} total harmonic coincidences")
        logger.info(f"Average coincidences per pair: {len(all_coincidences) / total_pairs:.2f}")

        self.coincidences = all_coincidences
        return all_coincidences

    def get_best_coincidence_per_pair(self) -> List[HarmonicCoincidence]:
        """
        Get the best (smallest beat frequency) coincidence for each molecule pair

        Returns:
            List of best coincidences, one per pair
        """
        if not self.coincidences:
            return []

        # Group by molecule pair
        pair_dict = {}
        for coin in self.coincidences:
            pair_key = (min(coin.molecule_i, coin.molecule_j),
                       max(coin.molecule_i, coin.molecule_j))

            if pair_key not in pair_dict:
                pair_dict[pair_key] = coin
            elif coin.beat_frequency_hz < pair_dict[pair_key].beat_frequency_hz:
                pair_dict[pair_key] = coin

        return list(pair_dict.values())

    def statistics(self) -> dict:
        """
        Calculate statistics on detected coincidences

        Returns:
            Dict with statistics
        """
        if not self.coincidences:
            return {}

        beat_freqs = [c.beat_frequency_hz for c in self.coincidences]
        qualities = [c.coincidence_quality for c in self.coincidences]

        return {
            'total_coincidences': len(self.coincidences),
            'mean_beat_frequency_hz': np.mean(beat_freqs),
            'std_beat_frequency_hz': np.std(beat_freqs),
            'min_beat_frequency_hz': np.min(beat_freqs),
            'max_beat_frequency_hz': np.max(beat_freqs),
            'mean_quality': np.mean(qualities),
            'best_quality': np.max(qualities)
        }


def calculate_beat_frequency_precision(beat_freq_hz: float,
                                      base_freq_hz: float) -> float:
    """
    Calculate precision enhancement from beat frequency

    Beat frequency analysis enables sub-cycle resolution:
    Precision_beat = (f_base / f_beat) × Precision_base

    Args:
        beat_freq_hz: Beat frequency between harmonics
        base_freq_hz: Base oscillation frequency

    Returns:
        Precision enhancement factor
    """
    if beat_freq_hz == 0:
        return np.inf

    enhancement = base_freq_hz / beat_freq_hz
    return enhancement


def find_coincidence_chains(coincidences: List[HarmonicCoincidence],
                            max_chain_length: int = 10) -> List[List[int]]:
    """
    Find chains of coincidences through the network

    A chain: mol_1 → mol_2 → mol_3 → ... → mol_n
    where each consecutive pair has a harmonic coincidence.

    These chains form the reflectance cascade paths.

    Args:
        coincidences: List of detected coincidences
        max_chain_length: Maximum chain length to find

    Returns:
        List of chains (each chain is list of molecule IDs)
    """
    if not coincidences:
        return []

    # Build adjacency list
    adjacency = {}
    for coin in coincidences:
        if coin.molecule_i not in adjacency:
            adjacency[coin.molecule_i] = []
        if coin.molecule_j not in adjacency:
            adjacency[coin.molecule_j] = []

        adjacency[coin.molecule_i].append(coin.molecule_j)
        adjacency[coin.molecule_j].append(coin.molecule_i)

    # Find chains using DFS
    chains = []
    visited_starts = set()

    for start_mol in adjacency:
        if start_mol in visited_starts:
            continue

        # DFS from this start
        stack = [(start_mol, [start_mol], {start_mol})]

        while stack:
            current, path, visited = stack.pop()

            if len(path) >= max_chain_length:
                chains.append(path)
                visited_starts.add(start_mol)
                continue

            if current not in adjacency:
                continue

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}
                    stack.append((neighbor, new_path, new_visited))

            # Also save current path if it's long enough
            if len(path) >= 3:
                chains.append(path)

    # Remove duplicates and sort by length
    unique_chains = []
    seen = set()
    for chain in chains:
        chain_tuple = tuple(sorted(chain))
        if chain_tuple not in seen:
            seen.add(chain_tuple)
            unique_chains.append(chain)

    unique_chains.sort(key=len, reverse=True)

    logger.info(f"Found {len(unique_chains)} coincidence chains")
    if unique_chains:
        logger.info(f"Longest chain: {len(unique_chains[0])} molecules")

    return unique_chains
