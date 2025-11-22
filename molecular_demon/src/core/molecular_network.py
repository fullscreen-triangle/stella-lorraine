"""
Harmonic Network Graph Construction
Builds graph where nodes = molecular oscillators, edges = frequency coincidence

This implements the core network structure from the trans-Planckian experiment:
- 260,000 nodes (molecular oscillators)
- 25.8 million edges (harmonic coincidences)
- Average degree ~198
- Graph enhancement factor ~7,176×
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MolecularOscillator:
    """
    Single molecular oscillator with frequency and categorical coordinates

    Identity: Oscillator ≡ Processor ≡ Maxwell Demon ≡ Spectrometer
    """
    id: int
    species: str  # 'N2', 'O2', 'H+', etc.
    frequency_hz: float
    phase_rad: float
    s_coordinates: Tuple[float, float, float]  # (S_k, S_t, S_e)
    harmonics_cached: Optional[np.ndarray] = field(default=None, repr=False)

    def harmonics(self, max_n: int = 150) -> np.ndarray:
        """
        Generate harmonic series: n × ω₀

        Higher harmonics enable sub-cycle precision through beat frequencies.
        Cached for performance since this is called frequently.
        """
        if self.harmonics_cached is None or len(self.harmonics_cached) != max_n:
            self.harmonics_cached = self.frequency_hz * np.arange(1, max_n + 1)
        return self.harmonics_cached

    def categorical_distance(self, other: 'MolecularOscillator') -> float:
        """
        Distance in S-entropy space

        Note: d_cat ⊥ d_spatial (categorical distance independent of physical distance)
        """
        s1 = np.array(self.s_coordinates)
        s2 = np.array(other.s_coordinates)
        return np.linalg.norm(s1 - s2)


class HarmonicNetworkGraph:
    """
    Construct graph of molecular oscillators with edges at frequency coincidences

    Key principle: When n₁·ω₁ ≈ n₂·ω₂, nodes are connected.
    These coincidences enable:
    1. Cross-validation of frequency measurements
    2. Beat frequency precision enhancement
    3. Reflectance cascade information propagation

    The graph topology directly determines precision enhancement factor.
    """

    def __init__(self,
                 molecules: List[MolecularOscillator],
                 coincidence_threshold_hz: float = 1e6,
                 max_harmonics: int = 150):
        """
        Initialize network builder

        Args:
            molecules: List of molecular oscillators
            coincidence_threshold_hz: Maximum frequency difference for edge (default 1 MHz)
            max_harmonics: Maximum harmonic order to check (default 150)
        """
        self.molecules = molecules
        self.threshold = coincidence_threshold_hz
        self.max_harmonics = max_harmonics
        self.graph = nx.Graph()

    def build_graph(self, progress_callback=None) -> nx.Graph:
        """
        Build complete harmonic network graph

        Returns graph with structure matching experimental data:
        - Nodes: molecular oscillators
        - Edges: harmonic coincidences
        - Edge attributes: harmonic orders, beat frequencies

        Args:
            progress_callback: Optional function(current, total) for progress tracking
        """
        logger.info(f"Building harmonic network for {len(self.molecules)} molecules...")
        logger.info(f"Coincidence threshold: {self.threshold:.2e} Hz")
        logger.info(f"Max harmonics: {self.max_harmonics}")

        # Add all nodes
        for mol in self.molecules:
            self.graph.add_node(
                mol.id,
                frequency=mol.frequency_hz,
                species=mol.species,
                s_coords=mol.s_coordinates,
                phase=mol.phase_rad
            )

        # Find harmonic coincidences
        total_pairs = len(self.molecules) * (len(self.molecules) - 1) // 2
        pair_count = 0
        edge_count = 0

        for i, mol_i in enumerate(self.molecules):
            harmonics_i = mol_i.harmonics(max_n=self.max_harmonics)

            for j, mol_j in enumerate(self.molecules[i+1:], start=i+1):
                pair_count += 1

                if progress_callback and pair_count % 10000 == 0:
                    progress_callback(pair_count, total_pairs)

                harmonics_j = mol_j.harmonics(max_n=self.max_harmonics)

                # Check all harmonic pairs for coincidence
                coincidences = []
                for n_i, freq_i in enumerate(harmonics_i, start=1):
                    for n_j, freq_j in enumerate(harmonics_j, start=1):

                        if abs(freq_i - freq_j) < self.threshold:
                            coincidences.append({
                                'harmonic_i': n_i,
                                'harmonic_j': n_j,
                                'frequency': (freq_i + freq_j) / 2,
                                'beat_frequency': abs(freq_i - freq_j)
                            })

                # Add edge if any coincidences found
                if coincidences:
                    # Use the coincidence with smallest beat frequency (best match)
                    best_coincidence = min(coincidences, key=lambda x: x['beat_frequency'])

                    self.graph.add_edge(
                        mol_i.id,
                        mol_j.id,
                        **best_coincidence,
                        n_coincidences=len(coincidences),
                        categorical_distance=mol_i.categorical_distance(mol_j)
                    )
                    edge_count += 1

        logger.info(f"Graph construction complete:")
        logger.info(f"  Nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"  Edges: {self.graph.number_of_edges():,}")
        logger.info(f"  Average degree: {self._avg_degree():.2f}")

        return self.graph

    def _avg_degree(self) -> float:
        """Calculate average node degree"""
        if self.graph.number_of_nodes() == 0:
            return 0.0
        return np.mean([d for n, d in self.graph.degree()])

    def graph_statistics(self) -> Dict:
        """
        Calculate comprehensive graph statistics

        Returns statistics matching experimental validation format
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        avg_degree = self._avg_degree()
        density = nx.density(self.graph)

        # Calculate enhancement factor from topology
        # Formula derived from experimental fit: F ≈ (avg_degree)² / (1 + density)
        enhancement = (avg_degree ** 2) / (1 + density) if density > 0 else avg_degree ** 2

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'avg_degree': avg_degree,
            'density': density,
            'redundancy_factor': avg_degree,  # Each node has this many validation paths
            'graph_enhancement': enhancement
        }

    def find_convergence_nodes(self, top_fraction: float = 0.01) -> List[int]:
        """
        Find high-centrality nodes where many paths converge

        These are optimal points for spectrometer materialization in cascade.

        Args:
            top_fraction: Fraction of nodes to return (default 1%)

        Returns:
            List of node IDs sorted by betweenness centrality
        """
        if self.graph.number_of_nodes() == 0:
            return []

        logger.info("Calculating betweenness centrality...")
        centrality = nx.betweenness_centrality(self.graph)

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        n_convergence = max(1, int(len(self.molecules) * top_fraction))
        convergence_nodes = [node for node, _ in sorted_nodes[:n_convergence]]

        logger.info(f"Found {len(convergence_nodes)} convergence nodes")

        return convergence_nodes

    def calculate_enhancement_factor(self) -> float:
        """
        Calculate precision enhancement from graph topology

        From experimental data: ~7,176× enhancement
        This arises from redundant measurement paths through the network.
        """
        stats = self.graph_statistics()
        return stats.get('graph_enhancement', 1.0)

    def get_edge_beat_frequencies(self) -> np.ndarray:
        """
        Extract all beat frequencies from graph edges

        Beat frequencies enable sub-harmonic precision enhancement
        """
        if self.graph.number_of_edges() == 0:
            return np.array([])

        beat_freqs = [data['beat_frequency']
                     for u, v, data in self.graph.edges(data=True)]
        return np.array(beat_freqs)

    def shortest_path_length(self, source: int, target: int) -> Optional[int]:
        """
        Categorical distance measured by graph path length

        Note: This is independent of physical distance
        """
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
