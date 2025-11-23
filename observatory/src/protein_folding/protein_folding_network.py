"""
Protein Folding Network - Phase-Locked H-Bond Networks.

Models a protein as a network of coupled proton oscillators (hydrogen bonds)
that must phase-lock collectively for stable folding.

Key insight from papers: Folding isn't sequential addition of H-bonds -
it's establishing phase-coherent oscillatory network synchronized to GroEL cavity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

from proton_maxwell_demon import (
    ProtonMaxwellDemon,
    HBondOscillator,
    O2_MASTER_CLOCK_HZ,
    GROEL_BASE_HZ
)

logger = logging.getLogger(__name__)


@dataclass
class PhaseCoherenceCluster:
    """
    Group of H-bonds that are phase-locked together.

    From papers: Phase-locked clusters form the "nucleation sites" for folding.
    """
    bonds: List[HBondOscillator]
    center_phase: float  # Central phase of cluster
    coherence: float  # Mean phase coherence within cluster
    coupling_strength: float  # How strongly coupled to GroEL


class ProteinFoldingNetwork:
    """
    Represents protein as network of proton demons (H-bond oscillators).

    Enhanced with phase-locking dynamics:
    - Tracks phase coherence across H-bond network
    - Identifies phase-locked clusters
    - Calculates network stability from collective phase-locking
    - Synchronizes with GroEL cavity cycles
    """

    def __init__(self, protein_name: str, temperature: float = 310.0):
        """
        Initialize protein folding network.

        Args:
            protein_name: Name/ID of protein
            temperature: System temperature (K)
        """
        self.protein_name = protein_name
        self.temperature = temperature
        self.demon = ProtonMaxwellDemon(temperature)

        self.h_bonds: List[HBondOscillator] = []
        self.residue_map: Dict[int, List[HBondOscillator]] = {}  # residue -> bonds

        # Phase-locking state
        self.current_cycle = 0  # GroEL ATP cycle number
        self.groel_phase = 0.0  # Current GroEL cavity phase
        self.groel_frequency = GROEL_BASE_HZ  # Current cavity frequency

        # Network state
        self.phase_clusters: List[PhaseCoherenceCluster] = []
        self.network_coherence = 0.0

        logger.info(f"Initialized ProteinFoldingNetwork for {protein_name}")

    def add_h_bond(self, bond: HBondOscillator) -> None:
        """Add hydrogen bond to network."""
        self.h_bonds.append(bond)

        # Update residue map
        for res in [bond.donor_residue, bond.acceptor_residue]:
            if res not in self.residue_map:
                self.residue_map[res] = []
            self.residue_map[res].append(bond)

    def get_bonds_for_residue(self, residue: int) -> List[HBondOscillator]:
        """Get all H-bonds involving a residue."""
        return self.residue_map.get(residue, [])

    def update_groel_state(self, cycle: int, phase: float, frequency: float) -> None:
        """
        Update GroEL cavity state for this cycle.

        From papers: GroEL operates in ATP-driven cycles, each cycle modulates
        the cavity frequency and phase.

        Args:
            cycle: ATP cycle number
            phase: Current cavity phase (radians)
            frequency: Current cavity frequency (Hz)
        """
        self.current_cycle = cycle
        self.groel_phase = phase
        self.groel_frequency = frequency

        logger.debug(f"Cycle {cycle}: GroEL freq={frequency:.2e} Hz, phase={phase:.2f} rad")

    def calculate_network_variance(self) -> float:
        """
        Calculate variance in phase coherence across network.

        From papers: Variance minimization is the driving principle.
        Low variance = stable fold, high variance = unstable.
        """
        if not self.h_bonds:
            return float('inf')

        coherences = [bond.phase_coherence for bond in self.h_bonds]
        return float(np.var(coherences))

    def calculate_network_stability(self) -> float:
        """
        Calculate overall network stability from phase-locking.

        Stability = (mean coherence) / (1 + variance)
        """
        evaluation = self.demon.evaluate_configuration(
            self.h_bonds,
            self.groel_frequency,
            self.groel_phase
        )

        self.network_coherence = evaluation['mean_coherence']

        return evaluation['network_stability']

    def find_phase_coherence_clusters(self, coherence_threshold: float = 0.7,
                                     phase_tolerance: float = 0.5) -> List[PhaseCoherenceCluster]:
        """
        Identify clusters of H-bonds that are phase-locked together.

        From papers: Phase-locked clusters = folding nuclei.
        These are the stable "islands" that guide folding progression.

        Args:
            coherence_threshold: Minimum coherence to be considered phase-locked
            phase_tolerance: Maximum phase difference within cluster (radians)

        Returns:
            List of phase-coherence clusters
        """
        # Filter for phase-locked bonds
        phase_locked = [b for b in self.h_bonds if b.phase_coherence > coherence_threshold]

        if not phase_locked:
            return []

        # Cluster by phase similarity
        clusters = []
        remaining = set(range(len(phase_locked)))

        while remaining:
            # Start new cluster with first remaining bond
            idx = min(remaining)
            seed_bond = phase_locked[idx]
            cluster_bonds = [seed_bond]
            remaining.remove(idx)

            # Add all bonds with similar phase
            seed_phase = seed_bond.phase
            to_remove = set()

            for other_idx in remaining:
                other_bond = phase_locked[other_idx]
                phase_diff = np.abs(other_bond.phase - seed_phase)
                # Account for circular phase (0 ≈ 2π)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)

                if phase_diff < phase_tolerance:
                    cluster_bonds.append(other_bond)
                    to_remove.add(other_idx)

            remaining -= to_remove

            # Create cluster
            phases = [b.phase for b in cluster_bonds]
            couplings = [b.groel_coupling for b in cluster_bonds]
            coherences = [b.phase_coherence for b in cluster_bonds]

            cluster = PhaseCoherenceCluster(
                bonds=cluster_bonds,
                center_phase=np.mean(phases),
                coherence=np.mean(coherences),
                coupling_strength=np.mean(couplings)
            )
            clusters.append(cluster)

        # Sort by size (largest first)
        clusters.sort(key=lambda c: len(c.bonds), reverse=True)

        self.phase_clusters = clusters
        return clusters

    def identify_folding_nucleus(self) -> Optional[PhaseCoherenceCluster]:
        """
        Identify the primary folding nucleus - largest phase-locked cluster.

        From papers: Folding nucleates from the most strongly phase-locked region.
        """
        clusters = self.find_phase_coherence_clusters()

        if not clusters:
            return None

        # Return largest, most coherent cluster
        return max(clusters, key=lambda c: len(c.bonds) * c.coherence)

    def calculate_residue_stability(self, residue: int) -> float:
        """
        Calculate stability of a specific residue based on its H-bonds.

        Residue is stable if its H-bonds are phase-locked.
        """
        bonds = self.get_bonds_for_residue(residue)

        if not bonds:
            return 0.0

        coherences = [b.phase_coherence for b in bonds]
        return float(np.mean(coherences))

    def get_network_summary(self) -> Dict:
        """Get comprehensive summary of network state."""
        if not self.h_bonds:
            return {
                'protein': self.protein_name,
                'total_bonds': 0,
                'network_coherence': 0.0,
                'variance': float('inf'),
                'stability': 0.0,
                'phase_locked_count': 0,
                'clusters': [],
                'folding_nucleus_size': 0
            }

        clusters = self.find_phase_coherence_clusters()
        nucleus = self.identify_folding_nucleus()

        evaluation = self.demon.evaluate_configuration(
            self.h_bonds,
            self.groel_frequency,
            self.groel_phase
        )

        return {
            'protein': self.protein_name,
            'total_bonds': len(self.h_bonds),
            'network_coherence': evaluation['mean_coherence'],
            'variance': evaluation['variance'],
            'stability': evaluation['network_stability'],
            'phase_locked_count': evaluation['phase_locked_count'],
            'clusters': [
                {
                    'size': len(c.bonds),
                    'center_phase': float(c.center_phase),
                    'coherence': float(c.coherence),
                    'coupling': float(c.coupling_strength)
                }
                for c in clusters
            ],
            'folding_nucleus_size': len(nucleus.bonds) if nucleus else 0,
            'current_cycle': self.current_cycle,
            'groel_frequency': self.groel_frequency
        }

    def simulate_cycle_step(self, dt: float = 0.001) -> None:
        """
        Simulate one time step of phase evolution.

        All H-bonds evolve their phases under GroEL coupling.

        Args:
            dt: Time step in seconds
        """
        for bond in self.h_bonds:
            bond.update_phase(dt, self.groel_phase, self.groel_frequency)

        # Update network state
        self.calculate_network_stability()

    def get_critical_h_bonds(self, top_n: int = 10) -> List[Tuple[HBondOscillator, float]]:
        """
        Identify most critical H-bonds for network stability.

        From papers: Critical bonds are those with strongest GroEL coupling
        and highest phase coherence - removing them destabilizes the network.

        Returns:
            List of (bond, criticality_score) tuples
        """
        criticality_scores = []

        for bond in self.h_bonds:
            # Criticality = coherence * coupling strength
            score = bond.phase_coherence * bond.groel_coupling
            criticality_scores.append((bond, score))

        # Sort by criticality
        criticality_scores.sort(key=lambda x: x[1], reverse=True)

        return criticality_scores[:top_n]
