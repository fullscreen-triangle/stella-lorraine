"""
Proton Maxwell Demon for protein folding - PHASE-LOCKED VERSION.

This module implements the Proton Maxwell Demon concept for analyzing
hydrogen bond networks in proteins, viewing H-bonds as proton oscillators
that phase-lock with GroEL cavity resonance and O₂ master clock.

Key insight from papers: Protons don't just oscillate at fixed frequencies -
they phase-lock to the GroEL cavity which modulates with ATP cycles, all
synchronized to the cytoplasmic O₂ master clock at 10^13 Hz.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Physical constants
PLANCK = 6.626e-34  # J⋅s
BOLTZMANN = 1.381e-23  # J/K
PROTON_MASS = 1.673e-27  # kg

# Master clock frequencies from papers
O2_MASTER_CLOCK_HZ = 1e13  # O₂ vibrational frequency (master categorical clock)
HPLUS_FIELD_HZ = 4e13  # H⁺ field oscillation (reality substrate)
GROEL_BASE_HZ = 1.0  # GroEL ATP cycle base frequency (~1 Hz)
PROTON_OSCILLATION_HZ = 1e14  # Typical H-bond proton oscillation


@dataclass
class SEntropyCoordinates:
    """
    S-Entropy coordinates for categorical state tracking.
    From papers: Information position in (knowledge, temporal, evolution) space.
    """
    S_knowledge: float  # Information content (bits)
    S_temporal: float  # Temporal coherence
    S_evolution: float  # Evolutionary stability

    def distance_to(self, other: 'SEntropyCoordinates') -> float:
        """Calculate categorical distance between states."""
        return np.sqrt(
            (self.S_knowledge - other.S_knowledge)**2 +
            (self.S_temporal - other.S_temporal)**2 +
            (self.S_evolution - other.S_evolution)**2
        )


@dataclass
class HBondOscillator:
    """
    Hydrogen bond as a proton oscillator with phase-locking capability.

    Key insight: The oscillator doesn't just have frequency - it has PHASE
    that can lock to external drivers (GroEL cavity, O₂ clock).
    """
    donor_atom: str
    acceptor_atom: str
    donor_residue: int
    acceptor_residue: int
    bond_length: float  # Angstroms
    bond_angle: float  # degrees

    # Oscillatory properties
    frequency: float  # Hz (proton oscillation frequency)
    phase: float  # radians (current phase)
    amplitude: float  # Angstroms (oscillation amplitude)

    # Phase-locking properties (NEW)
    phase_coherence: float = 0.0  # 0-1, coherence with GroEL cavity
    groel_coupling: float = 0.0  # Coupling strength to cavity
    o2_coupling: float = 0.0  # Coupling to O₂ master clock

    # S-entropy state
    s_entropy: Optional[SEntropyCoordinates] = None

    def update_phase(self, time: float, groel_phase: float, groel_frequency: float) -> None:
        """
        Update phase with coupling to GroEL cavity.

        From papers: Phase evolution includes coupling terms:
        dφ/dt = ω + K*sin(φ_groel - φ)
        """
        # Natural phase evolution
        natural_phase = self.phase + 2 * np.pi * self.frequency * time

        # Phase-locking term (Kuramoto coupling)
        phase_diff = groel_phase - self.phase
        coupling_term = self.groel_coupling * np.sin(phase_diff)

        # Update phase
        self.phase = (natural_phase + coupling_term) % (2 * np.pi)

        # Update coherence (how well we're phase-locked)
        self.phase_coherence = np.abs(np.cos(phase_diff))

    def calculate_phase_lock_strength(self, cavity_frequency: float) -> float:
        """
        Calculate how strongly this H-bond will phase-lock to GroEL cavity.

        From papers: Strong phase-locking when |ω_A - ω_B| < K_coupling

        For THz frequencies, K_coupling ~ 10^11 Hz (0.1 THz bandwidth)
        """
        freq_diff = np.abs(self.frequency - cavity_frequency)

        # Harmonic matching (including subharmonics and superharmonics)
        # Test integer ratios up to 20:1
        harmonic_ratios = [1, 2, 3, 4, 5, 7, 10, 0.5, 0.33, 0.25, 0.2, 0.1]

        best_match = float('inf')
        for ratio in harmonic_ratios:
            harmonic_freq = cavity_frequency * ratio
            match = np.abs(self.frequency - harmonic_freq)
            if match < best_match:
                best_match = match

        # Phase-lock strength decreases with frequency mismatch
        # K_coupling scales with frequency (stronger coupling at THz frequencies)
        # Typical coupling strength: ~10% of base frequency
        K_coupling = 0.1 * cavity_frequency  # Dynamic coupling strength

        if best_match < K_coupling:
            strength = 1.0 - (best_match / K_coupling)
        else:
            # Weak coupling even beyond threshold (exponential falloff)
            strength = np.exp(-best_match / K_coupling) * 0.1

        return strength


class ProtonMaxwellDemon:
    """
    Maxwell Demon operating on hydrogen bond proton oscillators.

    Enhanced with phase-locking dynamics from papers:
    - Sorts protein configurations based on phase coherence
    - Couples to GroEL cavity resonance
    - Synchronized to O₂ master clock
    - Operates through PCET (proton-coupled electron transfer) mechanisms
    """

    def __init__(self, temperature: float = 310.0):
        """
        Initialize demon.

        Args:
            temperature: System temperature in Kelvin (physiological = 310K)
        """
        self.temperature = temperature
        self.kT = BOLTZMANN * temperature

        logger.info(f"Initialized ProtonMaxwellDemon at T={temperature}K")
        logger.info(f"  O₂ master clock: {O2_MASTER_CLOCK_HZ:.2e} Hz")
        logger.info(f"  H⁺ field: {HPLUS_FIELD_HZ:.2e} Hz")
        logger.info(f"  GroEL base: {GROEL_BASE_HZ:.2f} Hz")

    def calculate_proton_frequency(self, h_bond: HBondOscillator) -> float:
        """
        Calculate proton oscillation frequency for an H-bond.

        From papers: Frequency depends on bond strength and mass.
        Typical range: 10^13 - 10^14 Hz for H-bonds.
        """
        # Bond strength from length (shorter = stronger = higher frequency)
        # Typical H-bond length: 2.5-3.5 Å
        # Use harmonic oscillator approximation

        # Force constant estimation (stronger for shorter bonds)
        k_spring = 500.0 / (h_bond.bond_length ** 2)  # N/m (approximate)

        # Proton oscillation frequency
        omega = np.sqrt(k_spring / PROTON_MASS)
        frequency = omega / (2 * np.pi)

        # Angular dependence (optimal at 180 degrees)
        angle_factor = np.abs(np.cos(np.radians(h_bond.bond_angle - 180)))
        frequency *= (0.5 + 0.5 * angle_factor)

        return frequency

    def calculate_s_entropy_state(self, h_bond: HBondOscillator,
                                   groel_coupling: float) -> SEntropyCoordinates:
        """
        Calculate S-entropy coordinates for H-bond state.

        From papers: S-entropy encodes (knowledge, temporal, evolution) position
        in categorical information space.
        """
        # Knowledge entropy: Information content of bond configuration
        S_k = -np.log2(h_bond.phase_coherence + 1e-10)  # Higher coherence = lower entropy

        # Temporal entropy: Phase stability over time
        freq_ratio = h_bond.frequency / O2_MASTER_CLOCK_HZ
        S_t = -np.log2(np.abs(freq_ratio) + 1e-10)

        # Evolution entropy: Coupling stability (how well it phase-locks)
        S_e = -np.log2(groel_coupling + 1e-10)

        return SEntropyCoordinates(S_k, S_t, S_e)

    def evaluate_configuration(self, h_bonds: List[HBondOscillator],
                               groel_frequency: float,
                               groel_phase: float) -> Dict:
        """
        Evaluate protein configuration based on phase-locking to GroEL.

        This is the demon's "filter" function - it sorts configurations
        based on harmonic coupling with the GroEL cavity.

        Args:
            h_bonds: List of H-bond oscillators
            groel_frequency: Current GroEL cavity frequency (varies with ATP cycle)
            groel_phase: Current GroEL cavity phase

        Returns:
            Dictionary with evaluation metrics
        """
        if not h_bonds:
            return {
                'mean_coherence': 0.0,
                'variance': float('inf'),
                'phase_locked_count': 0,
                'network_stability': 0.0,
                'passes_filter': False
            }

        coherences = []
        phase_locked_count = 0

        for bond in h_bonds:
            # Calculate phase-lock strength
            lock_strength = bond.calculate_phase_lock_strength(groel_frequency)
            bond.groel_coupling = lock_strength

            # Update phase with GroEL coupling
            bond.update_phase(time=0.001, groel_phase=groel_phase,
                            groel_frequency=groel_frequency)

            coherences.append(bond.phase_coherence)

            if bond.phase_coherence > 0.7:  # threshold for "phase-locked"
                phase_locked_count += 1

        coherences = np.array(coherences)
        mean_coherence = np.mean(coherences)
        variance = np.var(coherences)

        # Network stability: high mean coherence + low variance = stable
        network_stability = mean_coherence * (1.0 / (1.0 + variance))

        # Demon's filter criterion: passes if network is phase-locked
        passes_filter = network_stability > 0.5 and phase_locked_count > len(h_bonds) * 0.5

        return {
            'mean_coherence': float(mean_coherence),
            'variance': float(variance),
            'phase_locked_count': phase_locked_count,
            'total_bonds': len(h_bonds),
            'network_stability': float(network_stability),
            'passes_filter': passes_filter,
            'groel_frequency': groel_frequency,
            'groel_phase': groel_phase
        }

    def calculate_harmonic_network_entropy(self, h_bonds: List[HBondOscillator]) -> float:
        """
        Calculate total network entropy from harmonic coupling.

        From papers: Entropy = -k_B * sum(p_i * ln(p_i)) over categorical states
        """
        if not h_bonds:
            return float('inf')

        # Phase distribution defines categorical states
        phases = np.array([bond.phase for bond in h_bonds])

        # Bin phases into categorical states (8 bins)
        hist, _ = np.histogram(phases, bins=8, range=(0, 2*np.pi), density=True)

        # Calculate Shannon entropy
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist))

        return entropy


def create_h_bond_oscillator(donor: str, acceptor: str,
                              donor_res: int, acceptor_res: int,
                              length: float, angle: float,
                              temperature: float = 310.0) -> HBondOscillator:
    """
    Factory function to create H-bond oscillator with calculated properties.
    """
    # Calculate frequency
    bond = HBondOscillator(
        donor_atom=donor,
        acceptor_atom=acceptor,
        donor_residue=donor_res,
        acceptor_residue=acceptor_res,
        bond_length=length,
        bond_angle=angle,
        frequency=0.0,  # Will be calculated
        phase=np.random.uniform(0, 2*np.pi),  # Random initial phase
        amplitude=0.1  # Typical amplitude in Angstroms
    )

    # Calculate frequency using demon
    demon = ProtonMaxwellDemon(temperature)
    bond.frequency = demon.calculate_proton_frequency(bond)

    return bond
