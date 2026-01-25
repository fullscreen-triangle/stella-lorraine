"""
Categorical S-Entropy Coordinates for Membrane Transporters
============================================================

Maps membrane transporter conformational states to S-entropy space.
Based on Stella Lorraine Observatory categorical dynamics framework.

Key Insight: Transporters operate in BOTH physical space (membrane geometry)
AND categorical space (S-entropy coordinates). Substrate selection happens
in S-space through phase-locking, explaining Maxwell Demon behavior.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


@dataclass
class SEntropyCoordinates:
    """
    S-entropy coordinates for categorical molecular demons.

    S_k: Knowledge entropy (which state are we in?)
    S_t: Temporal entropy (when do transitions occur?)
    S_e: Evolution entropy (how does system evolve?)
    """
    S_k: float  # [0, ∞) - knowledge entropy
    S_t: float  # [0, 1] - temporal phase (normalized)
    S_e: float  # [0, ∞) - evolution/amplitude

    def distance_to(self, other: 'SEntropyCoordinates') -> float:
        """Categorical distance in S-space"""
        return np.sqrt(
            (self.S_k - other.S_k)**2 +
            (self.S_t - other.S_t)**2 +
            (self.S_e - other.S_e)**2
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            'S_k': self.S_k,
            'S_t': self.S_t,
            'S_e': self.S_e
        }


class TransporterState(Enum):
    """Conformational states of ABC transporter"""
    OPEN_OUTSIDE = "open_outside"      # Substrate binding from outside
    OCCLUDED = "occluded"              # Substrate trapped
    OPEN_INSIDE = "open_inside"        # Substrate release to inside
    RESETTING = "resetting"            # Return to initial state


@dataclass
class ConformationalState:
    """
    A single conformational state with both physical and categorical properties.
    """
    name: TransporterState

    # Physical properties
    cavity_volume: float              # Å³
    binding_site_frequency: float     # Hz (vibrational mode of binding site)
    transmembrane_distance: float     # Å (how far through membrane)

    # Categorical coordinates
    s_coordinates: SEntropyCoordinates

    # Energetics
    free_energy: float                # kJ/mol
    atp_bound: bool                   # Is ATP bound?

    def to_dict(self) -> Dict:
        return {
            'name': self.name.value,
            'cavity_volume': self.cavity_volume,
            'binding_site_frequency': self.binding_site_frequency,
            'transmembrane_distance': self.transmembrane_distance,
            's_coordinates': self.s_coordinates.to_dict(),
            'free_energy': self.free_energy,
            'atp_bound': self.atp_bound
        }


class TransporterConformationalLandscape:
    """
    Complete conformational landscape of a membrane transporter.

    Maps physical conformations to S-entropy coordinates, enabling
    categorical analysis of transport mechanism.
    """

    def __init__(self, transporter_type: str = "ABC_exporter"):
        self.transporter_type = transporter_type
        self.states = self._initialize_states()

        # ATP parameters
        self.atp_hydrolysis_frequency = 10.0  # Hz (typical turnover)
        self.atp_binding_energy = -30.0       # kJ/mol

        # Temperature
        self.temperature = 310.0  # K
        self.k_B = 1.38e-23       # J/K

    def _initialize_states(self) -> Dict[TransporterState, ConformationalState]:
        """
        Initialize conformational states based on structural data.

        Based on:
        - P-glycoprotein structures (PDB: 3G5U, 4M1M)
        - MsbA structures (PDB: 3B5W, 3B5X)
        - ABCB10 structures (PDB: 6VQU)
        """

        states = {}

        # OPEN_OUTSIDE: Ready to bind substrate from extracellular/periplasmic space
        states[TransporterState.OPEN_OUTSIDE] = ConformationalState(
            name=TransporterState.OPEN_OUTSIDE,
            cavity_volume=5000.0,              # Å³ (large, open cavity)
            binding_site_frequency=3.8e13,     # Hz (relaxed conformation)
            transmembrane_distance=20.0,       # Å (halfway through membrane)
            s_coordinates=SEntropyCoordinates(
                S_k=0.1,  # Low knowledge (many possible substrates)
                S_t=0.0,  # Start of cycle
                S_e=1.0   # High evolution potential
            ),
            free_energy=0.0,                   # Reference state
            atp_bound=True
        )

        # OCCLUDED: Substrate trapped, ATP hydrolysis triggers transition
        states[TransporterState.OCCLUDED] = ConformationalState(
            name=TransporterState.OCCLUDED,
            cavity_volume=3000.0,              # Å³ (compressed)
            binding_site_frequency=4.5e13,     # Hz (strained, higher frequency)
            transmembrane_distance=20.0,       # Å (midpoint)
            s_coordinates=SEntropyCoordinates(
                S_k=0.9,  # High knowledge (substrate identified)
                S_t=0.25, # 1/4 through cycle
                S_e=0.5   # Mid-evolution
            ),
            free_energy=15.0,                  # kJ/mol (strained)
            atp_bound=True  # About to hydrolyze
        )

        # OPEN_INSIDE: Substrate released to cytoplasm/inside
        states[TransporterState.OPEN_INSIDE] = ConformationalState(
            name=TransporterState.OPEN_INSIDE,
            cavity_volume=4500.0,              # Å³ (open inside)
            binding_site_frequency=3.2e13,     # Hz (relaxed, different mode)
            transmembrane_distance=40.0,       # Å (flipped orientation)
            s_coordinates=SEntropyCoordinates(
                S_k=0.2,  # Low knowledge (substrate released)
                S_t=0.5,  # Halfway through cycle
                S_e=0.3   # Low evolution (stable state)
            ),
            free_energy=-10.0,                 # kJ/mol (favorable, ATP hydrolyzed)
            atp_bound=False  # ADP bound
        )

        # RESETTING: Returning to initial state
        states[TransporterState.RESETTING] = ConformationalState(
            name=TransporterState.RESETTING,
            cavity_volume=4000.0,              # Å³ (transitioning)
            binding_site_frequency=3.5e13,     # Hz (intermediate)
            transmembrane_distance=30.0,       # Å (moving back)
            s_coordinates=SEntropyCoordinates(
                S_k=0.05, # Very low knowledge (empty)
                S_t=0.75, # 3/4 through cycle
                S_e=0.8   # High evolution (active transition)
            ),
            free_energy=5.0,                   # kJ/mol (barrier to cross)
            atp_bound=False  # Exchanging ADP for ATP
        )

        return states

    def get_state(self, state: TransporterState) -> ConformationalState:
        """Retrieve conformational state"""
        return self.states[state]

    def calculate_transition_rate(self,
                                  from_state: TransporterState,
                                  to_state: TransporterState,
                                  substrate_bound: bool = False) -> float:
        """
        Calculate transition rate between states using Kramers theory.

        Rate depends on:
        1. Energy barrier (ΔG)
        2. ATP binding/hydrolysis
        3. Substrate presence (stabilizes certain states)
        """

        state_from = self.states[from_state]
        state_to = self.states[to_state]

        # Energy barrier
        delta_G = state_to.free_energy - state_from.free_energy

        # ATP hydrolysis contribution
        if state_from.atp_bound and not state_to.atp_bound:
            delta_G += self.atp_binding_energy

        # Substrate stabilization (lowers barrier)
        if substrate_bound and from_state == TransporterState.OPEN_OUTSIDE:
            delta_G -= 10.0  # kJ/mol stabilization

        # Kramers rate: k = k0 * exp(-ΔG/kT)
        k_0 = 1e6  # s^-1 (attempt frequency)
        k_B_T = self.k_B * self.temperature * 6.022e20  # Convert to kJ/mol

        rate = k_0 * np.exp(-delta_G / k_B_T)

        return rate

    def calculate_s_space_trajectory(self,
                                     num_cycles: int = 5) -> List[SEntropyCoordinates]:
        """
        Calculate trajectory through S-entropy space during transport cycles.

        Returns list of S-coordinates visited during transport.
        """
        trajectory = []

        cycle_order = [
            TransporterState.OPEN_OUTSIDE,
            TransporterState.OCCLUDED,
            TransporterState.OPEN_INSIDE,
            TransporterState.RESETTING
        ]

        for cycle in range(num_cycles):
            for state in cycle_order:
                s_coord = self.states[state].s_coordinates
                # Add cycle offset to S_t (temporal coordinate)
                s_coord_copy = SEntropyCoordinates(
                    S_k=s_coord.S_k,
                    S_t=(s_coord.S_t + cycle) % 1.0,  # Wrap around
                    S_e=s_coord.S_e
                )
                trajectory.append(s_coord_copy)

        return trajectory

    def plot_conformational_cycle(self) -> Dict:
        """
        Generate data for plotting conformational cycle.

        Returns dict with:
        - states: list of state names
        - volumes: cavity volumes
        - frequencies: binding site frequencies
        - s_coordinates: S-entropy coordinates
        - free_energies: free energy values
        """

        cycle_order = [
            TransporterState.OPEN_OUTSIDE,
            TransporterState.OCCLUDED,
            TransporterState.OPEN_INSIDE,
            TransporterState.RESETTING,
            TransporterState.OPEN_OUTSIDE  # Close the cycle
        ]

        data = {
            'states': [],
            'volumes': [],
            'frequencies': [],
            's_k': [],
            's_t': [],
            's_e': [],
            'free_energies': []
        }

        for state in cycle_order:
            conf = self.states[state]
            data['states'].append(conf.name.value)
            data['volumes'].append(conf.cavity_volume)
            data['frequencies'].append(conf.binding_site_frequency)
            data['s_k'].append(conf.s_coordinates.S_k)
            data['s_t'].append(conf.s_coordinates.S_t)
            data['s_e'].append(conf.s_coordinates.S_e)
            data['free_energies'].append(conf.free_energy)

        return data


def example_usage():
    """Demonstrate transporter conformational landscape"""

    print("="*70)
    print("MEMBRANE TRANSPORTER CONFORMATIONAL LANDSCAPE")
    print("="*70)
    print()

    # Initialize landscape
    landscape = TransporterConformationalLandscape("ABC_exporter")

    print("Conformational States in S-Entropy Space:")
    print("-" * 70)

    for state_name, conf_state in landscape.states.items():
        print(f"\n{state_name.value.upper()}:")
        print(f"  Cavity volume: {conf_state.cavity_volume:.1f} Ų")
        print(f"  Binding frequency: {conf_state.binding_site_frequency:.2e} Hz")
        print(f"  S-coordinates: S_k={conf_state.s_coordinates.S_k:.2f}, "
              f"S_t={conf_state.s_coordinates.S_t:.2f}, "
              f"S_e={conf_state.s_coordinates.S_e:.2f}")
        print(f"  Free energy: {conf_state.free_energy:.1f} kJ/mol")
        print(f"  ATP bound: {conf_state.atp_bound}")

    print("\n" + "="*70)
    print("Transition Rates:")
    print("-" * 70)

    transitions = [
        (TransporterState.OPEN_OUTSIDE, TransporterState.OCCLUDED),
        (TransporterState.OCCLUDED, TransporterState.OPEN_INSIDE),
        (TransporterState.OPEN_INSIDE, TransporterState.RESETTING),
        (TransporterState.RESETTING, TransporterState.OPEN_OUTSIDE),
    ]

    for from_state, to_state in transitions:
        rate_empty = landscape.calculate_transition_rate(from_state, to_state, substrate_bound=False)
        rate_bound = landscape.calculate_transition_rate(from_state, to_state, substrate_bound=True)

        print(f"\n{from_state.value} → {to_state.value}:")
        print(f"  Empty: {rate_empty:.2e} s⁻¹")
        print(f"  Substrate-bound: {rate_bound:.2e} s⁻¹")
        print(f"  Enhancement: {rate_bound/rate_empty:.1f}×")

    print("\n" + "="*70)
    print("S-Space Trajectory (5 cycles):")
    print("-" * 70)

    trajectory = landscape.calculate_s_space_trajectory(num_cycles=5)

    print(f"Total trajectory points: {len(trajectory)}")
    print(f"S-space distance traveled: {sum(trajectory[i].distance_to(trajectory[i+1]) for i in range(len(trajectory)-1)):.2f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    example_usage()
