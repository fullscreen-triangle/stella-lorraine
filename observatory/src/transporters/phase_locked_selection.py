"""
Phase-Locked Substrate Selection in Membrane Transporters
==========================================================

Explains Maxwell Demon substrate selectivity through phase-locking.

KEY INSIGHT: Transporters don't "recognize" substrates through lock-and-key
geometry alone. They use PHASE-LOCKING of vibrational frequencies:

1. Binding site has characteristic frequency ω_site (THz range)
2. Substrate has vibrational modes ω_substrate
3. Phase-lock occurs when |ω_site - ω_substrate| < threshold
4. Only phase-locked substrates trigger conformational change
5. ATP modulates ω_site, scanning frequency space

This explains:
- High selectivity (10^6-fold discrimination)
- Promiscuity (multiple substrates if frequencies match)
- Drug resistance (mutations shift ω_site)
- Non-competitive inhibition (blocks phase-locking mechanism)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

from categorical_coordinates import (
    SEntropyCoordinates,
    TransporterState,
    TransporterConformationalLandscape
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubstrateVibrationalProfile:
    """
    Vibrational fingerprint of a potential substrate molecule.
    """
    name: str
    molecular_weight: float  # Da

    # Dominant vibrational modes (Hz)
    fundamental_frequency: float      # Primary stretch/bend mode
    harmonic_modes: List[float]       # Overtones and combinations

    # Categorical coordinates
    s_coordinates: SEntropyCoordinates

    # Binding properties
    charge: int                       # Net charge
    hydrophobicity: float             # log P

    def calculate_phase_lock_strength(self,
                                     site_frequency: float,
                                     threshold_hz: float = 1e12) -> float:
        """
        Calculate phase-locking strength to a binding site frequency.

        Returns value in [0, 1]:
        - 1.0 = perfect phase lock
        - 0.0 = no phase lock
        """

        # Check fundamental and harmonics
        all_frequencies = [self.fundamental_frequency] + self.harmonic_modes

        min_detuning = float('inf')
        for freq in all_frequencies:
            # Check if any harmonic of substrate matches site
            for n_substrate in range(1, 6):  # Check first 5 harmonics
                for n_site in range(1, 6):
                    detuning = abs(n_substrate * freq - n_site * site_frequency)
                    if detuning < min_detuning:
                        min_detuning = detuning

        # Convert detuning to phase-lock strength (Lorentzian lineshape)
        gamma = threshold_hz  # Linewidth
        phase_lock = 1.0 / (1.0 + (min_detuning / gamma)**2)

        return phase_lock

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'molecular_weight': self.molecular_weight,
            'fundamental_frequency': self.fundamental_frequency,
            'harmonic_modes': self.harmonic_modes,
            's_coordinates': self.s_coordinates.to_dict(),
            'charge': self.charge,
            'hydrophobicity': self.hydrophobicity
        }


class PhaseLockingTransporter:
    """
    Membrane transporter using phase-locking for substrate selection.

    Implements the Maxwell Demon through:
    1. MEASUREMENT: Detect substrate via frequency matching (phase-lock)
    2. FEEDBACK: Conformational change if phase-locked
    3. TRANSPORT: Move substrate across membrane
    4. RESET: ATP-driven return to initial state
    """

    def __init__(self, transporter_type: str = "ABC_exporter"):
        # Conformational landscape
        self.landscape = TransporterConformationalLandscape(transporter_type)
        self.current_state = TransporterState.OPEN_OUTSIDE

        # Phase-locking parameters
        self.phase_lock_threshold = 1e12  # Hz (1 THz threshold)
        self.min_phase_lock_strength = 0.3  # Minimum to trigger transport

        # ATP cycle modulation
        self.atp_modulation_frequency = 10.0  # Hz (ATP turnover)
        self.atp_modulation_amplitude = 0.5e13  # Hz (±5 THz scan)

        # Statistics
        self.transport_events = 0
        self.rejected_molecules = 0
        self.current_time = 0.0  # seconds

        # O₂ master clock (from protein folding framework)
        self.o2_master_clock = 1e13  # Hz

        # Temperature
        self.temperature = 310.0  # K
        self.k_B = 1.38e-23  # J/K

    def get_current_binding_frequency(self, time: float) -> float:
        """
        Calculate current binding site frequency at given time.

        ATP hydrolysis modulates frequency, scanning for substrates.
        This is analogous to GroEL's cavity frequency modulation.
        """

        conf_state = self.landscape.get_state(self.current_state)
        base_frequency = conf_state.binding_site_frequency

        # ATP cycle modulates frequency sinusoidally
        atp_phase = 2 * np.pi * self.atp_modulation_frequency * time
        modulation = self.atp_modulation_amplitude * np.sin(atp_phase)

        current_frequency = base_frequency + modulation

        return current_frequency

    def measure_substrate(self,
                         substrate: SubstrateVibrationalProfile,
                         time: float) -> Tuple[bool, float]:
        """
        MEASUREMENT STEP: Detect if substrate phase-locks to binding site.

        This is a CATEGORICAL measurement - happens in S-entropy space,
        no momentum transfer to substrate (zero backaction).

        Returns:
            (is_detected, phase_lock_strength)
        """

        # Get current binding site frequency (modulated by ATP)
        site_frequency = self.get_current_binding_frequency(time)

        # Calculate phase-lock strength
        phase_lock_strength = substrate.calculate_phase_lock_strength(
            site_frequency,
            self.phase_lock_threshold
        )

        # Check if above threshold
        is_detected = phase_lock_strength >= self.min_phase_lock_strength

        logger.debug(f"Measurement at t={time:.6f}s:")
        logger.debug(f"  Site frequency: {site_frequency:.2e} Hz")
        logger.debug(f"  Substrate: {substrate.name}")
        logger.debug(f"  Phase-lock strength: {phase_lock_strength:.3f}")
        logger.debug(f"  Detected: {is_detected}")

        return is_detected, phase_lock_strength

    def feedback_conformational_change(self,
                                      substrate: SubstrateVibrationalProfile,
                                      phase_lock_strength: float):
        """
        FEEDBACK STEP: Conformational change triggered by phase-locking.

        Strong phase-lock → conformational change → transport
        Weak phase-lock → no change → substrate rejected
        """

        if phase_lock_strength < self.min_phase_lock_strength:
            logger.debug(f"Rejected {substrate.name} (weak phase-lock)")
            self.rejected_molecules += 1
            return False

        # Trigger conformational transition
        if self.current_state == TransporterState.OPEN_OUTSIDE:
            # Bind substrate, transition to occluded
            self.current_state = TransporterState.OCCLUDED
            logger.info(f"✓ Substrate {substrate.name} bound (phase-lock: {phase_lock_strength:.3f})")
            return True

        return False

    def transport_cycle(self,
                       substrate: SubstrateVibrationalProfile,
                       time: float) -> Dict:
        """
        Complete transport cycle for one substrate molecule.

        Returns dict with cycle statistics.
        """

        cycle_start_time = time
        results = {
            'substrate': substrate.name,
            'start_time': cycle_start_time,
            'transported': False,
            'phase_lock_strength': 0.0,
            'cycle_duration': 0.0,
            'state_trajectory': []
        }

        # STEP 1: MEASUREMENT
        is_detected, phase_lock_strength = self.measure_substrate(substrate, time)
        results['phase_lock_strength'] = phase_lock_strength
        results['state_trajectory'].append(self.current_state.value)

        if not is_detected:
            results['transported'] = False
            return results

        # STEP 2: FEEDBACK (conformational change)
        bound = self.feedback_conformational_change(substrate, phase_lock_strength)

        if not bound:
            results['transported'] = False
            return results

        # STEP 3: TRANSPORT (state transitions)
        # Occluded → Open inside → Release

        # Transition to open_inside (ATP hydrolysis)
        transition_time_1 = 1.0 / self.landscape.calculate_transition_rate(
            TransporterState.OCCLUDED,
            TransporterState.OPEN_INSIDE,
            substrate_bound=True
        )
        time += transition_time_1
        self.current_state = TransporterState.OPEN_INSIDE
        results['state_trajectory'].append(self.current_state.value)
        logger.debug(f"  Transition to OPEN_INSIDE at t={time:.6f}s")

        # Release substrate (inside)
        logger.info(f"✓ Substrate {substrate.name} released inside")

        # STEP 4: RESET (return to initial state)
        # Resetting → Open outside

        transition_time_2 = 1.0 / self.landscape.calculate_transition_rate(
            TransporterState.OPEN_INSIDE,
            TransporterState.RESETTING,
            substrate_bound=False
        )
        time += transition_time_2
        self.current_state = TransporterState.RESETTING
        results['state_trajectory'].append(self.current_state.value)
        logger.debug(f"  Transition to RESETTING at t={time:.6f}s")

        transition_time_3 = 1.0 / self.landscape.calculate_transition_rate(
            TransporterState.RESETTING,
            TransporterState.OPEN_OUTSIDE,
            substrate_bound=False
        )
        time += transition_time_3
        self.current_state = TransporterState.OPEN_OUTSIDE
        results['state_trajectory'].append(self.current_state.value)
        logger.debug(f"  Transition to OPEN_OUTSIDE at t={time:.6f}s")

        # Complete
        results['transported'] = True
        results['cycle_duration'] = time - cycle_start_time
        self.transport_events += 1
        self.current_time = time

        return results

    def simulate_substrate_selection(self,
                                     substrates: List[SubstrateVibrationalProfile],
                                     simulation_time: float = 1.0) -> Dict:
        """
        Simulate Maxwell Demon operation over multiple substrates.

        Returns statistics on:
        - Which substrates were transported
        - Which were rejected
        - Phase-lock strengths
        - Transport rates
        """

        logger.info("="*70)
        logger.info("PHASE-LOCKED SUBSTRATE SELECTION SIMULATION")
        logger.info("="*70)

        results = {
            'substrates': [],
            'transported': [],
            'rejected': [],
            'phase_lock_strengths': {},
            'transport_rates': {},
            'total_time': simulation_time
        }

        time = 0.0
        substrate_idx = 0

        while time < simulation_time and substrate_idx < len(substrates):
            substrate = substrates[substrate_idx]

            # Attempt transport
            cycle_result = self.transport_cycle(substrate, time)

            results['substrates'].append(substrate.name)
            results['phase_lock_strengths'][substrate.name] = cycle_result['phase_lock_strength']

            if cycle_result['transported']:
                results['transported'].append(substrate.name)
                time = self.current_time
            else:
                results['rejected'].append(substrate.name)
                time += 0.001  # Small time step before next substrate

            substrate_idx += 1

        # Calculate transport rates
        for substrate in substrates:
            name = substrate.name
            if name in results['transported']:
                # Find cycle duration
                for sub_name in results['substrates']:
                    if sub_name == name:
                        results['transport_rates'][name] = self.atp_modulation_frequency
                        break
            else:
                results['transport_rates'][name] = 0.0

        # Summary statistics
        results['transport_efficiency'] = (
            len(results['transported']) / len(substrates) if substrates else 0.0
        )
        results['selectivity'] = self._calculate_selectivity(results['phase_lock_strengths'])

        logger.info(f"\nTransported: {len(results['transported'])}/{len(substrates)}")
        logger.info(f"Efficiency: {results['transport_efficiency']:.1%}")
        logger.info(f"Selectivity: {results['selectivity']:.2e}")

        return results

    def _calculate_selectivity(self, phase_lock_strengths: Dict[str, float]) -> float:
        """
        Calculate selectivity as ratio of max to min phase-lock strength.
        """
        if not phase_lock_strengths:
            return 1.0

        values = list(phase_lock_strengths.values())
        max_val = max(values)
        min_val = min([v for v in values if v > 0] + [1e-10])

        return max_val / min_val

    def get_statistics(self) -> Dict:
        """Get transporter statistics"""
        return {
            'transport_events': self.transport_events,
            'rejected_molecules': self.rejected_molecules,
            'current_state': self.current_state.value,
            'current_time': self.current_time,
            'atp_turnover_rate': self.atp_modulation_frequency,
            'selectivity_factor': self.transport_events / max(self.rejected_molecules, 1)
        }


def create_example_substrates() -> List[SubstrateVibrationalProfile]:
    """
    Create example substrate molecules for testing.

    Based on known P-glycoprotein substrates and non-substrates.
    """

    substrates = []

    # SUBSTRATE 1: Doxorubicin (anticancer drug, known P-gp substrate)
    # MW: 543.5 Da, positive charge, aromatic
    substrates.append(SubstrateVibrationalProfile(
        name="Doxorubicin",
        molecular_weight=543.5,
        fundamental_frequency=3.5e13,  # Hz (aromatic C-H, C=O)
        harmonic_modes=[7.0e13, 1.05e14],  # First two harmonics
        s_coordinates=SEntropyCoordinates(S_k=0.6, S_t=0.2, S_e=0.7),
        charge=+1,
        hydrophobicity=1.3
    ))

    # SUBSTRATE 2: Verapamil (calcium channel blocker, P-gp substrate)
    # MW: 454.6 Da, positive charge
    substrates.append(SubstrateVibrationalProfile(
        name="Verapamil",
        molecular_weight=454.6,
        fundamental_frequency=3.8e13,  # Hz (close to transporter frequency)
        harmonic_modes=[7.6e13, 1.14e14],
        s_coordinates=SEntropyCoordinates(S_k=0.7, S_t=0.3, S_e=0.8),
        charge=+1,
        hydrophobicity=3.8
    ))

    # NON-SUBSTRATE 1: Glucose (too small, wrong frequency)
    # MW: 180.2 Da, neutral
    substrates.append(SubstrateVibrationalProfile(
        name="Glucose",
        molecular_weight=180.2,
        fundamental_frequency=2.5e13,  # Hz (O-H stretch, different range)
        harmonic_modes=[5.0e13, 7.5e13],
        s_coordinates=SEntropyCoordinates(S_k=0.2, S_t=0.1, S_e=0.3),
        charge=0,
        hydrophobicity=-3.0
    ))

    # SUBSTRATE 3: Rhodamine 123 (fluorescent dye, P-gp substrate)
    # MW: 380.8 Da, positive charge
    substrates.append(SubstrateVibrationalProfile(
        name="Rhodamine_123",
        molecular_weight=380.8,
        fundamental_frequency=3.7e13,  # Hz (aromatic, similar to site)
        harmonic_modes=[7.4e13, 1.11e14],
        s_coordinates=SEntropyCoordinates(S_k=0.65, S_t=0.25, S_e=0.75),
        charge=+1,
        hydrophobicity=2.5
    ))

    # NON-SUBSTRATE 2: Metformin (too hydrophilic, wrong frequency)
    # MW: 129.2 Da, positive charge
    substrates.append(SubstrateVibrationalProfile(
        name="Metformin",
        molecular_weight=129.2,
        fundamental_frequency=2.8e13,  # Hz (N-H stretch, different)
        harmonic_modes=[5.6e13, 8.4e13],
        s_coordinates=SEntropyCoordinates(S_k=0.3, S_t=0.15, S_e=0.4),
        charge=+2,
        hydrophobicity=-1.4
    ))

    return substrates


def validate_phase_locking():
    """
    Main validation: Demonstrate phase-locked substrate selection.
    """

    print("\n" + "="*70)
    print("VALIDATION: PHASE-LOCKED SUBSTRATE SELECTION")
    print("="*70 + "\n")

    # Create transporter
    transporter = PhaseLockingTransporter("ABC_exporter")

    # Create test substrates
    substrates = create_example_substrates()

    print("Test Substrates:")
    print("-" * 70)
    for sub in substrates:
        print(f"{sub.name:20s} | MW={sub.molecular_weight:6.1f} Da | "
              f"f₀={sub.fundamental_frequency:.2e} Hz | charge={sub.charge:+d}")
    print()

    # Simulate selection
    results = transporter.simulate_substrate_selection(
        substrates,
        simulation_time=5.0  # 5 seconds
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print("\nPhase-Lock Strengths:")
    print("-" * 70)
    for name, strength in results['phase_lock_strengths'].items():
        status = "✓ TRANSPORTED" if name in results['transported'] else "✗ REJECTED"
        print(f"{name:20s} | {strength:.3f} | {status}")

    print(f"\nTransport Efficiency: {results['transport_efficiency']:.1%}")
    print(f"Selectivity Factor: {results['selectivity']:.2e}")

    print("\nTransporter Statistics:")
    stats = transporter.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)

    return results


if __name__ == "__main__":
    validate_phase_locking()
