"""
Ensemble Transporter Demon: Collective Maxwell Demon Behavior
==============================================================

KEY INSIGHT: A cell has ~1,000-10,000 copies of each transporter type.
Rather than tracking individual transporters, we model the ENSEMBLE as
a single collective demon that IS all transporters simultaneously.

This is analogous to:
- Atmospheric demon = all molecules at S-coordinate
- Pixel demon = all molecular demons in pixel
- Ensemble transporter demon = all transporters of one type

Emergent Ensemble Properties:
-----------------------------
1. ENHANCED THROUGHPUT: N transporters → N× transport rate
2. FREQUENCY COVERAGE: Statistical ensemble covers wider frequency range
3. COLLECTIVE SELECTIVITY: Ensemble statistics sharpen selectivity
4. COORDINATED SCANNING: ATP cycles can synchronize across ensemble
5. CATEGORICAL ADDRESSING: Access entire ensemble through S-coordinates

This explains:
- How cells handle multiple substrate types simultaneously
- Why transporter density affects drug resistance
- How membrane composition modulates transport
- Collective phase-locking in membrane domains
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
from phase_locked_selection import (
    PhaseLockingTransporter,
    SubstrateVibrationalProfile
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleStatistics:
    """
    Statistical properties of transporter ensemble.
    """
    num_transporters: int                    # Total transporters in ensemble
    num_active: int                          # Currently in transport cycle
    num_available: int                       # Ready to bind substrate
    total_transport_events: int              # Cumulative transports
    avg_cycle_time: float                    # Average cycle duration (s)
    ensemble_throughput: float               # Molecules/second
    collective_selectivity: float            # Ensemble selectivity factor
    s_coordinate_spread: float               # Dispersion in S-space

    def to_dict(self) -> Dict:
        return {
            'num_transporters': self.num_transporters,
            'num_active': self.num_active,
            'num_available': self.num_available,
            'total_transport_events': self.total_transport_events,
            'avg_cycle_time': self.avg_cycle_time,
            'ensemble_throughput': self.ensemble_throughput,
            'collective_selectivity': self.collective_selectivity,
            's_coordinate_spread': self.s_coordinate_spread
        }


class EnsembleTransporterDemon:
    """
    Collective Maxwell Demon representing ALL transporters of one type in a cell.

    Instead of tracking individual transporters, we model the ensemble as
    a single entity that operates in both:
    1. Physical space: Distributed across membrane
    2. S-entropy space: Collective categorical state

    The ensemble demon exhibits emergent properties not present in
    individual transporters.
    """

    def __init__(self,
                 transporter_type: str = "P-glycoprotein",
                 num_transporters: int = 5000,
                 membrane_area_um2: float = 1000.0):
        """
        Initialize ensemble transporter demon.

        Args:
            transporter_type: Type of transporter (e.g., P-glycoprotein, ABCB1)
            num_transporters: Number of copies in cell membrane
            membrane_area_um2: Total membrane area (μm²)
        """

        self.transporter_type = transporter_type
        self.num_transporters = num_transporters
        self.membrane_area_um2 = membrane_area_um2

        # Density (transporters per μm²)
        self.density = num_transporters / membrane_area_um2

        # Individual transporter template
        self.template = PhaseLockingTransporter("ABC_exporter")

        # Ensemble state distribution
        # Instead of tracking each transporter, we track probability distribution
        self.state_distribution = self._initialize_state_distribution()

        # Ensemble S-coordinates (collective categorical state)
        self.ensemble_s_coordinate = self._calculate_ensemble_s_coordinate()

        # Statistics
        self.total_transport_events = 0
        self.total_rejection_events = 0
        self.cycle_times: List[float] = []

        # Time
        self.current_time = 0.0

        # Temperature
        self.temperature = 310.0  # K

        logger.info(f"Initialized {transporter_type} ensemble demon:")
        logger.info(f"  Transporters: {num_transporters}")
        logger.info(f"  Membrane area: {membrane_area_um2:.1f} μm²")
        logger.info(f"  Density: {self.density:.2f} transporters/μm²")

    def _initialize_state_distribution(self) -> Dict[TransporterState, float]:
        """
        Initialize probability distribution over conformational states.

        At equilibrium (no substrate), most transporters are in OPEN_OUTSIDE
        waiting for substrates.
        """

        return {
            TransporterState.OPEN_OUTSIDE: 0.85,  # Most waiting
            TransporterState.OCCLUDED: 0.05,      # Few transitioning
            TransporterState.OPEN_INSIDE: 0.05,   # Few releasing
            TransporterState.RESETTING: 0.05      # Few resetting
        }

    def _calculate_ensemble_s_coordinate(self) -> SEntropyCoordinates:
        """
        Calculate collective S-entropy coordinate for ensemble.

        This is the ENSEMBLE state - the S-coordinate that represents
        ALL transporters simultaneously.
        """

        landscape = self.template.landscape

        # Weighted average of individual S-coordinates
        S_k_avg = sum(
            prob * landscape.get_state(state).s_coordinates.S_k
            for state, prob in self.state_distribution.items()
        )

        S_t_avg = sum(
            prob * landscape.get_state(state).s_coordinates.S_t
            for state, prob in self.state_distribution.items()
        )

        S_e_avg = sum(
            prob * landscape.get_state(state).s_coordinates.S_e
            for state, prob in self.state_distribution.items()
        )

        return SEntropyCoordinates(S_k_avg, S_t_avg, S_e_avg)

    def get_num_available(self) -> int:
        """Number of transporters available to bind substrate"""
        return int(self.num_transporters * self.state_distribution[TransporterState.OPEN_OUTSIDE])

    def get_num_active(self) -> int:
        """Number of transporters currently in transport cycle"""
        return self.num_transporters - self.get_num_available()

    def collective_phase_lock_strength(self,
                                      substrate: SubstrateVibrationalProfile,
                                      time: float) -> float:
        """
        Calculate collective phase-lock strength of ensemble.

        KEY INSIGHT: Ensemble has ENHANCED phase-locking because:
        1. Statistical averaging over many transporters
        2. Distributed ATP cycle phases → continuous frequency scanning
        3. Membrane domains may synchronize cycles

        Returns effective phase-lock strength (0-1).
        """

        # Each transporter has slightly different ATP cycle phase
        # This means ensemble continuously scans frequency range

        # Get template phase-lock
        template_freq = self.template.get_current_binding_frequency(time)
        base_phase_lock = substrate.calculate_phase_lock_strength(
            template_freq,
            self.template.phase_lock_threshold
        )

        # Ensemble enhancement factor
        # More transporters → better statistical coverage of frequency space
        enhancement = 1.0 + 0.5 * np.log10(self.num_transporters / 100)

        # Available transporters factor
        # More available → higher probability of match
        availability_factor = self.state_distribution[TransporterState.OPEN_OUTSIDE]

        # Collective phase-lock
        collective_strength = min(1.0, base_phase_lock * enhancement * (1 + availability_factor))

        return collective_strength

    def ensemble_transport_rate(self,
                                substrate: SubstrateVibrationalProfile,
                                time: float) -> float:
        """
        Calculate ensemble transport rate (molecules/second).

        This is the COLLECTIVE throughput - much higher than individual transporter.
        """

        # Phase-lock probability
        phase_lock = self.collective_phase_lock_strength(substrate, time)

        # Number of available transporters
        num_available = self.get_num_available()

        # Individual transporter rate (from ATP cycle)
        individual_rate = self.template.atp_modulation_frequency  # ~10 Hz

        # Ensemble rate = individual rate × num available × phase-lock probability
        ensemble_rate = individual_rate * num_available * phase_lock

        return ensemble_rate

    def transport_substrate_ensemble(self,
                                    substrate: SubstrateVibrationalProfile,
                                    num_molecules: int,
                                    duration: float = 1.0) -> Dict:
        """
        Transport multiple substrate molecules using ensemble.

        Args:
            substrate: Substrate to transport
            num_molecules: Number of substrate molecules available
            duration: Time period (seconds)

        Returns:
            Dict with transport statistics
        """

        logger.info(f"\nEnsemble transport of {substrate.name}:")
        logger.info(f"  Available molecules: {num_molecules}")
        logger.info(f"  Duration: {duration:.2f} s")
        logger.info(f"  Available transporters: {self.get_num_available()}")

        # Calculate ensemble transport rate
        transport_rate = self.ensemble_transport_rate(substrate, self.current_time)
        logger.info(f"  Ensemble transport rate: {transport_rate:.1f} molecules/s")

        # Calculate number transported
        # Limited by either transport capacity or available molecules
        max_transport = transport_rate * duration
        num_transported = min(int(max_transport), num_molecules)

        # Update statistics
        self.total_transport_events += num_transported
        self.total_rejection_events += (num_molecules - num_transported)

        # Update time
        self.current_time += duration

        # Calculate collective selectivity
        phase_lock = self.collective_phase_lock_strength(substrate, self.current_time)

        results = {
            'substrate': substrate.name,
            'molecules_available': num_molecules,
            'molecules_transported': num_transported,
            'molecules_rejected': num_molecules - num_transported,
            'transport_rate': transport_rate,
            'efficiency': num_transported / num_molecules if num_molecules > 0 else 0,
            'collective_phase_lock': phase_lock,
            'duration': duration,
            'transporters_active': int(num_transported / max(transport_rate * duration / self.get_num_available(), 1))
        }

        logger.info(f"  Transported: {num_transported}/{num_molecules} ({results['efficiency']:.1%})")
        logger.info(f"  Collective phase-lock: {phase_lock:.3f}")

        return results

    def multi_substrate_competition(self,
                                   substrates: List[SubstrateVibrationalProfile],
                                   substrate_concentrations: Dict[str, int],
                                   duration: float = 1.0) -> Dict:
        """
        Simulate competition between multiple substrates for ensemble.

        This reveals EMERGENT SELECTIVITY from collective demon behavior:
        - Substrates compete for available transporters
        - Phase-locking determines selection probabilities
        - Ensemble statistics sharpen selectivity

        Args:
            substrates: List of competing substrates
            substrate_concentrations: {substrate_name: num_molecules}
            duration: Competition time (seconds)

        Returns:
            Transport results for each substrate
        """

        logger.info("="*70)
        logger.info("MULTI-SUBSTRATE COMPETITION")
        logger.info("="*70)

        # Calculate phase-lock for each substrate
        phase_locks = {}
        for substrate in substrates:
            phase_locks[substrate.name] = self.collective_phase_lock_strength(
                substrate,
                self.current_time
            )

        logger.info("\nPhase-lock strengths:")
        for name, strength in phase_locks.items():
            logger.info(f"  {name:20s}: {strength:.3f}")

        # Calculate transport probabilities (normalized phase-locks)
        total_phase_lock = sum(phase_locks.values())
        transport_probs = {
            name: pl / total_phase_lock if total_phase_lock > 0 else 0
            for name, pl in phase_locks.items()
        }

        # Total available molecules
        total_molecules = sum(substrate_concentrations.values())

        # Ensemble transport capacity
        total_capacity = self.get_num_available() * self.template.atp_modulation_frequency * duration

        # Distribute capacity according to transport probabilities
        results = {}
        for substrate in substrates:
            name = substrate.name
            available = substrate_concentrations.get(name, 0)

            # Allocated capacity for this substrate
            allocated_capacity = total_capacity * transport_probs[name]

            # Actual transported (limited by availability and capacity)
            transported = min(int(allocated_capacity), available)

            results[name] = {
                'available': available,
                'transported': transported,
                'rejected': available - transported,
                'phase_lock': phase_locks[name],
                'transport_probability': transport_probs[name],
                'efficiency': transported / available if available > 0 else 0
            }

        # Update statistics
        total_transported = sum(r['transported'] for r in results.values())
        self.total_transport_events += total_transported
        self.total_rejection_events += sum(r['rejected'] for r in results.values())
        self.current_time += duration

        # Summary
        logger.info("\nTransport Results:")
        logger.info("-" * 70)
        for name, data in results.items():
            logger.info(f"{name:20s}: {data['transported']:4d}/{data['available']:4d} "
                       f"({data['efficiency']:5.1%}) | "
                       f"prob={data['transport_probability']:.3f}")

        logger.info(f"\nTotal transported: {total_transported}/{total_molecules} "
                   f"({total_transported/total_molecules:.1%})")
        logger.info("="*70)

        results_summary = {
            'substrates': results,
            'total_molecules': total_molecules,
            'total_transported': total_transported,
            'total_rejected': sum(r['rejected'] for r in results.values()),
            'ensemble_efficiency': total_transported / total_molecules if total_molecules > 0 else 0,
            'duration': duration,
            'collective_selectivity': max(phase_locks.values()) / min([v for v in phase_locks.values() if v > 0] + [1e-10])
        }

        return results_summary

    def get_ensemble_statistics(self) -> EnsembleStatistics:
        """Get current ensemble statistics"""

        avg_cycle = np.mean(self.cycle_times) if self.cycle_times else 0.1
        throughput = self.total_transport_events / max(self.current_time, 1e-6)

        # Calculate S-coordinate spread (ensemble coherence)
        spread = 0.1  # Simplified - would calculate from state distribution

        selectivity = self.total_transport_events / max(self.total_rejection_events, 1)

        return EnsembleStatistics(
            num_transporters=self.num_transporters,
            num_active=self.get_num_active(),
            num_available=self.get_num_available(),
            total_transport_events=self.total_transport_events,
            avg_cycle_time=avg_cycle,
            ensemble_throughput=throughput,
            collective_selectivity=selectivity,
            s_coordinate_spread=spread
        )

    def get_statistics_dict(self) -> Dict:
        """Get statistics as dict"""
        stats = self.get_ensemble_statistics()
        return {
            'transporter_type': self.transporter_type,
            **stats.to_dict(),
            'membrane_area_um2': self.membrane_area_um2,
            'density_per_um2': self.density,
            'current_time': self.current_time
        }


def validate_ensemble_demon():
    """
    Validate ensemble transporter demon.

    Demonstrates emergent collective behavior.
    """

    print("\n" + "="*70)
    print("ENSEMBLE TRANSPORTER DEMON VALIDATION")
    print("="*70 + "\n")

    # Create ensemble demon (5000 P-glycoprotein molecules)
    ensemble = EnsembleTransporterDemon(
        transporter_type="P-glycoprotein",
        num_transporters=5000,
        membrane_area_um2=1000.0
    )

    # Create substrates
    from phase_locked_selection import create_example_substrates
    substrates = create_example_substrates()

    print("\n" + "="*70)
    print("TEST 1: SINGLE SUBSTRATE ENSEMBLE TRANSPORT")
    print("="*70)

    # Transport Verapamil (known P-gp substrate)
    verapamil = substrates[1]
    result = ensemble.transport_substrate_ensemble(
        verapamil,
        num_molecules=10000,
        duration=1.0
    )

    print(f"\n✓ Ensemble transported {result['molecules_transported']} molecules in 1 second")
    print(f"✓ Transport rate: {result['transport_rate']:.1f} molecules/s")
    print(f"✓ Efficiency: {result['efficiency']:.1%}")

    print("\n" + "="*70)
    print("TEST 2: MULTI-SUBSTRATE COMPETITION")
    print("="*70)

    # Multiple competing substrates
    concentrations = {
        'Doxorubicin': 5000,
        'Verapamil': 5000,
        'Glucose': 5000,
        'Rhodamine_123': 5000,
        'Metformin': 5000
    }

    competition_results = ensemble.multi_substrate_competition(
        substrates,
        concentrations,
        duration=1.0
    )

    print(f"\n✓ Collective selectivity: {competition_results['collective_selectivity']:.2e}")
    print(f"✓ Overall efficiency: {competition_results['ensemble_efficiency']:.1%}")

    # Statistics
    stats = ensemble.get_statistics_dict()
    print("\n" + "="*70)
    print("ENSEMBLE STATISTICS")
    print("="*70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)

    return {
        'single_substrate': result,
        'competition': competition_results,
        'statistics': stats
    }


if __name__ == "__main__":
    results = validate_ensemble_demon()
