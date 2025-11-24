"""
GroEL Resonance Chamber - Phase-Locked Chaperonin Dynamics.

Models GroEL as an ATP-driven resonance chamber that operates in cycles,
each cycle providing a specific frequency environment for protein phase-locking.

Key insight from papers: GroEL doesn't just "encapsulate" - it's an active
oscillatory filter that cycles through frequency space, testing protein
configurations for phase-lock stability at each cycle.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from protein_folding_network import ProteinFoldingNetwork, PhaseCoherenceCluster
from proton_maxwell_demon import O2_MASTER_CLOCK_HZ, GROEL_BASE_HZ

logger = logging.getLogger(__name__)


@dataclass
class ATPCycleState:
    """
    State of GroEL during one ATP hydrolysis cycle.

    From papers: Each ATP cycle has specific phase where cavity
    provides optimal resonance for certain protein configurations.
    """
    cycle_number: int
    time_in_cycle: float  # seconds
    atp_state: str  # 'ATP_bound', 'transition', 'ADP_Pi', 'ADP_release'
    cavity_frequency: float  # Hz
    cavity_phase: float  # radians
    cavity_volume: float  # Angstrom^3
    temperature: float  # K


class GroELResonanceChamber:
    """
    GroEL chaperonin as phase-locked resonance chamber.

    Enhanced with cyclical dynamics from papers:
    - ATP-driven cycles modulate cavity frequency
    - Each cycle tests different frequency space
    - Protein folds through iterative phase-locking refinement
    - Synchronized to cytoplasmic O₂ master clock

    Key mechanism: GroEL doesn't fold protein directly - it provides
    a cyclic scanning of frequency space, allowing protein to find
    its native phase-locked state through variance minimization.
    """

    def __init__(self, temperature: float = 310.0):
        """
        Initialize GroEL resonance chamber.

        Args:
            temperature: System temperature (K)
        """
        self.temperature = temperature

        # ATP cycle parameters
        self.cycle_duration = 1.0  # seconds (typical GroEL cycle time)
        self.current_cycle = 0
        self.time_in_cycle = 0.0

        # Cavity properties
        self.cavity_base_frequency = GROEL_BASE_HZ  # ATP cycle frequency (1 Hz)
        self.cavity_vibrational_base = O2_MASTER_CLOCK_HZ  # Cavity coupled to O₂ master clock (10 THz)
        self.cavity_volume = 85000.0  # Angstrom^3 (approximate)

        # Frequency modulation during cycle (from papers)
        # ATP cycle modulates which vibrational harmonics are active
        # These harmonics span the range of H-bond frequencies (1-10× O₂ clock)
        self.frequency_harmonics = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # multiples of O₂ clock

        # Phase-locking tracking
        self.protein_history: List[Dict] = []  # History of protein states per cycle
        self.best_cycle: Optional[int] = None  # Cycle with lowest variance
        self.best_stability: float = 0.0

        logger.info(f"Initialized GroELResonanceChamber at T={temperature}K")
        logger.info(f"  ATP cycle frequency: {self.cavity_base_frequency} Hz")
        logger.info(f"  Cavity vibrational base: {self.cavity_vibrational_base:.2e} Hz")
        logger.info(f"  Cycle duration: {self.cycle_duration}s")
        logger.info(f"  Cavity volume: {self.cavity_volume} Ų")

    def get_current_cavity_state(self) -> ATPCycleState:
        """
        Get current state of GroEL cavity based on ATP cycle position.

        From papers: Cavity properties modulate through ATP cycle:
        - ATP binding: cavity contracts, frequency increases
        - Transition state: maximum frequency
        - ADP+Pi: cavity expands, frequency decreases
        - ADP release: return to baseline
        """
        # Determine ATP state from cycle time
        t_frac = self.time_in_cycle / self.cycle_duration

        if t_frac < 0.25:
            atp_state = 'ATP_bound'
            volume_factor = 0.9  # Contracted
            freq_multiplier = 2.0  # Higher frequency
        elif t_frac < 0.50:
            atp_state = 'transition'
            volume_factor = 0.85  # Most contracted
            freq_multiplier = 3.0  # Highest frequency
        elif t_frac < 0.75:
            atp_state = 'ADP_Pi'
            volume_factor = 1.1  # Expanded
            freq_multiplier = 1.5  # Medium frequency
        else:
            atp_state = 'ADP_release'
            volume_factor = 1.0  # Baseline
            freq_multiplier = 1.0  # Base frequency

        # Calculate cavity frequency (samples harmonic series of vibrational modes)
        # ATP cycle selects which harmonic, phase within cycle modulates it
        harmonic_idx = self.current_cycle % len(self.frequency_harmonics)
        harmonic = self.frequency_harmonics[harmonic_idx]
        cavity_freq = self.cavity_vibrational_base * harmonic * freq_multiplier

        # Cavity phase evolves continuously
        cavity_phase = (2 * np.pi * cavity_freq * self.time_in_cycle) % (2 * np.pi)

        # Volume modulation
        cavity_volume = self.cavity_volume * volume_factor

        return ATPCycleState(
            cycle_number=self.current_cycle,
            time_in_cycle=self.time_in_cycle,
            atp_state=atp_state,
            cavity_frequency=cavity_freq,
            cavity_phase=cavity_phase,
            cavity_volume=cavity_volume,
            temperature=self.temperature
        )

    def advance_cycle(self, protein: ProteinFoldingNetwork) -> Dict:
        """
        Advance one full ATP cycle, observing protein response.

        From papers: Each cycle is a "measurement" of protein configuration
        at a specific frequency. Protein responds by phase-locking (or not).

        Args:
            protein: Protein folding network

        Returns:
            Cycle summary with stability metrics
        """
        self.current_cycle += 1
        self.time_in_cycle = 0.0

        # Simulate cycle in small time steps
        n_steps = 100
        dt = self.cycle_duration / n_steps

        stability_trace = []
        variance_trace = []

        for step in range(n_steps):
            # Get current cavity state
            cavity_state = self.get_current_cavity_state()

            # Update protein with cavity state
            protein.update_groel_state(
                cycle=self.current_cycle,
                phase=cavity_state.cavity_phase,
                frequency=cavity_state.cavity_frequency
            )

            # Protein evolves phases under cavity coupling
            protein.simulate_cycle_step(dt)

            # Track metrics
            stability = protein.calculate_network_stability()
            variance = protein.calculate_network_variance()

            stability_trace.append(stability)
            variance_trace.append(variance)

            # Advance time
            self.time_in_cycle += dt

        # Cycle summary
        final_stability = stability_trace[-1]
        final_variance = variance_trace[-1]
        mean_stability = np.mean(stability_trace)

        # Check if this is best cycle so far
        if final_stability > self.best_stability:
            self.best_stability = final_stability
            self.best_cycle = self.current_cycle

        cycle_summary = {
            'cycle': self.current_cycle,
            'final_stability': float(final_stability),
            'final_variance': float(final_variance),
            'mean_stability': float(mean_stability),
            'min_variance': float(np.min(variance_trace)),
            'cavity_frequency_range': (
                float(np.min([self.get_current_cavity_state().cavity_frequency
                             for _ in range(4)])),
                float(np.max([self.get_current_cavity_state().cavity_frequency
                             for _ in range(4)]))
            ),
            'is_best_so_far': (self.current_cycle == self.best_cycle)
        }

        # Add to history
        self.protein_history.append(cycle_summary)

        logger.debug(f"Cycle {self.current_cycle} complete: "
                    f"stability={final_stability:.3f}, variance={final_variance:.3f}")

        return cycle_summary

    def run_folding_simulation(self, protein: ProteinFoldingNetwork,
                               max_cycles: int = 10,
                               stability_threshold: float = 0.8) -> Dict:
        """
        Run complete folding simulation through multiple ATP cycles.

        From papers: Protein folds through iterative refinement across cycles.
        Folding is complete when variance is minimized (high stability).

        Args:
            protein: Protein network to fold
            max_cycles: Maximum cycles to run
            stability_threshold: Stop if stability exceeds this

        Returns:
            Complete folding simulation results
        """
        logger.info(f"Starting GroEL folding simulation for {protein.protein_name}")
        logger.info(f"  Max cycles: {max_cycles}")
        logger.info(f"  Stability threshold: {stability_threshold}")

        # Reset state
        self.current_cycle = 0
        self.protein_history = []
        self.best_cycle = None
        self.best_stability = 0.0

        cycles_run = 0
        folding_complete = False

        for cycle_num in range(max_cycles):
            # Advance one cycle
            cycle_result = self.advance_cycle(protein)
            cycles_run += 1

            logger.info(f"  Cycle {cycle_num + 1}/{max_cycles}: "
                       f"stability={cycle_result['final_stability']:.3f}, "
                       f"variance={cycle_result['final_variance']:.4f}")

            # Check convergence
            if cycle_result['final_stability'] > stability_threshold:
                folding_complete = True
                logger.info(f"  ✓ Folding complete at cycle {cycle_num + 1}")
                break

            # Check if stuck (variance increasing over last 3 cycles)
            if len(self.protein_history) >= 3:
                recent_variances = [h['final_variance'] for h in self.protein_history[-3:]]
                if all(v1 < v2 for v1, v2 in zip(recent_variances[:-1], recent_variances[1:])):
                    logger.warning(f"  ⚠ Variance increasing - protein may be misfolding")

        # Final summary
        final_network_state = protein.get_network_summary()

        return {
            'protein': protein.protein_name,
            'cycles_run': cycles_run,
            'folding_complete': folding_complete,
            'best_cycle': self.best_cycle,
            'best_stability': self.best_stability,
            'final_stability': self.protein_history[-1]['final_stability'] if self.protein_history else 0.0,
            'final_variance': self.protein_history[-1]['final_variance'] if self.protein_history else float('inf'),
            'cycle_history': self.protein_history,
            'final_network_state': final_network_state
        }

    def identify_folding_pathway(self, protein: ProteinFoldingNetwork) -> List[Dict]:
        """
        Analyze cycle history to identify folding pathway.

        From papers: Folding pathway = sequence of phase-locked clusters
        that form across cycles.

        Returns:
            List of major folding events with cycle numbers
        """
        if not self.protein_history:
            return []

        pathway = []

        # Analyze each cycle
        for cycle_data in self.protein_history:
            cycle = cycle_data['cycle']

            # Get protein state at this cycle (need to re-simulate to get clusters)
            # For now, use stability changes as proxy
            if len(pathway) == 0:
                # First cycle
                pathway.append({
                    'cycle': cycle,
                    'event': 'initial_configuration',
                    'stability': cycle_data['final_stability']
                })
            else:
                # Check for significant stability increase (new cluster formed)
                prev_stability = pathway[-1]['stability']
                curr_stability = cycle_data['final_stability']

                if curr_stability > prev_stability + 0.1:  # Significant improvement
                    pathway.append({
                        'cycle': cycle,
                        'event': 'phase_lock_strengthened',
                        'stability': curr_stability,
                        'stability_gain': curr_stability - prev_stability
                    })
                elif curr_stability < prev_stability - 0.1:  # Destabilization
                    pathway.append({
                        'cycle': cycle,
                        'event': 'phase_lock_weakened',
                        'stability': curr_stability,
                        'stability_loss': prev_stability - curr_stability
                    })

        return pathway
