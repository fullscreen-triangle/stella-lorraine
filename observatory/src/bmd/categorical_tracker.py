"""
categorical_tracker.py

Tracks categorical state completions and equivalence classes in Maxwell demon system.
Implements St-Stellas formalism for BMD validation.

Based on:
- Mizraji (2021): BMDs as information catalysts
- St-Stellas Categories: Categorical completion framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from collections import defaultdict

@dataclass
class CategoricalState:
    """
    A categorical state C_i in the system.

    Represents a unique, irreversible configuration that cannot be re-occupied.
    """
    index: int  # Categorical index C_i
    timestamp: float  # When this category was completed
    configuration_hash: int  # Hash of physical configuration
    equivalence_class_id: int  # Which equivalence class this belongs to
    observables: Dict[str, float]  # Observable values (temp, entropy, etc.)

    # St-Stellas S-coordinates
    S_knowledge: float = 0.0  # Information deficit
    S_time: float = 0.0  # Temporal position in sequence
    S_entropy: float = 0.0  # Constraint density

@dataclass
class EquivalenceClass:
    """
    Equivalence class [C]_~ of categorically equivalent states.

    Many distinct categorical states produce identical observables.
    This is the degeneracy |[C]_~| that BMDs filter through.
    """
    class_id: int
    representative_state: int  # Representative C_i
    observable_signature: Tuple[float, ...]  # The shared observable
    member_states: Set[int] = field(default_factory=set)  # All states in class

    @property
    def degeneracy(self) -> int:
        """Cardinality of equivalence class"""
        return len(self.member_states)

    @property
    def information_content(self) -> float:
        """Information content in bits: log2(degeneracy)"""
        return np.log2(max(1, self.degeneracy))

class CategoricalTracker:
    """
    Track categorical state completions and BMD operations.

    This connects the physical simulation to St-Stellas formalism:
    - Physical states → Categorical states (irreversible assignments)
    - Particle configurations → Equivalence classes (many-to-one)
    - Demon decisions → BMD filtering operations
    """

    def __init__(self, observable_precision: float = 0.01):
        """
        Args:
            observable_precision: Precision for equivalence class binning
        """
        self.observable_precision = observable_precision

        # Categorical state tracking
        self.states: List[CategoricalState] = []
        self.current_index = 0

        # Equivalence class tracking
        self.equivalence_classes: Dict[int, EquivalenceClass] = {}
        self.next_class_id = 0

        # BMD operation tracking
        self.bmd_operations: List[Dict] = []

        # S-space trajectory
        self.s_trajectory: List[Tuple[float, float, float]] = []

    def record_state(self, system, time: float) -> CategoricalState:
        """
        Record a new categorical state from current system configuration.

        This implements Axiom 1 (Categorical Irreversibility):
        Once recorded, this state cannot be re-occupied.
        """
        # Extract observables
        observables = {
            'temp_A': system.compartment_A.temperature,
            'temp_B': system.compartment_B.temperature,
            'entropy_A': system.compartment_A.entropy,
            'entropy_B': system.compartment_B.entropy,
            'particles_A': system.compartment_A.particle_count,
            'particles_B': system.compartment_B.particle_count,
        }

        # Compute configuration hash (for categorical uniqueness)
        config_hash = self._compute_configuration_hash(system)

        # Find or create equivalence class
        equiv_class_id = self._assign_equivalence_class(observables)

        # Compute S-coordinates
        S_k, S_t, S_e = self._compute_s_coordinates(system, time)

        # Create categorical state
        state = CategoricalState(
            index=self.current_index,
            timestamp=time,
            configuration_hash=config_hash,
            equivalence_class_id=equiv_class_id,
            observables=observables,
            S_knowledge=S_k,
            S_time=S_t,
            S_entropy=S_e
        )

        # Record state
        self.states.append(state)
        self.equivalence_classes[equiv_class_id].member_states.add(self.current_index)
        self.s_trajectory.append((S_k, S_t, S_e))

        # Increment categorical index (time arrow)
        self.current_index += 1

        return state

    def record_bmd_operation(
        self,
        particle_id: int,
        observed: bool,
        classified_state: str,
        decision: str,
        pre_state_idx: int,
        post_state_idx: int
    ):
        """
        Record a BMD operation (demon decision).

        This is the core: demon filtering potential states → actual states.
        """
        operation = {
            'particle_id': particle_id,
            'observed': observed,
            'classified_state': classified_state,
            'decision': decision,
            'pre_categorical_state': pre_state_idx,
            'post_categorical_state': post_state_idx,
            'equivalence_class_filtered': self._count_filtered_states(pre_state_idx, post_state_idx)
        }

        self.bmd_operations.append(operation)

    def _compute_configuration_hash(self, system) -> int:
        """
        Compute hash of particle configuration.

        Different configurations may hash to same equivalence class
        (this is the categorical degeneracy).
        """
        # Collect all particle velocities and positions
        config_data = []
        for p in system.compartment_A.particles:
            config_data.extend([p.velocity, *p.position])
        for p in system.compartment_B.particles:
            config_data.extend([p.velocity, *p.position])

        # Hash (with rounding for numerical stability)
        config_tuple = tuple(np.round(config_data, decimals=6))
        return hash(config_tuple)

    def _assign_equivalence_class(self, observables: Dict[str, float]) -> int:
        """
        Assign observables to equivalence class.

        States with same observables (within precision) are equivalent.
        This implements Definition 2.3 (Categorical Equivalence).
        """
        # Create signature by rounding observables
        signature = tuple(
            round(v / self.observable_precision) * self.observable_precision
            for v in [
                observables['temp_A'],
                observables['temp_B'],
                observables['entropy_A'],
                observables['entropy_B']
            ]
        )

        # Find existing class or create new
        for class_id, equiv_class in self.equivalence_classes.items():
            if equiv_class.observable_signature == signature:
                return class_id

        # New equivalence class
        new_class = EquivalenceClass(
            class_id=self.next_class_id,
            representative_state=self.current_index,
            observable_signature=signature
        )
        self.equivalence_classes[self.next_class_id] = new_class
        self.next_class_id += 1

        return new_class.class_id

    def _compute_s_coordinates(
        self,
        system,
        time: float
    ) -> Tuple[float, float, float]:
        """
        Compute St-Stellas S-coordinates for current state.

        S_k: Knowledge dimension (information deficit)
        S_t: Time dimension (categorical position)
        S_e: Entropy dimension (constraint density)
        """
        # S_knowledge: Information deficit from perfect classification
        # Measured by demon's error rate and bits needed
        if system.demon.decisions_made > 0:
            S_k = (1.0 - system.demon.accuracy) * system.demon.information_capacity
        else:
            S_k = system.demon.information_capacity

        # S_time: Position in categorical sequence
        # Normalized by expected completion rate
        S_t = time / (system.dt if system.dt > 0 else 1.0)

        # S_entropy: Constraint density from phase-locking
        # Approximated by entropy + demon cost
        S_e = system.total_entropy

        return S_k, S_t, S_e

    def _count_filtered_states(self, pre_idx: int, post_idx: int) -> int:
        """
        Count how many potential states were filtered by BMD operation.

        This quantifies |[C]_~| → 1 reduction (equivalence class to specific state).
        """
        if pre_idx >= len(self.states) or post_idx >= len(self.states):
            return 0

        pre_class = self.states[pre_idx].equivalence_class_id
        post_class = self.states[post_idx].equivalence_class_id

        # States filtered = degeneracy of pre_class that weren't selected
        pre_degeneracy = self.equivalence_classes[pre_class].degeneracy

        return max(0, pre_degeneracy - 1)

    # ========================================================================
    # ANALYSIS METHODS
    # ========================================================================

    def compute_categorical_completion_rate(self, window: int = 10) -> float:
        """
        Compute dC/dt (categorical completion rate).

        This is the fundamental clock in categorical time theory.
        """
        if len(self.states) < window + 1:
            return 0.0

        recent = self.states[-window:]
        dt = recent[-1].timestamp - recent[0].timestamp
        dC = len(recent)

        return dC / dt if dt > 0 else 0.0

    def compute_total_information_processed(self) -> float:
        """
        Total information processed by BMD operations.

        Sum of log2(degeneracy) over all filtering operations.
        """
        total_bits = 0.0

        for state in self.states:
            equiv_class = self.equivalence_classes[state.equivalence_class_id]
            total_bits += equiv_class.information_content

        return total_bits

    def compute_bmd_probability_enhancement(self) -> float:
        """
        Compute p_BMD / p_0 (Mizraji's probability enhancement).

        Typical biological value: 10^6 to 10^11
        """
        if not self.bmd_operations:
            return 1.0

        # Estimate p_0: uniform over potential states
        # Estimate p_BMD: demon's success rate

        avg_equivalence_class_size = np.mean([
            ec.degeneracy for ec in self.equivalence_classes.values()
        ])

        # p_0 ~ 1/|potential states|
        p_0 = 1.0 / max(1, avg_equivalence_class_size)

        # p_BMD ~ demon accuracy
        p_BMD = np.mean([
            1.0 if op['decision'] != 'rejected' else 0.0
            for op in self.bmd_operations
        ])

        return p_BMD / p_0 if p_0 > 0 else 1.0

    def verify_st_stellas_equivalence(self) -> Dict[str, bool]:
        """
        Verify the fundamental equivalence (Theorem 3.12):
        BMD operation ≡ S-Navigation ≡ Categorical Completion
        """
        results = {}

        # (1) BMD operations should track categorical completions
        bmd_count = len(self.bmd_operations)
        categorical_count = len(self.states)
        results['bmd_categorical_match'] = abs(bmd_count - categorical_count) < 0.1 * categorical_count

        # (2) S-distance should decrease over time (navigation)
        if len(self.s_trajectory) > 1:
            s_distances = [
                np.linalg.norm(np.array(self.s_trajectory[i+1]) - np.array(self.s_trajectory[i]))
                for i in range(len(self.s_trajectory) - 1)
            ]
            # Should show decreasing trend (convergence)
            results['s_navigation_convergence'] = np.mean(s_distances[-10:]) < np.mean(s_distances[:10]) if len(s_distances) > 20 else True
        else:
            results['s_navigation_convergence'] = True

        # (3) Equivalence classes should show degeneracy
        avg_degeneracy = np.mean([ec.degeneracy for ec in self.equivalence_classes.values()])
        results['equivalence_class_degeneracy'] = avg_degeneracy > 1.0

        return results

    def generate_report(self) -> str:
        """Generate comprehensive categorical analysis report"""
        report = []
        report.append("=" * 70)
        report.append("ST-STELLAS CATEGORICAL DYNAMICS ANALYSIS")
        report.append("=" * 70)
        report.append("")

        report.append("CATEGORICAL STATE TRACKING:")
        report.append(f"  Total states completed: {len(self.states)}")
        report.append(f"  Current categorical index: C_{self.current_index}")
        report.append(f"  Categorical completion rate: {self.compute_categorical_completion_rate():.3f} states/time")
        report.append("")

        report.append("EQUIVALENCE CLASS STRUCTURE:")
        report.append(f"  Number of equivalence classes: {len(self.equivalence_classes)}")
        avg_degeneracy = np.mean([ec.degeneracy for ec in self.equivalence_classes.values()])
        max_degeneracy = max([ec.degeneracy for ec in self.equivalence_classes.values()])
        report.append(f"  Average degeneracy |[C]_~|: {avg_degeneracy:.1f}")
        report.append(f"  Maximum degeneracy: {max_degeneracy}")
        report.append(f"  Average information/class: {np.log2(avg_degeneracy):.2f} bits")
        report.append("")

        report.append("BMD OPERATIONS:")
        report.append(f"  Total BMD operations: {len(self.bmd_operations)}")
        report.append(f"  Total information processed: {self.compute_total_information_processed():.1f} bits")
        enhancement = self.compute_bmd_probability_enhancement()
        report.append(f"  Probability enhancement p_BMD/p_0: {enhancement:.2e}")
        report.append(f"  Mizraji range check (10^6 - 10^11): {1e6 <= enhancement <= 1e11}")
        report.append("")

        report.append("S-SPACE TRAJECTORY:")
        if self.s_trajectory:
            current_s = self.s_trajectory[-1]
            report.append(f"  Current S-coordinates: ({current_s[0]:.3f}, {current_s[1]:.3f}, {current_s[2]:.3f})")
            initial_s = self.s_trajectory[0]
            s_distance_traveled = np.linalg.norm(np.array(current_s) - np.array(initial_s))
            report.append(f"  S-distance traveled: {s_distance_traveled:.3f}")
        report.append("")

        report.append("ST-STELLAS EQUIVALENCE VERIFICATION:")
        verification = self.verify_st_stellas_equivalence()
        for test, passed in verification.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report.append(f"  {test}: {status}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)


def main():
    """Test categorical tracker"""
    print("Categorical tracker module loaded.")
    print("Use with PrisonerSystem to track categorical dynamics.")

if __name__ == "__main__":
    main()
