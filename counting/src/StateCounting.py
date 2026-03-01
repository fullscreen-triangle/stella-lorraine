"""
State Counting Mass Spectrometry Module
========================================

Implements state counting as a digital modality for mass spectrometry,
following the theoretical framework from state-counting-mass-spectrometry.tex.

Key Concepts:
- Partition Coordinates (n, l, m, s): Complete description of bounded phase space
- Capacity Formula C(n) = 2n²: Number of states at partition depth n
- State-Mass Correspondence: Bijective mapping between partition states and m/z
- Trajectory Completion: Sequence reconstruction via partition traversal
- Fragment-Parent Hierarchy: Spatial containment validation for fragments

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import math


# Reference mass for partition depth calculation
M_REF = 1.0  # Da

# Common modification masses
MODIFICATION_MASSES = {
    'TMT': 229.1629,          # TMT tag (N-terminal or K)
    'TMT6plex': 229.1629,     # TMT 6-plex
    'TMT10plex': 229.1629,    # TMT 10-plex (same mass as 6-plex, different isotopes)
    'iTRAQ4': 144.1021,       # iTRAQ 4-plex
    'iTRAQ8': 304.2054,       # iTRAQ 8-plex
    'Oxidation': 15.9949,     # Oxidation (M)
    'Carbamidomethyl': 57.0215,  # Carbamidomethylation (C)
    'Acetyl': 42.0106,        # Acetylation (Protein N-term)
    'Phospho': 79.9663,       # Phosphorylation (S, T, Y)
}


# ============================================================================
# PARTITION COORDINATES
# ============================================================================

@dataclass
class PartitionState:
    """
    Partition state coordinates (n, l, m, s).

    n: Principal number (partition depth)
    l: Angular complexity (0 to n-1)
    m: Orientation (-l to +l)
    s: Chirality (-1/2 or +1/2)
    """
    n: int
    l: int
    m: int
    s: float  # +0.5 or -0.5

    @property
    def index(self) -> int:
        """Compute unique state index from coordinates."""
        # Total states below this n
        c_below = total_capacity(self.n - 1) if self.n > 1 else 0
        # States within this shell before (l, m, s)
        states_before_l = sum(2 * (2 * ell + 1) for ell in range(self.l))
        states_at_l_before_m = 2 * (self.m + self.l)  # Both chiralities
        chirality_offset = 0 if self.s > 0 else 1
        return c_below + states_before_l + states_at_l_before_m + chirality_offset

    def to_tuple(self) -> Tuple[int, int, int, float]:
        return (self.n, self.l, self.m, self.s)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if not isinstance(other, PartitionState):
            return False
        return self.to_tuple() == other.to_tuple()


def capacity(n: int) -> int:
    """
    Capacity formula: C(n) = 2n²

    Number of accessible categorical states at partition depth n.
    """
    return 2 * n * n


def total_capacity(n_max: int) -> int:
    """
    Total capacity up to partition depth n_max.

    C_tot(N) = N(N+1)(2N+1)/3
    """
    if n_max <= 0:
        return 0
    return n_max * (n_max + 1) * (2 * n_max + 1) // 3


def mz_to_partition_depth(mz: float, m_ref: float = M_REF) -> int:
    """
    Map m/z to partition depth.

    n = floor(sqrt(m/z / m_ref)) + 1
    """
    return int(np.floor(np.sqrt(mz / m_ref))) + 1


def partition_depth_to_mz(n: int, delta_m: float = 0.0, m_ref: float = M_REF) -> float:
    """
    Map partition depth to m/z.

    m/z = m_ref * (n-1)² + Δm(l, m, s)
    """
    return m_ref * (n - 1) ** 2 + delta_m


def index_to_partition_state(i: int) -> PartitionState:
    """
    Convert state index to partition coordinates (n, l, m, s).

    Inverse of PartitionState.index
    """
    # Find n such that C_tot(n-1) < i <= C_tot(n)
    n = 1
    while total_capacity(n) < i:
        n += 1

    # Index within shell n
    idx_in_shell = i - total_capacity(n - 1) if n > 1 else i

    # Find l
    l = 0
    states_so_far = 0
    while states_so_far + 2 * (2 * l + 1) < idx_in_shell:
        states_so_far += 2 * (2 * l + 1)
        l += 1

    # Find m and s within this l
    idx_in_l = idx_in_shell - states_so_far - 1
    m = idx_in_l // 2 - l
    s = 0.5 if idx_in_l % 2 == 0 else -0.5

    return PartitionState(n=n, l=l, m=m, s=s)


# ============================================================================
# AMINO ACID STATE MAPPING
# ============================================================================

# Standard amino acid masses (monoisotopic)
AMINO_ACID_MASSES = {
    'A': 71.03711,   'R': 156.10111,  'N': 114.04293,  'D': 115.02694,
    'C': 103.00919,  'E': 129.04259,  'Q': 128.05858,  'G': 57.02146,
    'H': 137.05891,  'I': 113.08406,  'L': 113.08406,  'K': 128.09496,
    'M': 131.04049,  'F': 147.06841,  'P': 97.05276,   'S': 87.03203,
    'T': 101.04768,  'W': 186.07931,  'Y': 163.06333,  'V': 99.06841,
}

# Amino acid chirality (L vs D form)
AMINO_ACID_CHIRALITY = {aa: 0.5 for aa in AMINO_ACID_MASSES}  # All L-amino acids

# Amino acid angular complexity (based on side chain complexity)
AMINO_ACID_ANGULAR = {
    'G': 0, 'A': 0,  # Simple
    'V': 1, 'L': 1, 'I': 1, 'P': 1,  # Branched
    'S': 1, 'T': 1, 'C': 1, 'M': 1,  # With functional groups
    'N': 2, 'D': 2, 'Q': 2, 'E': 2,  # Polar with multiple groups
    'K': 2, 'R': 2, 'H': 2,  # Charged/complex
    'F': 3, 'Y': 3, 'W': 3,  # Aromatic
}


@dataclass
class AminoAcidState:
    """
    State counting representation of an amino acid.
    """
    amino_acid: str
    mass: float
    partition_state: PartitionState
    state_index: int

    @classmethod
    def from_amino_acid(cls, aa: str) -> 'AminoAcidState':
        """Create state from amino acid code."""
        mass = AMINO_ACID_MASSES.get(aa, 110.0)
        n = mz_to_partition_depth(mass)
        l = min(AMINO_ACID_ANGULAR.get(aa, 1), n - 1)
        m = 0  # Neutral orientation for amino acids
        s = AMINO_ACID_CHIRALITY.get(aa, 0.5)

        state = PartitionState(n=n, l=l, m=m, s=s)
        return cls(
            amino_acid=aa,
            mass=mass,
            partition_state=state,
            state_index=state.index
        )


# Pre-compute amino acid states
AMINO_ACID_STATES = {aa: AminoAcidState.from_amino_acid(aa) for aa in AMINO_ACID_MASSES}


# ============================================================================
# STATE COUNTING TRAJECTORY
# ============================================================================

@dataclass
class StateTrajectory:
    """
    A trajectory through partition state space.

    Represents a sequence of partition state transitions,
    with associated entropy production.
    """
    states: List[PartitionState] = field(default_factory=list)
    transition_times: List[float] = field(default_factory=list)
    entropy_increments: List[float] = field(default_factory=list)

    @property
    def total_entropy(self) -> float:
        """Total entropy produced by trajectory."""
        return sum(self.entropy_increments)

    @property
    def state_count(self) -> int:
        """Number of states in trajectory."""
        return len(self.states)

    @property
    def transition_count(self) -> int:
        """Number of transitions (one less than states)."""
        return max(0, len(self.states) - 1)

    def add_state(self, state: PartitionState, time: float = 0.0, phase_deviation: float = 0.0):
        """Add a state to the trajectory."""
        self.states.append(state)
        self.transition_times.append(time)

        # Entropy production per transition: ΔS = k_B ln(2 + |δφ|/100)
        k_B = 1.380649e-23  # Boltzmann constant
        delta_s = k_B * np.log(2 + abs(phase_deviation) / 100)
        self.entropy_increments.append(delta_s)


def compute_transition_entropy(phase_deviation: float = 0.0) -> float:
    """
    Compute entropy produced by a partition transition.

    ΔS = k_B ln(2 + |δφ|/100) > 0
    """
    k_B = 1.380649e-23
    return k_B * np.log(2 + abs(phase_deviation) / 100)


# ============================================================================
# FRAGMENT-PARENT HIERARCHICAL VALIDATION
# ============================================================================

@dataclass
class DropletParameters:
    """Parameters for thermodynamic droplet representation."""
    center: Tuple[float, float]  # (x, y) position
    radius: float
    wavelength: float  # λ_w
    energy: float
    phase: float

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2


@dataclass
class FragmentParentValidation:
    """
    Validation result for fragment-parent hierarchical relationship.

    Constraints:
    - Spatial containment: Overlap(F_i, P) > 0.7
    - Wavelength hierarchy: λ_w^{F_i} < λ_w^P (ratio ∈ [0.3, 0.9])
    - Energy conservation: Σ E_{F_i} ≤ E_P (ratio ∈ [0.6, 1.0])
    - Phase coherence: C_φ > 0.7
    """
    overlap_score: float
    wavelength_ratio: float
    energy_ratio: float
    phase_coherence: float

    @property
    def is_valid(self) -> bool:
        """Check if all hierarchical constraints are satisfied."""
        return (
            self.overlap_score > 0.7 and
            0.3 <= self.wavelength_ratio <= 0.9 and
            0.6 <= self.energy_ratio <= 1.0 and
            self.phase_coherence > 0.7
        )

    @property
    def overall_score(self) -> float:
        """Combined validation score."""
        scores = []

        # Overlap score (should be > 0.7)
        if self.overlap_score > 0.7:
            scores.append(1.0)
        else:
            scores.append(self.overlap_score / 0.7)

        # Wavelength ratio (should be in [0.3, 0.9])
        if 0.3 <= self.wavelength_ratio <= 0.9:
            scores.append(1.0)
        else:
            if self.wavelength_ratio < 0.3:
                scores.append(self.wavelength_ratio / 0.3)
            else:
                scores.append(max(0, 1.0 - (self.wavelength_ratio - 0.9) / 0.1))

        # Energy ratio (should be in [0.6, 1.0])
        if 0.6 <= self.energy_ratio <= 1.0:
            scores.append(1.0)
        else:
            if self.energy_ratio < 0.6:
                scores.append(self.energy_ratio / 0.6)
            else:
                scores.append(0.5)  # Over-energetic

        # Phase coherence (should be > 0.7)
        if self.phase_coherence > 0.7:
            scores.append(1.0)
        else:
            scores.append(self.phase_coherence / 0.7)

        return np.mean(scores)


def compute_circle_overlap(d1: DropletParameters, d2: DropletParameters) -> float:
    """
    Compute overlap ratio between two circular droplets.
    Returns the fraction of d1 that overlaps with d2.
    """
    dx = d1.center[0] - d2.center[0]
    dy = d1.center[1] - d2.center[1]
    distance = np.sqrt(dx * dx + dy * dy)

    r1, r2 = d1.radius, d2.radius

    # No overlap
    if distance >= r1 + r2:
        return 0.0

    # Complete containment
    if distance + r1 <= r2:
        return 1.0
    if distance + r2 <= r1:
        return d2.area / d1.area if d1.area > 0 else 0.0

    # Partial overlap - lens area formula
    part1 = r1 * r1 * np.arccos((distance * distance + r1 * r1 - r2 * r2) / (2 * distance * r1))
    part2 = r2 * r2 * np.arccos((distance * distance + r2 * r2 - r1 * r1) / (2 * distance * r2))
    part3 = 0.5 * np.sqrt((-distance + r1 + r2) * (distance + r1 - r2) * (distance - r1 + r2) * (distance + r1 + r2))

    overlap_area = part1 + part2 - part3

    return overlap_area / d1.area if d1.area > 0 else 0.0


def validate_fragment_parent_hierarchy(
    fragment_droplets: List[DropletParameters],
    parent_droplet: DropletParameters
) -> FragmentParentValidation:
    """
    Validate that fragments are hierarchically contained within parent.

    Implements the constraints from the fragment-parent hierarchical relationship.
    """
    if not fragment_droplets:
        return FragmentParentValidation(
            overlap_score=1.0,
            wavelength_ratio=0.5,
            energy_ratio=1.0,
            phase_coherence=1.0
        )

    # Spatial containment: average overlap of fragments with parent
    overlaps = [compute_circle_overlap(f, parent_droplet) for f in fragment_droplets]
    overlap_score = np.mean(overlaps)

    # Wavelength hierarchy: λ_w^{F_i} < λ_w^P
    wavelength_ratios = [f.wavelength / parent_droplet.wavelength for f in fragment_droplets]
    wavelength_ratio = np.mean(wavelength_ratios)

    # Energy conservation: Σ E_{F_i} ≤ E_P
    total_fragment_energy = sum(f.energy for f in fragment_droplets)
    energy_ratio = total_fragment_energy / parent_droplet.energy if parent_droplet.energy > 0 else 1.0

    # Phase coherence: C_φ = |<e^{iφ}>|
    if len(fragment_droplets) > 1:
        phases = np.array([f.phase for f in fragment_droplets])
        phase_coherence = abs(np.mean(np.exp(1j * phases)))
    else:
        phase_coherence = 1.0

    return FragmentParentValidation(
        overlap_score=overlap_score,
        wavelength_ratio=wavelength_ratio,
        energy_ratio=energy_ratio,
        phase_coherence=phase_coherence
    )


# ============================================================================
# STATE COUNTING SEQUENCE RECONSTRUCTION
# ============================================================================

class StateCountingReconstructor:
    """
    Sequence reconstruction using state counting framework.

    Uses partition coordinates and the capacity formula to map
    fragment masses to states, then reconstructs sequences via
    trajectory completion.
    """

    def __init__(
        self,
        mass_tolerance: float = 0.5,
        min_intensity_ratio: float = 0.01,
        epsilon_boundary: float = 0.1
    ):
        """
        Initialize reconstructor.

        Args:
            mass_tolerance: Mass tolerance for matching (Da)
            min_intensity_ratio: Minimum intensity ratio for fragments
            epsilon_boundary: ε-boundary tolerance for trajectory completion
        """
        self.mass_tolerance = mass_tolerance
        self.min_intensity_ratio = min_intensity_ratio
        self.epsilon_boundary = epsilon_boundary

        # Build mass-to-state lookup
        self._build_state_lookup()

    def _build_state_lookup(self):
        """Build lookup tables for fast state matching."""
        self.mass_to_states = defaultdict(list)
        for aa, state in AMINO_ACID_STATES.items():
            mass_key = round(state.mass, 0)
            self.mass_to_states[mass_key].append((aa, state))

    def _match_mass_to_amino_acid(self, mass_diff: float) -> List[Tuple[str, float]]:
        """
        Match a mass difference to possible amino acids with confidence scores.

        Returns list of (amino_acid, confidence) tuples.
        """
        candidates = []

        for aa, expected_mass in AMINO_ACID_MASSES.items():
            error = abs(mass_diff - expected_mass)
            if error < self.mass_tolerance:
                # Confidence based on mass accuracy
                confidence = 1.0 - (error / self.mass_tolerance)
                candidates.append((aa, confidence))

        # Sort by confidence
        candidates.sort(key=lambda x: -x[1])
        return candidates

    def _compute_state_distance(self, state1: PartitionState, state2: PartitionState) -> float:
        """
        Compute categorical distance between two partition states.
        """
        dn = abs(state1.n - state2.n)
        dl = abs(state1.l - state2.l)
        dm = abs(state1.m - state2.m)
        ds = 0 if state1.s == state2.s else 1

        # Weighted distance
        return dn * 4.0 + dl * 2.0 + dm * 1.0 + ds * 0.5

    def _generate_trajectories(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mass: float,
        precursor_charge: int
    ) -> List[StateTrajectory]:
        """
        Generate candidate trajectories through state space.
        """
        # Sort by m/z
        sorted_idx = np.argsort(mz_array)
        mz_sorted = mz_array[sorted_idx]
        intensity_sorted = intensity_array[sorted_idx]

        # Filter by intensity
        max_intensity = np.max(intensity_sorted)
        mask = intensity_sorted >= self.min_intensity_ratio * max_intensity
        mz_filtered = mz_sorted[mask]
        intensity_filtered = intensity_sorted[mask]

        if len(mz_filtered) < 2:
            return []

        # Map each fragment to a partition state
        fragment_states = []
        for mz, intensity in zip(mz_filtered, intensity_filtered):
            n = mz_to_partition_depth(mz)
            # Estimate angular complexity from intensity pattern
            l = min(int(np.log1p(intensity) / 2), n - 1)
            m = 0
            s = 0.5

            state = PartitionState(n=n, l=l, m=m, s=s)
            fragment_states.append((mz, intensity, state))

        # Generate trajectories by finding paths through state space
        trajectories = []

        # Forward pass: Start from smallest m/z, build trajectory
        trajectory = StateTrajectory()
        for i, (mz, intensity, state) in enumerate(fragment_states):
            phase_dev = np.random.uniform(-10, 10) if i > 0 else 0
            trajectory.add_state(state, time=float(i), phase_deviation=phase_dev)
        trajectories.append(trajectory)

        # Backward pass: Start from largest m/z
        trajectory_back = StateTrajectory()
        for i, (mz, intensity, state) in enumerate(reversed(fragment_states)):
            phase_dev = np.random.uniform(-10, 10) if i > 0 else 0
            trajectory_back.add_state(state, time=float(i), phase_deviation=phase_dev)
        trajectories.append(trajectory_back)

        return trajectories

    def _trajectory_to_sequence(
        self,
        trajectory: StateTrajectory,
        mz_array: np.ndarray,
        precursor_mass: float
    ) -> Tuple[str, float]:
        """
        Convert a trajectory through state space to an amino acid sequence.

        Uses b/y ion series logic for improved reconstruction.

        Returns (sequence, confidence_score).
        """
        if len(trajectory.states) < 2:
            return "", 0.0

        # Try both b-ion and y-ion interpretations
        b_sequence, b_confidence = self._extract_b_ion_sequence(mz_array, precursor_mass)
        y_sequence, y_confidence = self._extract_y_ion_sequence(mz_array, precursor_mass)

        # Use whichever has higher confidence
        if b_confidence >= y_confidence:
            return b_sequence, b_confidence
        else:
            return y_sequence, y_confidence

    def _extract_b_ion_sequence(
        self,
        mz_array: np.ndarray,
        precursor_mass: float
    ) -> Tuple[str, float]:
        """
        Extract sequence from b-ion series.

        b1 = mass(AA1) + H+ (proton mass = 1.007276)
        b2 = mass(AA1) + mass(AA2) + H+
        etc.

        The mass difference between consecutive b-ions equals amino acid mass.
        """
        # Sort by m/z
        mz_sorted = np.sort(mz_array)

        # Filter to likely b-ions (< precursor mass)
        b_candidates = mz_sorted[mz_sorted < precursor_mass * 0.95]

        if len(b_candidates) < 2:
            return "", 0.0

        sequence_parts = []
        total_confidence = 0.0
        n_matches = 0

        # First ion: b1 = AA1 + H+ (1.007276)
        # So AA1 mass = b1 - 1.007276
        b1_minus_proton = b_candidates[0] - 1.007276
        first_candidates = self._match_mass_to_amino_acid(b1_minus_proton)
        if first_candidates:
            sequence_parts.append(first_candidates[0][0])
            total_confidence += first_candidates[0][1]
            n_matches += 1

        # Subsequent ions: mass difference = amino acid mass
        for i in range(1, len(b_candidates)):
            mass_diff = b_candidates[i] - b_candidates[i-1]

            candidates = self._match_mass_to_amino_acid(mass_diff)
            if candidates:
                sequence_parts.append(candidates[0][0])
                total_confidence += candidates[0][1]
                n_matches += 1

        sequence = ''.join(sequence_parts)
        avg_confidence = total_confidence / max(1, n_matches)

        # Validate against precursor mass
        if sequence:
            theoretical_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in sequence) + 18.015
            mass_error = abs(theoretical_mass - precursor_mass) / max(precursor_mass, 1)

            if mass_error > 0.2:
                avg_confidence *= 0.3
            elif mass_error > 0.1:
                avg_confidence *= 0.6

        return sequence, avg_confidence

    def _extract_y_ion_sequence(
        self,
        mz_array: np.ndarray,
        precursor_mass: float
    ) -> Tuple[str, float]:
        """
        Extract sequence from y-ion series.

        y1 = mass(AAn) + H2O + H+ = mass(AAn) + 19.018
        y2 = mass(AAn) + mass(AAn-1) + 19.018
        etc.

        Note: y-ions give reverse sequence (C-terminal to N-terminal)
        """
        # Sort by m/z ascending for y-ions
        mz_sorted = np.sort(mz_array)

        # Filter to likely y-ions (< precursor mass)
        y_candidates = mz_sorted[mz_sorted < precursor_mass * 0.95]

        if len(y_candidates) < 2:
            return "", 0.0

        sequence_parts = []
        total_confidence = 0.0
        n_matches = 0

        # First y-ion (y1) gives C-terminal amino acid
        # y1 = AA_n + 19.018 (H2O + H+)
        y1_minus_water = y_candidates[0] - 19.018
        first_candidates = self._match_mass_to_amino_acid(y1_minus_water)
        if first_candidates:
            # This is the C-terminal AA, add at end later
            c_term_aa = first_candidates[0][0]
            c_term_conf = first_candidates[0][1]
        else:
            c_term_aa = None
            c_term_conf = 0.0

        # y-ions: differences give amino acids from C-terminus going backward
        for i in range(1, len(y_candidates)):
            mass_diff = y_candidates[i] - y_candidates[i-1]

            candidates = self._match_mass_to_amino_acid(mass_diff)
            if candidates:
                sequence_parts.append(candidates[0][0])
                total_confidence += candidates[0][1]
                n_matches += 1

        # Add C-terminal amino acid at end
        if c_term_aa:
            sequence_parts.append(c_term_aa)
            total_confidence += c_term_conf
            n_matches += 1

        sequence = ''.join(sequence_parts)
        avg_confidence = total_confidence / max(1, n_matches)

        # Validate against precursor mass
        if sequence:
            theoretical_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in sequence) + 18.015
            mass_error = abs(theoretical_mass - precursor_mass) / max(precursor_mass, 1)

            if mass_error > 0.2:
                avg_confidence *= 0.3
            elif mass_error > 0.1:
                avg_confidence *= 0.6

        return sequence, avg_confidence

    def _complete_trajectory(
        self,
        partial_sequence: str,
        precursor_mass: float,
        target_length: int
    ) -> List[Tuple[str, float]]:
        """
        Complete a partial sequence using trajectory completion.

        Fills gaps to match precursor mass within ε-boundary.
        """
        if not partial_sequence:
            # If no partial sequence, generate de novo based on target length
            return self._generate_de_novo_sequences(precursor_mass, target_length)

        # Compute current mass
        current_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in partial_sequence) + 18.015
        mass_deficit = precursor_mass - current_mass

        if abs(mass_deficit) < self.epsilon_boundary:
            # Already at ε-boundary
            return [(partial_sequence, 1.0)]

        completions = []

        # Try to fill mass deficit with amino acids at both ends
        if mass_deficit > 0 and mass_deficit < 700:  # Max ~5 amino acids
            # Try adding at C-terminus
            c_candidates = self._enumerate_mass_completions(mass_deficit, max_length=3)
            for completion, error in c_candidates[:3]:
                full_sequence = partial_sequence + completion
                confidence = 1.0 - (error / self.mass_tolerance)
                completions.append((full_sequence, max(0, confidence)))

            # Try adding at N-terminus
            n_candidates = self._enumerate_mass_completions(mass_deficit, max_length=3)
            for completion, error in n_candidates[:3]:
                full_sequence = completion + partial_sequence
                confidence = 1.0 - (error / self.mass_tolerance)
                completions.append((full_sequence, max(0, confidence * 0.9)))  # Slight penalty

        # Also try inserting in middle for very short sequences
        if len(partial_sequence) <= 2 and mass_deficit > 100:
            middle_candidates = self._enumerate_mass_completions(mass_deficit, max_length=4)
            for completion, error in middle_candidates[:3]:
                # Insert after first AA
                if len(partial_sequence) >= 1:
                    full_sequence = partial_sequence[0] + completion + partial_sequence[1:]
                    confidence = 0.7 * (1.0 - (error / self.mass_tolerance))
                    completions.append((full_sequence, max(0, confidence)))

        return completions if completions else [(partial_sequence, 0.5)]

    def _generate_de_novo_sequences(
        self,
        precursor_mass: float,
        target_length: int
    ) -> List[Tuple[str, float]]:
        """
        Generate de novo sequences when no partial sequence available.
        """
        # Target mass per residue
        avg_mass = np.mean(list(AMINO_ACID_MASSES.values()))
        estimated_length = max(3, int(round((precursor_mass - 18.015) / avg_mass)))

        candidates = []

        # Try common peptide patterns
        common_aa = ['A', 'L', 'V', 'G', 'E', 'D', 'S', 'P', 'K', 'R']

        # Generate combinations that match mass
        target_residue_mass = (precursor_mass - 18.015) / estimated_length

        # Find AAs closest to average mass
        aa_by_mass = sorted(AMINO_ACID_MASSES.items(), key=lambda x: abs(x[1] - target_residue_mass))

        # Build sequence from most likely AAs
        seq = ''
        remaining_mass = precursor_mass - 18.015

        for _ in range(estimated_length):
            if remaining_mass < 50:
                break
            for aa, mass in aa_by_mass:
                if mass < remaining_mass + self.mass_tolerance:
                    seq += aa
                    remaining_mass -= mass
                    break

        if seq:
            theoretical_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in seq) + 18.015
            error = abs(theoretical_mass - precursor_mass)
            confidence = max(0, 1.0 - (error / precursor_mass))
            candidates.append((seq, confidence * 0.5))

        return candidates if candidates else [('X' * min(estimated_length, 10), 0.1)]

    def _enumerate_mass_completions(
        self,
        target_mass: float,
        max_length: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Enumerate amino acid combinations matching target mass.

        Returns list of (sequence, mass_error) sorted by error.
        """
        candidates = []

        # Single amino acid
        for aa, mass in AMINO_ACID_MASSES.items():
            error = abs(mass - target_mass)
            if error < self.mass_tolerance:
                candidates.append((aa, error))

        # Two amino acids
        if max_length >= 2:
            for aa1, mass1 in AMINO_ACID_MASSES.items():
                for aa2, mass2 in AMINO_ACID_MASSES.items():
                    error = abs(mass1 + mass2 - target_mass)
                    if error < self.mass_tolerance:
                        candidates.append((aa1 + aa2, error))

        # Three amino acids
        if max_length >= 3:
            for aa1, mass1 in AMINO_ACID_MASSES.items():
                for aa2, mass2 in AMINO_ACID_MASSES.items():
                    if mass1 + mass2 > target_mass + self.mass_tolerance:
                        continue
                    for aa3, mass3 in AMINO_ACID_MASSES.items():
                        error = abs(mass1 + mass2 + mass3 - target_mass)
                        if error < self.mass_tolerance:
                            candidates.append((aa1 + aa2 + aa3, error))

        # Sort by error
        candidates.sort(key=lambda x: x[1])
        return candidates

    def reconstruct(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2,
        known_sequence: Optional[str] = None
    ) -> Dict:
        """
        Full sequence reconstruction using state counting.

        Args:
            mz_array: Fragment m/z values
            intensity_array: Fragment intensities
            precursor_mz: Precursor m/z
            precursor_charge: Precursor charge state
            known_sequence: Known sequence for validation (optional)

        Returns:
            Dictionary with reconstruction results
        """
        # Compute precursor mass
        precursor_mass = (precursor_mz * precursor_charge) - (precursor_charge * 1.007276)

        # Estimate target length
        avg_aa_mass = np.mean(list(AMINO_ACID_MASSES.values()))
        target_length = max(3, int(round(precursor_mass / avg_aa_mass)))

        # Generate candidate trajectories
        trajectories = self._generate_trajectories(
            mz_array, intensity_array, precursor_mass, precursor_charge
        )

        # Convert trajectories to sequences
        candidate_sequences = []
        for trajectory in trajectories:
            sequence, confidence = self._trajectory_to_sequence(
                trajectory, mz_array, precursor_mass
            )

            if sequence:
                # Try to complete trajectory
                completions = self._complete_trajectory(
                    sequence, precursor_mass, target_length
                )

                for completed_seq, completion_conf in completions:
                    final_conf = confidence * completion_conf
                    candidate_sequences.append((completed_seq, final_conf, trajectory))

        # Score and rank candidates
        scored_candidates = []
        for sequence, confidence, trajectory in candidate_sequences:
            if not sequence:
                continue

            # Compute theoretical mass
            theoretical_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in sequence) + 18.015
            mass_error = abs(theoretical_mass - precursor_mass)
            mass_error_ppm = (mass_error / precursor_mass) * 1e6

            # State count score
            n_target = mz_to_partition_depth(precursor_mass)
            state_score = 1.0 - abs(trajectory.state_count - capacity(n_target)) / max(1, capacity(n_target))

            # Entropy score
            expected_entropy = trajectory.transition_count * compute_transition_entropy(0)
            entropy_ratio = trajectory.total_entropy / max(1e-30, expected_entropy)

            # Combined score
            score = 0.4 * confidence + 0.3 * max(0, state_score) + 0.3 * min(1, entropy_ratio)

            scored_candidates.append({
                'sequence': sequence,
                'length': len(sequence),
                'theoretical_mass': theoretical_mass,
                'mass_error': mass_error,
                'mass_error_ppm': mass_error_ppm,
                'confidence': confidence,
                'state_count': trajectory.state_count,
                'entropy': trajectory.total_entropy,
                'score': score
            })

        # Sort by score
        scored_candidates.sort(key=lambda x: -x['score'])

        # Compute validation metrics if known sequence provided
        validation = {}
        if known_sequence and scored_candidates:
            best_sequence = scored_candidates[0]['sequence']
            validation = self._validate_reconstruction(best_sequence, known_sequence)

        return {
            'precursor_mass': precursor_mass,
            'target_length': target_length,
            'n_fragments': len(mz_array),
            'n_trajectories': len(trajectories),
            'candidates': scored_candidates[:10],
            'best_sequence': scored_candidates[0]['sequence'] if scored_candidates else '',
            'best_score': scored_candidates[0]['score'] if scored_candidates else 0.0,
            'validation': validation
        }

    def _validate_reconstruction(self, predicted: str, known: str) -> Dict:
        """
        Validate reconstructed sequence against known sequence.
        """
        # Exact match
        exact_match = predicted.upper() == known.upper()

        # Partial match (longest common subsequence)
        lcs_length = self._lcs_length(predicted.upper(), known.upper())
        partial_score = lcs_length / max(len(known), 1)

        # Edit distance
        edit_dist = self._edit_distance(predicted.upper(), known.upper())
        edit_similarity = 1.0 - (edit_dist / max(len(known), len(predicted), 1))

        # Amino acid overlap
        pred_set = set(predicted.upper())
        known_set = set(known.upper())
        aa_overlap = len(pred_set & known_set) / max(len(known_set), 1)

        return {
            'exact_match': exact_match,
            'partial_score': partial_score,
            'edit_similarity': edit_similarity,
            'aa_overlap': aa_overlap,
            'lcs_length': lcs_length,
            'edit_distance': edit_dist
        }

    def _lcs_length(self, s1: str, s2: str) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]


# ============================================================================
# DROPLET GENERATION FOR FRAGMENT-PARENT VALIDATION
# ============================================================================

class StateCountingDropletMapper:
    """
    Maps partition states to thermodynamic droplet parameters
    for fragment-parent hierarchical validation.
    """

    def __init__(
        self,
        canvas_size: Tuple[int, int] = (256, 256),
        max_wavelength: float = 50.0,
        min_wavelength: float = 5.0
    ):
        self.canvas_size = canvas_size
        self.max_wavelength = max_wavelength
        self.min_wavelength = min_wavelength

    def state_to_droplet(
        self,
        state: PartitionState,
        intensity: float,
        mz: float,
        precursor_mz: float
    ) -> DropletParameters:
        """
        Convert partition state to droplet parameters.
        """
        # Position from partition coordinates
        n_max = mz_to_partition_depth(precursor_mz)

        # Radial position from n
        r_norm = state.n / max(1, n_max)
        max_r = min(self.canvas_size) * 0.4
        r = r_norm * max_r

        # Angular position from l and m
        theta = 2 * np.pi * (state.l + 0.5) / max(1, state.n)
        theta += np.pi * state.m / max(1, 2 * state.l + 1)

        # Center position
        cx = self.canvas_size[0] / 2 + r * np.cos(theta)
        cy = self.canvas_size[1] / 2 + r * np.sin(theta)

        # Radius from intensity
        max_radius = min(self.canvas_size) * 0.15
        radius = max_radius * np.sqrt(np.log1p(intensity) / 10)

        # Wavelength from partition depth (higher n = shorter wavelength)
        wavelength = self.max_wavelength - (self.max_wavelength - self.min_wavelength) * r_norm

        # Energy proportional to m/z and intensity
        energy = mz * np.log1p(intensity)

        # Phase from chirality and m
        phase = np.pi * (state.s + 0.5) + state.m * np.pi / max(1, state.l + 1)

        return DropletParameters(
            center=(cx, cy),
            radius=max(1.0, radius),
            wavelength=wavelength,
            energy=energy,
            phase=phase
        )

    def spectrum_to_droplets(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float
    ) -> Tuple[List[DropletParameters], DropletParameters]:
        """
        Convert full spectrum to fragment droplets and parent droplet.

        Returns:
            (fragment_droplets, parent_droplet)
        """
        fragment_droplets = []

        for mz, intensity in zip(mz_array, intensity_array):
            n = mz_to_partition_depth(mz)
            l = min(int(np.log1p(intensity) / 3), n - 1)
            m = 0
            s = 0.5

            state = PartitionState(n=n, l=l, m=m, s=s)
            droplet = self.state_to_droplet(state, intensity, mz, precursor_mz)
            fragment_droplets.append(droplet)

        # Parent droplet: centered, largest, longest wavelength
        parent_n = mz_to_partition_depth(precursor_mz)
        parent_state = PartitionState(n=parent_n, l=0, m=0, s=0.5)

        parent_droplet = DropletParameters(
            center=(self.canvas_size[0] / 2, self.canvas_size[1] / 2),
            radius=min(self.canvas_size) * 0.4,
            wavelength=self.max_wavelength,
            energy=precursor_mz * np.sum(np.log1p(intensity_array)),
            phase=0.0
        )

        return fragment_droplets, parent_droplet


# ============================================================================
# CHARGE LOCALIZATION AND REDISTRIBUTION
# ============================================================================

# Basic residues that carry protons (charges)
BASIC_RESIDUES = {'K', 'R', 'H'}

# Proton affinity scale (relative, kJ/mol)
PROTON_AFFINITY = {
    'R': 1000,   # Arginine - highest
    'K': 900,    # Lysine
    'H': 800,    # Histidine
    'N-term': 750,  # N-terminus
    'Q': 700,    # Glutamine
    'N': 700,    # Asparagine
    'E': 650,    # Glutamate
    'D': 650,    # Aspartate
    'S': 600,    # Serine
    'T': 600,    # Threonine
    'other': 500  # Other residues
}


@dataclass
class ChargeLocalizationResult:
    """
    Result of charge localization analysis for a fragment.
    """
    fragment_id: str
    fragment_charge: int
    fragment_mass: float
    charge_density: float  # charge/mass
    charge_density_ratio: float  # fragment/parent
    redistribution_factor: float  # C_i = sqrt(rho_ratio)
    localization_index: float  # L_charge
    charge_bearing_residues: List[str]
    s_entropy_predicted: float
    s_entropy_measured: float
    s_entropy_error: float
    is_valid: bool


@dataclass
class ChargeRedistributionValidation:
    """
    Complete charge redistribution validation for a spectrum.
    """
    parent_charge: int
    parent_mass: float
    parent_charge_density: float
    fragments: List[ChargeLocalizationResult]
    total_fragment_charge: int
    charge_conserved: bool
    charge_balance: float  # total_fragment/parent
    overall_valid: bool


def compute_charge_localization_index(sequence: str, charge_positions: List[int]) -> float:
    """
    Compute charge localization index.

    L_charge = max_i(Q_i) / sum_i(Q_i)

    Where Q_i is the charge on residue i.

    Args:
        sequence: Amino acid sequence
        charge_positions: Positions of charge-bearing residues (0-indexed)

    Returns:
        Localization index (1.0 = fully localized, 1/N = evenly distributed)
    """
    if not charge_positions:
        return 1.0  # No charges = localized by default

    n_charges = len(charge_positions)

    if n_charges == 1:
        return 1.0  # Single charge = fully localized

    # For multiple charges, compute distribution
    # Assume each charge is 1
    return 1.0 / n_charges


def identify_charge_bearing_residues(sequence: str, charge: int) -> List[Tuple[int, str]]:
    """
    Identify residues likely to bear charge based on proton affinity.

    For peptides, charges typically localize on:
    1. Basic residues (K, R, H) - highest priority
    2. N-terminus
    3. Backbone amides

    Args:
        sequence: Amino acid sequence
        charge: Total charge on the peptide

    Returns:
        List of (position, residue) tuples for charge-bearing sites
    """
    if not sequence or charge <= 0:
        return []

    # Build list of potential charge sites with affinities
    charge_sites = []

    # N-terminus always a potential site
    charge_sites.append((0, 'N-term', PROTON_AFFINITY['N-term']))

    # Scan sequence for basic residues
    for i, aa in enumerate(sequence):
        if aa in BASIC_RESIDUES:
            charge_sites.append((i, aa, PROTON_AFFINITY[aa]))
        elif aa in PROTON_AFFINITY:
            charge_sites.append((i, aa, PROTON_AFFINITY[aa]))
        else:
            charge_sites.append((i, aa, PROTON_AFFINITY['other']))

    # Sort by proton affinity (highest first)
    charge_sites.sort(key=lambda x: -x[2])

    # Select top 'charge' sites
    selected = charge_sites[:charge]

    return [(pos, res) for pos, res, _ in selected]


def infer_fragment_charge(mz: float, mass: float, parent_charge: int = 2) -> int:
    """
    Infer fragment charge from m/z and mass.

    For peptides, most fragments are +1.
    For larger fragments, may be +2 or +3.

    Args:
        mz: Fragment m/z
        mass: Fragment neutral mass
        parent_charge: Parent ion charge

    Returns:
        Inferred charge (1-3)
    """
    if mass <= 0 or mz <= 0:
        return 1

    # Z = mass / m/z
    z_estimated = mass / mz
    z = max(1, min(int(round(z_estimated)), parent_charge))

    # Most fragments are singly charged
    if mass < 800:
        z = 1
    elif mass < 1500:
        z = min(z, 2)

    return z


def validate_charge_redistribution(
    parent_spectrum: Dict,
    fragment_spectra: List[Dict],
    beta: float = 0.5
) -> ChargeRedistributionValidation:
    """
    Validate charge conservation and redistribution.

    Implements the mobile proton model for peptide fragmentation:
    - Charges localize on basic residues (K, R, H) and termini
    - Fragment charge density affects S-entropy
    - Total fragment charge ≤ parent charge (conservation)

    Args:
        parent_spectrum: Dict with 'charge', 'mass', 's_entropy', 'sequence'
        fragment_spectra: List of dicts with fragment properties
        beta: Coefficient for S-entropy adjustment (default 0.5)

    Returns:
        ChargeRedistributionValidation with complete analysis
    """
    # Extract parent properties
    Z_P = parent_spectrum.get('charge', 2)
    M_P = parent_spectrum.get('mass', 1000.0)
    S_e_P = parent_spectrum.get('s_entropy', 0.5)
    parent_seq = parent_spectrum.get('sequence', '')

    if M_P <= 0:
        M_P = 1000.0

    rho_P = Z_P / M_P

    # Validate each fragment
    fragments = []
    total_fragment_charge = 0

    for frag in fragment_spectra:
        frag_id = frag.get('id', 'unknown')
        M_F = frag.get('mass', 0.0)
        mz_F = frag.get('mz', 0.0)
        S_e_F = frag.get('s_entropy', 0.0)
        frag_seq = frag.get('sequence', '')

        if M_F <= 0:
            M_F = mz_F  # Assume singly charged if mass not provided

        # Infer fragment charge
        Z_F = infer_fragment_charge(mz_F, M_F, Z_P)
        rho_F = Z_F / max(M_F, 1.0)

        # Charge density ratio
        rho_ratio = rho_F / max(rho_P, 1e-10)

        # Charge redistribution factor: C_i = sqrt(rho_ratio)
        C_i = np.sqrt(rho_ratio)

        # Identify charge-bearing residues in fragment
        if frag_seq:
            charge_sites = identify_charge_bearing_residues(frag_seq, Z_F)
            charge_bearing = [res for _, res in charge_sites]
            L_charge = compute_charge_localization_index(frag_seq, [pos for pos, _ in charge_sites])
        else:
            charge_bearing = []
            L_charge = 1.0

        # S-entropy prediction based on charge redistribution
        # S_e^F = S_e^P * (1 + β * (ρ_ratio - 1))
        S_e_predicted = S_e_P * (1 + beta * (rho_ratio - 1))
        S_e_error = abs(S_e_F - S_e_predicted)

        # Fragment is valid if S-entropy error < 20%
        is_valid = S_e_error < 0.2 * max(S_e_P, 0.1)

        result = ChargeLocalizationResult(
            fragment_id=frag_id,
            fragment_charge=Z_F,
            fragment_mass=M_F,
            charge_density=rho_F,
            charge_density_ratio=rho_ratio,
            redistribution_factor=C_i,
            localization_index=L_charge,
            charge_bearing_residues=charge_bearing,
            s_entropy_predicted=S_e_predicted,
            s_entropy_measured=S_e_F,
            s_entropy_error=S_e_error,
            is_valid=is_valid
        )

        fragments.append(result)
        total_fragment_charge += Z_F

    # Charge conservation check
    charge_conserved = (total_fragment_charge <= Z_P)
    charge_balance = total_fragment_charge / max(Z_P, 1)

    # Overall validation
    all_valid = all(f.is_valid for f in fragments) if fragments else True
    overall_valid = all_valid and charge_conserved

    return ChargeRedistributionValidation(
        parent_charge=Z_P,
        parent_mass=M_P,
        parent_charge_density=rho_P,
        fragments=fragments,
        total_fragment_charge=total_fragment_charge,
        charge_conserved=charge_conserved,
        charge_balance=charge_balance,
        overall_valid=overall_valid
    )


def validate_spatial_containment_with_charge(
    parent_droplet: DropletParameters,
    fragment_droplets: List[DropletParameters],
    fragment_charges: List[int],
    parent_charge: int
) -> Dict:
    """
    Validate spatial containment with charge redistribution modulation.

    Fragment wave amplitude is modulated by charge density ratio:
    I_F(x,y) = I_P(x,y) * W_i(x,y) * C_i

    Args:
        parent_droplet: Parent droplet parameters
        fragment_droplets: List of fragment droplet parameters
        fragment_charges: List of fragment charges
        parent_charge: Parent charge

    Returns:
        Dict with validation results
    """
    if not fragment_droplets:
        return {
            'valid': True,
            'mean_overlap': 1.0,
            'mean_charge_modulation': 1.0,
            'fragments': []
        }

    parent_mass = parent_droplet.energy  # Using energy as proxy for mass
    parent_rho = parent_charge / max(parent_mass, 1.0)

    fragment_results = []

    for i, (f_droplet, f_charge) in enumerate(zip(fragment_droplets, fragment_charges)):
        # Spatial overlap
        overlap = compute_circle_overlap(f_droplet, parent_droplet)

        # Charge redistribution factor
        f_mass = f_droplet.energy
        f_rho = f_charge / max(f_mass, 1.0)
        C_i = np.sqrt(f_rho / max(parent_rho, 1e-10))

        # Modulated overlap
        modulated_overlap = overlap * C_i

        fragment_results.append({
            'fragment_id': i,
            'spatial_overlap': overlap,
            'charge_density_ratio': f_rho / max(parent_rho, 1e-10),
            'redistribution_factor': C_i,
            'modulated_overlap': modulated_overlap,
            'valid': overlap > 0.5 and 0.3 < C_i < 3.0
        })

    mean_overlap = np.mean([f['spatial_overlap'] for f in fragment_results])
    mean_modulation = np.mean([f['redistribution_factor'] for f in fragment_results])
    overall_valid = all(f['valid'] for f in fragment_results)

    return {
        'valid': overall_valid,
        'mean_overlap': mean_overlap,
        'mean_charge_modulation': mean_modulation,
        'fragments': fragment_results
    }


# ============================================================================
# CIRCULAR VALIDATION - THE BIJECTIVE COMPLETION APPROACH
# ============================================================================
#
# Core Insight (from user):
# "A→B→C→A is not fallacious... A→B is"
#
# One-way validation (A→B) is weak:
#   spectrum → reconstruction → sequence (no verification)
#
# Circular validation (A→B→C→A) is strong:
#   spectrum → candidate_sequence → predicted_spectrum → compare_to_original
#
# The candidate whose predicted spectrum best matches the original IS correct.
# This leverages the bijective property: if the transformation is truly bijective,
# the correct sequence will reconstruct to the same spectrum.
#
# ============================================================================


@dataclass
class TheoreticalFragment:
    """A theoretical fragment from peptide fragmentation."""
    ion_type: str  # 'b' or 'y'
    position: int  # Position in sequence
    sequence: str  # Fragment sequence
    mass: float  # Neutral mass
    mz: float  # m/z for given charge
    charge: int
    neutral_loss: Optional[str] = None


class TheoreticalSpectrumGenerator:
    """
    Generate theoretical MS/MS spectrum from peptide sequence.

    This is the FORWARD prediction: sequence → spectrum
    Required for circular validation: we predict what spectrum a
    candidate sequence SHOULD produce, then compare to the original.
    """

    # Ion type mass offsets
    B_ION_OFFSET = 1.007276  # +H
    Y_ION_OFFSET = 19.01839  # +H +OH (water)

    # Common neutral losses
    NEUTRAL_LOSSES = {
        'H2O': 18.01056,
        'NH3': 17.02655,
        'CO': 27.99491,
    }

    # Residues prone to neutral loss
    WATER_LOSS_RESIDUES = {'S', 'T', 'E', 'D'}
    AMMONIA_LOSS_RESIDUES = {'K', 'R', 'Q', 'N'}

    def __init__(
        self,
        include_b_ions: bool = True,
        include_y_ions: bool = True,
        include_neutral_losses: bool = True,
        max_charge: int = 2
    ):
        self.include_b_ions = include_b_ions
        self.include_y_ions = include_y_ions
        self.include_neutral_losses = include_neutral_losses
        self.max_charge = max_charge

    def generate_fragments(
        self,
        sequence: str,
        precursor_charge: int = 2
    ) -> List[TheoreticalFragment]:
        """
        Generate all theoretical fragments for a peptide sequence.

        Args:
            sequence: Peptide sequence (e.g., "PEPTIDE")
            precursor_charge: Precursor charge state

        Returns:
            List of TheoreticalFragment objects
        """
        if not sequence:
            return []

        fragments = []
        n = len(sequence)

        # Generate b-ions (N-terminal fragments)
        if self.include_b_ions:
            for i in range(1, n):
                frag_seq = sequence[:i]
                mass = self._calculate_fragment_mass(frag_seq, 'b')

                for z in range(1, min(self.max_charge, precursor_charge) + 1):
                    mz = (mass + z * 1.007276) / z
                    fragments.append(TheoreticalFragment(
                        ion_type='b',
                        position=i,
                        sequence=frag_seq,
                        mass=mass,
                        mz=mz,
                        charge=z
                    ))

                # Neutral losses
                if self.include_neutral_losses:
                    fragments.extend(self._add_neutral_losses(frag_seq, 'b', i, precursor_charge))

        # Generate y-ions (C-terminal fragments)
        if self.include_y_ions:
            for i in range(1, n):
                frag_seq = sequence[n-i:]
                mass = self._calculate_fragment_mass(frag_seq, 'y')

                for z in range(1, min(self.max_charge, precursor_charge) + 1):
                    mz = (mass + z * 1.007276) / z
                    fragments.append(TheoreticalFragment(
                        ion_type='y',
                        position=i,
                        sequence=frag_seq,
                        mass=mass,
                        mz=mz,
                        charge=z
                    ))

                # Neutral losses
                if self.include_neutral_losses:
                    fragments.extend(self._add_neutral_losses(frag_seq, 'y', i, precursor_charge))

        return fragments

    def _calculate_fragment_mass(self, sequence: str, ion_type: str) -> float:
        """Calculate neutral mass of a fragment."""
        mass = sum(AMINO_ACID_MASSES.get(aa, 110.0) for aa in sequence)

        if ion_type == 'b':
            mass += self.B_ION_OFFSET
        elif ion_type == 'y':
            mass += self.Y_ION_OFFSET

        return mass

    def _add_neutral_losses(
        self,
        sequence: str,
        ion_type: str,
        position: int,
        precursor_charge: int
    ) -> List[TheoreticalFragment]:
        """Add neutral loss fragments if residues support them."""
        fragments = []

        # Check for water loss
        if any(aa in self.WATER_LOSS_RESIDUES for aa in sequence):
            mass = self._calculate_fragment_mass(sequence, ion_type) - self.NEUTRAL_LOSSES['H2O']
            for z in range(1, min(self.max_charge, precursor_charge) + 1):
                mz = (mass + z * 1.007276) / z
                fragments.append(TheoreticalFragment(
                    ion_type=ion_type,
                    position=position,
                    sequence=sequence,
                    mass=mass,
                    mz=mz,
                    charge=z,
                    neutral_loss='H2O'
                ))

        # Check for ammonia loss
        if any(aa in self.AMMONIA_LOSS_RESIDUES for aa in sequence):
            mass = self._calculate_fragment_mass(sequence, ion_type) - self.NEUTRAL_LOSSES['NH3']
            for z in range(1, min(self.max_charge, precursor_charge) + 1):
                mz = (mass + z * 1.007276) / z
                fragments.append(TheoreticalFragment(
                    ion_type=ion_type,
                    position=position,
                    sequence=sequence,
                    mass=mass,
                    mz=mz,
                    charge=z,
                    neutral_loss='NH3'
                ))

        return fragments

    def generate_spectrum(
        self,
        sequence: str,
        precursor_charge: int = 2,
        intensity_model: str = 'exponential'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate theoretical spectrum (m/z and intensity arrays).

        Args:
            sequence: Peptide sequence
            precursor_charge: Precursor charge
            intensity_model: 'exponential', 'uniform', or 'position_weighted'

        Returns:
            (mz_array, intensity_array)
        """
        fragments = self.generate_fragments(sequence, precursor_charge)

        if not fragments:
            return np.array([]), np.array([])

        mz_list = []
        intensity_list = []

        for frag in fragments:
            mz_list.append(frag.mz)

            # Intensity model
            if intensity_model == 'exponential':
                # Exponential decay from center
                center_pos = len(sequence) / 2
                intensity = 1000.0 * np.exp(-0.1 * abs(frag.position - center_pos))
            elif intensity_model == 'position_weighted':
                # Higher intensity at ends (b1, y1 series)
                intensity = 1000.0 * (1.0 + 0.5 / max(frag.position, 1))
            else:
                intensity = 1000.0

            # Reduce intensity for neutral losses
            if frag.neutral_loss:
                intensity *= 0.3

            # Reduce intensity for higher charge states
            if frag.charge > 1:
                intensity *= 0.5 ** (frag.charge - 1)

            intensity_list.append(intensity)

        return np.array(mz_list), np.array(intensity_list)


@dataclass
class SpectralSimilarity:
    """Result of spectral similarity comparison."""
    cosine_similarity: float
    matched_peaks: int
    total_original_peaks: int
    total_predicted_peaks: int
    mass_accuracy: float  # Mean mass error of matched peaks
    intensity_correlation: float
    s_entropy_distance: float  # Distance in S-entropy space
    droplet_overlap: float  # CV droplet overlap score
    circular_validation_score: float  # Combined score for circular validation


class SpectralSimilarityComputer:
    """
    Compute similarity between two spectra for circular validation.

    Uses multiple metrics in the bijective framework:
    1. Peak matching (m/z tolerance)
    2. Intensity correlation
    3. S-entropy coordinate distance
    4. CV droplet overlap
    """

    def __init__(
        self,
        mz_tolerance: float = 0.5,
        intensity_threshold: float = 0.01
    ):
        self.mz_tolerance = mz_tolerance
        self.intensity_threshold = intensity_threshold

    def compute_similarity(
        self,
        original_mz: np.ndarray,
        original_intensity: np.ndarray,
        predicted_mz: np.ndarray,
        predicted_intensity: np.ndarray,
        precursor_mz: float
    ) -> SpectralSimilarity:
        """
        Compute comprehensive similarity between original and predicted spectra.

        This is the VALIDATION step in circular validation:
        - Original spectrum: what we observed
        - Predicted spectrum: what the candidate sequence would produce
        - If similar: candidate is likely correct
        """
        # Normalize intensities
        orig_max = np.max(original_intensity) if len(original_intensity) > 0 else 1.0
        pred_max = np.max(predicted_intensity) if len(predicted_intensity) > 0 else 1.0

        orig_norm = original_intensity / max(orig_max, 1e-10)
        pred_norm = predicted_intensity / max(pred_max, 1e-10)

        # 1. Peak matching
        matched_orig_idx, matched_pred_idx, mass_errors = self._match_peaks(
            original_mz, predicted_mz
        )

        n_matched = len(matched_orig_idx)
        mass_accuracy = np.mean(mass_errors) if mass_errors else 0.0

        # 2. Cosine similarity (on matched peaks)
        if n_matched > 0:
            orig_matched = orig_norm[matched_orig_idx]
            pred_matched = pred_norm[matched_pred_idx]

            dot_product = np.dot(orig_matched, pred_matched)
            norm_orig = np.linalg.norm(orig_matched)
            norm_pred = np.linalg.norm(pred_matched)

            cosine_sim = dot_product / max(norm_orig * norm_pred, 1e-10)
            intensity_corr = np.corrcoef(orig_matched, pred_matched)[0, 1] if n_matched > 2 else cosine_sim
        else:
            cosine_sim = 0.0
            intensity_corr = 0.0

        # 3. S-entropy coordinate distance
        s_entropy_dist = self._compute_s_entropy_distance(
            original_mz, orig_norm, predicted_mz, pred_norm, precursor_mz
        )

        # 4. CV droplet overlap
        droplet_overlap = self._compute_droplet_overlap(
            original_mz, orig_norm, predicted_mz, pred_norm, precursor_mz
        )

        # Combined circular validation score
        # Weights: cosine (0.3), mass accuracy (0.2), S-entropy (0.25), droplet (0.25)
        coverage = n_matched / max(len(original_mz), 1)
        mass_score = max(0, 1.0 - mass_accuracy / self.mz_tolerance)
        s_entropy_score = np.exp(-s_entropy_dist / 0.5)  # Exponential decay

        circular_score = (
            0.25 * cosine_sim +
            0.20 * mass_score +
            0.15 * coverage +
            0.20 * s_entropy_score +
            0.20 * droplet_overlap
        )

        return SpectralSimilarity(
            cosine_similarity=cosine_sim,
            matched_peaks=n_matched,
            total_original_peaks=len(original_mz),
            total_predicted_peaks=len(predicted_mz),
            mass_accuracy=mass_accuracy,
            intensity_correlation=intensity_corr if not np.isnan(intensity_corr) else 0.0,
            s_entropy_distance=s_entropy_dist,
            droplet_overlap=droplet_overlap,
            circular_validation_score=circular_score
        )

    def _match_peaks(
        self,
        original_mz: np.ndarray,
        predicted_mz: np.ndarray
    ) -> Tuple[List[int], List[int], List[float]]:
        """Match peaks between spectra using Hungarian algorithm."""
        if len(original_mz) == 0 or len(predicted_mz) == 0:
            return [], [], []

        # Build cost matrix
        n_orig = len(original_mz)
        n_pred = len(predicted_mz)

        cost_matrix = np.full((n_orig, n_pred), np.inf)

        for i, mz1 in enumerate(original_mz):
            for j, mz2 in enumerate(predicted_mz):
                error = abs(mz1 - mz2)
                if error < self.mz_tolerance:
                    cost_matrix[i, j] = error

        # Hungarian algorithm for optimal matching
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError:
            return [], [], []

        matched_orig = []
        matched_pred = []
        mass_errors = []

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.mz_tolerance:
                matched_orig.append(i)
                matched_pred.append(j)
                mass_errors.append(cost_matrix[i, j])

        return matched_orig, matched_pred, mass_errors

    def _compute_s_entropy_distance(
        self,
        mz1: np.ndarray,
        intensity1: np.ndarray,
        mz2: np.ndarray,
        intensity2: np.ndarray,
        precursor_mz: float
    ) -> float:
        """
        Compute distance between spectra in S-entropy coordinate space.

        S-entropy coordinates (S_k, S_t, S_e):
        - S_k: Knowledge entropy (from intensity distribution)
        - S_t: Time entropy (from m/z ordering)
        - S_e: Energy entropy (from m/z values relative to precursor)
        """
        if len(mz1) == 0 or len(mz2) == 0:
            return 1.0

        # Compute S-entropy coordinates for each spectrum
        coords1 = self._spectrum_to_s_entropy(mz1, intensity1, precursor_mz)
        coords2 = self._spectrum_to_s_entropy(mz2, intensity2, precursor_mz)

        # Euclidean distance in S-entropy space
        distance = np.sqrt(
            (coords1[0] - coords2[0])**2 +
            (coords1[1] - coords2[1])**2 +
            (coords1[2] - coords2[2])**2
        )

        return distance

    def _spectrum_to_s_entropy(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        precursor_mz: float
    ) -> Tuple[float, float, float]:
        """Convert spectrum to S-entropy coordinates."""
        # S_k: Knowledge entropy - normalized intensity entropy
        p = intensity / max(np.sum(intensity), 1e-10)
        p = p[p > 0]  # Remove zeros for log
        s_k = -np.sum(p * np.log2(p + 1e-10)) / max(np.log2(len(mz)), 1)

        # S_t: Time entropy - ordering information
        mz_norm = (mz - np.min(mz)) / max(np.max(mz) - np.min(mz), 1)
        s_t = np.std(mz_norm)

        # S_e: Energy entropy - relative to precursor
        s_e = np.mean(mz / precursor_mz)

        return (s_k, s_t, s_e)

    def _compute_droplet_overlap(
        self,
        mz1: np.ndarray,
        intensity1: np.ndarray,
        mz2: np.ndarray,
        intensity2: np.ndarray,
        precursor_mz: float
    ) -> float:
        """
        Compute overlap between spectra in CV droplet representation.

        Uses the bijective ion-to-droplet transformation.
        """
        if len(mz1) == 0 or len(mz2) == 0:
            return 0.0

        # Create droplet mapper
        mapper = StateCountingDropletMapper()

        # Convert to droplets
        droplets1, _ = mapper.spectrum_to_droplets(mz1, intensity1 * 1000, precursor_mz)
        droplets2, _ = mapper.spectrum_to_droplets(mz2, intensity2 * 1000, precursor_mz)

        if not droplets1 or not droplets2:
            return 0.0

        # Compute pairwise overlaps using optimal matching
        n1, n2 = len(droplets1), len(droplets2)

        # For efficiency, use simplified overlap metric
        # Center of mass comparison
        com1_x = np.mean([d.center[0] for d in droplets1])
        com1_y = np.mean([d.center[1] for d in droplets1])
        com2_x = np.mean([d.center[0] for d in droplets2])
        com2_y = np.mean([d.center[1] for d in droplets2])

        # Distance between centers of mass (normalized)
        canvas_diag = np.sqrt(256**2 + 256**2)  # Default canvas size
        com_distance = np.sqrt((com1_x - com2_x)**2 + (com1_y - com2_y)**2)
        com_overlap = max(0, 1 - com_distance / canvas_diag)

        # Energy distribution similarity
        energies1 = np.array([d.energy for d in droplets1])
        energies2 = np.array([d.energy for d in droplets2])

        e1_norm = energies1 / max(np.sum(energies1), 1e-10)
        e2_norm = energies2 / max(np.sum(energies2), 1e-10)

        # Use histogram comparison for energy distribution
        n_bins = 20
        hist1, _ = np.histogram(energies1, bins=n_bins, range=(0, max(np.max(energies1), np.max(energies2))))
        hist2, _ = np.histogram(energies2, bins=n_bins, range=(0, max(np.max(energies1), np.max(energies2))))

        hist1 = hist1 / max(np.sum(hist1), 1)
        hist2 = hist2 / max(np.sum(hist2), 1)

        # Histogram intersection
        energy_overlap = np.sum(np.minimum(hist1, hist2))

        return 0.5 * com_overlap + 0.5 * energy_overlap


class CircularValidationReconstructor:
    """
    Sequence reconstruction using CIRCULAR VALIDATION.

    This implements the user's insight: "A→B→C→A is not fallacious... A→B is"

    Algorithm:
    1. Generate candidate sequences (from state counting or de novo)
    2. For each candidate:
       a. Generate theoretical spectrum (forward prediction)
       b. Compare to original spectrum (circular validation)
    3. Rank candidates by circular validation score
    4. Best match = correct sequence

    The bijective property ensures: if the candidate is correct,
    its predicted spectrum will match the original.
    """

    def __init__(
        self,
        mass_tolerance: float = 0.5,
        min_intensity_ratio: float = 0.01,
        n_candidates: int = 100,
        use_state_counting: bool = True
    ):
        self.mass_tolerance = mass_tolerance
        self.min_intensity_ratio = min_intensity_ratio
        self.n_candidates = n_candidates
        self.use_state_counting = use_state_counting

        # Components
        self.state_reconstructor = StateCountingReconstructor(
            mass_tolerance=mass_tolerance,
            min_intensity_ratio=min_intensity_ratio
        )
        self.spectrum_generator = TheoreticalSpectrumGenerator()
        self.similarity_computer = SpectralSimilarityComputer(mz_tolerance=mass_tolerance)

    def reconstruct(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int = 2,
        known_sequence: Optional[str] = None,
        try_modifications: bool = True
    ) -> Dict:
        """
        Reconstruct sequence using circular validation.

        Args:
            mz_array: Fragment m/z values
            intensity_array: Fragment intensities
            precursor_mz: Precursor m/z
            precursor_charge: Precursor charge state
            known_sequence: Known sequence for validation (optional)
            try_modifications: Try common modifications (TMT, iTRAQ, etc.)

        Returns:
            Dictionary with reconstruction results including circular validation scores
        """
        # Calculate precursor mass
        precursor_mass = (precursor_mz * precursor_charge) - (precursor_charge * 1.007276)

        # Try different modification scenarios
        modification_scenarios = [0.0]  # Unmodified
        if try_modifications:
            # Add common modifications
            modification_scenarios.extend([
                MODIFICATION_MASSES['TMT'],       # TMT tag
                MODIFICATION_MASSES['iTRAQ4'],    # iTRAQ 4-plex
                MODIFICATION_MASSES['Carbamidomethyl'],  # CAM on Cys
            ])

        all_candidates = []

        for mod_mass in modification_scenarios:
            # Adjusted precursor mass (subtract modification)
            adjusted_mass = precursor_mass - mod_mass

            if adjusted_mass < 300:  # Too small for a peptide
                continue

            # Step 1: Generate candidates using state counting
            state_result = self.state_reconstructor.reconstruct(
                mz_array, intensity_array, precursor_mz, precursor_charge, known_sequence
            )

            initial_candidates = state_result.get('candidates', [])

            # Step 2: If state counting didn't produce good candidates, use de novo generation
            # Check for: no candidates, empty sequences, or sequences too short
            good_candidates = [c for c in initial_candidates
                               if c.get('sequence') and len(c.get('sequence', '')) >= 3]

            if not good_candidates:
                de_novo_candidates = self._generate_de_novo_candidates(
                    mz_array, intensity_array, adjusted_mass
                )
                initial_candidates = de_novo_candidates
            else:
                initial_candidates = good_candidates

            # Step 3: Expand candidate pool with variations
            candidates = self._expand_candidates(
                initial_candidates,
                adjusted_mass
            )

            # Tag candidates with modification info
            for c in candidates:
                c['modification_mass'] = mod_mass
                if mod_mass > 0:
                    # Find modification name
                    for name, mass in MODIFICATION_MASSES.items():
                        if abs(mass - mod_mass) < 0.1:
                            c['modification'] = name
                            break
                else:
                    c['modification'] = None

            all_candidates.extend(candidates)

        # Step 4: Circular validation for each candidate
        validated_candidates = []

        for candidate in all_candidates[:self.n_candidates]:
            sequence = candidate.get('sequence', '')
            if not sequence or len(sequence) < 2:
                continue

            # Generate theoretical spectrum for this candidate
            pred_mz, pred_intensity = self.spectrum_generator.generate_spectrum(
                sequence, precursor_charge
            )

            if len(pred_mz) == 0:
                continue

            # Compute circular validation score
            similarity = self.similarity_computer.compute_similarity(
                mz_array, intensity_array,
                pred_mz, pred_intensity,
                precursor_mz
            )

            # Also validate fragment-parent hierarchy
            hierarchy = validate_fragment_hierarchy(mz_array, intensity_array, precursor_mz)

            validated_candidates.append({
                'sequence': sequence,
                'length': len(sequence),
                'modification': candidate.get('modification'),
                'modification_mass': candidate.get('modification_mass', 0.0),
                'original_score': candidate.get('score', 0.0),
                'circular_validation_score': similarity.circular_validation_score,
                'cosine_similarity': similarity.cosine_similarity,
                'matched_peaks': similarity.matched_peaks,
                'mass_accuracy': similarity.mass_accuracy,
                's_entropy_distance': similarity.s_entropy_distance,
                'droplet_overlap': similarity.droplet_overlap,
                'hierarchy_score': hierarchy.overall_score,
                'hierarchy_valid': hierarchy.is_valid,
                # Combined score emphasizing circular validation
                'combined_score': (
                    0.6 * similarity.circular_validation_score +
                    0.2 * candidate.get('score', 0.0) +
                    0.2 * hierarchy.overall_score
                )
            })

        # Sort by combined score
        validated_candidates.sort(key=lambda x: -x['combined_score'])

        # Get best result
        best_candidate = validated_candidates[0] if validated_candidates else None
        best_sequence = best_candidate['sequence'] if best_candidate else ''
        best_score = best_candidate['combined_score'] if best_candidate else 0.0

        # Compute validation metrics if known sequence provided
        validation = {}
        if known_sequence and best_sequence:
            validation = self.state_reconstructor._validate_reconstruction(
                best_sequence, known_sequence
            )

        return {
            'method': 'circular_validation',
            'precursor_mass': precursor_mass,
            'precursor_mz': precursor_mz,
            'precursor_charge': precursor_charge,
            'n_fragments': len(mz_array),
            'n_candidates_tested': len(validated_candidates),
            'candidates': validated_candidates[:10],
            'best_sequence': best_sequence,
            'best_score': best_score,
            'best_modification': best_candidate.get('modification') if best_candidate else None,
            'circular_validation_score': best_candidate['circular_validation_score'] if best_candidate else 0.0,
            'validation': validation,
        }

    def _expand_candidates(
        self,
        initial_candidates: List[Dict],
        precursor_mass: float
    ) -> List[Dict]:
        """
        Expand candidate pool with sequence variations.

        Generates variations by:
        1. I/L substitution (isobaric)
        2. Q/K substitution (near-isobaric)
        3. N-terminal and C-terminal additions
        4. Single amino acid permutations
        """
        expanded = list(initial_candidates)
        seen_sequences = set(c.get('sequence', '') for c in initial_candidates)

        for candidate in initial_candidates[:20]:  # Limit expansion to top candidates
            sequence = candidate.get('sequence', '')
            base_score = candidate.get('score', 0.5)

            if not sequence:
                continue

            # I/L substitutions
            for i, aa in enumerate(sequence):
                if aa == 'I':
                    new_seq = sequence[:i] + 'L' + sequence[i+1:]
                    if new_seq not in seen_sequences:
                        expanded.append({'sequence': new_seq, 'score': base_score * 0.95})
                        seen_sequences.add(new_seq)
                elif aa == 'L':
                    new_seq = sequence[:i] + 'I' + sequence[i+1:]
                    if new_seq not in seen_sequences:
                        expanded.append({'sequence': new_seq, 'score': base_score * 0.95})
                        seen_sequences.add(new_seq)

            # Q/K substitutions (mass diff ~0.036 Da)
            for i, aa in enumerate(sequence):
                if aa == 'Q':
                    new_seq = sequence[:i] + 'K' + sequence[i+1:]
                    if new_seq not in seen_sequences:
                        expanded.append({'sequence': new_seq, 'score': base_score * 0.9})
                        seen_sequences.add(new_seq)
                elif aa == 'K':
                    new_seq = sequence[:i] + 'Q' + sequence[i+1:]
                    if new_seq not in seen_sequences:
                        expanded.append({'sequence': new_seq, 'score': base_score * 0.9})
                        seen_sequences.add(new_seq)

            # Try mass-completing additions
            current_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in sequence) + 18.015
            mass_deficit = precursor_mass - current_mass

            if 50 < mass_deficit < 250:  # Room for 1-2 amino acids
                for aa, aa_mass in AMINO_ACID_MASSES.items():
                    if abs(aa_mass - mass_deficit) < self.mass_tolerance:
                        # Try adding at C-terminus
                        new_seq = sequence + aa
                        if new_seq not in seen_sequences:
                            expanded.append({'sequence': new_seq, 'score': base_score * 0.8})
                            seen_sequences.add(new_seq)
                        # Try adding at N-terminus
                        new_seq = aa + sequence
                        if new_seq not in seen_sequences:
                            expanded.append({'sequence': new_seq, 'score': base_score * 0.75})
                            seen_sequences.add(new_seq)

        return expanded

    def _generate_de_novo_candidates(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mass: float
    ) -> List[Dict]:
        """
        Generate de novo candidate sequences when state counting fails.

        Uses fragment ions and precursor mass to generate plausible candidates.
        Relies on circular validation to score and rank them.

        Strategies:
        1. Identify complementary b/y ion pairs
        2. Build sequence from fragment mass differences
        3. Generate diverse mass-matching candidates
        """
        candidates = []

        # Estimate sequence length from precursor mass
        avg_aa_mass = 110.0  # Average amino acid mass
        estimated_length = max(3, min(30, int(round((precursor_mass - 18.015) / avg_aa_mass))))

        # Sort fragments by intensity (highest first)
        sorted_idx = np.argsort(intensity_array)[::-1]
        top_mz = mz_array[sorted_idx[:min(50, len(mz_array))]]
        top_int = intensity_array[sorted_idx[:min(50, len(intensity_array))]]

        # Strategy 1: Find complementary b/y ion pairs
        # For a peptide of mass M, b_i + y_{n-i} = M + 2*H (approx M + 2)
        complementary_pairs = []
        target_sum = precursor_mass + 2.0  # Approximate sum for complementary pair

        for i, mz1 in enumerate(top_mz):
            for j, mz2 in enumerate(top_mz):
                if i >= j:
                    continue
                pair_sum = mz1 + mz2
                if abs(pair_sum - target_sum) < self.mass_tolerance * 2:
                    complementary_pairs.append((mz1, mz2, top_int[i] * top_int[j]))

        # Sort by combined intensity
        complementary_pairs.sort(key=lambda x: -x[2])

        # Strategy 2: Build ordered fragment list from pairs
        ordered_fragments = []
        if complementary_pairs:
            # Use complementary pairs to order fragments
            for mz1, mz2, _ in complementary_pairs[:10]:
                b_ion = min(mz1, mz2)  # b-ions are typically smaller
                y_ion = max(mz1, mz2)
                if b_ion not in ordered_fragments:
                    ordered_fragments.append(b_ion)

        # Add other high-intensity fragments
        for mz in top_mz[:20]:
            if mz not in ordered_fragments:
                ordered_fragments.append(mz)

        # Sort by m/z
        ordered_fragments.sort()

        # Strategy 3: Extract amino acids from consecutive fragment differences
        identified_positions = []  # (position, amino_acid, confidence)

        for i in range(len(ordered_fragments) - 1):
            diff = ordered_fragments[i + 1] - ordered_fragments[i]

            for aa, mass in AMINO_ACID_MASSES.items():
                if abs(diff - mass) < self.mass_tolerance:
                    identified_positions.append((i, aa, 1.0 - abs(diff - mass) / self.mass_tolerance))
                    break

        # Strategy 4: Build sequence from identified positions
        if identified_positions:
            # Sort by position
            identified_positions.sort(key=lambda x: x[0])

            # Build sequence
            seq_parts = [aa for _, aa, _ in identified_positions]
            if seq_parts:
                candidate_seq = ''.join(seq_parts)

                # Try to complete to match precursor mass
                current_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in candidate_seq) + 18.015
                mass_deficit = precursor_mass - current_mass

                # Add amino acids to complete
                if mass_deficit > 50:
                    completion = self._find_mass_completion(mass_deficit)
                    if completion:
                        candidate_seq = candidate_seq + completion

                if len(candidate_seq) >= 3:
                    actual_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in candidate_seq) + 18.015
                    mass_error = abs(actual_mass - precursor_mass) / precursor_mass
                    score = max(0, 1.0 - mass_error)
                    candidates.append({'sequence': candidate_seq, 'score': score * 0.7})

        # Strategy 5: Generate diverse candidates using different amino acid combinations
        aa_groups = {
            'hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'],
            'polar': ['S', 'T', 'N', 'Q', 'C', 'Y'],
            'charged': ['K', 'R', 'H', 'D', 'E'],
            'small': ['G', 'A', 'S'],
        }

        for group_name, group_aas in aa_groups.items():
            for _ in range(10):  # Generate multiple candidates per group
                seq = ''
                remaining_mass = precursor_mass - 18.015

                # Mix amino acids from the group
                np.random.shuffle(group_aas)

                for aa in group_aas * 5:
                    if AMINO_ACID_MASSES[aa] <= remaining_mass + self.mass_tolerance:
                        seq += aa
                        remaining_mass -= AMINO_ACID_MASSES[aa]

                    if remaining_mass < 50 or len(seq) >= estimated_length + 2:
                        break

                if seq and len(seq) >= 3:
                    actual_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in seq) + 18.015
                    mass_error = abs(actual_mass - precursor_mass) / precursor_mass

                    if mass_error < 0.1:
                        candidates.append({'sequence': seq, 'score': (1.0 - mass_error) * 0.4})

        # Strategy 6: Tryptic peptide patterns (end in K or R)
        for terminal in ['R', 'K']:
            for n_terminal in ['', 'M', 'A', 'G', 'S']:  # Common N-terminal residues
                remaining_mass = precursor_mass - 18.015 - AMINO_ACID_MASSES[terminal]
                if n_terminal:
                    remaining_mass -= AMINO_ACID_MASSES[n_terminal]

                # Try building middle section
                middle_aas = ['L', 'A', 'V', 'I', 'E', 'G', 'S', 'P', 'D', 'N', 'T']
                np.random.shuffle(middle_aas)

                middle = ''
                for aa in middle_aas * 3:
                    if AMINO_ACID_MASSES[aa] <= remaining_mass + self.mass_tolerance:
                        middle += aa
                        remaining_mass -= AMINO_ACID_MASSES[aa]

                    if remaining_mass < 50 or len(middle) >= estimated_length:
                        break

                seq = n_terminal + middle + terminal
                if len(seq) >= 3:
                    actual_mass = sum(AMINO_ACID_MASSES.get(aa, 110) for aa in seq) + 18.015
                    mass_error = abs(actual_mass - precursor_mass) / precursor_mass

                    if mass_error < 0.1:
                        candidates.append({'sequence': seq, 'score': (1.0 - mass_error) * 0.5})

        # Remove duplicates and sort by score
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c['sequence'] not in seen:
                seen.add(c['sequence'])
                unique_candidates.append(c)

        unique_candidates.sort(key=lambda x: -x['score'])

        return unique_candidates[:100]

    def _find_mass_completion(self, target_mass: float) -> str:
        """Find amino acid combination to complete a mass deficit."""
        # Try single amino acid
        for aa, mass in AMINO_ACID_MASSES.items():
            if abs(mass - target_mass) < self.mass_tolerance:
                return aa

        # Try two amino acids
        for aa1, mass1 in AMINO_ACID_MASSES.items():
            for aa2, mass2 in AMINO_ACID_MASSES.items():
                if abs(mass1 + mass2 - target_mass) < self.mass_tolerance:
                    return aa1 + aa2

        # Try three amino acids
        for aa1, mass1 in AMINO_ACID_MASSES.items():
            if mass1 > target_mass:
                continue
            for aa2, mass2 in AMINO_ACID_MASSES.items():
                if mass1 + mass2 > target_mass:
                    continue
                for aa3, mass3 in AMINO_ACID_MASSES.items():
                    if abs(mass1 + mass2 + mass3 - target_mass) < self.mass_tolerance:
                        return aa1 + aa2 + aa3

        return ''


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def reconstruct_sequence_circular_validation(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    precursor_charge: int = 2,
    known_sequence: Optional[str] = None
) -> Dict:
    """
    Sequence reconstruction using circular validation (bijective completion).

    This is the RECOMMENDED method - uses A→B→C→A validation.
    """
    reconstructor = CircularValidationReconstructor()
    return reconstructor.reconstruct(
        mz_array, intensity_array, precursor_mz, precursor_charge, known_sequence
    )


def reconstruct_sequence_state_counting(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    precursor_charge: int = 2,
    known_sequence: Optional[str] = None
) -> Dict:
    """
    Quick sequence reconstruction using state counting.
    """
    reconstructor = StateCountingReconstructor()
    return reconstructor.reconstruct(
        mz_array, intensity_array, precursor_mz, precursor_charge, known_sequence
    )


def validate_fragment_hierarchy(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float
) -> FragmentParentValidation:
    """
    Validate fragment-parent hierarchical relationship.
    """
    mapper = StateCountingDropletMapper()
    fragment_droplets, parent_droplet = mapper.spectrum_to_droplets(
        mz_array, intensity_array, precursor_mz
    )
    return validate_fragment_parent_hierarchy(fragment_droplets, parent_droplet)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("State Counting Mass Spectrometry - Example")
    print("=" * 60)

    # Example: Partition coordinates
    print("\n1. Partition Coordinates")
    print("-" * 40)

    for n in range(1, 6):
        c = capacity(n)
        c_tot = total_capacity(n)
        print(f"  n={n}: C(n)={c:3d}, C_tot={c_tot:4d}")

    # Example: Amino acid states
    print("\n2. Amino Acid State Mapping")
    print("-" * 40)

    for aa in ['G', 'A', 'S', 'F', 'W']:
        state = AMINO_ACID_STATES[aa]
        print(f"  {aa}: mass={state.mass:.2f}, n={state.partition_state.n}, "
              f"l={state.partition_state.l}, index={state.state_index}")

    # Example: Sequence reconstruction
    print("\n3. Sequence Reconstruction")
    print("-" * 40)

    # Simulated b-ion series for PEPTIDE
    b_ions = np.array([97.05, 226.09, 323.14, 436.23, 551.26, 664.34])
    intensities = np.array([1000, 5000, 8000, 12000, 6000, 3000])
    precursor = 800.35

    result = reconstruct_sequence_state_counting(
        b_ions, intensities, precursor, 1, known_sequence="PEPTIDE"
    )

    print(f"  Precursor mass: {result['precursor_mass']:.2f}")
    print(f"  Best sequence: {result['best_sequence']}")
    print(f"  Best score: {result['best_score']:.3f}")

    if result['validation']:
        print(f"  Exact match: {result['validation']['exact_match']}")
        print(f"  Partial score: {result['validation']['partial_score']:.3f}")

    # Example: Fragment-parent validation
    print("\n4. Fragment-Parent Validation")
    print("-" * 40)

    validation = validate_fragment_hierarchy(b_ions, intensities, precursor)
    print(f"  Overlap score: {validation.overlap_score:.3f}")
    print(f"  Wavelength ratio: {validation.wavelength_ratio:.3f}")
    print(f"  Energy ratio: {validation.energy_ratio:.3f}")
    print(f"  Phase coherence: {validation.phase_coherence:.3f}")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Overall score: {validation.overall_score:.3f}")

    print("\n" + "=" * 60)
