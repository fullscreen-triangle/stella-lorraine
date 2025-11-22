"""
Biological Maxwell Demon Recursive Decomposition

Each MD → 3 sub-MDs (S_k, S_t, S_e), creating 3^k parallel channels

Key identity: Harmonic ≡ Maxwell Demon ≡ Filter[states → specific ω]

This implements the recursive structure from the thermometry paper:
- Level 0: 1 MD
- Level 1: 3 MDs
- Level 2: 9 MDs
- Level k: 3^k MDs

All operate in parallel with zero chronological time (categorical simultaneity)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaxwellDemon:
    """
    Single Maxwell Demon operating on one frequency

    Identity: Each oscillation frequency IS a Maxwell Demon
    The demon filters phase space to select states with that frequency.

    Decomposition: Each MD splits into 3 sub-MDs along S-entropy axes
    """
    frequency_hz: float
    s_k: float  # Knowledge entropy coordinate
    s_t: float  # Temporal entropy coordinate
    s_e: float  # Evolution entropy coordinate
    depth: int  # Recursion depth in hierarchy
    parent: Optional['MaxwellDemon'] = field(default=None, repr=False)
    children: List['MaxwellDemon'] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Validate demon properties"""
        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be positive")
        if self.depth < 0:
            raise ValueError("Depth must be non-negative")

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf demon (no children)"""
        return len(self.children) == 0

    @property
    def total_entropy(self) -> float:
        """Total S-entropy of this demon"""
        return self.s_k + self.s_t + self.s_e

    def decompose(self) -> List['MaxwellDemon']:
        """
        Decompose into 3 sub-demons along S-entropy axes

        Creates:
        - Sub-demon 0: Filters along S_k (knowledge) axis
        - Sub-demon 1: Filters along S_t (temporal) axis
        - Sub-demon 2: Filters along S_e (evolution) axis

        Each sub-demon operates at a slightly shifted frequency
        to capture different aspects of the oscillation.

        Returns:
            List of 3 child demons
        """
        if self.children:
            # Already decomposed
            return self.children

        # Frequency shifts for each child (small perturbations)
        # These arise from different S-entropy projections
        freq_shifts = [
            1.0 + 0.001 * np.cos(2 * np.pi * self.s_k),  # S_k dependent
            1.0 + 0.001 * np.cos(2 * np.pi * self.s_t),  # S_t dependent
            1.0 + 0.001 * np.cos(2 * np.pi * self.s_e),  # S_e dependent
        ]

        # Create 3 children
        for i, shift in enumerate(freq_shifts):
            child = MaxwellDemon(
                frequency_hz=self.frequency_hz * shift,
                s_k=self.s_k + (1.0 if i == 0 else 0.0),  # Increment S_k for child 0
                s_t=self.s_t + (1.0 if i == 1 else 0.0),  # Increment S_t for child 1
                s_e=self.s_e + (1.0 if i == 2 else 0.0),  # Increment S_e for child 2
                depth=self.depth + 1,
                parent=self
            )
            self.children.append(child)

        return self.children

    def decompose_recursive(self, target_depth: int) -> List['MaxwellDemon']:
        """
        Recursively decompose to target depth

        Args:
            target_depth: Maximum depth to decompose

        Returns:
            All leaf demons at target depth (will be 3^target_depth demons)
        """
        if self.depth >= target_depth:
            return [self]

        # Decompose this demon
        children = self.decompose()

        # Recursively decompose children
        all_leaves = []
        for child in children:
            all_leaves.extend(child.decompose_recursive(target_depth))

        return all_leaves

    def act_as_source(self) -> float:
        """
        MD acts as frequency source

        Generates phase based on its S-entropy state
        """
        # Phase generation depends on temporal coordinate
        phase = self.frequency_hz * (self.s_t if self.s_t > 0 else 1e-10)
        return phase

    def act_as_detector(self, incoming_phase: float) -> float:
        """
        MD acts as frequency detector

        Receives and processes incoming phase information
        """
        # Detection depends on evolution coordinate
        detected_frequency = incoming_phase / (2 * np.pi * (self.s_e if self.s_e > 0 else 1.0))
        return detected_frequency

    def source_detector_unified(self, network_frequency: float) -> float:
        """
        Unified source-detector operation

        The SAME demon acts as both source AND detector simultaneously.
        This is valid in categorical space due to S_t evolution creating
        categorical distance from itself: d_cat(MD, MD) ≠ 0

        Args:
            network_frequency: Frequency from harmonic network node

        Returns:
            Combined frequency measurement
        """
        # As source
        phase_source = self.act_as_source()

        # As detector
        phase_detector = network_frequency * (2 * np.pi)

        # Categorical correlation (orthogonal S-space components)
        combined_phase = np.sqrt(phase_source**2 + phase_detector**2)

        # Convert back to frequency
        measured_frequency = combined_phase / (2 * np.pi)

        return measured_frequency


class BMDHierarchy:
    """
    Complete BMD decomposition tree

    Structure forms exponential expansion:
    - Level 0: 1 root MD
    - Level 1: 3 MDs (first decomposition)
    - Level 2: 9 MDs (second decomposition)
    - Level k: 3^k MDs

    All leaf MDs at depth k operate in parallel (zero chronological time)
    """

    def __init__(self, root_frequency: float, initial_s_coords: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize hierarchy with root demon

        Args:
            root_frequency: Base oscillation frequency (Hz)
            initial_s_coords: Initial (S_k, S_t, S_e) coordinates
        """
        self.root = MaxwellDemon(
            frequency_hz=root_frequency,
            s_k=initial_s_coords[0],
            s_t=initial_s_coords[1],
            s_e=initial_s_coords[2],
            depth=0
        )

    def build_hierarchy(self, depth: int) -> List[MaxwellDemon]:
        """
        Build complete hierarchy to specified depth

        Returns all leaf demons at target depth (operational MDs)

        Args:
            depth: Target decomposition depth

        Returns:
            List of 3^depth leaf demons
        """
        logger.info(f"Building BMD hierarchy to depth {depth}...")

        leaves = self.root.decompose_recursive(depth)

        n_leaves = len(leaves)
        expected = 3 ** depth

        if n_leaves != expected:
            logger.warning(f"Expected {expected} leaves, got {n_leaves}")
        else:
            logger.info(f"✓ BMD hierarchy complete: {n_leaves} parallel demons at depth {depth}")

        return leaves

    @staticmethod
    def count_at_depth(depth: int) -> int:
        """
        Calculate number of demons at given depth

        Formula: N(k) = 3^k
        """
        return 3 ** depth

    def total_parallel_channels(self, depth: int) -> int:
        """
        Total number of parallel measurement channels

        All leaf MDs operate simultaneously in categorical space
        (zero chronological time)
        """
        return self.count_at_depth(depth)

    def enhancement_factor(self, depth: int) -> float:
        """
        Precision enhancement from BMD parallelization

        Each channel provides independent categorical information.
        In categorical space (not phase space), information adds directly,
        not as sqrt(N) like classical statistics.

        Returns:
            Enhancement factor = 3^depth
        """
        return float(self.count_at_depth(depth))

    def get_all_frequencies(self, depth: int) -> np.ndarray:
        """
        Get frequencies of all demons at target depth

        Args:
            depth: Target depth

        Returns:
            Array of frequencies (Hz)
        """
        leaves = self.build_hierarchy(depth)
        return np.array([md.frequency_hz for md in leaves])

    def get_s_coordinates(self, depth: int) -> np.ndarray:
        """
        Get S-entropy coordinates of all demons at target depth

        Args:
            depth: Target depth

        Returns:
            Array of shape (3^depth, 3) with (S_k, S_t, S_e) coordinates
        """
        leaves = self.build_hierarchy(depth)
        coords = np.array([[md.s_k, md.s_t, md.s_e] for md in leaves])
        return coords


def verify_exponential_scaling(max_depth: int = 10) -> bool:
    """
    Verify that BMD decomposition follows 3^k scaling

    This is a critical validation of the theoretical structure.

    Args:
        max_depth: Maximum depth to test

    Returns:
        True if scaling matches 3^k for all depths
    """
    logger.info("Verifying BMD exponential scaling...")

    hierarchy = BMDHierarchy(root_frequency=1e13)

    all_match = True
    for k in range(max_depth + 1):
        leaves = hierarchy.build_hierarchy(k)
        actual = len(leaves)
        expected = 3 ** k

        match = (actual == expected)

        if match:
            logger.info(f"  Depth {k}: {actual} demons (✓ matches 3^{k})")
        else:
            logger.error(f"  Depth {k}: {actual} demons (✗ expected 3^{k} = {expected})")
            all_match = False

    if all_match:
        logger.info("✓ BMD exponential scaling verified")
    else:
        logger.error("✗ BMD scaling FAILED")

    return all_match
