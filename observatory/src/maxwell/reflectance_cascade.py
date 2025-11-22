"""
Reflectance Cascade: Quadratic Information Gain Through Multiple Observations
============================================================================

Key insight: Each "reflection" in categorical space ADDS information rather
than losing it. Information gain scales as (n+1)² where n is cascade depth.

This is the opposite of physical reflections which lose energy!

In categorical space:
- First observation: I₁ bits
- Second observation (cascade 1): I₁ + 4×I₁ = 5×I₁ bits
- Third observation (cascade 2): 5×I₁ + 9×I₁ = 14×I₁ bits
- nth observation: Σ(k+1)² for k=0 to n-1

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class CascadeObservation:
    """
    A single observation in the reflectance cascade
    """
    cascade_level: int
    observer_id: str
    s_state_observed: Any  # S-entropy coordinates
    information_bits: float
    cumulative_information: float
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cascade_level': self.cascade_level,
            'observer_id': self.observer_id,
            'information_bits': self.information_bits,
            'cumulative_information': self.cumulative_information,
            'timestamp': self.timestamp
        }


class ReflectanceCascade:
    """
    Implements reflectance cascade for quadratic information gain

    Each cascade level:
    1. Observes current state
    2. Calculates information gain: I_n = (n+1)² × I_base
    3. Updates cumulative information
    4. Optionally triggers next cascade
    """

    def __init__(
        self,
        base_information_bits: float = 1.0,
        max_cascade_depth: int = 10,
        name: str = "cascade"
    ):
        self.base_information = base_information_bits
        self.max_depth = max_cascade_depth
        self.name = name

        self.observations: List[CascadeObservation] = []
        self.cumulative_information = 0.0

        logger.debug(
            f"Created ReflectanceCascade '{name}' "
            f"(base_info={base_information_bits}, max_depth={max_cascade_depth})"
        )

    def observe(
        self,
        observer_id: str,
        s_state: Any,
        cascade_level: int = 0,
        timestamp: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> CascadeObservation:
        """
        Perform observation at specified cascade level

        Args:
            observer_id: ID of observer (pixel demon, molecular demon, etc.)
            s_state: S-entropy state being observed
            cascade_level: Current cascade depth (0 = first observation)
            timestamp: Time of observation
            metadata: Additional information

        Returns:
            CascadeObservation with information gain
        """
        # Information gain at this level: (n+1)² × base
        level_gain = ((cascade_level + 1) ** 2) * self.base_information

        # Update cumulative
        self.cumulative_information += level_gain

        observation = CascadeObservation(
            cascade_level=cascade_level,
            observer_id=observer_id,
            s_state_observed=s_state,
            information_bits=level_gain,
            cumulative_information=self.cumulative_information,
            timestamp=timestamp,
            metadata=metadata or {}
        )

        self.observations.append(observation)

        logger.debug(
            f"Cascade observation at level {cascade_level}: "
            f"+{level_gain:.2f} bits (total: {self.cumulative_information:.2f} bits)"
        )

        return observation

    def propagate_cascade(
        self,
        initial_observer_id: str,
        initial_s_state: Any,
        observer_network: Optional[Any] = None,
        max_depth: Optional[int] = None
    ) -> List[CascadeObservation]:
        """
        Propagate cascade through network of observers

        Args:
            initial_observer_id: Starting observer
            initial_s_state: Initial state to observe
            observer_network: Network containing observers (e.g., PixelDemonGrid)
            max_depth: Override max cascade depth

        Returns:
            List of all observations in cascade
        """
        max_depth = max_depth or self.max_depth

        # First observation
        current_s_state = initial_s_state
        current_observer = initial_observer_id

        cascade_observations = []

        for level in range(max_depth):
            # Observe at current level
            obs = self.observe(
                observer_id=current_observer,
                s_state=current_s_state,
                cascade_level=level,
                timestamp=level * 0.001  # Simulated time
            )

            cascade_observations.append(obs)

            # Find next observer (categorical reflection)
            if observer_network is not None:
                next_observer = self._find_next_observer(
                    current_observer,
                    current_s_state,
                    observer_network
                )

                if next_observer is None:
                    logger.debug(f"Cascade terminated at level {level} (no next observer)")
                    break

                current_observer = next_observer
                # S-state evolves through cascade
                current_s_state = self._evolve_s_state(current_s_state, level)
            else:
                # No network, just accumulate at same observer
                pass

        logger.info(
            f"Cascade completed: {len(cascade_observations)} observations, "
            f"{self.cumulative_information:.2f} total bits"
        )

        return cascade_observations

    def _find_next_observer(
        self,
        current_observer_id: str,
        current_s_state: Any,
        observer_network: Any
    ) -> Optional[str]:
        """
        Find next observer in cascade (categorical reflection)

        Next observer = categorically nearest to current state
        """
        # This would query the observer network
        # For now, return None to end cascade
        return None

    def _evolve_s_state(self, s_state: Any, cascade_level: int) -> Any:
        """
        Evolve S-entropy state through cascade

        State accumulates information (increases S_k)
        """
        if hasattr(s_state, 'S_k'):
            # Increase knowledge entropy
            s_state.S_k += 0.1 * (cascade_level + 1)

        return s_state

    def calculate_total_information(self, num_observations: int) -> float:
        """
        Calculate theoretical total information for n observations

        I_total = Σ(k+1)² × I_base for k=0 to n-1
                = I_base × Σ(k+1)²
                = I_base × [n(n+1)(2n+1)/6]
        """
        if num_observations == 0:
            return 0.0

        n = num_observations
        sum_of_squares = n * (n + 1) * (2 * n + 1) // 6

        return self.base_information * sum_of_squares

    def calculate_precision_enhancement(self, num_observations: int) -> float:
        """
        Calculate precision enhancement factor

        Precision scales with square root of information:
        σ_n = σ_0 / √I_total

        Enhancement = σ_0 / σ_n = √I_total
        """
        total_info = self.calculate_total_information(num_observations)
        return np.sqrt(total_info)

    def get_cascade_summary(self) -> Dict[str, Any]:
        """Get summary of cascade observations"""
        if not self.observations:
            return {
                'num_observations': 0,
                'cumulative_information': 0.0
            }

        return {
            'name': self.name,
            'num_observations': len(self.observations),
            'max_cascade_level': max(obs.cascade_level for obs in self.observations),
            'cumulative_information_bits': self.cumulative_information,
            'base_information_bits': self.base_information,
            'theoretical_max_info': self.calculate_total_information(len(self.observations)),
            'precision_enhancement': self.calculate_precision_enhancement(len(self.observations)),
            'observations': [obs.to_dict() for obs in self.observations]
        }


class PixelDemonCascade:
    """
    Reflectance cascade specifically for Pixel Maxwell Demons

    Integrates with PixelMaxwellDemon class for spatial cascades
    """

    def __init__(self, pixel_demon_grid):
        self.grid = pixel_demon_grid
        self.cascades: Dict[str, ReflectanceCascade] = {}

    def start_cascade_at_pixel(
        self,
        pixel_indices: tuple,
        base_information: float = 1.0,
        max_depth: int = 5
    ) -> ReflectanceCascade:
        """
        Start reflectance cascade at specific pixel

        Cascade propagates to neighboring pixels categorically
        """
        # Get pixel demon
        pixel_demon = self.grid.demons[pixel_indices]
        cascade_id = f"cascade_{pixel_demon.pixel_id}"

        # Create cascade
        cascade = ReflectanceCascade(
            base_information_bits=base_information,
            max_cascade_depth=max_depth,
            name=cascade_id
        )

        # Initial observation
        cascade.observe(
            observer_id=pixel_demon.pixel_id,
            s_state=pixel_demon.s_state,
            cascade_level=0
        )

        # Propagate to neighbors
        self._propagate_to_neighbors(
            pixel_indices,
            cascade,
            current_level=0,
            max_depth=max_depth
        )

        self.cascades[cascade_id] = cascade

        return cascade

    def _propagate_to_neighbors(
        self,
        pixel_indices: tuple,
        cascade: ReflectanceCascade,
        current_level: int,
        max_depth: int
    ):
        """Propagate cascade to neighboring pixels"""
        if current_level >= max_depth - 1:
            return

        # Get current pixel demon
        current_demon = self.grid.demons[pixel_indices]

        # Find neighbors (simple spatial neighbors)
        neighbors = self._get_spatial_neighbors(pixel_indices)

        # Propagate to each neighbor
        for neighbor_indices in neighbors:
            neighbor_demon = self.grid.demons[neighbor_indices]

            # Check categorical proximity
            cat_distance = current_demon.s_state.distance_to(neighbor_demon.s_state)

            # Only propagate if categorically close
            if cat_distance < 1.0:
                # Observe at next cascade level
                cascade.observe(
                    observer_id=neighbor_demon.pixel_id,
                    s_state=neighbor_demon.s_state,
                    cascade_level=current_level + 1
                )

                # Recursively propagate (but don't revisit)
                # In practice, track visited pixels

    def _get_spatial_neighbors(self, indices: tuple) -> List[tuple]:
        """Get spatially adjacent pixel indices"""
        neighbors = []

        if len(indices) == 2:
            # 2D grid
            y, x = indices
            ny, nx = self.grid.shape

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny_new, nx_new = y + dy, x + dx

                    if 0 <= ny_new < ny and 0 <= nx_new < nx:
                        neighbors.append((ny_new, nx_new))

        return neighbors

    def visualize_cascade_map(self) -> np.ndarray:
        """
        Create map showing cumulative information at each pixel
        """
        info_map = np.zeros(self.grid.shape)

        for cascade_id, cascade in self.cascades.items():
            for obs in cascade.observations:
                # Parse pixel coordinates from observer_id
                # Format: "grid_name_y{y}_x{x}"
                parts = obs.observer_id.split('_')
                if len(parts) >= 4:
                    try:
                        y = int(parts[-2][1:])  # Remove 'y' prefix
                        x = int(parts[-1][1:])  # Remove 'x' prefix

                        info_map[y, x] += obs.information_bits
                    except (ValueError, IndexError):
                        pass

        return info_map


def demonstrate_cascade_vs_linear():
    """
    Demonstrate quadratic information gain of cascade vs linear accumulation
    """
    print("=" * 80)
    print("REFLECTANCE CASCADE: Quadratic vs Linear Information Gain")
    print("=" * 80)

    max_observations = 10
    base_info = 1.0

    cascade = ReflectanceCascade(base_information_bits=base_info)

    print("\nObservation | Cascade Gain | Linear Gain | Cascade Total | Linear Total")
    print("-" * 80)

    linear_total = 0.0

    for n in range(1, max_observations + 1):
        # Cascade: (n)² gain at level n-1
        cascade_gain = (n ** 2) * base_info
        cascade_total = cascade.calculate_total_information(n)

        # Linear: constant gain
        linear_gain = base_info
        linear_total += linear_gain

        print(f"    {n:2d}      |    {cascade_gain:6.1f}      |    {linear_gain:4.1f}       |    {cascade_total:7.1f}      |    {linear_total:6.1f}")

    print("\n" + "=" * 80)
    print(f"After {max_observations} observations:")
    print(f"  Cascade total: {cascade_total:.1f} bits")
    print(f"  Linear total:  {linear_total:.1f} bits")
    print(f"  Advantage: {cascade_total / linear_total:.1f}× more information!")
    print(f"  Precision enhancement: {cascade.calculate_precision_enhancement(max_observations):.1f}×")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_cascade_vs_linear()
