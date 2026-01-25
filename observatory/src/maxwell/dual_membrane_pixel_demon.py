"""
Dual-Membrane Pixel Maxwell Demon: Complementary Front/Back States
==================================================================

A dual-membrane pixel demon has two conjugate states:
- FRONT state (visible to observer)
- BACK state (hidden from observer, categorical conjugate)

Key Concepts:
- Each pixel has a cognate on the opposite membrane face
- Only one face is observable at a time
- Changes to front state are reflected/transformed to back state
- The two states switch roles dynamically
- Like viewing a 3D object from 2D: one face visible, one hidden
- The "carbon copy" relationship: front pixels → transformed back pixels

This creates a categorical membrane with complementary information
on each side, enabling trans-dimensional information processing.

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import logging

from pixel_maxwell_demon import (
    PixelMaxwellDemon,
    SEntropyCoordinates,
    MolecularDemon,
    Hypothesis
)

logger = logging.getLogger(__name__)


class MembraneFace(Enum):
    """Which face of the membrane is currently observable"""
    FRONT = "front"
    BACK = "back"


@dataclass
class DualState:
    """
    Dual state representation: front and back faces of membrane

    The states are conjugate pairs in categorical space:
    - Front: Observable state (what measurement reveals)
    - Back: Hidden state (complementary categorical information)
    """
    front_s: SEntropyCoordinates
    back_s: SEntropyCoordinates

    def switch(self) -> 'DualState':
        """Switch front and back faces"""
        return DualState(front_s=self.back_s, back_s=self.front_s)

    def categorical_distance(self) -> float:
        """Distance between front and back states in S-space"""
        return self.front_s.distance_to(self.back_s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'front': self.front_s.to_dict(),
            'back': self.back_s.to_dict(),
            'separation': self.categorical_distance()
        }


class ConjugateTransform:
    """
    Transformation operator that maps front state to back state

    This is the categorical operation that creates the "carbon copy":
    changes in front pixels produce transformed back pixels.

    Types of transforms:
    - Phase conjugation: S_k → -S_k (knowledge inversion)
    - Temporal inversion: S_t → -S_t (time reversal)
    - Evolution complement: S_e → 1 - S_e (state complement)
    - Harmonic conjugation: complex conjugate in frequency space
    """

    def __init__(self, transform_type: str = 'phase_conjugate'):
        """
        Initialize conjugate transform

        Args:
            transform_type: Type of transformation
                - 'phase_conjugate': Invert S_k coordinate
                - 'temporal_inverse': Invert S_t coordinate
                - 'evolution_complement': Complement S_e coordinate
                - 'full_conjugate': Invert all coordinates
                - 'harmonic': Complex conjugate in frequency domain
                - 'custom': User-defined transformation
        """
        self.transform_type = transform_type
        self.custom_func: Optional[Callable] = None

    def apply(self, s_state: SEntropyCoordinates) -> SEntropyCoordinates:
        """Apply conjugate transformation to create back state from front state"""
        if self.transform_type == 'phase_conjugate':
            # Invert knowledge coordinate (what's known → what's unknown)
            return SEntropyCoordinates(
                S_k=-s_state.S_k,
                S_t=s_state.S_t,
                S_e=s_state.S_e
            )

        elif self.transform_type == 'temporal_inverse':
            # Invert temporal coordinate (forward time → backward time)
            return SEntropyCoordinates(
                S_k=s_state.S_k,
                S_t=-s_state.S_t,
                S_e=s_state.S_e
            )

        elif self.transform_type == 'evolution_complement':
            # Complement evolution coordinate (current state → alternative state)
            return SEntropyCoordinates(
                S_k=s_state.S_k,
                S_t=s_state.S_t,
                S_e=1.0 - s_state.S_e if s_state.S_e <= 1.0 else -s_state.S_e
            )

        elif self.transform_type == 'full_conjugate':
            # Full categorical conjugation (all coordinates inverted)
            return SEntropyCoordinates(
                S_k=-s_state.S_k,
                S_t=-s_state.S_t,
                S_e=-s_state.S_e
            )

        elif self.transform_type == 'harmonic':
            # Complex conjugate: phases flip, amplitudes preserved
            # In S-space: rotate by π around origin in (S_k, S_t) plane
            theta = np.arctan2(s_state.S_t, s_state.S_k)
            r = np.sqrt(s_state.S_k**2 + s_state.S_t**2)

            return SEntropyCoordinates(
                S_k=r * np.cos(theta + np.pi),
                S_t=r * np.sin(theta + np.pi),
                S_e=s_state.S_e
            )

        elif self.transform_type == 'custom' and self.custom_func is not None:
            return self.custom_func(s_state)

        else:
            # Default: identity (no transformation)
            return s_state

    def set_custom_function(self, func: Callable[[SEntropyCoordinates], SEntropyCoordinates]):
        """Set custom transformation function"""
        self.transform_type = 'custom'
        self.custom_func = func


class DualMembranePixelDemon(PixelMaxwellDemon):
    """
    Dual-membrane pixel demon with front and back conjugate states

    Extends PixelMaxwellDemon with:
    - Dual state (front/back)
    - Conjugate transformation
    - Face switching mechanism
    - Synchronized evolution of both states
    - Carbon-copy pixel transformations
    """

    def __init__(
        self,
        position: np.ndarray,
        pixel_id: Optional[str] = None,
        transform_type: str = 'phase_conjugate',
        switching_frequency: Optional[float] = None
    ):
        """
        Initialize dual-membrane pixel demon

        Args:
            position: Physical position in space
            pixel_id: Unique identifier for this pixel
            transform_type: Type of conjugate transformation
            switching_frequency: Frequency of front/back switching (Hz)
                                If None, switching is manual/event-driven
        """
        # Initialize base pixel demon
        super().__init__(position, pixel_id)

        # Conjugate transform operator
        self.transform = ConjugateTransform(transform_type)

        # Create dual state: front and back
        self.dual_state = self._initialize_dual_state()

        # Which face is currently observable
        self.observable_face = MembraneFace.FRONT

        # Switching parameters
        self.switching_frequency = switching_frequency
        self.last_switch_time = 0.0
        self.switch_count = 0

        # Dual molecular demon lattices (front and back)
        self.front_demons: Dict[str, MolecularDemon] = self.molecular_demons
        self.back_demons: Dict[str, MolecularDemon] = {}

        # Evolution tracking
        self.evolution_history: List[Dict[str, Any]] = []

        logger.info(
            f"Created DualMembranePixelDemon '{self.pixel_id}' "
            f"with {transform_type} transformation"
        )

    def _initialize_dual_state(self) -> DualState:
        """Initialize front and back states as conjugate pair"""
        front_s = self.s_state
        back_s = self.transform.apply(front_s)

        return DualState(front_s=front_s, back_s=back_s)

    def initialize_atmospheric_lattice(
        self,
        temperature_k: float = 288.15,
        pressure_pa: float = 101325.0,
        humidity_fraction: float = 0.5
    ):
        """
        Initialize BOTH front and back molecular demon lattices

        Front lattice: standard atmospheric composition
        Back lattice: conjugate transformation of front lattice
        """
        # Initialize front lattice (standard)
        super().initialize_atmospheric_lattice(
            temperature_k, pressure_pa, humidity_fraction
        )
        self.front_demons = self.molecular_demons.copy()

        # Create back lattice (conjugate of front)
        self.back_demons = {}
        for molecule, front_demon in self.front_demons.items():
            # Apply conjugate transformation to demon's S-state
            back_s_state = self.transform.apply(front_demon.s_state)

            # Create conjugate demon
            back_demon = MolecularDemon(
                molecule_type=f"{molecule}_back",
                s_state=back_s_state,
                vibrational_modes=front_demon.vibrational_modes,  # Same frequencies
                number_density=front_demon.number_density  # Same density
            )
            self.back_demons[molecule] = back_demon

        logger.info(
            f"Initialized dual atmospheric lattice: "
            f"{len(self.front_demons)} front + {len(self.back_demons)} back demons"
        )

    def get_observable_demons(self) -> Dict[str, MolecularDemon]:
        """Get currently observable molecular demon lattice"""
        if self.observable_face == MembraneFace.FRONT:
            return self.front_demons
        else:
            return self.back_demons

    def get_hidden_demons(self) -> Dict[str, MolecularDemon]:
        """Get currently hidden molecular demon lattice"""
        if self.observable_face == MembraneFace.FRONT:
            return self.back_demons
        else:
            return self.front_demons

    def switch_observable_face(self, current_time: float = 0.0):
        """
        Switch which face is observable (front ↔ back)

        This is like rotating the membrane: what was visible becomes hidden,
        what was hidden becomes visible.
        """
        # Switch observable face
        self.observable_face = (
            MembraneFace.BACK if self.observable_face == MembraneFace.FRONT
            else MembraneFace.FRONT
        )

        # Switch dual state
        self.dual_state = self.dual_state.switch()

        # Update current S-state to match observable face
        self.s_state = (
            self.dual_state.front_s if self.observable_face == MembraneFace.FRONT
            else self.dual_state.back_s
        )

        # Swap demon lattices
        self.front_demons, self.back_demons = self.back_demons, self.front_demons
        self.molecular_demons = self.get_observable_demons()

        # Update tracking
        self.last_switch_time = current_time
        self.switch_count += 1

        logger.debug(
            f"Switched to {self.observable_face.value} face "
            f"(switch #{self.switch_count})"
        )

    def auto_switch_if_needed(self, current_time: float):
        """
        Automatically switch faces if switching frequency is set
        """
        if self.switching_frequency is None:
            return

        # Check if it's time to switch
        dt = current_time - self.last_switch_time
        period = 1.0 / self.switching_frequency

        if dt >= period:
            self.switch_observable_face(current_time)

    def propagate_change(
        self,
        front_change: Dict[str, Any],
        current_time: float = 0.0
    ):
        """
        Propagate a change from front to back (or vice versa)

        This is the "carbon copy" mechanism: when pixels change on the
        observable face, the hidden face updates with a transformed version.

        Args:
            front_change: Dictionary describing change to observable face
                         e.g., {'molecule': 'O2', 'density_delta': 0.1}
            current_time: Current time for tracking
        """
        # Apply change to observable face
        if 'molecule' in front_change and 'density_delta' in front_change:
            molecule = front_change['molecule']
            delta = front_change['density_delta']

            observable_demons = self.get_observable_demons()
            hidden_demons = self.get_hidden_demons()

            if molecule in observable_demons:
                # Update observable demon
                observable_demons[molecule].number_density += delta

                # Transform and apply to hidden face
                # (Change in observable → conjugate change in hidden)
                if molecule in hidden_demons:
                    # Apply conjugate transformation to density change
                    # For conjugate: positive change → negative change
                    conjugate_delta = self._conjugate_transform_density(delta)
                    hidden_demons[molecule].number_density += conjugate_delta

        # Record evolution
        self.evolution_history.append({
            'time': current_time,
            'observable_face': self.observable_face.value,
            'change': front_change,
            'dual_state': self.dual_state.to_dict()
        })

        # Check if auto-switching is enabled
        self.auto_switch_if_needed(current_time)

    def _conjugate_transform_density(self, density_delta: float) -> float:
        """
        Apply conjugate transformation to density change

        Depending on transform type:
        - phase_conjugate: flip sign (increase ↔ decrease)
        - temporal_inverse: reverse change
        - evolution_complement: complementary change
        """
        if self.transform.transform_type == 'phase_conjugate':
            return -density_delta
        elif self.transform.transform_type == 'full_conjugate':
            return -density_delta
        elif self.transform.transform_type == 'evolution_complement':
            # Complementary change (in opposite direction but scaled)
            return -0.5 * density_delta
        else:
            # Default: mirror change
            return density_delta

    def evolve_dual_state(
        self,
        dt: float,
        current_time: float = 0.0
    ):
        """
        Evolve both front and back states together

        This is the coupled evolution: changes happen simultaneously
        on both faces, maintaining their conjugate relationship.

        Args:
            dt: Time step
            current_time: Current time
        """
        # Evolve front state (observable)
        front_s = self.dual_state.front_s
        back_s = self.dual_state.back_s

        # Simple evolution: states drift in S-space
        # Front: moves in +S_t direction (forward time)
        # Back: moves in -S_t direction (backward time, for temporal_inverse)

        if self.observable_face == MembraneFace.FRONT:
            # Observable front evolves normally
            new_front_s = SEntropyCoordinates(
                S_k=front_s.S_k + 0.01 * dt * np.random.randn(),
                S_t=front_s.S_t + dt * 0.1,
                S_e=front_s.S_e
            )

            # Hidden back evolves conjugately
            new_back_s = self.transform.apply(new_front_s)
        else:
            # Observable back evolves
            new_back_s = SEntropyCoordinates(
                S_k=back_s.S_k + 0.01 * dt * np.random.randn(),
                S_t=back_s.S_t + dt * 0.1,
                S_e=back_s.S_e
            )

            # Hidden front evolves conjugately
            new_front_s = self.transform.apply(new_back_s)

        # Update dual state
        self.dual_state = DualState(front_s=new_front_s, back_s=new_back_s)
        self.s_state = new_front_s if self.observable_face == MembraneFace.FRONT else new_back_s

        # Check for auto-switching
        self.auto_switch_if_needed(current_time)

    def measure_observable_face(self) -> Dict[str, Any]:
        """
        Measure the currently observable face

        Returns information about the visible face only.
        The hidden face remains inaccessible (Heisenberg-like complementarity).
        """
        observable_demons = self.get_observable_demons()

        measurement = {
            'pixel_id': self.pixel_id,
            'observable_face': self.observable_face.value,
            'position': self.position.tolist(),
            's_state': self.s_state.to_dict(),
            'molecular_composition': {
                mol: demon.number_density
                for mol, demon in observable_demons.items()
            },
            'information_density': self._calculate_information_density(observable_demons),
            'switch_count': self.switch_count
        }

        return measurement

    def _calculate_information_density(self, demons: Dict[str, MolecularDemon]) -> float:
        """Calculate oscillatory information density from molecular demons"""
        total = 0.0
        for demon in demons.values():
            for freq in demon.vibrational_modes:
                total += freq * demon.number_density * 1e-20  # Scaled
        return total

    def probe_hidden_face(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to probe the hidden face (should fail or give limited info)

        In true complementarity, you cannot access both faces simultaneously.
        This method demonstrates the limitation.

        Returns:
            Limited/uncertainty-limited information about hidden face
        """
        logger.warning(
            f"Attempting to probe hidden face while observing {self.observable_face.value}. "
            "This violates complementarity and will return limited information."
        )

        # Can only return highly uncertain information
        hidden_face = MembraneFace.BACK if self.observable_face == MembraneFace.FRONT else MembraneFace.FRONT

        return {
            'pixel_id': self.pixel_id,
            'hidden_face': hidden_face.value,
            'accessible': False,
            'uncertainty': 'infinite',  # Heisenberg-limited
            'note': 'Hidden face is categorically orthogonal to observable face'
        }

    def get_dual_state_summary(self) -> Dict[str, Any]:
        """Get complete summary of dual membrane state"""
        return {
            'pixel_id': self.pixel_id,
            'position': self.position.tolist(),
            'observable_face': self.observable_face.value,
            'transform_type': self.transform.transform_type,
            'switch_count': self.switch_count,
            'switching_frequency': self.switching_frequency,
            'dual_state': self.dual_state.to_dict(),
            'categorical_separation': self.dual_state.categorical_distance(),
            'front_demons': len(self.front_demons),
            'back_demons': len(self.back_demons),
            'evolution_history_length': len(self.evolution_history)
        }


class DualMembraneGrid:
    """
    Grid of dual-membrane pixel demons

    Like viewing a 3D object through a 2D screen: each pixel shows one face,
    but has a conjugate face on the back that you cannot see simultaneously.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        physical_extent: Tuple[float, float],
        transform_type: str = 'phase_conjugate',
        synchronized_switching: bool = True,
        switching_frequency: Optional[float] = None,
        name: str = "dual_membrane_grid"
    ):
        """
        Initialize dual-membrane grid

        Args:
            shape: (ny, nx) grid dimensions
            physical_extent: (ey, ex) physical size
            transform_type: Conjugate transformation type
            synchronized_switching: If True, all pixels switch together
            switching_frequency: Frequency of face switching (Hz)
            name: Grid identifier
        """
        self.shape = shape
        self.physical_extent = physical_extent
        self.transform_type = transform_type
        self.synchronized_switching = synchronized_switching
        self.switching_frequency = switching_frequency
        self.name = name

        # Create grid of dual membrane demons
        self.demons = self._initialize_grid()

        # Global time tracking
        self.current_time = 0.0

        logger.info(
            f"Created DualMembraneGrid '{name}' with shape {shape}, "
            f"synchronized={synchronized_switching}"
        )

    def _initialize_grid(self) -> np.ndarray:
        """Initialize grid of dual-membrane pixel demons"""
        ny, nx = self.shape
        ey, ex = self.physical_extent

        demons = np.empty(self.shape, dtype=object)

        for iy in range(ny):
            for ix in range(nx):
                # Physical position
                x = (ix / nx) * ex
                y = (iy / ny) * ey
                position = np.array([x, y, 0.0])

                # Create dual membrane demon
                demons[iy, ix] = DualMembranePixelDemon(
                    position=position,
                    pixel_id=f"{self.name}_y{iy}_x{ix}",
                    transform_type=self.transform_type,
                    switching_frequency=self.switching_frequency
                )

        return demons

    def initialize_all_atmospheric(
        self,
        temperature_k: float = 288.15,
        pressure_pa: float = 101325.0,
        humidity_fraction: float = 0.5
    ):
        """Initialize atmospheric lattices for all demons (front and back)"""
        flat_demons = self.demons.flatten()
        for demon in flat_demons:
            demon.initialize_atmospheric_lattice(
                temperature_k, pressure_pa, humidity_fraction
            )
        logger.info(
            f"Initialized dual atmospheric lattices for {len(flat_demons)} demons"
        )

    def switch_all_faces(self):
        """Switch all pixels simultaneously (synchronized switching)"""
        flat_demons = self.demons.flatten()
        for demon in flat_demons:
            demon.switch_observable_face(self.current_time)
        logger.info(f"Switched all {len(flat_demons)} pixel faces at t={self.current_time:.3f}s")

    def evolve_grid(self, dt: float):
        """Evolve entire grid by time step dt"""
        flat_demons = self.demons.flatten()
        for demon in flat_demons:
            demon.evolve_dual_state(dt, self.current_time)

        self.current_time += dt

    def measure_observable_grid(self) -> np.ndarray:
        """
        Measure all observable faces (creates 2D image)

        Returns:
            2D array of information density values (the "image")
        """
        ny, nx = self.shape
        image = np.zeros((ny, nx))

        for iy in range(ny):
            for ix in range(nx):
                measurement = self.demons[iy, ix].measure_observable_face()
                image[iy, ix] = measurement['information_density']

        return image

    def get_hidden_grid_estimate(self) -> np.ndarray:
        """
        Attempt to estimate hidden face (will be highly uncertain)

        This demonstrates complementarity: cannot access hidden face
        while observing front face.
        """
        ny, nx = self.shape
        hidden_estimate = np.full((ny, nx), np.nan)  # Undefined/inaccessible

        logger.warning(
            "Attempting to measure hidden grid violates complementarity. "
            "Returning NaN array."
        )

        return hidden_estimate

    def create_carbon_copy_pattern(
        self,
        front_pattern: np.ndarray
    ) -> np.ndarray:
        """
        Create back-face pattern as conjugate transform of front pattern

        This is the "carbon copy" mechanism: apply conjugate transformation
        to create the back image from the front image.

        Args:
            front_pattern: 2D array representing front face pattern

        Returns:
            2D array representing conjugate back face pattern
        """
        ny, nx = self.shape
        back_pattern = np.zeros((ny, nx))

        # Apply transformation pixel-by-pixel
        for iy in range(ny):
            for ix in range(nx):
                # Get front value
                front_val = front_pattern[iy, ix]

                # Apply conjugate transformation
                # (For simplicity, use demon's transform type)
                demon = self.demons[iy, ix]
                if demon.transform.transform_type == 'phase_conjugate':
                    back_val = -front_val
                elif demon.transform.transform_type == 'evolution_complement':
                    back_val = 1.0 - front_val if 0 <= front_val <= 1 else -front_val
                else:
                    back_val = front_val  # Identity

                back_pattern[iy, ix] = back_val

        return back_pattern
