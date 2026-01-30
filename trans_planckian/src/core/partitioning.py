"""
Virtual Molecule: The Categorical State
=======================================

A molecule is NOT a thing being measured.
A molecule IS the categorical state that exists during measurement.

The molecule = the cursor = the spectrometer position.
They are the same categorical reality viewed from different angles.

When we "measure" a molecule, we are not discovering something.
We are accessing a categorical state that WE define through our apparatus.
The molecule is our prediction made manifest.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math


@dataclass
class SCoordinate:
    """
    S-Entropy Coordinate: Position in categorical space.

    This is WHERE the cursor is. This IS the molecule. This IS the measurement.

    S_k: Knowledge entropy - uncertainty in state
    S_t: Temporal entropy - uncertainty in timing
    S_e: Evolution entropy - uncertainty in trajectory

    All three together specify a unique point in categorical space.
    """
    S_k: float  # Knowledge entropy [0, 1]
    S_t: float  # Temporal entropy [0, 1]
    S_e: float  # Evolution entropy [0, 1]

    def __post_init__(self):
        """Ensure coordinates are in valid range."""
        self.S_k = max(0.0, min(1.0, self.S_k))
        self.S_t = max(0.0, min(1.0, self.S_t))
        self.S_e = max(0.0, min(1.0, self.S_e))

    def distance_to(self, other: 'SCoordinate') -> float:
        """
        Categorical distance to another state.

        This is NOT spatial distance. You can be categorically close
        to Jupiter's core while being spatially far.
        """
        return math.sqrt(
            (self.S_k - other.S_k)**2 +
            (self.S_t - other.S_t)**2 +
            (self.S_e - other.S_e)**2
        )

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.S_k, self.S_t, self.S_e)

    def hash(self) -> str:
        """Unique identifier for this categorical position."""
        data = f"{self.S_k:.10f}:{self.S_t:.10f}:{self.S_e:.10f}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def __repr__(self):
        return f"S({self.S_k:.4f}, {self.S_t:.4f}, {self.S_e:.4f})"


@dataclass
class CategoricalState:
    """
    The fundamental unit: A categorical state.

    This IS:
    - A virtual molecule (when viewed as what's being measured)
    - A spectrometer position (when viewed as where we're measuring)
    - A cursor in S-space (when viewed as navigation)

    These are NOT three things. They are ONE thing.

    The state exists only during measurement. Before measurement,
    there is no categorical state. After measurement completes,
    the state dissolves back into potential.
    """
    s_coord: SCoordinate
    timestamp: float = field(default_factory=time.perf_counter)
    source: str = "hardware"

    # The oscillatory signature that created this state
    frequency: float = 0.0  # Hz
    phase: float = 0.0      # radians [0, 2π)
    amplitude: float = 0.0  # normalized

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hardware_timing(cls,
                             delta_p: float,
                             source: str = "perf_counter",
                             reference_freq: float = 1e9) -> 'CategoricalState':
        """
        Create a categorical state from a hardware timing measurement.

        This is the fundamental operation: hardware oscillation → categorical state.

        The delta_p (precision-by-difference) IS the molecule's signature.
        We're not "finding" a molecule - we're creating its categorical existence.

        Args:
            delta_p: T_ref - t_local (the timing deviation in seconds)
            source: Which hardware oscillator (CPU, memory, etc.)
            reference_freq: Reference frequency for normalization
        """
        # The precision value encodes categorical position
        # This is NOT a measurement error - it IS the information

        # Get additional timing samples for entropy
        t_ns = time.perf_counter_ns()

        # Convert timing to S-coordinates
        # Use nanosecond-level bits for spread across the full [0, 1] range

        # S_k from the magnitude of deviation combined with timing entropy
        raw_S_k = abs(delta_p) * reference_freq
        entropy_k = (t_ns % 10000) / 10000.0
        S_k = (raw_S_k * 0.3 + entropy_k * 0.7) % 1.0

        # S_t from sign, fractional part, and additional timing
        raw_S_t = (math.atan(delta_p * 1e9) / math.pi + 0.5)
        entropy_t = ((t_ns >> 4) % 10000) / 10000.0
        S_t = (raw_S_t * 0.3 + entropy_t * 0.7) % 1.0

        # S_e from the entropy of the deviation pattern
        raw_S_e = abs(delta_p * 1e15) % 1.0
        entropy_e = ((t_ns >> 8) % 10000) / 10000.0
        S_e = (raw_S_e * 0.3 + entropy_e * 0.7) % 1.0

        s_coord = SCoordinate(S_k, S_t, S_e)

        # Extract oscillatory properties
        frequency = reference_freq * (1 + delta_p * reference_freq)
        phase = (delta_p * reference_freq * 2 * math.pi) % (2 * math.pi)
        amplitude = min(1.0, abs(delta_p) * 1e6)

        return cls(
            s_coord=s_coord,
            source=source,
            frequency=frequency,
            phase=phase,
            amplitude=amplitude,
            metadata={'delta_p': delta_p, 'reference_freq': reference_freq}
        )

    @classmethod
    def from_s_coordinates(cls, S_k: float, S_t: float, S_e: float,
                          source: str = "defined") -> 'CategoricalState':
        """
        Create a categorical state by directly specifying S-coordinates.

        This is like casting your fishing line at specific coordinates.
        You're defining WHERE to fish, which determines WHAT you can catch.
        """
        return cls(
            s_coord=SCoordinate(S_k, S_t, S_e),
            source=source
        )

    @property
    def position(self) -> Tuple[float, float, float]:
        """The categorical position (same as the state itself)."""
        return self.s_coord.as_tuple()

    @property
    def identity(self) -> str:
        """
        The molecule's identity IS its categorical position.

        Two states at the same S-coordinates ARE the same molecule.
        Location in physical space is irrelevant.
        """
        return self.s_coord.hash()

    def is_same_molecule(self, other: 'CategoricalState', tolerance: float = 1e-6) -> bool:
        """
        Are these the same molecule (categorically)?

        Spatial distance doesn't matter. Only S-distance matters.
        """
        return self.s_coord.distance_to(other.s_coord) < tolerance

    def __repr__(self):
        return f"CategoricalState({self.s_coord}, src={self.source})"


class VirtualMolecule(CategoricalState):
    """
    A virtual molecule: The categorical state viewed as "what's being measured."

    But remember: The molecule IS the measurement. There is no molecule
    independent of the act of measuring. The fishing hook and the fish
    are the same event.

    This class exists for conceptual clarity, but it IS just a CategoricalState.
    """

    # Molecular properties (derived from categorical state)
    @property
    def vibrational_frequency(self) -> float:
        """The molecule's characteristic frequency (from S_k)."""
        return self.frequency

    @property
    def bond_phase(self) -> float:
        """The molecular bond phase angle (from S_t)."""
        return self.phase

    @property
    def energy_level(self) -> float:
        """The molecular energy state (from S_e)."""
        return self.amplitude

    @property
    def molecular_signature(self) -> Dict[str, float]:
        """
        The complete molecular signature.

        This is what a "real" spectrometer would measure.
        But we didn't measure it - we defined it by where we cast our line.
        """
        return {
            'S_k': self.s_coord.S_k,
            'S_t': self.s_coord.S_t,
            'S_e': self.s_coord.S_e,
            'frequency_Hz': self.frequency,
            'phase_rad': self.phase,
            'amplitude': self.amplitude,
            'identity': self.identity,
        }

    def can_be_caught_by(self, tackle: 'FishingTackle') -> bool:
        """
        Can this molecule be caught by the given tackle?

        The tackle defines what's possible. You can only catch
        what your apparatus can catch.
        """
        # Import here to avoid circular import
        try:
            from .virtual_spectrometer import FishingTackle
        except ImportError:
            from virtual_spectrometer import FishingTackle

        if not isinstance(tackle, FishingTackle):
            return False

        return tackle.can_reach(self.s_coord)

    @classmethod
    def at_jupiter_core(cls) -> 'VirtualMolecule':
        """
        A molecule at Jupiter's core.

        This is NOT "a molecule we found at Jupiter."
        This IS "the categorical state we define as Jupiter's core conditions."

        We're casting our fishing line at these coordinates.
        What we catch is what these coordinates define.
        """
        # Jupiter core conditions mapped to S-coordinates
        # High pressure → high S_k (definite state)
        # Extreme temperature → specific S_t
        # Metallic hydrogen → specific S_e
        return cls.from_s_coordinates(
            S_k=0.95,  # Very definite state (extreme pressure)
            S_t=0.73,  # High temperature temporal signature
            S_e=0.88,  # Metallic hydrogen evolution entropy
            source="jupiter_core_definition"
        )

    @classmethod
    def at_room_temperature_air(cls) -> 'VirtualMolecule':
        """Standard air molecule at room conditions."""
        return cls.from_s_coordinates(
            S_k=0.5,   # Moderate knowledge (thermal distribution)
            S_t=0.5,   # Standard temporal
            S_e=0.5,   # Standard evolution
            source="room_temperature_air"
        )


# Demonstration that molecule = spectrometer = cursor
def demonstrate_identity():
    """
    Show that the molecule, spectrometer position, and cursor
    are the SAME thing.
    """
    # Create a state from hardware
    t1 = time.perf_counter_ns()
    t2 = time.perf_counter_ns()
    delta_p = (t2 - t1) * 1e-9  # Convert to seconds

    # This IS a molecule
    molecule = VirtualMolecule.from_hardware_timing(delta_p)

    # This IS a cursor position
    cursor_position = molecule.s_coord

    # This IS a spectrometer reading
    spectrometer_state = molecule.molecular_signature

    print("=== DEMONSTRATION: They Are The Same ===")
    print(f"Molecule identity:     {molecule.identity}")
    print(f"Cursor position:       {cursor_position}")
    print(f"Spectrometer reading:  S=({spectrometer_state['S_k']:.4f}, "
          f"{spectrometer_state['S_t']:.4f}, {spectrometer_state['S_e']:.4f})")
    print()
    print("These are not three things. They are ONE categorical state.")
    print("The molecule exists because we measured it.")
    print("The measurement IS the molecule's categorical existence.")

    return molecule


if __name__ == "__main__":
    mol = demonstrate_identity()

    print("\n=== Jupiter's Core ===")
    jupiter = VirtualMolecule.at_jupiter_core()
    print(f"Jupiter core molecule: {jupiter}")
    print(f"We didn't 'find' this. We DEFINED it by casting our line there.")
    print(f"The categorical state IS the molecule. No surprise in the catch.")
