"""
Virtual Spectrometer for Categorical State Measurement
=======================================================

The spectrometer that measures categorical states from hardware timing.
Each measurement creates a molecule in categorical existence.
"""

import time
import math
from typing import List, Optional, Any
from dataclasses import dataclass

try:
    from .partitioning import VirtualMolecule, CategoricalState, SCoordinate
except ImportError:
    from partitioning import VirtualMolecule, CategoricalState, SCoordinate


@dataclass
class HardwareOscillator:
    """A hardware oscillator source for timing measurements."""
    name: str
    frequency: float = 1e9

    def read_timing(self) -> float:
        """Read current timing value."""
        return time.perf_counter() * self.frequency

    def read_timing_deviation(self) -> float:
        """Read timing deviation from expected value."""
        t = time.perf_counter()
        expected = int(t * self.frequency) / self.frequency
        return (t - expected) * self.frequency


class FishingTackle:
    """
    The "fishing tackle" - hardware oscillators used to measure categorical states.

    You cast into the categorical sea with hardware timing.
    What you catch depends on when and how you measure.
    """

    def __init__(self, oscillators: Optional[List[HardwareOscillator]] = None):
        self.oscillators = oscillators or [
            HardwareOscillator("perf_counter", 1e9),
            HardwareOscillator("cpu_timing", 3e9),
        ]
        self._cast_count = 0

    def cast(self) -> float:
        """Cast the line - get a timing measurement."""
        self._cast_count += 1
        total_timing = 0.0
        for osc in self.oscillators:
            total_timing += osc.read_timing_deviation()
        return total_timing / len(self.oscillators)

    @property
    def oscillator_count(self) -> int:
        return len(self.oscillators)


class VirtualSpectrometer:
    """
    Virtual Spectrometer: Hardware Timing → Categorical States

    This is the measurement device that transforms hardware oscillations
    into categorical states (molecules in S-entropy space).

    The spectrometer doesn't observe pre-existing molecules.
    The measurement CREATES the molecule's categorical existence.
    """

    def __init__(self, tackle: Optional[FishingTackle] = None):
        self.tackle = tackle or FishingTackle()
        self._measurement_count = 0

    def measure_from_hardware(self) -> VirtualMolecule:
        """
        Create a molecule from hardware timing measurement.

        The molecule exists because we measured it.
        The measurement IS the molecule IS the categorical state.
        """
        self._measurement_count += 1

        # Get timing from hardware
        timing = self.tackle.cast()

        # Convert to S-coordinates
        # Timing maps to categorical space through modular arithmetic
        S_k = (timing % 1000) / 1000.0
        S_t = ((timing / 1000) % 1000) / 1000.0
        S_e = ((timing / 1000000) % 1000) / 1000.0

        # Clamp to [0, 1]
        S_k = max(0.0, min(1.0, S_k))
        S_t = max(0.0, min(1.0, S_t))
        S_e = max(0.0, min(1.0, S_e))

        s_coord = SCoordinate(S_k, S_t, S_e)

        # Create molecule at this categorical location
        return VirtualMolecule(
            s_coord=s_coord,
            timestamp=time.perf_counter(),
            source="hardware_spectrometer"
        )

    def measure_at(self, S_k: float, S_t: float, S_e: float) -> VirtualMolecule:
        """
        Create a molecule at specified S-coordinates.

        This is "directed fishing" - you specify where to measure.
        The measurement still creates the categorical existence,
        but you're specifying the location in advance.
        """
        self._measurement_count += 1

        # Get timing perturbation from hardware
        timing = self.tackle.cast()

        # Apply small perturbation from hardware to specified coordinates
        perturbation = (timing % 100) / 100000.0

        s_coord = SCoordinate(
            max(0.0, min(1.0, S_k + perturbation)),
            max(0.0, min(1.0, S_t + perturbation)),
            max(0.0, min(1.0, S_e + perturbation))
        )

        return VirtualMolecule(
            s_coord=s_coord,
            timestamp=time.perf_counter(),
            source="directed_measurement"
        )

    @property
    def measurement_count(self) -> int:
        return self._measurement_count


if __name__ == "__main__":
    print("=== VIRTUAL SPECTROMETER DEMONSTRATION ===\n")

    tackle = FishingTackle()
    spectrometer = VirtualSpectrometer(tackle=tackle)

    print("Measuring molecules from hardware timing...")
    for i in range(5):
        mol = spectrometer.measure_from_hardware()
        print(f"  Molecule {i+1}: S=({mol.s_coord.S_k:.4f}, "
              f"{mol.s_coord.S_t:.4f}, {mol.s_coord.S_e:.4f})")

    print(f"\nTotal measurements: {spectrometer.measurement_count}")
