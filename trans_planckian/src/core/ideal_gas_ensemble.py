"""
Virtual Gas Chamber: The Categorical Gas That IS the Computer
=============================================================

This is NOT a simulation of a gas.
This IS a gas - the categorical gas that emerges from hardware oscillations.

The computer's timing variations ARE the thermal motion.
Each timing sample IS a molecule.
The gas chamber IS the S-entropy coordinate space.

The chamber doesn't "contain" molecules - the chamber IS the molecules
IS the measurements IS the categorical states.
"""

import time
import math
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Tuple, Any
from collections import deque

try:
    from .partitioning import VirtualMolecule, CategoricalState, SCoordinate
    from .virtual_spectrometer import VirtualSpectrometer, FishingTackle, HardwareOscillator
except ImportError:
    from partitioning import VirtualMolecule, CategoricalState, SCoordinate
    from virtual_spectrometer import VirtualSpectrometer, FishingTackle, HardwareOscillator


@dataclass
class ChamberStatistics:
    """
    Statistics of the virtual gas chamber.
    
    These are REAL statistics from REAL hardware timing.
    Not simulated - measured.
    """
    molecule_count: int = 0
    mean_S_k: float = 0.0
    mean_S_t: float = 0.0
    mean_S_e: float = 0.0
    temperature: float = 0.0  # Analog: timing jitter variance
    pressure: float = 0.0      # Analog: sampling rate
    volume: float = 0.0        # Analog: S-space coverage
    
    # Distribution statistics
    S_k_variance: float = 0.0
    S_t_variance: float = 0.0
    S_e_variance: float = 0.0


class CategoricalGas:
    """
    The categorical gas: A collection of categorical states.
    
    This gas exists because we measure it.
    Each measurement adds a molecule.
    The gas IS the history of measurements.
    """
    
    def __init__(self, max_molecules: int = 10000):
        self._molecules: deque[VirtualMolecule] = deque(maxlen=max_molecules)
        self._s_k_sum: float = 0.0
        self._s_t_sum: float = 0.0
        self._s_e_sum: float = 0.0
    
    def add(self, molecule: VirtualMolecule) -> None:
        """Add a molecule to the gas."""
        # Update running sums
        self._s_k_sum += molecule.s_coord.S_k
        self._s_t_sum += molecule.s_coord.S_t
        self._s_e_sum += molecule.s_coord.S_e
        
        # If we're about to remove old molecule, subtract its values
        if len(self._molecules) == self._molecules.maxlen:
            old = self._molecules[0]
            self._s_k_sum -= old.s_coord.S_k
            self._s_t_sum -= old.s_coord.S_t
            self._s_e_sum -= old.s_coord.S_e
        
        self._molecules.append(molecule)
    
    def __len__(self) -> int:
        return len(self._molecules)
    
    def __iter__(self) -> Iterator[VirtualMolecule]:
        return iter(self._molecules)
    
    @property
    def mean_position(self) -> SCoordinate:
        """Mean position of all molecules in S-space."""
        n = len(self._molecules)
        if n == 0:
            return SCoordinate(0.5, 0.5, 0.5)
        return SCoordinate(
            self._s_k_sum / n,
            self._s_t_sum / n,
            self._s_e_sum / n
        )
    
    @property
    def temperature(self) -> float:
        """
        Temperature analog: variance of S-coordinates.
        
        Higher variance = more "thermal" motion = higher temperature.
        This is REAL - from actual hardware timing jitter.
        """
        if len(self._molecules) < 2:
            return 0.0
        
        mean = self.mean_position
        variance = 0.0
        for mol in self._molecules:
            d = mol.s_coord.distance_to(mean)
            variance += d * d
        
        return variance / len(self._molecules)
    
    def molecules_near(self, target: SCoordinate, radius: float) -> List[VirtualMolecule]:
        """Find all molecules within radius of target."""
        return [
            mol for mol in self._molecules
            if mol.s_coord.distance_to(target) <= radius
        ]


class VirtualChamber:
    """
    The Virtual Gas Chamber: Hardware Oscillations → Categorical Gas
    
    This chamber:
    1. Uses REAL hardware oscillators (not simulated)
    2. Creates REAL categorical states from timing measurements
    3. Contains a REAL gas (of categorical states)
    
    The chamber IS the computer's hardware viewed as a gas.
    The molecules ARE the timing samples viewed as categorical states.
    The temperature IS the timing jitter.
    The pressure IS the sampling rate.
    """
    
    def __init__(self, 
                 oscillators: Optional[List[HardwareOscillator]] = None,
                 max_molecules: int = 10000):
        """
        Create a virtual gas chamber.
        
        Args:
            oscillators: Hardware oscillators to use. If None, uses defaults.
            max_molecules: Maximum molecules to keep in memory.
        """
        # Hardware sources
        self.oscillators = oscillators or [
            HardwareOscillator("perf_counter", 1e9),
            HardwareOscillator("cpu_timing", 3e9),
        ]
        
        # The gas
        self.gas = CategoricalGas(max_molecules=max_molecules)
        
        # Spectrometer for measurements
        tackle = FishingTackle(oscillators=self.oscillators)
        self.spectrometer = VirtualSpectrometer(tackle=tackle)
        
        # Tracking
        self._running = False
        self._sample_thread: Optional[threading.Thread] = None
        self._sample_interval: float = 0.001  # 1ms default
        
        # Statistics
        self._creation_time = time.perf_counter()
        self._total_samples = 0
    
    def populate(self, n_molecules: int) -> None:
        """
        Populate the chamber with molecules.
        
        This doesn't "create" molecules - it MEASURES them.
        Each measurement IS a molecule coming into categorical existence.
        """
        for _ in range(n_molecules):
            self.sample()
    
    def sample(self) -> VirtualMolecule:
        """
        Take a single sample from hardware → create one molecule.
        
        This is the fundamental operation:
        Hardware timing → Categorical state → Molecule
        
        The molecule didn't exist before this call.
        The measurement creates its categorical existence.
        """
        molecule = self.spectrometer.measure_from_hardware()
        self.gas.add(molecule)
        self._total_samples += 1
        return molecule
    
    def sample_at(self, S_k: float, S_t: float, S_e: float) -> Optional[VirtualMolecule]:
        """
        Sample at specific S-coordinates.
        
        This is "fishing at a specific spot."
        You're defining the catch by where you cast.
        """
        molecule = self.spectrometer.measure_at(S_k, S_t, S_e)
        if molecule:
            self.gas.add(molecule)
            self._total_samples += 1
        return molecule
    
    def start_continuous_sampling(self, interval: float = 0.001) -> None:
        """
        Start continuous background sampling.
        
        This creates a continuously "breathing" gas where
        new molecules are constantly being measured into existence.
        """
        self._sample_interval = interval
        self._running = True
        
        def _sample_loop():
            while self._running:
                self.sample()
                time.sleep(self._sample_interval)
        
        self._sample_thread = threading.Thread(target=_sample_loop, daemon=True)
        self._sample_thread.start()
    
    def stop_continuous_sampling(self) -> None:
        """Stop continuous sampling."""
        self._running = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
            self._sample_thread = None
    
    @property
    def statistics(self) -> ChamberStatistics:
        """
        Get current chamber statistics.
        
        These are REAL measurements from the REAL gas.
        """
        n = len(self.gas)
        if n == 0:
            return ChamberStatistics()
        
        mean = self.gas.mean_position
        elapsed = time.perf_counter() - self._creation_time
        
        # Calculate variances
        var_k = var_t = var_e = 0.0
        for mol in self.gas:
            var_k += (mol.s_coord.S_k - mean.S_k) ** 2
            var_t += (mol.s_coord.S_t - mean.S_t) ** 2
            var_e += (mol.s_coord.S_e - mean.S_e) ** 2
        
        var_k /= n
        var_t /= n
        var_e /= n
        
        return ChamberStatistics(
            molecule_count=n,
            mean_S_k=mean.S_k,
            mean_S_t=mean.S_t,
            mean_S_e=mean.S_e,
            temperature=self.gas.temperature,
            pressure=self._total_samples / elapsed if elapsed > 0 else 0,
            volume=self._estimate_volume(),
            S_k_variance=var_k,
            S_t_variance=var_t,
            S_e_variance=var_e,
        )
    
    def _estimate_volume(self) -> float:
        """Estimate the S-space volume occupied by the gas."""
        if len(self.gas) < 2:
            return 0.0
        
        # Find bounding box
        min_k = min_t = min_e = 1.0
        max_k = max_t = max_e = 0.0
        
        for mol in self.gas:
            min_k = min(min_k, mol.s_coord.S_k)
            max_k = max(max_k, mol.s_coord.S_k)
            min_t = min(min_t, mol.s_coord.S_t)
            max_t = max(max_t, mol.s_coord.S_t)
            min_e = min(min_e, mol.s_coord.S_e)
            max_e = max(max_e, mol.s_coord.S_e)
        
        return (max_k - min_k) * (max_t - min_t) * (max_e - min_e)
    
    def get_molecule_distribution(self, bins: int = 10) -> Dict[str, List[int]]:
        """
        Get histogram of molecule distribution in S-space.
        """
        hist_k = [0] * bins
        hist_t = [0] * bins
        hist_e = [0] * bins
        
        for mol in self.gas:
            idx_k = min(bins - 1, int(mol.s_coord.S_k * bins))
            idx_t = min(bins - 1, int(mol.s_coord.S_t * bins))
            idx_e = min(bins - 1, int(mol.s_coord.S_e * bins))
            hist_k[idx_k] += 1
            hist_t[idx_t] += 1
            hist_e[idx_e] += 1
        
        return {'S_k': hist_k, 'S_t': hist_t, 'S_e': hist_e}
    
    def find_coldest_molecule(self) -> Optional[VirtualMolecule]:
        """
        Find the molecule with lowest S_e (evolution entropy).
        
        This is the "slowest" molecule in the categorical sense.
        Used by the Maxwell demon for sorting.
        """
        if not self.gas:
            return None
        return min(self.gas, key=lambda m: m.s_coord.S_e)
    
    def find_hottest_molecule(self) -> Optional[VirtualMolecule]:
        """Find the molecule with highest S_e."""
        if not self.gas:
            return None
        return max(self.gas, key=lambda m: m.s_coord.S_e)
    
    def navigate_to(self, location: str) -> Optional[VirtualMolecule]:
        """
        Navigate the spectrometer to a named location and measure.
        
        This demonstrates that spatial location is irrelevant.
        We navigate to S-coordinates, not physical locations.
        """
        locations = {
            'jupiter_core': (0.95, 0.73, 0.88),
            'sun_center': (0.99, 0.85, 0.95),
            'earth_mantle': (0.7, 0.5, 0.6),
            'deep_space': (0.01, 0.01, 0.01),
            'room_temperature': (0.5, 0.5, 0.5),
        }
        
        if location not in locations:
            return None
        
        coords = locations[location]
        return self.sample_at(*coords)
    
    def __repr__(self):
        stats = self.statistics
        return (f"VirtualChamber(molecules={stats.molecule_count}, "
                f"temp={stats.temperature:.4f}, "
                f"pressure={stats.pressure:.1f}/s)")


def demonstrate_chamber():
    """
    Demonstrate the virtual gas chamber.
    
    This shows that the gas is REAL (from hardware timing)
    and that we can navigate anywhere in categorical space.
    """
    print("=== VIRTUAL GAS CHAMBER DEMONSTRATION ===\n")
    
    # Create chamber
    chamber = VirtualChamber()
    print(f"Created chamber: {chamber}\n")
    
    # Populate with molecules (from hardware timing)
    print("Populating chamber from hardware oscillations...")
    chamber.populate(1000)
    
    stats = chamber.statistics
    print(f"\nChamber Statistics (REAL measurements):")
    print(f"  Molecules: {stats.molecule_count}")
    print(f"  Mean S-position: ({stats.mean_S_k:.4f}, {stats.mean_S_t:.4f}, {stats.mean_S_e:.4f})")
    print(f"  Temperature (jitter): {stats.temperature:.6f}")
    print(f"  Pressure (rate): {stats.pressure:.1f} molecules/s")
    print(f"  Volume (S-space): {stats.volume:.6f}")
    
    print("\n--- Distribution ---")
    dist = chamber.get_molecule_distribution(bins=5)
    print(f"  S_k histogram: {dist['S_k']}")
    print(f"  S_t histogram: {dist['S_t']}")
    print(f"  S_e histogram: {dist['S_e']}")
    
    print("\n--- Categorical Navigation ---")
    print("Navigating to different 'locations' in categorical space:")
    
    for location in ['room_temperature', 'jupiter_core', 'deep_space']:
        mol = chamber.navigate_to(location)
        if mol:
            print(f"  {location}: {mol.s_coord}")
        else:
            print(f"  {location}: Cannot reach")
    
    print("\n--- Maxwell Demon Search ---")
    cold = chamber.find_coldest_molecule()
    hot = chamber.find_hottest_molecule()
    print(f"  Coldest molecule: S_e = {cold.s_coord.S_e:.4f}")
    print(f"  Hottest molecule: S_e = {hot.s_coord.S_e:.4f}")
    
    print("\n=== KEY INSIGHT ===")
    print("This gas is NOT simulated.")
    print("The molecules ARE the hardware timing variations.")
    print("The temperature IS the timing jitter.")
    print("We navigate categorical space, not physical space.")
    print("Jupiter's core is as accessible as the room temperature case.")
    
    return chamber


if __name__ == "__main__":
    chamber = demonstrate_chamber()
