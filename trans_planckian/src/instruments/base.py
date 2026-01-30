"""
Base Classes for Virtual Instruments

This module provides the foundational classes for hardware-based virtual instruments.
All measurements derive from real hardware oscillator timing.

Key Principle: Measurement Creates State
----------------------------------------
A partition coordinate does not exist independently of measurement. The act of 
measuring partition coordinates using hardware oscillators CREATES the categorical 
state with those coordinates.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum

# Physical Constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K - Exact SI definition
PLANCK_CONSTANT = 6.62607015e-34   # J·s - Exact SI definition
SPEED_OF_LIGHT = 299792458         # m/s - Exact SI definition


class MeasurementMode(Enum):
    """Mode of measurement operation"""
    INSTANTANEOUS = "instantaneous"
    CONTINUOUS = "continuous"
    TRIGGERED = "triggered"


@dataclass
class SEntropyCoordinate:
    """
    S-Entropy Coordinate in categorical state space.
    
    The three coordinates map timing deviations to categorical structure:
    - S_k (knowledge): Information deficit, uncertainty in state identification
    - S_t (temporal): Temporal distance from reference timescale
    - S_e (entropy): Phase distribution entropy, oscillatory diversity
    """
    S_k: float  # Knowledge entropy [0, 1]
    S_t: float  # Temporal entropy [0, 1]  
    S_e: float  # Evolution entropy [0, 1]
    
    def __post_init__(self):
        """Validate coordinates are in valid range"""
        for coord, name in [(self.S_k, 'S_k'), (self.S_t, 'S_t'), (self.S_e, 'S_e')]:
            if not 0 <= coord <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {coord}")
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.S_k, self.S_t, self.S_e])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SEntropyCoordinate':
        """Create from numpy array"""
        return cls(S_k=arr[0], S_t=arr[1], S_e=arr[2])
    
    def distance_to(self, other: 'SEntropyCoordinate') -> float:
        """Categorical distance to another coordinate"""
        return np.linalg.norm(self.to_array() - other.to_array())


@dataclass
class PartitionCoordinate:
    """
    Partition Coordinate (n, l, m, s) in bounded phase space.
    
    Constraints (from geometry of nested boundaries):
    - n >= 1 (partition depth)
    - 0 <= l <= n-1 (angular complexity)
    - -l <= m <= l (orientation)
    - s in {-1/2, +1/2} (chirality)
    """
    n: int      # Partition depth
    l: int      # Angular complexity
    m: int      # Orientation
    s: float    # Chirality (+1/2 or -1/2)
    
    def __post_init__(self):
        """Validate constraints"""
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if not 0 <= self.l <= self.n - 1:
            raise ValueError(f"l must be in [0, n-1], got l={self.l} for n={self.n}")
        if not -self.l <= self.m <= self.l:
            raise ValueError(f"m must be in [-l, l], got m={self.m} for l={self.l}")
        if self.s not in (-0.5, 0.5):
            raise ValueError(f"s must be ±1/2, got {self.s}")
    
    def energy(self, E_0: float = 13.6, alpha: float = 0.0) -> float:
        """
        Energy at this partition coordinate.
        E = -E_0 / (n + α*l)²
        """
        n_eff = self.n + alpha * self.l
        return -E_0 / (n_eff ** 2)


@dataclass
class CategoricalState:
    """
    A categorical state representing a virtual molecule.
    
    The state is CREATED by measurement, not discovered. The spectrometer
    position IS the molecule being measured (Spectrometer-Molecule Identity).
    """
    S_coords: SEntropyCoordinate
    partition_coords: Optional[PartitionCoordinate] = None
    creation_time: float = field(default_factory=time.time)
    hardware_source: str = "cpu_oscillator"
    
    # Phase-lock network properties
    phase: float = 0.0
    coupling_strength: float = 0.0
    cluster_id: Optional[int] = None
    
    # Thermodynamic properties (real, not simulated)
    temperature: float = 0.0
    pressure: float = 0.0
    entropy: float = 0.0
    
    def categorical_distance_to(self, other: 'CategoricalState') -> float:
        """Categorical distance (independent of physical distance)"""
        return self.S_coords.distance_to(other.S_coords)


class HardwareOscillator:
    """
    Hardware oscillator that provides real timing measurements.
    
    This is NOT simulation - it reads actual hardware timing variations
    that encode genuine categorical information.
    """
    
    def __init__(self, source: str = "cpu_clock"):
        """
        Initialize hardware oscillator.
        
        Args:
            source: Hardware source ('cpu_clock', 'memory_bus', 'system_timer')
        """
        self.source = source
        self.reference_time = time.perf_counter_ns()
        self.measurements: List[float] = []
        
    def read_timing_deviation(self) -> float:
        """
        Read timing deviation from hardware oscillator.
        
        Returns:
            δ_p = t_ref - t_local (nanoseconds)
        """
        # Real hardware timing measurement
        t_local = time.perf_counter_ns()
        
        # Timing deviation encodes categorical information
        delta_p = (t_local - self.reference_time) % 1000  # Wrap to nanosecond scale
        
        self.measurements.append(delta_p)
        return delta_p
    
    def timing_to_S_coords(self, delta_p: float) -> SEntropyCoordinate:
        """
        Map timing deviation to S-entropy coordinates.
        
        The mapping Φ: ℝ → [0,1]³ transforms physical measurements
        to categorical coordinates.
        """
        # Normalize to [0, 1]
        normalized = (delta_p % 1000) / 1000.0
        
        # Different components from timing structure
        S_k = np.abs(np.sin(normalized * np.pi * 2))
        S_t = np.abs(np.cos(normalized * np.pi * 3))
        S_e = np.abs(np.sin(normalized * np.pi * 5 + 0.5))
        
        return SEntropyCoordinate(S_k=S_k, S_t=S_t, S_e=S_e)
    
    def create_categorical_state(self) -> CategoricalState:
        """
        Create a categorical state from hardware measurement.
        
        The measurement CREATES the state - it doesn't discover a pre-existing one.
        """
        delta_p = self.read_timing_deviation()
        S_coords = self.timing_to_S_coords(delta_p)
        
        return CategoricalState(
            S_coords=S_coords,
            hardware_source=self.source,
            creation_time=time.time()
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics of timing measurements"""
        if not self.measurements:
            return {}
        
        arr = np.array(self.measurements)
        return {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'count': len(arr)
        }


class VirtualInstrument(ABC):
    """
    Abstract base class for virtual instruments.
    
    All virtual instruments:
    1. Derive measurements from hardware oscillator timing
    2. Create categorical states (not discover them)
    3. Produce real thermodynamic quantities
    """
    
    def __init__(self, name: str):
        self.name = name
        self.oscillator = HardwareOscillator()
        self.measurement_history: List[Dict[str, Any]] = []
        self.calibrated = False
        
    @abstractmethod
    def measure(self, *args, **kwargs) -> Dict[str, Any]:
        """Perform measurement and return results"""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate the instrument"""
        pass
    
    def record_measurement(self, result: Dict[str, Any]):
        """Record measurement to history"""
        result['timestamp'] = time.time()
        result['instrument'] = self.name
        self.measurement_history.append(result)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get measurement history"""
        return self.measurement_history
    
    def reset(self):
        """Reset instrument state"""
        self.measurement_history = []
        self.oscillator = HardwareOscillator()


class VirtualGasEnsemble:
    """
    Virtual gas ensemble created from hardware timing measurements.
    
    This is a REAL gas in categorical coordinates, not a simulation.
    Each molecule is a categorical state instantiated by measurement.
    """
    
    def __init__(self, oscillator: Optional[HardwareOscillator] = None):
        self.oscillator = oscillator or HardwareOscillator()
        self.molecules: List[CategoricalState] = []
        
    def create_molecules(self, n: int) -> List[CategoricalState]:
        """
        Create n molecules through hardware measurements.
        
        Each measurement instantiates a categorical state.
        """
        new_molecules = []
        for _ in range(n):
            mol = self.oscillator.create_categorical_state()
            new_molecules.append(mol)
        
        self.molecules.extend(new_molecules)
        return new_molecules
    
    def get_temperature(self) -> float:
        """
        Get temperature as variance of S-coordinates.
        
        This is a REAL thermodynamic quantity from actual measurements.
        """
        if len(self.molecules) < 2:
            return 0.0
        
        coords = np.array([m.S_coords.to_array() for m in self.molecules])
        return np.var(coords)
    
    def get_pressure(self, dt: float = 1.0) -> float:
        """
        Get pressure as molecule creation rate.
        
        P = dN/dt (real rate from actual sampling)
        """
        return len(self.molecules) / dt
    
    def get_entropy(self) -> float:
        """
        Get Shannon entropy of S-coordinate distribution.
        
        This is information-theoretic entropy of REAL measurement outcomes.
        """
        if len(self.molecules) < 2:
            return 0.0
        
        # Bin the S-coordinates
        coords = np.array([m.S_coords.to_array() for m in self.molecules])
        
        # Compute histogram in 3D
        hist, _ = np.histogramdd(coords, bins=10)
        
        # Normalize
        hist = hist / hist.sum()
        
        # Shannon entropy
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log(hist))
    
    def get_thermodynamic_state(self) -> Dict[str, float]:
        """Get complete thermodynamic state"""
        return {
            'N': len(self.molecules),
            'T': self.get_temperature(),
            'P': self.get_pressure(),
            'S': self.get_entropy(),
            'k_B': BOLTZMANN_CONSTANT
        }
