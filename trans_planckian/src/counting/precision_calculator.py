"""
Precision-by-Difference Calculator

The core mathematical engine that computes precision-by-difference values
and uses them to navigate the S-entropy hierarchy.

Key insight from Sango Rine Shumba:
    ΔP_i(k) = T_ref(k) - t_i(k)

This difference is NOT an error - it IS the address.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import time

from .s_entropy_address import SCoordinate, SEntropyAddress
from .hardware_oscillator import HardwareOscillatorCapture


@dataclass 
class PrecisionWindow:
    """
    A temporal coherence window for precision-by-difference.
    
    From the paper:
    W_i(k) = [T_ref(k) + min_j(ΔP_j), T_ref(k) + max_j(ΔP_j)]
    """
    start: float
    end: float
    reference_time: float
    precision_values: List[float] = field(default_factory=list)
    
    @property
    def width(self) -> float:
        return self.end - self.start
    
    @property
    def center(self) -> float:
        return (self.start + self.end) / 2
    
    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end


class PrecisionByDifferenceCalculator:
    """
    Computes precision-by-difference values and maps them to S-entropy addresses.
    
    The calculator maintains a reference clock and computes differences from
    actual hardware timing. These differences accumulate into trajectories
    that serve as addresses in the categorical hierarchy.
    """
    
    def __init__(self, oscillator: Optional[HardwareOscillatorCapture] = None):
        """
        Initialize the precision calculator.
        
        Args:
            oscillator: Hardware oscillator capture instance
        """
        self.oscillator = oscillator or HardwareOscillatorCapture()
        
        # Reference system
        self._reference_epoch = time.perf_counter()
        self._reference_ns = time.perf_counter_ns()
        
        # Active addresses (trajectories in progress)
        self._active_addresses: Dict[str, SEntropyAddress] = {}
        
        # Precision history
        self._precision_history: deque[float] = deque(maxlen=10000)
        
        # Statistics
        self._total_calculations = 0
        self._mean_precision = 0.0
        self._variance_precision = 0.0
        
    def calculate_precision_difference(self) -> float:
        """
        Calculate a single precision-by-difference value.
        
        This is ΔP = T_ref - t_local
        
        Returns:
            The precision-by-difference value
        """
        # Get current times
        current_perf = time.perf_counter()
        current_ns = time.perf_counter_ns()
        
        # Expected elapsed (from reference)
        expected_elapsed = current_perf - self._reference_epoch
        
        # Actual elapsed (from nanosecond counter)
        actual_elapsed = (current_ns - self._reference_ns) / 1e9
        
        # Precision-by-difference
        delta_p = expected_elapsed - actual_elapsed
        
        # Update statistics
        self._total_calculations += 1
        self._precision_history.append(delta_p)
        
        # Online mean and variance update
        old_mean = self._mean_precision
        self._mean_precision += (delta_p - old_mean) / self._total_calculations
        self._variance_precision += (delta_p - old_mean) * (delta_p - self._mean_precision)
        
        return delta_p
    
    def get_precision_window(self, n_samples: int = 10) -> PrecisionWindow:
        """
        Calculate a precision window from multiple samples.
        
        Returns:
            A PrecisionWindow defining the coherence region
        """
        reference_time = time.perf_counter() - self._reference_epoch
        
        # Collect precision samples
        samples = []
        for _ in range(n_samples):
            samples.append(self.calculate_precision_difference())
            
        # Build window
        window = PrecisionWindow(
            start=reference_time + min(samples),
            end=reference_time + max(samples),
            reference_time=reference_time,
            precision_values=samples
        )
        
        return window
    
    def create_address(self, key: str) -> SEntropyAddress:
        """
        Create a new S-entropy address and begin tracking its trajectory.
        
        Args:
            key: Unique identifier for this address
            
        Returns:
            A new SEntropyAddress
        """
        address = SEntropyAddress()
        self._active_addresses[key] = address
        
        # Initialize with first position
        self.update_address(key)
        
        return address
    
    def update_address(self, key: str) -> Optional[SCoordinate]:
        """
        Update an address's trajectory with new precision-by-difference data.
        
        Args:
            key: The address key
            
        Returns:
            The new S-coordinate, or None if address not found
        """
        address = self._active_addresses.get(key)
        if not address:
            return None
        
        # Calculate precision difference
        delta_p = self.calculate_precision_difference()
        
        # Get hardware signature
        signature = self.oscillator.get_precision_signature(n_samples=5)
        
        # Convert to S-coordinate
        coordinate = self.oscillator.signature_to_scoordinate(signature)
        
        # Record in trajectory
        address.record(coordinate, delta_p)
        
        return coordinate
    
    def get_address(self, key: str) -> Optional[SEntropyAddress]:
        """Get an active address by key."""
        return self._active_addresses.get(key)
    
    def navigate_between(self, from_key: str, to_key: str) -> Tuple[List, float]:
        """
        Compute navigation path between two addresses.
        
        This uses categorical completion - the path is determined by
        the accumulated precision-by-difference trajectories.
        
        Args:
            from_key: Source address key
            to_key: Destination address key
            
        Returns:
            Tuple of (path, cost)
        """
        from_addr = self._active_addresses.get(from_key)
        to_addr = self._active_addresses.get(to_key)
        
        if not from_addr or not to_addr:
            return [], float('inf')
        
        path = from_addr.navigate_to(to_addr)
        cost = from_addr.categorical_distance(to_addr)
        
        return path, cost
    
    def predict_optimal_location(self, key: str) -> Optional[SCoordinate]:
        """
        Predict where an address trajectory is heading.
        
        This is categorical completion - extracting the endpoint
        that is already encoded in the history.
        """
        address = self._active_addresses.get(key)
        if not address:
            return None
        
        return address.predict_completion()
    
    def compute_hierarchy_position(self, key: str) -> Tuple[int, List[int]]:
        """
        Compute the position in the 3^k hierarchy for an address.
        
        Returns:
            Tuple of (depth, branch_path)
        """
        address = self._active_addresses.get(key)
        if not address:
            return (0, [])
        
        branches = address.hierarchy_branch
        return (len(branches), branches)
    
    def synchronize_addresses(self, keys: List[str]) -> PrecisionWindow:
        """
        Synchronize multiple addresses to a common precision window.
        
        This enables collective state coordination as in the paper.
        """
        all_precisions = []
        
        for key in keys:
            address = self._active_addresses.get(key)
            if address:
                all_precisions.extend(list(address.precision_differences))
        
        if not all_precisions:
            return self.get_precision_window()
        
        reference_time = time.perf_counter() - self._reference_epoch
        
        return PrecisionWindow(
            start=reference_time + min(all_precisions),
            end=reference_time + max(all_precisions),
            reference_time=reference_time,
            precision_values=all_precisions
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """Get calculator statistics."""
        std = np.sqrt(self._variance_precision / max(1, self._total_calculations - 1))
        
        return {
            'total_calculations': self._total_calculations,
            'mean_precision': self._mean_precision,
            'std_precision': std,
            'active_addresses': len(self._active_addresses),
            'history_length': len(self._precision_history),
        }


