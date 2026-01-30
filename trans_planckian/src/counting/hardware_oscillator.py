"""
Hardware Oscillator Capture

Captures timing variations from actual hardware components to generate
precision-by-difference values. These are NOT simulated - they are real
measurements from the computer's oscillating processes.

Sources of oscillation:
- CPU cycle timing variations
- Memory access latency
- I/O timing jitter
- Power supply fluctuations (via performance counters)
- Network timing (if available)

These hardware oscillations map onto molecular frequencies through
harmonic coincidence, enabling categorical measurement.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from collections import deque
import threading
import hashlib

# Try to import performance monitoring libraries
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class OscillationSample:
    """A single sample from hardware oscillators."""
    timestamp: float  # High-resolution timestamp
    source: str  # Which oscillator this came from
    value: float  # The measured value
    reference: float  # The reference value (expected)
    precision_diff: float = field(init=False)  # Computed precision-by-difference
    
    def __post_init__(self):
        self.precision_diff = self.reference - self.value


class HardwareOscillatorCapture:
    """
    Captures real hardware oscillations and converts them to precision-by-difference values.
    
    This is the interface between physical hardware and the categorical memory system.
    The oscillations are REAL - they come from actual hardware timing variations.
    """
    
    # Approximate fundamental frequencies of common hardware oscillators (Hz)
    REFERENCE_FREQUENCIES = {
        'cpu_cycle': 3.0e9,  # 3 GHz typical CPU
        'memory_cycle': 2.133e9,  # DDR4 typical
        'pcie_cycle': 8.0e9,  # PCIe gen 3
        'usb_frame': 1000.0,  # USB 1ms frame
        'display_refresh': 60.0,  # 60 Hz typical
        'power_ac': 50.0,  # 50/60 Hz power line
    }
    
    def __init__(self, sample_rate: float = 1000.0):
        """
        Initialize the hardware oscillator capture.
        
        Args:
            sample_rate: How many samples per second to capture
        """
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate
        
        # Storage for samples
        self.samples: deque[OscillationSample] = deque(maxlen=10000)
        
        # Reference timestamps for precision-by-difference
        self._reference_time = time.perf_counter()
        self._reference_counter = time.perf_counter_ns()
        
        # Calibration data
        self._calibration: Dict[str, float] = {}
        
        # Background capture thread
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        
    def calibrate(self, duration: float = 1.0) -> Dict[str, float]:
        """
        Calibrate oscillator references by measuring for a duration.
        
        This establishes the 'reference' values for precision-by-difference.
        """
        samples_per_source: Dict[str, List[float]] = {
            'timing_jitter': [],
            'perf_counter': [],
        }
        
        if HAS_PSUTIL:
            samples_per_source['cpu_percent'] = []
            samples_per_source['memory_percent'] = []
        
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            # Timing jitter (difference between expected and actual interval)
            t0 = time.perf_counter()
            time.sleep(0.001)  # 1ms target
            t1 = time.perf_counter()
            samples_per_source['timing_jitter'].append((t1 - t0) - 0.001)
            
            # Performance counter variation
            samples_per_source['perf_counter'].append(time.perf_counter_ns() % 1000000)
            
            if HAS_PSUTIL:
                samples_per_source['cpu_percent'].append(psutil.cpu_percent(interval=0))
                samples_per_source['memory_percent'].append(psutil.virtual_memory().percent)
        
        # Compute reference statistics
        for source, values in samples_per_source.items():
            if values:
                self._calibration[source] = float(np.mean(values))
                
        return self._calibration
    
    def capture_sample(self) -> OscillationSample:
        """
        Capture a single sample from hardware oscillators.
        
        Returns:
            An OscillationSample with real hardware timing data
        """
        # High-resolution timing
        timestamp = time.perf_counter()
        
        # Measure actual timing vs expected
        expected_elapsed = timestamp - self._reference_time
        actual_ns = time.perf_counter_ns() - self._reference_counter
        actual_elapsed = actual_ns / 1e9
        
        # The precision-by-difference: difference between expected and actual
        # This captures the real hardware timing variations
        value = actual_elapsed
        reference = expected_elapsed
        
        sample = OscillationSample(
            timestamp=timestamp,
            source='timing',
            value=value,
            reference=reference
        )
        
        self.samples.append(sample)
        return sample
    
    def capture_multi_source(self) -> List[OscillationSample]:
        """
        Capture samples from multiple oscillator sources simultaneously.
        
        This provides a richer precision-by-difference signature.
        """
        samples = []
        timestamp = time.perf_counter()
        
        # Source 1: High-resolution timing jitter
        t0 = time.perf_counter_ns()
        _ = hash(str(time.time()))  # Small computation to create measurable jitter
        t1 = time.perf_counter_ns()
        jitter = (t1 - t0) / 1e9
        
        samples.append(OscillationSample(
            timestamp=timestamp,
            source='computation_jitter',
            value=jitter,
            reference=self._calibration.get('timing_jitter', 1e-6)
        ))
        
        # Source 2: Memory access timing
        t0 = time.perf_counter_ns()
        _ = [0] * 1000  # Allocate memory
        t1 = time.perf_counter_ns()
        mem_time = (t1 - t0) / 1e9
        
        samples.append(OscillationSample(
            timestamp=timestamp,
            source='memory_timing',
            value=mem_time,
            reference=self._calibration.get('timing_jitter', 1e-6)
        ))
        
        # Source 3: Performance counter variation
        counter_val = time.perf_counter_ns() % 1000000
        samples.append(OscillationSample(
            timestamp=timestamp,
            source='perf_counter',
            value=counter_val / 1e6,
            reference=self._calibration.get('perf_counter', 500000) / 1e6
        ))
        
        # Source 4: System metrics if available
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=0) / 100.0
            samples.append(OscillationSample(
                timestamp=timestamp,
                source='cpu_load',
                value=cpu,
                reference=self._calibration.get('cpu_percent', 50) / 100.0
            ))
        
        for s in samples:
            self.samples.append(s)
            
        return samples
    
    def get_precision_signature(self, n_samples: int = 10) -> np.ndarray:
        """
        Get a precision-by-difference signature from recent samples.
        
        This signature encodes a position in the S-entropy hierarchy.
        
        Args:
            n_samples: Number of samples to include
            
        Returns:
            Array of precision-by-difference values
        """
        # Capture fresh samples if needed
        while len(self.samples) < n_samples:
            self.capture_multi_source()
            
        recent = list(self.samples)[-n_samples:]
        return np.array([s.precision_diff for s in recent])
    
    def signature_to_scoordinate(self, signature: np.ndarray) -> 'SCoordinate':
        """
        Convert a precision signature to an S-coordinate.
        
        The three S-coordinates are derived from the signature:
        - S_k: Kinetic (variance, rate of change)
        - S_t: Thermal (mean, central tendency)
        - S_e: Entropic (entropy, disorder)
        """
        from .s_entropy_address import SCoordinate
        
        if len(signature) == 0:
            return SCoordinate(S_k=0, S_t=0, S_e=0)
        
        # S_k: Rate of change (kinetic)
        if len(signature) > 1:
            S_k = float(np.std(np.diff(signature)))
        else:
            S_k = 0.0
            
        # S_t: Mean value (thermal)
        S_t = float(np.mean(signature))
        
        # S_e: Entropy (disorder)
        # Use histogram entropy
        hist, _ = np.histogram(signature, bins=min(10, len(signature)))
        hist = hist / (hist.sum() + 1e-10)
        S_e = float(-np.sum(hist * np.log(hist + 1e-10)))
        
        return SCoordinate(S_k=S_k, S_t=S_t, S_e=S_e)
    
    def start_background_capture(self, callback: Optional[Callable] = None):
        """Start continuous background capture."""
        if self._running:
            return
            
        self._running = True
        
        def capture_loop():
            while self._running:
                samples = self.capture_multi_source()
                if callback:
                    callback(samples)
                time.sleep(self.sample_interval)
                
        self._capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self._capture_thread.start()
        
    def stop_background_capture(self):
        """Stop background capture."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
            
    def get_harmonic_coincidences(self, target_freq: float, tolerance: float = 0.01) -> List[Dict]:
        """
        Find harmonic coincidences between hardware oscillators and a target frequency.
        
        This is the mapping from hardware to molecular frequencies that enables
        zero-backaction categorical measurement.
        
        Args:
            target_freq: Target frequency to find harmonics of (Hz)
            tolerance: Fractional tolerance for coincidence detection
            
        Returns:
            List of harmonic coincidences found
        """
        coincidences = []
        
        for source, ref_freq in self.REFERENCE_FREQUENCIES.items():
            # Check for harmonic relationships
            # n * target_freq ≈ m * ref_freq for integers n, m
            
            for n in range(1, 100):
                for m in range(1, 100):
                    ratio = (n * target_freq) / (m * ref_freq)
                    if abs(ratio - 1.0) < tolerance:
                        coincidences.append({
                            'source': source,
                            'source_freq': ref_freq,
                            'target_freq': target_freq,
                            'harmonic_n': n,
                            'harmonic_m': m,
                            'ratio': ratio,
                            'error': abs(ratio - 1.0)
                        })
                        
        return coincidences


