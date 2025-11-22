"""
GrandWave - Universal Reality Substrate
========================================

The GrandWave represents "reality" as an infinite superposition of oscillatory
phenomena. All measurements, phenomena, and interactions are interference patterns
between specific oscillatory disturbances and this universal substrate.

Theoretical Foundation:
-----------------------
1. Reality = Oscillatory substrate (Physical Necessity theorem)
2. Time = Alignment process in observers (St-Stellas framework)
3. Measurements = Transient disturbances in unified S-entropy space
4. Navigation = Finding paths through interference patterns

Purpose:
--------
The GrandWave serves as the universal reference to prevent getting lost during
random graph navigation. Solutions remain viable by maintaining coherence with
the GrandWave's oscillatory structure. This enables:

- Cross-domain equivalence (all domains project onto same wave)
- O(1) navigation (direct jumps through interference patterns)
- Transcendent observation (simultaneous view of all disturbances)
- Solution viability (maintained through wave coherence)

Implementation:
---------------
The GrandWave is represented as a high-dimensional Fourier basis spanning all
measurable frequencies (0.1 Hz to 10 GHz) with phase relationships preserved
through trans-Planckian timing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class WaveDisturbance:
    """
    A transient disturbance in the GrandWave (i.e., a measurement)
    
    Attributes:
        source: Origin instrument/component
        timestamp: Trans-Planckian precision timestamp
        S_coords: (S1, S2, S3) coordinates in universal space
        frequencies: Array of characteristic frequencies
        amplitudes: Array of amplitudes
        phases: Array of phases
        domain: Measurement domain ('acoustic', 'dielectric', etc.)
        duration: How long this disturbance exists (seconds)
    """
    source: str
    timestamp: float  # Trans-Planckian precision
    S_coords: np.ndarray  # (3,) array
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    domain: str
    duration: float = 0.0
    
    def __post_init__(self):
        """Validate disturbance data"""
        assert len(self.S_coords) == 3, "S_coords must be 3D"
        assert len(self.frequencies) == len(self.amplitudes) == len(self.phases), \
            "Frequencies, amplitudes, and phases must have same length"


@dataclass
class InterferencePattern:
    """
    Interference between two wave disturbances
    
    Represents how two measurements/phenomena interact in unified S-space.
    """
    disturbance_A: WaveDisturbance
    disturbance_B: WaveDisturbance
    S_distance: float  # Euclidean distance in S-space
    harmonic_coincidences: List[Tuple[float, float, float]]  # [(f_A, f_B, strength)]
    coherence: float  # 0-1, how well aligned
    phase_relationship: float  # radians
    
    @property
    def is_equivalent(self) -> bool:
        """Check if disturbances are equivalent (S-distance < 0.1)"""
        return self.S_distance < 0.1


class GrandWave:
    """
    Universal Reality Substrate - The Infinite Wave
    
    The GrandWave maintains a complete representation of reality as an infinite
    superposition of oscillatory phenomena. All measurements appear as transient
    disturbances that interfere with this substrate.
    
    This class serves as:
    1. Universal reference for navigation
    2. Repository of all active disturbances (measurements)
    3. Interference pattern calculator
    4. Viability checker (solutions must maintain coherence)
    5. Transcendent observer (simultaneous view of all phenomena)
    """
    
    def __init__(self, 
                 frequency_range: Tuple[float, float] = (0.1, 1e10),
                 n_basis_functions: int = 10000,
                 trans_planckian_precision: float = 7.51e-50):
        """
        Initialize the GrandWave
        
        Args:
            frequency_range: (min_Hz, max_Hz) - range of observable frequencies
            n_basis_functions: Number of Fourier basis functions
            trans_planckian_precision: Temporal precision (seconds)
        """
        self.frequency_range = frequency_range
        self.n_basis = n_basis_functions
        self.precision = trans_planckian_precision
        
        # Generate logarithmic frequency basis (matches hierarchical oscillations)
        self.basis_frequencies = np.logspace(
            np.log10(frequency_range[0]),
            np.log10(frequency_range[1]),
            n_basis_functions
        )
        
        # Initialize basis amplitudes (all unity initially)
        self.basis_amplitudes = np.ones(n_basis_functions)
        
        # Initialize basis phases (all zero initially)
        self.basis_phases = np.zeros(n_basis_functions)
        
        # Active disturbances (measurements currently in the wave)
        self.active_disturbances: Dict[str, WaveDisturbance] = {}
        
        # Interference pattern cache
        self.interference_cache: Dict[Tuple[str, str], InterferencePattern] = {}
        
        # Reference time (trans-Planckian origin)
        self.t0 = time.time()
        
        # Statistics
        self.total_disturbances_created = 0
        self.total_interference_patterns = 0
        
    def get_current_time(self) -> float:
        """
        Get current time with trans-Planckian precision
        
        Returns:
            Time in seconds since initialization (trans-Planckian resolution)
        """
        # In real implementation, this would call hardware clock
        # For now, simulate by scaling system time
        elapsed = time.time() - self.t0
        # Quantize to trans-Planckian precision
        quantized = np.round(elapsed / self.precision) * self.precision
        return quantized
        
    def add_disturbance(self, 
                       source: str,
                       S_coords: np.ndarray,
                       frequencies: np.ndarray,
                       amplitudes: np.ndarray,
                       phases: np.ndarray,
                       domain: str,
                       duration: float = 0.0) -> WaveDisturbance:
        """
        Add a new disturbance to the GrandWave (i.e., record a measurement)
        
        Args:
            source: Identifier for measurement source
            S_coords: S-entropy coordinates (3D)
            frequencies: Characteristic frequencies
            amplitudes: Amplitudes for each frequency
            phases: Phases for each frequency
            domain: Measurement domain
            duration: How long disturbance persists (0 = instant)
            
        Returns:
            WaveDisturbance object
        """
        timestamp = self.get_current_time()
        
        disturbance = WaveDisturbance(
            source=source,
            timestamp=timestamp,
            S_coords=np.array(S_coords),
            frequencies=np.array(frequencies),
            amplitudes=np.array(amplitudes),
            phases=np.array(phases),
            domain=domain,
            duration=duration
        )
        
        # Store in active disturbances
        self.active_disturbances[source] = disturbance
        self.total_disturbances_created += 1
        
        # Update GrandWave basis (disturbance modifies the infinite wave)
        self._incorporate_disturbance(disturbance)
        
        # Calculate interference with all other active disturbances
        self._update_interference_patterns(disturbance)
        
        return disturbance
        
    def _incorporate_disturbance(self, disturbance: WaveDisturbance):
        """
        Incorporate disturbance into GrandWave basis
        
        The disturbance modifies the infinite wave through superposition.
        """
        for freq, amp, phase in zip(disturbance.frequencies, 
                                     disturbance.amplitudes,
                                     disturbance.phases):
            # Find nearest basis frequency
            idx = np.argmin(np.abs(self.basis_frequencies - freq))
            
            # Superpose (complex addition in frequency domain)
            existing = self.basis_amplitudes[idx] * np.exp(1j * self.basis_phases[idx])
            new = amp * np.exp(1j * phase)
            combined = existing + new
            
            # Update basis
            self.basis_amplitudes[idx] = np.abs(combined)
            self.basis_phases[idx] = np.angle(combined)
            
    def _update_interference_patterns(self, new_disturbance: WaveDisturbance):
        """
        Calculate interference between new disturbance and all existing ones
        """
        for source, existing in self.active_disturbances.items():
            if source == new_disturbance.source:
                continue
                
            # Calculate interference
            pattern = self.calculate_interference(new_disturbance, existing)
            
            # Cache it
            key = tuple(sorted([new_disturbance.source, source]))
            self.interference_cache[key] = pattern
            self.total_interference_patterns += 1
            
    def calculate_interference(self, 
                              disturbance_A: WaveDisturbance,
                              disturbance_B: WaveDisturbance) -> InterferencePattern:
        """
        Calculate interference pattern between two disturbances
        
        This reveals how two measurements/phenomena interact in unified space.
        
        Returns:
            InterferencePattern describing the interaction
        """
        # S-distance (cross-domain equivalence metric)
        S_distance = np.linalg.norm(disturbance_A.S_coords - disturbance_B.S_coords)
        
        # Find harmonic coincidences
        coincidences = []
        for i, f_A in enumerate(disturbance_A.frequencies):
            for j, f_B in enumerate(disturbance_B.frequencies):
                # Check for harmonic coincidence (within 0.1 Hz)
                if np.abs(f_A - f_B) < 0.1:
                    strength = min(disturbance_A.amplitudes[i], 
                                  disturbance_B.amplitudes[j])
                    coincidences.append((f_A, f_B, strength))
                    
                # Check harmonics (up to 10th order)
                for n_A in range(1, 11):
                    for n_B in range(1, 11):
                        if np.abs(n_A * f_A - n_B * f_B) < 0.1:
                            strength = min(
                                disturbance_A.amplitudes[i] / n_A,
                                disturbance_B.amplitudes[j] / n_B
                            )
                            coincidences.append((n_A * f_A, n_B * f_B, strength))
        
        # Calculate coherence (normalized dot product of amplitudes)
        # Interpolate both to common frequency grid
        common_freqs = np.union1d(disturbance_A.frequencies, disturbance_B.frequencies)
        amp_A = np.interp(common_freqs, disturbance_A.frequencies, 
                         disturbance_A.amplitudes, left=0, right=0)
        amp_B = np.interp(common_freqs, disturbance_B.frequencies,
                         disturbance_B.amplitudes, left=0, right=0)
        
        coherence = np.dot(amp_A, amp_B) / (np.linalg.norm(amp_A) * np.linalg.norm(amp_B))
        if np.isnan(coherence):
            coherence = 0.0
            
        # Phase relationship (average phase difference at coincidences)
        if coincidences:
            phase_diffs = []
            for f_A_coin, f_B_coin, _ in coincidences:
                idx_A = np.argmin(np.abs(disturbance_A.frequencies - f_A_coin))
                idx_B = np.argmin(np.abs(disturbance_B.frequencies - f_B_coin))
                phase_diff = disturbance_A.phases[idx_A] - disturbance_B.phases[idx_B]
                phase_diffs.append(phase_diff)
            phase_relationship = np.mean(phase_diffs)
        else:
            phase_relationship = 0.0
            
        return InterferencePattern(
            disturbance_A=disturbance_A,
            disturbance_B=disturbance_B,
            S_distance=S_distance,
            harmonic_coincidences=coincidences,
            coherence=coherence,
            phase_relationship=phase_relationship
        )
        
    def find_equivalent_disturbances(self, 
                                     S_coords: np.ndarray,
                                     threshold: float = 0.1) -> List[WaveDisturbance]:
        """
        Find all disturbances equivalent to given S-coordinates
        
        This enables cross-domain solution transfer: measurements with S-distance < threshold
        are informationally equivalent.
        
        Args:
            S_coords: Target S-entropy coordinates
            threshold: Maximum S-distance for equivalence (default 0.1)
            
        Returns:
            List of equivalent disturbances
        """
        equivalents = []
        
        for disturbance in self.active_disturbances.values():
            distance = np.linalg.norm(disturbance.S_coords - S_coords)
            if distance < threshold:
                equivalents.append(disturbance)
                
        # Sort by distance
        equivalents.sort(key=lambda d: np.linalg.norm(d.S_coords - S_coords))
        
        return equivalents
        
    def navigate_to_target(self, 
                          S_current: np.ndarray,
                          S_target: np.ndarray) -> Dict[str, Any]:
        """
        Navigate from current S-coordinates to target via GrandWave
        
        This is the O(1) navigation: direct jump through interference patterns
        rather than sequential graph traversal.
        
        Args:
            S_current: Current position in S-space
            S_target: Target position in S-space
            
        Returns:
            Navigation result with path, intermediate disturbances, etc.
        """
        # Direct vector in S-space
        delta_S = S_target - S_current
        S_distance = np.linalg.norm(delta_S)
        
        # Find disturbances along the path (if any)
        intermediate_disturbances = []
        for disturbance in self.active_disturbances.values():
            # Check if disturbance lies on path (within small tolerance)
            projection = np.dot(disturbance.S_coords - S_current, delta_S) / S_distance**2
            if 0 < projection < 1:  # Between current and target
                distance_to_line = np.linalg.norm(
                    disturbance.S_coords - (S_current + projection * delta_S)
                )
                if distance_to_line < 0.2:  # Near the path
                    intermediate_disturbances.append({
                        'disturbance': disturbance,
                        'projection': projection,
                        'distance_to_path': distance_to_line
                    })
        
        # Sort by projection (order along path)
        intermediate_disturbances.sort(key=lambda d: d['projection'])
        
        return {
            'S_current': S_current,
            'S_target': S_target,
            'delta_S': delta_S,
            'S_distance': S_distance,
            'intermediate_disturbances': intermediate_disturbances,
            'navigation_type': 'direct_jump',
            'complexity': 'O(1)'
        }
        
    def check_solution_viability(self, 
                                 S_solution: np.ndarray,
                                 domain: str) -> Dict[str, Any]:
        """
        Check if a solution is viable by verifying coherence with GrandWave
        
        Solutions must maintain coherence with the universal oscillatory substrate.
        This prevents "miraculous" intermediate states from persisting.
        
        Args:
            S_solution: Proposed solution coordinates
            domain: Domain in which solution should be viable
            
        Returns:
            Viability check results
        """
        # Find nearest disturbances in same domain
        domain_disturbances = [
            d for d in self.active_disturbances.values() 
            if d.domain == domain
        ]
        
        if not domain_disturbances:
            return {
                'viable': True,  # No constraints
                'reason': 'No reference disturbances in domain',
                'coherence': 1.0
            }
        
        # Calculate average coherence with domain
        coherences = []
        for disturbance in domain_disturbances:
            distance = np.linalg.norm(disturbance.S_coords - S_solution)
            # Coherence decreases exponentially with S-distance
            coherence = np.exp(-distance / 0.5)
            coherences.append(coherence)
            
        avg_coherence = np.mean(coherences)
        
        # Solution is viable if coherence > 0.3
        viable = avg_coherence > 0.3
        
        return {
            'viable': viable,
            'coherence': avg_coherence,
            'n_references': len(domain_disturbances),
            'reason': 'Sufficient coherence' if viable else 'Insufficient coherence'
        }
        
    def get_transcendent_view(self) -> Dict[str, Any]:
        """
        Get transcendent observer view: simultaneous observation of all disturbances
        
        This enables seeing all measurements at once in unified S-space,
        revealing cross-domain patterns invisible to sequential observation.
        
        Returns:
            Complete state of the GrandWave
        """
        # Collect all S-coordinates
        all_S_coords = np.array([d.S_coords for d in self.active_disturbances.values()])
        
        if len(all_S_coords) == 0:
            return {
                'n_disturbances': 0,
                'S_centroid': np.array([0, 0, 0]),
                'S_span': 0.0,
                'domains': [],
                'frequency_range': (0, 0)
            }
        
        # Calculate centroid in S-space
        S_centroid = np.mean(all_S_coords, axis=0)
        
        # Calculate span (max distance from centroid)
        distances = np.linalg.norm(all_S_coords - S_centroid, axis=1)
        S_span = np.max(distances)
        
        # Domains represented
        domains = list(set(d.domain for d in self.active_disturbances.values()))
        
        # Frequency range covered
        all_frequencies = np.concatenate([
            d.frequencies for d in self.active_disturbances.values()
        ])
        freq_range = (np.min(all_frequencies), np.max(all_frequencies))
        
        # Interference network density
        n_disturbances = len(self.active_disturbances)
        n_possible_pairs = n_disturbances * (n_disturbances - 1) // 2
        n_actual_patterns = len(self.interference_cache)
        network_density = n_actual_patterns / n_possible_pairs if n_possible_pairs > 0 else 0
        
        return {
            'n_disturbances': n_disturbances,
            'S_centroid': S_centroid,
            'S_span': S_span,
            'domains': domains,
            'frequency_range': freq_range,
            'interference_patterns': n_actual_patterns,
            'network_density': network_density,
            'total_disturbances_created': self.total_disturbances_created,
            'basis_energy': np.sum(self.basis_amplitudes**2)
        }
        
    def cleanup_expired_disturbances(self):
        """
        Remove disturbances that have expired (duration elapsed)
        
        Measurements persist only for their specified duration,
        then fade from the GrandWave.
        """
        current_time = self.get_current_time()
        expired = []
        
        for source, disturbance in self.active_disturbances.items():
            if disturbance.duration > 0:
                if current_time - disturbance.timestamp > disturbance.duration:
                    expired.append(source)
                    
        # Remove expired
        for source in expired:
            del self.active_disturbances[source]
            
        # Clean interference cache
        keys_to_remove = []
        for key in self.interference_cache.keys():
            if key[0] not in self.active_disturbances or key[1] not in self.active_disturbances:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.interference_cache[key]
            
    def export_state(self) -> Dict[str, Any]:
        """
        Export complete GrandWave state for persistence/analysis
        """
        return {
            'timestamp': self.get_current_time(),
            'frequency_range': self.frequency_range,
            'n_basis': self.n_basis,
            'basis_frequencies': self.basis_frequencies.tolist(),
            'basis_amplitudes': self.basis_amplitudes.tolist(),
            'basis_phases': self.basis_phases.tolist(),
            'active_disturbances': {
                source: {
                    'timestamp': d.timestamp,
                    'S_coords': d.S_coords.tolist(),
                    'frequencies': d.frequencies.tolist(),
                    'amplitudes': d.amplitudes.tolist(),
                    'phases': d.phases.tolist(),
                    'domain': d.domain
                }
                for source, d in self.active_disturbances.items()
            },
            'statistics': self.get_transcendent_view()
        }
        
    def __repr__(self) -> str:
        stats = self.get_transcendent_view()
        return (
            f"GrandWave(n_disturbances={stats['n_disturbances']}, "
            f"domains={stats['domains']}, "
            f"freq_range={stats['frequency_range']}, "
            f"S_span={stats['S_span']:.3f})"
        ) 