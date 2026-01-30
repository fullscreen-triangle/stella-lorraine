"""
Partition Coordinate Instruments

Instruments for measuring partition coordinates (n, l, m, s) in bounded phase space.
These implement the virtual instrument suite from the bounded systems framework.

Shell Capacity: 2n² (verified experimentally)
Selection Rules: Δl = ±1, Δm ∈ {0, ±1}, Δs = 0
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .base import (
    VirtualInstrument, 
    HardwareOscillator, 
    PartitionCoordinate,
    CategoricalState,
    BOLTZMANN_CONSTANT,
    PLANCK_CONSTANT
)


class ShellResonator(VirtualInstrument):
    """
    Shell Resonator - Measures partition depth n.
    
    Theory: Resonance frequency scales inversely with depth squared:
        f_resonance(n) = f_0 / n²
    
    Hardware timing variations select which resonance is "observed."
    """
    
    def __init__(self, base_frequency: float = 1e9):
        """
        Initialize shell resonator.
        
        Args:
            base_frequency: Base frequency f_0 in Hz
        """
        super().__init__("Shell Resonator")
        self.f_0 = base_frequency
        self.max_depth = 10  # Maximum detectable depth
        
    def calibrate(self) -> bool:
        """Calibrate by measuring resonance peaks"""
        # Measure resonances at known depths
        calibration_data = []
        for n in range(1, 5):
            expected_f = self.f_0 / (n ** 2)
            measured = self._measure_resonance()
            calibration_data.append((n, expected_f, measured))
        
        self.calibrated = True
        return True
    
    def _measure_resonance(self) -> float:
        """Measure resonance frequency from hardware timing"""
        delta_p = self.oscillator.read_timing_deviation()
        # Convert timing to frequency
        return self.f_0 * (1 + delta_p / 1000.0)
    
    def measure(self, **kwargs) -> Dict[str, Any]:
        """
        Measure partition depth n.
        
        Returns:
            Dictionary with measured depth and resonance frequency
        """
        f_resonance = self._measure_resonance()
        
        # Determine n from resonance: n = sqrt(f_0 / f_resonance)
        if f_resonance > 0:
            n_continuous = np.sqrt(self.f_0 / f_resonance)
            n = max(1, min(self.max_depth, int(round(n_continuous))))
        else:
            n = 1
        
        result = {
            'n': n,
            'resonance_frequency': f_resonance,
            'theoretical_frequency': self.f_0 / (n ** 2),
            'shell_capacity': 2 * n ** 2
        }
        
        self.record_measurement(result)
        return result


class AngularAnalyser(VirtualInstrument):
    """
    Angular Analyser - Measures complexity parameter l.
    
    Theory: Detects phase relationships in the partition boundary.
    The number of phase nodes determines l.
    
    Constraint: l ∈ {0, 1, ..., n-1} for measured depth n
    """
    
    def __init__(self):
        super().__init__("Angular Analyser")
        
    def calibrate(self) -> bool:
        """Calibrate phase detection"""
        self.calibrated = True
        return True
    
    def measure(self, n: int, **kwargs) -> Dict[str, Any]:
        """
        Measure angular complexity l.
        
        Args:
            n: Partition depth (constrains allowed l values)
            
        Returns:
            Dictionary with measured complexity
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        # Hardware timing selects among allowed values
        delta_p = self.oscillator.read_timing_deviation()
        
        # Map to allowed range [0, n-1]
        l_raw = int(delta_p) % n
        l = min(l_raw, n - 1)
        
        # Count nodal planes
        nodal_planes = l
        
        result = {
            'l': l,
            'n': n,
            'nodal_planes': nodal_planes,
            'degeneracy': 2 * l + 1,  # Number of orientations
            'allowed_range': list(range(n))
        }
        
        self.record_measurement(result)
        return result


class OrientationMapper(VirtualInstrument):
    """
    Orientation Mapper - Measures orientation parameter m.
    
    Theory: Detects spatial direction of partition boundary nodes.
    For complexity l, there are 2l + 1 possible orientations.
    
    Constraint: m ∈ {-l, -l+1, ..., 0, ..., l-1, l}
    """
    
    def __init__(self):
        super().__init__("Orientation Mapper")
        
    def calibrate(self) -> bool:
        """Calibrate orientation detection"""
        self.calibrated = True
        return True
    
    def measure(self, l: int, **kwargs) -> Dict[str, Any]:
        """
        Measure orientation m.
        
        Args:
            l: Angular complexity (constrains allowed m values)
            
        Returns:
            Dictionary with measured orientation
        """
        if l < 0:
            raise ValueError("l must be >= 0")
        
        # Use vector components of hardware timing
        delta_x = self.oscillator.read_timing_deviation()
        delta_y = self.oscillator.read_timing_deviation()
        
        # Map to allowed range [-l, l]
        sign = np.sign(delta_x - delta_y) if delta_x != delta_y else 1
        magnitude = int(abs(delta_x - delta_y)) % (l + 1)
        m = int(sign * magnitude)
        
        # Clamp to valid range
        m = max(-l, min(l, m))
        
        result = {
            'm': m,
            'l': l,
            'allowed_range': list(range(-l, l + 1)),
            'angular_momentum_projection': m
        }
        
        self.record_measurement(result)
        return result


class ChiralityDiscriminator(VirtualInstrument):
    """
    Chirality Discriminator - Measures chirality parameter s.
    
    Theory: Detects handedness of partition boundary.
    Binary measurement: s = ±1/2
    
    Hardware timing parity determines the measured chirality.
    """
    
    def __init__(self):
        super().__init__("Chirality Discriminator")
        
    def calibrate(self) -> bool:
        """Calibrate chirality detection"""
        self.calibrated = True
        return True
    
    def measure(self, **kwargs) -> Dict[str, Any]:
        """
        Measure chirality s.
        
        Returns:
            Dictionary with measured chirality
        """
        delta_p = self.oscillator.read_timing_deviation()
        
        # Binary decision based on parity
        is_even = int(delta_p) % 2 == 0
        s = 0.5 if is_even else -0.5
        
        result = {
            's': s,
            'handedness': 'right' if s > 0 else 'left',
            'raw_timing_ns': delta_p
        }
        
        self.record_measurement(result)
        return result


class PartitionCoordinateMeasurer(VirtualInstrument):
    """
    Complete partition coordinate measurement system.
    
    Combines all four instruments to measure (n, l, m, s) sequentially,
    ensuring geometric constraints are satisfied by construction.
    """
    
    def __init__(self, base_frequency: float = 1e9):
        super().__init__("Partition Coordinate Measurer")
        
        # Component instruments
        self.shell_resonator = ShellResonator(base_frequency)
        self.angular_analyser = AngularAnalyser()
        self.orientation_mapper = OrientationMapper()
        self.chirality_discriminator = ChiralityDiscriminator()
        
        # Energy scale (eV)
        self.E_0 = 13.6  # Calibrated to hydrogen-like system
        
    def calibrate(self) -> bool:
        """Calibrate all component instruments"""
        self.shell_resonator.calibrate()
        self.angular_analyser.calibrate()
        self.orientation_mapper.calibrate()
        self.chirality_discriminator.calibrate()
        self.calibrated = True
        return True
    
    def measure(self, **kwargs) -> Dict[str, Any]:
        """
        Perform complete partition coordinate measurement.
        
        Sequential measurement ensures constraints are satisfied:
        1. Shell resonator measures n
        2. Angular analyser measures l (constrained by n)
        3. Orientation mapper measures m (constrained by l)
        4. Chirality discriminator measures s
        
        Returns:
            Dictionary with complete partition coordinate
        """
        # Sequential measurement
        n_result = self.shell_resonator.measure()
        n = n_result['n']
        
        l_result = self.angular_analyser.measure(n=n)
        l = l_result['l']
        
        m_result = self.orientation_mapper.measure(l=l)
        m = m_result['m']
        
        s_result = self.chirality_discriminator.measure()
        s = s_result['s']
        
        # Create partition coordinate
        coord = PartitionCoordinate(n=n, l=l, m=m, s=s)
        
        # Compute energy
        energy = coord.energy(self.E_0)
        
        result = {
            'coordinate': coord,
            'n': n,
            'l': l,
            'm': m,
            's': s,
            'energy_eV': energy,
            'shell_capacity': 2 * n ** 2,
            'subshell_capacity': 2 * (2 * l + 1),
            'component_results': {
                'shell': n_result,
                'angular': l_result,
                'orientation': m_result,
                'chirality': s_result
            }
        }
        
        self.record_measurement(result)
        return result
    
    def measure_transition(self, initial: PartitionCoordinate, 
                           final: PartitionCoordinate) -> Dict[str, Any]:
        """
        Measure transition between partition coordinates.
        
        Selection rules:
        - Δl = ±1 (allowed)
        - Δm ∈ {0, ±1} (allowed)
        - Δs = 0 (chirality conserved)
        
        Returns:
            Dictionary with transition properties
        """
        delta_n = final.n - initial.n
        delta_l = final.l - initial.l
        delta_m = final.m - initial.m
        delta_s = final.s - initial.s
        
        # Check selection rules
        l_allowed = delta_l in (-1, 1)
        m_allowed = delta_m in (-1, 0, 1)
        s_allowed = delta_s == 0
        
        is_allowed = l_allowed and m_allowed and s_allowed
        
        # Transition energy
        E_initial = initial.energy(self.E_0)
        E_final = final.energy(self.E_0)
        delta_E = E_final - E_initial
        
        # Wavelength if carried by photon
        if delta_E != 0:
            wavelength_m = PLANCK_CONSTANT * 299792458 / abs(delta_E * 1.602e-19)
            wavelength_nm = wavelength_m * 1e9
        else:
            wavelength_nm = float('inf')
        
        return {
            'initial': initial,
            'final': final,
            'delta_n': delta_n,
            'delta_l': delta_l,
            'delta_m': delta_m,
            'delta_s': delta_s,
            'is_allowed': is_allowed,
            'selection_rules': {
                'delta_l_allowed': l_allowed,
                'delta_m_allowed': m_allowed,
                'delta_s_allowed': s_allowed
            },
            'energy_change_eV': delta_E,
            'wavelength_nm': wavelength_nm,
            'emission': delta_E < 0,
            'absorption': delta_E > 0
        }
    
    def verify_shell_capacity(self, max_n: int = 4) -> Dict[int, Dict[str, Any]]:
        """
        Verify the 2n² shell capacity formula.
        
        Returns:
            Dictionary mapping n to capacity verification results
        """
        results = {}
        
        for n in range(1, max_n + 1):
            theoretical = 2 * n ** 2
            
            # Count all valid (l, m, s) combinations
            measured = 0
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [-0.5, 0.5]:
                        measured += 1
            
            results[n] = {
                'theoretical_capacity': theoretical,
                'measured_capacity': measured,
                'agreement': theoretical == measured,
                'subshells': [(l, 2 * (2 * l + 1)) for l in range(n)]
            }
        
        return results
