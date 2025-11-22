"""
S-Entropy Calculator - Domain-Specific Calculations
====================================================

Calculates S-entropy coordinates from oscillatory signatures for each domain.
These coordinates enable cross-domain equivalence checking and O(1) navigation.

Domains:
--------
- Acoustic: Sound/vibration measurements
- Dielectric: Capacitance/permittivity
- Thermal: Temperature/heat flow
- Electromagnetic: EM fields/induction
- Mechanical: Force/displacement
- Optical: Light/photon interactions
- Chemical: Molecular/spectroscopic
"""

import numpy as np
from typing import Dict, Tuple
from .oscillatory_signatures import OscillatorySignature


class SEntropyCalculator:
    """Calculate S-entropy coordinates from oscillatory signatures"""
    
    def __init__(self, domain: str):
        """
        Initialize calculator for specific domain
        
        Args:
            domain: Measurement domain
        """
        self.domain = domain
        
    def calculate(self, signature: OscillatorySignature) -> np.ndarray:
        """
        Calculate S-entropy coordinates from oscillatory signature
        
        Args:
            signature: OscillatorySignature from analysis
            
        Returns:
            3D S-entropy coordinates
        """
        # Extract key data
        freqs = signature.frequencies
        amps = signature.amplitudes
        phases = signature.phases
        Q_factors = signature.Q_factors
        
        # S1: Frequency-weighted entropy
        S1 = self._calculate_frequency_entropy(freqs, amps)
        
        # S2: Phase coherence entropy
        S2 = self._calculate_phase_entropy(phases)
        
        # S3: Q-factor distribution entropy
        S3 = self._calculate_quality_entropy(Q_factors)
        
        # Apply domain-specific scaling
        S_coords = self._apply_domain_scaling(np.array([S1, S2, S3]))
        
        return S_coords
        
    def _calculate_frequency_entropy(self, freqs: np.ndarray, amps: np.ndarray) -> float:
        """S1: Frequency distribution entropy"""
        # Normalize amplitudes
        normalized_amps = amps / (np.sum(amps) + 1e-10)
        
        # Shannon entropy weighted by frequency
        entropy = -np.sum(normalized_amps * np.log(normalized_amps + 1e-10))
        
        # Weight by frequency spread
        freq_spread = (np.max(freqs) - np.min(freqs)) / (np.mean(freqs) + 1e-10)
        
        S1 = entropy * (1 + 0.1 * freq_spread)
        
        return S1
        
    def _calculate_phase_entropy(self, phases: np.ndarray) -> float:
        """S2: Phase relationship entropy"""
        if len(phases) < 2:
            return 0.0
            
        # Relative phases
        rel_phases = phases - phases[0]
        
        # Wrap to [0, 2π]
        rel_phases = np.mod(rel_phases, 2*np.pi)
        
        # Phase distribution entropy
        hist, _ = np.histogram(rel_phases, bins=8, range=(0, 2*np.pi))
        hist_norm = hist / (np.sum(hist) + 1e-10)
        
        S2 = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        
        return S2
        
    def _calculate_quality_entropy(self, Q_factors: np.ndarray) -> float:
        """S3: Q-factor distribution entropy"""
        if len(Q_factors) == 0:
            return 0.0
            
        # Log-scale Q-factors (high Q varies exponentially)
        log_Q = np.log10(Q_factors + 1)
        
        # Normalize
        log_Q_norm = log_Q / (np.sum(log_Q) + 1e-10)
        
        # Entropy
        S3 = -np.sum(log_Q_norm * np.log(log_Q_norm + 1e-10))
        
        return S3
        
    def _apply_domain_scaling(self, S_coords: np.ndarray) -> np.ndarray:
        """Apply domain-specific scaling factors"""
        domain_scales = {
            'acoustic': np.array([1.0, 1.1, 0.9]),
            'dielectric': np.array([1.2, 0.9, 1.0]),
            'thermal': np.array([0.9, 1.0, 1.1]),
            'electromagnetic': np.array([1.1, 1.0, 0.9]),
            'mechanical': np.array([1.05, 1.05, 1.0]),
            'optical': np.array([1.15, 0.95, 1.0]),
            'chemical': np.array([0.95, 1.05, 1.1])
        }
        
        scale = domain_scales.get(self.domain, np.array([1.0, 1.0, 1.0]))
        
        return S_coords * scale