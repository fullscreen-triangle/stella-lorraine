"""
Oscillatory Signatures - Pathway 1 (Oscillatory Analysis)
==========================================================

Extracts oscillatory signatures from measurements and calculates S-entropy
coordinates. This is the "rigorous" pathway that captures frequency-domain
information through FFT, harmonic extraction, and phase analysis.

Pipeline:
---------
Raw Signal → FFT → Harmonics → S-Entropy Coords → Predictions

Components:
-----------
1. FFTEngine: High-performance FFT with windowing
2. HarmonicExtractor: Peak finding and Q-factor estimation
3. PhaseAnalyzer: Phase relationships between harmonics
4. SEntropyCalculator: Domain-specific S-coordinate calculation
5. PredictionEngine: Domain-specific property prediction from S-coords
"""

import numpy as np
import scipy.signal
import scipy.fft
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class OscillatorySignature:
    """
    Complete oscillatory characterization of a measurement
    
    Attributes:
        frequencies: Dominant frequencies (Hz)
        amplitudes: Amplitudes for each frequency
        phases: Phases for each frequency (radians)
        Q_factors: Quality factors (resonance sharpness)
        power_spectrum: Full power spectrum
        time_signal: Original time-domain signal
        timestamps: Trans-Planckian precision timestamps
    """
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    Q_factors: np.ndarray
    power_spectrum: np.ndarray
    frequency_axis: np.ndarray
    time_signal: np.ndarray
    timestamps: np.ndarray
    
    @property
    def dominant_frequency(self) -> float:
        """Get dominant (highest amplitude) frequency"""
        return self.frequencies[np.argmax(self.amplitudes)]
        
    @property
    def n_harmonics(self) -> int:
        """Number of extracted harmonics"""
        return len(self.frequencies)


class FFTEngine:
    """
    High-performance FFT analysis with windowing and zero-padding
    """
    
    WINDOWS = {
        'hann': scipy.signal.windows.hann,
        'hamming': scipy.signal.windows.hamming,
        'blackman': scipy.signal.windows.blackman,
        'bartlett': scipy.signal.windows.bartlett,
        'none': None
    }
    
    def __init__(self, window_type: str = 'hann', zero_padding_factor: int = 4):
        """
        Initialize FFT engine
        
        Args:
            window_type: Window function ('hann', 'hamming', 'blackman', 'bartlett', 'none')
            zero_padding_factor: Zero-padding multiplier for better freq resolution
        """
        self.window_type = window_type
        self.zero_padding_factor = zero_padding_factor
        
    def analyze(self, signal: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Perform FFT analysis on signal
        
        Args:
            signal: Time-domain signal
            timestamps: Trans-Planckian precision timestamps
            
        Returns:
            Dictionary with FFT results
        """
        n_samples = len(signal)
        
        # Calculate sample rate
        dt = np.mean(np.diff(timestamps))
        sample_rate = 1.0 / dt
        
        # Apply window to reduce spectral leakage
        if self.window_type != 'none' and self.window_type in self.WINDOWS:
            window = self.WINDOWS[self.window_type](n_samples)
            signal_windowed = signal * window
        else:
            signal_windowed = signal
            
        # Zero-pad for better frequency resolution
        n_fft = 2**int(np.ceil(np.log2(n_samples))) * self.zero_padding_factor
        
        # FFT (using scipy for better performance)
        fft = scipy.fft.fft(signal_windowed, n=n_fft)
        freqs = scipy.fft.fftfreq(n_fft, dt)
        
        # Only positive frequencies
        positive_idx = freqs > 0
        freqs_positive = freqs[positive_idx]
        fft_positive = fft[positive_idx]
        
        # Calculate magnitudes and phases
        magnitudes = np.abs(fft_positive)
        phases = np.angle(fft_positive)
        
        # Power spectrum
        power_spectrum = magnitudes**2
        
        return {
            'frequencies': freqs_positive,
            'complex_amplitudes': fft_positive,
            'magnitudes': magnitudes,
            'phases': phases,
            'power_spectrum': power_spectrum,
            'sample_rate': sample_rate,
            'n_fft': n_fft,
            'window_type': self.window_type
        }


class HarmonicExtractor:
    """
    Extract dominant harmonics from FFT spectrum
    """
    
    def __init__(self, 
                 n_harmonics: int = 20,
                 threshold_fraction: float = 0.01,
                 min_peak_distance: int = 10):
        """
        Initialize harmonic extractor
        
        Args:
            n_harmonics: Maximum number of harmonics to extract
            threshold_fraction: Minimum peak height (fraction of max)
            min_peak_distance: Minimum distance between peaks (samples)
        """
        self.n_harmonics = n_harmonics
        self.threshold_fraction = threshold_fraction
        self.min_peak_distance = min_peak_distance
        
    def extract(self, fft_result: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract harmonics from FFT result
        
        Args:
            fft_result: Output from FFTEngine.analyze()
            
        Returns:
            Dictionary with harmonic data
        """
        freqs = fft_result['frequencies']
        power = fft_result['power_spectrum']
        complex_amps = fft_result['complex_amplitudes']
        
        # Find peaks in power spectrum
        threshold = np.max(power) * self.threshold_fraction
        peaks, properties = scipy.signal.find_peaks(
            power,
            height=threshold,
            distance=self.min_peak_distance
        )
        
        if len(peaks) == 0:
            # No peaks found - return DC component
            return {
                'frequencies': np.array([0.0]),
                'amplitudes': np.array([np.max(power)]),
                'phases': np.array([0.0]),
                'Q_factors': np.array([1.0])
            }
        
        # Sort peaks by power (descending)
        sorted_idx = np.argsort(power[peaks])[::-1][:self.n_harmonics]
        peak_idx = peaks[sorted_idx]
        
        # Extract harmonic data
        harm_freqs = freqs[peak_idx]
        harm_amps = np.abs(complex_amps[peak_idx])
        harm_phases = np.angle(complex_amps[peak_idx])
        
        # Estimate Q-factors (resonance sharpness)
        Q_factors = self._estimate_Q_factors(freqs, power, peak_idx)
        
        return {
            'frequencies': harm_freqs,
            'amplitudes': harm_amps,
            'phases': harm_phases,
            'Q_factors': Q_factors
        }
        
    def _estimate_Q_factors(self, 
                           freqs: np.ndarray,
                           power: np.ndarray,
                           peak_idx: np.ndarray) -> np.ndarray:
        """
        Estimate Q-factor for each peak: Q = f_0 / Δf (FWHM)
        """
        Q_factors = []
        
        for idx in peak_idx:
            f0 = freqs[idx]
            P0 = power[idx]
            
            # Find half-power points (FWHM)
            half_power = P0 / 2.0
            
            # Search left
            left_idx = idx
            while left_idx > 0 and power[left_idx] > half_power:
                left_idx -= 1
                
            # Search right
            right_idx = idx
            while right_idx < len(power) - 1 and power[right_idx] > half_power:
                right_idx += 1
                
            f_left = freqs[left_idx]
            f_right = freqs[right_idx]
            delta_f = f_right - f_left
            
            if delta_f > 0:
                Q = f0 / delta_f
            else:
                Q = 1000.0  # Very sharp peak (high Q)
                
            Q_factors.append(Q)
            
        return np.array(Q_factors)


class PhaseAnalyzer:
    """
    Analyze phase relationships between harmonics
    """
    
    def analyze(self, harmonics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze phase relationships
        
        Args:
            harmonics: Output from HarmonicExtractor.extract()
            
        Returns:
            Phase analysis results
        """
        phases = harmonics['phases']
        frequencies = harmonics['frequencies']
        
        if len(phases) < 2:
            return {
                'relative_phases': phases,
                'phase_coherence': 1.0,
                'phase_locked_pairs': []
            }
        
        # Relative phases (referenced to first harmonic)
        relative_phases = phases - phases[0]
        
        # Wrap to [-π, π]
        relative_phases = np.arctan2(
            np.sin(relative_phases),
            np.cos(relative_phases)
        )
        
        # Check for phase-locked pairs (phase difference < 0.1 rad)
        phase_locked_pairs = []
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                phase_diff = np.abs(relative_phases[i] - relative_phases[j])
                if phase_diff < 0.1:  # Phase locked
                    phase_locked_pairs.append((i, j, phase_diff))
                    
        # Calculate phase coherence (0-1)
        # High coherence = all phases near 0 or π
        phase_variance = np.var(np.abs(relative_phases))
        phase_coherence = np.exp(-phase_variance)
        
        return {
            'relative_phases': relative_phases,
            'phase_coherence': phase_coherence,
            'phase_locked_pairs': phase_locked_pairs
        }


class OscillatoryAnalysisEngine:
    """
    Complete oscillatory analysis pipeline (Pathway 1)
    
    Raw Signal → FFT → Harmonics → Phase → Oscillatory Signature
    """
    
    def __init__(self, 
                 fft_window: str = 'hann',
                 n_harmonics: int = 20):
        """
        Initialize analysis engine
        
        Args:
            fft_window: Window function for FFT
            n_harmonics: Number of harmonics to extract
        """
        self.fft_engine = FFTEngine(window_type=fft_window)
        self.harmonic_extractor = HarmonicExtractor(n_harmonics=n_harmonics)
        self.phase_analyzer = PhaseAnalyzer()
        
    def analyze(self, 
                signal: np.ndarray,
                timestamps: np.ndarray) -> OscillatorySignature:
        """
        Perform complete oscillatory analysis
        
        Args:
            signal: Time-domain signal
            timestamps: Trans-Planckian precision timestamps
            
        Returns:
            OscillatorySignature object
        """
        # Phase 1: FFT
        fft_result = self.fft_engine.analyze(signal, timestamps)
        
        # Phase 2: Extract harmonics
        harmonics = self.harmonic_extractor.extract(fft_result)
        
        # Phase 3: Phase analysis
        phase_analysis = self.phase_analyzer.analyze(harmonics)
        
        # Create oscillatory signature
        signature = OscillatorySignature(
            frequencies=harmonics['frequencies'],
            amplitudes=harmonics['amplitudes'],
            phases=phase_analysis['relative_phases'],
            Q_factors=harmonics['Q_factors'],
            power_spectrum=fft_result['power_spectrum'],
            frequency_axis=fft_result['frequencies'],
            time_signal=signal,
            timestamps=timestamps
        )
        
        return signature
        
    def analyze_with_details(self,
                            signal: np.ndarray,
                            timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Perform analysis with detailed intermediate results
        
        Returns:
            Dictionary with signature + all intermediate results
        """
        # Phase 1: FFT
        fft_result = self.fft_engine.analyze(signal, timestamps)
        
        # Phase 2: Extract harmonics
        harmonics = self.harmonic_extractor.extract(fft_result)
        
        # Phase 3: Phase analysis
        phase_analysis = self.phase_analyzer.analyze(harmonics)
        
        # Create signature
        signature = OscillatorySignature(
            frequencies=harmonics['frequencies'],
            amplitudes=harmonics['amplitudes'],
            phases=phase_analysis['relative_phases'],
            Q_factors=harmonics['Q_factors'],
            power_spectrum=fft_result['power_spectrum'],
            frequency_axis=fft_result['frequencies'],
            time_signal=signal,
            timestamps=timestamps
        )
        
        return {
            'signature': signature,
            'fft_result': fft_result,
            'harmonics': harmonics,
            'phase_analysis': phase_analysis
        }


def extract_oscillatory_signature(signal: np.ndarray,
                                  timestamps: np.ndarray,
                                  n_harmonics: int = 20) -> OscillatorySignature:
    """
    Convenience function: extract oscillatory signature from signal
    
    Args:
        signal: Time-domain signal
        timestamps: Trans-Planckian timestamps
        n_harmonics: Number of harmonics to extract
        
    Returns:
        OscillatorySignature
    """
    engine = OscillatoryAnalysisEngine(n_harmonics=n_harmonics)
    return engine.analyze(signal, timestamps)