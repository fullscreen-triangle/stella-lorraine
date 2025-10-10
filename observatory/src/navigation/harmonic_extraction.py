"""
Harmonic Multiplication Precision Enhancement
==============================================
Molecular harmonics provide precision multiplication through integer frequency relationships.
"""

import numpy as np
from typing import List, Tuple, Dict


class HarmonicExtractor:
    """
    Extracts and analyzes molecular harmonics for precision multiplication.
    Each harmonic n provides Δt_n = Δt_fundamental / n precision.
    """

    def __init__(self, fundamental_frequency: float = 7.1e13):
        """
        Initialize with fundamental molecular frequency

        Args:
            fundamental_frequency: Base frequency (Hz), default = 71 THz for N2
        """
        self.fundamental_freq = fundamental_frequency
        self.fundamental_period = 1.0 / fundamental_frequency

    def extract_harmonics(self, signal: np.ndarray, time_points: np.ndarray,
                         max_harmonic: int = 150) -> Dict:
        """
        Extract harmonic series from signal

        Args:
            signal: Time-domain signal
            time_points: Corresponding time values
            max_harmonic: Maximum harmonic number to extract

        Returns:
            Dictionary with harmonic frequencies, amplitudes, and precisions
        """
        # FFT to frequency domain
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(time_points), time_points[1] - time_points[0])
        magnitude = np.abs(fft_result)

        # Find harmonics (integer multiples of fundamental)
        harmonics = []
        for n in range(1, max_harmonic + 1):
            target_freq = n * self.fundamental_freq

            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - target_freq))

            if freqs[idx] > 0:  # Only positive frequencies
                harmonic_data = {
                    'number': n,
                    'frequency': freqs[idx],
                    'ideal_frequency': target_freq,
                    'amplitude': magnitude[idx],
                    'phase': np.angle(fft_result[idx]),
                    'precision': self.fundamental_period / n,
                    'precision_fs': (self.fundamental_period / n) * 1e15,
                    'precision_as': (self.fundamental_period / n) * 1e18
                }
                harmonics.append(harmonic_data)

        return {
            'fundamental': {
                'frequency': self.fundamental_freq,
                'period': self.fundamental_period,
                'precision': self.fundamental_period
            },
            'harmonics': harmonics,
            'max_harmonic': max_harmonic,
            'total_harmonics_found': len(harmonics)
        }

    def calculate_sub_harmonic_precision(self, harmonic_n: int,
                                        phase_resolution: float = 0.001) -> float:
        """
        Calculate sub-harmonic precision using phase-coherent detection

        Args:
            harmonic_n: Harmonic number
            phase_resolution: Phase resolution (δn/n), typically 0.001

        Returns:
            Effective precision in seconds
        """
        # Harmonic precision
        harmonic_precision = self.fundamental_period / harmonic_n

        # Sub-harmonic resolution from phase coherence
        sub_harmonic_factor = 1.0 / phase_resolution

        # Effective precision
        effective_precision = harmonic_precision / sub_harmonic_factor

        return effective_precision

    def find_optimal_harmonic(self, harmonics_data: Dict,
                            coherence_time: float = 741e-15) -> Dict:
        """
        Find optimal harmonic for maximum precision within coherence time

        Args:
            harmonics_data: Results from extract_harmonics()
            coherence_time: LED-enhanced coherence time (seconds)

        Returns:
            Optimal harmonic data
        """
        harmonics = harmonics_data['harmonics']

        # Filter by amplitude (must be detectable)
        amplitudes = [h['amplitude'] for h in harmonics]
        threshold = 0.1 * np.max(amplitudes)
        detectable = [h for h in harmonics if h['amplitude'] > threshold]

        if not detectable:
            return None

        # Filter by coherence time (period must be shorter than decoherence)
        coherent = [h for h in detectable
                   if 1.0/h['frequency'] < coherence_time]

        if not coherent:
            # Use highest detectable harmonic
            optimal = max(detectable, key=lambda h: h['number'])
        else:
            # Use highest coherent harmonic
            optimal = max(coherent, key=lambda h: h['number'])

        # Calculate sub-harmonic precision
        sub_harmonic_precision = self.calculate_sub_harmonic_precision(optimal['number'])

        return {
            **optimal,
            'sub_harmonic_precision': sub_harmonic_precision,
            'sub_harmonic_precision_as': sub_harmonic_precision * 1e18,
            'total_enhancement': self.fundamental_period / sub_harmonic_precision
        }

    def precision_cascade(self, max_harmonic: int = 150,
                         sub_harmonic_resolution: float = 0.001) -> List[Dict]:
        """
        Calculate precision cascade through harmonic series

        Returns:
            List of precision values for each harmonic level
        """
        cascade = []

        for n in range(1, max_harmonic + 1):
            # Direct harmonic precision
            harmonic_precision = self.fundamental_period / n

            # Sub-harmonic precision
            sub_harmonic_precision = self.calculate_sub_harmonic_precision(
                n, sub_harmonic_resolution
            )

            cascade.append({
                'harmonic': n,
                'frequency': n * self.fundamental_freq,
                'period': 1.0 / (n * self.fundamental_freq),
                'harmonic_precision': harmonic_precision,
                'sub_harmonic_precision': sub_harmonic_precision,
                'precision_fs': sub_harmonic_precision * 1e15,
                'precision_as': sub_harmonic_precision * 1e18
            })

        return cascade


def demonstrate_harmonic_precision():
    """Demonstrate harmonic precision multiplication"""

    print("=" * 70)
    print("   HARMONIC PRECISION MULTIPLICATION")
    print("=" * 70)

    # Create extractor for N2
    extractor = HarmonicExtractor(fundamental_frequency=7.1e13)

    print(f"\n📊 Fundamental Frequency:")
    print(f"   ν_0 = {extractor.fundamental_freq:.2e} Hz (71 THz)")
    print(f"   τ_0 = {extractor.fundamental_period*1e15:.2f} fs")

    # Generate test signal with harmonics
    duration = 100 * extractor.fundamental_period  # 100 cycles
    n_samples = 2**14  # 16384 samples
    time_points = np.linspace(0, duration, n_samples)

    # Signal with multiple harmonics
    signal = np.zeros(n_samples)
    harmonic_numbers = [1, 10, 50, 100, 150]
    for n in harmonic_numbers:
        amplitude = 1.0 / n  # Decreasing amplitude
        signal += amplitude * np.sin(2*np.pi * n * extractor.fundamental_freq * time_points)

    # Extract harmonics
    print(f"\n🔬 Extracting harmonics from signal...")
    harmonics_data = extractor.extract_harmonics(signal, time_points, max_harmonic=150)

    print(f"   Total harmonics found: {harmonics_data['total_harmonics_found']}")

    # Show key harmonics
    print(f"\n🎵 Key Harmonics:")
    key_harmonics = [h for h in harmonics_data['harmonics']
                    if h['number'] in [1, 10, 50, 100, 150]]

    for h in key_harmonics:
        print(f"   n={h['number']:3d}: "
              f"ν = {h['frequency']:.2e} Hz, "
              f"Δt = {h['precision_fs']:.2f} fs, "
              f"amplitude = {h['amplitude']:.2e}")

    # Find optimal harmonic
    optimal = extractor.find_optimal_harmonic(harmonics_data, coherence_time=741e-15)

    print(f"\n⚡ Optimal Harmonic (within coherence time):")
    print(f"   Harmonic number: {optimal['number']}")
    print(f"   Frequency: {optimal['frequency']:.2e} Hz")
    print(f"   Direct precision: {optimal['precision_as']:.1f} as")
    print(f"   Sub-harmonic precision: {optimal['sub_harmonic_precision_as']:.1f} as")
    print(f"   Total enhancement: {optimal['total_enhancement']:.0f}×")

    # Precision cascade
    print(f"\n📈 Precision Cascade (selected harmonics):")
    cascade = extractor.precision_cascade(max_harmonic=150, sub_harmonic_resolution=0.001)

    for n in [1, 10, 50, 100, 150]:
        level = cascade[n-1]
        if level['precision_as'] >= 1:
            print(f"   n={n:3d}: {level['precision_as']:6.1f} as")
        else:
            print(f"   n={n:3d}: {level['precision_as']*1000:6.1f} zs")

    # Ultimate precision
    ultimate = cascade[-1]  # Highest harmonic
    print(f"\n✨ ULTIMATE PRECISION (n=150 with sub-harmonic):")
    print(f"   {ultimate['precision_as']:.1f} attoseconds")
    print(f"   = {ultimate['precision_as']*1000:.1f} zeptoseconds")
    print(f"   {ultimate['precision_fs']*1e-3:.0f}× better than fundamental!")

    return extractor, harmonics_data, optimal


if __name__ == "__main__":
    extractor, harmonics, optimal = demonstrate_harmonic_precision()
