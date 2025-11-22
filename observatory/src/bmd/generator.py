import nump as np


class SourceWave:
    """Fundamental reality oscillation"""
    def __init__(self, omega_0=1e13, amplitude=1.0):
        self.omega_0 = omega_0  # 10^13 Hz (your framework)
        self.amplitude = amplitude

    def psi(self, x, t):
        """Wave function at position x, time t"""
        return self.amplitude * np.exp(1j * (self.omega_0 * t - self.k * x))


class InterferencePattern:
    """Pattern created by object-wave interaction"""
    def __init__(self, source_wave, object_properties):
        self.source = source_wave
        self.object = object_properties
        self.pattern = self._compute_interference()

    def _compute_interference(self):
        """Compute interference pattern"""
        # Scattered wave from object
        psi_scattered = self._scatter_wave()
        # Total = source + scattered
        return self.source.psi + psi_scattered


class CategoryExtractor:
    """Extract stable patterns (categories) from interference"""
    def extract_categories(self, interference_pattern):
        """Find stable nodes/antinodes in pattern"""
        # Fourier analysis to find dominant frequencies
        fft = np.fft.fft(interference_pattern)
        # Peaks = stable patterns = categories
        categories = self._find_peaks(fft)
        return categories

class CascadingInterference:
    """Multiple objects creating layered interference"""
    def __init__(self, source_wave):
        self.source = source_wave
        self.objects = []
        self.patterns = []

    def add_object(self, obj):
        """Add object and compute new interference"""
        # Previous pattern becomes input for next
        input_wave = self.patterns[-1] if self.patterns else self.source
        # New interference
        new_pattern = InterferencePattern(input_wave, obj)
        self.patterns.append(new_pattern)

