# interferometry/phase_correlation.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.constants as const
from typing import Tuple, List
from dataclasses import dataclass
from categorical_state import CategoricalState, EntropicCoordinates
from oscillator_synchronization import HydrogenOscillatorSync


@dataclass
class PhaseCorrelation:
    """Phase correlation between two stations"""
    baseline_vector: np.ndarray  # [m]
    phase_difference: float  # [rad]
    correlation_coefficient: float  # [0, 1]
    timestamp: float  # [s]

    def baseline_length(self) -> float:
        """Baseline length [m]"""
        return np.linalg.norm(self.baseline_vector)

    def angular_resolution(self, wavelength: float) -> float:
        """Angular resolution [rad]"""
        return wavelength / self.baseline_length()


class CategoricalPhaseAnalyzer:
    """
    Phase correlation in categorical space for trans-Planckian baselines

    Phase propagates via categorical prediction with effective velocity
    vcat/c ∈ [2.846, 65.71], maintaining coherence independent of baseline.
    """

    def __init__(self, wavelength: float):
        """
        Args:
            wavelength: Observation wavelength [m]
        """
        self.lambda_ = wavelength
        self.k = 2 * np.pi / wavelength  # Wave vector
        self.f = const.c / wavelength  # Frequency

        # Categorical propagation velocity range
        self.v_cat_min = 2.846 * const.c
        self.v_cat_max = 65.71 * const.c

        # Synchronization
        self.sync = HydrogenOscillatorSync()

    def extract_phase_from_categorical_state(self,
                                             cat_state: CategoricalState) -> float:
        """
        Extract phase information from categorical state

        Phase encoded in temporal entropy St

        Args:
            cat_state: Categorical state C(t)

        Returns:
            Phase [rad]
        """
        # Phase from temporal entropy
        # St encodes timing information → phase
        St = cat_state.S.St

        # Convert entropy to phase
        # φ = ωt where t ~ exp(St/kB)
        t_effective = np.exp(St / const.k) * self.sync.delta_t
        phase = (2 * np.pi * self.f * t_effective) % (2 * np.pi)

        return phase

    def correlate_phases(self,
                        cat_state_1: CategoricalState,
                        cat_state_2: CategoricalState,
                        baseline: np.ndarray,
                        source_direction: np.ndarray) -> PhaseCorrelation:
        """
        Correlate phases between two stations in categorical space

        Args:
            cat_state_1: Categorical state at station 1
            cat_state_2: Categorical state at station 2
            baseline: Baseline vector [m] (station 2 - station 1)
            source_direction: Unit vector toward source

        Returns:
            PhaseCorrelation
        """
        # Extract phases
        phi_1 = self.extract_phase_from_categorical_state(cat_state_1)
        phi_2 = self.extract_phase_from_categorical_state(cat_state_2)

        # Geometric delay
        geometric_delay = np.dot(baseline, source_direction) / const.c
        geometric_phase = self.k * np.dot(baseline, source_direction)

        # Phase difference (corrected for geometry)
        delta_phi = (phi_2 - phi_1 - geometric_phase) % (2 * np.pi)

        # Correlation coefficient from entropic similarity
        S1_total = cat_state_1.S.total_entropy()
        S2_total = cat_state_2.S.total_entropy()

        # Correlation ~ exp(-|ΔS|/kB)
        delta_S = abs(S2_total - S1_total)
        correlation = np.exp(-delta_S / const.k)

        # Timestamp (average)
        timestamp = 0.5 * (cat_state_1.t + cat_state_2.t)

        return PhaseCorrelation(
            baseline_vector=baseline,
            phase_difference=delta_phi,
            correlation_coefficient=correlation,
            timestamp=timestamp
        )

    def categorical_coherence_length(self, baseline_length: float) -> float:
        """
        Calculate coherence length in categorical space

        Unlike physical space (limited by atmosphere), categorical
        coherence is maintained via vcat propagation.

        Args:
            baseline_length: Physical baseline [m]

        Returns:
            Effective coherence length [m]
        """
        # Categorical propagation maintains coherence
        # Coherence length ~ vcat * τcoherence
        tau_coherence = 1.0 / self.f  # Coherence time ~ period

        # Use minimum vcat for conservative estimate
        L_coherence = self.v_cat_min * tau_coherence

        return L_coherence

    def is_baseline_coherent(self, baseline_length: float) -> bool:
        """
        Check if baseline maintains coherence

        Args:
            baseline_length: Baseline length [m]

        Returns:
            True if coherent
        """
        L_coh = self.categorical_coherence_length(baseline_length)
        return baseline_length < L_coh

    def atmospheric_immunity(self,
                            baseline_length: float,
                            r0_fried: float = 0.1) -> float:
        """
        Calculate atmospheric immunity factor

        Categorical phase correlation is immune to atmospheric
        turbulence since phase propagates in categorical space,
        not physical space.

        Args:
            baseline_length: Baseline [m]
            r0_fried: Fried parameter [m] (atmospheric coherence length)

        Returns:
            Immunity factor (1 = complete immunity)
        """
        # Conventional interferometry: degradation ~ (D/r0)^(-5/3)
        conventional_degradation = (baseline_length / r0_fried)**(-5/3)

        # Categorical interferometry: no atmospheric degradation
        categorical_degradation = 1.0

        # Immunity factor
        immunity = categorical_degradation / max(conventional_degradation, 1e-10)

        return min(immunity, 1.0)


class TransPlanckianInterferometer:
    """
    Trans-Planckian baseline interferometry

    Achieves θ ~ 10⁻⁵ microarcseconds with D = 10⁴ km baselines
    """

    def __init__(self,
                 wavelength: float,
                 station_positions: np.ndarray):
        """
        Args:
            wavelength: Observation wavelength [m]
            station_positions: Array of station positions [m] (N×3)
        """
        self.lambda_ = wavelength
        self.positions = station_positions
        self.N_stations = station_positions.shape[0]

        # Initialize analyzer
        self.analyzer = CategoricalPhaseAnalyzer(wavelength)

        # Compute baselines
        self.baselines = self._compute_baselines()

    def _compute_baselines(self) -> List[np.ndarray]:
        """Compute all baseline vectors"""
        baselines = []
        for i in range(self.N_stations):
            for j in range(i+1, self.N_stations):
                baseline = self.positions[j] - self.positions[i]
                baselines.append(baseline)
        return baselines

    def angular_resolution(self) -> float:
        """
        Calculate angular resolution

        θ_min ≈ λ/D_max

        Returns:
            Angular resolution [rad]
        """
        D_max = max([np.linalg.norm(b) for b in self.baselines])
        theta = self.lambda_ / D_max
        return theta

    def angular_resolution_microarcsec(self) -> float:
        """Angular resolution in microarcseconds"""
        theta_rad = self.angular_resolution()
        theta_arcsec = theta_rad * (180 * 3600 / np.pi)
        theta_microarcsec = theta_arcsec * 1e6
        return theta_microarcsec

    def visibility_function(self,
                           source_direction: np.ndarray,
                           cat_states: List[CategoricalState]) -> np.ndarray:
        """
        Calculate visibility function from phase correlations

        Args:
            source_direction: Unit vector toward source
            cat_states: List of categorical states (one per station)

        Returns:
            Complex visibility for each baseline
        """
        N_baselines = len(self.baselines)
        visibilities = np.zeros(N_baselines, dtype=complex)

        baseline_idx = 0
        for i in range(self.N_stations):
            for j in range(i+1, self.N_stations):
                # Correlate phases
                corr = self.analyzer.correlate_phases(
                    cat_states[i],
                    cat_states[j],
                    self.baselines[baseline_idx],
                    source_direction
                )

                # Complex visibility
                amplitude = corr.correlation_coefficient
                phase = corr.phase_difference
                visibilities[baseline_idx] = amplitude * np.exp(1j * phase)

                baseline_idx += 1

        return visibilities

    def image_reconstruction(self,
                            visibilities: np.ndarray,
                            image_size: int = 256) -> np.ndarray:
        """
        Reconstruct image from visibilities (simplified)

        Args:
            visibilities: Complex visibilities
            image_size: Image dimension [pixels]

        Returns:
            Reconstructed image
        """
        # Simplified inverse Fourier transform
        # Full implementation would use CLEAN or other algorithms

        # Create UV coverage
        uv_plane = np.zeros((image_size, image_size), dtype=complex)

        for i, baseline in enumerate(self.baselines):
            # UV coordinates
            u = baseline[0] / self.lambda_
            v = baseline[1] / self.lambda_

            # Map to image plane
            u_idx = int(u * image_size / (2 * np.max(np.abs([b[0] for b in self.baselines]))))
            v_idx = int(v * image_size / (2 * np.max(np.abs([b[1] for b in self.baselines]))))

            u_idx = (u_idx + image_size // 2) % image_size
            v_idx = (v_idx + image_size // 2) % image_size

            if i < len(visibilities):
                uv_plane[v_idx, u_idx] = visibilities[i]

        # Inverse FFT
        image = np.fft.ifft2(np.fft.ifftshift(uv_plane))
        image = np.abs(image)

        return image

    def exoplanet_detection_capability(self,
                                       planet_distance: float,
                                       planet_radius: float) -> Tuple[float, bool]:
        """
        Assess capability to detect exoplanet

        Args:
            planet_distance: Distance to planet [m]
            planet_radius: Planet radius [m]

        Returns:
            (angular_size [rad], detectable [bool])
        """
        # Angular size of planet
        theta_planet = planet_radius / planet_distance

        # Resolution
        theta_resolution = self.angular_resolution()

        # Detectable if planet > resolution
        detectable = theta_planet > theta_resolution

        return theta_planet, detectable


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("TRANS-PLANCKIAN BASELINE INTERFEROMETRY VALIDATION")
    print("=" * 60)

    # Parameters
    wavelength = 500e-9  # 500 nm (visible)

    # Planetary-scale network
    N_stations = 10
    baseline_scale = 1e7  # 10,000 km

    # Random station positions on Earth surface
    positions = np.random.randn(N_stations, 3)
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= baseline_scale

    # Create interferometer
    interferometer = TransPlanckianInterferometer(wavelength, positions)

    print(f"\nConfiguration:")
    print(f"  Wavelength: {wavelength*1e9:.0f} nm")
    print(f"  Number of stations: {N_stations}")
    print(f"  Maximum baseline: {max([np.linalg.norm(b) for b in interferometer.baselines])/1e3:.1f} km")

    # Angular resolution
    theta_rad = interferometer.angular_resolution()
    theta_microarcsec = interferometer.angular_resolution_microarcsec()

    print(f"\nAngular Resolution:")
    print(f"  θ = {theta_rad:.2e} rad")
    print(f"  θ = {theta_microarcsec:.2e} μas")

    # Compare with paper claim
    paper_claim = 1e-5  # 10⁻⁵ μas
    print(f"  Paper claim: {paper_claim:.2e} μas")
    print(f"  Ratio: {theta_microarcsec / paper_claim:.2f}")

    # Exoplanet detection
    print("\n" + "-" * 60)
    print("EXOPLANET DETECTION CAPABILITY")
    print("-" * 60)

    # Earth-like planet at 10 pc
    planet_distance = 10 * 3.086e16  # 10 parsecs in meters
    planet_radius = 6.371e6  # Earth radius

    theta_planet, detectable = interferometer.exoplanet_detection_capability(
        planet_distance, planet_radius
    )

    print(f"\nEarth-like planet at 10 pc:")
    print(f"  Angular size: {theta_planet * (180*3600/np.pi) * 1e6:.2e} μas")
    print(f"  Resolution: {theta_microarcsec:.2e} μas")
    print(f"  Detectable: {detectable}")

    # Atmospheric immunity
    print("\n" + "-" * 60)
    print("ATMOSPHERIC IMMUNITY")
    print("-" * 60)

    analyzer = CategoricalPhaseAnalyzer(wavelength)
    max_baseline = max([np.linalg.norm(b) for b in interferometer.baselines])

    immunity = analyzer.atmospheric_immunity(max_baseline)
    coherent = analyzer.is_baseline_coherent(max_baseline)

    print(f"\nBaseline: {max_baseline/1e3:.1f} km")
    print(f"Atmospheric immunity factor: {immunity:.6f}")
    print(f"Baseline coherent: {coherent}")
    print(f"Coherence length: {analyzer.categorical_coherence_length(max_baseline)/1e3:.1f} km")
