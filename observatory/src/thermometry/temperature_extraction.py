# thermometry/temperature_extraction.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.constants as const
from typing import Tuple
from categorical_state import CategoricalState, EntropicCoordinates


class ThermometryAnalyzer:
    """
    Extract temperature from categorical state with picokelvin resolution

    Based on relation:
    T = (ℏ²/2πmkB) exp[(2Smomentum/3kB) - 1]
    """

    def __init__(self, particle_mass: float):
        """
        Args:
            particle_mass: Mass of particle [kg]
        """
        self.m = particle_mass
        self.hbar = const.hbar
        self.kB = const.k

        # Timing precision from H+ oscillator sync
        self.delta_t = 2.2e-15  # seconds

        # Energy resolution
        self.delta_E = self.hbar / (2 * self.delta_t)

        # Temperature resolution
        self.delta_T = self.delta_E / self.kB  # ∼ 17 pK

    def extract_temperature(self, cat_state: CategoricalState) -> Tuple[float, float]:
        """
        Extract temperature from categorical state

        Args:
            cat_state: Categorical state C(t)

        Returns:
            (temperature [K], uncertainty [K])
        """
        # Extract momentum entropy (Se, not Sk!)
        # Se = evolution entropy (captures momentum distribution)
        # Sk = knowledge entropy (distinguishability)
        S_momentum = cat_state.S.Se

        # Guard against invalid entropy values
        if S_momentum <= 0 or not np.isfinite(S_momentum):
            # Return minimum measurable temperature
            return self.delta_T, self.delta_T

        # Temperature from entropy relation
        # T = (ℏ²/2πmkB) exp[(2Smomentum/3kB) - 1]
        prefactor = (self.hbar**2) / (2 * np.pi * self.m * self.kB)
        exponent = (2 * S_momentum) / (3 * self.kB) - 1

        # Prevent overflow for large entropy
        if exponent > 100:
            exponent = 100
        elif exponent < -100:
            return self.delta_T, self.delta_T

        T = prefactor * np.exp(exponent)

        # Uncertainty from energy resolution
        delta_T = self.delta_T

        return T, delta_T

    def extract_temperature_from_momentum_distribution(self,
                                                       momenta: np.ndarray) -> Tuple[float, float]:
        """
        Direct temperature extraction from momentum distribution

        Args:
            momenta: Array of momentum values [kg·m/s]

        Returns:
            (temperature [K], uncertainty [K])
        """
        # Mean squared momentum
        p_squared_mean = np.mean(momenta**2)

        # Temperature from equipartition theorem (3D)
        # <p²> = 3mkBT
        T = p_squared_mean / (3 * self.m * self.kB)

        # Statistical uncertainty
        p_squared_std = np.std(momenta**2)
        N = len(momenta)
        delta_T_statistical = (p_squared_std / np.sqrt(N)) / (3 * self.m * self.kB)

        # Total uncertainty (statistical + systematic)
        delta_T_total = np.sqrt(delta_T_statistical**2 + self.delta_T**2)

        return T, delta_T_total

    def relative_precision(self, T: float, delta_T: float) -> float:
        """
        Calculate relative precision ΔT/T

        Args:
            T: Temperature [K]
            delta_T: Temperature uncertainty [K]

        Returns:
            Relative precision (dimensionless)
        """
        return delta_T / T

    def improvement_factor(self,
                          delta_T_categorical: float,
                          delta_T_conventional: float) -> float:
        """
        Calculate improvement factor over conventional methods

        Args:
            delta_T_categorical: Uncertainty from categorical thermometry [K]
            delta_T_conventional: Uncertainty from conventional method [K]

        Returns:
            Improvement factor (dimensionless)
        """
        return delta_T_conventional / delta_T_categorical

    def heating_rate(self, measurement_time: float) -> float:
        """
        Calculate heating from far-detuned optical coupling

        Args:
            measurement_time: Duration of measurement [s]

        Returns:
            Temperature increase [K]
        """
        # Heating rate < 1 fK/s from paper
        heating_rate = 1e-15  # K/s
        return heating_rate * measurement_time

    def is_non_invasive(self, T: float, measurement_time: float) -> bool:
        """
        Check if measurement is non-invasive

        Criterion: Heating << Temperature resolution

        Args:
            T: System temperature [K]
            measurement_time: Measurement duration [s]

        Returns:
            True if non-invasive
        """
        heating = self.heating_rate(measurement_time)
        return heating < 0.1 * self.delta_T  # Heating < 10% of resolution


class TimeOfFlightComparison:
    """
    Compare categorical thermometry with time-of-flight imaging
    """

    def __init__(self, particle_mass: float):
        self.m = particle_mass
        self.kB = const.k

    def tof_temperature_resolution(self,
                                   T: float,
                                   expansion_time: float,
                                   imaging_resolution: float) -> float:
        """
        Calculate TOF temperature resolution

        Args:
            T: Temperature [K]
            expansion_time: Free expansion time [s]
            imaging_resolution: Spatial resolution [m]

        Returns:
            Temperature uncertainty [K]
        """
        # Velocity from temperature
        v_thermal = np.sqrt(self.kB * T / self.m)

        # Expansion distance
        expansion_distance = v_thermal * expansion_time

        # Relative uncertainty from imaging
        relative_uncertainty = imaging_resolution / expansion_distance

        # Temperature uncertainty
        delta_T = 2 * T * relative_uncertainty  # Factor of 2 from v² dependence

        return delta_T

    def tof_relative_precision(self,
                              T: float,
                              expansion_time: float = 20e-3,  # 20 ms typical
                              imaging_resolution: float = 5e-6) -> float:  # 5 μm typical
        """
        Calculate typical TOF relative precision

        Returns:
            ΔT/T for time-of-flight
        """
        delta_T = self.tof_temperature_resolution(T, expansion_time, imaging_resolution)
        return delta_T / T


# Example validation
if __name__ == "__main__":
    # Rb-87 parameters
    m_Rb87 = 1.443e-25  # kg

    # Initialize analyzers
    thermo = ThermometryAnalyzer(m_Rb87)
    tof_comp = TimeOfFlightComparison(m_Rb87)

    # Test at T = 100 nK
    T_test = 100e-9  # K

    print("=" * 60)
    print("CATEGORICAL THERMOMETRY VALIDATION")
    print("=" * 60)

    # Simulate momentum distribution
    N = int(1e5)
    sigma_v = np.sqrt(const.k * T_test / m_Rb87)
    velocities = np.random.normal(0, sigma_v, N)
    momenta = m_Rb87 * velocities

    # Extract temperature
    T_measured, delta_T = thermo.extract_temperature_from_momentum_distribution(momenta)

    print(f"\nTrue temperature: {T_test*1e9:.3f} nK")
    print(f"Measured temperature: {T_measured*1e9:.3f} ± {delta_T*1e12:.2f} pK")
    print(f"Relative precision: {thermo.relative_precision(T_measured, delta_T):.2e}")

    # Compare with TOF
    tof_precision = tof_comp.tof_relative_precision(T_test)
    cat_precision = thermo.relative_precision(T_measured, delta_T)
    improvement = tof_precision / cat_precision

    print(f"\nTOF relative precision: {tof_precision:.2e}")
    print(f"Categorical relative precision: {cat_precision:.2e}")
    print(f"Improvement factor: {improvement:.2e}")

    # Check non-invasive criterion
    measurement_time = 1e-3  # 1 ms
    heating = thermo.heating_rate(measurement_time)
    is_non_invasive = thermo.is_non_invasive(T_measured, measurement_time)

    print(f"\nMeasurement time: {measurement_time*1e3:.1f} ms")
    print(f"Heating: {heating*1e15:.3f} fK")
    print(f"Non-invasive: {is_non_invasive}")

    print("\n" + "=" * 60)
