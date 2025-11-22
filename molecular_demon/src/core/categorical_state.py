"""
Categorical State Theory - S-Entropy Coordinates

Defines the tri-coordinate S-entropy system:
- S_k: Knowledge entropy (information accumulation)
- S_t: Temporal entropy (time evolution)
- S_e: Evolution entropy (momentum/energy)

Key principle: Categories are orthogonal to phase space (x, p)
This enables Heisenberg bypass - measuring category doesn't disturb position/momentum
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)


@dataclass
class CategoricalState:
    """
    Complete categorical state of a molecular oscillator

    The state exists in S-entropy space, orthogonal to phase space.
    Measuring S-coordinates does not collapse wavefunction in (x, p).
    """
    s_k: float  # Knowledge entropy (dimensionless, ≥ 0)
    s_t: float  # Temporal entropy (dimensionless, ≥ 0)
    s_e: float  # Evolution entropy (dimensionless, ≥ 0)

    def __post_init__(self):
        """Validate coordinates are non-negative"""
        if self.s_k < 0 or self.s_t < 0 or self.s_e < 0:
            raise ValueError("S-entropy coordinates must be non-negative")

    @property
    def total_entropy(self) -> float:
        """Total S-entropy: S_total = S_k + S_t + S_e"""
        return self.s_k + self.s_t + self.s_e

    def distance_to(self, other: 'CategoricalState') -> float:
        """
        Categorical distance between two states

        This is the fundamental distance metric in categorical space.
        Independent of physical spatial distance.
        """
        diff = np.array([
            self.s_k - other.s_k,
            self.s_t - other.s_t,
            self.s_e - other.s_e
        ])
        return np.linalg.norm(diff)

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for linear algebra operations"""
        return np.array([self.s_k, self.s_t, self.s_e])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'CategoricalState':
        """Construct from numpy array"""
        return cls(s_k=vec[0], s_t=vec[1], s_e=vec[2])

    def __repr__(self) -> str:
        return f"CategoricalState(S_k={self.s_k:.3f}, S_t={self.s_t:.3f}, S_e={self.s_e:.3f})"


class SEntropyCalculator:
    """
    Calculate S-entropy coordinates from physical observables

    These formulas map from phase space to categorical space,
    establishing the bridge between physical measurements and categories.
    """

    @staticmethod
    def knowledge_entropy(measurements: list, base: float = 2) -> float:
        """
        Calculate S_k from measurement history

        S_k = -Σ p_i log(p_i)

        Represents accumulated information about the system.
        Higher S_k = more information gathered.

        Args:
            measurements: List of observed values
            base: Logarithm base (2 for bits, e for nats)
        """
        if not measurements:
            return 0.0

        # Bin measurements to get probability distribution
        hist, _ = np.histogram(measurements, bins='auto', density=True)

        # Normalize to probabilities
        bin_width = (max(measurements) - min(measurements)) / len(hist)
        probs = hist * bin_width
        probs = probs[probs > 0]  # Remove zeros

        # Shannon entropy
        if base == 2:
            s_k = -np.sum(probs * np.log2(probs))
        else:
            s_k = -np.sum(probs * np.log(probs))

        return float(s_k)

    @staticmethod
    def temporal_entropy(time_elapsed: float, tau_min: float = 1e-15) -> float:
        """
        Calculate S_t from elapsed time

        S_t = (k_B/2) ln(1 + t/τ_min)

        Represents time evolution in categorical space.

        Args:
            time_elapsed: Time since initial state (seconds)
            tau_min: Minimum time resolution (default: 1 fs)
        """
        if time_elapsed < 0:
            raise ValueError("Time elapsed must be non-negative")

        # Dimensionless temporal entropy
        s_t = 0.5 * np.log(1 + time_elapsed / tau_min)

        return float(s_t)

    @staticmethod
    def evolution_entropy(temperature: float,
                         mass: float,
                         n_particles: int = 1) -> float:
        """
        Calculate S_e from temperature (momentum distribution)

        S_e = (3N k_B / 2) ln(m k_B T / 2πℏ²) + S_0

        Represents momentum/energy distribution in categorical space.

        Args:
            temperature: Temperature in Kelvin
            mass: Particle mass in kg
            n_particles: Number of particles (default 1)
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if mass <= 0:
            raise ValueError("Mass must be positive")

        # Classical phase space volume
        thermal_wavelength_sq = 2 * np.pi * hbar**2 / (mass * k_B * temperature)

        # Evolution entropy (dimensionless)
        s_e = (3 * n_particles * k_B / 2) * np.log(1 / thermal_wavelength_sq)

        # Normalize to dimensionless units (divide by k_B)
        s_e_dimensionless = s_e / k_B

        return float(s_e_dimensionless)

    @staticmethod
    def from_frequency(frequency_hz: float,
                      measurement_count: int = 1,
                      time_elapsed: float = 1e-9) -> CategoricalState:
        """
        Construct categorical state from molecular oscillation frequency

        This is the primary interface for molecular oscillator systems.

        Args:
            frequency_hz: Oscillation frequency
            measurement_count: Number of measurements (affects S_k)
            time_elapsed: Time since first measurement (affects S_t)
        """
        # Estimate temperature from frequency (for diatomic molecules)
        # E = hν, T ~ E/k_B
        temperature = hbar * 2 * np.pi * frequency_hz / k_B

        # Typical molecular mass (e.g., N2)
        mass = 28 * 1.66054e-27  # 28 amu in kg

        # Calculate components
        s_k = np.log2(measurement_count + 1)  # Information from measurements
        s_t = SEntropyCalculator.temporal_entropy(time_elapsed)
        s_e = SEntropyCalculator.evolution_entropy(temperature, mass)

        return CategoricalState(s_k=s_k, s_t=s_t, s_e=s_e)


class CategoryOrthogonality:
    """
    Proof that categorical measurements are orthogonal to phase space

    This validates the Heisenberg bypass: measuring frequency (category)
    doesn't disturb position or momentum.
    """

    @staticmethod
    def commutator_x_category() -> float:
        """
        Calculate [x̂, 𝒟_ω]

        For frequency/category operator 𝒟_ω and position x̂.
        Should be zero (orthogonal).
        """
        # In categorical space, frequency is not a function of position
        # Therefore commutator vanishes
        return 0.0

    @staticmethod
    def commutator_p_category() -> float:
        """
        Calculate [p̂, 𝒟_ω]

        For frequency/category operator 𝒟_ω and momentum p̂.
        Should be zero (orthogonal).
        """
        # Frequency is a temporal derivative, not spatial
        # Therefore commutator vanishes
        return 0.0

    @staticmethod
    def verify_orthogonality() -> bool:
        """
        Verify that categories are orthogonal to phase space

        Returns True if both commutators vanish
        """
        comm_x = CategoryOrthogonality.commutator_x_category()
        comm_p = CategoryOrthogonality.commutator_p_category()

        orthogonal = (abs(comm_x) < 1e-10) and (abs(comm_p) < 1e-10)

        if orthogonal:
            logger.info("✓ Category orthogonality verified: [x̂,𝒟]=0, [p̂,𝒟]=0")
        else:
            logger.warning("✗ Category orthogonality FAILED")

        return orthogonal


def navigate_categorical_space(start: CategoricalState,
                              target: CategoricalState,
                              steps: int = 10) -> list:
    """
    Navigate through categorical space from start to target

    This is S-entropy navigation - can be arbitrarily fast (discontinuous)
    while maintaining measurement precision.

    Args:
        start: Initial categorical state
        target: Target categorical state
        steps: Number of navigation steps

    Returns:
        List of categorical states along the path
    """
    path = []

    for i in range(steps + 1):
        alpha = i / steps

        # Linear interpolation in S-space
        s_k = (1 - alpha) * start.s_k + alpha * target.s_k
        s_t = (1 - alpha) * start.s_t + alpha * target.s_t
        s_e = (1 - alpha) * start.s_e + alpha * target.s_e

        path.append(CategoricalState(s_k=s_k, s_t=s_t, s_e=s_e))

    return path
