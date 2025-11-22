#!/usr/bin/env python3
"""
Corrected Temperature Extraction from Categorical States

DERIVATION FROM FIRST PRINCIPLES:
=================================

1. Phase Space Volume for Ideal Gas:
   Ω(T) = (V/h³) ∫ exp(-p²/2mkBT) d³p

2. Momentum integral (Gaussian):
   ∫ exp(-p²/2mkBT) d³p = (2πmkBT)^(3/2)

3. Statistical entropy:
   S = kB ln(Ω) = kB ln[V(2πmkBT)^(3/2)/h³]

4. Isolate momentum contribution:
   S_momentum = (3kB/2) ln(mkBT/2πℏ²) + (3kB/2)

5. Solve for T:
   ln(mkBT/2πℏ²) = (2S_momentum/3kB) - 1
   mkBT/2πℏ² = exp[(2S_momentum/3kB) - 1]
   T = (2πℏ²/mkB) exp[(2S_momentum/3kB) - 1]

6. Connection to categorical state (Sk, St, Se):
   - Sk: knowledge entropy (distinguishability)
   - St: temporal entropy (rate of evolution)
   - Se: evolution entropy (irreversible completion)

   For momentum distribution:
   S_momentum ≈ Se (momentum is frozen during categorical measurement)

7. Uncertainty from timing precision:
   δT = |∂T/∂Se| × δSe
   where δSe comes from oscillator timing: δt ~ 2×10⁻¹⁵ s

CORRECTIONS FOR QUANTUM SYSTEMS:
================================

BEC (T < T_BEC):
- Thermal fraction: N_thermal/N = 1 - (T/T_BEC)^3
- Entropy from thermal atoms only
- Condensate has zero entropy (ground state)

Mean-field interactions (Gross-Pitaevskii):
- Energy shift: μ = gn₀ where g = 4πℏ²a_s/m
- Effective temperature includes interaction energy
"""

import numpy as np
import scipy.constants as const
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CategoricalState:
    """Categorical state (Sk, St, Se)"""
    Sk: float  # Knowledge entropy [J/K]
    St: float  # Temporal entropy [J/K]
    Se: float  # Evolution entropy [J/K]


@dataclass
class QuantumSystem:
    """Quantum system parameters"""
    mass: float  # Particle mass [kg]
    scattering_length: float = 0.0  # Scattering length [m] (0 for non-interacting)
    N_particles: int = 1000  # Number of particles


class CategoricalThermometry:
    """Extract temperature from categorical states with rigorous stat mech"""

    def __init__(self, system: QuantumSystem):
        self.system = system
        self.m = system.mass
        self.hbar = const.hbar
        self.kB = const.k

        # Oscillator timing precision (H+ at 71 THz)
        self.timing_precision = 2e-15  # s
        self.osc_freq = 71e12  # Hz

    def momentum_entropy_to_temperature(self, S_momentum: float) -> float:
        """
        Extract temperature from momentum entropy (ideal gas)

        T = (2πℏ²/mkB) exp[(2S_momentum/3kB) - 1]

        Args:
            S_momentum: Momentum contribution to entropy [J/K]

        Returns:
            Temperature [K]
        """
        if S_momentum <= 0:
            return 0.0

        # Direct inversion of entropy formula
        exponent = (2 * S_momentum) / (3 * self.kB) - 1

        # Prevent overflow for large entropy
        if exponent > 100:
            exponent = 100
        elif exponent < -100:
            return 0.0

        T = (2 * np.pi * self.hbar**2 / (self.m * self.kB)) * np.exp(exponent)

        return T

    def categorical_state_to_temperature(self, state: CategoricalState) -> float:
        """
        Extract temperature from categorical state (Sk, St, Se)

        MAPPING:
        --------
        Se ≈ S_momentum (momentum frozen during measurement)
        Sk ≈ distinguishability entropy (not used for T)
        St ≈ time evolution entropy (not used for T)

        Args:
            state: Categorical state

        Returns:
            Temperature [K]
        """
        # Use Se as momentum entropy
        S_momentum = state.Se

        return self.momentum_entropy_to_temperature(S_momentum)

    def temperature_uncertainty(self, T: float) -> float:
        """
        Uncertainty in temperature from timing precision

        ΔT = |∂T/∂Se| × ΔSe

        where ΔSe comes from oscillator timing uncertainty

        Args:
            T: Temperature [K]

        Returns:
            Temperature uncertainty [K]
        """
        # Energy uncertainty from timing
        # ΔE = ℏ × Δω = ℏ × (2π × Δf)
        # For oscillator: Δf/f ~ Δt × f (timing jitter)
        delta_f = self.timing_precision * self.osc_freq
        delta_E = self.hbar * 2 * np.pi * delta_f

        # Entropy uncertainty
        # ΔSe ~ ΔE/T (from thermodynamic relation)
        if T > 0:
            delta_Se = delta_E / T
        else:
            delta_Se = 0

        # Temperature uncertainty via chain rule
        # ∂T/∂Se = T × (2/3kB)  [from exponential formula]
        dT_dSe = T * (2 / (3 * self.kB))

        delta_T = abs(dT_dSe * delta_Se)

        # Minimum uncertainty from quantum limit
        # ΔT_min = ℏω/kB (energy resolution limit)
        delta_T_quantum = (self.hbar * 2 * np.pi * delta_f) / self.kB

        # Take larger of the two (conservative)
        return max(delta_T, delta_T_quantum)

    def bec_correction(self, T: float, density: float) -> Tuple[float, float]:
        """
        BEC corrections for T < T_BEC

        Args:
            T: Measured temperature [K]
            density: Number density [m⁻³]

        Returns:
            (T_corrected, thermal_fraction)
        """
        # BEC critical temperature
        # T_BEC = (2πℏ²/mkB) × (n/ζ(3/2))^(2/3)
        # where ζ(3/2) ≈ 2.612 (Riemann zeta)

        zeta_3_2 = 2.612
        T_BEC = (2 * np.pi * self.hbar**2 / (self.m * self.kB)) * \
                (density / zeta_3_2)**(2/3)

        if T >= T_BEC:
            # Above BEC transition - no correction needed
            return T, 1.0

        # Below transition - only thermal fraction contributes to entropy
        thermal_fraction = (T / T_BEC)**3

        # Corrected temperature accounts for condensate
        # (measured entropy comes only from thermal atoms)
        T_corrected = T / thermal_fraction**(1/3) if thermal_fraction > 0 else T

        return T_corrected, thermal_fraction

    def interaction_correction(self, T: float, density: float) -> float:
        """
        Mean-field interaction correction (Gross-Pitaevskii)

        Args:
            T: Temperature [K]
            density: Number density [m⁻³]

        Returns:
            Temperature correction ΔT [K]
        """
        if self.system.scattering_length == 0:
            return 0.0  # Non-interacting

        # Interaction parameter
        g = 4 * np.pi * self.hbar**2 * self.system.scattering_length / self.m

        # Mean-field energy shift
        # μ = g × n₀ where n₀ is condensate density
        # Approximation: n₀ ≈ density (for strong condensate)
        mu = g * density

        # Temperature correction (energy shift / kB)
        delta_T = mu / self.kB

        return delta_T


def validate_temperature_extraction():
    """Validation with known test cases"""

    print("=" * 70)
    print("TEMPERATURE EXTRACTION VALIDATION (CORRECTED)")
    print("=" * 70)

    # Rb-87 parameters
    m_Rb87 = 87 * const.u  # atomic mass unit
    a_s_Rb87 = 100 * const.physical_constants['Bohr radius'][0]  # scattering length

    system = QuantumSystem(
        mass=m_Rb87,
        scattering_length=a_s_Rb87,
        N_particles=10000
    )

    thermo = CategoricalThermometry(system)

    # Test case 1: Known temperature → entropy → temperature
    print("\n" + "-" * 70)
    print("TEST 1: Round-trip validation (T → S → T)")
    print("-" * 70)

    test_temperatures = [10e-9, 100e-9, 1e-6, 10e-6]  # nK to μK range

    for T_true in test_temperatures:
        # Calculate expected entropy for this temperature
        # S = (3kB/2)[ln(mkBT/2πℏ²) + 1]
        ln_term = np.log((m_Rb87 * const.k * T_true) / (2 * np.pi * const.hbar**2))
        S_expected = (3 * const.k / 2) * (ln_term + 1)

        # Create categorical state with this entropy
        state = CategoricalState(Sk=0, St=0, Se=S_expected)

        # Extract temperature
        T_measured = thermo.categorical_state_to_temperature(state)
        delta_T = thermo.temperature_uncertainty(T_measured)

        # Calculate error
        error_pct = abs(T_measured - T_true) / T_true * 100

        print(f"\nT_true = {T_true*1e9:.1f} nK:")
        print(f"  S_momentum = {S_expected:.6e} J/K")
        print(f"  T_measured = {T_measured*1e9:.3f} ± {delta_T*1e12:.2f} pK")
        print(f"  Error: {error_pct:.6f}%")
        print(f"  Relative precision: {delta_T/T_measured:.2e}")

    # Test case 2: Realistic categorical state (from actual measurement)
    print("\n" + "-" * 70)
    print("TEST 2: Realistic measurement scenario")
    print("-" * 70)

    T_target = 100e-9  # 100 nK

    # Simulate categorical state measurement
    # In real measurement: (Sk, St, Se) extracted from oscillator phase space

    # For T = 100 nK:
    ln_term = np.log((m_Rb87 * const.k * T_target) / (2 * np.pi * const.hbar**2))
    Se_measured = (3 * const.k / 2) * (ln_term + 1)

    # Add small measurement noise (0.1% of Se)
    Se_measured += np.random.normal(0, 0.001 * abs(Se_measured))

    state_measured = CategoricalState(
        Sk=1e-23,  # Distinguishability entropy (not used for T)
        St=0,      # Time evolution entropy
        Se=Se_measured
    )

    T_extracted = thermo.categorical_state_to_temperature(state_measured)
    delta_T_extracted = thermo.temperature_uncertainty(T_extracted)

    print(f"\nTarget temperature: {T_target*1e9:.3f} nK")
    print(f"Categorical state: Sk={state_measured.Sk:.2e}, St={state_measured.St:.2e}, Se={state_measured.Se:.2e} J/K")
    print(f"Extracted temperature: {T_extracted*1e9:.3f} ± {delta_T_extracted*1e12:.2f} pK")
    print(f"Relative error: {abs(T_extracted - T_target)/T_target*100:.4f}%")
    print(f"Relative precision: {delta_T_extracted/T_extracted:.2e}")

    # Compare with paper claim
    paper_claim_uncertainty = 17e-12  # 17 pK
    print(f"\nPaper claim uncertainty: {paper_claim_uncertainty*1e12:.0f} pK")
    print(f"Achieved uncertainty: {delta_T_extracted*1e12:.2f} pK")
    print(f"Claim validated: {delta_T_extracted <= paper_claim_uncertainty*1.1}")  # 10% tolerance

    # Test case 3: BEC corrections
    print("\n" + "-" * 70)
    print("TEST 3: BEC corrections")
    print("-" * 70)

    # Typical BEC density
    density = 1e14  # atoms/cm³ = 1e20 atoms/m³
    density_SI = density * 1e6  # m⁻³

    T_test = 50e-9  # 50 nK (likely below T_BEC)
    T_corrected, thermal_frac = thermo.bec_correction(T_test, density_SI)

    print(f"\nMeasured T = {T_test*1e9:.1f} nK")
    print(f"Density = {density:.1e} atoms/cm³")
    print(f"Thermal fraction = {thermal_frac:.4f}")
    print(f"Corrected T = {T_corrected*1e9:.3f} nK")

    # Test case 4: Interaction corrections
    print("\n" + "-" * 70)
    print("TEST 4: Mean-field interaction corrections")
    print("-" * 70)

    delta_T_interaction = thermo.interaction_correction(T_test, density_SI)

    print(f"\nT = {T_test*1e9:.1f} nK")
    print(f"Scattering length = {system.scattering_length/const.physical_constants['Bohr radius'][0]:.1f} a₀")
    print(f"Interaction correction ΔT = {delta_T_interaction*1e9:.3f} nK")
    print(f"Total T = {(T_test + delta_T_interaction)*1e9:.3f} nK")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    validate_temperature_extraction()
