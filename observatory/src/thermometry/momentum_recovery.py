# thermometry/momentum_recovery.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.constants as const
from typing import Tuple, Optional
from scipy.optimize import minimize
from categorical_state import CategoricalState, EntropicCoordinates


class MomentumRecovery:
    """
    Recover momentum distribution from configurational entropy Se

    Key relation from paper:
    T = (ℏ²/2πmkB) exp[(2Smomentum/3kB) - 1]

    Enables temperature determination without disturbing atomic ensemble.
    """

    def __init__(self, particle_mass: float, num_particles: int):
        """
        Args:
            particle_mass: Mass of particle [kg]
            num_particles: Number of particles in ensemble
        """
        self.m = particle_mass
        self.N = num_particles
        self.hbar = const.hbar
        self.kB = const.k

    def temperature_from_momentum_entropy(self, S_momentum: float) -> float:
        """
        Extract temperature from momentum entropy

        T = (ℏ²/2πmkB) exp[(2Smomentum/3kB) - 1]

        Args:
            S_momentum: Momentum entropy Sk [J/K]

        Returns:
            Temperature [K]
        """
        prefactor = (self.hbar**2) / (2 * np.pi * self.m * self.kB)
        exponent = (2 * S_momentum) / (3 * self.kB) - 1
        T = prefactor * np.exp(exponent)
        return T

    def momentum_entropy_from_temperature(self, T: float) -> float:
        """
        Calculate momentum entropy from temperature (inverse relation)

        Smomentum = (3kB/2) * [ln(2πmkBT/ℏ²) + 1]

        Args:
            T: Temperature [K]

        Returns:
            Momentum entropy [J/K]
        """
        argument = (2 * np.pi * self.m * self.kB * T) / (self.hbar**2)
        S_momentum = (3 * self.kB / 2) * (np.log(argument) + 1)
        return S_momentum

    def recover_momentum_distribution_parameters(self,
                                                 Se: float) -> Tuple[float, float]:
        """
        Recover momentum distribution parameters from environmental entropy

        For thermal distribution: p ~ exp(-p²/2mkBT)

        Args:
            Se: Environmental entropy [J/K]

        Returns:
            (temperature [K], momentum_width [kg·m/s])
        """
        # Environmental entropy relates to configuration space
        # For ideal gas: Se = NkB[ln(V/N) + 3/2 ln(2πmkBT/h²) + 5/2]

        # Simplified: extract temperature from entropy
        # Se ≈ (3/2) N kB [1 + ln(2πmkBT/h²)]

        # Solve for T
        def entropy_equation(T):
            if T <= 0:
                return 1e10
            predicted_Se = (3/2) * self.N * self.kB * (
                1 + np.log(2 * np.pi * self.m * self.kB * T / const.h**2)
            )
            return (predicted_Se - Se)**2

        # Optimize
        result = minimize(entropy_equation, x0=1e-6, bounds=[(1e-12, 1e-3)])
        T = result.x[0]

        # Momentum width from temperature
        sigma_p = np.sqrt(self.m * self.kB * T)

        return T, sigma_p

    def reconstruct_momentum_distribution(self,
                                         cat_state: CategoricalState,
                                         num_samples: int = 10000) -> np.ndarray:
        """
        Reconstruct full 3D momentum distribution from categorical state

        Args:
            cat_state: Categorical state C(t)
            num_samples: Number of momentum samples to generate

        Returns:
            Momentum array [kg·m/s] (num_samples, 3)
        """
        # Extract entropy components
        Sk = cat_state.S.Sk
        Se = cat_state.S.Se

        # Method 1: From momentum entropy
        T_from_Sk = self.temperature_from_momentum_entropy(Sk)

        # Method 2: From environmental entropy
        T_from_Se, sigma_p_Se = self.recover_momentum_distribution_parameters(Se)

        # Use average (both should agree)
        T_avg = 0.5 * (T_from_Sk + T_from_Se)

        # Generate Maxwell-Boltzmann distribution
        sigma_p = np.sqrt(self.m * self.kB * T_avg)

        momenta = np.random.normal(0, sigma_p, (num_samples, 3))

        return momenta

    def validate_reconstruction(self,
                               original_momenta: np.ndarray,
                               reconstructed_momenta: np.ndarray) -> dict:
        """
        Validate momentum reconstruction accuracy

        Args:
            original_momenta: Original momentum distribution [kg·m/s]
            reconstructed_momenta: Reconstructed distribution [kg·m/s]

        Returns:
            Dictionary with validation metrics
        """
        # Compare distributions
        p_orig = np.linalg.norm(original_momenta, axis=1)
        p_recon = np.linalg.norm(reconstructed_momenta, axis=1)

        # Statistical tests
        mean_orig = np.mean(p_orig)
        mean_recon = np.mean(p_recon)

        std_orig = np.std(p_orig)
        std_recon = np.std(p_recon)

        # Temperature comparison
        T_orig = np.mean(original_momenta**2) / (3 * self.m * self.kB)
        T_recon = np.mean(reconstructed_momenta**2) / (3 * self.m * self.kB)

        # Kolmogorov-Smirnov test
        from scipy.stats import ks_2samp
        ks_statistic, ks_pvalue = ks_2samp(p_orig, p_recon)

        return {
            'mean_momentum_original': mean_orig,
            'mean_momentum_reconstructed': mean_recon,
            'mean_momentum_error': abs(mean_recon - mean_orig) / mean_orig,
            'std_original': std_orig,
            'std_reconstructed': std_recon,
            'std_error': abs(std_recon - std_orig) / std_orig,
            'temperature_original': T_orig,
            'temperature_reconstructed': T_recon,
            'temperature_error': abs(T_recon - T_orig) / T_orig,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'distributions_match': ks_pvalue > 0.05
        }

    def entropy_consistency_check(self, cat_state: CategoricalState) -> dict:
        """
        Check consistency between different entropy components

        Sk, St, Se should be mutually consistent for physical system

        Args:
            cat_state: Categorical state

        Returns:
            Dictionary with consistency metrics
        """
        Sk = cat_state.S.Sk
        Se = cat_state.S.Se
        St = cat_state.S.St

        # Temperature from momentum entropy
        T_from_Sk = self.temperature_from_momentum_entropy(Sk)

        # Temperature from environmental entropy
        T_from_Se, _ = self.recover_momentum_distribution_parameters(Se)

        # Consistency ratio (should be ~1)
        consistency_ratio = T_from_Sk / T_from_Se

        # Total entropy should satisfy thermodynamic relations
        S_total = Sk + St + Se

        # For ideal gas: S_total ≈ NkB[ln(V/N) + (3/2)ln(2πmkBT/h²) + 5/2]
        S_expected = self.N * self.kB * (
            np.log(1e-15 / self.N) +  # Assume V ~ 1 μm³
            (3/2) * np.log(2 * np.pi * self.m * self.kB * T_from_Sk / const.h**2) +
            5/2
        )

        return {
            'T_from_momentum_entropy': T_from_Sk,
            'T_from_environmental_entropy': T_from_Se,
            'consistency_ratio': consistency_ratio,
            'consistent': 0.9 < consistency_ratio < 1.1,
            'total_entropy': S_total,
            'expected_entropy': S_expected,
            'entropy_ratio': S_total / S_expected
        }


class QuantumBackactionAnalyzer:
    """
    Analyze quantum backaction in measurement

    Categorical measurement bypasses quantum backaction since it
    doesn't require momentum measurement (Heisenberg uncertainty).
    """

    def __init__(self, particle_mass: float):
        self.m = particle_mass
        self.hbar = const.hbar

    def heisenberg_momentum_uncertainty(self,
                                       position_uncertainty: float) -> float:
        """
        Calculate minimum momentum uncertainty from Heisenberg principle

        Δp · Δx ≥ ℏ/2

        Args:
            position_uncertainty: Position uncertainty Δx [m]

        Returns:
            Momentum uncertainty Δp [kg·m/s]
        """
        return self.hbar / (2 * position_uncertainty)

    def photon_recoil_energy(self, wavelength: float) -> float:
        """
        Calculate photon recoil energy

        Erecoil = ℏ²k²/2m = h²/(2mλ²)

        Args:
            wavelength: Photon wavelength [m]

        Returns:
            Recoil energy [J]
        """
        return const.h**2 / (2 * self.m * wavelength**2)

    def photon_recoil_temperature(self, wavelength: float) -> float:
        """
        Convert photon recoil energy to temperature

        Args:
            wavelength: Photon wavelength [m]

        Returns:
            Equivalent temperature [K]
        """
        E_recoil = self.photon_recoil_energy(wavelength)
        return E_recoil / const.k

    def conventional_measurement_backaction(self,
                                           wavelength: float,
                                           num_photons: int) -> dict:
        """
        Calculate backaction from conventional optical measurement

        Args:
            wavelength: Probe wavelength [m]
            num_photons: Number of probe photons

        Returns:
            Dictionary with backaction metrics
        """
        # Single photon recoil
        E_recoil_single = self.photon_recoil_energy(wavelength)
        T_recoil_single = self.photon_recoil_temperature(wavelength)

        # Total heating from multiple photons
        E_total = num_photons * E_recoil_single
        T_heating = E_total / const.k

        # Momentum kick
        p_recoil = const.h / wavelength
        delta_p_total = np.sqrt(num_photons) * p_recoil  # Random walk

        return {
            'single_photon_recoil_energy': E_recoil_single,
            'single_photon_recoil_temperature': T_recoil_single,
            'total_heating_energy': E_total,
            'total_heating_temperature': T_heating,
            'momentum_kick': delta_p_total,
            'num_photons': num_photons
        }

    def categorical_measurement_backaction(self,
                                          measurement_time: float,
                                          detuning: float,
                                          probe_intensity: float) -> dict:
        """
        Calculate backaction from categorical measurement

        Far-detuned optical coupling: Δ >> Γ
        Heating < 1 fK/s from paper

        Args:
            measurement_time: Measurement duration [s]
            detuning: Laser detuning [Hz]
            probe_intensity: Probe intensity [W/m²]

        Returns:
            Dictionary with backaction metrics
        """
        # Heating rate from paper
        heating_rate = 1e-15  # K/s (< 1 fK/s)

        # Total heating
        total_heating = heating_rate * measurement_time

        # Effective photon scattering rate (far-detuned)
        # Γ_eff = Γ * (Ω/Δ)² where Γ is natural linewidth
        Gamma_natural = 2 * np.pi * 6e6  # ~6 MHz for alkali atoms

        # Rabi frequency from intensity
        # Ω ∝ √I (simplified)
        Omega = np.sqrt(probe_intensity / 1e-3) * Gamma_natural  # Normalized

        Gamma_eff = Gamma_natural * (Omega / detuning)**2

        # Number of photons scattered
        num_photons_scattered = Gamma_eff * measurement_time

        return {
            'heating_rate': heating_rate,
            'measurement_time': measurement_time,
            'total_heating': total_heating,
            'effective_scattering_rate': Gamma_eff,
            'photons_scattered': num_photons_scattered,
            'detuning': detuning,
            'far_detuned': detuning > 10 * Gamma_natural
        }

    def backaction_comparison(self,
                             T_system: float,
                             wavelength: float = 780e-9,
                             measurement_time: float = 1e-3) -> dict:
        """
        Compare backaction: conventional vs categorical

        Args:
            T_system: System temperature [K]
            wavelength: Probe wavelength [m]
            measurement_time: Measurement duration [s]

        Returns:
            Comparison dictionary
        """
        # Conventional (TOF imaging)
        # Requires ~1000 photons for good SNR
        conventional = self.conventional_measurement_backaction(wavelength, 1000)

        # Categorical (far-detuned)
        detuning = 2 * np.pi * 1e9  # 1 GHz detuning
        categorical = self.categorical_measurement_backaction(
            measurement_time, detuning, 1e-3
        )

        # Improvement factor
        improvement = conventional['total_heating_temperature'] / categorical['total_heating']

        # Is measurement non-invasive?
        conventional_invasive = conventional['total_heating_temperature'] > 0.01 * T_system
        categorical_invasive = categorical['total_heating'] > 0.01 * T_system

        return {
            'system_temperature': T_system,
            'conventional_heating': conventional['total_heating_temperature'],
            'categorical_heating': categorical['total_heating'],
            'improvement_factor': improvement,
            'conventional_invasive': conventional_invasive,
            'categorical_invasive': categorical_invasive,
            'categorical_advantage': not categorical_invasive and conventional_invasive
        }


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("MOMENTUM RECOVERY VALIDATION")
    print("=" * 60)

    # Rb-87 parameters
    m_Rb87 = 1.443e-25  # kg
    N_atoms = 1e5
    T_true = 100e-9  # 100 nK

    # Initialize recovery
    recovery = MomentumRecovery(m_Rb87, N_atoms)

    # Generate original momentum distribution
    sigma_p = np.sqrt(m_Rb87 * const.k * T_true)
    original_momenta = np.random.normal(0, sigma_p, (int(N_atoms), 3))

    print(f"\nOriginal System:")
    print(f"  Temperature: {T_true*1e9:.1f} nK")
    print(f"  Momentum width: {sigma_p:.2e} kg·m/s")

    # Create categorical state from momenta
    from categorical_state import CategoricalStateEstimator
    estimator = CategoricalStateEstimator(m_Rb87, N_atoms)
    cat_state = estimator.from_momentum_distribution(
        original_momenta.flatten(), 0.0
    )

    print(f"\nCategorical State:")
    print(f"  Sk = {cat_state.S.Sk:.6e} J/K")
    print(f"  St = {cat_state.S.St:.6e} J/K")
    print(f"  Se = {cat_state.S.Se:.6e} J/K")

    # Recover temperature from entropy
    T_recovered = recovery.temperature_from_momentum_entropy(cat_state.S.Sk)
    print(f"\nRecovered Temperature:")
    print(f"  T = {T_recovered*1e9:.3f} nK")
    print(f"  Error: {abs(T_recovered - T_true)/T_true * 100:.2f}%")

    # Reconstruct momentum distribution
    reconstructed_momenta = recovery.reconstruct_momentum_distribution(cat_state, 10000)

    # Validate reconstruction
    validation = recovery.validate_reconstruction(original_momenta, reconstructed_momenta)

    print(f"\nReconstruction Validation:")
    print(f"  Temperature error: {validation['temperature_error']*100:.2f}%")
    print(f"  Momentum width error: {validation['std_error']*100:.2f}%")
    print(f"  KS test p-value: {validation['ks_pvalue']:.4f}")
    print(f"  Distributions match: {validation['distributions_match']}")

    # Entropy consistency check
    consistency = recovery.entropy_consistency_check(cat_state)
    print(f"\nEntropy Consistency:")
    print(f"  T from Sk: {consistency['T_from_momentum_entropy']*1e9:.3f} nK")
    print(f"  T from Se: {consistency['T_from_environmental_entropy']*1e9:.3f} nK")
    print(f"  Consistency ratio: {consistency['consistency_ratio']:.4f}")
    print(f"  Consistent: {consistency['consistent']}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    p_orig = np.linalg.norm(original_momenta, axis=1)
    p_recon = np.linalg.norm(reconstructed_momenta, axis=1)

    axes[0].hist(p_orig * 1e27, bins=50, alpha=0.5, label='Original', density=True)
    axes[0].hist(p_recon * 1e27, bins=50, alpha=0.5, label='Reconstructed', density=True)
    axes[0].set_xlabel('Momentum Magnitude [10⁻²⁷ kg·m/s]')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('Momentum Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2D scatter
    axes[1].scatter(original_momenta[:1000, 0] * 1e27,
                   original_momenta[:1000, 1] * 1e27,
                   alpha=0.3, s=1, label='Original')
    axes[1].scatter(reconstructed_momenta[:1000, 0] * 1e27,
                   reconstructed_momenta[:1000, 1] * 1e27,
                   alpha=0.3, s=1, label='Reconstructed')
    axes[1].set_xlabel('px [10⁻²⁷ kg·m/s]')
    axes[1].set_ylabel('py [10⁻²⁷ kg·m/s]')
    axes[1].set_title('Momentum Space (2D)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.savefig('momentum_recovery_validation.png', dpi=150)
    print("\nPlot saved: momentum_recovery_validation.png")

    # ===== QUANTUM BACKACTION ANALYSIS =====
    print("\n" + "=" * 60)
    print("QUANTUM BACKACTION ANALYSIS")
    print("=" * 60)

    backaction = QuantumBackactionAnalyzer(m_Rb87)

    # Photon recoil for Rb-87
    wavelength_D2 = 780e-9  # D2 line
    T_recoil = backaction.photon_recoil_temperature(wavelength_D2)

    print(f"\nPhoton Recoil (λ = {wavelength_D2*1e9:.0f} nm):")
    print(f"  Recoil energy: {backaction.photon_recoil_energy(wavelength_D2):.2e} J")
    print(f"  Recoil temperature: {T_recoil*1e9:.1f} nK")
    print(f"  Paper claim: ~280 nK")

    # Compare backaction
    comparison = backaction.backaction_comparison(T_true, wavelength_D2, 1e-3)

    print(f"\nBackaction Comparison (T = {T_true*1e9:.1f} nK):")
    print(f"  Conventional heating: {comparison['conventional_heating']*1e9:.2f} nK")
    print(f"  Categorical heating: {comparison['categorical_heating']*1e15:.2f} fK")
    print(f"  Improvement factor: {comparison['improvement_factor']:.2e}")
    print(f"  Conventional invasive: {comparison['conventional_invasive']}")
    print(f"  Categorical invasive: {comparison['categorical_invasive']}")
    print(f"  Categorical advantage: {comparison['categorical_advantage']}")

    print("\n" + "=" * 60)
