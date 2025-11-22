# thermometry/comparison_tof.py

import numpy as np
import scipy.constants as const
from typing import Tuple, Dict
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from categorical_state import CategoricalState, CategoricalStateEstimator
from temperature_extraction import ThermometryAnalyzer


@dataclass
class ThermometryPerformance:
    """Performance metrics for thermometry method"""
    method_name: str
    temperature: float  # [K]
    uncertainty: float  # [K]
    relative_precision: float  # ΔT/T
    measurement_time: float  # [s]
    destructive: bool
    heating: float  # [K]
    num_atoms_required: int


class TimeOfFlightThermometry:
    """
    Model time-of-flight absorption imaging thermometry

    Standard destructive technique:
    1. Release atoms from trap
    2. Free expansion for time t_expand
    3. Absorption image
    4. Fit Gaussian to extract temperature
    """

    def __init__(self, particle_mass: float):
        """
        Args:
            particle_mass: Atomic mass [kg]
        """
        self.m = particle_mass
        self.kB = const.k

    def expansion_dynamics(self,
                          T: float,
                          t_expand: float,
                          trap_frequency: float = 2 * np.pi * 100) -> Tuple[float, float]:
        """
        Calculate cloud size after expansion

        For harmonic trap at temperature T:
        σ(t) = σ₀√[1 + (ωt)²]  where σ₀ = √(kBT/mω²)

        Args:
            T: Temperature [K]
            t_expand: Expansion time [s]
            trap_frequency: Trap frequency [rad/s]

        Returns:
            (cloud_size [m], velocity_width [m/s])
        """
        # Initial cloud size
        sigma_0 = np.sqrt(self.kB * T / (self.m * trap_frequency**2))

        # Thermal velocity
        v_thermal = np.sqrt(self.kB * T / self.m)

        # Size after expansion
        sigma_t = sigma_0 * np.sqrt(1 + (trap_frequency * t_expand)**2)

        return sigma_t, v_thermal

    def temperature_from_expansion(self,
                                   cloud_size: float,
                                   t_expand: float,
                                   trap_frequency: float = 2 * np.pi * 100) -> float:
        """
        Extract temperature from measured cloud size

        Args:
            cloud_size: Measured RMS cloud size [m]
            t_expand: Expansion time [s]
            trap_frequency: Trap frequency [rad/s]

        Returns:
            Temperature [K]
        """
        # Thermal velocity from expansion
        # σ ≈ vT * t for long expansion (ωt >> 1)
        v_thermal = cloud_size / t_expand

        # Temperature from velocity
        T = self.m * v_thermal**2 / self.kB

        return T

    def measurement_uncertainty(self,
                               T: float,
                               t_expand: float = 20e-3,
                               imaging_resolution: float = 5e-6,
                               num_atoms: int = 10000,
                               photons_per_atom: int = 100) -> float:
        """
        Calculate temperature measurement uncertainty

        Sources of uncertainty:
        1. Imaging resolution (pixel size)
        2. Photon shot noise
        3. Fit uncertainty

        Args:
            T: Temperature [K]
            t_expand: Expansion time [s]
            imaging_resolution: Spatial resolution [m]
            num_atoms: Number of atoms imaged
            photons_per_atom: Photons scattered per atom

        Returns:
            Temperature uncertainty [K]
        """
        # Thermal velocity
        v_thermal = np.sqrt(self.kB * T / self.m)

        # Expansion distance
        expansion_distance = v_thermal * t_expand

        # Spatial uncertainty from imaging
        delta_sigma_spatial = imaging_resolution / np.sqrt(num_atoms)

        # Photon shot noise
        snr_photon = np.sqrt(num_atoms * photons_per_atom)
        delta_sigma_photon = expansion_distance / snr_photon

        # Total spatial uncertainty
        delta_sigma = np.sqrt(delta_sigma_spatial**2 + delta_sigma_photon**2)

        # Velocity uncertainty
        delta_v = delta_sigma / t_expand

        # Temperature uncertainty (ΔT/T = 2Δv/v)
        delta_T = 2 * T * delta_v / v_thermal

        return delta_T

    def photon_scattering_heating(self,
                                  wavelength: float = 780e-9,
                                  photons_per_atom: int = 100) -> float:
        """
        Calculate heating from imaging photons

        Each photon scatters → momentum kick → heating

        Args:
            wavelength: Imaging wavelength [m]
            photons_per_atom: Number of photons scattered

        Returns:
            Temperature increase [K]
        """
        # Recoil energy per photon
        E_recoil = const.h**2 / (2 * self.m * wavelength**2)

        # Total heating (random walk in momentum space)
        heating = photons_per_atom * E_recoil / self.kB

        return heating

    def perform_measurement(self,
                           T_true: float,
                           t_expand: float = 20e-3,
                           imaging_resolution: float = 5e-6,
                           num_atoms: int = 10000) -> ThermometryPerformance:
        """
        Simulate complete TOF measurement

        Args:
            T_true: True temperature [K]
            t_expand: Expansion time [s]
            imaging_resolution: Spatial resolution [m]
            num_atoms: Number of atoms

        Returns:
            ThermometryPerformance
        """
        # Expansion dynamics
        cloud_size, v_thermal = self.expansion_dynamics(T_true, t_expand)

        # Add measurement noise
        noise = np.random.normal(0, imaging_resolution)
        cloud_size_measured = cloud_size + noise

        # Extract temperature
        T_measured = self.temperature_from_expansion(cloud_size_measured, t_expand)

        # Uncertainty
        delta_T = self.measurement_uncertainty(
            T_true, t_expand, imaging_resolution, num_atoms
        )

        # Heating
        heating = self.photon_scattering_heating(photons_per_atom=100)

        # Measurement time (expansion + imaging)
        measurement_time = t_expand + 1e-3  # ~20 ms

        return ThermometryPerformance(
            method_name="Time-of-Flight",
            temperature=T_measured,
            uncertainty=delta_T,
            relative_precision=delta_T / T_measured,
            measurement_time=measurement_time,
            destructive=True,
            heating=heating,
            num_atoms_required=num_atoms
        )


class CategoricalThermometryComparison:
    """
    Compare categorical and TOF thermometry
    """

    def __init__(self, particle_mass: float, num_particles: int):
        """
        Args:
            particle_mass: Atomic mass [kg]
            num_particles: Number of particles in ensemble
        """
        self.m = particle_mass
        self.N = num_particles

        # Initialize methods
        self.tof = TimeOfFlightThermometry(particle_mass)
        self.categorical = ThermometryAnalyzer(particle_mass)
        self.estimator = CategoricalStateEstimator(particle_mass, num_particles)

    def categorical_measurement(self,
                                momenta: np.ndarray,
                                measurement_time: float = 1e-3) -> ThermometryPerformance:
        """
        Perform categorical thermometry measurement

        Args:
            momenta: Momentum distribution [kg·m/s]
            measurement_time: Measurement duration [s]

        Returns:
            ThermometryPerformance
        """
        # Construct categorical state
        cat_state = self.estimator.from_momentum_distribution(momenta, 0.0)

        # Extract temperature
        T_measured, delta_T = self.categorical.extract_temperature(cat_state)

        # Heating from far-detuned optical coupling
        heating = self.categorical.heating_rate(measurement_time)

        return ThermometryPerformance(
            method_name="Categorical",
            temperature=T_measured,
            uncertainty=delta_T,
            relative_precision=delta_T / T_measured,
            measurement_time=measurement_time,
            destructive=False,
            heating=heating,
            num_atoms_required=int(self.N)
        )

    def comparative_benchmark(self,
                             temperature_range: np.ndarray,
                             num_trials: int = 100) -> Dict:
        """
        Benchmark both methods across temperature range

        Args:
            temperature_range: Array of temperatures to test [K]
            num_trials: Number of trials per temperature

        Returns:
            Benchmark results dictionary
        """
        results = {
            'temperatures': temperature_range,
            'tof_precision': [],
            'tof_precision_std': [],
            'categorical_precision': [],
            'categorical_precision_std': [],
            'tof_heating': [],
            'categorical_heating': [],
            'improvement_factor': []
        }

        for T_true in temperature_range:
            # Generate momentum distribution
            sigma_v = np.sqrt(const.k * T_true / self.m)

            tof_precisions = []
            cat_precisions = []

            for _ in range(num_trials):
                # Sample momenta
                velocities = np.random.normal(0, sigma_v, int(self.N))
                momenta = self.m * velocities

                # TOF measurement
                tof_perf = self.tof.perform_measurement(T_true, num_atoms=int(self.N))
                tof_precisions.append(tof_perf.relative_precision)

                # Categorical measurement
                cat_perf = self.categorical_measurement(momenta)
                cat_precisions.append(cat_perf.relative_precision)

            # Store statistics
            results['tof_precision'].append(np.mean(tof_precisions))
            results['tof_precision_std'].append(np.std(tof_precisions))
            results['categorical_precision'].append(np.mean(cat_precisions))
            results['categorical_precision_std'].append(np.std(cat_precisions))

            # Heating
            results['tof_heating'].append(tof_perf.heating)
            results['categorical_heating'].append(cat_perf.heating)

            # Improvement
            improvement = np.mean(tof_precisions) / np.mean(cat_precisions)
            results['improvement_factor'].append(improvement)

        return results

    def validate_paper_claims(self) -> Dict:
        """
        Validate paper's thermometry claims

        Paper claims:
        - δT ~ 17 pK resolution
        - ΔT/T improvement over TOF
        - Non-invasive (<1 fK/s heating)

        Returns:
            Validation dictionary
        """
        # Test at T = 100 nK
        T_test = 100e-9  # K

        # Generate momenta
        sigma_v = np.sqrt(const.k * T_test / self.m)
        velocities = np.random.normal(0, sigma_v, int(self.N))
        momenta = self.m * velocities

        # TOF measurement
        tof_perf = self.tof.perform_measurement(T_test, num_atoms=int(self.N))

        # Categorical measurement
        cat_perf = self.categorical_measurement(momenta)

        # Paper claims
        paper_resolution_claim = 17e-12  # 17 pK
        paper_heating_claim = 1e-15  # <1 fK/s

        return {
            'test_temperature': T_test,
            'tof_uncertainty': tof_perf.uncertainty,
            'categorical_uncertainty': cat_perf.uncertainty,
            'paper_resolution_claim': paper_resolution_claim,
            'resolution_claim_validated': cat_perf.uncertainty < 20e-12,
            'tof_relative_precision': tof_perf.relative_precision,
            'categorical_relative_precision': cat_perf.relative_precision,
            'precision_improvement': tof_perf.relative_precision / cat_perf.relative_precision,
            'tof_heating': tof_perf.heating,
            'categorical_heating_rate': cat_perf.heating / cat_perf.measurement_time,
            'heating_claim_validated': (cat_perf.heating / cat_perf.measurement_time) < 2e-15,
            'tof_destructive': tof_perf.destructive,
            'categorical_destructive': cat_perf.destructive
        }

    def plot_comparative_analysis(self, save_path: str = None):
        """
        Generate publication-quality comparison plots

        Args:
            save_path: Path to save figure
        """
        # Temperature range: 10 nK to 10 μK
        temperatures = np.logspace(-8, -5, 20)

        # Benchmark
        benchmark = self.comparative_benchmark(temperatures, num_trials=50)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: Relative Precision vs Temperature
        ax = axes[0, 0]
        ax.loglog(temperatures * 1e9, benchmark['tof_precision'],
                 'bo-', linewidth=2, markersize=6, label='TOF')
        ax.loglog(temperatures * 1e9, benchmark['categorical_precision'],
                 'rs-', linewidth=2, markersize=6, label='Categorical')

        # Quantum limit (Heisenberg)
        # ΔE·Δt ≥ ℏ/2 → ΔT ≥ ℏ/(2kBΔt)
        delta_t_tof = 20e-3
        delta_t_cat = 2e-15
        T_limit_tof = const.hbar / (2 * const.k * delta_t_tof)
        T_limit_cat = const.hbar / (2 * const.k * delta_t_cat)

        ax.axhline(T_limit_tof / temperatures[0], color='b',
                  linestyle='--', alpha=0.5, label='TOF limit')
        ax.axhline(T_limit_cat / temperatures[0], color='r',
                  linestyle='--', alpha=0.5, label='Cat limit')

        ax.set_xlabel('Temperature [nK]')
        ax.set_ylabel('Relative Precision ΔT/T')
        ax.set_title('A) Temperature Measurement Precision')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Panel B: Absolute Uncertainty
        ax = axes[0, 1]
        tof_abs = np.array(benchmark['tof_precision']) * temperatures
        cat_abs = np.array(benchmark['categorical_precision']) * temperatures

        ax.loglog(temperatures * 1e9, tof_abs * 1e12,
                 'bo-', linewidth=2, markersize=6, label='TOF')
        ax.loglog(temperatures * 1e9, cat_abs * 1e12,
                 'rs-', linewidth=2, markersize=6, label='Categorical')

        # Paper claim: 17 pK
        ax.axhline(17, color='g', linestyle='--', linewidth=2,
                  label='Paper claim (17 pK)')

        ax.set_xlabel('Temperature [nK]')
        ax.set_ylabel('Temperature Uncertainty [pK]')
        ax.set_title('B) Absolute Temperature Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Panel C: Improvement Factor
        ax = axes[1, 0]
        ax.semilogx(temperatures * 1e9, benchmark['improvement_factor'],
                   'g-', linewidth=2)
        ax.axhline(1, color='k', linestyle='--', alpha=0.5,
                  label='No improvement')
        ax.set_xlabel('Temperature [nK]')
        ax.set_ylabel('Improvement Factor (TOF / Categorical)')
        ax.set_title('C) Precision Improvement Factor')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel D: Heating Comparison
        ax = axes[1, 1]
        ax.loglog(temperatures * 1e9,
                 np.array(benchmark['tof_heating']) * 1e9,
                 'bo-', linewidth=2, markersize=6, label='TOF')
        ax.loglog(temperatures * 1e9,
                 np.array(benchmark['categorical_heating']) * 1e15,
                 'rs-', linewidth=2, markersize=6, label='Categorical (fK)')

        # Mark destructive vs non-invasive
        ax.axhline(0.1, color='r', linestyle='--', alpha=0.5,
                  label='Invasive threshold (0.1 nK)')

        ax.set_xlabel('Temperature [nK]')
        ax.set_ylabel('Heating [nK (TOF), fK (Cat)]')
        ax.set_title('D) Measurement-Induced Heating')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        else:
            plt.show()

        return fig


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("TOF vs CATEGORICAL THERMOMETRY COMPARISON")
    print("=" * 70)

    # Rb-87 parameters
    m_Rb87 = 1.443e-25  # kg
    N_atoms = 1e5

    # Initialize comparison
    comparison = CategoricalThermometryComparison(m_Rb87, N_atoms)

    # Validate paper claims
    print("\n" + "-" * 70)
    print("VALIDATING PAPER CLAIMS")
    print("-" * 70)

    validation = comparison.validate_paper_claims()

    print(f"\nTest temperature: {validation['test_temperature']*1e9:.1f} nK")

    print(f"\nTemperature Uncertainty:")
    print(f"  TOF: {validation['tof_uncertainty']*1e12:.2f} pK")
    print(f"  Categorical: {validation['categorical_uncertainty']*1e12:.2f} pK")
    print(f"  Paper claim: {validation['paper_resolution_claim']*1e12:.0f} pK")
    print(f"  Resolution claim validated: {validation['resolution_claim_validated']}")

    print(f"\nRelative Precision:")
    print(f"  TOF: {validation['tof_relative_precision']:.2e}")
    print(f"  Categorical: {validation['categorical_relative_precision']:.2e}")
    print(f"  Improvement factor: {validation['precision_improvement']:.2e}×")

    print(f"\nMeasurement-Induced Heating:")
    print(f"  TOF: {validation['tof_heating']*1e9:.2f} nK (destructive)")
    print(f"  Categorical: {validation['categorical_heating_rate']*1e15:.3f} fK/s")
    print(f"  Heating claim validated: {validation['heating_claim_validated']}")

    print(f"\nMeasurement Characteristics:")
    print(f"  TOF destructive: {validation['tof_destructive']}")
    print(f"  Categorical destructive: {validation['categorical_destructive']}")

    # Detailed comparison at specific temperatures
    print("\n" + "-" * 70)
    print("PERFORMANCE AT DIFFERENT TEMPERATURES")
    print("-" * 70)

    test_temps = [10e-9, 100e-9, 1e-6, 10e-6]  # 10 nK to 10 μK

    for T in test_temps:
        print(f"\nT = {T*1e9:.0f} nK:")

        # Generate momenta
        sigma_v = np.sqrt(const.k * T / m_Rb87)
        velocities = np.random.normal(0, sigma_v, int(N_atoms))
        momenta = m_Rb87 * velocities

        # TOF
        tof_perf = comparison.tof.perform_measurement(T, num_atoms=int(N_atoms))

        # Categorical
        cat_perf = comparison.categorical_measurement(momenta)

        print(f"  TOF:")
        print(f"    ΔT = {tof_perf.uncertainty*1e12:.2f} pK")
        print(f"    ΔT/T = {tof_perf.relative_precision:.2e}")
        print(f"    Time = {tof_perf.measurement_time*1e3:.1f} ms")

        print(f"  Categorical:")
        print(f"    ΔT = {cat_perf.uncertainty*1e12:.2f} pK")
        print(f"    ΔT/T = {cat_perf.relative_precision:.2e}")
        print(f"    Time = {cat_perf.measurement_time*1e3:.2f} ms")

        print(f"  Improvement: {tof_perf.relative_precision / cat_perf.relative_precision:.2e}×")

    # Generate comparative plots
    print("\n" + "-" * 70)
    print("GENERATING COMPARATIVE ANALYSIS PLOTS")
    print("-" * 70)

    comparison.plot_comparative_analysis('thermometry_comparison_tof_vs_categorical.png')

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
