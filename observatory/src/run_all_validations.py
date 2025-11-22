#!/usr/bin/env python3
"""
Master Validation Script

Runs comprehensive validation experiments for:
1. Categorical state representation and oscillator synchronization
2. Trans-Planckian interferometry (angular resolution, atmospheric immunity, baseline coherence)
3. Categorical quantum thermometry (temperature extraction, momentum recovery, TOF comparison)

Generates validation reports and publication-quality figures.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import validation modules
from categorical.categorical_state import (
    CategoricalState, CategoricalStateEstimator, EntropicCoordinates
)
from categorical.oscillator_synchronization import (
    HydrogenOscillatorSync, MultiStationSync
)
from interferometry.angular_resolution import (
    AngularResolutionCalculator, TransPlanckianResolutionValidator
)
from interferometry.atmospheric_effects import (
    AtmosphericComparisonExperiment
)
from interferometry.baseline_coherence import (
    BaselineCoherenceAnalyzer, FringeVisibilityExperiment
)
from interferometry.phase_correlation import (
    CategoricalPhaseAnalyzer, TransPlanckianInterferometer
)
from thermometry.temperature_extraction import (
    ThermometryAnalyzer, TimeOfFlightComparison
)
from thermometry.momentum_recovery import (
    MomentumRecovery, QuantumBackactionAnalyzer
)
from thermometry.real_time_monitor import (
    RealTimeThermometer, EvaporativeCoolingSimulator
)
from thermometry.comparison_tof import (
    CategoricalThermometryComparison
)


class ValidationReport:
    """Generate comprehensive validation report"""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}

    def add_section(self, section_name: str, data: dict):
        """Add validation section"""
        self.results[section_name] = data

    def save_json(self):
        """Save results as JSON"""
        output_file = self.output_dir / f"validation_report_{self.timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ JSON report saved: {output_file}")

    def generate_markdown_report(self):
        """Generate human-readable markdown report"""
        output_file = self.output_dir / f"validation_report_{self.timestamp}.md"

        with open(output_file, 'w') as f:
            f.write("# Comprehensive Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Table of contents
            f.write("## Table of Contents\n\n")
            for i, section in enumerate(self.results.keys(), 1):
                f.write(f"{i}. [{section}](#{section.lower().replace(' ', '-')})\n")
            f.write("\n---\n\n")

            # Sections
            for section_name, data in self.results.items():
                f.write(f"## {section_name}\n\n")
                self._write_section(f, data, level=3)
                f.write("\n---\n\n")

        print(f"✓ Markdown report saved: {output_file}")

    def _write_section(self, f, data, level=3):
        """Recursively write section data"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    f.write(f"{'#' * level} {key}\n\n")
                    self._write_section(f, value, level + 1)
                else:
                    f.write(f"**{key}:** {value}\n\n")
        elif isinstance(data, list):
            for item in data:
                f.write(f"- {item}\n")
            f.write("\n")
        else:
            f.write(f"{data}\n\n")


def validate_categorical_framework(report: ValidationReport):
    """Validate categorical state representation"""
    print("\n" + "=" * 70)
    print("VALIDATING CATEGORICAL FRAMEWORK")
    print("=" * 70)

    results = {}

    # Rb-87 parameters
    m_Rb87 = 1.443e-25  # kg
    N_atoms = 1e5

    # Test categorical state construction
    print("\n1. Categorical State Construction...")
    estimator = CategoricalStateEstimator(m_Rb87, N_atoms)

    # Generate momentum distribution at T = 100 nK
    T_test = 100e-9  # K
    sigma_v = np.sqrt(1.380649e-23 * T_test / m_Rb87)
    velocities = np.random.normal(0, sigma_v, int(N_atoms))
    momenta = m_Rb87 * velocities

    cat_state = estimator.from_momentum_distribution(momenta, 0.0)

    results['categorical_state'] = {
        'temperature_test': f"{T_test * 1e9:.1f} nK",
        'Sk_kinetic_entropy': f"{cat_state.S.Sk:.6e} J/K",
        'St_temporal_entropy': f"{cat_state.S.St:.6e} J/K",
        'Se_environmental_entropy': f"{cat_state.S.Se:.6e} J/K",
        'total_entropy': f"{cat_state.S.total_entropy():.6e} J/K"
    }

    # Test H+ oscillator synchronization
    print("2. H+ Oscillator Synchronization...")
    sync = HydrogenOscillatorSync()

    # Timing precision
    results['oscillator_sync'] = {
        'frequency': f"{sync.f_osc / 1e12:.1f} THz",
        'timing_precision': f"{sync.delta_t * 1e15:.2f} fs",
        'energy_resolution': f"{sync.delta_E:.2e} J",
        'temperature_resolution': f"{sync.delta_E / 1.380649e-23 * 1e12:.1f} pK"
    }

    # Multi-station synchronization
    print("3. Multi-Station Network Synchronization...")
    N_stations = 10
    positions = np.random.randn(N_stations, 3) * 1e6  # ~1000 km scale

    network = MultiStationSync(N_stations)
    network.initialize_network(positions)

    results['multi_station_sync'] = {
        'num_stations': N_stations,
        'network_scale_km': f"{np.max(np.linalg.norm(positions, axis=1)) / 1e3:.1f}",
        'synchronization_error_fs': f"{network.synchronization_error() * 1e15:.3f}",
        'max_baseline_delay_ms': f"{np.max(np.abs(network.get_baseline_delays())) * 1e3:.3f}"
    }

    print("✓ Categorical framework validation complete")
    report.add_section("Categorical Framework", results)


def validate_interferometry(report: ValidationReport):
    """Validate trans-Planckian interferometry"""
    print("\n" + "=" * 70)
    print("VALIDATING TRANS-PLANCKIAN INTERFEROMETRY")
    print("=" * 70)

    results = {}
    wavelength = 500e-9  # 500 nm

    # 1. Angular Resolution
    print("\n1. Angular Resolution Validation...")
    validator = TransPlanckianResolutionValidator()
    validation = validator.validate_paper_claim()

    results['angular_resolution'] = {
        'baseline_km': validation['baseline_km'],
        'wavelength_nm': validation['wavelength_nm'],
        'paper_claim_microarcsec': f"{validation['paper_claim_microarcsec']:.2e}",
        'calculated_microarcsec': f"{validation['calculated_microarcsec']:.2e}",
        'agreement': validation['agreement']
    }

    # Exoplanet detection capability
    survey = validator.exoplanet_survey_capability()
    results['exoplanet_imaging'] = {
        'num_scenarios': survey['num_scenarios'],
        'num_resolvable': survey['num_resolvable'],
        'num_imageable': survey['num_imageable'],
        'success_rate_resolve': f"{survey['success_rate_resolve'] * 100:.0f}%",
        'success_rate_image': f"{survey['success_rate_image'] * 100:.0f}%"
    }

    # 2. Atmospheric Immunity
    print("2. Atmospheric Immunity Validation...")
    atm_experiment = AtmosphericComparisonExperiment(wavelength)
    atm_validation = atm_experiment.categorical.validate_paper_claims()

    results['atmospheric_immunity'] = {
        'baseline_paper_km': atm_validation['baseline_paper_claim'] / 1e3,
        'conventional_limit_m': f"{atm_validation['conventional_baseline_limit']:.2f}",
        'categorical_limit_km': atm_validation['categorical_baseline_limit'] / 1e3,
        'baseline_extension_factor': f"{atm_validation['baseline_extension_factor']:.2e}",
        'conventional_visibility_at_10000km': f"{atm_validation['conventional_visibility_at_10000km']:.2e}",
        'categorical_coherence_at_10000km': f"{atm_validation['categorical_coherence_at_10000km']:.6f}",
        'immunity_factor': f"{atm_validation['atmospheric_immunity_factor']:.2e}",
        'claim_validated': atm_validation['paper_claim_validated']
    }

    # Generate atmospheric comparison plot
    print("   Generating atmospheric immunity plots...")
    atm_experiment.plot_atmospheric_comparison(
        report.output_dir / f"atmospheric_immunity_{report.timestamp}.png"
    )

    # 3. Baseline Coherence
    print("3. Baseline Coherence Validation...")
    coherence_analyzer = BaselineCoherenceAnalyzer(wavelength)

    # Test at paper's baseline
    baseline_paper = 1e7  # 10,000 km
    conv_coherence = coherence_analyzer.conventional_baseline_coherence(baseline_paper, 1e-3)
    cat_coherence = coherence_analyzer.categorical_baseline_coherence(baseline_paper, 1e-3)

    results['baseline_coherence'] = {
        'test_baseline_km': baseline_paper / 1e3,
        'conventional_visibility': f"{conv_coherence.fringe_visibility:.2e}",
        'categorical_visibility': f"{cat_coherence.fringe_visibility:.6f}",
        'conventional_snr': f"{conv_coherence.snr:.2f}",
        'categorical_snr': f"{cat_coherence.snr:.2f}",
        'coherence_advantage': f"{cat_coherence.fringe_visibility / max(conv_coherence.fringe_visibility, 1e-10):.2e}"
    }

    # Generate coherence plots
    print("   Generating baseline coherence plots...")
    fringe_experiment = FringeVisibilityExperiment(wavelength)
    fringe_experiment.plot_coherence_validation(
        report.output_dir / f"baseline_coherence_{report.timestamp}.png"
    )

    # 4. Phase Correlation
    print("4. Phase Correlation Validation...")
    N_stations = 10
    positions = np.random.randn(N_stations, 3)
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= 1e7  # 10,000 km scale

    interferometer = TransPlanckianInterferometer(wavelength, positions)

    results['phase_correlation'] = {
        'num_stations': N_stations,
        'max_baseline_km': f"{max([np.linalg.norm(b) for b in interferometer.baselines]) / 1e3:.1f}",
        'angular_resolution_microarcsec': f"{interferometer.angular_resolution_microarcsec():.2e}",
        'paper_claim_microarcsec': "1.0e-05",
        'ratio': f"{interferometer.angular_resolution_microarcsec() / 1e-5:.2f}"
    }

    print("✓ Interferometry validation complete")
    report.add_section("Trans-Planckian Interferometry", results)


def validate_thermometry(report: ValidationReport):
    """Validate categorical quantum thermometry"""
    print("\n" + "=" * 70)
    print("VALIDATING CATEGORICAL QUANTUM THERMOMETRY")
    print("=" * 70)

    results = {}

    # Rb-87 parameters
    m_Rb87 = 1.443e-25  # kg
    N_atoms = 1e5

    # 1. Temperature Extraction
    print("\n1. Temperature Extraction Validation...")
    thermometer = ThermometryAnalyzer(m_Rb87)

    # Test at T = 100 nK
    T_test = 100e-9  # K
    sigma_v = np.sqrt(1.380649e-23 * T_test / m_Rb87)
    velocities = np.random.normal(0, sigma_v, int(N_atoms))
    momenta = m_Rb87 * velocities

    T_measured, delta_T = thermometer.extract_temperature_from_momentum_distribution(momenta)

    results['temperature_extraction'] = {
        'true_temperature_nK': f"{T_test * 1e9:.3f}",
        'measured_temperature_nK': f"{T_measured * 1e9:.3f}",
        'uncertainty_pK': f"{delta_T * 1e12:.2f}",
        'relative_precision': f"{delta_T / T_measured:.2e}",
        'paper_claim_pK': "17",
        'claim_validated': delta_T < 20e-12
    }

    # 2. Momentum Recovery
    print("2. Momentum Recovery Validation...")
    recovery = MomentumRecovery(m_Rb87, N_atoms)

    # Create categorical state
    from categorical.categorical_state import CategoricalStateEstimator
    estimator = CategoricalStateEstimator(m_Rb87, N_atoms)
    cat_state = estimator.from_momentum_distribution(momenta, 0.0)

    # Recover temperature
    T_recovered = recovery.temperature_from_momentum_entropy(cat_state.S.Sk)

    # Reconstruct momentum distribution
    reconstructed_momenta = recovery.reconstruct_momentum_distribution(cat_state, 10000)
    validation_metrics = recovery.validate_reconstruction(
        momenta[:10000].reshape(-1, 1),
        reconstructed_momenta
    )

    results['momentum_recovery'] = {
        'recovered_temperature_nK': f"{T_recovered * 1e9:.3f}",
        'temperature_error': f"{validation_metrics['temperature_error'] * 100:.2f}%",
        'momentum_error': f"{validation_metrics['mean_momentum_error'] * 100:.2f}%",
        'ks_test_pvalue': f"{validation_metrics['ks_pvalue']:.4f}",
        'distributions_match': validation_metrics['distributions_match']
    }

    # 3. Quantum Backaction
    print("3. Quantum Backaction Analysis...")
    backaction = QuantumBackactionAnalyzer(m_Rb87)

    wavelength_D2 = 780e-9  # Rb D2 line
    T_recoil = backaction.photon_recoil_temperature(wavelength_D2)

    comparison = backaction.backaction_comparison(T_test, wavelength_D2, 1e-3)

    results['quantum_backaction'] = {
        'photon_recoil_temperature_nK': f"{T_recoil * 1e9:.1f}",
        'conventional_heating_nK': f"{comparison['conventional_heating'] * 1e9:.2f}",
        'categorical_heating_fK': f"{comparison['categorical_heating'] * 1e15:.2f}",
        'improvement_factor': f"{comparison['improvement_factor']:.2e}",
        'conventional_invasive': comparison['conventional_invasive'],
        'categorical_invasive': comparison['categorical_invasive'],
        'categorical_advantage': comparison['categorical_advantage']
    }

    # 4. TOF Comparison
    print("4. TOF vs Categorical Comparison...")
    tof_comparison = CategoricalThermometryComparison(m_Rb87, N_atoms)
    validation_tof = tof_comparison.validate_paper_claims()

    results['tof_comparison'] = {
        'test_temperature_nK': f"{validation_tof['test_temperature'] * 1e9:.1f}",
        'tof_uncertainty_pK': f"{validation_tof['tof_uncertainty'] * 1e12:.2f}",
        'categorical_uncertainty_pK': f"{validation_tof['categorical_uncertainty'] * 1e12:.2f}",
        'resolution_claim_validated': validation_tof['resolution_claim_validated'],
        'tof_relative_precision': f"{validation_tof['tof_relative_precision']:.2e}",
        'categorical_relative_precision': f"{validation_tof['categorical_relative_precision']:.2e}",
        'precision_improvement': f"{validation_tof['precision_improvement']:.2e}",
        'heating_claim_validated': validation_tof['heating_claim_validated'],
        'tof_destructive': validation_tof['tof_destructive'],
        'categorical_non_destructive': not validation_tof['categorical_destructive']
    }

    # Generate TOF comparison plots
    print("   Generating TOF comparison plots...")
    tof_comparison.plot_comparative_analysis(
        report.output_dir / f"thermometry_tof_comparison_{report.timestamp}.png"
    )

    print("✓ Thermometry validation complete")
    report.add_section("Categorical Quantum Thermometry", results)


def main():
    """Run all validation experiments"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("Categorical State Propagation & Trans-Planckian Measurements")
    print("=" * 70)

    start_time = time.time()

    # Initialize report
    report = ValidationReport()

    # Run validation modules
    try:
        validate_categorical_framework(report)
        validate_interferometry(report)
        validate_thermometry(report)

        # Generate reports
        print("\n" + "=" * 70)
        print("GENERATING VALIDATION REPORTS")
        print("=" * 70)

        report.save_json()
        report.generate_markdown_report()

        # Summary
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        print(f"\n✓ Total time: {elapsed_time:.2f} seconds")
        print(f"✓ Reports saved to: {report.output_dir}/")
        print(f"\nGenerated files:")
        print(f"  - validation_report_{report.timestamp}.json")
        print(f"  - validation_report_{report.timestamp}.md")
        print(f"  - atmospheric_immunity_{report.timestamp}.png")
        print(f"  - baseline_coherence_{report.timestamp}.png")
        print(f"  - thermometry_tof_comparison_{report.timestamp}.png")

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
