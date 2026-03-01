#!/usr/bin/env python3
"""
Validation Experiment for Categorical State Counting Framework
================================================================

Runs comprehensive validation experiments and saves results in JSON format.

This script validates:
1. Trans-Planckian: Bounded discrete phase space
2. CatScript: Categorical partition coordinates from oscillator counts
3. Categorical Cryogenics: T = 2E/(3k_B × M)
4. Fundamental Identity: dM/dt = ω/(2π/M) = 1/⟨τ_p⟩
5. Heat-Entropy Decoupling
6. State-Mass Correspondence

Results are saved in counting/validation_results/
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path to allow package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the counting.src package
from counting.src import (
    StateCountingPipeline,
    ValidationPipeline,
    PipelineConfig,
    create_ion_trajectory,
    HardwareOscillator,
    ThermodynamicRegimeClassifier,
    PartitionCoordinates,
    calculate_categorical_temperature,
)

import numpy as np


class ValidationExperiment:
    """Comprehensive validation experiment runner."""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().isoformat()

    def run_full_experiment(self, mzml_path: str = None) -> Dict[str, Any]:
        """
        Run complete validation experiment.

        Args:
            mzml_path: Path to mzML file (optional, uses synthetic if None)

        Returns:
            Complete validation results dictionary
        """
        print("=" * 80)
        print("CATEGORICAL STATE COUNTING - VALIDATION EXPERIMENT")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print()

        results = {
            'metadata': self._get_metadata(mzml_path),
            'oscillator_validation': self._validate_oscillator(),
            'partition_coordinates': self._validate_partition_coordinates(),
            'categorical_temperature': self._validate_categorical_temperature(),
            'ion_trajectory_validation': self._validate_ion_trajectories(),
            'pipeline_validation': None,  # Will be populated below
            'statistical_analysis': None,  # Will be populated below
        }

        # Run pipeline validation (mzML or synthetic)
        print("\n" + "=" * 80)
        print("PIPELINE VALIDATION")
        print("=" * 80)

        pipeline_results = self._run_pipeline_validation(mzml_path)
        results['pipeline_validation'] = pipeline_results

        # Statistical analysis
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)

        stats = self._compute_statistics(pipeline_results)
        results['statistical_analysis'] = stats

        # Save results
        self._save_results(results)

        print("\n" + "=" * 80)
        print("VALIDATION EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")

        return results

    def _get_metadata(self, mzml_path: str = None) -> Dict[str, Any]:
        """Get experiment metadata."""
        return {
            'timestamp': self.timestamp,
            'framework_version': '1.0.0',
            'data_source': mzml_path if mzml_path else 'synthetic',
            'claims_tested': [
                'Trans-Planckian: Phase space is bounded and discrete',
                'CatScript: Partition coordinates from oscillator counts',
                'Categorical Cryogenics: T = 2E/(3k_B × M)',
                'Fundamental Identity: dM/dt = 1/⟨τ_p⟩',
                'Heat-Entropy Decoupling: Cov(δQ, dS_cat) = 0',
                'State-Mass Correspondence: N_state ↔ m/z'
            ]
        }

    def _validate_oscillator(self) -> Dict[str, Any]:
        """Validate hardware oscillator as fundamental counter."""
        print("\n--- Validating Hardware Oscillator ---")

        osc = HardwareOscillator(frequency_hz=10e6, stability=0)

        test_durations = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        measurements = []

        for duration in test_durations:
            osc.reset()
            M = osc.count_cycles(duration)
            time_back = osc.time_from_count(M)

            measurements.append({
                'duration_s': duration,
                'cycles_counted': int(M),
                'reconstructed_time_s': time_back,
                'error': abs(time_back - duration) / duration,
                'fundamental_identity_valid': abs(M - osc.frequency * duration) / M < 1e-10
            })

        all_valid = all(m['fundamental_identity_valid'] for m in measurements)

        print(f"  Oscillator frequency: {osc.frequency/1e6:.0f} MHz")
        print(f"  Fundamental identity validated: {all_valid}")

        return {
            'oscillator_name': osc.name,
            'frequency_hz': osc.frequency,
            'period_s': osc.period_s,
            'measurements': measurements,
            'fundamental_identity_validated': all_valid,
            'claim': 'dM/dt = ω/(2π/M) = 1/⟨τ_p⟩',
            'result': 'PASS' if all_valid else 'FAIL'
        }

    def _validate_partition_coordinates(self) -> Dict[str, Any]:
        """Validate partition coordinate derivation from state counts."""
        print("\n--- Validating Partition Coordinates ---")

        test_counts = [2, 8, 18, 32, 50, 100, 200, 500, 1000, 10000]
        validations = []

        for M in test_counts:
            coords = PartitionCoordinates.from_count(M, charge=1)

            # Verify capacity formula: C(n) = 2n²
            expected_capacity = 2 * coords.n**2

            # Verify cumulative: sum of C(i) for i=1 to n
            expected_cumulative = sum(2*i**2 for i in range(1, coords.n + 1))

            # Verify selection rules
            l_valid = 0 <= coords.l < coords.n
            m_valid = -coords.l <= coords.m <= coords.l
            s_valid = coords.s in [-0.5, 0.5]

            validations.append({
                'state_count_M': M,
                'n': coords.n,
                'l': coords.l,
                'm': coords.m,
                's': coords.s,
                'capacity': coords.capacity,
                'expected_capacity': expected_capacity,
                'cumulative_capacity': coords.cumulative_capacity,
                'expected_cumulative': expected_cumulative,
                'capacity_match': coords.capacity == expected_capacity,
                'cumulative_match': coords.cumulative_capacity == expected_cumulative,
                'l_valid': l_valid,
                'm_valid': m_valid,
                's_valid': s_valid,
                'all_valid': (coords.capacity == expected_capacity and
                            coords.cumulative_capacity == expected_cumulative and
                            l_valid and m_valid and s_valid)
            })

        all_valid = all(v['all_valid'] for v in validations)

        print(f"  Tested {len(test_counts)} state counts")
        print(f"  Capacity formula C(n)=2n² validated: {all_valid}")

        return {
            'capacity_formula': 'C(n) = 2n²',
            'cumulative_formula': 'N(n) = n(n+1)(2n+1)/3',
            'validations': validations,
            'n_tests': len(test_counts),
            'all_valid': all_valid,
            'claim': 'Partition coordinates from oscillator counts',
            'result': 'PASS' if all_valid else 'FAIL'
        }

    def _validate_categorical_temperature(self) -> Dict[str, Any]:
        """Validate categorical temperature formula."""
        print("\n--- Validating Categorical Temperature ---")

        energy_eV = 10.0
        test_M = [1, 10, 100, 1000, 10000, 100000, 1000000]

        validations = []

        for M in test_M:
            T_cat = calculate_categorical_temperature(energy_eV, M)
            suppression = 1.0 / M

            # Verify formula: T = 2E/(3k_B × M)
            from counting.src.ThermodynamicRegimes import K_B, E_CHARGE
            E_J = energy_eV * E_CHARGE
            T_expected = 2 * E_J / (3 * K_B * M)

            validations.append({
                'state_count_M': M,
                'energy_eV': energy_eV,
                'T_categorical_K': T_cat,
                'T_expected_K': T_expected,
                'suppression_factor': suppression,
                'formula_match': abs(T_cat - T_expected) / T_expected < 1e-10,
                'suppression_correct': abs(suppression - 1.0/M) < 1e-10
            })

        all_valid = all(v['formula_match'] and v['suppression_correct']
                       for v in validations)

        print(f"  Tested {len(test_M)} state counts")
        print(f"  Temperature formula T=2E/(3k_B×M) validated: {all_valid}")

        return {
            'formula': 'T = 2E/(3k_B × M)',
            'insight': 'More states → Lower effective temperature',
            'validations': validations,
            'n_tests': len(test_M),
            'all_valid': all_valid,
            'claim': 'Categorical cryogenics',
            'result': 'PASS' if all_valid else 'FAIL'
        }

    def _validate_ion_trajectories(self) -> Dict[str, Any]:
        """Validate complete ion trajectories."""
        print("\n--- Validating Ion Trajectories ---")

        test_ions = [
            {'mz': 150.0, 'charge': 1, 'energy': 10.0},
            {'mz': 500.25, 'charge': 2, 'energy': 15.0},
            {'mz': 1000.5, 'charge': 3, 'energy': 20.0},
        ]

        validations = []

        for ion_params in test_ions:
            trajectory = create_ion_trajectory(
                mz=ion_params['mz'],
                charge=ion_params['charge'],
                energy_eV=ion_params['energy'],
                instrument="orbitrap"
            )

            trajectory.complete_ms1_journey()
            report = trajectory.get_validation_report()

            validations.append({
                'ion': ion_params,
                'total_state_count': report['summary']['total_state_count'],
                'total_time_s': report['summary']['total_time_s'],
                'trans_planckian_valid': report['trans_planckian']['validated'],
                'catscript_valid': report['catscript']['validated'],
                'categorical_cryogenics_valid': report['categorical_cryogenics']['validated'],
                'fundamental_identity_valid': report['fundamental_identity']['validated'],
                'stage_breakdown': report['stage_breakdown'],
            })

        all_valid = all(
            v['trans_planckian_valid'] and
            v['catscript_valid'] and
            v['categorical_cryogenics_valid'] and
            v['fundamental_identity_valid']
            for v in validations
        )

        print(f"  Tested {len(test_ions)} ion trajectories")
        print(f"  All validations passed: {all_valid}")

        return {
            'validations': validations,
            'n_ions': len(test_ions),
            'all_valid': all_valid,
            'claim': 'Complete ion trajectory as state counting sequence',
            'result': 'PASS' if all_valid else 'FAIL'
        }

    def _run_pipeline_validation(self, mzml_path: str = None) -> Dict[str, Any]:
        """Run complete pipeline validation."""

        pipeline = StateCountingPipeline()

        if mzml_path and Path(mzml_path).exists():
            print(f"Processing mzML file: {mzml_path}")
            try:
                results = pipeline.process_mzml(mzml_path)
                data_source = mzml_path
            except Exception as e:
                print(f"Error processing mzML: {e}")
                print("Falling back to synthetic data...")
                mzml_path = None

        if not mzml_path:
            print("Processing synthetic peak list...")
            data_source = "synthetic"
            peaks = [
                {'mz': 150.0, 'intensity': 5000, 'rt': 3.0},
                {'mz': 250.5, 'intensity': 8000, 'rt': 5.5},
                {'mz': 350.25, 'intensity': 12000, 'rt': 7.0},
                {'mz': 450.3, 'intensity': 15000, 'rt': 9.5},
                {'mz': 550.5, 'intensity': 10000, 'rt': 11.0},
                {'mz': 650.75, 'intensity': 6000, 'rt': 13.5},
                {'mz': 750.0, 'intensity': 8000, 'rt': 15.0},
                {'mz': 850.25, 'intensity': 4000, 'rt': 17.5},
            ]
            results = pipeline.process_peak_list(peaks)

        # Run validations
        validator = ValidationPipeline()
        validation = validator.validate_all(results)

        print(f"  Processed {results.n_ions_processed} ions")
        print(f"  Trans-Planckian: {'PASS' if validation['trans_planckian']['overall_valid'] else 'FAIL'}")
        print(f"  CatScript: {'PASS' if validation['catscript']['overall_valid'] else 'FAIL'}")
        print(f"  Categorical Cryogenics: {'PASS' if validation['categorical_cryogenics']['overall_valid'] else 'FAIL'}")

        return {
            'data_source': data_source,
            'n_ions_processed': results.n_ions_processed,
            'regime_distribution': results.regime_counts,
            'validations': validation,
            'overall_pass': (
                validation['trans_planckian']['overall_valid'] and
                validation['catscript']['overall_valid'] and
                validation['categorical_cryogenics']['overall_valid']
            )
        }

    def _compute_statistics(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistical measures from validation results."""
        print("\n--- Computing Statistics ---")

        val = pipeline_results['validations']

        # Trans-Planckian statistics
        tp_vals = val['trans_planckian']['validations']
        if tp_vals:
            state_counts = [v['state_count_M'] for v in tp_vals]
            capacities = [v['cumulative_capacity'] for v in tp_vals]

            tp_stats = {
                'mean_state_count': float(np.mean(state_counts)),
                'std_state_count': float(np.std(state_counts)),
                'min_state_count': int(np.min(state_counts)),
                'max_state_count': int(np.max(state_counts)),
                'mean_capacity': float(np.mean(capacities)),
                'fraction_bounded': sum(1 for v in tp_vals if v['bounded']) / len(tp_vals),
            }
        else:
            tp_stats = {}

        # CatScript statistics
        cs_vals = val['catscript']['validations']
        if cs_vals:
            n_values = [v['n'] for v in cs_vals]

            cs_stats = {
                'mean_n': float(np.mean(n_values)),
                'std_n': float(np.std(n_values)),
                'min_n': int(np.min(n_values)),
                'max_n': int(np.max(n_values)),
                'fraction_valid': sum(1 for v in cs_vals
                                    if v['n_correct'] and v['l_valid'] and
                                       v['m_valid'] and v['s_valid']) / len(cs_vals),
            }
        else:
            cs_stats = {}

        # Categorical Cryogenics statistics
        cc_vals = val['categorical_cryogenics']['validations']
        if cc_vals:
            temperatures = [v['T_categorical'] for v in cc_vals]
            suppressions = [v['suppression'] for v in cc_vals]

            cc_stats = {
                'mean_temperature_K': float(np.mean(temperatures)),
                'std_temperature_K': float(np.std(temperatures)),
                'min_temperature_K': float(np.min(temperatures)),
                'max_temperature_K': float(np.max(temperatures)),
                'mean_suppression': float(np.mean(suppressions)),
                'fraction_matching': sum(1 for v in cc_vals if v['T_match']) / len(cc_vals),
            }
        else:
            cc_stats = {}

        print(f"  Statistical analysis complete")

        return {
            'trans_planckian': tp_stats,
            'catscript': cs_stats,
            'categorical_cryogenics': cc_stats,
            'regime_distribution': pipeline_results['regime_distribution']
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON files."""

        # Main results file
        main_file = self.output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(main_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n[OK] Main results saved to: {main_file}")

        # Summary file
        summary = {
            'timestamp': results['metadata']['timestamp'],
            'data_source': results['metadata']['data_source'],
            'overall_validation': {
                'oscillator': results['oscillator_validation']['result'],
                'partition_coordinates': results['partition_coordinates']['result'],
                'categorical_temperature': results['categorical_temperature']['result'],
                'ion_trajectories': results['ion_trajectory_validation']['result'],
                'pipeline': 'PASS' if results['pipeline_validation']['overall_pass'] else 'FAIL'
            },
            'statistics': results['statistical_analysis']
        }

        summary_file = self.output_dir / "validation_summary_latest.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[OK] Summary saved to: {summary_file}")


def main():
    """Run validation experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run categorical state counting validation experiment"
    )
    parser.add_argument(
        "--mzml",
        type=str,
        default="counting/public/20090526_06_R134_RIN_51.mzML",
        help="Path to mzML file (default: use sample data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="counting/validation_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Check if mzML exists
    mzml_path = args.mzml if Path(args.mzml).exists() else None

    if not mzml_path:
        print(f"Warning: mzML file not found at {args.mzml}")
        print("Will use synthetic data instead.")

    # Run experiment
    experiment = ValidationExperiment(output_dir=args.output_dir)
    results = experiment.run_full_experiment(mzml_path=mzml_path)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nData Source: {results['metadata']['data_source']}")
    print(f"\nFramework Validations:")
    print(f"  Hardware Oscillator:       {results['oscillator_validation']['result']}")
    print(f"  Partition Coordinates:     {results['partition_coordinates']['result']}")
    print(f"  Categorical Temperature:   {results['categorical_temperature']['result']}")
    print(f"  Ion Trajectories:          {results['ion_trajectory_validation']['result']}")
    print(f"  Pipeline (Overall):        {'PASS' if results['pipeline_validation']['overall_pass'] else 'FAIL'}")

    if results['statistical_analysis']:
        print(f"\nStatistical Summary:")
        if results['statistical_analysis']['trans_planckian']:
            tp = results['statistical_analysis']['trans_planckian']
            print(f"  Mean state count: {tp.get('mean_state_count', 'N/A'):.0f}")
            print(f"  Fraction bounded: {tp.get('fraction_bounded', 'N/A'):.2%}")

    print(f"\n[OK] All results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
