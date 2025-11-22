#!/usr/bin/env python3
"""
Master Experiment Runner - Transcendent Observer BMD System
=============================================================
Runs all navigation experiments and saves results for each stage.
Demonstrates the complete BMD system operating at the transcendent observer level.

This script:
1. Runs each module independently
2. Saves results at each stage in accessible formats (JSON + figures)
3. Creates a comprehensive system-level summary
4. Validates BMD equivalence across all pathways
"""

import os
import sys
import json
from datetime import datetime
import numpy as np

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# JSON serialization helper for numpy types
def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def create_results_structure():
    """Create organized directory structure for results"""
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join(base_dir, f'transcendent_observer_{timestamp}')

    subdirs = [
        'entropy_navigation',
        'finite_observer',
        'fourier_transform',
        'recursive_observers',
        'harmonic_extraction',
        'harmonic_network',
        'molecular_vibrations',
        'multidomain_seft',
        'led_excitation',
        'hardware_clock',
        'bmd_equivalence',
        'system_summary'
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir, timestamp


def run_experiment(name, func, results_dir):
    """Run a single experiment and handle errors"""
    print(f"\n{'='*70}")
    print(f"   EXPERIMENT: {name.upper()}")
    print(f"{'='*70}\n")

    try:
        result = func(results_dir)
        print(f"\n✓ {name} completed successfully")
        return {'status': 'success', 'result': result}
    except Exception as e:
        print(f"\n✗ {name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def experiment_1_entropy_navigation(results_dir):
    """Experiment 1: S-Entropy Miraculous Navigation"""
    from entropy_navigation import SEntropyNavigator

    navigator = SEntropyNavigator(precision=47e-21)

    # Test navigation
    states = [
        np.array([10, 100, 50, 42]),  # Initial
        np.array([50, 500, 5, 7])      # Target
    ]

    nav_result = navigator.navigate_s_space(states[0], states[1])
    decoupling_demo = navigator.demonstrate_decoupling()

    # Save results
    results = {
        'navigation': {
            'initial_state': states[0].tolist(),
            'target_state': states[1].tolist(),
            'navigation_time_s': float(nav_result['navigation_time']),
            'temporal_precision_zs': float(nav_result['temporal_precision'] * 1e21),
            'speed_ratio': float(nav_result['speed_ratio'])
        },
        'decoupling': decoupling_demo
    }

    with open(os.path.join(results_dir, 'entropy_navigation', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_2_finite_observer(results_dir):
    """Experiment 2: Finite Observer Verification"""
    from finite_observer_verification import FiniteObserverSimulator

    simulator = FiniteObserverSimulator(true_frequency=7.1e13)

    # Compare traditional vs miraculous
    comparison = simulator.compare_traditional_vs_miraculous()

    # Save results
    results = {
        'traditional': comparison['traditional'],
        'miraculous': comparison['miraculous'],
        'speedup_factor': float(comparison['miraculous_speedup'])
    }

    with open(os.path.join(results_dir, 'finite_observer', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_3_fourier_transform(results_dir):
    """Experiment 3: Multi-Domain S-Entropy Fourier Transform"""
    from fourier_transform_coordinates import MultiDomainSEFT, SEFTParameters

    seft = MultiDomainSEFT()

    # Generate test signal
    duration = 100e-15
    n_samples = 2048
    time_points = np.linspace(0, duration, n_samples)
    signal = np.sin(2*np.pi*7.1e13*time_points) + 0.1*np.random.randn(n_samples)

    # Run 4-pathway SEFT
    params = SEFTParameters()
    seft_result = seft.four_pathway_seft(signal, time_points, params)

    # Save results
    results = {
        'baseline_precision_zs': float(seft_result['baseline_precision'] * 1e21),
        'enhanced_precision_zs': float(seft_result['enhanced_precision'] * 1e21),
        'enhancement_factor': float(seft_result['enhancement_factor']),
        'pathway_contributions': {
            'standard_time': float(seft_result['pathways']['standard_time']['precision_fs']),
            'entropy': float(seft_result['pathways']['entropy']['precision_fs']),
            'convergence': float(seft_result['pathways']['convergence']['precision_fs']),
            'information': float(seft_result['pathways']['information']['precision_fs'])
        }
    }

    with open(os.path.join(results_dir, 'fourier_transform', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_4_recursive_observers(results_dir):
    """Experiment 4: Recursive Observer Nesting"""
    from gas_molecule_lattice import RecursiveObserverLattice

    lattice = RecursiveObserverLattice(n_molecules=100, chamber_size=1e-3)

    # Run recursive observation
    recursion_result = lattice.recursive_observe(recursion_depth=3, sample_size=20)

    # Save results
    results = {
        'recursion_levels': recursion_result['recursion_levels'],
        'precision_cascade_zs': [float(p * 1e21) for p in recursion_result['precision_cascade'][:3]],
        'active_observers': recursion_result['active_observers'][:3],
        'observation_paths': recursion_result['observation_paths'][:3]
    }

    with open(os.path.join(results_dir, 'recursive_observers', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_5_harmonic_extraction(results_dir):
    """Experiment 5: Harmonic Precision Multiplication"""
    from harmonic_extraction import HarmonicExtractor

    extractor = HarmonicExtractor(fundamental_frequency=7.1e13)

    # Generate signal with harmonics
    duration = 100 / 7.1e13
    n_samples = 2048
    time_points = np.linspace(0, duration, n_samples)

    signal = np.zeros(n_samples)
    for n in [1, 10, 50, 100]:
        signal += (1.0/n) * np.sin(2*np.pi*n*7.1e13*time_points)

    # Extract harmonics
    harmonics_data = extractor.extract_harmonics(signal, time_points, max_harmonic=100)
    optimal = extractor.find_optimal_harmonic(harmonics_data, coherence_time=247e-15)

    # Save results
    results = {
        'total_harmonics_found': harmonics_data['total_harmonics_found'],
        'optimal_harmonic': {
            'number': optimal['number'],
            'frequency_Hz': float(optimal['frequency']),
            'precision_as': float(optimal['precision_as']),
            'total_enhancement': float(optimal['total_enhancement'])
        }
    }

    with open(os.path.join(results_dir, 'harmonic_extraction', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_6_harmonic_network(results_dir):
    """Experiment 6: Harmonic Network Graph"""
    from harmonic_network_graph import HarmonicNetworkGraph

    network = HarmonicNetworkGraph(frequency_tolerance=1e11)

    # Build network
    stats = network.build_from_recursive_observations(
        n_molecules=30,
        base_frequency=7.1e13,
        max_depth=2,
        harmonics_per_molecule=10
    )

    # Calculate graph enhancement
    enhancement = network.precision_enhancement_from_graph()

    # Save results
    results = {
        'network_statistics': {
            'total_nodes': stats['total_nodes'],
            'total_edges': stats['total_edges'],
            'avg_degree': float(stats['avg_degree']),
            'graph_density': float(stats['density'])
        },
        'precision_enhancement': {
            'redundancy_factor': float(enhancement['redundancy_factor']),
            'amplification_factor': float(enhancement['amplification_factor']),
            'total_graph_enhancement': float(enhancement['total_graph_enhancement'])
        }
    }

    with open(os.path.join(results_dir, 'harmonic_network', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_7_molecular_vibrations(results_dir):
    """Experiment 7: Quantum Molecular Vibrations"""
    from molecular_vibrations import QuantumVibrationalAnalyzer

    analyzer = QuantumVibrationalAnalyzer(frequency=7.1e13, coherence_time=247e-15)

    # Calculate properties
    energies = analyzer.calculate_energy_levels(max_level=5)
    linewidth = analyzer.heisenberg_linewidth()
    precision = analyzer.temporal_precision()
    led_props = analyzer.led_enhanced_coherence(base_coherence=100e-15, led_enhancement=2.47)

    # Save results
    results = {
        'frequency_Hz': float(analyzer.frequency),
        'coherence_time_fs': float(analyzer.coherence_time * 1e15),
        'heisenberg_linewidth_Hz': float(linewidth),
        'temporal_precision_fs': float(precision * 1e15),
        'led_enhancement': {
            'enhanced_coherence_fs': float(led_props['enhanced_coherence_time'] * 1e15),
            'enhanced_precision_fs': float(led_props['enhanced_precision'] * 1e15),
            'precision_improvement': float(led_props['precision_improvement'])
        }
    }

    with open(os.path.join(results_dir, 'molecular_vibrations', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_8_multidomain_seft(results_dir):
    """Experiment 8: Miraculous Measurement System"""
    from multidomain_seft import MiraculousMeasurementSystem

    system = MiraculousMeasurementSystem(baseline_precision=47e-21)

    # Perform miraculous measurement
    true_freq = 7.1e13
    result = system.miraculous_frequency_measurement(true_freq, initial_uncertainty=0.1)

    # Save results
    results = {
        'true_frequency_Hz': float(result['true_frequency']),
        'initial_estimate_Hz': float(result['initial_estimate']),
        'measured_frequency_Hz': float(result['measured_frequency']),
        'temporal_precision_zs': float(result['temporal_precision'] * 1e21),
        'measurement_time_s': float(result['measurement_time']),
        'gap_analysis': {
            'relative_gap_percent': float(result['gap_analysis']['relative_gap'] * 100),
            'acceptable': result['gap_analysis']['acceptable']
        }
    }

    with open(os.path.join(results_dir, 'multidomain_seft', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_9_led_excitation(results_dir):
    """Experiment 9: LED Spectroscopy"""
    from led_excitation import LEDSpectroscopySystem

    led_system = LEDSpectroscopySystem()

    # Test on benzene
    pattern = 'c1ccccc1'
    analysis = led_system.analyze_molecular_fluorescence(pattern, 470)
    fluor_props = led_system.predict_fluorescence_properties(pattern)

    # Save results
    results = {
        'molecular_pattern': pattern,
        'led_wavelengths_nm': led_system.led_wavelengths,
        'excitation_efficiency': float(analysis['excitation_efficiency']),
        'fluorescence_intensity': float(analysis['fluorescence_intensity']),
        'fluorescence_properties': {
            'quantum_yield': float(fluor_props['quantum_yield']),
            'lifetime_ns': float(fluor_props['lifetime_ns']),
            'coherence_enhancement': float(fluor_props['coherence_enhancement_factor'])
        }
    }

    with open(os.path.join(results_dir, 'led_excitation', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_10_hardware_clock(results_dir):
    """Experiment 10: Hardware Clock Integration"""
    from hardware_clock_integration import HardwareClockSync

    sync = HardwareClockSync()

    # Test with sample patterns
    patterns = ['c1ccccc1', 'CCO', 'CC(=O)O']
    sync_results = sync.synchronize_molecular_hardware(patterns)

    # Calculate average efficiency
    efficiencies = [r['coordination_efficiency'] for r in sync_results.values()]
    avg_efficiency = np.mean(efficiencies)

    # Save results
    results = {
        'cpu_frequency_GHz': float(sync.cpu_frequency / 1e9),
        'patterns_tested': len(patterns),
        'avg_coordination_efficiency': float(avg_efficiency),
        'synchronization_results': {
            pattern: {
                'molecular_frequency_THz': float(data['molecular_frequency_THz']),
                'coordination_efficiency': float(data['coordination_efficiency'])
            }
            for pattern, data in sync_results.items()
        }
    }

    with open(os.path.join(results_dir, 'hardware_clock', 'results.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def experiment_11_bmd_equivalence(results_dir):
    """Experiment 11: BMD Equivalence Validation"""
    from bmd_equivalence import BMDEquivalenceValidator

    validator = BMDEquivalenceValidator()

    # Generate test signal
    signal, time_points = validator.generate_test_signal(n_samples=1024, frequency=7.1e13)

    # Validate BMD equivalence
    bmd_results = validator.validate_bmd_equivalence(signal, n_iterations=40)

    # Save detailed results
    validator.save_results(os.path.join(results_dir, 'bmd_equivalence'))

    # Simplified summary
    results = {
        'equivalence_achieved': bmd_results['equivalence_achieved'],
        'convergence_analysis': {
            'mean_final_variance': float(bmd_results['convergence_analysis']['mean_final_variance']),
            'variance_spread': float(bmd_results['convergence_analysis']['variance_spread']),
            'relative_spread': float(bmd_results['convergence_analysis']['relative_spread']),
            'final_variances': {k: float(v) for k, v in bmd_results['convergence_analysis']['final_variances'].items()}
        },
        'statistical_tests': {
            'f_statistic': float(bmd_results['statistical_tests']['f_statistic']),
            'p_value': float(bmd_results['statistical_tests']['p_value']),
            'equivalence_hypothesis': bmd_results['statistical_tests']['equivalence_hypothesis']
        }
    }

    with open(os.path.join(results_dir, 'bmd_equivalence', 'summary.json'), 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    return results


def create_system_summary(exp_dir, all_results):
    """Create comprehensive system-level summary"""
    summary = {
        'experiment_name': 'Transcendent Observer BMD System',
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(all_results),
        'successful_experiments': sum(1 for r in all_results.values() if r['status'] == 'success'),
        'failed_experiments': sum(1 for r in all_results.values() if r['status'] == 'failed'),
        'experiment_results': {}
    }

    # Extract key metrics from each experiment
    for name, result in all_results.items():
        if result['status'] == 'success':
            summary['experiment_results'][name] = result['result']
        else:
            summary['experiment_results'][name] = {'error': result.get('error', 'Unknown error')}

    # Overall system status
    summary['system_status'] = 'OPERATIONAL' if summary['failed_experiments'] == 0 else 'PARTIAL'

    # Save summary (convert numpy types to Python types)
    summary_serializable = convert_to_serializable(summary)
    with open(os.path.join(exp_dir, 'system_summary', 'complete_system_summary.json'), 'w') as f:
        json.dump(summary_serializable, f, indent=2)

    return summary


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("   TRANSCENDENT OBSERVER BMD SYSTEM")
    print("   Complete Experimental Suite")
    print("="*70)

    # Create results structure
    print("\n📁 Creating results directory structure...")
    exp_dir, timestamp = create_results_structure()
    print(f"   Results directory: {exp_dir}")

    # Define all experiments
    experiments = [
        ('Entropy Navigation', experiment_1_entropy_navigation),
        ('Finite Observer', experiment_2_finite_observer),
        ('Fourier Transform', experiment_3_fourier_transform),
        ('Recursive Observers', experiment_4_recursive_observers),
        ('Harmonic Extraction', experiment_5_harmonic_extraction),
        ('Harmonic Network', experiment_6_harmonic_network),
        ('Molecular Vibrations', experiment_7_molecular_vibrations),
        ('Multidomain SEFT', experiment_8_multidomain_seft),
        ('LED Excitation', experiment_9_led_excitation),
        ('Hardware Clock', experiment_10_hardware_clock),
        ('BMD Equivalence', experiment_11_bmd_equivalence)
    ]

    # Run all experiments
    all_results = {}

    for i, (name, func) in enumerate(experiments, 1):
        print(f"\n\n{'#'*70}")
        print(f"   [{i}/{len(experiments)}] {name}")
        print(f"{'#'*70}")

        result = run_experiment(name, func, exp_dir)
        all_results[name.lower().replace(' ', '_')] = result

    # Create system summary
    print(f"\n\n{'='*70}")
    print("   CREATING SYSTEM SUMMARY")
    print(f"{'='*70}")

    summary = create_system_summary(exp_dir, all_results)

    # Print final summary
    print(f"\n\n{'='*70}")
    print("   TRANSCENDENT OBSERVER BMD SYSTEM - COMPLETE")
    print(f"{'='*70}")
    print(f"\n   Timestamp: {timestamp}")
    print(f"   Total Experiments: {summary['total_experiments']}")
    print(f"   Successful: {summary['successful_experiments']}")
    print(f"   Failed: {summary['failed_experiments']}")
    print(f"   System Status: {summary['system_status']}")
    print(f"\n   Results Location: {exp_dir}")
    print(f"\n{'='*70}\n")

    # List all result files
    print("📊 RESULTS SAVED:")
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith('.json'):
                rel_path = os.path.relpath(os.path.join(root, file), exp_dir)
                print(f"   ✓ {rel_path}")

    print(f"\n{'='*70}")
    print("   ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}\n")

    return exp_dir, all_results, summary


if __name__ == "__main__":
    exp_dir, results, summary = main()
