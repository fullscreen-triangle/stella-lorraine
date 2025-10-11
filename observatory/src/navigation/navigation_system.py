#!/usr/bin/env python3
"""
Navigation Module Test Script
==============================
Tests ALL components in the navigation module independently.

Components tested:
- entropy_navigation
- finite_observer_verification
- fourier_transform_coordinates
- gas_molecule_lattice
- hardware_clock_integration
- harmonic_extraction
- harmonic_network_graph
- led_excitation
- molecular_vibrations
- multidomain_seft
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("   NAVIGATION MODULE COMPREHENSIVE TEST")
print("="*70)

# Setup results directory
results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'navigation_module')
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    'timestamp': timestamp,
    'module': 'navigation',
    'components_tested': []
}

# Test 1: Entropy Navigation
print("\n[1/10] Testing: entropy_navigation.py")
try:
    from entropy_navigation import SEntropyNavigator

    navigator = SEntropyNavigator(precision=47e-21)

    # Test navigation
    current_state = {'S': 0.0, 'tau': 1e-9, 'I': 5.0, 't': 0.0}
    target_state = {'S': 100.0, 'tau': 1e-12, 'I': 8.0, 't': 1e-12}
    nav_result = navigator.navigate(current_state, target_state, lambda_steps=50, allow_miraculous=True)

    # Test decoupling demonstration
    decoupling = navigator.demonstrate_decoupling()

    results['components_tested'].append({
        'component': 'entropy_navigation',
        'status': 'success',
        'tests': {
            'navigation_speed': float(nav_result['navigation_velocity']),
            'temporal_precision': float(nav_result['temporal_precision']),
            'miraculous_states': sum(1 for s in nav_result['path'] if s['miraculous']),
            'decoupling_scenarios': len(decoupling['scenarios'])
        }
    })
    print("   ✓ SEntropyNavigator working")

except Exception as e:
    results['components_tested'].append({
        'component': 'entropy_navigation',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 2: Finite Observer Verification
print("\n[2/10] Testing: finite_observer_verification.py")
try:
    from finite_observer_verification import FiniteObserverSimulator

    simulator = FiniteObserverSimulator(true_frequency=7.1e13)

    # Test traditional measurement
    trad_result = simulator.traditional_measurement(n_cycles=50, fft_samples=2048)

    # Test miraculous navigation
    mirac_result = simulator.miraculous_navigation()

    # Test comparison
    comparison = simulator.compare_methods(n_trials=3)

    results['components_tested'].append({
        'component': 'finite_observer_verification',
        'status': 'success',
        'tests': {
            'traditional_time_ms': float(trad_result['total_time'] * 1e3),
            'miraculous_time_us': float(mirac_result['total_time'] * 1e6),
            'speed_advantage': float(comparison['speed_advantage']),
            'miraculous_precision_zs': float(mirac_result['temporal_precision'] * 1e21)
        }
    })
    print("   ✓ FiniteObserverSimulator working")

except Exception as e:
    results['components_tested'].append({
        'component': 'finite_observer_verification',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 3: Fourier Transform Coordinates
print("\n[3/10] Testing: fourier_transform_coordinates.py")
try:
    from fourier_transform_coordinates import MultiDomainSEFT, SEFTParameters

    seft = MultiDomainSEFT()

    # Create test signal
    duration = 100e-15
    n_samples = 2**10
    time_points = np.linspace(0, duration, n_samples)
    signal = np.sin(2*np.pi*7.1e13*time_points) + 0.05*np.random.randn(n_samples)

    # Test all domain transforms
    entropy_coords = np.cumsum(np.abs(np.random.randn(n_samples))**2)
    convergence_times = np.exp(-np.linspace(0, 5, n_samples)) * 1e-9
    information_coords = np.linspace(0, 10, n_samples)

    transform_results = seft.transform_all_domains(
        signal, time_points, entropy_coords, convergence_times, information_coords
    )

    results['components_tested'].append({
        'component': 'fourier_transform_coordinates',
        'status': 'success',
        'tests': {
            'total_enhancement': float(transform_results['total_enhancement']),
            'entropy_enhancement': float(transform_results['entropy']['precision_enhancement']),
            'convergence_enhancement': float(transform_results['convergence']['precision_enhancement']),
            'information_enhancement': float(transform_results['information']['precision_enhancement']),
            'consensus_frequency_Hz': float(transform_results['consensus_frequency'])
        }
    })
    print("   ✓ MultiDomainSEFT working")

except Exception as e:
    results['components_tested'].append({
        'component': 'fourier_transform_coordinates',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 4: Gas Molecule Lattice
print("\n[4/10] Testing: gas_molecule_lattice.py")
try:
    from gas_molecule_lattice import RecursiveObserverLattice, MolecularObserver

    # Create small lattice for testing
    lattice = RecursiveObserverLattice(n_molecules=100, chamber_size=1e-3)

    # Test recursive observation
    recursion_results = lattice.recursive_observe(recursion_depth=3, sample_size=10)

    # Test precision calculation
    final_precision = recursion_results['precision_cascade'][-1]
    planck_analysis = lattice.calculate_precision_vs_planck(final_precision)

    results['components_tested'].append({
        'component': 'gas_molecule_lattice',
        'status': 'success',
        'tests': {
            'n_molecules': lattice.n_molecules,
            'recursion_levels': len(recursion_results['recursion_levels']),
            'final_precision_s': float(final_precision),
            'vs_planck_ratio': float(planck_analysis['ratio']),
            'total_observation_paths': int(recursion_results['observation_paths'][-1])
        }
    })
    print("   ✓ RecursiveObserverLattice working")

except Exception as e:
    results['components_tested'].append({
        'component': 'gas_molecule_lattice',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 5: Harmonic Extraction
print("\n[5/10] Testing: harmonic_extraction.py")
try:
    from harmonic_extraction import HarmonicExtractor

    extractor = HarmonicExtractor(fundamental_frequency=7.1e13)

    # Generate test signal with harmonics
    duration = 50 * extractor.fundamental_period
    n_samples = 2**12
    time_points = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    for n in [1, 10, 50]:
        signal += (1.0/n) * np.sin(2*np.pi * n * extractor.fundamental_freq * time_points)

    # Extract harmonics
    harmonics_data = extractor.extract_harmonics(signal, time_points, max_harmonic=100)

    # Find optimal harmonic
    optimal = extractor.find_optimal_harmonic(harmonics_data, coherence_time=741e-15)

    # Get precision cascade
    cascade = extractor.precision_cascade(max_harmonic=100, sub_harmonic_resolution=0.001)

    results['components_tested'].append({
        'component': 'harmonic_extraction',
        'status': 'success',
        'tests': {
            'fundamental_frequency_Hz': float(extractor.fundamental_freq),
            'total_harmonics_found': harmonics_data['total_harmonics_found'],
            'optimal_harmonic': int(optimal['number']),
            'sub_harmonic_precision_as': float(optimal['sub_harmonic_precision_as']),
            'total_enhancement': float(optimal['total_enhancement'])
        }
    })
    print("   ✓ HarmonicExtractor working")

except Exception as e:
    results['components_tested'].append({
        'component': 'harmonic_extraction',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 6: Harmonic Network Graph
print("\n[6/10] Testing: harmonic_network_graph.py")
try:
    from harmonic_network_graph import HarmonicNetworkGraph, HarmonicNode

    network = HarmonicNetworkGraph(frequency_tolerance=1e11)

    # Build small network for testing
    stats = network.build_from_recursive_observations(
        n_molecules=20,
        base_frequency=7.1e13,
        max_depth=2,
        harmonics_per_molecule=10
    )

    # Calculate network statistics
    network_stats = network.calculate_network_statistics()

    # Test precision enhancement from graph
    graph_enhancement = network.precision_enhancement_from_graph()

    results['components_tested'].append({
        'component': 'harmonic_network_graph',
        'status': 'success',
        'tests': {
            'total_nodes': network_stats['total_nodes'],
            'total_edges': network_stats['total_edges'],
            'avg_degree': float(network_stats['avg_degree']),
            'graph_density': float(network_stats['density']),
            'redundancy_factor': float(graph_enhancement['redundancy_factor']),
            'total_graph_enhancement': float(graph_enhancement['total_graph_enhancement'])
        }
    })
    print("   ✓ HarmonicNetworkGraph working")

except Exception as e:
    results['components_tested'].append({
        'component': 'harmonic_network_graph',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 7: Molecular Vibrations
print("\n[7/10] Testing: molecular_vibrations.py")
try:
    from molecular_vibrations import QuantumVibrationalAnalyzer

    analyzer = QuantumVibrationalAnalyzer(frequency=7.1e13, coherence_time=247e-15)

    # Test energy levels
    energies = analyzer.calculate_energy_levels(max_level=10)

    # Test Heisenberg linewidth
    linewidth = analyzer.heisenberg_linewidth()
    precision = analyzer.temporal_precision()

    # Test LED enhancement
    led_props = analyzer.led_enhanced_coherence(base_coherence=100e-15, led_enhancement=2.47)

    # Test thermal population
    populations = analyzer.thermal_population(temperature=300.0, max_level=10)

    results['components_tested'].append({
        'component': 'molecular_vibrations',
        'status': 'success',
        'tests': {
            'frequency_Hz': float(analyzer.frequency),
            'heisenberg_linewidth_Hz': float(linewidth),
            'temporal_precision_fs': float(precision * 1e15),
            'led_enhanced_precision_fs': float(led_props['enhanced_precision'] * 1e15),
            'ground_state_population': float(populations[0])
        }
    })
    print("   ✓ QuantumVibrationalAnalyzer working")

except Exception as e:
    results['components_tested'].append({
        'component': 'molecular_vibrations',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 8: Multidomain SEFT
print("\n[8/10] Testing: multidomain_seft.py")
try:
    from multidomain_seft import MiraculousMeasurementSystem

    system = MiraculousMeasurementSystem(baseline_precision=47e-21)

    # Test frequency estimation
    estimate = system.estimate_frequency(target=7.1e13, uncertainty=0.1)

    # Test miraculous path creation
    path = system.create_miraculous_path(estimate, 7.1e13, lambda_steps=50)

    # Test gap verification
    gap_analysis = system.verify_gap(estimate, 7.1e13)

    # Test navigation vs accuracy table
    table = system.navigation_vs_accuracy_table()

    results['components_tested'].append({
        'component': 'multidomain_seft',
        'status': 'success',
        'tests': {
            'baseline_precision_zs': float(system.baseline_precision * 1e21),
            'miraculous_path_length': len(path),
            'miraculous_states_count': sum(1 for s in path if s['miraculous']),
            'navigation_scenarios': len(table['scenarios'])
        }
    })
    print("   ✓ MiraculousMeasurementSystem working")

except Exception as e:
    results['components_tested'].append({
        'component': 'multidomain_seft',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 9: LED Excitation (if standalone functions exist)
print("\n[9/10] Testing: led_excitation.py")
try:
    from led_excitation import LEDSpectroscopySystem

    led_system = LEDSpectroscopySystem()

    # Test molecular fluorescence analysis
    analysis = led_system.analyze_molecular_fluorescence('c1ccccc1', 470)

    # Test fluorescence property prediction
    fluor_props = led_system.predict_fluorescence_properties('c1ccccc1')

    results['components_tested'].append({
        'component': 'led_excitation',
        'status': 'success',
        'tests': {
            'led_wavelengths': led_system.led_wavelengths,
            'excitation_efficiency': float(analysis['excitation_efficiency']),
            'detection_efficiency': float(analysis['detection_efficiency']),
            'fluorescence_intensity': float(analysis['fluorescence_intensity'])
        }
    })
    print("   ✓ LEDSpectroscopySystem working")

except Exception as e:
    results['components_tested'].append({
        'component': 'led_excitation',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 10: Hardware Clock Integration
print("\n[10/10] Testing: hardware_clock_integration.py")
try:
    from hardware_clock_integration import HardwareClockSync

    sync = HardwareClockSync()

    # Test with sample molecular patterns
    patterns = ['c1ccccc1', 'CCO', 'CC(=O)O']
    sync_results = sync.synchronize_molecular_hardware(patterns)

    results['components_tested'].append({
        'component': 'hardware_clock_integration',
        'status': 'success',
        'tests': {
            'cpu_frequency_GHz': float(sync.cpu_frequency / 1e9),
            'patterns_synchronized': len(sync_results),
            'avg_coordination_efficiency': float(np.mean([r['coordination_efficiency'] for r in sync_results.values()]))
        }
    })
    print("   ✓ HardwareClockSync working")

except Exception as e:
    results['components_tested'].append({
        'component': 'hardware_clock_integration',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Save results
results_file = os.path.join(results_dir, f'navigation_test_{timestamp}.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Generate summary figure
fig = plt.figure(figsize=(16, 10))

# Panel 1: Component status
ax1 = plt.subplot(2, 3, 1)
statuses = [c['status'] for c in results['components_tested']]
success_count = statuses.count('success')
failed_count = statuses.count('failed')

ax1.pie([success_count, failed_count], labels=['Success', 'Failed'],
        colors=['#4CAF50', '#F44336'], autopct='%1.0f%%', startangle=90)
ax1.set_title('Navigation Module Component Status', fontweight='bold')

# Panel 2: Component list
ax2 = plt.subplot(2, 3, 2)
ax2.axis('off')
components_text = "NAVIGATION COMPONENTS TESTED:\n\n"
for i, comp in enumerate(results['components_tested'], 1):
    status_icon = "✓" if comp['status'] == 'success' else "✗"
    components_text += f"{i}. {comp['component']}\n   {status_icon} {comp['status']}\n"

ax2.text(0.1, 0.9, components_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace')

# Panel 3-6: Key metrics from successful tests
successful_tests = [c for c in results['components_tested'] if c['status'] == 'success']

if successful_tests:
    # Show some key metrics
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    metrics_text = "KEY METRICS:\n\n"
    for comp in successful_tests[:5]:  # First 5 successful
        if 'tests' in comp:
            metrics_text += f"{comp['component']}:\n"
            for key, value in list(comp['tests'].items())[:2]:
                metrics_text += f"  {key}: {value}\n"
            metrics_text += "\n"

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace')

# Summary
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
summary_text = f"""
NAVIGATION MODULE TEST SUMMARY

Timestamp: {timestamp}

Components Tested: {len(results['components_tested'])}
Success: {success_count}
Failed: {failed_count}
Success Rate: {success_count/len(results['components_tested'])*100:.1f}%

Status: {'✓ PASSED' if failed_count == 0 else '⚠ PARTIAL'}
"""
ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Navigation Module Comprehensive Test', fontsize=16, fontweight='bold')
plt.tight_layout()

figure_file = os.path.join(results_dir, f'navigation_test_{timestamp}.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: {figure_file}")
plt.show()

# Print summary
print("\n" + "="*70)
print("   NAVIGATION MODULE TEST COMPLETE")
print("="*70)
print(f"\n   Success: {success_count}/{len(results['components_tested'])}")
print(f"   Results: {results_file}")
print(f"   Figure: {figure_file}")
print("\n")

if __name__ == "__main__":
    pass
