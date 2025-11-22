#!/usr/bin/env python3
"""
Quick Test Script - Verify All Modules Work and Save Results
=============================================================
Tests each module briefly and confirms results are saved.
"""

import os
import sys
from datetime import datetime

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module(name, test_func):
    """Test a single module"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        test_func()
        print(f"✓ {name} - PASSED")
        return True
    except Exception as e:
        print(f"✗ {name} - FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_bmd_equivalence():
    """Test BMD equivalence module"""
    from bmd_equivalence import BMDEquivalenceValidator
    import numpy as np

    validator = BMDEquivalenceValidator()
    signal, time_points = validator.generate_test_signal(n_samples=512, frequency=7.1e13)
    results = validator.validate_bmd_equivalence(signal, n_iterations=10)

    # Check that results were saved
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'bmd_equivalence')
    assert os.path.exists(results_dir), "Results directory not created"

    # Find JSON files
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    assert len(json_files) > 0, "No JSON results saved"

    print(f"   Results saved in: {results_dir}")
    print(f"   JSON files: {len(json_files)}")


def test_multidomain_seft():
    """Test multidomain SEFT module"""
    from multidomain_seft import MiraculousMeasurementSystem

    system = MiraculousMeasurementSystem(baseline_precision=47e-21)
    result = system.miraculous_frequency_measurement(7.1e13, initial_uncertainty=0.05)

    # Check that results were saved
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'multidomain_seft')
    assert os.path.exists(results_dir), "Results directory not created"

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    assert len(json_files) > 0, "No JSON results saved"

    print(f"   Results saved in: {results_dir}")
    print(f"   JSON files: {len(json_files)}")


def test_molecular_vibrations():
    """Test molecular vibrations module"""
    from molecular_vibrations import QuantumVibrationalAnalyzer

    analyzer = QuantumVibrationalAnalyzer(frequency=7.1e13, coherence_time=247e-15)
    energies = analyzer.calculate_energy_levels(max_level=5)
    linewidth = analyzer.heisenberg_linewidth()

    # Check that results were saved
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'molecular_vibrations')
    assert os.path.exists(results_dir), "Results directory not created"

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    assert len(json_files) > 0, "No JSON results saved"

    print(f"   Results saved in: {results_dir}")
    print(f"   JSON files: {len(json_files)}")


def test_led_excitation():
    """Test LED excitation module"""
    from led_excitation import LEDSpectroscopySystem

    led_system = LEDSpectroscopySystem()
    analysis = led_system.analyze_molecular_fluorescence('c1ccccc1', 470)

    # Check analysis results
    assert 'fluorescence_intensity' in analysis
    assert 'excitation_efficiency' in analysis

    print(f"   Fluorescence intensity: {analysis['fluorescence_intensity']:.3f}")
    print(f"   Excitation efficiency: {analysis['excitation_efficiency']:.3f}")


def main():
    """Run all quick tests"""
    print("\n" + "="*60)
    print("   QUICK TEST SUITE - Navigation Modules")
    print("   Testing result saving functionality")
    print("="*60)

    tests = [
        ('BMD Equivalence', test_bmd_equivalence),
        ('Multidomain SEFT', test_multidomain_seft),
        ('Molecular Vibrations', test_molecular_vibrations),
        ('LED Excitation', test_led_excitation)
    ]

    results = {}
    for name, func in tests:
        results[name] = test_module(name, func)

    # Summary
    print(f"\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"   {icon} {name}: {'PASSED' if status else 'FAILED'}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print(f"\n   🎉 ALL TESTS PASSED - Results are being saved correctly!")
    else:
        print(f"\n   ⚠ Some tests failed - check errors above")

    print("="*60 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
