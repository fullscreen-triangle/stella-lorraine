#!/usr/bin/env python3
"""
Experiment Diagnostics
======================
Quickly check which experiments can run successfully.
"""

import os
import sys

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

experiments = [
    ('simulation/Molecule.py', 'Molecular Clock'),
    ('simulation/GasChamber.py', 'Gas Chamber'),
    ('navigation/harmonic_extraction.py', 'Harmonic Extraction'),
    ('navigation/molecular_vibrations.py', 'Quantum Vibrations'),
    ('navigation/fourier_transform_coordinates.py', 'Multi-Domain SEFT'),
    ('navigation/entropy_navigation.py', 'S-Entropy Navigation'),
    ('navigation/multidomain_seft.py', 'Miraculous Measurement'),
    ('navigation/finite_observer_verification.py', 'Finite Observer'),
    ('navigation/gas_molecule_lattice.py', 'Recursive Observer Nesting'),
    ('navigation/harmonic_network_graph.py', 'Harmonic Network Graph'),
]

print("="*70)
print("   EXPERIMENT DIAGNOSTICS")
print("="*70)
print("\nChecking which experiments can be imported...\n")

results = []

for script_path, name in experiments:
    module_path = script_path.replace('/', '.').replace('.py', '')

    try:
        # Try to import the module
        exec(f"import {module_path}")

        # Check if main() exists
        module = eval(module_path)
        has_main = hasattr(module, 'main')

        if has_main:
            status = "✓ READY"
            color = '\033[92m'  # Green
        else:
            status = "⚠ NO main()"
            color = '\033[93m'  # Yellow

        print(f"{color}{status}\033[0m  {name}")
        print(f"       Path: {script_path}")
        print(f"       Has main(): {has_main}")

        results.append((name, 'ready' if has_main else 'no_main', None))

    except Exception as e:
        status = "✗ FAILED"
        color = '\033[91m'  # Red
        print(f"{color}{status}\033[0m  {name}")
        print(f"       Path: {script_path}")
        print(f"       Error: {str(e)[:100]}")

        results.append((name, 'failed', str(e)))

    print()

# Summary
print("="*70)
print("   SUMMARY")
print("="*70)

ready = sum(1 for _, status, _ in results if status == 'ready')
no_main = sum(1 for _, status, _ in results if status == 'no_main')
failed = sum(1 for _, status, _ in results if status == 'failed')

print(f"\nTotal experiments: {len(results)}")
print(f"✓ Ready to run: {ready}")
print(f"⚠ Need main(): {no_main}")
print(f"✗ Failed import: {failed}")

if failed > 0:
    print(f"\n❌ Fix these {failed} failed experiments first:")
    for name, status, error in results:
        if status == 'failed':
            print(f"   • {name}")
            print(f"     Error: {error[:100]}")

if no_main > 0:
    print(f"\n⚠️  Add main() function to these {no_main} experiments:")
    for name, status, _ in results:
        if status == 'no_main':
            print(f"   • {name}")

if ready == len(results):
    print(f"\n🎉 All experiments ready! Run validation suite:")
    print(f"   python run_validation_suite.py")

print()
