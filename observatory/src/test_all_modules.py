#!/usr/bin/env python3
"""
Master Module Test Runner
==========================
Runs all module test scripts independently.

Simply executes each module's test script:
- navigation/navigation_system.py
- simulation/simulation_dynamics.py
- oscillatory/oscillatory_system.py
- signal/signal_system.py
- recursion/recursive_precision.py

No orchestration - just runs each independently.
"""

import subprocess
import sys
import os
from datetime import datetime

print("╔" + "═" * 68 + "╗")
print("║" + " " * 68 + "║")
print("║" + "         MODULE-BY-MODULE COMPREHENSIVE TESTING".center(68) + "║")
print("║" + " " * 68 + "║")
print("╚" + "═" * 68 + "╝")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n🔬 Test Run: {timestamp}")
print("\nTesting each module independently...\n")

# List of module test scripts
test_scripts = [
    ('navigation', 'navigation/navigation_system.py'),
    ('simulation', 'simulation/simulation_dynamics.py'),
    ('oscillatory', 'oscillatory/oscillatory_system.py'),
    ('signal', 'signal/signal_system.py'),
    ('recursion', 'recursion/recursive_precision.py')
]

results = []

for i, (module_name, script_path) in enumerate(test_scripts, 1):
    print(f"\n{'='*70}")
    print(f"   [{i}/{len(test_scripts)}] TESTING MODULE: {module_name.upper()}")
    print(f"   Script: {script_path}")
    print(f"{'='*70}\n")

    full_path = os.path.join(os.path.dirname(__file__), script_path)

    if not os.path.exists(full_path):
        print(f"   ✗ Script not found: {full_path}")
        results.append((module_name, 'not_found'))
        continue

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, full_path],
            cwd=os.path.dirname(__file__),
            capture_output=False,  # Show output in real-time
            timeout=300
        )

        if result.returncode == 0:
            print(f"\n   ✓ {module_name.upper()} module test COMPLETED")
            results.append((module_name, 'success'))
        else:
            print(f"\n   ✗ {module_name.upper()} module test FAILED (code {result.returncode})")
            results.append((module_name, 'failed'))

    except subprocess.TimeoutExpired:
        print(f"\n   ✗ {module_name.upper()} module test TIMEOUT")
        results.append((module_name, 'timeout'))
    except Exception as e:
        print(f"\n   ✗ {module_name.upper()} module test ERROR: {e}")
        results.append((module_name, 'error'))

# Summary
print("\n\n" + "="*70)
print("   FINAL SUMMARY")
print("="*70)

success = sum(1 for _, status in results if status == 'success')
failed = sum(1 for _, status in results if status != 'success')

print(f"\n   Total modules tested: {len(results)}")
print(f"   Successful: {success}")
print(f"   Failed/Errors: {failed}")
print(f"\n   Results by module:")

for module_name, status in results:
    icon = "✓" if status == 'success' else "✗"
    print(f"      {icon} {module_name.ljust(15)} - {status}")

print(f"\n   Overall: {'✓ ALL PASSED' if failed == 0 else f'⚠ {failed} FAILED'}")

print(f"\n   Check results in: results/")
print(f"      - navigation_module/")
print(f"      - simulation_module/")
print(f"      - oscillatory_module/")
print(f"      - signal_module/")
print(f"      - recursion_module/")

print("\n")

if __name__ == "__main__":
    sys.exit(0 if failed == 0 else 1)
