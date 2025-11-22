#!/usr/bin/env python3
"""
Master Validation Runner for Virtual Systems

Runs all validation tests for:
1. Virtual light sources
2. Complete virtual interferometry
3. Cooling cascade thermometry

Generates comprehensive report
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json


def run_validation_script(script_name: str):
    """Run a validation script and capture results"""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}\n")

    script_path = Path(__file__).parent / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        if result.stderr:
            print("STDERR:", result.stderr)

        return True, result.stdout

    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False, e.stdout


def generate_master_report(results: dict):
    """Generate master validation report"""

    print("\n" + "="*70)
    print("MASTER VALIDATION REPORT")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Count successes
    n_total = len(results)
    n_passed = sum(1 for success, _ in results.values() if success)

    print(f"\nValidation Summary:")
    print(f"  Total tests: {n_total}")
    print(f"  Passed: {n_passed}")
    print(f"  Failed: {n_total - n_passed}")
    print(f"  Success rate: {n_passed/n_total*100:.0f}%")

    print(f"\nIndividual Results:")
    for test_name, (success, output) in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name:40s}: {status}")

    # Save master report
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / f"master_validation_report_{timestamp}.txt"

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VIRTUAL SYSTEMS VALIDATION - MASTER REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("VALIDATION SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total tests run: {n_total}\n")
        f.write(f"Tests passed: {n_passed}\n")
        f.write(f"Tests failed: {n_total - n_passed}\n")
        f.write(f"Success rate: {n_passed/n_total*100:.0f}%\n\n")

        f.write("INDIVIDUAL TEST RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for test_name, (success, output) in results.items():
            f.write(f"{'='*70}\n")
            f.write(f"TEST: {test_name}\n")
            f.write(f"STATUS: {'PASSED ✓' if success else 'FAILED ✗'}\n")
            f.write(f"{'='*70}\n\n")
            f.write(output)
            f.write("\n\n")

        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")

    print(f"\n✓ Master report saved: {report_path}")

    # Summary JSON
    summary = {
        'timestamp': timestamp,
        'total_tests': n_total,
        'passed': n_passed,
        'failed': n_total - n_passed,
        'success_rate': n_passed / n_total,
        'tests': {
            name: 'passed' if success else 'failed'
            for name, (success, _) in results.items()
        }
    }

    summary_path = output_dir / f"validation_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary JSON saved: {summary_path}")

    return n_passed == n_total


def main():
    """Run all validations"""

    print("="*70)
    print("VIRTUAL SYSTEMS - COMPLETE VALIDATION SUITE")
    print("="*70)
    print(f"\nStarting validation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # List of validation scripts
    validation_scripts = [
        "validate_virtual_light_source.py",
        "validate_complete_virtual_interferometry.py",
        "validate_cooling_cascade.py",
        "validate_triangular_cooling_amplification.py"
    ]

    results = {}

    # Run each validation
    for script in validation_scripts:
        success, output = run_validation_script(script)
        test_name = script.replace("validate_", "").replace(".py", "").replace("_", " ").title()
        results[test_name] = (success, output)

    # Generate master report
    all_passed = generate_master_report(results)

    print(f"\n{'='*70}")
    if all_passed:
        print("ALL VALIDATIONS PASSED ✓")
        print("Ready to proceed with paper writing!")
    else:
        print("SOME VALIDATIONS FAILED ✗")
        print("Please review the master report for details.")
    print(f"{'='*70}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
