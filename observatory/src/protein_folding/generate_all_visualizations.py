"""
Master Visualization Script

Runs all three visualization scripts and generates a complete
visual analysis of the cycle-by-cycle phase-locked protein folding.
"""

import subprocess
import sys
from pathlib import Path

print("="*70)
print("GENERATING PROTEIN FOLDING VISUALIZATIONS")
print("="*70)
print()

# Get script directory
script_dir = Path(__file__).parent

# List of visualization scripts
scripts = [
    ("visualize_folding_dynamics.py", "Folding Dynamics Overview"),
    ("visualize_network_structure.py", "Network Structure & Dependencies"),
    ("visualize_phase_space.py", "Phase-Space Dynamics")
]

results = []

for script_name, description in scripts:
    print(f"Generating: {description}")
    print(f"  Script: {script_name}")

    script_path = script_dir / script_name

    try:
        # Run script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"  ✓ Success")
            results.append((script_name, True, result.stdout))
        else:
            print(f"  ✗ Failed")
            print(f"  Error: {result.stderr}")
            results.append((script_name, False, result.stderr))

    except Exception as e:
        print(f"  ✗ Error: {e}")
        results.append((script_name, False, str(e)))

    print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)

success_count = sum(1 for _, success, _ in results if success)
total_count = len(results)

print(f"Generated {success_count}/{total_count} visualizations successfully")
print()

for script_name, success, output in results:
    status = "✓" if success else "✗"
    print(f"{status} {script_name}")
    if success and output:
        # Extract saved file path from output
        for line in output.split('\n'):
            if 'Saved:' in line:
                print(f"    {line.strip()}")

print()
print("="*70)

# List all generated files
results_dir = script_dir / 'results'
if results_dir.exists():
    png_files = list(results_dir.glob('*.png'))
    if png_files:
        print(f"Generated {len(png_files)} visualization panels:")
        for png_file in sorted(png_files):
            print(f"  • {png_file.name}")
    print()
    print(f"All visualizations saved to: {results_dir}")
else:
    print("Results directory not found.")

print("="*70)
