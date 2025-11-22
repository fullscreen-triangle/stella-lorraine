#!/usr/bin/env python3
"""
Generate All Publication Figures
=================================

Master script to generate all visualization figures for the publication.

Runs all visualization scripts:
1. Categorical-Spacetime Mapping
2. Triangular Amplification
3. Phase-Lock Network Completion
4. Zero-Delay Positioning
"""

import subprocess
import sys
import os
from datetime import datetime

def run_visualization(script_name, description):
    """Run a visualization script"""
    print(f"\n{'='*70}")
    print(f" {description}")
    print(f"{'='*70}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)

    try:
        result = subprocess.run([sys.executable, script_path],
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Info: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main execution"""
    start_time = datetime.now()

    print("\n" + "="*70)
    print(" PUBLICATION FIGURES GENERATION")
    print(" Molecular Spectroscopy via Categorical State Propagation")
    print("="*70)
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Define visualizations
    visualizations = [
        ('categorical_mapping_results.py',
         'Figure 1: Categorical-Spacetime Mapping'),
        ('phase_lock_network_completion.py',
         'Figure 2: Phase-Lock Network Categorical Completion'),
        ('triangular_algorithm.py',
         'Figure 3: Triangular Amplification Algorithm'),
        ('no_delay_positioning.py',
         'Figure 4: Zero-Delay Positioning'),
    ]

    # Run all visualizations
    results = []
    for script_name, description in visualizations:
        success = run_visualization(script_name, description)
        results.append((description, success))

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print(" GENERATION SUMMARY")
    print("="*70)

    successful = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nCompleted: {successful}/{total} figures")
    print(f"Duration: {duration.total_seconds():.2f} seconds")
    print(f"\nDetailed Results:")

    for description, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {status}: {description}")

    print("\n" + "="*70)

    if successful == total:
        print("\n🎉 All figures generated successfully!")
        print("\nFigures saved to: observatory/results/figures/")
        return 0
    else:
        print(f"\n⚠️  {total - successful} figure(s) failed to generate.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
