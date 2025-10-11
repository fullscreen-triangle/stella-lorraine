#!/usr/bin/env python3
"""
Test Single Experiment
======================
Quick test to verify a single experiment runs correctly.
"""

import sys
import os

# Test the recursive observer nesting
print("Testing: Recursive Observer Nesting")
print("=" * 70)

try:
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import and run
    from navigation.gas_molecule_lattice import main

    print("\nRunning experiment...")
    results, figure = main()

    print("\n✓ SUCCESS!")
    print(f"Results type: {type(results)}")
    print(f"Figure path: {figure}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
