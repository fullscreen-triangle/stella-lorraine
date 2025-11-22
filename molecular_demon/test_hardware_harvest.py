"""
Quick test of hardware harvesting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing hardware harvesting...")

try:
    print("\n1. Importing harvester...")
    from physics.hardware_harvesting import HardwareFrequencyHarvester
    print("✓ Import successful")

    print("\n2. Creating harvester...")
    harvester = HardwareFrequencyHarvester()
    print("✓ Harvester created")

    print("\n3. Harvesting base frequencies...")
    hardware_oscillators = harvester.harvest_all()
    print(f"✓ Harvested {len(hardware_oscillators)} oscillators")

    print("\n4. Generating harmonics...")
    all_oscillators = harvester.generate_harmonics(hardware_oscillators, max_harmonic=10)
    print(f"✓ Generated {len(all_oscillators)} total (with harmonics)")

    print("\n5. Converting to molecular oscillators...")
    molecular_oscillators = harvester.to_molecular_oscillators(all_oscillators)
    print(f"✓ Converted {len(molecular_oscillators)} oscillators")

    print("\n6. Checking first oscillator...")
    first = molecular_oscillators[0]
    print(f"  ID: {first.id}")
    print(f"  Species: {first.species}")
    print(f"  Frequency: {first.frequency_hz:.2e} Hz")
    print(f"  S-coords: {first.s_coordinates}")
    print("✓ All checks passed!")

    print("\n" + "="*70)
    print("HARDWARE HARVESTING TEST: SUCCESS")
    print("="*70)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
