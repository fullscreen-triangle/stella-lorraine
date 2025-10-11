#!/usr/bin/env python3
"""
Precision Cascade Runner
=========================
Runs all precision observer levels independently.

Each precision level is a finite observer that works independently:
- Nanosecond (1e-9 s) - Hardware clocks
- Picosecond (1e-12 s) - N2 molecules + spectroscopy
- Femtosecond (1e-13 s) - Fundamental harmonic
- Attosecond (9.4e-17 s) - Standard FFT
- Zeptosecond (4.7e-20 s) - Multi-Domain SEFT
- Planck (~5e-44 s) - Recursive observer nesting
- Trans-Planckian (< 1e-44 s) - Harmonic network graph

No dependencies - each observer functions independently.
"""

import subprocess
import sys
import os
from datetime import datetime
import json

print("╔" + "═" * 68 + "╗")
print("║" + " " * 68 + "║")
print("║" + "     PRECISION CASCADE - OBSERVER BY OBSERVER".center(68) + "║")
print("║" + " " * 68 + "║")
print("╚" + "═" * 68 + "╝")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n🔬 Cascade Run: {timestamp}")
print("\n⚛️  Each precision level is a finite observer")
print("   They work independently - no cascading failures\n")

# Precision observers (in order)
observers = [
    ('nanosecond', '1e-9 s', 'Hardware Clock Aggregation'),
    ('picosecond', '1e-12 s', 'N₂ Molecules + Virtual Spectroscopy'),
    ('femtosecond', '1e-13 s', 'Fundamental Gas Harmonic'),
    ('attosecond', '9.4e-17 s', 'Standard FFT on Harmonics'),
    ('zeptosecond', '4.7e-20 s', 'Multi-Domain SEFT'),
    ('planck_time', '~5e-44 s', 'Recursive Observer Nesting'),
    ('trans_planckian', '< 1e-44 s', 'Harmonic Network Graph')
]

results = []
cascade_data = []

for i, (observer_name, target, method) in enumerate(observers, 1):
    print(f"\n{'='*70}")
    print(f"   [{i}/{len(observers)}] PRECISION OBSERVER: {observer_name.upper().replace('_', ' ')}")
    print(f"   Target: {target}")
    print(f"   Method: {method}")
    print(f"{'='*70}\n")

    script_path = os.path.join(os.path.dirname(__file__), f'{observer_name}.py')

    if not os.path.exists(script_path):
        print(f"   ✗ Script not found: {script_path}")
        results.append((observer_name, 'not_found'))
        continue

    try:
        # Run observer
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            capture_output=False,  # Show output in real-time
            timeout=120
        )

        if result.returncode == 0:
            print(f"\n   ✓ {observer_name.upper()} observer COMPLETED")
            results.append((observer_name, 'success'))

            # Try to load result JSON
            try:
                results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'precision_cascade')
                # Find latest result file
                result_files = [f for f in os.listdir(results_dir) if f.startswith(observer_name) and f.endswith('.json')]
                if result_files:
                    latest_file = sorted(result_files)[-1]
                    with open(os.path.join(results_dir, latest_file), 'r') as f:
                        observer_data = json.load(f)
                        cascade_data.append({
                            'observer': observer_name,
                            'precision_achieved': observer_data.get('precision_achieved_s', None),
                            'status': observer_data.get('status', 'unknown')
                        })
            except Exception as e:
                print(f"   (Could not load result data: {e})")
        else:
            print(f"\n   ✗ {observer_name.upper()} observer FAILED (code {result.returncode})")
            results.append((observer_name, 'failed'))

    except subprocess.TimeoutExpired:
        print(f"\n   ✗ {observer_name.upper()} observer TIMEOUT")
        results.append((observer_name, 'timeout'))
    except Exception as e:
        print(f"\n   ✗ {observer_name.upper()} observer ERROR: {e}")
        results.append((observer_name, 'error'))

# Final summary
print("\n\n" + "="*70)
print("   PRECISION CASCADE COMPLETE")
print("="*70)

success = sum(1 for _, status in results if status == 'success')
failed = sum(1 for _, status in results if status != 'success')

print(f"\n   Total observers: {len(results)}")
print(f"   Successful: {success}")
print(f"   Failed/Errors: {failed}")

print(f"\n   Results by observer:")
for (observer_name, target, method), (_, status) in zip(observers, results):
    icon = "✓" if status == 'success' else "✗"
    print(f"      {icon} {observer_name.ljust(20)} {target.ljust(12)} - {status}")

if cascade_data:
    print(f"\n   Precision Achieved:")
    for data in cascade_data:
        if data['precision_achieved']:
            print(f"      {data['observer'].ljust(20)} {data['precision_achieved']:.2e} s")

print(f"\n   Overall: {'✓ ALL OBSERVERS PASSED' if failed == 0 else f'⚠ {failed} FAILED'}")
print(f"\n   Results saved in: results/precision_cascade/")
print(f"      Each observer has its own JSON + PNG files")

# Save cascade summary
if cascade_data:
    summary_file = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'precision_cascade',
                                f'cascade_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'observers': cascade_data,
            'success_count': success,
            'total_count': len(results)
        }, f, indent=2)
    print(f"      Cascade summary: cascade_summary_{timestamp}.json")

print("\n🎯 Each observer functions independently (finite observer principle)")
print("   If one fails, others continue - no cascading failures!\n")

if __name__ == "__main__":
    sys.exit(0 if failed == 0 else 1)
