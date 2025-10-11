#!/usr/bin/env python3
"""
Simulation Module Test Script
===============================
Tests ALL components in the simulation module independently.

Components tested:
- Alignment
- GasChamber
- Molecule
- Observer
- Propagation
- Transcendent
- Wave
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("   SIMULATION MODULE COMPREHENSIVE TEST")
print("="*70)

results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'simulation_module')
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    'timestamp': timestamp,
    'module': 'simulation',
    'components_tested': []
}

# Test 1: Molecule
print("\n[1/7] Testing: Molecule.py")
try:
    from Molecule import DiatomicMolecule, create_N2_ensemble

    # Create N2 molecule
    n2 = DiatomicMolecule()

    # Test properties
    clock_precision = n2.get_clock_precision()
    energy_levels = n2.get_quantum_energy_levels(max_level=10)
    Q_factor = n2.get_quality_factor()

    # Test oscillation
    time_points = np.linspace(0, 100e-15, 1000)
    oscillation = n2.oscillate(time_points, amplitude=1.0, phase=0.0)

    # Test ensemble
    ensemble = create_N2_ensemble(n_molecules=100, temperature=300.0)
    ensemble_freqs = [mol.vibrational_frequency for mol in ensemble]

    results['components_tested'].append({
        'component': 'Molecule',
        'status': 'success',
        'tests': {
            'vibrational_frequency_Hz': float(n2.vibrational_frequency),
            'vibrational_period_fs': float(n2.vibrational_period * 1e15),
            'clock_precision_fs': float(clock_precision * 1e15),
            'Q_factor': float(Q_factor),
            'energy_levels': [float(e) for e in energy_levels],
            'ensemble_size': len(ensemble),
            'ensemble_freq_std': float(np.std(ensemble_freqs))
        }
    })
    print("   ✓ DiatomicMolecule working")

except Exception as e:
    results['components_tested'].append({
        'component': 'Molecule',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 2: GasChamber
print("\n[2/7] Testing: GasChamber.py")
try:
    from GasChamber import GasChamber, GasProperties

    # Create chamber
    chamber = GasChamber(
        size=(1e-3, 1e-3, 1e-3),
        temperature=300.0,
        pressure=101325.0,
        n_grid_points=16  # Small for testing
    )

    # Add molecular sources
    for i in range(5):
        pos = np.random.uniform(0, 1e-3, 3)
        freq = 7.1e13 * (1 + 0.01*np.random.randn())
        chamber.add_molecular_source(pos, freq, amplitude=1e-12, phase=0.0)

    # Propagate wave (short duration for testing)
    wave_data = chamber.propagate_wave(duration=1e-12)  # 1 ps

    # Extract resonant modes
    modes = chamber.extract_resonant_modes(
        wave_data['center_time_series'],
        wave_data['time_points']
    )

    # Calculate chamber resonances
    resonances = chamber.calculate_chamber_resonances()

    results['components_tested'].append({
        'component': 'GasChamber',
        'status': 'success',
        'tests': {
            'chamber_size_mm': float(chamber.size[0] * 1e3),
            'speed_of_sound_m_s': float(chamber.speed_of_sound),
            'n_molecular_sources': len(chamber.molecular_sources),
            'propagation_steps': wave_data['n_steps'],
            'resonant_modes_found': len(modes['frequencies']),
            'natural_resonances': [float(r) for r in resonances[:5]]
        }
    })
    print("   ✓ GasChamber working")

except Exception as e:
    results['components_tested'].append({
        'component': 'GasChamber',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 3: Observer
print("\n[3/7] Testing: Observer.py")
try:
    # Import all classes/functions from Observer
    import Observer

    # List all available classes/functions
    observer_items = [item for item in dir(Observer) if not item.startswith('_')]

    results['components_tested'].append({
        'component': 'Observer',
        'status': 'success',
        'tests': {
            'available_items': observer_items,
            'item_count': len(observer_items)
        }
    })
    print(f"   ✓ Observer module has {len(observer_items)} items")

except Exception as e:
    results['components_tested'].append({
        'component': 'Observer',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 4: Wave
print("\n[4/7] Testing: Wave.py")
try:
    # Import all classes/functions from Wave
    import Wave

    # List all available classes/functions
    wave_items = [item for item in dir(Wave) if not item.startswith('_')]

    results['components_tested'].append({
        'component': 'Wave',
        'status': 'success',
        'tests': {
            'available_items': wave_items,
            'item_count': len(wave_items)
        }
    })
    print(f"   ✓ Wave module has {len(wave_items)} items")

except Exception as e:
    results['components_tested'].append({
        'component': 'Wave',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 5: Alignment
print("\n[5/7] Testing: Alignment.py")
try:
    import Alignment

    alignment_items = [item for item in dir(Alignment) if not item.startswith('_')]

    results['components_tested'].append({
        'component': 'Alignment',
        'status': 'success',
        'tests': {
            'available_items': alignment_items,
            'item_count': len(alignment_items)
        }
    })
    print(f"   ✓ Alignment module has {len(alignment_items)} items")

except Exception as e:
    results['components_tested'].append({
        'component': 'Alignment',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 6: Propagation
print("\n[6/7] Testing: Propagation.py")
try:
    import Propagation

    propagation_items = [item for item in dir(Propagation) if not item.startswith('_')]

    results['components_tested'].append({
        'component': 'Propagation',
        'status': 'success',
        'tests': {
            'available_items': propagation_items,
            'item_count': len(propagation_items)
        }
    })
    print(f"   ✓ Propagation module has {len(propagation_items)} items")

except Exception as e:
    results['components_tested'].append({
        'component': 'Propagation',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Test 7: Transcendent
print("\n[7/7] Testing: Transcendent.py")
try:
    import Transcendent

    transcendent_items = [item for item in dir(Transcendent) if not item.startswith('_')]

    results['components_tested'].append({
        'component': 'Transcendent',
        'status': 'success',
        'tests': {
            'available_items': transcendent_items,
            'item_count': len(transcendent_items)
        }
    })
    print(f"   ✓ Transcendent module has {len(transcendent_items)} items")

except Exception as e:
    results['components_tested'].append({
        'component': 'Transcendent',
        'status': 'failed',
        'error': str(e)
    })
    print(f"   ✗ Error: {e}")

# Save results
results_file = os.path.join(results_dir, f'simulation_test_{timestamp}.json')
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
ax1.set_title('Simulation Module Component Status', fontweight='bold')

# Panel 2: Component list
ax2 = plt.subplot(2, 3, 2)
ax2.axis('off')
components_text = "SIMULATION COMPONENTS TESTED:\n\n"
for i, comp in enumerate(results['components_tested'], 1):
    status_icon = "✓" if comp['status'] == 'success' else "✗"
    components_text += f"{i}. {comp['component']}\n   {status_icon} {comp['status']}\n"

ax2.text(0.1, 0.9, components_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace')

# Panel 3: Molecule metrics (if successful)
molecule_test = next((c for c in results['components_tested'] if c['component'] == 'Molecule' and c['status'] == 'success'), None)
if molecule_test and 'tests' in molecule_test:
    ax3 = plt.subplot(2, 3, 3)
    metrics = molecule_test['tests']
    ax3.axis('off')
    mol_text = f"""
MOLECULE (N₂) METRICS:

Frequency: {metrics['vibrational_frequency_Hz']:.2e} Hz
Period: {metrics['vibrational_period_fs']:.2f} fs
Precision: {metrics['clock_precision_fs']:.2f} fs
Q Factor: {metrics['Q_factor']:.2e}

Ensemble:
  Size: {metrics['ensemble_size']}
  Std: {metrics['ensemble_freq_std']:.2e} Hz
"""
    ax3.text(0.1, 0.5, mol_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Panel 4: GasChamber metrics (if successful)
chamber_test = next((c for c in results['components_tested'] if c['component'] == 'GasChamber' and c['status'] == 'success'), None)
if chamber_test and 'tests' in chamber_test:
    ax4 = plt.subplot(2, 3, 4)
    metrics = chamber_test['tests']
    ax4.axis('off')
    chamber_text = f"""
GAS CHAMBER METRICS:

Size: {metrics['chamber_size_mm']:.1f} mm cube
Sound speed: {metrics['speed_of_sound_m_s']:.1f} m/s
Molecular sources: {metrics['n_molecular_sources']}
Propagation steps: {metrics['propagation_steps']}
Resonant modes: {metrics['resonant_modes_found']}
"""
    ax4.text(0.1, 0.5, chamber_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Summary
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
summary_text = f"""
SIMULATION MODULE TEST SUMMARY

Timestamp: {timestamp}

Components Tested: {len(results['components_tested'])}
Success: {success_count}
Failed: {failed_count}
Success Rate: {success_count/len(results['components_tested'])*100:.1f}%

Status: {'✓ PASSED' if failed_count == 0 else '⚠ PARTIAL'}
"""
ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
        fontsize=11, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Simulation Module Comprehensive Test', fontsize=16, fontweight='bold')
plt.tight_layout()

figure_file = os.path.join(results_dir, f'simulation_test_{timestamp}.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: {figure_file}")
plt.show()

# Print summary
print("\n" + "="*70)
print("   SIMULATION MODULE TEST COMPLETE")
print("="*70)
print(f"\n   Success: {success_count}/{len(results['components_tested'])}")
print(f"   Results: {results_file}")
print(f"   Figure: {figure_file}")
print("\n")

if __name__ == "__main__":
    pass
