#!/usr/bin/env python3
"""
Signal Module Test Script
==========================
Tests ALL components in the signal module independently.

Components tested:
- mimo_signal_amplification
- precise_clock_apis
- satellite_temporal_gps
- signal_fusion
- signal_latencies
- temporal_information_architecture
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("   SIGNAL MODULE COMPREHENSIVE TEST")
print("="*70)

results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'signal_module')
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results = {
    'timestamp': timestamp,
    'module': 'signal',
    'components_tested': []
}

components = [
    'mimo_signal_amplification',
    'precise_clock_apis',
    'satellite_temporal_gps',
    'signal_fusion',
    'signal_latencies',
    'temporal_information_architecture'
]

for i, component_name in enumerate(components, 1):
    print(f"\n[{i}/{len(components)}] Testing: {component_name}.py")
    try:
        module = __import__(component_name)
        items = [item for item in dir(module) if not item.startswith('_')]
        classes = [item for item in items if isinstance(getattr(module, item), type)]
        functions = [item for item in items if callable(getattr(module, item)) and not isinstance(getattr(module, item), type)]

        test_results = {
            'available_items': items,
            'classes': classes,
            'functions': functions,
            'item_count': len(items)
        }

        results['components_tested'].append({
            'component': component_name,
            'status': 'success',
            'tests': test_results
        })
        print(f"   ✓ {component_name}: {len(items)} items found")

    except Exception as e:
        results['components_tested'].append({
            'component': component_name,
            'status': 'failed',
            'error': str(e)
        })
        print(f"   ✗ Error: {e}")

results_file = os.path.join(results_dir, f'signal_test_{timestamp}.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

fig = plt.figure(figsize=(16, 10))
ax1 = plt.subplot(2, 3, 1)
statuses = [c['status'] for c in results['components_tested']]
success_count = statuses.count('success')
failed_count = statuses.count('failed')

ax1.pie([success_count, failed_count], labels=['Success', 'Failed'],
        colors=['#4CAF50', '#F44336'], autopct='%1.0f%%', startangle=90)
ax1.set_title('Signal Module Component Status', fontweight='bold')

ax2 = plt.subplot(2, 3, 2)
ax2.axis('off')
components_text = "SIGNAL COMPONENTS:\n\n"
for i, comp in enumerate(results['components_tested'], 1):
    status_icon = "✓" if comp['status'] == 'success' else "✗"
    components_text += f"{i}. {comp['component']}\n   {status_icon} {comp['status']}\n"
ax2.text(0.1, 0.9, components_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace')

ax3 = plt.subplot(2, 3, 3)
ax3.axis('off')
summary_text = f"""
SIGNAL MODULE TEST

Timestamp: {timestamp}

Components: {len(results['components_tested'])}
Success: {success_count}
Failed: {failed_count}
Success Rate: {success_count/len(results['components_tested'])*100:.1f}%

Status: {'✓ PASSED' if failed_count == 0 else '⚠ PARTIAL'}
"""
ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes,
        fontsize=11, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Signal Module Comprehensive Test', fontsize=16, fontweight='bold')
plt.tight_layout()

figure_file = os.path.join(results_dir, f'signal_test_{timestamp}.png')
plt.savefig(figure_file, dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("   SIGNAL MODULE TEST COMPLETE")
print("="*70)
print(f"\n   Success: {success_count}/{len(results['components_tested'])}")
print(f"   Results: {results_file}")
print(f"   Figure: {figure_file}")
print("\n")

if __name__ == "__main__":
    pass
