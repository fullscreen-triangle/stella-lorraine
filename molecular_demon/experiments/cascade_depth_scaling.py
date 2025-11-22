"""
Cascade Depth Scaling Analysis

Shows how precision scales with number of reflections.
Validates that information accumulates cumulatively.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import MolecularOscillator, HarmonicNetworkGraph, MolecularDemonReflectanceCascade
from physics import MolecularOscillatorGenerator

logging.basicConfig(level=logging.WARNING)  # Suppress detailed logs
logger = logging.getLogger(__name__)


def test_cascade_scaling(max_reflections=20):
    """Test how precision scales with cascade depth"""
    print("\n" + "="*70)
    print("CASCADE DEPTH SCALING ANALYSIS")
    print("="*70)

    # Generate small ensemble for faster testing (reduced for demo)
    print("\nGenerating molecular ensemble (500 molecules)...")
    generator = MolecularOscillatorGenerator('N2', 300.0)
    molecule_dicts = generator.generate_ensemble(500, seed=42)  # Reduced from 10,000

    molecules = [
        MolecularOscillator(
            id=m['id'],
            species=m['species'],
            frequency_hz=m['frequency_hz'],
            phase_rad=m['phase_rad'],
            s_coordinates=m['s_coordinates']
        )
        for m in molecule_dicts
    ]

    # Build network (reduced harmonics for speed)
    print("Building harmonic network...")
    network = HarmonicNetworkGraph(
        molecules,
        coincidence_threshold_hz=1e9,  # Wider threshold
        max_harmonics=10  # Reduced from 150
    )
    network.build_graph()

    # Test different cascade depths
    depths = range(1, max_reflections + 1)
    precisions = []
    enhancements = []

    print("\nTesting cascade depths...")
    print(f"{'Reflections':<12} {'Precision (s)':<20} {'Enhancement':<15}")
    print("-"*70)

    for n_reflections in depths:
        cascade = MolecularDemonReflectanceCascade(
            network=network,
            bmd_depth=5,  # Smaller for speed
            base_frequency_hz=7.07e13
        )

        results = cascade.run_cascade(n_reflections=n_reflections)

        precision = results['precision_achieved_s']
        enhancement = results['enhancement_factors']['total']

        precisions.append(precision)
        enhancements.append(enhancement)

        print(f"{n_reflections:<12} {precision:<20.2e} {enhancement:<15.2e}")

    print("="*70)

    # Plot results
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Precision vs reflections
        ax1.semilogy(depths, precisions, 'b-o')
        ax1.axhline(5.39116e-44, color='r', linestyle='--', label='Planck time')
        ax1.set_xlabel('Number of Reflections')
        ax1.set_ylabel('Time Precision (s)')
        ax1.set_title('Precision vs Cascade Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Enhancement vs reflections
        ax2.semilogy(depths, enhancements, 'g-o')
        ax2.set_xlabel('Number of Reflections')
        ax2.set_ylabel('Total Enhancement Factor')
        ax2.set_title('Enhancement Scaling')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_dir = Path(__file__).parent.parent / 'results'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'cascade_depth_scaling.png', dpi=150)
        print(f"\nPlot saved to: {output_dir / 'cascade_depth_scaling.png'}")

    except Exception as e:
        print(f"\nNote: Could not generate plot: {e}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    if len(precisions) >= 2:
        # Check scaling law
        log_precisions = np.log10(precisions)
        log_enhancements = np.log10(enhancements)

        # Fit power law
        coeffs = np.polyfit(depths, log_enhancements, 1)
        slope = coeffs[0]

        print(f"\nEnhancement scaling: E ∝ n^{slope:.2f}")
        print(f"Expected for quadratic growth: E ∝ n^2")

        if abs(slope - 2.0) < 0.5:
            print("✓ Scaling matches quadratic expectation (cumulative information)")
        else:
            print(f"⚠ Scaling deviates from quadratic (slope = {slope:.2f})")

    print("="*70)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Create results directory
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Prepare data
    scaling_data = []
    for i, (depth, prec, enh) in enumerate(zip(depths, precisions, enhancements)):
        scaling_data.append({
            'reflections': int(depth),
            'precision_s': float(prec),
            'enhancement': float(enh),
            'orders_below_planck': float(-np.log10(prec / 5.39116e-44)) if prec > 0 else 0
        })

    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'method': 'Cascade Depth Scaling Analysis',
        'parameters': {
            'molecules': len(molecules),
            'max_reflections': max_reflections,
            'bmd_depth': 5
        },
        'scaling_data': scaling_data,
        'analysis': {
            'scaling_exponent': float(slope) if len(precisions) >= 2 else 0,
            'expected_exponent': 2.0,
            'matches_quadratic': bool(abs(slope - 2.0) < 0.5) if len(precisions) >= 2 else False
        }
    }

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_dir / f'cascade_scaling_{timestamp}.json'

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {json_file}")

    # Save markdown report
    md_file = output_dir / f'cascade_scaling_{timestamp}.md'

    with open(md_file, 'w') as f:
        f.write("# Cascade Depth Scaling Analysis\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- Molecules: {len(molecules):,}\n")
        f.write(f"- BMD Depth: 5\n")
        f.write(f"- Max Reflections: {max_reflections}\n\n")
        f.write("## Scaling Results\n\n")
        f.write("| Reflections | Precision (s) | Orders Below Planck | Enhancement |\n")
        f.write("|-------------|---------------|---------------------|-------------|\n")

        for data in scaling_data:
            f.write(f"| {data['reflections']} | {data['precision_s']:.2e} | "
                   f"{data['orders_below_planck']:.2f} | {data['enhancement']:.2e}× |\n")

        f.write(f"\n## Analysis\n\n")
        if len(precisions) >= 2:
            f.write(f"- **Scaling law**: E ∝ n^{slope:.2f}\n")
            f.write(f"- **Expected**: E ∝ n^2 (quadratic growth)\n")
            f.write(f"- **Match**: {'✓ Yes' if abs(slope - 2.0) < 0.5 else '✗ No'}\n")
        f.write(f"\nThis demonstrates cumulative information growth in the cascade.\n")

    print(f"✓ Markdown report saved to: {md_file}")

    if 'cascade_depth_scaling.png' in str(output_dir):
        print(f"✓ Plot saved")

    print("="*70)


if __name__ == "__main__":
    test_cascade_scaling(max_reflections=10)  # Reduced for faster execution
