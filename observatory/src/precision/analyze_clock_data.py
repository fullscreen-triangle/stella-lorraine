#!/usr/bin/env python3
"""
Live Clock Data Analyzer
=========================
Analyzes data from live clock runs and creates comprehensive visualizations.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import glob

def load_latest_clock_data():
    """Load the most recent clock run data"""
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'live_clock')

    if not os.path.exists(results_dir):
        print(f"❌ No clock data found in {results_dir}")
        return None, None

    # Find latest files
    npz_files = glob.glob(os.path.join(results_dir, 'clock_run_data_*.npz'))
    json_files = glob.glob(os.path.join(results_dir, 'clock_run_data_*.json'))
    metadata_files = glob.glob(os.path.join(results_dir, 'clock_run_metadata_*.json'))

    if not npz_files and not json_files:
        print(f"❌ No clock data files found")
        return None, None

    # Load latest data
    data_file = sorted(npz_files + json_files)[-1]
    metadata_file = sorted(metadata_files)[-1]

    print(f"📊 Loading: {os.path.basename(data_file)}")
    print(f"📊 Metadata: {os.path.basename(metadata_file)}")

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load data
    if data_file.endswith('.npz'):
        data = dict(np.load(data_file))
        # Convert numpy arrays to lists for easier handling
        data = {k: np.array(v) for k, v in data.items()}
    else:
        with open(data_file, 'r') as f:
            data = json.load(f)
            data = {k: np.array(v) for k, v in data.items()}

    return metadata, data

def create_comprehensive_analysis(metadata, data):
    """Create comprehensive visualization of clock data"""

    print("\n📈 Creating visualizations...")

    # Convert reference times to seconds from start
    ref_times = (data['reference_ns'] - data['reference_ns'][0]) / 1e9

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: All precision layers over time
    ax1 = fig.add_subplot(gs[0, :])

    layers = [
        ('nanosecond', 'Nanosecond', '#FF6B6B'),
        ('picosecond', 'Picosecond', '#4ECDC4'),
        ('femtosecond', 'Femtosecond', '#45B7D1'),
        ('attosecond', 'Attosecond', '#96CEB4'),
        ('zeptosecond', 'Zeptosecond', '#FFEAA7'),
        ('planck', 'Planck', '#DFE6E9'),
        ('trans_planckian', 'Trans-Planckian', '#00B894')
    ]

    for layer, label, color in layers:
        precisions = data[f'{layer}_precision']
        ax1.semilogy(ref_times, precisions, label=label, color=color, alpha=0.7, linewidth=1.5)

    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Trans-Planckian Clock: All Precision Layers Over Time',
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: Trans-Planckian precision detail
    ax2 = fig.add_subplot(gs[1, 0])
    trans_p = data['trans_planckian_precision']
    ax2.plot(ref_times, trans_p * 1e50, color='#00B894', linewidth=1)
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (×10⁻⁵⁰ s)', fontsize=11, fontweight='bold')
    ax2.set_title('Trans-Planckian Precision Detail', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Precision stability (std dev)
    ax3 = fig.add_subplot(gs[1, 1])
    layer_names = [label for _, label, _ in layers]
    stabilities = []
    for layer, _, _ in layers:
        precisions = data[f'{layer}_precision']
        stability = np.std(precisions) / np.mean(precisions) * 100  # Percentage
        stabilities.append(stability)

    bars = ax3.bar(range(len(layer_names)), stabilities,
                   color=[c for _, _, c in layers], alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(layer_names)))
    ax3.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Stability (Std/Mean %)', fontsize=11, fontweight='bold')
    ax3.set_title('Precision Stability by Layer', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Histogram of trans-Planckian measurements
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(trans_p * 1e50, bins=50, color='#00B894', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Precision (×10⁻⁵⁰ s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Trans-Planckian Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Sample rate analysis
    ax5 = fig.add_subplot(gs[2, 0])
    # Calculate inter-sample intervals
    intervals = np.diff(ref_times)
    ax5.plot(ref_times[1:], intervals * 1000, color='#E74C3C', linewidth=0.5, alpha=0.7)
    ax5.axhline(np.mean(intervals) * 1000, color='blue', linestyle='--',
                label=f'Mean: {np.mean(intervals)*1000:.2f} ms')
    ax5.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Inter-sample Interval (ms)', fontsize=11, fontweight='bold')
    ax5.set_title('Sample Rate Consistency', fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Precision cascade comparison
    ax6 = fig.add_subplot(gs[2, 1])
    mean_precisions = [np.mean(data[f'{layer}_precision']) for layer, _, _ in layers]
    ax6.barh(range(len(layer_names)), mean_precisions,
            color=[c for _, _, c in layers], alpha=0.7, edgecolor='black')
    ax6.set_xscale('log')
    ax6.set_yticks(range(len(layer_names)))
    ax6.set_yticklabels(layer_names, fontsize=10)
    ax6.set_xlabel('Mean Precision (seconds)', fontsize=11, fontweight='bold')
    ax6.set_title('Precision Cascade Summary', fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both', axis='x')

    # Panel 7: Cumulative precision improvement
    ax7 = fig.add_subplot(gs[2, 2])
    improvements = [mean_precisions[0] / p for p in mean_precisions]
    ax7.semilogy(range(len(layer_names)), improvements, 'o-',
                color='#9B59B6', linewidth=2, markersize=8)
    ax7.set_xticks(range(len(layer_names)))
    ax7.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax7.set_ylabel('Improvement Factor', fontsize=11, fontweight='bold')
    ax7.set_title('Cumulative Enhancement', fontweight='bold')
    ax7.grid(True, alpha=0.3, which='both')

    # Panel 8: Statistics summary
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')

    summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                               ║
║                         TRANS-PLANCKIAN CLOCK RUN ANALYSIS                                    ║
║                                                                                               ║
║  Run Duration: {metadata['duration_s']:.3f} seconds                                                              ║
║  Total Measurements: {metadata['total_measurements']:,}                                                  ║
║  Sample Rate: {metadata['actual_sample_rate_hz']:.1f} Hz (target: {metadata['sample_rate_hz']} Hz)                               ║
║                                                                                               ║
║  PRECISION STATISTICS:                                                                        ║
║  ────────────────────────────────────────────────────────────────────────────────────────────║
║  Trans-Planckian: {np.mean(trans_p):.2e} ± {np.std(trans_p):.2e} s                                   ║
║  Zeptosecond:     {np.mean(data['zeptosecond_precision']):.2e} ± {np.std(data['zeptosecond_precision']):.2e} s                                   ║
║  Attosecond:      {np.mean(data['attosecond_precision']):.2e} ± {np.std(data['attosecond_precision']):.2e} s                                   ║
║                                                                                               ║
║  STABILITY (Coefficient of Variation):                                                        ║
║  ────────────────────────────────────────────────────────────────────────────────────────────║
║  Trans-Planckian: {stabilities[-1]:.4f}%                                                                ║
║  All Layers:      {np.mean(stabilities):.4f}% average                                                          ║
║                                                                                               ║
║  Total Enhancement: {improvements[-1]:.2e}×                                                              ║
║                                                                                               ║
║  Status: ✓ OPERATIONAL                                                                       ║
║                                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

    ax8.text(0.5, 0.5, summary_text, ha='center', va='center',
            transform=ax8.transAxes, fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3,
                     edgecolor='green', linewidth=2))

    plt.suptitle(f'Trans-Planckian Live Clock Analysis\nRun: {datetime.fromtimestamp(metadata["start_time_ns"]/1e9).strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=18, fontweight='bold')

    # Save figure
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'live_clock')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_file = os.path.join(results_dir, f'clock_analysis_{timestamp}.png')
    plt.savefig(fig_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {fig_file}")

    plt.show()

    return fig_file

def print_summary_statistics(metadata, data):
    """Print detailed statistics"""

    print("\n" + "="*70)
    print("   DETAILED STATISTICS")
    print("="*70)

    print(f"\n   Run Information:")
    print(f"   ─────────────────")
    print(f"   Duration: {metadata['duration_s']:.3f} s")
    print(f"   Measurements: {metadata['total_measurements']:,}")
    print(f"   Sample Rate: {metadata['actual_sample_rate_hz']:.2f} Hz")

    print(f"\n   Precision by Layer:")
    print(f"   ───────────────────")

    layers = ['nanosecond', 'picosecond', 'femtosecond', 'attosecond',
              'zeptosecond', 'planck', 'trans_planckian']

    for layer in layers:
        precisions = data[f'{layer}_precision']
        mean_p = np.mean(precisions)
        std_p = np.std(precisions)
        min_p = np.min(precisions)
        max_p = np.max(precisions)

        print(f"   {layer.title():20} {mean_p:.2e} ± {std_p:.2e} s")
        print(f"   {'':20} Range: [{min_p:.2e}, {max_p:.2e}]")

    print(f"\n   Planck Time Comparison:")
    print(f"   ───────────────────────")
    planck_time = 5.39e-44
    trans_p_mean = np.mean(data['trans_planckian_precision'])
    orders_below = -np.log10(trans_p_mean / planck_time)
    print(f"   Trans-Planckian: {trans_p_mean:.2e} s")
    print(f"   Planck Time: {planck_time:.2e} s")
    print(f"   Orders Below: {orders_below:.1f}")

def main():
    """Main analysis"""
    print("="*70)
    print("   TRANS-PLANCKIAN CLOCK DATA ANALYZER")
    print("="*70)

    # Load data
    metadata, data = load_latest_clock_data()

    if metadata is None:
        print("\n❌ No data to analyze. Run the clock first:")
        print("   python run_live_clock.py")
        return

    print(f"\n✓ Loaded {metadata['total_measurements']:,} measurements")

    # Print statistics
    print_summary_statistics(metadata, data)

    # Create visualizations
    print("\n" + "="*70)
    create_comprehensive_analysis(metadata, data)

    print("\n" + "="*70)
    print("   ✓ ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
