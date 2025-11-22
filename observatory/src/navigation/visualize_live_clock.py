#!/usr/bin/env python3
"""
Live Clock Data Visualization
==============================
Extracts and visualizes data from live clock NPZ files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import glob

def load_npz_data(npz_file):
    """Load data from NPZ file"""
    print(f"\n📂 Loading: {os.path.basename(npz_file)}")

    data = np.load(npz_file)

    # List all arrays in the file
    print(f"\n📊 Arrays in file:")
    for key in data.files:
        array = data[key]
        print(f"   - {key}: shape={array.shape}, dtype={array.dtype}")

    return data

def extract_clock_data(data):
    """Extract clock data from NPZ file"""
    extracted = {}

    # Common field names in clock experiments
    possible_fields = [
        'timestamps', 'time', 't',
        'frequencies', 'frequency', 'freq',
        'cpu_cycles', 'cycles',
        'molecular_freq', 'mol_freq',
        'coordination', 'sync',
        'precision', 'accuracy',
        'hardware_time', 'molecular_time',
        'phase', 'drift',
        'counters', 'performance_counters'
    ]

    # Extract available fields
    for field in data.files:
        extracted[field] = data[field]

        # Check if field is one of the common ones
        for possible in possible_fields:
            if possible in field.lower():
                print(f"   ✓ Found {possible}-related data: {field}")

    return extracted

def visualize_live_clock_data(npz_file, output_dir=None):
    """Create comprehensive visualization of live clock data with bar charts and radar charts"""

    # Load data
    data = load_npz_data(npz_file)
    extracted = extract_clock_data(data)

    # Setup output
    if output_dir is None:
        output_dir = os.path.dirname(npz_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with multiple visualization types
    fig = plt.figure(figsize=(20, 12))

    print(f"\n📈 Creating visualizations...")

    # Separate numeric arrays for processing
    numeric_arrays = {}
    for key, array in extracted.items():
        if np.issubdtype(array.dtype, np.number) and array.ndim == 1:
            numeric_arrays[key] = array

    n_arrays = len(numeric_arrays)

    if n_arrays == 0:
        print("   ⚠ No numeric 1D arrays found for visualization")
        return None, extracted

    # Layout: 3 rows
    # Row 1: Bar charts (distributions/histograms)
    # Row 2: Bar charts (statistics comparison)
    # Row 3: Radar chart (all arrays combined)

    # Row 1: Histograms as bar charts
    n_cols = min(4, n_arrays)
    for idx, (key, array) in enumerate(list(numeric_arrays.items())[:4], 1):
        ax = plt.subplot(3, 4, idx)

        # Create histogram
        if len(np.unique(array)) > 1:
            counts, bins = np.histogram(array, bins=30)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            ax.bar(bin_centers, counts, width=(bins[1]-bins[0])*0.8,
                   alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{key}\nDistribution', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            # Constant value
            ax.bar([0], [len(array)], alpha=0.7, color='coral', edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title(f'{key}\nConstant: {array[0]:.2e}', fontweight='bold', fontsize=10)
            ax.set_xticks([])

    # Row 2: Statistics comparison as grouped bar chart
    ax_stats = plt.subplot(3, 2, 3)

    stats_data = []
    labels = []
    for key, array in list(numeric_arrays.items())[:8]:
        stats_data.append([
            np.min(array),
            np.mean(array),
            np.max(array)
        ])
        labels.append(key[:15])  # Truncate long names

    if stats_data:
        x = np.arange(len(labels))
        width = 0.25

        mins = [s[0] for s in stats_data]
        means = [s[1] for s in stats_data]
        maxs = [s[2] for s in stats_data]

        ax_stats.bar(x - width, mins, width, label='Min', alpha=0.8, color='#3498db')
        ax_stats.bar(x, means, width, label='Mean', alpha=0.8, color='#2ecc71')
        ax_stats.bar(x + width, maxs, width, label='Max', alpha=0.8, color='#e74c3c')

        ax_stats.set_ylabel('Value')
        ax_stats.set_title('Statistical Comparison (Min/Mean/Max)', fontweight='bold')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax_stats.legend()
        ax_stats.grid(True, alpha=0.3, axis='y')

    # Row 2: Variance/Std deviation comparison
    ax_var = plt.subplot(3, 2, 4)

    variances = []
    var_labels = []
    for key, array in list(numeric_arrays.items())[:8]:
        variances.append(np.std(array))
        var_labels.append(key[:15])

    if variances:
        colors = plt.cm.viridis(np.linspace(0, 1, len(variances)))
        bars = ax_var.barh(range(len(var_labels)), variances, alpha=0.8, color=colors, edgecolor='black')
        ax_var.set_yticks(range(len(var_labels)))
        ax_var.set_yticklabels(var_labels, fontsize=8)
        ax_var.set_xlabel('Standard Deviation')
        ax_var.set_title('Variability Comparison', fontweight='bold')
        ax_var.grid(True, alpha=0.3, axis='x')

    # Row 3: Radar chart combining all arrays
    ax_radar = plt.subplot(3, 2, 5, projection='polar')

    # Prepare radar chart data
    radar_data = []
    radar_labels = []
    for key, array in list(numeric_arrays.items())[:8]:
        # Normalize to 0-1 range for radar chart
        if np.std(array) > 0:
            normalized = (array - np.min(array)) / (np.max(array) - np.min(array))
            radar_data.append(np.mean(normalized))
        else:
            radar_data.append(0.5)  # Center for constant values
        radar_labels.append(key[:15])

    if radar_data:
        angles = np.linspace(0, 2 * np.pi, len(radar_data), endpoint=False).tolist()
        radar_data += radar_data[:1]  # Complete the circle
        angles += angles[:1]

        ax_radar.plot(angles, radar_data, 'o-', linewidth=2, color='#e74c3c', alpha=0.8)
        ax_radar.fill(angles, radar_data, alpha=0.25, color='#e74c3c')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(radar_labels, fontsize=8)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Normalized Values (Radar View)', fontweight='bold', pad=20)
        ax_radar.grid(True)

    # Row 3: Time series sample (if applicable)
    ax_time = plt.subplot(3, 2, 6)

    # Show first array as time series with markers
    if numeric_arrays:
        first_key = list(numeric_arrays.keys())[0]
        first_array = numeric_arrays[first_key]

        # Sample data if too large
        sample_size = min(500, len(first_array))
        indices = np.linspace(0, len(first_array)-1, sample_size, dtype=int)

        ax_time.plot(indices, first_array[indices], 'o-', linewidth=1,
                    markersize=3, alpha=0.7, color='#9b59b6')
        ax_time.set_xlabel('Sample Index')
        ax_time.set_ylabel('Value')
        ax_time.set_title(f'Time Series: {first_key}', fontweight='bold')
        ax_time.grid(True, alpha=0.3)

        # Add trend line if data varies
        if np.std(first_array) > 0:
            z = np.polyfit(indices, first_array[indices], 1)
            p = np.poly1d(z)
            ax_time.plot(indices, p(indices), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax_time.legend()

    plt.suptitle(f'Live Clock Data Analysis\n{os.path.basename(npz_file)}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    fig_file = os.path.join(output_dir, f'{base_name}_barchart_radar_{timestamp}.png')
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Bar chart & radar visualization saved: {fig_file}")

    return fig_file, extracted

def create_detailed_analysis(extracted, npz_file, output_dir=None):
    """Create detailed statistical analysis"""

    if output_dir is None:
        output_dir = os.path.dirname(npz_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    analysis = {
        'timestamp': timestamp,
        'source_file': os.path.basename(npz_file),
        'arrays': {}
    }

    print(f"\n📊 Statistical Analysis:")

    for key, array in extracted.items():
        stats = {
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'size': int(array.size)
        }

        # Calculate statistics for numeric arrays
        if np.issubdtype(array.dtype, np.number):
            stats.update({
                'min': float(np.min(array)),
                'max': float(np.max(array)),
                'mean': float(np.mean(array)),
                'std': float(np.std(array)),
                'median': float(np.median(array))
            })

            print(f"\n   {key}:")
            print(f"      Shape: {array.shape}")
            print(f"      Range: [{stats['min']:.2e}, {stats['max']:.2e}]")
            print(f"      Mean: {stats['mean']:.2e} ± {stats['std']:.2e}")
        else:
            print(f"\n   {key}: {array.shape} ({array.dtype})")

        analysis['arrays'][key] = stats

    # Save analysis
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    json_file = os.path.join(output_dir, f'{base_name}_analysis_{timestamp}.json')

    with open(json_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n💾 Analysis saved: {json_file}")

    return json_file, analysis

def process_all_npz_files(directory):
    """Process all NPZ files in directory"""

    print("\n" + "="*70)
    print("   LIVE CLOCK DATA EXTRACTION & VISUALIZATION")
    print("="*70)

    # Find all NPZ files
    npz_files = glob.glob(os.path.join(directory, '*.npz'))

    if not npz_files:
        print(f"\n❌ No NPZ files found in: {directory}")
        return

    print(f"\n📂 Found {len(npz_files)} NPZ file(s):")
    for f in npz_files:
        print(f"   - {os.path.basename(f)}")

    results = []

    for npz_file in npz_files:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(npz_file)}")
        print(f"{'='*70}")

        try:
            # Visualize
            fig_file, extracted = visualize_live_clock_data(npz_file)

            # Analyze
            json_file, analysis = create_detailed_analysis(extracted, npz_file)

            results.append({
                'npz_file': npz_file,
                'visualization': fig_file,
                'analysis': json_file,
                'status': 'success'
            })

        except Exception as e:
            print(f"\n❌ Error processing {os.path.basename(npz_file)}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'npz_file': npz_file,
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    print(f"\n" + "="*70)
    print(f"   PROCESSING COMPLETE")
    print(f"="*70)

    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n   Successfully processed: {successful}/{len(npz_files)} files")

    for result in results:
        if result['status'] == 'success':
            print(f"\n   ✓ {os.path.basename(result['npz_file'])}")
            print(f"      Visualization: {os.path.basename(result['visualization'])}")
            print(f"      Analysis: {os.path.basename(result['analysis'])}")
        else:
            print(f"\n   ✗ {os.path.basename(result['npz_file'])}: {result.get('error', 'Unknown error')}")

    print(f"\n{'='*70}\n")

    return results

def main():
    """Main function"""
    import sys

    # Get directory from command line or use default
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Default to live_clock directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.join(current_dir, '..', '..', 'results', 'live_clock')

    # Process all NPZ files
    results = process_all_npz_files(directory)

    return results

if __name__ == "__main__":
    results = main()
