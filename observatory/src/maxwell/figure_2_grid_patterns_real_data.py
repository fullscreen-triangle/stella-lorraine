"""
figure_2_grid_patterns_real_data.py

Publication figure using ACTUAL experimental data from validation.
Uses the .npy files from Test 5 results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import json
from pathlib import Path

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def find_latest_validation_data(data_dir='results/dual_membrane_validation'):
    """Find the most recent validation data files"""
    data_dir = Path(data_dir)

    # Find all validation result files
    json_files = list(data_dir.glob('validation_results_*.json'))

    if not json_files:
        raise FileNotFoundError(f"No validation results found in {data_dir}")

    # Get the most recent one
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

    # Extract timestamp
    timestamp = latest_json.stem.split('_', 2)[-1]

    print(f"\n✓ Using validation data from: {timestamp}")

    return data_dir, timestamp

def load_validation_data(data_dir=None, timestamp=None):
    """Load actual validation data from .npy files"""

    if data_dir is None or timestamp is None:
        data_dir, timestamp = find_latest_validation_data()

    data_dir = Path(data_dir)

    print("\nLoading numpy arrays...")

    # Load numpy arrays
    front_sk = np.load(data_dir / f'front_sk_image_{timestamp}.npy')
    back_sk = np.load(data_dir / f'back_sk_image_{timestamp}.npy')
    test_pattern = np.load(data_dir / f'test_pattern_{timestamp}.npy')
    carbon_copy = np.load(data_dir / f'carbon_copy_{timestamp}.npy')

    print(f"  ✓ front_sk: {front_sk.shape}")
    print(f"  ✓ back_sk: {back_sk.shape}")
    print(f"  ✓ test_pattern: {test_pattern.shape}")
    print(f"  ✓ carbon_copy: {carbon_copy.shape}")

    # Load validation results JSON
    json_file = data_dir / f'validation_results_{timestamp}.json'
    with open(json_file, 'r') as f:
        validation_results = json.load(f)

    print(f"  ✓ validation_results loaded")

    return {
        'front_sk': front_sk,
        'back_sk': back_sk,
        'test_pattern': test_pattern,
        'carbon_copy': carbon_copy,
        'validation_results': validation_results,
        'timestamp': timestamp
    }

def create_figure_2_real_data(data_dir='results/dual_membrane_validation'):
    """
    Figure 2: Dual-Membrane Grid Spatial Patterns (REAL DATA)

    Panel A: Front face S_k map (actual data)
    Panel B: Back face S_k map (actual data)
    Panel C: Sum (front + back) showing zero
    Panel D: Difference
    Panel E: Test pattern (input)
    Panel F: Carbon copy (output)
    """

    # Load real data
    data = load_validation_data(data_dir)
    front_sk = data['front_sk']
    back_sk = data['back_sk']
    test_pattern = data['test_pattern']
    carbon_copy = data['carbon_copy']

    # Calculate derived quantities
    sum_sk = front_sk + back_sk
    diff_sk = front_sk - back_sk

    # Create figure
    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.92, top=0.93, bottom=0.05)

    # Panel A: Front face
    ax_a = fig.add_subplot(gs[0, 0])
    plot_grid_heatmap(ax_a, front_sk, 'Front Face $S_k$ (Measured)', 'A', 'RdBu_r')

    # Panel B: Back face
    ax_b = fig.add_subplot(gs[0, 1])
    plot_grid_heatmap(ax_b, back_sk, 'Back Face $S_k$ (Conjugate)', 'B', 'RdBu_r')

    # Panel C: Sum (should be ~0)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_grid_heatmap(ax_c, sum_sk, 'Sum: Front + Back ≈ 0', 'C', 'RdYlGn',
                     vmin=-0.1, vmax=0.1)

    # Panel D: Difference (shows structure)
    ax_d = fig.add_subplot(gs[1, 1])
    plot_grid_heatmap(ax_d, diff_sk, 'Difference: Front - Back', 'D', 'viridis')

    # Panel E: Test pattern
    ax_e = fig.add_subplot(gs[2, 0])
    plot_grid_heatmap(ax_e, test_pattern, 'Test Pattern (Input)', 'E', 'plasma')

    # Panel F: Carbon copy
    ax_f = fig.add_subplot(gs[2, 1])
    plot_grid_heatmap(ax_f, carbon_copy, 'Carbon Copy (Output)', 'F', 'plasma')

    # Overall title with timestamp
    timestamp = data['timestamp']
    fig.suptitle(f'Dual-Membrane Grid: Experimental Data\nValidation: {timestamp}',
                fontsize=11, fontweight='bold', y=0.98)

    return fig, data

def plot_grid_heatmap(ax, data, title, label, cmap, vmin=None, vmax=None):
    """Plot a single grid heatmap with actual data"""

    # Determine color scale
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # Create centered colormap if data crosses zero
    if vmin < 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # When using norm, don't pass vmin/vmax to imshow
        im = ax.imshow(data, cmap=cmap, aspect='auto', norm=norm, interpolation='nearest')
    else:
        norm = None
        # When not using norm, pass vmin/vmax to imshow
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    # Grid lines
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(np.arange(data.shape[1]))
    ax.set_yticklabels(np.arange(data.shape[0]))
    ax.tick_params(labelsize=7)

    # Minor ticks for grid
    ax.set_xticks(np.arange(data.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1)

    # Labels
    ax.set_xlabel('Pixel X', fontsize=8)
    ax.set_ylabel('Pixel Y', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=10)

    # Panel label
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    # Add statistics
    stats_text = (f'μ={np.mean(data):.4f}\n'
                 f'σ={np.std(data):.4f}\n'
                 f'min={np.min(data):.4f}\n'
                 f'max={np.max(data):.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=6.5, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    """Main execution"""

    print("\n" + "="*70)
    print("FIGURE 2: REAL EXPERIMENTAL DATA")
    print("="*70)

    # Generate figure with real data
    fig, data = create_figure_2_real_data()

    # Save
    output_dir = Path('results/dual_membrane_validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_file = output_dir / 'figure_2_grid_patterns_REAL_DATA.pdf'
    png_file = output_dir / 'figure_2_grid_patterns_REAL_DATA.png'

    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')

    # Print statistics
    print(f"\nFront face S_k:")
    print(f"  Mean: {np.mean(data['front_sk']):.6f}")
    print(f"  Std:  {np.std(data['front_sk']):.6f}")
    print(f"  Min:  {np.min(data['front_sk']):.6f}")
    print(f"  Max:  {np.max(data['front_sk']):.6f}")

    print(f"\nBack face S_k:")
    print(f"  Mean: {np.mean(data['back_sk']):.6f}")
    print(f"  Std:  {np.std(data['back_sk']):.6f}")
    print(f"  Min:  {np.min(data['back_sk']):.6f}")
    print(f"  Max:  {np.max(data['back_sk']):.6f}")

    print(f"\nSum (Front + Back):")
    sum_data = data['front_sk'] + data['back_sk']
    print(f"  Mean: {np.mean(sum_data):.6e}")
    print(f"  Std:  {np.std(sum_data):.6e}")
    print(f"  Max deviation from zero: {np.max(np.abs(sum_data)):.6e}")

    print(f"\nConjugate verification:")
    correlation = np.corrcoef(data['front_sk'].flatten(),
                             data['back_sk'].flatten())[0, 1]
    print(f"  Correlation coefficient: {correlation:.6f}")
    print(f"  Expected for perfect conjugate: -1.000000")
    print(f"  Match: {abs(correlation + 1.0) < 0.01}")

    print("\n✓ Figure 2 saved with REAL experimental data")
    print(f"  {pdf_file}")
    print(f"  {png_file}")

    plt.close()

if __name__ == '__main__':
    main()
