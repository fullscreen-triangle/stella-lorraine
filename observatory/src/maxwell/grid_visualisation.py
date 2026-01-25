"""
grid_visualisation.py

Publication figure showing dual-membrane grid with spatial patterns.
Demonstrates conjugate relationship visually.
Can use synthetic data or load from validation results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from pathlib import Path
import argparse

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

def load_real_data():
    """Load real validation data if available"""
    data_dir = Path('results/dual_membrane_validation')

    # Find latest validation files
    json_files = list(data_dir.glob('validation_results_*.json'))

    if not json_files:
        return None

    # Get the most recent one
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
    timestamp = latest_json.stem.split('_', 2)[-1]

    try:
        front_sk = np.load(data_dir / f'front_sk_image_{timestamp}.npy')
        back_sk = np.load(data_dir / f'back_sk_image_{timestamp}.npy')
        test_pattern = np.load(data_dir / f'test_pattern_{timestamp}.npy')
        carbon_copy = np.load(data_dir / f'carbon_copy_{timestamp}.npy')

        print(f"✓ Loaded real data from {timestamp}")
        return {
            'front_sk': front_sk,
            'back_sk': back_sk,
            'test_pattern': test_pattern,
            'carbon_copy': carbon_copy,
            'timestamp': timestamp,
            'real_data': True
        }
    except Exception as e:
        print(f"Could not load real data: {e}")
        return None

def generate_synthetic_data():
    """Generate synthetic data matching Test 5 results"""
    np.random.seed(42)
    grid_size = 8

    # Create spatial pattern (sinusoidal as in test)
    x = np.linspace(0, np.pi, grid_size)
    y = np.linspace(0, np.pi, grid_size)
    X, Y = np.meshgrid(x, y)

    # Front face: mean=0.506, std=0.177
    base_pattern = np.sin(X) * np.cos(Y)
    front_sk = 0.506 + 0.177 * base_pattern + np.random.normal(0, 0.02, (grid_size, grid_size))

    # Back face: conjugate (negative)
    back_sk = -front_sk + np.random.normal(0, 0.01, (grid_size, grid_size))

    # Test pattern and carbon copy
    test_pattern = np.random.rand(grid_size, grid_size) * 2 - 1
    carbon_copy = -test_pattern

    print("✓ Generated synthetic data")
    return {
        'front_sk': front_sk,
        'back_sk': back_sk,
        'test_pattern': test_pattern,
        'carbon_copy': carbon_copy,
        'timestamp': 'synthetic',
        'real_data': False
    }

def create_figure_2(data):
    """
    Figure 2: Dual-Membrane Grid Spatial Patterns

    Panel A: Front face S_k map
    Panel B: Back face S_k map
    Panel C: Sum (front + back) showing zero
    Panel D: Difference showing spatial structure
    """

    front_sk = data['front_sk']
    back_sk = data['back_sk']

    # Sum and difference
    sum_sk = front_sk + back_sk
    diff_sk = front_sk - back_sk

    fig = plt.figure(figsize=(7.5, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.92, top=0.93, bottom=0.07)

    # Panel A: Front face
    ax_a = fig.add_subplot(gs[0, 0])
    plot_grid_heatmap(ax_a, front_sk, 'Front Face $S_k$', 'A', 'RdBu_r')

    # Panel B: Back face
    ax_b = fig.add_subplot(gs[0, 1])
    plot_grid_heatmap(ax_b, back_sk, 'Back Face $S_k$ (Conjugate)', 'B', 'RdBu_r')

    # Panel C: Sum (should be ~0)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_grid_heatmap(ax_c, sum_sk, 'Sum: Front + Back ≈ 0', 'C', 'RdYlGn',
                     vmin=-0.05, vmax=0.05)

    # Panel D: Difference (shows structure)
    ax_d = fig.add_subplot(gs[1, 1])
    plot_grid_heatmap(ax_d, diff_sk, 'Difference: Front - Back', 'D', 'viridis')

    # Add overall title
    data_type = "Real Data" if data['real_data'] else "Synthetic Data"
    fig.suptitle(f'Dual-Membrane Grid: Spatial Conjugate Patterns ({data_type})',
                fontsize=12, fontweight='bold', y=0.98)

    return fig

def plot_grid_heatmap(ax, data, title, label, cmap, vmin=None, vmax=None):
    """Plot a single grid heatmap"""

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
    stats_text = f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=7, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Generate dual-membrane grid visualization')
    parser.add_argument('--synthetic', action='store_true',
                       help='Force use of synthetic data')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("DUAL-MEMBRANE GRID VISUALIZATION")
    print("="*70 + "\n")

    # Load data
    if args.synthetic:
        data = generate_synthetic_data()
    else:
        data = load_real_data()
        if data is None:
            print("Real data not found, using synthetic data")
            data = generate_synthetic_data()

    # Create figure
    print("Creating visualization...")
    fig = create_figure_2(data)

    # Save
    output_dir = Path('results/dual_membrane_validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = 'synthetic' if not data['real_data'] else 'real'
    pdf_file = output_dir / f'figure_2_grid_patterns_{suffix}.pdf'
    png_file = output_dir / f'figure_2_grid_patterns_{suffix}.png'

    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')

    print(f"\n✓ Figure 2 saved:")
    print(f"  {pdf_file}")
    print(f"  {png_file}")

    # Print statistics
    print("\nStatistics:")
    print(f"  Front mean: {np.mean(data['front_sk']):.4f}")
    print(f"  Back mean: {np.mean(data['back_sk']):.4f}")
    print(f"  Sum mean: {np.mean(data['front_sk'] + data['back_sk']):.6e}")
    correlation = np.corrcoef(data['front_sk'].flatten(),
                             data['back_sk'].flatten())[0, 1]
    print(f"  Correlation: {correlation:.6f}")

    plt.close()

if __name__ == '__main__':
    main()
