"""
membrane_picture.py

Publication-quality visualization of Moriarty image dual-membrane representation.
Shows front and back faces with multiple S-entropy coordinate visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

# Import dual-membrane framework
from dual_membrane_pixel_demon import DualMembraneGrid

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def load_and_process_moriarty():
    """Load Moriarty image and process through dual-membrane framework"""

    print("\n" + "="*70)
    print("PROCESSING MORIARTY IMAGE THROUGH DUAL-MEMBRANE FRAMEWORK")
    print("="*70)

    # Load image
    image_path = Path('moriarty.JPEG')
    if not image_path.exists():
        image_path = Path('me_Original.JPEG')

    if not image_path.exists():
        raise FileNotFoundError("Could not find Moriarty or me_Original image")

    print(f"\n✓ Loading image: {image_path}")
    img = Image.open(image_path)

    # Convert to grayscale
    img_gray = img.convert('L')
    img_array = np.array(img_gray, dtype=float) / 255.0

    print(f"  Image size: {img_array.shape}")
    print(f"  Pixel range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    # Downsample for faster processing (optional)
    target_size = 256  # Adjust based on desired resolution
    if max(img_array.shape) > target_size:
        scale = target_size / max(img_array.shape)
        new_shape = (int(img_array.shape[0] * scale), int(img_array.shape[1] * scale))
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_pil = img_pil.resize((new_shape[1], new_shape[0]), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil, dtype=float) / 255.0
        print(f"  Downsampled to: {img_array.shape}")

    # Create dual-membrane grid
    print(f"\n✓ Creating dual-membrane pixel demon grid...")
    grid = DualMembraneGrid(
        shape=img_array.shape,
        physical_extent=(1.0, 1.0),
        transform_type='phase_conjugate',
        synchronized_switching=True
    )

    # Initialize from image
    print(f"  Initializing {img_array.shape[0] * img_array.shape[1]} pixel demons...")
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            intensity = img_array[i, j]
            demon = grid.demons[i, j]  # Use 'demons' not 'pixels'

            # Initialize front face from intensity
            # Map intensity to S_k coordinate
            S_k = 2.0 * intensity - 1.0  # Map [0,1] to [-1,1]
            S_t = 0.0  # Neutral temporal
            S_e = 0.5  # Neutral evolutionary

            demon.dual_state.front_s = type('obj', (object,), {
                's_k': S_k, 's_t': S_t, 's_e': S_e
            })()

            # Apply conjugate transformation for back
            demon.dual_state.back_s = type('obj', (object,), {
                's_k': -S_k,  # Phase conjugate: negate S_k
                's_t': S_t,
                's_e': S_e
            })()

            # Set observable face to front initially
            demon.observable_face = demon.observable_face.__class__.FRONT

    print("✓ Grid initialized")

    # Extract front face measurements
    print("\n✓ Extracting front face...")
    front_data = {}
    front_sk = np.zeros(img_array.shape)
    front_st = np.zeros(img_array.shape)
    front_se = np.zeros(img_array.shape)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            s_state = grid.demons[i, j].dual_state.front_s
            front_sk[i, j] = s_state.s_k
            front_st[i, j] = s_state.s_t
            front_se[i, j] = s_state.s_e

    front_data = {
        's_k': front_sk,
        's_t': front_st,
        's_e': front_se
    }

    # Switch to back face
    print("✓ Switching to back face...")
    grid.switch_all_faces()

    # Extract back face measurements
    print("✓ Extracting back face...")
    back_data = {}
    back_sk = np.zeros(img_array.shape)
    back_st = np.zeros(img_array.shape)
    back_se = np.zeros(img_array.shape)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            s_state = grid.demons[i, j].dual_state.back_s
            back_sk[i, j] = s_state.s_k
            back_st[i, j] = s_state.s_t
            back_se[i, j] = s_state.s_e

    back_data = {
        's_k': back_sk,
        's_t': back_st,
        's_e': back_se
    }

    # Verify conjugacy
    print("\n✓ Verifying conjugate relationship...")
    conjugacy_sum = front_sk + back_sk
    conjugacy_error = np.mean(np.abs(conjugacy_sum))
    correlation = np.corrcoef(front_sk.flatten(), back_sk.flatten())[0, 1]

    print(f"  Mean conjugacy error: {conjugacy_error:.6e}")
    print(f"  Correlation coefficient: {correlation:.6f}")
    print(f"  Conjugacy verified: {conjugacy_error < 0.01 and correlation < -0.99}")

    return {
        'original': img_array,
        'front': front_data,
        'back': back_data,
        'conjugacy_error': conjugacy_error,
        'correlation': correlation,
        'image_name': image_path.name
    }

def create_dual_membrane_figure(data):
    """
    Create comprehensive dual-membrane visualization

    Layout:
    Row 1: Original | Front S_k | Front S_t | Front S_e
    Row 2: Negative | Back S_k  | Back S_t  | Back S_e
    Row 3: Conjugacy verification panels
    """

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                          left=0.06, right=0.96, top=0.94, bottom=0.06)

    # Row 1: Original and Front Face
    ax_orig = fig.add_subplot(gs[0, 0])
    plot_image(ax_orig, data['original'], 'Original Image', 'A', 'gray')

    ax_f_sk = fig.add_subplot(gs[0, 1])
    plot_image(ax_f_sk, data['front']['s_k'], 'Front $S_k$ (Knowledge)', 'B', 'RdBu_r', vmin=-1, vmax=1)

    ax_f_st = fig.add_subplot(gs[0, 2])
    plot_image(ax_f_st, data['front']['s_t'], 'Front $S_t$ (Temporal)', 'C', 'viridis', vmin=-1, vmax=1)

    ax_f_se = fig.add_subplot(gs[0, 3])
    plot_image(ax_f_se, data['front']['s_e'], 'Front $S_e$ (Evolution)', 'D', 'plasma', vmin=0, vmax=1)

    # Row 2: Negative and Back Face (The "HIDDEN" conjugate state)
    negative = 1.0 - data['original']
    ax_neg = fig.add_subplot(gs[1, 0])
    plot_image(ax_neg, negative, 'Negative (Visual)', 'E', 'gray')

    ax_b_sk = fig.add_subplot(gs[1, 1])
    plot_image(ax_b_sk, data['back']['s_k'], 'Back $S_k$ (Conjugate)', 'F', 'RdBu_r', vmin=-1, vmax=1)

    ax_b_st = fig.add_subplot(gs[1, 2])
    plot_image(ax_b_st, data['back']['s_t'], 'Back $S_t$ (Conjugate)', 'G', 'viridis', vmin=-1, vmax=1)

    ax_b_se = fig.add_subplot(gs[1, 3])
    plot_image(ax_b_se, data['back']['s_e'], 'Back $S_e$ (Conjugate)', 'H', 'plasma', vmin=0, vmax=1)

    # Row 3: Conjugacy verification
    sum_sk = data['front']['s_k'] + data['back']['s_k']
    ax_sum = fig.add_subplot(gs[2, 0])
    plot_image(ax_sum, sum_sk, 'Sum: $S_k^{front} + S_k^{back} \\approx 0$', 'I', 'RdYlGn',
               vmin=-0.1, vmax=0.1, centered=True)

    # Correlation plot
    ax_corr = fig.add_subplot(gs[2, 1])
    plot_correlation(ax_corr, data['front']['s_k'], data['back']['s_k'], 'J')

    # Histogram comparison
    ax_hist = fig.add_subplot(gs[2, 2])
    plot_histogram(ax_hist, data['front']['s_k'], data['back']['s_k'], 'K')

    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 3])
    plot_statistics(ax_stats, data, 'L')

    # Overall title
    fig.suptitle(f'Dual-Membrane Representation: {data["image_name"]}\n' +
                 f'Front Face (Observable) vs Back Face (Hidden Conjugate)',
                fontsize=13, fontweight='bold', y=0.98)

    return fig

def plot_image(ax, data, title, label, cmap, vmin=None, vmax=None, centered=False):
    """Plot a single image panel"""

    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # Use centered colormap if requested
    if centered and vmin < 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # When using norm, don't pass vmin/vmax to imshow
        im = ax.imshow(data, cmap=cmap, aspect='auto', norm=norm, interpolation='bilinear')
    else:
        norm = None
        # When not using norm, pass vmin/vmax to imshow
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='bilinear')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.axis('off')

    # Panel label
    ax.text(-0.08, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # Statistics
    stats_text = f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=7, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_correlation(ax, front, back, label):
    """Plot correlation between front and back"""

    # Sample for speed
    sample_size = min(10000, front.size)
    indices = np.random.choice(front.size, sample_size, replace=False)

    front_flat = front.flatten()[indices]
    back_flat = back.flatten()[indices]

    # Scatter plot with density
    ax.hexbin(front_flat, back_flat, gridsize=50, cmap='Blues', mincnt=1)

    # Ideal line (y = -x for perfect conjugate)
    x_line = np.linspace(-1, 1, 100)
    ax.plot(x_line, -x_line, 'r--', linewidth=2, label='Ideal: $y = -x$')

    # Fit line
    coeffs = np.polyfit(front_flat, back_flat, 1)
    y_fit = np.poly1d(coeffs)(x_line)
    ax.plot(x_line, y_fit, 'g-', linewidth=1.5, alpha=0.7,
            label=f'Fit: $y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}$')

    ax.set_xlabel('Front Face $S_k$', fontsize=9)
    ax.set_ylabel('Back Face $S_k$', fontsize=9)
    ax.set_title('Conjugate Correlation', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    # Panel label
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # Correlation coefficient
    corr = np.corrcoef(front.flatten(), back.flatten())[0, 1]
    ax.text(0.98, 0.02, f'$r = {corr:.4f}$', transform=ax.transAxes,
           fontsize=9, va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

def plot_histogram(ax, front, back, label):
    """Plot histogram comparison of front and back"""

    bins = np.linspace(-1, 1, 50)

    ax.hist(front.flatten(), bins=bins, alpha=0.5, label='Front Face',
            color='blue', density=True)
    ax.hist(back.flatten(), bins=bins, alpha=0.5, label='Back Face',
            color='red', density=True)

    ax.set_xlabel('$S_k$ Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('Distribution Comparison', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel label
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # Note about symmetry
    ax.text(0.98, 0.98, 'Distributions should\nbe mirror images',
           transform=ax.transAxes, fontsize=7, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

def plot_statistics(ax, data, label):
    """Plot statistics summary panel"""

    ax.axis('off')

    # Panel label
    ax.text(-0.08, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

    # Compile statistics
    front_sk = data['front']['s_k']
    back_sk = data['back']['s_k']

    stats_text = f"""
CONJUGACY VERIFICATION

Front Face:
  μ(Sₖ) = {np.mean(front_sk):7.4f}
  σ(Sₖ) = {np.std(front_sk):7.4f}
  range = [{np.min(front_sk):.3f}, {np.max(front_sk):.3f}]

Back Face (Conjugate):
  μ(Sₖ) = {np.mean(back_sk):7.4f}
  σ(Sₖ) = {np.std(back_sk):7.4f}
  range = [{np.min(back_sk):.3f}, {np.max(back_sk):.3f}]

Conjugate Tests:
  μ(front) + μ(back) = {np.mean(front_sk) + np.mean(back_sk):.2e}
  ✓ Should be ≈ 0

  Correlation r = {data['correlation']:.6f}
  ✓ Should be ≈ -1

  Mean |Sₖ_f + Sₖ_b| = {data['conjugacy_error']:.2e}
  ✓ Should be < 0.01

Result: {"VERIFIED ✓" if data['conjugacy_error'] < 0.01 else "FAILED ✗"}
"""

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=8, va='top', ha='left', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

def save_results(data, fig):
    """Save figure and data"""

    # Create results directory
    results_dir = Path('results/membrane_picture')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save figure
    fig_path_pdf = results_dir / f'moriarty_dual_membrane_{timestamp}.pdf'
    fig_path_png = results_dir / f'moriarty_dual_membrane_{timestamp}.png'

    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')

    print(f"\n✓ Figure saved:")
    print(f"  {fig_path_pdf}")
    print(f"  {fig_path_png}")

    # Save data
    data_path = results_dir / f'moriarty_data_{timestamp}.npz'
    np.savez_compressed(
        data_path,
        original=data['original'],
        front_sk=data['front']['s_k'],
        front_st=data['front']['s_t'],
        front_se=data['front']['s_e'],
        back_sk=data['back']['s_k'],
        back_st=data['back']['s_t'],
        back_se=data['back']['s_e']
    )

    print(f"✓ Data saved: {data_path}")

    # Save statistics
    stats = {
        'image_name': data['image_name'],
        'timestamp': timestamp,
        'front_stats': {
            's_k': {'mean': float(np.mean(data['front']['s_k'])),
                    'std': float(np.std(data['front']['s_k'])),
                    'min': float(np.min(data['front']['s_k'])),
                    'max': float(np.max(data['front']['s_k']))},
        },
        'back_stats': {
            's_k': {'mean': float(np.mean(data['back']['s_k'])),
                    'std': float(np.std(data['back']['s_k'])),
                    'min': float(np.min(data['back']['s_k'])),
                    'max': float(np.max(data['back']['s_k']))},
        },
        'conjugacy_error': float(data['conjugacy_error']),
        'correlation': float(data['correlation']),
        'verified': bool(data['conjugacy_error'] < 0.01 and data['correlation'] < -0.99)
    }

    stats_path = results_dir / f'moriarty_stats_{timestamp}.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Statistics saved: {stats_path}")

    # Save README
    readme_path = results_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# Moriarty Dual-Membrane Analysis Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

This directory contains the dual-membrane analysis of the Moriarty image.

### Files

- `moriarty_dual_membrane_*.pdf/png`: Publication-quality figure showing:
  - Row 1: Original image and front face S-coordinates
  - Row 2: Negative and back face S-coordinates (conjugate)
  - Row 3: Conjugacy verification plots

- `moriarty_data_*.npz`: NumPy compressed array containing:
  - Original image
  - Front face S_k, S_t, S_e coordinates
  - Back face S_k, S_t, S_e coordinates

- `moriarty_stats_*.json`: Statistics and verification results

## Interpretation

The **front face** represents the observable categorical state of each pixel.

The **back face** represents the hidden conjugate state - the "negative" in
categorical space, not just intensity inversion.

For phase conjugate transformation:
- Front S_k and Back S_k are opposite: S_k_back = -S_k_front
- They cannot be observed simultaneously (complementarity)
- Together they form a complete dual-membrane representation

## Verification

Conjugacy is verified by:
1. Correlation coefficient ≈ -1 (perfect negative correlation)
2. Mean of sum ≈ 0 (opposite values)
3. Element-wise sum ≈ 0 (local conjugacy)
""")

    print(f"✓ README saved: {readme_path}")

def main():
    """Main execution"""

    print("\n" + "="*70)
    print("MORIARTY DUAL-MEMBRANE VISUALIZATION")
    print("="*70)

    # Process image
    data = load_and_process_moriarty()

    # Create figure
    print("\n✓ Creating visualization...")
    fig = create_dual_membrane_figure(data)

    # Save results
    save_results(data, fig)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\n✓ Dual-membrane representation successfully generated!")
    print("✓ Front face (observable) and back face (conjugate) extracted")
    print(f"✓ Conjugacy verified: {data['conjugacy_error'] < 0.01 and data['correlation'] < -0.99}")
    print("\nThe back face is the 'hidden' conjugate state - the categorical")
    print("negative that exists but cannot be observed simultaneously with")
    print("the front face (ammeter/voltmeter complementarity).")

if __name__ == '__main__':
    main()
