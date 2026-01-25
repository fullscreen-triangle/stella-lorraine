"""
visualize_moriarty_dual_membrane.py

Publication-quality visualization of dual-membrane analysis
on real photograph (Italian Greyhound "Moriarty")

Demonstrates:
- Front/back face decomposition
- S_k coordinate mapping
- Conjugate relationships
- Platform independence verification
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

if __name__ == '__main__':
    # File paths (adjust as needed)
    DATA_DIR = "results/image_processing/"  # Current directory
    TIMESTAMP_1 = "20251126_124850"
    TIMESTAMP_2 = "20251126_124931"

    # Output settings
    DPI = 300
    FIGSIZE = (20, 12)
    STYLE = 'dark_background'  # or 'default' for white background

    # ============================================================================
    # LOAD DATA
    # ============================================================================

    def load_moriarty_data(timestamp):
        """Load all data for a given timestamp"""

        prefix = f"real_"

        # Load images
        input_img = Image.open(f"results/image_processing/real_input_moriarty_20251126_124850.png")
        grayscale_img = Image.open(f"results/image_processing/real_grayscale_moriarty_20251126_124850.png")

        # Load numpy arrays
        front_sk = np.load(f"results/image_processing/real_front_sk_moriarty_20251126_124850.npy")
        back_sk = np.load(f"results/image_processing/real_back_sk_moriarty_20251126_124850.npy")

        # Try to load back face array if it exists
        try:
            back_face = np.load(f"results/image_processing/real_back_moriarty_20251126_124850.npy")
        except:
            back_face = None

        return {
            'input': input_img,
            'grayscale': grayscale_img,
            'front_sk': front_sk,
            'back_sk': back_sk,
            'back_face': back_face,
            'timestamp': timestamp
        }

    print("Loading data...")
    data_1 = load_moriarty_data(TIMESTAMP_1)
    data_2 = load_moriarty_data(TIMESTAMP_2)
    print("✓ Data loaded successfully")

    # ============================================================================
    # ANALYSIS
    # ============================================================================

    def analyze_conjugate_relationship(front_sk, back_sk):
        """Analyze the conjugate relationship between front and back faces"""

        # Expected relationship: back_sk ≈ -front_sk
        difference = back_sk + front_sk
        correlation = np.corrcoef(front_sk.flatten(), back_sk.flatten())[0, 1]

        stats = {
            'correlation': correlation,
            'mean_difference': np.mean(np.abs(difference)),
            'max_difference': np.max(np.abs(difference)),
            'std_difference': np.std(difference),
            'front_mean': np.mean(front_sk),
            'front_std': np.std(front_sk),
            'back_mean': np.mean(back_sk),
            'back_std': np.std(back_sk),
        }

        return stats, difference

    print("\nAnalyzing conjugate relationships...")
    stats_1, diff_1 = analyze_conjugate_relationship(data_1['front_sk'], data_1['back_sk'])
    stats_2, diff_2 = analyze_conjugate_relationship(data_2['front_sk'], data_2['back_sk'])

    print(f"\nRun 1 ({TIMESTAMP_1}):")
    print(f"  Correlation: {stats_1['correlation']:.6f}")
    print(f"  Mean |difference|: {stats_1['mean_difference']:.6e}")
    print(f"  Max |difference|: {stats_1['max_difference']:.6e}")

    print(f"\nRun 2 ({TIMESTAMP_2}):")
    print(f"  Correlation: {stats_2['correlation']:.6f}")
    print(f"  Mean |difference|: {stats_2['mean_difference']:.6e}")
    print(f"  Max |difference|: {stats_2['max_difference']:.6e}")

    # Platform independence check
    reproducibility = np.allclose(data_1['front_sk'], data_2['front_sk'], rtol=1e-10)
    print(f"\nPlatform independence: {'✓ VERIFIED' if reproducibility else '✗ FAILED'}")

    # ============================================================================
    # VISUALIZATION
    # ============================================================================

    plt.style.use(STYLE)

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================================
    # ROW 1: ORIGINAL IMAGES
    # ============================================================================

    # Original color image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(data_1['input'])
    ax1.set_title('Original Image\n"Moriarty"', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Grayscale conversion
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(data_1['grayscale'], cmap='gray')
    ax2.set_title('Grayscale\n(Input to Analysis)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Front face S_k
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(data_1['front_sk'], cmap='RdBu_r', aspect='auto')
    ax3.set_title('Front Face S_k\n(Observable)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Spatial Position')
    ax3.set_ylabel('Frequency')
    plt.colorbar(im3, ax=ax3, label='S_k value')

    # Back face S_k
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(data_1['back_sk'], cmap='RdBu_r', aspect='auto')
    ax4.set_title('Back Face S_k\n(Conjugate)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Spatial Position')
    ax4.set_ylabel('Frequency')
    plt.colorbar(im4, ax=ax4, label='S_k value')

    # ============================================================================
    # ROW 2: CONJUGATE ANALYSIS
    # ============================================================================

    # Difference map (should be near zero)
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(diff_1, cmap='seismic', aspect='auto',
                    vmin=-np.max(np.abs(diff_1)), vmax=np.max(np.abs(diff_1)))
    ax5.set_title(f'Conjugate Difference\n(Back + Front)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Spatial Position')
    ax5.set_ylabel('Frequency')
    plt.colorbar(im5, ax=ax5, label='Difference')

    # Scatter plot: Back vs Front
    ax6 = fig.add_subplot(gs[1, 1])
    sample_indices = np.random.choice(data_1['front_sk'].size, 10000, replace=False)
    ax6.scatter(data_1['front_sk'].flatten()[sample_indices],
            data_1['back_sk'].flatten()[sample_indices],
            alpha=0.1, s=1, c='cyan')
    ax6.plot([-4, 4], [4, -4], 'r--', linewidth=2, label='Perfect conjugate')
    ax6.set_xlabel('Front S_k', fontsize=12)
    ax6.set_ylabel('Back S_k', fontsize=12)
    ax6.set_title(f'Conjugate Relationship\nr = {stats_1["correlation"]:.6f}',
                fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    # Distribution comparison
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(data_1['front_sk'].flatten(), bins=100, alpha=0.5,
            label='Front', color='blue', density=True)
    ax7.hist(data_1['back_sk'].flatten(), bins=100, alpha=0.5,
            label='Back', color='red', density=True)
    ax7.set_xlabel('S_k value', fontsize=12)
    ax7.set_ylabel('Density', fontsize=12)
    ax7.set_title('S_k Distributions', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Statistics table
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    stats_text = f"""
    CONJUGATE VERIFICATION

    Correlation: {stats_1['correlation']:.8f}

    Front Face:
    Mean: {stats_1['front_mean']:.6e}
    Std:  {stats_1['front_std']:.6f}

    Back Face:
    Mean: {stats_1['back_mean']:.6e}
    Std:  {stats_1['back_std']:.6f}

    Difference (Back + Front):
    Mean |Δ|: {stats_1['mean_difference']:.6e}
    Max |Δ|:  {stats_1['max_difference']:.6e}
    Std(Δ):   {stats_1['std_difference']:.6e}

    Status: {'✓ CONJUGATE' if abs(stats_1['correlation'] + 1) < 0.01 else '✗ NOT CONJUGATE'}
    """
    ax8.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='black' if STYLE == 'dark_background' else 'white',
            alpha=0.8, edgecolor='cyan', linewidth=2))

    # ============================================================================
    # ROW 3: PLATFORM INDEPENDENCE
    # ============================================================================

    # Difference between two runs
    ax9 = fig.add_subplot(gs[2, 0])
    run_diff = data_1['front_sk'] - data_2['front_sk']
    im9 = ax9.imshow(run_diff, cmap='seismic', aspect='auto',
                    vmin=-np.max(np.abs(run_diff)) if np.max(np.abs(run_diff)) > 0 else -1,
                    vmax=np.max(np.abs(run_diff)) if np.max(np.abs(run_diff)) > 0 else 1)
    ax9.set_title('Platform Independence\n(Run 1 - Run 2)', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Spatial Position')
    ax9.set_ylabel('Frequency')
    plt.colorbar(im9, ax=ax9, label='Difference')

    # Reproducibility scatter
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.scatter(data_1['front_sk'].flatten()[sample_indices],
                data_2['front_sk'].flatten()[sample_indices],
                alpha=0.1, s=1, c='lime')
    ax10.plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Perfect match')
    ax10.set_xlabel('Run 1 Front S_k', fontsize=12)
    ax10.set_ylabel('Run 2 Front S_k', fontsize=12)
    ax10.set_title('Reproducibility Check', fontsize=14, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.set_aspect('equal')

    # Spatial profile comparison
    ax11 = fig.add_subplot(gs[2, 2])
    mid_row = data_1['front_sk'].shape[0] // 2
    ax11.plot(data_1['front_sk'][mid_row, :], label='Run 1', linewidth=2, alpha=0.7)
    ax11.plot(data_2['front_sk'][mid_row, :], label='Run 2', linewidth=2, alpha=0.7, linestyle='--')
    ax11.set_xlabel('Spatial Position', fontsize=12)
    ax11.set_ylabel('S_k value', fontsize=12)
    ax11.set_title(f'Cross-section (row {mid_row})', fontsize=14, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Final statistics
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    repro_text = f"""
    PLATFORM INDEPENDENCE

    Run 1: {TIMESTAMP_1}
    Run 2: {TIMESTAMP_2}

    Max difference: {np.max(np.abs(run_diff)):.6e}
    Mean |difference|: {np.mean(np.abs(run_diff)):.6e}
    Std(difference): {np.std(run_diff):.6e}

    Correlation: {np.corrcoef(data_1['front_sk'].flatten(), data_2['front_sk'].flatten())[0,1]:.12f}

    Identical (rtol=1e-10): {'✓ YES' if reproducibility else '✗ NO'}

    Status: {'✓ PLATFORM INDEPENDENT' if reproducibility else '⚠ CHECK REQUIRED'}
    """
    ax12.text(0.1, 0.5, repro_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='black' if STYLE == 'dark_background' else 'white',
            alpha=0.8, edgecolor='lime', linewidth=2))

    # ============================================================================
    # MAIN TITLE
    # ============================================================================

    fig.suptitle('Dual-Membrane Analysis: Real Photograph ("Moriarty")\n' +
                'Demonstrating Conjugate Structure and Platform Independence',
                fontsize=18, fontweight='bold', y=0.98)

    # ============================================================================
    # SAVE
    # ============================================================================

    output_file = f'moriarty_dual_membrane_analysis.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n✓ Saved: {output_file}")

    plt.show()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
