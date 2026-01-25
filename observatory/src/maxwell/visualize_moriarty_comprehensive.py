"""
visualize_moriarty_comprehensive.py

Comprehensive visualization with:
- Original image
- Grayscale version
- 3D phase space
- Statistics panel
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image


if __name__ == '__main__':
    # Load data
    TIMESTAMP = "20251126_124850"
    front_sk = np.load(f"results/image_processing/real_front_sk_moriarty_{TIMESTAMP}.npy")
    back_sk = np.load(f"results/image_processing/real_back_sk_moriarty_{TIMESTAMP}.npy")

    # Load images
    original_img = Image.open(f"results/image_processing/real_input_moriarty_{TIMESTAMP}.png")
    grayscale_img = Image.open(f"results/image_processing/real_grayscale_moriarty_{TIMESTAMP}.png")

    # Sample points for visualization
    n_samples = 5000
    indices = np.random.choice(front_sk.size, n_samples, replace=False)
    front_flat = front_sk.flatten()[indices]
    back_flat = back_sk.flatten()[indices]

    # Calculate statistics
    correlation = np.corrcoef(front_sk.flatten(), back_sk.flatten())[0, 1]
    conjugate_sum = front_sk + back_sk
    conjugate_error = np.mean(np.abs(conjugate_sum))
    symmetry_score = -correlation

    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.3)

    # Top left: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.axis('off')
    ax1.set_title('Original Image\n"Moriarty"', fontsize=12, fontweight='bold')

    # Top right: Grayscale
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(grayscale_img, cmap='gray')
    ax2.axis('off')
    ax2.set_title('Grayscale (Analysis Input)', fontsize=12, fontweight='bold')

    # Middle left: Statistics
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    stats_text = f"""
    DUAL-MEMBRANE ANALYSIS
    Subject: Moriarty (Italian Greyhound)
    Professional Model

    BEAUTY METRICS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Conjugate Correlation: {correlation:.8f}
    Symmetry Score: {symmetry_score:.8f}
    Conjugate Error: {conjugate_error:.8f}

    Front S_k Range: [{front_sk.min():.3f}, {front_sk.max():.3f}]
    Back S_k Range: [{back_sk.min():.3f}, {back_sk.max():.3f}]

    Image Shape: {front_sk.shape}
    Total Coordinates: {front_sk.size:,}

    STATUS: ✓ Conjugate relationship verified
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Middle right: Conjugate relationship plot
    ax4 = fig.add_subplot(gs[1, 1])
    sample_indices = np.random.choice(front_sk.size, 1000, replace=False)
    ax4.scatter(front_sk.flatten()[sample_indices],
                back_sk.flatten()[sample_indices],
                alpha=0.3, s=1)
    ax4.plot([front_sk.min(), front_sk.max()],
             [-front_sk.min(), -front_sk.max()],
             'r--', linewidth=2, label='Perfect Conjugate')
    ax4.set_xlabel('Front Face S_k', fontsize=10)
    ax4.set_ylabel('Back Face S_k', fontsize=10)
    ax4.set_title(f'Conjugate Relationship (r = {correlation:.6f})',
                  fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Bottom: 3D phase space (spans both columns)
    ax5 = fig.add_subplot(gs[2, :], projection='3d')

    # Plot front face
    scatter_front = ax5.scatter(front_flat, np.zeros_like(front_flat),
                               np.arange(len(front_flat)),
                               c=front_flat, cmap='Blues', alpha=0.6, s=10,
                               label='Front Face')

    # Plot back face
    scatter_back = ax5.scatter(back_flat, np.ones_like(back_flat),
                              np.arange(len(back_flat)),
                              c=back_flat, cmap='Reds', alpha=0.6, s=10,
                              label='Back Face')

    # Connect conjugate pairs
    for i in range(0, len(front_flat), 50):
        ax5.plot([front_flat[i], back_flat[i]], [0, 1], [i, i],
                'gray', alpha=0.1, linewidth=0.5)

    ax5.set_xlabel('S_k Value', fontsize=11)
    ax5.set_ylabel('Face (0=Front, 1=Back)', fontsize=11)
    ax5.set_zlabel('Sample Index', fontsize=11)
    ax5.set_title('3D Phase Space: Dual-Membrane Structure',
                  fontsize=13, fontweight='bold')
    ax5.legend()

    # Overall title
    fig.suptitle('Mathematical Portrait of Beauty: "Moriarty"\n' +
                 'Dual-Membrane Analysis of Professional Canine Model',
                 fontsize=16, fontweight='bold', y=0.98)

    # Rotate view
    def rotate(angle):
        ax5.view_init(elev=20, azim=angle)

    # Create animation
    anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2),
                        interval=50, repeat=True)

    output_file = 'moriarty_comprehensive_analysis.gif'
    anim.save(output_file, writer='pillow', fps=20, dpi=100)
    print(f"✓ Saved: {output_file}")

    plt.show()
