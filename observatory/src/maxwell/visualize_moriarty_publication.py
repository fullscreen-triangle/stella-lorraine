"""
visualize_moriarty_publication.py

Clean, minimal publication-quality figure
suitable for Nature, Science, or similar journals
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib as mpl


if __name__ == '__main__':

    # Set publication style
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5

    # Load data
    TIMESTAMP = "20251126_124850"
    input_img = Image.open(f"results/image_processing/real_input_moriarty_{TIMESTAMP}.png")
    grayscale = Image.open(f"results/image_processing/real_grayscale_moriarty_{TIMESTAMP}.png")
    front_sk = np.load(f"results/image_processing/real_front_sk_moriarty_{TIMESTAMP}.npy")
    back_sk = np.load(f"results/image_processing/real_back_sk_moriarty_{TIMESTAMP}.npy")

    # Analysis
    correlation = np.corrcoef(front_sk.flatten(), back_sk.flatten())[0, 1]
    difference = front_sk + back_sk

    # Create figure (single column width: 3.5 inches)
    fig = plt.figure(figsize=(7, 8))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Panel A: Original image
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(input_img)
    ax_a.set_title('a', loc='left', fontweight='bold', fontsize=10)
    ax_a.axis('off')

    # Panel B: Grayscale
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(grayscale, cmap='gray')
    ax_b.set_title('b', loc='left', fontweight='bold', fontsize=10)
    ax_b.axis('off')

    # Panel C: Front S_k
    ax_c = fig.add_subplot(gs[1, 0])
    im_c = ax_c.imshow(front_sk, cmap='RdBu_r', aspect='auto')
    ax_c.set_title('c', loc='left', fontweight='bold', fontsize=10)
    ax_c.set_xlabel('Position', fontsize=8)
    ax_c.set_ylabel('Frequency', fontsize=8)
    cbar_c = plt.colorbar(im_c, ax=ax_c)
    cbar_c.set_label('$S_k$', fontsize=8)

    # Panel D: Back S_k
    ax_d = fig.add_subplot(gs[1, 1])
    im_d = ax_d.imshow(back_sk, cmap='RdBu_r', aspect='auto')
    ax_d.set_title('d', loc='left', fontweight='bold', fontsize=10)
    ax_d.set_xlabel('Position', fontsize=8)
    ax_d.set_ylabel('Frequency', fontsize=8)
    cbar_d = plt.colorbar(im_d, ax=ax_d)
    cbar_d.set_label('$S_k^*$', fontsize=8)

    # Panel E: Conjugate scatter
    ax_e = fig.add_subplot(gs[2, 0])
    sample_idx = np.random.choice(front_sk.size, 5000, replace=False)
    ax_e.scatter(front_sk.flatten()[sample_idx],
                back_sk.flatten()[sample_idx],
                s=0.5, alpha=0.3, c='black', rasterized=True)
    ax_e.plot([-4, 4], [4, -4], 'r-', linewidth=1, label='$S_k^* = -S_k$')
    ax_e.set_xlabel('$S_k$ (front)', fontsize=8)
    ax_e.set_ylabel('$S_k^*$ (back)', fontsize=8)
    ax_e.set_title('e', loc='left', fontweight='bold', fontsize=10)
    ax_e.legend(fontsize=6)
    ax_e.text(0.05, 0.95, f'$r = {correlation:.4f}$',
            transform=ax_e.transAxes, fontsize=7,
            verticalalignment='top')
    ax_e.set_aspect('equal')

    # Panel F: Difference
    ax_f = fig.add_subplot(gs[2, 1])
    im_f = ax_f.imshow(difference, cmap='seismic', aspect='auto',
                    vmin=-np.max(np.abs(difference)),
                    vmax=np.max(np.abs(difference)))
    ax_f.set_title('f', loc='left', fontweight='bold', fontsize=10)
    ax_f.set_xlabel('Position', fontsize=8)
    ax_f.set_ylabel('Frequency', fontsize=8)
    cbar_f = plt.colorbar(im_f, ax=ax_f)
    cbar_f.set_label('$S_k + S_k^*$', fontsize=8)

    # Save
    plt.savefig('moriarty_publication_figure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('moriarty_publication_figure.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: moriarty_publication_figure.pdf")
    print("✓ Saved: moriarty_publication_figure.png")
    plt.show()
