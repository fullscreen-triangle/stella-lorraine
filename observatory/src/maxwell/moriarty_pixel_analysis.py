"""
visualize_moriarty_pixel_analysis.py

Detailed pixel-level analysis showing how individual pixels
map to S_k coordinates on both faces
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


if __name__ == '__main__':
    # Load data
    TIMESTAMP = "20251126_124850"
    grayscale = np.array(Image.open(f"results/image_processing/real_grayscale_moriarty_{TIMESTAMP}.png"))
    front_sk = np.load(f"results/image_processing/real_front_sk_moriarty_{TIMESTAMP}.npy")
    back_sk = np.load(f"results/image_processing/real_back_sk_moriarty_{TIMESTAMP}.npy")

    # Select interesting regions (dog's eye, nose, background)
    regions = {
        'Eye': (60, 50, 80, 70),      # (row_start, col_start, row_end, col_end)
        'Nose': (90, 45, 110, 65),
        'Background': (20, 80, 40, 100)
    }

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(len(regions), 4, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (region_name, (r1, c1, r2, c2)) in enumerate(regions.items()):

        # Extract region
        gray_region = grayscale[r1:r2, c1:c2]
        front_region = front_sk[:, c1:c2] if front_sk.shape[1] > c2 else front_sk
        back_region = back_sk[:, c1:c2] if back_sk.shape[1] > c2 else back_sk

        # Plot grayscale region
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(gray_region, cmap='gray')
        ax1.set_title(f'{region_name}\n(Grayscale)', fontweight='bold')
        ax1.axis('off')

        # Plot front S_k
        ax2 = fig.add_subplot(gs[idx, 1])
        im2 = ax2.imshow(front_region, cmap='RdBu_r', aspect='auto')
        ax2.set_title(f'{region_name}\n(Front S_k)', fontweight='bold')
        plt.colorbar(im2, ax=ax2)

        # Plot back S_k
        ax3 = fig.add_subplot(gs[idx, 2])
        im3 = ax3.imshow(back_region, cmap='RdBu_r', aspect='auto')
        ax3.set_title(f'{region_name}\n(Back S_k)', fontweight='bold')
        plt.colorbar(im3, ax=ax3)

        # Plot conjugate verification
        ax4 = fig.add_subplot(gs[idx, 3])
        diff_region = front_region + back_region
        im4 = ax4.imshow(diff_region, cmap='seismic', aspect='auto',
                        vmin=-np.max(np.abs(diff_region)),
                        vmax=np.max(np.abs(diff_region)))
        ax4.set_title(f'{region_name}\n(Front + Back)', fontweight='bold')
        plt.colorbar(im4, ax=ax4)

    fig.suptitle('Pixel-Level Dual-Membrane Analysis: "Moriarty"\n' +
                'Regional Analysis of Conjugate Structure',
                fontsize=16, fontweight='bold')

    plt.savefig('moriarty_pixel_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: moriarty_pixel_analysis.png")
    plt.show()
