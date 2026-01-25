"""
visualize_moriarty_3d_phase_space_vertical.py

Vertical layout with image on top
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

    # Load original image
    original_img = Image.open(f"results/image_processing/real_input_moriarty_{TIMESTAMP}.png")

    # Sample points for visualization
    n_samples = 5000
    indices = np.random.choice(front_sk.size, n_samples, replace=False)
    front_flat = front_sk.flatten()[indices]
    back_flat = back_sk.flatten()[indices]

    # Create figure with vertical layout
    fig = plt.figure(figsize=(16, 20))

    # Top: Original image
    ax_img = fig.add_subplot(211)
    ax_img.imshow(original_img)
    ax_img.axis('off')
    ax_img.set_title('Original Image: "Moriarty" - Professional Model, Italian Greyhound\n' +
                     'Grandson of "Hypnotic Poison", Photographed in Croatia',
                     fontsize=14, fontweight='bold', pad=20)

    # Bottom: 3D phase space
    ax = fig.add_subplot(212, projection='3d')

    # Plot front face
    scatter_front = ax.scatter(front_flat, np.zeros_like(front_flat), np.arange(len(front_flat)),
                              c=front_flat, cmap='Blues', alpha=0.6, s=10, label='Front Face')

    # Plot back face
    scatter_back = ax.scatter(back_flat, np.ones_like(back_flat), np.arange(len(back_flat)),
                             c=back_flat, cmap='Reds', alpha=0.6, s=10, label='Back Face')

    # Connect conjugate pairs
    for i in range(0, len(front_flat), 50):
        ax.plot([front_flat[i], back_flat[i]], [0, 1], [i, i],
                'gray', alpha=0.1, linewidth=0.5)

    ax.set_xlabel('S_k Value', fontsize=12)
    ax.set_ylabel('Face (0=Front, 1=Back)', fontsize=12)
    ax.set_zlabel('Sample Index', fontsize=12)
    ax.set_title('Dual-Membrane Structure in S_k Phase Space',
                 fontsize=14, fontweight='bold')
    ax.legend()

    # Add correlation text
    correlation = np.corrcoef(front_sk.flatten(), back_sk.flatten())[0, 1]
    fig.text(0.5, 0.48, f'Conjugate Correlation: r = {correlation:.6f}',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Rotate view
    def rotate(angle):
        ax.view_init(elev=20, azim=angle)

    # Create animation
    anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2),
                        interval=50, repeat=True)

    output_file = 'moriarty_3d_phase_space_vertical.gif'
    anim.save(output_file, writer='pillow', fps=20, dpi=100)
    print(f"✓ Saved: {output_file}")

    plt.show()
