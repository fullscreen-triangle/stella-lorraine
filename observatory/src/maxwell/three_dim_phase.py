"""
visualize_moriarty_3d_phase_space.py

Interactive 3D visualization of S_k coordinate space
showing the dual-membrane structure in phase space
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

    # Sample points for visualization
    n_samples = 5000
    indices = np.random.choice(front_sk.size, n_samples, replace=False)
    front_flat = front_sk.flatten()[indices]
    back_flat = back_sk.flatten()[indices]

    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot front face
    ax.scatter(front_flat, np.zeros_like(front_flat), np.arange(len(front_flat)),
            c=front_flat, cmap='Blues', alpha=0.6, s=10, label='Front Face')

    # Plot back face
    ax.scatter(back_flat, np.ones_like(back_flat), np.arange(len(back_flat)),
            c=back_flat, cmap='Reds', alpha=0.6, s=10, label='Back Face')

    # Connect conjugate pairs
    for i in range(0, len(front_flat), 50):  # Every 50th point to avoid clutter
        ax.plot([front_flat[i], back_flat[i]], [0, 1], [i, i],
            'gray', alpha=0.1, linewidth=0.5)

    ax.set_xlabel('S_k Value', fontsize=12)
    ax.set_ylabel('Face (0=Front, 1=Back)', fontsize=12)
    ax.set_zlabel('Sample Index', fontsize=12)
    ax.set_title('3D Phase Space: Dual-Membrane Structure\n"Moriarty"',
                fontsize=16, fontweight='bold')
    ax.legend()

    # Rotate view
    def rotate(angle):
        ax.view_init(elev=20, azim=angle)

    # Create animation
    anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2),
                        interval=50, repeat=True)

    output_file = 'moriarty_3d_phase_space.gif'
    anim.save(output_file, writer='pillow', fps=20, dpi=100)
    print(f"✓ Saved: {output_file}")

    plt.show()
