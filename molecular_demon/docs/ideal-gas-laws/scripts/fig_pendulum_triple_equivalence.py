"""
Figure: Pendulum Triple Equivalence
Shows how a pendulum's motion can be expressed equivalently as:
- Oscillation (continuous periodic motion)
- Categories (discrete positional states)
- Partitions (temporal segments of the period)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
import json
import os

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

colors = {
    'oscillatory': '#3498db',
    'categorical': '#2ecc71',
    'partition': '#e74c3c',
    'pendulum': '#2c3e50',
    'trajectory': '#f39c12',
}

def draw_pendulum(ax, theta, L=2.0, pivot=(0, 0), bob_size=0.15):
    """Draw a pendulum at angle theta"""
    x_bob = pivot[0] + L * np.sin(theta)
    y_bob = pivot[1] - L * np.cos(theta)
    
    # Rod
    ax.plot([pivot[0], x_bob], [pivot[1], y_bob], 'k-', linewidth=2)
    
    # Bob
    circle = Circle((x_bob, y_bob), bob_size, facecolor=colors['pendulum'], 
                    edgecolor='black', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    
    # Pivot
    ax.plot(*pivot, 'ko', markersize=8)
    
    return x_bob, y_bob

def create_figure():
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    
    # =========================================
    # Row 1: Oscillatory Perspective
    # =========================================
    
    # Panel A1: Pendulum swinging (animation snapshot)
    ax = axes[0, 0]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('OSCILLATORY VIEW', fontsize=12, fontweight='bold', color=colors['oscillatory'])
    
    L = 2.0
    pivot = (0, 0)
    theta_max = 0.5  # radians (~30 degrees)
    
    # Draw trajectory arc
    theta_arc = np.linspace(-theta_max, theta_max, 100)
    x_arc = L * np.sin(theta_arc)
    y_arc = -L * np.cos(theta_arc)
    ax.plot(x_arc, y_arc, '--', color=colors['trajectory'], linewidth=2, alpha=0.5)
    
    # Draw pendulum at multiple positions (ghosted)
    for theta in np.linspace(-theta_max, theta_max, 7):
        alpha = 0.2 if theta != 0 else 1.0
        x_bob = L * np.sin(theta)
        y_bob = -L * np.cos(theta)
        ax.plot([0, x_bob], [0, y_bob], '-', color='gray', alpha=alpha, linewidth=1)
        circle = Circle((x_bob, y_bob), 0.12, facecolor=colors['oscillatory'], 
                        alpha=alpha, edgecolor='none')
        ax.add_patch(circle)
    
    # Current position (center)
    draw_pendulum(ax, 0, L, pivot)
    
    # Arrows showing motion
    ax.annotate('', xy=(-1.2, -1.8), xytext=(-0.5, -2),
                arrowprops=dict(arrowstyle='<->', color=colors['oscillatory'], lw=2))
    
    ax.text(0, -2.7, r'$\theta(t) = \theta_{max} \cos(\omega t)$', ha='center', fontsize=11)
    ax.text(0, 0.5, 'Pivot', ha='center', fontsize=9)
    
    # Panel A2: x(t) and v(t) plots
    ax = axes[0, 1]
    t = np.linspace(0, 4*np.pi, 500)
    omega = 1
    theta_t = theta_max * np.cos(omega * t)
    omega_t = -theta_max * omega * np.sin(omega * t)
    
    ax.plot(t, theta_t, '-', color=colors['oscillatory'], linewidth=2, label=r'$\theta(t)$')
    ax.plot(t, omega_t, '--', color=colors['oscillatory'], linewidth=2, alpha=0.6, label=r'$\dot{\theta}(t)$')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (t/T)')
    ax.set_ylabel('Angle / Angular velocity')
    ax.set_title('Continuous Periodic Motion', fontsize=11)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 4*np.pi)
    ax.grid(True, alpha=0.3)
    
    # Mark period
    ax.annotate('', xy=(2*np.pi, -0.4), xytext=(0, -0.4),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(np.pi, -0.5, 'Period T', ha='center', fontsize=10)
    
    # Panel A3: Phase space
    ax = axes[0, 2]
    ax.plot(theta_t, omega_t, '-', color=colors['oscillatory'], linewidth=2)
    ax.scatter([0], [theta_max * omega], s=100, c='red', zorder=10, label='Current state')
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    ax.set_title('Phase Space (Ellipse)', fontsize=11)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # =========================================
    # Row 2: Categorical and Partition Perspectives
    # =========================================
    
    # Panel B1: Categorical view - discrete positions
    ax = axes[1, 0]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CATEGORICAL VIEW', fontsize=12, fontweight='bold', color=colors['categorical'])
    
    # Discrete categories
    n_categories = 8
    theta_cats = np.linspace(-theta_max, theta_max, n_categories)
    
    for i, theta in enumerate(theta_cats):
        x_bob = L * np.sin(theta)
        y_bob = -L * np.cos(theta)
        
        # Draw position
        circle = Circle((x_bob, y_bob), 0.15, facecolor=colors['categorical'], 
                        edgecolor='darkgreen', linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x_bob, y_bob + 0.3, f'$C_{{{i+1}}}$', ha='center', fontsize=9, fontweight='bold')
        
        # Rod (faint)
        ax.plot([0, x_bob], [0, y_bob], '-', color='gray', alpha=0.3, linewidth=1)
    
    ax.plot(0, 0, 'ko', markersize=8)
    
    ax.text(0, -2.7, f'M = {n_categories} categories', ha='center', fontsize=11)
    ax.text(0, -3.1, r'Each $C_i$ is a distinguishable state', ha='center', fontsize=10, style='italic')
    
    # Panel B2: Categories as discrete states
    ax = axes[1, 1]
    
    # Show the categories as a discrete distribution
    categories = np.arange(1, n_categories + 1)
    occupancy = np.exp(-np.abs(categories - n_categories/2 - 0.5) / 2)  # Gaussian-like
    
    ax.bar(categories, occupancy, color=colors['categorical'], edgecolor='darkgreen', 
           linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Category index $m$')
    ax.set_ylabel('Time in category')
    ax.set_title('Discrete State Structure', fontsize=11)
    ax.set_xticks(categories)
    ax.set_xticklabels([f'$C_{i}$' for i in categories])
    
    # Arrow showing traversal
    ax.annotate('', xy=(7.5, 0.7), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(0.5, 0.5), xytext=(7.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(4, 0.8, 'Traversal', ha='center', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B3: Partition view - temporal segments
    ax = axes[1, 2]
    ax.set_title('PARTITION VIEW', fontsize=12, fontweight='bold', color=colors['partition'])
    
    # Show period as a pie/ring divided into partitions
    # Use a different visualization: timeline
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Timeline
    ax.arrow(0.5, 2, 9, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.text(10, 2, 't', fontsize=12)
    ax.text(0.3, 2, '0', fontsize=10)
    ax.text(9.5, 1.5, 'T', fontsize=10)
    
    # Partitions
    n_partitions = 8
    partition_width = 8.5 / n_partitions
    partition_colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_partitions))
    
    for i in range(n_partitions):
        x_start = 0.5 + i * partition_width
        rect = plt.Rectangle((x_start, 1.3), partition_width * 0.9, 1.4,
                              facecolor=partition_colors[i], edgecolor='darkred',
                              linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x_start + partition_width * 0.45, 2, f'$P_{{{i+1}}}$', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.text(5, 3.3, r'$T = \sum_{i=1}^{M} \tau_i$', ha='center', fontsize=12)
    ax.text(5, 0.7, 'Each partition = one category transition', ha='center', fontsize=10, style='italic')
    ax.text(5, 0.2, r'$\langle\tau_p\rangle = T/M$', ha='center', fontsize=11)
    
    plt.tight_layout()
    
    # Add main equivalence statement
    fig.text(0.5, 0.02, 
             r'$\mathbf{TRIPLE\ EQUIVALENCE:}$ Oscillation = Category Traversal = Period Partition',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    fig.text(0.5, -0.02,
             r'Fundamental Identity: $\frac{dM}{dt} = \frac{\omega}{2\pi/M} = \frac{1}{\langle\tau_p\rangle}$',
             ha='center', fontsize=12, style='italic')
    
    return fig

def save_data():
    """Save pendulum data"""
    data = {
        'description': 'Pendulum triple equivalence demonstration',
        'oscillatory': {
            'description': 'Continuous periodic motion',
            'equation': 'theta(t) = theta_max * cos(omega*t)',
            'period': 'T = 2*pi/omega',
            'phase_space': 'Ellipse in (theta, omega) plane'
        },
        'categorical': {
            'description': 'Discrete positional states',
            'M_categories': 8,
            'traversal': 'System visits each category during oscillation'
        },
        'partition': {
            'description': 'Temporal segments of period',
            'relation': 'tau_p = T/M',
            'total': 'T = sum(tau_i)'
        },
        'equivalence': 'dM/dt = omega/(2*pi/M) = 1/<tau_p>'
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_pendulum_triple_equivalence.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_pendulum_triple_equivalence.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

