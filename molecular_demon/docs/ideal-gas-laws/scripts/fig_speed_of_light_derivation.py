"""
Figure: Derivation of the Speed of Light
Shows how c emerges as the maximum categorical transition rate
from the container scaling thought experiment.

Key insight: If you scale a container while preserving configuration snapshots,
velocities must scale proportionally. But velocities cannot exceed c.
Therefore, c is the limit of categorical transition rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import json
import os

# Constants
c = 299792458  # m/s

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
    'particles': '#3498db',
    'trajectory': '#2ecc71',
    'forbidden': '#e74c3c',
    'limit': '#9b59b6',
    'categorical': '#27ae60',
}

def generate_gas_ensemble(n_particles, box_size, seed=42):
    """Generate random particle positions in a box"""
    np.random.seed(seed)
    positions = np.random.rand(n_particles, 3) * box_size
    return positions

def generate_velocities(n_particles, v_scale, seed=42):
    """Generate random velocities with given scale"""
    np.random.seed(seed + 1)
    velocities = (np.random.rand(n_particles, 3) - 0.5) * v_scale
    return velocities

def create_figure():
    fig = plt.figure(figsize=(12, 10))
    
    # 2x2 grid
    # Panel A: Original container with gas ensemble
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Original configuration
    n_particles = 30
    box_size_original = 1.0
    positions_original = generate_gas_ensemble(n_particles, box_size_original)
    velocities_original = generate_velocities(n_particles, 0.1)
    
    # Plot particles
    ax1.scatter(positions_original[:, 0], positions_original[:, 1], positions_original[:, 2],
                s=50, c=colors['particles'], alpha=0.7, edgecolors='white')
    
    # Plot velocity arrows
    for i in range(min(10, n_particles)):
        ax1.quiver(positions_original[i, 0], positions_original[i, 1], positions_original[i, 2],
                   velocities_original[i, 0], velocities_original[i, 1], velocities_original[i, 2],
                   color=colors['trajectory'], arrow_length_ratio=0.3, linewidth=1)
    
    # Draw box
    box_edges = [
        [[0, 1], [0, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 1]],
        [[1, 1], [0, 1], [0, 0]], [[1, 1], [0, 0], [0, 1]], [[0, 1], [1, 1], [0, 0]],
        [[0, 0], [1, 1], [0, 1]], [[0, 1], [0, 0], [1, 1]], [[0, 0], [0, 1], [1, 1]],
        [[1, 1], [1, 1], [0, 1]], [[1, 1], [0, 1], [1, 1]], [[0, 1], [1, 1], [1, 1]],
    ]
    for edge in box_edges:
        ax1.plot3D(*edge, 'k-', alpha=0.3)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('A: Original Container (V = V₀)\nv ~ 100 m/s')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    
    # Panel B: Expanded container (same configurations, higher velocities needed)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    scale_factor = 2.0
    positions_scaled = positions_original * scale_factor
    velocities_scaled = velocities_original * scale_factor  # Must scale to maintain configuration sequence!
    
    ax2.scatter(positions_scaled[:, 0], positions_scaled[:, 1], positions_scaled[:, 2],
                s=50, c=colors['particles'], alpha=0.7, edgecolors='white')
    
    for i in range(min(10, n_particles)):
        ax2.quiver(positions_scaled[i, 0], positions_scaled[i, 1], positions_scaled[i, 2],
                   velocities_scaled[i, 0], velocities_scaled[i, 1], velocities_scaled[i, 2],
                   color=colors['trajectory'], arrow_length_ratio=0.3, linewidth=1)
    
    for edge in box_edges:
        scaled_edge = [[e * scale_factor for e in dim] for dim in edge]
        ax2.plot3D(*scaled_edge, 'k-', alpha=0.3)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title(f'B: Scaled Container (V = {scale_factor}³V₀)\nv ~ {100*scale_factor:.0f} m/s required!')
    ax2.set_xlim(0, scale_factor)
    ax2.set_ylim(0, scale_factor)
    ax2.set_zlim(0, scale_factor)
    
    # Panel C: Required velocity vs scale factor
    ax3 = fig.add_subplot(2, 2, 3)
    
    scale_factors = np.linspace(1, 1e8, 1000)
    v_original = 500  # m/s (typical thermal velocity)
    v_required = v_original * scale_factors
    
    # The limit
    v_limit = c * np.ones_like(scale_factors)
    
    ax3.loglog(scale_factors, v_required, '-', color=colors['trajectory'], linewidth=2,
               label='Required velocity (v = k·v₀)')
    ax3.loglog(scale_factors, v_limit, '--', color=colors['limit'], linewidth=2,
               label='Speed of light (c)')
    
    # Mark the critical scale
    k_critical = c / v_original
    ax3.axvline(x=k_critical, color=colors['forbidden'], linestyle=':', alpha=0.7)
    ax3.axhline(y=c, color=colors['limit'], linestyle=':', alpha=0.3)
    
    # Shade forbidden region
    ax3.fill_between(scale_factors[v_required > c], v_required[v_required > c], 
                     v_limit[v_required > c], alpha=0.3, color=colors['forbidden'],
                     label='Forbidden (v > c)')
    
    ax3.scatter([k_critical], [c], s=100, c=colors['limit'], zorder=10, 
                marker='*', edgecolors='black')
    ax3.annotate(f'Critical scale\nk = c/v₀ ≈ {k_critical:.0e}', 
                 xy=(k_critical, c), xytext=(k_critical*10, c/10),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Scale factor k (container size / original size)')
    ax3.set_ylabel('Required velocity (m/s)')
    ax3.set_title('C: Velocity Scaling with Container Size')
    ax3.legend(loc='lower right')
    ax3.set_xlim(1, 1e8)
    ax3.set_ylim(1e2, 1e10)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: The categorical interpretation
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Show categorical transition rate vs physical limit
    transition_rates = np.logspace(0, 50, 500)
    
    # Maximum transition rate (Planck frequency)
    omega_planck = 1.85e43  # rad/s
    
    # Physical transition rates
    physical_rates = np.clip(transition_rates, 0, omega_planck)
    
    ax4.loglog(transition_rates, transition_rates, '--', color='gray', alpha=0.5,
               label='If unlimited')
    ax4.loglog(transition_rates, np.minimum(transition_rates, omega_planck), '-', 
               color=colors['categorical'], linewidth=2, label='Categorical (bounded)')
    ax4.axhline(y=omega_planck, color=colors['limit'], linestyle=':', linewidth=2)
    
    # Fill forbidden region
    ax4.fill_between(transition_rates[transition_rates > omega_planck], 
                     omega_planck, transition_rates[transition_rates > omega_planck],
                     alpha=0.2, color=colors['forbidden'])
    
    ax4.text(1e45, omega_planck * 1.5, r'$\omega_{Planck}$', fontsize=10, color=colors['limit'])
    ax4.text(1e42, 1e47, 'FORBIDDEN\n(impossible)', fontsize=12, color=colors['forbidden'],
             ha='center', fontweight='bold')
    
    # Add the key insight
    textbox = (r'$\mathbf{Key\ Insight:}$' + '\n\n'
               r'$\frac{\Delta x}{\Delta t} \leq c$' + '\n\n'
               r'$\Rightarrow$ Maximum categorical' + '\n'
               r'transition rate exists' + '\n\n'
               r'$\omega_{max} = \omega_{Planck}$' + '\n'
               r'$v_{max} = c$')
    ax4.text(0.02, 0.98, textbox, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.set_xlabel('Attempted transition rate (rad/s)')
    ax4.set_ylabel('Actual transition rate (rad/s)')
    ax4.set_title('D: The Speed of Light as Categorical Limit')
    ax4.legend(loc='lower right')
    ax4.set_xlim(1, 1e50)
    ax4.set_ylim(1, 1e50)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add main title
    fig.suptitle('Derivation of the Speed of Light from Categorical Structure', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    return fig

def save_data():
    """Save derivation data"""
    data = {
        'description': 'Speed of light derivation from categorical limits',
        'key_insight': 'If configurations are scaled, velocities must scale proportionally. '
                       'But velocities cannot exceed c. Therefore c is the maximum categorical transition rate.',
        'derivation': {
            'step1': 'Container scaling by factor k requires v -> k*v to maintain configuration sequence',
            'step2': 'But v cannot exceed c (relativistic limit)',
            'step3': 'Therefore k_max = c/v_0, beyond which configurations are impossible',
            'step4': 'This proves c is the maximum rate of categorical transition: Δx/Δt <= c'
        },
        'constants': {
            'c_m_s': 299792458,
            'omega_Planck_rad_s': 1.85e43
        }
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_speed_of_light_derivation.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_speed_of_light_derivation.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

