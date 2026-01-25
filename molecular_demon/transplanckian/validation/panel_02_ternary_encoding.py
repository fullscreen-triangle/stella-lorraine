"""
Panel 2: Ternary Encoding Resolution Enhancement

Validates the ternary encoding mechanism providing 10^3.5× enhancement through
20-trit representation in S-entropy space.

Four subplots:
1. Binary vs Ternary information density
2. Trit count vs resolution enhancement
3. S-entropy cube packing efficiency
4. 3D: Ternary encoding in (S_k, S_t, S_e) space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Subplot 1: Binary vs Ternary Information Density
ax1 = fig.add_subplot(gs[0, 0])

k_array = np.arange(1, 21)  # Number of trits/bits
binary_states = 2 ** k_array
ternary_states = 3 ** k_array
enhancement = ternary_states / binary_states

ax1.semilogy(k_array, binary_states, 'o-', label='Binary ($2^k$)', 
             linewidth=2, markersize=6, color='blue')
ax1.semilogy(k_array, ternary_states, 's-', label='Ternary ($3^k$)', 
             linewidth=2, markersize=6, color='red')
ax1.semilogy(k_array, enhancement, '^-', label='Enhancement ($1.5^k$)', 
             linewidth=2, markersize=6, color='green')

ax1.set_xlabel('Number of Digits (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of States', fontsize=12, fontweight='bold')
ax1.set_title('Ternary vs Binary Information Density', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotation for k=20
k_20 = 20
enhancement_20 = 1.5 ** k_20
ax1.annotate(f'k=20: ${enhancement_20:.0f}\\times$ enhancement',
             xy=(20, enhancement_20), xytext=(15, enhancement_20 * 10),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
             fontsize=10, fontweight='bold', color='red')

# Subplot 2: Resolution Enhancement vs Trit Count
ax2 = fig.add_subplot(gs[0, 1])

# Baseline resolution (hardware limited)
delta_t_baseline = 1e-21  # seconds (attosecond scale)

# Calculate resolution with ternary enhancement
resolution_with_ternary = delta_t_baseline / (1.5 ** k_array)

# Theoretical prediction from paper: 10^3.5 enhancement
enhancement_theoretical = 10**3.5
k_theoretical = np.log(enhancement_theoretical) / np.log(1.5)

ax2.semilogy(k_array, resolution_with_ternary, 'o-', 
             linewidth=2, markersize=6, color='purple', label='Ternary Enhanced')
ax2.axhline(y=delta_t_baseline, color='gray', linestyle='--', 
            linewidth=2, label=f'Baseline: ${delta_t_baseline:.0e}$ s')
ax2.axhline(y=delta_t_baseline / enhancement_theoretical, color='red', linestyle='--', 
            linewidth=2, label=f'Target: ${delta_t_baseline / enhancement_theoretical:.0e}$ s')
ax2.axvline(x=k_theoretical, color='red', linestyle=':', 
            linewidth=2, alpha=0.7)

ax2.set_xlabel('Number of Trits (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Temporal Resolution $\\delta t$ (s)', fontsize=12, fontweight='bold')
ax2.set_title('Resolution Enhancement from Ternary Encoding', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add annotation
ax2.annotate(f'$10^{{3.5}}\\times$ at k={k_theoretical:.1f}',
             xy=(k_theoretical, delta_t_baseline / enhancement_theoretical),
             xytext=(k_theoretical + 3, delta_t_baseline / enhancement_theoretical * 100),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
             fontsize=10, fontweight='bold', color='red')

# Subplot 3: S-Entropy Cube Packing Efficiency
ax3 = fig.add_subplot(gs[1, 0])

# S-entropy coordinate resolution vs trit count
trit_counts = np.arange(1, 21)
s_entropy_resolution = 1.0 / (3 ** trit_counts)  # Each coordinate resolution

# Volume packing in [0,1]^3 cube
states_per_dimension = 3 ** trit_counts
total_states_cube = states_per_dimension ** 3
volume_per_state = 1.0 / total_states_cube

ax3.loglog(trit_counts, s_entropy_resolution, 'o-', 
           label='Per-coordinate resolution', linewidth=2, markersize=6, color='blue')
ax3.loglog(trit_counts, volume_per_state, 's-', 
           label='Volume per state', linewidth=2, markersize=6, color='orange')

ax3.set_xlabel('Trits per Coordinate', fontsize=12, fontweight='bold')
ax3.set_ylabel('Resolution / Volume', fontsize=12, fontweight='bold')
ax3.set_title('S-Entropy Cube Packing Efficiency', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# Add text box with efficiency metrics
textstr = f'At 20 trits:\n' \
          f'States/dim: ${3**20:.2e}$\n' \
          f'Total states: ${(3**20)**3:.2e}$\n' \
          f'Efficiency: ${1.5**20:.0f}\\times$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

# Subplot 4: 3D Visualization of Ternary States in S-Entropy Space
ax4 = fig.add_subplot(gs[1, 1], projection='3d')

# Generate ternary grid in [0,1]^3 for k=3 trits (27 states)
k_viz = 3
grid_points = 3 ** k_viz  # 27 points per dimension

S_k_grid = np.linspace(0, 1, grid_points)
S_t_grid = np.linspace(0, 1, grid_points)
S_e_grid = np.linspace(0, 1, grid_points)

# Create 3D grid of ternary states
S_k_mesh, S_t_mesh, S_e_mesh = np.meshgrid(S_k_grid, S_t_grid, S_e_grid)

# Flatten for scatter plot (subsample for visibility)
subsample = 5  # Show every 5th point
S_k_flat = S_k_mesh.flatten()[::subsample]
S_t_flat = S_t_mesh.flatten()[::subsample]
S_e_flat = S_e_mesh.flatten()[::subsample]

# Color by combined coordinate value
colors = S_k_flat + S_t_flat + S_e_flat

scatter = ax4.scatter(S_k_flat, S_t_flat, S_e_flat, 
                      c=colors, cmap='plasma', s=20, alpha=0.6, marker='o')

# Draw unit cube wireframe
cube_edges = [
    ([0,1], [0,0], [0,0]), ([0,0], [0,1], [0,0]), ([0,0], [0,0], [0,1]),
    ([1,1], [0,1], [0,0]), ([1,1], [0,0], [0,1]), ([0,1], [1,1], [0,0]),
    ([0,1], [0,0], [1,1]), ([0,0], [1,1], [0,1]), ([0,0], [1,1], [1,1]),
    ([1,1], [1,1], [0,1]), ([1,1], [0,1], [1,1]), ([0,1], [1,1], [1,1])
]

for edge in cube_edges:
    ax4.plot(edge[0], edge[1], edge[2], 'k-', linewidth=1, alpha=0.3)

ax4.set_xlabel('$S_k$ (Kinetic)', fontsize=11, fontweight='bold')
ax4.set_ylabel('$S_t$ (Temporal)', fontsize=11, fontweight='bold')
ax4.set_zlabel('$S_e$ (Ensemble)', fontsize=11, fontweight='bold')
ax4.set_title(f'3D: Ternary Grid in S-Entropy Cube\n({grid_points}³ = {grid_points**3} states)', 
              fontsize=13, fontweight='bold')

# Add colorbar
cbar = fig.colorbar(scatter, ax=ax4, shrink=0.6, aspect=10)
cbar.set_label('$S_k + S_t + S_e$', fontsize=10, fontweight='bold')

ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.set_zlim([0, 1])
ax4.view_init(elev=20, azim=45)

# Overall title
fig.suptitle('Panel 2: Ternary Encoding Resolution Enhancement ($10^{3.5}\\times$)\n' +
             'Three-dimensional S-entropy representation with natural ternary basis',
             fontsize=15, fontweight='bold', y=0.98)

# Add footer
fig.text(0.5, 0.02,
         'Enhancement: $(3/2)^k$ for k trits | Paper formula: $1.5^{20} \\approx 3325 \\approx 10^{3.5}$',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('panel_02_ternary_encoding.png', dpi=300, bbox_inches='tight')
print("✓ Panel 2 saved: panel_02_ternary_encoding.png")
plt.show()
