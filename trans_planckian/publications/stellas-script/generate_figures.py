"""
Generate 4-panel figure for CatScript categorical thermodynamics paper
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11

# Create figure with 4 panels
fig = plt.figure(figsize=(16, 4))

# ============================================================================
# Panel 1: 3D S-Entropy Space Trajectories
# ============================================================================
ax1 = fig.add_subplot(141, projection='3d')

# Generate categorical trajectories through S-entropy space
n_trajectories = 5
n_points = 100

for i in range(n_trajectories):
    # Parametric trajectory through (S_k, S_t, S_e) space
    t = np.linspace(0, 2*np.pi, n_points)

    # Different trajectory shapes
    S_k = 1e-23 * (1 + 0.5*np.sin(2*t + i*np.pi/5))
    S_t = 1e-24 * (1 + 0.3*np.cos(3*t + i*np.pi/3))
    S_e = 1e-25 * (1 + 0.4*np.sin(t + i*np.pi/4))

    # Color by progress along trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))

    for j in range(n_points-1):
        ax1.plot(S_k[j:j+2], S_t[j:j+2], S_e[j:j+2],
                color=colors[j], alpha=0.6, linewidth=1.5)

# Mark start and end points
ax1.scatter([1e-23], [1e-24], [0], c='green', s=100, marker='o',
           edgecolors='black', linewidths=2, label='Start', zorder=10)
ax1.scatter([1e-23], [1e-24], [1e-25], c='red', s=100, marker='s',
           edgecolors='black', linewidths=2, label='End', zorder=10)

ax1.set_xlabel('$S_k$ (kinetic)', fontsize=9, labelpad=1)
ax1.set_ylabel('$S_t$ (temporal)', fontsize=9, labelpad=1)
ax1.set_zlabel('$S_e$ (evolution)', fontsize=9, labelpad=1)
ax1.set_title('Categorical Trajectories', fontweight='bold', pad=5)

# Format tick labels in scientific notation
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
ax1.tick_params(axis='both', which='major', labelsize=7, pad=-2)

# Adjust viewing angle
ax1.view_init(elev=20, azim=45)
ax1.legend(loc='upper right', fontsize=7, framealpha=0.8)

# ============================================================================
# Panel 2: Enhancement Mechanism Scaling
# ============================================================================
ax2 = fig.add_subplot(142)

mechanisms = ['Ternary', 'Multi\nmodal', 'Harmonic', 'Poincaré', 'Refine\nment']
enhancements_log = [3.52, 5.00, 3.00, 66.00, 43.43]
colors_mech = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

bars = ax2.bar(mechanisms, enhancements_log, color=colors_mech,
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bar, val in zip(bars, enhancements_log):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'$10^{{{val:.1f}}}$',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_ylabel('log$_{10}$(Enhancement)', fontweight='bold', fontsize=10)
ax2.set_title('Enhancement Factors', fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(enhancements_log) * 1.15)

# Add total line
total = sum(enhancements_log)
ax2.axhline(y=total, color='red', linestyle='--', linewidth=2,
           label=f'Total: $10^{{{total:.1f}}}$', alpha=0.7)
ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)

ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8)

# ============================================================================
# Panel 3: Trans-Planckian Resolution vs Frequency
# ============================================================================
ax3 = fig.add_subplot(143)

# Frequency range from molecular to Planck scale
frequencies = np.logspace(13, 43, 100)  # Hz
t_planck = 5.391e-44  # s
enhancement_total = 10**120.95

# Categorical temporal resolution
delta_t = t_planck / (enhancement_total * (frequencies / 1.855e43))

# Plot
ax3.loglog(frequencies, delta_t, linewidth=3, color='#2E86AB',
          label='Categorical Resolution')

# Reference lines
ax3.axhline(y=t_planck, color='red', linestyle='--', linewidth=2,
           label='Planck Time', alpha=0.7)
ax3.axhline(y=1e-18, color='orange', linestyle=':', linewidth=2,
           label='Attosecond Limit', alpha=0.7)

# Mark specific regimes
molecular = 5.13e13
electronic = 2.47e15
nuclear = 1.24e20

ax3.scatter([molecular], [t_planck / (enhancement_total * (molecular/1.855e43))],
           s=100, c='green', marker='o', edgecolors='black', linewidths=2, zorder=10)
ax3.scatter([electronic], [t_planck / (enhancement_total * (electronic/1.855e43))],
           s=100, c='blue', marker='s', edgecolors='black', linewidths=2, zorder=10)
ax3.scatter([nuclear], [t_planck / (enhancement_total * (nuclear/1.855e43))],
           s=100, c='purple', marker='^', edgecolors='black', linewidths=2, zorder=10)

ax3.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=10)
ax3.set_ylabel('Temporal Resolution (s)', fontweight='bold', fontsize=10)
ax3.set_title('Trans-Planckian Resolution', fontweight='bold', pad=10)
ax3.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax3.grid(True, alpha=0.3, which='both', linestyle='--')
ax3.tick_params(labelsize=8)

# ============================================================================
# Panel 4: Thermodynamic State Space (Temperature vs Entropy)
# ============================================================================
ax4 = fig.add_subplot(144)

# Temperature range
T = np.logspace(-15, 3, 200)  # K
k_B = 1.380649e-23  # J/K

# Different numbers of oscillators
M_values = [10, 100, 1000, 10000]
colors_thermo = ['#E63946', '#457B9D', '#1D3557', '#F77F00']
labels_M = ['M=10', 'M=100', 'M=1000', 'M=10⁴']

for M, color, label in zip(M_values, colors_thermo, labels_M):
    # Entropy from partition counting: S = k_B * M * ln(n)
    # where n depends on temperature (Boltzmann distribution)
    # For simplicity: n_eff ≈ T/T_0 for T > T_0
    T_0 = 1  # K (reference)
    n_eff = np.maximum(1, T/T_0)
    S = k_B * M * np.log(n_eff)

    ax4.loglog(T, S, linewidth=2.5, color=color, label=label, alpha=0.8)

# Mark special temperatures
T_room = 300
T_cmb = 2.7
T_absolute = 1e-15

for T_mark, name, y_offset in [(T_room, 'Room', 1.5),
                                (T_cmb, 'CMB', 1.5),
                                (1e-9, 'nK', 1.5)]:
    ax4.axvline(x=T_mark, color='gray', linestyle=':', alpha=0.5, linewidth=1)

ax4.set_xlabel('Temperature (K)', fontweight='bold', fontsize=10)
ax4.set_ylabel('Entropy (J/K)', fontweight='bold', fontsize=10)
ax4.set_title('Categorical Thermodynamics', fontweight='bold', pad=10)
ax4.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax4.grid(True, alpha=0.3, which='both', linestyle='--')
ax4.tick_params(labelsize=8)

# Highlight regions
ax4.axvspan(1e-15, 1e-9, alpha=0.1, color='blue', label='_nolegend_')
ax4.axvspan(1, 1000, alpha=0.1, color='red', label='_nolegend_')

# ============================================================================
# Overall layout
# ============================================================================
plt.tight_layout(pad=1.5)

# Save figure
plt.savefig('figV4_thermodynamics.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved as: figV4_thermodynamics.png")

plt.show()
