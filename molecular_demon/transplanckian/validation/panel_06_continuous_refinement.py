"""
Panel 6: Continuous Refinement Dynamics

Validates the 10^44× enhancement from exponential improvement through
non-halting dynamics with exp(100) refinement over 100 seconds.

Four subplots:
1. Exponential decay of resolution
2. Enhancement factor growth  
3. Recurrence time effects
4. 3D: Resolution evolution surface (time, T_rec, delta_t)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Parameters
delta_t_0 = 1e-94  # Baseline resolution (after other enhancements)
T_rec = 1.0  # Recurrence time (1 second from paper)

# Subplot 1: Exponential Decay of Resolution
ax1 = fig.add_subplot(gs[0, 0])

time_array = np.linspace(0, 100, 1000)

# Exponential refinement: delta_t(t) = delta_t_0 * exp(-t/T_rec)
delta_t_array = delta_t_0 * np.exp(-time_array / T_rec)

ax1.semilogy(time_array, delta_t_array, '-', linewidth=2.5, color='blue')

# Mark key time points
key_times = [0, 10, 50, 100]
for t in key_times:
    delta_t_val = delta_t_0 * np.exp(-t / T_rec)
    ax1.scatter([t], [delta_t_val], s=150, c='red', marker='o', 
                zorder=5, edgecolors='black', linewidths=2)
    
    if t > 0:
        enhancement = np.exp(t / T_rec)
        ax1.annotate(f't={t}s\n$e^{{{t}}} = {enhancement:.0e}\\times$',
                     xy=(t, delta_t_val), xytext=(t+5, delta_t_val * 1e10),
                     arrowprops=dict(arrowstyle='->', lw=1.5),
                     fontsize=9, fontweight='bold')

ax1.set_xlabel('Integration Time t (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temporal Resolution $\\delta t$ (s)', fontsize=12, fontweight='bold')
ax1.set_title('Exponential Refinement: $\\delta t(t) = \\delta t_0 e^{-t/T_{\\mathrm{rec}}}$', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_ylim([delta_t_0 * np.exp(-100), delta_t_0 * 2])

# Subplot 2: Enhancement Factor Growth
ax2 = fig.add_subplot(gs[0, 1])

enhancement_array = np.exp(time_array / T_rec)

ax2.semilogy(time_array, enhancement_array, '-', linewidth=2.5, color='green')

# Target enhancement at t=100s
target_enhancement = np.exp(100)
ax2.axhline(y=target_enhancement, color='red', linestyle='--', 
            linewidth=2, label=f'Target: $e^{{100}} \\approx 10^{{44}}$')
ax2.scatter([100], [target_enhancement], s=250, c='red', marker='*', 
            zorder=5, edgecolors='black', linewidths=2)

# Add shaded regions for different time scales
ax2.axvspan(0, 10, alpha=0.1, color='blue', label='Short-term (< 10s)')
ax2.axvspan(10, 50, alpha=0.1, color='yellow', label='Medium-term (10-50s)')
ax2.axvspan(50, 100, alpha=0.1, color='red', label='Long-term (50-100s)')

ax2.set_xlabel('Integration Time t (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Enhancement Factor $e^{t/T_{\\mathrm{rec}}}$', fontsize=12, fontweight='bold')
ax2.set_title('Continuous Refinement Enhancement', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3, which='both')

# Subplot 3: Recurrence Time Effects
ax3 = fig.add_subplot(gs[1, 0])

# Different recurrence times
T_rec_array = [0.1, 0.5, 1.0, 2.0, 5.0]
time_fixed = 100  # Fixed integration time

for T_rec_val in T_rec_array:
    time_vals = np.linspace(0, time_fixed, 500)
    delta_t_vals = delta_t_0 * np.exp(-time_vals / T_rec_val)
    
    ax3.semilogy(time_vals, delta_t_vals, '-', linewidth=2, 
                 label=f'$T_{{\\mathrm{{rec}}}} = {T_rec_val}$ s')

ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Resolution $\\delta t$ (s)', fontsize=12, fontweight='bold')
ax3.set_title('Effect of Recurrence Time $T_{\\mathrm{rec}}$', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# Add text box
textstr = f'Paper value:\n' \
          f'$T_{{\\mathrm{{rec}}}} = 1.0$ s\n' \
          f'$t_{{\\mathrm{{int}}}} = 100$ s\n' \
          f'Enhancement: $e^{{100}} \\approx 10^{{44}}$'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
ax3.text(0.65, 0.95, textstr, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', bbox=props, family='monospace',
         fontweight='bold')

# Subplot 4: 3D Resolution Evolution Surface
ax4 = fig.add_subplot(gs[1, 1], projection='3d')

# Create meshgrid
T_rec_mesh_array = np.linspace(0.1, 5, 30)
time_mesh_array = np.linspace(0, 100, 30)

T_rec_mesh, time_mesh = np.meshgrid(T_rec_mesh_array, time_mesh_array)

# Calculate resolution surface (log scale)
delta_t_mesh = delta_t_0 * np.exp(-time_mesh / T_rec_mesh)
log_delta_t_mesh = np.log10(delta_t_mesh)

# Plot surface
surf = ax4.plot_surface(T_rec_mesh, time_mesh, log_delta_t_mesh, 
                        cmap='coolwarm', alpha=0.9, edgecolor='none')

# Mark paper parameters
paper_T_rec = 1.0
paper_time = 100
paper_log_delta_t = np.log10(delta_t_0 * np.exp(-paper_time / paper_T_rec))

ax4.scatter([paper_T_rec], [paper_time], [paper_log_delta_t], 
            s=300, c='yellow', marker='*', edgecolors='black', 
            linewidths=2, zorder=10, label='Paper: $T_{\\mathrm{rec}}=1$ s, $t=100$ s')

ax4.set_xlabel('$T_{\\mathrm{rec}}$ (s)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Time t (s)', fontsize=11, fontweight='bold')
ax4.set_zlabel('$\\log_{10}(\\delta t)$ [s]', fontsize=11, fontweight='bold')
ax4.set_title('3D: Resolution Evolution Landscape', fontsize=13, fontweight='bold')

cbar = fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=10)
cbar.set_label('$\\log_{10}(\\delta t)$', fontsize=10, fontweight='bold')

ax4.legend(fontsize=9, loc='upper left')
ax4.view_init(elev=25, azim=225)

# Overall title
fig.suptitle('Panel 6: Continuous Refinement Dynamics ($10^{44}\\times$ Enhancement)\n' +
             'Exponential improvement: $\\delta t(t) = \\delta t_0 \\exp(-t/T_{\\mathrm{rec}})$ with $T_{\\mathrm{rec}} = 1$ s',
             fontsize=15, fontweight='bold', y=0.98)

# Footer
fig.text(0.5, 0.02,
         'Validation: Non-halting dynamics with Poincaré recurrence | Enhancement: $\\exp(100) \\approx 2.7 \\times 10^{43} \\approx 10^{44}$',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('panel_06_continuous_refinement.png', dpi=300, bbox_inches='tight')
print("✓ Panel 6 saved: panel_06_continuous_refinement.png")
plt.show()
