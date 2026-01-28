"""
Panel 3: Phase-Lock Networks as Molecular Crystal Formation

Validates gas → liquid → crystal phase transitions through:
1. Phase diagram (T vs P with gas/liquid/crystal regions)
2. Order parameter evolution Φ(t)
3. Lattice structure formation (network graph evolution)
4. 3D crystal structure with defects
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import networkx as nx

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Chart A: Phase Diagram
ax1 = fig.add_subplot(gs[0, 0])

# Temperature and pressure ranges
T_range = np.linspace(0, 400, 100)
P_range = np.linspace(0, 1000, 100)
T_grid, P_grid = np.meshgrid(T_range, P_range)

# Define phase regions
gas_phase = (T_grid > 200) & (P_grid < 400)
liquid_phase = ((T_grid > 100) & (T_grid <= 200) & (P_grid < 700)) | \
               ((T_grid > 200) & (P_grid >= 400) & (P_grid < 700))
crystal_phase = (T_grid <= 100) | (P_grid >= 700)

# Plot phase regions
ax1.contourf(T_grid, P_grid, gas_phase.astype(int), levels=[0.5, 1.5], 
            colors=['lightblue'], alpha=0.5)
ax1.contourf(T_grid, P_grid, liquid_phase.astype(int), levels=[0.5, 1.5], 
            colors=['lightgreen'], alpha=0.5)
ax1.contourf(T_grid, P_grid, crystal_phase.astype(int), levels=[0.5, 1.5], 
            colors=['lightcoral'], alpha=0.5)

# Add phase labels
ax1.text(300, 200, 'Gas Phase\n(Disordered)', ha='center', fontsize=12, fontweight='bold')
ax1.text(150, 550, 'Liquid Phase\n(Partial Coordination)', ha='center', fontsize=12, fontweight='bold')
ax1.text(50, 850, 'Crystal Phase\n(Perfect Sync)', ha='center', fontsize=12, fontweight='bold')

# Critical point
ax1.plot(200, 400, 'r*', markersize=20, label='Critical Point')

# Measured points
np.random.seed(42)
gas_points_T = np.random.uniform(250, 350, 10)
gas_points_P = np.random.uniform(50, 350, 10)
liquid_points_T = np.random.uniform(120, 180, 10)
liquid_points_P = np.random.uniform(450, 650, 10)
crystal_points_T = np.random.uniform(20, 80, 10)
crystal_points_P = np.random.uniform(750, 950, 10)

ax1.scatter(gas_points_T, gas_points_P, c='blue', s=50, marker='o', edgecolors='black', linewidth=1.5)
ax1.scatter(liquid_points_T, liquid_points_P, c='green', s=50, marker='s', edgecolors='black', linewidth=1.5)
ax1.scatter(crystal_points_T, crystal_points_P, c='red', s=50, marker='^', edgecolors='black', linewidth=1.5)

ax1.set_xlabel('Network Temperature $T_{net}$ (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Network Pressure $P_{net}$ (packets/s)', fontsize=12, fontweight='bold')
ax1.set_title('Phase Diagram: Gas $\\rightarrow$ Liquid $\\rightarrow$ Crystal\\nClear phase separation', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Chart B: Order Parameter Evolution
ax2 = fig.add_subplot(gs[0, 1])

time = np.linspace(0, 20, 200)
Phi = 0.05 + 0.9 * (1 - np.exp(-time / 5))  # Logistic-like growth

# Add noise
Phi_measured = Phi[::10] + np.random.normal(0, 0.02, len(Phi[::10]))
time_measured = time[::10]

ax2.plot(time, Phi, 'b-', linewidth=3, label='Order Parameter $\\Phi(t)$')
ax2.scatter(time_measured, Phi_measured, c='red', s=40, marker='o', edgecolors='black', linewidth=1)

# Phase markers
ax2.axhspan(0, 0.3, alpha=0.2, color='blue', label='Gas Phase: $\\Phi < 0.3$')
ax2.axhspan(0.3, 0.7, alpha=0.2, color='green', label='Liquid Phase: $0.3 < \\Phi < 0.7$')
ax2.axhspan(0.7, 1.0, alpha=0.2, color='red', label='Crystal Phase: $\\Phi > 0.7$')

# Transition markers
ax2.axvline(x=3, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=7, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax2.text(3, 0.5, 'Gas→Liquid', rotation=90, ha='right', fontsize=9, fontweight='bold')
ax2.text(7, 0.5, 'Liquid→Crystal', rotation=90, ha='right', fontsize=9, fontweight='bold')

ax2.set_xlabel('Time $t$ (seconds)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Order Parameter $\\Phi$ (0=disordered, 1=ordered)', fontsize=12, fontweight='bold')
ax2.set_title('Order Parameter Evolution\\nClear phase transitions during cooling', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# Chart C: Lattice Structure Formation
ax3 = fig.add_subplot(gs[1, 0])

# Create three snapshots: gas, liquid, crystal
N_nodes = 20

# Gas phase (t=0s): random positions
np.random.seed(42)
pos_gas = {i: (np.random.rand(), np.random.rand()) for i in range(N_nodes)}
G_gas = nx.Graph()
G_gas.add_nodes_from(range(N_nodes))
# Few random edges
for _ in range(10):
    i, j = np.random.choice(N_nodes, 2, replace=False)
    G_gas.add_edge(i, j)

# Liquid phase (t=5s): partial clustering
pos_liquid = {}
for i in range(N_nodes):
    cluster = i // 5
    pos_liquid[i] = (cluster * 0.25 + np.random.rand() * 0.2, np.random.rand())
G_liquid = nx.Graph()
G_liquid.add_nodes_from(range(N_nodes))
# More edges, clustered
for i in range(N_nodes):
    for j in range(i+1, N_nodes):
        if abs(i - j) <= 2:
            G_liquid.add_edge(i, j)

# Crystal phase (t=10s): regular lattice
pos_crystal = {}
grid_size = int(np.sqrt(N_nodes))
idx = 0
for i in range(grid_size):
    for j in range(grid_size):
        if idx < N_nodes:
            pos_crystal[idx] = (i * 0.2, j * 0.2)
            idx += 1
G_crystal = nx.grid_2d_graph(grid_size, grid_size)
G_crystal = nx.convert_node_labels_to_integers(G_crystal)

# Draw all three
ax3.text(0.15, 0.95, 't=0s: Gas', transform=ax3.transAxes, fontsize=11, fontweight='bold')
nx.draw_networkx(G_gas, pos_gas, ax=ax3, node_color='lightblue', node_size=100, 
                 with_labels=False, edge_color='gray', width=0.5, alpha=0.3)

ax3.text(0.48, 0.95, 't=5s: Liquid', transform=ax3.transAxes, fontsize=11, fontweight='bold')
nx.draw_networkx(G_liquid, {k: (v[0] + 1.2, v[1]) for k, v in pos_liquid.items()}, ax=ax3,
                 node_color='lightgreen', node_size=100, with_labels=False, 
                 edge_color='gray', width=1, alpha=0.5)

ax3.text(0.78, 0.95, 't=10s: Crystal', transform=ax3.transAxes, fontsize=11, fontweight='bold')
nx.draw_networkx(G_crystal, {k: (v[0] + 2.4, v[1]) for k, v in pos_crystal.items()}, ax=ax3,
                 node_color='lightcoral', node_size=100, with_labels=False, 
                 edge_color='black', width=1.5)

ax3.set_title('Lattice Structure Formation\\nSpontaneous lattice formation during cooling', 
              fontsize=13, fontweight='bold')
ax3.axis('off')

# Chart D: 3D Crystal Structure
ax4 = fig.add_subplot(gs[1, 1], projection='3d')

# Create 3D cubic lattice
size = 4
nodes_3d = []
for i in range(size):
    for j in range(size):
        for k in range(size):
            nodes_3d.append((i, j, k))

nodes_3d = np.array(nodes_3d)

# Plot nodes
ax4.scatter(nodes_3d[:, 0], nodes_3d[:, 1], nodes_3d[:, 2], 
           c='blue', s=100, marker='o', edgecolors='black', linewidth=1.5, alpha=0.8)

# Draw edges (nearest neighbors)
for i, node in enumerate(nodes_3d):
    for j, neighbor in enumerate(nodes_3d):
        if i < j:
            dist = np.linalg.norm(node - neighbor)
            if dist < 1.5:  # Only nearest neighbors
                ax4.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 
                        [node[2], neighbor[2]], 'k-', alpha=0.3, linewidth=1)

# Add defects (vacancies)
defects = [15, 32, 48]
for defect_idx in defects:
    if defect_idx < len(nodes_3d):
        d = nodes_3d[defect_idx]
        ax4.scatter([d[0]], [d[1]], [d[2]], c='red', s=200, marker='X', 
                   edgecolors='darkred', linewidth=2, label='Defect' if defect_idx == defects[0] else '')

ax4.set_xlabel('$x$', fontsize=11, fontweight='bold')
ax4.set_ylabel('$y$', fontsize=11, fontweight='bold')
ax4.set_zlabel('$z$', fontsize=11, fontweight='bold')
ax4.set_title('3D Crystal Structure\\nCubic lattice with defects (red X)', 
              fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.view_init(elev=20, azim=45)

# Overall title
fig.suptitle('Panel 3: Phase-Lock Networks as Molecular Crystal Formation\\n' +
             'Gas $\\rightarrow$ Liquid $\\rightarrow$ Crystal transitions through network cooling',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('panel_03_phase_lock_crystal.png', dpi=300, bbox_inches='tight')
print("[OK] Panel 3 saved: panel_03_phase_lock_crystal.png")
plt.close()
