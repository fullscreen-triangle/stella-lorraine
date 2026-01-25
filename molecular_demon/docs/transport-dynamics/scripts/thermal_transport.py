"""
Thermal Transport Visualization
Multiple panels for phonon dynamics, thermal properties, and mode analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import json
import os

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Set style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['figure.facecolor'] = '#0a0a0a'

#==============================================================================
# Panel 1: Vibrational Field Analysis
#==============================================================================

def draw_vibrational_field(ax):
    """Vibrational field mapper under different heat conditions"""
    ax.set_title('Vibrational Field: Heat Conditions', fontsize=10, color='#ff4400')
    
    # Create lattice
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 8, 16)
    X, Y = np.meshgrid(x, y)
    
    # Temperature gradient (left hot, right cold)
    T = 400 - 20 * X  # Temperature field
    
    # Vibration amplitude proportional to sqrt(T)
    amplitude = 0.15 * np.sqrt(T / 300)
    
    # Random phase for each atom
    np.random.seed(42)
    phase = np.random.uniform(0, 2*np.pi, X.shape)
    
    # Displacement at this moment
    t = 0.5
    omega = 5  # Frequency
    U = amplitude * np.sin(omega * t + phase)
    V = amplitude * np.cos(omega * t + phase + np.pi/4)
    
    # Plot as quiver
    colors = plt.cm.hot(T / T.max())
    ax.quiver(X, Y, U, V, T, cmap='hot', alpha=0.8, scale=8)
    
    # Add temperature colorbar indication
    ax.text(0.5, 7.5, 'HOT', fontsize=10, color='red', fontweight='bold')
    ax.text(9, 7.5, 'COLD', fontsize=10, color='blue', fontweight='bold')
    
    ax.set_xlabel('x (lattice units)', fontsize=8)
    ax.set_ylabel('y (lattice units)', fontsize=8)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)

def draw_amplitude_temperature(ax):
    """Vibration amplitude vs temperature"""
    ax.set_title('Vibration Amplitude vs Temperature', fontsize=10, color='#ffcc00')
    
    T = np.linspace(10, 500, 100)
    
    # Debye model: <u²> ∝ T at high T, ∝ T² at low T
    theta_D = 350  # Debye temperature (K)
    
    # Classical regime (high T)
    u2_classical = T / theta_D
    
    # Quantum corrections at low T
    x = theta_D / T
    # Approximate Debye function
    u2_quantum = np.where(T > theta_D/3, 
                          T / theta_D,
                          (T / theta_D)**2 * 3)
    
    ax.plot(T, np.sqrt(u2_classical) * 0.1, '--', color='#888888', 
           linewidth=2, label='Classical (∝√T)')
    ax.plot(T, np.sqrt(u2_quantum) * 0.1, '-', color='#ffcc00', 
           linewidth=2, label='With quantum effects')
    
    ax.axvline(theta_D, color='#00ffff', linestyle=':', alpha=0.5, label=f'Θ_D = {theta_D} K')
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('RMS amplitude (Å)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.set_xlim(0, 500)
    ax.grid(True, alpha=0.3)

def draw_phonon_surface(ax):
    """Phonon dispersion surface (gradient visualization)"""
    ax.set_title('Phonon Dispersion Surface', fontsize=10, color='#00ff88')
    ax.remove()
    ax = plt.subplot(2, 2, 3, projection='3d')
    ax.set_facecolor('#0a0a0a')
    
    # Wavevector space
    kx = np.linspace(-np.pi, np.pi, 50)
    ky = np.linspace(-np.pi, np.pi, 50)
    KX, KY = np.meshgrid(kx, ky)
    
    # Simple 2D phonon dispersion (acoustic branch)
    a = 1  # Lattice constant
    omega_max = 10  # Max frequency (THz)
    
    # Acoustic phonon dispersion
    omega_acoustic = omega_max * np.sqrt(np.sin(KX*a/2)**2 + np.sin(KY*a/2)**2)
    
    # Plot surface
    surf = ax.plot_surface(KX, KY, omega_acoustic, cmap='viridis', 
                           alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('kₓ', fontsize=8, labelpad=-2)
    ax.set_ylabel('kᵧ', fontsize=8, labelpad=-2)
    ax.set_zlabel('ω (THz)', fontsize=8, labelpad=-2)
    ax.set_title('Phonon Dispersion Surface', fontsize=10, color='#00ff88', pad=-5)
    
    return ax

def draw_force_network(ax):
    """Network chart showing interatomic forces"""
    ax.set_title('Interatomic Force Network', fontsize=10, color='#ff00ff')
    
    # 2D lattice with force connections
    np.random.seed(44)
    nx_atoms, ny_atoms = 6, 5
    
    G = nx.Graph()
    positions = {}
    
    # Create regular lattice positions with thermal displacement
    T = 300  # Temperature (K)
    amplitude = 0.08  # Thermal amplitude
    
    node_id = 0
    for i in range(nx_atoms):
        for j in range(ny_atoms):
            x = i * 1.5 + amplitude * np.random.randn()
            y = j * 1.5 + amplitude * np.random.randn()
            positions[node_id] = (x, y)
            G.add_node(node_id, pos=(x, y))
            node_id += 1
    
    # Add edges (springs) with force based on displacement
    k_spring = 10  # Spring constant
    for i in positions:
        for j in positions:
            if i < j:
                xi, yi = positions[i]
                xj, yj = positions[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < 2.0:  # Neighbor threshold
                    # Force = k * (r - r0)
                    r0 = 1.5
                    force = abs(k_spring * (dist - r0))
                    G.add_edge(i, j, force=force)
    
    # Draw network
    pos = nx.get_node_attributes(G, 'pos')
    edges = G.edges()
    forces = [G[u][v]['force'] for u, v in edges]
    max_force = max(forces) if forces else 1
    
    # Color edges by force
    edge_colors = [plt.cm.plasma(f/max_force) for f in forces]
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                          width=2, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#ff00ff', 
                          node_size=150, alpha=0.9)
    
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 7)
    ax.set_xlabel('x (Å)', fontsize=8)
    ax.set_ylabel('y (Å)', fontsize=8)
    ax.set_aspect('equal')

# Create Panel 1
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle('Thermal Transport: Vibrational Dynamics', fontsize=14, color='white', y=0.98)

draw_vibrational_field(axes1[0, 0])
draw_amplitude_temperature(axes1[0, 1])

# Special handling for 3D plot
axes1[1, 0].remove()
ax3d = fig1.add_subplot(2, 2, 3, projection='3d')
ax3d.set_facecolor('#0a0a0a')
kx = np.linspace(-np.pi, np.pi, 50)
ky = np.linspace(-np.pi, np.pi, 50)
KX, KY = np.meshgrid(kx, ky)
omega_acoustic = 10 * np.sqrt(np.sin(KX/2)**2 + np.sin(KY/2)**2)
ax3d.plot_surface(KX, KY, omega_acoustic, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('kₓ', fontsize=8)
ax3d.set_ylabel('kᵧ', fontsize=8)
ax3d.set_zlabel('ω (THz)', fontsize=8)
ax3d.set_title('Phonon Dispersion Surface', fontsize=10, color='#00ff88')

draw_force_network(axes1[1, 1])

plt.tight_layout()
fig1.savefig('figures/panel_thermal_vibrational.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig1)

#==============================================================================
# Panel 2: Thermal Properties
#==============================================================================

def draw_thermal_conductivity(ax):
    """Thermal conductivity vs temperature for different materials"""
    ax.set_title('Thermal Conductivity vs Temperature', fontsize=10, color='#00ffff')
    
    T = np.linspace(50, 800, 100)
    
    materials = {
        'Copper': {'k_300': 400, 'type': 'metal', 'color': '#ff6600'},
        'Aluminum': {'k_300': 237, 'type': 'metal', 'color': '#cccccc'},
        'Silicon': {'k_300': 150, 'type': 'semiconductor', 'color': '#00ff00'},
        'Diamond': {'k_300': 2000, 'type': 'insulator', 'color': '#00ffff'},
        'Glass': {'k_300': 1.0, 'type': 'amorphous', 'color': '#ff00ff'},
    }
    
    for name, props in materials.items():
        if props['type'] == 'metal':
            # Wiedemann-Franz: κ ≈ const at high T
            k = props['k_300'] * (300 / T)**0.1
        elif props['type'] == 'semiconductor':
            # Peak at low T, 1/T at high T
            k = props['k_300'] * (300 / T)**1.2
        elif props['type'] == 'insulator':
            # Strong 1/T dependence
            k = props['k_300'] * (300 / T)**1.5
        else:  # amorphous
            # Weak T dependence
            k = props['k_300'] * (T / 300)**0.3
        
        ax.loglog(T, k, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

def draw_thermal_diffusivity_heatmap(ax):
    """Thermal diffusivity heatmap"""
    ax.set_title('Thermal Diffusivity (α = κ/ρcₚ)', fontsize=10, color='#ff6600')
    
    # Materials vs Temperature
    materials = ['Cu', 'Al', 'Fe', 'Si', 'SiO₂', 'H₂O']
    temperatures = np.array([100, 200, 300, 400, 500, 600])
    
    # Diffusivity data (m²/s × 10⁶)
    alpha = np.array([
        [150, 130, 117, 105, 95, 88],   # Cu
        [110, 100, 97, 93, 88, 82],     # Al
        [25, 22, 18, 15, 13, 11],       # Fe
        [120, 80, 60, 45, 35, 28],      # Si
        [0.9, 0.85, 0.83, 0.82, 0.81, 0.80],  # SiO2
        [0.14, 0.14, 0.14, 0.0, 0.0, 0.0],   # H2O (liquid only)
    ])
    
    im = ax.imshow(alpha, cmap='inferno', aspect='auto')
    ax.set_xticks(range(len(temperatures)))
    ax.set_xticklabels(temperatures)
    ax.set_yticks(range(len(materials)))
    ax.set_yticklabels(materials)
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Material', fontsize=8)
    plt.colorbar(im, ax=ax, label='α (mm²/s)')

def draw_thermal_effusivity_heatmap(ax):
    """Thermal effusivity heatmap"""
    ax.set_title('Thermal Effusivity (e = √(κρcₚ))', fontsize=10, color='#00ff88')
    
    materials = ['Cu', 'Al', 'Fe', 'Si', 'SiO₂', 'H₂O']
    temperatures = np.array([100, 200, 300, 400, 500, 600])
    
    # Effusivity data (W·s^0.5/(m²·K) × 10⁻³)
    e = np.array([
        [38, 37, 36, 35, 34, 33],   # Cu
        [24, 23, 22, 21, 20, 19],   # Al
        [16, 15, 14, 13, 12, 11],   # Fe
        [18, 15, 12, 10, 8, 7],     # Si
        [1.5, 1.4, 1.4, 1.3, 1.3, 1.2],  # SiO2
        [1.6, 1.6, 1.6, 0.0, 0.0, 0.0],  # H2O
    ])
    
    im = ax.imshow(e, cmap='plasma', aspect='auto')
    ax.set_xticks(range(len(temperatures)))
    ax.set_xticklabels(temperatures)
    ax.set_yticks(range(len(materials)))
    ax.set_yticklabels(materials)
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Material', fontsize=8)
    plt.colorbar(im, ax=ax, label='e (kW·s⁰·⁵/m²·K)')

def draw_thermal_inertia_heatmap(ax):
    """Thermal inertia heatmap"""
    ax.set_title('Thermal Inertia (I = ρcₚ)', fontsize=10, color='#ff00ff')
    
    materials = ['Cu', 'Al', 'Fe', 'Si', 'SiO₂', 'H₂O']
    temperatures = np.array([100, 200, 300, 400, 500, 600])
    
    # Thermal inertia data (J/(m³·K) × 10⁻⁶)
    I = np.array([
        [3.4, 3.4, 3.4, 3.5, 3.5, 3.6],   # Cu
        [2.4, 2.4, 2.4, 2.5, 2.5, 2.6],   # Al
        [3.5, 3.6, 3.7, 3.8, 3.9, 4.0],   # Fe
        [1.6, 1.7, 1.7, 1.8, 1.8, 1.9],   # Si
        [1.6, 1.7, 1.7, 1.7, 1.8, 1.8],   # SiO2
        [4.2, 4.2, 4.2, 0.0, 0.0, 0.0],   # H2O
    ])
    
    im = ax.imshow(I, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(temperatures)))
    ax.set_xticklabels(temperatures)
    ax.set_yticks(range(len(materials)))
    ax.set_yticklabels(materials)
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Material', fontsize=8)
    plt.colorbar(im, ax=ax, label='I (MJ/m³·K)')

# Create Panel 2
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Thermal Transport: Material Properties', fontsize=14, color='white', y=0.98)

draw_thermal_conductivity(axes2[0, 0])
draw_thermal_diffusivity_heatmap(axes2[0, 1])
draw_thermal_effusivity_heatmap(axes2[1, 0])
draw_thermal_inertia_heatmap(axes2[1, 1])

plt.tight_layout()
fig2.savefig('figures/panel_thermal_properties.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig2)

#==============================================================================
# Panel 3: Phonon Analysis
#==============================================================================

def draw_mode_matching_network(ax):
    """Network chart for phonon mode matching"""
    ax.set_title('Phonon Mode-Matching Network', fontsize=10, color='#ffcc00')
    
    # Create network where nodes are phonon modes
    G = nx.Graph()
    
    # Add acoustic and optical modes
    modes = {
        'LA': (0, 2),    # Longitudinal acoustic
        'TA1': (1, 1),   # Transverse acoustic 1
        'TA2': (1, 3),   # Transverse acoustic 2
        'LO': (3, 2),    # Longitudinal optical
        'TO1': (4, 1),   # Transverse optical 1
        'TO2': (4, 3),   # Transverse optical 2
    }
    
    for mode, pos in modes.items():
        G.add_node(mode, pos=pos)
    
    # Mode coupling (scattering channels)
    couplings = [
        ('LA', 'TA1', 0.8), ('LA', 'TA2', 0.8),
        ('TA1', 'TA2', 0.5),
        ('LA', 'LO', 0.3), ('TA1', 'TO1', 0.3), ('TA2', 'TO2', 0.3),
        ('LO', 'TO1', 0.6), ('LO', 'TO2', 0.6), ('TO1', 'TO2', 0.4),
    ]
    
    for m1, m2, strength in couplings:
        G.add_edge(m1, m2, weight=strength)
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw with edge thickness proportional to coupling
    edges = G.edges()
    weights = [G[u][v]['weight'] * 3 for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, ax=ax, width=weights, 
                          edge_color='#ffcc00', alpha=0.7)
    
    # Color nodes by type (acoustic vs optical)
    acoustic = ['LA', 'TA1', 'TA2']
    optical = ['LO', 'TO1', 'TO2']
    
    nx.draw_networkx_nodes(G, pos, nodelist=acoustic, ax=ax, 
                          node_color='#00ff00', node_size=500, label='Acoustic')
    nx.draw_networkx_nodes(G, pos, nodelist=optical, ax=ax, 
                          node_color='#ff6600', node_size=500, label='Optical')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white')
    
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(0, 4)
    ax.axis('off')

def draw_umklapp_constraints(ax):
    """Umklapp scattering constraints"""
    ax.set_title('Umklapp Scattering in k-Space', fontsize=10, color='#ff4400')
    
    # Brillouin zone boundaries
    bz = 1  # Brillouin zone edge (normalized)
    
    # Draw Brillouin zone
    square = plt.Rectangle((-bz, -bz), 2*bz, 2*bz, fill=False, 
                           edgecolor='#00ffff', linewidth=2)
    ax.add_patch(square)
    
    # Normal scattering: k1 + k2 = k3
    k1 = np.array([0.3, 0.2])
    k2 = np.array([0.2, 0.4])
    k3_normal = k1 + k2
    
    ax.arrow(0, 0, k1[0], k1[1], head_width=0.05, head_length=0.03, 
            fc='#00ff00', ec='#00ff00', label='k₁')
    ax.arrow(k1[0], k1[1], k2[0], k2[1], head_width=0.05, head_length=0.03, 
            fc='#00ff00', ec='#00ff00')
    ax.arrow(0, 0, k3_normal[0], k3_normal[1], head_width=0.05, head_length=0.03, 
            fc='yellow', ec='yellow', label='k₃ (Normal)')
    
    # Umklapp scattering: k1 + k2 = k3 + G
    k1_u = np.array([0.6, 0.5])
    k2_u = np.array([0.5, 0.6])
    k3_outside = k1_u + k2_u  # Outside BZ
    G = np.array([2, 0])  # Reciprocal lattice vector
    k3_umklapp = k3_outside - G  # Folded back
    
    ax.arrow(-0.5, -0.5, k1_u[0], k1_u[1], head_width=0.05, head_length=0.03, 
            fc='#ff6600', ec='#ff6600')
    ax.arrow(-0.5+k1_u[0], -0.5+k1_u[1], k2_u[0], k2_u[1], head_width=0.05, 
            head_length=0.03, fc='#ff6600', ec='#ff6600')
    ax.arrow(-0.5, -0.5, k3_umklapp[0], k3_umklapp[1], head_width=0.05, 
            head_length=0.03, fc='red', ec='red', label='k₃ (Umklapp)')
    
    # G vector
    ax.annotate('', xy=(1.5, 0), xytext=(1.1, 0.6),
                arrowprops=dict(arrowstyle='->', color='#00ffff', lw=2))
    ax.text(1.3, 0.3, 'G', fontsize=10, color='#00ffff')
    
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('kₓ (π/a)', fontsize=8)
    ax.set_ylabel('kᵧ (π/a)', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.text(0, -1.3, 'Normal: k₁+k₂=k₃', fontsize=8, color='#00ff00')
    ax.text(0, -1.5, 'Umklapp: k₁+k₂=k₃+G', fontsize=8, color='#ff6600')

def draw_interface_scattering(ax):
    """Interface scattering visualization"""
    ax.set_title('Phonon Interface Scattering', fontsize=10, color='#00ff88')
    
    # Two materials with different properties
    x = np.linspace(0, 10, 200)
    interface = 5
    
    # Incident, reflected, transmitted phonons
    k_inc = 2
    k_trans = 1.5  # Different due to impedance mismatch
    
    # Incident wave
    y_inc = np.where(x < interface, np.sin(k_inc * x), 0)
    
    # Reflected wave
    R = 0.3  # Reflection coefficient
    y_ref = np.where(x < interface, R * np.sin(-k_inc * x + 2*k_inc*interface), 0)
    
    # Transmitted wave
    T = 0.7  # Transmission coefficient
    y_trans = np.where(x >= interface, T * np.sin(k_trans * (x - interface)), 0)
    
    ax.fill_between([0, interface], [-1.5, -1.5], [1.5, 1.5], 
                    alpha=0.2, color='#00ff00', label='Material 1')
    ax.fill_between([interface, 10], [-1.5, -1.5], [1.5, 1.5], 
                    alpha=0.2, color='#ff6600', label='Material 2')
    
    ax.plot(x, y_inc, 'g-', linewidth=2, label='Incident')
    ax.plot(x, y_ref, 'r--', linewidth=2, label='Reflected')
    ax.plot(x, y_trans, 'b-', linewidth=2, label='Transmitted')
    
    ax.axvline(interface, color='white', linestyle='-', linewidth=2)
    ax.text(interface, 1.6, 'Interface', fontsize=8, ha='center', color='white')
    
    ax.set_xlabel('Position (nm)', fontsize=8)
    ax.set_ylabel('Displacement', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.8)

def draw_phonon_geometry(ax):
    """Geometric properties of phonon - D3-like visualization"""
    ax.set_title('Phonon Wavepacket Geometry', fontsize=10, color='#ff00ff')
    ax.remove()
    ax = plt.subplot(2, 2, 4, projection='3d')
    ax.set_facecolor('#0a0a0a')
    
    # Phonon wavepacket
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian envelope with oscillation
    k = 2  # Wave vector
    sigma = 1.5  # Envelope width
    Z = np.exp(-(X**2 + Y**2)/(2*sigma**2)) * np.cos(k * X)
    
    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('Amplitude', fontsize=8)
    ax.set_title('Phonon Wavepacket', fontsize=10, color='#ff00ff')
    
    return ax

# Create Panel 3
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
fig3.suptitle('Phonon Transport: Mode Analysis', fontsize=14, color='white', y=0.98)

draw_mode_matching_network(axes3[0, 0])
draw_umklapp_constraints(axes3[0, 1])
draw_interface_scattering(axes3[1, 0])

# Special handling for 3D plot
axes3[1, 1].remove()
ax3d = fig3.add_subplot(2, 2, 4, projection='3d')
ax3d.set_facecolor('#0a0a0a')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)/4.5) * np.cos(2 * X)
ax3d.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax3d.set_xlabel('x', fontsize=8)
ax3d.set_ylabel('y', fontsize=8)
ax3d.set_zlabel('Amplitude', fontsize=8)
ax3d.set_title('Phonon Wavepacket', fontsize=10, color='#ff00ff')

plt.tight_layout()
fig3.savefig('figures/panel_phonon_analysis.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig3)

#==============================================================================
# Save data
#==============================================================================

thermal_data = {
    'materials': {
        'Copper': {
            'thermal_conductivity_W_mK': 400,
            'thermal_diffusivity_mm2_s': 117,
            'Debye_temperature_K': 343
        },
        'Silicon': {
            'thermal_conductivity_W_mK': 150,
            'thermal_diffusivity_mm2_s': 60,
            'Debye_temperature_K': 645
        },
        'Diamond': {
            'thermal_conductivity_W_mK': 2000,
            'thermal_diffusivity_mm2_s': 500,
            'Debye_temperature_K': 2230
        }
    },
    'phonon_modes': {
        'acoustic': ['LA', 'TA1', 'TA2'],
        'optical': ['LO', 'TO1', 'TO2'],
        'mode_coupling_strengths': {
            'LA-TA': 0.8,
            'LA-LO': 0.3,
            'LO-TO': 0.6
        }
    },
    'scattering_mechanisms': {
        'normal': 'k1 + k2 = k3',
        'umklapp': 'k1 + k2 = k3 + G',
        'boundary': 'lambda = L (sample size)',
        'impurity': 'tau^-1 ∝ omega^4'
    }
}

with open('data/thermal_transport_data.json', 'w') as f:
    json.dump(thermal_data, f, indent=2)

print("Generated: figures/panel_thermal_vibrational.png")
print("Generated: figures/panel_thermal_properties.png")
print("Generated: figures/panel_phonon_analysis.png")
print("Generated: data/thermal_transport_data.json")
