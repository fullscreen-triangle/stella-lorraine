"""
Viscous Transport Visualization
2D cross-section networks, molecular vibrations, viscosity temperature dependence, surface potential
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
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
# Panel: Viscous Transport Characteristics
#==============================================================================

def draw_2d_cross_section_network(ax):
    """2D cross-section network chart showing molecular connectivity"""
    ax.set_title('2D Cross-Section: Molecular Network', fontsize=10, color='#00ffcc')
    
    # Create hexagonal lattice-like network for liquid
    np.random.seed(42)
    n_nodes = 40
    
    # Generate positions with some disorder (liquid-like)
    theta = np.linspace(0, 2*np.pi, 7)[:-1]
    positions = {}
    node_id = 0
    
    # Central node
    positions[node_id] = (5, 5)
    node_id += 1
    
    # Rings of nodes with increasing disorder
    for ring in range(1, 5):
        n_in_ring = 6 * ring
        for i in range(n_in_ring):
            angle = 2 * np.pi * i / n_in_ring + np.random.uniform(-0.2, 0.2)
            r = ring * 1.2 + np.random.uniform(-0.2, 0.2)
            x = 5 + r * np.cos(angle)
            y = 5 + r * np.sin(angle)
            if 0 < x < 10 and 0 < y < 10:
                positions[node_id] = (x, y)
                node_id += 1
    
    # Create graph and add edges based on distance
    G = nx.Graph()
    for i in positions:
        G.add_node(i, pos=positions[i])
    
    for i in positions:
        for j in positions:
            if i < j:
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
                if dist < 1.8:  # Neighbor threshold
                    G.add_edge(i, j, weight=1.0/dist)
    
    # Draw network
    pos = nx.get_node_attributes(G, 'pos')
    
    # Edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#00ffcc', 
                          width=[w*2 for w in weights], alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#00ffcc', 
                          node_size=100, alpha=0.8)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('x (nm)', fontsize=8)
    ax.set_ylabel('y (nm)', fontsize=8)
    ax.set_aspect('equal')

def draw_molecular_vibration_mapper(ax):
    """Molecular vibration mapper showing different vibrational modes"""
    ax.set_title('Molecular Vibration Mapper', fontsize=10, color='#ff6600')
    
    t = np.linspace(0, 5, 500)
    
    # Different molecules with characteristic vibrations
    molecules = {
        'H₂O (bend)': {'freq': 1.6, 'amp': 1.0, 'color': '#00aaff'},
        'H₂O (stretch)': {'freq': 3.4, 'amp': 0.8, 'color': '#0066ff'},
        'CO₂ (asymm)': {'freq': 2.3, 'amp': 0.9, 'color': '#ff6600'},
        'CH₄ (C-H)': {'freq': 3.0, 'amp': 0.7, 'color': '#00ff00'},
    }
    
    offset = 0
    for name, props in molecules.items():
        # Damped oscillation (representing energy exchange)
        damping = 0.3
        vibration = props['amp'] * np.exp(-damping * t) * np.sin(2 * np.pi * props['freq'] * t)
        ax.plot(t, vibration + offset, color=props['color'], linewidth=1.5, label=name)
        offset += 2.5
    
    ax.set_xlabel('Time (ps)', fontsize=8)
    ax.set_ylabel('Displacement (arb. units)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, 5)

def draw_viscosity_temperature(ax):
    """Temperature dependence of viscosity for different materials"""
    ax.set_title('Viscosity vs Temperature', fontsize=10, color='#ff00ff')
    
    T = np.linspace(200, 600, 100)
    
    # Viscosity data (Arrhenius-like for liquids)
    materials = {
        'Water': {'eta_0': 1e-3, 'E_a': 2000, 'color': '#00aaff'},
        'Glycerol': {'eta_0': 1.5, 'E_a': 5000, 'color': '#ff6600'},
        'Honey': {'eta_0': 5.0, 'E_a': 6000, 'color': '#ffcc00'},
        'Engine Oil': {'eta_0': 0.1, 'E_a': 3500, 'color': '#00ff00'},
    }
    
    R = 8.314  # Gas constant
    
    for name, props in materials.items():
        # Arrhenius equation for viscosity
        eta = props['eta_0'] * np.exp(props['E_a'] / (R * T))
        ax.semilogy(T, eta * 1000, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Viscosity (mPa·s)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(200, 600)
    ax.grid(True, alpha=0.3)

def draw_surface_potential(ax):
    """Surface potential of water - wave formation potential"""
    ax.set_title('Surface Wave Potential (Water)', fontsize=10, color='#00ffff')
    
    # Create meshgrid
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Surface tension potential landscape
    # Capillary waves with different wavelengths
    gamma = 0.072  # Surface tension of water (N/m)
    rho = 1000  # Density (kg/m³)
    
    # Superposition of wave modes
    k1, k2 = 2, 5  # Wave numbers
    omega1 = np.sqrt(gamma * k1**3 / rho)  # Dispersion relation
    omega2 = np.sqrt(gamma * k2**3 / rho)
    
    # Potential energy landscape
    Z = 0.5 * np.sin(k1 * X) * np.cos(k1 * Y) + 0.3 * np.sin(k2 * X + k2 * Y)
    
    # Plot as contour with colorbar
    contour = ax.contourf(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.8)
    ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    plt.colorbar(contour, ax=ax, label='Surface potential (mJ/m²)')
    
    ax.set_xlabel('x (mm)', fontsize=8)
    ax.set_ylabel('y (mm)', fontsize=8)
    ax.set_aspect('equal')

# Create Panel
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Viscous Transport: Molecular Dynamics', fontsize=14, color='white', y=0.98)

draw_2d_cross_section_network(axes[0, 0])
draw_molecular_vibration_mapper(axes[0, 1])
draw_viscosity_temperature(axes[1, 0])
draw_surface_potential(axes[1, 1])

plt.tight_layout()
fig.savefig('figures/panel_viscous_transport.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0a', edgecolor='none')
plt.close(fig)

#==============================================================================
# Save data
#==============================================================================

viscosity_data = {
    'materials': {
        'Water': {
            'eta_0_Pa_s': 1e-3,
            'activation_energy_J_mol': 2000,
            'surface_tension_N_m': 0.072
        },
        'Glycerol': {
            'eta_0_Pa_s': 1.5,
            'activation_energy_J_mol': 5000,
            'surface_tension_N_m': 0.063
        },
        'Honey': {
            'eta_0_Pa_s': 5.0,
            'activation_energy_J_mol': 6000,
            'surface_tension_N_m': 0.055
        },
        'Engine_Oil': {
            'eta_0_Pa_s': 0.1,
            'activation_energy_J_mol': 3500,
            'surface_tension_N_m': 0.030
        }
    },
    'molecular_vibrations': {
        'H2O_bend_THz': 1.6,
        'H2O_stretch_THz': 3.4,
        'CO2_asymmetric_THz': 2.3,
        'CH4_stretch_THz': 3.0
    },
    'partition_parameters': {
        'collision_aperture_selectivity': 0.15,
        'partition_lag_ps': 0.5,
        'categorical_potential_kT': 3.2
    }
}

with open('data/viscous_transport_data.json', 'w') as f:
    json.dump(viscosity_data, f, indent=2)

print("Generated: figures/panel_viscous_transport.png")
print("Generated: data/viscous_transport_data.json")
