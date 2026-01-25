"""
Flux Visualization: Transport through apertures and flow types
Generates panel charts showing carrier transport through different aperture types
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
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
# Panel 1: Carriers through different aperture types
#==============================================================================

def draw_lattice_scattering(ax):
    """Electrons through lattice scattering apertures"""
    ax.set_title('Electrons Through Lattice Apertures', fontsize=10, color='#00ffff')
    
    # Draw lattice atoms
    lattice_x = np.arange(0, 10, 1.5)
    lattice_y = np.arange(0, 8, 1.5)
    
    for x in lattice_x:
        for y in lattice_y:
            # Atoms with thermal vibration offset
            offset_x = 0.1 * np.sin(2 * np.pi * (x + y) / 3)
            offset_y = 0.1 * np.cos(2 * np.pi * (x - y) / 3)
            circle = Circle((x + offset_x, y + offset_y), 0.3, 
                           facecolor='#4444ff', edgecolor='#8888ff', alpha=0.7)
            ax.add_patch(circle)
    
    # Draw electron path through apertures
    electron_path_x = np.linspace(0, 9, 50)
    electron_path_y = 4 + 0.8 * np.sin(electron_path_x * 1.5) + 0.3 * np.random.randn(50)
    ax.plot(electron_path_x, electron_path_y, 'c-', linewidth=2, alpha=0.8)
    
    # Electron positions
    for i in range(0, 50, 8):
        ax.scatter(electron_path_x[i], electron_path_y[i], c='#00ffff', s=80, 
                  marker='o', edgecolors='white', zorder=5)
    
    # Aperture indicators
    for x in [2, 5, 8]:
        ax.axvline(x, color='#ffff00', alpha=0.3, linestyle='--')
    
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 8)
    ax.set_xlabel('Position (lattice units)', fontsize=8)
    ax.set_ylabel('Transverse displacement', fontsize=8)
    ax.set_aspect('equal')

def draw_phonon_mode_matching(ax):
    """Phonons through mode-matching apertures"""
    ax.set_title('Phonons Through Mode-Matching Apertures', fontsize=10, color='#ff6600')
    
    # Frequency space
    omega = np.linspace(0, 10, 200)
    
    # Source phonon spectrum
    source_spectrum = np.exp(-0.5 * ((omega - 3) / 1.5)**2) + 0.5 * np.exp(-0.5 * ((omega - 7) / 1)**2)
    
    # Aperture transmission function (mode matching)
    aperture_transmission = 0.8 * np.exp(-0.5 * ((omega - 3.5) / 2)**2) + 0.6 * np.exp(-0.5 * ((omega - 8) / 0.8)**2)
    
    # Transmitted spectrum
    transmitted = source_spectrum * aperture_transmission
    
    ax.fill_between(omega, source_spectrum, alpha=0.3, color='#ff6600', label='Source')
    ax.fill_between(omega, aperture_transmission, alpha=0.3, color='#00ff00', label='Aperture')
    ax.plot(omega, transmitted, 'w-', linewidth=2, label='Transmitted')
    
    ax.set_xlabel('Frequency ω (THz)', fontsize=8)
    ax.set_ylabel('Spectral density', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, 10)

def draw_viscous_collision(ax):
    """Molecules through collision apertures - Viscous fluid"""
    ax.set_title('Viscous Fluid: Collision Apertures', fontsize=10, color='#00ff88')
    
    # Dense molecular arrangement
    np.random.seed(42)
    n_molecules = 60
    x = np.random.uniform(0, 10, n_molecules)
    y = np.random.uniform(0, 8, n_molecules)
    
    # Velocities (correlated for viscous flow)
    base_vx = 0.3 * (y / 8)  # Shear flow
    vx = base_vx + 0.05 * np.random.randn(n_molecules)
    vy = 0.05 * np.random.randn(n_molecules)
    
    # Draw molecules
    for i in range(n_molecules):
        circle = Circle((x[i], y[i]), 0.25, 
                        facecolor='#00ff88', edgecolor='white', alpha=0.7)
        ax.add_patch(circle)
        ax.arrow(x[i], y[i], vx[i]*3, vy[i]*3, head_width=0.1, 
                head_length=0.05, fc='yellow', ec='yellow', alpha=0.7)
    
    # Collision aperture visualization
    ax.axhline(4, color='#ff0000', alpha=0.5, linestyle=':', label='Shear plane')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_xlabel('x (molecular diameters)', fontsize=8)
    ax.set_ylabel('y (molecular diameters)', fontsize=8)
    ax.set_aspect('equal')

def draw_ideal_gas_collision(ax):
    """Molecules through collision apertures - Ideal gas"""
    ax.set_title('Ideal Gas: Sparse Collision Apertures', fontsize=10, color='#ff00ff')
    
    # Sparse molecular arrangement
    np.random.seed(43)
    n_molecules = 25
    x = np.random.uniform(0, 10, n_molecules)
    y = np.random.uniform(0, 8, n_molecules)
    
    # Random velocities (uncorrelated)
    speeds = np.random.exponential(0.5, n_molecules)
    angles = np.random.uniform(0, 2*np.pi, n_molecules)
    vx = speeds * np.cos(angles)
    vy = speeds * np.sin(angles)
    
    # Draw molecules and trajectories
    for i in range(n_molecules):
        circle = Circle((x[i], y[i]), 0.2, 
                        facecolor='#ff00ff', edgecolor='white', alpha=0.8)
        ax.add_patch(circle)
        ax.arrow(x[i], y[i], vx[i]*2, vy[i]*2, head_width=0.12, 
                head_length=0.06, fc='cyan', ec='cyan', alpha=0.8)
    
    # Mean free path indicator
    mfp = 2.5  # Mean free path
    ax.annotate('', xy=(5 + mfp, 4), xytext=(5, 4),
                arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
    ax.text(5 + mfp/2, 4.3, 'λ (MFP)', fontsize=8, ha='center', color='yellow')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_xlabel('x (molecular diameters)', fontsize=8)
    ax.set_ylabel('y (molecular diameters)', fontsize=8)
    ax.set_aspect('equal')

# Create Panel 1
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle('Transport Through Apertures: Carrier Types', fontsize=14, color='white', y=0.98)

draw_lattice_scattering(axes1[0, 0])
draw_phonon_mode_matching(axes1[0, 1])
draw_viscous_collision(axes1[1, 0])
draw_ideal_gas_collision(axes1[1, 1])

plt.tight_layout()
fig1.savefig('figures/panel_aperture_carriers.png', dpi=150, bbox_inches='tight', 
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig1)

#==============================================================================
# Panel 2: Four fundamental flows
#==============================================================================

def draw_gas_vibrations(ax):
    """Gas molecular vibrations"""
    ax.set_title('Gas Molecular Vibrations', fontsize=10, color='#ffcc00')
    
    t = np.linspace(0, 10, 500)
    
    # Multiple molecules with different frequencies
    frequencies = [1.2, 1.8, 2.5, 3.2]
    amplitudes = [1.0, 0.8, 0.6, 0.4]
    colors = ['#ff6666', '#66ff66', '#6666ff', '#ffff66']
    
    for i, (f, a, c) in enumerate(zip(frequencies, amplitudes, colors)):
        # 3D-like vibration projected to 2D
        x = a * np.sin(2 * np.pi * f * t)
        y = a * np.cos(2 * np.pi * f * t + np.pi/4)
        displacement = np.sqrt(x**2 + y**2)
        ax.plot(t, displacement + i*2.5, color=c, linewidth=1.5, 
               label=f'ω = {f:.1f} THz')
    
    ax.set_xlabel('Time (ps)', fontsize=8)
    ax.set_ylabel('Displacement amplitude', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, 10)

def draw_current_flow(ax):
    """Electrical current flow - Newton's cradle propagation"""
    ax.set_title('Current Flow: Newton\'s Cradle', fontsize=10, color='#00ffff')
    
    # Wire representation
    n_electrons = 20
    x_positions = np.linspace(0, 10, n_electrons)
    
    # Time evolution of displacement wave
    times = [0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.cool(np.linspace(0, 1, len(times)))
    
    for t, c in zip(times, colors):
        # Signal propagates as wave
        displacement = 0.3 * np.exp(-((x_positions - 2 - t*8)**2) / 2)
        y = 1 + displacement
        ax.plot(x_positions, y + t*0.8, 'o-', color=c, markersize=8, 
               linewidth=2, alpha=0.8, label=f't = {t:.2f} ns')
    
    ax.arrow(1, 0.3, 8, 0, head_width=0.15, head_length=0.3, 
            fc='yellow', ec='yellow')
    ax.text(5, 0.1, 'Signal propagation →', fontsize=8, ha='center', color='yellow')
    
    ax.set_xlabel('Position along wire (nm)', fontsize=8)
    ax.set_ylabel('Electron displacement', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(-0.5, 11)

def draw_heat_flow(ax):
    """Heat flow visualization"""
    ax.set_title('Heat Flow: Phonon Cascade', fontsize=10, color='#ff4400')
    
    x = np.linspace(0, 10, 100)
    
    # Temperature profile evolving in time
    times = [0, 0.5, 1.0, 2.0, 5.0]
    T_hot, T_cold = 400, 300
    L = 10
    alpha = 1.5  # Thermal diffusivity
    
    for t in times:
        if t == 0:
            T = np.where(x < L/2, T_hot, T_cold)
        else:
            # Diffusion solution
            T = (T_hot + T_cold)/2 + (T_hot - T_cold)/2 * np.exp(-alpha * t) * np.cos(np.pi * x / L)
        
        ax.plot(x, T, linewidth=2, label=f't = {t:.1f} s', alpha=0.8)
    
    ax.fill_between([0, 2], [280, 280], [420, 420], alpha=0.2, color='red', label='Hot')
    ax.fill_between([8, 10], [280, 280], [420, 420], alpha=0.2, color='blue', label='Cold')
    
    ax.set_xlabel('Position (cm)', fontsize=8)
    ax.set_ylabel('Temperature (K)', fontsize=8)
    ax.legend(loc='right', fontsize=7)
    ax.set_xlim(0, 10)
    ax.set_ylim(280, 420)

def draw_mass_flow(ax):
    """Mass flow / diffusion"""
    ax.set_title('Mass Flow: Diffusive Transport', fontsize=10, color='#00ff00')
    
    x = np.linspace(0, 10, 100)
    
    # Concentration profile evolving (Fick's law)
    times = [0, 0.1, 0.5, 1.0, 3.0]
    D = 1.0  # Diffusivity
    
    for t in times:
        if t == 0:
            C = np.where(x < 3, 1.0, 0.0)
        else:
            # Error function solution
            from scipy.special import erfc
            C = 0.5 * erfc((x - 3) / (2 * np.sqrt(D * t)))
        
        ax.plot(x, C, linewidth=2, label=f't = {t:.1f} s', alpha=0.8)
    
    ax.axvline(3, color='white', linestyle='--', alpha=0.3, label='Initial boundary')
    
    ax.set_xlabel('Position (mm)', fontsize=8)
    ax.set_ylabel('Concentration (normalized)', fontsize=8)
    ax.legend(loc='right', fontsize=7)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 1.1)

# Create Panel 2
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Fundamental Transport Flows', fontsize=14, color='white', y=0.98)

draw_gas_vibrations(axes2[0, 0])
draw_current_flow(axes2[0, 1])
draw_heat_flow(axes2[1, 0])
draw_mass_flow(axes2[1, 1])

plt.tight_layout()
fig2.savefig('figures/panel_fundamental_flows.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig2)

#==============================================================================
# Save data
#==============================================================================

# Aperture data
aperture_data = {
    'electron_lattice': {
        'description': 'Electrons through lattice scattering apertures',
        'selectivity': 0.3,  # Fraction passing without scattering
        'partition_lag_ps': 0.1,  # 100 femtoseconds
        'categorical_potential_meV': 25.0
    },
    'phonon_modes': {
        'description': 'Phonons through mode-matching apertures',
        'acoustic_selectivity': 0.8,
        'optical_selectivity': 0.2,
        'umklapp_rate_THz': 0.5
    },
    'viscous_fluid': {
        'description': 'Molecules through collision apertures (viscous)',
        'collision_frequency_THz': 10.0,
        'mean_free_path_nm': 0.5,
        'selectivity': 0.1
    },
    'ideal_gas': {
        'description': 'Molecules through collision apertures (ideal gas)',
        'collision_frequency_GHz': 5.0,
        'mean_free_path_nm': 100.0,
        'selectivity': 0.9
    }
}

flow_data = {
    'gas_vibrations': {
        'frequencies_THz': [1.2, 1.8, 2.5, 3.2],
        'amplitudes_angstrom': [0.1, 0.08, 0.06, 0.04]
    },
    'current_flow': {
        'signal_velocity_m_s': 2e8,
        'drift_velocity_m_s': 1e-4,
        'electron_density_m3': 8.5e28
    },
    'heat_flow': {
        'T_hot_K': 400,
        'T_cold_K': 300,
        'thermal_diffusivity_m2_s': 1.5e-5
    },
    'mass_flow': {
        'diffusivity_m2_s': 1e-9,
        'initial_concentration': 1.0
    }
}

with open('data/aperture_transport_data.json', 'w') as f:
    json.dump(aperture_data, f, indent=2)

with open('data/flow_transport_data.json', 'w') as f:
    json.dump(flow_data, f, indent=2)

print("Generated: figures/panel_aperture_carriers.png")
print("Generated: figures/panel_fundamental_flows.png")
print("Generated: data/aperture_transport_data.json")
print("Generated: data/flow_transport_data.json")
