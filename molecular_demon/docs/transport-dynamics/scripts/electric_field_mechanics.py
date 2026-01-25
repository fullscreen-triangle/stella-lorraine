"""
Electric Field Mechanics Visualization
Electric and magnetic field vectors, electron trajectories, potential landscapes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
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
# Panel 1: Electromagnetic Field Mechanics
#==============================================================================

def draw_electric_field(ax):
    """Electric field vectors"""
    ax.set_title('Electric Field Configuration', fontsize=10, color='#00ffff')
    
    # Create grid
    x = np.linspace(-5, 5, 15)
    y = np.linspace(-4, 4, 12)
    X, Y = np.meshgrid(x, y)
    
    # Point charge electric field
    q1_pos = (-2, 0)
    q2_pos = (2, 0)
    q1, q2 = 1, -1  # Dipole
    
    # Electric field from two charges
    r1 = np.sqrt((X - q1_pos[0])**2 + (Y - q1_pos[1])**2 + 0.1)
    r2 = np.sqrt((X - q2_pos[0])**2 + (Y - q2_pos[1])**2 + 0.1)
    
    Ex = q1 * (X - q1_pos[0]) / r1**3 + q2 * (X - q2_pos[0]) / r2**3
    Ey = q1 * (Y - q1_pos[1]) / r1**3 + q2 * (Y - q2_pos[1]) / r2**3
    
    # Normalize for visualization
    E_mag = np.sqrt(Ex**2 + Ey**2)
    Ex_norm = Ex / (E_mag + 0.01)
    Ey_norm = Ey / (E_mag + 0.01)
    
    # Plot field vectors
    ax.quiver(X, Y, Ex_norm, Ey_norm, E_mag, cmap='plasma', alpha=0.8)
    
    # Plot charges
    ax.scatter(*q1_pos, c='red', s=200, marker='+', linewidths=3, label='+q')
    ax.scatter(*q2_pos, c='blue', s=200, marker='_', linewidths=3, label='-q')
    
    # Field lines
    for start_y in np.linspace(-3, 3, 7):
        # Start from positive charge
        x_line, y_line = [q1_pos[0]], [q1_pos[1] + 0.5 * np.sign(start_y)]
        for _ in range(100):
            r1 = np.sqrt((x_line[-1] - q1_pos[0])**2 + (y_line[-1] - q1_pos[1])**2 + 0.1)
            r2 = np.sqrt((x_line[-1] - q2_pos[0])**2 + (y_line[-1] - q2_pos[1])**2 + 0.1)
            
            ex = q1 * (x_line[-1] - q1_pos[0]) / r1**3 + q2 * (x_line[-1] - q2_pos[0]) / r2**3
            ey = q1 * (y_line[-1] - q1_pos[1]) / r1**3 + q2 * (y_line[-1] - q2_pos[1]) / r2**3
            
            e_mag = np.sqrt(ex**2 + ey**2) + 0.01
            dx = ex / e_mag * 0.2
            dy = ey / e_mag * 0.2
            
            x_line.append(x_line[-1] + dx)
            y_line.append(y_line[-1] + dy)
            
            if abs(x_line[-1]) > 5 or abs(y_line[-1]) > 4:
                break
        
        ax.plot(x_line, y_line, 'c-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('x (nm)', fontsize=8)
    ax.set_ylabel('y (nm)', fontsize=8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=7)

def draw_magnetic_field(ax):
    """Magnetic field vectors"""
    ax.set_title('Magnetic Field (Wire Cross-Section)', fontsize=10, color='#ff6600')
    
    # Create grid
    x = np.linspace(-5, 5, 15)
    y = np.linspace(-5, 5, 15)
    X, Y = np.meshgrid(x, y)
    
    # Wire at origin, current into page
    I = 1  # Current
    
    # Magnetic field from infinite wire
    r = np.sqrt(X**2 + Y**2) + 0.5
    
    # B = μ₀I/(2πr) in φ direction
    Bx = -I * Y / (2 * np.pi * r**2)
    By = I * X / (2 * np.pi * r**2)
    
    # Normalize
    B_mag = np.sqrt(Bx**2 + By**2)
    Bx_norm = Bx / (B_mag + 0.01)
    By_norm = By / (B_mag + 0.01)
    
    # Plot field vectors
    ax.quiver(X, Y, Bx_norm, By_norm, B_mag, cmap='cool', alpha=0.8)
    
    # Current symbol
    ax.scatter(0, 0, c='yellow', s=300, marker='o', edgecolors='white', linewidths=2)
    ax.scatter(0, 0, c='yellow', s=100, marker='x', linewidths=2)
    ax.text(0.5, 0.5, 'I ⊗', fontsize=10, color='yellow')
    
    # Circular field lines
    for r_circle in [1.5, 2.5, 3.5]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_circle * np.cos(theta), r_circle * np.sin(theta), 
               'orange', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('x (nm)', fontsize=8)
    ax.set_ylabel('y (nm)', fontsize=8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

def draw_electron_trajectories(ax):
    """3D electron trajectories in electromagnetic field"""
    ax.remove()
    ax = plt.subplot(2, 3, 3, projection='3d')
    ax.set_facecolor('#0a0a0a')
    ax.set_title('Electron Trajectories in EM Field', fontsize=10, color='#00ff88')
    
    # Electromagnetic field parameters
    E = np.array([0, 0, 0.1])  # Electric field in z
    B = np.array([0, 0, 1])    # Magnetic field in z
    
    # Electron parameters
    q = -1  # Charge
    m = 1   # Mass (normalized)
    
    # Initial conditions for multiple electrons
    trajectories = []
    colors = ['#00ffff', '#ff6600', '#00ff00', '#ff00ff']
    
    for i, (vx0, vy0) in enumerate([(1, 0), (0, 1), (1, 1), (0.5, 1.5)]):
        t = np.linspace(0, 20, 1000)
        dt = t[1] - t[0]
        
        x, y, z = [0], [0], [i * 0.5]
        vx, vy, vz = vx0, vy0, 0
        
        for _ in range(len(t) - 1):
            # Lorentz force: F = q(E + v × B)
            v = np.array([vx, vy, vz])
            F = q * (E + np.cross(v, B))
            
            # Update velocity and position
            ax_acc, ay, az = F / m
            vx += ax_acc * dt
            vy += ay * dt
            vz += az * dt
            
            x.append(x[-1] + vx * dt)
            y.append(y[-1] + vy * dt)
            z.append(z[-1] + vz * dt)
        
        ax.plot(x, y, z, color=colors[i], linewidth=1.5, alpha=0.8)
        trajectories.append({'x': x, 'y': y, 'z': z})
    
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('z', fontsize=8)
    
    return ax, trajectories

def draw_newton_cradle_wave(ax):
    """Wave function for electron movement showing resistance"""
    ax.set_title('Newton\'s Cradle: Resistance as Damping', fontsize=10, color='#ffcc00')
    
    x = np.linspace(0, 20, 200)
    
    # Different resistance levels
    resistances = [0.0, 0.05, 0.15, 0.3]
    colors = ['#00ff00', '#ffff00', '#ff6600', '#ff0000']
    labels = ['Superconductor (R=0)', 'Low R', 'Medium R', 'High R']
    
    for R, c, label in zip(resistances, colors, labels):
        # Damped wave propagation
        k = 2  # Wave number
        omega = 5  # Angular frequency
        t = 2  # Snapshot time
        
        # Wave with exponential decay (resistance)
        psi = np.exp(-R * x) * np.sin(k * x - omega * t)
        
        ax.plot(x, psi, color=c, linewidth=2, label=label, alpha=0.8)
    
    ax.set_xlabel('Position along wire (nm)', fontsize=8)
    ax.set_ylabel('Wave amplitude', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, 20)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

def draw_potential_landscape(ax):
    """Potential energy landscape surface plot"""
    ax.remove()
    ax = plt.subplot(2, 3, 5, projection='3d')
    ax.set_facecolor('#0a0a0a')
    ax.set_title('Potential Energy Landscape', fontsize=10, color='#ff00ff')
    
    # Create meshgrid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Periodic potential with barriers (resistance)
    a = 1.5  # Lattice constant
    V0 = 1   # Barrier height
    
    # Sinusoidal potential (simplified crystal)
    V = V0 * (np.cos(2 * np.pi * X / a)**2 + np.cos(2 * np.pi * Y / a)**2)
    
    # Add random defects
    np.random.seed(42)
    for _ in range(5):
        xd, yd = np.random.uniform(-4, 4, 2)
        V += 0.5 * np.exp(-((X - xd)**2 + (Y - yd)**2) / 0.5)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, V, cmap='inferno', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('x (Å)', fontsize=8)
    ax.set_ylabel('y (Å)', fontsize=8)
    ax.set_zlabel('V (eV)', fontsize=8)
    
    return ax

def draw_material_resistance(ax):
    """Different materials and their electrical properties"""
    ax.set_title('Material Resistance Comparison', fontsize=10, color='#00ffff')
    
    T = np.linspace(10, 500, 100)
    
    materials = {
        'Copper': {'rho_0': 1.68e-8, 'alpha': 0.0039, 'color': '#ff6600'},
        'Aluminum': {'rho_0': 2.65e-8, 'alpha': 0.0043, 'color': '#cccccc'},
        'Tungsten': {'rho_0': 5.6e-8, 'alpha': 0.0045, 'color': '#ffcc00'},
        'Nichrome': {'rho_0': 1.1e-6, 'alpha': 0.0004, 'color': '#00ff00'},
        'Germanium': {'rho_0': 0.46, 'alpha': -0.05, 'color': '#ff00ff'},  # Semiconductor
    }
    
    for name, props in materials.items():
        # Linear approximation for metals, exponential for semiconductors
        if props['alpha'] > 0:
            rho = props['rho_0'] * (1 + props['alpha'] * (T - 293))
        else:
            rho = props['rho_0'] * np.exp(-3000 / T)  # Semiconductor
        
        ax.semilogy(T, rho * 1e8, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Resistivity (μΩ·cm)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(10, 500)

# Create Panel 1
fig1 = plt.figure(figsize=(16, 12))
fig1.suptitle('Electromagnetic Field Mechanics', fontsize=14, color='white', y=0.98)

ax1 = fig1.add_subplot(2, 3, 1)
draw_electric_field(ax1)

ax2 = fig1.add_subplot(2, 3, 2)
draw_magnetic_field(ax2)

ax3 = fig1.add_subplot(2, 3, 3, projection='3d')
ax3.set_facecolor('#0a0a0a')
# Electron trajectories
E = np.array([0, 0, 0.1])
B = np.array([0, 0, 1])
colors = ['#00ffff', '#ff6600', '#00ff00', '#ff00ff']
trajectory_data = []
for i, (vx0, vy0) in enumerate([(1, 0), (0, 1), (1, 1), (0.5, 1.5)]):
    t = np.linspace(0, 20, 500)
    dt = t[1] - t[0]
    x, y, z = [0], [0], [i * 0.5]
    vx, vy, vz = vx0, vy0, 0
    for _ in range(len(t) - 1):
        v = np.array([vx, vy, vz])
        F = -1 * (E + np.cross(v, B))
        vx += F[0] * dt
        vy += F[1] * dt
        vz += F[2] * dt
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
        z.append(z[-1] + vz * dt)
    ax3.plot(x, y, z, color=colors[i], linewidth=1.5, alpha=0.8)
    trajectory_data.append({'x': x[-100:], 'y': y[-100:], 'z': z[-100:]})
ax3.set_xlabel('x', fontsize=8)
ax3.set_ylabel('y', fontsize=8)
ax3.set_zlabel('z', fontsize=8)
ax3.set_title('Electron Trajectories', fontsize=10, color='#00ff88')

ax4 = fig1.add_subplot(2, 3, 4)
draw_newton_cradle_wave(ax4)

ax5 = fig1.add_subplot(2, 3, 5, projection='3d')
ax5.set_facecolor('#0a0a0a')
x = np.linspace(-5, 5, 80)
y = np.linspace(-5, 5, 80)
X, Y = np.meshgrid(x, y)
V = np.cos(2 * np.pi * X / 1.5)**2 + np.cos(2 * np.pi * Y / 1.5)**2
ax5.plot_surface(X, Y, V, cmap='inferno', alpha=0.8)
ax5.set_xlabel('x (Å)', fontsize=8)
ax5.set_ylabel('y (Å)', fontsize=8)
ax5.set_zlabel('V (eV)', fontsize=8)
ax5.set_title('Potential Landscape', fontsize=10, color='#ff00ff')

ax6 = fig1.add_subplot(2, 3, 6)
draw_material_resistance(ax6)

plt.tight_layout()
fig1.savefig('figures/panel_electric_field_mechanics.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig1)

#==============================================================================
# Panel 2: Material Properties
#==============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Material Electrical Properties', fontsize=14, color='white', y=0.98)

# Band gap comparison
ax = axes2[0, 0]
ax.set_title('Band Gap vs Resistivity', fontsize=10, color='#00ffff')

materials_band = {
    'Metals': {'gap': 0, 'rho': 1e-8, 'color': '#ff6600', 'examples': 'Cu, Al, Au'},
    'Semimetals': {'gap': 0.01, 'rho': 1e-5, 'color': '#ffcc00', 'examples': 'Bi, Sb'},
    'Semiconductors': {'gap': 1.0, 'rho': 1e-2, 'color': '#00ff00', 'examples': 'Si, Ge'},
    'Insulators': {'gap': 5.0, 'rho': 1e12, 'color': '#ff00ff', 'examples': 'SiO₂, Diamond'},
}

for name, props in materials_band.items():
    ax.scatter(props['gap'], props['rho'], c=props['color'], s=200, label=name)
    ax.annotate(props['examples'], (props['gap'], props['rho']), 
               textcoords="offset points", xytext=(10, 5), fontsize=7, color=props['color'])

ax.set_xlabel('Band Gap (eV)', fontsize=8)
ax.set_ylabel('Resistivity (Ω·m)', fontsize=8)
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=7)
ax.grid(True, alpha=0.3)

# Fermi surface
ax = axes2[0, 1]
ax.set_title('Fermi Surface (2D slice)', fontsize=10, color='#ff6600')

theta = np.linspace(0, 2*np.pi, 100)
# Different Fermi surface shapes
r_free = 1 + 0 * theta  # Free electron (circle)
r_bcc = 1 + 0.2 * np.cos(4 * theta)  # BCC metal
r_fcc = 1 + 0.15 * np.cos(6 * theta)  # FCC metal

ax.plot(r_free * np.cos(theta), r_free * np.sin(theta), 
       'c-', linewidth=2, label='Free electron')
ax.plot(r_bcc * np.cos(theta), r_bcc * np.sin(theta), 
       'orange', linewidth=2, label='BCC metal')
ax.plot(r_fcc * np.cos(theta), r_fcc * np.sin(theta), 
       'lime', linewidth=2, label='FCC metal')

ax.set_xlabel('kₓ (π/a)', fontsize=8)
ax.set_ylabel('kᵧ (π/a)', fontsize=8)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=7)
ax.grid(True, alpha=0.3)

# Mean free path
ax = axes2[1, 0]
ax.set_title('Mean Free Path vs Temperature', fontsize=10, color='#00ff88')

T = np.linspace(10, 500, 100)

materials_mfp = {
    'Copper (pure)': {'l_0': 40, 'T_char': 100, 'color': '#ff6600'},
    'Copper (alloy)': {'l_0': 10, 'T_char': 200, 'color': '#ff8844'},
    'Aluminum': {'l_0': 15, 'T_char': 150, 'color': '#cccccc'},
    'Gold': {'l_0': 35, 'T_char': 120, 'color': '#ffcc00'},
}

for name, props in materials_mfp.items():
    # MFP decreases with T due to phonon scattering
    mfp = props['l_0'] * props['T_char'] / T
    mfp = np.clip(mfp, 1, None)  # Minimum MFP ~ lattice constant
    ax.semilogy(T, mfp, color=props['color'], linewidth=2, label=name)

ax.set_xlabel('Temperature (K)', fontsize=8)
ax.set_ylabel('Mean Free Path (nm)', fontsize=8)
ax.legend(loc='upper right', fontsize=7)
ax.grid(True, alpha=0.3)

# Mobility
ax = axes2[1, 1]
ax.set_title('Carrier Mobility', fontsize=10, color='#ff00ff')

materials_mob = {
    'Silicon (electrons)': 1400,
    'Silicon (holes)': 450,
    'GaAs (electrons)': 8500,
    'Germanium (electrons)': 3900,
    'InSb (electrons)': 78000,
}

names = list(materials_mob.keys())
values = list(materials_mob.values())
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))

ax.barh(names, values, color=colors)
ax.set_xlabel('Mobility (cm²/V·s)', fontsize=8)
ax.set_xscale('log')

plt.tight_layout()
fig2.savefig('figures/panel_material_properties.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig2)

#==============================================================================
# Save data
#==============================================================================

electric_data = {
    'field_configurations': {
        'dipole': {
            'q1': 1,
            'q2': -1,
            'separation_nm': 4
        }
    },
    'materials': {
        'Copper': {
            'resistivity_Ohm_m': 1.68e-8,
            'temp_coefficient_K-1': 0.0039,
            'mean_free_path_nm': 40,
            'Fermi_velocity_m_s': 1.57e6
        },
        'Silicon': {
            'resistivity_Ohm_m': 2300,
            'band_gap_eV': 1.12,
            'electron_mobility_cm2_V_s': 1400,
            'hole_mobility_cm2_V_s': 450
        }
    },
    'electron_trajectories': {
        'E_field': [0, 0, 0.1],
        'B_field': [0, 0, 1],
        'cyclotron_frequency': 'eB/m'
    },
    'partition_parameters': {
        'scattering_aperture_selectivity': 0.25,
        'partition_lag_fs': 10,
        'verification_gap_as': 3
    }
}

with open('data/electric_field_data.json', 'w') as f:
    json.dump(electric_data, f, indent=2)

print("Generated: figures/panel_electric_field_mechanics.png")
print("Generated: figures/panel_material_properties.png")
print("Generated: data/electric_field_data.json")
