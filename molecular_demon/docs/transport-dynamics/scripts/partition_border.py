"""
Partition Border Visualization
Categorical potential, enthalpy, aperture selectivity, and partition lag
for electric, diffusive, thermal, and viscous transport
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
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

# Constants
kB = 1.380649e-23  # Boltzmann constant
T = 300  # Temperature (K)

#==============================================================================
# Panel 1: Categorical Potential
#==============================================================================

def categorical_potential(selectivity, T=300):
    """Calculate categorical potential from selectivity: Φ = -kB*T*ln(s)"""
    s = np.clip(selectivity, 1e-10, 1)
    return -kB * T * np.log(s) / (kB * T)  # In units of kB*T

def draw_categorical_potential_electric(ax):
    """Categorical potential for electrical transport"""
    ax.set_title('Electric: Categorical Potential', fontsize=10, color='#00ffff')
    
    # Selectivity for different scattering mechanisms
    mechanisms = {
        'Phonon (T=300K)': 0.3,
        'Phonon (T=100K)': 0.6,
        'Impurity': 0.4,
        'Boundary': 0.5,
        'Electron-electron': 0.7,
    }
    
    T_range = np.linspace(50, 500, 100)
    
    for name, s_base in mechanisms.items():
        # Temperature dependence of selectivity
        if 'Phonon' in name:
            s = s_base * (300 / T_range)**0.5
        elif 'Electron' in name:
            s = 1 - (1 - s_base) * (T_range / 300)**2
        else:
            s = s_base * np.ones_like(T_range)
        
        s = np.clip(s, 0.01, 0.99)
        phi = categorical_potential(s)
        
        ax.plot(T_range, phi, linewidth=2, label=name, alpha=0.8)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Φ / kB T', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 500)

def draw_categorical_potential_diffusive(ax):
    """Categorical potential for diffusive transport"""
    ax.set_title('Diffusive: Categorical Potential', fontsize=10, color='#00ff00')
    
    T_range = np.linspace(200, 800, 100)
    
    # Diffusion activation energies (in kB*T units at 300K)
    mechanisms = {
        'Vacancy diffusion': {'E_a': 1.5, 'color': '#00ff00'},
        'Interstitial': {'E_a': 0.8, 'color': '#66ff66'},
        'Grain boundary': {'E_a': 0.5, 'color': '#00cc00'},
        'Surface diffusion': {'E_a': 0.3, 'color': '#00aa00'},
    }
    
    for name, props in mechanisms.items():
        # Selectivity from Arrhenius: s = exp(-E_a / kB T)
        s = np.exp(-props['E_a'] * 300 / T_range)
        phi = categorical_potential(s)
        
        ax.plot(T_range, phi, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Φ / kB T', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_categorical_potential_thermal(ax):
    """Categorical potential for thermal transport"""
    ax.set_title('Thermal: Categorical Potential', fontsize=10, color='#ff6600')
    
    omega = np.linspace(0.1, 15, 100)  # Phonon frequency (THz)
    omega_D = 10  # Debye frequency
    
    # Mode-dependent selectivity
    modes = {
        'Acoustic (LA)': {'s0': 0.9, 'omega_c': 3, 'color': '#ff6600'},
        'Acoustic (TA)': {'s0': 0.85, 'omega_c': 2.5, 'color': '#ff8800'},
        'Optical': {'s0': 0.3, 'omega_c': 8, 'color': '#ffaa00'},
    }
    
    for name, props in modes.items():
        # Selectivity decreases at high frequency (Umklapp)
        s = props['s0'] * np.exp(-(omega / props['omega_c'])**2)
        phi = categorical_potential(s)
        
        ax.plot(omega, phi, color=props['color'], linewidth=2, label=name)
    
    ax.axvline(omega_D, color='white', linestyle='--', alpha=0.5, label='Debye frequency')
    ax.set_xlabel('Phonon frequency (THz)', fontsize=8)
    ax.set_ylabel('Φ / kB T', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_categorical_potential_viscous(ax):
    """Categorical potential for viscous transport"""
    ax.set_title('Viscous: Categorical Potential', fontsize=10, color='#ff00ff')
    
    shear_rate = np.logspace(-2, 4, 100)  # 1/s
    
    # Different fluids
    fluids = {
        'Water': {'s0': 0.6, 'gamma_c': 1e3, 'color': '#00aaff'},
        'Glycerol': {'s0': 0.2, 'gamma_c': 1e1, 'color': '#ff00ff'},
        'Polymer melt': {'s0': 0.1, 'gamma_c': 1e0, 'color': '#ff6666'},
        'Ideal gas': {'s0': 0.95, 'gamma_c': 1e6, 'color': '#66ff66'},
    }
    
    for name, props in fluids.items():
        # Shear thinning: selectivity increases at high shear
        s = props['s0'] + (1 - props['s0']) * (1 - np.exp(-shear_rate / props['gamma_c']))
        phi = categorical_potential(s)
        
        ax.semilogx(shear_rate, phi, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Shear rate (1/s)', fontsize=8)
    ax.set_ylabel('Φ / kB T', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

# Create Panel 1
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle('Categorical Potential: Transport Types', fontsize=14, color='white', y=0.98)

draw_categorical_potential_electric(axes1[0, 0])
draw_categorical_potential_diffusive(axes1[0, 1])
draw_categorical_potential_thermal(axes1[1, 0])
draw_categorical_potential_viscous(axes1[1, 1])

plt.tight_layout()
fig1.savefig('figures/panel_categorical_potential.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig1)

#==============================================================================
# Panel 2: Categorical Enthalpy
#==============================================================================

def draw_enthalpy_electric(ax):
    """Categorical enthalpy for electrical transport"""
    ax.set_title('Electric: Categorical Enthalpy', fontsize=10, color='#00ffff')
    
    # Enthalpy = U + Σ n_a Φ_a
    # For a wire: proportional to resistance
    
    T_range = np.linspace(10, 500, 100)
    
    materials = {
        'Copper': {'rho_0': 1.68e-8, 'alpha': 0.0039, 'color': '#ff6600'},
        'Aluminum': {'rho_0': 2.65e-8, 'alpha': 0.0043, 'color': '#cccccc'},
        'Tungsten': {'rho_0': 5.6e-8, 'alpha': 0.0045, 'color': '#ffcc00'},
    }
    
    for name, props in materials.items():
        rho = props['rho_0'] * (1 + props['alpha'] * (T_range - 293))
        # Enthalpy proportional to resistivity (aperture potentials)
        H = rho / props['rho_0']  # Normalized
        
        ax.plot(T_range, H, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('H / H₀ (normalized)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_enthalpy_diffusive(ax):
    """Categorical enthalpy for diffusive transport"""
    ax.set_title('Diffusive: Categorical Enthalpy', fontsize=10, color='#00ff00')
    
    T_range = np.linspace(300, 1200, 100)
    
    # Different diffusion processes
    processes = {
        'Bulk diffusion': {'H_0': 2.0, 'dH': 0.5, 'color': '#00ff00'},
        'Grain boundary': {'H_0': 0.8, 'dH': 0.3, 'color': '#00cc00'},
        'Surface': {'H_0': 0.4, 'dH': 0.15, 'color': '#00aa00'},
    }
    
    for name, props in processes.items():
        # Enthalpy includes activation barrier
        H = props['H_0'] + props['dH'] * (T_range - 300) / 300
        
        ax.plot(T_range, H, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('H (eV/atom)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_enthalpy_thermal(ax):
    """Categorical enthalpy for thermal transport"""
    ax.set_title('Thermal: Categorical Enthalpy', fontsize=10, color='#ff6600')
    
    T_range = np.linspace(10, 500, 100)
    
    # Phonon enthalpy (Debye model)
    theta_D = 350  # Debye temperature
    
    # Different materials
    materials = {
        'Diamond': {'theta_D': 2230, 'color': '#00ffff'},
        'Silicon': {'theta_D': 645, 'color': '#00ff00'},
        'Copper': {'theta_D': 343, 'color': '#ff6600'},
        'Lead': {'theta_D': 105, 'color': '#888888'},
    }
    
    for name, props in materials.items():
        x = props['theta_D'] / T_range
        # Debye enthalpy approximation
        H = np.where(x < 1, 
                    3 * T_range / props['theta_D'],  # High T limit
                    3 * np.exp(-x))  # Low T limit
        H = H / H.max()  # Normalize
        
        ax.plot(T_range, H, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('H / H_max', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_enthalpy_viscous(ax):
    """Categorical enthalpy for viscous transport"""
    ax.set_title('Viscous: Categorical Enthalpy', fontsize=10, color='#ff00ff')
    
    T_range = np.linspace(200, 600, 100)
    
    # Viscosity enthalpy (Arrhenius)
    fluids = {
        'Water': {'H_0': 0.18, 'color': '#00aaff'},
        'Glycerol': {'H_0': 0.65, 'color': '#ff00ff'},
        'Silicone oil': {'H_0': 0.35, 'color': '#ffaa00'},
    }
    
    for name, props in fluids.items():
        # Enthalpy of activation for viscous flow
        H = props['H_0'] * (1 - 0.001 * (T_range - 300))  # Slight T dependence
        H = np.maximum(H, 0.05)
        
        ax.plot(T_range, H, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('H (eV/molecule)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

# Create Panel 2
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Categorical Enthalpy: Transport Types', fontsize=14, color='white', y=0.98)

draw_enthalpy_electric(axes2[0, 0])
draw_enthalpy_diffusive(axes2[0, 1])
draw_enthalpy_thermal(axes2[1, 0])
draw_enthalpy_viscous(axes2[1, 1])

plt.tight_layout()
fig2.savefig('figures/panel_categorical_enthalpy.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig2)

#==============================================================================
# Panel 3: Aperture Selectivity (Transport Coefficients)
#==============================================================================

def draw_resistance(ax):
    """Electrical resistance from aperture selectivity"""
    ax.set_title('Resistance (ρ = N⁻¹Σ τᵢⱼgᵢⱼ)', fontsize=10, color='#00ffff')
    
    T_range = np.linspace(10, 500, 100)
    
    # Selectivity determines resistance
    materials = {
        'Copper': {'s': 0.7, 'color': '#ff6600'},
        'Superconductor (T<Tc)': {'s': 1.0, 'Tc': 90, 'color': '#00ffff'},
        'Insulator': {'s': 0.001, 'color': '#888888'},
    }
    
    for name, props in materials.items():
        if 'Tc' in props:
            # Superconductor: s=1 below Tc
            s = np.where(T_range < props['Tc'], 1.0, 0.3)
            rho = (1 - s) * 10  # Resistance proportional to (1-s)
        elif props['s'] < 0.01:
            # Insulator
            rho = 100 * np.exp(1000 / T_range)
        else:
            # Normal metal
            s = props['s'] - 0.001 * T_range  # Selectivity decreases with T
            rho = (1 - s) * 10
        
        ax.semilogy(T_range, rho + 0.01, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Resistivity (μΩ·cm)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_viscosity(ax):
    """Viscosity from aperture selectivity"""
    ax.set_title('Viscosity (μ = Σ τᵢⱼgᵢⱼ)', fontsize=10, color='#ff00ff')
    
    T_range = np.linspace(200, 600, 100)
    
    fluids = {
        'Water': {'eta_0': 1e-3, 'E_a': 2000, 'color': '#00aaff'},
        'Glycerol': {'eta_0': 1.5, 'E_a': 5000, 'color': '#ff00ff'},
        'Superfluid He (T<Tλ)': {'Tc': 2.17, 'color': '#00ff00'},
    }
    
    R = 8.314
    
    for name, props in fluids.items():
        if 'Tc' in props:
            # Superfluid: viscosity = 0 below Tc
            T_he = np.linspace(0.5, 4, 100)
            eta = np.where(T_he < props['Tc'], 0, 1e-6 * (T_he - props['Tc'])**2)
            ax.semilogy(T_he * 100 + 200, eta + 1e-10, color=props['color'], 
                       linewidth=2, label=name)
        else:
            eta = props['eta_0'] * np.exp(props['E_a'] / (R * T_range))
            ax.semilogy(T_range, eta * 1000, color=props['color'], 
                       linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Viscosity (mPa·s)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_diffusivity(ax):
    """Diffusivity from aperture selectivity"""
    ax.set_title('Diffusivity (D⁻¹ ∝ Σ τᵢⱼgᵢⱼ)', fontsize=10, color='#00ff00')
    
    T_range = np.linspace(300, 1200, 100)
    
    processes = {
        'Cu in Cu': {'D_0': 2e-5, 'Q': 2.0, 'color': '#ff6600'},
        'C in Fe': {'D_0': 6e-7, 'Q': 0.8, 'color': '#cccccc'},
        'H in Pd': {'D_0': 2.9e-7, 'Q': 0.23, 'color': '#00ff00'},
    }
    
    for name, props in processes.items():
        # Arrhenius diffusion
        D = props['D_0'] * np.exp(-props['Q'] * 11604 / T_range)
        
        ax.semilogy(T_range, D, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Diffusivity (m²/s)', fontsize=8)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_thermal_inertia(ax):
    """Thermal inertia from aperture selectivity"""
    ax.set_title('Thermal Conductivity (κ⁻¹ ∝ Σ τᵢⱼgᵢⱼ)', fontsize=10, color='#ff6600')
    
    T_range = np.linspace(10, 500, 100)
    
    materials = {
        'Diamond': {'k_300': 2000, 'n': 1.5, 'color': '#00ffff'},
        'Copper': {'k_300': 400, 'n': 0.1, 'color': '#ff6600'},
        'Silicon': {'k_300': 150, 'n': 1.2, 'color': '#00ff00'},
        'Glass': {'k_300': 1.0, 'n': -0.3, 'color': '#888888'},
    }
    
    for name, props in materials.items():
        k = props['k_300'] * (300 / T_range)**props['n']
        
        ax.loglog(T_range, k, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Thermal Conductivity (W/m·K)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

# Create Panel 3
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
fig3.suptitle('Aperture Selectivity → Transport Coefficients', fontsize=14, color='white', y=0.98)

draw_resistance(axes3[0, 0])
draw_viscosity(axes3[0, 1])
draw_diffusivity(axes3[1, 0])
draw_thermal_inertia(axes3[1, 1])

plt.tight_layout()
fig3.savefig('figures/panel_aperture_selectivity.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig3)

#==============================================================================
# Panel 4: Partition Lag
#==============================================================================

def draw_partition_lag_electric(ax):
    """Partition lag for electrical transport"""
    ax.set_title('Electric: Partition Lag τₚ', fontsize=10, color='#00ffff')
    
    T_range = np.linspace(10, 500, 100)
    
    mechanisms = {
        'Phonon scattering': {'tau_0': 10, 'n': 1, 'color': '#ff6600'},
        'Impurity': {'tau_0': 50, 'n': 0, 'color': '#00ff00'},
        'e-e scattering': {'tau_0': 1000, 'n': 2, 'color': '#ff00ff'},
    }
    
    for name, props in mechanisms.items():
        if props['n'] == 0:
            tau = props['tau_0'] * np.ones_like(T_range)
        else:
            tau = props['tau_0'] * (300 / T_range)**props['n']
        
        ax.semilogy(T_range, tau, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Partition lag τₚ (fs)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_partition_lag_diffusive(ax):
    """Partition lag for diffusive transport"""
    ax.set_title('Diffusive: Partition Lag τₚ', fontsize=10, color='#00ff00')
    
    T_range = np.linspace(300, 1200, 100)
    
    processes = {
        'Vacancy jump': {'tau_0': 1e-13, 'E_a': 0.8, 'color': '#00ff00'},
        'Interstitial': {'tau_0': 1e-14, 'E_a': 0.3, 'color': '#66ff66'},
        'GB diffusion': {'tau_0': 1e-12, 'E_a': 0.5, 'color': '#00cc00'},
    }
    
    for name, props in processes.items():
        # Partition lag from jump frequency
        tau = props['tau_0'] * np.exp(props['E_a'] * 11604 / T_range)
        
        ax.semilogy(T_range, tau * 1e15, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Partition lag τₚ (fs)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_partition_lag_thermal(ax):
    """Partition lag for thermal transport"""
    ax.set_title('Thermal: Partition Lag τₚ', fontsize=10, color='#ff6600')
    
    omega = np.linspace(0.1, 15, 100)  # THz
    
    mechanisms = {
        'Normal scattering': {'tau_0': 100, 'n': 2, 'color': '#00ff00'},
        'Umklapp': {'tau_0': 10, 'n': 3, 'color': '#ff6600'},
        'Boundary': {'tau_0': 1000, 'n': 0, 'color': '#00ffff'},
        'Impurity': {'tau_0': 50, 'n': 4, 'color': '#ff00ff'},
    }
    
    for name, props in mechanisms.items():
        tau = props['tau_0'] / (1 + (omega / 5)**props['n'])
        
        ax.semilogy(omega, tau, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Phonon frequency (THz)', fontsize=8)
    ax.set_ylabel('Partition lag τₚ (ps)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

def draw_partition_lag_viscous(ax):
    """Partition lag for viscous transport"""
    ax.set_title('Viscous: Partition Lag τₚ', fontsize=10, color='#ff00ff')
    
    T_range = np.linspace(200, 600, 100)
    
    fluids = {
        'Water': {'tau_0': 1, 'E_a': 0.18, 'color': '#00aaff'},
        'Glycerol': {'tau_0': 100, 'E_a': 0.65, 'color': '#ff00ff'},
        'n-Hexane': {'tau_0': 0.5, 'E_a': 0.08, 'color': '#00ff00'},
    }
    
    for name, props in fluids.items():
        tau = props['tau_0'] * np.exp(props['E_a'] * 11604 / T_range)
        
        ax.semilogy(T_range, tau, color=props['color'], linewidth=2, label=name)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Partition lag τₚ (ps)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)

# Create Panel 4
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))
fig4.suptitle('Partition Lag: Transport Types', fontsize=14, color='white', y=0.98)

draw_partition_lag_electric(axes4[0, 0])
draw_partition_lag_diffusive(axes4[0, 1])
draw_partition_lag_thermal(axes4[1, 0])
draw_partition_lag_viscous(axes4[1, 1])

plt.tight_layout()
fig4.savefig('figures/panel_partition_lag.png', dpi=150, bbox_inches='tight',
             facecolor='#0a0a0a', edgecolor='none')
plt.close(fig4)

#==============================================================================
# Save data
#==============================================================================

partition_data = {
    'categorical_potential': {
        'formula': 'Φ = -kB*T*ln(s)',
        'units': 'kB*T',
        'description': 'Energy cost of aperture selectivity'
    },
    'categorical_enthalpy': {
        'formula': 'H = U + Σ n_a Φ_a',
        'description': 'Internal energy plus aperture potentials'
    },
    'transport_coefficients': {
        'electrical': 'ρ = (ne²)⁻¹ Σ τᵢⱼ gᵢⱼ',
        'viscous': 'μ = Σ τᵢⱼ gᵢⱼ',
        'diffusive': 'D⁻¹ ∝ Σ τᵢⱼ gᵢⱼ',
        'thermal': 'κ⁻¹ ∝ Σ τᵢⱼ gᵢⱼ'
    },
    'partition_lag': {
        'description': 'Time for partition operation to complete',
        'electric_phonon_fs': 10,
        'diffusion_vacancy_fs': 1e5,
        'thermal_umklapp_ps': 1,
        'viscous_collision_ps': 10
    },
    'phase_transitions': {
        'superconductor': {
            'mechanism': 'Cooper pairing → s = 1 → τₚ = 0',
            'result': 'ρ = 0 exactly'
        },
        'superfluid': {
            'mechanism': 'Bose condensation → s = 1 → τₚ = 0',
            'result': 'μ = 0 exactly'
        }
    }
}

with open('data/partition_border_data.json', 'w') as f:
    json.dump(partition_data, f, indent=2)

print("Generated: figures/panel_categorical_potential.png")
print("Generated: figures/panel_categorical_enthalpy.png")
print("Generated: figures/panel_aperture_selectivity.png")
print("Generated: figures/panel_partition_lag.png")
print("Generated: data/partition_border_data.json")
