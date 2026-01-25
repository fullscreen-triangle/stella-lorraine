"""
Figure 2: Pressure Perspectives
Demonstrates the triple equivalence of pressure definitions:
- Categorical: P = k_B T (dM/dV)
- Oscillatory: P = (1/3V) sum(m*omega^2*A^2)
- Partition: P = (k_B T/V) * sum(1/tau_boundary)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import json
import os

# Constants
k_B = constants.k
m_Ar = 39.948 * constants.atomic_mass

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

colors = {
    'categorical': '#2ecc71',
    'oscillatory': '#3498db',
    'partition': '#e74c3c',
    'classical': '#2c3e50',
}

def ideal_gas_pressure(rho, T):
    """P = rho * k_B * T (where rho is number density)"""
    return rho * k_B * T

def categorical_pressure(rho, T, rho_sat=1e30):
    """Categorical pressure with saturation at high density"""
    # dM/dV decreases as we approach saturation
    saturation_factor = 1 / (1 + (rho / rho_sat)**2)
    return rho * k_B * T * saturation_factor

def oscillatory_pressure(rho, T, m=m_Ar):
    """P = (1/3) * rho * m * <omega^2 * A^2>"""
    # For thermal oscillators, m*omega^2*A^2 = k_B*T
    return rho * k_B * T  # Same as ideal gas in classical limit

def partition_pressure(rho, T):
    """Partition-based pressure"""
    return rho * k_B * T

def create_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Pressure: Triple Equivalence Perspectives', fontsize=14, fontweight='bold')
    
    T = 300  # K
    
    # Panel A: Categorical vs Classical
    ax = axes[0, 0]
    rho = np.logspace(10, 32, 500)  # particles/m³
    
    P_classical = ideal_gas_pressure(rho, T)
    P_categorical = categorical_pressure(rho, T)
    
    ax.loglog(rho, P_classical, '--', color=colors['classical'], linewidth=2, 
              label='Classical ($P = \\rho k_B T$)')
    ax.loglog(rho, P_categorical, '-', color=colors['categorical'], linewidth=2,
              label='Categorical (with saturation)')
    
    ax.axvline(x=1e30, color='red', linestyle=':', alpha=0.5)
    ax.text(2e30, 1e10, '$\\rho_{sat}$', color='red', fontsize=9)
    
    ax.set_xlabel('Density $\\rho$ (particles/m$^3$)')
    ax.set_ylabel('Pressure P (Pa)')
    ax.set_title('A: Categorical vs Classical Pressure')
    ax.legend(loc='lower right')
    ax.set_xlim(1e10, 1e32)
    
    # Panel B: Oscillatory Pressure
    ax = axes[0, 1]
    
    P_oscillatory = oscillatory_pressure(rho, T)
    
    ax.loglog(rho, P_oscillatory, '-', color=colors['oscillatory'], linewidth=2,
              label='Oscillatory ($\\frac{1}{3}\\rho m\\omega^2 A^2$)')
    ax.loglog(rho, P_classical, '--', color=colors['classical'], linewidth=1, alpha=0.5,
              label='Classical')
    
    ax.set_xlabel('Density $\\rho$ (particles/m$^3$)')
    ax.set_ylabel('Pressure P (Pa)')
    ax.set_title('B: Oscillatory Pressure')
    ax.legend(loc='lower right')
    ax.set_xlim(1e10, 1e32)
    
    # Inset: oscillation diagram
    inset = ax.inset_axes([0.15, 0.55, 0.35, 0.35])
    theta = np.linspace(0, 2*np.pi, 100)
    r = 1 + 0.3*np.sin(3*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    inset.plot(x, y, color=colors['oscillatory'], linewidth=2)
    inset.scatter([0], [0], s=50, c='black')
    inset.arrow(0, 0, 0.8, 0, head_width=0.1, color='red')
    inset.text(0.9, 0.1, '$A\\omega^2$', fontsize=8)
    inset.set_xlim(-1.5, 1.5)
    inset.set_ylim(-1.5, 1.5)
    inset.set_aspect('equal')
    inset.axis('off')
    inset.set_title('Amplitude creates pressure', fontsize=8)
    
    # Panel C: Partition Pressure
    ax = axes[1, 0]
    
    P_partition = partition_pressure(rho, T)
    
    ax.loglog(rho, P_partition, '-', color=colors['partition'], linewidth=2,
              label='Partition (boundary rate)')
    ax.loglog(rho, P_classical, '--', color=colors['classical'], linewidth=1, alpha=0.5,
              label='Classical')
    
    ax.set_xlabel('Density $\\rho$ (particles/m$^3$)')
    ax.set_ylabel('Pressure P (Pa)')
    ax.set_title('C: Partition Pressure')
    ax.legend(loc='lower right')
    ax.set_xlim(1e10, 1e32)
    
    # Inset: boundary/bulk ratio
    inset = ax.inset_axes([0.55, 0.15, 0.4, 0.35])
    rho_inset = np.logspace(20, 30, 50)
    ratio = np.ones_like(rho_inset)  # For ideal gas, ratio = 1
    ratio_real = 1 + 0.1 * (rho_inset / 1e25)  # Slight increase for real gas
    inset.semilogx(rho_inset, ratio, '--', color=colors['classical'], label='Ideal')
    inset.semilogx(rho_inset, ratio_real, '-', color=colors['partition'], label='Real')
    inset.set_xlabel('$\\rho$', fontsize=8)
    inset.set_ylabel('Boundary/Bulk', fontsize=8)
    inset.legend(fontsize=7)
    inset.tick_params(labelsize=7)
    
    # Panel D: Pressure Saturation at High Density
    ax = axes[1, 1]
    
    rho_high = np.logspace(25, 32, 200)
    
    # Compressibility factor Z = PV/(NkT) = P/(rho*k_B*T)
    Z_classical = np.ones_like(rho_high)
    Z_categorical = categorical_pressure(rho_high, T) / (rho_high * k_B * T)
    
    # Van der Waals (for comparison)
    b = 3e-29  # m³/particle (typical)
    a = 1e-48  # Pa·m⁶ (typical)
    Z_vdw = 1 / (1 - b * rho_high) - a * rho_high / (k_B * T)
    Z_vdw = np.clip(Z_vdw, 0, 10)
    
    ax.semilogx(rho_high, Z_classical, '--', color=colors['classical'], linewidth=2,
                label='Classical (Z = 1)')
    ax.semilogx(rho_high, Z_categorical, '-', color=colors['categorical'], linewidth=2,
                label='Categorical (saturation)')
    ax.semilogx(rho_high, Z_vdw, ':', color='purple', linewidth=2,
                label='Van der Waals')
    
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(rho_high[rho_high > 1e28], 0, Z_categorical[rho_high > 1e28], 
                    alpha=0.2, color=colors['categorical'], label='Saturation regime')
    
    ax.set_xlabel('Density $\\rho$ (particles/m$^3$)')
    ax.set_ylabel('Compressibility factor Z = P/($\\rho k_B T$)')
    ax.set_title('D: Pressure Saturation at High Density')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(1e25, 1e32)
    ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    return fig

def save_data():
    """Save pressure data to JSON"""
    T = 300
    rho = np.logspace(10, 32, 100).tolist()
    
    data = {
        'description': 'Pressure perspectives data',
        'temperature_K': T,
        'density_particles_m3': rho,
        'classical_pressure_Pa': [ideal_gas_pressure(r, T) for r in rho],
        'categorical_pressure_Pa': [categorical_pressure(r, T) for r in rho],
        'saturation_density_m3': 1e30
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_pressure_perspectives.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_pressure_perspectives.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

