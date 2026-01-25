"""
Figure 3: Internal Energy Perspectives
Demonstrates the triple equivalence of internal energy:
- Categorical: U = k_B T * M_active
- Oscillatory: U = sum(hbar*omega*(n + 1/2))
- Partition: U = sum(Phi_a * N_a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import json
import os

# Constants
k_B = constants.k
hbar = constants.hbar
N_A = constants.Avogadro

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
    'quantum': '#9b59b6',
}

def classical_energy(T, N, f=3):
    """Classical equipartition: U = (f/2)*N*k_B*T"""
    return (f/2) * N * k_B * T

def categorical_energy(T, N, T_activation=None):
    """U = k_B*T * M_active, where M_active depends on T"""
    if T_activation is None:
        T_activation = [1, 100, 3000]  # trans, rot, vib
    
    M_active = 0
    # Translational (always active)
    M_active += 3
    # Rotational (activates above ~100 K)
    M_active += 2 * (1 / (1 + np.exp(-(T - 100)/20)))
    # Vibrational (activates above ~3000 K)
    M_active += 2 * (1 / (1 + np.exp(-(T - 3000)/500)))
    
    return k_B * T * N * M_active / 2

def oscillatory_energy(T, N, omega=1e13):
    """Quantum oscillator: U = hbar*omega*(n + 1/2)"""
    x = hbar * omega / (k_B * T)
    # Avoid overflow
    x = np.clip(x, 0, 700)
    n_avg = 1 / (np.exp(x) - 1 + 1e-10)
    return N * hbar * omega * (n_avg + 0.5)

def einstein_heat_capacity(T, theta_E):
    """Einstein model heat capacity"""
    x = theta_E / T
    x = np.clip(x, 0, 700)
    return 3 * N_A * k_B * (x**2 * np.exp(x)) / (np.exp(x) - 1)**2

def create_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Internal Energy: Triple Equivalence Perspectives', fontsize=14, fontweight='bold')
    
    N = N_A  # 1 mole
    
    # Panel A: Categorical Energy vs Temperature
    ax = axes[0, 0]
    T = np.logspace(-1, 4, 500)
    
    U_classical = classical_energy(T, N, f=3)
    U_categorical = categorical_energy(T, N)
    
    # Normalized by N*k_B*T
    ax.semilogx(T, U_classical / (N * k_B * T), '--', color=colors['classical'], 
                linewidth=2, label='Classical (3/2)')
    ax.semilogx(T, U_categorical / (N * k_B * T), '-', color=colors['categorical'],
                linewidth=2, label='Categorical ($M_{active}/2$)')
    
    # Mark activation temperatures
    ax.axvline(x=100, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=3000, color='red', linestyle=':', alpha=0.5)
    ax.text(120, 2.5, 'Rotation\nactivates', fontsize=8, color='orange')
    ax.text(3500, 3.0, 'Vibration\nactivates', fontsize=8, color='red')
    
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel('U / ($Nk_BT$)')
    ax.set_title('A: Categorical Energy vs Temperature')
    ax.legend(loc='upper left')
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1, 4)
    
    # Panel B: Oscillatory Energy (Quantum)
    ax = axes[0, 1]
    
    omega = 1e13  # rad/s (typical molecular vibration)
    T_range = np.logspace(0, 4, 200)
    
    U_osc = oscillatory_energy(T_range, N, omega)
    U_zp = N * hbar * omega / 2  # Zero-point energy
    
    ax.semilogy(T_range, U_osc, '-', color=colors['oscillatory'], linewidth=2,
                label='Oscillatory ($\\sum\\hbar\\omega(n+\\frac{1}{2})$)')
    ax.axhline(y=U_zp, color='purple', linestyle='--', linewidth=1.5,
               label=f'Zero-point $U_0 = N\\hbar\\omega/2$')
    
    # Classical limit
    U_classical_limit = N * k_B * T_range
    ax.semilogy(T_range, U_classical_limit, ':', color=colors['classical'], 
                linewidth=1.5, alpha=0.7, label='Classical limit $Nk_BT$')
    
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel('Energy U (J)')
    ax.set_title('B: Oscillatory Energy (Quantum)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(1, 1e4)
    
    # Panel C: Partition Energy (Aperture)
    ax = axes[1, 0]
    
    # Show energy stored in different aperture types
    T_part = np.logspace(1, 4, 100)
    
    # Different aperture contributions
    U_trans = 1.5 * N * k_B * T_part  # Translational
    U_rot = N * k_B * T_part * (1 / (1 + np.exp(-(T_part - 100)/20)))  # Rotational
    U_vib = N * k_B * T_part * (1 / (1 + np.exp(-(T_part - 3000)/500)))  # Vibrational
    
    ax.fill_between(T_part, 0, U_trans/(N*k_B*T_part), alpha=0.5, 
                    color='green', label='Translational')
    ax.fill_between(T_part, U_trans/(N*k_B*T_part), 
                    (U_trans + U_rot)/(N*k_B*T_part), alpha=0.5,
                    color='orange', label='Rotational')
    ax.fill_between(T_part, (U_trans + U_rot)/(N*k_B*T_part),
                    (U_trans + U_rot + U_vib)/(N*k_B*T_part), alpha=0.5,
                    color='red', label='Vibrational')
    
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel('$\\sum\\Phi_a N_a$ / ($Nk_BT$)')
    ax.set_title('C: Partition Energy (Aperture Contributions)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xscale('log')
    ax.set_xlim(10, 1e4)
    ax.set_ylim(0, 4)
    
    # Panel D: Heat Capacity
    ax = axes[1, 1]
    
    T_cv = np.logspace(0, 4, 300)
    
    # Classical
    Cv_classical = np.ones_like(T_cv) * 1.5  # 3/2 for monatomic
    
    # Categorical (with mode activation)
    Cv_categorical = np.zeros_like(T_cv)
    # Translational
    Cv_categorical += 1.5
    # Rotational activation
    dM_rot_dT = 2 * (1/20) * np.exp(-(T_cv - 100)/20) / (1 + np.exp(-(T_cv - 100)/20))**2
    Cv_categorical += 1.0 * (1 / (1 + np.exp(-(T_cv - 100)/20))) + T_cv * dM_rot_dT * 0.1
    # Vibrational activation
    dM_vib_dT = 2 * (1/500) * np.exp(-(T_cv - 3000)/500) / (1 + np.exp(-(T_cv - 3000)/500))**2
    Cv_categorical += 1.0 * (1 / (1 + np.exp(-(T_cv - 3000)/500))) + T_cv * dM_vib_dT * 0.05
    
    # Einstein model for comparison (diatomic, theta_E ~ 3000 K)
    theta_E = 3000
    x = theta_E / T_cv
    x = np.clip(x, 0, 50)
    Cv_einstein = 2.5 + (x**2 * np.exp(x)) / (np.exp(x) - 1 + 1e-10)**2
    Cv_einstein = np.clip(Cv_einstein, 2.5, 3.5)
    
    ax.semilogx(T_cv, Cv_classical, '--', color=colors['classical'], linewidth=2,
                label='Classical (3/2)')
    ax.semilogx(T_cv, Cv_categorical, '-', color=colors['categorical'], linewidth=2,
                label='Categorical')
    ax.semilogx(T_cv, Cv_einstein, ':', color=colors['quantum'], linewidth=2,
                label='Einstein model')
    
    # Annotations
    ax.annotate('Quantum\nfreeze-out', xy=(2, 1.55), fontsize=8,
                bbox=dict(facecolor='lightblue', alpha=0.5))
    ax.annotate('Classical\nplateau', xy=(300, 2.6), fontsize=8,
                bbox=dict(facecolor='lightgreen', alpha=0.5))
    ax.annotate('Vibrational\nactivation', xy=(5000, 3.2), fontsize=8,
                bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel('$C_V / (Nk_B)$')
    ax.set_title('D: Heat Capacity (Mode Activation)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1, 4)
    
    plt.tight_layout()
    return fig

def save_data():
    """Save energy data to JSON"""
    N = N_A
    T = np.logspace(-1, 4, 100).tolist()
    
    data = {
        'description': 'Internal energy perspectives data',
        'N_particles': N,
        'temperature_K': T,
        'classical_energy_J': [classical_energy(t, N) for t in T],
        'categorical_energy_J': [categorical_energy(t, N) for t in T],
        'activation_temperatures_K': {
            'rotational': 100,
            'vibrational': 3000
        }
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_internal_energy.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_internal_energy.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

