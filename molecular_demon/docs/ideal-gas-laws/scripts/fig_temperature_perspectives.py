"""
Figure 1: Temperature Perspectives
Demonstrates the triple equivalence of temperature definitions:
- Categorical: T = (hbar/k_B) * dM/dt
- Oscillatory: T = (hbar/k_B) * <omega>
- Partition: T = (hbar/k_B) / <tau_p>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import json
import os

# Constants
hbar = 1.054571817e-34  # J·s
k_B = 1.380649e-23      # J/K
c = 299792458           # m/s
G = 6.67430e-11         # m³/(kg·s²)

# Planck temperature
T_Planck = np.sqrt(hbar * c**5 / (G * k_B**2))  # ~1.4e32 K
omega_Planck = np.sqrt(c**5 / (hbar * G))       # ~1.9e43 rad/s

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

# Color scheme
colors = {
    'categorical': '#2ecc71',  # Green
    'oscillatory': '#3498db',  # Blue  
    'partition': '#e74c3c',    # Red
    'classical': '#2c3e50',    # Dark gray
}

def categorical_actualization_rate(T):
    """dM/dt = k_B * T / hbar"""
    return k_B * T / hbar

def oscillatory_frequency(T, saturate=True):
    """<omega> = k_B * T / hbar, with optional Planck saturation"""
    omega = k_B * T / hbar
    if saturate:
        # Saturation at Planck frequency
        omega = omega_Planck * np.tanh(omega / omega_Planck)
    return omega

def partition_lag(T):
    """<tau_p> = hbar / (k_B * T)"""
    return hbar / (k_B * T)

def create_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Temperature: Triple Equivalence Perspectives', fontsize=14, fontweight='bold')
    
    # Temperature range: 1 mK to 10^13 K
    T = np.logspace(-3, 13, 500)
    
    # Panel A: Categorical Actualization Rate
    ax = axes[0, 0]
    dM_dt = categorical_actualization_rate(T)
    ax.loglog(T, dM_dt, color=colors['categorical'], linewidth=2, label='Categorical')
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel('dM/dt (transitions/s)')
    ax.set_title('A: Categorical Actualization Rate')
    
    # Regime annotations
    ax.axvspan(1e-3, 1, alpha=0.1, color='blue', label='Quantum')
    ax.axvspan(1, 1e6, alpha=0.1, color='green', label='Classical')
    ax.axvspan(1e6, 1e13, alpha=0.1, color='orange', label='Relativistic')
    ax.legend(loc='lower right')
    ax.set_xlim(1e-3, 1e13)
    
    # Panel B: Oscillatory Frequency
    ax = axes[0, 1]
    omega_sat = oscillatory_frequency(T, saturate=True)
    omega_unsat = oscillatory_frequency(T, saturate=False)
    ax.loglog(T, omega_unsat, '--', color=colors['classical'], linewidth=1, alpha=0.5, label='Classical (no bound)')
    ax.loglog(T, omega_sat, color=colors['oscillatory'], linewidth=2, label='Categorical')
    ax.axhline(y=omega_Planck, color='purple', linestyle=':', linewidth=1.5, label=r'$\omega_{Planck}$')
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel(r'$\langle\omega\rangle$ (rad/s)')
    ax.set_title('B: Oscillatory Frequency')
    ax.legend(loc='lower right')
    ax.set_xlim(1e-3, 1e13)
    ax.set_ylim(1e8, 1e50)
    ax.text(1e11, omega_Planck*5, r'$\omega_{Planck} = 1.85 \times 10^{43}$ rad/s', fontsize=8, color='purple')
    
    # Panel C: Partition Lag
    ax = axes[1, 0]
    tau_p = partition_lag(T)
    ax.loglog(T, tau_p, color=colors['partition'], linewidth=2)
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel(r'$\langle\tau_p\rangle$ (s)')
    ax.set_title('C: Partition Lag')
    
    # Annotations
    ax.annotate('Long lag\n(cold)', xy=(1e-2, 1e-9), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.annotate('Short lag\n(hot)', xy=(1e10, 1e-32), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax.set_xlim(1e-3, 1e13)
    
    # Panel D: Equivalence Test
    ax = axes[1, 1]
    
    # Normalized quantities (all should equal 1 when properly normalized)
    T_test = np.logspace(0, 10, 50)  # 1 K to 10^10 K
    
    # Calculate the three perspectives
    dM_dt_norm = categorical_actualization_rate(T_test) * hbar / k_B  # Should give T
    omega_norm = oscillatory_frequency(T_test, saturate=False) * hbar / k_B  # Should give T
    tau_inv_norm = (1/partition_lag(T_test)) * hbar / k_B  # Should give T
    
    # Plot as ratios to T (should all be 1)
    ax.semilogx(T_test, dM_dt_norm / T_test, 'o', color=colors['categorical'], 
                markersize=6, label='Categorical', alpha=0.7)
    ax.semilogx(T_test, omega_norm / T_test, 's', color=colors['oscillatory'], 
                markersize=5, label='Oscillatory', alpha=0.7)
    ax.semilogx(T_test, tau_inv_norm / T_test, '^', color=colors['partition'], 
                markersize=5, label='Partition', alpha=0.7)
    
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Temperature T (K)')
    ax.set_ylabel('Ratio to Classical T')
    ax.set_title('D: Equivalence Test')
    ax.set_ylim(0.9, 1.1)
    ax.legend(loc='upper right')
    ax.set_xlim(1, 1e10)
    
    plt.tight_layout()
    return fig

def save_data(T, dM_dt, omega, tau_p):
    """Save computed data to JSON"""
    data = {
        'description': 'Temperature perspectives data',
        'temperature_K': T.tolist(),
        'categorical_dM_dt': dM_dt.tolist(),
        'oscillatory_omega': omega.tolist(),
        'partition_tau_p': tau_p.tolist(),
        'constants': {
            'hbar': hbar,
            'k_B': k_B,
            'T_Planck': T_Planck,
            'omega_Planck': omega_Planck
        }
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_temperature_perspectives.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    # Generate figure
    fig = create_figure()
    
    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_temperature_perspectives.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    # Save data
    T = np.logspace(-3, 13, 500)
    save_data(T, categorical_actualization_rate(T), 
              oscillatory_frequency(T, saturate=True), partition_lag(T))
    
    plt.show()

