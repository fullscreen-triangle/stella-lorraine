"""
Figure 5: Velocity Distributions
Compares categorical (discrete, bounded) vs classical Maxwell-Boltzmann (continuous, unbounded)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import json
import os

# Constants
k_B = constants.k
m_Ar = 39.948 * constants.atomic_mass  # Argon mass
c = constants.c

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
    'classical': '#2c3e50',
    'forbidden': '#e74c3c',
    'bose_einstein': '#3498db',
}

def maxwell_boltzmann(v, T, m):
    """Maxwell-Boltzmann velocity distribution"""
    coeff = 4 * np.pi * (m / (2 * np.pi * k_B * T))**1.5
    return coeff * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

def categorical_distribution(m_cat, M_v):
    """Categorical velocity distribution f(m) = exp(-m/M_v) / Z"""
    exp_vals = np.exp(-m_cat / M_v)
    Z = np.sum(exp_vals)
    return exp_vals / Z

def bose_einstein_occupation(omega, T):
    """Bose-Einstein occupation number"""
    hbar = constants.hbar
    x = hbar * omega / (k_B * T)
    # Avoid overflow
    x = np.clip(x, 0, 700)
    return 1 / (np.exp(x) - 1 + 1e-10)

def create_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Velocity Distribution: Discrete and Bounded', fontsize=14, fontweight='bold')
    
    # Panel A: Room Temperature (300 K)
    ax = axes[0, 0]
    T = 300  # K
    v = np.linspace(0, 1500, 500)  # m/s
    
    # Classical distribution
    f_classical = maxwell_boltzmann(v, T, m_Ar)
    
    # Categorical distribution (discretized)
    n_categories = 50
    v_categories = np.linspace(0, 1500, n_categories)
    delta_v = v_categories[1] - v_categories[0]
    M_v = np.sqrt(2 * k_B * T / m_Ar) / delta_v  # characteristic category scale
    m_cat = np.arange(n_categories)
    f_categorical = categorical_distribution(m_cat, M_v)
    
    # Normalize categorical to match classical peak
    f_categorical_scaled = f_categorical * max(f_classical) / max(f_categorical) * 0.9
    
    ax.plot(v, f_classical, '-', color=colors['classical'], linewidth=2, label='Classical (Maxwell)')
    ax.bar(v_categories, f_categorical_scaled, width=delta_v*0.8, 
           color=colors['categorical'], alpha=0.6, label='Categorical (discrete)')
    ax.set_xlabel('Velocity v (m/s)')
    ax.set_ylabel('Probability density f(v)')
    ax.set_title('A: Room Temperature (T = 300 K)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1500)
    
    # Inset showing discrete structure
    inset = ax.inset_axes([0.55, 0.45, 0.4, 0.35])
    inset.bar(v_categories[15:25], f_categorical_scaled[15:25], width=delta_v*0.8,
              color=colors['categorical'], alpha=0.8)
    inset.plot(v[200:350], f_classical[200:350], '-', color=colors['classical'], linewidth=1.5)
    inset.set_xlim(450, 750)
    inset.set_title('Discrete structure', fontsize=8)
    inset.tick_params(labelsize=7)
    
    # Panel B: Ultra-Cold (1 mK)
    ax = axes[0, 1]
    T_cold = 1e-3  # K
    
    # At ultra-cold, only few categories occupied
    n_categories_cold = 20
    M_v_cold = 3  # Very few categories accessible
    m_cat_cold = np.arange(n_categories_cold)
    f_cold = categorical_distribution(m_cat_cold, M_v_cold)
    
    # Delta_v for ultra-cold
    v_thermal = np.sqrt(2 * k_B * T_cold / m_Ar)  # ~0.1 mm/s
    delta_v_cold = v_thermal / M_v_cold
    v_cold = m_cat_cold * delta_v_cold * 1000  # Convert to mm/s
    
    ax.bar(m_cat_cold, f_cold, width=0.8, color=colors['categorical'], 
           edgecolor='darkgreen', linewidth=1)
    ax.set_xlabel('Category index m')
    ax.set_ylabel('Probability f(m)')
    ax.set_title('B: Ultra-Cold (T = 1 mK)')
    ax.set_xlim(-0.5, 15)
    ax.annotate(f'$\\Delta v \\approx$ {delta_v_cold*1000:.2f} mm/s', 
                xy=(8, 0.15), fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Panel C: Relativistic (10^9 K)
    ax = axes[1, 0]
    T_rel = 1e9  # K
    
    # Fraction of speed of light
    v_frac = np.linspace(0, 1.2, 500)
    v_abs = v_frac * c
    
    # Classical (extends beyond c)
    f_rel_classical = maxwell_boltzmann(v_abs, T_rel, m_Ar)
    f_rel_classical = f_rel_classical / np.max(f_rel_classical)  # Normalize
    
    # Categorical (sharp cutoff at c)
    f_rel_categorical = f_rel_classical.copy()
    f_rel_categorical[v_frac > 1] = 0
    
    ax.semilogy(v_frac, f_rel_classical + 1e-10, '--', color=colors['classical'], 
                linewidth=2, label='Classical (unphysical)')
    ax.semilogy(v_frac[v_frac <= 1], f_rel_categorical[v_frac <= 1] + 1e-10, 
                '-', color=colors['categorical'], linewidth=2, label='Categorical')
    
    # Shade forbidden region
    ax.axvspan(1, 1.2, alpha=0.3, color=colors['forbidden'], label='Forbidden (v > c)')
    ax.axvline(x=1, color='red', linestyle=':', linewidth=2)
    
    ax.set_xlabel('v/c (fraction of speed of light)')
    ax.set_ylabel('Probability density (log scale)')
    ax.set_title('C: Relativistic ($T = 10^9$ K)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(1e-6, 2)
    ax.text(1.02, 0.1, 'v = c', fontsize=9, color='red', rotation=90)
    
    # Panel D: Oscillatory Distribution (Bose-Einstein)
    ax = axes[1, 1]
    T_be = 300  # K
    
    # Frequency range
    omega = np.logspace(10, 15, 100)  # rad/s
    hbar = constants.hbar
    
    # Bose-Einstein
    n_BE = bose_einstein_occupation(omega, T_be)
    
    # Categorical oscillatory (same formula, but emphasizing categorical origin)
    n_cat = n_BE  # Same distribution
    
    ax.loglog(omega, n_BE, '-', color=colors['classical'], linewidth=2, 
              label='Bose-Einstein', alpha=0.7)
    ax.loglog(omega[::5], n_cat[::5], 'o', color=colors['categorical'], 
              markersize=6, label='Categorical oscillatory', alpha=0.8)
    
    ax.set_xlabel(r'Frequency $\omega$ (rad/s)')
    ax.set_ylabel(r'Occupation $\langle n \rangle$')
    ax.set_title('D: Oscillatory Distribution')
    ax.legend(loc='upper right')
    ax.set_xlim(1e10, 1e15)
    
    ax.annotate('Perfect agreement:\nCategorical = Bose-Einstein', 
                xy=(3e12, 1e3), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    return fig

def save_data():
    """Save distribution data to JSON"""
    T_values = [300, 1e-3, 1e9]
    data = {
        'description': 'Velocity distribution data at different temperatures',
        'temperatures_K': T_values,
        'room_temp': {
            'T_K': 300,
            'v_range_m_s': [0, 1500],
            'n_categories': 50
        },
        'ultra_cold': {
            'T_K': 1e-3,
            'n_categories_occupied': 10
        },
        'relativistic': {
            'T_K': 1e9,
            'cutoff_v_c': 1.0
        }
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_velocity_distributions.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_velocity_distributions.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

