"""
Lindemann Amplitude Monitor (LAM)
Monitors atomic oscillation amplitude to predict melting via partition extinction
"""

import numpy as np
import matplotlib.pyplot as plt
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

# Physical constants
kB = 1.380649e-23
hbar = 1.054571e-34
amu = 1.66054e-27

class LindemannAmplitudeMonitor:
    """
    Monitors Lindemann parameter η = <u²>^(1/2) / a
    Predicts melting when η exceeds critical threshold (~0.1-0.2)
    """
    
    def __init__(self, material_params=None):
        self.params = material_params or self._default_params()
        self.eta_c = 0.15  # Critical Lindemann parameter
        
    def _default_params(self):
        """Default parameters for copper."""
        return {
            'name': 'Copper',
            'M': 63.55,  # Atomic mass (amu)
            'a': 3.61e-10,  # Lattice constant (m)
            'Theta_D': 343,  # Debye temperature (K)
            'T_m': 1358,  # Melting temperature (K)
            'omega_D': 4.8e13  # Debye frequency (rad/s)
        }
    
    def mean_square_displacement(self, T):
        """
        Calculate mean square displacement <u²> at temperature T.
        Uses Debye model approximation.
        """
        M = self.params['M'] * amu
        omega_D = self.params['omega_D']
        Theta_D = self.params['Theta_D']
        
        # Classical contribution (high T)
        u2_classical = 3 * kB * T / (M * omega_D**2)
        
        # Quantum correction (low T)
        if T < Theta_D:
            x = Theta_D / T
            # Debye function approximation
            u2_quantum = u2_classical * (1 + (x/4)**2)
        else:
            u2_quantum = u2_classical
        
        return u2_quantum
    
    def lindemann_parameter(self, T):
        """
        Calculate Lindemann parameter η = sqrt(<u²>) / a
        """
        u2 = self.mean_square_displacement(T)
        u_rms = np.sqrt(u2)
        eta = u_rms / self.params['a']
        return eta
    
    def predict_melting_temperature(self):
        """
        Predict melting temperature from Lindemann criterion.
        Find T where η = η_c
        """
        T_range = np.linspace(100, 2000, 1000)
        eta_values = [self.lindemann_parameter(T) for T in T_range]
        
        # Find crossing
        for i, eta in enumerate(eta_values):
            if eta >= self.eta_c:
                return T_range[i]
        
        return None
    
    def scan_temperature(self, T_range):
        """Scan Lindemann parameter vs temperature."""
        results = []
        for T in T_range:
            eta = self.lindemann_parameter(T)
            u2 = self.mean_square_displacement(T)
            
            results.append({
                'temperature_K': T,
                'eta': eta,
                'u_rms_angstrom': np.sqrt(u2) * 1e10,
                'melted': eta >= self.eta_c
            })
        
        return results


# Material database
MATERIALS = {
    'Copper': {'M': 63.55, 'a': 3.61e-10, 'Theta_D': 343, 'T_m': 1358, 'omega_D': 4.8e13},
    'Aluminum': {'M': 26.98, 'a': 4.05e-10, 'Theta_D': 428, 'T_m': 933, 'omega_D': 6.0e13},
    'Gold': {'M': 196.97, 'a': 4.08e-10, 'Theta_D': 165, 'T_m': 1337, 'omega_D': 2.3e13},
    'Iron': {'M': 55.85, 'a': 2.87e-10, 'Theta_D': 470, 'T_m': 1811, 'omega_D': 6.6e13},
    'Lead': {'M': 207.2, 'a': 4.95e-10, 'Theta_D': 105, 'T_m': 600, 'omega_D': 1.5e13},
}


def visualize_lam_results():
    """Create visualization of LAM measurements."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Lindemann Amplitude Monitor (LAM) Results', fontsize=14, color='#ffcc00', y=0.98)
    
    # Plot 1: Lindemann parameter vs temperature (copper)
    ax = axes[0, 0]
    ax.set_title('Lindemann Parameter η vs Temperature (Cu)', fontsize=10, color='#ff6600')
    
    lam_cu = LindemannAmplitudeMonitor()
    T_range = np.linspace(50, 1500, 200)
    results_cu = lam_cu.scan_temperature(T_range)
    
    eta = [r['eta'] for r in results_cu]
    T_m_actual = lam_cu.params['T_m']
    T_m_predicted = lam_cu.predict_melting_temperature()
    
    ax.plot(T_range, eta, color='#ff6600', linewidth=2)
    ax.axhline(lam_cu.eta_c, color='#ff0000', linestyle='--', label=f'ηc = {lam_cu.eta_c}')
    ax.axvline(T_m_actual, color='#00ffff', linestyle=':', label=f'Tm (actual) = {T_m_actual} K')
    if T_m_predicted is not None:
        ax.axvline(T_m_predicted, color='#00ff00', linestyle='-.', label=f'Tm (predicted) = {T_m_predicted:.0f} K')
    
    ax.fill_between(T_range, 0, eta, where=[e >= lam_cu.eta_c for e in eta], 
                   alpha=0.3, color='red', label='MELTED (site partition extinct)')
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Lindemann parameter η', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Multiple materials comparison
    ax = axes[0, 1]
    ax.set_title('Lindemann Parameter: Material Comparison', fontsize=10, color='#00ffff')
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(MATERIALS)))
    
    for (name, params), c in zip(MATERIALS.items(), colors):
        params_full = {**params, 'name': name}
        lam = LindemannAmplitudeMonitor(params_full)
        
        # Normalize temperature by melting point
        T_norm = np.linspace(0.1, 1.2, 100)
        T_actual = T_norm * params['T_m']
        eta = [lam.lindemann_parameter(T) for T in T_actual]
        
        ax.plot(T_norm, eta, color=c, linewidth=2, label=name)
    
    ax.axhline(0.15, color='white', linestyle='--', alpha=0.5, label='ηc ≈ 0.15')
    ax.axvline(1.0, color='white', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('T / Tm (normalized)', fontsize=8)
    ax.set_ylabel('Lindemann parameter η', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMS displacement vs temperature
    ax = axes[1, 0]
    ax.set_title('RMS Displacement vs Temperature', fontsize=10, color='#00ff88')
    
    for (name, params), c in zip(MATERIALS.items(), colors):
        params_full = {**params, 'name': name}
        lam = LindemannAmplitudeMonitor(params_full)
        
        T_range = np.linspace(50, 2000, 100)
        u_rms = [np.sqrt(lam.mean_square_displacement(T)) * 1e10 for T in T_range]  # Angstroms
        
        ax.plot(T_range, u_rms, color=c, linewidth=2, label=name)
        
        # Mark melting point
        ax.scatter([params['T_m']], [u_rms[np.argmin(np.abs(T_range - params['T_m']))]], 
                  color=c, s=100, marker='x', zorder=5)
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('RMS displacement (Å)', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Site assignment partition interpretation
    ax = axes[1, 1]
    ax.set_title('Site Assignment Partition: Solid → Liquid', fontsize=10, color='#ff00ff')
    
    # Schematic of partition extinction
    T_norm = np.linspace(0, 1.5, 100)
    
    # Probability of correct site assignment
    p_correct = np.where(T_norm < 1, 
                         1 - 0.5 * (T_norm)**2,  # Decreases as η increases
                         0.5)  # Random (liquid)
    
    # Categorical potential
    phi = -np.log(np.clip(p_correct, 0.01, 1))
    
    ax.plot(T_norm, p_correct, color='#00ff00', linewidth=2, label='Site assignment probability')
    ax.plot(T_norm, phi / phi.max(), color='#ff00ff', linewidth=2, label='Categorical potential Φ (norm)')
    
    ax.axvline(1.0, color='#ffcc00', linestyle='--', label='Melting (T = Tm)')
    ax.fill_between(T_norm[T_norm >= 1], 0, 1, alpha=0.2, color='#ff0000')
    
    ax.text(0.5, 0.8, 'SOLID\nSite partition\ndefined', fontsize=9, ha='center', color='#00ff00')
    ax.text(1.25, 0.8, 'LIQUID\nSite partition\nextinct', fontsize=9, ha='center', color='#ff6666')
    
    ax.set_xlabel('T / Tm (normalized)', fontsize=8)
    ax.set_ylabel('Probability / Potential', fontsize=8)
    ax.legend(loc='center right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    
    plt.tight_layout()
    fig.savefig('figures/panel_lam_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    plt.close(fig)
    
    return lam_cu


# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Lindemann Amplitude Monitor (LAM)")
    print("Melting Prediction Through Partition Extinction")
    print("=" * 60)
    
    lam_cu = visualize_lam_results()
    
    # Analyze all materials
    print("\nMaterial Analysis:")
    print("-" * 50)
    
    material_results = []
    for name, params in MATERIALS.items():
        params_full = {**params, 'name': name}
        lam = LindemannAmplitudeMonitor(params_full)
        T_m_pred = lam.predict_melting_temperature()
        T_m_actual = params['T_m']
        
        if T_m_pred is not None:
            error = abs(T_m_pred - T_m_actual) / T_m_actual * 100
            print(f"{name:12s}: Tm(actual) = {T_m_actual:4d} K, Tm(pred) = {T_m_pred:4.0f} K, error = {error:.1f}%")
        else:
            error = None
            print(f"{name:12s}: Tm(actual) = {T_m_actual:4d} K, Tm(pred) = N/A")
        
        material_results.append({
            'name': name,
            'T_m_actual_K': T_m_actual,
            'T_m_predicted_K': T_m_pred if T_m_pred is not None else 0,
            'error_percent': error if error is not None else 0,
            'Theta_D_K': params['Theta_D']
        })
    
    # Save data
    output_data = {
        'instrument': 'Lindemann Amplitude Monitor',
        'principle': 'Melting occurs when η = √<u²>/a exceeds ηc ≈ 0.1-0.2',
        'eta_critical': 0.15,
        'interpretation': 'Site assignment partition becomes extinct at melting',
        'materials': material_results
    }
    
    with open('data/lam_measurements.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nGenerated: figures/panel_lam_results.png")
    print(f"Generated: data/lam_measurements.json")

