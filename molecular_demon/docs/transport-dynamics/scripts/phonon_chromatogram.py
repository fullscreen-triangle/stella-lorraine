"""
Phonon Chromatograph (PC)
Separates thermal transport by phonon mode, measuring mode-specific transport properties
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

class PhononChromatograph:
    """
    Categorical instrument for phonon mode separation.
    Treats thermal transport as chromatography through phonon space.
    """
    
    def __init__(self, material_params=None):
        self.params = material_params or self._default_params()
        self.omega_max = self.params['omega_D']  # Debye frequency (THz)
        
    def _default_params(self):
        """Default parameters for silicon."""
        return {
            'name': 'Silicon',
            'omega_D': 15.0,  # Debye frequency (THz)
            'v_LA': 8400,    # Longitudinal acoustic velocity (m/s)
            'v_TA': 5800,    # Transverse acoustic velocity (m/s)
            'Theta_D': 645,  # Debye temperature (K)
            'kappa_300K': 150  # Thermal conductivity at 300K (W/m·K)
        }
    
    def phonon_spectrum(self, omega, T):
        """
        Calculate phonon occupation at frequency omega and temperature T.
        Bose-Einstein distribution.
        """
        x = hbar * omega * 1e12 / (kB * T)  # Convert THz to rad/s
        n = 1 / (np.exp(np.clip(x, -100, 100)) - 1 + 1e-10)
        return n
    
    def mode_lifetime(self, omega, T, mode='LA'):
        """
        Calculate phonon lifetime for given mode.
        Different scattering mechanisms dominate at different frequencies.
        """
        omega_D = self.params['omega_D']
        
        # Normal scattering: τ_N^-1 ∝ ω²T
        tau_N = 1e-9 / (omega/omega_D)**2 / (T/300)
        
        # Umklapp scattering: τ_U^-1 ∝ ω²T*exp(-Θ/bT)
        Theta = self.params['Theta_D']
        tau_U = 1e-9 / (omega/omega_D)**2 / T * np.exp(Theta / (3*T))
        
        # Impurity scattering: τ_I^-1 ∝ ω⁴
        tau_I = 1e-6 / (omega/omega_D)**4
        
        # Boundary scattering (sample size dependent)
        L = 1e-3  # 1 mm sample
        v = self.params['v_LA'] if mode == 'LA' else self.params['v_TA']
        tau_B = L / v
        
        # Matthiessen's rule
        tau_total = 1 / (1/tau_N + 1/tau_U + 1/tau_I + 1/tau_B)
        
        return {
            'total': tau_total,
            'normal': tau_N,
            'umklapp': tau_U,
            'impurity': tau_I,
            'boundary': tau_B
        }
    
    def mean_free_path(self, omega, T, mode='LA'):
        """Calculate phonon mean free path."""
        tau = self.mode_lifetime(omega, T, mode)['total']
        v = self.params['v_LA'] if mode == 'LA' else self.params['v_TA']
        return v * tau
    
    def mode_conductivity(self, omega, T, mode='LA'):
        """
        Calculate thermal conductivity contribution from mode at frequency omega.
        κ(ω) = C(ω) * v² * τ(ω)
        """
        # Mode heat capacity
        n = self.phonon_spectrum(omega, T)
        x = hbar * omega * 1e12 / (kB * T)
        C = kB * x**2 * np.exp(x) / (np.exp(x) - 1 + 1e-10)**2
        
        # Velocity
        v = self.params['v_LA'] if mode == 'LA' else self.params['v_TA']
        
        # Lifetime
        tau = self.mode_lifetime(omega, T, mode)['total']
        
        # Conductivity contribution
        kappa = C * v**2 * tau
        
        return kappa
    
    def run_chromatography(self, T, n_modes=100):
        """
        Run phonon chromatography: separate thermal transport by mode.
        Returns 'elution profile' - conductivity vs frequency.
        """
        omega = np.linspace(0.1, self.omega_max, n_modes)
        
        results = {
            'omega_THz': omega.tolist(),
            'LA': [],
            'TA': [],
            'total': []
        }
        
        for w in omega:
            kappa_LA = self.mode_conductivity(w, T, 'LA')
            kappa_TA = 2 * self.mode_conductivity(w, T, 'TA')  # 2 TA branches
            
            results['LA'].append(kappa_LA)
            results['TA'].append(kappa_TA)
            results['total'].append(kappa_LA + kappa_TA)
        
        # Integrate for total conductivity
        results['kappa_total'] = np.trapz(results['total'], omega)
        results['kappa_LA'] = np.trapz(results['LA'], omega)
        results['kappa_TA'] = np.trapz(results['TA'], omega)
        
        return results
    
    def temperature_scan(self, T_range):
        """Run chromatography at multiple temperatures."""
        scans = []
        for T in T_range:
            result = self.run_chromatography(T)
            result['temperature_K'] = T
            scans.append(result)
        return scans


def visualize_pc_results():
    """Create visualization of Phonon Chromatograph results."""
    
    pc = PhononChromatograph()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Phonon Chromatograph (PC) Results', fontsize=14, color='#ff6600', y=0.98)
    
    # Plot 1: Elution profile at 300K
    ax = axes[0, 0]
    ax.set_title('Phonon "Elution Profile" at 300K', fontsize=10, color='#ff6600')
    
    result = pc.run_chromatography(300)
    omega = result['omega_THz']
    
    ax.fill_between(omega, result['LA'], alpha=0.5, color='#ff6600', label='LA branch')
    ax.fill_between(omega, result['TA'], alpha=0.5, color='#00ff00', label='TA branches')
    ax.plot(omega, result['total'], 'w-', linewidth=2, label='Total')
    
    ax.set_xlabel('Phonon frequency ω (THz)', fontsize=8)
    ax.set_ylabel('κ(ω) contribution', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, pc.omega_max)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean free path spectrum
    ax = axes[0, 1]
    ax.set_title('Mean Free Path Spectrum', fontsize=10, color='#00ffff')
    
    omega = np.linspace(0.1, pc.omega_max, 100)
    mfp_LA = [pc.mean_free_path(w, 300, 'LA') * 1e9 for w in omega]  # nm
    mfp_TA = [pc.mean_free_path(w, 300, 'TA') * 1e9 for w in omega]
    
    ax.semilogy(omega, mfp_LA, color='#ff6600', linewidth=2, label='LA')
    ax.semilogy(omega, mfp_TA, color='#00ff00', linewidth=2, label='TA')
    
    ax.axhline(1e6, color='#888888', linestyle='--', alpha=0.5, label='Sample size (1mm)')
    ax.axhline(1e3, color='#ffcc00', linestyle=':', alpha=0.5, label='1 μm')
    
    ax.set_xlabel('Phonon frequency ω (THz)', fontsize=8)
    ax.set_ylabel('Mean free path (nm)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Temperature evolution
    ax = axes[1, 0]
    ax.set_title('Elution Profile vs Temperature', fontsize=10, color='#00ff88')
    
    T_values = [100, 200, 300, 400, 500]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(T_values)))
    
    for T, c in zip(T_values, colors):
        result = pc.run_chromatography(T)
        ax.plot(result['omega_THz'], result['total'], color=c, 
               linewidth=2, label=f'T={T}K')
    
    ax.set_xlabel('Phonon frequency ω (THz)', fontsize=8)
    ax.set_ylabel('κ(ω) contribution', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Total conductivity decomposition
    ax = axes[1, 1]
    ax.set_title('Branch Contribution vs Temperature', fontsize=10, color='#ff00ff')
    
    T_range = np.linspace(50, 600, 30)
    scans = pc.temperature_scan(T_range)
    
    kappa_LA = [s['kappa_LA'] for s in scans]
    kappa_TA = [s['kappa_TA'] for s in scans]
    kappa_total = [s['kappa_total'] for s in scans]
    
    ax.fill_between(T_range, 0, kappa_LA, alpha=0.5, color='#ff6600', label='LA')
    ax.fill_between(T_range, kappa_LA, np.array(kappa_LA) + np.array(kappa_TA), 
                   alpha=0.5, color='#00ff00', label='TA')
    ax.plot(T_range, kappa_total, 'w-', linewidth=2, label='Total κ')
    
    ax.set_xlabel('Temperature (K)', fontsize=8)
    ax.set_ylabel('Thermal conductivity (arb. units)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('figures/panel_pc_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    plt.close(fig)
    
    return pc, scans


# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Phonon Chromatograph (PC)")
    print("Mode-Resolved Thermal Transport Analysis")
    print("=" * 60)
    
    pc, scans = visualize_pc_results()
    
    # Save data
    output_data = {
        'instrument': 'Phonon Chromatograph',
        'principle': 'Separates thermal transport by phonon mode (chromatography analogy)',
        'material': pc.params['name'],
        'omega_D_THz': pc.params['omega_D'],
        'v_LA_m_s': pc.params['v_LA'],
        'v_TA_m_s': pc.params['v_TA'],
        'Theta_D_K': pc.params['Theta_D'],
        'n_temperatures': len(scans),
        'temperature_range_K': [scans[0]['temperature_K'], scans[-1]['temperature_K']]
    }
    
    with open('data/pc_measurements.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nMaterial: {pc.params['name']}")
    print(f"Debye frequency: {pc.params['omega_D']} THz")
    print(f"Temperature range: {scans[0]['temperature_K']:.0f} - {scans[-1]['temperature_K']:.0f} K")
    print(f"\nGenerated: figures/panel_pc_results.png")
    print(f"Generated: data/pc_measurements.json")

