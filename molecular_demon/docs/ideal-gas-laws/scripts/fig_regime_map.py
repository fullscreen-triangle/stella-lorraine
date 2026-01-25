"""
Figure 11: Regime Map
2D phase diagram showing different thermodynamic regimes:
- Quantum, Classical, Relativistic, Saturation, Planck
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import json
import os

# Constants
k_B = 1.380649e-23  # J/K
hbar = 1.054571817e-34  # J·s
c = 299792458  # m/s
G = 6.67430e-11  # m³/(kg·s²)

# Planck temperature
T_Planck = np.sqrt(hbar * c**5 / (G * k_B**2))  # ~1.4e32 K

# Style settings
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

def create_figure():
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Temperature range: 10^-3 to 10^33 K
    # Density range: 10^-10 to 10^35 particles/m³
    T_min, T_max = 1e-3, 1e33
    rho_min, rho_max = 1e-10, 1e35
    
    ax.set_xlim(T_min, T_max)
    ax.set_ylim(rho_min, rho_max)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Define regions with colors
    regions = [
        # (T_low, T_high, rho_low, rho_high, color, label, text_pos)
        {'T': (1e-3, 10), 'rho': (1e-10, 1e22), 'color': '#a8d5e5', 
         'label': 'Quantum Regime', 'text': (0.3, 1e15), 'note': 'Velocity quantization\nobservable'},
        {'T': (10, 1e6), 'rho': (1e15, 1e26), 'color': '#90EE90',
         'label': 'Classical Regime', 'text': (1e3, 1e21), 'note': 'Categorical = Classical\n(within 0.1%)'},
        {'T': (1e6, 1e32), 'rho': (1e-10, 1e28), 'color': '#FFD580',
         'label': 'Relativistic Regime', 'text': (1e10, 1e10), 'note': 'Cutoff at v = c'},
        {'T': (1e-3, 1e33), 'rho': (1e28, 1e35), 'color': '#FF6B6B',
         'label': 'Saturation Regime', 'text': (1e5, 1e32), 'note': 'P saturates\nM approaches M_max'},
        {'T': (1e31, 1e33), 'rho': (1e-10, 1e35), 'color': '#9B59B6',
         'label': 'Planck Regime', 'text': (5e31, 1e2), 'note': 'T_max = T_Planck'},
    ]
    
    # Draw regions (overlapping, so order matters)
    for reg in regions:
        T_low, T_high = reg['T']
        rho_low, rho_high = reg['rho']
        
        # Create rectangle in log space
        width = np.log10(T_high) - np.log10(T_low)
        height = np.log10(rho_high) - np.log10(rho_low)
        
        rect = plt.Rectangle((T_low, rho_low), T_high - T_low, rho_high - rho_low,
                              facecolor=reg['color'], alpha=0.4, edgecolor='black',
                              linewidth=0.5, transform=ax.transData)
        ax.add_patch(rect)
        
        # Add text
        ax.text(reg['text'][0], reg['text'][1], 
                f"{reg['label']}\n{reg['note']}", 
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Characteristic lines
    # Debye temperature line (approximate)
    T_Debye = np.logspace(-3, 5, 100)
    rho_Debye = 1e22 * (T_Debye / 300)**0  # Roughly constant
    ax.plot([300, 300], [1e15, 1e28], '--', color='blue', linewidth=1.5, label='Debye T (~300 K)')
    
    # Fermi temperature line (approximate for electrons)
    T_Fermi = np.logspace(3, 6, 100)
    ax.plot([1e4, 1e4], [1e25, 1e35], '--', color='green', linewidth=1.5, label='Fermi T')
    
    # Planck temperature line
    ax.axvline(x=T_Planck, color='purple', linestyle=':', linewidth=2, label='Planck T')
    ax.text(T_Planck*1.5, 1e-5, f'$T_{{Planck}}$\n$1.4\\times10^{{32}}$ K', 
            fontsize=8, color='purple')
    
    # Nuclear density line
    ax.axhline(y=1e44/1e15, color='red', linestyle=':', linewidth=1.5, label='Nuclear density')
    ax.text(1e-2, 3e29, 'Nuclear density', fontsize=8, color='red')
    
    # Data points for real systems
    systems = [
        {'name': 'Room air', 'T': 300, 'rho': 2.5e25, 'marker': 'o'},
        {'name': 'Liquid He', 'T': 4, 'rho': 2.2e28, 'marker': 's'},
        {'name': 'Sun core', 'T': 1.5e7, 'rho': 1.5e32, 'marker': '^'},
        {'name': 'Ultra-cold atoms', 'T': 1e-6, 'rho': 1e18, 'marker': 'D'},
        {'name': 'RHIC QGP', 'T': 2e12, 'rho': 1e32, 'marker': 'p'},
        {'name': 'Early universe', 'T': 1e10, 'rho': 1e15, 'marker': '*'},
        {'name': 'Neutron star', 'T': 1e8, 'rho': 1e44, 'marker': 'h'},
        {'name': 'BEC', 'T': 1e-7, 'rho': 1e19, 'marker': 'v'},
    ]
    
    for sys in systems:
        ax.scatter(sys['T'], sys['rho'], marker=sys['marker'], s=100, 
                   c='black', edgecolors='white', linewidth=1.5, zorder=10)
        
        # Add label with offset
        offset = (1.5, 1.5)
        if sys['name'] == 'Neutron star':
            offset = (0.3, 0.5)
        ax.annotate(sys['name'], (sys['T'], sys['rho']), 
                    xytext=(sys['T']*offset[0], sys['rho']*offset[1]),
                    fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    ax.set_xlabel('Temperature T (K)', fontsize=12)
    ax.set_ylabel('Density $\\rho$ (particles/m$^3$)', fontsize=12)
    ax.set_title('Thermodynamic Regime Map: Categorical vs Classical', fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='#a8d5e5', alpha=0.6, label='Quantum'),
        plt.Rectangle((0,0), 1, 1, facecolor='#90EE90', alpha=0.6, label='Classical'),
        plt.Rectangle((0,0), 1, 1, facecolor='#FFD580', alpha=0.6, label='Relativistic'),
        plt.Rectangle((0,0), 1, 1, facecolor='#FF6B6B', alpha=0.6, label='Saturation'),
        plt.Rectangle((0,0), 1, 1, facecolor='#9B59B6', alpha=0.6, label='Planck'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    
    plt.tight_layout()
    return fig

def save_data():
    """Save regime boundaries to JSON"""
    data = {
        'description': 'Thermodynamic regime boundaries',
        'regimes': {
            'quantum': {'T_max_K': 10, 'note': 'Velocity quantization observable'},
            'classical': {'T_range_K': [10, 1e6], 'note': 'Categorical = Classical'},
            'relativistic': {'T_min_K': 1e6, 'note': 'v = c cutoff active'},
            'saturation': {'rho_min_m3': 1e28, 'note': 'Categorical saturation'},
            'planck': {'T_K': 1.4e32, 'note': 'Maximum temperature'}
        },
        'characteristic_systems': [
            {'name': 'Room air', 'T_K': 300, 'rho_m3': 2.5e25},
            {'name': 'Liquid He', 'T_K': 4, 'rho_m3': 2.2e28},
            {'name': 'Sun core', 'T_K': 1.5e7, 'rho_m3': 1.5e32},
        ]
    }
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, '..', 'figures', 'fig_regime_map.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    fig = create_figure()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(output_dir, '..', 'figures', 'fig_regime_map.png')
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    save_data()
    plt.show()

