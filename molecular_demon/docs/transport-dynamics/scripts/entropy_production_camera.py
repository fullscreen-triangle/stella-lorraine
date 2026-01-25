"""
Entropy Production Camera (EPC)
Real-time visualization of entropy production during transport
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
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

class EntropyProductionCamera:
    """
    Maps entropy production during transport processes.
    Visualizes where dissipation occurs spatially.
    """
    
    def __init__(self, grid_size=(50, 50)):
        self.Nx, self.Ny = grid_size
        self.x = np.linspace(0, 10, self.Nx)  # mm
        self.y = np.linspace(0, 8, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def compute_entropy_production(self, T_field, J_field, sigma_field):
        """
        Compute local entropy production rate.
        σ̇ = J · ∇(1/T) for heat transport
        σ̇ = J² / (σT) for electrical transport
        """
        # Temperature gradient
        dTdx = np.gradient(T_field, self.x, axis=1)
        dTdy = np.gradient(T_field, self.y, axis=0)
        
        # Entropy production: σ̇ = κ|∇T|²/T²
        # Simplified: proportional to |∇T|²/T
        grad_T_sq = dTdx**2 + dTdy**2
        S_dot = grad_T_sq / (T_field + 1)**2
        
        return S_dot
    
    def create_temperature_field(self, scenario='gradient'):
        """Create temperature field for different scenarios."""
        if scenario == 'gradient':
            # Simple left-right gradient
            T = 400 - 20 * self.X
            
        elif scenario == 'hotspot':
            # Central hot spot
            r = np.sqrt((self.X - 5)**2 + (self.Y - 4)**2)
            T = 300 + 100 * np.exp(-r**2 / 4)
            
        elif scenario == 'defect':
            # Gradient with defect (high resistance region)
            T = 400 - 20 * self.X
            defect_mask = ((self.X - 5)**2 + (self.Y - 4)**2 < 1)
            T[defect_mask] += 50  # Hot spot at defect
            
        elif scenario == 'superconductor':
            # Superconducting region (no entropy production)
            T = 400 - 20 * self.X
            sc_region = self.X < 3
            T[sc_region] = T[sc_region].mean()  # Isothermal in SC region
            
        return T
    
    def create_current_field(self, T_field, rho_field):
        """Create current density field."""
        # J = -κ∇T for heat
        # Simplified: J proportional to -∇T
        dTdx = np.gradient(T_field, self.x, axis=1)
        dTdy = np.gradient(T_field, self.y, axis=0)
        
        Jx = -dTdx / rho_field
        Jy = -dTdy / rho_field
        
        return Jx, Jy
    
    def scan_scenarios(self):
        """Scan multiple transport scenarios."""
        scenarios = ['gradient', 'hotspot', 'defect', 'superconductor']
        results = {}
        
        for scenario in scenarios:
            T = self.create_temperature_field(scenario)
            
            # Resistivity field (higher near defects)
            rho = np.ones_like(T)
            if scenario == 'defect':
                defect_mask = ((self.X - 5)**2 + (self.Y - 4)**2 < 1)
                rho[defect_mask] = 5
            elif scenario == 'superconductor':
                rho[self.X < 3] = 0.001  # Near-zero resistance
            
            Jx, Jy = self.create_current_field(T, rho)
            S_dot = self.compute_entropy_production(T, (Jx, Jy), rho)
            
            results[scenario] = {
                'T': T,
                'rho': rho,
                'Jx': Jx,
                'Jy': Jy,
                'S_dot': S_dot,
                'total_S_dot': np.sum(S_dot)
            }
        
        return results


def visualize_epc_results():
    """Create visualization of EPC measurements."""
    
    epc = EntropyProductionCamera()
    results = epc.scan_scenarios()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Entropy Production Camera (EPC) Results', fontsize=14, color='#ff4400', y=0.98)
    
    scenarios = ['gradient', 'hotspot', 'defect', 'superconductor']
    titles = [
        'Uniform Temperature Gradient',
        'Central Hot Spot',
        'Defect (High Resistance Region)',
        'Superconducting Region (x < 3mm)'
    ]
    
    for ax, scenario, title in zip(axes.flat, scenarios, titles):
        data = results[scenario]
        
        # Plot entropy production rate
        im = ax.pcolormesh(epc.X, epc.Y, data['S_dot'], 
                          cmap='inferno', shading='auto')
        
        # Overlay temperature contours
        cs = ax.contour(epc.X, epc.Y, data['T'], levels=10, 
                       colors='white', alpha=0.3, linewidths=0.5)
        
        # Overlay current vectors (subsampled)
        skip = 5
        ax.quiver(epc.X[::skip, ::skip], epc.Y[::skip, ::skip],
                 data['Jx'][::skip, ::skip], data['Jy'][::skip, ::skip],
                 color='cyan', alpha=0.5, scale=50)
        
        ax.set_title(f'{title}\nΣσ̇ = {data["total_S_dot"]:.2f}', fontsize=10, color='#ff4400')
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        
        plt.colorbar(im, ax=ax, label='σ̇ (entropy production rate)')
        
        if scenario == 'superconductor':
            ax.axvline(3, color='#00ffff', linestyle='--', linewidth=2)
            ax.text(1.5, 7, 'SC\nσ̇ = 0', fontsize=10, color='#00ffff', ha='center')
    
    plt.tight_layout()
    fig.savefig('figures/panel_epc_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    plt.close(fig)
    
    # Create comparison bar chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    ax.set_title('Total Entropy Production by Scenario', fontsize=12, color='#ff4400')
    
    names = ['Gradient', 'Hot Spot', 'Defect', 'Superconductor']
    values = [results[s]['total_S_dot'] for s in scenarios]
    colors = ['#ff6600', '#ffcc00', '#ff0000', '#00ffff']
    
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel('Total entropy production Σσ̇', fontsize=10, color='white')
    ax.tick_params(colors='white')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, 
               f'{val:.1f}', ha='center', fontsize=9, color='white')
    
    plt.tight_layout()
    fig2.savefig('figures/panel_epc_comparison.png', dpi=150, bbox_inches='tight',
                 facecolor='#0a0a0a', edgecolor='none')
    plt.close(fig2)
    
    return epc, results


# Main execution
if __name__ == '__main__':
    print("=" * 60)
    print("Entropy Production Camera (EPC)")
    print("Real-Time Dissipation Mapping")
    print("=" * 60)
    
    epc, results = visualize_epc_results()
    
    # Save data
    output_data = {
        'instrument': 'Entropy Production Camera',
        'principle': 'Maps local entropy production σ̇ = J·∇(1/T)',
        'grid_size': [epc.Nx, epc.Ny],
        'domain_mm': [10, 8],
        'scenarios': {}
    }
    
    for scenario in results:
        output_data['scenarios'][scenario] = {
            'total_entropy_production': float(results[scenario]['total_S_dot']),
            'max_T_K': float(results[scenario]['T'].max()),
            'min_T_K': float(results[scenario]['T'].min())
        }
    
    with open('data/epc_measurements.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\nScenario Analysis:")
    print("-" * 40)
    for scenario in results:
        print(f"{scenario:15s}: S_dot_total = {results[scenario]['total_S_dot']:.2f}")
    
    print(f"\nGenerated: figures/panel_epc_results.png")
    print(f"Generated: figures/panel_epc_comparison.png")
    print(f"Generated: data/epc_measurements.json")

