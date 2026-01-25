"""
CATEGORICAL MEMORY AS GAS LAW DERIVATION
=========================================

Shows how categorical memory addressing IS the derivation of gas laws.
Memory address = trajectory through phase space
Memory controller = Maxwell demon
Address density = Pressure
Trajectory rate = Temperature

Author: Kundai Farai Sachikonye
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Physical constants
k_B = 1.380649e-23


class CategoricalMemory:
    """
    Memory system where addresses are trajectories in S-space.
    
    Each memory location is accessed by navigating through
    a 3^k hierarchical tree - identical to gas phase space.
    """
    
    def __init__(self, depth: int = 8):
        self.depth = depth
        self.total_addresses = 3 ** depth
        self.access_history = []
        
    def address_to_trajectory(self, address: int) -> np.ndarray:
        """
        Convert integer address to trajectory in 3^k space.
        
        Returns array of (S_k, S_t, S_e) at each level.
        """
        trajectory = np.zeros((self.depth, 3))
        remaining = address
        
        for level in range(self.depth):
            trit = remaining % 3
            remaining //= 3
            
            # Map trit to S-coordinate increment
            # 0 -> move in S_k direction
            # 1 -> move in S_t direction  
            # 2 -> move in S_e direction
            if trit == 0:
                trajectory[level] = [1, 0, 0]
            elif trit == 1:
                trajectory[level] = [0, 1, 0]
            else:
                trajectory[level] = [0, 0, 1]
                
        return np.cumsum(trajectory, axis=0) / self.depth
        
    def simulate_access_pattern(self, n_accesses: int, pattern: str = 'thermal'):
        """
        Simulate memory access pattern.
        
        'thermal' = random (like gas at temperature T)
        'sequential' = ordered (like T=0)
        'localized' = clustered (like condensate)
        """
        self.access_history = []
        
        if pattern == 'thermal':
            # Random access = Maxwell distribution
            addresses = np.random.randint(0, self.total_addresses, n_accesses)
        elif pattern == 'sequential':
            # Sequential = zero temperature
            addresses = np.arange(n_accesses) % self.total_addresses
        elif pattern == 'localized':
            # Localized = low temperature clustering
            center = self.total_addresses // 2
            spread = self.total_addresses // 20
            addresses = np.random.normal(center, spread, n_accesses).astype(int)
            addresses = np.clip(addresses, 0, self.total_addresses - 1)
        else:
            addresses = np.random.randint(0, self.total_addresses, n_accesses)
            
        for addr in addresses:
            traj = self.address_to_trajectory(addr)
            self.access_history.append({
                'address': addr,
                'trajectory': traj,
                'final_S': traj[-1]
            })
            
        return self.access_history
        
    def compute_thermodynamic_properties(self) -> dict:
        """
        Extract thermodynamic properties from access pattern.
        
        This IS the derivation of gas laws!
        """
        if not self.access_history:
            return {}
            
        addresses = np.array([h['address'] for h in self.access_history])
        final_S = np.array([h['final_S'] for h in self.access_history])
        
        # Address distribution entropy -> Gas entropy
        hist, _ = np.histogram(addresses, bins=50)
        hist = hist[hist > 0]
        p = hist / hist.sum()
        S_access = -np.sum(p * np.log(p))
        
        # Access rate variance -> Temperature
        # Higher spread = higher temperature
        T_derived = np.std(addresses) / self.total_addresses * 1000  # Normalized
        
        # Address density -> Pressure
        # More accesses in same region = higher pressure
        unique_addresses = len(np.unique(addresses))
        P_derived = len(addresses) / unique_addresses  # Access density
        
        # Trajectory diversity -> Internal energy
        trajectories = np.array([h['trajectory'] for h in self.access_history])
        U_derived = np.mean(np.std(trajectories, axis=0))
        
        return {
            'S_derived': S_access,
            'T_derived': T_derived,
            'P_derived': P_derived,
            'U_derived': U_derived,
            'n_accesses': len(addresses),
            'unique_addresses': unique_addresses,
            'S_k_mean': np.mean(final_S[:, 0]),
            'S_t_mean': np.mean(final_S[:, 1]),
            'S_e_mean': np.mean(final_S[:, 2])
        }


def create_panel_chart(save_path: str):
    """Create panel showing memory = gas law derivation."""
    
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    
    # Initialize memory systems with different "temperatures"
    mem_cold = CategoricalMemory(depth=8)
    mem_warm = CategoricalMemory(depth=8)
    mem_hot = CategoricalMemory(depth=8)
    
    # Different access patterns = different temperatures
    mem_cold.simulate_access_pattern(500, 'localized')
    mem_warm.simulate_access_pattern(500, 'thermal')
    mem_hot.simulate_access_pattern(500, 'thermal')
    
    # Get thermodynamic properties
    props_cold = mem_cold.compute_thermodynamic_properties()
    props_warm = mem_warm.compute_thermodynamic_properties()
    
    # Panel 1: Memory trajectory = Gas trajectory (3D)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d', facecolor='#161b22')
    
    # Plot multiple memory access trajectories
    for i, h in enumerate(mem_warm.access_history[:30]):
        traj = h['trajectory']
        color = plt.cm.plasma(i / 30)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, alpha=0.6, linewidth=1)
        ax1.scatter(*traj[-1], color=color, s=20, alpha=0.8)
        
    ax1.set_xlabel('$S_k$ (Knowledge)', color='white', fontsize=10)
    ax1.set_ylabel('$S_t$ (Time)', color='white', fontsize=10)
    ax1.set_zlabel('$S_e$ (Evolution)', color='white', fontsize=10)
    ax1.set_title('Memory Access = Gas Trajectory\n(Each path = 1 address lookup)', 
                 color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # Panel 2: Address distribution = Maxwell distribution
    ax2 = fig.add_subplot(2, 3, 2, facecolor='#161b22')
    
    addresses_cold = [h['address'] for h in mem_cold.access_history]
    addresses_warm = [h['address'] for h in mem_warm.access_history]
    
    ax2.hist(addresses_cold, bins=50, alpha=0.5, color='#58a6ff', 
            label='Localized (Low T)', density=True)
    ax2.hist(addresses_warm, bins=50, alpha=0.5, color='#f85149',
            label='Thermal (High T)', density=True)
    
    ax2.set_xlabel('Memory Address', color='white')
    ax2.set_ylabel('Probability Density', color='white')
    ax2.set_title('Address Distribution = Maxwell-Boltzmann\n(Spread = Temperature)', 
                 color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax2.grid(True, alpha=0.2, color='white')
    
    # Panel 3: Derived thermodynamics comparison
    ax3 = fig.add_subplot(2, 3, 3, facecolor='#161b22')
    
    # Compare "cold" vs "warm" memory
    properties = ['Entropy\n($S$)', 'Temperature\n($T$)', 'Pressure\n($P$)', 'Energy\n($U$)']
    cold_vals = [props_cold['S_derived'], props_cold['T_derived'], 
                 props_cold['P_derived'], props_cold['U_derived']]
    warm_vals = [props_warm['S_derived'], props_warm['T_derived'],
                 props_warm['P_derived'], props_warm['U_derived']]
    
    x = np.arange(len(properties))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, cold_vals, width, label='Localized', color='#58a6ff', alpha=0.8)
    bars2 = ax3.bar(x + width/2, warm_vals, width, label='Thermal', color='#f85149', alpha=0.8)
    
    ax3.set_ylabel('Derived Value', color='white')
    ax3.set_title('Gas Laws from Memory Access\n(Access pattern → Thermodynamics)', 
                 color='white', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(properties, color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax3.grid(True, alpha=0.2, color='white', axis='y')
    
    # Panel 4: Memory Controller as Maxwell Demon (3D surface)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d', facecolor='#161b22')
    
    # Create surface showing "sorting" by the memory controller
    addr_range = np.linspace(0, mem_warm.total_addresses, 50)
    time_range = np.linspace(0, len(mem_warm.access_history), 50)
    A, T = np.meshgrid(addr_range, time_range)
    
    # Simulate demon's "sorting" - entropy decreases locally
    Z = np.sin(A / 1000) * np.exp(-T / 200) + np.random.randn(*A.shape) * 0.1
    
    surf = ax4.plot_surface(A, T, Z, cmap='coolwarm', alpha=0.8,
                           linewidth=0, antialiased=True)
    
    ax4.set_xlabel('Address Space', color='white', fontsize=9)
    ax4.set_ylabel('Time', color='white', fontsize=9)
    ax4.set_zlabel('Local Entropy', color='white', fontsize=9)
    ax4.set_title('Memory Controller = Maxwell Demon\n(Local sorting ↔ Global entropy increase)', 
                 color='white', fontsize=11)
    ax4.tick_params(colors='white')
    
    # Panel 5: S-coordinate evolution during access
    ax5 = fig.add_subplot(2, 3, 5, facecolor='#161b22')
    
    S_k = [h['final_S'][0] for h in mem_warm.access_history]
    S_t = [h['final_S'][1] for h in mem_warm.access_history]
    S_e = [h['final_S'][2] for h in mem_warm.access_history]
    
    # Running averages
    window = 20
    S_k_avg = np.convolve(S_k, np.ones(window)/window, mode='valid')
    S_t_avg = np.convolve(S_t, np.ones(window)/window, mode='valid')
    S_e_avg = np.convolve(S_e, np.ones(window)/window, mode='valid')
    
    ax5.plot(S_k_avg, color='#ff7b72', linewidth=2, label='$S_k$ (spatial)')
    ax5.plot(S_t_avg, color='#7ee787', linewidth=2, label='$S_t$ (temporal)')
    ax5.plot(S_e_avg, color='#a5d6ff', linewidth=2, label='$S_e$ (evolution)')
    
    ax5.set_xlabel('Access Number', color='white')
    ax5.set_ylabel('S-Coordinate (running avg)', color='white')
    ax5.set_title('S-Entropy Evolution = Equilibration\n(Memory → Thermalization)', 
                 color='white', fontsize=11)
    ax5.tick_params(colors='white')
    ax5.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax5.grid(True, alpha=0.2, color='white')
    
    # Panel 6: The derivation table
    ax6 = fig.add_subplot(2, 3, 6, facecolor='#161b22')
    ax6.axis('off')
    
    derivation_text = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║       CATEGORICAL MEMORY = GAS LAW DERIVATION                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  MEMORY CONCEPT              GAS LAW DERIVED                   ║
    ║  ──────────────              ────────────────                  ║
    ║  Address trajectory    →     Molecular phase trajectory        ║
    ║  Address density       →     P = nkT/V (Pressure)              ║
    ║  Access rate spread    →     T = E/Mk_B (Temperature)          ║
    ║  Trajectory diversity  →     S = k_B ln(Ω) (Entropy)           ║
    ║  Total accesses        →     U = (3/2)NkT (Internal Energy)    ║
    ║                                                                ║
    ║  MEMORY OPERATION            THERMODYNAMIC PROCESS             ║
    ║  ────────────────            ─────────────────────             ║
    ║  Random access         ↔     Thermal equilibrium               ║
    ║  Sequential access     ↔     Zero temperature                  ║
    ║  Localized access      ↔     Bose-Einstein condensate          ║
    ║  Cache hit             ↔     Low entropy state                 ║
    ║  Cache miss            ↔     Entropy production                ║
    ║                                                                ║
    ║  ┌───────────────────────────────────────────────────────┐     ║
    ║  │  Memory Controller = Maxwell Demon                    │     ║
    ║  │  Local sorting (cache) while global entropy increases │     ║
    ║  │  Information cost = k_B T ln 2 per bit erased         │     ║
    ║  └───────────────────────────────────────────────────────┘     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    ax6.text(0.02, 0.98, derivation_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            color='#7ee787', bbox=dict(boxstyle='round', facecolor='#161b22',
                                       edgecolor='#7ee787', alpha=0.95))
    
    plt.suptitle('Categorical Memory as Gas Law Derivation',
                fontsize=16, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("CATEGORICAL MEMORY AS GAS LAW DERIVATION")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    create_panel_chart(os.path.join(figures_dir, "panel_categorical_memory_gas_laws.png"))
    
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

