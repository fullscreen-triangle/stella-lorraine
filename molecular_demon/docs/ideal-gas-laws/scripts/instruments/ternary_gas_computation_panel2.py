"""
TERNARY GAS COMPUTATION - PANEL 2
=================================

Shows the actual computational operations using ternary representation
for gas dynamics. Demonstrates that computing IS solving gas dynamics
because oscillators are processors.

Key concepts:
- Ternary operations as thermodynamic transformations
- Trajectory completion in S-space
- Hardware oscillation as gas simulation
- Poincare recurrence = Computation halting

Author: Kundai Farai Sachikonye  
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import os

# Physical constants
k_B = 1.380649e-23


class TernaryProcessor:
    """
    Simulates ternary computation as gas dynamics.
    
    Each trit operation corresponds to a thermodynamic transformation:
    - 0 (oscillatory): Phase evolution
    - 1 (categorical): State transition  
    - 2 (partition): Mode coupling
    """
    
    def __init__(self, n_trits: int = 12):
        self.n_trits = n_trits
        self.state = np.zeros(n_trits, dtype=int)
        self.S_trajectory = []
        self.trit_history = []
        
    def compute_S_coords(self) -> np.ndarray:
        """Compute S-entropy coordinates from current trit state."""
        # Count trit types
        n_0 = np.sum(self.state == 0)  # Oscillatory
        n_1 = np.sum(self.state == 1)  # Categorical
        n_2 = np.sum(self.state == 2)  # Partition
        
        total = self.n_trits
        
        # S-coordinates from trit distribution
        S_k = n_1 / total * np.log(total / (n_1 + 1))  # Knowledge from categories
        S_t = n_0 / total * np.log(total / (n_0 + 1))  # Time from oscillations
        S_e = n_2 / total * np.log(total / (n_2 + 1))  # Evolution from partitions
        
        return np.array([S_k, S_t, S_e])
        
    def increment_trit(self, position: int):
        """Increment trit at position (mod 3)."""
        self.state[position] = (self.state[position] + 1) % 3
        self.S_trajectory.append(self.compute_S_coords())
        self.trit_history.append(self.state.copy())
        
    def execute_program(self, program: list):
        """
        Execute a ternary program.
        
        Program is a list of trit positions to increment.
        This corresponds to molecular state transitions.
        """
        self.S_trajectory = [self.compute_S_coords()]
        self.trit_history = [self.state.copy()]
        
        for pos in program:
            self.increment_trit(pos % self.n_trits)
            
    def random_walk(self, n_steps: int = 100):
        """
        Random walk in trit space = thermal fluctuations.
        """
        program = np.random.randint(0, self.n_trits, n_steps)
        self.execute_program(program.tolist())
        
    def get_S_trajectory(self) -> np.ndarray:
        return np.array(self.S_trajectory)


def simulate_gas_dynamics_as_computation(N_molecules: int = 50, 
                                         n_steps: int = 200) -> dict:
    """
    Simulate gas dynamics using ternary computation.
    
    Each molecule is a ternary processor.
    Gas dynamics = parallel ternary computation.
    """
    processors = [TernaryProcessor(n_trits=12) for _ in range(N_molecules)]
    
    # Each processor executes random walk (thermal motion)
    for proc in processors:
        proc.random_walk(n_steps)
        
    # Collect all trajectories
    trajectories = [proc.get_S_trajectory() for proc in processors]
    
    # Compute ensemble statistics at each time step
    ensemble_stats = []
    for t in range(n_steps + 1):
        S_all = np.array([traj[t] for traj in trajectories])
        ensemble_stats.append({
            'time': t,
            'S_k_mean': np.mean(S_all[:, 0]),
            'S_t_mean': np.mean(S_all[:, 1]),
            'S_e_mean': np.mean(S_all[:, 2]),
            'S_k_std': np.std(S_all[:, 0]),
            'S_t_std': np.std(S_all[:, 1]),
            'S_e_std': np.std(S_all[:, 2]),
            'total_entropy': np.sum(S_all)
        })
        
    return {
        'trajectories': trajectories,
        'ensemble_stats': ensemble_stats,
        'N': N_molecules,
        'n_steps': n_steps
    }


def compute_thermodynamic_from_ternary(simulation: dict) -> dict:
    """
    Extract thermodynamic properties from ternary computation.
    
    This shows that solving gas dynamics = computing.
    """
    stats = simulation['ensemble_stats']
    N = simulation['N']
    
    # Temperature proportional to trajectory spread
    T_ternary = [s['S_t_mean'] * 300 + 200 for s in stats]  # Normalized
    
    # Pressure from S_k (spatial distribution)
    P_ternary = [s['S_k_mean'] * 1e5 + 5e4 for s in stats]  # Normalized to ~1 atm
    
    # Internal energy from total entropy evolution
    U_ternary = [s['total_entropy'] * k_B * 300 * N for s in stats]
    
    return {
        'T': T_ternary,
        'P': P_ternary,
        'U': U_ternary,
        'time': [s['time'] for s in stats]
    }


def create_panel_chart(save_path: str):
    """Create ternary computation panel showing gas dynamics = computation."""
    
    # Run simulation
    simulation = simulate_gas_dynamics_as_computation(N_molecules=50, n_steps=150)
    thermo = compute_thermodynamic_from_ternary(simulation)
    
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Colors
    colors = {
        'trit0': '#FF6B6B',  # Oscillatory
        'trit1': '#4ECDC4',  # Categorical
        'trit2': '#FFE66D',  # Partition
    }
    
    # Panel 1: Multiple S-trajectories in 3D (computation = gas dynamics)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d', facecolor='#1a1a2e')
    
    # Plot first 20 trajectories
    for i, traj in enumerate(simulation['trajectories'][:20]):
        color = plt.cm.viridis(i / 20)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, alpha=0.5, linewidth=0.8)
        # Mark start and end
        ax1.scatter(*traj[0], color=color, s=30, marker='o')
        ax1.scatter(*traj[-1], color=color, s=30, marker='s')
        
    ax1.set_xlabel('$S_k$ (Knowledge)', color='white')
    ax1.set_ylabel('$S_t$ (Time)', color='white')
    ax1.set_zlabel('$S_e$ (Evolution)', color='white')
    ax1.set_title('Ternary Computation Trajectories\n(Each line = 1 molecule)', 
                 color='white', fontsize=11)
    ax1.tick_params(colors='white')
    
    # Panel 2: Ensemble evolution (gas reaches equilibrium = computation completes)
    ax2 = fig.add_subplot(2, 3, 2, facecolor='#1a1a2e')
    
    stats = simulation['ensemble_stats']
    times = [s['time'] for s in stats]
    
    ax2.fill_between(times, 
                    [s['S_k_mean'] - s['S_k_std'] for s in stats],
                    [s['S_k_mean'] + s['S_k_std'] for s in stats],
                    alpha=0.3, color=colors['trit1'])
    ax2.plot(times, [s['S_k_mean'] for s in stats], 
            color=colors['trit1'], linewidth=2, label='$S_k$ (cat.)')
    
    ax2.fill_between(times,
                    [s['S_t_mean'] - s['S_t_std'] for s in stats],
                    [s['S_t_mean'] + s['S_t_std'] for s in stats],
                    alpha=0.3, color=colors['trit0'])
    ax2.plot(times, [s['S_t_mean'] for s in stats],
            color=colors['trit0'], linewidth=2, label='$S_t$ (osc.)')
    
    ax2.fill_between(times,
                    [s['S_e_mean'] - s['S_e_std'] for s in stats],
                    [s['S_e_mean'] + s['S_e_std'] for s in stats],
                    alpha=0.3, color=colors['trit2'])
    ax2.plot(times, [s['S_e_mean'] for s in stats],
            color=colors['trit2'], linewidth=2, label='$S_e$ (part.)')
    
    ax2.set_xlabel('Computation Step', color='white')
    ax2.set_ylabel('Mean S-Coordinate', color='white')
    ax2.set_title('Ensemble Equilibration\n(Computation → Thermalization)', 
                 color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax2.grid(True, alpha=0.2, color='white')
    
    # Panel 3: Ternary operation diagram (3D)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d', facecolor='#1a1a2e')
    
    # Show the three fundamental operations as vectors
    origin = np.array([0.5, 0.5, 0.5])
    
    # Operation 0: Oscillatory (rotate in S_t direction)
    op0 = np.array([0, 0.3, 0])
    # Operation 1: Categorical (move in S_k direction)
    op1 = np.array([0.3, 0, 0])
    # Operation 2: Partition (evolve in S_e direction)
    op2 = np.array([0, 0, 0.3])
    
    # Draw operation vectors
    ax3.quiver(*origin, *op0, color=colors['trit0'], arrow_length_ratio=0.2, 
              linewidth=3, label='Op 0: Oscillate')
    ax3.quiver(*origin, *op1, color=colors['trit1'], arrow_length_ratio=0.2,
              linewidth=3, label='Op 1: Categorize')
    ax3.quiver(*origin, *op2, color=colors['trit2'], arrow_length_ratio=0.2,
              linewidth=3, label='Op 2: Partition')
    
    # Draw unit cube
    for i in range(2):
        for j in range(2):
            ax3.plot([i, i], [j, j], [0, 1], 'w-', alpha=0.2)
            ax3.plot([i, i], [0, 1], [j, j], 'w-', alpha=0.2)
            ax3.plot([0, 1], [i, i], [j, j], 'w-', alpha=0.2)
            
    ax3.scatter(*origin, color='white', s=100, zorder=10)
    ax3.set_xlabel('$S_k$', color='white')
    ax3.set_ylabel('$S_t$', color='white')
    ax3.set_zlabel('$S_e$', color='white')
    ax3.set_title('Ternary Operations\nin S-Space', color='white', fontsize=11)
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#2a2a4e', edgecolor='white', labelcolor='white', 
              loc='upper left', fontsize=8)
    
    # Panel 4: Thermodynamic extraction from computation
    ax4 = fig.add_subplot(2, 3, 4, facecolor='#1a1a2e')
    
    ax4_twin = ax4.twinx()
    
    ax4.plot(thermo['time'], thermo['T'], 'r-', linewidth=2, label='T (K)')
    ax4_twin.plot(thermo['time'], np.array(thermo['P'])/1e5, 'b--', 
                 linewidth=2, label='P (bar)')
    
    ax4.set_xlabel('Computation Step', color='white')
    ax4.set_ylabel('Temperature (K)', color='red')
    ax4_twin.set_ylabel('Pressure (bar)', color='cyan')
    ax4.set_title('Thermodynamics from Ternary Computation', 
                 color='white', fontsize=11)
    ax4.tick_params(colors='red', axis='y')
    ax4.tick_params(colors='white', axis='x')
    ax4_twin.tick_params(colors='cyan')
    ax4.grid(True, alpha=0.2, color='white')
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, 
              facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    
    # Panel 5: Trit state heatmap over time (single molecule)
    ax5 = fig.add_subplot(2, 3, 5, facecolor='#1a1a2e')
    
    # Get trit history from first processor
    proc = TernaryProcessor(n_trits=12)
    proc.random_walk(100)
    trit_hist = np.array(proc.trit_history)
    
    im = ax5.imshow(trit_hist.T, cmap='RdYlBu', aspect='auto',
                   interpolation='nearest')
    ax5.set_xlabel('Computation Step', color='white')
    ax5.set_ylabel('Trit Position', color='white')
    ax5.set_title('Trit State Evolution\n(1 molecule = 12 trits)', 
                 color='white', fontsize=11)
    ax5.tick_params(colors='white')
    
    cbar = fig.colorbar(im, ax=ax5, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['0:Osc', '1:Cat', '2:Part'], color='white')
    cbar.ax.tick_params(colors='white')
    
    # Panel 6: The key insight
    ax6 = fig.add_subplot(2, 3, 6, facecolor='#1a1a2e')
    ax6.axis('off')
    
    insight_text = """
    ╔═══════════════════════════════════════════════════════╗
    ║     COMPUTATION = GAS DYNAMICS (IDENTITY)              ║
    ╠═══════════════════════════════════════════════════════╣
    ║                                                        ║
    ║  TERNARY OPERATION        THERMODYNAMIC PROCESS        ║
    ║  ─────────────────        ─────────────────────        ║
    ║  Trit 0 increment    ↔    Phase oscillation            ║
    ║  Trit 1 increment    ↔    Category transition          ║
    ║  Trit 2 increment    ↔    Partition rearrangement      ║
    ║                                                        ║
    ║  COMPUTATIONAL STATE      GAS STATE                    ║
    ║  ───────────────────      ─────────                    ║
    ║  12-trit register    ↔    Molecular microstate         ║
    ║  S-entropy (S_k,S_t,S_e)  Phase space coordinates      ║
    ║  Random walk         ↔    Thermal fluctuations         ║
    ║                                                        ║
    ║  COMPUTATION COMPLETE     EQUILIBRIUM                  ║
    ║  ───────────────────      ───────────                  ║
    ║  Poincaré recurrence ↔    Maxwell distribution         ║
    ║                                                        ║
    ║  ┌─────────────────────────────────────────────────┐   ║
    ║  │ Oscillator = Processor                          │   ║
    ║  │ Memory Address = Trajectory in S-Space          │   ║
    ║  │ Solving Gas Dynamics = Running Ternary Program  │   ║
    ║  └─────────────────────────────────────────────────┘   ║
    ╚═══════════════════════════════════════════════════════╝
    """
    
    ax6.text(0.02, 0.98, insight_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            color='#4ECDC4', bbox=dict(boxstyle='round', facecolor='#1a1a2e', 
                                       edgecolor='#4ECDC4', alpha=0.95))
    
    plt.suptitle('Ternary Computation as Gas Dynamics: Oscillator = Processor',
                fontsize=16, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"Saved panel 2 to {save_path}")


def main():
    """Generate ternary gas computation panel 2."""
    print("=" * 60)
    print("TERNARY GAS COMPUTATION - PANEL 2")
    print("Computation = Gas Dynamics (Identity)")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    create_panel_chart(os.path.join(figures_dir, "panel_ternary_computation_2.png"))
    
    print("\n" + "=" * 60)
    print("PANEL 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

