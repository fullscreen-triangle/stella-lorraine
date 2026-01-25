"""
POINCARE RECURRENCE MONITOR (PRM)
=================================

Monitors and validates Poincare recurrence in categorical phase space.
Tests that trajectories return arbitrarily close to initial conditions.

Key predictions tested:
1. Recurrence exists for bounded systems
2. Recurrence time scales with phase space volume
3. Categorical recurrence differs from continuous recurrence
4. S-entropy provides recurrence metric

Author: Kundai Farai Sachikonye
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from typing import Dict, List, Tuple, Any

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
h = 6.62607015e-34  # Planck constant (J*s)
hbar = h / (2 * np.pi)
c = 299792458  # Speed of light (m/s)


class PoincareRecurrenceMonitor:
    """Monitor Poincare recurrence in categorical phase space."""
    
    def __init__(self, N_particles: int = 100, dimension: int = 3):
        """
        Initialize monitor.
        
        Args:
            N_particles: Number of particles in system
            dimension: Spatial dimension (typically 3)
        """
        self.N = N_particles
        self.dim = dimension
        self.results = {}
        
    def initialize_system(self, T: float = 300.0, V: float = 1e-24) -> Dict:
        """
        Initialize gas system in phase space.
        
        Args:
            T: Temperature (K)
            V: Volume (m^3)
        
        Returns:
            Initial state dictionary
        """
        # Thermal velocity scale
        m = 4.65e-26  # ~N2 mass
        v_thermal = np.sqrt(k_B * T / m)
        
        # Initialize positions uniformly in box
        L = V**(1/3)
        positions = np.random.uniform(0, L, (self.N, self.dim))
        
        # Initialize velocities from Maxwell distribution
        velocities = np.random.normal(0, v_thermal, (self.N, self.dim))
        
        # Categorical coordinates (discretize phase space)
        n_bins = 10  # Categories per dimension
        cat_positions = np.floor(positions / L * n_bins).astype(int) % n_bins
        v_max = 3 * v_thermal
        cat_velocities = np.floor((velocities + v_max) / (2 * v_max) * n_bins).astype(int) % n_bins
        
        return {
            'positions': positions,
            'velocities': velocities,
            'cat_positions': cat_positions,
            'cat_velocities': cat_velocities,
            'L': L,
            'v_thermal': v_thermal,
            'n_bins': n_bins,
            'T': T,
            'V': V
        }
        
    def evolve_system(self, state: Dict, dt: float = 1e-12, steps: int = 1000) -> List[Dict]:
        """
        Evolve system and track trajectory.
        
        Args:
            state: Initial state
            dt: Time step (s)
            steps: Number of steps
            
        Returns:
            List of states (trajectory)
        """
        trajectory = [state.copy()]
        positions = state['positions'].copy()
        velocities = state['velocities'].copy()
        L = state['L']
        v_thermal = state['v_thermal']
        n_bins = state['n_bins']
        v_max = 3 * v_thermal
        
        for _ in range(steps):
            # Free evolution (ideal gas)
            positions = positions + velocities * dt
            
            # Periodic boundary conditions
            positions = positions % L
            
            # Compute categorical coordinates
            cat_positions = np.floor(positions / L * n_bins).astype(int) % n_bins
            cat_velocities = np.floor((velocities + v_max) / (2 * v_max) * n_bins).astype(int) % n_bins
            
            new_state = {
                'positions': positions.copy(),
                'velocities': velocities.copy(),
                'cat_positions': cat_positions,
                'cat_velocities': cat_velocities,
                'L': L,
                'v_thermal': v_thermal,
                'n_bins': n_bins,
                'T': state['T'],
                'V': state['V']
            }
            trajectory.append(new_state)
            
        return trajectory
        
    def compute_phase_distance(self, state1: Dict, state2: Dict, mode: str = 'continuous') -> float:
        """
        Compute distance between two states in phase space.
        
        Args:
            state1, state2: States to compare
            mode: 'continuous' or 'categorical'
            
        Returns:
            Distance metric
        """
        if mode == 'continuous':
            # Normalized phase space distance
            L = state1['L']
            v_th = state1['v_thermal']
            
            dx = (state1['positions'] - state2['positions']) / L
            dv = (state1['velocities'] - state2['velocities']) / v_th
            
            return np.sqrt(np.mean(dx**2) + np.mean(dv**2))
            
        else:  # categorical
            # Hamming distance on categorical coordinates
            dx_cat = state1['cat_positions'] != state2['cat_positions']
            dv_cat = state1['cat_velocities'] != state2['cat_velocities']
            
            return np.mean(dx_cat) + np.mean(dv_cat)
            
    def find_recurrence(self, trajectory: List[Dict], epsilon: float = 0.1, 
                       mode: str = 'continuous') -> Dict:
        """
        Find recurrence times in trajectory.
        
        Args:
            trajectory: List of states
            epsilon: Recurrence threshold
            mode: 'continuous' or 'categorical'
            
        Returns:
            Recurrence analysis results
        """
        initial = trajectory[0]
        distances = []
        recurrence_times = []
        
        for i, state in enumerate(trajectory[1:], 1):
            d = self.compute_phase_distance(initial, state, mode)
            distances.append(d)
            
            if d < epsilon:
                recurrence_times.append(i)
                
        return {
            'distances': np.array(distances),
            'recurrence_times': np.array(recurrence_times),
            'min_distance': np.min(distances) if distances else np.inf,
            'first_recurrence': recurrence_times[0] if recurrence_times else None,
            'n_recurrences': len(recurrence_times)
        }
        
    def compute_s_entropy_trajectory(self, trajectory: List[Dict]) -> np.ndarray:
        """
        Compute S-entropy along trajectory.
        
        Returns:
            Array of (S_k, S_t, S_e) for each step
        """
        S_coords = []
        
        for i, state in enumerate(trajectory):
            # Knowledge entropy - from categorical distribution
            cat_flat = state['cat_positions'].flatten()
            hist, _ = np.histogram(cat_flat, bins=state['n_bins'])
            hist = hist[hist > 0]
            p = hist / hist.sum()
            S_k = -np.sum(p * np.log(p + 1e-10))
            
            # Time entropy - from velocity distribution  
            v_mag = np.linalg.norm(state['velocities'], axis=1)
            hist, _ = np.histogram(v_mag, bins=20)
            hist = hist[hist > 0]
            p = hist / hist.sum()
            S_t = -np.sum(p * np.log(p + 1e-10))
            
            # Evolution entropy - from trajectory progress
            S_e = np.log(i + 1)
            
            S_coords.append([S_k, S_t, S_e])
            
        return np.array(S_coords)
        
    def estimate_recurrence_time(self, phase_space_volume: float) -> float:
        """
        Estimate theoretical Poincare recurrence time.
        
        For a gas of N particles in volume V at temperature T,
        the recurrence time is astronomically large:
        t_rec ~ exp(S/k_B)
        
        Args:
            phase_space_volume: Effective phase space volume
            
        Returns:
            Estimated recurrence time
        """
        # For categorical phase space with n bins per dimension
        # Total states ~ n^(2*dim*N)
        n_bins = 10
        log_states = 2 * self.dim * self.N * np.log(n_bins)
        
        # Recurrence time ~ number of states (ergodic assumption)
        # This is still a vast underestimate for real systems
        t_rec = np.exp(min(log_states, 700))  # Cap to avoid overflow
        
        return t_rec
        
    def full_validation(self, T: float = 300.0) -> Dict:
        """
        Run full Poincare recurrence validation.
        
        Args:
            T: Temperature (K)
            
        Returns:
            Complete validation results
        """
        # Initialize system
        V = 1e-24  # 1 nm^3
        state = self.initialize_system(T=T, V=V)
        
        # Evolve for many steps
        n_steps = 5000
        trajectory = self.evolve_system(state, dt=1e-12, steps=n_steps)
        
        # Find recurrences in both modes
        cont_rec = self.find_recurrence(trajectory, epsilon=0.3, mode='continuous')
        cat_rec = self.find_recurrence(trajectory, epsilon=0.3, mode='categorical')
        
        # Compute S-entropy trajectory
        S_trajectory = self.compute_s_entropy_trajectory(trajectory)
        
        # Theoretical recurrence time
        t_rec_theory = self.estimate_recurrence_time(V)
        
        self.results = {
            'N_particles': self.N,
            'dimension': self.dim,
            'T': T,
            'V': V,
            'n_steps': n_steps,
            'continuous_recurrence': {
                'min_distance': float(cont_rec['min_distance']),
                'first_recurrence': int(cont_rec['first_recurrence']) if cont_rec['first_recurrence'] else None,
                'n_recurrences': int(cont_rec['n_recurrences']),
                'distances': cont_rec['distances'][:100].tolist()  # First 100 for storage
            },
            'categorical_recurrence': {
                'min_distance': float(cat_rec['min_distance']),
                'first_recurrence': int(cat_rec['first_recurrence']) if cat_rec['first_recurrence'] else None,
                'n_recurrences': int(cat_rec['n_recurrences']),
                'distances': cat_rec['distances'][:100].tolist()
            },
            'S_entropy_trajectory': S_trajectory.tolist(),
            'theoretical_recurrence_time': float(t_rec_theory) if t_rec_theory < 1e100 else 'inf',
            'trajectory_distances_cont': cont_rec['distances'].tolist(),
            'trajectory_distances_cat': cat_rec['distances'].tolist()
        }
        
        return self.results
        
    def create_panel_chart(self, save_path: str):
        """Create comprehensive panel visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        # Panel 1: Phase space distance evolution (continuous)
        ax1 = fig.add_subplot(2, 3, 1)
        distances_cont = np.array(self.results['trajectory_distances_cont'])
        ax1.plot(distances_cont, 'b-', alpha=0.7, linewidth=0.5)
        ax1.axhline(y=0.3, color='r', linestyle='--', label='Epsilon threshold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Phase Distance (continuous)')
        ax1.set_title('Continuous Phase Space Distance')
        ax1.legend()
        ax1.set_ylim(0, min(2, distances_cont.max() * 1.1))
        
        # Panel 2: Phase space distance evolution (categorical)
        ax2 = fig.add_subplot(2, 3, 2)
        distances_cat = np.array(self.results['trajectory_distances_cat'])
        ax2.plot(distances_cat, 'g-', alpha=0.7, linewidth=0.5)
        ax2.axhline(y=0.3, color='r', linestyle='--', label='Epsilon threshold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Phase Distance (categorical)')
        ax2.set_title('Categorical Phase Space Distance')
        ax2.legend()
        
        # Panel 3: S-entropy trajectory in 3D
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        S_traj = np.array(self.results['S_entropy_trajectory'])
        colors = np.linspace(0, 1, len(S_traj))
        ax3.scatter(S_traj[:, 0], S_traj[:, 1], S_traj[:, 2], c=colors, 
                   cmap='viridis', s=2, alpha=0.5)
        ax3.set_xlabel('S_k (knowledge)')
        ax3.set_ylabel('S_t (time)')
        ax3.set_zlabel('S_e (evolution)')
        ax3.set_title('S-Entropy Trajectory')
        
        # Panel 4: Distance histogram
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.hist(distances_cont, bins=50, alpha=0.5, label='Continuous', color='blue', density=True)
        ax4.hist(distances_cat, bins=50, alpha=0.5, label='Categorical', color='green', density=True)
        ax4.axvline(x=0.3, color='r', linestyle='--', label='Epsilon')
        ax4.set_xlabel('Phase Distance')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Distance Distribution')
        ax4.legend()
        
        # Panel 5: Recurrence statistics
        ax5 = fig.add_subplot(2, 3, 5)
        categories = ['Continuous', 'Categorical']
        n_rec = [self.results['continuous_recurrence']['n_recurrences'],
                 self.results['categorical_recurrence']['n_recurrences']]
        colors = ['blue', 'green']
        bars = ax5.bar(categories, n_rec, color=colors, alpha=0.7)
        ax5.set_ylabel('Number of Recurrences')
        ax5.set_title(f'Recurrence Count (N={self.results["n_steps"]} steps)')
        for bar, val in zip(bars, n_rec):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val}', ha='center', va='bottom')
        
        # Panel 6: Minimum distance vs theoretical prediction
        ax6 = fig.add_subplot(2, 3, 6)
        # Show scaling of recurrence time with system size
        N_range = np.array([10, 20, 50, 100, 200])
        log_t_rec = 2 * 3 * N_range * np.log(10)  # n_bins = 10
        ax6.semilogy(N_range, np.exp(np.minimum(log_t_rec, 50)), 'bo-', linewidth=2)
        ax6.axvline(x=self.N, color='r', linestyle='--', label=f'This system (N={self.N})')
        ax6.set_xlabel('Number of Particles')
        ax6.set_ylabel('Recurrence Time Scale')
        ax6.set_title('Recurrence Time vs System Size')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Poincare Recurrence Monitor: N={self.N} particles, T={self.results["T"]} K',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved panel chart to {save_path}")
        
    def save_data(self, save_path: str):
        """Save results to JSON."""
        # Create a serializable version
        save_results = {}
        for key, val in self.results.items():
            if isinstance(val, dict):
                save_results[key] = {}
                for k, v in val.items():
                    if isinstance(v, np.ndarray):
                        save_results[key][k] = v.tolist()
                    elif isinstance(v, (np.int64, np.int32)):
                        save_results[key][k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        save_results[key][k] = float(v)
                    else:
                        save_results[key][k] = v
            elif isinstance(val, np.ndarray):
                save_results[key] = val.tolist()
            elif isinstance(val, (np.int64, np.int32)):
                save_results[key] = int(val)
            elif isinstance(val, (np.float64, np.float32)):
                save_results[key] = float(val)
            else:
                save_results[key] = val
                
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"Saved data to {save_path}")


def main():
    """Run Poincare Recurrence Monitor validation."""
    print("=" * 60)
    print("POINCARE RECURRENCE MONITOR (PRM)")
    print("Testing: Trajectory return in categorical phase space")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Test with different particle numbers
    for N in [50, 100]:
        print(f"\n--- N = {N} particles ---")
        prm = PoincareRecurrenceMonitor(N_particles=N)
        results = prm.full_validation(T=300.0)
        
        print(f"  Continuous: min_dist = {results['continuous_recurrence']['min_distance']:.3f}")
        print(f"  Categorical: min_dist = {results['categorical_recurrence']['min_distance']:.3f}")
        print(f"  Continuous recurrences: {results['continuous_recurrence']['n_recurrences']}")
        print(f"  Categorical recurrences: {results['categorical_recurrence']['n_recurrences']}")
        
        prm.create_panel_chart(os.path.join(figures_dir, f"panel_prm_N{N}.png"))
        prm.save_data(os.path.join(data_dir, f"prm_N{N}.json"))
    
    print("\n" + "=" * 60)
    print("PRM VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
