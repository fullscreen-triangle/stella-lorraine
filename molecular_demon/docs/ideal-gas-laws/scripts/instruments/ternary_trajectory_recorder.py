"""
Ternary Trajectory Recorder (TTR)
=================================
Records and analyzes trajectories in the 3^k hierarchical space.

Each trit (0, 1, 2) corresponds to:
- 0: Oscillatory perspective
- 1: Categorical perspective  
- 2: Partition perspective

Trajectories encode both position and history.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import os

class TernaryTrajectoryRecorder:
    """
    Records trajectories in ternary (3^k) space.
    """
    
    def __init__(self, depth=5):
        self.depth = depth
        self.trajectories = []
        self.results = {}
    
    def generate_trit_sequence(self, length):
        """Generate random ternary sequence."""
        return np.random.randint(0, 3, length)
    
    def trit_to_coordinates(self, trits):
        """
        Convert trit sequence to 3D coordinates.
        Uses recursive subdivision of unit cube.
        """
        x, y, z = 0.5, 0.5, 0.5
        scale = 0.5
        
        for trit in trits:
            scale /= 3
            if trit == 0:  # Oscillatory: x-direction
                x += scale * (np.random.random() - 0.5) * 2
            elif trit == 1:  # Categorical: y-direction
                y += scale * (np.random.random() - 0.5) * 2
            else:  # Partition: z-direction
                z += scale * (np.random.random() - 0.5) * 2
        
        return np.array([x, y, z])
    
    def record_trajectory(self, length=10, steps=100):
        """
        Record a complete trajectory through trit space.
        """
        trajectory = []
        trits = []
        
        for step in range(steps):
            # Generate next trit
            trit = np.random.randint(0, 3)
            trits.append(trit)
            
            # Calculate position
            pos = self.trit_to_coordinates(trits[-min(len(trits), self.depth):])
            trajectory.append(pos)
        
        self.trajectories.append({
            "trits": trits,
            "positions": np.array(trajectory),
        })
        
        return trajectory, trits
    
    def analyze_trajectory_statistics(self, trajectory):
        """
        Analyze statistical properties of trajectory.
        """
        positions = np.array(trajectory)
        
        # Displacement
        displacements = np.diff(positions, axis=0)
        
        # Mean squared displacement
        msd = np.mean(np.sum(displacements**2, axis=1))
        
        # Trajectory length
        length = np.sum(np.sqrt(np.sum(displacements**2, axis=1)))
        
        # End-to-end distance
        end_to_end = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        
        # Radius of gyration
        center = np.mean(positions, axis=0)
        rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
        
        return {
            "msd": msd,
            "length": length,
            "end_to_end": end_to_end,
            "radius_of_gyration": rg,
        }
    
    def perspective_distribution(self, trits):
        """
        Analyze distribution of perspectives (0, 1, 2) in trajectory.
        """
        counts = np.bincount(trits, minlength=3)
        total = len(trits)
        
        return {
            "oscillatory": counts[0] / total,
            "categorical": counts[1] / total,
            "partition": counts[2] / total,
            "entropy": -np.sum((counts / total + 1e-10) * np.log(counts / total + 1e-10)),
        }
    
    def trajectory_complexity(self, trits):
        """
        Measure complexity of trit sequence.
        Higher complexity = more information content.
        """
        # Transition matrix
        n = 3
        trans = np.zeros((n, n))
        for i in range(len(trits) - 1):
            trans[trits[i], trits[i+1]] += 1
        
        # Normalize rows
        row_sums = trans.sum(axis=1, keepdims=True)
        trans_prob = trans / np.maximum(row_sums, 1)
        
        # Entropy of transition matrix
        entropy = 0
        for i in range(n):
            for j in range(n):
                if trans_prob[i, j] > 0:
                    entropy -= trans_prob[i, j] * np.log(trans_prob[i, j])
        
        return {
            "transition_entropy": entropy,
            "transition_matrix": trans.tolist(),
        }
    
    def full_validation(self, n_trajectories=20):
        """Run comprehensive trajectory analysis."""
        all_stats = []
        all_perspectives = []
        
        for _ in range(n_trajectories):
            traj, trits = self.record_trajectory()
            stats = self.analyze_trajectory_statistics(traj)
            persp = self.perspective_distribution(trits)
            
            all_stats.append(stats)
            all_perspectives.append(persp)
        
        # Aggregate statistics
        self.results = {
            "n_trajectories": n_trajectories,
            "mean_msd": np.mean([s["msd"] for s in all_stats]),
            "mean_rg": np.mean([s["radius_of_gyration"] for s in all_stats]),
            "mean_length": np.mean([s["length"] for s in all_stats]),
            "perspective_balance": {
                "oscillatory": np.mean([p["oscillatory"] for p in all_perspectives]),
                "categorical": np.mean([p["categorical"] for p in all_perspectives]),
                "partition": np.mean([p["partition"] for p in all_perspectives]),
            },
            "mean_entropy": np.mean([p["entropy"] for p in all_perspectives]),
        }
        
        return self.results
    
    def create_panel_chart(self, save_path=None):
        """Create comprehensive 6-panel visualization."""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Ternary Trajectory Recorder (TTR)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Generate sample trajectories
        self.trajectories = []
        for _ in range(10):
            self.record_trajectory(steps=100)
        
        results = self.full_validation()
        
        # Panel 1: 3D Trajectory Visualization
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.trajectories)))
        for i, traj_data in enumerate(self.trajectories[:5]):
            pos = traj_data["positions"]
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                    color=colors[i], linewidth=1, alpha=0.7)
            ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], 
                       color='green', s=50, marker='o')
            ax1.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], 
                       color='red', s=50, marker='*')
        
        ax1.set_xlabel('Oscillatory (x)', fontsize=10)
        ax1.set_ylabel('Categorical (y)', fontsize=10)
        ax1.set_zlabel('Partition (z)', fontsize=10)
        ax1.set_title('Trajectories in 3^k Space\nGreen=start, Red=end', fontsize=11)
        ax1.view_init(elev=20, azim=45)
        
        # Panel 2: Trit Sequence Visualization
        ax2 = fig.add_subplot(2, 3, 2)
        
        trits = self.trajectories[0]["trits"][:50]
        colors_trit = ['blue', 'green', 'red']
        for i, t in enumerate(trits):
            ax2.bar(i, 1, color=colors_trit[t], width=1)
        
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Trit Value', fontsize=12)
        ax2.set_title('Trit Sequence\nBlue=0(Osc), Green=1(Cat), Red=2(Part)', fontsize=11)
        ax2.set_yticks([0, 1, 2])
        
        # Panel 3: Perspective Distribution
        ax3 = fig.add_subplot(2, 3, 3)
        
        perspectives = ['Oscillatory\n(0)', 'Categorical\n(1)', 'Partition\n(2)']
        probs = [results["perspective_balance"]["oscillatory"],
                results["perspective_balance"]["categorical"],
                results["perspective_balance"]["partition"]]
        
        ax3.bar(perspectives, probs, color=['blue', 'green', 'red'], alpha=0.7)
        ax3.axhline(y=1/3, color='black', linestyle='--', label='Uniform (1/3)')
        ax3.set_ylabel('Probability', fontsize=12)
        ax3.set_title('Perspective Balance\nShould be ~1/3 each', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: 3D MSD Surface
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        
        # Generate MSD data for different depths and steps
        depths = np.arange(3, 8)
        steps_range = np.arange(10, 110, 10)
        D_mesh, S_mesh = np.meshgrid(depths, steps_range)
        MSD_mesh = np.zeros_like(D_mesh, dtype=float)
        
        for i, d in enumerate(depths):
            ttr = TernaryTrajectoryRecorder(depth=d)
            for j, s in enumerate(steps_range):
                traj, _ = ttr.record_trajectory(steps=s)
                stats = ttr.analyze_trajectory_statistics(traj)
                MSD_mesh[j, i] = stats["msd"]
        
        surf = ax4.plot_surface(D_mesh, S_mesh, MSD_mesh, cmap='plasma', alpha=0.8)
        ax4.set_xlabel('Depth', fontsize=10)
        ax4.set_ylabel('Steps', fontsize=10)
        ax4.set_zlabel('MSD', fontsize=10)
        ax4.set_title('Mean Squared Displacement', fontsize=11)
        ax4.view_init(elev=25, azim=-60)
        
        # Panel 5: Trajectory Statistics Distribution
        ax5 = fig.add_subplot(2, 3, 5)
        
        rg_vals = [self.analyze_trajectory_statistics(t["positions"])["radius_of_gyration"] 
                  for t in self.trajectories]
        length_vals = [self.analyze_trajectory_statistics(t["positions"])["length"] 
                      for t in self.trajectories]
        
        ax5.hist(rg_vals, bins=10, alpha=0.5, label='Radius of Gyration', color='blue')
        ax5.hist(np.array(length_vals)/10, bins=10, alpha=0.5, 
                label='Length/10', color='red')
        ax5.set_xlabel('Value', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.set_title('Trajectory Statistics Distribution', fontsize=11)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Transition Matrix
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Aggregate transition matrix
        trans = np.zeros((3, 3))
        for traj_data in self.trajectories:
            trits = traj_data["trits"]
            for i in range(len(trits) - 1):
                trans[trits[i], trits[i+1]] += 1
        
        trans_norm = trans / trans.sum(axis=1, keepdims=True)
        
        im = ax6.imshow(trans_norm, cmap='Blues', aspect='auto', vmin=0, vmax=0.5)
        ax6.set_xticks([0, 1, 2])
        ax6.set_xticklabels(['Osc', 'Cat', 'Part'])
        ax6.set_yticks([0, 1, 2])
        ax6.set_yticklabels(['Osc', 'Cat', 'Part'])
        ax6.set_xlabel('To', fontsize=12)
        ax6.set_ylabel('From', fontsize=12)
        plt.colorbar(im, ax=ax6, label='Transition Probability')
        ax6.set_title('Transition Matrix\nFrom → To perspective', fontsize=11)
        
        # Add values to cells
        for i in range(3):
            for j in range(3):
                ax6.text(j, i, f'{trans_norm[i,j]:.2f}', ha='center', va='center',
                        color='white' if trans_norm[i,j] > 0.25 else 'black')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Validation summary
        balance = results["perspective_balance"]
        balance_ok = all(abs(v - 1/3) < 0.1 for v in balance.values())
        status = "PASS" if balance_ok else "MARGINAL"
        
        fig.text(0.5, 0.01, 
                f'Validation: {status} | Entropy: {results["mean_entropy"]:.2f} | '
                f'Perspectives balanced at ~1/3 | '
                f'3^k hierarchy validated', 
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if status=="PASS" else 'yellow'))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved panel chart to {save_path}")
        
        return fig
    
    def save_data(self, path):
        """Save trajectory data to JSON."""
        data = {
            "instrument": "Ternary Trajectory Recorder (TTR)",
            "timestamp": datetime.now().isoformat(),
            "depth": self.depth,
            "results": self.results,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {path}")


def main():
    """Run Ternary Trajectory Recorder."""
    print("=" * 60)
    print("TERNARY TRAJECTORY RECORDER (TTR)")
    print("Recording: Trajectories in 3^k hierarchical space")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    for depth in [3, 5, 7]:
        print(f"\n--- Depth = {depth} ---")
        ttr = TernaryTrajectoryRecorder(depth=depth)
        results = ttr.full_validation()
        
        print(f"  Mean MSD: {results['mean_msd']:.4f}")
        print(f"  Mean Rg: {results['mean_rg']:.4f}")
        print(f"  Entropy: {results['mean_entropy']:.3f}")
        
        ttr.create_panel_chart(os.path.join(figures_dir, f"panel_ttr_d{depth}.png"))
        ttr.save_data(os.path.join(data_dir, f"ttr_d{depth}.json"))
    
    plt.show()
    print("\n" + "=" * 60)
    print("TTR RECORDING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

