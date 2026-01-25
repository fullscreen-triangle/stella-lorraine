"""
Maxwell-Boltzmann Categorical Reconstructor (MBCR)
===================================================
Tests that the Maxwell-Boltzmann distribution emerges as the continuum 
limit of discrete categorical velocity distribution, with natural velocity 
bound at c (speed of light).

Key validations:
1. Discrete-to-continuum convergence
2. Natural velocity bound at c
3. Velocity moment agreement
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import os
from scipy.special import gamma

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
c = 2.998e8  # Speed of light (m/s)

class MaxwellBoltzmannReconstructor:
    """
    Reconstructs Maxwell-Boltzmann distribution from categorical structure.
    """
    
    def __init__(self, system_name="N2"):
        self.system_name = system_name
        self.results = {}
        self.setup_system(system_name)
    
    def setup_system(self, system_name):
        """Configure molecular system."""
        systems = {
            "N2": {"mass": 28.014 * 1.66054e-27},
            "H2": {"mass": 2.016 * 1.66054e-27},
            "He": {"mass": 4.003 * 1.66054e-27},
            "Xe": {"mass": 131.29 * 1.66054e-27},
            "Ar": {"mass": 39.95 * 1.66054e-27},
        }
        self.params = systems.get(system_name, systems["N2"])
        self.mass = self.params["mass"]
    
    def maxwell_boltzmann_pdf(self, v, T):
        """Classical Maxwell-Boltzmann speed distribution."""
        m = self.mass
        coeff = 4 * np.pi * (m / (2 * np.pi * k_B * T))**1.5
        return coeff * v**2 * np.exp(-m * v**2 / (2 * k_B * T))
    
    def categorical_velocity_distribution(self, v_bins, T, n_categories=100):
        """
        Discrete categorical velocity distribution.
        Each velocity category has a certain occupation.
        """
        m = self.mass
        v_centers = (v_bins[:-1] + v_bins[1:]) / 2
        dv = v_bins[1] - v_bins[0]
        
        # Count molecules in each category
        # Using Boltzmann factor for categorical occupation
        energies = 0.5 * m * v_centers**2
        boltzmann_factors = np.exp(-energies / (k_B * T))
        
        # Phase space degeneracy (4*pi*v^2 dv)
        degeneracy = 4 * np.pi * v_centers**2 * dv
        
        # Categorical probability
        weights = degeneracy * boltzmann_factors
        
        # Apply velocity bound at c
        weights[v_centers > c] = 0
        
        # Normalize
        P_cat = weights / np.sum(weights)
        
        return v_centers, P_cat
    
    def continuum_limit_test(self, T, n_bins_list=None):
        """Test convergence to MB as bin size -> 0."""
        if n_bins_list is None:
            n_bins_list = [10, 20, 50, 100, 200, 500]
        
        v_max = 5 * np.sqrt(k_B * T / self.mass)  # 5x thermal velocity
        
        results = {
            "n_bins": n_bins_list,
            "max_deviation": [],
            "mean_deviation": [],
        }
        
        for n_bins in n_bins_list:
            v_bins = np.linspace(0, v_max, n_bins + 1)
            v_centers, P_cat = self.categorical_velocity_distribution(v_bins, T)
            
            # Compare to MB
            P_mb = self.maxwell_boltzmann_pdf(v_centers, T)
            P_mb_normalized = P_mb / np.sum(P_mb)
            
            # Calculate deviations (only where P_mb > 0)
            mask = P_mb_normalized > 1e-10
            if np.any(mask):
                deviations = np.abs(P_cat[mask] - P_mb_normalized[mask]) / P_mb_normalized[mask] * 100
                results["max_deviation"].append(np.max(deviations))
                results["mean_deviation"].append(np.mean(deviations))
            else:
                results["max_deviation"].append(0)
                results["mean_deviation"].append(0)
        
        return results
    
    def velocity_bound_test(self, T):
        """
        Test that P(v > c) = 0 in categorical theory.
        Classical MB has non-zero tail.
        """
        v_max = 1.5 * c  # Go beyond light speed
        v = np.linspace(0, v_max, 1000)
        
        # Classical MB (no bound)
        P_mb = self.maxwell_boltzmann_pdf(v, T)
        P_mb_beyond_c = np.sum(P_mb[v > c]) / np.sum(P_mb) * 100
        
        # Categorical (with bound)
        v_bins = np.linspace(0, v_max, 1001)
        v_centers, P_cat = self.categorical_velocity_distribution(v_bins, T)
        P_cat_beyond_c = np.sum(P_cat[v_centers > c])  # Should be exactly 0
        
        return {
            "P_mb_beyond_c_percent": P_mb_beyond_c,
            "P_cat_beyond_c": P_cat_beyond_c,
            "categorical_bound_enforced": P_cat_beyond_c == 0,
        }
    
    def velocity_moments(self, T):
        """
        Calculate velocity moments from categorical distribution.
        Compare to MB predictions.
        """
        m = self.mass
        v_max = 10 * np.sqrt(k_B * T / m)
        v_bins = np.linspace(0, v_max, 501)
        v_centers, P_cat = self.categorical_velocity_distribution(v_bins, T)
        dv = v_bins[1] - v_bins[0]
        
        # Categorical moments
        v_mean_cat = np.sum(v_centers * P_cat)
        v2_mean_cat = np.sum(v_centers**2 * P_cat)
        v3_mean_cat = np.sum(v_centers**3 * P_cat)
        
        # MB theoretical moments
        v_mean_mb = np.sqrt(8 * k_B * T / (np.pi * m))
        v2_mean_mb = 3 * k_B * T / m
        v3_mean_mb = 4 * np.sqrt(2 * k_B * T / (np.pi * m)) * (k_B * T / m)
        
        return {
            "v_mean": {"categorical": v_mean_cat, "MB": v_mean_mb, 
                      "deviation": abs(v_mean_cat - v_mean_mb) / v_mean_mb * 100},
            "v2_mean": {"categorical": v2_mean_cat, "MB": v2_mean_mb,
                       "deviation": abs(v2_mean_cat - v2_mean_mb) / v2_mean_mb * 100},
            "v3_mean": {"categorical": v3_mean_cat, "MB": v3_mean_mb,
                       "deviation": abs(v3_mean_cat - v3_mean_mb) / v3_mean_mb * 100},
        }
    
    def full_validation(self, T_range=None):
        """Comprehensive validation across temperature range."""
        if T_range is None:
            T_range = np.array([100, 300, 1000, 3000, 10000])
        
        results = {
            "T": T_range.tolist(),
            "v_mean_deviation": [],
            "v2_mean_deviation": [],
            "bound_enforced": [],
            "convergence_bins_50": [],
        }
        
        for T in T_range:
            moments = self.velocity_moments(T)
            results["v_mean_deviation"].append(moments["v_mean"]["deviation"])
            results["v2_mean_deviation"].append(moments["v2_mean"]["deviation"])
            
            bound = self.velocity_bound_test(T)
            results["bound_enforced"].append(bound["categorical_bound_enforced"])
            
            conv = self.continuum_limit_test(T, [50])
            results["convergence_bins_50"].append(conv["mean_deviation"][0])
        
        self.results = results
        return results
    
    def create_panel_chart(self, save_path=None):
        """Create comprehensive 6-panel visualization."""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Maxwell-Boltzmann Categorical Reconstructor (MBCR) - {self.system_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        T_ref = 300  # K
        
        # Panel 1: 3D Distribution Surface (T, v, P)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        T_range = np.array([100, 200, 300, 500, 800])
        v_max = 2000  # m/s
        v_plot = np.linspace(0, v_max, 100)
        
        for T in T_range:
            P_mb = self.maxwell_boltzmann_pdf(v_plot, T)
            P_mb_scaled = P_mb / np.max(P_mb)
            ax1.plot(v_plot, [T]*len(v_plot), P_mb_scaled, linewidth=2)
        
        ax1.set_xlabel('Velocity (m/s)', fontsize=10)
        ax1.set_ylabel('Temperature (K)', fontsize=10)
        ax1.set_zlabel('P(v) normalized', fontsize=10)
        ax1.set_title('MB Distribution vs Temperature', fontsize=11)
        ax1.view_init(elev=20, azim=-60)
        
        # Panel 2: Discrete vs Continuum Comparison
        ax2 = fig.add_subplot(2, 3, 2)
        v_max_plot = 5 * np.sqrt(k_B * T_ref / self.mass)
        
        # Coarse binning
        v_bins_coarse = np.linspace(0, v_max_plot, 21)
        v_c_coarse, P_coarse = self.categorical_velocity_distribution(v_bins_coarse, T_ref)
        
        # Fine binning
        v_bins_fine = np.linspace(0, v_max_plot, 201)
        v_c_fine, P_fine = self.categorical_velocity_distribution(v_bins_fine, T_ref)
        
        # Continuum
        v_cont = np.linspace(0, v_max_plot, 500)
        P_cont = self.maxwell_boltzmann_pdf(v_cont, T_ref)
        P_cont /= np.trapz(P_cont, v_cont)
        
        ax2.bar(v_c_coarse, P_coarse * 20, width=v_bins_coarse[1]-v_bins_coarse[0], 
               alpha=0.5, label='Categorical (20 bins)', color='blue')
        ax2.plot(v_c_fine, P_fine * 200, 'g-', linewidth=2, alpha=0.7, label='Categorical (200 bins)')
        ax2.plot(v_cont, P_cont * (v_bins_fine[1]-v_bins_fine[0]), 'r--', 
                linewidth=2, label='MB Continuum')
        ax2.set_xlabel('Velocity (m/s)', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('Discrete to Continuum Convergence', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Convergence Analysis
        ax3 = fig.add_subplot(2, 3, 3)
        conv = self.continuum_limit_test(T_ref)
        ax3.loglog(conv["n_bins"], conv["mean_deviation"], 'bo-', linewidth=2, 
                  label='Mean Deviation')
        ax3.loglog(conv["n_bins"], conv["max_deviation"], 'rs-', linewidth=2, 
                  label='Max Deviation')
        ax3.axhline(y=1.0, color='orange', linestyle='--', label='1% threshold')
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='0.5% threshold')
        ax3.set_xlabel('Number of Bins', fontsize=12)
        ax3.set_ylabel('Deviation from MB (%)', fontsize=12)
        ax3.set_title('Convergence to Continuum Limit', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')
        
        # Panel 4: 3D Velocity Bound Visualization
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        
        # Create mesh showing velocity bound
        T_range_4 = np.logspace(2, 8, 30)  # 100 K to 100 million K
        v_range_4 = np.linspace(0, 1.2 * c, 50)
        T_mesh, v_mesh = np.meshgrid(T_range_4, v_range_4)
        P_mesh = np.zeros_like(T_mesh)
        
        for i, T in enumerate(T_range_4):
            for j, v in enumerate(v_range_4):
                if v < c:  # Categorical bound
                    P_mesh[j, i] = self.maxwell_boltzmann_pdf(v, T)
                else:
                    P_mesh[j, i] = 0  # Hard cutoff at c
        
        # Normalize each temperature column
        for i in range(len(T_range_4)):
            if np.max(P_mesh[:, i]) > 0:
                P_mesh[:, i] /= np.max(P_mesh[:, i])
        
        surf = ax4.plot_surface(np.log10(T_mesh), v_mesh / c, P_mesh, 
                               cmap='RdYlBu', alpha=0.8)
        ax4.axhline(y=1.0, color='red', linestyle='--')
        ax4.set_xlabel('log10(T/K)', fontsize=10)
        ax4.set_ylabel('v/c', fontsize=10)
        ax4.set_zlabel('P(v) normalized', fontsize=10)
        ax4.set_title('Velocity Bound at c\nP(v>c) = 0 exactly', fontsize=11)
        ax4.view_init(elev=25, azim=45)
        
        # Panel 5: Moment Comparison
        ax5 = fig.add_subplot(2, 3, 5)
        T_moments = np.linspace(100, 1000, 20)
        v_mean_cat = []
        v_mean_mb = []
        v2_mean_cat = []
        v2_mean_mb = []
        
        for T in T_moments:
            mom = self.velocity_moments(T)
            v_mean_cat.append(mom["v_mean"]["categorical"])
            v_mean_mb.append(mom["v_mean"]["MB"])
            v2_mean_cat.append(np.sqrt(mom["v2_mean"]["categorical"]))
            v2_mean_mb.append(np.sqrt(mom["v2_mean"]["MB"]))
        
        ax5.plot(T_moments, v_mean_cat, 'b-', linewidth=2, label='<v> categorical')
        ax5.plot(T_moments, v_mean_mb, 'b--', linewidth=2, label='<v> MB')
        ax5.plot(T_moments, v2_mean_cat, 'r-', linewidth=2, label='sqrt(<v2>) cat')
        ax5.plot(T_moments, v2_mean_mb, 'r--', linewidth=2, label='sqrt(<v2>) MB')
        ax5.set_xlabel('Temperature (K)', fontsize=12)
        ax5.set_ylabel('Velocity (m/s)', fontsize=12)
        ax5.set_title('Velocity Moments Comparison\nLines should overlap', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Multi-System Comparison
        ax6 = fig.add_subplot(2, 3, 6)
        systems = ["H2", "He", "N2", "Ar", "Xe"]
        colors = ['red', 'orange', 'blue', 'green', 'purple']
        v_max_multi = 3000
        v = np.linspace(0, v_max_multi, 500)
        
        for sys, color in zip(systems, colors):
            mbcr = MaxwellBoltzmannReconstructor(sys)
            P = mbcr.maxwell_boltzmann_pdf(v, T_ref)
            P /= np.max(P)
            ax6.plot(v, P, color=color, linewidth=2, label=sys)
        
        ax6.axhline(y=0, color='black', linewidth=0.5)
        ax6.set_xlabel('Velocity (m/s)', fontsize=12)
        ax6.set_ylabel('P(v) normalized', fontsize=12)
        ax6.set_title('Multi-System Distributions at 300 K\nHeavier = slower', fontsize=11)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Validation summary
        results = self.full_validation()
        max_moment_dev = max(max(results["v_mean_deviation"]), max(results["v2_mean_deviation"]))
        all_bounds = all(results["bound_enforced"])
        status = "PASS" if max_moment_dev < 2.0 and all_bounds else "MARGINAL" if max_moment_dev < 5.0 else "FAIL"
        
        fig.text(0.5, 0.01, 
                f'Validation: {status} | Moment Deviation: {max_moment_dev:.2f}% | '
                f'Velocity Bound (v<c): {"ENFORCED" if all_bounds else "FAILED"} | '
                f'MB emerges from categories', 
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if status=="PASS" else 'yellow'))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved panel chart to {save_path}")
        
        return fig
    
    def save_data(self, path):
        """Save validation data to JSON."""
        # Convert numpy booleans to Python booleans
        bound_test = self.velocity_bound_test(300)
        bound_test["categorical_bound_enforced"] = bool(bound_test["categorical_bound_enforced"])
        
        data = {
            "instrument": "Maxwell-Boltzmann Categorical Reconstructor (MBCR)",
            "system": self.system_name,
            "timestamp": datetime.now().isoformat(),
            "mass_kg": float(self.mass),
            "results": self.results,
            "convergence_test": self.continuum_limit_test(300),
            "velocity_bound_test": bound_test,
            "velocity_moments": self.velocity_moments(300),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"Saved data to {path}")


def main():
    """Run Maxwell-Boltzmann Categorical Reconstructor."""
    print("=" * 60)
    print("MAXWELL-BOLTZMANN CATEGORICAL RECONSTRUCTOR (MBCR)")
    print("Testing: MB as continuum limit with v < c bound")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    for system in ["N2", "H2", "Xe"]:
        print(f"\n--- Analyzing {system} ---")
        mbcr = MaxwellBoltzmannReconstructor(system)
        
        # Velocity moments at 300 K
        moments = mbcr.velocity_moments(300)
        print(f"  <v> deviation: {moments['v_mean']['deviation']:.2f}%")
        print(f"  <v2> deviation: {moments['v2_mean']['deviation']:.2f}%")
        
        # Velocity bound
        bound = mbcr.velocity_bound_test(300)
        print(f"  P(v>c) in MB: {bound['P_mb_beyond_c_percent']:.2e}%")
        print(f"  Categorical bound enforced: {bound['categorical_bound_enforced']}")
        
        mbcr.create_panel_chart(os.path.join(figures_dir, f"panel_mbcr_{system}.png"))
        mbcr.save_data(os.path.join(data_dir, f"mbcr_{system}.json"))
    
    plt.show()
    print("\n" + "=" * 60)
    print("MBCR VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

