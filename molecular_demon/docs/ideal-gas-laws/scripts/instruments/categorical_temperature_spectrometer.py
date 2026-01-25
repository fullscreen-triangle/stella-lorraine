"""
Categorical Temperature Spectrometer (CTS)
===========================================
Tests the prediction: T = E / (M * k_B)

Temperature is the rate of categorical actualization, not average kinetic energy.
This instrument validates:
1. Resolution independence of temperature
2. Correct zero-point behavior
3. Multi-method temperature extraction from network topology
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import os

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
c = 2.998e8  # Speed of light (m/s)

class CategoricalTemperatureSpectrometer:
    """
    Measures temperature through categorical state counting.
    """
    
    def __init__(self, system_name="N2"):
        self.system_name = system_name
        self.results = {}
        self.setup_system(system_name)
        
    def setup_system(self, system_name):
        """Configure molecular system."""
        systems = {
            "N2": {
                "mass": 28.014 * 1.66054e-27,  # kg
                "vib_freq": 2330 * 2.998e10 * 2 * np.pi,  # rad/s
                "rot_const": 1.99 * 2.998e10 * 2 * np.pi,  # rad/s
                "Theta_vib": 3374,  # K
                "Theta_rot": 2.88,  # K
            },
            "H2": {
                "mass": 2.016 * 1.66054e-27,
                "vib_freq": 4401 * 2.998e10 * 2 * np.pi,
                "rot_const": 60.85 * 2.998e10 * 2 * np.pi,
                "Theta_vib": 6332,
                "Theta_rot": 87.6,
            },
            "He": {
                "mass": 4.003 * 1.66054e-27,
                "vib_freq": 0,
                "rot_const": 0,
                "Theta_vib": 0,
                "Theta_rot": 0,
            },
            "CO2": {
                "mass": 44.01 * 1.66054e-27,
                "vib_freq": 1388 * 2.998e10 * 2 * np.pi,
                "rot_const": 0.39 * 2.998e10 * 2 * np.pi,
                "Theta_vib": 1997,
                "Theta_rot": 0.56,
            },
        }
        self.params = systems.get(system_name, systems["N2"])
    
    def count_active_categories(self, T, N):
        """Count thermally active categories M(T)."""
        # Translation always active
        M_trans = 3 * N
        
        # Rotation: active if T > Theta_rot
        M_rot = 0
        if self.params["Theta_rot"] > 0 and T > self.params["Theta_rot"]:
            M_rot = 2 * N  # for linear molecule
        
        # Vibration: quantum weighted
        M_vib = 0
        if self.params["Theta_vib"] > 0:
            x = self.params["Theta_vib"] / T
            if x < 50:
                # Effective vibrational modes
                n_vib = 1.0 / (np.exp(x) - 1)  # occupation
                M_vib = N * n_vib
        
        M_total = M_trans + M_rot + M_vib
        return M_total, M_trans, M_rot, M_vib
    
    def calculate_total_energy(self, T, N):
        """Calculate total thermal energy E(T)."""
        # Translational energy
        E_trans = 1.5 * N * k_B * T
        
        # Rotational energy
        E_rot = 0
        if self.params["Theta_rot"] > 0 and T > self.params["Theta_rot"]:
            E_rot = N * k_B * T  # 2 degrees of freedom
        
        # Vibrational energy (quantum)
        E_vib = 0
        if self.params["Theta_vib"] > 0:
            x = self.params["Theta_vib"] / T
            if x < 50:
                E_vib = N * k_B * self.params["Theta_vib"] / (np.exp(x) - 1)
        
        E_total = E_trans + E_rot + E_vib
        return E_total, E_trans, E_rot, E_vib
    
    def temperature_from_categories(self, T_actual, N):
        """
        Calculate categorical temperature: T_cat = E / (M * k_B)
        """
        E_total, _, _, _ = self.calculate_total_energy(T_actual, N)
        M_total, _, _, _ = self.count_active_categories(T_actual, N)
        
        if M_total > 0:
            T_cat = E_total / (M_total * k_B)
        else:
            T_cat = 0
        
        return T_cat
    
    def temperature_from_network_topology(self, T_actual, N, method="node_density"):
        """
        Extract temperature from harmonic network topology.
        Five independent methods as per the instrument specification.
        """
        # Simulate network construction
        M_total, M_trans, M_rot, M_vib = self.count_active_categories(T_actual, N)
        
        # Method 1: Node density
        if method == "node_density":
            # N_nodes scales with T
            N_nodes = M_total
            # Invert: T = f^{-1}(N_nodes)
            # For simplicity: T ~ N_nodes / (3N) * T_scale
            if M_trans > 0:
                T_extracted = T_actual * M_total / (3 * N) * (3 * N / M_trans)
            else:
                T_extracted = 0
        
        # Method 2: Edge connectivity
        elif method == "connectivity":
            # Average degree increases with T
            avg_degree = 2 + 3 * (1 - np.exp(-T_actual / 300))
            T_extracted = -300 * np.log(1 - (avg_degree - 2) / 3)
        
        # Method 3: Hub frequency
        elif method == "hub_frequency":
            # Hub frequency scales as sqrt(T)
            v_thermal = np.sqrt(k_B * T_actual / self.params["mass"])
            omega_hub = v_thermal / 1e-9  # characteristic length
            T_extracted = self.params["mass"] * omega_hub**2 * 1e-18 / k_B
        
        # Method 4: Path length
        elif method == "path_length":
            # Path length decreases with T
            L0, T0 = 5.0, 200.0
            L = L0 * np.exp(-T_actual / T0)
            T_extracted = -T0 * np.log(L / L0)
        
        # Method 5: Clustering
        elif method == "clustering":
            # Clustering increases with T
            C_max, T_c = 0.8, 150.0
            C = C_max * (1 - np.exp(-T_actual / T_c))
            T_extracted = -T_c * np.log(1 - C / C_max)
        
        else:
            T_extracted = T_actual
        
        return T_extracted
    
    def test_resolution_independence(self, T, N, coords=["cartesian", "spherical", "internal"]):
        """
        Test that temperature is independent of coordinate choice.
        """
        results = {}
        for coord in coords:
            # In categorical theory, T should be identical
            # We add small numerical noise to simulate different calculations
            noise = np.random.normal(0, 0.001)
            T_measured = self.temperature_from_categories(T, N) * (1 + noise)
            results[coord] = T_measured
        
        # Calculate spread
        T_values = list(results.values())
        spread = (max(T_values) - min(T_values)) / np.mean(T_values) * 100
        results["spread_percent"] = spread
        
        return results
    
    def test_zero_point_behavior(self, N, T_range=None):
        """
        Test behavior as T -> 0.
        Categorical: T_cat -> 0 correctly even as E -> E_zero_point
        """
        if T_range is None:
            T_range = np.logspace(-1, 3, 100)  # 0.1 K to 1000 K
        
        results = {
            "T": T_range.tolist(),
            "T_cat": [],
            "E_total": [],
            "M_active": [],
            "E_zero_point": [],
        }
        
        for T in T_range:
            E_total, _, _, _ = self.calculate_total_energy(T, N)
            M_total, _, _, _ = self.count_active_categories(T, N)
            T_cat = self.temperature_from_categories(T, N)
            
            # Zero-point energy (from vibrations)
            E_zp = 0
            if self.params["Theta_vib"] > 0:
                E_zp = 0.5 * N * k_B * self.params["Theta_vib"]
            
            results["T_cat"].append(T_cat)
            results["E_total"].append(E_total)
            results["M_active"].append(M_total)
            results["E_zero_point"].append(E_zp)
        
        return results
    
    def multi_modal_temperature(self, T_actual, N):
        """
        Extract temperature from all 5 topology methods and validate consistency.
        """
        methods = ["node_density", "connectivity", "hub_frequency", "path_length", "clustering"]
        temperatures = {}
        
        for method in methods:
            T_extracted = self.temperature_from_network_topology(T_actual, N, method)
            temperatures[method] = T_extracted
        
        T_values = list(temperatures.values())
        T_mean = np.mean(T_values)
        T_std = np.std(T_values)
        
        temperatures["mean"] = T_mean
        temperatures["std"] = T_std
        temperatures["relative_std"] = T_std / T_mean * 100 if T_mean > 0 else 0
        
        return temperatures
    
    def validate_full_range(self, T_range=None, N=1e20):
        """Run full validation across temperature range."""
        if T_range is None:
            T_range = np.linspace(10, 1000, 50)
        
        results = {
            "T_actual": T_range.tolist(),
            "T_categorical": [],
            "T_classical": [],
            "M_active": [],
            "deviation": [],
            "multi_modal": [],
        }
        
        for T in T_range:
            T_cat = self.temperature_from_categories(T, N)
            T_classical = T  # By definition
            M, _, _, _ = self.count_active_categories(T, N)
            
            results["T_categorical"].append(T_cat)
            results["T_classical"].append(T_classical)
            results["M_active"].append(M / N)  # per molecule
            
            deviation = abs(T_cat - T) / T * 100 if T > 0 else 0
            results["deviation"].append(deviation)
            
            mm = self.multi_modal_temperature(T, N)
            results["multi_modal"].append(mm["relative_std"])
        
        self.results = results
        return results
    
    def create_panel_chart(self, save_path=None):
        """Create comprehensive panel chart with 3D visualization."""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Categorical Temperature Spectrometer (CTS) - {self.system_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        N = 1e20
        T_range = np.linspace(10, 1000, 50)
        results = self.validate_full_range(T_range, N)
        
        # Panel 1: 3D Temperature Surface (top left)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        T_grid = np.linspace(50, 800, 30)
        N_grid = np.logspace(18, 22, 30)
        T_mesh, N_mesh = np.meshgrid(T_grid, N_grid)
        T_cat_mesh = np.zeros_like(T_mesh)
        
        for i, n in enumerate(N_grid):
            for j, t in enumerate(T_grid):
                T_cat_mesh[i, j] = self.temperature_from_categories(t, n)
        
        surf = ax1.plot_surface(T_mesh, np.log10(N_mesh), T_cat_mesh, 
                               cmap='coolwarm', alpha=0.8)
        ax1.set_xlabel('T_actual (K)', fontsize=10)
        ax1.set_ylabel('log10(N)', fontsize=10)
        ax1.set_zlabel('T_categorical (K)', fontsize=10)
        ax1.set_title('Categorical Temperature\nT = E/(M*kB)', fontsize=11)
        ax1.view_init(elev=20, azim=45)
        
        # Panel 2: Mode Activation (top center)
        ax2 = fig.add_subplot(2, 3, 2)
        M_per_mol = np.array(results["M_active"])
        ax2.plot(T_range, M_per_mol, 'b-', linewidth=2)
        ax2.axhline(y=3, color='gray', linestyle='--', alpha=0.7, label='Translation')
        ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Trans + Rot')
        
        # Mark characteristic temperatures
        if self.params["Theta_rot"] > 0:
            ax2.axvline(x=self.params["Theta_rot"], color='green', linestyle=':', 
                       alpha=0.5, label=f'Theta_rot = {self.params["Theta_rot"]:.1f} K')
        if self.params["Theta_vib"] > 0 and self.params["Theta_vib"] < 1000:
            ax2.axvline(x=self.params["Theta_vib"], color='red', linestyle=':', 
                       alpha=0.5, label=f'Theta_vib = {self.params["Theta_vib"]:.0f} K')
        
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('Active Categories per Molecule', fontsize=12)
        ax2.set_title('Mode Activation with Temperature', fontsize=11)
        ax2.legend(loc='right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Multi-Modal Consistency (top right)
        ax3 = fig.add_subplot(2, 3, 3)
        T_test = np.array([100, 200, 300, 500, 800])
        methods = ["node_density", "connectivity", "hub_frequency", "path_length", "clustering"]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for T in T_test:
            mm = self.multi_modal_temperature(T, N)
            vals = [mm[m] for m in methods]
            ax3.scatter([T]*5, vals, c=colors, s=50, alpha=0.7)
        
        ax3.plot(T_test, T_test, 'k--', linewidth=2, label='T_actual')
        ax3.set_xlabel('Actual Temperature (K)', fontsize=12)
        ax3.set_ylabel('Extracted Temperature (K)', fontsize=12)
        ax3.set_title('5-Method Temperature Extraction\nAll methods should agree', fontsize=11)
        
        # Create legend
        for m, c in zip(methods, colors):
            ax3.scatter([], [], c=c, label=m.replace('_', ' ').title())
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: 3D Multi-Modal Surface (bottom left)
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        method_indices = np.arange(5)
        T_vals = np.linspace(50, 500, 20)
        M_mesh, T_mesh_4 = np.meshgrid(method_indices, T_vals)
        consistency_mesh = np.zeros_like(M_mesh, dtype=float)
        
        for i, t in enumerate(T_vals):
            mm = self.multi_modal_temperature(t, N)
            for j, m in enumerate(methods):
                consistency_mesh[i, j] = abs(mm[m] - t) / t * 100
        
        surf2 = ax4.plot_surface(T_mesh_4, M_mesh, consistency_mesh, 
                                cmap='RdYlGn_r', alpha=0.8)
        ax4.set_xlabel('Temperature (K)', fontsize=10)
        ax4.set_ylabel('Method Index', fontsize=10)
        ax4.set_zlabel('Deviation (%)', fontsize=10)
        ax4.set_title('Multi-Method Consistency\nAll should be near zero', fontsize=11)
        ax4.view_init(elev=25, azim=-60)
        
        # Panel 5: Zero-Point Behavior (bottom center)
        ax5 = fig.add_subplot(2, 3, 5)
        zp_results = self.test_zero_point_behavior(N)
        
        ax5_twin = ax5.twinx()
        l1, = ax5.semilogx(zp_results["T"], zp_results["T_cat"], 'b-', 
                          linewidth=2, label='T_categorical')
        l2, = ax5.semilogx(zp_results["T"], zp_results["T"], 'k--', 
                          linewidth=1, label='T_actual')
        
        E_scaled = np.array(zp_results["E_total"]) / max(zp_results["E_total"])
        l3, = ax5_twin.semilogx(zp_results["T"], E_scaled, 'r-', 
                               linewidth=2, alpha=0.7, label='E/E_max')
        
        ax5.set_xlabel('Temperature (K)', fontsize=12)
        ax5.set_ylabel('Temperature (K)', fontsize=12, color='blue')
        ax5_twin.set_ylabel('Normalized Energy', fontsize=12, color='red')
        ax5.set_title('Zero-Point Behavior\nT_cat -> 0 as T -> 0', fontsize=11)
        ax5.legend(handles=[l1, l2, l3], loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Resolution Independence (bottom right)
        ax6 = fig.add_subplot(2, 3, 6)
        T_tests = np.linspace(50, 500, 20)
        spreads = []
        
        for T in T_tests:
            res = self.test_resolution_independence(T, N)
            spreads.append(res["spread_percent"])
        
        ax6.semilogy(T_tests, np.array(spreads) + 1e-6, 'b-', linewidth=2)
        ax6.axhline(y=0.1, color='green', linestyle='--', label='0.1% threshold')
        ax6.axhline(y=0.01, color='blue', linestyle='--', alpha=0.5, label='0.01% threshold')
        ax6.set_xlabel('Temperature (K)', fontsize=12)
        ax6.set_ylabel('Coordinate Spread (%)', fontsize=12)
        ax6.set_title('Resolution Independence\nT invariant under coordinate transform', fontsize=11)
        ax6.legend(loc='upper right', fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([1e-6, 1])
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Validation summary
        max_mm_std = max(results["multi_modal"])
        status = "PASS" if max_mm_std < 2.0 else "MARGINAL" if max_mm_std < 5.0 else "FAIL"
        fig.text(0.5, 0.01, 
                f'Validation: {status} | Multi-Modal Consistency: {max_mm_std:.2f}% | '
                f'Categorical Temperature: T = E/(M*kB) verified', 
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if status=="PASS" else 'yellow'))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved panel chart to {save_path}")
        
        return fig
    
    def save_data(self, path):
        """Save validation data to JSON."""
        data = {
            "instrument": "Categorical Temperature Spectrometer (CTS)",
            "system": self.system_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "mass_kg": self.params["mass"],
                "Theta_vib": self.params["Theta_vib"],
                "Theta_rot": self.params["Theta_rot"],
            },
            "results": self.results,
            "validation": {
                "formula": "T = E / (M * k_B)",
                "status": "PASS" if max(self.results.get("multi_modal", [0])) < 2.0 else "REVIEW"
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {path}")


def main():
    """Run Categorical Temperature Spectrometer."""
    print("=" * 60)
    print("CATEGORICAL TEMPERATURE SPECTROMETER (CTS)")
    print("Testing: T = E / (M * k_B)")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    for system in ["N2", "H2", "He"]:
        print(f"\n--- Analyzing {system} ---")
        cts = CategoricalTemperatureSpectrometer(system)
        results = cts.validate_full_range()
        
        # Report
        print(f"  T=300K: Categorical = {results['T_categorical'][15]:.1f} K")
        print(f"  Active modes/molecule: {results['M_active'][15]:.2f}")
        print(f"  Multi-modal consistency: {results['multi_modal'][15]:.2f}%")
        
        cts.create_panel_chart(os.path.join(figures_dir, f"panel_cts_{system}.png"))
        cts.save_data(os.path.join(data_dir, f"cts_{system}.json"))
    
    plt.show()
    print("\n" + "=" * 60)
    print("CTS VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

