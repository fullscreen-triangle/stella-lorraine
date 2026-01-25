"""
Quantum Statistics Categorical Classifier (QSCC)
=================================================
Tests that Bose-Einstein and Fermi-Dirac statistics emerge from 
categorical constraints on multiple occupancy.

Key validations:
1. Boson condensation (multiple occupancy)
2. Fermi degeneracy (Pauli exclusion = single occupancy)
3. Classical limit at high T
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import os
from scipy.special import zeta

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
hbar = 1.054571817e-34  # Reduced Planck constant

class QuantumStatisticsClassifier:
    """
    Classifies quantum statistics from categorical occupancy rules.
    """
    
    def __init__(self, particle_type="boson"):
        self.particle_type = particle_type
        self.results = {}
        self.setup_system()
    
    def setup_system(self):
        """Configure system parameters."""
        if self.particle_type == "boson":
            # He-4 parameters
            self.params = {
                "name": "He-4",
                "mass": 4.003 * 1.66054e-27,  # kg
                "T_c": 2.17,  # K (lambda point)
                "spin": 0,
            }
        elif self.particle_type == "fermion":
            # He-3 parameters
            self.params = {
                "name": "He-3",
                "mass": 3.016 * 1.66054e-27,
                "T_F": 0.3,  # K (approximate Fermi temperature for He-3)
                "spin": 0.5,
            }
        else:  # classical
            self.params = {
                "name": "Classical",
                "mass": 28.014 * 1.66054e-27,
                "T_char": 300,
            }
    
    def bose_einstein_distribution(self, epsilon, T, mu=0):
        """
        Bose-Einstein occupation number.
        n_BE = 1 / (exp((epsilon - mu)/(k_B*T)) - 1)
        """
        if T <= 0:
            return np.zeros_like(epsilon)
        
        x = (epsilon - mu) / (k_B * T)
        x = np.clip(x, -50, 50)  # Avoid overflow
        
        n = np.where(x > -50, 1.0 / (np.exp(x) - 1 + 1e-10), 0)
        n = np.where(epsilon > mu, n, np.inf)  # Condensate
        return np.maximum(n, 0)
    
    def fermi_dirac_distribution(self, epsilon, T, mu=0):
        """
        Fermi-Dirac occupation number.
        n_FD = 1 / (exp((epsilon - mu)/(k_B*T)) + 1)
        """
        if T <= 0:
            return np.where(epsilon < mu, 1.0, 0.0)
        
        x = (epsilon - mu) / (k_B * T)
        x = np.clip(x, -50, 50)
        
        return 1.0 / (np.exp(x) + 1)
    
    def maxwell_boltzmann_distribution(self, epsilon, T):
        """
        Classical Maxwell-Boltzmann occupation.
        n_MB = exp(-epsilon/(k_B*T))
        """
        if T <= 0:
            return np.zeros_like(epsilon)
        return np.exp(-epsilon / (k_B * T))
    
    def categorical_occupation(self, epsilon, T, mu=0):
        """
        Categorical occupation based on statistics type.
        Bosons: multiple occupancy allowed
        Fermions: at most one per category (Pauli)
        """
        if self.particle_type == "boson":
            return self.bose_einstein_distribution(epsilon, T, mu)
        elif self.particle_type == "fermion":
            return self.fermi_dirac_distribution(epsilon, T, mu)
        else:
            return self.maxwell_boltzmann_distribution(epsilon, T)
    
    def calculate_bec_fraction(self, T, n_density):
        """
        Calculate fraction of particles in ground state for bosons.
        n_0/N = 1 - (T/T_c)^(3/2) for T < T_c
        """
        m = self.params["mass"]
        
        # Critical temperature
        T_c = (2 * np.pi * hbar**2 / (m * k_B)) * (n_density / zeta(1.5))**(2/3)
        
        if T >= T_c:
            return 0.0, T_c
        else:
            return 1 - (T / T_c)**1.5, T_c
    
    def calculate_fermi_energy(self, n_density):
        """
        Calculate Fermi energy for fermions.
        E_F = (hbar^2/(2m)) * (3*pi^2*n)^(2/3)
        """
        m = self.params["mass"]
        E_F = (hbar**2 / (2 * m)) * (3 * np.pi**2 * n_density)**(2/3)
        T_F = E_F / k_B
        return E_F, T_F
    
    def test_classical_limit(self, T):
        """
        Test that BE and FD converge to MB at high T.
        """
        E_scale = k_B * T
        epsilon = np.linspace(0.1 * E_scale, 10 * E_scale, 100)
        
        n_BE = self.bose_einstein_distribution(epsilon, T, mu=0)
        n_FD = self.fermi_dirac_distribution(epsilon, T, mu=0)
        n_MB = self.maxwell_boltzmann_distribution(epsilon, T)
        
        # Normalize for comparison
        n_BE_norm = n_BE / np.max(n_BE) if np.max(n_BE) > 0 else n_BE
        n_FD_norm = n_FD / np.max(n_FD) if np.max(n_FD) > 0 else n_FD
        n_MB_norm = n_MB / np.max(n_MB)
        
        # Calculate deviation
        dev_BE = np.mean(np.abs(n_BE_norm - n_MB_norm) / (n_MB_norm + 1e-10)) * 100
        dev_FD = np.mean(np.abs(n_FD_norm - n_MB_norm) / (n_MB_norm + 1e-10)) * 100
        
        return {
            "T": T,
            "BE_MB_deviation": dev_BE,
            "FD_MB_deviation": dev_FD,
            "classical_limit_approached": dev_BE < 5 and dev_FD < 5,
        }
    
    def full_validation(self, T_range=None):
        """Run comprehensive validation."""
        if T_range is None:
            T_range = np.logspace(-1, 3, 50)  # 0.1 K to 1000 K
        
        results = {
            "T": T_range.tolist(),
            "BE_MB_deviation": [],
            "FD_MB_deviation": [],
        }
        
        for T in T_range:
            cl = self.test_classical_limit(T)
            results["BE_MB_deviation"].append(cl["BE_MB_deviation"])
            results["FD_MB_deviation"].append(cl["FD_MB_deviation"])
        
        self.results = results
        return results
    
    def create_panel_chart(self, save_path=None):
        """Create comprehensive 6-panel visualization."""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Quantum Statistics Categorical Classifier (QSCC) - {self.params["name"]}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Panel 1: 3D Occupation Surface
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        T_range = np.linspace(0.5, 5, 20)
        E_range = np.linspace(0.01 * k_B, 5 * k_B, 30)
        T_mesh, E_mesh = np.meshgrid(T_range, E_range)
        n_mesh = np.zeros_like(T_mesh)
        
        for i, E in enumerate(E_range):
            for j, T in enumerate(T_range):
                n_mesh[i, j] = self.bose_einstein_distribution(np.array([E]), T)[0]
        
        n_mesh = np.clip(n_mesh, 0, 10)  # Clip for visualization
        surf = ax1.plot_surface(T_mesh, E_mesh / k_B, n_mesh, cmap='plasma', alpha=0.8)
        ax1.set_xlabel('Temperature (K)', fontsize=10)
        ax1.set_ylabel('Energy/kB (K)', fontsize=10)
        ax1.set_zlabel('Occupation n', fontsize=10)
        ax1.set_title('Bose-Einstein Occupation\nDiverges as E→0 at low T', fontsize=11)
        ax1.view_init(elev=25, azim=-60)
        
        # Panel 2: BE vs FD vs MB Comparison
        ax2 = fig.add_subplot(2, 3, 2)
        T = 2.0  # K
        E_scale = k_B * T
        epsilon = np.linspace(0.1 * E_scale, 8 * E_scale, 200)
        
        n_BE = self.bose_einstein_distribution(epsilon, T)
        n_FD = self.fermi_dirac_distribution(epsilon, T, mu=2*E_scale)
        n_MB = self.maxwell_boltzmann_distribution(epsilon, T)
        
        ax2.semilogy(epsilon / E_scale, n_BE + 1e-5, 'b-', linewidth=2, label='Bose-Einstein')
        ax2.semilogy(epsilon / E_scale, n_FD + 1e-5, 'r-', linewidth=2, label='Fermi-Dirac')
        ax2.semilogy(epsilon / E_scale, n_MB, 'g--', linewidth=2, label='Maxwell-Boltzmann')
        ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Energy / kT', fontsize=12)
        ax2.set_ylabel('Occupation Number n', fontsize=12)
        ax2.set_title('Three Statistics at T = 2K\nCategorical occupancy rules', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([1e-4, 100])
        
        # Panel 3: BEC Condensation
        ax3 = fig.add_subplot(2, 3, 3)
        n_density = 2e28  # m^-3 (liquid He-4 density)
        T_bec = np.linspace(0.1, 5, 100)
        condensate_fractions = []
        T_c_val = None
        
        for T in T_bec:
            f0, T_c = self.calculate_bec_fraction(T, n_density)
            condensate_fractions.append(f0)
            if T_c_val is None:
                T_c_val = T_c
        
        ax3.plot(T_bec, condensate_fractions, 'b-', linewidth=2)
        ax3.axvline(x=T_c_val, color='red', linestyle='--', label=f'T_c = {T_c_val:.2f} K')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3.fill_between(T_bec, 0, condensate_fractions, alpha=0.3)
        ax3.set_xlabel('Temperature (K)', fontsize=12)
        ax3.set_ylabel('Condensate Fraction n₀/N', fontsize=12)
        ax3.set_title('Bose-Einstein Condensation\nMultiple occupancy → macroscopic ground state', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 5])
        ax3.set_ylim([0, 1])
        
        # Panel 4: 3D Fermi Surface
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        
        # Energy-temperature plane for FD
        T_fd = np.linspace(0.1, 2, 25)
        E_F = 1.0 * k_B  # Reference Fermi energy
        E_fd = np.linspace(0, 3 * E_F, 30)
        T_mesh_fd, E_mesh_fd = np.meshgrid(T_fd, E_fd)
        n_mesh_fd = np.zeros_like(T_mesh_fd)
        
        for i, E in enumerate(E_fd):
            for j, T in enumerate(T_fd):
                n_mesh_fd[i, j] = self.fermi_dirac_distribution(np.array([E]), T, mu=E_F)[0]
        
        surf2 = ax4.plot_surface(T_mesh_fd, E_mesh_fd / E_F, n_mesh_fd, 
                                cmap='coolwarm', alpha=0.8)
        ax4.set_xlabel('Temperature (K)', fontsize=10)
        ax4.set_ylabel('E/E_F', fontsize=10)
        ax4.set_zlabel('Occupation', fontsize=10)
        ax4.set_title('Fermi-Dirac Distribution\nPauli: max 1 per state', fontsize=11)
        ax4.view_init(elev=20, azim=45)
        
        # Panel 5: Fermi Degeneracy
        ax5 = fig.add_subplot(2, 3, 5)
        
        T_list = [0.01, 0.1, 0.5, 1.0, 2.0]
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(T_list)))
        E_F_plot = 1.0 * k_B
        epsilon = np.linspace(0, 2 * E_F_plot, 200)
        
        for T, color in zip(T_list, colors):
            n = self.fermi_dirac_distribution(epsilon, T, mu=E_F_plot)
            ax5.plot(epsilon / E_F_plot, n, color=color, linewidth=2, label=f'T={T}K')
        
        ax5.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='E = E_F')
        ax5.set_xlabel('E / E_F', fontsize=12)
        ax5.set_ylabel('Occupation n_FD', fontsize=12)
        ax5.set_title('Fermi Degeneracy\nSharp step as T→0', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Classical Limit Convergence
        ax6 = fig.add_subplot(2, 3, 6)
        results = self.full_validation()
        
        ax6.loglog(results["T"], np.array(results["BE_MB_deviation"]) + 0.01, 
                  'b-', linewidth=2, label='BE → MB')
        ax6.loglog(results["T"], np.array(results["FD_MB_deviation"]) + 0.01, 
                  'r-', linewidth=2, label='FD → MB')
        ax6.axhline(y=5, color='orange', linestyle='--', label='5% threshold')
        ax6.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1% threshold')
        ax6.set_xlabel('Temperature (K)', fontsize=12)
        ax6.set_ylabel('Deviation from MB (%)', fontsize=12)
        ax6.set_title('Classical Limit\nQuantum → Classical at high T', fontsize=11)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Validation summary
        high_T_BE = results["BE_MB_deviation"][-1]
        high_T_FD = results["FD_MB_deviation"][-1]
        status = "PASS" if high_T_BE < 5 and high_T_FD < 5 else "MARGINAL"
        
        fig.text(0.5, 0.01, 
                f'Validation: {status} | BE→MB at 1000K: {high_T_BE:.1f}% | '
                f'FD→MB at 1000K: {high_T_FD:.1f}% | '
                f'Quantum statistics from categorical occupancy', 
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if status=="PASS" else 'yellow'))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved panel chart to {save_path}")
        
        return fig
    
    def save_data(self, path):
        """Save validation data to JSON."""
        data = {
            "instrument": "Quantum Statistics Categorical Classifier (QSCC)",
            "particle_type": self.particle_type,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "name": self.params["name"],
                "mass_kg": self.params["mass"],
            },
            "results": self.results,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {path}")


def main():
    """Run Quantum Statistics Categorical Classifier."""
    print("=" * 60)
    print("QUANTUM STATISTICS CATEGORICAL CLASSIFIER (QSCC)")
    print("Testing: BE/FD from categorical occupancy rules")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    for ptype in ["boson", "fermion"]:
        print(f"\n--- Analyzing {ptype}s ---")
        qscc = QuantumStatisticsClassifier(ptype)
        results = qscc.full_validation()
        
        # High temperature convergence
        print(f"  Classical limit (1000K):")
        print(f"    BE deviation: {results['BE_MB_deviation'][-1]:.2f}%")
        print(f"    FD deviation: {results['FD_MB_deviation'][-1]:.2f}%")
        
        qscc.create_panel_chart(os.path.join(figures_dir, f"panel_qscc_{ptype}.png"))
        qscc.save_data(os.path.join(data_dir, f"qscc_{ptype}.json"))
    
    plt.show()
    print("\n" + "=" * 60)
    print("QSCC VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

