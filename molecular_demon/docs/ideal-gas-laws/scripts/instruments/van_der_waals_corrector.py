"""
Van der Waals Categorical Corrector (VWCC)
==========================================
Tests that Van der Waals corrections emerge from categorical finite-size 
and interaction effects:

(P + a*N^2/V^2)(V - Nb) = Nk_BT

Key validations:
1. Predict a and b from categorical structure
2. Pressure-volume isotherms
3. Critical point prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import os

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
N_A = 6.02214076e23  # Avogadro number

class VanDerWaalsCorrector:
    """
    Derives Van der Waals corrections from categorical finite-size effects.
    """
    
    def __init__(self, system_name="N2"):
        self.system_name = system_name
        self.results = {}
        self.setup_system(system_name)
    
    def setup_system(self, system_name):
        """Configure molecular system with known VdW parameters."""
        # Experimental VdW parameters (for validation)
        systems = {
            "N2": {
                "a_exp": 1.370,  # L^2*atm/mol^2
                "b_exp": 0.0387,  # L/mol
                "diameter": 3.64e-10,  # m (molecular diameter)
                "epsilon": 95.05 * k_B,  # J (Lennard-Jones depth)
                "T_c": 126.2,  # K (critical temperature)
                "P_c": 33.9,  # atm (critical pressure)
                "V_c": 0.0901,  # L/mol (critical molar volume)
            },
            "CO2": {
                "a_exp": 3.658,
                "b_exp": 0.04267,
                "diameter": 3.30e-10,
                "epsilon": 190 * k_B,
                "T_c": 304.2,
                "P_c": 72.9,
                "V_c": 0.0940,
            },
            "Ar": {
                "a_exp": 1.355,
                "b_exp": 0.0320,
                "diameter": 3.40e-10,
                "epsilon": 119.8 * k_B,
                "T_c": 150.8,
                "P_c": 48.7,
                "V_c": 0.0752,
            },
            "H2O": {
                "a_exp": 5.537,
                "b_exp": 0.0305,
                "diameter": 2.75e-10,
                "epsilon": 380 * k_B,
                "T_c": 647.3,
                "P_c": 218.3,
                "V_c": 0.0560,
            },
            "He": {
                "a_exp": 0.0346,
                "b_exp": 0.0238,
                "diameter": 2.60e-10,
                "epsilon": 10.2 * k_B,
                "T_c": 5.19,
                "P_c": 2.27,
                "V_c": 0.0578,
            },
        }
        self.params = systems.get(system_name, systems["N2"])
    
    def calculate_categorical_b(self):
        """
        Calculate excluded volume b from categorical structure.
        b = (2/3) * pi * d^3 * N_A per mol
        Each molecule occupies a categorical "space" of diameter d.
        """
        d = self.params["diameter"]
        # Excluded volume per molecule
        b_molecular = (2.0/3.0) * np.pi * d**3  # m^3 per molecule
        # Per mole, in L/mol
        b_categorical = b_molecular * N_A * 1000  # L/mol
        return b_categorical
    
    def calculate_categorical_a(self):
        """
        Calculate attraction parameter a from categorical structure.
        a = integral of -U(r) * 4*pi*r^2 dr for r > d
        Using Lennard-Jones potential.
        """
        d = self.params["diameter"]
        epsilon = self.params["epsilon"]
        
        # Lennard-Jones integral (simplified)
        # For LJ: a = (32/9) * pi * epsilon * d^3 * N_A^2
        a_molecular = (32.0/9.0) * np.pi * epsilon * d**3  # J*m^3
        
        # Convert to L^2*atm/mol^2
        # 1 J*m^3 = 1 Pa*m^6 = (1/101325) atm * (1000)^2 L^2 = 9.869e-3 L^2*atm
        a_categorical = a_molecular * N_A**2 * 9.869e-3
        
        return a_categorical
    
    def pressure_ideal(self, n, V, T):
        """Ideal gas pressure: P = nRT/V"""
        return n * 8.314 * T / V  # Pa (R in J/(mol*K), V in L gives ~correct units)
    
    def pressure_vdw(self, n, V, T, a=None, b=None):
        """
        Van der Waals pressure.
        P = nRT/(V-nb) - a*n^2/V^2
        n = moles, V = liters, T = Kelvin
        Returns pressure in atm
        """
        if a is None:
            a = self.params["a_exp"]
        if b is None:
            b = self.params["b_exp"]
        
        R = 0.08206  # L*atm/(mol*K)
        
        if V <= n * b:
            return np.nan
        
        P = n * R * T / (V - n * b) - a * n**2 / V**2
        return P
    
    def pressure_categorical(self, n, V, T):
        """
        Pressure from categorical corrections.
        Uses a and b calculated from categorical structure.
        """
        a_cat = self.calculate_categorical_a()
        b_cat = self.calculate_categorical_b()
        return self.pressure_vdw(n, V, T, a_cat, b_cat)
    
    def calculate_critical_point(self, a=None, b=None):
        """
        Calculate critical point from VdW parameters.
        T_c = 8a/(27Rb)
        P_c = a/(27b^2)
        V_c = 3nb
        """
        if a is None:
            a = self.params["a_exp"]
        if b is None:
            b = self.params["b_exp"]
        
        R = 0.08206  # L*atm/(mol*K)
        
        T_c = 8 * a / (27 * R * b)
        P_c = a / (27 * b**2)
        V_c = 3 * b  # per mole
        
        return {"T_c": T_c, "P_c": P_c, "V_c": V_c}
    
    def generate_isotherms(self, T_list, n=1, V_range=None):
        """Generate P-V isotherms at multiple temperatures."""
        if V_range is None:
            b = self.params["b_exp"]
            V_range = np.linspace(n * b * 1.1, 5.0, 200)  # L
        
        isotherms = {}
        for T in T_list:
            P_ideal = []
            P_vdw = []
            P_cat = []
            
            for V in V_range:
                P_ideal.append(n * 0.08206 * T / V)
                P_vdw.append(self.pressure_vdw(n, V, T))
                P_cat.append(self.pressure_categorical(n, V, T))
            
            isotherms[T] = {
                "V": V_range.tolist(),
                "P_ideal": P_ideal,
                "P_vdw": P_vdw,
                "P_categorical": P_cat,
            }
        
        return isotherms
    
    def validate_parameters(self):
        """Compare categorical predictions to experimental values."""
        a_cat = self.calculate_categorical_a()
        b_cat = self.calculate_categorical_b()
        a_exp = self.params["a_exp"]
        b_exp = self.params["b_exp"]
        
        return {
            "a_categorical": a_cat,
            "a_experimental": a_exp,
            "a_deviation_percent": abs(a_cat - a_exp) / a_exp * 100,
            "b_categorical": b_cat,
            "b_experimental": b_exp,
            "b_deviation_percent": abs(b_cat - b_exp) / b_exp * 100,
        }
    
    def validate_critical_point(self):
        """Compare critical point predictions."""
        # From experimental a, b
        cp_exp = self.calculate_critical_point()
        
        # From categorical a, b
        a_cat = self.calculate_categorical_a()
        b_cat = self.calculate_categorical_b()
        cp_cat = self.calculate_critical_point(a_cat, b_cat)
        
        return {
            "T_c_exp": self.params["T_c"],
            "T_c_vdw": cp_exp["T_c"],
            "T_c_categorical": cp_cat["T_c"],
            "P_c_exp": self.params["P_c"],
            "P_c_vdw": cp_exp["P_c"],
            "P_c_categorical": cp_cat["P_c"],
        }
    
    def full_validation(self):
        """Run comprehensive validation."""
        results = {
            "parameters": self.validate_parameters(),
            "critical_point": self.validate_critical_point(),
        }
        self.results = results
        return results
    
    def create_panel_chart(self, save_path=None):
        """Create comprehensive 6-panel visualization."""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Van der Waals Categorical Corrector (VWCC) - {self.system_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        results = self.full_validation()
        
        # Panel 1: 3D P-V-T Surface
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        T_range = np.linspace(150, 500, 20)
        b = self.params["b_exp"]
        V_range = np.linspace(b * 1.5, 2.0, 30)
        T_mesh, V_mesh = np.meshgrid(T_range, V_range)
        P_mesh = np.zeros_like(T_mesh)
        
        for i, V in enumerate(V_range):
            for j, T in enumerate(T_range):
                P = self.pressure_vdw(1, V, T)
                P_mesh[i, j] = P if not np.isnan(P) else 0
        
        surf = ax1.plot_surface(T_mesh, V_mesh, P_mesh, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Temperature (K)', fontsize=10)
        ax1.set_ylabel('Volume (L/mol)', fontsize=10)
        ax1.set_zlabel('Pressure (atm)', fontsize=10)
        ax1.set_title('Van der Waals P-V-T Surface', fontsize=11)
        ax1.view_init(elev=20, azim=-60)
        
        # Panel 2: Isotherms Comparison
        ax2 = fig.add_subplot(2, 3, 2)
        T_list = [self.params["T_c"] * 0.8, self.params["T_c"], self.params["T_c"] * 1.2]
        colors = ['blue', 'red', 'green']
        
        for T, color in zip(T_list, colors):
            iso = self.generate_isotherms([T])
            V = np.array(iso[T]["V"])
            P_ideal = np.array(iso[T]["P_ideal"])
            P_vdw = np.array(iso[T]["P_vdw"])
            
            ax2.plot(V, P_ideal, color=color, linestyle='--', alpha=0.5)
            ax2.plot(V, P_vdw, color=color, linewidth=2, 
                    label=f'T={T:.0f}K ({T/self.params["T_c"]:.1f}Tc)')
        
        ax2.axhline(y=self.params["P_c"], color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=self.params["V_c"], color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Volume (L/mol)', fontsize=12)
        ax2.set_ylabel('Pressure (atm)', fontsize=12)
        ax2.set_title('P-V Isotherms\nDashed = Ideal, Solid = VdW', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 2])
        ax2.set_ylim([0, 100])
        
        # Panel 3: Parameter Comparison
        ax3 = fig.add_subplot(2, 3, 3)
        params = results["parameters"]
        
        categories = ['a (L²atm/mol²)', 'b (L/mol)']
        exp_vals = [params["a_experimental"], params["b_experimental"] * 100]  # scale b
        cat_vals = [params["a_categorical"], params["b_categorical"] * 100]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, exp_vals, width, label='Experimental', color='blue')
        bars2 = ax3.bar(x + width/2, cat_vals, width, label='Categorical', color='orange')
        
        ax3.set_ylabel('Parameter Value', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['a', 'b × 100'])
        ax3.set_title(f'VdW Parameters\na: {params["a_deviation_percent"]:.1f}% dev, '
                     f'b: {params["b_deviation_percent"]:.1f}% dev', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: 3D Critical Point Visualization
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        
        # Generate multiple isochores near critical point
        V_list = np.linspace(self.params["V_c"] * 0.5, self.params["V_c"] * 2, 10)
        T_plot = np.linspace(self.params["T_c"] * 0.7, self.params["T_c"] * 1.5, 50)
        
        for V in V_list:
            P_vals = [self.pressure_vdw(1, V, T) for T in T_plot]
            ax4.plot([V]*len(T_plot), T_plot, P_vals, alpha=0.5)
        
        # Mark critical point
        ax4.scatter([self.params["V_c"]], [self.params["T_c"]], [self.params["P_c"]],
                   color='red', s=100, marker='*', label='Critical Point')
        
        ax4.set_xlabel('Volume (L/mol)', fontsize=10)
        ax4.set_ylabel('Temperature (K)', fontsize=10)
        ax4.set_zlabel('Pressure (atm)', fontsize=10)
        ax4.set_title('Critical Point Region', fontsize=11)
        ax4.view_init(elev=25, azim=45)
        
        # Panel 5: Critical Point Prediction
        ax5 = fig.add_subplot(2, 3, 5)
        cp = results["critical_point"]
        
        properties = ['T_c (K)', 'P_c (atm)']
        exp_vals = [cp["T_c_exp"], cp["P_c_exp"]]
        vdw_vals = [cp["T_c_vdw"], cp["P_c_vdw"]]
        cat_vals = [cp["T_c_categorical"], cp["P_c_categorical"]]
        
        x = np.arange(len(properties))
        width = 0.25
        
        ax5.bar(x - width, exp_vals, width, label='Experimental', color='blue')
        ax5.bar(x, vdw_vals, width, label='VdW (exp a,b)', color='green')
        ax5.bar(x + width, cat_vals, width, label='Categorical', color='orange')
        
        ax5.set_ylabel('Value', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(properties)
        ax5.set_title('Critical Point Predictions', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Panel 6: Multi-System Comparison
        ax6 = fig.add_subplot(2, 3, 6)
        systems = ["He", "Ar", "N2", "CO2", "H2O"]
        a_devs = []
        b_devs = []
        
        for sys in systems:
            vwcc = VanDerWaalsCorrector(sys)
            params = vwcc.validate_parameters()
            a_devs.append(params["a_deviation_percent"])
            b_devs.append(params["b_deviation_percent"])
        
        x = np.arange(len(systems))
        width = 0.35
        
        ax6.bar(x - width/2, a_devs, width, label='a deviation', color='red')
        ax6.bar(x + width/2, b_devs, width, label='b deviation', color='blue')
        ax6.axhline(y=10, color='orange', linestyle='--', label='10% threshold')
        ax6.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
        
        ax6.set_ylabel('Deviation from Experimental (%)', fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(systems)
        ax6.set_title('Multi-System Parameter Accuracy', fontsize=11)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # Validation summary
        a_dev = results["parameters"]["a_deviation_percent"]
        b_dev = results["parameters"]["b_deviation_percent"]
        status = "PASS" if a_dev < 20 and b_dev < 20 else "MARGINAL" if a_dev < 50 else "FAIL"
        
        fig.text(0.5, 0.01, 
                f'Validation: {status} | a deviation: {a_dev:.1f}% | b deviation: {b_dev:.1f}% | '
                f'VdW corrections emerge from categorical finite-size effects', 
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if status=="PASS" else 'yellow'))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved panel chart to {save_path}")
        
        return fig
    
    def save_data(self, path):
        """Save validation data to JSON."""
        data = {
            "instrument": "Van der Waals Categorical Corrector (VWCC)",
            "system": self.system_name,
            "timestamp": datetime.now().isoformat(),
            "molecular_parameters": {
                "diameter_m": self.params["diameter"],
                "epsilon_J": self.params["epsilon"],
            },
            "results": self.results,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {path}")


def main():
    """Run Van der Waals Categorical Corrector."""
    print("=" * 60)
    print("VAN DER WAALS CATEGORICAL CORRECTOR (VWCC)")
    print("Testing: (P + aN²/V²)(V - Nb) = NkT from categories")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    for system in ["N2", "CO2", "Ar"]:
        print(f"\n--- Analyzing {system} ---")
        vwcc = VanDerWaalsCorrector(system)
        results = vwcc.full_validation()
        
        params = results["parameters"]
        print(f"  a: categorical={params['a_categorical']:.3f}, "
              f"exp={params['a_experimental']:.3f}, "
              f"dev={params['a_deviation_percent']:.1f}%")
        print(f"  b: categorical={params['b_categorical']:.4f}, "
              f"exp={params['b_experimental']:.4f}, "
              f"dev={params['b_deviation_percent']:.1f}%")
        
        cp = results["critical_point"]
        print(f"  Tc: exp={cp['T_c_exp']:.1f}K, cat={cp['T_c_categorical']:.1f}K")
        
        vwcc.create_panel_chart(os.path.join(figures_dir, f"panel_vwcc_{system}.png"))
        vwcc.save_data(os.path.join(data_dir, f"vwcc_{system}.json"))
    
    plt.show()
    print("\n" + "=" * 60)
    print("VWCC VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

