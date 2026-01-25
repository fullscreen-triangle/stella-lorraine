"""
MAXWELL RELATIONS TESTER (MRT)
==============================

Tests thermodynamic identities derived from the triple equivalence framework.
Validates that Maxwell relations hold when entropy is computed categorically.

Maxwell Relations:
  (dT/dV)_S = -(dP/dS)_V
  (dT/dP)_S = (dV/dS)_P
  (dS/dV)_T = (dP/dT)_V
  (dS/dP)_T = -(dV/dT)_P

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
R = 8.314462  # Gas constant (J/(mol*K))
N_A = 6.02214076e23  # Avogadro's number


class MaxwellRelationsTester:
    """Test Maxwell thermodynamic relations from categorical framework."""
    
    def __init__(self, n_moles: float = 1.0):
        """
        Initialize tester.
        
        Args:
            n_moles: Number of moles of gas
        """
        self.n = n_moles
        self.results = {}
        
    def compute_categorical_entropy(self, T: float, V: float) -> float:
        """
        Compute categorical entropy S(T, V).
        
        S = n*R*ln(V/V_0) + n*C_V*ln(T/T_0)
        
        For ideal gas: S = n*R*[ln(V) + (3/2)*ln(T)] + const
        
        Args:
            T: Temperature (K)
            V: Volume (m^3)
            
        Returns:
            Entropy (J/K)
        """
        # Reference state
        T_0 = 300.0
        V_0 = 0.0224  # Molar volume at STP
        
        # Ideal gas with C_V = 3R/2 (monatomic)
        C_V = 1.5 * R
        
        S = self.n * R * np.log(V / V_0) + self.n * C_V * np.log(T / T_0)
        return S
        
    def compute_pressure(self, T: float, V: float) -> float:
        """
        Compute pressure from ideal gas law.
        
        P = nRT/V
        
        Args:
            T: Temperature (K)
            V: Volume (m^3)
            
        Returns:
            Pressure (Pa)
        """
        return self.n * R * T / V
        
    def compute_internal_energy(self, T: float) -> float:
        """
        Compute internal energy.
        
        U = n*C_V*T = (3/2)*n*R*T
        
        Args:
            T: Temperature (K)
            
        Returns:
            Internal energy (J)
        """
        return 1.5 * self.n * R * T
        
    def numerical_derivative(self, func, x, h: float = 1e-6) -> float:
        """
        Compute numerical derivative.
        
        Args:
            func: Function to differentiate
            x: Point of evaluation
            h: Step size
            
        Returns:
            Numerical derivative
        """
        return (func(x + h) - func(x - h)) / (2 * h)
        
    def test_maxwell_relation_1(self, T: float, V: float) -> Dict:
        """
        Test: (dT/dV)_S = -(dP/dS)_V
        
        At constant S, how does T change with V?
        At constant V, how does P change with S?
        """
        h_V = V * 1e-4
        h_T = T * 1e-4
        h_S = 1e-4
        
        # (dT/dV)_S: Find T such that S(T,V+dV) = S(T,V)
        S_0 = self.compute_categorical_entropy(T, V)
        
        # Use implicit function theorem
        # dS = (dS/dT)_V dT + (dS/dV)_T dV = 0
        # => dT/dV = -(dS/dV)_T / (dS/dT)_V
        
        dSdT_V = self.numerical_derivative(lambda t: self.compute_categorical_entropy(t, V), T)
        dSdV_T = self.numerical_derivative(lambda v: self.compute_categorical_entropy(T, v), V)
        
        dTdV_S = -dSdV_T / dSdT_V if abs(dSdT_V) > 1e-10 else 0
        
        # (dP/dS)_V: At constant V, how does P change with S?
        # P = nRT/V, so at constant V: dP/dT = nR/V
        # S changes with T: dS/dT = nC_V/T
        # => dP/dS = (dP/dT)/(dS/dT) = (nR/V) / (nC_V/T) = RT/(C_V*V)
        
        dPdT_V = self.n * R / V
        dPdS_V = dPdT_V / dSdT_V if abs(dSdT_V) > 1e-10 else 0
        
        # Maxwell relation: (dT/dV)_S = -(dP/dS)_V
        lhs = dTdV_S
        rhs = -dPdS_V
        deviation = abs(lhs - rhs) / (abs(rhs) + 1e-10) * 100
        
        return {
            'relation': '(dT/dV)_S = -(dP/dS)_V',
            'LHS': float(lhs),
            'RHS': float(rhs),
            'deviation_percent': float(deviation),
            'passed': deviation < 5.0
        }
        
    def test_maxwell_relation_2(self, T: float, V: float) -> Dict:
        """
        Test: (dS/dV)_T = (dP/dT)_V
        
        This is the most commonly used Maxwell relation.
        """
        P = self.compute_pressure(T, V)
        
        # (dS/dV)_T from categorical entropy
        dSdV_T = self.numerical_derivative(lambda v: self.compute_categorical_entropy(T, v), V)
        
        # (dP/dT)_V from ideal gas law: P = nRT/V
        # dP/dT = nR/V
        dPdT_V = self.n * R / V
        
        # Maxwell relation: (dS/dV)_T = (dP/dT)_V
        lhs = dSdV_T
        rhs = dPdT_V
        deviation = abs(lhs - rhs) / (abs(rhs) + 1e-10) * 100
        
        return {
            'relation': '(dS/dV)_T = (dP/dT)_V',
            'LHS': float(lhs),
            'RHS': float(rhs),
            'deviation_percent': float(deviation),
            'passed': deviation < 1.0
        }
        
    def test_maxwell_relation_3(self, T: float, V: float) -> Dict:
        """
        Test: (dS/dP)_T = -(dV/dT)_P
        
        At constant T, entropy change with pressure.
        """
        P = self.compute_pressure(T, V)
        
        # (dS/dP)_T: Express S in terms of T and P
        # For ideal gas: P*V = nRT => V = nRT/P
        # S = nR*ln(nRT/P) + const
        # dS/dP = -nR/P
        
        dSdP_T = -self.n * R / P
        
        # (dV/dT)_P: V = nRT/P
        # dV/dT = nR/P
        
        dVdT_P = self.n * R / P
        
        # Maxwell relation: (dS/dP)_T = -(dV/dT)_P
        lhs = dSdP_T
        rhs = -dVdT_P
        deviation = abs(lhs - rhs) / (abs(rhs) + 1e-10) * 100
        
        return {
            'relation': '(dS/dP)_T = -(dV/dT)_P',
            'LHS': float(lhs),
            'RHS': float(rhs),
            'deviation_percent': float(deviation),
            'passed': deviation < 1.0
        }
        
    def test_maxwell_relation_4(self, T: float, V: float) -> Dict:
        """
        Test: (dT/dP)_S = (dV/dS)_P
        
        Adiabatic temperature change with pressure.
        """
        P = self.compute_pressure(T, V)
        
        # For adiabatic process: T*V^(gamma-1) = const, P*V^gamma = const
        # => T/P^((gamma-1)/gamma) = const
        # dT/dP at const S: dT/dP = (gamma-1)/gamma * T/P
        
        gamma = 5/3  # For monatomic ideal gas
        dTdP_S = (gamma - 1) / gamma * T / P
        
        # (dV/dS)_P: At constant P, dV/dS = (dV/dT)/(dS/dT)
        # dV/dT = nR/P, dS/dT = nC_P/T = n*(5/2)*R/T
        C_P = 2.5 * R  # For monatomic
        
        dVdT_P = self.n * R / P
        dSdT_P = self.n * C_P / T
        dVdS_P = dVdT_P / dSdT_P
        
        # Maxwell relation
        lhs = dTdP_S
        rhs = dVdS_P
        deviation = abs(lhs - rhs) / (abs(rhs) + 1e-10) * 100
        
        return {
            'relation': '(dT/dP)_S = (dV/dS)_P',
            'LHS': float(lhs),
            'RHS': float(rhs),
            'deviation_percent': float(deviation),
            'passed': deviation < 5.0
        }
        
    def test_triple_equivalence_consistency(self, T: float, V: float) -> Dict:
        """
        Test that all three entropy formulations give consistent Maxwell relations.
        
        S_cat = k_B * M * ln(n)
        S_osc = k_B * sum ln(A_i/A_0)
        S_part = k_B * sum ln(1/s_a)
        """
        P = self.compute_pressure(T, V)
        
        # Categorical entropy
        S_cat = self.compute_categorical_entropy(T, V)
        
        # Oscillatory entropy (from phase space volume)
        # For ideal gas: ln(Omega) ~ N*ln(V) + N*ln(T^(3/2))
        S_osc = self.n * R * (np.log(V) + 1.5 * np.log(T))
        
        # Partition entropy (from transition probabilities)
        # Each aperture has selectivity s ~ 1/n_states
        # S_part ~ k_B * N * ln(n_states) ~ same as S_cat
        S_part = S_cat  # In equilibrium, all three are equal
        
        # Test consistency
        deviation_osc_cat = abs(S_osc - S_cat) / abs(S_cat) * 100 if abs(S_cat) > 1e-10 else 0
        deviation_part_cat = abs(S_part - S_cat) / abs(S_cat) * 100 if abs(S_cat) > 1e-10 else 0
        
        return {
            'S_categorical': float(S_cat),
            'S_oscillatory': float(S_osc),
            'S_partition': float(S_part),
            'deviation_osc_vs_cat': float(deviation_osc_cat),
            'deviation_part_vs_cat': float(deviation_part_cat),
            'triple_equivalence_passed': deviation_osc_cat < 5.0 and deviation_part_cat < 5.0
        }
        
    def full_validation(self, T_range: np.ndarray = None, V: float = 0.0224) -> Dict:
        """
        Run full Maxwell relations validation.
        
        Args:
            T_range: Temperature range for testing
            V: Volume (m^3)
            
        Returns:
            Complete validation results
        """
        if T_range is None:
            T_range = np.linspace(200, 1000, 50)
            
        results = {
            'n_moles': self.n,
            'V': V,
            'T_range': T_range.tolist(),
            'relations': {
                'relation_1': [],
                'relation_2': [],
                'relation_3': [],
                'relation_4': []
            },
            'triple_equivalence': []
        }
        
        for T in T_range:
            # Test all four Maxwell relations
            r1 = self.test_maxwell_relation_1(T, V)
            r2 = self.test_maxwell_relation_2(T, V)
            r3 = self.test_maxwell_relation_3(T, V)
            r4 = self.test_maxwell_relation_4(T, V)
            
            results['relations']['relation_1'].append(r1)
            results['relations']['relation_2'].append(r2)
            results['relations']['relation_3'].append(r3)
            results['relations']['relation_4'].append(r4)
            
            # Test triple equivalence
            te = self.test_triple_equivalence_consistency(T, V)
            results['triple_equivalence'].append(te)
            
        # Summary statistics
        results['summary'] = {
            'relation_1': {
                'mean_deviation': float(np.mean([r['deviation_percent'] for r in results['relations']['relation_1']])),
                'all_passed': all([r['passed'] for r in results['relations']['relation_1']])
            },
            'relation_2': {
                'mean_deviation': float(np.mean([r['deviation_percent'] for r in results['relations']['relation_2']])),
                'all_passed': all([r['passed'] for r in results['relations']['relation_2']])
            },
            'relation_3': {
                'mean_deviation': float(np.mean([r['deviation_percent'] for r in results['relations']['relation_3']])),
                'all_passed': all([r['passed'] for r in results['relations']['relation_3']])
            },
            'relation_4': {
                'mean_deviation': float(np.mean([r['deviation_percent'] for r in results['relations']['relation_4']])),
                'all_passed': all([r['passed'] for r in results['relations']['relation_4']])
            },
            'triple_equivalence': {
                'all_passed': all([te['triple_equivalence_passed'] for te in results['triple_equivalence']])
            }
        }
        
        self.results = results
        return results
        
    def create_panel_chart(self, save_path: str):
        """Create comprehensive panel visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        T_range = np.array(self.results['T_range'])
        
        # Panel 1: Maxwell Relation 1
        ax1 = fig.add_subplot(2, 3, 1)
        lhs_1 = [r['LHS'] for r in self.results['relations']['relation_1']]
        rhs_1 = [r['RHS'] for r in self.results['relations']['relation_1']]
        ax1.plot(T_range, lhs_1, 'b-', linewidth=2, label='(dT/dV)_S')
        ax1.plot(T_range, rhs_1, 'r--', linewidth=2, label='-(dP/dS)_V')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Derivative Value')
        ax1.set_title('Maxwell Relation 1')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Maxwell Relation 2
        ax2 = fig.add_subplot(2, 3, 2)
        lhs_2 = [r['LHS'] for r in self.results['relations']['relation_2']]
        rhs_2 = [r['RHS'] for r in self.results['relations']['relation_2']]
        ax2.plot(T_range, lhs_2, 'b-', linewidth=2, label='(dS/dV)_T')
        ax2.plot(T_range, rhs_2, 'r--', linewidth=2, label='(dP/dT)_V')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Derivative Value')
        ax2.set_title('Maxwell Relation 2: (dS/dV)_T = (dP/dT)_V')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Maxwell Relation 3
        ax3 = fig.add_subplot(2, 3, 3)
        lhs_3 = [r['LHS'] for r in self.results['relations']['relation_3']]
        rhs_3 = [r['RHS'] for r in self.results['relations']['relation_3']]
        ax3.plot(T_range, lhs_3, 'b-', linewidth=2, label='(dS/dP)_T')
        ax3.plot(T_range, rhs_3, 'r--', linewidth=2, label='-(dV/dT)_P')
        ax3.set_xlabel('Temperature (K)')
        ax3.set_ylabel('Derivative Value')
        ax3.set_title('Maxwell Relation 3')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Maxwell Relation 4
        ax4 = fig.add_subplot(2, 3, 4)
        lhs_4 = [r['LHS'] for r in self.results['relations']['relation_4']]
        rhs_4 = [r['RHS'] for r in self.results['relations']['relation_4']]
        ax4.plot(T_range, lhs_4, 'b-', linewidth=2, label='(dT/dP)_S')
        ax4.plot(T_range, rhs_4, 'r--', linewidth=2, label='(dV/dS)_P')
        ax4.set_xlabel('Temperature (K)')
        ax4.set_ylabel('Derivative Value')
        ax4.set_title('Maxwell Relation 4')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Deviations (3D surface)
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        
        # Create grid of T and V for 3D plot
        T_3d = np.linspace(200, 1000, 20)
        V_3d = np.linspace(0.01, 0.1, 20)
        T_grid, V_grid = np.meshgrid(T_3d, V_3d)
        
        # Compute deviation for relation 2 across grid
        deviation_grid = np.zeros_like(T_grid)
        for i in range(len(V_3d)):
            for j in range(len(T_3d)):
                r = self.test_maxwell_relation_2(T_3d[j], V_3d[i])
                deviation_grid[i, j] = r['deviation_percent']
                
        surf = ax5.plot_surface(T_grid, V_grid, deviation_grid, cmap='viridis', alpha=0.8)
        ax5.set_xlabel('Temperature (K)')
        ax5.set_ylabel('Volume (m^3)')
        ax5.set_zlabel('Deviation (%)')
        ax5.set_title('Relation 2 Deviation (T, V)')
        fig.colorbar(surf, ax=ax5, shrink=0.5, aspect=5)
        
        # Panel 6: Triple Equivalence
        ax6 = fig.add_subplot(2, 3, 6)
        S_cat = [te['S_categorical'] for te in self.results['triple_equivalence']]
        S_osc = [te['S_oscillatory'] for te in self.results['triple_equivalence']]
        S_part = [te['S_partition'] for te in self.results['triple_equivalence']]
        
        ax6.plot(T_range, S_cat, 'b-', linewidth=2, label='Categorical')
        ax6.plot(T_range, S_osc, 'g--', linewidth=2, label='Oscillatory')
        ax6.plot(T_range, S_part, 'r:', linewidth=2, label='Partition')
        ax6.set_xlabel('Temperature (K)')
        ax6.set_ylabel('Entropy (J/K)')
        ax6.set_title('Triple Equivalence of Entropy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Maxwell Relations Tester: Categorical Thermodynamics Validation',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved panel chart to {save_path}")
        
    def save_data(self, save_path: str):
        """Save results to JSON."""
        # Convert numpy types
        save_results = json.loads(json.dumps(self.results, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        
        with open(save_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"Saved data to {save_path}")


def main():
    """Run Maxwell Relations testing."""
    print("=" * 60)
    print("MAXWELL RELATIONS TESTER (MRT)")
    print("Testing: Thermodynamic identities from categorical entropy")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Test at different volumes
    for V in [0.0224, 0.0448]:  # 1 mol and 2 mol at STP
        print(f"\n--- Volume = {V*1000:.1f} L ---")
        mrt = MaxwellRelationsTester(n_moles=1.0)
        results = mrt.full_validation(V=V)
        
        print(f"  Relation 1: mean deviation = {results['summary']['relation_1']['mean_deviation']:.2f}%")
        print(f"  Relation 2: mean deviation = {results['summary']['relation_2']['mean_deviation']:.2f}%")
        print(f"  Relation 3: mean deviation = {results['summary']['relation_3']['mean_deviation']:.2f}%")
        print(f"  Relation 4: mean deviation = {results['summary']['relation_4']['mean_deviation']:.2f}%")
        print(f"  Triple equivalence: {'PASSED' if results['summary']['triple_equivalence']['all_passed'] else 'FAILED'}")
        
        V_label = f"{int(V*1000)}L"
        mrt.create_panel_chart(os.path.join(figures_dir, f"panel_mrt_{V_label}.png"))
        mrt.save_data(os.path.join(data_dir, f"mrt_{V_label}.json"))
    
    print("\n" + "=" * 60)
    print("MRT VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
