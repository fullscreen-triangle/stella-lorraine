"""
thermodynamics.py

Thermodynamic calculations for Maxwell demon system.
Tracks entropy, free energy, work extraction.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Boltzmann constant (normalized to 1 for simplicity)
KB = 1.0

@dataclass
class ThermodynamicState:
    """Complete thermodynamic state of system"""
    temperature_A: float
    temperature_B: float
    entropy_A: float
    entropy_B: float
    demon_entropy_cost: float
    particles_A: int
    particles_B: int

    @property
    def total_entropy(self) -> float:
        """Total system entropy"""
        return self.entropy_A + self.entropy_B + self.demon_entropy_cost

    @property
    def temperature_gradient(self) -> float:
        """Temperature difference"""
        return self.temperature_B - self.temperature_A

    @property
    def carnot_efficiency(self) -> float:
        """Maximum theoretical efficiency if used as heat engine"""
        if self.temperature_B == 0:
            return 0.0
        return 1.0 - (self.temperature_A / self.temperature_B)

    @property
    def max_work_extractable(self) -> float:
        """Maximum work extractable from temperature gradient"""
        # W_max = Q_h * η_carnot
        # Approximate Q_h from temperature and particle count
        Q_h = self.temperature_B * self.particles_B * KB
        return Q_h * self.carnot_efficiency

class ThermodynamicsAnalyzer:
    """Analyze thermodynamic properties of Maxwell demon system"""

    def __init__(self):
        self.states: List[ThermodynamicState] = []

    def record_state(self, system) -> ThermodynamicState:
        """Record current thermodynamic state"""
        state = ThermodynamicState(
            temperature_A=system.compartment_A.temperature,
            temperature_B=system.compartment_B.temperature,
            entropy_A=system.compartment_A.entropy,
            entropy_B=system.compartment_B.entropy,
            demon_entropy_cost=system.demon.entropy_cost,
            particles_A=system.compartment_A.particle_count,
            particles_B=system.compartment_B.particle_count
        )
        self.states.append(state)
        return state

    def entropy_production_rate(self, window: int = 10) -> float:
        """Calculate rate of entropy production"""
        if len(self.states) < window + 1:
            return 0.0

        recent_states = self.states[-window:]
        initial_entropy = recent_states[0].total_entropy
        final_entropy = recent_states[-1].total_entropy

        return (final_entropy - initial_entropy) / window

    def second_law_violated(self, tolerance: float = 1e-6) -> bool:
        """Check if second law is violated (total entropy decreases)"""
        if len(self.states) < 2:
            return False

        initial_entropy = self.states[0].total_entropy
        current_entropy = self.states[-1].total_entropy

        return current_entropy < (initial_entropy - tolerance)

    def information_to_work_conversion(self) -> Dict[str, float]:
        """Analyze conversion of information to useful work"""
        if len(self.states) < 2:
            return {'work_extracted': 0.0, 'information_used': 0.0, 'efficiency': 0.0}

        initial_state = self.states[0]
        final_state = self.states[-1]

        # Work extracted = increase in available work
        work_extracted = (final_state.max_work_extractable -
                         initial_state.max_work_extractable)

        # Information used = demon's entropy cost
        information_used = final_state.demon_entropy_cost

        # Efficiency = work / information cost
        efficiency = work_extracted / information_used if information_used > 0 else 0.0

        return {
            'work_extracted': work_extracted,
            'information_used': information_used,
            'efficiency': efficiency
        }

    def landauer_limit_check(self) -> Dict[str, float]:
        """Check if system respects Landauer's limit"""
        if len(self.states) < 2:
            return {'landauer_limit': 0.0, 'actual_cost': 0.0, 'ratio': 0.0}

        final_state = self.states[-1]

        # Landauer limit: kT ln(2) per bit erased
        # Assuming temperature ~ 1, this is ~ 0.693 per bit
        landauer_limit = 0.693

        # Actual cost per bit
        actual_cost = (final_state.demon_entropy_cost /
                      final_state.demon_entropy_cost if final_state.demon_entropy_cost > 0 else 1.0)

        return {
            'landauer_limit': landauer_limit,
            'actual_cost': actual_cost,
            'ratio': actual_cost / landauer_limit if landauer_limit > 0 else 0.0
        }

    def generate_report(self) -> str:
        """Generate comprehensive thermodynamic report"""
        if not self.states:
            return "No data recorded."

        initial = self.states[0]
        final = self.states[-1]

        report = []
        report.append("=" * 60)
        report.append("THERMODYNAMIC ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("INITIAL STATE:")
        report.append(f"  Temperature A: {initial.temperature_A:.4f}")
        report.append(f"  Temperature B: {initial.temperature_B:.4f}")
        report.append(f"  Total entropy: {initial.total_entropy:.4f}")
        report.append("")

        report.append("FINAL STATE:")
        report.append(f"  Temperature A: {final.temperature_A:.4f}")
        report.append(f"  Temperature B: {final.temperature_B:.4f}")
        report.append(f"  Temperature gradient: {final.temperature_gradient:.4f}")
        report.append(f"  Total entropy: {final.total_entropy:.4f}")
        report.append("")

        report.append("ENTROPY ANALYSIS:")
        entropy_change = final.total_entropy - initial.total_entropy
        report.append(f"  Total entropy change: {entropy_change:.4f}")
        report.append(f"  Entropy production rate: {self.entropy_production_rate():.6f}")
        report.append(f"  Second law violated: {self.second_law_violated()}")
        report.append("")

        report.append("INFORMATION-WORK CONVERSION:")
        conversion = self.information_to_work_conversion()
        report.append(f"  Work extracted: {conversion['work_extracted']:.4f}")
        report.append(f"  Information used: {conversion['information_used']:.4f}")
        report.append(f"  Efficiency: {conversion['efficiency']:.4f}")
        report.append("")

        report.append("LANDAUER LIMIT:")
        landauer = self.landauer_limit_check()
        report.append(f"  Landauer limit: {landauer['landauer_limit']:.4f}")
        report.append(f"  Actual cost: {landauer['actual_cost']:.4f}")
        report.append(f"  Ratio: {landauer['ratio']:.4f}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

def main():
    """Test thermodynamics module"""
    print("Testing thermodynamics module...")

    # Create mock states
    analyzer = ThermodynamicsAnalyzer()

    # Simulate states
    for i in range(100):
        # Mock state with increasing temperature gradient
        from types import SimpleNamespace
        mock_system = SimpleNamespace(
            compartment_A=SimpleNamespace(
                temperature=1.0 - 0.01 * i,
                entropy=1.0 + 0.001 * i,
                particle_count=50
            ),
            compartment_B=SimpleNamespace(
                temperature=1.0 + 0.01 * i,
                entropy=1.0 + 0.001 * i,
                particle_count=50
            ),
            demon=SimpleNamespace(
                entropy_cost=0.01 * i
            )
        )
        analyzer.record_state(mock_system)

    print(analyzer.generate_report())

if __name__ == "__main__":
    main()
