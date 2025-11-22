# core/categorical_state.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
from dataclasses import dataclass
from typing import Tuple, Optional
import scipy.constants as const

@dataclass
class EntropicCoordinates:
    """
    Entropic coordinates S = (Sk, St, Se)

    Sk: Kinetic entropy (momentum distribution)
    St: Temporal entropy (timing uncertainty)
    Se: Environmental entropy (configuration space)
    """
    Sk: float  # Kinetic entropy [J/K]
    St: float  # Temporal entropy [J/K]
    Se: float  # Environmental entropy [J/K]

    def total_entropy(self) -> float:
        """Total entropy S = Sk + St + Se"""
        return self.Sk + self.St + self.Se

    def momentum_component(self) -> float:
        """Momentum-related entropy for temperature extraction"""
        return self.Sk

    def configurational_component(self) -> float:
        """Configuration space entropy"""
        return self.Se


class CategoricalState:
    """
    Categorical state C(t) representation

    Encodes system state through entropic coordinates
    rather than wavefunction or density matrix.
    """

    def __init__(self,
                 entropy_coords: EntropicCoordinates,
                 timestamp: float,
                 system_params: Optional[dict] = None):
        """
        Initialize categorical state

        Args:
            entropy_coords: Entropic coordinates (Sk, St, Se)
            timestamp: Time of measurement [s]
            system_params: System-specific parameters (mass, etc.)
        """
        self.S = entropy_coords
        self.t = timestamp
        self.params = system_params or {}

    def evolve(self, dt: float) -> 'CategoricalState':
        """
        Evolve categorical state by time dt

        Uses categorical prediction with effective velocity
        vcat/c ∈ [2.846, 65.71]
        """
        # Simplified evolution (implement full dynamics as needed)
        new_S = EntropicCoordinates(
            Sk=self.S.Sk,  # Conserved in isolated system
            St=self.S.St + self._temporal_diffusion(dt),
            Se=self.S.Se + self._environmental_coupling(dt)
        )
        return CategoricalState(new_S, self.t + dt, self.params)

    def _temporal_diffusion(self, dt: float) -> float:
        """Temporal entropy increase"""
        # Based on timing precision δt ∼ 2×10⁻¹⁵ s
        delta_t_precision = 2e-15  # seconds
        return const.k * np.log(1 + dt / delta_t_precision)

    def _environmental_coupling(self, dt: float) -> float:
        """Environmental entropy coupling"""
        # Heating rate < 1 fK/s from far-detuned optical coupling
        heating_rate = 1e-15  # K/s
        return const.k * heating_rate * dt / self.get_temperature()

    def get_temperature(self) -> float:
        """
        Extract temperature from categorical state

        Returns:
            Temperature [K]
        """
        # Will be implemented in thermometry module
        raise NotImplementedError("Use ThermometryAnalyzer.extract_temperature()")

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'Sk': self.S.Sk,
            'St': self.S.St,
            'Se': self.S.Se,
            'timestamp': self.t,
            'params': self.params
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CategoricalState':
        """Deserialize from dictionary"""
        S = EntropicCoordinates(
            Sk=data['Sk'],
            St=data['St'],
            Se=data['Se']
        )
        return cls(S, data['timestamp'], data.get('params', {}))


class CategoricalStateEstimator:
    """
    Estimate categorical state from measurements
    """

    def __init__(self, system_mass: float, num_particles: int):
        """
        Args:
            system_mass: Particle mass [kg] (e.g., Rb-87: 1.443×10⁻²⁵ kg)
            num_particles: Number of particles in ensemble
        """
        self.m = system_mass
        self.N = num_particles

    def from_momentum_distribution(self,
                                   momenta: np.ndarray,
                                   timestamp: float) -> CategoricalState:
        """
        Construct categorical state from momentum measurements

        Args:
            momenta: Array of momentum values [kg·m/s]
            timestamp: Measurement time [s]

        Returns:
            CategoricalState
        """
        # Calculate momentum entropy
        p_squared = momenta**2
        mean_p_squared = np.mean(p_squared)

        # Kinetic entropy from momentum distribution width
        # Sk ∼ kB ln(σp) where σp is momentum width
        sigma_p = np.std(momenta)
        Sk = const.k * np.log(sigma_p / const.hbar + 1)

        # Temporal entropy from measurement timing uncertainty
        # δt ∼ 2×10⁻¹⁵ s from H+ oscillator sync
        delta_t = 2e-15
        St = const.k * np.log(timestamp / delta_t + 1)

        # Environmental entropy from configuration space
        # Se ∼ kB ln(Ω) where Ω is accessible phase space volume
        # For thermal distribution: Se ∼ (3/2) N kB
        Se = 1.5 * self.N * const.k

        S = EntropicCoordinates(Sk=Sk, St=St, Se=Se)

        params = {
            'mass': self.m,
            'num_particles': self.N,
            'mean_p_squared': mean_p_squared,
            'sigma_p': sigma_p
        }

        return CategoricalState(S, timestamp, params)

    def from_position_distribution(self,
                                   positions: np.ndarray,
                                   velocities: np.ndarray,
                                   timestamp: float) -> CategoricalState:
        """
        Construct categorical state from position/velocity measurements

        Args:
            positions: Array of positions [m]
            velocities: Array of velocities [m/s]
            timestamp: Measurement time [s]

        Returns:
            CategoricalState
        """
        # Convert velocities to momenta
        momenta = self.m * velocities

        # Use momentum-based construction
        cat_state = self.from_momentum_distribution(momenta, timestamp)

        # Add position-based environmental entropy
        sigma_x = np.std(positions)
        Se_position = const.k * np.log(sigma_x / (const.hbar / (self.m * np.std(velocities))) + 1)

        # Update environmental entropy
        cat_state.S.Se = max(cat_state.S.Se, Se_position)

        return cat_state


# Example usage
if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from datetime import datetime
    from pathlib import Path

    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("CATEGORICAL STATE VALIDATION")
    print("=" * 70)

    # Rb-87 BEC parameters
    m_Rb87 = 1.443e-25  # kg
    N_atoms = 1e5

    # Test multiple temperatures
    temperatures = [10e-9, 50e-9, 100e-9, 500e-9, 1e-6]  # nK to μK
    results = {
        'timestamp': timestamp,
        'particle_mass_kg': m_Rb87,
        'num_particles': N_atoms,
        'temperature_tests': []
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect data
    temps_list = []
    Sk_list = []
    St_list = []
    Se_list = []
    S_total_list = []

    for T in temperatures:
        # Simulate momentum distribution
        sigma_v = np.sqrt(const.k * T / m_Rb87)
        velocities = np.random.normal(0, sigma_v, int(N_atoms))

        # Estimate categorical state
        estimator = CategoricalStateEstimator(m_Rb87, N_atoms)
        cat_state = estimator.from_momentum_distribution(
            m_Rb87 * velocities,
            timestamp=0.0
        )

        print(f"\nCategorical State at T = {T*1e9:.1f} nK:")
        print(f"  Sk = {cat_state.S.Sk:.6e} J/K")
        print(f"  St = {cat_state.S.St:.6e} J/K")
        print(f"  Se = {cat_state.S.Se:.6e} J/K")
        print(f"  Total S = {cat_state.S.total_entropy():.6e} J/K")

        # Store results
        temps_list.append(T * 1e9)
        Sk_list.append(cat_state.S.Sk)
        St_list.append(cat_state.S.St)
        Se_list.append(cat_state.S.Se)
        S_total_list.append(cat_state.S.total_entropy())

        results['temperature_tests'].append({
            'temperature_nK': T * 1e9,
            'Sk': cat_state.S.Sk,
            'St': cat_state.S.St,
            'Se': cat_state.S.Se,
            'S_total': cat_state.S.total_entropy()
        })

    # Panel A: Entropy components vs temperature
    ax = axes[0, 0]
    ax.loglog(temps_list, Sk_list, 'o-', label='Sk (Kinetic)', linewidth=2)
    ax.loglog(temps_list, St_list, 's-', label='St (Temporal)', linewidth=2)
    ax.loglog(temps_list, Se_list, '^-', label='Se (Environmental)', linewidth=2)
    ax.set_xlabel('Temperature [nK]')
    ax.set_ylabel('Entropy [J/K]')
    ax.set_title('A) Entropy Components vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Panel B: Total entropy
    ax = axes[0, 1]
    ax.loglog(temps_list, S_total_list, 'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature [nK]')
    ax.set_ylabel('Total Entropy [J/K]')
    ax.set_title('B) Total Entropy S = Sk + St + Se')
    ax.grid(True, alpha=0.3, which='both')

    # Panel C: Entropy ratios
    ax = axes[1, 0]
    Sk_ratio = np.array(Sk_list) / np.array(S_total_list)
    St_ratio = np.array(St_list) / np.array(S_total_list)
    Se_ratio = np.array(Se_list) / np.array(S_total_list)
    ax.semilogx(temps_list, Sk_ratio, 'o-', label='Sk/S_total', linewidth=2)
    ax.semilogx(temps_list, St_ratio, 's-', label='St/S_total', linewidth=2)
    ax.semilogx(temps_list, Se_ratio, '^-', label='Se/S_total', linewidth=2)
    ax.set_xlabel('Temperature [nK]')
    ax.set_ylabel('Entropy Fraction')
    ax.set_title('C) Entropy Component Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Temperature scaling
    ax = axes[1, 1]
    # Theoretical: Sk ~ (3/2) N kB ln(T)
    ax.loglog(temps_list, Sk_list, 'o-', label='Measured Sk', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature [nK]')
    ax.set_ylabel('Kinetic Entropy Sk [J/K]')
    ax.set_title('D) Kinetic Entropy Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"categorical_state_validation_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")

    # Save JSON results
    json_path = output_dir / f"categorical_state_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {json_path}")

    print("\n" + "=" * 70)
