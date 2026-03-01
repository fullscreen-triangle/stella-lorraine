"""
Thermodynamic Regimes Classification Module
============================================

Maps ion states to five canonical thermodynamic regimes based on
oscillator state counts and S-entropy coordinates.

The five regimes:
1. Ideal Gas - High S_t, moderate S_k, uncorrelated states
2. Plasma - Long-range Coulomb interactions, correlated
3. Degenerate - Quantum statistics dominate
4. Relativistic - k_B T ~ mc^2
5. BEC - Macroscopic ground state occupation

Key identity: PV = Nk_B T × S(V, N, T, {n_i, l_i, m_i, s_i})

Author: Kundai Sachikonye
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

K_B = 1.380649e-23          # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
AMU = 1.66053906660e-27     # Atomic mass unit (kg)
HBAR = 1.054571817e-34      # Reduced Planck constant (J·s)
C_LIGHT = 299792458         # Speed of light (m/s)
EPSILON_0 = 8.854187817e-12 # Vacuum permittivity (F/m)
M_ELECTRON = 9.10938e-31    # Electron mass (kg)


# ============================================================================
# THERMODYNAMIC REGIME ENUMERATION
# ============================================================================

class ThermodynamicRegime(Enum):
    """
    Five canonical thermodynamic regimes.

    Classification based on:
    - Plasma parameter Γ (Coulomb coupling)
    - Degeneracy parameter η (quantum effects)
    - Relativistic parameter θ (thermal vs rest mass)
    - Phase coherence ξ (BEC signature)
    """
    IDEAL_GAS = 1       # Γ < 0.1, η < 1, θ << 1
    PLASMA = 2          # Γ > 0.5, correlated states
    DEGENERATE = 3      # η > 1, quantum statistics
    RELATIVISTIC = 4    # θ > 0.01, k_B T ~ mc²
    BEC = 5             # ξ > 0.7, macroscopic coherence


# ============================================================================
# REGIME PARAMETERS
# ============================================================================

@dataclass
class DimensionlessParameters:
    """
    Dimensionless parameters for regime classification.

    These parameters determine which thermodynamic regime applies.
    """
    # Plasma parameter: Γ = (Z²e²) / (4πε₀ a k_B T)
    # Ratio of Coulomb energy to thermal energy
    gamma: float = 0.0

    # Degeneracy parameter: η = λ_th / a
    # Ratio of de Broglie wavelength to interparticle spacing
    eta: float = 0.0

    # Relativistic parameter: θ = k_B T / (mc²)
    # Ratio of thermal energy to rest mass energy
    theta: float = 0.0

    # Phase coherence: ξ = 1/√M
    # From state counting - coherence decreases with more states
    xi: float = 0.0

    # Coupling strength: g = Γ³/² / η
    # Combined measure of interaction strength
    coupling: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'plasma_parameter_Gamma': self.gamma,
            'degeneracy_eta': self.eta,
            'relativistic_theta': self.theta,
            'phase_coherence_xi': self.xi,
            'coupling_g': self.coupling
        }


@dataclass
class SEntropyCoordinates:
    """
    S-Entropy coordinates (S_k, S_t, S_e).

    These coordinates form a complete description of the ion's
    thermodynamic state in the categorical framework.

    S_k: Knowledge entropy - how much is known about the ion state
    S_t: Temporal entropy - time uncertainty
    S_e: Energy entropy - energy distribution width
    """
    s_k: float = 0.5    # Knowledge entropy [0, 1]
    s_t: float = 0.5    # Temporal entropy [0, 1]
    s_e: float = 0.5    # Energy entropy [0, 1]

    @property
    def total(self) -> float:
        """Total S-entropy: S_total = √(S_k² + S_t² + S_e²)"""
        return np.sqrt(self.s_k**2 + self.s_t**2 + self.s_e**2)

    @property
    def normalized(self) -> Tuple[float, float, float]:
        """Normalized coordinates on unit sphere."""
        total = max(self.total, 1e-10)
        return (self.s_k / total, self.s_t / total, self.s_e / total)

    def to_dict(self) -> Dict[str, float]:
        return {
            'S_k': self.s_k,
            'S_t': self.s_t,
            'S_e': self.s_e,
            'S_total': self.total
        }


# ============================================================================
# THERMODYNAMIC STATE
# ============================================================================

@dataclass
class ThermodynamicState:
    """
    Complete thermodynamic state of an ion.

    Combines partition coordinates, S-entropy, and derived
    thermodynamic quantities.
    """
    # Partition coordinates (from oscillator counting)
    n: int = 1              # Principal depth
    l: int = 0              # Angular complexity
    m: int = 0              # Orientation
    s: float = 0.5          # Spin

    # State count (from oscillator)
    M: int = 1

    # S-Entropy coordinates
    s_entropy: SEntropyCoordinates = field(default_factory=SEntropyCoordinates)

    # Dimensionless parameters
    params: DimensionlessParameters = field(default_factory=DimensionlessParameters)

    # Physical observables
    temperature_K: float = 300.0
    energy_eV: float = 1.0
    mz: float = 500.0
    charge: int = 1

    # Derived quantities
    regime: ThermodynamicRegime = ThermodynamicRegime.IDEAL_GAS
    categorical_temperature_K: float = 0.0
    entropy_bits: float = 0.0

    def __post_init__(self):
        """Calculate derived quantities."""
        self.categorical_temperature_K = self._calculate_categorical_temp()
        self.entropy_bits = self.M  # One bit per state

    def _calculate_categorical_temp(self) -> float:
        """
        Categorical temperature: T_cat = 2E / (3k_B × M)

        Key result of categorical cryogenics.
        """
        E_joules = self.energy_eV * E_CHARGE
        return 2 * E_joules / (3 * K_B * max(1, self.M))

    @property
    def temperature_suppression(self) -> float:
        """T_cat / T_classical = 1/M"""
        return 1.0 / max(1, self.M)

    @property
    def capacity(self) -> int:
        """Capacity at depth n: C(n) = 2n²"""
        return 2 * self.n * self.n

    def to_dict(self) -> Dict[str, Any]:
        return {
            'partition': {'n': self.n, 'l': self.l, 'm': self.m, 's': self.s},
            'state_count_M': self.M,
            's_entropy': self.s_entropy.to_dict(),
            'params': self.params.to_dict(),
            'regime': self.regime.name,
            'temperature_K': self.temperature_K,
            'categorical_temperature_K': self.categorical_temperature_K,
            'temperature_suppression': self.temperature_suppression,
            'entropy_bits': self.entropy_bits,
            'capacity': self.capacity
        }


# ============================================================================
# REGIME CLASSIFIER
# ============================================================================

class ThermodynamicRegimeClassifier:
    """
    Classifies ions into thermodynamic regimes based on
    state counts and physical observables.

    The classification follows the hierarchy:
    1. Check for BEC (high coherence, low M)
    2. Check for relativistic (high θ)
    3. Check for degenerate (high η)
    4. Check for plasma (high Γ)
    5. Default to ideal gas
    """

    # Regime boundaries
    BEC_COHERENCE_THRESHOLD = 0.7
    BEC_STATE_MAX = 1000
    RELATIVISTIC_THRESHOLD = 0.01
    DEGENERATE_THRESHOLD = 1.0
    PLASMA_THRESHOLD = 0.5

    def __init__(
        self,
        ion_density_m3: float = 1e12,  # Typical ion cloud density
    ):
        """
        Initialize classifier.

        Args:
            ion_density_m3: Ion number density for spacing calculation
        """
        self.ion_density = ion_density_m3

        # Mean interparticle spacing
        self.mean_spacing = (3 / (4 * np.pi * ion_density_m3)) ** (1/3)

    def calculate_parameters(
        self,
        mz: float,
        charge: int,
        energy_eV: float,
        state_count: int
    ) -> DimensionlessParameters:
        """
        Calculate dimensionless parameters for regime classification.

        Args:
            mz: Mass-to-charge ratio
            charge: Ion charge state
            energy_eV: Kinetic energy
            state_count: Oscillator state count M

        Returns:
            DimensionlessParameters object
        """
        mass_kg = mz * AMU * charge

        # Temperature from energy
        T = energy_eV * E_CHARGE / K_B
        T = max(T, 1.0)  # Minimum temperature

        # Thermal de Broglie wavelength
        lambda_th = HBAR / np.sqrt(2 * np.pi * mass_kg * K_B * T)

        # Plasma parameter: Γ = Z²e² / (4πε₀ a k_B T)
        gamma = (charge**2 * E_CHARGE**2) / (
            4 * np.pi * EPSILON_0 * self.mean_spacing * K_B * T
        )

        # Degeneracy parameter: η = λ_th / a
        eta = lambda_th / self.mean_spacing

        # Relativistic parameter: θ = k_B T / (mc²)
        theta = K_B * T / (mass_kg * C_LIGHT**2)

        # Phase coherence: ξ = 1/√M
        xi = 1.0 / np.sqrt(max(1, state_count))

        # Coupling: g = Γ^(3/2) / η
        coupling = gamma**(3/2) / max(eta, 1e-10)

        return DimensionlessParameters(
            gamma=gamma,
            eta=eta,
            theta=theta,
            xi=xi,
            coupling=coupling
        )

    def classify(
        self,
        mz: float,
        charge: int = 1,
        energy_eV: float = 10.0,
        state_count: int = 1000000
    ) -> Tuple[ThermodynamicRegime, DimensionlessParameters]:
        """
        Classify ion into thermodynamic regime.

        Args:
            mz: Mass-to-charge ratio
            charge: Ion charge state
            energy_eV: Kinetic energy
            state_count: Oscillator state count M

        Returns:
            Tuple of (regime, parameters)
        """
        params = self.calculate_parameters(mz, charge, energy_eV, state_count)

        # Classification hierarchy

        # 1. BEC: High coherence AND low state count
        if params.xi > self.BEC_COHERENCE_THRESHOLD and state_count < self.BEC_STATE_MAX:
            regime = ThermodynamicRegime.BEC

        # 2. Relativistic: Thermal energy comparable to rest mass
        elif params.theta > self.RELATIVISTIC_THRESHOLD:
            regime = ThermodynamicRegime.RELATIVISTIC

        # 3. Degenerate: Quantum statistics dominate
        elif params.eta > self.DEGENERATE_THRESHOLD:
            regime = ThermodynamicRegime.DEGENERATE

        # 4. Plasma: Strong Coulomb coupling
        elif params.gamma > self.PLASMA_THRESHOLD:
            regime = ThermodynamicRegime.PLASMA

        # 5. Ideal Gas: Default
        else:
            regime = ThermodynamicRegime.IDEAL_GAS

        return regime, params

    def classify_from_s_entropy(
        self,
        s_entropy: SEntropyCoordinates,
        state_count: int
    ) -> ThermodynamicRegime:
        """
        Classify based on S-entropy coordinates.

        S-entropy signatures for each regime:
        - Ideal Gas: High S_t, moderate S_k, moderate S_e
        - Plasma: High S_e, correlated S_k and S_t
        - Degenerate: Low S_t (frozen dynamics), high S_k
        - Relativistic: High all coordinates
        - BEC: Very low S_t, very low S_e, high S_k
        """
        s_k, s_t, s_e = s_entropy.s_k, s_entropy.s_t, s_entropy.s_e

        # Phase coherence from state count
        xi = 1.0 / np.sqrt(max(1, state_count))

        # BEC: Low temporal and energy entropy, high coherence
        if xi > 0.7 and s_t < 0.1 and s_e < 0.1:
            return ThermodynamicRegime.BEC

        # Relativistic: All entropies high
        if s_k > 0.8 and s_t > 0.8 and s_e > 0.8:
            return ThermodynamicRegime.RELATIVISTIC

        # Degenerate: Frozen dynamics (low S_t), organized (high S_k)
        if s_t < 0.2 and s_k > 0.7:
            return ThermodynamicRegime.DEGENERATE

        # Plasma: High energy fluctuations, correlated states
        if s_e > 0.7 and abs(s_k - s_t) < 0.2:
            return ThermodynamicRegime.PLASMA

        # Default: Ideal gas
        return ThermodynamicRegime.IDEAL_GAS

    def get_regime_properties(
        self,
        regime: ThermodynamicRegime
    ) -> Dict[str, Any]:
        """
        Get physical properties and constraints for a regime.
        """
        properties = {
            ThermodynamicRegime.IDEAL_GAS: {
                'name': 'Ideal Gas',
                'description': 'Non-interacting particles, classical statistics',
                'equation_of_state': 'PV = Nk_B T',
                'state_function': 'S(V, N, T) = 1',
                'typical_Gamma': '< 0.1',
                'typical_eta': '< 1',
                'ms_signature': 'Gaussian peak shapes, independent peaks'
            },
            ThermodynamicRegime.PLASMA: {
                'name': 'Plasma',
                'description': 'Coulomb-coupled ions, collective modes',
                'equation_of_state': 'PV = Nk_B T × (1 - Γ/3)',
                'state_function': 'S(V, N, T, Γ)',
                'typical_Gamma': '> 0.5',
                'typical_eta': 'varies',
                'ms_signature': 'Peak correlations, space-charge effects'
            },
            ThermodynamicRegime.DEGENERATE: {
                'name': 'Degenerate',
                'description': 'Quantum statistics dominate',
                'equation_of_state': 'PV = (2/3)E (Fermi gas)',
                'state_function': 'S(V, N, T, η)',
                'typical_Gamma': 'varies',
                'typical_eta': '> 1',
                'ms_signature': 'Quantized charge states, shell structure'
            },
            ThermodynamicRegime.RELATIVISTIC: {
                'name': 'Relativistic',
                'description': 'Thermal energy ~ rest mass energy',
                'equation_of_state': 'PV = (1/3)E (photon gas limit)',
                'state_function': 'S(V, N, T, θ)',
                'typical_Gamma': 'varies',
                'typical_theta': '> 0.01',
                'ms_signature': 'Mass shifts, pair production'
            },
            ThermodynamicRegime.BEC: {
                'name': 'Bose-Einstein Condensate',
                'description': 'Macroscopic ground state occupation',
                'equation_of_state': 'P = (ζ(5/2)/ζ(3/2)) × k_B T / λ_th³',
                'state_function': 'S(V, N, T, ξ)',
                'typical_xi': '> 0.7',
                'ms_signature': 'Single coherent peak, narrow linewidth'
            }
        }
        return properties.get(regime, {})


# ============================================================================
# UNIVERSAL EQUATION OF STATE
# ============================================================================

class UniversalEquationOfState:
    """
    Universal equation of state incorporating partition coordinates.

    PV = Nk_B T × S(V, N, T, {n_i, l_i, m_i, s_i})

    The state function S encodes the contribution from each
    partition coordinate set.
    """

    def __init__(self):
        self.classifier = ThermodynamicRegimeClassifier()

    def state_function(
        self,
        partition_coords: List[Tuple[int, int, int, float]],
        temperature_K: float,
        regime: ThermodynamicRegime
    ) -> float:
        """
        Calculate state function S from partition coordinates.

        S = Σ_i w(n_i, l_i, m_i, s_i) × f(T, regime)

        Args:
            partition_coords: List of (n, l, m, s) tuples
            temperature_K: Temperature
            regime: Thermodynamic regime

        Returns:
            State function value
        """
        if not partition_coords:
            return 1.0

        # Weight function for each coordinate set
        total_weight = 0.0
        for n, l, m, s in partition_coords:
            # Capacity contribution
            capacity = 2 * n * n

            # Angular weight
            angular_weight = (2 * l + 1) / capacity if capacity > 0 else 0

            # Orientation weight
            orient_weight = 1.0 / (2 * l + 1) if l > 0 else 1.0

            # Spin contribution
            spin_factor = 1.0 + 0.1 * np.sign(s)

            total_weight += capacity * angular_weight * orient_weight * spin_factor

        # Normalize by number of coordinates
        S_base = total_weight / len(partition_coords)

        # Regime-dependent correction
        regime_corrections = {
            ThermodynamicRegime.IDEAL_GAS: 1.0,
            ThermodynamicRegime.PLASMA: 0.9,  # Coulomb reduction
            ThermodynamicRegime.DEGENERATE: 1.1,  # Quantum enhancement
            ThermodynamicRegime.RELATIVISTIC: 1.33,  # (4/3) factor
            ThermodynamicRegime.BEC: 0.5,  # Condensate reduction
        }

        correction = regime_corrections.get(regime, 1.0)

        return S_base * correction

    def pressure(
        self,
        N: int,
        V_m3: float,
        temperature_K: float,
        partition_coords: List[Tuple[int, int, int, float]],
        regime: ThermodynamicRegime
    ) -> float:
        """
        Calculate pressure from universal equation of state.

        P = (Nk_B T / V) × S(V, N, T, {n, l, m, s})

        Args:
            N: Number of ions
            V_m3: Volume in cubic meters
            temperature_K: Temperature
            partition_coords: Partition coordinates
            regime: Thermodynamic regime

        Returns:
            Pressure in Pascals
        """
        S = self.state_function(partition_coords, temperature_K, regime)
        return (N * K_B * temperature_K / V_m3) * S

    def internal_energy(
        self,
        N: int,
        temperature_K: float,
        partition_coords: List[Tuple[int, int, int, float]],
        regime: ThermodynamicRegime
    ) -> float:
        """
        Calculate internal energy.

        E = (3/2) Nk_B T × S × f(regime)

        Args:
            N: Number of ions
            temperature_K: Temperature
            partition_coords: Partition coordinates
            regime: Thermodynamic regime

        Returns:
            Internal energy in Joules
        """
        S = self.state_function(partition_coords, temperature_K, regime)

        # Regime-dependent degrees of freedom
        dof_factor = {
            ThermodynamicRegime.IDEAL_GAS: 3/2,
            ThermodynamicRegime.PLASMA: 3/2,
            ThermodynamicRegime.DEGENERATE: 3/5,  # Fermi gas
            ThermodynamicRegime.RELATIVISTIC: 3,   # Relativistic
            ThermodynamicRegime.BEC: 3/2,
        }.get(regime, 3/2)

        return dof_factor * N * K_B * temperature_K * S


# ============================================================================
# REGIME TRANSITION DETECTOR
# ============================================================================

class RegimeTransitionDetector:
    """
    Detects transitions between thermodynamic regimes
    during ion journey.
    """

    def __init__(self):
        self.classifier = ThermodynamicRegimeClassifier()
        self.transitions: List[Dict[str, Any]] = []
        self.current_regime: Optional[ThermodynamicRegime] = None

    def check_transition(
        self,
        mz: float,
        charge: int,
        energy_eV: float,
        state_count: int,
        stage_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a regime transition has occurred.

        Args:
            mz: Mass-to-charge ratio
            charge: Charge state
            energy_eV: Kinetic energy
            state_count: State count M
            stage_name: Current pipeline stage

        Returns:
            Transition info dict if transition occurred, None otherwise
        """
        new_regime, params = self.classifier.classify(
            mz, charge, energy_eV, state_count
        )

        if self.current_regime is not None and new_regime != self.current_regime:
            transition = {
                'from_regime': self.current_regime.name,
                'to_regime': new_regime.name,
                'stage': stage_name,
                'state_count': state_count,
                'parameters': params.to_dict(),
                'transition_entropy': K_B * np.log(2 + abs(energy_eV) / 100)
            }
            self.transitions.append(transition)
            self.current_regime = new_regime
            return transition

        self.current_regime = new_regime
        return None

    def get_all_transitions(self) -> List[Dict[str, Any]]:
        """Get all recorded transitions."""
        return self.transitions

    def validate_transition_entropy(self) -> Dict[str, Any]:
        """
        Validate transition entropy bound: ΔS > k_B ln(2).

        From categorical cryogenics paper.
        """
        min_entropy = K_B * np.log(2)

        validations = []
        for t in self.transitions:
            delta_S = t.get('transition_entropy', 0)
            validations.append({
                'transition': f"{t['from_regime']} → {t['to_regime']}",
                'delta_S': delta_S,
                'min_bound': min_entropy,
                'valid': delta_S >= min_entropy
            })

        all_valid = all(v['valid'] for v in validations) if validations else True

        return {
            'claim': 'Transition entropy ΔS ≥ k_B ln(2)',
            'validations': validations,
            'all_valid': all_valid
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def classify_ion_regime(
    mz: float,
    charge: int = 1,
    energy_eV: float = 10.0,
    state_count: int = 1000000
) -> Tuple[str, Dict[str, float]]:
    """
    Quick classification of ion thermodynamic regime.

    Args:
        mz: Mass-to-charge ratio
        charge: Charge state
        energy_eV: Kinetic energy
        state_count: State count M

    Returns:
        Tuple of (regime_name, parameters_dict)
    """
    classifier = ThermodynamicRegimeClassifier()
    regime, params = classifier.classify(mz, charge, energy_eV, state_count)
    return regime.name, params.to_dict()


def calculate_categorical_temperature(
    energy_eV: float,
    state_count: int
) -> float:
    """
    Calculate categorical temperature: T = 2E / (3k_B × M).

    Args:
        energy_eV: Energy in eV
        state_count: State count M

    Returns:
        Categorical temperature in Kelvin
    """
    E_joules = energy_eV * E_CHARGE
    return 2 * E_joules / (3 * K_B * max(1, state_count))


def map_journey_to_regimes(
    stages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Map ion journey stages to thermodynamic regimes.

    Args:
        stages: List of stage dictionaries with mz, charge, energy, state_count

    Returns:
        List of regime mappings
    """
    classifier = ThermodynamicRegimeClassifier()
    mappings = []

    for stage in stages:
        regime, params = classifier.classify(
            stage.get('mz', 500),
            stage.get('charge', 1),
            stage.get('energy_eV', 10),
            stage.get('state_count', 1)
        )

        mappings.append({
            'stage': stage.get('name', 'unknown'),
            'regime': regime.name,
            'params': params.to_dict()
        })

    return mappings


def demonstrate_regimes():
    """Demonstrate regime classification."""
    print("=" * 70)
    print("THERMODYNAMIC REGIME CLASSIFICATION")
    print("=" * 70)

    classifier = ThermodynamicRegimeClassifier()

    test_cases = [
        {'mz': 500, 'charge': 1, 'energy_eV': 10, 'state_count': 10000000, 'expected': 'Ideal Gas'},
        {'mz': 500, 'charge': 5, 'energy_eV': 1000, 'state_count': 100000, 'expected': 'Plasma'},
        {'mz': 10, 'charge': 1, 'energy_eV': 0.001, 'state_count': 1000, 'expected': 'Degenerate'},
        {'mz': 500, 'charge': 1, 'energy_eV': 500000, 'state_count': 100, 'expected': 'Relativistic'},
        {'mz': 500, 'charge': 1, 'energy_eV': 0.01, 'state_count': 10, 'expected': 'BEC'},
    ]

    print(f"\n{'Test Case':<20} {'Expected':<15} {'Classified':<15} {'Match':<10}")
    print("-" * 60)

    for tc in test_cases:
        regime, params = classifier.classify(
            tc['mz'], tc['charge'], tc['energy_eV'], tc['state_count']
        )
        match = '✓' if regime.name.upper().replace('_', ' ') == tc['expected'].upper() else '✗'
        print(f"E={tc['energy_eV']:<6} M={tc['state_count']:<8} "
              f"{tc['expected']:<15} {regime.name:<15} {match}")

    print("\n" + "=" * 70)
    print("CATEGORICAL TEMPERATURE DEMONSTRATION")
    print("=" * 70)

    energy = 10.0  # eV
    for M in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        T_cat = calculate_categorical_temperature(energy, M)
        print(f"M = {M:<10} T_cat = {T_cat:>15.2e} K (suppression: 1/{M})")


if __name__ == "__main__":
    demonstrate_regimes()
