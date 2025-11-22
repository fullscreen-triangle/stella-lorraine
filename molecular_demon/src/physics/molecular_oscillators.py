"""
Molecular Oscillator Physical Properties

Database of molecular species used in trans-Planckian measurements:
- N2: Nitrogen (primary, used in experiments)
- O2: Oxygen
- H+: Hydrogen ion (Lyman-alpha)
- H2O: Water
- CO2: Carbon dioxide

Each molecule provides natural oscillation frequencies for timekeeping.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Physical constants
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit to kg
K_BOLTZMANN = 1.380649e-23  # J/K
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 299792458  # m/s


@dataclass
class MolecularSpecies:
    """Physical properties of a molecular species"""
    name: str
    formula: str
    mass_amu: float  # Atomic mass units
    vibrational_frequency_hz: float  # Fundamental vibrational frequency
    rotational_constant_hz: float  # Rotational constant B
    harmonic_constant: float  # Force constant (N/m)
    max_harmonics: int  # Maximum observable harmonic order
    q_factor: float  # Quality factor
    coherence_time_s: float  # Typical coherence time

    @property
    def mass_kg(self) -> float:
        """Mass in kilograms"""
        return self.mass_amu * AMU_TO_KG

    @property
    def period_s(self) -> float:
        """Vibrational period"""
        return 1.0 / self.vibrational_frequency_hz

    @property
    def wavelength_m(self) -> float:
        """Associated electromagnetic wavelength"""
        return C_LIGHT / self.vibrational_frequency_hz


# Database of molecular species
MOLECULAR_DATABASE: Dict[str, MolecularSpecies] = {
    'N2': MolecularSpecies(
        name='Nitrogen',
        formula='N₂',
        mass_amu=28.014,
        vibrational_frequency_hz=7.07e13,  # 2359 cm⁻¹
        rotational_constant_hz=5.96e10,  # 1.99 cm⁻¹
        harmonic_constant=2294,  # N/m
        max_harmonics=150,
        q_factor=1e6,
        coherence_time_s=10e-9  # 10 ns
    ),

    'O2': MolecularSpecies(
        name='Oxygen',
        formula='O₂',
        mass_amu=31.998,
        vibrational_frequency_hz=4.74e13,  # 1580 cm⁻¹
        rotational_constant_hz=4.28e10,  # 1.44 cm⁻¹
        harmonic_constant=1177,  # N/m
        max_harmonics=100,
        q_factor=8e5,
        coherence_time_s=8e-9  # 8 ns
    ),

    'H+': MolecularSpecies(
        name='Hydrogen ion',
        formula='H⁺',
        mass_amu=1.008,
        vibrational_frequency_hz=2.47e15,  # Lyman-alpha
        rotational_constant_hz=0,  # Atomic, not molecular
        harmonic_constant=0,
        max_harmonics=200,
        q_factor=1e7,
        coherence_time_s=100e-9  # 100 ns
    ),

    'H2O': MolecularSpecies(
        name='Water',
        formula='H₂O',
        mass_amu=18.015,
        vibrational_frequency_hz=1.10e14,  # 3657 cm⁻¹ (O-H stretch)
        rotational_constant_hz=8.36e11,  # 27.88 cm⁻¹
        harmonic_constant=840,  # N/m (O-H)
        max_harmonics=120,
        q_factor=5e5,
        coherence_time_s=5e-9  # 5 ns
    ),

    'CO2': MolecularSpecies(
        name='Carbon dioxide',
        formula='CO₂',
        mass_amu=44.010,
        vibrational_frequency_hz=7.05e13,  # 2349 cm⁻¹ (asymmetric stretch)
        rotational_constant_hz=1.17e10,  # 0.39 cm⁻¹
        harmonic_constant=1480,  # N/m
        max_harmonics=80,
        q_factor=3e5,
        coherence_time_s=3e-9  # 3 ns
    ),
}


class MolecularOscillatorGenerator:
    """
    Generate ensemble of molecular oscillators with realistic properties

    Includes thermal broadening, Doppler shifts, and environmental variations
    """

    def __init__(self, species: str = 'N2', temperature_k: float = 300.0):
        """
        Initialize generator

        Args:
            species: Molecular species from database
            temperature_k: Temperature in Kelvin
        """
        if species not in MOLECULAR_DATABASE:
            raise ValueError(f"Unknown species: {species}. Available: {list(MOLECULAR_DATABASE.keys())}")

        self.species_data = MOLECULAR_DATABASE[species]
        self.temperature = temperature_k

    def generate_ensemble(self, n_molecules: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Generate ensemble of molecules with frequency variations

        Variations arise from:
        1. Thermal motion (Doppler broadening)
        2. Local field variations
        3. Collisional perturbations
        4. Quantum state distribution

        Args:
            n_molecules: Number of molecules to generate
            seed: Random seed for reproducibility

        Returns:
            List of molecule dictionaries with frequencies and properties
        """
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Generating {n_molecules} {self.species_data.name} molecules at {self.temperature} K")

        base_freq = self.species_data.vibrational_frequency_hz

        # Thermal velocity distribution (Maxwell-Boltzmann)
        v_thermal = np.sqrt(2 * K_BOLTZMANN * self.temperature / self.species_data.mass_kg)

        molecules = []
        for i in range(n_molecules):
            # Velocity components (3D)
            vx, vy, vz = np.random.normal(0, v_thermal, 3)
            v_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

            # Doppler shift: Δf/f = v/c
            doppler_shift = v_magnitude / C_LIGHT
            freq_doppler = base_freq * (1 + doppler_shift * np.random.choice([-1, 1]))

            # Local field variation (pressure, nearby molecules)
            field_variation = np.random.normal(0, base_freq * 1e-6)

            # Quantum state (thermal distribution over vibrational levels)
            # P(n) ∝ exp(-n·hν/kT)
            hnu_over_kt = (H_PLANCK * base_freq) / (K_BOLTZMANN * self.temperature)
            vibrational_level = np.random.geometric(1 - np.exp(-hnu_over_kt)) - 1
            freq_quantum = base_freq * (1 + vibrational_level * 0.001)  # Anharmonicity

            # Combined frequency
            frequency = freq_doppler + field_variation + (freq_quantum - base_freq)

            # Random phase
            phase = np.random.uniform(0, 2 * np.pi)

            # S-entropy coordinates (from categorical_state module)
            s_k = np.random.exponential(1.0)  # Knowledge accumulation
            s_t = np.random.exponential(1.0)  # Temporal evolution

            # S_e from temperature
            thermal_wavelength = H_PLANCK / np.sqrt(2 * np.pi * self.species_data.mass_kg * K_BOLTZMANN * self.temperature)
            s_e = np.log(1 / thermal_wavelength**3) if thermal_wavelength > 0 else 10.0

            molecules.append({
                'id': i,
                'species': self.species_data.formula,
                'frequency_hz': frequency,
                'phase_rad': phase,
                's_coordinates': (s_k, s_t, s_e),
                'velocity_m_s': (vx, vy, vz),
                'vibrational_level': vibrational_level,
                'temperature_k': self.temperature
            })

        logger.info(f"Generated ensemble:")
        logger.info(f"  Mean frequency: {np.mean([m['frequency_hz'] for m in molecules]):.2e} Hz")
        logger.info(f"  Std deviation: {np.std([m['frequency_hz'] for m in molecules]):.2e} Hz")
        logger.info(f"  Frequency range: {np.ptp([m['frequency_hz'] for m in molecules]):.2e} Hz")

        return molecules

    def doppler_broadening(self) -> float:
        """
        Calculate Doppler broadening width

        Returns:
            FWHM of Doppler-broadened line (Hz)
        """
        v_thermal = np.sqrt(2 * K_BOLTZMANN * self.temperature / self.species_data.mass_kg)

        # Δf_FWHM = (2·f₀/c)·√(2·ln(2)·kT/m)
        delta_f = (2 * self.species_data.vibrational_frequency_hz / C_LIGHT) * v_thermal * np.sqrt(2 * np.log(2))

        return delta_f


def get_species_properties(species: str) -> MolecularSpecies:
    """
    Get physical properties of a molecular species

    Args:
        species: Species name (N2, O2, H+, H2O, CO2)

    Returns:
        MolecularSpecies dataclass with properties
    """
    if species not in MOLECULAR_DATABASE:
        raise ValueError(f"Unknown species: {species}")

    return MOLECULAR_DATABASE[species]


def list_available_species() -> List[str]:
    """List all available molecular species"""
    return list(MOLECULAR_DATABASE.keys())


def compare_species() -> None:
    """Print comparison table of all species"""
    print("\n" + "="*80)
    print("MOLECULAR OSCILLATOR DATABASE")
    print("="*80)
    print(f"{'Species':<10} {'Formula':<8} {'Freq (THz)':<12} {'Period':<12} {'Max Harm':<10}")
    print("-"*80)

    for name, spec in MOLECULAR_DATABASE.items():
        freq_thz = spec.vibrational_frequency_hz / 1e12
        period_str = f"{spec.period_s*1e15:.1f} fs"
        print(f"{name:<10} {spec.formula:<8} {freq_thz:<12.2f} {period_str:<12} {spec.max_harmonics:<10}")

    print("="*80)
