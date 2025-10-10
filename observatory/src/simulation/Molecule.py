"""
Molecular Clock Theorem Implementation
========================================
For diatomic molecule with reduced mass μ and force constant k,
the vibrational frequency provides natural clock:
ν_vib = (1/2π)√(k/μ) ≈ 10^13 - 10^14 Hz
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MolecularProperties:
    """Physical properties of a diatomic molecule"""
    mass1: float  # Mass of atom 1 (kg)
    mass2: float  # Mass of atom 2 (kg)
    force_constant: float  # Force constant k (N/m)
    bond_length: float  # Equilibrium bond length (m)


class DiatomicMolecule:
    """
    Represents a diatomic molecule as a natural atomic clock
    with femtosecond-scale vibrational periods
    """

    # Physical constants
    N2_MASS = 14.0067 * 1.66054e-27  # N atom mass in kg
    N2_FORCE_CONSTANT = 2295.0  # N/m for N2 triple bond
    N2_BOND_LENGTH = 1.098e-10  # meters

    def __init__(self, mass1: float = None, mass2: float = None,
                 force_constant: float = None, bond_length: float = None):
        """
        Initialize molecule with physical properties
        Defaults to N2 molecule if no parameters provided
        """
        self.mass1 = mass1 or self.N2_MASS
        self.mass2 = mass2 or self.N2_MASS
        self.force_constant = force_constant or self.N2_FORCE_CONSTANT
        self.bond_length = bond_length or self.N2_BOND_LENGTH

        # Calculate derived properties
        self.reduced_mass = self._calculate_reduced_mass()
        self.vibrational_frequency = self._calculate_vibrational_frequency()
        self.vibrational_period = 1.0 / self.vibrational_frequency
        self.angular_frequency = 2 * np.pi * self.vibrational_frequency

    def _calculate_reduced_mass(self) -> float:
        """Calculate reduced mass μ = m1*m2/(m1+m2)"""
        return (self.mass1 * self.mass2) / (self.mass1 + self.mass2)

    def _calculate_vibrational_frequency(self) -> float:
        """
        Calculate vibrational frequency using harmonic oscillator model
        ν = (1/2π) * √(k/μ)
        """
        omega = np.sqrt(self.force_constant / self.reduced_mass)
        frequency = omega / (2 * np.pi)
        return frequency

    def get_clock_precision(self) -> float:
        """
        Return the fundamental clock precision (one vibrational period)
        For N2: ~14.1 femtoseconds
        """
        return self.vibrational_period

    def get_quantum_energy_levels(self, max_level: int = 10) -> np.ndarray:
        """
        Calculate quantum vibrational energy levels
        E_v = ℏω(v + 1/2) where v = 0,1,2,...
        """
        hbar = 1.054571817e-34  # J·s
        levels = np.arange(0, max_level + 1)
        energies = hbar * self.angular_frequency * (levels + 0.5)
        return energies

    def get_transition_frequency(self, v_initial: int, v_final: int) -> float:
        """
        Calculate transition frequency between vibrational levels
        For harmonic oscillator, all transitions have same frequency ν_vib
        """
        if v_final <= v_initial:
            raise ValueError("Final level must be higher than initial level")

        # For harmonic oscillator, frequency is independent of v
        return self.vibrational_frequency * (v_final - v_initial)

    def oscillate(self, time_points: np.ndarray, amplitude: float = 1.0,
                  phase: float = 0.0) -> np.ndarray:
        """
        Generate molecular oscillation waveform

        Args:
            time_points: Array of time values (seconds)
            amplitude: Oscillation amplitude
            phase: Phase offset (radians)

        Returns:
            Array of displacement values
        """
        return amplitude * np.cos(self.angular_frequency * time_points + phase)

    def get_quality_factor(self, damping_rate: float = 1e10) -> float:
        """
        Calculate quality factor Q = ω/γ where γ is damping rate
        For N2 in gas phase, Q ≈ 10^6
        """
        return self.angular_frequency / damping_rate

    def __repr__(self) -> str:
        return (f"DiatomicMolecule("
                f"ν={self.vibrational_frequency:.2e} Hz, "
                f"τ={self.vibrational_period*1e15:.2f} fs, "
                f"μ={self.reduced_mass*1e27:.2f}×10⁻²⁷ kg)")


def create_N2_ensemble(n_molecules: int = 1000,
                      temperature: float = 300.0) -> list[DiatomicMolecule]:
    """
    Create ensemble of N2 molecules with thermal distribution

    Args:
        n_molecules: Number of molecules
        temperature: Temperature in Kelvin

    Returns:
        List of DiatomicMolecule instances
    """
    kb = 1.380649e-23  # Boltzmann constant

    molecules = []
    np.random.seed(42)

    for _ in range(n_molecules):
        # Add thermal frequency variation (±1% typical)
        thermal_factor = 1.0 + 0.01 * np.random.randn()
        k_effective = DiatomicMolecule.N2_FORCE_CONSTANT * thermal_factor

        molecule = DiatomicMolecule(force_constant=k_effective)
        molecules.append(molecule)

    return molecules


if __name__ == "__main__":
    print("=" * 60)
    print("   MOLECULAR CLOCK: N2 as Natural Atomic Clock")
    print("=" * 60)

    # Create N2 molecule
    n2 = DiatomicMolecule()

    print(f"\n📊 N2 Molecule Properties:")
    print(f"   Reduced mass: {n2.reduced_mass*1e27:.4f} × 10⁻²⁷ kg")
    print(f"   Force constant: {n2.force_constant:.1f} N/m")
    print(f"   Vibrational frequency: {n2.vibrational_frequency:.3e} Hz")
    print(f"   Vibrational period: {n2.vibrational_period*1e15:.2f} fs")
    print(f"   Quality factor (Q): {n2.get_quality_factor():.2e}")

    print(f"\n⏰ Clock Precision:")
    print(f"   Fundamental: {n2.get_clock_precision()*1e15:.2f} fs")
    print(f"   With harmonics (n=150): {n2.get_clock_precision()/150*1e18:.1f} as")

    print(f"\n⚡ Quantum Energy Levels (first 5):")
    energies = n2.get_quantum_energy_levels(max_level=5)
    for v, E in enumerate(energies):
        print(f"   v={v}: E = {E*1e20:.4f} × 10⁻²⁰ J")

    # Create ensemble
    ensemble = create_N2_ensemble(n_molecules=100)
    frequencies = [mol.vibrational_frequency for mol in ensemble]

    print(f"\n🌡️  Thermal Ensemble (T=300K, n=100):")
    print(f"   Mean frequency: {np.mean(frequencies):.3e} Hz")
    print(f"   Std deviation: {np.std(frequencies):.3e} Hz")
    print(f"   Frequency spread: {np.std(frequencies)/np.mean(frequencies)*100:.2f}%")

    print(f"\n✨ N2 molecules are nature's femtosecond clocks!")
