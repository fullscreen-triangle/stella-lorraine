"""
Quantum Molecular Vibrations
=============================
Quantum mechanical treatment of molecular vibrations with Heisenberg-limited precision.
"""

import numpy as np
from typing import Dict, Tuple


class QuantumVibrationalAnalyzer:
    """
    Analyzes quantum molecular vibrations with natural linewidths
    limited by Heisenberg uncertainty principle.
    """

    # Physical constants
    HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
    KB = 1.380649e-23  # Boltzmann constant (J/K)

    def __init__(self, frequency: float = 7.1e13, coherence_time: float = 247e-15):
        """
        Initialize with molecular frequency

        Args:
            frequency: Vibrational frequency (Hz), default = 71 THz for N2
            coherence_time: Coherence time from LED spectroscopy (s)
        """
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.coherence_time = coherence_time

    def calculate_energy_levels(self, max_level: int = 10) -> np.ndarray:
        """
        Calculate quantum vibrational energy levels
        E_v = ℏω(v + 1/2) where v = 0,1,2,...

        Args:
            max_level: Maximum vibrational quantum number

        Returns:
            Array of energy levels (Joules)
        """
        v = np.arange(0, max_level + 1)
        energies = self.HBAR * self.omega * (v + 0.5)
        return energies

    def heisenberg_linewidth(self, coherence_time: float = None) -> float:
        """
        Calculate natural linewidth from Heisenberg uncertainty
        Δν · Δt ≥ 1/(4π)

        Args:
            coherence_time: Coherence time (uses instance default if None)

        Returns:
            Frequency uncertainty (Hz)
        """
        if coherence_time is None:
            coherence_time = self.coherence_time

        delta_nu = 1.0 / (4 * np.pi * coherence_time)
        return delta_nu

    def temporal_precision(self, coherence_time: float = None) -> float:
        """
        Calculate achievable temporal precision from Heisenberg limit

        Args:
            coherence_time: Coherence time (uses instance default if None)

        Returns:
            Temporal precision (seconds)
        """
        if coherence_time is None:
            coherence_time = self.coherence_time

        delta_nu = self.heisenberg_linewidth(coherence_time)

        # Δt = 1/Δν for frequency-to-time conversion
        delta_t = 1.0 / delta_nu

        return delta_t

    def led_enhanced_coherence(self, base_coherence: float = 100e-15,
                               led_enhancement: float = 2.47) -> Dict:
        """
        Calculate LED-enhanced coherence properties

        Args:
            base_coherence: Base molecular coherence time (s)
            led_enhancement: Enhancement factor from LED phase-locking

        Returns:
            Dictionary with enhanced properties
        """
        enhanced_coherence = base_coherence * led_enhancement

        # Natural linewidth
        natural_linewidth = self.heisenberg_linewidth(base_coherence)

        # LED-enhanced linewidth
        enhanced_linewidth = self.heisenberg_linewidth(enhanced_coherence)

        # Precision improvement
        natural_precision = self.temporal_precision(base_coherence)
        enhanced_precision = self.temporal_precision(enhanced_coherence)

        return {
            'base_coherence_time': base_coherence,
            'enhanced_coherence_time': enhanced_coherence,
            'enhancement_factor': led_enhancement,
            'natural_linewidth': natural_linewidth,
            'enhanced_linewidth': enhanced_linewidth,
            'natural_precision': natural_precision,
            'enhanced_precision': enhanced_precision,
            'precision_improvement': natural_precision / enhanced_precision
        }

    def quantum_state_evolution(self, initial_state: np.ndarray,
                               time_points: np.ndarray,
                               damping: float = 0.0) -> np.ndarray:
        """
        Evolve quantum state under harmonic oscillator Hamiltonian

        Args:
            initial_state: Initial state vector (complex amplitudes)
            time_points: Time points for evolution
            damping: Damping rate (1/s)

        Returns:
            State evolution array [time, state]
        """
        n_levels = len(initial_state)
        evolution = np.zeros((len(time_points), n_levels), dtype=complex)

        for i, t in enumerate(time_points):
            for v in range(n_levels):
                # Energy of level v
                E_v = self.HBAR * self.omega * (v + 0.5)

                # Time evolution with damping
                phase = -E_v * t / self.HBAR
                amplitude = initial_state[v] * np.exp(1j * phase - damping * t / 2)

                evolution[i, v] = amplitude

        return evolution

    def calculate_transition_rate(self, v_initial: int, v_final: int,
                                  temperature: float = 300.0) -> float:
        """
        Calculate transition rate between vibrational levels

        Args:
            v_initial: Initial level
            v_final: Final level
            temperature: Temperature (K)

        Returns:
            Transition rate (1/s)
        """
        # Energy difference
        delta_E = self.HBAR * self.omega * (v_final - v_initial)

        # Boltzmann factor
        boltzmann = np.exp(-abs(delta_E) / (self.KB * temperature))

        # Transition rate (simplified, using Fermi's golden rule approximation)
        rate = (self.omega / (2 * np.pi)) * boltzmann

        return rate

    def thermal_population(self, temperature: float = 300.0,
                          max_level: int = 10) -> np.ndarray:
        """
        Calculate thermal population distribution (Boltzmann)

        Args:
            temperature: Temperature (K)
            max_level: Maximum level to calculate

        Returns:
            Population probabilities for each level
        """
        energies = self.calculate_energy_levels(max_level)

        # Boltzmann distribution
        beta = 1.0 / (self.KB * temperature)
        populations = np.exp(-beta * energies)

        # Normalize
        populations = populations / np.sum(populations)

        return populations


def demonstrate_quantum_vibrations():
    """Demonstrate quantum vibrational analysis"""

    print("=" * 70)
    print("   QUANTUM MOLECULAR VIBRATIONS")
    print("=" * 70)

    # Create analyzer for N2 with LED enhancement
    analyzer = QuantumVibrationalAnalyzer(
        frequency=7.1e13,
        coherence_time=247e-15  # LED-enhanced
    )

    print(f"\n📊 N2 Vibrational Properties:")
    print(f"   Frequency: {analyzer.frequency:.2e} Hz (71 THz)")
    print(f"   Angular frequency: {analyzer.omega:.2e} rad/s")
    print(f"   LED-enhanced coherence: {analyzer.coherence_time*1e15:.0f} fs")

    # Energy levels
    energies = analyzer.calculate_energy_levels(max_level=5)
    print(f"\n⚡ Quantum Energy Levels (first 5):")
    for v, E in enumerate(energies):
        print(f"   v={v}: E = {E*1e20:.4f} × 10⁻²⁰ J = {E/(analyzer.HBAR*analyzer.omega):.1f} ℏω")

    # Heisenberg limits
    linewidth = analyzer.heisenberg_linewidth()
    precision = analyzer.temporal_precision()

    print(f"\n🔬 Heisenberg Uncertainty Limits:")
    print(f"   Natural linewidth: Δν = {linewidth:.2e} Hz")
    print(f"   Temporal precision: Δt = {precision*1e15:.2f} fs")
    print(f"   Uncertainty product: Δν·Δt = {linewidth*precision:.4f} (≥ 0.0796)")

    # LED enhancement
    led_props = analyzer.led_enhanced_coherence(
        base_coherence=100e-15,
        led_enhancement=2.47
    )

    print(f"\n💡 LED Enhancement Analysis:")
    print(f"   Base coherence: {led_props['base_coherence_time']*1e15:.0f} fs")
    print(f"   Enhanced coherence: {led_props['enhanced_coherence_time']*1e15:.0f} fs")
    print(f"   Enhancement factor: {led_props['enhancement_factor']:.2f}×")
    print(f"\n   Natural linewidth: {led_props['natural_linewidth']:.2e} Hz")
    print(f"   Enhanced linewidth: {led_props['enhanced_linewidth']:.2e} Hz")
    print(f"\n   Natural precision: {led_props['natural_precision']*1e15:.2f} fs")
    print(f"   Enhanced precision: {led_props['enhanced_precision']*1e15:.2f} fs")
    print(f"   Precision improvement: {led_props['precision_improvement']:.2f}×")

    # Thermal population
    populations = analyzer.thermal_population(temperature=300.0, max_level=10)

    print(f"\n🌡️  Thermal Population (T=300K):")
    print(f"   Ground state (v=0): {populations[0]*100:.1f}%")
    print(f"   First excited (v=1): {populations[1]*100:.3f}%")
    print(f"   Second excited (v=2): {populations[2]*100:.5f}%")

    # Quantum state evolution
    print(f"\n🌊 Quantum State Evolution:")
    initial_state = np.zeros(5, dtype=complex)
    initial_state[0] = 1.0  # Start in ground state

    time_points = np.linspace(0, 10e-15, 100)  # 10 fs
    evolution = analyzer.quantum_state_evolution(initial_state, time_points, damping=1e12)

    final_populations = np.abs(evolution[-1,:])**2
    print(f"   After 10 fs evolution:")
    for v in range(len(final_populations)):
        if final_populations[v] > 0.001:
            print(f"   v={v}: {final_populations[v]*100:.2f}%")

    print(f"\n✨ Quantum coherence maintained at {analyzer.coherence_time*1e15:.0f} fs!")

    return analyzer, led_props


if __name__ == "__main__":
    analyzer, led_props = demonstrate_quantum_vibrations()
