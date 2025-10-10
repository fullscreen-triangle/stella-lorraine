"""
Gas Chamber Wave Propagation Dynamics
======================================
Wave propagation in molecular gas chamber with resonant coupling:
∂²ψ/∂t² = c²∇²ψ - γ(∂ψ/∂t) + Σ α_j δ(r-r_j) cos(ω_j t + φ_j)

where:
- ψ(r,t) is wave amplitude
- c = √(γRT/M) is speed of sound in gas
- γ_damp is damping coefficient
- α_j is coupling strength to molecule j
- ω_j = 2πν_vib,j is molecular vibrational frequency
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GasProperties:
    """Thermodynamic properties of gas"""
    temperature: float  # Kelvin
    pressure: float  # Pascal
    gamma_adiabatic: float  # Adiabatic index (1.4 for N2)
    molar_mass: float  # kg/mol
    damping_coefficient: float  # 1/s


class GasChamber:
    """
    Simulates wave propagation in sealed gas chamber with molecular coupling.
    Supports resonant harmonic generation through molecular vibrations.
    """

    # N2 gas properties at STP
    N2_MOLAR_MASS = 28.014e-3  # kg/mol
    N2_GAMMA = 1.4  # Adiabatic index
    R_GAS = 8.314  # J/(mol·K)

    def __init__(self,
                 size: Tuple[float, float, float] = (1e-3, 1e-3, 1e-3),
                 temperature: float = 300.0,
                 pressure: float = 101325.0,
                 n_grid_points: int = 64):
        """
        Initialize gas chamber

        Args:
            size: Chamber dimensions (x, y, z) in meters
            temperature: Gas temperature in Kelvin
            pressure: Gas pressure in Pascal
            n_grid_points: Number of grid points per dimension
        """
        self.size = np.array(size)
        self.temperature = temperature
        self.pressure = pressure
        self.n_grid = n_grid_points

        # Gas properties
        self.gas_props = GasProperties(
            temperature=temperature,
            pressure=pressure,
            gamma_adiabatic=self.N2_GAMMA,
            molar_mass=self.N2_MOLAR_MASS,
            damping_coefficient=1e8  # Typical for N2
        )

        # Calculate speed of sound
        self.speed_of_sound = self._calculate_speed_of_sound()

        # Initialize grid
        self.dx = self.size[0] / n_grid_points
        self.dy = self.size[1] / n_grid_points
        self.dz = self.size[2] / n_grid_points

        # Wave field ψ(x,y,z,t)
        self.psi = np.zeros((n_grid_points, n_grid_points, n_grid_points))
        self.psi_prev = np.zeros_like(self.psi)

        # Molecular positions and properties
        self.molecular_sources = []

    def _calculate_speed_of_sound(self) -> float:
        """Calculate speed of sound: c = √(γRT/M)"""
        c = np.sqrt(
            self.gas_props.gamma_adiabatic *
            self.R_GAS *
            self.temperature /
            self.gas_props.molar_mass
        )
        return c

    def add_molecular_source(self,
                            position: np.ndarray,
                            frequency: float,
                            amplitude: float = 1.0,
                            phase: float = 0.0):
        """
        Add molecular vibrational source

        Args:
            position: (x,y,z) position in chamber (meters)
            frequency: Vibrational frequency (Hz)
            amplitude: Coupling strength α
            phase: Phase offset (radians)
        """
        source = {
            'position': position,
            'frequency': frequency,
            'omega': 2 * np.pi * frequency,
            'amplitude': amplitude,
            'phase': phase,
            'grid_idx': self._position_to_grid(position)
        }
        self.molecular_sources.append(source)

    def _position_to_grid(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert physical position to grid indices"""
        idx = (position / self.size * self.n_grid).astype(int)
        idx = np.clip(idx, 0, self.n_grid - 1)
        return tuple(idx)

    def propagate_wave(self, duration: float, dt: float = None) -> dict:
        """
        Propagate wave through chamber using FDTD method

        Args:
            duration: Simulation duration (seconds)
            dt: Time step (auto-calculated if None)

        Returns:
            Dictionary with wave evolution data
        """
        # CFL condition: dt ≤ dx/(c√3) for 3D
        if dt is None:
            dt = 0.5 * self.dx / (self.speed_of_sound * np.sqrt(3))

        n_steps = int(duration / dt)

        # Storage for time series at chamber center
        center_idx = (self.n_grid//2, self.n_grid//2, self.n_grid//2)
        time_series = np.zeros(n_steps)
        time_points = np.linspace(0, duration, n_steps)

        print(f"🌊 Wave propagation:")
        print(f"   Duration: {duration*1e12:.1f} ps")
        print(f"   Time step: {dt*1e15:.2f} fs")
        print(f"   Total steps: {n_steps:,}")
        print(f"   Speed of sound: {self.speed_of_sound:.1f} m/s")

        # FDTD coefficients
        c_squared = self.speed_of_sound**2
        dt_squared = dt**2
        damping = self.gas_props.damping_coefficient

        coeff_laplacian = c_squared * dt_squared / (self.dx**2)
        coeff_damping = damping * dt

        # Time stepping
        for step in range(n_steps):
            t = step * dt

            # Laplacian using finite differences
            laplacian = (
                np.roll(self.psi, 1, axis=0) + np.roll(self.psi, -1, axis=0) +
                np.roll(self.psi, 1, axis=1) + np.roll(self.psi, -1, axis=1) +
                np.roll(self.psi, 1, axis=2) + np.roll(self.psi, -1, axis=2) -
                6 * self.psi
            )

            # Add molecular sources
            source_term = np.zeros_like(self.psi)
            for src in self.molecular_sources:
                i, j, k = src['grid_idx']
                # cos(ωt + φ) forcing
                source_term[i, j, k] += src['amplitude'] * np.cos(
                    src['omega'] * t + src['phase']
                )

            # Update wave field (with damping)
            psi_new = (
                2 * self.psi - self.psi_prev +
                coeff_laplacian * laplacian +
                dt_squared * source_term
            ) / (1 + coeff_damping)

            # Update history
            self.psi_prev = self.psi.copy()
            self.psi = psi_new

            # Record center point
            time_series[step] = self.psi[center_idx]

            if step % (n_steps // 10) == 0:
                progress = step / n_steps * 100
                max_amp = np.max(np.abs(self.psi))
                print(f"   Progress: {progress:.0f}% | Max amplitude: {max_amp:.2e}")

        return {
            'time_points': time_points,
            'center_time_series': time_series,
            'final_field': self.psi.copy(),
            'speed_of_sound': self.speed_of_sound,
            'dt': dt,
            'n_steps': n_steps
        }

    def extract_resonant_modes(self, time_series: np.ndarray,
                               time_points: np.ndarray) -> dict:
        """Extract resonant harmonic modes from time series"""
        # FFT analysis
        fft_result = np.fft.fft(time_series)
        freqs = np.fft.fftfreq(len(time_points), time_points[1] - time_points[0])

        # Find peaks (resonant modes)
        magnitude = np.abs(fft_result)
        threshold = 0.1 * np.max(magnitude)

        positive_freqs = freqs > 0
        peaks_mask = (magnitude > threshold) & positive_freqs

        resonant_freqs = freqs[peaks_mask]
        resonant_amps = magnitude[peaks_mask]

        # Sort by amplitude
        sorted_indices = np.argsort(resonant_amps)[::-1]

        return {
            'frequencies': resonant_freqs[sorted_indices],
            'amplitudes': resonant_amps[sorted_indices],
            'fft_full': fft_result,
            'frequency_axis': freqs
        }

    def calculate_chamber_resonances(self) -> np.ndarray:
        """
        Calculate natural chamber resonance frequencies
        f_mnp = (c/2) * √[(m/Lx)² + (n/Ly)² + (p/Lz)²]
        """
        resonances = []
        for m in range(1, 5):
            for n in range(1, 5):
                for p in range(1, 5):
                    f = (self.speed_of_sound / 2) * np.sqrt(
                        (m / self.size[0])**2 +
                        (n / self.size[1])**2 +
                        (p / self.size[2])**2
                    )
                    resonances.append(f)

        return np.sort(resonances)[:10]  # Return first 10 modes


def demonstrate_gas_chamber_wave_propagation():
    """Demonstrate wave propagation in N2 gas chamber"""

    print("=" * 70)
    print("   GAS CHAMBER WAVE PROPAGATION WITH MOLECULAR COUPLING")
    print("=" * 70)

    # Create 1mm cube chamber
    chamber = GasChamber(
        size=(1e-3, 1e-3, 1e-3),
        temperature=300.0,
        pressure=101325.0,
        n_grid_points=32  # Reduced for speed
    )

    print(f"\n📊 Chamber Properties:")
    print(f"   Size: {chamber.size[0]*1e3:.1f} mm cube")
    print(f"   Temperature: {chamber.temperature:.1f} K")
    print(f"   Pressure: {chamber.pressure/1e3:.1f} kPa")
    print(f"   Speed of sound: {chamber.speed_of_sound:.1f} m/s")
    print(f"   Grid resolution: {chamber.n_grid}³ = {chamber.n_grid**3:,} points")

    # Add molecular sources (N2 molecules)
    n_molecules = 10
    np.random.seed(42)
    base_freq = 7.1e13  # 71 THz for N2

    print(f"\n🔬 Adding {n_molecules} molecular vibrational sources...")
    for i in range(n_molecules):
        pos = np.random.uniform(0, 1e-3, 3)
        freq = base_freq * (1 + 0.01 * np.random.randn())
        amp = 1e-12  # Small amplitude
        phase = np.random.uniform(0, 2*np.pi)

        chamber.add_molecular_source(pos, freq, amp, phase)

    # Natural resonances
    resonances = chamber.calculate_chamber_resonances()
    print(f"\n🎵 Chamber natural resonances (first 5):")
    for i, f in enumerate(resonances[:5]):
        print(f"   Mode {i+1}: {f*1e-9:.3f} GHz")

    # Propagate wave
    duration = 10e-12  # 10 picoseconds
    wave_data = chamber.propagate_wave(duration)

    # Extract harmonics
    modes = chamber.extract_resonant_modes(
        wave_data['center_time_series'],
        wave_data['time_points']
    )

    print(f"\n🎼 Extracted resonant modes (top 5):")
    for i in range(min(5, len(modes['frequencies']))):
        print(f"   f_{i+1} = {modes['frequencies'][i]*1e-12:.2f} THz, "
              f"amplitude = {modes['amplitudes'][i]:.2e}")

    print(f"\n✨ Wave successfully coupled to molecular vibrations!")

    return chamber, wave_data, modes


if __name__ == "__main__":
    chamber, wave_data, modes = demonstrate_gas_chamber_wave_propagation()
