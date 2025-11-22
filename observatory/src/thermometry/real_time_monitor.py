# thermometry/real_time_monitor.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.constants as const
from typing import List, Callable, Optional
from dataclasses import dataclass
import time
from collections import deque
import sys
from pathlib import Path

# Local imports from thermometry package
from categorical_state import CategoricalState, CategoricalStateEstimator
from temperature_extraction import ThermometryAnalyzer


@dataclass
class TemperatureSnapshot:
    """Single temperature measurement snapshot"""
    timestamp: float
    temperature: float
    uncertainty: float
    categorical_state: CategoricalState

    def relative_precision(self) -> float:
        return self.uncertainty / self.temperature


class RealTimeThermometer:
    """
    Real-time temperature monitoring during evaporative cooling

    Enables adaptive protocol optimization impossible with
    destructive methods (TOF).
    """

    def __init__(self,
                 particle_mass: float,
                 num_particles: int,
                 sampling_rate: float = 1000.0):  # Hz
        """
        Args:
            particle_mass: Mass of particle [kg]
            num_particles: Number of particles
            sampling_rate: Measurement rate [Hz]
        """
        self.m = particle_mass
        self.N = num_particles
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate

        # Initialize analyzers
        self.thermometer = ThermometryAnalyzer(particle_mass)
        self.estimator = CategoricalStateEstimator(particle_mass, num_particles)

        # Data storage
        self.history: deque = deque(maxlen=10000)  # Keep last 10k measurements

        # Monitoring state
        self.is_monitoring = False
        self.start_time = 0.0

    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_monitoring = True
        self.start_time = time.time()
        self.history.clear()

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False

    def measure_temperature(self,
                           momenta: np.ndarray,
                           timestamp: Optional[float] = None) -> TemperatureSnapshot:
        """
        Single temperature measurement

        Args:
            momenta: Momentum distribution [kg·m/s]
            timestamp: Measurement time [s] (None = use current time)

        Returns:
            TemperatureSnapshot
        """
        if timestamp is None:
            timestamp = time.time() - self.start_time

        # Construct categorical state
        cat_state = self.estimator.from_momentum_distribution(momenta, timestamp)

        # Extract temperature
        T, delta_T = self.thermometer.extract_temperature(cat_state)

        # Create snapshot
        snapshot = TemperatureSnapshot(
            timestamp=timestamp,
            temperature=T,
            uncertainty=delta_T,
            categorical_state=cat_state
        )

        # Store in history
        if self.is_monitoring:
            self.history.append(snapshot)

        return snapshot

    def get_temperature_trajectory(self) -> tuple:
        """
        Get temperature vs time trajectory

        Returns:
            (times [s], temperatures [K], uncertainties [K])
        """
        if len(self.history) == 0:
            return np.array([]), np.array([]), np.array([])

        times = np.array([s.timestamp for s in self.history])
        temps = np.array([s.temperature for s in self.history])
        uncerts = np.array([s.uncertainty for s in self.history])

        return times, temps, uncerts

    def get_cooling_rate(self, window_size: int = 100) -> float:
        """
        Calculate instantaneous cooling rate dT/dt

        Args:
            window_size: Number of points for derivative estimation

        Returns:
            Cooling rate [K/s]
        """
        if len(self.history) < window_size:
            return 0.0

        recent = list(self.history)[-window_size:]
        times = np.array([s.timestamp for s in recent])
        temps = np.array([s.temperature for s in recent])

        # Linear fit
        coeffs = np.polyfit(times, temps, 1)
        cooling_rate = coeffs[0]  # dT/dt

        return cooling_rate

    def predict_target_time(self, target_temperature: float) -> float:
        """
        Predict time to reach target temperature

        Args:
            target_temperature: Target T [K]

        Returns:
            Estimated time to target [s]
        """
        if len(self.history) < 100:
            return np.inf

        current_T = self.history[-1].temperature
        cooling_rate = self.get_cooling_rate()

        if cooling_rate >= 0:  # Not cooling
            return np.inf

        delta_T = target_temperature - current_T
        time_to_target = delta_T / cooling_rate

        return max(0, time_to_target)

    def detect_phase_transition(self,
                               threshold_derivative: float = -1e-6) -> bool:
        """
        Detect BEC phase transition from cooling rate change

        Args:
            threshold_derivative: d²T/dt² threshold [K/s²]

        Returns:
            True if phase transition detected
        """
        if len(self.history) < 200:
            return False

        # Calculate second derivative
        recent = list(self.history)[-200:]
        times = np.array([s.timestamp for s in recent])
        temps = np.array([s.temperature for s in recent])

        # Smooth data
        from scipy.signal import savgol_filter
        temps_smooth = savgol_filter(temps, 51, 3)

        # First derivative
        dt = times[1] - times[0]
        dT_dt = np.gradient(temps_smooth, dt)

        # Second derivative
        d2T_dt2 = np.gradient(dT_dt, dt)

        # Check for sudden change in cooling rate
        return np.min(d2T_dt2) < threshold_derivative

    def adaptive_cooling_control(self,
                                 current_T: float,
                                 target_T: float,
                                 cooling_power: float) -> float:
        """
        Adaptive control for evaporative cooling

        Args:
            current_T: Current temperature [K]
            target_T: Target temperature [K]
            cooling_power: Current cooling power [W]

        Returns:
            Adjusted cooling power [W]
        """
        # Get cooling rate
        cooling_rate = self.get_cooling_rate()

        # Calculate desired cooling rate
        time_to_target = self.predict_target_time(target_T)
        desired_rate = (target_T - current_T) / max(time_to_target, 1.0)

        # Proportional control
        error = desired_rate - cooling_rate
        Kp = 0.1  # Proportional gain

        adjustment = Kp * error
        new_power = cooling_power * (1 + adjustment)

        return new_power


class EvaporativeCoolingSimulator:
    """
    Simulate evaporative cooling with real-time monitoring
    """

    def __init__(self,
                 particle_mass: float,
                 num_particles: int,
                 initial_temperature: float,
                 target_temperature: float):
        """
        Args:
            particle_mass: Mass [kg]
            num_particles: Number of particles
            initial_temperature: Starting T [K]
            target_temperature: Target T [K]
        """
        self.m = particle_mass
        self.N = num_particles
        self.T_initial = initial_temperature
        self.T_target = target_temperature

        # Initialize monitor
        self.monitor = RealTimeThermometer(particle_mass, num_particles)

    def simulate_cooling(self,
                        duration: float,
                        cooling_rate: float = -1e-7,  # K/s
                        noise_level: float = 0.01) -> List[TemperatureSnapshot]:
        """
        Simulate evaporative cooling trajectory

        Args:
            duration: Simulation duration [s]
            cooling_rate: Base cooling rate [K/s]
            noise_level: Relative noise amplitude

        Returns:
            List of TemperatureSnapshot
        """
        self.monitor.start_monitoring()

        dt = self.monitor.dt
        num_steps = int(duration / dt)

        T_current = self.T_initial
        snapshots = []

        for step in range(num_steps):
            t = step * dt

            # Temperature evolution with noise
            T_current += cooling_rate * dt
            T_current += noise_level * T_current * np.random.randn() * np.sqrt(dt)
            T_current = max(T_current, self.T_target)  # Don't go below target

            # Generate momentum distribution
            sigma_v = np.sqrt(const.k * T_current / self.m)
            velocities = np.random.normal(0, sigma_v, int(self.N))
            momenta = self.m * velocities

            # Measure temperature
            snapshot = self.monitor.measure_temperature(momenta, t)
            snapshots.append(snapshot)

            # Check for phase transition
            if step > 200 and self.monitor.detect_phase_transition():
                print(f"Phase transition detected at t = {t:.3f} s, T = {T_current*1e9:.1f} nK")

        self.monitor.stop_monitoring()

        return snapshots


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)

    # Rb-87 BEC parameters
    m_Rb87 = 1.443e-25  # kg
    N_atoms = 1e5
    T_initial = 1e-6  # 1 μK
    T_target = 50e-9  # 50 nK

    print("=" * 60)
    print("REAL-TIME EVAPORATIVE COOLING SIMULATION")
    print("=" * 60)

    # Create simulator
    simulator = EvaporativeCoolingSimulator(
        m_Rb87, N_atoms, T_initial, T_target
    )

    # Run simulation
    print(f"\nCooling from {T_initial*1e6:.1f} μK to {T_target*1e9:.1f} nK...")
    snapshots = simulator.simulate_cooling(
        duration=10.0,  # 10 seconds
        cooling_rate=-1e-7,  # -100 nK/s
        noise_level=0.01
    )

    # Extract data
    times = np.array([s.timestamp for s in snapshots])
    temps = np.array([s.temperature for s in snapshots])
    uncerts = np.array([s.uncertainty for s in snapshots])

    # Calculate statistics
    final_T = temps[-1]
    final_uncert = uncerts[-1]
    relative_precision = final_uncert / final_T

    print(f"\nFinal temperature: {final_T*1e9:.3f} ± {final_uncert*1e12:.2f} pK")
    print(f"Relative precision: {relative_precision:.2e}")
    print(f"Total measurements: {len(snapshots)}")
    print(f"Average cooling rate: {(temps[-1] - temps[0]) / times[-1]:.2e} K/s")

    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Temperature trajectory
    axes[0].errorbar(times, temps * 1e9, yerr=uncerts * 1e12,
                     fmt='b-', alpha=0.5, elinewidth=2, label='Measured')
    axes[0].axhline(T_target * 1e9, color='r', linestyle='--', label='Target')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Temperature [nK]')
    axes[0].set_title('Real-Time Temperature Monitoring')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative precision
    rel_precision = uncerts / temps
    axes[1].semilogy(times, rel_precision, 'g-', alpha=0.7)
    axes[1].axhline(1e-5, color='r', linestyle='--',
                    label='TOF typical (ΔT/T ~ 10⁻²)')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Relative Precision ΔT/T')
    axes[1].set_title('Measurement Precision vs Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to validation results
    fig_path = output_dir / 'evaporative_cooling_monitor.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {fig_path}")
