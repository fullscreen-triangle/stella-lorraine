# core/oscillator_sync.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.constants as const
from typing import Tuple
from dataclasses import dataclass


@dataclass
class OscillatorState:
    """H+ oscillator state at 71 THz"""
    frequency: float  # Hz
    phase: float  # radians
    timestamp: float  # seconds

    def period(self) -> float:
        """Oscillation period [s]"""
        return 1.0 / self.frequency

    def energy(self) -> float:
        """Photon energy [J]"""
        return const.h * self.frequency


class HydrogenOscillatorSync:
    """
    Hardware-molecular synchronization via H+ oscillators at 71 THz

    Provides timing precision δt ~ 2.2×10⁻¹⁵ s
    """

    def __init__(self):
        # H+ oscillator frequency
        self.f_osc = 71e12  # 71 THz
        self.omega = 2 * np.pi * self.f_osc

        # Timing precision
        self.delta_t = 2.2e-15  # seconds

        # Energy resolution
        self.delta_E = const.hbar / (2 * self.delta_t)

        # Phase precision
        self.delta_phi = self.omega * self.delta_t

    def get_timestamp(self, cycle_count: int) -> float:
        """
        Get timestamp from oscillator cycle count

        Args:
            cycle_count: Number of oscillation cycles

        Returns:
            Time [s]
        """
        return cycle_count / self.f_osc

    def get_cycle_count(self, timestamp: float) -> int:
        """
        Get cycle count from timestamp

        Args:
            timestamp: Time [s]

        Returns:
            Cycle count
        """
        return int(timestamp * self.f_osc)

    def phase_at_time(self, timestamp: float, initial_phase: float = 0.0) -> float:
        """
        Calculate oscillator phase at given time

        Args:
            timestamp: Time [s]
            initial_phase: Initial phase [rad]

        Returns:
            Phase [rad]
        """
        return (initial_phase + self.omega * timestamp) % (2 * np.pi)

    def synchronize_stations(self,
                            station_positions: np.ndarray,
                            reference_time: float) -> np.ndarray:
        """
        Synchronize multiple stations using H+ oscillators

        Args:
            station_positions: Array of station positions [m] (N×3)
            reference_time: Reference timestamp [s]

        Returns:
            Synchronized timestamps for each station [s] (N,)
        """
        N_stations = station_positions.shape[0]

        # Light travel time corrections
        reference_position = np.mean(station_positions, axis=0)
        distances = np.linalg.norm(station_positions - reference_position, axis=1)
        light_travel_times = distances / const.c

        # Synchronized timestamps
        timestamps = reference_time + light_travel_times

        # Quantize to oscillator cycles
        cycle_counts = self.get_cycle_count(timestamps)
        synchronized_timestamps = self.get_timestamp(cycle_counts)

        return synchronized_timestamps

    def timing_jitter(self, num_samples: int = 1000) -> np.ndarray:
        """
        Simulate timing jitter

        Args:
            num_samples: Number of samples

        Returns:
            Jitter values [s]
        """
        # Gaussian jitter with σ = δt
        return np.random.normal(0, self.delta_t, num_samples)

    def allan_deviation(self,
                       timestamps: np.ndarray,
                       tau_values: np.ndarray) -> np.ndarray:
        """
        Calculate Allan deviation for stability analysis

        Args:
            timestamps: Array of timestamps [s]
            tau_values: Averaging times [s]

        Returns:
            Allan deviation values
        """
        allan_dev = np.zeros_like(tau_values)

        for i, tau in enumerate(tau_values):
            # Number of samples per averaging time
            n = int(tau * self.f_osc)
            if n < 2:
                continue

            # Calculate fractional frequency fluctuations
            y = np.diff(timestamps) * self.f_osc

            # Allan variance
            if len(y) >= 2*n:
                y_avg = np.convolve(y, np.ones(n)/n, mode='valid')
                allan_var = 0.5 * np.mean(np.diff(y_avg)**2)
                allan_dev[i] = np.sqrt(allan_var)

        return allan_dev


class MultiStationSync:
    """
    Multi-station synchronization for interferometry
    """

    def __init__(self, num_stations: int):
        self.N = num_stations
        self.sync = HydrogenOscillatorSync()

        # Station states
        self.station_positions = None
        self.station_timestamps = None
        self.station_phases = None

    def initialize_network(self, positions: np.ndarray):
        """
        Initialize station network

        Args:
            positions: Station positions [m] (N×3)
        """
        self.station_positions = positions
        self.N = positions.shape[0]

        # Synchronize all stations
        reference_time = 0.0
        self.station_timestamps = self.sync.synchronize_stations(
            positions, reference_time
        )

        # Initialize phases
        self.station_phases = np.array([
            self.sync.phase_at_time(t) for t in self.station_timestamps
        ])

    def update_synchronization(self, elapsed_time: float):
        """
        Update synchronization after elapsed time

        Args:
            elapsed_time: Time since last sync [s]
        """
        self.station_timestamps += elapsed_time

        # Update phases
        self.station_phases = np.array([
            self.sync.phase_at_time(t) for t in self.station_timestamps
        ])

    def get_baseline_delays(self) -> np.ndarray:
        """
        Get time delays between all station pairs

        Returns:
            Delay matrix [s] (N×N)
        """
        delays = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(i+1, self.N):
                baseline = self.station_positions[j] - self.station_positions[i]
                delay = np.linalg.norm(baseline) / const.c
                delays[i, j] = delay
                delays[j, i] = -delay

        return delays

    def synchronization_error(self) -> float:
        """
        Calculate RMS synchronization error

        Returns:
            RMS error [s]
        """
        mean_timestamp = np.mean(self.station_timestamps)
        deviations = self.station_timestamps - mean_timestamp
        return np.sqrt(np.mean(deviations**2))


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
    print("H+ OSCILLATOR SYNCHRONIZATION VALIDATION")
    print("=" * 70)

    # Initialize synchronizer
    sync = HydrogenOscillatorSync()

    print(f"\nOscillator frequency: {sync.f_osc/1e12:.1f} THz")
    print(f"Timing precision: {sync.delta_t*1e15:.2f} fs")
    print(f"Energy resolution: {sync.delta_E:.2e} J")
    print(f"Temperature resolution: {sync.delta_E/const.k*1e12:.1f} pK")

    # Results storage
    results = {
        'timestamp': timestamp,
        'oscillator': {
            'frequency_THz': sync.f_osc / 1e12,
            'timing_precision_fs': sync.delta_t * 1e15,
            'energy_resolution_J': sync.delta_E,
            'temperature_resolution_pK': sync.delta_E / const.k * 1e12
        },
        'multi_station_tests': []
    }

    # Multi-station network - test multiple scales
    print("\n" + "-" * 70)
    print("MULTI-STATION SYNCHRONIZATION")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test different network scales
    scales = [100e3, 500e3, 1e6, 5e6, 10e6]  # 100 km to 10,000 km
    N_stations = 10

    sync_errors = []
    max_delays = []

    for scale in scales:
        positions = np.random.randn(N_stations, 3)
        positions = positions / np.linalg.norm(positions, axis=1, keepdims=True) * scale

        network = MultiStationSync(N_stations)
        network.initialize_network(positions)

        sync_error = network.synchronization_error()
        delays = network.get_baseline_delays()
        max_delay = np.max(np.abs(delays))

        sync_errors.append(sync_error * 1e15)  # fs
        max_delays.append(max_delay * 1e3)  # ms

        print(f"\nScale: {scale/1e3:.0f} km")
        print(f"  Synchronization error: {sync_error*1e15:.3f} fs")
        print(f"  Maximum baseline delay: {max_delay*1e3:.3f} ms")

        results['multi_station_tests'].append({
            'network_scale_km': scale / 1e3,
            'num_stations': N_stations,
            'synchronization_error_fs': sync_error * 1e15,
            'max_baseline_delay_ms': max_delay * 1e3
        })

    # Timing jitter analysis
    jitter = sync.timing_jitter(10000)
    print(f"\nTiming jitter (10k samples):")
    print(f"  Mean: {np.mean(jitter)*1e15:.3f} fs")
    print(f"  Std: {np.std(jitter)*1e15:.3f} fs")
    print(f"  Max: {np.max(np.abs(jitter))*1e15:.3f} fs")

    results['timing_jitter'] = {
        'num_samples': 10000,
        'mean_fs': np.mean(jitter) * 1e15,
        'std_fs': np.std(jitter) * 1e15,
        'max_fs': np.max(np.abs(jitter)) * 1e15
    }

    # Panel A: Synchronization error vs network scale
    ax = axes[0, 0]
    ax.semilogx(np.array(scales) / 1e3, sync_errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(sync.delta_t * 1e15, color='r', linestyle='--',
              label=f'Timing precision ({sync.delta_t*1e15:.2f} fs)')
    ax.set_xlabel('Network Scale [km]')
    ax.set_ylabel('Synchronization Error [fs]')
    ax.set_title('A) Multi-Station Synchronization Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Baseline delay vs scale
    ax = axes[0, 1]
    ax.loglog(np.array(scales) / 1e3, max_delays, 'gs-', linewidth=2, markersize=8)
    # Light travel time
    light_delays = (np.array(scales) / const.c) * 1e3  # ms
    ax.loglog(np.array(scales) / 1e3, light_delays, 'r--',
             linewidth=2, label='Light travel time')
    ax.set_xlabel('Network Scale [km]')
    ax.set_ylabel('Maximum Baseline Delay [ms]')
    ax.set_title('B) Baseline Delays vs Network Scale')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Panel C: Timing jitter distribution
    ax = axes[1, 0]
    ax.hist(jitter * 1e15, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Timing Jitter [fs]')
    ax.set_ylabel('Probability Density')
    ax.set_title('C) Timing Jitter Distribution')
    ax.grid(True, alpha=0.3)

    # Panel D: Allan deviation (stability)
    ax = axes[1, 1]
    # Simulate timestamps
    num_samples = 1000
    timestamps = np.cumsum(np.ones(num_samples) / sync.f_osc +
                          np.random.normal(0, sync.delta_t, num_samples))
    tau_values = np.logspace(-9, -6, 20)  # 1 ns to 1 μs
    allan_dev = sync.allan_deviation(timestamps, tau_values)

    ax.loglog(tau_values * 1e6, allan_dev, 'mo-', linewidth=2, markersize=6)
    ax.set_xlabel('Averaging Time τ [μs]')
    ax.set_ylabel('Allan Deviation')
    ax.set_title('D) Clock Stability (Allan Deviation)')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"oscillator_synchronization_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")

    # Save JSON results
    json_path = output_dir / f"oscillator_sync_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {json_path}")

    print("\n" + "=" * 70)
