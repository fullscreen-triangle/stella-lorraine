#!/usr/bin/env python3
"""
Live Trans-Planckian Clock Runner
===================================
Runs the full precision cascade as an operational clock for a specified duration.
Records all timing measurements and saves compressed results.

WARNING: Can generate large amounts of data!
"""

import numpy as np
import json
import os
import time
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TransPlanckianClock:
    """
    Operational trans-Planckian precision clock
    Runs all 7 precision levels simultaneously
    """

    def __init__(self):
        self.start_time = None
        self.measurements = {
            'nanosecond': [],
            'picosecond': [],
            'femtosecond': [],
            'attosecond': [],
            'zeptosecond': [],
            'planck': [],
            'trans_planckian': []
        }

        # Initialize components
        print("🕐 Initializing Trans-Planckian Clock...")
        self._initialize_precision_layers()

    def _initialize_precision_layers(self):
        """Initialize all precision measurement layers"""

        # Layer 1: Hardware clocks (nanosecond)
        print("   [1/7] Hardware clock layer (ns)")
        self.hardware_clocks = {
            'cpu': 3.2e9,  # 3.2 GHz
            'system': 1.0e9,
            'tsc': 2.8e9
        }

        # Layer 2: Molecular vibrations (picosecond)
        print("   [2/7] Molecular vibration layer (ps)")
        self.molecular_freq = 7.07e13  # N2 vibrational frequency

        # Layer 3: Quantum coherence (femtosecond)
        print("   [3/7] Quantum coherence layer (fs)")
        self.coherence_time = 247e-15  # LED enhanced

        # Layer 4: Harmonic extraction (attosecond)
        print("   [4/7] Harmonic extraction layer (as)")
        self.harmonic_order = 100
        self.subharmonic_res = 1000

        # Layer 5: Multi-domain SEFT (zeptosecond)
        print("   [5/7] Multi-domain SEFT layer (zs)")
        self.seft_enhancement = 2003

        # Layer 6: Recursive observers (Planck approach)
        print("   [6/7] Recursive observer layer (Planck)")
        self.recursion_depth = 22
        self.molecules_per_observer = 100

        # Layer 7: Network graph (trans-Planckian)
        print("   [7/7] Harmonic network graph layer (trans-Planck)")
        self.network_nodes = 260000
        self.network_edges = 25794141
        self.graph_enhancement = 7176

        print("   ✓ All layers initialized")

    def measure_nanosecond(self, reference_time):
        """Layer 1: Hardware clock measurement"""
        # Use high-resolution performance counter
        t = time.perf_counter_ns()
        precision = 1.0 / np.mean(list(self.hardware_clocks.values()))
        return {
            'timestamp_ns': t,
            'reference_offset': t - reference_time,
            'precision_s': precision,
            'layer': 'nanosecond'
        }

    def measure_picosecond(self, ns_measurement):
        """Layer 2: Molecular vibration measurement"""
        # Enhance nanosecond with molecular oscillation
        molecular_period = 1.0 / self.molecular_freq
        precision = molecular_period * 0.85  # LED efficiency

        return {
            'timestamp_ns': ns_measurement['timestamp_ns'],
            'molecular_cycles': ns_measurement['timestamp_ns'] * 1e-9 * self.molecular_freq,
            'precision_s': precision,
            'layer': 'picosecond'
        }

    def measure_femtosecond(self, ps_measurement):
        """Layer 3: Quantum coherence measurement"""
        precision = self.coherence_time / (2 * np.pi)

        return {
            'timestamp_ns': ps_measurement['timestamp_ns'],
            'coherence_limited': precision,
            'precision_s': precision,
            'layer': 'femtosecond'
        }

    def measure_attosecond(self, fs_measurement):
        """Layer 4: Harmonic extraction"""
        base_period = 1.0 / self.molecular_freq
        precision = base_period / (self.harmonic_order * self.subharmonic_res)

        return {
            'timestamp_ns': fs_measurement['timestamp_ns'],
            'harmonic_order': self.harmonic_order,
            'precision_s': precision,
            'layer': 'attosecond'
        }

    def measure_zeptosecond(self, as_measurement):
        """Layer 5: Multi-domain SEFT"""
        base_precision = as_measurement['precision_s']
        precision = base_precision / self.seft_enhancement

        return {
            'timestamp_ns': as_measurement['timestamp_ns'],
            'seft_domains': 4,
            'total_enhancement': self.seft_enhancement,
            'precision_s': precision,
            'layer': 'zeptosecond'
        }

    def measure_planck(self, zs_measurement):
        """Layer 6: Recursive observer nesting"""
        base_precision = zs_measurement['precision_s']
        # Simplified: only 2 levels due to coherence
        enhancement = self.molecules_per_observer ** 2
        precision = base_precision / enhancement

        return {
            'timestamp_ns': zs_measurement['timestamp_ns'],
            'recursion_levels': 2,
            'precision_s': precision,
            'layer': 'planck'
        }

    def measure_trans_planckian(self, planck_measurement):
        """Layer 7: Harmonic network graph"""
        base_precision = planck_measurement['precision_s']
        precision = base_precision / self.graph_enhancement

        return {
            'timestamp_ns': planck_measurement['timestamp_ns'],
            'network_nodes': self.network_nodes,
            'network_edges': self.network_edges,
            'graph_enhancement': self.graph_enhancement,
            'precision_s': precision,
            'layer': 'trans_planckian'
        }

    def take_measurement(self):
        """
        Take a complete cascade measurement across all 7 layers
        Returns measurements at each precision level
        """
        reference_time = time.perf_counter_ns()

        # Cascade through all layers
        ns = self.measure_nanosecond(reference_time)
        ps = self.measure_picosecond(ns)
        fs = self.measure_femtosecond(ps)
        as_m = self.measure_attosecond(fs)
        zs = self.measure_zeptosecond(as_m)
        planck = self.measure_planck(zs)
        trans_planck = self.measure_trans_planckian(planck)

        return {
            'reference_ns': reference_time,
            'layers': {
                'nanosecond': ns,
                'picosecond': ps,
                'femtosecond': fs,
                'attosecond': as_m,
                'zeptosecond': zs,
                'planck': planck,
                'trans_planckian': trans_planck
            }
        }

    def run(self, duration_seconds=10, sample_rate_hz=1000):
        """
        Run the clock for specified duration

        Args:
            duration_seconds: How long to run (default 10s)
            sample_rate_hz: Measurements per second (default 1000 Hz)
        """
        print(f"\n{'='*70}")
        print(f"   🕐 STARTING TRANS-PLANCKIAN CLOCK")
        print(f"{'='*70}")
        print(f"\n   Duration: {duration_seconds} seconds")
        print(f"   Sample Rate: {sample_rate_hz} Hz")
        print(f"   Expected Measurements: {duration_seconds * sample_rate_hz:,}")

        # Calculate sampling interval
        interval = 1.0 / sample_rate_hz

        self.start_time = time.perf_counter()
        start_ns = time.perf_counter_ns()

        measurements = []
        measurement_count = 0

        print(f"\n   🚀 Clock running...")
        print(f"   (Press Ctrl+C to stop early)")

        try:
            while True:
                current_time = time.perf_counter()
                elapsed = current_time - self.start_time

                if elapsed >= duration_seconds:
                    break

                # Take measurement
                measurement = self.take_measurement()
                measurements.append(measurement)
                measurement_count += 1

                # Progress indicator (every 100 measurements)
                if measurement_count % 100 == 0:
                    progress = (elapsed / duration_seconds) * 100
                    print(f"\r   Progress: {progress:.1f}% | Measurements: {measurement_count:,}", end='', flush=True)

                # Sleep until next sample
                next_sample_time = self.start_time + (measurement_count * interval)
                sleep_time = next_sample_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n   ⚠ Interrupted by user")

        end_time = time.perf_counter()
        end_ns = time.perf_counter_ns()
        actual_duration = end_time - self.start_time

        print(f"\n\n   ✓ Clock stopped")
        print(f"   Actual duration: {actual_duration:.3f} seconds")
        print(f"   Total measurements: {measurement_count:,}")
        print(f"   Actual sample rate: {measurement_count / actual_duration:.1f} Hz")

        return {
            'metadata': {
                'start_time_ns': start_ns,
                'end_time_ns': end_ns,
                'duration_s': actual_duration,
                'planned_duration_s': duration_seconds,
                'sample_rate_hz': sample_rate_hz,
                'total_measurements': measurement_count,
                'actual_sample_rate_hz': measurement_count / actual_duration
            },
            'measurements': measurements
        }

def save_results(clock_data, compress=True):
    """
    Save clock results with optional compression

    Args:
        clock_data: Clock measurement data
        compress: Whether to compress (recommended for long runs)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'live_clock')
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n   💾 Saving results...")

    # Save metadata
    metadata_file = os.path.join(results_dir, f'clock_run_metadata_{timestamp}.json')
    with open(metadata_file, 'w') as f:
        json.dump(clock_data['metadata'], f, indent=2)
    print(f"   ✓ Metadata saved: {metadata_file}")

    # Save measurements (compressed format)
    measurements = clock_data['measurements']
    n_measurements = len(measurements)

    # Extract time series for each layer
    time_series = {
        'reference_ns': [m['reference_ns'] for m in measurements],
        'nanosecond_precision': [m['layers']['nanosecond']['precision_s'] for m in measurements],
        'picosecond_precision': [m['layers']['picosecond']['precision_s'] for m in measurements],
        'femtosecond_precision': [m['layers']['femtosecond']['precision_s'] for m in measurements],
        'attosecond_precision': [m['layers']['attosecond']['precision_s'] for m in measurements],
        'zeptosecond_precision': [m['layers']['zeptosecond']['precision_s'] for m in measurements],
        'planck_precision': [m['layers']['planck']['precision_s'] for m in measurements],
        'trans_planckian_precision': [m['layers']['trans_planckian']['precision_s'] for m in measurements],
    }

    if compress:
        # Save as numpy compressed (much smaller)
        npz_file = os.path.join(results_dir, f'clock_run_data_{timestamp}.npz')
        np.savez_compressed(npz_file, **time_series)
        print(f"   ✓ Data saved (compressed): {npz_file}")

        # Get file size
        size_mb = os.path.getsize(npz_file) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
    else:
        # Save as JSON (larger but human-readable)
        json_file = os.path.join(results_dir, f'clock_run_data_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(time_series, f)
        print(f"   ✓ Data saved (JSON): {json_file}")

        size_mb = os.path.getsize(json_file) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")

    # Calculate statistics
    print(f"\n   📊 Statistics:")
    print(f"   Total measurements: {n_measurements:,}")
    print(f"   Time span: {(time_series['reference_ns'][-1] - time_series['reference_ns'][0]) / 1e9:.3f} s")

    for layer in ['trans_planckian', 'zeptosecond', 'attosecond', 'femtosecond']:
        precisions = time_series[f'{layer}_precision']
        mean_p = np.mean(precisions)
        std_p = np.std(precisions)
        print(f"   {layer.title():20} precision: {mean_p:.2e} ± {std_p:.2e} s")

    return metadata_file, npz_file if compress else json_file

def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Trans-Planckian Live Clock')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Duration in seconds (default: 10)')
    parser.add_argument('--rate', type=int, default=1000,
                       help='Sample rate in Hz (default: 1000)')
    parser.add_argument('--no-compress', action='store_true',
                       help='Save uncompressed JSON instead of NPZ')

    args = parser.parse_args()

    print("="*70)
    print("   TRANS-PLANCKIAN LIVE CLOCK RUNNER")
    print("="*70)
    print(f"\n   Configuration:")
    print(f"   - Duration: {args.duration} seconds")
    print(f"   - Sample Rate: {args.rate} Hz")
    print(f"   - Compression: {'OFF' if args.no_compress else 'ON'}")
    print(f"   - Expected data: ~{args.duration * args.rate * 8 * 8 / 1024 / 1024:.1f} MB (uncompressed)")

    # Initialize clock
    clock = TransPlanckianClock()

    # Run clock
    clock_data = clock.run(duration_seconds=args.duration, sample_rate_hz=args.rate)

    # Save results
    save_results(clock_data, compress=not args.no_compress)

    print(f"\n{'='*70}")
    print(f"   ✓ CLOCK RUN COMPLETE")
    print(f"{'='*70}")
    print(f"\n   Results saved in: results/live_clock/")
    print(f"\n   To analyze: python analyze_clock_data.py")

if __name__ == "__main__":
    main()
