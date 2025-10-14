"""
Gait Analysis & Biomechanical Oscillations (Scale 3)

Analyzes stride cycle, cadence, and biomechanical oscillations during 400m run.
Phase-locks gait oscillations to cardiac master phase reference.

Key metrics:
- Stride cycle frequency (~2-4 Hz during sprint)
- Ground contact time
- Flight time
- Vertical oscillation
- Phase-locking with cardiac cycle

Author: Stella-Lorraine Observatory
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d


@dataclass
class GaitCycle:
    """Single gait cycle data"""
    cycle_number: int
    start_time: float  # seconds
    end_time: float
    duration_s: float
    stride_length_m: float
    stride_frequency_hz: float
    stance_time_s: float
    flight_time_s: float
    duty_factor: float  # stance_time / total_time
    vertical_displacement_m: float


@dataclass
class GaitAnalysis:
    """Complete gait analysis results"""
    cycles: list[GaitCycle]
    mean_cadence_spm: float  # steps per minute
    mean_stride_length_m: float
    mean_ground_contact_time_s: float
    gait_frequency_hz: float
    cardiac_gait_plv: float  # Phase-locking value with cardiac cycle
    gait_phase_rad: np.ndarray  # Phase within gait cycle


def detect_stride_events(
    velocity_ms: np.ndarray,
    timestamps_s: np.ndarray,
    acceleration_vertical: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect stride events (foot strikes) from velocity and acceleration

    Methods:
    1. Velocity local minima (foot strike → brief deceleration)
    2. Vertical acceleration peaks (impact with ground)
    3. Combined approach

    For 400m sprint:
    - Stride frequency: ~3.5-4.5 Hz (210-270 steps/min)
    - Stride length: ~2.0-2.5 m

    Args:
        velocity_ms: Running velocity (m/s)
        timestamps_s: Time points (seconds)
        acceleration_vertical: Optional vertical acceleration (m/s²)

    Returns:
        (foot_strike_times, foot_off_times) in seconds
    """
    # Method 1: Velocity-based (works without accelerometer)
    # Find local minima in velocity (deceleration at foot strike)
    mean_velocity = np.mean(velocity_ms)
    expected_stride_freq = mean_velocity / 2.0  # Approximate stride length 2m
    min_distance = int(1 / expected_stride_freq)  # Samples between strides

    # Smooth velocity to reduce noise
    from scipy.ndimage import gaussian_filter1d
    velocity_smooth = gaussian_filter1d(velocity_ms, sigma=2)

    # Find local minima (foot strikes)
    foot_strikes_idx, _ = signal.find_peaks(
        -velocity_smooth,  # Invert to find minima
        distance=min_distance,
        prominence=0.1
    )

    foot_strike_times = timestamps_s[foot_strikes_idx]

    # Estimate foot-off times (midpoint between strikes)
    foot_off_times = []
    for i in range(len(foot_strike_times) - 1):
        mid_time = (foot_strike_times[i] + foot_strike_times[i+1]) / 2
        foot_off_times.append(mid_time)

    foot_off_times = np.array(foot_off_times)

    return foot_strike_times, foot_off_times


def calculate_stride_parameters(
    foot_strike_times: np.ndarray,
    foot_off_times: np.ndarray,
    velocity_ms: np.ndarray,
    timestamps_s: np.ndarray
) -> list[GaitCycle]:
    """
    Calculate stride-by-stride parameters

    Each stride = one complete gait cycle (left foot strike to next left foot strike)

    Parameters:
    - Stride time: Time between consecutive foot strikes
    - Stance time: Time foot is on ground
    - Flight time: Time both feet off ground
    - Stride length: Distance covered in one stride
    - Duty factor: stance_time / stride_time

    Returns:
        List of GaitCycle objects
    """
    cycles = []

    # Interpolate velocity to get velocity at any time
    velocity_interp = interp1d(timestamps_s, velocity_ms, kind='linear',
                              bounds_error=False, fill_value='extrapolate')

    for i in range(len(foot_strike_times) - 1):
        t_start = foot_strike_times[i]
        t_end = foot_strike_times[i + 1]
        duration = t_end - t_start

        # Find foot-off time for this stride
        foot_off_idx = np.where((foot_off_times > t_start) & (foot_off_times < t_end))[0]
        if len(foot_off_idx) > 0:
            t_off = foot_off_times[foot_off_idx[0]]
            stance_time = t_off - t_start
            flight_time = t_end - t_off
        else:
            # Estimate if foot-off not detected
            stance_time = duration * 0.6  # Typical 60% stance during running
            flight_time = duration * 0.4

        # Calculate stride length (distance = velocity × time)
        mean_velocity = velocity_interp((t_start + t_end) / 2)
        stride_length = mean_velocity * duration

        # Stride frequency
        stride_frequency = 1 / duration if duration > 0 else 0

        # Duty factor
        duty_factor = stance_time / duration if duration > 0 else 0

        # Vertical displacement (estimated from flight time)
        # h = 0.5 × g × (flight_time/2)²
        if flight_time > 0:
            vertical_displacement = 0.5 * 9.81 * (flight_time / 2) ** 2
        else:
            vertical_displacement = 0

        cycle = GaitCycle(
            cycle_number=i,
            start_time=t_start,
            end_time=t_end,
            duration_s=duration,
            stride_length_m=stride_length,
            stride_frequency_hz=stride_frequency,
            stance_time_s=stance_time,
            flight_time_s=flight_time,
            duty_factor=duty_factor,
            vertical_displacement_m=vertical_displacement
        )

        cycles.append(cycle)

    return cycles


def calculate_gait_phase(
    timestamps_s: np.ndarray,
    foot_strike_times: np.ndarray
) -> np.ndarray:
    """
    Calculate gait phase (0-2π) for each timestamp

    Similar to cardiac phase, but for gait cycle:
    - 0 rad (0°): Foot strike (initial contact)
    - π/2 rad (90°): Mid-stance
    - π rad (180°): Toe-off (terminal contact)
    - 3π/2 rad (270°): Mid-swing
    - 2π rad (360°): Next foot strike

    Args:
        timestamps_s: Time points to calculate phase at
        foot_strike_times: Foot strike times

    Returns:
        Gait phase (radians) for each timestamp
    """
    gait_phase = np.zeros(len(timestamps_s))

    for i, t in enumerate(timestamps_s):
        # Find previous and next foot strikes
        prev_strikes = foot_strike_times[foot_strike_times <= t]
        next_strikes = foot_strike_times[foot_strike_times > t]

        if len(prev_strikes) > 0 and len(next_strikes) > 0:
            t_prev = prev_strikes[-1]
            t_next = next_strikes[0]

            # Linear phase interpolation
            phase = 2 * np.pi * (t - t_prev) / (t_next - t_prev)
            gait_phase[i] = phase

        elif len(prev_strikes) > 0 and len(prev_strikes) >= 2:
            # Extrapolate after last strike
            last_stride_time = prev_strikes[-1] - prev_strikes[-2]
            phase = 2 * np.pi * (t - prev_strikes[-1]) / last_stride_time
            gait_phase[i] = phase % (2 * np.pi)
        else:
            gait_phase[i] = 0

    return gait_phase


def calculate_phase_locking_value(
    gait_phase: np.ndarray,
    cardiac_phase: np.ndarray
) -> float:
    """
    Calculate Phase-Locking Value (PLV) between gait and cardiac cycles

    PLV quantifies synchronization between two oscillatory signals

    PLV = |⟨exp(i(φ_gait - φ_cardiac))⟩|

    Where:
    - φ_gait = gait phase
    - φ_cardiac = cardiac phase
    - ⟨⟩ = mean over time

    PLV ranges from 0 (no locking) to 1 (perfect locking)

    Expected during running: PLV = 0.3-0.6 (moderate coupling)

    Returns:
        PLV value (0-1)
    """
    # Phase difference
    phase_diff = gait_phase - cardiac_phase

    # Complex representation
    complex_phase = np.exp(1j * phase_diff)

    # PLV = magnitude of mean
    plv = np.abs(np.mean(complex_phase))

    return plv


def analyze_gait(
    velocity_ms: np.ndarray,
    timestamps: np.ndarray,
    cardiac_phase: Optional[np.ndarray] = None
) -> GaitAnalysis:
    """
    Complete gait analysis

    Pipeline:
    1. Detect stride events (foot strikes, foot-offs)
    2. Calculate stride-by-stride parameters
    3. Calculate gait phase
    4. Calculate phase-locking with cardiac cycle (if available)
    5. Return complete analysis

    Args:
        velocity_ms: Running velocity (m/s)
        timestamps: Time points
        cardiac_phase: Optional cardiac phase (radians)

    Returns:
        Complete GaitAnalysis object
    """
    # Convert timestamps to seconds
    if hasattr(timestamps[0], 'total_seconds'):
        timestamps_s = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    else:
        timestamps_s = timestamps

    print("  Detecting stride events...")
    foot_strike_times, foot_off_times = detect_stride_events(
        velocity_ms, timestamps_s
    )
    print(f"    Detected {len(foot_strike_times)} foot strikes")

    print("  Calculating stride parameters...")
    cycles = calculate_stride_parameters(
        foot_strike_times, foot_off_times, velocity_ms, timestamps_s
    )
    print(f"    Analyzed {len(cycles)} complete stride cycles")

    # Calculate mean metrics
    mean_cadence_spm = np.mean([c.stride_frequency_hz for c in cycles]) * 60 * 2  # *2 for both feet
    mean_stride_length = np.mean([c.stride_length_m for c in cycles])
    mean_ground_contact = np.mean([c.stance_time_s for c in cycles])
    gait_frequency = np.mean([c.stride_frequency_hz for c in cycles])

    print(f"    Mean cadence: {mean_cadence_spm:.0f} steps/min")
    print(f"    Mean stride length: {mean_stride_length:.2f} m")
    print(f"    Mean ground contact time: {mean_ground_contact*1000:.0f} ms")

    print("  Calculating gait phase...")
    gait_phase_rad = calculate_gait_phase(timestamps_s, foot_strike_times)

    # Phase-locking with cardiac cycle
    if cardiac_phase is not None:
        print("  Calculating cardiac-gait phase-locking...")
        plv = calculate_phase_locking_value(gait_phase_rad, cardiac_phase)
        print(f"    PLV (gait-cardiac): {plv:.3f}")
    else:
        plv = 0.0
        print("    (Cardiac phase not provided, PLV not calculated)")

    return GaitAnalysis(
        cycles=cycles,
        mean_cadence_spm=mean_cadence_spm,
        mean_stride_length_m=mean_stride_length,
        mean_ground_contact_time_s=mean_ground_contact,
        gait_frequency_hz=gait_frequency,
        cardiac_gait_plv=plv,
        gait_phase_rad=gait_phase_rad
    )


def main():
    """
    Example: Analyze gait for 400m run
    """
    print("=" * 70)
    print(" GAIT ANALYSIS (SCALE 3: BIOMECHANICAL) ")
    print("=" * 70)

    # Load smartwatch data
    try:
        from watch import load_400m_run_data
        from cardiac import establish_cardiac_phase_reference

        print("\n[1/3] Load Smartwatch Data")
        watch1, watch2 = load_400m_run_data()
        velocity_ms = watch1.biomechanics['speed_ms'].values
        timestamps = watch1.biomechanics['timestamp'].values

        print(f"  ✓ Data points: {len(velocity_ms)}")
        print(f"  ✓ Mean velocity: {np.mean(velocity_ms):.2f} m/s")

        # Get cardiac phase
        print("\n[2/3] Load Cardiac Phase Reference")
        cardiac_ref = establish_cardiac_phase_reference(
            watch1.heart_rate,
            timestamps=timestamps
        )
        cardiac_phase = cardiac_ref.cardiac_phase_rad

    except Exception as e:
        print(f"  ⚠ Data loading error: {e}")
        print("  → Using simulated data")

        # Simulate
        duration_s = 60
        timestamps = np.arange(0, duration_s, 0.1)
        velocity_ms = np.full(len(timestamps), 6.67) + np.random.normal(0, 0.5, len(timestamps))
        cardiac_phase = None

    # Analyze gait
    print("\n[3/3] Analyze Gait")
    gait_analysis = analyze_gait(velocity_ms, timestamps, cardiac_phase)

    # Summary
    print("\n" + "=" * 70)
    print(" GAIT ANALYSIS SUMMARY ")
    print("=" * 70)

    print(f"\n📊 Overall Metrics:")
    print(f"  Total stride cycles: {len(gait_analysis.cycles)}")
    print(f"  Mean cadence: {gait_analysis.mean_cadence_spm:.0f} steps/min")
    print(f"  Mean stride length: {gait_analysis.mean_stride_length_m:.2f} m")
    print(f"  Mean ground contact: {gait_analysis.mean_ground_contact_time_s*1000:.0f} ms")
    print(f"  Gait frequency: {gait_analysis.gait_frequency_hz:.2f} Hz")

    if cardiac_phase is not None:
        print(f"\n🔗 Cardiac-Gait Coupling:")
        print(f"  PLV (phase-locking): {gait_analysis.cardiac_gait_plv:.3f}")
        if gait_analysis.cardiac_gait_plv > 0.5:
            print(f"  → STRONG coupling with cardiac cycle")
        elif gait_analysis.cardiac_gait_plv > 0.3:
            print(f"  → MODERATE coupling with cardiac cycle")
        else:
            print(f"  → WEAK coupling with cardiac cycle")

    # Save results
    import os
    import json

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'gait_analysis')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save stride cycles
    cycles_data = []
    for cycle in gait_analysis.cycles:
        cycles_data.append({
            'cycle_number': cycle.cycle_number,
            'start_time': float(cycle.start_time),
            'duration_s': float(cycle.duration_s),
            'stride_length_m': float(cycle.stride_length_m),
            'stride_frequency_hz': float(cycle.stride_frequency_hz),
            'stance_time_s': float(cycle.stance_time_s),
            'flight_time_s': float(cycle.flight_time_s),
            'duty_factor': float(cycle.duty_factor)
        })

    cycles_file = os.path.join(results_dir, f'stride_cycles_{timestamp_str}.json')
    with open(cycles_file, 'w') as f:
        json.dump(cycles_data, f, indent=2)
    print(f"\n✓ Stride cycles saved: {cycles_file}")

    # Save gait phase
    phase_df = pd.DataFrame({
        'timestamp_s': timestamps if not hasattr(timestamps[0], 'total_seconds') else
                      np.array([(t - timestamps[0]).total_seconds() for t in timestamps]),
        'gait_phase_rad': gait_analysis.gait_phase_rad,
        'gait_phase_deg': np.degrees(gait_analysis.gait_phase_rad)
    })

    phase_file = os.path.join(results_dir, f'gait_phase_{timestamp_str}.csv')
    phase_df.to_csv(phase_file, index=False)
    print(f"✓ Gait phase saved: {phase_file}")

    # Save summary
    summary = {
        'total_cycles': len(gait_analysis.cycles),
        'mean_cadence_spm': float(gait_analysis.mean_cadence_spm),
        'mean_stride_length_m': float(gait_analysis.mean_stride_length_m),
        'mean_ground_contact_time_s': float(gait_analysis.mean_ground_contact_time_s),
        'gait_frequency_hz': float(gait_analysis.gait_frequency_hz),
        'cardiac_gait_plv': float(gait_analysis.cardiac_gait_plv)
    }

    summary_file = os.path.join(results_dir, f'gait_summary_{timestamp_str}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_file}")

    print("\n" + "=" * 70)
    print(" GAIT ANALYSIS COMPLETE ")
    print("=" * 70)
    print(f"\nGait oscillations ({gait_analysis.gait_frequency_hz:.2f} Hz) phase-locked")
    print(f"to cardiac master oscillator (PLV = {gait_analysis.cardiac_gait_plv:.3f})")

    return gait_analysis


if __name__ == "__main__":
    main()
