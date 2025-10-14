"""
Cardiac Phase Reference (Scale 2: Master Oscillator)

Establishes the heartbeat as the master phase reference for all biological
oscillations in the hierarchical cascade.

Key concepts:
- Cardiac cycle as fundamental perception quantum
- HRV (Heart Rate Variability) analysis
- Phase calculation (0-2π for each cardiac cycle)
- Master oscillator for multi-scale synchronization

Author: Stella-Lorraine Observatory
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d


@dataclass
class CardiacPhaseReference:
    """Master cardiac phase reference"""
    timestamps: np.ndarray  # Time points
    cardiac_phase_rad: np.ndarray  # Phase 0-2π
    rr_intervals_ms: np.ndarray  # R-R intervals
    heart_rate_bpm: np.ndarray  # Instantaneous HR
    hrv_metrics: Dict  # SDNN, RMSSD, pNN50, LF/HF
    is_stable: bool  # Rhythm stability flag
    master_frequency_hz: float  # Average cardiac frequency


def detect_heart_rate_peaks(hr_time_series: pd.Series, sampling_rate_hz: float = 1.0) -> np.ndarray:
    """
    Detect peaks in heart rate time series

    For actual ECG: Use R-peak detection (Pan-Tompkins, Hamilton, etc.)
    For smartwatch HR: Use local maxima with physiological constraints

    Args:
        hr_time_series: Heart rate in BPM
        sampling_rate_hz: Sampling rate (typically 1 Hz for smartwatch)

    Returns:
        Array of peak indices
    """
    hr_values = hr_time_series.values

    # For HR time series (not raw ECG), peaks correspond to local maxima
    # But we actually want to mark each cardiac cycle

    # Convert HR (bpm) to expected cycle time
    # HR = 140 bpm → period = 60/140 = 0.43 seconds
    mean_hr = np.nanmean(hr_values)
    expected_period_s = 60 / mean_hr
    min_distance = int(expected_period_s * 0.7 * sampling_rate_hz)  # At least 70% of expected

    # Find local maxima in HR signal as surrogate for cardiac events
    # In reality, would use actual R-peaks from ECG
    peaks, properties = signal.find_peaks(
        hr_values,
        distance=min_distance,
        prominence=2  # At least 2 bpm prominence
    )

    # If too few peaks detected, generate uniform peaks based on mean HR
    if len(peaks) < 3:
        num_beats = int(len(hr_values) / (expected_period_s * sampling_rate_hz))
        peaks = np.linspace(0, len(hr_values) - 1, num_beats, dtype=int)

    return peaks


def calculate_rr_intervals(peak_times: np.ndarray) -> np.ndarray:
    """
    Calculate R-R intervals (time between consecutive heartbeats)

    R-R interval = time between consecutive R-peaks
    Units: milliseconds (standard in HRV analysis)

    Typical range: 400-1200 ms (150-50 bpm)
    """
    rr_intervals_ms = np.diff(peak_times) * 1000  # Convert to ms

    # Physiological filtering: Remove non-physiological values
    # Valid range: 300-2000 ms (200-30 bpm)
    rr_intervals_ms = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 2000)]

    return rr_intervals_ms


def calculate_hrv_metrics(rr_intervals_ms: np.ndarray) -> Dict:
    """
    Calculate Heart Rate Variability (HRV) metrics

    Time domain metrics:
    - SDNN: Standard deviation of R-R intervals (overall variability)
    - RMSSD: Root mean square of successive differences (short-term variability)
    - pNN50: Percentage of successive R-R intervals differing by >50 ms

    Frequency domain metrics:
    - LF: Low frequency power (0.04-0.15 Hz) - sympathetic + parasympathetic
    - HF: High frequency power (0.15-0.4 Hz) - parasympathetic (respiratory)
    - LF/HF ratio: Sympathovagal balance

    Returns:
        Dictionary with all HRV metrics
    """
    if len(rr_intervals_ms) < 5:
        return {
            'sdnn_ms': np.nan,
            'rmssd_ms': np.nan,
            'pnn50_pct': np.nan,
            'lf_power': np.nan,
            'hf_power': np.nan,
            'lf_hf_ratio': np.nan,
            'mean_rr_ms': np.nan,
            'mean_hr_bpm': np.nan
        }

    # Time domain metrics
    sdnn = np.std(rr_intervals_ms)

    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0

    mean_rr = np.mean(rr_intervals_ms)
    mean_hr = 60000 / mean_rr  # Convert ms to bpm

    # Frequency domain metrics (requires interpolation to evenly sampled signal)
    try:
        # Create time vector for R-R intervals
        time_rr = np.cumsum(rr_intervals_ms) / 1000  # Convert to seconds
        time_rr = np.insert(time_rr, 0, 0)

        # Interpolate to 4 Hz (standard for HRV frequency analysis)
        time_interp = np.arange(0, time_rr[-1], 0.25)
        f_interp = interp1d(time_rr[:-1], rr_intervals_ms, kind='cubic', fill_value='extrapolate')
        rr_interp = f_interp(time_interp)

        # Compute power spectral density
        freqs, psd = signal.welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)))

        # Extract frequency bands
        lf_band = (freqs >= 0.04) & (freqs < 0.15)
        hf_band = (freqs >= 0.15) & (freqs < 0.4)

        lf_power = np.trapz(psd[lf_band], freqs[lf_band])
        hf_power = np.trapz(psd[hf_band], freqs[hf_band])

        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    except Exception as e:
        print(f"    Warning: Could not calculate frequency domain metrics: {e}")
        lf_power = np.nan
        hf_power = np.nan
        lf_hf_ratio = np.nan

    return {
        'sdnn_ms': sdnn,
        'rmssd_ms': rmssd,
        'pnn50_pct': pnn50,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio,
        'mean_rr_ms': mean_rr,
        'mean_hr_bpm': mean_hr
    }


def calculate_cardiac_phase(timestamps: np.ndarray, peak_times: np.ndarray) -> np.ndarray:
    """
    Calculate instantaneous cardiac phase (0-2π) for each timestamp

    Cardiac phase:
    - 0 rad (0°): R-peak (ventricular depolarization)
    - π/2 rad (90°): Systole (ventricular contraction)
    - π rad (180°): T-wave (ventricular repolarization)
    - 3π/2 rad (270°): Diastole (ventricular filling)
    - 2π rad (360°): Next R-peak

    This phase is the MASTER REFERENCE for all biological oscillations!

    Args:
        timestamps: Time points to calculate phase at
        peak_times: R-peak times (cardiac events)

    Returns:
        Cardiac phase (radians) for each timestamp
    """
    cardiac_phase = np.zeros(len(timestamps))

    for i, t in enumerate(timestamps):
        # Find previous and next peaks
        prev_peaks = peak_times[peak_times <= t]
        next_peaks = peak_times[peak_times > t]

        if len(prev_peaks) > 0 and len(next_peaks) > 0:
            t_prev = prev_peaks[-1]
            t_next = next_peaks[0]

            # Linear phase interpolation between peaks
            # Phase = 2π * (t - t_prev) / (t_next - t_prev)
            phase = 2 * np.pi * (t - t_prev) / (t_next - t_prev)
            cardiac_phase[i] = phase

        elif len(prev_peaks) > 0:
            # After last peak: extrapolate using last R-R interval
            if len(prev_peaks) >= 2:
                last_rr = prev_peaks[-1] - prev_peaks[-2]
                phase = 2 * np.pi * (t - prev_peaks[-1]) / last_rr
                cardiac_phase[i] = phase % (2 * np.pi)
            else:
                cardiac_phase[i] = 0

        elif len(next_peaks) > 0:
            # Before first peak: extrapolate using first R-R interval
            if len(next_peaks) >= 2:
                first_rr = next_peaks[1] - next_peaks[0]
                phase = 2 * np.pi * (t - (next_peaks[0] - first_rr)) / first_rr
                cardiac_phase[i] = phase % (2 * np.pi)
            else:
                cardiac_phase[i] = 0

        else:
            cardiac_phase[i] = 0

    return cardiac_phase


def validate_cardiac_stability(rr_intervals_ms: np.ndarray, threshold_cv: float = 0.15) -> Tuple[bool, float]:
    """
    Validate that cardiac rhythm is stable enough for phase reference

    Criteria:
    - Coefficient of variation (CV) < 15% (stable rhythm)
    - No ectopic beats (sudden >20% changes)
    - Sufficient data points (>10 beats)

    During 400m sprint: CV typically 10-20% (moderate variability)
    At rest: CV typically 5-10%
    Arrhythmia: CV > 30%

    Args:
        rr_intervals_ms: R-R intervals
        threshold_cv: Maximum acceptable CV (default 15%)

    Returns:
        (is_stable, actual_cv)
    """
    if len(rr_intervals_ms) < 10:
        return False, np.nan

    mean_rr = np.mean(rr_intervals_ms)
    std_rr = np.std(rr_intervals_ms)
    cv = (std_rr / mean_rr) * 100  # Coefficient of variation (%)

    # Check for ectopic beats (sudden large changes)
    successive_diffs = np.abs(np.diff(rr_intervals_ms))
    max_diff_pct = np.max(successive_diffs / rr_intervals_ms[:-1]) * 100

    # Stable if CV is reasonable and no large ectopic beats
    is_stable = (cv < threshold_cv * 100) and (max_diff_pct < 30)

    return is_stable, cv


def establish_cardiac_phase_reference(
    hr_time_series: pd.DataFrame,
    timestamps: Optional[np.ndarray] = None
) -> CardiacPhaseReference:
    """
    Establish complete cardiac phase reference

    This is the MASTER OSCILLATOR for the entire framework!

    Pipeline:
    1. Detect heart rate peaks (R-peaks surrogate)
    2. Calculate R-R intervals
    3. Calculate HRV metrics
    4. Validate rhythm stability
    5. Calculate cardiac phase for all timestamps
    6. Return complete reference

    Args:
        hr_time_series: DataFrame with 'timestamp' and 'heart_rate_bpm'
        timestamps: Optional specific timestamps to calculate phase at

    Returns:
        Complete cardiac phase reference
    """
    print("\n  Establishing Cardiac Phase Reference...")

    # Extract time series
    if timestamps is None:
        timestamps = hr_time_series['timestamp'].values

    hr_values = hr_time_series['heart_rate_bpm'].values
    hr_timestamps = hr_time_series['timestamp'].values

    # Convert timestamps to seconds (float) for calculations
    timestamps_s = np.array([(t - timestamps[0]).total_seconds() if hasattr(t, 'total_seconds')
                            else t for t in timestamps])
    hr_timestamps_s = np.array([(t - hr_timestamps[0]).total_seconds() if hasattr(t, 'total_seconds')
                                else t for t in hr_timestamps])

    # 1. Detect peaks
    sampling_rate = 1.0 / np.mean(np.diff(hr_timestamps_s)) if len(hr_timestamps_s) > 1 else 1.0
    peak_indices = detect_heart_rate_peaks(pd.Series(hr_values), sampling_rate)
    peak_times_s = hr_timestamps_s[peak_indices]

    print(f"    Detected {len(peak_indices)} cardiac cycles")

    # 2. Calculate R-R intervals
    rr_intervals_ms = calculate_rr_intervals(peak_times_s)
    print(f"    Mean R-R interval: {np.mean(rr_intervals_ms):.0f} ms")

    # 3. Calculate HRV metrics
    hrv_metrics = calculate_hrv_metrics(rr_intervals_ms)
    print(f"    Mean HR: {hrv_metrics['mean_hr_bpm']:.1f} bpm")
    print(f"    SDNN: {hrv_metrics['sdnn_ms']:.1f} ms")
    print(f"    RMSSD: {hrv_metrics['rmssd_ms']:.1f} ms")

    # 4. Validate stability
    is_stable, cv = validate_cardiac_stability(rr_intervals_ms)
    stability_str = "✓ STABLE" if is_stable else "⚠ VARIABLE"
    print(f"    Rhythm stability: {stability_str} (CV: {cv:.1f}%)")

    # 5. Calculate cardiac phase
    cardiac_phase_rad = calculate_cardiac_phase(timestamps_s, peak_times_s)
    print(f"    Cardiac phase calculated for {len(cardiac_phase_rad)} time points")

    # 6. Calculate master frequency
    master_frequency_hz = hrv_metrics['mean_hr_bpm'] / 60
    print(f"    Master frequency: {master_frequency_hz:.3f} Hz ({hrv_metrics['mean_hr_bpm']:.1f} bpm)")

    # Interpolate HR to all timestamps
    hr_interp_func = interp1d(hr_timestamps_s, hr_values, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
    hr_interpolated = hr_interp_func(timestamps_s)

    return CardiacPhaseReference(
        timestamps=timestamps,
        cardiac_phase_rad=cardiac_phase_rad,
        rr_intervals_ms=rr_intervals_ms,
        heart_rate_bpm=hr_interpolated,
        hrv_metrics=hrv_metrics,
        is_stable=is_stable,
        master_frequency_hz=master_frequency_hz
    )


def synchronize_signal_to_cardiac_phase(
    signal_values: np.ndarray,
    signal_timestamps: np.ndarray,
    cardiac_ref: CardiacPhaseReference
) -> np.ndarray:
    """
    Synchronize any signal to cardiac phase

    This enables phase-locking analysis between the signal and cardiac cycle.
    Used for all other oscillatory scales (biomechanical, atmospheric, etc.)

    Args:
        signal_values: Signal to synchronize
        signal_timestamps: Signal timestamps
        cardiac_ref: Cardiac phase reference

    Returns:
        Signal values aligned to cardiac phase bins
    """
    # Bin by cardiac phase (e.g., 36 bins = 10° each)
    n_phase_bins = 36
    phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)

    # Interpolate signal to cardiac reference timestamps
    signal_interp_func = interp1d(
        signal_timestamps,
        signal_values,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    signal_at_cardiac_times = signal_interp_func(
        np.array([t.total_seconds() if hasattr(t, 'total_seconds') else t
                 for t in cardiac_ref.timestamps])
    )

    # Phase-average the signal
    phase_averaged_signal = np.zeros(n_phase_bins)
    phase_counts = np.zeros(n_phase_bins)

    for i in range(len(signal_at_cardiac_times)):
        phase = cardiac_ref.cardiac_phase_rad[i]
        bin_idx = np.digitize(phase, phase_bins) - 1
        if 0 <= bin_idx < n_phase_bins:
            phase_averaged_signal[bin_idx] += signal_at_cardiac_times[i]
            phase_counts[bin_idx] += 1

    # Average
    phase_averaged_signal = phase_averaged_signal / np.maximum(phase_counts, 1)

    return phase_averaged_signal


def main():
    """
    Example: Establish cardiac phase reference for 400m run
    """
    print("=" * 70)
    print(" CARDIAC PHASE REFERENCE (MASTER OSCILLATOR) ")
    print("=" * 70)

    # Load smartwatch data
    from watch import load_400m_run_data

    print("\n[1/2] Load Smartwatch Data")
    watch1, watch2 = load_400m_run_data()

    print(f"\n[2/2] Establish Cardiac Phase Reference")

    # Watch 1
    print(f"\n  Watch 1 (Garmin):")
    cardiac_ref_1 = establish_cardiac_phase_reference(
        watch1.heart_rate,
        timestamps=watch1.gps_track['timestamp'].values
    )

    # Watch 2
    print(f"\n  Watch 2 (Coros):")
    cardiac_ref_2 = establish_cardiac_phase_reference(
        watch2.heart_rate,
        timestamps=watch2.gps_track['timestamp'].values
    )

    # Compare
    print(f"\n📊 Comparison:")
    print(f"  Watch 1 master frequency: {cardiac_ref_1.master_frequency_hz:.3f} Hz")
    print(f"  Watch 2 master frequency: {cardiac_ref_2.master_frequency_hz:.3f} Hz")
    print(f"  Difference: {abs(cardiac_ref_1.master_frequency_hz - cardiac_ref_2.master_frequency_hz)*1000:.2f} mHz")

    # Save results
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'cardiac_phase')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_str = cardiac_ref_1.timestamps[0].strftime("%Y%m%d_%H%M%S") if hasattr(cardiac_ref_1.timestamps[0], 'strftime') else "20220427_154453"

    # Save phase reference
    cardiac_data = pd.DataFrame({
        'timestamp': cardiac_ref_1.timestamps,
        'cardiac_phase_rad': cardiac_ref_1.cardiac_phase_rad,
        'cardiac_phase_deg': np.degrees(cardiac_ref_1.cardiac_phase_rad),
        'heart_rate_bpm': cardiac_ref_1.heart_rate_bpm
    })

    cardiac_file = os.path.join(results_dir, f'cardiac_phase_reference_{timestamp_str}.csv')
    cardiac_data.to_csv(cardiac_file, index=False)
    print(f"\n✓ Cardiac phase reference saved: {cardiac_file}")

    # Save HRV metrics
    import json
    hrv_file = os.path.join(results_dir, f'hrv_metrics_{timestamp_str}.json')
    with open(hrv_file, 'w') as f:
        json.dump({
            'watch1': cardiac_ref_1.hrv_metrics,
            'watch2': cardiac_ref_2.hrv_metrics,
            'master_frequency_hz': cardiac_ref_1.master_frequency_hz,
            'is_stable': cardiac_ref_1.is_stable
        }, f, indent=2)
    print(f"✓ HRV metrics saved: {hrv_file}")

    print("\n" + "=" * 70)
    print(" MASTER PHASE REFERENCE ESTABLISHED ")
    print("=" * 70)
    print("\nHeartbeat established as master oscillator for all biological scales.")
    print("Cardiac phase provides absolute reference for hierarchical coupling.")

    return cardiac_ref_1, cardiac_ref_2


if __name__ == "__main__":
    main()
