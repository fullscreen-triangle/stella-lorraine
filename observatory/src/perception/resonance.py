"""
Neural Oscillations as Gas Molecular Dynamics (Scale 1)

Models neural activity as a thermodynamic gas molecular system where:
- Neurons = gas molecules
- Neural oscillations = molecular motion
- Information processing = variance minimization
- Consciousness = BMD frame selection rate

Based on the Gas Molecular Information Model and Neural Gas Dynamics framework.

Author: Stella-Lorraine Observatory
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Neural oscillation frequency bands (Hz)
NEURAL_BANDS = {
    'delta': (0.5, 4),      # Deep sleep
    'theta': (4, 8),        # Drowsiness, meditation
    'alpha': (8, 13),       # Relaxed, eyes closed
    'beta': (13, 30),       # Alert, active thinking
    'gamma': (30, 100),     # Cognitive processing, consciousness
    'high_gamma': (100, 200)  # Advanced processing
}


@dataclass
class NeuralGasState:
    """Neural gas molecular state"""
    n_molecules: int  # Number of neural "molecules"
    temperature: float  # System "temperature" (activity level)
    pressure: float  # System "pressure" (cognitive load)
    variance: float  # Entropy/disorder
    mean_frequency_hz: float  # Mean oscillation frequency
    band_powers: Dict[str, float]  # Power in each frequency band


def simulate_neural_gas_from_cardiac(
    cardiac_phase: np.ndarray,
    heart_rate_bpm: np.ndarray,
    cognitive_load: float = 0.7
) -> NeuralGasState:
    """
    Simulate neural gas molecular dynamics from cardiac input

    Cardiac cycle acts as perturbation to neural gas system:
    - Each heartbeat introduces variance (information/perturbation)
    - Neural gas must minimize variance before next beat
    - Rate of variance minimization = rate of perception

    Args:
        cardiac_phase: Cardiac cycle phase (radians)
        heart_rate_bpm: Heart rate (bpm)
        cognitive_load: Cognitive demand (0-1), higher = more processing

    Returns:
        Neural gas state
    """
    # Number of "molecules" (neural assemblies) scales with cognitive load
    # Typical: 10^4 - 10^6 neural assemblies active during task
    n_molecules = int(1e5 * cognitive_load)

    # Temperature (activity level) scales with HR and cognitive load
    # At rest: HR ~60 bpm, T_neural ~0.5
    # During exercise: HR ~150 bpm, T_neural ~0.8
    mean_hr = np.mean(heart_rate_bpm)
    temperature = 0.3 + (mean_hr / 200) * 0.5 + cognitive_load * 0.2
    temperature = np.clip(temperature, 0.3, 1.0)

    # Pressure (cognitive load) directly from input
    pressure = cognitive_load

    # Variance increases with cardiac perturbations
    # Calculate variance from cardiac phase variability
    phase_diff = np.diff(cardiac_phase)
    phase_diff = phase_diff[phase_diff > 0]  # Remove phase wraps
    variance = np.std(phase_diff) if len(phase_diff) > 0 else 0.1

    # Mean neural frequency from cardiac frequency
    cardiac_freq_hz = mean_hr / 60
    # Neural oscillations are typically 5-50× faster than cardiac
    mean_frequency_hz = cardiac_freq_hz * 20  # ~20 Hz (beta range during activity)

    # Band powers (normalized)
    # During cognitive task: beta and gamma dominate
    band_powers = {
        'delta': 0.05 * (1 - cognitive_load),  # Decreases with load
        'theta': 0.10 * (1 - cognitive_load),
        'alpha': 0.15,  # Moderate baseline
        'beta': 0.40 * cognitive_load,  # Increases with load
        'gamma': 0.25 * cognitive_load,  # Increases with load
        'high_gamma': 0.05 * temperature  # Increases with activity
    }

    # Normalize
    total_power = sum(band_powers.values())
    band_powers = {k: v/total_power for k, v in band_powers.items()}

    return NeuralGasState(
        n_molecules=n_molecules,
        temperature=temperature,
        pressure=pressure,
        variance=variance,
        mean_frequency_hz=mean_frequency_hz,
        band_powers=band_powers
    )


def calculate_variance_minimization_rate(
    cardiac_period_s: float,
    neural_variance: float,
    target_variance: float = 0.05
) -> float:
    """
    Calculate rate of variance minimization (rate of perception)

    Key insight from gas molecular model:
    - Heartbeat introduces variance
    - Neural gas must minimize variance before next beat
    - Time to minimize = rate of perception

    Rate = (initial_variance - target_variance) / cardiac_period

    Args:
        cardiac_period_s: Time between heartbeats
        neural_variance: Current variance
        target_variance: Target equilibrium variance

    Returns:
        Variance minimization rate (variance units / second)
    """
    if cardiac_period_s <= 0:
        return 0.0

    variance_reduction = max(0, neural_variance - target_variance)
    minimization_rate = variance_reduction / cardiac_period_s

    return minimization_rate


def calculate_consciousness_quality(
    neural_gas: NeuralGasState,
    cardiac_heart_rate_bpm: float,
    oxygen_coupling_factor: float = 1.0
) -> float:
    """
    Calculate consciousness quality score

    From framework:
    Q_consciousness = f(neural_activity, cardiac_coupling, O₂_coupling)

    Components:
    - Neural activity: Temperature, gamma/beta power
    - Cardiac coupling: How well neural gas responds to cardiac rhythm
    - O₂ coupling: Enhancement from atmospheric oxygen

    Returns:
        Consciousness quality (0-1)
    """
    # Component 1: Neural activity level
    activity_score = neural_gas.temperature

    # Component 2: High-frequency processing (gamma/beta)
    processing_score = (neural_gas.band_powers.get('gamma', 0) +
                       neural_gas.band_powers.get('beta', 0))

    # Component 3: Variance control (low variance = high quality)
    variance_score = 1 - min(neural_gas.variance, 1.0)

    # Component 4: Cardiac synchronization (optimal HR = better consciousness)
    # Optimal range: 60-100 bpm
    hr_optimal = 80
    hr_score = 1 - min(abs(cardiac_heart_rate_bpm - hr_optimal) / hr_optimal, 1.0)

    # Combine with weights
    consciousness_quality = (
        0.25 * activity_score +
        0.35 * processing_score +
        0.25 * variance_score +
        0.15 * hr_score
    ) * oxygen_coupling_factor

    return np.clip(consciousness_quality, 0, 1)


def analyze_neural_resonance(
    cardiac_phase: np.ndarray,
    heart_rate_bpm: np.ndarray,
    cognitive_load: float = 0.7,
    oxygen_coupling: float = 1.0
) -> Dict:
    """
    Complete neural resonance analysis

    Pipeline:
    1. Simulate neural gas from cardiac input
    2. Calculate variance minimization rate
    3. Calculate consciousness quality
    4. Return complete analysis

    Args:
        cardiac_phase: Cardiac phase time series
        heart_rate_bpm: Heart rate time series
        cognitive_load: Cognitive demand (0-1)
        oxygen_coupling: O₂ coupling factor (default 1.0 = normal)

    Returns:
        Dictionary with complete analysis
    """
    print("  Simulating neural gas molecular dynamics...")
    neural_gas = simulate_neural_gas_from_cardiac(
        cardiac_phase, heart_rate_bpm, cognitive_load
    )

    print(f"    Neural molecules (assemblies): {neural_gas.n_molecules:,}")
    print(f"    System temperature: {neural_gas.temperature:.3f}")
    print(f"    System pressure (load): {neural_gas.pressure:.3f}")
    print(f"    Variance: {neural_gas.variance:.3f}")
    print(f"    Mean frequency: {neural_gas.mean_frequency_hz:.1f} Hz")

    print("  Calculating variance minimization rate...")
    mean_hr = np.mean(heart_rate_bpm)
    cardiac_period_s = 60 / mean_hr

    minimization_rate = calculate_variance_minimization_rate(
        cardiac_period_s,
        neural_gas.variance
    )

    print(f"    Cardiac period: {cardiac_period_s:.3f} s")
    print(f"    Minimization rate: {minimization_rate:.3f} variance/s")
    print(f"    → This is the RATE OF PERCEPTION!")

    print("  Calculating consciousness quality...")
    consciousness_q = calculate_consciousness_quality(
        neural_gas, mean_hr, oxygen_coupling
    )

    print(f"    Consciousness quality: {consciousness_q:.3f}")
    if consciousness_q > 0.7:
        print(f"    → HIGH quality consciousness")
    elif consciousness_q > 0.5:
        print(f"    → MODERATE quality consciousness")
    else:
        print(f"    → LOW quality consciousness")

    # Calculate perception bandwidth
    # From theory: BMD frame selection ~50 bits/second (consciousness bandwidth)
    # With O₂ enhancement: Can access 10³⁰× more information
    base_bandwidth_bits_per_s = 50  # Psychological limit
    effective_bandwidth = base_bandwidth_bits_per_s * oxygen_coupling

    print(f"\n  Perception bandwidth:")
    print(f"    Base: {base_bandwidth_bits_per_s} bits/s")
    print(f"    With O₂ coupling: {effective_bandwidth:.0f} bits/s")

    return {
        'neural_gas_state': neural_gas,
        'variance_minimization_rate': minimization_rate,
        'consciousness_quality': consciousness_q,
        'perception_bandwidth_bits_per_s': effective_bandwidth,
        'cardiac_period_s': cardiac_period_s,
        'cognitive_load': cognitive_load,
        'oxygen_coupling_factor': oxygen_coupling
    }


def main():
    """
    Example: Analyze neural resonance for 400m run
    """
    print("=" * 70)
    print(" NEURAL RESONANCE (SCALE 1: CELLULAR/NEURAL) ")
    print("=" * 70)

    # Load cardiac phase reference
    try:
        from cardiac import establish_cardiac_phase_reference
        from watch import load_400m_run_data

        print("\n[1/2] Load Cardiac Phase Reference")
        watch1, watch2 = load_400m_run_data()

        cardiac_ref = establish_cardiac_phase_reference(
            watch1.heart_rate,
            timestamps=watch1.gps_track['timestamp'].values
        )

        cardiac_phase = cardiac_ref.cardiac_phase_rad
        heart_rate_bpm = cardiac_ref.heart_rate_bpm

        print(f"  ✓ Cardiac data: {len(cardiac_phase)} time points")
        print(f"  ✓ Mean HR: {np.mean(heart_rate_bpm):.1f} bpm")

    except Exception as e:
        print(f"  ⚠ Data loading error: {e}")
        print("  → Using simulated data")

        # Simulate
        duration_s = 60
        timestamps_s = np.arange(0, duration_s, 0.1)
        cardiac_phase = np.linspace(0, 2*np.pi*70, len(timestamps_s)) % (2*np.pi)
        heart_rate_bpm = np.full(len(timestamps_s), 150.0)

    # Analyze neural resonance
    print("\n[2/2] Analyze Neural Resonance")

    # During 400m run: high cognitive load (pacing, pain management, effort control)
    cognitive_load = 0.8

    # With normal oxygen coupling (can vary with altitude, etc.)
    oxygen_coupling = 1.0

    analysis = analyze_neural_resonance(
        cardiac_phase, heart_rate_bpm,
        cognitive_load, oxygen_coupling
    )

    # Summary
    print("\n" + "=" * 70)
    print(" NEURAL RESONANCE SUMMARY ")
    print("=" * 70)

    neural_gas = analysis['neural_gas_state']

    print(f"\n🧠 Neural Gas State:")
    print(f"  Molecules (assemblies): {neural_gas.n_molecules:,}")
    print(f"  Temperature (activity): {neural_gas.temperature:.3f}")
    print(f"  Pressure (load): {neural_gas.pressure:.3f}")
    print(f"  Variance: {neural_gas.variance:.3f}")
    print(f"  Mean frequency: {neural_gas.mean_frequency_hz:.1f} Hz")

    print(f"\n📊 Frequency Band Powers:")
    for band, power in neural_gas.band_powers.items():
        freq_range = NEURAL_BANDS.get(band, (0, 0))
        print(f"  {band:12s} ({freq_range[0]:3.0f}-{freq_range[1]:3.0f} Hz): {power:.3f}")

    print(f"\n⚡ Perception Dynamics:")
    print(f"  Variance minimization rate: {analysis['variance_minimization_rate']:.3f} /s")
    print(f"  Cardiac period: {analysis['cardiac_period_s']:.3f} s")
    print(f"  → Neural gas must minimize variance in {analysis['cardiac_period_s']*1000:.0f} ms!")

    print(f"\n✨ Consciousness:")
    print(f"  Quality score: {analysis['consciousness_quality']:.3f}")
    print(f"  Perception bandwidth: {analysis['perception_bandwidth_bits_per_s']:.0f} bits/s")
    print(f"  Cognitive load: {analysis['cognitive_load']:.1f}")
    print(f"  O₂ coupling: {analysis['oxygen_coupling_factor']:.1f}x")

    # Save results
    import os
    import json

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'neural_resonance')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary = {
        'neural_gas': {
            'n_molecules': neural_gas.n_molecules,
            'temperature': float(neural_gas.temperature),
            'pressure': float(neural_gas.pressure),
            'variance': float(neural_gas.variance),
            'mean_frequency_hz': float(neural_gas.mean_frequency_hz),
            'band_powers': {k: float(v) for k, v in neural_gas.band_powers.items()}
        },
        'perception': {
            'variance_minimization_rate': float(analysis['variance_minimization_rate']),
            'consciousness_quality': float(analysis['consciousness_quality']),
            'perception_bandwidth_bits_per_s': float(analysis['perception_bandwidth_bits_per_s']),
            'cardiac_period_s': float(analysis['cardiac_period_s'])
        },
        'parameters': {
            'cognitive_load': float(analysis['cognitive_load']),
            'oxygen_coupling_factor': float(analysis['oxygen_coupling_factor'])
        }
    }

    summary_file = os.path.join(results_dir, f'neural_resonance_{timestamp_str}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved: {summary_file}")

    print("\n" + "=" * 70)
    print(" NEURAL RESONANCE ANALYSIS COMPLETE ")
    print("=" * 70)
    print(f"\nNeural gas ({neural_gas.mean_frequency_hz:.1f} Hz) minimizes variance")
    print(f"at rate {analysis['variance_minimization_rate']:.3f} /s, synchronized to")
    print(f"cardiac master oscillator ({1/analysis['cardiac_period_s']:.2f} Hz)")
    print(f"\nConsciousness quality: {analysis['consciousness_quality']:.3f}")

    return analysis


if __name__ == "__main__":
    main()
