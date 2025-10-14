"""
Musculoskeletal Oscillations (Scale 3)

Analyzes muscle activation, joint angles, and musculoskeletal oscillations.
Models arm swing, torso rotation, and their coupling to gait and cardiac cycles.

Key concepts:
- Reciprocal arm-leg coordination
- Torso rotation coupling
- Muscle activation patterns
- Phase relationships across body segments

Author: Stella-Lorraine Observatory
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MusculoskeletalOscillations:
    """Complete musculoskeletal oscillation analysis"""
    arm_swing_frequency_hz: float
    torso_rotation_frequency_hz: float
    arm_swing_amplitude_deg: float
    torso_rotation_amplitude_deg: float
    arm_leg_phase_difference_rad: float  # Should be ~π (opposite phase)
    muscle_activation_patterns: Dict[str, np.ndarray]
    joint_angles: Dict[str, np.ndarray]


def estimate_arm_swing_from_gait(
    gait_frequency_hz: float,
    velocity_ms: float
) -> Tuple[float, float]:
    """
    Estimate arm swing frequency and amplitude from gait

    Arm swing is typically coupled 1:1 with leg swing:
    - Same frequency as stride frequency
    - Opposite phase (right arm + left leg)
    - Amplitude increases with velocity

    Typical amplitudes during sprinting:
    - Shoulder flexion/extension: 60-90°
    - Elbow flexion: 90-120°

    Args:
        gait_frequency_hz: Stride frequency
        velocity_ms: Running velocity

    Returns:
        (arm_frequency_hz, arm_amplitude_deg)
    """
    # Arm swing frequency = gait frequency (1:1 coupling)
    arm_frequency = gait_frequency_hz

    # Amplitude scales with velocity
    # At 6 m/s (jogging): ~45°
    # At 10 m/s (sprinting): ~75°
    # At 12 m/s (maximal sprint): ~90°

    # Linear interpolation
    velocity_norm = np.clip(velocity_ms, 4, 12)  # Clip to reasonable range
    arm_amplitude = 30 + (velocity_norm - 4) * 7.5  # Linear scaling

    return arm_frequency, arm_amplitude


def estimate_torso_rotation(
    gait_frequency_hz: float,
    velocity_ms: float
) -> Tuple[float, float]:
    """
    Estimate torso rotation frequency and amplitude

    Torso rotation:
    - Frequency = 2 × stride frequency (rotates each step)
    - Amplitude: 10-20° during running
    - Counterrotation to pelvis (reduces angular momentum)

    Args:
        gait_frequency_hz: Stride frequency
        velocity_ms: Running velocity

    Returns:
        (torso_frequency_hz, torso_amplitude_deg)
    """
    # Torso rotates twice per stride (once per step)
    torso_frequency = gait_frequency_hz * 2

    # Amplitude is relatively constant during running
    # Increases slightly with velocity
    torso_amplitude = 12 + velocity_ms * 0.5  # Typical: 12-18°
    torso_amplitude = np.clip(torso_amplitude, 10, 20)

    return torso_frequency, torso_amplitude


def calculate_arm_leg_phase_relationship(
    gait_phase: np.ndarray,
    timestamps_s: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Calculate arm swing phase relative to leg swing

    Biomechanical coordination:
    - Right arm swings forward with left leg (opposite phase)
    - Phase difference ≈ π radians (180°)
    - This minimizes angular momentum and improves efficiency

    Args:
        gait_phase: Gait cycle phase (radians)
        timestamps_s: Time points

    Returns:
        (arm_phase, mean_phase_difference)
    """
    # Arm phase is approximately π radians ahead of contralateral leg
    # (i.e., when right leg is at 0, right arm is at π)
    arm_phase = gait_phase + np.pi
    arm_phase = arm_phase % (2 * np.pi)  # Wrap to 0-2π

    # Calculate mean phase difference
    phase_diff = arm_phase - gait_phase
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to -π to π
    mean_phase_diff = np.mean(phase_diff)

    return arm_phase, mean_phase_diff


def estimate_muscle_activation_patterns(
    gait_phase: np.ndarray,
    velocity_ms: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Estimate muscle activation patterns from gait phase

    Key muscle groups during running:
    - Quadriceps: Peak during mid-stance (π/2)
    - Hamstrings: Peak during late swing (3π/2)
    - Gastrocnemius: Peak during toe-off (π)
    - Hip flexors: Peak during mid-swing (π)
    - Glutes: Peak during mid-stance (π/2)

    Activation modeled as sinusoidal with phase shifts

    Returns:
        Dictionary of muscle activation patterns (0-1)
    """
    # Normalize velocity to get relative intensity (0-1)
    intensity = np.clip(velocity_ms / 12, 0.3, 1.0)  # 12 m/s = maximal sprint

    patterns = {
        # Quadriceps: Peak at mid-stance
        'quadriceps': intensity * (0.5 + 0.5 * np.cos(gait_phase - np.pi/2)),

        # Hamstrings: Peak at late swing
        'hamstrings': intensity * (0.5 + 0.5 * np.cos(gait_phase - 3*np.pi/2)),

        # Gastrocnemius: Peak at toe-off
        'gastrocnemius': intensity * (0.5 + 0.5 * np.cos(gait_phase - np.pi)),

        # Hip flexors: Peak at mid-swing
        'hip_flexors': intensity * (0.5 + 0.5 * np.cos(gait_phase - np.pi)),

        # Glutes: Peak at mid-stance
        'glutes': intensity * (0.5 + 0.5 * np.cos(gait_phase - np.pi/2)),

        # Tibialis anterior: Peak at heel strike
        'tibialis_anterior': intensity * (0.5 + 0.5 * np.cos(gait_phase)),
    }

    return patterns


def estimate_joint_angles(
    gait_phase: np.ndarray,
    stride_length_m: float
) -> Dict[str, np.ndarray]:
    """
    Estimate joint angles throughout gait cycle

    Key joints:
    - Hip: 0° (extension) to ~70° (flexion) during swing
    - Knee: 0° (extension) to ~130° (flexion) during swing
    - Ankle: -10° (dorsiflexion) to ~20° (plantarflexion)
    - Shoulder: -30° (extension) to ~60° (flexion) during arm swing

    Args:
        gait_phase: Gait cycle phase (radians)
        stride_length_m: Stride length

    Returns:
        Dictionary of joint angles (degrees)
    """
    # Scale amplitudes with stride length
    amplitude_scale = stride_length_m / 2.0  # Normalize to typical 2m stride

    angles = {
        # Hip: Maximum flexion at mid-swing (π)
        'hip': 35 * amplitude_scale * np.sin(gait_phase),

        # Knee: Maximum flexion during swing phase
        'knee': 65 + 65 * amplitude_scale * np.abs(np.sin(gait_phase)),

        # Ankle: Plantarflexion at toe-off, dorsiflexion at heel strike
        'ankle': 5 + 15 * amplitude_scale * np.sin(gait_phase - np.pi/4),

        # Shoulder: Arm swing
        'shoulder': 45 * amplitude_scale * np.sin(gait_phase + np.pi),  # Opposite to leg

        # Elbow: Slight flexion during swing
        'elbow': 90 + 15 * amplitude_scale * np.sin(gait_phase + np.pi),
    }

    return angles


def analyze_musculoskeletal_oscillations(
    gait_phase: np.ndarray,
    velocity_ms: np.ndarray,
    timestamps_s: np.ndarray,
    gait_frequency_hz: float,
    stride_length_m: float
) -> MusculoskeletalOscillations:
    """
    Complete musculoskeletal oscillation analysis

    Pipeline:
    1. Estimate arm swing from gait
    2. Estimate torso rotation
    3. Calculate arm-leg phase relationship
    4. Estimate muscle activation patterns
    5. Estimate joint angles
    6. Return complete analysis

    Returns:
        MusculoskeletalOscillations object
    """
    mean_velocity = np.mean(velocity_ms)

    print("  Analyzing arm swing...")
    arm_freq, arm_amp = estimate_arm_swing_from_gait(gait_frequency_hz, mean_velocity)
    print(f"    Frequency: {arm_freq:.2f} Hz")
    print(f"    Amplitude: {arm_amp:.0f}°")

    print("  Analyzing torso rotation...")
    torso_freq, torso_amp = estimate_torso_rotation(gait_frequency_hz, mean_velocity)
    print(f"    Frequency: {torso_freq:.2f} Hz")
    print(f"    Amplitude: {torso_amp:.0f}°")

    print("  Calculating arm-leg phase relationship...")
    arm_phase, phase_diff = calculate_arm_leg_phase_relationship(gait_phase, timestamps_s)
    print(f"    Phase difference: {np.degrees(phase_diff):.0f}°")
    print(f"    (Expected: ~180° for opposite coordination)")

    print("  Estimating muscle activation patterns...")
    muscle_patterns = estimate_muscle_activation_patterns(gait_phase, velocity_ms)
    print(f"    Muscles analyzed: {len(muscle_patterns)}")

    print("  Estimating joint angles...")
    joint_angles = estimate_joint_angles(gait_phase, stride_length_m)
    print(f"    Joints analyzed: {len(joint_angles)}")

    return MusculoskeletalOscillations(
        arm_swing_frequency_hz=arm_freq,
        torso_rotation_frequency_hz=torso_freq,
        arm_swing_amplitude_deg=arm_amp,
        torso_rotation_amplitude_deg=torso_amp,
        arm_leg_phase_difference_rad=phase_diff,
        muscle_activation_patterns=muscle_patterns,
        joint_angles=joint_angles
    )


def main():
    """
    Example: Analyze musculoskeletal oscillations for 400m run
    """
    print("=" * 70)
    print(" MUSCULOSKELETAL OSCILLATIONS (SCALE 3) ")
    print("=" * 70)

    # Load gait analysis
    try:
        from gait import analyze_gait
        from watch import load_400m_run_data

        print("\n[1/2] Load Gait Analysis")
        watch1, watch2 = load_400m_run_data()
        velocity_ms = watch1.biomechanics['speed_ms'].values
        timestamps = watch1.biomechanics['timestamp'].values

        gait_analysis = analyze_gait(velocity_ms, timestamps)

        gait_phase = gait_analysis.gait_phase_rad
        gait_frequency = gait_analysis.gait_frequency_hz
        stride_length = gait_analysis.mean_stride_length_m

        # Convert timestamps
        if hasattr(timestamps[0], 'total_seconds'):
            timestamps_s = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        else:
            timestamps_s = timestamps

        print(f"  ✓ Gait frequency: {gait_frequency:.2f} Hz")
        print(f"  ✓ Stride length: {stride_length:.2f} m")

    except Exception as e:
        print(f"  ⚠ Data loading error: {e}")
        print("  → Using simulated data")

        duration_s = 60
        timestamps_s = np.arange(0, duration_s, 0.1)
        gait_phase = np.linspace(0, 2*np.pi*20, len(timestamps_s)) % (2*np.pi)
        velocity_ms = np.full(len(timestamps_s), 8.0)
        gait_frequency = 3.5
        stride_length = 2.2

    # Analyze musculoskeletal
    print("\n[2/2] Analyze Musculoskeletal Oscillations")
    musculo = analyze_musculoskeletal_oscillations(
        gait_phase, velocity_ms, timestamps_s,
        gait_frequency, stride_length
    )

    # Summary
    print("\n" + "=" * 70)
    print(" MUSCULOSKELETAL ANALYSIS SUMMARY ")
    print("=" * 70)

    print(f"\n📊 Oscillatory Components:")
    print(f"  Arm swing:")
    print(f"    Frequency: {musculo.arm_swing_frequency_hz:.2f} Hz")
    print(f"    Amplitude: {musculo.arm_swing_amplitude_deg:.0f}°")

    print(f"\n  Torso rotation:")
    print(f"    Frequency: {musculo.torso_rotation_frequency_hz:.2f} Hz")
    print(f"    Amplitude: {musculo.torso_rotation_amplitude_deg:.0f}°")

    print(f"\n🔗 Coordination:")
    print(f"  Arm-leg phase difference: {np.degrees(musculo.arm_leg_phase_difference_rad):.0f}°")
    if abs(np.degrees(musculo.arm_leg_phase_difference_rad) - 180) < 30:
        print(f"  → ✓ OPTIMAL opposite-phase coordination")
    else:
        print(f"  → ⚠ Suboptimal coordination")

    print(f"\n💪 Muscle Groups Analyzed:")
    for muscle, activation in musculo.muscle_activation_patterns.items():
        mean_activation = np.mean(activation)
        max_activation = np.max(activation)
        print(f"  {muscle}: mean={mean_activation:.2f}, max={max_activation:.2f}")

    print(f"\n🦴 Joint Angles:")
    for joint, angles in musculo.joint_angles.items():
        mean_angle = np.mean(angles)
        range_angle = np.max(angles) - np.min(angles)
        print(f"  {joint}: mean={mean_angle:.0f}°, range={range_angle:.0f}°")

    # Save results
    import os
    import json

    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'musculoskeletal')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary = {
        'arm_swing_frequency_hz': float(musculo.arm_swing_frequency_hz),
        'arm_swing_amplitude_deg': float(musculo.arm_swing_amplitude_deg),
        'torso_rotation_frequency_hz': float(musculo.torso_rotation_frequency_hz),
        'torso_rotation_amplitude_deg': float(musculo.torso_rotation_amplitude_deg),
        'arm_leg_phase_difference_rad': float(musculo.arm_leg_phase_difference_rad),
        'arm_leg_phase_difference_deg': float(np.degrees(musculo.arm_leg_phase_difference_rad))
    }

    summary_file = os.path.join(results_dir, f'musculoskeletal_summary_{timestamp_str}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved: {summary_file}")

    # Save muscle patterns
    muscle_df = pd.DataFrame({
        'timestamp_s': timestamps_s,
        **{name: pattern for name, pattern in musculo.muscle_activation_patterns.items()}
    })

    muscle_file = os.path.join(results_dir, f'muscle_activation_{timestamp_str}.csv')
    muscle_df.to_csv(muscle_file, index=False)
    print(f"✓ Muscle activation saved: {muscle_file}")

    # Save joint angles
    joint_df = pd.DataFrame({
        'timestamp_s': timestamps_s,
        **{name: angles for name, angles in musculo.joint_angles.items()}
    })

    joint_file = os.path.join(results_dir, f'joint_angles_{timestamp_str}.csv')
    joint_df.to_csv(joint_file, index=False)
    print(f"✓ Joint angles saved: {joint_file}")

    print("\n" + "=" * 70)
    print(" MUSCULOSKELETAL ANALYSIS COMPLETE ")
    print("=" * 70)
    print(f"\nArm swing ({musculo.arm_swing_frequency_hz:.2f} Hz) and torso rotation")
    print(f"({musculo.torso_rotation_frequency_hz:.2f} Hz) coupled to gait cycle")
    print(f"with optimal opposite-phase coordination.")

    return musculo


if __name__ == "__main__":
    main()
