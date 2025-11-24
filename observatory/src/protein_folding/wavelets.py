"""
FIXED ADVANCED STATISTICAL VISUALIZATION FOR PHASE-LOCKED FOLDING
Rigorous evidence: 3D phase space, polar analysis, wavelets, cosinor
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Wedge
import json
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("="*80)
    print("ADVANCED STATISTICAL ANALYSIS: PHASE-LOCKED FOLDING")
    print("="*80)

    # ============================================================
    # LOAD DATA
    # ============================================================

    with open('phase_locked_folding_results.json', 'r') as f:
        phase_results = json.load(f)

    with open('cycle_by_cycle_validation.json', 'r') as f:
        validation_results = json.load(f)

    # Extract ubiquitin data
    ubiquitin_cycles = validation_results['test_4']['cycle_history']
    cycles = np.array([c['cycle'] for c in ubiquitin_cycles])
    stability = np.array([c['final_stability'] for c in ubiquitin_cycles])
    variance = np.array([c['final_variance'] for c in ubiquitin_cycles])
    mean_stability = np.array([c['mean_stability'] for c in ubiquitin_cycles])
    min_variance = np.array([c['min_variance'] for c in ubiquitin_cycles])

    # Calculate phase (from variance - low variance = high phase coherence)
    phase_coherence = 1 - variance / np.max(variance)

    print(f"\n✓ Data loaded: {len(cycles)} cycles")
    print(f"  Stability range: {np.min(stability):.3f} - {np.max(stability):.3f}")
    print(f"  Variance range: {np.min(variance):.3f} - {np.max(variance):.3f}")
    print(f"  Phase coherence range: {np.min(phase_coherence):.3f} - {np.max(phase_coherence):.3f}")

    # ============================================================
    # FIGURE 5: 3D PHASE SPACE ANALYSIS
    # ============================================================

    print("\n" + "="*80)
    print("FIGURE 5: 3D PHASE SPACE TRAJECTORIES")
    print("="*80)

    fig5 = plt.figure(figsize=(28, 24))
    gs5 = GridSpec(4, 3, figure=fig5, hspace=0.4, wspace=0.4)

    # Panel 5A: Main 3D Phase Space (Stability, Variance, Phase)
    ax5a = fig5.add_subplot(gs5[0:2, 0:2], projection='3d')

    # Plot trajectory
    scatter = ax5a.scatter(stability, variance, phase_coherence,
                        c=cycles, s=200, cmap='viridis',
                        alpha=0.8, edgecolor='black', linewidth=1.5)

    # Connect points with line
    ax5a.plot(stability, variance, phase_coherence,
            'k-', linewidth=2, alpha=0.3)

    # Mark start and end
    ax5a.scatter([stability[0]], [variance[0]], [phase_coherence[0]],
                s=500, c='red', marker='o', edgecolor='black', linewidth=3,
                label='Start', zorder=10)
    ax5a.scatter([stability[-1]], [variance[-1]], [phase_coherence[-1]],
                s=500, c='green', marker='*', edgecolor='black', linewidth=3,
                label='Folded', zorder=10)

    # Mark critical transitions (fixed dimension check)
    stability_diff = np.diff(stability)
    critical_indices = np.where(stability_diff > 0.1)[0]
    if len(critical_indices) > 0:
        ax5a.scatter(stability[critical_indices], variance[critical_indices],
                    phase_coherence[critical_indices],
                    s=400, c='gold', marker='D', edgecolor='black', linewidth=2,
                    label='Critical', zorder=9)

    ax5a.set_xlabel('Stability', fontsize=12, fontweight='bold', labelpad=10)
    ax5a.set_ylabel('Variance', fontsize=12, fontweight='bold', labelpad=10)
    ax5a.set_zlabel('Phase Coherence', fontsize=12, fontweight='bold', labelpad=10)
    ax5a.set_title('(A) 3D Phase Space Trajectory\nStability-Variance-Coherence',
                fontsize=14, fontweight='bold', pad=20)
    ax5a.legend(fontsize=10, loc='upper left')

    cbar = plt.colorbar(scatter, ax=ax5a, pad=0.1, shrink=0.8)
    cbar.set_label('Cycle', fontsize=10, fontweight='bold')

    # Panel 5B: Projection onto Stability-Variance plane
    ax5b = fig5.add_subplot(gs5[0, 2])

    scatter_sv = ax5b.scatter(stability, variance, c=cycles, s=150,
                            cmap='viridis', alpha=0.8, edgecolor='black', linewidth=1)
    ax5b.plot(stability, variance, 'k-', linewidth=1.5, alpha=0.3)

    ax5b.scatter([stability[0]], [variance[0]], s=300, c='red', marker='o',
                edgecolor='black', linewidth=2, zorder=5)
    ax5b.scatter([stability[-1]], [variance[-1]], s=300, c='green', marker='*',
                edgecolor='black', linewidth=2, zorder=5)

    ax5b.set_xlabel('Stability', fontsize=11, fontweight='bold')
    ax5b.set_ylabel('Variance', fontsize=11, fontweight='bold')
    ax5b.set_title('(B) S-V Projection',
                fontsize=12, fontweight='bold')
    ax5b.grid(alpha=0.3, linestyle='--')

    # Panel 5C: Projection onto Stability-Phase plane
    ax5c = fig5.add_subplot(gs5[1, 2])

    scatter_sp = ax5c.scatter(stability, phase_coherence, c=cycles, s=150,
                            cmap='viridis', alpha=0.8, edgecolor='black', linewidth=1)
    ax5c.plot(stability, phase_coherence, 'k-', linewidth=1.5, alpha=0.3)

    ax5c.scatter([stability[0]], [phase_coherence[0]], s=300, c='red', marker='o',
                edgecolor='black', linewidth=2, zorder=5)
    ax5c.scatter([stability[-1]], [phase_coherence[-1]], s=300, c='green', marker='*',
                edgecolor='black', linewidth=2, zorder=5)

    ax5c.set_xlabel('Stability', fontsize=11, fontweight='bold')
    ax5c.set_ylabel('Phase Coherence', fontsize=11, fontweight='bold')
    ax5c.set_title('(C) S-P Projection',
                fontsize=12, fontweight='bold')
    ax5c.grid(alpha=0.3, linestyle='--')

    # Panel 5D: Phase Space Velocity (FIXED)
    ax5d = fig5.add_subplot(gs5[2, :2])

    # Calculate velocity in phase space
    ds = np.diff(stability)
    dv = np.diff(variance)
    dp = np.diff(phase_coherence)
    velocity = np.sqrt(ds**2 + dv**2 + dp**2)

    # Use cycles[1:] to match velocity length
    ax5d.plot(cycles[1:], velocity, 'o-', linewidth=2.5, markersize=8,
            color='#e74c3c', alpha=0.8)
    ax5d.fill_between(cycles[1:], velocity, alpha=0.3, color='#e74c3c')

    # Mark high velocity points
    high_vel_threshold = np.mean(velocity) + np.std(velocity)
    high_vel_indices = np.where(velocity > high_vel_threshold)[0]
    if len(high_vel_indices) > 0:
        ax5d.scatter(cycles[1:][high_vel_indices], velocity[high_vel_indices],
                    s=300, c='gold', marker='*', edgecolor='black', linewidth=2,
                    zorder=5, label='High velocity')
        ax5d.legend(fontsize=10)

    ax5d.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax5d.set_ylabel('Phase Space Velocity', fontsize=12, fontweight='bold')
    ax5d.set_title('(D) Phase Space Velocity\nRate of Change in S-V-P Space',
                fontsize=13, fontweight='bold')
    ax5d.grid(alpha=0.3, linestyle='--')

    # Panel 5E: Phase Space Distance from Origin
    ax5e = fig5.add_subplot(gs5[2, 2])

    # Normalize and calculate distance
    stability_norm = (stability - np.min(stability)) / (np.max(stability) - np.min(stability))
    variance_norm = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))
    phase_norm = (phase_coherence - np.min(phase_coherence)) / (np.max(phase_coherence) - np.min(phase_coherence))

    distance = np.sqrt(stability_norm**2 + variance_norm**2 + phase_norm**2)

    ax5e.plot(cycles, distance, 'o-', linewidth=2.5, markersize=8,
            color='#9b59b6', alpha=0.8)

    ax5e.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax5e.set_ylabel('Distance from Origin', fontsize=11, fontweight='bold')
    ax5e.set_title('(E) Phase Space Distance',
                fontsize=12, fontweight='bold')
    ax5e.grid(alpha=0.3, linestyle='--')

    # Panel 5F: Statistics
    ax5f = fig5.add_subplot(gs5[3, :])
    ax5f.axis('off')

    # Calculate statistics
    mean_velocity = np.mean(velocity)
    std_velocity = np.std(velocity)
    max_velocity = np.max(velocity)
    total_distance = np.sum(velocity)

    # Calculate trajectory curvature (FIXED)
    if len(ds) > 1 and len(dv) > 1:
        angles = np.arctan2(dv, ds)
        curvature = np.abs(np.diff(angles))
        mean_curvature = np.mean(curvature)
        max_curvature = np.max(curvature)
    else:
        mean_curvature = 0
        max_curvature = 0

    stats_text = f"""
    3D PHASE SPACE STATISTICS:

    TRAJECTORY METRICS:
    Total cycles: {len(cycles)}
    Start point: S={stability[0]:.3f}, V={variance[0]:.3f}, P={phase_coherence[0]:.3f}
    End point: S={stability[-1]:.3f}, V={variance[-1]:.3f}, P={phase_coherence[-1]:.3f}

    Δ Stability: {stability[-1] - stability[0]:.3f} ({(stability[-1] - stability[0])/stability[0]*100:.1f}%)
    Δ Variance: {variance[-1] - variance[0]:.3f} ({(variance[0] - variance[-1])/variance[0]*100:.1f}% reduction)
    Δ Phase coherence: {phase_coherence[-1] - phase_coherence[0]:.3f}

    VELOCITY ANALYSIS:
    Mean velocity: {mean_velocity:.4f} ± {std_velocity:.4f}
    Max velocity: {max_velocity:.4f} (cycle {cycles[1:][np.argmax(velocity)]})
    Total path length: {total_distance:.3f}
    High velocity events: {len(high_vel_indices)} (>{high_vel_threshold:.4f})

    TRAJECTORY SHAPE:
    Mean curvature: {mean_curvature:.4f} rad
    Max curvature: {max_curvature:.4f} rad
    Straightness: {np.linalg.norm([stability[-1]-stability[0], variance[-1]-variance[0], phase_coherence[-1]-phase_coherence[0]]) / total_distance:.3f}

    CONVERGENCE METRICS:
    Distance to target (final): {distance[-1]:.3f}
    Convergence rate: {(distance[0] - distance[-1]) / len(cycles):.4f} per cycle
    Monotonic improvement: {np.sum(np.diff(stability) > 0) / len(np.diff(stability)) * 100:.1f}% of cycles

    CRITICAL TRANSITIONS:
    Number of jumps (ΔS > 0.1): {len(critical_indices)}
    Cycles with jumps: {cycles[critical_indices].tolist() if len(critical_indices) > 0 else 'None'}
    """

    ax5f.text(0.05, 0.95, stats_text, transform=ax5f.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    fig5.suptitle('3D Phase Space Analysis: Stability-Variance-Coherence Trajectory\n'
                'Quantitative Evidence for Phase-Locked Convergence',
                fontsize=18, fontweight='bold', y=0.998)

    plt.savefig('FIGURE_5_3D_PHASE_SPACE.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('FIGURE_5_3D_PHASE_SPACE.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 5 saved")

    # ============================================================
    # FIGURE 6: POLAR HISTOGRAMS & CIRCULAR STATISTICS
    # ============================================================

    print("\n" + "="*80)
    print("FIGURE 6: POLAR ANALYSIS & CIRCULAR STATISTICS")
    print("="*80)

    fig6 = plt.figure(figsize=(28, 24))
    gs6 = GridSpec(4, 4, figure=fig6, hspace=0.45, wspace=0.45)

    # Convert cycle to phase angle (0 to 2π)
    cycle_phase = 2 * np.pi * cycles / np.max(cycles)

    # Panel 6A: Stability Polar Plot
    ax6a = fig6.add_subplot(gs6[0, 0], projection='polar')

    ax6a.plot(cycle_phase, stability, 'o-', linewidth=2.5, markersize=8,
            color='#9b59b6', alpha=0.8)
    ax6a.fill(cycle_phase, stability, alpha=0.2, color='#9b59b6')

    ax6a.scatter([cycle_phase[0]], [stability[0]], s=300, c='red', marker='o',
                edgecolor='black', linewidth=2, zorder=5)
    ax6a.scatter([cycle_phase[-1]], [stability[-1]], s=300, c='green', marker='*',
                edgecolor='black', linewidth=2, zorder=5)

    ax6a.set_title('(A) Stability Polar Plot\nRadial = Stability, Angular = Cycle',
                fontsize=11, fontweight='bold', pad=20)
    ax6a.set_ylim(0, 1)

    # Panel 6B: Variance Polar Plot
    ax6b = fig6.add_subplot(gs6[0, 1], projection='polar')

    ax6b.plot(cycle_phase, variance, 'o-', linewidth=2.5, markersize=8,
            color='#e74c3c', alpha=0.8)
    ax6b.fill(cycle_phase, variance, alpha=0.2, color='#e74c3c')

    ax6b.scatter([cycle_phase[0]], [variance[0]], s=300, c='red', marker='o',
                edgecolor='black', linewidth=2, zorder=5)
    ax6b.scatter([cycle_phase[-1]], [variance[-1]], s=300, c='green', marker='*',
                edgecolor='black', linewidth=2, zorder=5)

    ax6b.set_title('(B) Variance Polar Plot\nRadial = Variance, Angular = Cycle',
                fontsize=11, fontweight='bold', pad=20)

    # Panel 6C: Phase Coherence Polar Plot
    ax6c = fig6.add_subplot(gs6[0, 2], projection='polar')

    ax6c.plot(cycle_phase, phase_coherence, 'o-', linewidth=2.5, markersize=8,
            color='#2ecc71', alpha=0.8)
    ax6c.fill(cycle_phase, phase_coherence, alpha=0.2, color='#2ecc71')

    ax6c.scatter([cycle_phase[0]], [phase_coherence[0]], s=300, c='red', marker='o',
                edgecolor='black', linewidth=2, zorder=5)
    ax6c.scatter([cycle_phase[-1]], [phase_coherence[-1]], s=300, c='green', marker='*',
                edgecolor='black', linewidth=2, zorder=5)

    ax6c.set_title('(C) Phase Coherence Polar\nRadial = Coherence, Angular = Cycle',
                fontsize=11, fontweight='bold', pad=20)
    ax6c.set_ylim(0, 1)

    # Panel 6D: Combined Polar Histogram
    ax6d = fig6.add_subplot(gs6[0, 3], projection='polar')

    # Create bins
    n_bins = 12
    bin_edges = np.linspace(0, 2*np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Histogram of stability by phase
    stability_counts, _ = np.histogram(cycle_phase, bins=bin_edges)
    # Avoid division by zero
    stability_counts_safe = np.where(stability_counts > 0, stability_counts, 1)
    stability_hist, _ = np.histogram(cycle_phase, bins=bin_edges, weights=stability)
    stability_hist = stability_hist / stability_counts_safe

    bars = ax6d.bar(bin_centers, stability_hist, width=2*np.pi/n_bins,
                alpha=0.7, color='#3498db', edgecolor='black', linewidth=1.5)

    ax6d.set_title('(D) Stability Histogram\nBinned by Cycle Phase',
                fontsize=11, fontweight='bold', pad=20)

    # Panel 6E: Rose Diagram (Directional Histogram) - FIXED
    ax6e = fig6.add_subplot(gs6[1, 0], projection='polar')

    # Calculate direction of movement in phase space
    if len(ds) > 0 and len(dv) > 0:
        directions = np.arctan2(dv, ds)
        directions = (directions + 2*np.pi) % (2*np.pi)  # Normalize to [0, 2π]

        # Rose diagram
        n_petals = 16
        petal_edges = np.linspace(0, 2*np.pi, n_petals + 1)
        petal_centers = (petal_edges[:-1] + petal_edges[1:]) / 2

        direction_hist, _ = np.histogram(directions, bins=petal_edges)

        bars = ax6e.bar(petal_centers, direction_hist, width=2*np.pi/n_petals,
                    alpha=0.7, color='#f39c12', edgecolor='black', linewidth=1.5)

    ax6e.set_title('(E) Rose Diagram\nDirection of Movement',
                fontsize=11, fontweight='bold', pad=20)

    # Panel 6F: Circular Mean & Variance
    ax6f = fig6.add_subplot(gs6[1, 1], projection='polar')

    # Calculate circular statistics
    cos_mean = np.sum(stability * np.cos(cycle_phase)) / np.sum(stability)
    sin_mean = np.sum(stability * np.sin(cycle_phase)) / np.sum(stability)
    circular_mean = np.arctan2(sin_mean, cos_mean)
    circular_variance = 1 - np.sqrt(cos_mean**2 + sin_mean**2)

    # Plot
    ax6f.plot(cycle_phase, stability, 'o', markersize=8, alpha=0.5, color='gray')

    # Draw mean direction
    ax6f.arrow(0, 0, circular_mean, 0.8, head_width=0.15, head_length=0.1,
            fc='red', ec='black', linewidth=3, alpha=0.8, zorder=5)

    ax6f.set_title('(F) Circular Mean & Variance\nRed Arrow = Mean Direction',
                fontsize=11, fontweight='bold', pad=20)
    ax6f.set_ylim(0, 1)

    # Panel 6G: Phase Synchronization Index
    ax6g = fig6.add_subplot(gs6[1, 2:])

    # Calculate phase synchronization (Kuramoto order parameter)
    order_parameter = []
    for i in range(len(cycles)):
        phases = cycle_phase[:i+1]
        amplitudes = stability[:i+1]

        r = np.abs(np.sum(amplitudes * np.exp(1j * phases))) / np.sum(amplitudes)
        order_parameter.append(r)

    order_parameter = np.array(order_parameter)

    ax6g.plot(cycles, order_parameter, 'o-', linewidth=3, markersize=10,
            color='#9b59b6', alpha=0.8)
    ax6g.fill_between(cycles, order_parameter, alpha=0.3, color='#9b59b6')

    # Mark synchronization threshold
    ax6g.axhline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax6g.text(cycles[-1]*0.5, 0.72, 'Synchronization Threshold',
            fontsize=10, fontweight='bold', color='red')

    ax6g.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax6g.set_ylabel('Order Parameter (r)', fontsize=12, fontweight='bold')
    ax6g.set_title('(G) Phase Synchronization Index (Kuramoto)\nr = 1: Perfect Sync, r = 0: Random',
                fontsize=13, fontweight='bold')
    ax6g.set_ylim(0, 1)
    ax6g.grid(alpha=0.3, linestyle='--')

    # Panel 6H: Circular Statistics Summary
    ax6h = fig6.add_subplot(gs6[2, :2])
    ax6h.axis('off')

    # Calculate additional circular statistics
    R = np.sqrt(cos_mean**2 + sin_mean**2)
    n = len(cycles)
    z = n * R**2
    p_rayleigh = np.exp(-z) if z < 100 else 0  # Avoid overflow

    mean_resultant = R
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else 0

    circular_stats_text = f"""
    CIRCULAR STATISTICS:

    BASIC METRICS:
    Sample size: {n}
    Circular mean: {circular_mean:.4f} rad ({np.degrees(circular_mean):.1f}°)
    Circular variance: {circular_variance:.4f}
    Circular std dev: {circular_std:.4f} rad ({np.degrees(circular_std):.1f}°)

    CONCENTRATION:
    Mean resultant length (R): {mean_resultant:.4f}
    Interpretation: {'High concentration' if mean_resultant > 0.7 else 'Moderate' if mean_resultant > 0.3 else 'Low'}

    UNIFORMITY TEST (Rayleigh):
    Test statistic (z): {z:.4f}
    p-value: {p_rayleigh:.6f}
    Result: {'REJECT uniformity' if p_rayleigh < 0.05 else 'Cannot reject'}

    SYNCHRONIZATION:
    Final order parameter: {order_parameter[-1]:.4f}
    Status: {'YES (r > 0.7)' if order_parameter[-1] > 0.7 else 'PARTIAL' if order_parameter[-1] > 0.3 else 'NO'}
    Cycles to sync: {np.where(order_parameter > 0.7)[0][0] + 1 if np.any(order_parameter > 0.7) else 'N/A'}
    """

    ax6h.text(0.05, 0.95, circular_stats_text, transform=ax6h.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    # Panel 6I: Angular Velocity - FIXED
    ax6i = fig6.add_subplot(gs6[2, 2:])

    # Calculate angular velocity (change in phase direction)
    if len(directions) > 1:
        angular_velocity = np.diff(directions)
        # Handle wraparound
        angular_velocity = (angular_velocity + np.pi) % (2*np.pi) - np.pi

        # Match array lengths: angular_velocity has length len(directions)-1
        # directions has length len(ds), which is len(cycles)-1
        # So angular_velocity has length len(cycles)-2
        cycles_for_angular = cycles[2:]  # Skip first 2 cycles

        ax6i.plot(cycles_for_angular, angular_velocity, 'o-', linewidth=2.5, markersize=8,
                color='#e74c3c', alpha=0.8)
        ax6i.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax6i.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax6i.set_ylabel('Angular Velocity (rad/cycle)', fontsize=12, fontweight='bold')
        ax6i.set_title('(I) Angular Velocity in Phase Space\nRate of Direction Change',
                    fontsize=13, fontweight='bold')
        ax6i.grid(alpha=0.3, linestyle='--')
    else:
        ax6i.text(0.5, 0.5, 'Insufficient data\nfor angular velocity',
                ha='center', va='center', transform=ax6i.transAxes,
                fontsize=12, fontweight='bold')
        ax6i.axis('off')

    # Panel 6J: Vector Field
    ax6j = fig6.add_subplot(gs6[3, :], projection='polar')

    # Create simplified vector field
    theta_points = cycle_phase[::2]  # Every other point
    r_points = stability[::2]

    # Calculate local gradients
    if len(ds) > 0:
        U_points = ds[::2] * 0.1 if len(ds) >= len(theta_points) else ds[:len(theta_points)] * 0.1
        V_points = np.diff(cycle_phase)[::2] * 0.1 if len(np.diff(cycle_phase)) >= len(theta_points) else np.diff(cycle_phase)[:len(theta_points)] * 0.1

        # Ensure same length
        min_len = min(len(theta_points), len(r_points), len(U_points), len(V_points))
        theta_points = theta_points[:min_len]
        r_points = r_points[:min_len]
        U_points = U_points[:min_len]
        V_points = V_points[:min_len]

        ax6j.quiver(theta_points, r_points, U_points, V_points,
                alpha=0.5, color='blue', scale=0.5)

    # Overlay actual trajectory
    ax6j.plot(cycle_phase, stability, 'r-', linewidth=3, alpha=0.8, label='Actual trajectory')
    ax6j.scatter([cycle_phase[0]], [stability[0]], s=300, c='red', marker='o',
                edgecolor='black', linewidth=2, zorder=5)
    ax6j.scatter([cycle_phase[-1]], [stability[-1]], s=300, c='green', marker='*',
                edgecolor='black', linewidth=2, zorder=5)

    ax6j.set_title('(J) Phase Space Vector Field\nFlow Dynamics in Polar Coordinates',
                fontsize=13, fontweight='bold', pad=20)
    ax6j.set_ylim(0, 1)
    ax6j.legend(fontsize=10)

    fig6.suptitle('Polar Analysis & Circular Statistics\n'
                'Evidence for Phase-Locked Circular Dynamics',
                fontsize=18, fontweight='bold', y=0.998)

    plt.savefig('FIGURE_6_POLAR_ANALYSIS.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('FIGURE_6_POLAR_ANALYSIS.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 6 saved")

    # ============================================================
    # FIGURE 7: PHASE RESPONSE CURVES
    # ============================================================

    print("\n" + "="*80)
    print("FIGURE 7: PHASE RESPONSE CURVES & PERTURBATION ANALYSIS")
    print("="*80)

    fig7 = plt.figure(figsize=(28, 24))
    gs7 = GridSpec(4, 3, figure=fig7, hspace=0.45, wspace=0.35)

    # Panel 7A: Stability Phase Response Curve
    ax7a = fig7.add_subplot(gs7[0, :])

    # Calculate phase response: how does stability change with cycle phase?
    def phase_response_func(phase, A, B, C, D):
        return A + B * np.cos(phase + C) + D * np.cos(2 * phase)

    try:
        popt_stability, _ = curve_fit(phase_response_func, cycle_phase, stability,
                                    p0=[0.6, 0.2, 0, 0.1], maxfev=10000)

        # Generate smooth curve
        phase_smooth = np.linspace(0, 2*np.pi, 200)
        stability_fit = phase_response_func(phase_smooth, *popt_stability)

        ax7a.plot(phase_smooth, stability_fit, '-', linewidth=3,
                color='#9b59b6', alpha=0.8, label='Fitted PRC')
        ax7a.plot(cycle_phase, stability, 'o', markersize=10,
                color='#3498db', alpha=0.6, label='Data')

        # Mark peaks and troughs
        peaks = signal.find_peaks(stability_fit)[0]
        troughs = signal.find_peaks(-stability_fit)[0]

        if len(peaks) > 0:
            ax7a.scatter(phase_smooth[peaks], stability_fit[peaks],
                        s=400, c='green', marker='^', edgecolor='black',
                        linewidth=2, zorder=5, label='Peaks')
        if len(troughs) > 0:
            ax7a.scatter(phase_smooth[troughs], stability_fit[troughs],
                        s=400, c='red', marker='v', edgecolor='black',
                        linewidth=2, zorder=5, label='Troughs')

        fit_success = True
    except Exception as e:
        print(f"  Warning: PRC fitting failed: {e}")
        ax7a.plot(cycle_phase, stability, 'o-', linewidth=2, markersize=10,
                color='#9b59b6', alpha=0.8)
        fit_success = False

    ax7a.set_xlabel('Cycle Phase (rad)', fontsize=12, fontweight='bold')
    ax7a.set_ylabel('Stability', fontsize=12, fontweight='bold')
    ax7a.set_title('(A) Stability Phase Response Curve (PRC)\nPeriodic Response to Cycle Phase',
                fontsize=14, fontweight='bold')
    ax7a.set_xlim(0, 2*np.pi)
    ax7a.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax7a.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax7a.legend(fontsize=10)
    ax7a.grid(alpha=0.3, linestyle='--')

    # Panel 7B: Variance Phase Response Curve
    ax7b = fig7.add_subplot(gs7[1, :])

    try:
        popt_variance, _ = curve_fit(phase_response_func, cycle_phase, variance,
                                    p0=[0.08, 0.05, 0, 0.02], maxfev=10000)

        variance_fit = phase_response_func(phase_smooth, *popt_variance)

        ax7b.plot(phase_smooth, variance_fit, '-', linewidth=3,
                color='#e74c3c', alpha=0.8, label='Fitted PRC')
        ax7b.plot(cycle_phase, variance, 'o', markersize=10,
                color='#f39c12', alpha=0.6, label='Data')

        peaks_v = signal.find_peaks(variance_fit)[0]
        troughs_v = signal.find_peaks(-variance_fit)[0]

        if len(peaks_v) > 0:
            ax7b.scatter(phase_smooth[peaks_v], variance_fit[peaks_v],
                        s=400, c='red', marker='^', edgecolor='black',
                        linewidth=2, zorder=5, label='Peaks')
        if len(troughs_v) > 0:
            ax7b.scatter(phase_smooth[troughs_v], variance_fit[troughs_v],
                        s=400, c='green', marker='v', edgecolor='black',
                        linewidth=2, zorder=5, label='Troughs')

    except Exception as e:
        print(f"  Warning: Variance PRC fitting failed: {e}")
        ax7b.plot(cycle_phase, variance, 'o-', linewidth=2, markersize=10,
                color='#e74c3c', alpha=0.8)

    ax7b.set_xlabel('Cycle Phase (rad)', fontsize=12, fontweight='bold')
    ax7b.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax7b.set_title('(B) Variance Phase Response Curve (PRC)\nAnti-Correlated with Stability',
                fontsize=14, fontweight='bold')
    ax7b.set_xlim(0, 2*np.pi)
    ax7b.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax7b.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax7b.legend(fontsize=10)
    ax7b.grid(alpha=0.3, linestyle='--')

    # Panel 7C: Phase Sensitivity
    ax7c = fig7.add_subplot(gs7[2, :2])

    if fit_success:
        phase_sensitivity = np.gradient(stability_fit, phase_smooth)

        ax7c.plot(phase_smooth, phase_sensitivity, linewidth=3,
                color='#2ecc71', alpha=0.8)
        ax7c.fill_between(phase_smooth, phase_sensitivity, alpha=0.3, color='#2ecc71')

        # Mark zero crossings
        zero_crossings = np.where(np.diff(np.sign(phase_sensitivity)))[0]
        if len(zero_crossings) > 0:
            ax7c.scatter(phase_smooth[zero_crossings], phase_sensitivity[zero_crossings],
                        s=300, c='red', marker='o', edgecolor='black', linewidth=2, zorder=5)

        ax7c.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    else:
        phase_sensitivity = np.gradient(stability, cycle_phase)
        ax7c.plot(cycle_phase, phase_sensitivity, 'o-', linewidth=2, markersize=8,
                color='#2ecc71', alpha=0.8)

    ax7c.set_xlabel('Cycle Phase (rad)', fontsize=12, fontweight='bold')
    ax7c.set_ylabel('dS/dφ (Phase Sensitivity)', fontsize=12, fontweight='bold')
    ax7c.set_title('(C) Phase Sensitivity\nRate of Change of Stability with Phase',
                fontsize=13, fontweight='bold')
    ax7c.set_xlim(0, 2*np.pi)
    ax7c.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax7c.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax7c.grid(alpha=0.3, linestyle='--')

    # Panel 7D: PRC Statistics
    ax7d = fig7.add_subplot(gs7[2, 2])
    ax7d.axis('off')

    if fit_success:
        prc_stats_text = f"""
    PRC STATISTICS:

    STABILITY PRC:
    Amplitude: {popt_stability[1]:.4f}
    Phase shift: {popt_stability[2]:.4f} rad
    2nd harmonic: {popt_stability[3]:.4f}
    Mean: {popt_stability[0]:.4f}

    Peaks: {len(peaks)}
    Troughs: {len(troughs)}

    INTERPRETATION:
    {'✓ Strong periodic' if abs(popt_stability[1]) > 0.1 else '⧗ Weak periodic'}
    {'✓ Phase-locked' if len(peaks) > 0 else '⧗ No phase-lock'}
    """
    else:
        prc_stats_text = """
    PRC STATISTICS:

    Fitting unsuccessful.
    Data may not show
    clear periodic structure.
    """

    ax7d.text(0.1, 0.9, prc_stats_text, transform=ax7d.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    # Panel 7E: Perturbation Response
    ax7e = fig7.add_subplot(gs7[3, :])

    # Simulate perturbation response
    perturbation_phases = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    perturbation_responses = []

    for pert_phase in perturbation_phases:
        idx = np.argmin(np.abs(cycle_phase - pert_phase))

        if idx < len(stability) - 3:
            response = stability[idx+3] - stability[idx]
            perturbation_responses.append(response)
        else:
            perturbation_responses.append(0)

    perturbation_responses = np.array(perturbation_responses)

    ax7e.bar(perturbation_phases, perturbation_responses,
            width=np.pi/8, alpha=0.7, color='#f39c12',
            edgecolor='black', linewidth=2)

    ax7e.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax7e.set_xlabel('Perturbation Phase (rad)', fontsize=12, fontweight='bold')
    ax7e.set_ylabel('Response (ΔStability)', fontsize=12, fontweight='bold')
    ax7e.set_title('(E) Perturbation Response Analysis\nSystem Response to Phase-Specific Perturbations',
                fontsize=13, fontweight='bold')
    ax7e.set_xlim(0, 2*np.pi)
    ax7e.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax7e.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax7e.grid(alpha=0.3, linestyle='--', axis='y')

    fig7.suptitle('Phase Response Curves & Perturbation Analysis\n'
                'Evidence for Phase-Dependent Dynamics',
                fontsize=18, fontweight='bold', y=0.998)

    plt.savefig('FIGURE_7_PHASE_RESPONSE.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('FIGURE_7_PHASE_RESPONSE.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 7 saved")

    print("\n" + "="*80)
    print("ADVANCED STATISTICAL ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated figures:")
    print("  5. FIGURE_5_3D_PHASE_SPACE.pdf/png")
    print("  6. FIGURE_6_POLAR_ANALYSIS.pdf/png")
    print("  7. FIGURE_7_PHASE_RESPONSE.pdf/png")
    print("\n✓ All dimension mismatches fixed!")
    print("="*80)
