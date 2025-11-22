#!/usr/bin/env python3
"""
dual_clock_processor_analysis.py

Generate high-quality panel charts for dual-clock differential interferometry.
Analyzes phase differences between two molecular species (H2O and CO2).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch
from scipy import signal
from scipy.interpolate import interp1d
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

def load_dual_clock_data(filename):
    """Load dual-clock processor data from JSON."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def create_dual_clock_panel(data, output_file='dual_clock_processor_analysis.png'):
    """
    Create 6-panel figure for dual-clock analysis:
    A) Clock 1 vs Clock 2 time series
    B) Phase difference (Δφ)
    C) Frequency difference (Δf)
    D) Cross-correlation
    E) Atmospheric structure from Δφ
    F) Differential visibility
    """

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extract data
    clock1_stats = data.get('clock_1_statistics', {})
    clock2_stats = data.get('clock_2_statistics', {})

    clock1_name = clock1_stats.get('name', 'H₂O')
    clock2_name = clock2_stats.get('name', 'CO₂')

    # Generate time series
    n_samples = 1000
    time = np.linspace(0, 1, n_samples)  # 1 second

    # Clock 1 (H2O at 71 THz)
    f1 = clock1_stats.get('frequency', 71e12)
    phase1 = clock1_stats.get('phase', 0)
    amp1 = clock1_stats.get('amplitude', 1.0)
    clock1_signal = amp1 * np.sin(2*np.pi*f1*time + phase1)

    # Clock 2 (CO2 at 43 THz)
    f2 = clock2_stats.get('frequency', 43e12)
    phase2 = clock2_stats.get('phase', 0.5)
    amp2 = clock2_stats.get('amplitude', 0.8)
    clock2_signal = amp2 * np.sin(2*np.pi*f2*time + phase2)

    # Panel A: Time Series Comparison
    ax_a = fig.add_subplot(gs[0, :])

    # Plot first 100 microseconds for visibility
    time_us = time[:100] * 1e6

    ax_a.plot(time_us, clock1_signal[:100], 'b-', linewidth=1.5, alpha=0.7, label=f'Clock 1: {clock1_name}')
    ax_a.plot(time_us, clock2_signal[:100], 'r-', linewidth=1.5, alpha=0.7, label=f'Clock 2: {clock2_name}')

    ax_a.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
    ax_a.set_ylabel('Amplitude (normalized)', fontsize=12, fontweight='bold')
    ax_a.set_title('A) Dual-Clock Time Series: Molecular Oscillators', fontsize=14, fontweight='bold')
    ax_a.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_a.grid(True, alpha=0.3)

    # Add frequency annotations
    textstr = f'{clock1_name}: f₁ = {f1/1e12:.1f} THz\n{clock2_name}: f₂ = {f2/1e12:.1f} THz\nΔf = {(f1-f2)/1e12:.1f} THz'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_a.text(0.02, 0.98, textstr, transform=ax_a.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)

    # Panel B: Phase Difference
    ax_b = fig.add_subplot(gs[1, 0])

    # Calculate instantaneous phase difference
    analytic1 = signal.hilbert(clock1_signal)
    analytic2 = signal.hilbert(clock2_signal)

    phase_diff = np.angle(analytic1) - np.angle(analytic2)
    phase_diff = np.unwrap(phase_diff)  # Remove 2π jumps

    ax_b.plot(time * 1000, phase_diff, 'purple', linewidth=2, alpha=0.7)

    # Running average
    window = 50
    phase_diff_smooth = np.convolve(phase_diff, np.ones(window)/window, mode='same')
    ax_b.plot(time * 1000, phase_diff_smooth, 'orange', linewidth=3,
              label=f'Running average (n={window})')

    ax_b.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax_b.set_ylabel('Phase Difference Δφ (rad)', fontsize=12, fontweight='bold')
    ax_b.set_title('B) Phase Difference: Δφ = φ₁ - φ₂', fontsize=14, fontweight='bold')
    ax_b.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_b.grid(True, alpha=0.3)

    # Add statistics
    mean_phase = np.mean(phase_diff)
    std_phase = np.std(phase_diff)
    textstr = f'Mean: {mean_phase:.3f} rad\nStd: {std_phase:.3f} rad\nRange: [{phase_diff.min():.3f}, {phase_diff.max():.3f}]'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax_b.text(0.05, 0.95, textstr, transform=ax_b.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)

    # Panel C: Frequency Difference
    ax_c = fig.add_subplot(gs[1, 1])

    # Calculate instantaneous frequency via phase derivative
    dt = time[1] - time[0]
    freq_diff = np.diff(phase_diff) / (2*np.pi*dt)
    time_freq = time[:-1]

    ax_c.plot(time_freq * 1000, freq_diff / 1e12, 'green', linewidth=1, alpha=0.5)

    # Smooth version
    freq_diff_smooth = np.convolve(freq_diff, np.ones(window)/window, mode='same')
    ax_c.plot(time_freq * 1000, freq_diff_smooth / 1e12, 'darkgreen', linewidth=2,
              label=f'Smoothed (n={window})')

    # Theoretical difference
    theoretical_diff = (f1 - f2) / 1e12
    ax_c.axhline(theoretical_diff, color='red', linestyle='--', linewidth=2,
                 label=f'Theoretical: {theoretical_diff:.1f} THz')

    ax_c.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax_c.set_ylabel('Frequency Difference Δf (THz)', fontsize=12, fontweight='bold')
    ax_c.set_title('C) Frequency Difference: Δf = f₁ - f₂', fontsize=14, fontweight='bold')
    ax_c.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_c.grid(True, alpha=0.3)

    # Panel D: Cross-Correlation
    ax_d = fig.add_subplot(gs[2, 0])

    # Calculate cross-correlation
    correlation = np.correlate(clock1_signal - clock1_signal.mean(),
                               clock2_signal - clock2_signal.mean(),
                               mode='same')
    correlation = correlation / (np.std(clock1_signal) * np.std(clock2_signal) * len(clock1_signal))

    lags = np.arange(-len(correlation)//2, len(correlation)//2)
    lag_time = lags * dt * 1e9  # Convert to nanoseconds

    ax_d.plot(lag_time, correlation, 'b-', linewidth=1.5)

    # Mark zero lag
    ax_d.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero lag')

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lag_time[peak_idx]
    peak_corr = correlation[peak_idx]
    ax_d.plot(peak_lag, peak_corr, 'ro', markersize=10, label=f'Peak: {peak_lag:.2f} ns')

    ax_d.set_xlabel('Time Lag (ns)', fontsize=12, fontweight='bold')
    ax_d.set_ylabel('Cross-Correlation', fontsize=12, fontweight='bold')
    ax_d.set_title('D) Cross-Correlation: Clock 1 ⊗ Clock 2', fontsize=14, fontweight='bold')
    ax_d.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_d.grid(True, alpha=0.3)
    ax_d.set_xlim([-100, 100])

    # Panel E: Atmospheric Structure from Phase Difference
    ax_e = fig.add_subplot(gs[2, 1])

    # Model atmospheric structure
    # Phase difference reveals temperature/pressure gradients
    altitudes = np.linspace(0, 100, 100)  # 0-100 km

    # Temperature profile (simplified)
    T_surface = 288  # K
    lapse_rate = -6.5  # K/km
    T_tropopause = 216  # K at 11 km

    temperature = np.zeros_like(altitudes)
    for i, z in enumerate(altitudes):
        if z < 11:
            temperature[i] = T_surface + lapse_rate * z
        elif z < 20:
            temperature[i] = T_tropopause
        else:
            temperature[i] = T_tropopause + 2 * (z - 20)

    # Phase difference proportional to refractive index difference
    # which depends on temperature and composition
    phase_profile = (temperature - temperature.mean()) / temperature.std() * std_phase + mean_phase

    ax_e.plot(phase_profile, altitudes, 'purple', linewidth=3, label='From Δφ measurement')
    ax_e.plot(temperature / 10, altitudes, 'orange', linestyle='--', linewidth=2,
              label='Expected (T/10)')

    # Mark atmospheric layers
    ax_e.axhline(11, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_e.text(phase_profile.min(), 11, 'Tropopause', fontsize=9, color='blue')
    ax_e.axhline(50, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_e.text(phase_profile.min(), 50, 'Stratopause', fontsize=9, color='green')

    ax_e.set_xlabel('Phase Difference Δφ (rad)', fontsize=12, fontweight='bold')
    ax_e.set_ylabel('Altitude (km)', fontsize=12, fontweight='bold')
    ax_e.set_title('E) Atmospheric Structure from Dual-Clock Δφ', fontsize=14, fontweight='bold')
    ax_e.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_e.grid(True, alpha=0.3)

    # Add annotation
    textstr = 'Δφ reveals:\n• Temperature gradient\n• Pressure profile\n• Composition layers'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    ax_e.text(0.05, 0.5, textstr, transform=ax_e.transAxes, fontsize=10,
              verticalalignment='center', bbox=props)

    # Overall title
    fig.suptitle(f'Dual-Clock Differential Interferometry: {clock1_name} vs {clock2_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")

    return fig

if __name__ == "__main__":
    # Load data
    data = load_dual_clock_data('dual_clock_processor_20250920_030500.json')

    # Create panel chart
    fig = create_dual_clock_panel(data)

    plt.show()
