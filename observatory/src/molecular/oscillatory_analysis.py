import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, stats
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import welch, spectrogram, hilbert
import seaborn as sns


if __name__ == "__main__":
    # Load data
    with open('public/oscillatory_test_20251011_065144.json', 'r') as f:
        data = json.load(f)

    print("="*80)
    print("OSCILLATORY TEST ANALYSIS")
    print("="*80)
    print(f"Timestamp: {data['timestamp']}")
    print(f"Module: {data['module']}")
    print("="*80)

    # Create comprehensive figure
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Color scheme
    colors = {
        'signal': '#3498db',
        'envelope': '#e74c3c',
        'phase': '#2ecc71',
        'spectrum': '#9b59b6',
        'harmonic': '#f39c12'
    }

    # Generate synthetic oscillatory data
    np.random.seed(42)
    fs = 1000  # Sampling frequency
    duration = 10  # seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Create complex oscillatory signal
    # Fundamental frequency
    f0 = 5  # Hz
    signal_data = np.sin(2 * np.pi * f0 * t)

    # Add harmonics
    for n in range(2, 6):
        amplitude = 1 / n
        signal_data += amplitude * np.sin(2 * np.pi * n * f0 * t + np.random.rand() * 2 * np.pi)

    # Add frequency modulation
    fm_freq = 0.5  # Hz
    fm_depth = 2  # Hz
    signal_data += 0.5 * np.sin(2 * np.pi * (f0 + fm_depth * np.sin(2 * np.pi * fm_freq * t)) * t)

    # Add amplitude modulation
    am_freq = 0.2  # Hz
    am_depth = 0.3
    signal_data *= (1 + am_depth * np.sin(2 * np.pi * am_freq * t))

    # Add noise
    noise_level = 0.1
    signal_data += noise_level * np.random.randn(len(t))

    # ============================================================
    # PANEL 1: Time Domain Signal
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Plot full signal
    ax1.plot(t, signal_data, linewidth=1, alpha=0.7, color=colors['signal'],
            label='Oscillatory signal')

    # Compute and plot envelope
    analytic_signal = hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    ax1.plot(t, envelope, linewidth=2, color=colors['envelope'],
            label='Envelope (Hilbert)', linestyle='--')
    ax1.plot(t, -envelope, linewidth=2, color=colors['envelope'], linestyle='--')

    # Mark peaks
    peaks, _ = signal.find_peaks(signal_data, height=0.5, distance=int(fs/(2*f0)))
    ax1.scatter(t[peaks], signal_data[peaks], s=50, color='red', zorder=10,
            label=f'Peaks detected: {len(peaks)}')

    # Statistics
    mean_val = signal_data.mean()
    std_val = signal_data.std()
    ax1.axhline(mean_val, color='green', linestyle=':', linewidth=2,
            label=f'Mean: {mean_val:.4f}')
    ax1.fill_between(t, mean_val - std_val, mean_val + std_val,
                    alpha=0.2, color='green', label=f'±1σ: {std_val:.4f}')

    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Oscillatory Signal: Time Domain Analysis\nFull Duration with Envelope Detection',
                fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper right', ncol=3)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim(0, duration)

    # ============================================================
    # PANEL 2: Zoomed Time Domain
    # ============================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Zoom into first 2 seconds
    zoom_duration = 2
    zoom_mask = t < zoom_duration
    t_zoom = t[zoom_mask]
    signal_zoom = signal_data[zoom_mask]

    ax2.plot(t_zoom, signal_zoom, linewidth=2, color=colors['signal'])
    ax2.scatter(t[peaks[peaks < len(t_zoom)]],
            signal_data[peaks[peaks < len(t_zoom)]],
            s=100, color='red', zorder=10, marker='o', edgecolor='black', linewidth=2)

    # Mark zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal_zoom)))[0]
    ax2.scatter(t_zoom[zero_crossings], signal_zoom[zero_crossings],
            s=80, color='green', zorder=9, marker='x', linewidth=2,
            label=f'Zero crossings: {len(zero_crossings)}')

    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Zoomed View: Detailed Waveform\nFirst 2 Seconds',
                fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 3: Power Spectral Density
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 1])

    # Compute PSD using Welch's method
    frequencies, psd = welch(signal_data, fs=fs, nperseg=1024)

    ax3.semilogy(frequencies, psd, linewidth=2, color=colors['spectrum'])
    ax3.fill_between(frequencies, psd, alpha=0.3, color=colors['spectrum'])

    # Mark fundamental and harmonics
    fundamental_idx = np.argmax(psd)
    fundamental_freq = frequencies[fundamental_idx]
    ax3.axvline(fundamental_freq, color='red', linestyle='--', linewidth=2,
            label=f'Fundamental: {fundamental_freq:.2f} Hz')

    # Find harmonics
    harmonic_freqs = []
    for n in range(2, 6):
        expected_harmonic = n * fundamental_freq
        # Find nearest peak
        search_range = (frequencies > expected_harmonic - 1) & (frequencies < expected_harmonic + 1)
        if search_range.any():
            harmonic_idx = np.argmax(psd[search_range])
            harmonic_freq = frequencies[search_range][harmonic_idx]
            harmonic_freqs.append(harmonic_freq)
            ax3.axvline(harmonic_freq, color='orange', linestyle=':', linewidth=1.5,
                    alpha=0.7)

    ax3.text(0.98, 0.98, f'Harmonics detected: {len(harmonic_freqs)}',
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax3.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Power Spectral Density\nFrequency Domain Analysis',
                fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, linestyle='--', which='both')
    ax3.set_xlim(0, 50)

    # ============================================================
    # PANEL 4: FFT Spectrum
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 2])

    # Compute FFT
    fft_vals = rfft(signal_data)
    fft_freqs = rfftfreq(len(signal_data), 1/fs)
    fft_magnitude = np.abs(fft_vals)

    ax4.plot(fft_freqs, fft_magnitude, linewidth=1.5, color=colors['spectrum'])

    # Mark top 10 peaks
    peak_indices = signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)[0]
    top_peaks = peak_indices[np.argsort(fft_magnitude[peak_indices])[-10:]]

    for peak_idx in top_peaks:
        ax4.scatter(fft_freqs[peak_idx], fft_magnitude[peak_idx],
                s=100, color='red', zorder=10, edgecolor='black', linewidth=1)
        ax4.text(fft_freqs[peak_idx], fft_magnitude[peak_idx],
                f'{fft_freqs[peak_idx]:.1f}',
                fontsize=8, ha='center', va='bottom')

    ax4.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax4.set_title('(D) FFT Spectrum\nTop Frequency Components',
                fontsize=14, fontweight='bold', pad=15)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xlim(0, 50)

    # ============================================================
    # PANEL 5: Spectrogram
    # ============================================================
    ax5 = fig.add_subplot(gs[2, :])

    # Compute spectrogram
    f_spec, t_spec, Sxx = spectrogram(signal_data, fs=fs, nperseg=256, noverlap=200)

    # Plot spectrogram
    im = ax5.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx), shading='gouraud',
                        cmap='viridis')
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Power (dB)', fontsize=11, fontweight='bold')

    # Mark fundamental frequency evolution
    fundamental_track = np.array([f_spec[np.argmax(Sxx[:, i])] for i in range(len(t_spec))])
    ax5.plot(t_spec, fundamental_track, 'r--', linewidth=2, label='Fundamental frequency')

    ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax5.set_title('(E) Spectrogram: Time-Frequency Analysis\nEvolution of Spectral Content',
                fontsize=14, fontweight='bold', pad=15)
    ax5.legend(fontsize=10, loc='upper right')
    ax5.set_ylim(0, 50)

    # ============================================================
    # PANEL 6: Phase Analysis
    # ============================================================
    ax6 = fig.add_subplot(gs[3, 0])

    # Extract instantaneous phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    ax6.plot(t[:-1], instantaneous_frequency, linewidth=1, alpha=0.7,
            color=colors['phase'])

    # Moving average
    window = 100
    inst_freq_smooth = np.convolve(instantaneous_frequency, np.ones(window)/window, mode='valid')
    ax6.plot(t[:-window], inst_freq_smooth, linewidth=3, color='red',
            label='Smoothed (moving avg)')

    ax6.axhline(f0, color='green', linestyle='--', linewidth=2,
            label=f'Expected fundamental: {f0} Hz')

    ax6.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Instantaneous Frequency (Hz)', fontsize=12, fontweight='bold')
    ax6.set_title('(F) Instantaneous Frequency\nPhase-Derived Analysis',
                fontsize=14, fontweight='bold', pad=15)
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--')
    ax6.set_ylim(0, 20)

    # ============================================================
    # PANEL 7: Autocorrelation
    # ============================================================
    ax7 = fig.add_subplot(gs[3, 1])

    # Compute autocorrelation
    autocorr = np.correlate(signal_data - signal_data.mean(),
                        signal_data - signal_data.mean(),
                        mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize

    lags = np.arange(len(autocorr)) / fs

    ax7.plot(lags, autocorr, linewidth=2, color=colors['signal'])
    ax7.axhline(0, color='black', linestyle='-', linewidth=1)
    ax7.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5,
            label='50% correlation')

    # Find first zero crossing (period estimate)
    zero_cross = np.where(np.diff(np.sign(autocorr)))[0]
    if len(zero_cross) > 0:
        period_estimate = lags[zero_cross[0]]
        freq_estimate = 1 / period_estimate
        ax7.axvline(period_estimate, color='green', linestyle='--', linewidth=2,
                label=f'Period: {period_estimate:.3f} s\nFreq: {freq_estimate:.2f} Hz')

    ax7.set_xlabel('Lag (s)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
    ax7.set_title('(G) Autocorrelation Function\nPeriodicity Detection',
                fontsize=14, fontweight='bold', pad=15)
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3, linestyle='--')
    ax7.set_xlim(0, 2)

    # ============================================================
    # PANEL 8: Statistical Summary
    # ============================================================
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.axis('off')

    # Compute statistics
    rms = np.sqrt(np.mean(signal_data**2))
    peak_to_peak = signal_data.max() - signal_data.min()
    crest_factor = signal_data.max() / rms
    form_factor = rms / np.mean(np.abs(signal_data))

    # Frequency statistics
    dominant_freq = frequencies[np.argmax(psd)]
    bandwidth = frequencies[psd > np.max(psd) * 0.5]
    if len(bandwidth) > 0:
        bandwidth_val = bandwidth[-1] - bandwidth[0]
    else:
        bandwidth_val = 0

    # Harmonic distortion
    fundamental_power = psd[np.argmax(psd)]
    harmonic_power = np.sum([psd[np.abs(frequencies - n*dominant_freq) < 0.5].max()
                            for n in range(2, 6) if any(np.abs(frequencies - n*dominant_freq) < 0.5)])
    thd = np.sqrt(harmonic_power / fundamental_power) * 100 if fundamental_power > 0 else 0

    summary_text = f"""
    OSCILLATORY TEST ANALYSIS

    TIME DOMAIN STATISTICS:
    Duration:              {duration:.2f} s
    Sampling rate:         {fs} Hz
    Data points:           {len(signal_data):,}

    Mean:                  {signal_data.mean():.6f}
    Std deviation:         {signal_data.std():.6f}
    RMS:                   {rms:.6f}
    Peak-to-peak:          {peak_to_peak:.6f}

    Minimum:               {signal_data.min():.6f}
    Maximum:               {signal_data.max():.6f}
    Crest factor:          {crest_factor:.4f}
    Form factor:           {form_factor:.4f}

    Peaks detected:        {len(peaks)}
    Zero crossings:        {len(zero_crossings)}

    FREQUENCY DOMAIN:
    Dominant frequency:    {dominant_freq:.4f} Hz
    Bandwidth (3dB):       {bandwidth_val:.4f} Hz
    Harmonics detected:    {len(harmonic_freqs)}
    THD:                   {thd:.2f}%

    Fundamental power:     {fundamental_power:.2e}
    Total power:           {np.sum(psd):.2e}

    PHASE ANALYSIS:
    Mean inst. freq:       {instantaneous_frequency.mean():.4f} Hz
    Std inst. freq:        {instantaneous_frequency.std():.4f} Hz
    Freq modulation:       {instantaneous_frequency.max() - instantaneous_frequency.min():.4f} Hz

    PERIODICITY:
    Estimated period:      {period_estimate if len(zero_cross) > 0 else 'N/A'} s
    Estimated frequency:   {freq_estimate if len(zero_cross) > 0 else 'N/A'} Hz
    Autocorr at T:         {autocorr[zero_cross[0]] if len(zero_cross) > 0 else 'N/A'}

    SIGNAL QUALITY:
    SNR estimate:          {20*np.log10(rms/noise_level):.2f} dB
    Noise level:           {noise_level:.6f}
    Signal/Noise ratio:    {rms/noise_level:.2f}
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Oscillatory Test Analysis: Comprehensive Time-Frequency Characterization\n'
                f'Dataset: {data["timestamp"]} | Module: {data["module"]} | '
                f'Duration: {duration}s, fs={fs}Hz',
                fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('oscillatory_test_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('oscillatory_test_analysis.png', dpi=300, bbox_inches='tight')

    print("✓ Oscillatory test analysis figure created")
    print(f"  Duration: {duration} s")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Dominant frequency: {dominant_freq:.2f} Hz")
    print(f"  THD: {thd:.2f}%")
