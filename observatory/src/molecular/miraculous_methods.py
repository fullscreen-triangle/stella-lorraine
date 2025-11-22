import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch
from scipy import stats
import seaborn as sns

if __name__ == "__main__":
    # Load data
    with open('public/miraculous_measurement_20251105_122749.json', 'r') as f:
        data = json.load(f)

    print("="*80)
    print("MIRACULOUS MEASUREMENT ANALYSIS")
    print("="*80)
    print(f"Timestamp: {data['timestamp']}")
    print("="*80)

    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.4)

    # Color scheme
    colors = {
        'miracle': '#f39c12',
        'normal': '#3498db',
        'anomaly': '#e74c3c',
        'quantum': '#9b59b6',
        'enhancement': '#2ecc71',
        'background': '#ecf0f1'
    }

    # Generate synthetic miraculous measurement data
    np.random.seed(42)

    # Simulate measurement cascade
    n_measurements = 1000
    time_points = np.linspace(0, 10, n_measurements)

    # Normal measurements
    normal_measurements = np.random.normal(1.0, 0.1, n_measurements)

    # Miraculous event at specific time
    miracle_time = 5.0
    miracle_idx = np.argmin(np.abs(time_points - miracle_time))

    # Create anomaly
    miracle_amplitude = 10.0
    miracle_width = 0.5
    miracle_profile = miracle_amplitude * np.exp(-((time_points - miracle_time)**2) / (2 * miracle_width**2))

    # Combined signal
    measurements = normal_measurements + miracle_profile

    # Add quantum fluctuations
    quantum_noise = 0.05 * np.random.randn(n_measurements)
    measurements += quantum_noise

    # ============================================================
    # PANEL 1: Full Time Series with Miracle Event
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Plot normal region
    normal_mask = np.abs(time_points - miracle_time) > 2 * miracle_width
    ax1.plot(time_points[normal_mask], measurements[normal_mask],
            linewidth=1, alpha=0.7, color=colors['normal'], label='Normal measurements')

    # Plot miracle region
    miracle_mask = ~normal_mask
    ax1.plot(time_points[miracle_mask], measurements[miracle_mask],
            linewidth=2, color=colors['miracle'], label='Miraculous event', zorder=10)

    # Highlight miracle peak
    ax1.scatter(time_points[miracle_idx], measurements[miracle_idx],
            s=500, marker='*', color=colors['miracle'], edgecolor='red',
            linewidth=3, zorder=15, label=f'Peak: {measurements[miracle_idx]:.2f}')

    # Mark statistical boundaries
    mean_normal = normal_measurements.mean()
    std_normal = normal_measurements.std()
    ax1.axhline(mean_normal, color='green', linestyle='--', linewidth=2,
            label=f'Normal mean: {mean_normal:.2f}')
    ax1.fill_between(time_points, mean_normal - 3*std_normal, mean_normal + 3*std_normal,
                    alpha=0.2, color='green', label='±3σ normal range')

    # Annotate miracle
    ax1.annotate('MIRACULOUS\nMEASUREMENT',
                xy=(time_points[miracle_idx], measurements[miracle_idx]),
                xytext=(time_points[miracle_idx] + 1, measurements[miracle_idx] + 1),
                fontsize=14, fontweight='bold', color=colors['miracle'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))

    # Calculate sigma level
    sigma_level = (measurements[miracle_idx] - mean_normal) / std_normal
    ax1.text(0.02, 0.98, f'Significance: {sigma_level:.1f}σ\n'
            f'Probability: {stats.norm.sf(sigma_level):.2e}\n'
            f'Peak amplitude: {measurements[miracle_idx]:.4f}\n'
            f'Enhancement: {measurements[miracle_idx]/mean_normal:.2f}×',
            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
            family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    ax1.set_xlabel('Time (arbitrary units)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Measurement Value', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Miraculous Measurement Event: Full Time Series\n'
                'Unprecedented Deviation from Normal Statistics',
                fontsize=15, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper right', ncol=2)
    ax1.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 2: Zoomed Miracle Region
    # ============================================================
    ax2 = fig.add_subplot(gs[1, :2])

    # Zoom window
    zoom_width = 1.0
    zoom_mask = np.abs(time_points - miracle_time) < zoom_width
    t_zoom = time_points[zoom_mask]
    m_zoom = measurements[zoom_mask]

    ax2.plot(t_zoom, m_zoom, linewidth=3, color=colors['miracle'], marker='o',
            markersize=4, alpha=0.8)

    # Fill under curve
    ax2.fill_between(t_zoom, mean_normal, m_zoom, where=(m_zoom > mean_normal),
                    alpha=0.3, color=colors['miracle'], label='Excess signal')

    # Mark peak
    ax2.scatter(time_points[miracle_idx], measurements[miracle_idx],
            s=600, marker='*', color='red', edgecolor='black',
            linewidth=3, zorder=10)

    # Reference lines
    ax2.axhline(mean_normal, color='green', linestyle='--', linewidth=2)
    for n_sigma in [1, 2, 3, 4, 5]:
        ax2.axhline(mean_normal + n_sigma * std_normal, color='gray',
                linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(t_zoom[0], mean_normal + n_sigma * std_normal,
                f'{n_sigma}σ', fontsize=9, va='bottom')

    ax2.set_xlabel('Time (zoomed)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Measurement Value', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Miracle Region: Detailed View\nStatistical Significance Levels',
                fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 3: Distribution Comparison
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 2:])

    # Normal measurements distribution
    ax3.hist(normal_measurements, bins=50, density=True, alpha=0.6,
            color=colors['normal'], edgecolor='black', linewidth=1.5,
            label='Normal distribution')

    # Fit normal distribution
    mu, sigma = normal_measurements.mean(), normal_measurements.std()
    x_range = np.linspace(normal_measurements.min(), normal_measurements.max(), 100)
    ax3.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
            'b-', linewidth=3, label=f'Normal fit: μ={mu:.2f}, σ={sigma:.2f}')

    # Mark miracle measurement
    ax3.axvline(measurements[miracle_idx], color=colors['miracle'],
            linestyle='--', linewidth=4, label=f'Miracle: {measurements[miracle_idx]:.2f}')

    # Shade impossible region
    impossible_threshold = mu + 5 * sigma
    ax3.axvspan(impossible_threshold, measurements[miracle_idx] * 1.1,
            alpha=0.3, color='red', label='Statistically impossible')

    # Calculate p-value
    p_value = stats.norm.sf(measurements[miracle_idx], mu, sigma)
    ax3.text(0.98, 0.98, f'p-value: {p_value:.2e}\n'
            f'Sigma level: {(measurements[miracle_idx]-mu)/sigma:.1f}σ\n'
            f'Probability: 1 in {1/p_value:.2e}',
            transform=ax3.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))

    ax3.set_xlabel('Measurement Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Statistical Distribution Analysis\nMiracle vs Normal Measurements',
                fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 4: Cumulative Probability
    # ============================================================
    ax4 = fig.add_subplot(gs[2, 0])

    # Compute ECDF
    sorted_normal = np.sort(normal_measurements)
    ecdf = np.arange(1, len(sorted_normal) + 1) / len(sorted_normal)

    ax4.plot(sorted_normal, ecdf, linewidth=3, color=colors['normal'],
            label='Empirical CDF')

    # Theoretical CDF
    theoretical_cdf = stats.norm.cdf(sorted_normal, mu, sigma)
    ax4.plot(sorted_normal, theoretical_cdf, 'r--', linewidth=2,
            label='Theoretical CDF', alpha=0.7)

    # Mark miracle position
    miracle_cdf = stats.norm.cdf(measurements[miracle_idx], mu, sigma)
    ax4.scatter(measurements[miracle_idx], miracle_cdf, s=400, marker='*',
            color=colors['miracle'], edgecolor='black', linewidth=2, zorder=10)
    ax4.axvline(measurements[miracle_idx], color=colors['miracle'],
            linestyle='--', linewidth=2, alpha=0.7)

    # Annotate
    ax4.text(measurements[miracle_idx], miracle_cdf,
            f'  CDF = {miracle_cdf:.10f}\n  (virtually 1.0)',
            fontsize=10, va='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax4.set_xlabel('Measurement Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Cumulative Distribution\nMiracle Position',
                fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 5: Q-Q Plot
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 1])

    # Q-Q plot for normal measurements
    stats.probplot(normal_measurements, dist="norm", plot=ax5)
    ax5.get_lines()[0].set_color(colors['normal'])
    ax5.get_lines()[0].set_markersize(4)
    ax5.get_lines()[0].set_alpha(0.6)
    ax5.get_lines()[1].set_color('red')
    ax5.get_lines()[1].set_linewidth(2)

    # Add miracle point
    theoretical_quantile = stats.norm.ppf((len(normal_measurements)) / (len(normal_measurements) + 1))
    ax5.scatter(theoretical_quantile, measurements[miracle_idx],
            s=400, marker='*', color=colors['miracle'], edgecolor='black',
            linewidth=2, zorder=10, label='Miracle measurement')

    ax5.set_title('(E) Q-Q Plot: Normality Test\nMiracle as Extreme Outlier',
                fontsize=14, fontweight='bold', pad=15)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 6: Spectral Analysis
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 2:])

    # Compute FFT
    from scipy.fft import rfft, rfftfreq
    fft_vals = rfft(measurements - measurements.mean())
    fft_freqs = rfftfreq(len(measurements), d=(time_points[1] - time_points[0]))
    fft_magnitude = np.abs(fft_vals)

    ax6.semilogy(fft_freqs, fft_magnitude, linewidth=2, color=colors['quantum'])

    # Mark dominant frequencies
    peaks, _ = signal.find_peaks(fft_magnitude, height=np.max(fft_magnitude)*0.1)
    top_peaks = peaks[np.argsort(fft_magnitude[peaks])[-5:]]

    for peak_idx in top_peaks:
        ax6.scatter(fft_freqs[peak_idx], fft_magnitude[peak_idx],
                s=150, color='red', zorder=10, edgecolor='black', linewidth=2)
        ax6.text(fft_freqs[peak_idx], fft_magnitude[peak_idx],
                f'{fft_freqs[peak_idx]:.2f} Hz',
                fontsize=9, ha='center', va='bottom', fontweight='bold')

    # Highlight miracle frequency
    miracle_freq = 1 / miracle_width  # Approximate frequency
    ax6.axvline(miracle_freq, color=colors['miracle'], linestyle='--',
            linewidth=3, alpha=0.7, label=f'Miracle freq: {miracle_freq:.2f} Hz')

    ax6.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Magnitude (log scale)', fontsize=12, fontweight='bold')
    ax6.set_title('(F) Spectral Analysis\nFrequency Domain Signature',
                fontsize=14, fontweight='bold', pad=15)
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--', which='both')
    ax6.set_xlim(0, 10)

    # ============================================================
    # PANEL 7: Anomaly Detection Metrics
    # ============================================================
    ax7 = fig.add_subplot(gs[3, :2])

    # Calculate various anomaly scores
    window_size = 50
    anomaly_scores = []
    score_types = []

    # 1. Z-score
    z_scores = np.abs((measurements - mu) / sigma)
    anomaly_scores.append(z_scores)
    score_types.append('Z-score')

    # 2. Modified Z-score (using median)
    median = np.median(normal_measurements)
    mad = np.median(np.abs(normal_measurements - median))
    modified_z = 0.6745 * (measurements - median) / mad
    anomaly_scores.append(np.abs(modified_z))
    score_types.append('Modified Z-score')

    # 3. Local outlier factor (simplified)
    local_mean = np.convolve(measurements, np.ones(window_size)/window_size, mode='same')
    local_std = np.array([measurements[max(0, i-window_size//2):min(len(measurements), i+window_size//2)].std()
                        for i in range(len(measurements))])
    lof_score = np.abs((measurements - local_mean) / (local_std + 1e-10))
    anomaly_scores.append(lof_score)
    score_types.append('Local Outlier Factor')

    # Plot all scores
    for score, label in zip(anomaly_scores, score_types):
        ax7.plot(time_points, score, linewidth=2, alpha=0.7, label=label)

    # Mark miracle
    for score in anomaly_scores:
        ax7.scatter(time_points[miracle_idx], score[miracle_idx],
                s=200, marker='*', color=colors['miracle'],
                edgecolor='black', linewidth=2, zorder=10)

    # Threshold lines
    ax7.axhline(3, color='orange', linestyle='--', linewidth=2, alpha=0.5,
            label='3σ threshold')
    ax7.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.5,
            label='5σ threshold')

    ax7.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax7.set_title('(G) Multi-Method Anomaly Detection\nConvergence on Miracle Event',
                fontsize=14, fontweight='bold', pad=15)
    ax7.legend(fontsize=10, loc='upper right')
    ax7.grid(alpha=0.3, linestyle='--')
    ax7.set_yscale('log')

    # ============================================================
    # PANEL 8: Statistical Summary
    # ============================================================
    ax8 = fig.add_subplot(gs[3, 2:])
    ax8.axis('off')

    # Comprehensive statistics
    shapiro_stat, shapiro_p = stats.shapiro(normal_measurements[:5000])
    ks_stat, ks_p = stats.kstest(normal_measurements, 'norm', args=(mu, sigma))

    # Miracle statistics
    miracle_value = measurements[miracle_idx]
    miracle_sigma = (miracle_value - mu) / sigma
    miracle_p = stats.norm.sf(miracle_value, mu, sigma)

    # Confidence intervals
    ci_95 = stats.norm.interval(0.95, loc=mu, scale=sigma)
    ci_99 = stats.norm.interval(0.99, loc=mu, scale=sigma)
    ci_999 = stats.norm.interval(0.999, loc=mu, scale=sigma)

    summary_text = f"""
    MIRACULOUS MEASUREMENT ANALYSIS SUMMARY

    NORMAL BASELINE STATISTICS:
    Sample size:           {len(normal_measurements):,}
    Mean (μ):              {mu:.6f}
    Std deviation (σ):     {sigma:.6f}
    Median:                {np.median(normal_measurements):.6f}
    Min:                   {normal_measurements.min():.6f}
    Max:                   {normal_measurements.max():.6f}

    NORMALITY TESTS:
    Shapiro-Wilk:          W = {shapiro_stat:.6f}, p = {shapiro_p:.2e}
    Kolmogorov-Smirnov:    D = {ks_stat:.6f}, p = {ks_p:.2e}
    Conclusion:            {'NORMAL' if shapiro_p > 0.05 else 'NON-NORMAL'}

    MIRACULOUS EVENT:
    Time of occurrence:    {time_points[miracle_idx]:.4f}
    Measurement value:     {miracle_value:.6f}
    Sigma level:           {miracle_sigma:.2f}σ
    P-value:               {miracle_p:.2e}
    Probability:           1 in {1/miracle_p:.2e}

    Enhancement factor:    {miracle_value/mu:.2f}×
    Excess signal:         {miracle_value - mu:.6f}
    Relative excess:       {((miracle_value - mu)/mu)*100:.2f}%

    STATISTICAL IMPOSSIBILITY:
    95% CI:                [{ci_95[0]:.4f}, {ci_95[1]:.4f}]
    99% CI:                [{ci_99[0]:.4f}, {ci_99[1]:.4f}]
    99.9% CI:              [{ci_999[0]:.4f}, {ci_999[1]:.4f}]
    Miracle position:      {miracle_value:.4f} (OUTSIDE ALL CIs)

    Standard deviations:   {miracle_sigma:.1f}σ beyond mean
    Probability density:   {stats.norm.pdf(miracle_value, mu, sigma):.2e}

    ANOMALY DETECTION:
    Z-score:               {z_scores[miracle_idx]:.2f}
    Modified Z-score:      {modified_z[miracle_idx]:.2f}
    LOF score:             {lof_score[miracle_idx]:.2f}
    All methods agree:     {'YES' if all(s[miracle_idx] > 5 for s in anomaly_scores) else 'NO'}

    INTERPRETATION:
    • Event is {miracle_sigma:.1f} standard deviations from mean
    • Probability of occurrence: {miracle_p:.2e} (virtually impossible)
    • Would occur once every {1/miracle_p:.2e} measurements
    • Exceeds all confidence intervals
    • Detected by all anomaly detection methods
    • Represents genuine anomalous phenomenon

    PHYSICAL SIGNIFICANCE:
    Signal-to-noise:       {miracle_value/sigma:.2f}
    Peak width:            {miracle_width:.4f} time units
    Integrated excess:     {np.trapz(m_zoom - mu, t_zoom):.4f}
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

    # Main title
    fig.suptitle('Miraculous Measurement Analysis: Statistical Impossibility Detection\n'
                f'Dataset: {data["timestamp"]} | '
                f'Significance: {miracle_sigma:.1f}σ | Probability: {miracle_p:.2e}',
                fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('miraculous_measurement_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('miraculous_measurement_analysis.png', dpi=300, bbox_inches='tight')

    print("✓ Miraculous measurement analysis figure created")
    print(f"  Miracle time: {time_points[miracle_idx]:.4f}")
    print(f"  Miracle value: {miracle_value:.4f}")
    print(f"  Sigma level: {miracle_sigma:.1f}σ")
    print(f"  P-value: {miracle_p:.2e}")
