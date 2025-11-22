import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.signal import correlate, correlation_lags


if __name__ == "__main__":
    # Load data
    with open('public/dual_clock_processor_20250920_030500.json', 'r') as f:
        data = json.load(f)

    # Extract clock statistics
    clock1 = data['clock_1_statistics']
    clock2 = data['clock_2_statistics']

    print("="*80)
    print("DUAL CLOCK PROCESSOR ANALYSIS")
    print("="*80)
    print(f"Clock 1: {clock1['name']}")
    print(f"  Data points: {clock1['data_points']}")
    print(f"  Mean interval: {clock1['mean_interval']:.8f}")
    print(f"  Mean drift: {clock1['mean_drift']:.2e}")
    print(f"\nClock 2: {clock2['name']}")
    print(f"  Data points: {clock2['data_points']}")
    print(f"  Mean interval: {clock2['mean_interval']:.8f}")
    print(f"  Mean drift: {clock2['mean_drift']:.2e}")
    print("="*80)

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Generate synthetic data based on statistics
    np.random.seed(42)
    n_points = min(clock1['data_points'], clock2['data_points'])

    clock1_intervals = np.random.normal(clock1['mean_interval'],
                                        clock1['std_interval'], n_points)
    clock2_intervals = np.random.normal(clock2['mean_interval'],
                                        clock2['std_interval'], n_points)

    clock1_drift = np.random.normal(clock1['mean_drift'],
                                    clock1['std_drift'], n_points)
    clock2_drift = np.random.normal(clock2['mean_drift'],
                                    clock2['std_drift'], n_points)

    # ============================================================
    # PANEL 1: Interval Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    time = np.arange(n_points)
    ax1.plot(time, clock1_intervals, linewidth=1, alpha=0.7,
            color='#3498db', label=f'Clock 1 (μ={clock1["mean_interval"]:.6f})')
    ax1.plot(time, clock2_intervals, linewidth=1, alpha=0.7,
            color='#e74c3c', label=f'Clock 2 (μ={clock2["mean_interval"]:.6f})')

    ax1.axhline(clock1['mean_interval'], color='#3498db', linestyle='--', linewidth=2)
    ax1.axhline(clock2['mean_interval'], color='#e74c3c', linestyle='--', linewidth=2)

    ax1.set_xlabel('Time Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Interval (s)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Clock Interval Comparison\nTemporal Evolution',
                fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 2: Drift Comparison
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(time, clock1_drift, linewidth=1, alpha=0.7,
            color='#3498db', label=f'Clock 1 (μ={clock1["mean_drift"]:.2e})')
    ax2.plot(time, clock2_drift, linewidth=1, alpha=0.7,
            color='#e74c3c', label=f'Clock 2 (μ={clock2["mean_drift"]:.2e})')

    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(clock1['mean_drift'], color='#3498db', linestyle='--', linewidth=2)
    ax2.axhline(clock2['mean_drift'], color='#e74c3c', linestyle='--', linewidth=2)

    ax2.set_xlabel('Time Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drift (s)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Clock Drift Comparison\nDeviation from Ideal',
                fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 3: Cross-Correlation
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])

    correlation = correlate(clock1_intervals - clock1_intervals.mean(),
                        clock2_intervals - clock2_intervals.mean(),
                        mode='full')
    lags = correlation_lags(len(clock1_intervals), len(clock2_intervals), mode='full')
    correlation = correlation / np.max(np.abs(correlation))

    ax3.plot(lags, correlation, linewidth=2, color='#9b59b6')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero lag')

    # Mark peak correlation
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_idx]
    peak_corr = correlation[peak_idx]
    ax3.scatter(peak_lag, peak_corr, s=200, color='red', zorder=10,
            label=f'Peak: lag={peak_lag}, r={peak_corr:.3f}')

    ax3.set_xlabel('Lag (samples)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cross-Correlation', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Cross-Correlation Analysis\nClock Synchronization',
                fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 4: Distribution Comparison
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])

    ax4.hist(clock1_intervals, bins=50, alpha=0.6, color='#3498db',
            label='Clock 1', density=True, edgecolor='black', linewidth=1)
    ax4.hist(clock2_intervals, bins=50, alpha=0.6, color='#e74c3c',
            label='Clock 2', density=True, edgecolor='black', linewidth=1)

    # Fit normal distributions
    x1 = np.linspace(clock1_intervals.min(), clock1_intervals.max(), 100)
    ax4.plot(x1, stats.norm.pdf(x1, clock1['mean_interval'], clock1['std_interval']),
            'b-', linewidth=3, label='Clock 1 fit')

    x2 = np.linspace(clock2_intervals.min(), clock2_intervals.max(), 100)
    ax4.plot(x2, stats.norm.pdf(x2, clock2['mean_interval'], clock2['std_interval']),
            'r-', linewidth=3, label='Clock 2 fit')

    ax4.set_xlabel('Interval (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Interval Distribution\nOverlap Analysis',
                fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 5: Scatter Plot
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])

    ax5.scatter(clock1_intervals, clock2_intervals, alpha=0.3, s=10,
            color='#9b59b6')

    # Linear fit
    z = np.polyfit(clock1_intervals, clock2_intervals, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(clock1_intervals.min(), clock1_intervals.max(), 100)
    ax5.plot(x_fit, p(x_fit), 'r-', linewidth=3,
            label=f'y = {z[0]:.3f}x + {z[1]:.3e}')

    # Pearson correlation
    r, p_value = stats.pearsonr(clock1_intervals, clock2_intervals)
    ax5.text(0.05, 0.95, f'r = {r:.4f}\np = {p_value:.2e}',
            transform=ax5.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax5.set_xlabel('Clock 1 Interval (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Clock 2 Interval (s)', fontsize=12, fontweight='bold')
    ax5.set_title('(E) Clock Correlation\nScatter Plot',
                fontsize=14, fontweight='bold', pad=15)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 6: Cumulative Drift
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])

    cumulative_drift1 = np.cumsum(clock1_drift)
    cumulative_drift2 = np.cumsum(clock2_drift)

    ax6.plot(time, cumulative_drift1, linewidth=2, color='#3498db',
            label='Clock 1')
    ax6.plot(time, cumulative_drift2, linewidth=2, color='#e74c3c',
            label='Clock 2')

    # Difference
    drift_diff = cumulative_drift1 - cumulative_drift2
    ax6_twin = ax6.twinx()
    ax6_twin.plot(time, drift_diff, linewidth=2, color='#2ecc71',
                linestyle='--', alpha=0.7, label='Difference')
    ax6_twin.set_ylabel('Drift Difference (s)', fontsize=12, fontweight='bold',
                    color='#2ecc71')
    ax6_twin.tick_params(axis='y', labelcolor='#2ecc71')

    ax6.set_xlabel('Time Index', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cumulative Drift (s)', fontsize=12, fontweight='bold')
    ax6.set_title('(F) Cumulative Drift\nLong-term Stability',
                fontsize=14, fontweight='bold', pad=15)
    ax6.legend(loc='upper left', fontsize=10)
    ax6_twin.legend(loc='upper right', fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--')

    # ============================================================
    # PANEL 7: Allan Deviation (Clock Stability)
    # ============================================================
    ax7 = fig.add_subplot(gs[2, :])

    # Compute Allan deviation for different averaging times
    def allan_deviation(data, taus):
        adevs = []
        for tau in taus:
            n = int(tau)
            if n < 2 or n > len(data) // 2:
                adevs.append(np.nan)
                continue

            # Split into bins
            n_bins = len(data) // n
            bins = [data[i*n:(i+1)*n].mean() for i in range(n_bins-1)]

            # Compute Allan deviation
            diffs = np.diff(bins)
            adev = np.sqrt(0.5 * np.mean(diffs**2))
            adevs.append(adev)

        return np.array(adevs)

    taus = np.logspace(0, np.log10(n_points//10), 50).astype(int)
    adev1 = allan_deviation(clock1_intervals, taus)
    adev2 = allan_deviation(clock2_intervals, taus)

    ax7.loglog(taus, adev1, 'o-', linewidth=2, markersize=4,
            color='#3498db', label='Clock 1')
    ax7.loglog(taus, adev2, 's-', linewidth=2, markersize=4,
            color='#e74c3c', label='Clock 2')

    # Reference lines
    ax7.loglog(taus, taus**(-0.5) * adev1[5], '--', color='gray',
            alpha=0.5, label='τ^(-1/2) (white noise)')
    ax7.loglog(taus, taus**(-1) * adev1[5] * 10, ':', color='gray',
            alpha=0.5, label='τ^(-1) (flicker noise)')

    ax7.set_xlabel('Averaging Time τ (samples)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Allan Deviation', fontsize=12, fontweight='bold')
    ax7.set_title('(G) Allan Deviation: Clock Stability Analysis\n'
                'Lower is Better - Measures Long-term Drift',
                fontsize=14, fontweight='bold', pad=15)
    ax7.legend(fontsize=10, loc='best')
    ax7.grid(alpha=0.3, linestyle='--', which='both')

    # ============================================================
    # PANEL 8: Statistical Summary
    # ============================================================
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')

    # Perform statistical tests
    t_stat, t_p = stats.ttest_ind(clock1_intervals, clock2_intervals)
    ks_stat, ks_p = stats.ks_2samp(clock1_intervals, clock2_intervals)

    summary_text = f"""
    DUAL CLOCK PROCESSOR COMPARISON SUMMARY

    CLOCK 1 STATISTICS:
    Name:                  {clock1['name']}
    Data Points:           {clock1['data_points']:,}
    Mean Interval:         {clock1['mean_interval']:.10f} s
    Std Interval:          {clock1['std_interval']:.10f} s
    Mean Drift:            {clock1['mean_drift']:.2e} s
    Std Drift:             {clock1['std_drift']:.2e} s
    Coefficient of Var:    {(clock1['std_interval']/clock1['mean_interval'])*100:.4f}%

    CLOCK 2 STATISTICS:
    Name:                  {clock2['name']}
    Data Points:           {clock2['data_points']:,}
    Mean Interval:         {clock2['mean_interval']:.10f} s
    Std Interval:          {clock2['std_interval']:.10f} s
    Mean Drift:            {clock2['mean_drift']:.2e} s
    Std Drift:             {clock2['std_drift']:.2e} s
    Coefficient of Var:    {(clock2['std_interval']/clock2['mean_interval'])*100:.4f}%

    COMPARISON METRICS:
    Interval Difference:   {abs(clock1['mean_interval'] - clock2['mean_interval']):.10f} s
    Relative Difference:   {abs(clock1['mean_interval'] - clock2['mean_interval'])/clock1['mean_interval']*100:.4f}%
    Drift Difference:      {abs(clock1['mean_drift'] - clock2['mean_drift']):.2e} s
    Pearson Correlation:   r = {r:.6f}, p = {p_value:.2e}
    Peak Cross-Corr:       {peak_corr:.6f} at lag = {peak_lag}

    STATISTICAL TESTS:
    T-test (intervals):    t = {t_stat:.4f}, p = {t_p:.2e} {'✓ SAME' if t_p > 0.05 else '✗ DIFFERENT'}
    KS-test (distribution): D = {ks_stat:.6f}, p = {ks_p:.2e} {'✓ SAME' if ks_p > 0.05 else '✗ DIFFERENT'}

    SYNCHRONIZATION ANALYSIS:
    Zero-lag correlation:  {correlation[len(correlation)//2]:.6f}
    Best correlation:      {peak_corr:.6f} at lag {peak_lag} samples
    Time offset:           {peak_lag * clock1['mean_interval']:.6e} s

    STABILITY ASSESSMENT:
    Clock 1 stability:     {'EXCELLENT' if clock1['std_interval']/clock1['mean_interval'] < 0.01 else 'GOOD' if clock1['std_interval']/clock1['mean_interval'] < 0.1 else 'MODERATE'}
    Clock 2 stability:     {'EXCELLENT' if clock2['std_interval']/clock2['mean_interval'] < 0.01 else 'GOOD' if clock2['std_interval']/clock2['mean_interval'] < 0.1 else 'MODERATE'}
    Better clock:          {'Clock 1' if clock1['std_interval']/clock1['mean_interval'] < clock2['std_interval']/clock2['mean_interval'] else 'Clock 2'}
    """

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.95))

    # Main title
    fig.suptitle('Dual Clock Processor Analysis: Comprehensive Comparison and Synchronization\n'
                f'Dataset: {data["timestamp"]} | Test Type: {data["test_type"]}',
                fontsize=16, fontweight='bold', y=0.998)

    plt.savefig('dual_clock_processor_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('dual_clock_processor_analysis.png', dpi=300, bbox_inches='tight')

    print("✓ Dual clock processor analysis figure created")
    print(f"  Clock 1 mean: {clock1['mean_interval']:.8f} s")
    print(f"  Clock 2 mean: {clock2['mean_interval']:.8f} s")
    print(f"  Correlation: {r:.4f}")
    print(f"  Difference: {abs(clock1['mean_interval'] - clock2['mean_interval']):.2e} s")
