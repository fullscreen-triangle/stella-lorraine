# Script 1: Tetration Growth and Comparison to Known Large Numbers
# File: tetration_growth_analysis.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns


if __name__ == "__main__":
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    def tetration(n, t):
        """Compute n↑↑t (tetration) safely using logarithms"""
        if t == 0:
            return 1
        if t == 1:
            return n

        # For t >= 2, we work with log10 to avoid overflow
        # n↑↑t = n^(n↑↑(t-1))
        # log10(n↑↑t) = (n↑↑(t-1)) * log10(n)

        result_log = n  # Start with n↑↑2 = n^n
        for i in range(t - 2):
            # At each step: log10(n^x) = x * log10(n)
            result_log = (10**result_log) * np.log10(n) if result_log < 10 else result_log * np.log10(n)
            if result_log > 1e10:  # Cap to prevent overflow
                return result_log

        return result_log

    def log_tower_height(n, t):
        """Compute the 'height' of log tower needed to reduce n↑↑t to manageable size"""
        if t <= 1:
            return 0
        return t - 1

    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Tetration growth for different bases
    ax1 = fig.add_subplot(gs[0, :2])
    t_values = np.arange(0, 7)
    bases = [2, 3, 4, 10]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(bases)))

    for idx, n in enumerate(bases):
        log_values = []
        for t in t_values:
            if t <= 1:
                log_values.append(np.log10(max(tetration(n, t), 1)))
            else:
                # For t >= 2, tetration returns log10 already
                val = tetration(n, t)
                log_values.append(val if val > 1 else np.log10(max(n**n, 1)))

        ax1.plot(t_values, log_values, 'o-', label=f'n={n}',
                color=colors[idx], linewidth=2, markersize=8)

    ax1.set_xlabel('Categorical Depth (t)', fontweight='bold')
    ax1.set_ylabel('log₁₀(C(t))', fontweight='bold')
    ax1.set_title('A. Tetration Growth: C(t) = n↑↑t', fontweight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    # Panel 2: Comparison with known large numbers
    ax2 = fig.add_subplot(gs[0, 2])

    # Define known large numbers (as log10 of log10 for visualization)
    large_numbers = {
        'Googol\n(10¹⁰⁰)': 100,
        'Googolplex\n(10^googol)': 1e100,
        "Graham's\nNumber": 1e10,  # Approximate
        'TREE(3)': 1e15,  # Very approximate
        '2↑↑5': tetration(2, 5),
        '2↑↑6': tetration(2, 6),
    }

    # Take log10 twice for visualization
    names = list(large_numbers.keys())
    values = [np.log10(np.log10(v)) if v > 10 else np.log10(v) for v in large_numbers.values()]

    bars = ax2.barh(names, values, color=plt.cm.plasma(np.linspace(0.2, 0.9, len(names))))
    ax2.set_xlabel('log₁₀(log₁₀(N))', fontweight='bold')
    ax2.set_title('B. Comparison with\nKnown Large Numbers', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax2.text(val + 0.1, i, f'{val:.1f}', va='center', fontsize=8)

    # Panel 3: Recursive structure visualization
    ax3 = fig.add_subplot(gs[1, :])

    # Show the recursive structure C(t+1) = n^C(t)
    t_max = 5
    n = 2
    positions = []
    sizes = []
    labels = []

    for t in range(t_max + 1):
        if t == 0:
            val = 1
        elif t == 1:
            val = n
        else:
            val = tetration(n, t)

        positions.append(t)
        sizes.append(np.log10(val) if val > 1 else 1)

        if t <= 2:
            labels.append(f'C({t})={int(val)}')
        else:
            labels.append(f'C({t})=2^C({t-1})')

    ax3.scatter(positions, sizes, s=500, alpha=0.6, c=plt.cm.coolwarm(np.linspace(0, 1, len(positions))))

    # Draw arrows showing recursion
    for i in range(len(positions) - 1):
        ax3.annotate('', xy=(positions[i+1], sizes[i+1]), xytext=(positions[i], sizes[i]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))

    # Add labels
    for i, (pos, size, label) in enumerate(zip(positions, sizes, labels)):
        ax3.text(pos, size + 0.5, label, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_xlabel('Categorical Depth (t)', fontweight='bold')
    ax3.set_ylabel('log₁₀(C(t))', fontweight='bold')
    ax3.set_title('C. Recursive Structure: C(t+1) = n^C(t)', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(-0.5, t_max + 0.5)

    # Panel 4: Observer-dependent categorical horizon
    ax4 = fig.add_subplot(gs[2, 0])

    # Simulate observer measurements
    np.random.seed(42)
    n_measurements = 20
    observed = []
    unobserved = []

    C_total = 100  # Total categories (simplified)
    observed_count = 0

    for m in range(n_measurements):
        # Each measurement observes some categories
        new_observed = np.random.randint(1, 5)
        observed_count += new_observed
        observed.append(observed_count)
        unobserved.append(C_total - observed_count)

    measurements = np.arange(n_measurements)
    ax4.fill_between(measurements, 0, observed, alpha=0.6, label='Observed', color='steelblue')
    ax4.fill_between(measurements, observed, C_total, alpha=0.6, label='Unobserved', color='coral')
    ax4.plot(measurements, observed, 'o-', color='darkblue', linewidth=2, markersize=4)

    ax4.set_xlabel('Number of Measurements', fontweight='bold')
    ax4.set_ylabel('Categorical Count', fontweight='bold')
    ax4.set_title('D. Observer-Dependent\nCategorical Horizon', fontweight='bold', pad=20)
    ax4.legend(loc='right', frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Panel 5: Dark matter ratio prediction
    ax5 = fig.add_subplot(gs[2, 1])

    # Theoretical prediction: R_DM = C(t) / |C_t^act|
    t_values_dm = np.arange(1, 6)
    n = 2

    ratios_simple = []
    for t in t_values_dm:
        C_t = tetration(n, t)
        if t <= 2:
            C_t_act = 1  # Simplified
            ratio = C_t / C_t_act
        else:
            # For higher t, use logarithmic approximation
            ratio = 10**(tetration(n, t) - 2)  # Simplified
        ratios_simple.append(min(ratio, 1e6))  # Cap for visualization

    # Observed ratio
    observed_ratio = 5.4

    ax5.semilogy(t_values_dm, ratios_simple, 'o-', linewidth=2, markersize=8,
                label='Predicted Ratio', color='darkgreen')
    ax5.axhline(y=observed_ratio, color='red', linestyle='--', linewidth=2,
            label=f'Observed Ratio ≈ {observed_ratio}')

    ax5.set_xlabel('Categorical Depth (t)', fontweight='bold')
    ax5.set_ylabel('Dark Matter Ratio', fontweight='bold')
    ax5.set_title('E. Dark Matter Ratio:\nC(t)/|C_t^act|', fontweight='bold', pad=20)
    ax5.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Panel 6: Entropy as shortest path
    ax6 = fig.add_subplot(gs[2, 2])

    # Simulate entropy growth
    time_steps = np.linspace(0, 10, 100)
    # S ∝ ln(C(t))
    entropy = np.log(1 + time_steps**2)  # Simplified model

    ax6.plot(time_steps, entropy, linewidth=3, color='purple')
    ax6.fill_between(time_steps, 0, entropy, alpha=0.3, color='purple')

    ax6.set_xlabel('Time', fontweight='bold')
    ax6.set_ylabel('Entropy S', fontweight='bold')
    ax6.set_title('F. Entropy Growth:\nS ∝ ln(C(t))', fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Add arrow indicating direction
    ax6.annotate('Arrow of Time', xy=(7, entropy[70]), xytext=(5, entropy[70] + 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, fontweight='bold', color='red')

    plt.suptitle('Categorical Dynamics: Tetration Growth and Physical Predictions',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('tetration_analysis_panel.png', dpi=300, bbox_inches='tight')
    plt.savefig('tetration_analysis_panel.pdf', dpi=300, bbox_inches='tight')
    print("Saved: tetration_analysis_panel.png and .pdf")
    plt.show()
