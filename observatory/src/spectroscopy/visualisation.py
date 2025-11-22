"""
Publication-Quality Visualization of Experimental Spectral Data
================================================================
This script generates multi-panel plots for statistical analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Load data
data = {
    "descriptive_statistics": {
        "real_spectra_stats": {
            "intensity_mean": 6.4840613331629005,
            "intensity_std": 80.35648394152072,
            "intensity_median": -1.347589,
            "intensity_iqr": 2.5873573515625,
            "intensity_skewness": 22.665732067472838,
            "intensity_kurtosis": 570.7297075038166,
            "wavelength_range": [-47.748917, 840.0],
            "n_spectra": 7,
            "total_data_points": 59981
        },
        "virtual_spectra_stats": {
            "intensity_mean": 0.10851885417212011,
            "intensity_std": 0.09561933524805676,
            "intensity_median": 0.10078870414687893,
            "intensity_iqr": 0.0992705121368986,
            "intensity_skewness": 4.511368449756141,
            "intensity_kurtosis": 30.898999167613532,
            "wavelength_range": [200.0, 800.0],
            "n_spectra": 10,
            "total_data_points": 5000
        },
        "comparative_stats": {
            "intensity_mean_ratio": 0.016736247329601526,
            "intensity_std_ratio": 0.0011899392626193493,
            "wavelength_overlap": 100.0
        }
    },
    "hypothesis_testing": {
        "correlation_vs_zero": {
            "test": "One-sample t-test (correlation > 0)",
            "statistic": 2.1420802237094096,
            "p_value": 0.03571874644398494,
            "significant": "True",
            "interpretation": "Correlations significantly greater than 0"
        },
        "peak_f1_vs_random": {
            "test": "One-sample t-test (peak F1 > random)",
            "statistic": -57.29112553544405,
            "p_value": 6.395670415104749e-60,
            "significant": "True",
            "interpretation": "Peak detection significantly better than random"
        },
        "rmse_vs_unity": {
            "test": "One-sample t-test (RMSE < 1)",
            "statistic": -17.46409551818724,
            "p_value": 5.015262868230228e-27,
            "significant": "True",
            "interpretation": "RMSE significantly less than unity"
        },
        "correlation_normality": {
            "test": "Shapiro-Wilk normality test",
            "statistic": 0.9682663761410345,
            "p_value": 0.07270772688977212,
            "is_normal": "True",
            "interpretation": "Correlations are normally distributed"
        },
        "led_anova": {
            "test": "One-way ANOVA (LED responses)",
            "statistic": 1.6652938570541551,
            "p_value": 0.24247350837939344,
            "significant": "False",
            "interpretation": "No significant differences between LED responses"
        }
    },
    "effect_sizes": {
        "correlation_cohens_d": {
            "value": 0.25602755668669047,
            "interpretation": "Small effect"
        },
        "peak_f1_cohens_d": {
            "value": -6.847599230093096,
            "interpretation": "Large effect"
        },
        "mean_r_squared": {
            "value": -40.45169976828709,
            "interpretation": "Very weak relationship"
        }
    },
    "confidence_intervals": {
        "correlation": {
            "mean": 0.01655477460939986,
            "ci_lower": 0.0011371112839967026,
            "ci_upper": 0.03197243793480302,
            "sem": 0.007728363497391426
        },
        "peak_f1": {
            "mean": 0.05529472564640519,
            "ci_lower": 0.04572915990728496,
            "ci_upper": 0.06486029138552542,
            "sem": 0.004794900986604707
        },
        "rmse": {
            "mean": 0.4369203297134734,
            "ci_lower": 0.3725990366119569,
            "ci_upper": 0.5012416228149899,
            "sem": 0.03224213184703045
        }
    },
    "power_analysis": {
        "sample_size": 70,
        "effect_size": 0.25602755668669047,
        "alpha": 0.05,
        "observed_power": 0.5582726545225962,
        "power_interpretation": "Low power - consider increasing sample size",
        "recommended_sample_size": 120
    },
    "multivariate_analysis": {
        "pca": {
            "explained_variance_ratio": [0.9999811686508913, 1.566617142974857e-05, 2.318919737213518e-06],
            "cumulative_variance": [0.9999811686508913, 0.999996834822321, 0.9999991537420583],
            "components": [
                [0.00010483733298973985, 0.0020658178069060673, -0.0037895684878343537, 0.9999881863779571,
                 -0.0022333116940142154],
                [-0.033794855017016526, -0.6444376778751377, -0.11033607288476552, 0.002604887458826477,
                 0.7558949349552319],
                [-0.2726946959100839, -0.05942905659754463, 0.9571755535952094, 0.003950305872826229,
                 0.07684493957085191]
            ],
            "n_components": 3
        },
        "clustering": {
            "n_clusters": 3,
            "cluster_labels": [2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0,
                               2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "cluster_centers": [
                [9.22272859854528, -0.06918657713031119, 0.10556274333851151],
                [-157.54504094558706, 0.021338846547134062, -0.03697764848091578],
                [35.43007971503638, 0.031782811647826335, -0.047360028853835376]
            ],
            "inertia": 1091.4080623700722
        }
    }
}


def create_figure_1_descriptive_stats(data):
    """
    Figure 1: Descriptive Statistics Comparison
    4-panel plot showing real vs virtual spectra statistics
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    real_stats = data['descriptive_statistics']['real_spectra_stats']
    virtual_stats = data['descriptive_statistics']['virtual_spectra_stats']

    # Panel A: Intensity Statistics Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Mean', 'Std Dev', 'Median', 'IQR']
    real_values = [real_stats['intensity_mean'], real_stats['intensity_std'],
                   real_stats['intensity_median'], real_stats['intensity_iqr']]
    virtual_values = [virtual_stats['intensity_mean'], virtual_stats['intensity_std'],
                      virtual_stats['intensity_median'], virtual_stats['intensity_iqr']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, real_values, width, label='Real Spectra',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width / 2, virtual_values, width, label='Virtual Spectra',
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax1.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax1.set_title('A) Intensity Statistics', fontsize=12, fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Panel B: Distribution Shape (Skewness & Kurtosis)
    ax2 = fig.add_subplot(gs[0, 1])
    shape_categories = ['Skewness', 'Kurtosis']
    real_shape = [real_stats['intensity_skewness'], real_stats['intensity_kurtosis']]
    virtual_shape = [virtual_stats['intensity_skewness'], virtual_stats['intensity_kurtosis']]

    x2 = np.arange(len(shape_categories))
    bars3 = ax2.bar(x2 - width / 2, real_shape, width, label='Real Spectra',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x2 + width / 2, virtual_shape, width, label='Virtual Spectra',
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(shape_categories)
    ax2.legend(frameon=True, fancybox=False, edgecolor='black')
    ax2.set_title('B) Distribution Shape', fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_yscale('log')

    # Panel C: Sample Size Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    sample_categories = ['N Spectra', 'Total Points']
    real_samples = [real_stats['n_spectra'], real_stats['total_data_points']]
    virtual_samples = [virtual_stats['n_spectra'], virtual_stats['total_data_points']]

    x3 = np.arange(len(sample_categories))
    bars5 = ax3.bar(x3 - width / 2, real_samples, width, label='Real Spectra',
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars6 = ax3.bar(x3 + width / 2, virtual_samples, width, label='Virtual Spectra',
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(sample_categories)
    ax3.legend(frameon=True, fancybox=False, edgecolor='black')
    ax3.set_title('C) Sample Sizes', fontsize=12, fontweight='bold', loc='left')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.set_yscale('log')

    # Panel D: Wavelength Range
    ax4 = fig.add_subplot(gs[1, 1])
    real_range = real_stats['wavelength_range']
    virtual_range = virtual_stats['wavelength_range']

    ax4.barh(0, real_range[1] - real_range[0], left=real_range[0], height=0.3,
             label='Real Spectra', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.barh(1, virtual_range[1] - virtual_range[0], left=virtual_range[0], height=0.3,
             label='Virtual Spectra', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)

    ax4.set_xlabel('Wavelength (nm)', fontsize=11, fontweight='bold')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Real', 'Virtual'])
    ax4.set_title('D) Wavelength Coverage', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    plt.savefig('figure1_descriptive_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_descriptive_statistics.pdf', bbox_inches='tight')
    print("✓ Figure 1 saved: figure1_descriptive_statistics.png/pdf")
    return fig


def create_figure_2_hypothesis_testing(data):
    """
    Figure 2: Hypothesis Testing Results
    4-panel plot showing statistical test results
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    hyp_tests = data['hypothesis_testing']

    # Panel A: P-values for all tests
    ax1 = fig.add_subplot(gs[0, 0])
    test_names = ['Correlation\nvs Zero', 'Peak F1\nvs Random', 'RMSE\nvs Unity',
                  'Correlation\nNormality', 'LED\nANOVA']
    p_values = [
        hyp_tests['correlation_vs_zero']['p_value'],
        hyp_tests['peak_f1_vs_random']['p_value'],
        hyp_tests['rmse_vs_unity']['p_value'],
        hyp_tests['correlation_normality']['p_value'],
        hyp_tests['led_anova']['p_value']
    ]

    # Use log scale for p-values
    p_values_log = [-np.log10(p) for p in p_values]
    colors = ['#06A77D' if p < 0.05 else '#D62839' for p in p_values]

    bars = ax1.bar(range(len(test_names)), p_values_log, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.5,
                label='α = 0.05', zorder=10)

    ax1.set_ylabel('-log₁₀(p-value)', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, fontsize=9)
    ax1.set_title('A) Statistical Significance', fontsize=12, fontweight='bold', loc='left')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Panel B: Test Statistics
    ax2 = fig.add_subplot(gs[0, 1])
    stat_names = ['Correlation\nvs Zero', 'Peak F1\nvs Random', 'RMSE\nvs Unity',
                  'Normality\n(Shapiro)', 'LED\nANOVA']
    statistics = [
        hyp_tests['correlation_vs_zero']['statistic'],
        hyp_tests['peak_f1_vs_random']['statistic'],
        hyp_tests['rmse_vs_unity']['statistic'],
        hyp_tests['correlation_normality']['statistic'],
        hyp_tests['led_anova']['statistic']
    ]

    colors2 = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars2 = ax2.bar(range(len(stat_names)), statistics, color=colors2,
                    alpha=0.8, edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('Test Statistic', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(stat_names)))
    ax2.set_xticklabels(stat_names, fontsize=9)
    ax2.set_title('B) Test Statistics', fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)

    # Panel C: Significance Summary
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    # Create table
    table_data = []
    for name, key in [('Correlation vs Zero', 'correlation_vs_zero'),
                      ('Peak F1 vs Random', 'peak_f1_vs_random'),
                      ('RMSE vs Unity', 'rmse_vs_unity'),
                      ('Normality Test', 'correlation_normality'),
                      ('LED ANOVA', 'led_anova')]:
        test = hyp_tests[key]
        sig = '✓' if test.get('significant', test.get('is_normal')) == 'True' else '✗'
        p_val = f"{test['p_value']:.4f}" if test['p_value'] >= 0.0001 else f"{test['p_value']:.2e}"
        table_data.append([name, sig, p_val])

    table = ax3.table(cellText=table_data,
                      colLabels=['Test', 'Sig.', 'p-value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.5, 0.15, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style cells
    for i in range(1, 6):
        for j in range(3):
            if j == 1:  # Significance column
                if table_data[i - 1][1] == '✓':
                    table[(i, j)].set_facecolor('#D5F4E6')
                else:
                    table[(i, j)].set_facecolor('#FADBD8')
            else:
                table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
            table[(i, j)].set_edgecolor('black')

    ax3.set_title('C) Significance Summary', fontsize=12, fontweight='bold',
                  loc='left', pad=20)

    # Panel D: Effect Sizes
    ax4 = fig.add_subplot(gs[1, 1])
    effect_data = data['effect_sizes']
    effect_names = ['Correlation\nCohen\'s d', 'Peak F1\nCohen\'s d', 'Mean\nR²']
    effect_values = [
        effect_data['correlation_cohens_d']['value'],
        effect_data['peak_f1_cohens_d']['value'],
        effect_data['mean_r_squared']['value']
    ]

    colors3 = ['#3498DB', '#E74C3C', '#F39C12']
    bars3 = ax4.barh(range(len(effect_names)), effect_values, color=colors3,
                     alpha=0.8, edgecolor='black', linewidth=1.2)

    ax4.set_xlabel('Effect Size', fontsize=11, fontweight='bold')
    ax4.set_yticks(range(len(effect_names)))
    ax4.set_yticklabels(effect_names)
    ax4.set_title('D) Effect Sizes', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.8)

    plt.savefig('figure2_hypothesis_testing.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_hypothesis_testing.pdf', bbox_inches='tight')
    print("✓ Figure 2 saved: figure2_hypothesis_testing.png/pdf")
    return fig


def create_figure_3_confidence_intervals(data):
    """
    Figure 3: Confidence Intervals and Uncertainty
    3-panel plot showing confidence intervals for key metrics
    """
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    ci_data = data['confidence_intervals']

    metrics = ['correlation', 'peak_f1', 'rmse']
    titles = ['A) Correlation Coefficient', 'B) Peak F1 Score', 'C) RMSE']
    colors = ['#3498DB', '#2ECC71', '#E74C3C']

    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = fig.add_subplot(gs[0, idx])

        metric_data = ci_data[metric]
        mean = metric_data['mean']
        ci_lower = metric_data['ci_lower']
        ci_upper = metric_data['ci_upper']
        sem = metric_data['sem']

        # Plot mean with error bars
        ax.errorbar([0], [mean], yerr=[[mean - ci_lower], [ci_upper - mean]],
                    fmt='o', markersize=12, color=color, markeredgecolor='black',
                    markeredgewidth=1.5, capsize=10, capthick=2, elinewidth=2,
                    label='95% CI')

        # Add horizontal lines for CI bounds
        ax.axhline(ci_lower, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(ci_upper, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(mean, color=color, linestyle='-', alpha=0.7, linewidth=2)

        # Shade CI region
        ax.axhspan(ci_lower, ci_upper, alpha=0.2, color=color)

        # Add reference line if applicable
        if metric == 'correlation':
            ax.axhline(0, color='black', linestyle=':', linewidth=1.5, label='Null (r=0)')
        elif metric == 'rmse':
            ax.axhline(1, color='black', linestyle=':', linewidth=1.5, label='Unity')

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add text annotation
        text_str = f'Mean: {mean:.4f}\nSEM: {sem:.4f}\nCI: [{ci_lower:.4f}, {ci_upper:.4f}]'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig('figure3_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_confidence_intervals.pdf', bbox_inches='tight')
    print("✓ Figure 3 saved: figure3_confidence_intervals.png/pdf")
    return fig


def create_figure_4_power_analysis(data):
    """
    Figure 4: Power Analysis
    2-panel plot showing power analysis results
    """
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    power_data = data['power_analysis']

    # Panel A: Current vs Recommended Sample Size
    ax1 = fig.add_subplot(gs[0, 0])

    categories = ['Current\nSample Size', 'Recommended\nSample Size']
    values = [power_data['sample_size'], power_data['recommended_sample_size']]
    colors = ['#E74C3C', '#2ECC71']

    bars = ax1.bar(range(len(categories)), values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'n = {int(val)}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Sample Size (n)', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_title('A) Sample Size Comparison', fontsize=12, fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, max(values) * 1.15)

    # Panel B: Power Analysis Summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Create summary box
    summary_text = f"""
    POWER ANALYSIS SUMMARY
    {'=' * 40}

    Current Sample Size:     {power_data['sample_size']}
    Effect Size (Cohen's d): {power_data['effect_size']:.4f}
    Significance Level (α):  {power_data['alpha']}

    Observed Power:          {power_data['observed_power']:.4f}
    Power Interpretation:    {power_data['power_interpretation']}

    Recommended Sample Size: {power_data['recommended_sample_size']}
    Additional Samples Needed: {power_data['recommended_sample_size'] - power_data['sample_size']}

    {'=' * 40}
    Note: Power analysis assumes α = 0.05 and 
    desired power = 0.80 (80%)
    """

    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ECF0F1',
                       edgecolor='black', linewidth=1.5, alpha=0.9))

    ax2.set_title('B) Power Analysis Details', fontsize=12, fontweight='bold',
                  loc='left', pad=20)

    # Add power gauge
    ax2_inset = fig.add_axes([0.65, 0.15, 0.25, 0.25])

    # Create gauge
    theta = np.linspace(0, np.pi
                        , 100)
    r = 1

    # Background arc
    ax2_inset.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=8, alpha=0.2)

    # Power arc
    power_theta = theta[0:int(power_data['observed_power'] * len(theta))]
    power_color = '#E74C3C' if power_data['observed_power'] < 0.8 else '#2ECC71'
    ax2_inset.plot(r * np.cos(power_theta), r * np.sin(power_theta),
                   color=power_color, linewidth=8)

    # Target power line (0.8)
    target_theta = np.pi * 0.8
    ax2_inset.plot([0, r * np.cos(target_theta)], [0, r * np.sin(target_theta)],
                   'k--', linewidth=2, alpha=0.5)

    # Current power indicator
    current_theta = np.pi * power_data['observed_power']
    ax2_inset.plot([0, r * np.cos(current_theta)], [0, r * np.sin(current_theta)],
                   color=power_color, linewidth=3, marker='o', markersize=10,
                   markerfacecolor=power_color, markeredgecolor='black', markeredgewidth=1.5)

    ax2_inset.set_xlim(-1.2, 1.2)
    ax2_inset.set_ylim(-0.2, 1.2)
    ax2_inset.set_aspect('equal')
    ax2_inset.axis('off')

    # Add labels
    ax2_inset.text(0, -0.15, f'{power_data["observed_power"]:.2%}',
                   ha='center', fontsize=14, fontweight='bold')
    ax2_inset.text(0, -0.35, 'Observed Power', ha='center', fontsize=10)
    ax2_inset.text(-1, 0, '0%', ha='right', fontsize=8)
    ax2_inset.text(1, 0, '100%', ha='left', fontsize=8)
    ax2_inset.text(r * np.cos(target_theta) - 0.15, r * np.sin(target_theta) + 0.1,
                   'Target\n(80%)', ha='center', fontsize=8, style='italic')

    plt.savefig('figure4_power_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure4_power_analysis.pdf', bbox_inches='tight')
    print("✓ Figure 4 saved: figure4_power_analysis.png/pdf")
    return fig


def create_figure_5_multivariate(data):
    """
    Figure 5: Multivariate Analysis
    4-panel plot showing PCA and clustering results
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    pca_data = data['multivariate_analysis']['pca']
    cluster_data = data['multivariate_analysis']['clustering']

    # Panel A: PCA Explained Variance
    ax1 = fig.add_subplot(gs[0, 0])

    n_components = len(pca_data['explained_variance_ratio'])
    components = [f'PC{i + 1}' for i in range(n_components)]
    variance_ratio = [v * 100 for v in pca_data['explained_variance_ratio']]

    bars = ax1.bar(range(n_components), variance_ratio, color='#3498DB',
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, variance_ratio)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(n_components))
    ax1.set_xticklabels(components)
    ax1.set_title('A) PCA Explained Variance', fontsize=12, fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Panel B: Cumulative Variance
    ax2 = fig.add_subplot(gs[0, 1])

    cumulative_var = [v * 100 for v in pca_data['cumulative_variance']]

    ax2.plot(range(n_components), cumulative_var, marker='o', markersize=10,
             linewidth=3, color='#E74C3C', markeredgecolor='black',
             markeredgewidth=1.5, label='Cumulative Variance')
    ax2.axhline(95, color='black', linestyle='--', linewidth=1.5,
                label='95% Threshold', alpha=0.7)

    # Fill area under curve
    ax2.fill_between(range(n_components), cumulative_var, alpha=0.3, color='#E74C3C')

    # Add value labels
    for i, val in enumerate(cumulative_var):
        ax2.text(i, val + 0.00003, f'{val:.4f}%', ha='center',
                 fontsize=8, fontweight='bold')

    ax2.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Components', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(n_components))
    ax2.set_xticklabels([str(i + 1) for i in range(n_components)])
    ax2.set_title('B) Cumulative Explained Variance', fontsize=12,
                  fontweight='bold', loc='left')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_ylim([99.999, 100.0001])

    # Panel C: PCA Component Loadings Heatmap
    ax3 = fig.add_subplot(gs[1, 0])

    components_matrix = np.array(pca_data['components'])
    n_features = components_matrix.shape[1]

    im = ax3.imshow(components_matrix, cmap='RdBu_r', aspect='auto',
                    vmin=-1, vmax=1, interpolation='nearest')

    ax3.set_yticks(range(n_components))
    ax3.set_yticklabels([f'PC{i + 1}' for i in range(n_components)])
    ax3.set_xticks(range(n_features))
    ax3.set_xticklabels([f'F{i + 1}' for i in range(n_features)], fontsize=9)
    ax3.set_xlabel('Features', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Principal Components', fontsize=11, fontweight='bold')
    ax3.set_title('C) PCA Component Loadings', fontsize=12, fontweight='bold', loc='left')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Loading', fontsize=10, fontweight='bold')

    # Add loading values as text
    for i in range(n_components):
        for j in range(n_features):
            text = ax3.text(j, i, f'{components_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=7)

    # Panel D: Clustering Results
    ax4 = fig.add_subplot(gs[1, 1])

    cluster_labels = np.array(cluster_data['cluster_labels'])
    n_clusters = cluster_data['n_clusters']

    # Count samples per cluster
    cluster_counts = [np.sum(cluster_labels == i) for i in range(n_clusters)]

    colors_cluster = ['#3498DB', '#E74C3C', '#2ECC71']
    bars = ax4.bar(range(n_clusters), cluster_counts, color=colors_cluster,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, cluster_counts)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'n = {count}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(n_clusters))
    ax4.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
    ax4.set_title('D) Cluster Distribution', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_axisbelow(True)

    # Add inertia annotation
    inertia_text = f'Inertia: {cluster_data["inertia"]:.2f}'
    ax4.text(0.98, 0.98, inertia_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.savefig('figure5_multivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure5_multivariate_analysis.pdf', bbox_inches='tight')
    print("✓ Figure 5 saved: figure5_multivariate_analysis.png/pdf")
    return fig


def create_figure_6_cluster_centers(data):
    """
    Figure 6: Cluster Centers in PCA Space
    Visualization of cluster centers in 3D PCA space
    """
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    cluster_data = data['multivariate_analysis']['clustering']
    cluster_centers = np.array(cluster_data['cluster_centers'])
    n_clusters = cluster_data['n_clusters']

    # Panel A: 2D projection (PC1 vs PC2)
    ax1 = fig.add_subplot(gs[0, 0])

    colors_cluster = ['#3498DB', '#E74C3C', '#2ECC71']
    markers = ['o', 's', '^']

    for i in range(n_clusters):
        ax1.scatter(cluster_centers[i, 0], cluster_centers[i, 1],
                    s=300, c=colors_cluster[i], marker=markers[i],
                    edgecolors='black', linewidths=2, alpha=0.8,
                    label=f'Cluster {i}', zorder=10)

        # Add cluster label
        ax1.annotate(f'C{i}', (cluster_centers[i, 0], cluster_centers[i, 1]),
                     fontsize=12, fontweight='bold', ha='center', va='center')

    ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax1.set_title('A) Cluster Centers (PC1 vs PC2)', fontsize=12,
                  fontweight='bold', loc='left')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Panel B: 2D projection (PC1 vs PC3)
    ax2 = fig.add_subplot(gs[0, 1])

    for i in range(n_clusters):
        ax2.scatter(cluster_centers[i, 0], cluster_centers[i, 2],
                    s=300, c=colors_cluster[i], marker=markers[i],
                    edgecolors='black', linewidths=2, alpha=0.8,
                    label=f'Cluster {i}', zorder=10)

        # Add cluster label
        ax2.annotate(f'C{i}', (cluster_centers[i, 0], cluster_centers[i, 2]),
                     fontsize=12, fontweight='bold', ha='center', va='center')

    ax2.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PC3', fontsize=11, fontweight='bold')
    ax2.set_title('B) Cluster Centers (PC1 vs PC3)', fontsize=12,
                  fontweight='bold', loc='left')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.savefig('figure6_cluster_centers.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure6_cluster_centers.pdf', bbox_inches='tight')
    print("✓ Figure 6 saved: figure6_cluster_centers.png/pdf")
    return fig


def create_comprehensive_summary(data):
    """
    Create a comprehensive summary figure with key metrics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Title
    fig.suptitle('Comprehensive Statistical Analysis Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Key Metrics Overview
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    ci_data = data['confidence_intervals']
    power_data = data['power_analysis']

    metrics_text = f"""
        KEY PERFORMANCE METRICS
        {'=' * 120}

        Correlation:  {ci_data['correlation']['mean']:.4f}  [95% CI: {ci_data['correlation']['ci_lower']:.4f}, {ci_data['correlation']['ci_upper']:.4f}]
        Peak F1:      {ci_data['peak_f1']['mean']:.4f}  [95% CI: {ci_data['peak_f1']['ci_lower']:.4f}, {ci_data['peak_f1']['ci_upper']:.4f}]
        RMSE:         {ci_data['rmse']['mean']:.4f}  [95% CI: {ci_data['rmse']['ci_lower']:.4f}, {ci_data['rmse']['ci_upper']:.4f}]

        Statistical Power: {power_data['observed_power']:.2%}  |  Sample Size: {power_data['sample_size']}  |  Recommended: {power_data['recommended_sample_size']}
        """

    ax1.text(0.05, 0.5, metrics_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#E8F8F5',
                       edgecolor='black', linewidth=2))

    # Panel 2: Spectra Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    real_stats = data['descriptive_statistics']['real_spectra_stats']
    virtual_stats = data['descriptive_statistics']['virtual_spectra_stats']

    comparison_data = [
        ['Parameter', 'Real', 'Virtual', 'Ratio'],
        ['Mean Intensity', f"{real_stats['intensity_mean']:.2f}",
         f"{virtual_stats['intensity_mean']:.2f}",
         f"{data['descriptive_statistics']['comparative_stats']['intensity_mean_ratio']:.4f}"],
        ['Std Dev', f"{real_stats['intensity_std']:.2f}",
         f"{virtual_stats['intensity_std']:.2f}",
         f"{data['descriptive_statistics']['comparative_stats']['intensity_std_ratio']:.4f}"],
        ['N Spectra', f"{real_stats['n_spectra']}",
         f"{virtual_stats['n_spectra']}", 'N/A'],
        ['Data Points', f"{real_stats['total_data_points']}",
         f"{virtual_stats['total_data_points']}", 'N/A']
    ]

    table = ax2.table(cellText=comparison_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    for i in range(4):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, 5):
        for j in range(4):
            table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
            table[(i, j)].set_edgecolor('black')

    ax2.set_title('Spectra Statistics Comparison', fontsize=11, fontweight='bold', pad=10)
    ax2.axis('off')

    # Panel 3: Hypothesis Tests Summary
    ax3 = fig.add_subplot(gs[1, 1])
    hyp_tests = data['hypothesis_testing']

    test_summary = [
        ['Test', 'p-value', 'Result'],
        ['Correlation > 0', f"{hyp_tests['correlation_vs_zero']['p_value']:.4f}",
         '✓' if hyp_tests['correlation_vs_zero']['significant'] == 'True' else '✗'],
        ['Peak F1 > Random', f"{hyp_tests['peak_f1_vs_random']['p_value']:.2e}",
         '✓' if hyp_tests['peak_f1_vs_random']['significant'] == 'True' else '✗'],
        ['RMSE < 1', f"{hyp_tests['rmse_vs_unity']['p_value']:.2e}",
         '✓' if hyp_tests['rmse_vs_unity']['significant'] == 'True' else '✗'],
        ['Normality', f"{hyp_tests['correlation_normality']['p_value']:.4f}",
         '✓' if hyp_tests['correlation_normality']['is_normal'] == 'True' else '✗'],
        ['LED ANOVA', f"{hyp_tests['led_anova']['p_value']:.4f}",
         '✓' if hyp_tests['led_anova']['significant'] == 'True' else '✗']
    ]

    table2 = ax3.table(cellText=test_summary, cellLoc='center', loc='center',
                       colWidths=[0.4, 0.3, 0.3])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 2.2)

    for i in range(3):
        table2[(0, i)].set_facecolor('#34495E')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, 6):
        for j in range(3):
            if j == 2:
                if test_summary[i][2] == '✓':
                    table2[(i, j)].set_facecolor('#D5F4E6')
                else:
                    table2[(i, j)].set_facecolor('#FADBD8')
            else:
                table2[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
            table2[(i, j)].set_edgecolor('black')

    ax3.set_title('Hypothesis Testing Results', fontsize=11, fontweight='bold', pad=10)
    ax3.axis('off')

    # Panel 4: Effect Sizes
    ax4 = fig.add_subplot(gs[1, 2])
    effect_data = data['effect_sizes']

    effect_summary = [
        ['Metric', 'Value', 'Interpretation'],
        ['Correlation d', f"{effect_data['correlation_cohens_d']['value']:.3f}",
         effect_data['correlation_cohens_d']['interpretation']],
        ['Peak F1 d', f"{effect_data['peak_f1_cohens_d']['value']:.3f}",
         effect_data['peak_f1_cohens_d']['interpretation']],
        ['Mean R²', f"{effect_data['mean_r_squared']['value']:.3f}",
         effect_data['mean_r_squared']['interpretation']]
    ]

    table3 = ax4.table(cellText=effect_summary, cellLoc='center', loc='center',
                       colWidths=[0.3, 0.25, 0.45])
    table3.auto_set_font_size(False)
    table3.set_fontsize(8)
    table3.scale(1, 3)

    for i in range(3):
        table3[(0, i)].set_facecolor('#34495E')
        table3[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, 4):
        for j in range(3):
            table3[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
            table3[(i, j)].set_edgecolor('black')

    ax4.set_title('Effect Sizes', fontsize=11, fontweight='bold', pad=10)
    ax4.axis('off')

    # Panel 5: PCA Variance
    ax5 = fig.add_subplot(gs[2, 0])
    pca_data = data['multivariate_analysis']['pca']
    variance_pct = [v * 100 for v in pca_data['explained_variance_ratio']]

    ax5.bar(range(len(variance_pct)), variance_pct, color='#9B59B6',
            alpha=0.8, edgecolor='black', linewidth=1.2)
    ax5.set_ylabel('Variance (%)', fontsize=10, fontweight='bold')
    ax5.set_xlabel('PC', fontsize=10, fontweight='bold')
    ax5.set_xticks(range(len(variance_pct)))
    ax5.set_xticklabels([f'PC{i + 1}' for i in range(len(variance_pct))])
    ax5.set_title('PCA Explained Variance', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.set_axisbelow(True)

    # Panel 6: Cluster Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    cluster_data = data['multivariate_analysis']['clustering']
    cluster_labels = np.array(cluster_data['cluster_labels'])
    cluster_counts = [np.sum(cluster_labels == i) for i in range(cluster_data['n_clusters'])]

    colors_pie = ['#3498DB', '#E74C3C', '#2ECC71']
    wedges, texts, autotexts = ax6.pie(cluster_counts, labels=[f'Cluster {i}' for i in range(len(cluster_counts))],
                                       autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                       wedgeprops=dict(edgecolor='black', linewidth=1.5))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax6.set_title('Cluster Distribution', fontsize=11, fontweight='bold')

    # Panel 7: Confidence Intervals Comparison
    ax7 = fig.add_subplot(gs[2, 2])

    metrics = ['Correlation', 'Peak F1', 'RMSE']
    means = [ci_data['correlation']['mean'], ci_data['peak_f1']['mean'],
             ci_data['rmse']['mean']]
    errors_lower = [ci_data['correlation']['mean'] - ci_data['correlation']['ci_lower'],
                    ci_data['peak_f1']['mean'] - ci_data['peak_f1']['ci_lower'],
                    ci_data['rmse']['mean'] - ci_data['rmse']['ci_lower']]
    errors_upper = [ci_data['correlation']['ci_upper'] - ci_data['correlation']['mean'],
                    ci_data['peak_f1']['ci_upper'] - ci_data['peak_f1']['mean'],
                    ci_data['rmse']['ci_upper'] - ci_data['rmse']['mean']]

    colors_ci = ['#3498DB', '#2ECC71', '#E74C3C']

    for i, (metric, mean, err_low, err_up, color) in enumerate(
            zip(metrics, means, errors_lower, errors_upper, colors_ci)):
        ax7.errorbar(i, mean, yerr=[[err_low], [err_up]], fmt='o', markersize=10,
                     color=color, markeredgecolor='black', markeredgewidth=1.5,
                     capsize=8, capthick=2, elinewidth=2, label=metric)

    ax7.set_xticks(range(len(metrics)))
    ax7.set_xticklabels(metrics, rotation=45, ha='right')
    ax7.set_ylabel('Value', fontsize=10, fontweight='bold')
    ax7.set_title('Confidence Intervals (95%)', fontsize=11, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3, linestyle='--')
    ax7.set_axisbelow(True)

    plt.savefig('figure7_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure7_comprehensive_summary.pdf', bbox_inches='tight')
    print("✓ Figure 7 saved: figure7_comprehensive_summary.png/pdf")
    return fig


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60 + "\n")

    # Generate all figures
    fig1 = create_figure_1_descriptive_stats(data)
    plt.close(fig1)

    fig2 = create_figure_2_hypothesis_testing(data)
    plt.close(fig2)

    fig3 = create_figure_3_confidence_intervals(data)
    plt.close(fig3)

    fig4 = create_figure_4_power_analysis(data)
    plt.close(fig4)

    fig5 = create_figure_5_multivariate(data)
    plt.close(fig5)

    fig6 = create_figure_6_cluster_centers(data)
    plt.close(fig6)

    fig7 = create_comprehensive_summary(data)
    plt.close(fig7)

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • figure1_descriptive_statistics.png/pdf")
    print("  • figure2_hypothesis_testing.png/pdf")
    print("  • figure3_confidence_intervals.png/pdf")
    print("  • figure4_power_analysis.png/pdf")
    print("  • figure5_multivariate_analysis.png/pdf")
    print("  • figure6_cluster_centers.png/pdf")
    print("  • figure7_comprehensive_summary.png/pdf")
    print("\n" + "=" * 60 + "\n")

