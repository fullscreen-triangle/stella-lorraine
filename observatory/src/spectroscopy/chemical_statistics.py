#!/usr/bin/env python3
"""
Statistical Analysis for Hardware-Based Spectroscopy Validation
=============================================================

Comprehensive statistical methods to validate our hardware-based spectroscopy
system against established spectrometers using rigorous statistical tests.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import json
import warnings
warnings.filterwarnings('ignore')

class SpectroscopyStatistics:
    """
    Comprehensive statistical analysis framework for spectroscopy validation
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.results = {}

    def descriptive_statistics(self, real_data, virtual_data):
        """Calculate comprehensive descriptive statistics"""
        print("📊 Computing descriptive statistics...")

        stats_results = {
            'real_spectra_stats': {},
            'virtual_spectra_stats': {},
            'comparative_stats': {}
        }

        # Real spectra statistics
        real_intensities = []
        real_wavelengths = []

        for spectrum in real_data.values():
            real_intensities.extend(spectrum['intensities'])
            real_wavelengths.extend(spectrum['wavelengths'])

        real_intensities = np.array(real_intensities)
        real_wavelengths = np.array(real_wavelengths)

        stats_results['real_spectra_stats'] = {
            'intensity_mean': np.mean(real_intensities),
            'intensity_std': np.std(real_intensities),
            'intensity_median': np.median(real_intensities),
            'intensity_iqr': np.percentile(real_intensities, 75) - np.percentile(real_intensities, 25),
            'intensity_skewness': stats.skew(real_intensities),
            'intensity_kurtosis': stats.kurtosis(real_intensities),
            'wavelength_range': [np.min(real_wavelengths), np.max(real_wavelengths)],
            'n_spectra': len(real_data),
            'total_data_points': len(real_intensities)
        }

        # Virtual spectra statistics
        virtual_intensities = []
        virtual_wavelengths = []

        for spectrum in virtual_data.values():
            virtual_intensities.extend(spectrum['combined_intensities'])
            virtual_wavelengths.extend(spectrum['combined_wavelengths'])

        virtual_intensities = np.array(virtual_intensities)
        virtual_wavelengths = np.array(virtual_wavelengths)

        stats_results['virtual_spectra_stats'] = {
            'intensity_mean': np.mean(virtual_intensities),
            'intensity_std': np.std(virtual_intensities),
            'intensity_median': np.median(virtual_intensities),
            'intensity_iqr': np.percentile(virtual_intensities, 75) - np.percentile(virtual_intensities, 25),
            'intensity_skewness': stats.skew(virtual_intensities),
            'intensity_kurtosis': stats.kurtosis(virtual_intensities),
            'wavelength_range': [np.min(virtual_wavelengths), np.max(virtual_wavelengths)],
            'n_spectra': len(virtual_data),
            'total_data_points': len(virtual_intensities)
        }

        # Comparative statistics
        stats_results['comparative_stats'] = {
            'intensity_mean_ratio': stats_results['virtual_spectra_stats']['intensity_mean'] /
                                  stats_results['real_spectra_stats']['intensity_mean'],
            'intensity_std_ratio': stats_results['virtual_spectra_stats']['intensity_std'] /
                                 stats_results['real_spectra_stats']['intensity_std'],
            'wavelength_overlap': self._calculate_wavelength_overlap(
                stats_results['real_spectra_stats']['wavelength_range'],
                stats_results['virtual_spectra_stats']['wavelength_range']
            )
        }

        return stats_results

    def _calculate_wavelength_overlap(self, real_range, virtual_range):
        """Calculate wavelength range overlap percentage"""
        overlap_start = max(real_range[0], virtual_range[0])
        overlap_end = min(real_range[1], virtual_range[1])

        if overlap_end <= overlap_start:
            return 0.0

        overlap_length = overlap_end - overlap_start
        real_length = real_range[1] - real_range[0]
        virtual_length = virtual_range[1] - virtual_range[0]

        # Calculate overlap as percentage of smaller range
        min_length = min(real_length, virtual_length)
        return (overlap_length / min_length) * 100 if min_length > 0 else 0.0

    def hypothesis_testing(self, validation_results):
        """Perform comprehensive hypothesis testing"""
        print("🧪 Performing hypothesis testing...")

        hypothesis_results = {}

        # Extract data for testing
        correlations = [r['pearson_correlation'] for r in validation_results['spectral_similarity_results']]
        peak_f1_scores = [r['peak_f1_score'] for r in validation_results['peak_detection_results']]
        rmse_values = [r['rmse'] for r in validation_results['spectral_similarity_results']]

        # Test 1: Correlation significantly greater than 0
        correlation_test = stats.ttest_1samp(correlations, 0)
        hypothesis_results['correlation_vs_zero'] = {
            'test': 'One-sample t-test (correlation > 0)',
            'statistic': correlation_test.statistic,
            'p_value': correlation_test.pvalue,
            'significant': correlation_test.pvalue < 0.05,
            'interpretation': 'Correlations significantly greater than 0' if correlation_test.pvalue < 0.05 else 'No significant correlation'
        }

        # Test 2: Peak F1 scores significantly greater than random (0.33)
        peak_test = stats.ttest_1samp(peak_f1_scores, 0.33)
        hypothesis_results['peak_f1_vs_random'] = {
            'test': 'One-sample t-test (peak F1 > random)',
            'statistic': peak_test.statistic,
            'p_value': peak_test.pvalue,
            'significant': peak_test.pvalue < 0.05,
            'interpretation': 'Peak detection significantly better than random' if peak_test.pvalue < 0.05 else 'Peak detection not significantly better than random'
        }

        # Test 3: RMSE significantly less than 1 (normalized scale)
        rmse_test = stats.ttest_1samp(rmse_values, 1.0)
        hypothesis_results['rmse_vs_unity'] = {
            'test': 'One-sample t-test (RMSE < 1)',
            'statistic': rmse_test.statistic,
            'p_value': rmse_test.pvalue,
            'significant': rmse_test.pvalue < 0.05,
            'interpretation': 'RMSE significantly less than unity' if rmse_test.pvalue < 0.05 else 'RMSE not significantly less than unity'
        }

        # Test 4: Normality tests
        correlation_normality = stats.shapiro(correlations)
        hypothesis_results['correlation_normality'] = {
            'test': 'Shapiro-Wilk normality test',
            'statistic': correlation_normality.statistic,
            'p_value': correlation_normality.pvalue,
            'is_normal': correlation_normality.pvalue > 0.05,
            'interpretation': 'Correlations are normally distributed' if correlation_normality.pvalue > 0.05 else 'Correlations are not normally distributed'
        }

        # Test 5: LED wavelength response ANOVA
        led_responses = validation_results['led_validation']
        blue_responses = led_responses['blue']['responses']
        green_responses = led_responses['green']['responses']
        red_responses = led_responses['red']['responses']

        led_anova = stats.f_oneway(blue_responses, green_responses, red_responses)
        hypothesis_results['led_anova'] = {
            'test': 'One-way ANOVA (LED responses)',
            'statistic': led_anova.statistic,
            'p_value': led_anova.pvalue,
            'significant': led_anova.pvalue < 0.05,
            'interpretation': 'Significant differences between LED responses' if led_anova.pvalue < 0.05 else 'No significant differences between LED responses'
        }

        return hypothesis_results

    def effect_size_analysis(self, validation_results):
        """Calculate effect sizes for validation metrics"""
        print("📏 Calculating effect sizes...")

        effect_sizes = {}

        correlations = [r['pearson_correlation'] for r in validation_results['spectral_similarity_results']]
        peak_f1_scores = [r['peak_f1_score'] for r in validation_results['peak_detection_results']]

        # Cohen's d for correlation vs zero
        correlation_mean = np.mean(correlations)
        correlation_std = np.std(correlations, ddof=1)
        cohens_d_correlation = correlation_mean / correlation_std if correlation_std > 0 else 0

        effect_sizes['correlation_cohens_d'] = {
            'value': cohens_d_correlation,
            'interpretation': self._interpret_cohens_d(cohens_d_correlation)
        }

        # Cohen's d for peak F1 vs random (0.33)
        peak_mean = np.mean(peak_f1_scores)
        peak_std = np.std(peak_f1_scores, ddof=1)
        cohens_d_peak = (peak_mean - 0.33) / peak_std if peak_std > 0 else 0

        effect_sizes['peak_f1_cohens_d'] = {
            'value': cohens_d_peak,
            'interpretation': self._interpret_cohens_d(cohens_d_peak)
        }

        # R-squared (coefficient of determination) as effect size
        r_squared_values = [r['r2_score'] for r in validation_results['spectral_similarity_results']]
        mean_r_squared = np.mean(r_squared_values)

        effect_sizes['mean_r_squared'] = {
            'value': mean_r_squared,
            'interpretation': self._interpret_r_squared(mean_r_squared)
        }

        return effect_sizes

    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"

    def _interpret_r_squared(self, r2):
        """Interpret R-squared values"""
        if r2 < 0.1:
            return "Very weak relationship"
        elif r2 < 0.3:
            return "Weak relationship"
        elif r2 < 0.5:
            return "Moderate relationship"
        elif r2 < 0.7:
            return "Strong relationship"
        else:
            return "Very strong relationship"

    def confidence_intervals(self, validation_results, confidence_level=0.95):
        """Calculate confidence intervals for key metrics"""
        print("📊 Computing confidence intervals...")

        alpha = 1 - confidence_level

        ci_results = {}

        # Correlation confidence intervals
        correlations = [r['pearson_correlation'] for r in validation_results['spectral_similarity_results']]
        correlation_mean = np.mean(correlations)
        correlation_sem = stats.sem(correlations)
        correlation_ci = stats.t.interval(confidence_level, len(correlations)-1,
                                        loc=correlation_mean, scale=correlation_sem)

        ci_results['correlation'] = {
            'mean': correlation_mean,
            'ci_lower': correlation_ci[0],
            'ci_upper': correlation_ci[1],
            'sem': correlation_sem
        }

        # Peak F1 confidence intervals
        peak_f1_scores = [r['peak_f1_score'] for r in validation_results['peak_detection_results']]
        peak_mean = np.mean(peak_f1_scores)
        peak_sem = stats.sem(peak_f1_scores)
        peak_ci = stats.t.interval(confidence_level, len(peak_f1_scores)-1,
                                 loc=peak_mean, scale=peak_sem)

        ci_results['peak_f1'] = {
            'mean': peak_mean,
            'ci_lower': peak_ci[0],
            'ci_upper': peak_ci[1],
            'sem': peak_sem
        }

        # RMSE confidence intervals
        rmse_values = [r['rmse'] for r in validation_results['spectral_similarity_results']]
        rmse_mean = np.mean(rmse_values)
        rmse_sem = stats.sem(rmse_values)
        rmse_ci = stats.t.interval(confidence_level, len(rmse_values)-1,
                                 loc=rmse_mean, scale=rmse_sem)

        ci_results['rmse'] = {
            'mean': rmse_mean,
            'ci_lower': rmse_ci[0],
            'ci_upper': rmse_ci[1],
            'sem': rmse_sem
        }

        return ci_results

    def power_analysis(self, validation_results):
        """Perform statistical power analysis"""
        print("⚡ Performing power analysis...")

        correlations = [r['pearson_correlation'] for r in validation_results['spectral_similarity_results']]
        n = len(correlations)

        # Power analysis for correlation test
        correlation_mean = np.mean(correlations)
        correlation_std = np.std(correlations, ddof=1)

        # Calculate observed effect size
        effect_size = correlation_mean / correlation_std if correlation_std > 0 else 0

        # Estimate power using t-distribution
        alpha = 0.05
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        t_observed = effect_size * np.sqrt(n)

        power = 1 - stats.t.cdf(t_critical, df, loc=t_observed)

        power_results = {
            'sample_size': n,
            'effect_size': effect_size,
            'alpha': alpha,
            'observed_power': power,
            'power_interpretation': self._interpret_power(power),
            'recommended_sample_size': self._recommend_sample_size(effect_size, alpha, 0.8)
        }

        return power_results

    def _interpret_power(self, power):
        """Interpret statistical power"""
        if power < 0.5:
            return "Very low power - high risk of Type II error"
        elif power < 0.8:
            return "Low power - consider increasing sample size"
        elif power < 0.9:
            return "Adequate power"
        else:
            return "High power"

    def _recommend_sample_size(self, effect_size, alpha, desired_power):
        """Recommend sample size for desired power"""
        if effect_size <= 0:
            return "Cannot calculate - effect size too small"

        # Approximate sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(desired_power)

        n_recommended = ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n_recommended))

    def multivariate_analysis(self, validation_results):
        """Perform multivariate statistical analysis"""
        print("🔍 Performing multivariate analysis...")

        # Prepare data matrix
        features = []
        labels = []

        for result in validation_results['spectral_similarity_results']:
            feature_vector = [
                result['pearson_correlation'],
                result['cosine_similarity'],
                result['rmse'],
                result['r2_score'],
                result['spectral_angle']
            ]
            features.append(feature_vector)
            labels.append(result['real_spectrum'])

        features = np.array(features)

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        multivariate_results = {}

        # Principal Component Analysis
        if len(features) > 1:
            pca = PCA(n_components=min(3, features.shape[1]))
            pca_features = pca.fit_transform(features)

            multivariate_results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': pca.components_.tolist(),
                'n_components': pca.n_components_
            }

            # K-means clustering on PCA features
            if len(pca_features) >= 2:
                n_clusters = min(3, len(pca_features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(pca_features)

                multivariate_results['clustering'] = {
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'inertia': kmeans.inertia_
                }

        return multivariate_results

    def create_statistical_visualizations(self, stats_results, hypothesis_results,
                                        effect_sizes, ci_results, power_results,
                                        multivariate_results, save_dir):
        """Create comprehensive statistical visualizations"""

        fig = plt.figure(figsize=(20, 16))

        # 1. Descriptive Statistics Comparison
        ax1 = plt.subplot(4, 4, 1)
        real_stats = stats_results['real_spectra_stats']
        virtual_stats = stats_results['virtual_spectra_stats']

        metrics = ['intensity_mean', 'intensity_std', 'intensity_median']
        real_values = [real_stats[m] for m in metrics]
        virtual_values = [virtual_stats[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width/2, real_values, width, label='Real', alpha=0.7)
        plt.bar(x + width/2, virtual_values, width, label='Virtual', alpha=0.7)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Descriptive Statistics Comparison')
        plt.xticks(x, ['Mean', 'Std', 'Median'])
        plt.legend()

        # 2. Hypothesis Testing Results
        ax2 = plt.subplot(4, 4, 2)
        test_names = list(hypothesis_results.keys())
        p_values = [hypothesis_results[test].get('p_value', 1.0) for test in test_names]
        significant = [hypothesis_results[test].get('significant', False) for test in test_names]

        colors = ['green' if sig else 'red' for sig in significant]
        bars = plt.bar(range(len(test_names)), p_values, color=colors, alpha=0.7)
        plt.axhline(y=0.05, color='black', linestyle='--', label='α = 0.05')
        plt.xlabel('Tests')
        plt.ylabel('p-value')
        plt.title('Hypothesis Testing Results')
        plt.xticks(range(len(test_names)), [name.replace('_', '\n') for name in test_names], rotation=45)
        plt.legend()
        plt.yscale('log')

        # 3. Effect Sizes
        ax3 = plt.subplot(4, 4, 3)
        effect_names = list(effect_sizes.keys())
        effect_values = [effect_sizes[name]['value'] for name in effect_names]

        plt.bar(range(len(effect_names)), effect_values, alpha=0.7, color='purple')
        plt.xlabel('Effect Size Measures')
        plt.ylabel('Effect Size')
        plt.title('Effect Size Analysis')
        plt.xticks(range(len(effect_names)), [name.replace('_', '\n') for name in effect_names])

        # 4. Confidence Intervals
        ax4 = plt.subplot(4, 4, 4)
        ci_metrics = list(ci_results.keys())
        ci_means = [ci_results[metric]['mean'] for metric in ci_metrics]
        ci_lowers = [ci_results[metric]['ci_lower'] for metric in ci_metrics]
        ci_uppers = [ci_results[metric]['ci_upper'] for metric in ci_metrics]

        x_pos = range(len(ci_metrics))
        plt.errorbar(x_pos, ci_means,
                    yerr=[np.array(ci_means) - np.array(ci_lowers),
                          np.array(ci_uppers) - np.array(ci_means)],
                    fmt='o', capsize=5, capthick=2)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('95% Confidence Intervals')
        plt.xticks(x_pos, ci_metrics)

        # 5. Power Analysis
        ax5 = plt.subplot(4, 4, 5)
        power_info = [
            ('Sample Size', power_results['sample_size']),
            ('Effect Size', power_results['effect_size']),
            ('Observed Power', power_results['observed_power']),
            ('Recommended N', power_results['recommended_sample_size'] if isinstance(power_results['recommended_sample_size'], (int, float)) else 0)
        ]

        labels, values = zip(*power_info)
        plt.bar(range(len(labels)), values, alpha=0.7, color='orange')
        plt.xlabel('Power Analysis Components')
        plt.ylabel('Values')
        plt.title('Statistical Power Analysis')
        plt.xticks(range(len(labels)), labels, rotation=45)

        # 6. PCA Explained Variance
        if 'pca' in multivariate_results:
            ax6 = plt.subplot(4, 4, 6)
            pca_data = multivariate_results['pca']
            components = range(1, len(pca_data['explained_variance_ratio']) + 1)

            plt.bar(components, pca_data['explained_variance_ratio'], alpha=0.7, color='blue')
            plt.plot(components, pca_data['cumulative_variance'], 'ro-', label='Cumulative')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.legend()

        # 7-12. Distribution plots for key metrics
        # (These would be populated with actual data from validation_results)

        # 13. Statistical Summary
        ax13 = plt.subplot(4, 4, 13)
        ax13.axis('off')

        summary_text = f"""
        Statistical Analysis Summary
        ===========================

        Descriptive Statistics:
        - Real spectra: {real_stats['n_spectra']} samples
        - Virtual spectra: {virtual_stats['n_spectra']} samples
        - Wavelength overlap: {stats_results['comparative_stats']['wavelength_overlap']:.1f}%

        Hypothesis Testing:
        - Significant correlations: {hypothesis_results['correlation_vs_zero'].get('significant', False)}
        - Peak detection > random: {hypothesis_results['peak_f1_vs_random'].get('significant', False)}
        - RMSE < unity: {hypothesis_results['rmse_vs_unity'].get('significant', False)}

        Effect Sizes:
        - Correlation effect: {effect_sizes['correlation_cohens_d']['interpretation']}
        - Peak F1 effect: {effect_sizes['peak_f1_cohens_d']['interpretation']}

        Power Analysis:
        - Observed power: {power_results['observed_power']:.3f}
        - Power interpretation: {power_results['power_interpretation']}
        """

        ax13.text(0.1, 0.9, summary_text, transform=ax13.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def comprehensive_statistical_analysis(self, validation_results, real_data, virtual_data, save_dir):
        """Run complete statistical analysis pipeline"""
        print("📊 Running Comprehensive Statistical Analysis")
        print("=" * 50)

        # 1. Descriptive Statistics
        stats_results = self.descriptive_statistics(real_data, virtual_data)

        # 2. Hypothesis Testing
        hypothesis_results = self.hypothesis_testing(validation_results)

        # 3. Effect Size Analysis
        effect_sizes = self.effect_size_analysis(validation_results)

        # 4. Confidence Intervals
        ci_results = self.confidence_intervals(validation_results)

        # 5. Power Analysis
        power_results = self.power_analysis(validation_results)

        # 6. Multivariate Analysis
        multivariate_results = self.multivariate_analysis(validation_results)

        # Compile all results
        complete_results = {
            'descriptive_statistics': stats_results,
            'hypothesis_testing': hypothesis_results,
            'effect_sizes': effect_sizes,
            'confidence_intervals': ci_results,
            'power_analysis': power_results,
            'multivariate_analysis': multivariate_results
        }

        # Create visualizations
        self.create_statistical_visualizations(
            stats_results, hypothesis_results, effect_sizes,
            ci_results, power_results, multivariate_results, save_dir
        )

        # Save results
        with open(os.path.join(save_dir, 'statistical_analysis_results.json'), 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)

        # Print summary
        self.print_statistical_summary(complete_results)

        return complete_results

    def print_statistical_summary(self, results):
        """Print comprehensive statistical summary"""
        print("\n📊 Statistical Analysis Summary")
        print("=" * 40)

        # Descriptive statistics
        real_stats = results['descriptive_statistics']['real_spectra_stats']
        virtual_stats = results['descriptive_statistics']['virtual_spectra_stats']

        print(f"\n📈 Descriptive Statistics:")
        print(f"   Real Spectra: {real_stats['n_spectra']} samples, {real_stats['total_data_points']} data points")
        print(f"   Virtual Spectra: {virtual_stats['n_spectra']} samples, {virtual_stats['total_data_points']} data points")
        print(f"   Intensity correlation: {results['descriptive_statistics']['comparative_stats']['intensity_mean_ratio']:.3f}")

        # Hypothesis testing
        print(f"\n🧪 Hypothesis Testing:")
        for test_name, test_result in results['hypothesis_testing'].items():
            significance = "✅" if test_result.get('significant', False) else "❌"
            print(f"   {significance} {test_result.get('interpretation', 'No interpretation available')}")

        # Effect sizes
        print(f"\n📏 Effect Sizes:")
        for effect_name, effect_result in results['effect_sizes'].items():
            print(f"   {effect_name}: {effect_result['value']:.3f} ({effect_result['interpretation']})")

        # Confidence intervals
        print(f"\n📊 95% Confidence Intervals:")
        for metric, ci_data in results['confidence_intervals'].items():
            print(f"   {metric}: {ci_data['mean']:.3f} [{ci_data['ci_lower']:.3f}, {ci_data['ci_upper']:.3f}]")

        # Power analysis
        power_data = results['power_analysis']
        print(f"\n⚡ Power Analysis:")
        print(f"   Sample size: {power_data['sample_size']}")
        print(f"   Observed power: {power_data['observed_power']:.3f}")
        print(f"   Power assessment: {power_data['power_interpretation']}")

        # Overall assessment
        print(f"\n🎯 Overall Statistical Assessment:")

        # Check if validation is statistically significant
        correlation_significant = results['hypothesis_testing']['correlation_vs_zero'].get('significant', False)
        peak_significant = results['hypothesis_testing']['peak_f1_vs_random'].get('significant', False)

        if correlation_significant and peak_significant:
            print("   ✅ Hardware-based spectroscopy shows statistically significant performance")
        elif correlation_significant or peak_significant:
            print("   ⚠️ Hardware-based spectroscopy shows partial statistical validation")
        else:
            print("   ❌ Hardware-based spectroscopy lacks statistical validation")

        # Check effect sizes
        correlation_effect = results['effect_sizes']['correlation_cohens_d']['value']
        if correlation_effect > 0.8:
            print("   ✅ Large effect size demonstrates practical significance")
        elif correlation_effect > 0.5:
            print("   ⚠️ Medium effect size shows moderate practical significance")
        else:
            print("   ❌ Small effect size indicates limited practical significance")

def main():
    """Main statistical analysis execution"""
    print("📊 Statistical Analysis for Hardware-Based Spectroscopy")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    spectra_dir = os.path.join(base_dir, 'public', 'spectra')
    results_dir = os.path.join(base_dir, 'results')

    # Import the validation module to use its data loading
    from validation import SpectroscopyValidator, load_molecular_patterns
    from led_spectroscopy import LEDSpectroscopySystem
    from spectral_analysis_algorithm import SpectralAnalyzer
    from rgb_chemical_mapping import RGBChemicalMapper

    # Initialize systems
    validator = SpectroscopyValidator()
    led_system = LEDSpectroscopySystem()
    spectral_analyzer = SpectralAnalyzer()
    rgb_mapper = RGBChemicalMapper()

    # Load real spectra data
    print("📊 Loading real instrument spectra...")
    real_data = validator.load_real_spectra(spectra_dir)

    # Load molecular patterns and generate virtual spectra
    print("🧪 Loading molecular patterns...")
    molecular_patterns = load_molecular_patterns()

    print("🔬 Generating virtual spectra using hardware systems...")
    virtual_data = {}

    for i, pattern in enumerate(molecular_patterns):
        # LED spectroscopy analysis
        led_results = {}
        for led_color, wavelength in led_system.led_wavelengths.items():
            led_analysis = led_system.analyze_molecular_fluorescence(pattern, wavelength)
            led_results[led_color] = led_analysis

        # Spectral analysis
        spectral_analysis = spectral_analyzer.analyze_spectrum(pattern)

        # RGB mapping
        rgb_mapping = rgb_mapper.map_pattern_to_rgb(pattern)

        virtual_data[f'pattern_{i}'] = {
            'pattern': pattern,
            'led_results': led_results,
            'spectral_analysis': spectral_analysis,
            'rgb_mapping': rgb_mapping,
            'combined_wavelengths': spectral_analysis['wavelengths'],
            'combined_intensities': spectral_analysis['intensities']
        }

    # Run validation if we have both real and virtual data
    if real_data and virtual_data:
        print("⚗️ Running validation experiments...")
        validation_results = validator.comprehensive_validation(spectra_dir, molecular_patterns)

        if validation_results:
            # Initialize statistical analyzer
            stats_analyzer = SpectroscopyStatistics()

            # Run comprehensive analysis with real data
            statistical_results = stats_analyzer.comprehensive_statistical_analysis(
                validation_results, real_data, virtual_data, results_dir
            )

            print("\n💾 Statistical analysis complete!")
            print(f"   Results saved to: {results_dir}")
        else:
            print("❌ Validation experiments failed.")
    else:
        print("❌ Insufficient data for analysis.")
        if not real_data:
            print(f"   No real spectra found in: {spectra_dir}")
        if not virtual_data:
            print("   Failed to generate virtual spectra.")

if __name__ == "__main__":
    main()
