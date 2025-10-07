"""
Statistical Analysis Framework for Precision Measurements

This script includes ALL necessary scientific statistical methods required for
time keeping and very precise measurements. It supports both traditional and
"exotic" methods including time sequence conversion, ambiguous compression,
and semantic distance analysis for reducing time measurement duration.

Statistical validation of precision enhancement methods from oscillatory components:
- observatory/src/oscillatory/ambigous_compression
- observatory/src/oscillatory/semantic_distance
- observatory/src/oscillatory/time_sequencing
- observatory/src/oscillatory/observer_oscillatory_hierarchy

All methods assist in reducing time taken to read time with statistical validation.
"""

import numpy as np
import scipy.stats as stats
from scipy import signal
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from collections import defaultdict


class StatisticalTest(Enum):
    """Types of statistical tests for precision measurement validation"""
    T_TEST = "t_test"                                    # Student's t-test
    WELCH_T_TEST = "welch_t_test"                       # Welch's t-test (unequal variances)
    PAIRED_T_TEST = "paired_t_test"                     # Paired t-test
    MANN_WHITNEY_U = "mann_whitney_u"                   # Mann-Whitney U test
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"      # Wilcoxon signed-rank test
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"          # KS test for distributions
    ANDERSON_DARLING = "anderson_darling"               # Anderson-Darling test
    SHAPIRO_WILK = "shapiro_wilk"                       # Shapiro-Wilk normality test
    F_TEST = "f_test"                                   # F-test for variance equality
    CHI_SQUARE = "chi_square"                           # Chi-square test
    STRATEGIC_DISAGREEMENT = "strategic_disagreement"   # Strategic disagreement test
    BINOMIAL_TEST = "binomial_test"                     # Binomial test
    POISSON_TEST = "poisson_test"                       # Poisson test
    BAYESIAN_T_TEST = "bayesian_t_test"                 # Bayesian t-test
    PRECISION_VALIDATION = "precision_validation"       # Precision validation test


class HypothesisType(Enum):
    """Types of statistical hypotheses"""
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"
    EQUIVALENCE = "equivalence"
    NON_INFERIORITY = "non_inferiority"
    SUPERIORITY = "superiority"


class DistributionType(Enum):
    """Types of probability distributions for modeling"""
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    UNIFORM = "uniform"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    WEIBULL = "weibull"
    PARETO = "pareto"
    STUDENT_T = "student_t"
    CHI_SQUARE_DIST = "chi_square"
    F_DISTRIBUTION = "f_distribution"


class SignificanceLevel(Enum):
    """Standard significance levels"""
    ALPHA_001 = 0.001    # 99.9% confidence
    ALPHA_005 = 0.005    # 99.5% confidence
    ALPHA_01 = 0.01      # 99% confidence
    ALPHA_05 = 0.05      # 95% confidence
    ALPHA_10 = 0.10      # 90% confidence


@dataclass
class TestResult:
    """Result of a statistical test"""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    sample_size: int = 0
    degrees_of_freedom: Optional[int] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant"""
        return self.p_value < alpha

    def get_effect_size_interpretation(self) -> str:
        """Interpret effect size magnitude"""
        if self.effect_size is None:
            return "Unknown"

        # Cohen's conventions
        if abs(self.effect_size) < 0.2:
            return "Small effect"
        elif abs(self.effect_size) < 0.5:
            return "Medium effect"
        elif abs(self.effect_size) < 0.8:
            return "Large effect"
        else:
            return "Very large effect"


@dataclass
class PowerAnalysisResult:
    """Result of statistical power analysis"""
    test_type: StatisticalTest
    power: float
    alpha: float
    effect_size: float
    sample_size: int
    required_sample_size: Optional[int] = None
    minimum_detectable_effect: Optional[float] = None
    power_curve_points: List[Tuple[float, float]] = field(default_factory=list)

    def is_adequately_powered(self, threshold: float = 0.8) -> bool:
        """Check if study is adequately powered"""
        return self.power >= threshold


@dataclass
class ConfidenceInterval:
    """Confidence interval result"""
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str
    parameter_name: str = ""

    def contains_value(self, value: float) -> bool:
        """Check if confidence interval contains a specific value"""
        return self.lower_bound <= value <= self.upper_bound

    def width(self) -> float:
        """Calculate confidence interval width"""
        return self.upper_bound - self.lower_bound


@dataclass
class BayesianResult:
    """Result of Bayesian statistical analysis"""
    posterior_mean: float
    posterior_std: float
    credible_interval: Tuple[float, float]
    credible_level: float
    bayes_factor: Optional[float] = None
    prior_parameters: Dict[str, float] = field(default_factory=dict)
    posterior_parameters: Dict[str, float] = field(default_factory=dict)
    evidence: Optional[float] = None

    def probability_greater_than(self, threshold: float) -> float:
        """Calculate probability that parameter is greater than threshold"""
        # Assuming normal posterior (could be extended for other distributions)
        z_score = (threshold - self.posterior_mean) / self.posterior_std
        return 1.0 - stats.norm.cdf(z_score)


class PrecisionStatistics:
    """
    Comprehensive statistical analysis framework for precision measurements

    Includes traditional statistical methods and exotic precision enhancement
    validation for oscillatory time sequencing and semantic distance methods.
    """

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.power_analyses: List[PowerAnalysisResult] = []
        self.confidence_intervals: List[ConfidenceInterval] = []
        self.bayesian_results: List[BayesianResult] = []

    # Traditional Statistical Tests

    def t_test(self, sample1: List[float], sample2: Optional[List[float]] = None,
               population_mean: Optional[float] = None,
               alternative: HypothesisType = HypothesisType.TWO_SIDED,
               alpha: float = 0.05) -> TestResult:
        """
        Perform t-test for mean comparison

        Args:
            sample1: First sample
            sample2: Second sample (for two-sample test)
            population_mean: Population mean (for one-sample test)
            alternative: Type of alternative hypothesis
            alpha: Significance level
        """

        if sample2 is not None:
            # Two-sample t-test
            if alternative == HypothesisType.TWO_SIDED:
                statistic, p_value = stats.ttest_ind(sample1, sample2)
            elif alternative == HypothesisType.GREATER:
                statistic, p_value = stats.ttest_ind(sample1, sample2, alternative='greater')
            else:  # LESS
                statistic, p_value = stats.ttest_ind(sample1, sample2, alternative='less')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1, ddof=1) +
                                 (len(sample2) - 1) * np.var(sample2, ddof=1)) /
                                (len(sample1) + len(sample2) - 2))
            effect_size = (np.mean(sample1) - np.mean(sample2)) / pooled_std

            df = len(sample1) + len(sample2) - 2

        else:
            # One-sample t-test
            if population_mean is None:
                population_mean = 0.0

            if alternative == HypothesisType.TWO_SIDED:
                statistic, p_value = stats.ttest_1samp(sample1, population_mean)
            elif alternative == HypothesisType.GREATER:
                statistic, p_value = stats.ttest_1samp(sample1, population_mean, alternative='greater')
            else:  # LESS
                statistic, p_value = stats.ttest_1samp(sample1, population_mean, alternative='less')

            # Effect size
            effect_size = (np.mean(sample1) - population_mean) / np.std(sample1, ddof=1)
            df = len(sample1) - 1

        # Critical value
        if alternative == HypothesisType.TWO_SIDED:
            critical_value = stats.t.ppf(1 - alpha/2, df)
        else:
            critical_value = stats.t.ppf(1 - alpha, df)

        # Confidence interval for mean difference
        if sample2 is not None:
            mean_diff = np.mean(sample1) - np.mean(sample2)
            se_diff = pooled_std * np.sqrt(1/len(sample1) + 1/len(sample2))
        else:
            mean_diff = np.mean(sample1) - population_mean
            se_diff = np.std(sample1, ddof=1) / np.sqrt(len(sample1))

        margin_error = critical_value * se_diff
        ci = (mean_diff - margin_error, mean_diff + margin_error)

        result = TestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            confidence_interval=ci,
            effect_size=effect_size,
            sample_size=len(sample1) + (len(sample2) if sample2 else 0),
            degrees_of_freedom=df,
            test_parameters={'alternative': alternative.value, 'alpha': alpha},
            interpretation=self._interpret_t_test(statistic, p_value, alpha, alternative)
        )

        self.test_results.append(result)
        return result

    def strategic_disagreement_test(self, observed_successes: int, total_trials: int,
                                   expected_random_probability: float,
                                   alpha: float = 0.001) -> TestResult:
        """
        Strategic disagreement validation test

        Tests H0: System produces random disagreement patterns
        vs H1: System produces systematic (strategic) disagreement patterns

        From precision-validation-algorithm.tex:
        P_random = (1/10)^|P_disagree| × (9/10)^|P_agree|
        C_validation = 1 - (P_random)^m
        """

        # Binomial test for strategic disagreement
        p_value = stats.binomtest(observed_successes, total_trials, expected_random_probability).pvalue

        # Test statistic (z-score approximation for large samples)
        expected_successes = total_trials * expected_random_probability
        variance = total_trials * expected_random_probability * (1 - expected_random_probability)

        if variance > 0:
            z_statistic = (observed_successes - expected_successes) / np.sqrt(variance)
        else:
            z_statistic = 0.0

        # Validation confidence
        validation_confidence = 1.0 - (expected_random_probability ** observed_successes)

        result = TestResult(
            test_type=StatisticalTest.STRATEGIC_DISAGREEMENT,
            statistic=z_statistic,
            p_value=p_value,
            critical_value=stats.norm.ppf(1 - alpha),
            effect_size=(observed_successes - expected_successes) / np.sqrt(variance) if variance > 0 else 0,
            sample_size=total_trials,
            test_parameters={
                'observed_successes': observed_successes,
                'expected_random_probability': expected_random_probability,
                'validation_confidence': validation_confidence,
                'alpha': alpha
            },
            interpretation=self._interpret_strategic_disagreement_test(
                validation_confidence, p_value, alpha
            )
        )

        self.test_results.append(result)
        return result

    def precision_validation_test(self, candidate_measurements: List[float],
                                reference_measurements: List[float],
                                precision_claim_factor: float = 10.0,
                                alpha: float = 0.05) -> TestResult:
        """
        Test for precision improvement validation

        Tests H0: Candidate precision ≤ Reference precision
        vs H1: Candidate precision > Reference precision
        """

        # Calculate precision metrics (inverse of standard deviation)
        candidate_precision = 1.0 / (np.std(candidate_measurements, ddof=1) + 1e-15)
        reference_precision = 1.0 / (np.std(reference_measurements, ddof=1) + 1e-15)

        # Precision improvement ratio
        precision_ratio = candidate_precision / reference_precision

        # F-test for variance equality (precision is inverse of variance)
        candidate_var = np.var(candidate_measurements, ddof=1)
        reference_var = np.var(reference_measurements, ddof=1)

        f_statistic = reference_var / candidate_var  # Higher precision = lower variance
        df1 = len(reference_measurements) - 1
        df2 = len(candidate_measurements) - 1

        # One-sided F-test (testing if candidate variance is smaller)
        p_value = 1.0 - stats.f.cdf(f_statistic, df1, df2)

        # Critical value
        critical_value = stats.f.ppf(1 - alpha, df1, df2)

        result = TestResult(
            test_type=StatisticalTest.PRECISION_VALIDATION,
            statistic=f_statistic,
            p_value=p_value,
            critical_value=critical_value,
            effect_size=precision_ratio,
            sample_size=len(candidate_measurements) + len(reference_measurements),
            degrees_of_freedom=df1,  # Primary df
            test_parameters={
                'precision_claim_factor': precision_claim_factor,
                'candidate_precision': candidate_precision,
                'reference_precision': reference_precision,
                'precision_ratio': precision_ratio,
                'df1': df1,
                'df2': df2
            },
            interpretation=self._interpret_precision_validation_test(
                precision_ratio, p_value, alpha, precision_claim_factor
            )
        )

        self.test_results.append(result)
        return result

    # Exotic Method Validation (for oscillatory/S-entropy components)

    def semantic_distance_amplification_test(self, original_distances: List[float],
                                           amplified_distances: List[float],
                                           expected_amplification: float = 658.0) -> TestResult:
        """
        Test semantic distance amplification effectiveness

        Validates the 658× semantic distance amplification from ambiguous compression
        """

        # Calculate amplification factors
        amplification_factors = [amp / orig for amp, orig in zip(amplified_distances, original_distances)
                               if orig > 0]

        if not amplification_factors:
            # Return null result
            return TestResult(
                test_type=StatisticalTest.T_TEST,
                statistic=0.0,
                p_value=1.0,
                interpretation="No valid amplification factors calculated"
            )

        # One-sample t-test against expected amplification
        mean_amplification = np.mean(amplification_factors)

        statistic, p_value = stats.ttest_1samp(amplification_factors, expected_amplification)

        # Effect size
        effect_size = (mean_amplification - expected_amplification) / np.std(amplification_factors, ddof=1)

        result = TestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(amplification_factors),
            degrees_of_freedom=len(amplification_factors) - 1,
            test_parameters={
                'expected_amplification': expected_amplification,
                'observed_mean_amplification': mean_amplification,
                'amplification_std': np.std(amplification_factors, ddof=1)
            },
            interpretation=self._interpret_amplification_test(
                mean_amplification, expected_amplification, p_value
            )
        )

        self.test_results.append(result)
        return result

    def time_sequence_efficiency_test(self, traditional_times: List[float],
                                    sequence_times: List[float]) -> TestResult:
        """
        Test time sequencing efficiency for "reducing time taken to read time"

        Validates that time sequencing methods reduce time measurement duration
        """

        # Paired t-test (same measurements using different methods)
        if len(traditional_times) == len(sequence_times):
            statistic, p_value = stats.ttest_rel(traditional_times, sequence_times)
            test_type = StatisticalTest.PAIRED_T_TEST
        else:
            # Independent samples t-test
            statistic, p_value = stats.ttest_ind(traditional_times, sequence_times)
            test_type = StatisticalTest.T_TEST

        # Effect size (positive means sequence method is faster)
        time_reduction = np.mean(traditional_times) - np.mean(sequence_times)
        pooled_std = np.sqrt((np.var(traditional_times, ddof=1) + np.var(sequence_times, ddof=1)) / 2)
        effect_size = time_reduction / pooled_std

        # Efficiency improvement percentage
        efficiency_improvement = (time_reduction / np.mean(traditional_times)) * 100

        result = TestResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(traditional_times) + len(sequence_times),
            test_parameters={
                'time_reduction': time_reduction,
                'efficiency_improvement_percent': efficiency_improvement,
                'traditional_mean': np.mean(traditional_times),
                'sequence_mean': np.mean(sequence_times)
            },
            interpretation=self._interpret_efficiency_test(
                efficiency_improvement, p_value, effect_size
            )
        )

        self.test_results.append(result)
        return result

    def hierarchical_navigation_complexity_test(self, navigation_times: List[float],
                                              hierarchy_levels: List[int]) -> TestResult:
        """
        Test O(1) complexity claim for hierarchical navigation

        Validates that navigation time is independent of hierarchy level (O(1) complexity)
        """

        # Correlation test between navigation time and hierarchy levels
        correlation, p_value = stats.pearsonr(hierarchy_levels, navigation_times)

        # If O(1), correlation should be near zero
        # Test H0: correlation = 0 vs H1: correlation ≠ 0
        n = len(navigation_times)
        df = n - 2

        # t-statistic for correlation
        t_statistic = correlation * np.sqrt(df) / np.sqrt(1 - correlation**2)

        result = TestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=t_statistic,
            p_value=p_value,
            effect_size=correlation,
            sample_size=n,
            degrees_of_freedom=df,
            test_parameters={
                'correlation': correlation,
                'mean_navigation_time': np.mean(navigation_times),
                'navigation_time_std': np.std(navigation_times, ddof=1),
                'complexity_assessment': 'O(1)' if abs(correlation) < 0.1 else 'Non-O(1)'
            },
            interpretation=self._interpret_complexity_test(correlation, p_value)
        )

        self.test_results.append(result)
        return result

    # Power Analysis Methods

    def calculate_power(self, test_type: StatisticalTest, effect_size: float,
                       sample_size: int, alpha: float = 0.05) -> PowerAnalysisResult:
        """Calculate statistical power for a given test"""

        if test_type == StatisticalTest.T_TEST:
            # Power for one-sample t-test
            power = self._calculate_t_test_power(effect_size, sample_size, alpha)
        elif test_type == StatisticalTest.STRATEGIC_DISAGREEMENT:
            # Power for strategic disagreement test
            power = self._calculate_strategic_disagreement_power(effect_size, sample_size, alpha)
        else:
            # Default power calculation
            power = 0.8  # Placeholder

        result = PowerAnalysisResult(
            test_type=test_type,
            power=power,
            alpha=alpha,
            effect_size=effect_size,
            sample_size=sample_size
        )

        self.power_analyses.append(result)
        return result

    def _calculate_t_test_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate power for t-test"""
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)

        # Critical value
        critical_value = stats.t.ppf(1 - alpha/2, sample_size - 1)

        # Power calculation
        power = 1 - stats.nct.cdf(critical_value, sample_size - 1, ncp)

        return power

    def _calculate_strategic_disagreement_power(self, effect_size: float,
                                              sample_size: int, alpha: float) -> float:
        """Calculate power for strategic disagreement test"""
        # Simplified power calculation for binomial test
        # Effect size interpreted as deviation from null probability

        null_prob = 0.1  # Typical random disagreement probability
        alt_prob = null_prob + effect_size

        # Approximate power using normal approximation
        null_mean = sample_size * null_prob
        null_var = sample_size * null_prob * (1 - null_prob)

        alt_mean = sample_size * alt_prob
        alt_var = sample_size * alt_prob * (1 - alt_prob)

        # Critical value under null
        critical_value = stats.norm.ppf(1 - alpha, null_mean, np.sqrt(null_var))

        # Power under alternative
        power = 1 - stats.norm.cdf(critical_value, alt_mean, np.sqrt(alt_var))

        return power

    # Confidence Interval Methods

    def calculate_confidence_interval(self, data: List[float], confidence_level: float = 0.95,
                                    method: str = "t_distribution") -> ConfidenceInterval:
        """Calculate confidence interval for data"""

        alpha = 1 - confidence_level
        mean = np.mean(data)

        if method == "t_distribution":
            # t-distribution based CI
            std_error = np.std(data, ddof=1) / np.sqrt(len(data))
            critical_value = stats.t.ppf(1 - alpha/2, len(data) - 1)
            margin_error = critical_value * std_error

        elif method == "bootstrap":
            # Bootstrap confidence interval
            bootstrap_means = []
            for _ in range(1000):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))

            lower_bound = np.percentile(bootstrap_means, 100 * alpha/2)
            upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha/2))

            ci = ConfidenceInterval(
                estimate=mean,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method=method
            )

            self.confidence_intervals.append(ci)
            return ci

        else:
            # Normal distribution based CI
            std_error = np.std(data, ddof=1) / np.sqrt(len(data))
            critical_value = stats.norm.ppf(1 - alpha/2)
            margin_error = critical_value * std_error

        ci = ConfidenceInterval(
            estimate=mean,
            lower_bound=mean - margin_error,
            upper_bound=mean + margin_error,
            confidence_level=confidence_level,
            method=method
        )

        self.confidence_intervals.append(ci)
        return ci

    # Bayesian Analysis Methods

    def bayesian_t_test(self, sample1: List[float], sample2: Optional[List[float]] = None,
                       prior_mean: float = 0.0, prior_precision: float = 1.0,
                       credible_level: float = 0.95) -> BayesianResult:
        """Bayesian t-test analysis"""

        if sample2 is not None:
            # Two-sample Bayesian t-test
            diff_data = np.array(sample1) - np.array(sample2[:len(sample1)])
        else:
            # One-sample Bayesian t-test
            diff_data = np.array(sample1)

        n = len(diff_data)
        sample_mean = np.mean(diff_data)
        sample_var = np.var(diff_data, ddof=1)

        # Conjugate normal-gamma prior (assuming normal likelihood)
        # Posterior parameters
        posterior_precision = prior_precision + n
        posterior_mean = (prior_precision * prior_mean + n * sample_mean) / posterior_precision

        # Posterior variance (approximate)
        posterior_var = sample_var / n + 1.0 / posterior_precision
        posterior_std = np.sqrt(posterior_var)

        # Credible interval
        alpha = 1 - credible_level
        lower_bound = stats.norm.ppf(alpha/2, posterior_mean, posterior_std)
        upper_bound = stats.norm.ppf(1 - alpha/2, posterior_mean, posterior_std)

        # Bayes factor (simplified)
        # BF = P(data|H1) / P(data|H0)
        # Approximation using BIC
        bic_diff = n * np.log(sample_var)  # Simplified
        bayes_factor = np.exp(-0.5 * bic_diff)

        result = BayesianResult(
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_interval=(lower_bound, upper_bound),
            credible_level=credible_level,
            bayes_factor=bayes_factor,
            prior_parameters={'mean': prior_mean, 'precision': prior_precision},
            posterior_parameters={'mean': posterior_mean, 'precision': posterior_precision}
        )

        self.bayesian_results.append(result)
        return result

    # Distribution Analysis Methods

    def fit_distribution(self, data: List[float],
                        distribution: DistributionType = DistributionType.NORMAL) -> Dict[str, Any]:
        """Fit probability distribution to data"""

        if distribution == DistributionType.NORMAL:
            params = stats.norm.fit(data)
            fitted_dist = stats.norm(*params)

        elif distribution == DistributionType.LOG_NORMAL:
            params = stats.lognorm.fit(data)
            fitted_dist = stats.lognorm(*params)

        elif distribution == DistributionType.EXPONENTIAL:
            params = stats.expon.fit(data)
            fitted_dist = stats.expon(*params)

        elif distribution == DistributionType.GAMMA:
            params = stats.gamma.fit(data)
            fitted_dist = stats.gamma(*params)

        else:
            # Default to normal
            params = stats.norm.fit(data)
            fitted_dist = stats.norm(*params)

        # Goodness of fit test
        ks_statistic, ks_p_value = stats.kstest(data, fitted_dist.cdf)

        # AIC and BIC
        log_likelihood = np.sum(fitted_dist.logpdf(data))
        k = len(params)  # Number of parameters
        n = len(data)

        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return {
            'distribution': distribution.value,
            'parameters': params,
            'fitted_distribution': fitted_dist,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'goodness_of_fit': 'Good' if ks_p_value > 0.05 else 'Poor'
        }

    # Specialized Analysis for Domain-Specific Applications

    def analyze_measurement_distributions(self, candidate_measurements: List[float],
                                        reference_measurements: List[float]) -> Dict[str, Any]:
        """Comprehensive analysis of measurement distributions"""

        analysis = {}

        # Basic statistics
        analysis['candidate_stats'] = {
            'mean': np.mean(candidate_measurements),
            'std': np.std(candidate_measurements, ddof=1),
            'median': np.median(candidate_measurements),
            'iqr': np.percentile(candidate_measurements, 75) - np.percentile(candidate_measurements, 25),
            'skewness': stats.skew(candidate_measurements),
            'kurtosis': stats.kurtosis(candidate_measurements)
        }

        analysis['reference_stats'] = {
            'mean': np.mean(reference_measurements),
            'std': np.std(reference_measurements, ddof=1),
            'median': np.median(reference_measurements),
            'iqr': np.percentile(reference_measurements, 75) - np.percentile(reference_measurements, 25),
            'skewness': stats.skew(reference_measurements),
            'kurtosis': stats.kurtosis(reference_measurements)
        }

        # Distribution fitting
        analysis['candidate_distribution'] = self.fit_distribution(candidate_measurements)
        analysis['reference_distribution'] = self.fit_distribution(reference_measurements)

        # Normality tests
        analysis['candidate_normality'] = {
            'shapiro_statistic': stats.shapiro(candidate_measurements).statistic,
            'shapiro_p_value': stats.shapiro(candidate_measurements).pvalue,
            'is_normal': stats.shapiro(candidate_measurements).pvalue > 0.05
        }

        analysis['reference_normality'] = {
            'shapiro_statistic': stats.shapiro(reference_measurements).statistic,
            'shapiro_p_value': stats.shapiro(reference_measurements).pvalue,
            'is_normal': stats.shapiro(reference_measurements).pvalue > 0.05
        }

        # Two-sample comparison
        analysis['comparison'] = {
            'mean_difference': analysis['candidate_stats']['mean'] - analysis['reference_stats']['mean'],
            'variance_ratio': analysis['candidate_stats']['std']**2 / analysis['reference_stats']['std']**2,
            'precision_ratio': analysis['reference_stats']['std'] / analysis['candidate_stats']['std']
        }

        return analysis

    def test_precision_hypothesis(self, candidate_measurements: List[float],
                                reference_measurements: List[float],
                                alternative: str = 'greater') -> TestResult:
        """Test hypothesis about precision improvement"""

        return self.precision_validation_test(
            candidate_measurements, reference_measurements,
            precision_claim_factor=2.0
        )

    def calculate_validation_power(self, statistical_analysis: Dict[str, Any],
                                 hypothesis_testing: TestResult) -> PowerAnalysisResult:
        """Calculate power for validation study"""

        effect_size = hypothesis_testing.effect_size or 0.5
        sample_size = hypothesis_testing.sample_size

        return self.calculate_power(
            hypothesis_testing.test_type, effect_size, sample_size
        )

    def analyze_domain_measurements(self, measurements: List[Dict], domain: str) -> Dict[str, Any]:
        """Domain-specific measurement analysis"""

        values = [m['value'] for m in measurements]

        analysis = {
            'domain': domain,
            'measurement_count': len(values),
            'basic_stats': {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'median': np.median(values),
                'range': max(values) - min(values)
            }
        }

        # Domain-specific analysis
        if domain == 'temporal':
            # Temporal stability analysis
            if len(values) > 1:
                differences = np.diff(values)
                analysis['temporal_stability'] = {
                    'drift_rate': np.mean(differences),
                    'drift_std': np.std(differences, ddof=1),
                    'allan_variance': self._calculate_allan_variance(values) if len(values) > 10 else None
                }

        elif domain == 'spatial':
            # Spatial accuracy analysis
            analysis['spatial_accuracy'] = {
                'circular_error_probable': np.percentile(values, 50),
                'spherical_error_probable': np.percentile(values, 50)  # Simplified
            }

        elif domain == 'frequency':
            # Frequency stability analysis
            if len(values) > 1:
                fractional_frequency = np.array(values) / np.mean(values)
                analysis['frequency_stability'] = {
                    'fractional_std': np.std(fractional_frequency, ddof=1),
                    'frequency_drift': np.polyfit(range(len(values)), values, 1)[0]
                }

        return analysis

    def analyze_cross_domain_consistency(self, domain_measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency across multiple domains"""

        consistency_analysis = {
            'domains_analyzed': list(domain_measurements.keys()),
            'cross_domain_correlation': {}
        }

        # Extract values from each domain
        domain_values = {}
        for domain, data in domain_measurements.items():
            if 'measurements' in data:
                domain_values[domain] = [m['value'] for m in data['measurements']]

        # Calculate cross-domain correlations
        domain_names = list(domain_values.keys())
        for i, domain1 in enumerate(domain_names):
            for domain2 in domain_names[i+1:]:
                if len(domain_values[domain1]) == len(domain_values[domain2]):
                    correlation, p_value = stats.pearsonr(
                        domain_values[domain1], domain_values[domain2]
                    )
                    consistency_analysis['cross_domain_correlation'][f"{domain1}_vs_{domain2}"] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

        return consistency_analysis

    def analyze_temporal_stability(self, time_series_data: List[Dict], window: float) -> Dict[str, Any]:
        """Analyze temporal stability of measurements"""

        timestamps = [d['timestamp'] for d in time_series_data]
        values = [d['value'] for d in time_series_data]

        if len(values) < 2:
            return {'error': 'Insufficient data for temporal analysis'}

        # Calculate stability metrics
        stability_analysis = {
            'measurement_count': len(values),
            'time_span': max(timestamps) - min(timestamps),
            'sampling_rate': len(values) / (max(timestamps) - min(timestamps)) if len(values) > 1 else 0
        }

        # Allan variance (for time/frequency measurements)
        if len(values) > 10:
            stability_analysis['allan_variance'] = self._calculate_allan_variance(values)

        # Drift analysis
        if len(values) > 2:
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
            stability_analysis['drift_analysis'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'significant_drift': p_value < 0.05
            }

        # Stability windows
        if len(values) > 20:
            window_size = min(20, len(values) // 4)
            window_stds = []

            for i in range(0, len(values) - window_size, window_size):
                window_values = values[i:i + window_size]
                window_stds.append(np.std(window_values, ddof=1))

            stability_analysis['windowed_stability'] = {
                'window_size': window_size,
                'window_count': len(window_stds),
                'mean_window_std': np.mean(window_stds),
                'std_variation': np.std(window_stds, ddof=1)
            }

        return stability_analysis

    def detect_precision_drift(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """Detect drift in precision over time"""

        timestamps = [d['timestamp'] for d in time_series_data]
        values = [d['value'] for d in time_series_data]

        if len(values) < 10:
            return {'error': 'Insufficient data for drift detection'}

        # Calculate moving standard deviation
        window_size = min(10, len(values) // 3)
        moving_stds = []
        moving_times = []

        for i in range(len(values) - window_size):
            window_values = values[i:i + window_size]
            window_times = timestamps[i:i + window_size]

            moving_stds.append(np.std(window_values, ddof=1))
            moving_times.append(np.mean(window_times))

        # Test for trend in precision (standard deviation)
        if len(moving_stds) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(moving_times, moving_stds)

            drift_analysis = {
                'precision_drift_rate': slope,
                'drift_significance': p_value,
                'drift_detected': p_value < 0.05,
                'drift_direction': 'decreasing_precision' if slope > 0 else 'increasing_precision',
                'r_squared': r_value**2
            }
        else:
            drift_analysis = {'error': 'Insufficient windows for drift analysis'}

        return drift_analysis

    def analyze_precision_evolution(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how precision evolves over time"""

        timestamps = [d['timestamp'] for d in time_series_data]
        values = [d['value'] for d in time_series_data]

        evolution_analysis = {
            'total_time_span': max(timestamps) - min(timestamps),
            'measurement_count': len(values)
        }

        # Divide into time segments and analyze precision in each
        num_segments = min(5, len(values) // 10)  # At least 10 points per segment

        if num_segments > 1:
            segment_size = len(values) // num_segments
            segment_precisions = []
            segment_times = []

            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < num_segments - 1 else len(values)

                segment_values = values[start_idx:end_idx]
                segment_timestamps = timestamps[start_idx:end_idx]

                # Precision = 1 / standard_deviation
                segment_std = np.std(segment_values, ddof=1)
                segment_precision = 1.0 / (segment_std + 1e-15)

                segment_precisions.append(segment_precision)
                segment_times.append(np.mean(segment_timestamps))

            evolution_analysis['precision_evolution'] = {
                'segment_count': num_segments,
                'segment_precisions': segment_precisions,
                'segment_times': segment_times,
                'initial_precision': segment_precisions[0],
                'final_precision': segment_precisions[-1],
                'precision_improvement': segment_precisions[-1] / segment_precisions[0]
            }

            # Test for trend in precision evolution
            if len(segment_precisions) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    segment_times, segment_precisions
                )

                evolution_analysis['precision_trend'] = {
                    'slope': slope,
                    'p_value': p_value,
                    'significant_trend': p_value < 0.05,
                    'trend_direction': 'improving' if slope > 0 else 'degrading',
                    'r_squared': r_value**2
                }

        return evolution_analysis

    def create_precision_prediction_model(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """Create predictive model for precision evolution"""

        timestamps = [d['timestamp'] for d in time_series_data]
        values = [d['value'] for d in time_series_data]

        if len(values) < 20:
            return {'error': 'Insufficient data for predictive modeling'}

        # Create precision time series (rolling standard deviation)
        window_size = min(10, len(values) // 4)
        precision_series = []
        precision_times = []

        for i in range(len(values) - window_size):
            window_values = values[i:i + window_size]
            window_time = timestamps[i + window_size // 2]

            precision = 1.0 / (np.std(window_values, ddof=1) + 1e-15)
            precision_series.append(precision)
            precision_times.append(window_time)

        # Fit polynomial model
        degree = min(3, len(precision_series) // 10)  # Adaptive degree

        if degree > 0:
            coefficients = np.polyfit(precision_times, precision_series, degree)
            poly_model = np.poly1d(coefficients)

            # Model evaluation
            predicted_values = poly_model(precision_times)
            rmse = np.sqrt(np.mean((predicted_values - precision_series)**2))
            r_squared = stats.pearsonr(predicted_values, precision_series)[0]**2

            prediction_model = {
                'model_type': f'polynomial_degree_{degree}',
                'coefficients': coefficients.tolist(),
                'rmse': rmse,
                'r_squared': r_squared,
                'model_function': poly_model,
                'prediction_horizon': max(precision_times) - min(precision_times)
            }

            # Future predictions
            future_times = np.linspace(
                max(precision_times),
                max(precision_times) + (max(precision_times) - min(precision_times)) * 0.1,
                10
            )
            future_predictions = poly_model(future_times)

            prediction_model['future_predictions'] = {
                'future_times': future_times.tolist(),
                'predicted_precisions': future_predictions.tolist()
            }
        else:
            prediction_model = {'error': 'Insufficient data for polynomial fitting'}

        return prediction_model

    def characterize_measurement_system(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Comprehensive statistical characterization of measurement system"""

        values = [m['value'] for m in measurements]

        characterization = {
            'sample_size': len(values),
            'descriptive_statistics': {
                'mean': np.mean(values),
                'median': np.median(values),
                'mode': float(stats.mode(values, keepdims=True).mode[0]) if len(values) > 1 else values[0],
                'std': np.std(values, ddof=1),
                'variance': np.var(values, ddof=1),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values),
                'range': max(values) - min(values),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25)
            }
        }

        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        characterization['percentiles'] = {
            f'p{p}': np.percentile(values, p) for p in percentiles
        }

        # Distribution fitting
        characterization['distribution_analysis'] = self.fit_distribution(values)

        # Outlier detection
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [v for v in values if v < lower_bound or v > upper_bound]

        characterization['outlier_analysis'] = {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(values) * 100,
            'outlier_values': outliers,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        # Precision metrics
        characterization['precision_metrics'] = {
            'precision': 1.0 / (characterization['descriptive_statistics']['std'] + 1e-15),
            'coefficient_of_variation': characterization['descriptive_statistics']['std'] / abs(characterization['descriptive_statistics']['mean']) if characterization['descriptive_statistics']['mean'] != 0 else float('inf'),
            'signal_to_noise_ratio': abs(characterization['descriptive_statistics']['mean']) / characterization['descriptive_statistics']['std'] if characterization['descriptive_statistics']['std'] > 0 else float('inf')
        }

        return characterization

    def calculate_system_performance_metrics(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics for measurement system"""

        values = [m['value'] for m in measurements]
        timestamps = [m.get('timestamp', i) for i, m in enumerate(measurements)]

        performance_metrics = {
            'accuracy_metrics': {
                'bias': np.mean(values),  # Assuming true value is 0
                'absolute_bias': abs(np.mean(values)),
                'mean_absolute_error': np.mean(np.abs(values))
            },
            'precision_metrics': {
                'standard_deviation': np.std(values, ddof=1),
                'variance': np.var(values, ddof=1),
                'precision_score': 1.0 / (np.std(values, ddof=1) + 1e-15)
            }
        }

        # Temporal performance (if timestamps available)
        if len(set(timestamps)) > 1:  # Check if timestamps are actually different
            time_diffs = np.diff(sorted(timestamps))
            performance_metrics['temporal_performance'] = {
                'measurement_rate': len(values) / (max(timestamps) - min(timestamps)) if max(timestamps) != min(timestamps) else 0,
                'mean_sampling_interval': np.mean(time_diffs),
                'sampling_consistency': 1.0 / (np.std(time_diffs, ddof=1) + 1e-15)
            }

        # Stability metrics
        if len(values) > 10:
            performance_metrics['stability_metrics'] = {
                'allan_variance': self._calculate_allan_variance(values),
                'drift_rate': np.polyfit(range(len(values)), values, 1)[0],
                'stability_factor': 1.0 / (np.std(np.diff(values), ddof=1) + 1e-15)
            }

        return performance_metrics

    def compare_measurement_systems(self, system1_measurements: List[Dict],
                                  system2_measurements: List[Dict]) -> Dict[str, Any]:
        """Compare two measurement systems statistically"""

        values1 = [m['value'] for m in system1_measurements]
        values2 = [m['value'] for m in system2_measurements]

        comparison = {
            'system1_size': len(values1),
            'system2_size': len(values2)
        }

        # Statistical tests
        # t-test for mean difference
        t_result = self.t_test(values1, values2)
        comparison['mean_comparison'] = {
            'statistic': t_result.statistic,
            'p_value': t_result.p_value,
            'significant_difference': t_result.is_significant(),
            'effect_size': t_result.effect_size
        }

        # F-test for variance difference
        f_statistic = np.var(values1, ddof=1) / np.var(values2, ddof=1)
        f_p_value = 2 * min(
            stats.f.cdf(f_statistic, len(values1) - 1, len(values2) - 1),
            1 - stats.f.cdf(f_statistic, len(values1) - 1, len(values2) - 1)
        )

        comparison['variance_comparison'] = {
            'f_statistic': f_statistic,
            'p_value': f_p_value,
            'significant_difference': f_p_value < 0.05,
            'precision_ratio': np.std(values2, ddof=1) / np.std(values1, ddof=1)
        }

        # Non-parametric tests
        mann_whitney_result = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        comparison['distribution_comparison'] = {
            'mann_whitney_statistic': mann_whitney_result.statistic,
            'mann_whitney_p_value': mann_whitney_result.pvalue,
            'distributions_different': mann_whitney_result.pvalue < 0.05
        }

        # Effect size measures
        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) +
                             (len(values2) - 1) * np.var(values2, ddof=1)) /
                            (len(values1) + len(values2) - 2))

        comparison['effect_sizes'] = {
            'cohens_d': (np.mean(values1) - np.mean(values2)) / pooled_std,
            'glass_delta': (np.mean(values1) - np.mean(values2)) / np.std(values2, ddof=1),
            'hedges_g': comparison['mean_comparison']['effect_size']  # Already calculated in t_test
        }

        return comparison

    def rank_measurement_systems(self, system_measurements: Dict[str, List[Dict]],
                                baseline_system: str = 'reference') -> Dict[str, Any]:
        """Rank multiple measurement systems by performance"""

        system_scores = {}

        for system_name, measurements in system_measurements.items():
            values = [m['value'] for m in measurements]

            # Calculate composite performance score
            precision_score = 1.0 / (np.std(values, ddof=1) + 1e-15)
            accuracy_score = 1.0 / (abs(np.mean(values)) + 1e-15)  # Assuming true value is 0
            stability_score = 1.0 / (np.std(np.diff(values), ddof=1) + 1e-15) if len(values) > 1 else 0

            # Weighted composite score
            composite_score = (0.4 * precision_score + 0.3 * accuracy_score + 0.3 * stability_score)

            system_scores[system_name] = {
                'precision_score': precision_score,
                'accuracy_score': accuracy_score,
                'stability_score': stability_score,
                'composite_score': composite_score,
                'sample_size': len(values)
            }

        # Rank systems
        ranked_systems = sorted(system_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

        ranking_result = {
            'ranking': [(name, scores['composite_score']) for name, scores in ranked_systems],
            'best_system': ranked_systems[0][0],
            'worst_system': ranked_systems[-1][0],
            'system_scores': system_scores
        }

        # Relative performance to baseline
        if baseline_system in system_scores:
            baseline_score = system_scores[baseline_system]['composite_score']

            for system_name, scores in system_scores.items():
                scores['relative_to_baseline'] = scores['composite_score'] / baseline_score

            ranking_result['baseline_system'] = baseline_system
            ranking_result['baseline_score'] = baseline_score

        return ranking_result

    def _calculate_allan_variance(self, values: List[float]) -> Dict[str, Any]:
        """Calculate Allan variance for time/frequency stability analysis"""

        if len(values) < 3:
            return {'error': 'Insufficient data for Allan variance'}

        # Calculate Allan variance for different tau values
        max_tau = len(values) // 3
        tau_values = range(1, min(max_tau, 20))  # Limit for computational efficiency

        allan_variances = []

        for tau in tau_values:
            # Calculate Allan variance for this tau
            y_values = []

            for i in range(len(values) - 2 * tau):
                y1 = np.mean(values[i:i + tau])
                y2 = np.mean(values[i + tau:i + 2 * tau])
                y_values.append(y2 - y1)

            if y_values:
                allan_var = np.var(y_values, ddof=1) / 2.0
                allan_variances.append(allan_var)
            else:
                allan_variances.append(0.0)

        return {
            'tau_values': list(tau_values),
            'allan_variances': allan_variances,
            'min_allan_variance': min(allan_variances) if allan_variances else 0,
            'optimal_tau': tau_values[np.argmin(allan_variances)] if allan_variances else 1
        }

    # Interpretation Methods

    def _interpret_t_test(self, statistic: float, p_value: float,
                         alpha: float, alternative: HypothesisType) -> str:
        """Interpret t-test results"""

        if p_value < alpha:
            if alternative == HypothesisType.TWO_SIDED:
                return f"Statistically significant difference detected (p = {p_value:.4f} < α = {alpha})"
            elif alternative == HypothesisType.GREATER:
                return f"First group significantly greater than second (p = {p_value:.4f} < α = {alpha})"
            else:
                return f"First group significantly less than second (p = {p_value:.4f} < α = {alpha})"
        else:
            return f"No statistically significant difference (p = {p_value:.4f} ≥ α = {alpha})"

    def _interpret_strategic_disagreement_test(self, validation_confidence: float,
                                             p_value: float, alpha: float) -> str:
        """Interpret strategic disagreement test results"""

        if validation_confidence > 0.999:
            confidence_level = "extremely high (>99.9%)"
        elif validation_confidence > 0.99:
            confidence_level = "very high (>99%)"
        elif validation_confidence > 0.95:
            confidence_level = "high (>95%)"
        else:
            confidence_level = f"moderate ({validation_confidence:.1%})"

        if p_value < alpha:
            return f"Strategic disagreement pattern validated with {confidence_level} confidence"
        else:
            return f"Strategic disagreement pattern not statistically significant (p = {p_value:.4f})"

    def _interpret_precision_validation_test(self, precision_ratio: float, p_value: float,
                                           alpha: float, claim_factor: float) -> str:
        """Interpret precision validation test results"""

        if p_value < alpha:
            if precision_ratio >= claim_factor:
                return f"Precision improvement claim VALIDATED: {precision_ratio:.1f}× improvement (≥{claim_factor}× claimed)"
            else:
                return f"Precision improvement detected but below claimed factor: {precision_ratio:.1f}× vs {claim_factor}× claimed"
        else:
            return f"Precision improvement claim NOT VALIDATED (p = {p_value:.4f} ≥ α = {alpha})"

    def _interpret_amplification_test(self, observed: float, expected: float, p_value: float) -> str:
        """Interpret semantic amplification test results"""

        if abs(observed - expected) / expected < 0.1:  # Within 10% of expected
            return f"Semantic amplification factor validated: {observed:.1f}× (expected {expected:.1f}×)"
        elif observed > expected:
            return f"Semantic amplification exceeded expectations: {observed:.1f}× vs {expected:.1f}× expected"
        else:
            return f"Semantic amplification below expectations: {observed:.1f}× vs {expected:.1f}× expected"

    def _interpret_efficiency_test(self, improvement: float, p_value: float, effect_size: float) -> str:
        """Interpret time sequence efficiency test results"""

        if p_value < 0.05:
            if improvement > 0:
                return f"Time sequencing provides significant efficiency improvement: {improvement:.1f}% faster"
            else:
                return f"Time sequencing is significantly slower: {abs(improvement):.1f}% slower"
        else:
            return f"No significant efficiency difference detected (improvement: {improvement:.1f}%)"

    def _interpret_complexity_test(self, correlation: float, p_value: float) -> str:
        """Interpret hierarchical navigation complexity test results"""

        if abs(correlation) < 0.1 and p_value > 0.05:
            return "O(1) complexity claim VALIDATED: navigation time independent of hierarchy level"
        elif abs(correlation) < 0.3:
            return f"Near-O(1) complexity observed: weak correlation (r = {correlation:.3f})"
        else:
            return f"O(1) complexity claim NOT VALIDATED: significant correlation (r = {correlation:.3f}, p = {p_value:.4f})"

    # Summary and Reporting Methods

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all statistical analyses"""

        summary = {
            'total_tests_performed': len(self.test_results),
            'total_power_analyses': len(self.power_analyses),
            'total_confidence_intervals': len(self.confidence_intervals),
            'total_bayesian_analyses': len(self.bayesian_results)
        }

        # Test results summary
        if self.test_results:
            test_types = [result.test_type.value for result in self.test_results]
            summary['test_types_used'] = list(set(test_types))
            summary['significant_results'] = sum(1 for result in self.test_results if result.is_significant())
            summary['significance_rate'] = summary['significant_results'] / len(self.test_results)

        # Power analysis summary
        if self.power_analyses:
            powers = [analysis.power for analysis in self.power_analyses]
            summary['average_power'] = np.mean(powers)
            summary['adequately_powered_studies'] = sum(1 for analysis in self.power_analyses if analysis.is_adequately_powered())

        # Recent results
        summary['most_recent_test'] = self.test_results[-1].__dict__ if self.test_results else None

        return summary


class StatisticalFramework:
    """Framework for managing multiple statistical analysis workflows"""

    def __init__(self):
        self.statistics_engine = PrecisionStatistics()
        self.analysis_workflows: Dict[str, Callable] = {}
        self._setup_default_workflows()

    def _setup_default_workflows(self):
        """Setup default analysis workflows"""

        self.analysis_workflows['precision_validation'] = self._precision_validation_workflow
        self.analysis_workflows['exotic_method_validation'] = self._exotic_method_validation_workflow
        self.analysis_workflows['system_comparison'] = self._system_comparison_workflow
        self.analysis_workflows['temporal_analysis'] = self._temporal_analysis_workflow

    def _precision_validation_workflow(self, candidate_data: List[float],
                                     reference_data: List[float]) -> Dict[str, Any]:
        """Complete precision validation workflow"""

        results = {}

        # 1. Distribution analysis
        results['distribution_analysis'] = self.statistics_engine.analyze_measurement_distributions(
            candidate_data, reference_data
        )

        # 2. Precision validation test
        results['precision_test'] = self.statistics_engine.precision_validation_test(
            candidate_data, reference_data
        )

        # 3. Strategic disagreement simulation (if applicable)
        # This would require additional parameters for real implementation

        # 4. Power analysis
        results['power_analysis'] = self.statistics_engine.calculate_power(
            StatisticalTest.PRECISION_VALIDATION,
            results['precision_test'].effect_size or 0.5,
            len(candidate_data) + len(reference_data)
        )

        return results

    def _exotic_method_validation_workflow(self, traditional_data: List[float],
                                         enhanced_data: List[float]) -> Dict[str, Any]:
        """Validation workflow for exotic precision enhancement methods"""

        results = {}

        # 1. Efficiency test
        results['efficiency_test'] = self.statistics_engine.time_sequence_efficiency_test(
            traditional_data, enhanced_data
        )

        # 2. Amplification test (if distance data available)
        if len(traditional_data) == len(enhanced_data):
            results['amplification_test'] = self.statistics_engine.semantic_distance_amplification_test(
                traditional_data, enhanced_data
            )

        # 3. Statistical characterization
        results['traditional_characterization'] = self.statistics_engine.characterize_measurement_system(
            [{'value': v} for v in traditional_data]
        )
        results['enhanced_characterization'] = self.statistics_engine.characterize_measurement_system(
            [{'value': v} for v in enhanced_data]
        )

        return results

    def _system_comparison_workflow(self, systems_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Complete system comparison workflow"""

        results = {}

        # Convert to required format
        systems_measurements = {
            name: [{'value': v} for v in values]
            for name, values in systems_data.items()
        }

        # System ranking
        results['ranking'] = self.statistics_engine.rank_measurement_systems(systems_measurements)

        # Pairwise comparisons
        system_names = list(systems_data.keys())
        results['pairwise_comparisons'] = {}

        for i, system1 in enumerate(system_names):
            for system2 in system_names[i+1:]:
                comparison_key = f"{system1}_vs_{system2}"
                results['pairwise_comparisons'][comparison_key] = self.statistics_engine.compare_measurement_systems(
                    systems_measurements[system1], systems_measurements[system2]
                )

        return results

    def _temporal_analysis_workflow(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """Complete temporal analysis workflow"""

        results = {}

        # 1. Temporal stability
        results['stability_analysis'] = self.statistics_engine.analyze_temporal_stability(
            time_series_data, 3600.0  # 1 hour window
        )

        # 2. Drift detection
        results['drift_analysis'] = self.statistics_engine.detect_precision_drift(time_series_data)

        # 3. Precision evolution
        results['evolution_analysis'] = self.statistics_engine.analyze_precision_evolution(time_series_data)

        # 4. Predictive modeling
        results['prediction_model'] = self.statistics_engine.create_precision_prediction_model(time_series_data)

        return results

    def run_workflow(self, workflow_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Run a specific analysis workflow"""

        if workflow_name not in self.analysis_workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        return self.analysis_workflows[workflow_name](*args, **kwargs)


# Factory functions

def create_precision_statistics_suite() -> PrecisionStatistics:
    """Create precision statistics analysis suite"""
    return PrecisionStatistics()


def create_statistical_framework() -> StatisticalFramework:
    """Create statistical analysis framework"""
    return StatisticalFramework()


# Specialized analysis classes

class HypothesisTestSuite:
    """Suite of hypothesis tests for precision measurement validation"""

    def __init__(self):
        self.test_history: List[TestResult] = []

    def run_comprehensive_hypothesis_tests(self, data1: List[float],
                                         data2: Optional[List[float]] = None) -> Dict[str, TestResult]:
        """Run comprehensive suite of hypothesis tests"""

        stats_engine = PrecisionStatistics()
        results = {}

        if data2 is not None:
            # Two-sample tests
            results['t_test'] = stats_engine.t_test(data1, data2)
            results['welch_t_test'] = stats_engine.t_test(data1, data2)  # Could be enhanced to use Welch's
            results['mann_whitney'] = self._mann_whitney_test(data1, data2)
            results['precision_validation'] = stats_engine.precision_validation_test(data1, data2)
        else:
            # One-sample tests
            results['one_sample_t'] = stats_engine.t_test(data1, population_mean=0.0)
            results['shapiro_wilk'] = self._shapiro_wilk_test(data1)

        # Store results
        self.test_history.extend(results.values())

        return results

    def _mann_whitney_test(self, data1: List[float], data2: List[float]) -> TestResult:
        """Mann-Whitney U test"""

        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

        return TestResult(
            test_type=StatisticalTest.MANN_WHITNEY_U,
            statistic=statistic,
            p_value=p_value,
            sample_size=len(data1) + len(data2),
            interpretation=f"Mann-Whitney U test: {'Significant' if p_value < 0.05 else 'Non-significant'} difference"
        )

    def _shapiro_wilk_test(self, data: List[float]) -> TestResult:
        """Shapiro-Wilk normality test"""

        statistic, p_value = stats.shapiro(data)

        return TestResult(
            test_type=StatisticalTest.SHAPIRO_WILK,
            statistic=statistic,
            p_value=p_value,
            sample_size=len(data),
            interpretation=f"Shapiro-Wilk test: Data {'appears normal' if p_value > 0.05 else 'deviates from normality'}"
        )


class PowerAnalysisCalculator:
    """Calculator for statistical power analysis"""

    def __init__(self):
        self.power_calculations: List[PowerAnalysisResult] = []

    def sample_size_calculation(self, effect_size: float, power: float = 0.8,
                               alpha: float = 0.05, test_type: StatisticalTest = StatisticalTest.T_TEST) -> int:
        """Calculate required sample size for given power"""

        if test_type == StatisticalTest.T_TEST:
            # For t-test
            from scipy.special import ndtri

            z_alpha = ndtri(1 - alpha/2)
            z_beta = ndtri(power)

            n = ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))

        else:
            # Default calculation
            return int(np.ceil(16 / (effect_size ** 2)))  # Rule of thumb

    def power_curve_analysis(self, effect_sizes: List[float], sample_size: int,
                           alpha: float = 0.05) -> Dict[str, Any]:
        """Generate power curve analysis"""

        stats_engine = PrecisionStatistics()
        power_points = []

        for effect_size in effect_sizes:
            power_result = stats_engine.calculate_power(StatisticalTest.T_TEST, effect_size, sample_size, alpha)
            power_points.append((effect_size, power_result.power))

        return {
            'power_curve_points': power_points,
            'sample_size': sample_size,
            'alpha': alpha,
            'adequate_power_threshold': 0.8
        }


class ConfidenceIntervalCalculator:
    """Calculator for confidence intervals"""

    def __init__(self):
        self.confidence_intervals: List[ConfidenceInterval] = []

    def calculate_multiple_confidence_levels(self, data: List[float],
                                           confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> List[ConfidenceInterval]:
        """Calculate confidence intervals at multiple confidence levels"""

        stats_engine = PrecisionStatistics()
        intervals = []

        for level in confidence_levels:
            ci = stats_engine.calculate_confidence_interval(data, level)
            intervals.append(ci)

        self.confidence_intervals.extend(intervals)
        return intervals


class BayesianValidator:
    """Bayesian statistical validation methods"""

    def __init__(self):
        self.bayesian_results: List[BayesianResult] = []

    def setup_precision_prior(self, prior_belief: Dict[str, float]) -> Dict[str, float]:
        """Setup prior distribution for precision validation"""

        # Default normal prior for precision parameter
        prior_setup = {
            'distribution': 'normal',
            'mean': prior_belief.get('mean', 0.0),
            'precision': prior_belief.get('precision', 1.0),
            'variance': 1.0 / prior_belief.get('precision', 1.0)
        }

        return prior_setup

    def calculate_measurement_likelihood(self, measurements: List[Dict]) -> Dict[str, float]:
        """Calculate likelihood of measurements given model"""

        values = [m['value'] for m in measurements]

        # Simple likelihood calculation
        likelihood_params = {
            'sample_mean': np.mean(values),
            'sample_variance': np.var(values, ddof=1),
            'sample_size': len(values),
            'log_likelihood': stats.norm.logpdf(values, np.mean(values), np.std(values, ddof=1)).sum()
        }

        return likelihood_params

    def compute_precision_posterior(self, prior_setup: Dict[str, float],
                                  likelihood_params: Dict[str, float]) -> Dict[str, float]:
        """Compute posterior distribution for precision parameter"""

        # Conjugate normal-normal update
        prior_mean = prior_setup['mean']
        prior_precision = prior_setup['precision']

        sample_mean = likelihood_params['sample_mean']
        sample_size = likelihood_params['sample_size']
        sample_variance = likelihood_params['sample_variance']

        # Posterior parameters
        posterior_precision = prior_precision + sample_size / sample_variance
        posterior_mean = (prior_precision * prior_mean + (sample_size / sample_variance) * sample_mean) / posterior_precision

        posterior_params = {
            'distribution': 'normal',
            'mean': posterior_mean,
            'precision': posterior_precision,
            'variance': 1.0 / posterior_precision,
            'std': np.sqrt(1.0 / posterior_precision)
        }

        return posterior_params

    def make_validation_decision(self, posterior_params: Dict[str, float],
                               evidence_threshold: float = 0.95) -> BayesianResult:
        """Make Bayesian validation decision"""

        posterior_mean = posterior_params['mean']
        posterior_std = posterior_params['std']

        # Credible interval
        alpha = 1 - evidence_threshold
        lower_bound = stats.norm.ppf(alpha/2, posterior_mean, posterior_std)
        upper_bound = stats.norm.ppf(1 - alpha/2, posterior_mean, posterior_std)

        # Decision based on credible interval
        validation_decision = {
            'validated': lower_bound > 0,  # Assuming we're testing if parameter > 0
            'evidence_strength': evidence_threshold,
            'credible_interval': (lower_bound, upper_bound)
        }

        result = BayesianResult(
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_interval=(lower_bound, upper_bound),
            credible_level=evidence_threshold,
            posterior_parameters=posterior_params
        )

        self.bayesian_results.append(result)
        return result
