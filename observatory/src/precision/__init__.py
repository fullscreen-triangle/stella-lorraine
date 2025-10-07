"""
Precision Module - Statistical Validation and Analysis Framework

This module imports all statistics and validation algorithms and packages
some of them for sequences of analysis. Instead of importing each function
individually, common analysis sequences are packaged for convenience.

Components:
- Strategic Disagreement Validation (SDV) framework
- Comprehensive statistical analysis methods for precision measurements
- Validation algorithms for ground truth-free precision verification
- Hypothesis testing and power analysis for measurement systems
"""

# Import validation framework components
from .validation import (
    ValidationFramework,
    StrategicDisagreementValidator,
    ConsensusCalculator,
    AgreementAnalyzer,
    ValidationConfidenceCalculator,
    ValidationAlgorithm,
    # Enums and types
    ValidationMethod,
    MeasurementSystem,
    DisagreementType,
    ValidationResult,
    MeasurementRecord,
    StrategicDisagreementPattern,
    # Factory function
    create_validation_framework
)

# Import statistical analysis components
from .statistics import (
    PrecisionStatistics,
    StatisticalFramework,
    HypothesisTestSuite,
    PowerAnalysisCalculator,
    ConfidenceIntervalCalculator,
    BayesianValidator,
    # Enums and types
    StatisticalTest,
    HypothesisType,
    DistributionType,
    SignificanceLevel,
    TestResult,
    PowerAnalysisResult,
    ConfidenceInterval,
    BayesianResult,
    # Factory function
    create_precision_statistics_suite
)

def create_strategic_disagreement_validator():
    """Create strategic disagreement validator for precision validation"""
    return StrategicDisagreementValidator()

def create_complete_validation_suite():
    """
    Create complete validation suite with all statistical and validation components

    Returns:
        dict: Complete validation system with statistical analysis and SDV
    """

    # Create core validation components
    validator = create_strategic_disagreement_validator()
    statistics_suite = create_precision_statistics_suite()
    validation_framework = create_validation_framework()

    # Create integrated validation suite
    validation_suite = {
        'strategic_validator': validator,
        'statistics_suite': statistics_suite,
        'validation_framework': validation_framework,
        'suite_type': 'complete_validation',
        'creation_timestamp': __import__('time').time()
    }

    # Setup packaged analysis sequences
    validation_suite['analysis_packages'] = create_analysis_packages(
        validator, statistics_suite, validation_framework
    )

    return validation_suite

def create_analysis_packages(validator, statistics_suite, validation_framework):
    """
    Create packaged analysis sequences for common validation workflows

    This packages multiple analysis steps into single function calls for convenience.
    """

    packages = {}

    # Package 1: Complete Precision Validation Sequence
    def complete_precision_validation(candidate_measurements, reference_measurements,
                                    predicted_disagreement_positions, confidence_target=0.999):
        """
        Complete precision validation sequence:
        1. Statistical analysis of measurement distributions
        2. Hypothesis testing for precision claims
        3. Strategic disagreement pattern creation and validation
        4. Confidence calculation and reporting
        """

        results = {'validation_sequence': 'complete_precision_validation'}

        # Step 1: Statistical Analysis
        results['statistical_analysis'] = statistics_suite.analyze_measurement_distributions(
            candidate_measurements, reference_measurements
        )

        # Step 2: Hypothesis Testing
        results['hypothesis_testing'] = statistics_suite.test_precision_hypothesis(
            candidate_measurements, reference_measurements,
            alternative='greater'  # Testing if candidate > reference precision
        )

        # Step 3: Strategic Disagreement Validation
        pattern_id = f"precision_validation_{int(__import__('time').time())}"

        # Create strategic pattern
        validator.create_strategic_disagreement_pattern(
            pattern_id=pattern_id,
            candidate_system=MeasurementSystem.CANDIDATE_SYSTEM,
            reference_systems=[MeasurementSystem.REFERENCE_CONSENSUS],
            predicted_disagreement_positions=predicted_disagreement_positions,
            disagreement_type=DisagreementType.POSITION_SPECIFIC
        )

        # Add measurements to validator
        for measurement in candidate_measurements:
            validator.add_measurement_record(
                MeasurementSystem.CANDIDATE_SYSTEM,
                measurement['value'],
                measurement.get('precision_digits', 15),
                measurement.get('uncertainty', 1e-15)
            )

        for measurement in reference_measurements:
            validator.add_measurement_record(
                MeasurementSystem.REFERENCE_CONSENSUS,
                measurement['value'],
                measurement.get('precision_digits', 12),
                measurement.get('uncertainty', 1e-12)
            )

        # Validate strategic disagreement
        results['strategic_validation'] = validator.validate_strategic_disagreement_pattern(pattern_id)

        # Step 4: Comprehensive Confidence Calculation
        results['validation_confidence'] = results['strategic_validation'].validation_confidence
        results['meets_confidence_target'] = results['validation_confidence'] >= confidence_target

        # Step 5: Power Analysis
        results['power_analysis'] = statistics_suite.calculate_validation_power(
            results['statistical_analysis'], results['hypothesis_testing']
        )

        return results

    packages['complete_precision_validation'] = complete_precision_validation

    # Package 2: Multi-Domain Validation Sequence
    def multi_domain_validation(domain_measurements, confidence_target=0.999):
        """
        Multi-domain validation sequence for temporal, spatial, frequency domains:
        1. Domain-specific statistical analysis
        2. Cross-domain consistency checking
        3. Strategic disagreement validation per domain
        4. Overall validation confidence calculation
        """

        results = {'validation_sequence': 'multi_domain_validation', 'domains': {}}

        overall_confidences = []

        for domain_name, domain_data in domain_measurements.items():
            domain_results = {}

            # Domain-specific analysis
            domain_results['statistical_analysis'] = statistics_suite.analyze_domain_measurements(
                domain_data['measurements'], domain_name
            )

            # Strategic validation for this domain
            if 'predicted_positions' in domain_data:
                pattern_id = f"{domain_name}_validation_{int(__import__('time').time())}"

                validator.create_strategic_disagreement_pattern(
                    pattern_id=pattern_id,
                    candidate_system=MeasurementSystem.CANDIDATE_SYSTEM,
                    reference_systems=domain_data.get('reference_systems', [MeasurementSystem.REFERENCE_CONSENSUS]),
                    predicted_disagreement_positions=domain_data['predicted_positions']
                )

                # Add measurements for this domain
                for measurement in domain_data['measurements']:
                    validator.add_measurement_record(
                        measurement.get('system_type', MeasurementSystem.CANDIDATE_SYSTEM),
                        measurement['value'],
                        measurement.get('precision_digits', 15),
                        measurement.get('uncertainty', 1e-15)
                    )

                domain_results['strategic_validation'] = validator.validate_strategic_disagreement_pattern(pattern_id)
                overall_confidences.append(domain_results['strategic_validation'].validation_confidence)

            results['domains'][domain_name] = domain_results

        # Cross-domain consistency analysis
        if len(overall_confidences) > 1:
            results['cross_domain_consistency'] = statistics_suite.analyze_cross_domain_consistency(
                domain_measurements
            )

        # Overall validation confidence (geometric mean for conservative estimate)
        if overall_confidences:
            import numpy as np
            results['overall_validation_confidence'] = float(np.prod(overall_confidences) ** (1.0 / len(overall_confidences)))
            results['meets_confidence_target'] = results['overall_validation_confidence'] >= confidence_target

        return results

    packages['multi_domain_validation'] = multi_domain_validation

    # Package 3: Time Series Precision Analysis
    def time_series_precision_analysis(time_series_data, analysis_window=3600.0):
        """
        Time series precision analysis sequence:
        1. Temporal stability analysis
        2. Drift detection and quantification
        3. Precision evolution over time
        4. Predictive precision modeling
        """

        results = {'validation_sequence': 'time_series_precision_analysis'}

        # Temporal stability analysis
        results['stability_analysis'] = statistics_suite.analyze_temporal_stability(
            time_series_data, analysis_window
        )

        # Drift detection
        results['drift_analysis'] = statistics_suite.detect_precision_drift(
            time_series_data
        )

        # Precision evolution
        results['precision_evolution'] = statistics_suite.analyze_precision_evolution(
            time_series_data
        )

        # Predictive modeling
        results['predictive_model'] = statistics_suite.create_precision_prediction_model(
            time_series_data
        )

        return results

    packages['time_series_precision_analysis'] = time_series_precision_analysis

    # Package 4: Comparative System Analysis
    def comparative_system_analysis(system_measurements, baseline_system='reference'):
        """
        Comparative analysis between multiple measurement systems:
        1. System-by-system statistical characterization
        2. Pairwise comparison analysis
        3. Ranking and performance metrics
        4. Relative validation confidence
        """

        results = {'validation_sequence': 'comparative_system_analysis', 'systems': {}}

        system_names = list(system_measurements.keys())

        # System-by-system analysis
        for system_name, measurements in system_measurements.items():
            system_results = {}

            # Statistical characterization
            system_results['statistical_profile'] = statistics_suite.characterize_measurement_system(
                measurements
            )

            # Performance metrics
            system_results['performance_metrics'] = statistics_suite.calculate_system_performance_metrics(
                measurements
            )

            results['systems'][system_name] = system_results

        # Pairwise comparisons
        results['pairwise_comparisons'] = {}

        for i, system1 in enumerate(system_names):
            for system2 in system_names[i+1:]:
                comparison_key = f"{system1}_vs_{system2}"

                comparison_result = statistics_suite.compare_measurement_systems(
                    system_measurements[system1], system_measurements[system2]
                )

                results['pairwise_comparisons'][comparison_key] = comparison_result

        # System ranking
        results['system_ranking'] = statistics_suite.rank_measurement_systems(
            system_measurements, baseline_system
        )

        return results

    packages['comparative_system_analysis'] = comparative_system_analysis

    # Package 5: Bayesian Precision Validation
    def bayesian_precision_validation(measurements, prior_precision_belief, evidence_threshold=0.95):
        """
        Bayesian precision validation sequence:
        1. Prior precision distribution setup
        2. Likelihood calculation from measurements
        3. Posterior precision distribution
        4. Bayesian validation decision
        """

        results = {'validation_sequence': 'bayesian_precision_validation'}

        # Setup Bayesian validator
        bayesian_validator = BayesianValidator()

        # Prior setup
        results['prior_setup'] = bayesian_validator.setup_precision_prior(prior_precision_belief)

        # Likelihood calculation
        results['likelihood_analysis'] = bayesian_validator.calculate_measurement_likelihood(measurements)

        # Posterior computation
        results['posterior_analysis'] = bayesian_validator.compute_precision_posterior(
            results['prior_setup'], results['likelihood_analysis']
        )

        # Bayesian validation decision
        results['bayesian_validation'] = bayesian_validator.make_validation_decision(
            results['posterior_analysis'], evidence_threshold
        )

        return results

    packages['bayesian_precision_validation'] = bayesian_precision_validation

    return packages

def run_packaged_analysis(package_name, validation_suite, *args, **kwargs):
    """
    Run a packaged analysis sequence

    Args:
        package_name: Name of the analysis package to run
        validation_suite: Complete validation suite containing the packages
        *args, **kwargs: Arguments to pass to the specific analysis package

    Returns:
        Analysis results from the specified package
    """

    if 'analysis_packages' not in validation_suite:
        raise ValueError("Validation suite does not contain analysis packages")

    if package_name not in validation_suite['analysis_packages']:
        available_packages = list(validation_suite['analysis_packages'].keys())
        raise ValueError(f"Package '{package_name}' not available. Available: {available_packages}")

    analysis_function = validation_suite['analysis_packages'][package_name]
    return analysis_function(*args, **kwargs)

def get_precision_module_status():
    """Get status of precision validation module"""

    status = {
        'module': 'precision_validation',
        'validation_framework_available': True,
        'statistics_suite_available': True,
        'strategic_disagreement_validation': True,
        'packaged_analysis_sequences': 5,
        'supported_validation_methods': [method.value for method in ValidationMethod],
        'supported_statistical_tests': [test.value for test in StatisticalTest],
        'ground_truth_free_validation': True,
        'bayesian_validation_support': True
    }

    return status

# Module exports
__all__ = [
    # Main validation classes
    'ValidationFramework',
    'StrategicDisagreementValidator',
    'PrecisionStatistics',
    'StatisticalFramework',
    'BayesianValidator',

    # Analysis and calculation classes
    'ConsensusCalculator',
    'AgreementAnalyzer',
    'ValidationConfidenceCalculator',
    'HypothesisTestSuite',
    'PowerAnalysisCalculator',
    'ConfidenceIntervalCalculator',

    # Enums and types
    'ValidationMethod',
    'MeasurementSystem',
    'DisagreementType',
    'StatisticalTest',
    'HypothesisType',
    'DistributionType',
    'SignificanceLevel',

    # Data classes
    'ValidationResult',
    'MeasurementRecord',
    'StrategicDisagreementPattern',
    'TestResult',
    'PowerAnalysisResult',
    'ConfidenceInterval',
    'BayesianResult',

    # Factory functions
    'create_strategic_disagreement_validator',
    'create_precision_statistics_suite',
    'create_validation_framework',
    'create_complete_validation_suite',

    # Packaged analysis functions
    'run_packaged_analysis',
    'get_precision_module_status'
]
