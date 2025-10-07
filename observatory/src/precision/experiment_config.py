"""
Experiment Configuration Module

Implements comprehensive configuration management for validation experiments
within the Strategic Disagreement Validation framework.

This module provides configuration for:
- Wave simulation experiments
- Precision measurement validation experiments
- Exotic method evaluation experiments
- Multi-system comparison experiments
- Statistical validation experiments
- S-Entropy alignment experiments

All experiments are designed around the validation framework described in
docs/algorithm/precision-validation-algorithm.tex and support the exotic
precision enhancement methods from the oscillatory modules.
"""

import time
import json
import toml
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np


class ExperimentType(Enum):
    """Types of validation experiments"""
    STRATEGIC_DISAGREEMENT = "strategic_disagreement"
    WAVE_SIMULATION = "wave_simulation"
    EXOTIC_METHOD_VALIDATION = "exotic_method_validation"
    PRECISION_SYSTEM_COMPARISON = "precision_system_comparison"
    MULTI_DOMAIN_VALIDATION = "multi_domain_validation"
    STATISTICAL_VALIDATION = "statistical_validation"
    S_ENTROPY_ALIGNMENT = "s_entropy_alignment"
    INTERFERENCE_PATTERN_ANALYSIS = "interference_pattern_analysis"
    HIERARCHICAL_NAVIGATION_TEST = "hierarchical_navigation_test"
    SEMANTIC_DISTANCE_TEST = "semantic_distance_test"
    TEMPORAL_COHERENCE_TEST = "temporal_coherence_test"
    BAYESIAN_VALIDATION = "bayesian_validation"


class MeasurementSystemType(Enum):
    """Measurement system types for experiments"""
    ATOMIC_CLOCK_CESIUM = "atomic_clock_cesium"
    ATOMIC_CLOCK_RUBIDIUM = "atomic_clock_rubidium"
    ATOMIC_CLOCK_HYDROGEN_MASER = "atomic_clock_hydrogen_maser"
    GPS_REFERENCE = "gps_reference"
    NTP_SERVER = "ntp_server"
    QUANTUM_SENSOR = "quantum_sensor"
    ENHANCED_GPS_SYSTEM = "enhanced_gps_system"
    MIMO_VIRTUAL_INFRASTRUCTURE = "mimo_virtual_infrastructure"
    MOLECULAR_SATELLITE_MESH = "molecular_satellite_mesh"
    SEMANTIC_DISTANCE_SYSTEM = "semantic_distance_system"
    TIME_SEQUENCING_SYSTEM = "time_sequencing_system"
    HIERARCHICAL_NAVIGATION_SYSTEM = "hierarchical_navigation_system"
    AMBIGUOUS_COMPRESSION_SYSTEM = "ambiguous_compression_system"
    S_ENTROPY_ALIGNMENT_SYSTEM = "s_entropy_alignment_system"
    TRANSCENDENT_OBSERVER_SYSTEM = "transcendent_observer_system"
    WAVE_REALITY_SIMULATOR = "wave_reality_simulator"
    OBSERVER_INTERFERENCE_NETWORK = "observer_interference_network"


class ValidationMethodType(Enum):
    """Validation method types"""
    POSITION_WISE_DISAGREEMENT = "position_wise_disagreement"
    PATTERN_RECOGNITION = "pattern_recognition"
    STATISTICAL_HYPOTHESIS_TEST = "statistical_hypothesis_test"
    CONFIDENCE_INTERVAL_ANALYSIS = "confidence_interval_analysis"
    BAYESIAN_INFERENCE = "bayesian_inference"
    INTERFERENCE_SUBSET_ANALYSIS = "interference_subset_analysis"
    INFORMATION_LOSS_QUANTIFICATION = "information_loss_quantification"
    PRECISION_IMPROVEMENT_FACTOR = "precision_improvement_factor"
    CATEGORICAL_ALIGNMENT_TEST = "categorical_alignment_test"
    S_ENTROPY_OPTIMIZATION = "s_entropy_optimization"


@dataclass
class SystemConfiguration:
    """Configuration for measurement system"""
    system_id: str
    system_type: MeasurementSystemType
    precision_digits: int = 15
    base_uncertainty: float = 1e-15
    sampling_rate: float = 1e9  # Hz
    measurement_duration: float = 1e-6  # seconds
    calibration_parameters: Dict[str, Any] = field(default_factory=dict)
    system_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_effective_uncertainty(self, measurement_conditions: Dict = None) -> float:
        """Calculate effective uncertainty under specific conditions"""
        effective_uncertainty = self.base_uncertainty

        if measurement_conditions:
            # Temperature effects
            temperature = measurement_conditions.get('temperature', 20.0)  # Celsius
            temp_coefficient = self.calibration_parameters.get('temperature_coefficient', 0.0)
            effective_uncertainty += abs(temperature - 20.0) * temp_coefficient

            # Pressure effects
            pressure = measurement_conditions.get('pressure', 1013.25)  # hPa
            pressure_coefficient = self.calibration_parameters.get('pressure_coefficient', 0.0)
            effective_uncertainty += abs(pressure - 1013.25) * pressure_coefficient

            # Humidity effects
            humidity = measurement_conditions.get('humidity', 50.0)  # %
            humidity_coefficient = self.calibration_parameters.get('humidity_coefficient', 0.0)
            effective_uncertainty += abs(humidity - 50.0) * humidity_coefficient

            # Network latency effects (for remote systems)
            network_latency = measurement_conditions.get('network_latency', 0.0)  # seconds
            effective_uncertainty += network_latency * 1e-6  # Assume 1μs uncertainty per second latency

        return max(effective_uncertainty, self.base_uncertainty * 0.1)  # Minimum 10% of base uncertainty


@dataclass
class DisagreementPrediction:
    """Configuration for predicted disagreement patterns"""
    prediction_id: str
    predicted_positions: List[int]
    disagreement_type: str = "position_specific"
    prediction_confidence: float = 0.9
    prediction_rationale: str = ""
    expected_pattern_strength: float = 0.95
    statistical_model: str = "binomial"
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_random_probability(self) -> float:
        """Calculate probability of random system matching this pattern"""
        if not self.predicted_positions:
            return 1.0

        # Strategic disagreement theorem: P_random = (1/10)^n
        n_positions = len(self.predicted_positions)

        if self.statistical_model == "binomial":
            return (0.1) ** n_positions
        elif self.statistical_model == "gaussian":
            # For continuous measurements, use different model
            return (0.05) ** n_positions
        else:
            return (0.1) ** n_positions


@dataclass
class WaveSimulationConfig:
    """Configuration for wave simulation experiments"""
    reality_complexity: float = 1.5
    num_reality_layers: int = 10
    simulation_duration: float = 1e-6  # seconds
    sampling_rate: float = 1e9  # Hz
    num_observers: int = 3
    observer_configurations: List[Dict[str, Any]] = field(default_factory=list)
    propagation_constraints: Dict[str, Any] = field(default_factory=dict)
    expected_information_loss: float = 0.3
    categorical_completion_rate: float = 1e6  # slots/second/complexity_unit
    dark_reality_percentage: float = 0.95
    matter_energy_percentage: float = 0.05

    def __post_init__(self):
        if not self.observer_configurations:
            # Default observer configurations
            self.observer_configurations = [
                {
                    'observer_type': 'FINITE_OBSERVER',
                    'interaction_mode': 'DISTURBING',
                    'size': 0.05,
                    'perception_spectrum': (1e6, 2e9)
                },
                {
                    'observer_type': 'ADAPTIVE_OBSERVER',
                    'interaction_mode': 'REFLECTIVE',
                    'size': 0.07,
                    'perception_spectrum': (5e6, 3e9)
                },
                {
                    'observer_type': 'S_ENTROPY_OBSERVER',
                    'interaction_mode': 'TRANSCENDENT',
                    'size': 0.03,
                    'perception_spectrum': (1e7, 5e9)
                }
            ][:self.num_observers]


@dataclass
class ExoticMethodConfig:
    """Configuration for exotic method validation"""
    method_name: str
    traditional_system_config: SystemConfiguration
    enhanced_system_config: SystemConfiguration
    validation_metrics: List[str] = field(default_factory=list)
    expected_improvement_factor: float = 2.0
    measurement_sample_size: int = 100
    validation_threshold: float = 0.95
    performance_baseline: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.validation_metrics:
            # Default metrics based on method
            if self.method_name == "semantic_distance":
                self.validation_metrics = ["amplification_factor", "pattern_complexity", "processing_efficiency"]
                self.expected_improvement_factor = 658.0  # 658× amplification
            elif self.method_name == "time_sequencing":
                self.validation_metrics = ["processing_time", "precision_improvement", "sequence_efficiency"]
                self.expected_improvement_factor = 5.0
            elif self.method_name == "hierarchical_navigation":
                self.validation_metrics = ["navigation_time", "complexity_order", "precision_gain"]
                self.expected_improvement_factor = 10.0
            elif self.method_name == "ambiguous_compression":
                self.validation_metrics = ["compression_resistance", "information_preservation", "precision_maintenance"]
                self.expected_improvement_factor = 3.0
            else:
                self.validation_metrics = ["precision_improvement", "efficiency_gain"]


@dataclass
class StatisticalValidationConfig:
    """Configuration for statistical validation"""
    confidence_level: float = 0.999  # 99.9% confidence
    significance_level: float = 0.001  # 0.1% significance
    minimum_sample_size: int = 30
    maximum_sample_size: int = 1000
    hypothesis_test_type: str = "binomial"  # binomial, t_test, chi_square
    power_analysis: bool = True
    effect_size_detection: float = 0.5  # Minimum detectable effect size
    multiple_comparisons_correction: str = "bonferroni"  # bonferroni, fdr, none
    bootstrap_iterations: int = 1000
    monte_carlo_simulations: int = 10000


@dataclass
class ExperimentConfiguration:
    """Master experiment configuration"""
    experiment_id: str
    experiment_type: ExperimentType
    experiment_name: str
    description: str = ""

    # System configurations
    measurement_systems: List[SystemConfiguration] = field(default_factory=list)
    candidate_system_id: str = ""
    reference_system_ids: List[str] = field(default_factory=list)

    # Validation configuration
    disagreement_predictions: List[DisagreementPrediction] = field(default_factory=list)
    validation_methods: List[ValidationMethodType] = field(default_factory=list)
    statistical_config: StatisticalValidationConfig = field(default_factory=StatisticalValidationConfig)

    # Experiment-specific configurations
    wave_simulation_config: Optional[WaveSimulationConfig] = None
    exotic_method_config: Optional[ExoticMethodConfig] = None

    # Execution parameters
    num_measurement_cycles: int = 10
    measurement_interval: float = 1.0  # seconds between measurements
    warm_up_cycles: int = 3
    cool_down_cycles: int = 2

    # Environmental conditions
    environmental_conditions: Dict[str, Any] = field(default_factory=dict)

    # Output configuration
    output_directory: str = "experiment_results"
    save_raw_data: bool = True
    save_analysis_plots: bool = True
    save_validation_reports: bool = True
    real_time_monitoring: bool = True

    # Metadata
    creation_timestamp: float = field(default_factory=time.time)
    created_by: str = "stella_lorraine_framework"
    experiment_version: str = "1.0.0"
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize default environmental conditions
        if not self.environmental_conditions:
            self.environmental_conditions = {
                'temperature': 20.0,  # Celsius
                'pressure': 1013.25,  # hPa
                'humidity': 50.0,     # %
                'network_latency': 1e-3,  # seconds
                'electromagnetic_interference': 'minimal'
            }

        # Set default validation methods based on experiment type
        if not self.validation_methods:
            if self.experiment_type == ExperimentType.STRATEGIC_DISAGREEMENT:
                self.validation_methods = [
                    ValidationMethodType.POSITION_WISE_DISAGREEMENT,
                    ValidationMethodType.STATISTICAL_HYPOTHESIS_TEST,
                    ValidationMethodType.CONFIDENCE_INTERVAL_ANALYSIS
                ]
            elif self.experiment_type == ExperimentType.WAVE_SIMULATION:
                self.validation_methods = [
                    ValidationMethodType.INTERFERENCE_SUBSET_ANALYSIS,
                    ValidationMethodType.INFORMATION_LOSS_QUANTIFICATION,
                    ValidationMethodType.CATEGORICAL_ALIGNMENT_TEST
                ]
            elif self.experiment_type == ExperimentType.EXOTIC_METHOD_VALIDATION:
                self.validation_methods = [
                    ValidationMethodType.PRECISION_IMPROVEMENT_FACTOR,
                    ValidationMethodType.PATTERN_RECOGNITION,
                    ValidationMethodType.STATISTICAL_HYPOTHESIS_TEST
                ]
            else:
                self.validation_methods = [ValidationMethodType.STATISTICAL_HYPOTHESIS_TEST]

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate experiment configuration"""
        issues = []

        # Check required fields
        if not self.experiment_id:
            issues.append("experiment_id is required")

        if not self.experiment_name:
            issues.append("experiment_name is required")

        # Check measurement systems
        if not self.measurement_systems:
            issues.append("At least one measurement system must be configured")

        # Check candidate system
        if self.experiment_type in [ExperimentType.STRATEGIC_DISAGREEMENT,
                                  ExperimentType.PRECISION_SYSTEM_COMPARISON]:
            if not self.candidate_system_id:
                issues.append("candidate_system_id required for this experiment type")

            candidate_found = any(sys.system_id == self.candidate_system_id
                                for sys in self.measurement_systems)
            if not candidate_found:
                issues.append(f"Candidate system '{self.candidate_system_id}' not found in measurement_systems")

        # Check reference systems
        if self.experiment_type in [ExperimentType.STRATEGIC_DISAGREEMENT,
                                  ExperimentType.PRECISION_SYSTEM_COMPARISON,
                                  ExperimentType.MULTI_DOMAIN_VALIDATION]:
            if not self.reference_system_ids:
                issues.append("reference_system_ids required for this experiment type")

            for ref_id in self.reference_system_ids:
                ref_found = any(sys.system_id == ref_id for sys in self.measurement_systems)
                if not ref_found:
                    issues.append(f"Reference system '{ref_id}' not found in measurement_systems")

        # Check disagreement predictions for strategic disagreement
        if self.experiment_type == ExperimentType.STRATEGIC_DISAGREEMENT:
            if not self.disagreement_predictions:
                issues.append("disagreement_predictions required for strategic disagreement validation")

        # Check wave simulation config
        if self.experiment_type == ExperimentType.WAVE_SIMULATION:
            if not self.wave_simulation_config:
                issues.append("wave_simulation_config required for wave simulation experiment")

        # Check exotic method config
        if self.experiment_type == ExperimentType.EXOTIC_METHOD_VALIDATION:
            if not self.exotic_method_config:
                issues.append("exotic_method_config required for exotic method validation")

        # Validate statistical configuration
        stat_config = self.statistical_config
        if not (0 < stat_config.confidence_level < 1):
            issues.append("confidence_level must be between 0 and 1")

        if not (0 < stat_config.significance_level < 1):
            issues.append("significance_level must be between 0 and 1")

        if stat_config.minimum_sample_size > stat_config.maximum_sample_size:
            issues.append("minimum_sample_size cannot be greater than maximum_sample_size")

        # Validate measurement parameters
        if self.num_measurement_cycles < 1:
            issues.append("num_measurement_cycles must be at least 1")

        if self.measurement_interval <= 0:
            issues.append("measurement_interval must be positive")

        return len(issues) == 0, issues

    def get_total_experiment_duration(self) -> float:
        """Calculate total experiment duration in seconds"""
        total_cycles = self.warm_up_cycles + self.num_measurement_cycles + self.cool_down_cycles
        return total_cycles * self.measurement_interval

    def get_system_by_id(self, system_id: str) -> Optional[SystemConfiguration]:
        """Get system configuration by ID"""
        for system in self.measurement_systems:
            if system.system_id == system_id:
                return system
        return None

    def get_candidate_system(self) -> Optional[SystemConfiguration]:
        """Get candidate system configuration"""
        return self.get_system_by_id(self.candidate_system_id)

    def get_reference_systems(self) -> List[SystemConfiguration]:
        """Get reference system configurations"""
        return [self.get_system_by_id(ref_id) for ref_id in self.reference_system_ids
                if self.get_system_by_id(ref_id) is not None]

    def export_config(self, file_path: str, format: str = "toml") -> bool:
        """Export configuration to file"""
        try:
            config_dict = asdict(self)

            # Convert enums to strings for serialization
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif isinstance(obj, Enum):
                    return obj.value
                else:
                    return obj

            config_dict = convert_enums(config_dict)

            if format.lower() == "toml":
                with open(file_path, 'w') as f:
                    toml.dump(config_dict, f)
            elif format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return True
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return False


class ExperimentConfigBuilder:
    """Builder pattern for creating experiment configurations"""

    def __init__(self):
        self.config = ExperimentConfiguration(
            experiment_id="",
            experiment_type=ExperimentType.STRATEGIC_DISAGREEMENT,
            experiment_name=""
        )

    def set_basic_info(self, experiment_id: str, experiment_name: str,
                      experiment_type: ExperimentType, description: str = "") -> 'ExperimentConfigBuilder':
        """Set basic experiment information"""
        self.config.experiment_id = experiment_id
        self.config.experiment_name = experiment_name
        self.config.experiment_type = experiment_type
        self.config.description = description
        return self

    def add_system(self, system_id: str, system_type: MeasurementSystemType,
                   precision_digits: int = 15, base_uncertainty: float = 1e-15,
                   **kwargs) -> 'ExperimentConfigBuilder':
        """Add measurement system"""
        system_config = SystemConfiguration(
            system_id=system_id,
            system_type=system_type,
            precision_digits=precision_digits,
            base_uncertainty=base_uncertainty,
            **kwargs
        )
        self.config.measurement_systems.append(system_config)
        return self

    def set_candidate_system(self, system_id: str) -> 'ExperimentConfigBuilder':
        """Set candidate system"""
        self.config.candidate_system_id = system_id
        return self

    def add_reference_system(self, system_id: str) -> 'ExperimentConfigBuilder':
        """Add reference system"""
        self.config.reference_system_ids.append(system_id)
        return self

    def add_disagreement_prediction(self, prediction_id: str, predicted_positions: List[int],
                                  **kwargs) -> 'ExperimentConfigBuilder':
        """Add disagreement prediction"""
        prediction = DisagreementPrediction(
            prediction_id=prediction_id,
            predicted_positions=predicted_positions,
            **kwargs
        )
        self.config.disagreement_predictions.append(prediction)
        return self

    def set_wave_simulation(self, **kwargs) -> 'ExperimentConfigBuilder':
        """Set wave simulation configuration"""
        self.config.wave_simulation_config = WaveSimulationConfig(**kwargs)
        return self

    def set_exotic_method(self, method_name: str, traditional_system_id: str,
                         enhanced_system_id: str, **kwargs) -> 'ExperimentConfigBuilder':
        """Set exotic method configuration"""
        traditional_system = self.config.get_system_by_id(traditional_system_id)
        enhanced_system = self.config.get_system_by_id(enhanced_system_id)

        if not traditional_system or not enhanced_system:
            raise ValueError("Traditional and enhanced systems must be added before setting exotic method config")

        self.config.exotic_method_config = ExoticMethodConfig(
            method_name=method_name,
            traditional_system_config=traditional_system,
            enhanced_system_config=enhanced_system,
            **kwargs
        )
        return self

    def set_statistical_params(self, **kwargs) -> 'ExperimentConfigBuilder':
        """Set statistical validation parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config.statistical_config, key):
                setattr(self.config.statistical_config, key, value)
        return self

    def set_execution_params(self, num_cycles: int = None, measurement_interval: float = None,
                           warm_up: int = None, cool_down: int = None) -> 'ExperimentConfigBuilder':
        """Set execution parameters"""
        if num_cycles is not None:
            self.config.num_measurement_cycles = num_cycles
        if measurement_interval is not None:
            self.config.measurement_interval = measurement_interval
        if warm_up is not None:
            self.config.warm_up_cycles = warm_up
        if cool_down is not None:
            self.config.cool_down_cycles = cool_down
        return self

    def set_environmental_conditions(self, **conditions) -> 'ExperimentConfigBuilder':
        """Set environmental conditions"""
        self.config.environmental_conditions.update(conditions)
        return self

    def build(self) -> ExperimentConfiguration:
        """Build and validate configuration"""
        is_valid, issues = self.config.validate_configuration()

        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(issues)}")

        return self.config


class ConfigurationTemplates:
    """Pre-defined configuration templates for common experiment types"""

    @staticmethod
    def strategic_disagreement_basic() -> ExperimentConfiguration:
        """Basic strategic disagreement validation experiment"""

        builder = ExperimentConfigBuilder()

        config = (builder
                 .set_basic_info(
                     experiment_id="strategic_disagreement_basic",
                     experiment_name="Basic Strategic Disagreement Validation",
                     experiment_type=ExperimentType.STRATEGIC_DISAGREEMENT,
                     description="Validates precision system using strategic disagreement method"
                 )
                 .add_system("cesium_reference", MeasurementSystemType.ATOMIC_CLOCK_CESIUM,
                           precision_digits=12, base_uncertainty=1e-12)
                 .add_system("rubidium_reference", MeasurementSystemType.ATOMIC_CLOCK_RUBIDIUM,
                           precision_digits=11, base_uncertainty=5e-12)
                 .add_system("enhanced_candidate", MeasurementSystemType.S_ENTROPY_ALIGNMENT_SYSTEM,
                           precision_digits=15, base_uncertainty=1e-15)
                 .set_candidate_system("enhanced_candidate")
                 .add_reference_system("cesium_reference")
                 .add_reference_system("rubidium_reference")
                 .add_disagreement_prediction("pred_001", [2, 7, 11],
                                            prediction_confidence=0.95,
                                            prediction_rationale="Enhanced system predicts disagreement at these positions")
                 .set_execution_params(num_cycles=50, measurement_interval=2.0)
                 .build())

        return config

    @staticmethod
    def wave_simulation_basic() -> ExperimentConfiguration:
        """Basic wave simulation experiment"""

        builder = ExperimentConfigBuilder()

        config = (builder
                 .set_basic_info(
                     experiment_id="wave_simulation_basic",
                     experiment_name="Basic Wave Simulation Validation",
                     experiment_type=ExperimentType.WAVE_SIMULATION,
                     description="Validates categorical alignment using wave interference patterns"
                 )
                 .add_system("reality_wave", MeasurementSystemType.WAVE_REALITY_SIMULATOR,
                           precision_digits=20, base_uncertainty=1e-20)
                 .add_system("observer_network", MeasurementSystemType.OBSERVER_INTERFERENCE_NETWORK,
                           precision_digits=15, base_uncertainty=1e-15)
                 .set_wave_simulation(
                     reality_complexity=1.5,
                     num_observers=3,
                     simulation_duration=1e-6,
                     expected_information_loss=0.3
                 )
                 .set_execution_params(num_cycles=20, measurement_interval=1.0)
                 .build())

        return config

    @staticmethod
    def semantic_distance_validation() -> ExperimentConfiguration:
        """Semantic distance method validation experiment"""

        builder = ExperimentConfigBuilder()

        config = (builder
                 .set_basic_info(
                     experiment_id="semantic_distance_validation",
                     experiment_name="Semantic Distance Amplification Validation",
                     experiment_type=ExperimentType.EXOTIC_METHOD_VALIDATION,
                     description="Validates 658× semantic distance amplification method"
                 )
                 .add_system("traditional_clock", MeasurementSystemType.ATOMIC_CLOCK_CESIUM,
                           precision_digits=12, base_uncertainty=1e-12)
                 .add_system("semantic_enhanced", MeasurementSystemType.SEMANTIC_DISTANCE_SYSTEM,
                           precision_digits=15, base_uncertainty=1e-15)
                 .set_exotic_method(
                     method_name="semantic_distance",
                     traditional_system_id="traditional_clock",
                     enhanced_system_id="semantic_enhanced",
                     expected_improvement_factor=658.0
                 )
                 .set_execution_params(num_cycles=100, measurement_interval=0.5)
                 .build())

        return config

    @staticmethod
    def hierarchical_navigation_test() -> ExperimentConfiguration:
        """Hierarchical navigation O(1) test"""

        builder = ExperimentConfigBuilder()

        config = (builder
                 .set_basic_info(
                     experiment_id="hierarchical_navigation_test",
                     experiment_name="O(1) Hierarchical Navigation Test",
                     experiment_type=ExperimentType.HIERARCHICAL_NAVIGATION_TEST,
                     description="Tests O(1) complexity hierarchical navigation using gear ratios"
                 )
                 .add_system("linear_navigator", MeasurementSystemType.GPS_REFERENCE,
                           precision_digits=10, base_uncertainty=1e-10)
                 .add_system("hierarchical_navigator", MeasurementSystemType.HIERARCHICAL_NAVIGATION_SYSTEM,
                           precision_digits=15, base_uncertainty=1e-15)
                 .set_exotic_method(
                     method_name="hierarchical_navigation",
                     traditional_system_id="linear_navigator",
                     enhanced_system_id="hierarchical_navigator",
                     expected_improvement_factor=10.0
                 )
                 .set_execution_params(num_cycles=200, measurement_interval=0.1)
                 .build())

        return config

    @staticmethod
    def multi_domain_comprehensive() -> ExperimentConfiguration:
        """Comprehensive multi-domain validation"""

        builder = ExperimentConfigBuilder()

        config = (builder
                 .set_basic_info(
                     experiment_id="multi_domain_comprehensive",
                     experiment_name="Comprehensive Multi-Domain Validation",
                     experiment_type=ExperimentType.MULTI_DOMAIN_VALIDATION,
                     description="Comprehensive validation across multiple precision enhancement domains"
                 )
                 .add_system("cesium_ref", MeasurementSystemType.ATOMIC_CLOCK_CESIUM)
                 .add_system("rubidium_ref", MeasurementSystemType.ATOMIC_CLOCK_RUBIDIUM)
                 .add_system("gps_ref", MeasurementSystemType.GPS_REFERENCE)
                 .add_system("semantic_system", MeasurementSystemType.SEMANTIC_DISTANCE_SYSTEM)
                 .add_system("time_sequence_system", MeasurementSystemType.TIME_SEQUENCING_SYSTEM)
                 .add_system("hierarchical_system", MeasurementSystemType.HIERARCHICAL_NAVIGATION_SYSTEM)
                 .add_system("s_entropy_system", MeasurementSystemType.S_ENTROPY_ALIGNMENT_SYSTEM)
                 .set_candidate_system("s_entropy_system")
                 .add_reference_system("cesium_ref")
                 .add_reference_system("rubidium_ref")
                 .add_reference_system("gps_ref")
                 .add_disagreement_prediction("multi_pred_001", [1, 3, 5, 7, 9],
                                            prediction_confidence=0.98)
                 .set_execution_params(num_cycles=500, measurement_interval=1.0)
                 .set_statistical_params(confidence_level=0.9999, significance_level=0.0001)
                 .build())

        return config


class ConfigurationLoader:
    """Utility for loading configurations from files"""

    @staticmethod
    def load_from_file(file_path: str) -> ExperimentConfiguration:
        """Load configuration from TOML or JSON file"""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            if path.suffix.lower() == '.toml':
                with open(file_path, 'r') as f:
                    config_dict = toml.load(f)
            elif path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Convert string enum values back to enums
            config_dict = ConfigurationLoader._convert_strings_to_enums(config_dict)

            # Create configuration object
            config = ExperimentConfiguration(**config_dict)

            # Validate loaded configuration
            is_valid, issues = config.validate_configuration()
            if not is_valid:
                raise ValueError(f"Invalid loaded configuration: {'; '.join(issues)}")

            return config

        except Exception as e:
            raise ValueError(f"Error loading configuration from {file_path}: {e}")

    @staticmethod
    def _convert_strings_to_enums(obj):
        """Convert string values back to enum objects"""

        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == 'experiment_type' and isinstance(value, str):
                    result[key] = ExperimentType(value)
                elif key == 'system_type' and isinstance(value, str):
                    result[key] = MeasurementSystemType(value)
                elif key in ['validation_methods'] and isinstance(value, list):
                    result[key] = [ValidationMethodType(v) if isinstance(v, str) else v for v in value]
                else:
                    result[key] = ConfigurationLoader._convert_strings_to_enums(value)
            return result
        elif isinstance(obj, list):
            return [ConfigurationLoader._convert_strings_to_enums(item) for item in obj]
        else:
            return obj


# Utility functions

def create_experiment_config(experiment_type: str = "strategic_disagreement") -> ExperimentConfiguration:
    """Create experiment configuration using templates"""

    templates = {
        "strategic_disagreement": ConfigurationTemplates.strategic_disagreement_basic,
        "wave_simulation": ConfigurationTemplates.wave_simulation_basic,
        "semantic_distance": ConfigurationTemplates.semantic_distance_validation,
        "hierarchical_navigation": ConfigurationTemplates.hierarchical_navigation_test,
        "multi_domain": ConfigurationTemplates.multi_domain_comprehensive
    }

    if experiment_type not in templates:
        available_types = list(templates.keys())
        raise ValueError(f"Unknown experiment type: {experiment_type}. Available: {available_types}")

    return templates[experiment_type]()


def validate_experiment_config(config: ExperimentConfiguration) -> Tuple[bool, List[str]]:
    """Validate experiment configuration"""
    return config.validate_configuration()


# Main execution for testing
if __name__ == "__main__":
    # Test configuration creation
    print("Testing experiment configuration...")

    # Test basic strategic disagreement config
    config = ConfigurationTemplates.strategic_disagreement_basic()
    is_valid, issues = config.validate_configuration()

    print(f"Configuration ID: {config.experiment_id}")
    print(f"Configuration valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")

    print(f"Total duration: {config.get_total_experiment_duration():.2f} seconds")
    print(f"Measurement systems: {len(config.measurement_systems)}")
    print(f"Disagreement predictions: {len(config.disagreement_predictions)}")

    # Test export
    success = config.export_config("test_config.toml")
    print(f"Export successful: {success}")

    print("Configuration module initialized successfully!")
