#!/usr/bin/env python3
"""
Observatory Experiment Configuration
===================================
This is the config file for experiments, defining all parameters required for any process.
Provides comprehensive parameter management for Stella-Lorraine observatory experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timezone
import json
from enum import Enum, auto


class ExperimentType(Enum):
    """Types of experiments supported by the observatory"""
    PRECISION_TIMING = auto()
    OSCILLATORY_ANALYSIS = auto()
    CONSCIOUSNESS_TARGETING = auto()
    MEMORIAL_FRAMEWORK = auto()
    TEMPORAL_SYNCHRONIZATION = auto()
    QUANTUM_MOLECULAR = auto()
    BAYESIAN_OPTIMIZATION = auto()
    MULTI_OBJECTIVE = auto()


class OptimizationGoal(Enum):
    """Optimization goals for Bayesian Network"""
    MAXIMIZE_PRECISION = auto()
    MINIMIZE_LATENCY = auto()
    MAXIMIZE_ACCURACY = auto()
    MINIMIZE_ERROR = auto()
    MAXIMIZE_CONVERGENCE = auto()
    MINIMIZE_RESOURCE_USAGE = auto()
    MAXIMIZE_STABILITY = auto()
    PARETO_OPTIMIZATION = auto()


@dataclass
class TemporalPrecisionConfig:
    """Configuration for temporal precision experiments"""
    target_precision_ns: float = 0.1  # Sub-nanosecond target
    sampling_rate_hz: int = 1000000000    # 1 MHz sampling
    measurement_duration_s: float = 30.0
    reference_clock_source: str = "atomic"
    oscillatory_coupling_enabled: bool = True
    multi_scale_frequencies: List[float] = field(default_factory=lambda: [1e3, 1e6, 1e9, 1e12])
    environmental_corrections: bool = True
    quantum_enhancement: bool = True


@dataclass
class OscillatoryAnalysisConfig:
    """Configuration for oscillatory framework analysis"""
    frequency_range_hz: Tuple[float, float] = (1e-3, 1e15)  # Cosmic to quantum
    oscillator_count: int = 10000
    coupling_strength: float = 0.5
    convergence_threshold: float = 1e-9
    max_iterations: int = 10000
    multi_scale_coupling: bool = True
    self_organization_enabled: bool = True
    temporal_emergence_tracking: bool = True


@dataclass
class ConsciousnessTargetingConfig:
    """Configuration for consciousness targeting experiments"""
    population_size: int = 10000
    consciousness_dimensions: int = 4
    targeting_accuracy_threshold: float = 0.90
    free_will_belief_distribution: str = "beta(2,2)"
    death_proximity_model: str = "exponential(0.5)"
    nordic_paradox_enabled: bool = True
    functional_delusion_tracking: bool = True
    inheritance_efficiency_target: float = 0.95


@dataclass
class MemorialFrameworkConfig:
    """Configuration for memorial framework experiments"""
    buhera_model_enabled: bool = True
    consciousness_inheritance_rate: float = 0.92
    capitalism_elimination_target: float = 0.99
    expertise_transfer_efficiency: float = 0.88
    temporal_persistence_requirement: float = 0.98
    cost_reduction_target: float = 0.99
    memorial_system_comparisons: List[str] = field(default_factory=lambda: ["traditional", "digital", "stella_lorraine"])


@dataclass
class BayesianNetworkConfig:
    """Configuration for Bayesian Network orchestrator"""
    node_count: int = 100
    prior_distribution: str = "uniform"
    posterior_update_method: str = "variational_inference"
    convergence_criteria: float = 1e-6
    max_iterations: int = 1000
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    evidence_weight: float = 1.0
    causal_structure_learning: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for optimization process"""
    primary_goal: OptimizationGoal = OptimizationGoal.MAXIMIZE_PRECISION
    secondary_goals: List[OptimizationGoal] = field(default_factory=list)
    pareto_weights: Dict[str, float] = field(default_factory=dict)
    optimization_algorithm: str = "bayesian_optimization"
    acquisition_function: str = "expected_improvement"
    gaussian_process_kernel: str = "matern52"
    budget_iterations: int = 1000
    early_stopping_patience: int = 50
    target_improvement_threshold: float = 0.01


@dataclass
class ResourceConfig:
    """Configuration for computational resources"""
    max_cpu_cores: int = -1  # -1 means use all available
    max_memory_gb: float = 32.0
    gpu_enabled: bool = True
    gpu_memory_gb: float = 8.0
    parallel_experiments: int = 4
    distributed_computing: bool = False
    cluster_nodes: List[str] = field(default_factory=list)
    timeout_minutes: int = 60


@dataclass
class DataConfig:
    """Configuration for data management"""
    output_directory: Path = Path("./observatory_results")
    experiment_id_prefix: str = "stella_exp"
    save_intermediate_results: bool = True
    save_raw_data: bool = True
    compression_enabled: bool = True
    backup_enabled: bool = True
    data_retention_days: int = 365
    result_formats: List[str] = field(default_factory=lambda: ["json", "hdf5", "parquet"])


@dataclass
class ValidationConfig:
    """Configuration for experiment validation"""
    cross_validation_folds: int = 5
    statistical_significance_level: float = 0.05
    confidence_interval: float = 0.95
    bootstrap_samples: int = 1000
    outlier_detection_enabled: bool = True
    outlier_threshold_std: float = 3.0
    reproducibility_seed: Optional[int] = 42
    validation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])


@dataclass
class MonitoringConfig:
    """Configuration for experiment monitoring"""
    logging_level: str = "INFO"
    progress_reporting_interval: int = 10  # seconds
    performance_monitoring: bool = True
    memory_usage_tracking: bool = True
    real_time_visualization: bool = True
    alert_on_errors: bool = True
    alert_on_completion: bool = True
    dashboard_port: int = 8080


class ExperimentConfig:
    """
    Comprehensive experiment configuration manager
    Handles all parameters required for observatory experiments
    """

    def __init__(self, config_file: Optional[str] = None):
        self.timestamp = datetime.now(timezone.utc)
        self.config_version = "1.0.0"

        # Core configuration components
        self.experiment_type = ExperimentType.PRECISION_TIMING
        self.experiment_name = f"stella_experiment_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.description = "Stella-Lorraine Observatory Experiment"

        # Specific experiment configurations
        self.temporal_precision = TemporalPrecisionConfig()
        self.oscillatory_analysis = OscillatoryAnalysisConfig()
        self.consciousness_targeting = ConsciousnessTargetingConfig()
        self.memorial_framework = MemorialFrameworkConfig()
        self.bayesian_network = BayesianNetworkConfig()
        self.optimization = OptimizationConfig()
        self.resources = ResourceConfig()
        self.data = DataConfig()
        self.validation = ValidationConfig()
        self.monitoring = MonitoringConfig()

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "config_version": self.config_version,
                "experiment_type": self.experiment_type.name,
                "experiment_name": self.experiment_name,
                "description": self.description
            },
            "temporal_precision": self._dataclass_to_dict(self.temporal_precision),
            "oscillatory_analysis": self._dataclass_to_dict(self.oscillatory_analysis),
            "consciousness_targeting": self._dataclass_to_dict(self.consciousness_targeting),
            "memorial_framework": self._dataclass_to_dict(self.memorial_framework),
            "bayesian_network": self._dataclass_to_dict(self.bayesian_network),
            "optimization": self._dataclass_to_dict(self.optimization),
            "resources": self._dataclass_to_dict(self.resources),
            "data": self._dataclass_to_dict(self.data),
            "validation": self._dataclass_to_dict(self.validation),
            "monitoring": self._dataclass_to_dict(self.monitoring)
        }
        return config_dict

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary with enum handling"""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.name
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.name for item in value]
            else:
                result[key] = value
        return result

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def load_from_file(self, filepath: str) -> None:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Load metadata
        if "metadata" in config_dict:
            meta = config_dict["metadata"]
            self.experiment_name = meta.get("experiment_name", self.experiment_name)
            self.description = meta.get("description", self.description)
            if "experiment_type" in meta:
                self.experiment_type = ExperimentType[meta["experiment_type"]]

        # Load component configurations
        self._load_component_config(config_dict, "temporal_precision", self.temporal_precision)
        self._load_component_config(config_dict, "oscillatory_analysis", self.oscillatory_analysis)
        self._load_component_config(config_dict, "consciousness_targeting", self.consciousness_targeting)
        self._load_component_config(config_dict, "memorial_framework", self.memorial_framework)
        self._load_component_config(config_dict, "bayesian_network", self.bayesian_network)
        self._load_component_config(config_dict, "optimization", self.optimization)
        self._load_component_config(config_dict, "resources", self.resources)
        self._load_component_config(config_dict, "data", self.data)
        self._load_component_config(config_dict, "validation", self.validation)
        self._load_component_config(config_dict, "monitoring", self.monitoring)

    def _load_component_config(self, config_dict: Dict, component_name: str, component_obj: Any) -> None:
        """Load component configuration with type conversion"""
        if component_name in config_dict:
            component_dict = config_dict[component_name]
            for key, value in component_dict.items():
                if hasattr(component_obj, key):
                    # Handle enum conversion
                    current_value = getattr(component_obj, key)
                    if isinstance(current_value, Enum):
                        enum_class = type(current_value)
                        setattr(component_obj, key, enum_class[value])
                    elif isinstance(current_value, Path):
                        setattr(component_obj, key, Path(value))
                    else:
                        setattr(component_obj, key, value)

    def get_experiment_parameters(self, experiment_type: ExperimentType) -> Dict[str, Any]:
        """Get parameters specific to experiment type"""
        base_params = {
            "experiment_name": self.experiment_name,
            "experiment_type": experiment_type.name,
            "timestamp": self.timestamp.isoformat(),
            "bayesian_network": self._dataclass_to_dict(self.bayesian_network),
            "optimization": self._dataclass_to_dict(self.optimization),
            "resources": self._dataclass_to_dict(self.resources),
            "data": self._dataclass_to_dict(self.data),
            "validation": self._dataclass_to_dict(self.validation),
            "monitoring": self._dataclass_to_dict(self.monitoring)
        }

        # Add experiment-specific parameters
        if experiment_type == ExperimentType.PRECISION_TIMING:
            base_params["temporal_precision"] = self._dataclass_to_dict(self.temporal_precision)
        elif experiment_type == ExperimentType.OSCILLATORY_ANALYSIS:
            base_params["oscillatory_analysis"] = self._dataclass_to_dict(self.oscillatory_analysis)
        elif experiment_type == ExperimentType.CONSCIOUSNESS_TARGETING:
            base_params["consciousness_targeting"] = self._dataclass_to_dict(self.consciousness_targeting)
        elif experiment_type == ExperimentType.MEMORIAL_FRAMEWORK:
            base_params["memorial_framework"] = self._dataclass_to_dict(self.memorial_framework)
        else:
            # Include all for multi-type experiments
            base_params.update({
                "temporal_precision": self._dataclass_to_dict(self.temporal_precision),
                "oscillatory_analysis": self._dataclass_to_dict(self.oscillatory_analysis),
                "consciousness_targeting": self._dataclass_to_dict(self.consciousness_targeting),
                "memorial_framework": self._dataclass_to_dict(self.memorial_framework)
            })

        return base_params

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters and return (is_valid, error_messages)"""
        errors = []

        # Validate temporal precision config
        if self.temporal_precision.target_precision_ns <= 0:
            errors.append("Target precision must be positive")

        if self.temporal_precision.sampling_rate_hz <= 0:
            errors.append("Sampling rate must be positive")

        # Validate oscillatory analysis config
        if self.oscillatory_analysis.oscillator_count <= 0:
            errors.append("Oscillator count must be positive")

        if not (0 <= self.oscillatory_analysis.coupling_strength <= 1):
            errors.append("Coupling strength must be between 0 and 1")

        # Validate consciousness targeting config
        if self.consciousness_targeting.population_size <= 0:
            errors.append("Population size must be positive")

        if not (0 <= self.consciousness_targeting.targeting_accuracy_threshold <= 1):
            errors.append("Targeting accuracy threshold must be between 0 and 1")

        # Validate resource config
        if self.resources.max_memory_gb <= 0:
            errors.append("Max memory must be positive")

        if self.resources.parallel_experiments <= 0:
            errors.append("Parallel experiments must be positive")

        # Validate optimization config
        if self.optimization.budget_iterations <= 0:
            errors.append("Budget iterations must be positive")

        if not (0 <= self.optimization.target_improvement_threshold <= 1):
            errors.append("Target improvement threshold must be between 0 and 1")

        return len(errors) == 0, errors

    def create_experiment_preset(self, preset_name: str) -> None:
        """Create predefined experiment configurations"""
        if preset_name == "high_precision_timing":
            self.experiment_type = ExperimentType.PRECISION_TIMING
            self.temporal_precision.target_precision_ns = 0.01  # 10 picoseconds
            self.temporal_precision.sampling_rate_hz = 10000000  # 10 MHz
            self.optimization.primary_goal = OptimizationGoal.MAXIMIZE_PRECISION

        elif preset_name == "consciousness_research":
            self.experiment_type = ExperimentType.CONSCIOUSNESS_TARGETING
            self.consciousness_targeting.population_size = 50000
            self.consciousness_targeting.targeting_accuracy_threshold = 0.95
            self.optimization.primary_goal = OptimizationGoal.MAXIMIZE_ACCURACY

        elif preset_name == "oscillatory_framework":
            self.experiment_type = ExperimentType.OSCILLATORY_ANALYSIS
            self.oscillatory_analysis.oscillator_count = 10000
            self.oscillatory_analysis.frequency_range_hz = (1e-6, 1e18)  # Extended range
            self.optimization.primary_goal = OptimizationGoal.MAXIMIZE_CONVERGENCE

        elif preset_name == "memorial_optimization":
            self.experiment_type = ExperimentType.MEMORIAL_FRAMEWORK
            self.memorial_framework.consciousness_inheritance_rate = 0.99
            self.memorial_framework.capitalism_elimination_target = 0.995
            self.optimization.primary_goal = OptimizationGoal.MAXIMIZE_ACCURACY

        elif preset_name == "multi_objective":
            self.experiment_type = ExperimentType.MULTI_OBJECTIVE
            self.optimization.primary_goal = OptimizationGoal.PARETO_OPTIMIZATION
            self.optimization.secondary_goals = [
                OptimizationGoal.MAXIMIZE_PRECISION,
                OptimizationGoal.MINIMIZE_LATENCY,
                OptimizationGoal.MAXIMIZE_ACCURACY
            ]

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ExperimentConfig(name='{self.experiment_name}', type={self.experiment_type.name})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"ExperimentConfig(name='{self.experiment_name}', "
                f"type={self.experiment_type.name}, "
                f"timestamp='{self.timestamp.isoformat()}')")


# Convenience factory functions
def create_precision_timing_config() -> ExperimentConfig:
    """Create configuration optimized for precision timing experiments"""
    config = ExperimentConfig()
    config.create_experiment_preset("high_precision_timing")
    return config


def create_consciousness_research_config() -> ExperimentConfig:
    """Create configuration optimized for consciousness research"""
    config = ExperimentConfig()
    config.create_experiment_preset("consciousness_research")
    return config


def create_oscillatory_framework_config() -> ExperimentConfig:
    """Create configuration optimized for oscillatory framework analysis"""
    config = ExperimentConfig()
    config.create_experiment_preset("oscillatory_framework")
    return config


def create_memorial_optimization_config() -> ExperimentConfig:
    """Create configuration optimized for memorial framework"""
    config = ExperimentConfig()
    config.create_experiment_preset("memorial_optimization")
    return config


def create_multi_objective_config() -> ExperimentConfig:
    """Create configuration for multi-objective optimization"""
    config = ExperimentConfig()
    config.create_experiment_preset("multi_objective")
    return config
