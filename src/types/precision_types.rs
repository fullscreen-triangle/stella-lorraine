use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use crate::types::temporal_types::{TemporalPosition, PrecisionLevel};

/// Precision measurement configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionMeasurementConfig {
    /// Target precision level
    pub target_precision: PrecisionLevel,
    /// Measurement duration
    pub measurement_duration: Duration,
    /// Sampling rate in Hz
    pub sampling_rate: f64,
    /// Number of measurement cycles
    pub measurement_cycles: usize,
    /// Noise reduction parameters
    pub noise_reduction: NoiseReductionConfig,
    /// Calibration parameters
    pub calibration: CalibrationConfig,
    /// Environmental compensation
    pub environmental_compensation: EnvironmentalCompensationConfig,
}

/// Noise reduction configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseReductionConfig {
    /// Enable thermal noise reduction
    pub thermal_noise_reduction: bool,
    /// Enable shot noise reduction
    pub shot_noise_reduction: bool,
    /// Enable flicker noise reduction
    pub flicker_noise_reduction: bool,
    /// Enable quantization noise reduction
    pub quantization_noise_reduction: bool,
    /// Custom noise filters
    pub custom_filters: Vec<NoiseFilter>,
}

/// Noise filter specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseFilter {
    /// Filter type
    pub filter_type: NoiseFilterType,
    /// Filter parameters
    pub parameters: Vec<f64>,
    /// Filter enabled
    pub enabled: bool,
}

/// Types of noise filters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseFilterType {
    /// Low-pass filter
    LowPass,
    /// High-pass filter
    HighPass,
    /// Band-pass filter
    BandPass,
    /// Band-stop filter
    BandStop,
    /// Notch filter
    Notch,
    /// Adaptive filter
    Adaptive,
    /// Kalman filter
    Kalman,
}

/// Calibration configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Reference frequency source
    pub reference_frequency: f64,
    /// Calibration interval
    pub calibration_interval: Duration,
    /// Calibration accuracy requirement
    pub accuracy_requirement: f64,
    /// Temperature calibration
    pub temperature_calibration: bool,
    /// Pressure calibration
    pub pressure_calibration: bool,
    /// Humidity calibration
    pub humidity_calibration: bool,
}

/// Environmental compensation configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentalCompensationConfig {
    /// Temperature compensation
    pub temperature_compensation: TemperatureCompensation,
    /// Pressure compensation
    pub pressure_compensation: PressureCompensation,
    /// Humidity compensation
    pub humidity_compensation: HumidityCompensation,
    /// Magnetic field compensation
    pub magnetic_field_compensation: MagneticFieldCompensation,
    /// Vibration compensation
    pub vibration_compensation: VibrationCompensation,
}

/// Temperature compensation parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureCompensation {
    /// Enable temperature compensation
    pub enabled: bool,
    /// Temperature coefficient (ppm/°C)
    pub temperature_coefficient: f64,
    /// Reference temperature (°C)
    pub reference_temperature: f64,
    /// Temperature measurement uncertainty (°C)
    pub temperature_uncertainty: f64,
}

/// Pressure compensation parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureCompensation {
    /// Enable pressure compensation
    pub enabled: bool,
    /// Pressure coefficient (ppm/Pa)
    pub pressure_coefficient: f64,
    /// Reference pressure (Pa)
    pub reference_pressure: f64,
    /// Pressure measurement uncertainty (Pa)
    pub pressure_uncertainty: f64,
}

/// Humidity compensation parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumidityCompensation {
    /// Enable humidity compensation
    pub enabled: bool,
    /// Humidity coefficient (ppm/%)
    pub humidity_coefficient: f64,
    /// Reference humidity (%)
    pub reference_humidity: f64,
    /// Humidity measurement uncertainty (%)
    pub humidity_uncertainty: f64,
}

/// Magnetic field compensation parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MagneticFieldCompensation {
    /// Enable magnetic field compensation
    pub enabled: bool,
    /// Magnetic field coefficient (ppm/T)
    pub magnetic_field_coefficient: f64,
    /// Reference magnetic field (T)
    pub reference_magnetic_field: f64,
    /// Magnetic field measurement uncertainty (T)
    pub magnetic_field_uncertainty: f64,
}

/// Vibration compensation parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VibrationCompensation {
    /// Enable vibration compensation
    pub enabled: bool,
    /// Vibration sensitivity (ppm/(m/s²))
    pub vibration_sensitivity: f64,
    /// Vibration frequency range (Hz)
    pub frequency_range: (f64, f64),
    /// Vibration measurement uncertainty (m/s²)
    pub vibration_uncertainty: f64,
}

/// Precision measurement result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionMeasurementResult {
    /// Measured temporal position
    pub temporal_position: TemporalPosition,
    /// Measurement uncertainty
    pub uncertainty: f64,
    /// Achieved precision level
    pub precision_level: PrecisionLevel,
    /// Measurement confidence
    pub confidence: f64,
    /// Measurement statistics
    pub statistics: MeasurementStatistics,
    /// Quality metrics
    pub quality_metrics: MeasurementQualityMetrics,
    /// Measurement timestamp
    pub measurement_time: SystemTime,
}

/// Measurement statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub standard_deviation: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub minimum: f64,
    /// Maximum value
    pub maximum: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Allan variance
    pub allan_variance: AllanVariance,
}

/// Allan variance analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllanVariance {
    /// Allan variance values
    pub values: Vec<f64>,
    /// Corresponding tau values (averaging times)
    pub tau_values: Vec<f64>,
    /// Allan deviation values
    pub deviation_values: Vec<f64>,
    /// Noise identification
    pub noise_identification: NoiseIdentification,
}

/// Noise identification from Allan variance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseIdentification {
    /// Dominant noise types
    pub dominant_noise_types: Vec<NoiseType>,
    /// Noise floor level
    pub noise_floor: f64,
    /// Stability at 1 second
    pub stability_1s: f64,
    /// Stability at 10 seconds
    pub stability_10s: f64,
    /// Stability at 100 seconds
    pub stability_100s: f64,
}

/// Types of noise identified
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseType {
    /// White phase noise
    WhitePhase,
    /// Flicker phase noise
    FlickerPhase,
    /// White frequency noise
    WhiteFrequency,
    /// Flicker frequency noise
    FlickerFrequency,
    /// Random walk frequency noise
    RandomWalkFrequency,
}

/// Measurement quality metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementQualityMetrics {
    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: f64,
    /// Effective number of bits
    pub effective_bits: f64,
    /// Spurious-free dynamic range
    pub spurious_free_dynamic_range: f64,
    /// Total harmonic distortion
    pub total_harmonic_distortion: f64,
    /// Measurement linearity
    pub linearity: f64,
    /// Measurement stability
    pub stability: f64,
    /// Measurement repeatability
    pub repeatability: f64,
}

/// Precision validation result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionValidationResult {
    /// Validation passed
    pub validation_passed: bool,
    /// Target precision
    pub target_precision: f64,
    /// Achieved precision
    pub achieved_precision: f64,
    /// Precision ratio (achieved/target)
    pub precision_ratio: f64,
    /// Validation confidence
    pub validation_confidence: f64,
    /// Validation details
    pub validation_details: Vec<ValidationDetail>,
}

/// Precision validation detail
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationDetail {
    /// Validation aspect
    pub aspect: ValidationAspect,
    /// Validation result
    pub result: bool,
    /// Validation value
    pub value: f64,
    /// Validation threshold
    pub threshold: f64,
    /// Validation notes
    pub notes: String,
}

/// Aspects of precision validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationAspect {
    /// Temporal precision
    TemporalPrecision,
    /// Spatial precision
    SpatialPrecision,
    /// Measurement uncertainty
    MeasurementUncertainty,
    /// Allan variance
    AllanVariance,
    /// Noise performance
    NoisePerformance,
    /// Environmental stability
    EnvironmentalStability,
    /// Calibration accuracy
    CalibrationAccuracy,
}

/// Precision budget analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionBudget {
    /// Total precision budget
    pub total_budget: f64,
    /// Budget components
    pub components: Vec<PrecisionBudgetComponent>,
    /// Budget utilization
    pub utilization: f64,
    /// Budget margin
    pub margin: f64,
}

/// Precision budget component
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionBudgetComponent {
    /// Component name
    pub name: String,
    /// Component type
    pub component_type: PrecisionComponentType,
    /// Allocated precision
    pub allocated_precision: f64,
    /// Actual precision
    pub actual_precision: f64,
    /// Utilization ratio
    pub utilization: f64,
}

/// Types of precision budget components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrecisionComponentType {
    /// Quantum level precision
    Quantum,
    /// Molecular level precision
    Molecular,
    /// Biological level precision
    Biological,
    /// Consciousness level precision
    Consciousness,
    /// Environmental level precision
    Environmental,
    /// Measurement system precision
    MeasurementSystem,
    /// Calibration precision
    Calibration,
    /// Environmental compensation precision
    EnvironmentalCompensation,
}

/// Precision optimization parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionOptimizationParams {
    /// Optimization target
    pub target: OptimizationTarget,
    /// Optimization constraints
    pub constraints: Vec<OptimizationConstraint>,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
}

/// Optimization targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Minimize measurement uncertainty
    MinimizeUncertainty,
    /// Maximize precision
    MaximizePrecision,
    /// Minimize measurement time
    MinimizeMeasurementTime,
    /// Maximize signal-to-noise ratio
    MaximizeSignalToNoise,
    /// Minimize power consumption
    MinimizePowerConsumption,
}

/// Optimization constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Constraint value
    pub value: f64,
    /// Constraint operator
    pub operator: ConstraintOperator,
}

/// Types of optimization constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Maximum measurement time
    MaxMeasurementTime,
    /// Maximum power consumption
    MaxPowerConsumption,
    /// Minimum signal-to-noise ratio
    MinSignalToNoise,
    /// Maximum environmental sensitivity
    MaxEnvironmentalSensitivity,
    /// Minimum measurement confidence
    MinMeasurementConfidence,
}

/// Constraint operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintOperator {
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Equal
    Equal,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Gradient descent
    GradientDescent,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Particle swarm optimization
    ParticleSwarmOptimization,
    /// Bayesian optimization
    BayesianOptimization,
}

/// Convergence criteria
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Relative tolerance
    pub relative_tolerance: f64,
    /// Absolute tolerance
    pub absolute_tolerance: f64,
    /// Maximum function evaluations
    pub max_function_evaluations: usize,
    /// Stall generations
    pub stall_generations: usize,
}

impl PrecisionMeasurementConfig {
    /// Creates a default configuration for target precision
    pub fn new(target_precision: PrecisionLevel) -> Self {
        Self {
            target_precision,
            measurement_duration: Duration::from_secs(60),
            sampling_rate: 1e6, // 1 MHz
            measurement_cycles: 100,
            noise_reduction: NoiseReductionConfig::default(),
            calibration: CalibrationConfig::default(),
            environmental_compensation: EnvironmentalCompensationConfig::default(),
        }
    }

    /// Creates configuration for ultra-precision measurements
    pub fn ultra_precision() -> Self {
        Self {
            target_precision: PrecisionLevel::Ultimate,
            measurement_duration: Duration::from_secs(600), // 10 minutes
            sampling_rate: 1e9, // 1 GHz
            measurement_cycles: 1000,
            noise_reduction: NoiseReductionConfig::ultra_precision(),
            calibration: CalibrationConfig::ultra_precision(),
            environmental_compensation: EnvironmentalCompensationConfig::ultra_precision(),
        }
    }
}

impl NoiseReductionConfig {
    /// Creates default noise reduction configuration
    pub fn default() -> Self {
        Self {
            thermal_noise_reduction: true,
            shot_noise_reduction: true,
            flicker_noise_reduction: true,
            quantization_noise_reduction: true,
            custom_filters: vec![],
        }
    }

    /// Creates ultra-precision noise reduction configuration
    pub fn ultra_precision() -> Self {
        Self {
            thermal_noise_reduction: true,
            shot_noise_reduction: true,
            flicker_noise_reduction: true,
            quantization_noise_reduction: true,
            custom_filters: vec![
                NoiseFilter {
                    filter_type: NoiseFilterType::LowPass,
                    parameters: vec![1e6], // 1 MHz cutoff
                    enabled: true,
                },
                NoiseFilter {
                    filter_type: NoiseFilterType::Notch,
                    parameters: vec![50.0, 1.0], // 50 Hz notch, 1 Hz bandwidth
                    enabled: true,
                },
                NoiseFilter {
                    filter_type: NoiseFilterType::Adaptive,
                    parameters: vec![0.01, 0.99], // Learning rate, forgetting factor
                    enabled: true,
                },
            ],
        }
    }
}

impl CalibrationConfig {
    /// Creates default calibration configuration
    pub fn default() -> Self {
        Self {
            reference_frequency: 10e6, // 10 MHz
            calibration_interval: Duration::from_secs(3600), // 1 hour
            accuracy_requirement: 1e-12,
            temperature_calibration: true,
            pressure_calibration: true,
            humidity_calibration: true,
        }
    }

    /// Creates ultra-precision calibration configuration
    pub fn ultra_precision() -> Self {
        Self {
            reference_frequency: 10e6, // 10 MHz
            calibration_interval: Duration::from_secs(600), // 10 minutes
            accuracy_requirement: 1e-15,
            temperature_calibration: true,
            pressure_calibration: true,
            humidity_calibration: true,
        }
    }
}

impl EnvironmentalCompensationConfig {
    /// Creates default environmental compensation configuration
    pub fn default() -> Self {
        Self {
            temperature_compensation: TemperatureCompensation::default(),
            pressure_compensation: PressureCompensation::default(),
            humidity_compensation: HumidityCompensation::default(),
            magnetic_field_compensation: MagneticFieldCompensation::default(),
            vibration_compensation: VibrationCompensation::default(),
        }
    }

    /// Creates ultra-precision environmental compensation configuration
    pub fn ultra_precision() -> Self {
        Self {
            temperature_compensation: TemperatureCompensation::ultra_precision(),
            pressure_compensation: PressureCompensation::ultra_precision(),
            humidity_compensation: HumidityCompensation::ultra_precision(),
            magnetic_field_compensation: MagneticFieldCompensation::ultra_precision(),
            vibration_compensation: VibrationCompensation::ultra_precision(),
        }
    }
}

impl TemperatureCompensation {
    /// Creates default temperature compensation
    pub fn default() -> Self {
        Self {
            enabled: true,
            temperature_coefficient: 1e-6, // 1 ppm/°C
            reference_temperature: 20.0,   // 20°C
            temperature_uncertainty: 0.1,  // 0.1°C
        }
    }

    /// Creates ultra-precision temperature compensation
    pub fn ultra_precision() -> Self {
        Self {
            enabled: true,
            temperature_coefficient: 1e-9, // 1 ppb/°C
            reference_temperature: 20.0,   // 20°C
            temperature_uncertainty: 0.001, // 1 mK
        }
    }
}

impl PressureCompensation {
    /// Creates default pressure compensation
    pub fn default() -> Self {
        Self {
            enabled: true,
            pressure_coefficient: 1e-11, // 10 ppb/Pa
            reference_pressure: 101325.0, // 1 atm
            pressure_uncertainty: 10.0,   // 10 Pa
        }
    }

    /// Creates ultra-precision pressure compensation
    pub fn ultra_precision() -> Self {
        Self {
            enabled: true,
            pressure_coefficient: 1e-12, // 1 ppb/Pa
            reference_pressure: 101325.0, // 1 atm
            pressure_uncertainty: 0.1,    // 0.1 Pa
        }
    }
}

impl HumidityCompensation {
    /// Creates default humidity compensation
    pub fn default() -> Self {
        Self {
            enabled: true,
            humidity_coefficient: 1e-6, // 1 ppm/%
            reference_humidity: 50.0,   // 50%
            humidity_uncertainty: 1.0,  // 1%
        }
    }

    /// Creates ultra-precision humidity compensation
    pub fn ultra_precision() -> Self {
        Self {
            enabled: true,
            humidity_coefficient: 1e-9, // 1 ppb/%
            reference_humidity: 50.0,   // 50%
            humidity_uncertainty: 0.01, // 0.01%
        }
    }
}

impl MagneticFieldCompensation {
    /// Creates default magnetic field compensation
    pub fn default() -> Self {
        Self {
            enabled: true,
            magnetic_field_coefficient: 1e-6, // 1 ppm/T
            reference_magnetic_field: 5e-5,   // ~Earth's field
            magnetic_field_uncertainty: 1e-6, // 1 µT
        }
    }

    /// Creates ultra-precision magnetic field compensation
    pub fn ultra_precision() -> Self {
        Self {
            enabled: true,
            magnetic_field_coefficient: 1e-9, // 1 ppb/T
            reference_magnetic_field: 5e-5,   // ~Earth's field
            magnetic_field_uncertainty: 1e-9, // 1 nT
        }
    }
}

impl VibrationCompensation {
    /// Creates default vibration compensation
    pub fn default() -> Self {
        Self {
            enabled: true,
            vibration_sensitivity: 1e-6, // 1 ppm/(m/s²)
            frequency_range: (1.0, 1000.0), // 1 Hz to 1 kHz
            vibration_uncertainty: 1e-3, // 1 mm/s²
        }
    }

    /// Creates ultra-precision vibration compensation
    pub fn ultra_precision() -> Self {
        Self {
            enabled: true,
            vibration_sensitivity: 1e-9, // 1 ppb/(m/s²)
            frequency_range: (0.1, 10000.0), // 0.1 Hz to 10 kHz
            vibration_uncertainty: 1e-6, // 1 µm/s²
        }
    }
}

impl NoiseType {
    /// Gets the Allan variance slope for this noise type
    pub fn allan_variance_slope(&self) -> f64 {
        match self {
            NoiseType::WhitePhase => -1.0,
            NoiseType::FlickerPhase => -1.0,
            NoiseType::WhiteFrequency => 0.0,
            NoiseType::FlickerFrequency => 0.0,
            NoiseType::RandomWalkFrequency => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_measurement_config() {
        let config = PrecisionMeasurementConfig::new(PrecisionLevel::Target);
        assert_eq!(config.target_precision, PrecisionLevel::Target);
        assert!(config.sampling_rate > 0.0);
        assert!(config.measurement_cycles > 0);
    }

    #[test]
    fn test_ultra_precision_config() {
        let config = PrecisionMeasurementConfig::ultra_precision();
        assert_eq!(config.target_precision, PrecisionLevel::Ultimate);
        assert!(config.sampling_rate >= 1e9);
        assert!(config.measurement_cycles >= 1000);
    }

    #[test]
    fn test_noise_reduction_config() {
        let config = NoiseReductionConfig::ultra_precision();
        assert!(config.thermal_noise_reduction);
        assert!(config.shot_noise_reduction);
        assert!(config.flicker_noise_reduction);
        assert!(config.quantization_noise_reduction);
        assert!(!config.custom_filters.is_empty());
    }

    #[test]
    fn test_environmental_compensation() {
        let config = EnvironmentalCompensationConfig::ultra_precision();
        assert!(config.temperature_compensation.enabled);
        assert!(config.pressure_compensation.enabled);
        assert!(config.humidity_compensation.enabled);
        assert!(config.magnetic_field_compensation.enabled);
        assert!(config.vibration_compensation.enabled);
    }

    #[test]
    fn test_noise_type_slopes() {
        assert_eq!(NoiseType::WhitePhase.allan_variance_slope(), -1.0);
        assert_eq!(NoiseType::WhiteFrequency.allan_variance_slope(), 0.0);
        assert_eq!(NoiseType::RandomWalkFrequency.allan_variance_slope(), 1.0);
    }
} 