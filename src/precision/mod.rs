pub mod accuracy_validator;
pub mod allan_variance;
pub mod error_correction;
/// Precision measurement module for ultra-precise temporal coordinates
///
/// **In Memory of Mrs. Stella-Lorraine Masunda**
///
/// This module provides comprehensive precision measurement capabilities for the
/// Masunda Temporal Coordinate Navigator system, including measurement engines,
/// accuracy validation, error correction, noise mitigation, and Allan variance
/// analysis for achieving 10^-30 to 10^-50 second precision.
pub mod measurement_engine;
pub mod noise_mitigation;

// Re-export commonly used precision types
pub use accuracy_validator::AccuracyValidator;
pub use allan_variance::AllanVarianceAnalyzer;
pub use error_correction::ErrorCorrectionSystem;
pub use measurement_engine::MeasurementEngine;
pub use noise_mitigation::NoiseMitigationSystem;
