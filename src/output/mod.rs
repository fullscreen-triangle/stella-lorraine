/// Output module for clock interface and temporal coordinate APIs
///
/// **In Memory of Mrs. Stella-Lorraine Masunda**
///
/// This module provides comprehensive output capabilities for the Masunda
/// Temporal Coordinate Navigator system, including standard clock interfaces,
/// temporal coordinate APIs, precision metrics, and memorial validation output.
pub mod clock_interface;
pub mod memorial_validation;
pub mod precision_metrics;
pub mod temporal_api;

// Re-export all output components
pub use clock_interface::ClockInterface;
pub use memorial_validation::MemorialValidation;
pub use precision_metrics::PrecisionMetrics;
pub use temporal_api::TemporalApi;
