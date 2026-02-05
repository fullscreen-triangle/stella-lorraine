//! Error types for the CatCount library

use thiserror::Error;

/// Result type alias for CatCount operations
pub type Result<T> = std::result::Result<T, CatCountError>;

/// Errors that can occur in CatCount operations
#[derive(Error, Debug)]
pub enum CatCountError {
    /// Invalid frequency value
    #[error("Invalid frequency: {0} Hz (must be positive and finite)")]
    InvalidFrequency(f64),

    /// Invalid oscillator count
    #[error("Invalid oscillator count: {0} (must be positive)")]
    InvalidOscillatorCount(u64),

    /// Invalid state count
    #[error("Invalid state count: {0} (must be at least 2)")]
    InvalidStateCount(u64),

    /// Invalid partition quantum number
    #[error("Invalid partition number {name}: {value} (constraint: {constraint})")]
    InvalidPartitionNumber {
        name: &'static str,
        value: i64,
        constraint: String,
    },

    /// Invalid S-entropy coordinate
    #[error("Invalid S-entropy coordinate {name}: {value} (must be non-negative)")]
    InvalidSEntropyCoord {
        name: &'static str,
        value: f64,
    },

    /// Invalid enhancement parameter
    #[error("Invalid enhancement parameter: {0}")]
    InvalidEnhancementParam(String),

    /// Computation overflow
    #[error("Computation overflow: {0}")]
    Overflow(String),

    /// Computation underflow
    #[error("Computation underflow: {0}")]
    Underflow(String),

    /// Division by zero
    #[error("Division by zero in {0}")]
    DivisionByZero(&'static str),

    /// Spectroscopy error
    #[error("Spectroscopy error: {0}")]
    SpectroscopyError(String),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl CatCountError {
    /// Create an invalid frequency error
    pub fn invalid_frequency(freq: f64) -> Self {
        Self::InvalidFrequency(freq)
    }

    /// Create an invalid partition number error
    pub fn invalid_partition(name: &'static str, value: i64, constraint: impl Into<String>) -> Self {
        Self::InvalidPartitionNumber {
            name,
            value,
            constraint: constraint.into(),
        }
    }

    /// Create an invalid S-entropy coordinate error
    pub fn invalid_s_entropy(name: &'static str, value: f64) -> Self {
        Self::InvalidSEntropyCoord { name, value }
    }
}
