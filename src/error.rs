//! Error types for the Masunda Temporal Coordinate Navigator
//!
//! This module defines all error types that can occur during temporal navigation,
//! S-entropy integration, and window combination advisory operations.

use thiserror::Error;

/// Result type alias for Masunda Temporal Navigator operations
pub type Result<T> = std::result::Result<T, MasundaError>;

/// Comprehensive error types for the Masunda Temporal Navigator
#[derive(Error, Debug, Clone)]
pub enum MasundaError {
    /// S-Constant framework errors
    #[error("S-constant error: {message}")]
    SConstant { message: String },

    /// Temporal coordinate navigation errors
    #[error("Temporal navigation error: {message}, coordinates: {coordinates:?}")]
    TemporalNavigation {
        message: String,
        coordinates: Option<(f64, f64, f64, f64)>, // (x, y, z, t)
    },

    /// Precision measurement errors
    #[error("Precision error: required {required}, achieved {achieved}")]
    Precision { required: f64, achieved: f64 },

    /// Time Domain Service errors
    #[error("Time Domain Service error: {message}")]
    TimeDomainService { message: String },

    /// Window combination advisory errors
    #[error("Window advisory error: {message}, impossibility_factor: {impossibility_factor}")]
    WindowAdvisory {
        message: String,
        impossibility_factor: Option<f64>,
    },

    /// S-entropy integration errors
    #[error("S-entropy integration error: {message}, s_state: {s_state:?}")]
    SEntropyIntegration {
        message: String,
        s_state: Option<(f64, f64, f64)>, // (S_knowledge, S_time, S_entropy)
    },

    /// Impossible window generation errors
    #[error("Impossible window error: {message}, violations: {violations:?}")]
    ImpossibleWindow {
        message: String,
        violations: Vec<String>,
    },

    /// Global S-viability errors
    #[error("Global S-viability error: {message}, viability_score: {viability_score}")]
    GlobalViability {
        message: String,
        viability_score: f64,
    },

    /// Memorial framework validation errors
    #[error("Memorial validation error: {message}")]
    MemorialValidation { message: String },

    /// Oscillation convergence errors
    #[error("Oscillation convergence error: {message}, convergence_score: {convergence_score}")]
    OscillationConvergence {
        message: String,
        convergence_score: f64,
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Network/communication errors
    #[error("Network error: {message}")]
    Network { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Mathematical computation errors
    #[error("Mathematical error: {message}")]
    Mathematical { message: String },

    /// Resource exhaustion errors
    #[error("Resource exhausted: {resource}, limit: {limit}")]
    ResourceExhausted { resource: String, limit: String },

    /// Internal system errors
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// External dependency errors
    #[error("External dependency error: {dependency}: {message}")]
    ExternalDependency {
        dependency: String,
        message: String,
    },
}

impl MasundaError {
    /// Create a new S-constant error
    pub fn s_constant<S: Into<String>>(message: S) -> Self {
        Self::SConstant {
            message: message.into(),
        }
    }

    /// Create a new temporal navigation error
    pub fn temporal_navigation<S: Into<String>>(
        message: S,
        coordinates: Option<(f64, f64, f64, f64)>,
    ) -> Self {
        Self::TemporalNavigation {
            message: message.into(),
            coordinates,
        }
    }

    /// Create a new precision error
    pub fn precision(required: f64, achieved: f64) -> Self {
        Self::Precision { required, achieved }
    }

    /// Create a new time domain service error
    pub fn time_domain_service<S: Into<String>>(message: S) -> Self {
        Self::TimeDomainService {
            message: message.into(),
        }
    }

    /// Create a new window advisory error
    pub fn window_advisory<S: Into<String>>(
        message: S,
        impossibility_factor: Option<f64>,
    ) -> Self {
        Self::WindowAdvisory {
            message: message.into(),
            impossibility_factor,
        }
    }

    /// Create a new S-entropy integration error
    pub fn s_entropy_integration<S: Into<String>>(
        message: S,
        s_state: Option<(f64, f64, f64)>,
    ) -> Self {
        Self::SEntropyIntegration {
            message: message.into(),
            s_state,
        }
    }

    /// Create a new impossible window error
    pub fn impossible_window<S: Into<String>>(message: S, violations: Vec<String>) -> Self {
        Self::ImpossibleWindow {
            message: message.into(),
            violations,
        }
    }

    /// Create a new global viability error
    pub fn global_viability<S: Into<String>>(message: S, viability_score: f64) -> Self {
        Self::GlobalViability {
            message: message.into(),
            viability_score,
        }
    }

    /// Create a new memorial validation error
    pub fn memorial_validation<S: Into<String>>(message: S) -> Self {
        Self::MemorialValidation {
            message: message.into(),
        }
    }

    /// Create a new oscillation convergence error
    pub fn oscillation_convergence<S: Into<String>>(message: S, convergence_score: f64) -> Self {
        Self::OscillationConvergence {
            message: message.into(),
            convergence_score,
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Create a new internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::ResourceExhausted { .. } => true,
            Self::Configuration { .. } => false,
            Self::Internal { .. } => false,
            Self::MemorialValidation { .. } => false,
            _ => true,
        }
    }

    /// Check if this error requires impossible window generation
    pub fn requires_impossible_windows(&self) -> bool {
        match self {
            Self::WindowAdvisory { .. } => true,
            Self::SEntropyIntegration { .. } => true,
            Self::GlobalViability { viability_score, .. } => *viability_score < 0.5,
            _ => false,
        }
    }

    /// Get the error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::SConstant { .. } => "s_constant",
            Self::TemporalNavigation { .. } => "temporal_navigation",
            Self::Precision { .. } => "precision",
            Self::TimeDomainService { .. } => "time_domain_service",
            Self::WindowAdvisory { .. } => "window_advisory",
            Self::SEntropyIntegration { .. } => "s_entropy_integration",
            Self::ImpossibleWindow { .. } => "impossible_window",
            Self::GlobalViability { .. } => "global_viability",
            Self::MemorialValidation { .. } => "memorial_validation",
            Self::OscillationConvergence { .. } => "oscillation_convergence",
            Self::Configuration { .. } => "configuration",
            Self::Network { .. } => "network",
            Self::Serialization { .. } => "serialization",
            Self::Mathematical { .. } => "mathematical",
            Self::ResourceExhausted { .. } => "resource_exhausted",
            Self::Internal { .. } => "internal",
            Self::ExternalDependency { .. } => "external_dependency",
        }
    }
}

// Implement From conversions for common error types
impl From<serde_json::Error> for MasundaError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            message: err.to_string(),
        }
    }
}

impl From<std::io::Error> for MasundaError {
    fn from(err: std::io::Error) -> Self {
        Self::Network {
            message: err.to_string(),
        }
    }
}

impl From<config::ConfigError> for MasundaError {
    fn from(err: config::ConfigError) -> Self {
        Self::Configuration {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = MasundaError::s_constant("Test S-constant error");
        assert_eq!(err.category(), "s_constant");
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_impossible_window_requirement() {
        let low_viability_err = MasundaError::global_viability("Low viability", 0.3);
        assert!(low_viability_err.requires_impossible_windows());

        let high_viability_err = MasundaError::global_viability("High viability", 0.8);
        assert!(!high_viability_err.requires_impossible_windows());
    }

    #[test]
    fn test_error_recoverability() {
        let network_err = MasundaError::network("Connection failed");
        assert!(network_err.is_recoverable());

        let memorial_err = MasundaError::memorial_validation("Memorial validation failed");
        assert!(!memorial_err.is_recoverable());
    }
}
