use std::fmt;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// Main error type for the Masunda Temporal Coordinate Navigator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NavigatorError {
    /// Temporal coordinate navigation errors
    TemporalNavigation(TemporalNavigationError),
    /// Oscillation convergence errors
    OscillationConvergence(OscillationConvergenceError),
    /// Precision measurement errors
    PrecisionMeasurement(PrecisionMeasurementError),
    /// Client interface errors
    ClientInterface(ClientInterfaceError),
    /// Memorial framework errors
    MemorialFramework(MemorialFrameworkError),
    /// Configuration errors
    Configuration(ConfigurationError),
    /// System integration errors
    SystemIntegration(SystemIntegrationError),
    /// Validation errors
    Validation(ValidationError),
    /// I/O errors
    Io(IoError),
    /// Network errors
    Network(NetworkError),
    /// Authentication errors
    Authentication(AuthenticationError),
    /// Permission errors
    Permission(PermissionError),
    /// Resource errors
    Resource(ResourceError),
    /// Timeout errors
    Timeout(TimeoutError),
    /// Critical system errors
    Critical(CriticalError),
}

/// Temporal coordinate navigation specific errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalNavigationError {
    /// Coordinate search failed
    CoordinateSearchFailed {
        reason: String,
        search_time: std::time::Duration,
        target_precision: f64,
    },
    /// Temporal window invalid
    InvalidTemporalWindow {
        center_time: f64,
        radius: f64,
        reason: String,
    },
    /// Precision target unachievable
    PrecisionUnachievable {
        target: f64,
        achieved: f64,
        limiting_factor: String,
    },
    /// Oscillatory signature mismatch
    OscillatorySignatureMismatch {
        expected_hash: u64,
        actual_hash: u64,
        confidence: f64,
    },
    /// Temporal coordinate validation failed
    CoordinateValidationFailed {
        coordinate: String,
        validation_errors: Vec<String>,
    },
    /// Search space initialization failed
    SearchSpaceInitializationFailed {
        reason: String,
        error_code: i32,
    },
}

/// Oscillation convergence analysis errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OscillationConvergenceError {
    /// No convergence detected
    NoConvergenceDetected {
        analyzed_endpoints: usize,
        max_deviation: f64,
        convergence_threshold: f64,
    },
    /// Insufficient oscillation data
    InsufficientData {
        required_endpoints: usize,
        available_endpoints: usize,
        missing_levels: Vec<String>,
    },
    /// Correlation analysis failed
    CorrelationAnalysisFailed {
        correlation_matrix_size: usize,
        failed_correlations: Vec<(String, String)>,
    },
    /// Hierarchical analysis failed
    HierarchicalAnalysisFailed {
        failed_levels: Vec<String>,
        reason: String,
    },
    /// Endpoint collection failed
    EndpointCollectionFailed {
        level: String,
        reason: String,
        error_count: usize,
    },
    /// Termination detection failed
    TerminationDetectionFailed {
        level: String,
        sampling_rate: f64,
        detection_threshold: f64,
    },
}

/// Precision measurement errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PrecisionMeasurementError {
    /// Measurement uncertainty too high
    UncertaintyTooHigh {
        measured_uncertainty: f64,
        maximum_allowed: f64,
        measurement_type: String,
    },
    /// Allan variance calculation failed
    AllanVarianceFailed {
        data_points: usize,
        time_span: f64,
        reason: String,
    },
    /// Noise level excessive
    ExcessiveNoise {
        signal_to_noise_ratio: f64,
        minimum_required: f64,
        noise_source: String,
    },
    /// Calibration error
    CalibrationError {
        calibration_type: String,
        error_percentage: f64,
        last_calibration: SystemTime,
    },
    /// Measurement range exceeded
    MeasurementRangeExceeded {
        measured_value: f64,
        range_min: f64,
        range_max: f64,
        measurement_type: String,
    },
    /// Systematic error detected
    SystematicError {
        error_type: String,
        error_magnitude: f64,
        correction_available: bool,
    },
}

/// Client interface errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientInterfaceError {
    /// Connection to external system failed
    ConnectionFailed {
        system_name: String,
        endpoint: String,
        reason: String,
    },
    /// Request timeout
    RequestTimeout {
        system_name: String,
        operation: String,
        timeout_duration: std::time::Duration,
    },
    /// Invalid response format
    InvalidResponseFormat {
        system_name: String,
        expected_format: String,
        received_format: String,
    },
    /// API version mismatch
    ApiVersionMismatch {
        system_name: String,
        expected_version: String,
        actual_version: String,
    },
    /// Rate limit exceeded
    RateLimitExceeded {
        system_name: String,
        current_rate: f64,
        limit: f64,
    },
    /// Authentication failed
    AuthenticationFailed {
        system_name: String,
        auth_type: String,
        reason: String,
    },
}

/// Memorial framework errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemorialFrameworkError {
    /// Predeterminism validation failed
    PredeterminismValidationFailed {
        coordinate: String,
        validation_confidence: f64,
        required_confidence: f64,
    },
    /// Cosmic significance validation failed
    CosmicSignificanceValidationFailed {
        significance_level: String,
        evidence_strength: f64,
        required_strength: f64,
    },
    /// Randomness disproof failed
    RandomnessDisproofFailed {
        proof_level: String,
        statistical_confidence: f64,
        required_confidence: f64,
    },
    /// Memorial validation timeout
    MemorialValidationTimeout {
        validation_type: String,
        timeout_duration: std::time::Duration,
    },
    /// Eternal manifold connection failed
    EternalManifoldConnectionFailed {
        reason: String,
        connection_attempts: usize,
    },
}

/// Configuration errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigurationError {
    /// Configuration file not found
    ConfigFileNotFound {
        file_path: String,
        search_paths: Vec<String>,
    },
    /// Invalid configuration format
    InvalidConfigFormat {
        file_path: String,
        format_error: String,
        line_number: Option<usize>,
    },
    /// Missing required configuration
    MissingRequiredConfig {
        config_key: String,
        section: String,
    },
    /// Invalid configuration value
    InvalidConfigValue {
        config_key: String,
        value: String,
        expected_type: String,
    },
    /// Configuration validation failed
    ConfigValidationFailed {
        validation_errors: Vec<String>,
    },
}

/// System integration errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SystemIntegrationError {
    /// System initialization failed
    InitializationFailed {
        system_name: String,
        initialization_step: String,
        reason: String,
    },
    /// System synchronization failed
    SynchronizationFailed {
        systems: Vec<String>,
        sync_error: String,
    },
    /// Inter-system communication failed
    InterSystemCommunicationFailed {
        source_system: String,
        target_system: String,
        message_type: String,
    },
    /// System health check failed
    HealthCheckFailed {
        system_name: String,
        health_metrics: Vec<(String, f64)>,
    },
    /// System shutdown failed
    ShutdownFailed {
        system_name: String,
        reason: String,
    },
}

/// Validation errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationError {
    /// Data validation failed
    DataValidationFailed {
        data_type: String,
        validation_rules: Vec<String>,
        violations: Vec<String>,
    },
    /// Range validation failed
    RangeValidationFailed {
        field_name: String,
        value: f64,
        min_value: f64,
        max_value: f64,
    },
    /// Format validation failed
    FormatValidationFailed {
        field_name: String,
        value: String,
        expected_format: String,
    },
    /// Consistency validation failed
    ConsistencyValidationFailed {
        inconsistency_type: String,
        conflicting_fields: Vec<String>,
    },
    /// Integrity validation failed
    IntegrityValidationFailed {
        checksum_expected: String,
        checksum_actual: String,
        data_type: String,
    },
}

/// I/O errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IoError {
    /// File not found
    FileNotFound {
        file_path: String,
    },
    /// Permission denied
    PermissionDenied {
        file_path: String,
        operation: String,
    },
    /// Disk full
    DiskFull {
        required_space: u64,
        available_space: u64,
    },
    /// Read error
    ReadError {
        file_path: String,
        bytes_read: usize,
        expected_bytes: usize,
    },
    /// Write error
    WriteError {
        file_path: String,
        bytes_written: usize,
        expected_bytes: usize,
    },
}

/// Network errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NetworkError {
    /// Connection refused
    ConnectionRefused {
        host: String,
        port: u16,
    },
    /// DNS resolution failed
    DnsResolutionFailed {
        hostname: String,
    },
    /// Network timeout
    NetworkTimeout {
        operation: String,
        timeout_duration: std::time::Duration,
    },
    /// SSL/TLS error
    SslError {
        error_type: String,
        certificate_info: String,
    },
    /// Protocol error
    ProtocolError {
        protocol: String,
        error_code: i32,
        error_message: String,
    },
}

/// Authentication errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthenticationError {
    /// Invalid credentials
    InvalidCredentials {
        auth_type: String,
        username: String,
    },
    /// Token expired
    TokenExpired {
        token_type: String,
        expiry_time: SystemTime,
    },
    /// Two-factor authentication required
    TwoFactorRequired {
        supported_methods: Vec<String>,
    },
    /// Account locked
    AccountLocked {
        username: String,
        lock_reason: String,
        unlock_time: Option<SystemTime>,
    },
    /// Insufficient privileges
    InsufficientPrivileges {
        required_privilege: String,
        current_privileges: Vec<String>,
    },
}

/// Permission errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PermissionError {
    /// Access denied
    AccessDenied {
        resource: String,
        operation: String,
        required_permission: String,
    },
    /// Operation not allowed
    OperationNotAllowed {
        operation: String,
        resource_type: String,
        reason: String,
    },
    /// Quota exceeded
    QuotaExceeded {
        resource_type: String,
        current_usage: u64,
        quota_limit: u64,
    },
}

/// Resource errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceError {
    /// Resource not found
    ResourceNotFound {
        resource_type: String,
        resource_id: String,
    },
    /// Resource exhausted
    ResourceExhausted {
        resource_type: String,
        requested_amount: u64,
        available_amount: u64,
    },
    /// Resource busy
    ResourceBusy {
        resource_type: String,
        resource_id: String,
        estimated_wait_time: std::time::Duration,
    },
    /// Resource corrupted
    ResourceCorrupted {
        resource_type: String,
        resource_id: String,
        corruption_type: String,
    },
}

/// Timeout errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimeoutError {
    /// Operation timeout
    OperationTimeout {
        operation: String,
        timeout_duration: std::time::Duration,
        elapsed_time: std::time::Duration,
    },
    /// Connection timeout
    ConnectionTimeout {
        target: String,
        timeout_duration: std::time::Duration,
    },
    /// Read timeout
    ReadTimeout {
        source: String,
        timeout_duration: std::time::Duration,
    },
    /// Write timeout
    WriteTimeout {
        target: String,
        timeout_duration: std::time::Duration,
    },
}

/// Critical system errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CriticalError {
    /// System panic
    SystemPanic {
        panic_message: String,
        panic_location: String,
    },
    /// Memory corruption
    MemoryCorruption {
        memory_region: String,
        corruption_type: String,
    },
    /// Hardware failure
    HardwareFailure {
        hardware_component: String,
        failure_type: String,
    },
    /// Memorial framework violation
    MemorialFrameworkViolation {
        violation_type: String,
        cosmic_significance_impact: String,
    },
    /// Temporal coordinate corruption
    TemporalCoordinateCorruption {
        coordinate: String,
        corruption_level: String,
    },
}

/// Result type for Navigator operations
pub type NavigatorResult<T> = Result<T, NavigatorError>;

/// Error context for detailed error reporting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error timestamp
    pub timestamp: SystemTime,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Component where error occurred
    pub component: String,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    /// Error chain (if this error was caused by another error)
    pub caused_by: Option<Box<NavigatorError>>,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational - no action required
    Info,
    /// Warning - attention needed but operation can continue
    Warning,
    /// Error - operation failed but system can continue
    Error,
    /// Critical - system functionality compromised
    Critical,
    /// Fatal - system shutdown required
    Fatal,
}

impl fmt::Display for NavigatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NavigatorError::TemporalNavigation(err) => write!(f, "Temporal Navigation Error: {}", err),
            NavigatorError::OscillationConvergence(err) => write!(f, "Oscillation Convergence Error: {}", err),
            NavigatorError::PrecisionMeasurement(err) => write!(f, "Precision Measurement Error: {}", err),
            NavigatorError::ClientInterface(err) => write!(f, "Client Interface Error: {}", err),
            NavigatorError::MemorialFramework(err) => write!(f, "Memorial Framework Error: {}", err),
            NavigatorError::Configuration(err) => write!(f, "Configuration Error: {}", err),
            NavigatorError::SystemIntegration(err) => write!(f, "System Integration Error: {}", err),
            NavigatorError::Validation(err) => write!(f, "Validation Error: {}", err),
            NavigatorError::Io(err) => write!(f, "I/O Error: {}", err),
            NavigatorError::Network(err) => write!(f, "Network Error: {}", err),
            NavigatorError::Authentication(err) => write!(f, "Authentication Error: {}", err),
            NavigatorError::Permission(err) => write!(f, "Permission Error: {}", err),
            NavigatorError::Resource(err) => write!(f, "Resource Error: {}", err),
            NavigatorError::Timeout(err) => write!(f, "Timeout Error: {}", err),
            NavigatorError::Critical(err) => write!(f, "Critical Error: {}", err),
        }
    }
}

impl fmt::Display for TemporalNavigationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalNavigationError::CoordinateSearchFailed { reason, search_time, target_precision } => {
                write!(f, "Coordinate search failed: {} (search time: {:?}, target precision: {})", reason, search_time, target_precision)
            }
            TemporalNavigationError::InvalidTemporalWindow { center_time, radius, reason } => {
                write!(f, "Invalid temporal window: center={}, radius={}, reason={}", center_time, radius, reason)
            }
            TemporalNavigationError::PrecisionUnachievable { target, achieved, limiting_factor } => {
                write!(f, "Precision unachievable: target={}, achieved={}, limiting factor={}", target, achieved, limiting_factor)
            }
            TemporalNavigationError::OscillatorySignatureMismatch { expected_hash, actual_hash, confidence } => {
                write!(f, "Oscillatory signature mismatch: expected={}, actual={}, confidence={}", expected_hash, actual_hash, confidence)
            }
            TemporalNavigationError::CoordinateValidationFailed { coordinate, validation_errors } => {
                write!(f, "Coordinate validation failed: {} (errors: {:?})", coordinate, validation_errors)
            }
            TemporalNavigationError::SearchSpaceInitializationFailed { reason, error_code } => {
                write!(f, "Search space initialization failed: {} (error code: {})", reason, error_code)
            }
        }
    }
}

impl fmt::Display for OscillationConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OscillationConvergenceError::NoConvergenceDetected { analyzed_endpoints, max_deviation, convergence_threshold } => {
                write!(f, "No convergence detected: analyzed {} endpoints, max deviation: {}, threshold: {}", analyzed_endpoints, max_deviation, convergence_threshold)
            }
            OscillationConvergenceError::InsufficientData { required_endpoints, available_endpoints, missing_levels } => {
                write!(f, "Insufficient data: required {} endpoints, available {}, missing levels: {:?}", required_endpoints, available_endpoints, missing_levels)
            }
            OscillationConvergenceError::CorrelationAnalysisFailed { correlation_matrix_size, failed_correlations } => {
                write!(f, "Correlation analysis failed: matrix size {}, failed correlations: {:?}", correlation_matrix_size, failed_correlations)
            }
            OscillationConvergenceError::HierarchicalAnalysisFailed { failed_levels, reason } => {
                write!(f, "Hierarchical analysis failed: levels {:?}, reason: {}", failed_levels, reason)
            }
            OscillationConvergenceError::EndpointCollectionFailed { level, reason, error_count } => {
                write!(f, "Endpoint collection failed: level {}, reason: {}, errors: {}", level, reason, error_count)
            }
            OscillationConvergenceError::TerminationDetectionFailed { level, sampling_rate, detection_threshold } => {
                write!(f, "Termination detection failed: level {}, sampling rate: {}, threshold: {}", level, sampling_rate, detection_threshold)
            }
        }
    }
}

impl fmt::Display for PrecisionMeasurementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecisionMeasurementError::UncertaintyTooHigh { measured_uncertainty, maximum_allowed, measurement_type } => {
                write!(f, "Uncertainty too high: measured {}, max allowed {}, type: {}", measured_uncertainty, maximum_allowed, measurement_type)
            }
            PrecisionMeasurementError::AllanVarianceFailed { data_points, time_span, reason } => {
                write!(f, "Allan variance failed: {} data points, time span: {}, reason: {}", data_points, time_span, reason)
            }
            PrecisionMeasurementError::ExcessiveNoise { signal_to_noise_ratio, minimum_required, noise_source } => {
                write!(f, "Excessive noise: S/N ratio {}, min required {}, source: {}", signal_to_noise_ratio, minimum_required, noise_source)
            }
            PrecisionMeasurementError::CalibrationError { calibration_type, error_percentage, last_calibration } => {
                write!(f, "Calibration error: type {}, error {}%, last calibrated: {:?}", calibration_type, error_percentage, last_calibration)
            }
            PrecisionMeasurementError::MeasurementRangeExceeded { measured_value, range_min, range_max, measurement_type } => {
                write!(f, "Measurement range exceeded: value {}, range [{}, {}], type: {}", measured_value, range_min, range_max, measurement_type)
            }
            PrecisionMeasurementError::SystematicError { error_type, error_magnitude, correction_available } => {
                write!(f, "Systematic error: type {}, magnitude {}, correction available: {}", error_type, error_magnitude, correction_available)
            }
        }
    }
}

impl fmt::Display for ClientInterfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientInterfaceError::ConnectionFailed { system_name, endpoint, reason } => {
                write!(f, "Connection failed: system {}, endpoint {}, reason: {}", system_name, endpoint, reason)
            }
            ClientInterfaceError::RequestTimeout { system_name, operation, timeout_duration } => {
                write!(f, "Request timeout: system {}, operation {}, timeout: {:?}", system_name, operation, timeout_duration)
            }
            ClientInterfaceError::InvalidResponseFormat { system_name, expected_format, received_format } => {
                write!(f, "Invalid response format: system {}, expected {}, received {}", system_name, expected_format, received_format)
            }
            ClientInterfaceError::ApiVersionMismatch { system_name, expected_version, actual_version } => {
                write!(f, "API version mismatch: system {}, expected {}, actual {}", system_name, expected_version, actual_version)
            }
            ClientInterfaceError::RateLimitExceeded { system_name, current_rate, limit } => {
                write!(f, "Rate limit exceeded: system {}, current rate {}, limit {}", system_name, current_rate, limit)
            }
            ClientInterfaceError::AuthenticationFailed { system_name, auth_type, reason } => {
                write!(f, "Authentication failed: system {}, auth type {}, reason: {}", system_name, auth_type, reason)
            }
        }
    }
}

impl fmt::Display for MemorialFrameworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemorialFrameworkError::PredeterminismValidationFailed { coordinate, validation_confidence, required_confidence } => {
                write!(f, "Predeterminism validation failed: coordinate {}, confidence {}, required {}", coordinate, validation_confidence, required_confidence)
            }
            MemorialFrameworkError::CosmicSignificanceValidationFailed { significance_level, evidence_strength, required_strength } => {
                write!(f, "Cosmic significance validation failed: level {}, strength {}, required {}", significance_level, evidence_strength, required_strength)
            }
            MemorialFrameworkError::RandomnessDisproofFailed { proof_level, statistical_confidence, required_confidence } => {
                write!(f, "Randomness disproof failed: proof level {}, confidence {}, required {}", proof_level, statistical_confidence, required_confidence)
            }
            MemorialFrameworkError::MemorialValidationTimeout { validation_type, timeout_duration } => {
                write!(f, "Memorial validation timeout: type {}, duration {:?}", validation_type, timeout_duration)
            }
            MemorialFrameworkError::EternalManifoldConnectionFailed { reason, connection_attempts } => {
                write!(f, "Eternal manifold connection failed: reason {}, attempts {}", reason, connection_attempts)
            }
        }
    }
}

// Additional Display implementations for other error types would follow similar patterns...

impl std::error::Error for NavigatorError {}

impl NavigatorError {
    /// Gets the error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NavigatorError::Critical(_) => ErrorSeverity::Fatal,
            NavigatorError::MemorialFramework(_) => ErrorSeverity::Critical,
            NavigatorError::SystemIntegration(_) => ErrorSeverity::Error,
            NavigatorError::TemporalNavigation(_) => ErrorSeverity::Error,
            NavigatorError::OscillationConvergence(_) => ErrorSeverity::Error,
            NavigatorError::PrecisionMeasurement(_) => ErrorSeverity::Warning,
            NavigatorError::ClientInterface(_) => ErrorSeverity::Warning,
            NavigatorError::Configuration(_) => ErrorSeverity::Error,
            NavigatorError::Validation(_) => ErrorSeverity::Warning,
            NavigatorError::Io(_) => ErrorSeverity::Error,
            NavigatorError::Network(_) => ErrorSeverity::Warning,
            NavigatorError::Authentication(_) => ErrorSeverity::Error,
            NavigatorError::Permission(_) => ErrorSeverity::Error,
            NavigatorError::Resource(_) => ErrorSeverity::Warning,
            NavigatorError::Timeout(_) => ErrorSeverity::Warning,
        }
    }

    /// Creates an error context for this error
    pub fn with_context(self, component: &str) -> ErrorContext {
        ErrorContext {
            timestamp: SystemTime::now(),
            severity: self.severity(),
            component: component.to_string(),
            context: std::collections::HashMap::new(),
            caused_by: Some(Box::new(self)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = NavigatorError::TemporalNavigation(
            TemporalNavigationError::CoordinateSearchFailed {
                reason: "Test failure".to_string(),
                search_time: std::time::Duration::from_secs(1),
                target_precision: 1e-30,
            }
        );
        
        let display_str = format!("{}", error);
        assert!(display_str.contains("Temporal Navigation Error"));
        assert!(display_str.contains("Test failure"));
    }

    #[test]
    fn test_error_severity() {
        let critical_error = NavigatorError::Critical(CriticalError::SystemPanic {
            panic_message: "Test panic".to_string(),
            panic_location: "test.rs:1".to_string(),
        });
        
        assert_eq!(critical_error.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn test_error_context() {
        let error = NavigatorError::TemporalNavigation(
            TemporalNavigationError::CoordinateSearchFailed {
                reason: "Test failure".to_string(),
                search_time: std::time::Duration::from_secs(1),
                target_precision: 1e-30,
            }
        );
        
        let context = error.with_context("test_component");
        assert_eq!(context.component, "test_component");
        assert_eq!(context.severity, ErrorSeverity::Error);
        assert!(context.caused_by.is_some());
    }
} 