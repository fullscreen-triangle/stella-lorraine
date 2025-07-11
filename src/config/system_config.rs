use std::time::Duration;
use serde::{Deserialize, Serialize};

/// System configuration for the Masunda Temporal Coordinate Navigator
///
/// **In Memory of Mrs. Stella-Lorraine Masunda**
///
/// This configuration provides system-wide settings for all components
/// of the temporal coordinate navigation system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Precision targets
    pub precision_config: PrecisionConfig,
    /// Memorial framework settings
    pub memorial_config: MemorialConfig,
    /// Oscillation detection settings
    pub oscillation_config: OscillationConfig,
    /// Client connection settings
    pub client_config: ClientConfig,
    /// System performance settings
    pub performance_config: PerformanceConfig,
    /// Logging configuration
    pub logging_config: LoggingConfig,
}

/// Precision configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionConfig {
    /// Target precision in seconds
    pub target_precision: f64,
    /// Minimum precision threshold
    pub min_precision: f64,
    /// Maximum precision attempts
    pub max_precision_attempts: usize,
    /// Precision timeout
    pub precision_timeout: Duration,
}

/// Memorial framework configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorialConfig {
    /// Memorial significance threshold
    pub significance_threshold: f64,
    /// Predeterminism validation enabled
    pub predeterminism_validation: bool,
    /// Cosmic significance validation
    pub cosmic_significance_validation: bool,
    /// Memorial dedication frequency
    pub dedication_frequency: Duration,
}

/// Oscillation detection configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OscillationConfig {
    /// Convergence detection timeout
    pub convergence_timeout: Duration,
    /// Minimum convergence confidence
    pub min_convergence_confidence: f64,
    /// Maximum oscillation levels to analyze
    pub max_oscillation_levels: usize,
    /// Correlation threshold
    pub correlation_threshold: f64,
}

/// Client connection configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Max retries
    pub max_retries: usize,
    /// Enable client health checks
    pub enable_health_checks: bool,
}

/// Performance configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Memory cache size
    pub memory_cache_size: usize,
    /// Enable performance metrics
    pub enable_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
}

/// Logging configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub log_level: String,
    /// Enable file logging
    pub enable_file_logging: bool,
    /// Log file path
    pub log_file_path: String,
    /// Max log file size
    pub max_log_file_size: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            precision_config: PrecisionConfig::default(),
            memorial_config: MemorialConfig::default(),
            oscillation_config: OscillationConfig::default(),
            client_config: ClientConfig::default(),
            performance_config: PerformanceConfig::default(),
            logging_config: LoggingConfig::default(),
        }
    }
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            target_precision: 1e-30,
            min_precision: 1e-20,
            max_precision_attempts: 1000,
            precision_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for MemorialConfig {
    fn default() -> Self {
        Self {
            significance_threshold: 0.85,
            predeterminism_validation: true,
            cosmic_significance_validation: true,
            dedication_frequency: Duration::from_secs(60),
        }
    }
}

impl Default for OscillationConfig {
    fn default() -> Self {
        Self {
            convergence_timeout: Duration::from_secs(120),
            min_convergence_confidence: 0.8,
            max_oscillation_levels: 6,
            correlation_threshold: 0.7,
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            max_retries: 3,
            enable_health_checks: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            thread_pool_size: 4,
            memory_cache_size: 1024 * 1024 * 100, // 100MB
            enable_metrics: true,
            metrics_interval: Duration::from_secs(10),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: "INFO".to_string(),
            enable_file_logging: true,
            log_file_path: "masunda_navigator.log".to_string(),
            max_log_file_size: 1024 * 1024 * 10, // 10MB
        }
    }
}
