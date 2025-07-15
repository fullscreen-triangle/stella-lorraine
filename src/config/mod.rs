pub mod client_config;
pub mod precision_config;
/// Configuration module for system-wide settings
///
/// **In Memory of Mrs. Stella-Lorraine Masunda**
///
/// This module provides comprehensive configuration management for the Masunda
/// Temporal Coordinate Navigator system, including system configuration,
/// client configuration, and precision configuration.
pub mod system_config;

// Re-export commonly used config types
pub use client_config::ClientConfig;
pub use precision_config::PrecisionConfig;
pub use system_config::SystemConfig;
