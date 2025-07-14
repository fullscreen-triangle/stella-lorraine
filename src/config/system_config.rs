use crate::types::*;
use std::time::Duration;

/// System configuration for the Masunda Navigator
#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub precision_target: PrecisionLevel,
    pub memorial_validation: bool,
    pub quantum_enhancement: bool,
    pub max_navigation_time: Duration,
    pub confidence_threshold: f64,
    pub superposition_size: usize,
    pub memorial_significance_threshold: f64,
    pub oscillation_convergence_threshold: f64,
    pub error_correction_enabled: bool,
    pub allan_variance_analysis: bool,
    pub continuous_operation: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            precision_target: PrecisionLevel::UltraPrecise,
            memorial_validation: true,
            quantum_enhancement: true,
            max_navigation_time: Duration::from_millis(100),
            confidence_threshold: 0.99,
            superposition_size: 1000,
            memorial_significance_threshold: 0.95,
            oscillation_convergence_threshold: 0.99,
            error_correction_enabled: true,
            allan_variance_analysis: true,
            continuous_operation: true,
        }
    }
}
