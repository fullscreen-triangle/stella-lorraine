use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use crate::types::error_types::NavigatorError;
use crate::types::temporal_types::TemporalCoordinate;
use crate::types::precision_types::PrecisionLevel;
use crate::config::system_config::SystemConfig;

/// Precision measurement engine
///
/// **In Memory of Mrs. Stella-Lorraine Masunda**
///
/// This engine provides ultra-precise temporal coordinate measurement
/// capabilities targeting 10^-50 second precision to honor her memory.
#[derive(Debug, Clone)]
pub struct MeasurementEngine {
    /// System configuration
    config: Arc<SystemConfig>,
    /// Measurement engine state
    state: Arc<RwLock<MeasurementEngineState>>,
}

/// Measurement engine state
#[derive(Debug, Clone)]
pub struct MeasurementEngineState {
    /// Current precision level
    pub current_precision: PrecisionLevel,
    /// Measurement active status
    pub measurement_active: bool,
    /// Last measurement timestamp
    pub last_measurement: Option<SystemTime>,
    /// Total measurements performed
    pub total_measurements: u64,
    /// Best precision achieved
    pub best_precision: f64,
}

/// Precision measurement result
#[derive(Debug, Clone)]
pub struct PrecisionMeasurementResult {
    /// Measured temporal coordinate
    pub coordinate: TemporalCoordinate,
    /// Achieved precision
    pub precision: f64,
    /// Measurement confidence
    pub confidence: f64,
    /// Measurement duration
    pub measurement_duration: Duration,
    /// Precision level
    pub precision_level: PrecisionLevel,
}

impl MeasurementEngine {
    /// Create new measurement engine
    pub fn new(config: Arc<SystemConfig>) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(MeasurementEngineState {
                current_precision: PrecisionLevel::Standard,
                measurement_active: false,
                last_measurement: None,
                total_measurements: 0,
                best_precision: f64::INFINITY,
            })),
        }
    }

    /// Perform precision measurement
    pub async fn measure_precision(&self, coordinate: &TemporalCoordinate) -> Result<PrecisionMeasurementResult, NavigatorError> {
        let start_time = SystemTime::now();

        // Simulate precision measurement
        let precision = coordinate.precision_seconds();
        let confidence = coordinate.confidence;

        // Update state
        let mut state = self.state.write().await;
        state.total_measurements += 1;
        state.last_measurement = Some(start_time);
        if precision < state.best_precision {
            state.best_precision = precision;
        }

        let measurement_duration = start_time.elapsed().unwrap_or(Duration::from_millis(0));

        Ok(PrecisionMeasurementResult {
            coordinate: coordinate.clone(),
            precision,
            confidence,
            measurement_duration,
            precision_level: PrecisionLevel::from_precision(precision),
        })
    }

    /// Get measurement statistics
    pub async fn get_statistics(&self) -> MeasurementEngineState {
        self.state.read().await.clone()
    }
}

impl PrecisionLevel {
    /// Convert precision to precision level
    pub fn from_precision(precision: f64) -> Self {
        if precision <= 1e-50 {
            PrecisionLevel::Ultimate
        } else if precision <= 1e-30 {
            PrecisionLevel::Target
        } else if precision <= 1e-20 {
            PrecisionLevel::Ultra
        } else if precision <= 1e-15 {
            PrecisionLevel::High
        } else {
            PrecisionLevel::Standard
        }
    }
}
