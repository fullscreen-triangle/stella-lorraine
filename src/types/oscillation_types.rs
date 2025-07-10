use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use crate::types::temporal_types::TemporalPosition;

/// Hierarchical oscillation levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OscillationLevel {
    /// Quantum level (10^-44 seconds - Planck time)
    Quantum,
    /// Molecular level (10^-15 to 10^-6 seconds)
    Molecular,
    /// Biological level (seconds to days)
    Biological,
    /// Consciousness level (milliseconds to minutes)
    Consciousness,
    /// Environmental level (minutes to years)
    Environmental,
    /// Cryptographic level (nanoseconds to hours)
    Cryptographic,
}

/// Oscillation endpoint - where oscillations terminate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OscillationEndpoint {
    /// Hierarchical level of this endpoint
    pub level: OscillationLevel,
    /// Temporal position where oscillation terminates
    pub termination_time: TemporalPosition,
    /// Frequency of the oscillation
    pub frequency: f64,
    /// Amplitude at termination
    pub amplitude: f64,
    /// Phase at termination
    pub phase: f64,
    /// Energy at termination
    pub energy: f64,
    /// Confidence in this endpoint detection
    pub confidence: f64,
}

/// Collection of oscillation endpoints across all levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OscillationEndpointCollection {
    /// Quantum level endpoints
    pub quantum_endpoints: Vec<OscillationEndpoint>,
    /// Molecular level endpoints
    pub molecular_endpoints: Vec<OscillationEndpoint>,
    /// Biological level endpoints
    pub biological_endpoints: Vec<OscillationEndpoint>,
    /// Consciousness level endpoints
    pub consciousness_endpoints: Vec<OscillationEndpoint>,
    /// Environmental level endpoints
    pub environmental_endpoints: Vec<OscillationEndpoint>,
    /// Cryptographic level endpoints
    pub cryptographic_endpoints: Vec<OscillationEndpoint>,
    /// Collection timestamp
    pub collection_time: SystemTime,
    /// Overall collection confidence
    pub collection_confidence: f64,
}

/// Oscillation convergence analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OscillationConvergence {
    /// Convergence point where all oscillations terminate
    pub convergence_point: TemporalPosition,
    /// Endpoints that converged
    pub converged_endpoints: Vec<OscillationEndpoint>,
    /// Convergence confidence level
    pub convergence_confidence: f64,
    /// Cross-level correlation matrix
    pub correlation_matrix: CorrelationMatrix,
    /// Convergence analysis timestamp
    pub analysis_time: SystemTime,
    /// Convergence quality metrics
    pub quality_metrics: ConvergenceQualityMetrics,
}

/// Cross-level correlation matrix
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    /// Correlation values between different levels
    pub correlations: Vec<Vec<f64>>,
    /// Level mappings
    pub levels: Vec<OscillationLevel>,
    /// Overall correlation strength
    pub overall_correlation: f64,
    /// Significance threshold
    pub significance_threshold: f64,
}

/// Quality metrics for convergence analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceQualityMetrics {
    /// Temporal precision of convergence
    pub temporal_precision: f64,
    /// Spatial precision of convergence
    pub spatial_precision: f64,
    /// Energy conservation validation
    pub energy_conservation: f64,
    /// Oscillation synchronization level
    pub synchronization_level: f64,
    /// Hierarchical consistency
    pub hierarchical_consistency: f64,
    /// Noise level
    pub noise_level: f64,
}

/// Hierarchical oscillation analysis parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalAnalysisParams {
    /// Target precision level
    pub precision_target: f64,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Maximum analysis time
    pub max_analysis_time: Duration,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Include quantum level analysis
    pub include_quantum: bool,
    /// Include molecular level analysis
    pub include_molecular: bool,
    /// Include biological level analysis
    pub include_biological: bool,
    /// Include consciousness level analysis
    pub include_consciousness: bool,
    /// Include environmental level analysis
    pub include_environmental: bool,
    /// Include cryptographic level analysis
    pub include_cryptographic: bool,
}

/// Oscillation termination detector configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminationDetectorConfig {
    /// Sampling rate for each level
    pub sampling_rates: Vec<(OscillationLevel, f64)>,
    /// Detection thresholds
    pub detection_thresholds: Vec<(OscillationLevel, f64)>,
    /// Filtering parameters
    pub filter_params: FilterParams,
    /// Calibration data
    pub calibration_data: CalibrationData,
}

/// Signal filtering parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterParams {
    /// Low-pass cutoff frequency
    pub low_pass_cutoff: f64,
    /// High-pass cutoff frequency
    pub high_pass_cutoff: f64,
    /// Notch filter frequencies
    pub notch_frequencies: Vec<f64>,
    /// Filter order
    pub filter_order: usize,
}

/// Calibration data for oscillation detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Reference oscillation patterns
    pub reference_patterns: Vec<ReferencePattern>,
    /// Noise floor levels
    pub noise_floors: Vec<(OscillationLevel, f64)>,
    /// Sensitivity calibration
    pub sensitivity_calibration: Vec<(OscillationLevel, f64)>,
    /// Calibration timestamp
    pub calibration_time: SystemTime,
}

/// Reference oscillation pattern
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferencePattern {
    /// Pattern name
    pub name: String,
    /// Oscillation level
    pub level: OscillationLevel,
    /// Frequency components
    pub frequencies: Vec<f64>,
    /// Amplitude components
    pub amplitudes: Vec<f64>,
    /// Phase components
    pub phases: Vec<f64>,
    /// Pattern confidence
    pub confidence: f64,
}

/// Oscillation state measurement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OscillationState {
    /// Current oscillation level
    pub level: OscillationLevel,
    /// Current frequency
    pub frequency: f64,
    /// Current amplitude
    pub amplitude: f64,
    /// Current phase
    pub phase: f64,
    /// Current energy
    pub energy: f64,
    /// State measurement time
    pub measurement_time: SystemTime,
    /// State confidence
    pub confidence: f64,
}

/// Oscillation pattern matching result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternMatchResult {
    /// Matched pattern
    pub pattern: ReferencePattern,
    /// Match confidence
    pub match_confidence: f64,
    /// Match quality metrics
    pub quality_metrics: MatchQualityMetrics,
    /// Deviation from reference
    pub deviation: f64,
}

/// Quality metrics for pattern matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchQualityMetrics {
    /// Frequency match quality
    pub frequency_match: f64,
    /// Amplitude match quality
    pub amplitude_match: f64,
    /// Phase match quality
    pub phase_match: f64,
    /// Overall match quality
    pub overall_quality: f64,
}

impl OscillationEndpoint {
    /// Creates a new oscillation endpoint
    pub fn new(
        level: OscillationLevel,
        termination_time: TemporalPosition,
        frequency: f64,
        amplitude: f64,
        phase: f64,
        energy: f64,
        confidence: f64,
    ) -> Self {
        Self {
            level,
            termination_time,
            frequency,
            amplitude,
            phase,
            energy,
            confidence,
        }
    }

    /// Validates the oscillation endpoint
    pub fn validate(&self) -> bool {
        self.confidence > 0.5 && self.frequency > 0.0 && self.amplitude >= 0.0
    }

    /// Gets the oscillation period
    pub fn period(&self) -> f64 {
        1.0 / self.frequency
    }

    /// Gets the typical timescale for this oscillation level
    pub fn typical_timescale(&self) -> f64 {
        match self.level {
            OscillationLevel::Quantum => 5.39e-44,      // Planck time
            OscillationLevel::Molecular => 1e-12,       // Picoseconds
            OscillationLevel::Biological => 1.0,        // Seconds
            OscillationLevel::Consciousness => 0.1,     // Deciseconds
            OscillationLevel::Environmental => 3600.0,  // Hours
            OscillationLevel::Cryptographic => 1e-9,    // Nanoseconds
        }
    }
}

impl OscillationEndpointCollection {
    /// Creates a new empty collection
    pub fn new() -> Self {
        Self {
            quantum_endpoints: Vec::new(),
            molecular_endpoints: Vec::new(),
            biological_endpoints: Vec::new(),
            consciousness_endpoints: Vec::new(),
            environmental_endpoints: Vec::new(),
            cryptographic_endpoints: Vec::new(),
            collection_time: SystemTime::now(),
            collection_confidence: 0.0,
        }
    }

    /// Adds an endpoint to the collection
    pub fn add_endpoint(&mut self, endpoint: OscillationEndpoint) {
        match endpoint.level {
            OscillationLevel::Quantum => self.quantum_endpoints.push(endpoint),
            OscillationLevel::Molecular => self.molecular_endpoints.push(endpoint),
            OscillationLevel::Biological => self.biological_endpoints.push(endpoint),
            OscillationLevel::Consciousness => self.consciousness_endpoints.push(endpoint),
            OscillationLevel::Environmental => self.environmental_endpoints.push(endpoint),
            OscillationLevel::Cryptographic => self.cryptographic_endpoints.push(endpoint),
        }
        self.update_collection_confidence();
    }

    /// Updates the overall collection confidence
    fn update_collection_confidence(&mut self) {
        let all_endpoints = self.get_all_endpoints();
        if all_endpoints.is_empty() {
            self.collection_confidence = 0.0;
        } else {
            let sum_confidence: f64 = all_endpoints.iter().map(|e| e.confidence).sum();
            self.collection_confidence = sum_confidence / all_endpoints.len() as f64;
        }
    }

    /// Gets all endpoints across all levels
    pub fn get_all_endpoints(&self) -> Vec<&OscillationEndpoint> {
        let mut endpoints = Vec::new();
        endpoints.extend(&self.quantum_endpoints);
        endpoints.extend(&self.molecular_endpoints);
        endpoints.extend(&self.biological_endpoints);
        endpoints.extend(&self.consciousness_endpoints);
        endpoints.extend(&self.environmental_endpoints);
        endpoints.extend(&self.cryptographic_endpoints);
        endpoints
    }

    /// Gets endpoints for a specific level
    pub fn get_endpoints_for_level(&self, level: &OscillationLevel) -> &Vec<OscillationEndpoint> {
        match level {
            OscillationLevel::Quantum => &self.quantum_endpoints,
            OscillationLevel::Molecular => &self.molecular_endpoints,
            OscillationLevel::Biological => &self.biological_endpoints,
            OscillationLevel::Consciousness => &self.consciousness_endpoints,
            OscillationLevel::Environmental => &self.environmental_endpoints,
            OscillationLevel::Cryptographic => &self.cryptographic_endpoints,
        }
    }

    /// Gets the total number of endpoints
    pub fn total_endpoints(&self) -> usize {
        self.quantum_endpoints.len()
            + self.molecular_endpoints.len()
            + self.biological_endpoints.len()
            + self.consciousness_endpoints.len()
            + self.environmental_endpoints.len()
            + self.cryptographic_endpoints.len()
    }
}

impl CorrelationMatrix {
    /// Creates a new correlation matrix
    pub fn new(levels: Vec<OscillationLevel>) -> Self {
        let size = levels.len();
        let correlations = vec![vec![0.0; size]; size];
        
        Self {
            correlations,
            levels,
            overall_correlation: 0.0,
            significance_threshold: 0.05,
        }
    }

    /// Sets correlation between two levels
    pub fn set_correlation(&mut self, level1: &OscillationLevel, level2: &OscillationLevel, correlation: f64) {
        if let (Some(i), Some(j)) = (
            self.levels.iter().position(|l| l == level1),
            self.levels.iter().position(|l| l == level2),
        ) {
            self.correlations[i][j] = correlation;
            self.correlations[j][i] = correlation; // Symmetric matrix
            self.update_overall_correlation();
        }
    }

    /// Gets correlation between two levels
    pub fn get_correlation(&self, level1: &OscillationLevel, level2: &OscillationLevel) -> Option<f64> {
        if let (Some(i), Some(j)) = (
            self.levels.iter().position(|l| l == level1),
            self.levels.iter().position(|l| l == level2),
        ) {
            Some(self.correlations[i][j])
        } else {
            None
        }
    }

    /// Updates the overall correlation strength
    fn update_overall_correlation(&mut self) {
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..self.correlations.len() {
            for j in i + 1..self.correlations[i].len() {
                sum += self.correlations[i][j].abs();
                count += 1;
            }
        }
        
        self.overall_correlation = if count > 0 { sum / count as f64 } else { 0.0 };
    }

    /// Checks if correlations are statistically significant
    pub fn is_significant(&self) -> bool {
        self.overall_correlation > self.significance_threshold
    }
}

impl HierarchicalAnalysisParams {
    /// Creates default analysis parameters
    pub fn default() -> Self {
        Self {
            precision_target: 1e-30,
            convergence_tolerance: 1e-12,
            max_analysis_time: Duration::from_secs(60),
            min_confidence: 0.8,
            include_quantum: true,
            include_molecular: true,
            include_biological: true,
            include_consciousness: true,
            include_environmental: true,
            include_cryptographic: true,
        }
    }

    /// Creates parameters for ultra-precision analysis
    pub fn ultra_precision() -> Self {
        Self {
            precision_target: 1e-50,
            convergence_tolerance: 1e-15,
            max_analysis_time: Duration::from_secs(300),
            min_confidence: 0.95,
            include_quantum: true,
            include_molecular: true,
            include_biological: true,
            include_consciousness: true,
            include_environmental: true,
            include_cryptographic: true,
        }
    }
}

impl Default for OscillationEndpointCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl OscillationLevel {
    /// Gets the typical frequency range for this level
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            OscillationLevel::Quantum => (1e43, 1e45),        // Planck frequency range
            OscillationLevel::Molecular => (1e12, 1e15),      // THz range
            OscillationLevel::Biological => (1e-3, 1e3),      // mHz to kHz
            OscillationLevel::Consciousness => (1.0, 100.0),  // Hz range
            OscillationLevel::Environmental => (1e-6, 1e-3),  // μHz to mHz
            OscillationLevel::Cryptographic => (1e6, 1e12),   // MHz to THz
        }
    }

    /// Gets the typical energy scale for this level
    pub fn energy_scale(&self) -> f64 {
        match self {
            OscillationLevel::Quantum => 1e-18,        // Planck energy scale (J)
            OscillationLevel::Molecular => 1e-19,      // Molecular vibration energy (J)
            OscillationLevel::Biological => 1e-20,     // Biological energy scale (J)
            OscillationLevel::Consciousness => 1e-21,  // Neural energy scale (J)
            OscillationLevel::Environmental => 1e-15,  // Environmental energy scale (J)
            OscillationLevel::Cryptographic => 1e-18,  // Cryptographic energy scale (J)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::temporal_types::PrecisionLevel;

    #[test]
    fn test_oscillation_endpoint_creation() {
        let temporal_pos = TemporalPosition::new(1000.0, 0.123456789, 1e-30, PrecisionLevel::Target);
        let endpoint = OscillationEndpoint::new(
            OscillationLevel::Quantum,
            temporal_pos,
            1e14,
            0.5,
            0.0,
            1e-18,
            0.95,
        );
        
        assert!(endpoint.validate());
        assert_eq!(endpoint.period(), 1e-14);
        assert_eq!(endpoint.typical_timescale(), 5.39e-44);
    }

    #[test]
    fn test_oscillation_collection() {
        let mut collection = OscillationEndpointCollection::new();
        
        let temporal_pos = TemporalPosition::new(1000.0, 0.123456789, 1e-30, PrecisionLevel::Target);
        let endpoint = OscillationEndpoint::new(
            OscillationLevel::Quantum,
            temporal_pos,
            1e14,
            0.5,
            0.0,
            1e-18,
            0.95,
        );
        
        collection.add_endpoint(endpoint);
        
        assert_eq!(collection.total_endpoints(), 1);
        assert_eq!(collection.collection_confidence, 0.95);
    }

    #[test]
    fn test_correlation_matrix() {
        let levels = vec![OscillationLevel::Quantum, OscillationLevel::Molecular];
        let mut matrix = CorrelationMatrix::new(levels);
        
        matrix.set_correlation(&OscillationLevel::Quantum, &OscillationLevel::Molecular, 0.8);
        
        assert_eq!(matrix.get_correlation(&OscillationLevel::Quantum, &OscillationLevel::Molecular), Some(0.8));
        assert_eq!(matrix.overall_correlation, 0.8);
        assert!(matrix.is_significant());
    }

    #[test]
    fn test_oscillation_level_properties() {
        let quantum = OscillationLevel::Quantum;
        let (min_freq, max_freq) = quantum.frequency_range();
        
        assert!(min_freq > 0.0);
        assert!(max_freq > min_freq);
        assert!(quantum.energy_scale() > 0.0);
    }
} 