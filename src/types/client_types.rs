use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::types::temporal_types::TemporalCoordinate;
use crate::types::oscillation_types::{OscillationEndpoint, OscillationLevel};

/// Client configuration for external system connections
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientConfig {
    /// System endpoint URL
    pub endpoint: String,
    /// API version
    pub api_version: String,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Connection timeout
    pub timeout: Duration,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
}

/// Authentication configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// Authentication credentials
    pub credentials: HashMap<String, String>,
    /// Token refresh interval
    pub refresh_interval: Option<Duration>,
}

/// Authentication types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthType {
    /// No authentication
    None,
    /// API key authentication
    ApiKey,
    /// OAuth2 authentication
    OAuth2,
    /// JWT token authentication
    JWT,
    /// Custom authentication
    Custom(String),
}

/// Retry configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: usize,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Retry strategy
    pub strategy: RetryStrategy,
}

/// Retry strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetryStrategy {
    /// Fixed delay
    Fixed,
    /// Exponential backoff
    ExponentialBackoff,
    /// Linear backoff
    LinearBackoff,
    /// Jittered backoff
    JitteredBackoff,
}

/// Rate limiting configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per second
    pub max_requests_per_second: f64,
    /// Burst capacity
    pub burst_capacity: usize,
    /// Rate limit strategy
    pub strategy: RateLimitStrategy,
}

/// Rate limiting strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RateLimitStrategy {
    /// Token bucket
    TokenBucket,
    /// Leaky bucket
    LeakyBucket,
    /// Fixed window
    FixedWindow,
    /// Sliding window
    SlidingWindow,
}

/// Generic client request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientRequest {
    /// Request ID
    pub request_id: String,
    /// Request type
    pub request_type: RequestType,
    /// Request parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Request timestamp
    pub timestamp: SystemTime,
    /// Request priority
    pub priority: RequestPriority,
}

/// Request types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RequestType {
    /// Get oscillation data
    GetOscillationData,
    /// Get system status
    GetSystemStatus,
    /// Perform analysis
    PerformAnalysis,
    /// Get configuration
    GetConfiguration,
    /// Set configuration
    SetConfiguration,
    /// Health check
    HealthCheck,
    /// Custom request
    Custom(String),
}

/// Request priority levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RequestPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Generic client response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientResponse {
    /// Request ID this response corresponds to
    pub request_id: String,
    /// Response status
    pub status: ResponseStatus,
    /// Response data
    pub data: Option<serde_json::Value>,
    /// Response timestamp
    pub timestamp: SystemTime,
    /// Response metadata
    pub metadata: HashMap<String, String>,
}

/// Response status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseStatus {
    /// Success
    Success,
    /// Error
    Error(String),
    /// Partial success
    PartialSuccess(String),
    /// Timeout
    Timeout,
    /// Rate limited
    RateLimited,
}

/// Kambuzuma client specific types
pub mod kambuzuma {
    use super::*;

    /// Kambuzuma oscillation data request
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct OscillationDataRequest {
        /// Quantum coherence level
        pub coherence_level: f64,
        /// Measurement duration
        pub duration: Duration,
        /// Precision requirement
        pub precision_requirement: f64,
        /// Quantum state filters
        pub state_filters: Vec<QuantumStateFilter>,
    }

    /// Quantum state filter
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct QuantumStateFilter {
        /// Filter type
        pub filter_type: String,
        /// Filter parameters
        pub parameters: Vec<f64>,
    }

    /// Kambuzuma oscillation data response
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct OscillationDataResponse {
        /// Quantum oscillation endpoints
        pub quantum_endpoints: Vec<OscillationEndpoint>,
        /// Coherence measurements
        pub coherence_measurements: Vec<CoherenceMeasurement>,
        /// Quantum state data
        pub quantum_state_data: QuantumStateData,
    }

    /// Coherence measurement
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CoherenceMeasurement {
        /// Measurement time
        pub time: f64,
        /// Coherence value
        pub coherence: f64,
        /// Decoherence rate
        pub decoherence_rate: f64,
    }

    /// Quantum state data
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct QuantumStateData {
        /// State vector
        pub state_vector: Vec<f64>,
        /// Probability amplitudes
        pub probability_amplitudes: Vec<f64>,
        /// Entanglement measures
        pub entanglement_measures: Vec<f64>,
    }
}

/// Kwasa-kwasa client specific types
pub mod kwasa_kwasa {
    use super::*;

    /// Kwasa-kwasa semantic analysis request
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SemanticAnalysisRequest {
        /// Pattern recognition parameters
        pub pattern_params: PatternRecognitionParams,
        /// Catalysis requirements
        pub catalysis_requirements: CatalysisRequirements,
        /// Reconstruction targets
        pub reconstruction_targets: Vec<ReconstructionTarget>,
    }

    /// Pattern recognition parameters
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PatternRecognitionParams {
        /// Pattern complexity threshold
        pub complexity_threshold: f64,
        /// Pattern matching tolerance
        pub matching_tolerance: f64,
        /// Maximum pattern depth
        pub max_depth: usize,
    }

    /// Catalysis requirements
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CatalysisRequirements {
        /// Minimum catalysis efficiency
        pub min_efficiency: f64,
        /// Maximum catalysis time
        pub max_time: Duration,
        /// Catalysis quality threshold
        pub quality_threshold: f64,
    }

    /// Reconstruction target
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ReconstructionTarget {
        /// Target type
        pub target_type: String,
        /// Target parameters
        pub parameters: HashMap<String, f64>,
        /// Reconstruction fidelity requirement
        pub fidelity_requirement: f64,
    }

    /// Kwasa-kwasa semantic analysis response
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SemanticAnalysisResponse {
        /// Pattern match results
        pub pattern_matches: Vec<PatternMatch>,
        /// Catalysis results
        pub catalysis_results: Vec<CatalysisResult>,
        /// Reconstruction results
        pub reconstruction_results: Vec<ReconstructionResult>,
    }

    /// Pattern match result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PatternMatch {
        /// Pattern ID
        pub pattern_id: String,
        /// Match confidence
        pub confidence: f64,
        /// Pattern complexity
        pub complexity: f64,
        /// Match data
        pub match_data: Vec<f64>,
    }

    /// Catalysis result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CatalysisResult {
        /// Catalysis efficiency
        pub efficiency: f64,
        /// Catalysis time
        pub time: Duration,
        /// Catalysis quality
        pub quality: f64,
        /// Catalysis data
        pub data: Vec<f64>,
    }

    /// Reconstruction result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ReconstructionResult {
        /// Reconstruction target
        pub target: ReconstructionTarget,
        /// Reconstruction fidelity
        pub fidelity: f64,
        /// Reconstruction data
        pub data: Vec<f64>,
    }
}

/// Mzekezeke client specific types
pub mod mzekezeke {
    use super::*;

    /// Mzekezeke authentication request
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AuthenticationRequest {
        /// 12-dimensional authentication data
        pub auth_data: [f64; 12],
        /// Authentication challenge
        pub challenge: String,
        /// Security level requirement
        pub security_level: SecurityLevel,
    }

    /// Security levels
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum SecurityLevel {
        /// Standard security
        Standard,
        /// High security
        High,
        /// Ultra security
        Ultra,
        /// Thermodynamic security (10^44 Joules)
        Thermodynamic,
    }

    /// Mzekezeke authentication response
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AuthenticationResponse {
        /// Authentication result
        pub result: AuthenticationResult,
        /// Validated dimensions
        pub validated_dimensions: Vec<usize>,
        /// Security level achieved
        pub security_level: SecurityLevel,
        /// Thermodynamic proof
        pub thermodynamic_proof: Option<ThermodynamicProof>,
    }

    /// Authentication result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum AuthenticationResult {
        /// Authentication successful
        Success,
        /// Authentication failed
        Failed(String),
        /// Partial authentication
        Partial(Vec<usize>),
    }

    /// Thermodynamic proof
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ThermodynamicProof {
        /// Energy requirement (Joules)
        pub energy_requirement: f64,
        /// Proof validity
        pub proof_validity: bool,
        /// Proof data
        pub proof_data: Vec<u8>,
    }
}

/// Buhera client specific types
pub mod buhera {
    use super::*;

    /// Buhera environmental data request
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EnvironmentalDataRequest {
        /// Environmental parameters
        pub parameters: Vec<EnvironmentalParameter>,
        /// Measurement duration
        pub duration: Duration,
        /// Coupling analysis requirements
        pub coupling_requirements: CouplingRequirements,
    }

    /// Environmental parameter
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EnvironmentalParameter {
        /// Parameter type
        pub parameter_type: EnvironmentalParameterType,
        /// Measurement precision
        pub precision: f64,
        /// Sampling rate
        pub sampling_rate: f64,
    }

    /// Environmental parameter types
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum EnvironmentalParameterType {
        /// Temperature
        Temperature,
        /// Pressure
        Pressure,
        /// Humidity
        Humidity,
        /// Magnetic field
        MagneticField,
        /// Atmospheric oscillations
        AtmosphericOscillations,
        /// Gravitational variations
        GravitationalVariations,
    }

    /// Coupling analysis requirements
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CouplingRequirements {
        /// Minimum coupling strength
        pub min_coupling_strength: f64,
        /// Correlation threshold
        pub correlation_threshold: f64,
        /// Analysis depth
        pub analysis_depth: usize,
    }

    /// Buhera environmental data response
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EnvironmentalDataResponse {
        /// Environmental measurements
        pub measurements: Vec<EnvironmentalMeasurement>,
        /// Coupling analysis results
        pub coupling_results: Vec<CouplingResult>,
        /// Environmental oscillation data
        pub oscillation_data: Vec<OscillationEndpoint>,
    }

    /// Environmental measurement
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EnvironmentalMeasurement {
        /// Parameter type
        pub parameter_type: EnvironmentalParameterType,
        /// Measurement value
        pub value: f64,
        /// Measurement uncertainty
        pub uncertainty: f64,
        /// Measurement time
        pub time: SystemTime,
    }

    /// Coupling result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct CouplingResult {
        /// Coupling strength
        pub strength: f64,
        /// Correlation coefficient
        pub correlation: f64,
        /// Coupling type
        pub coupling_type: String,
        /// Coupling data
        pub data: Vec<f64>,
    }
}

/// Consciousness client specific types
pub mod consciousness {
    use super::*;

    /// Consciousness enhancement request
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ConsciousnessEnhancementRequest {
        /// Enhancement parameters
        pub enhancement_params: EnhancementParams,
        /// Fire-adaptation requirements
        pub fire_adaptation: FireAdaptationRequirements,
        /// Prediction targets
        pub prediction_targets: Vec<PredictionTarget>,
    }

    /// Enhancement parameters
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EnhancementParams {
        /// Enhancement level
        pub level: EnhancementLevel,
        /// Alpha wave synchronization
        pub alpha_sync: bool,
        /// Neural network coupling
        pub neural_coupling: bool,
        /// Consciousness expansion factor
        pub expansion_factor: f64,
    }

    /// Enhancement levels
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum EnhancementLevel {
        /// Basic enhancement
        Basic,
        /// Advanced enhancement
        Advanced,
        /// Fire-adapted enhancement
        FireAdapted,
        /// Consciousness-targeting enhancement
        ConsciousnessTargeting,
    }

    /// Fire adaptation requirements
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct FireAdaptationRequirements {
        /// Fire-wavelength processing
        pub wavelength_processing: bool,
        /// Acoustic environment adaptation
        pub acoustic_adaptation: bool,
        /// Audio image generation
        pub audio_image_generation: bool,
        /// Consciousness targeting
        pub consciousness_targeting: bool,
    }

    /// Prediction target
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PredictionTarget {
        /// Target type
        pub target_type: String,
        /// Prediction horizon
        pub prediction_horizon: Duration,
        /// Accuracy requirement
        pub accuracy_requirement: f64,
    }

    /// Consciousness enhancement response
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ConsciousnessEnhancementResponse {
        /// Enhancement results
        pub enhancement_results: Vec<EnhancementResult>,
        /// Fire adaptation results
        pub fire_adaptation_results: FireAdaptationResults,
        /// Prediction results
        pub prediction_results: Vec<PredictionResult>,
    }

    /// Enhancement result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EnhancementResult {
        /// Enhancement factor achieved
        pub factor: f64,
        /// Enhancement quality
        pub quality: f64,
        /// Enhancement data
        pub data: Vec<f64>,
    }

    /// Fire adaptation results
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct FireAdaptationResults {
        /// Wavelength processing results
        pub wavelength_results: Vec<f64>,
        /// Acoustic adaptation results
        pub acoustic_results: Vec<f64>,
        /// Audio image quality
        pub audio_image_quality: f64,
        /// Consciousness targeting accuracy
        pub targeting_accuracy: f64,
    }

    /// Prediction result
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PredictionResult {
        /// Prediction target
        pub target: PredictionTarget,
        /// Prediction value
        pub value: f64,
        /// Prediction confidence
        pub confidence: f64,
        /// Prediction data
        pub data: Vec<f64>,
    }
}

/// Client status information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientStatus {
    /// Client name
    pub name: String,
    /// Connection status
    pub connection_status: ConnectionStatus,
    /// Last successful request time
    pub last_success: Option<SystemTime>,
    /// Last error
    pub last_error: Option<String>,
    /// Request statistics
    pub request_stats: RequestStats,
    /// Health metrics
    pub health_metrics: HashMap<String, f64>,
}

/// Connection status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    /// Connected
    Connected,
    /// Disconnected
    Disconnected,
    /// Connecting
    Connecting,
    /// Error
    Error(String),
}

/// Request statistics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestStats {
    /// Total requests sent
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Average response time
    pub avg_response_time: Duration,
    /// Request rate (requests per second)
    pub request_rate: f64,
}

impl ClientConfig {
    /// Creates a new client configuration
    pub fn new(endpoint: String, api_version: String) -> Self {
        Self {
            endpoint,
            api_version,
            auth: AuthConfig::default(),
            timeout: Duration::from_secs(30),
            retry: RetryConfig::default(),
            rate_limit: RateLimitConfig::default(),
        }
    }

    /// Creates configuration for Kambuzuma client
    pub fn kambuzuma(endpoint: String) -> Self {
        Self::new(endpoint, "v1".to_string())
    }

    /// Creates configuration for Kwasa-kwasa client
    pub fn kwasa_kwasa(endpoint: String) -> Self {
        Self::new(endpoint, "v1".to_string())
    }

    /// Creates configuration for Mzekezeke client
    pub fn mzekezeke(endpoint: String) -> Self {
        Self::new(endpoint, "v1".to_string())
    }

    /// Creates configuration for Buhera client
    pub fn buhera(endpoint: String) -> Self {
        Self::new(endpoint, "v1".to_string())
    }

    /// Creates configuration for Consciousness client
    pub fn consciousness(endpoint: String) -> Self {
        Self::new(endpoint, "v1".to_string())
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            auth_type: AuthType::None,
            credentials: HashMap::new(),
            refresh_interval: None,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            strategy: RetryStrategy::ExponentialBackoff,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests_per_second: 10.0,
            burst_capacity: 100,
            strategy: RateLimitStrategy::TokenBucket,
        }
    }
}

impl RequestStats {
    /// Creates new request statistics
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time: Duration::from_millis(0),
            request_rate: 0.0,
        }
    }

    /// Gets the success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    /// Gets the failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.failed_requests as f64 / self.total_requests as f64
        }
    }
}

impl Default for RequestStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_creation() {
        let config = ClientConfig::new("http://localhost:8080".to_string(), "v1".to_string());
        assert_eq!(config.endpoint, "http://localhost:8080");
        assert_eq!(config.api_version, "v1");
        assert_eq!(config.auth.auth_type, AuthType::None);
    }

    #[test]
    fn test_kambuzuma_config() {
        let config = ClientConfig::kambuzuma("http://kambuzuma:8080".to_string());
        assert_eq!(config.endpoint, "http://kambuzuma:8080");
        assert_eq!(config.api_version, "v1");
    }

    #[test]
    fn test_request_stats() {
        let mut stats = RequestStats::new();
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.failure_rate(), 0.0);
        
        stats.total_requests = 10;
        stats.successful_requests = 8;
        stats.failed_requests = 2;
        
        assert_eq!(stats.success_rate(), 0.8);
        assert_eq!(stats.failure_rate(), 0.2);
    }

    #[test]
    fn test_response_status() {
        let success = ResponseStatus::Success;
        let error = ResponseStatus::Error("Test error".to_string());
        
        assert_eq!(success, ResponseStatus::Success);
        match error {
            ResponseStatus::Error(msg) => assert_eq!(msg, "Test error"),
            _ => panic!("Expected error response"),
        }
    }
} 