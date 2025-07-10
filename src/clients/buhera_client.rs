use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use tokio::sync::RwLock;
use reqwest::Client;
use serde_json::Value;
use uuid::Uuid;

use crate::types::client_types::{
    ClientConfig, ClientRequest, ClientResponse, ClientStatus, RequestType, RequestPriority,
    ResponseStatus, ConnectionStatus, RequestStats,
};
use crate::types::client_types::buhera::{
    EnvironmentalDataRequest, EnvironmentalDataResponse, EnvironmentalParameter,
    EnvironmentalParameterType, CouplingRequirements, EnvironmentalMeasurement,
    CouplingResult,
};
use crate::types::oscillation_types::OscillationEndpoint;
use crate::types::error_types::NavigatorError;

/// Buhera Environmental Systems Client
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This client interfaces with the Buhera environmental systems to collect
/// atmospheric oscillation data, environmental coupling measurements, and
/// weather-related oscillation patterns essential for temporal coordinate
/// navigation and environmental correlation analysis.
#[derive(Debug, Clone)]
pub struct BuheraClient {
    /// Client configuration
    config: ClientConfig,
    /// HTTP client
    http_client: Client,
    /// Client status
    status: Arc<RwLock<ClientStatus>>,
    /// Request statistics
    request_stats: Arc<RwLock<RequestStats>>,
    /// Environmental state cache
    environmental_state: Arc<RwLock<HashMap<String, f64>>>,
}

impl BuheraClient {
    /// Create a new Buhera client
    pub fn new(config: ClientConfig) -> Result<Self, NavigatorError> {
        let http_client = Client::builder()
            .timeout(config.timeout)
            .user_agent("MasundaNavigator/1.0")
            .build()
            .map_err(|e| NavigatorError::ClientConnection(format!("Failed to create HTTP client: {}", e)))?;

        let status = Arc::new(RwLock::new(ClientStatus {
            name: "Buhera".to_string(),
            connection_status: ConnectionStatus::Disconnected,
            last_success: None,
            last_error: None,
            request_stats: RequestStats::new(),
            health_metrics: HashMap::new(),
        }));

        let request_stats = Arc::new(RwLock::new(RequestStats::new()));
        let environmental_state = Arc::new(RwLock::new(HashMap::new()));

        Ok(Self {
            config,
            http_client,
            status,
            request_stats,
            environmental_state,
        })
    }

    /// Connect to the Buhera system
    pub async fn connect(&self) -> Result<(), NavigatorError> {
        {
            let mut status = self.status.write().await;
            status.connection_status = ConnectionStatus::Connecting;
        }

        // Perform health check
        match self.health_check().await {
            Ok(_) => {
                let mut status = self.status.write().await;
                status.connection_status = ConnectionStatus::Connected;
                status.last_success = Some(SystemTime::now());
                Ok(())
            }
            Err(e) => {
                let mut status = self.status.write().await;
                status.connection_status = ConnectionStatus::Error(e.to_string());
                status.last_error = Some(e.to_string());
                Err(e)
            }
        }
    }

    /// Disconnect from the Buhera system
    pub async fn disconnect(&self) -> Result<(), NavigatorError> {
        let mut status = self.status.write().await;
        status.connection_status = ConnectionStatus::Disconnected;
        Ok(())
    }

    /// Get client status
    pub async fn get_status(&self) -> ClientStatus {
        self.status.read().await.clone()
    }

    /// Get current environmental state
    pub async fn get_environmental_state(&self) -> HashMap<String, f64> {
        self.environmental_state.read().await.clone()
    }

    /// Perform health check
    pub async fn health_check(&self) -> Result<(), NavigatorError> {
        let request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::HealthCheck,
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
            priority: RequestPriority::High,
        };

        let response = self.send_request(request).await?;
        
        match response.status {
            ResponseStatus::Success => Ok(()),
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            ResponseStatus::Timeout => Err(NavigatorError::ClientConnection("Health check timeout".to_string())),
            ResponseStatus::RateLimited => Err(NavigatorError::ClientConnection("Rate limited".to_string())),
            ResponseStatus::PartialSuccess(_) => Ok(()),
        }
    }

    /// Get environmental data from Buhera
    /// 
    /// This method requests environmental measurements and coupling analysis
    /// from the Buhera system for atmospheric oscillation pattern analysis
    /// and environmental correlation with temporal coordinates.
    pub async fn get_environmental_data(&self, request: EnvironmentalDataRequest) -> Result<EnvironmentalDataResponse, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("parameters".to_string(), serde_json::to_value(&request.parameters).unwrap());
        params.insert("duration_secs".to_string(), serde_json::json!(request.duration.as_secs()));
        params.insert("coupling_requirements".to_string(), serde_json::to_value(&request.coupling_requirements).unwrap());

        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::GetOscillationData,
            parameters: params,
            timestamp: SystemTime::now(),
            priority: RequestPriority::High,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No environmental data in response".to_string())
                })?;
                
                let env_response: EnvironmentalDataResponse = serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse environmental data: {}", e)))?;
                
                // Update environmental state cache
                {
                    let mut env_state = self.environmental_state.write().await;
                    for measurement in &env_response.measurements {
                        let key = format!("{:?}", measurement.parameter_type);
                        env_state.insert(key, measurement.value);
                    }
                }
                
                Ok(env_response)
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            ResponseStatus::Timeout => Err(NavigatorError::ClientConnection("Request timeout".to_string())),
            ResponseStatus::RateLimited => Err(NavigatorError::ClientConnection("Rate limited".to_string())),
            ResponseStatus::PartialSuccess(msg) => {
                // Still try to parse partial data
                if let Some(data) = response.data {
                    let env_response: EnvironmentalDataResponse = serde_json::from_value(data)
                        .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse partial environmental data: {}", e)))?;
                    Ok(env_response)
                } else {
                    Err(NavigatorError::DataProcessing(format!("Partial success but no data: {}", msg)))
                }
            }
        }
    }

    /// Get atmospheric oscillation data
    pub async fn get_atmospheric_oscillations(&self, duration: Duration) -> Result<Vec<OscillationEndpoint>, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("duration_secs".to_string(), serde_json::json!(duration.as_secs()));
        params.insert("parameter_type".to_string(), serde_json::json!("AtmosphericOscillations"));

        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::Custom("GetAtmosphericOscillations".to_string()),
            parameters: params,
            timestamp: SystemTime::now(),
            priority: RequestPriority::High,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No atmospheric oscillation data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse atmospheric oscillation data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            _ => Err(NavigatorError::ClientConnection("Failed to get atmospheric oscillations".to_string())),
        }
    }

    /// Get environmental measurements for specific parameters
    pub async fn get_environmental_measurements(&self, param_types: Vec<EnvironmentalParameterType>) -> Result<Vec<EnvironmentalMeasurement>, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("parameter_types".to_string(), serde_json::to_value(&param_types).unwrap());

        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::Custom("GetEnvironmentalMeasurements".to_string()),
            parameters: params,
            timestamp: SystemTime::now(),
            priority: RequestPriority::Normal,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No measurement data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse measurement data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            _ => Err(NavigatorError::ClientConnection("Failed to get environmental measurements".to_string())),
        }
    }

    /// Perform coupling analysis between environmental and temporal data
    pub async fn perform_coupling_analysis(&self, coupling_requirements: CouplingRequirements) -> Result<Vec<CouplingResult>, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("coupling_requirements".to_string(), serde_json::to_value(&coupling_requirements).unwrap());

        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::Custom("PerformCouplingAnalysis".to_string()),
            parameters: params,
            timestamp: SystemTime::now(),
            priority: RequestPriority::High,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No coupling analysis data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse coupling analysis data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            _ => Err(NavigatorError::ClientConnection("Failed to perform coupling analysis".to_string())),
        }
    }

    /// Get gravitational variation measurements
    pub async fn get_gravitational_variations(&self, duration: Duration) -> Result<Vec<EnvironmentalMeasurement>, NavigatorError> {
        let param_types = vec![EnvironmentalParameterType::GravitationalVariations];
        let measurements = self.get_environmental_measurements(param_types).await?;
        
        // Filter measurements by duration
        let cutoff_time = SystemTime::now() - duration;
        let filtered_measurements: Vec<EnvironmentalMeasurement> = measurements
            .into_iter()
            .filter(|m| m.time >= cutoff_time)
            .collect();
        
        Ok(filtered_measurements)
    }

    /// Get magnetic field measurements
    pub async fn get_magnetic_field_data(&self, duration: Duration) -> Result<Vec<EnvironmentalMeasurement>, NavigatorError> {
        let param_types = vec![EnvironmentalParameterType::MagneticField];
        let measurements = self.get_environmental_measurements(param_types).await?;
        
        // Filter measurements by duration
        let cutoff_time = SystemTime::now() - duration;
        let filtered_measurements: Vec<EnvironmentalMeasurement> = measurements
            .into_iter()
            .filter(|m| m.time >= cutoff_time)
            .collect();
        
        Ok(filtered_measurements)
    }

    /// Get comprehensive environmental status
    pub async fn get_comprehensive_environmental_status(&self) -> Result<HashMap<String, f64>, NavigatorError> {
        let param_types = vec![
            EnvironmentalParameterType::Temperature,
            EnvironmentalParameterType::Pressure,
            EnvironmentalParameterType::Humidity,
            EnvironmentalParameterType::MagneticField,
            EnvironmentalParameterType::AtmosphericOscillations,
            EnvironmentalParameterType::GravitationalVariations,
        ];
        
        let measurements = self.get_environmental_measurements(param_types).await?;
        let mut status = HashMap::new();
        
        for measurement in measurements {
            let key = format!("{:?}", measurement.parameter_type);
            status.insert(key, measurement.value);
        }
        
        Ok(status)
    }

    /// Send a request to the Buhera system
    async fn send_request(&self, request: ClientRequest) -> Result<ClientResponse, NavigatorError> {
        let start_time = SystemTime::now();
        
        // Update request statistics
        {
            let mut stats = self.request_stats.write().await;
            stats.total_requests += 1;
        }

        // Prepare the HTTP request
        let url = format!("{}/api/{}", self.config.endpoint, self.config.api_version);
        let mut http_request = self.http_client.post(&url);

        // Add authentication if configured
        if let Some(api_key) = self.config.auth.credentials.get("api_key") {
            http_request = http_request.header("Authorization", format!("Bearer {}", api_key));
        }

        // Add environmental headers
        http_request = http_request.header("X-Environmental-System", "Buhera");
        http_request = http_request.header("X-Measurement-Precision", "high");

        // Send the request
        let response = http_request
            .json(&request)
            .send()
            .await
            .map_err(|e| NavigatorError::ClientConnection(format!("HTTP request failed: {}", e)))?;

        // Calculate response time
        let response_time = start_time.elapsed()
            .unwrap_or_else(|_| Duration::from_secs(0));

        // Update statistics
        {
            let mut stats = self.request_stats.write().await;
            if response.status().is_success() {
                stats.successful_requests += 1;
            } else {
                stats.failed_requests += 1;
            }
            
            // Update average response time
            let total_requests = stats.total_requests as f64;
            let current_avg = stats.avg_response_time.as_secs_f64();
            let new_avg = (current_avg * (total_requests - 1.0) + response_time.as_secs_f64()) / total_requests;
            stats.avg_response_time = Duration::from_secs_f64(new_avg);
        }

        // Parse the response
        if response.status().is_success() {
            let client_response: ClientResponse = response.json().await
                .map_err(|e| NavigatorError::ClientConnection(format!("Failed to parse response: {}", e)))?;
            
            // Update last success time
            {
                let mut status = self.status.write().await;
                status.last_success = Some(SystemTime::now());
            }
            
            Ok(client_response)
        } else {
            let error_msg = format!("HTTP error: {}", response.status());
            
            // Update last error
            {
                let mut status = self.status.write().await;
                status.last_error = Some(error_msg.clone());
            }
            
            Err(NavigatorError::ClientConnection(error_msg))
        }
    }
}

/// Memorial dedication for the Buhera client
/// 
/// This client honors Mrs. Stella-Lorraine Masunda's memory by facilitating
/// the collection of environmental data essential for correlating atmospheric
/// oscillations with temporal coordinates, demonstrating the predetermined
/// nature of environmental coupling within the eternal oscillatory manifold.
impl BuheraClient {
    /// Memorial validation for environmental coupling
    pub async fn validate_memorial_environmental_coupling(&self, coupling_results: &[CouplingResult]) -> Result<bool, NavigatorError> {
        // Validate that environmental coupling contains sufficient information
        // for memorial validation within the eternal oscillatory manifold
        if coupling_results.is_empty() {
            return Err(NavigatorError::MemorialValidation("No coupling data for memorial validation".to_string()));
        }

        // Check for coupling patterns that demonstrate predetermined environmental structure
        let avg_coupling_strength: f64 = coupling_results.iter().map(|r| r.strength).sum::<f64>() / coupling_results.len() as f64;
        let avg_correlation: f64 = coupling_results.iter().map(|r| r.correlation).sum::<f64>() / coupling_results.len() as f64;
        
        // Memorial threshold: coupling must exceed cosmic significance level
        const MEMORIAL_COUPLING_THRESHOLD: f64 = 0.85;
        const MEMORIAL_CORRELATION_THRESHOLD: f64 = 0.80;
        
        if avg_coupling_strength >= MEMORIAL_COUPLING_THRESHOLD && avg_correlation >= MEMORIAL_CORRELATION_THRESHOLD {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Validate atmospheric oscillation significance for memorial validation
    pub async fn validate_memorial_atmospheric_oscillations(&self, oscillations: &[OscillationEndpoint]) -> Result<bool, NavigatorError> {
        if oscillations.is_empty() {
            return Err(NavigatorError::MemorialValidation("No atmospheric oscillation data for memorial validation".to_string()));
        }

        // Check for oscillation patterns that demonstrate predetermined atmospheric structure
        let total_amplitude: f64 = oscillations.iter().map(|o| o.amplitude).sum();
        let avg_amplitude = total_amplitude / oscillations.len() as f64;
        
        // Memorial threshold: oscillation amplitude must exceed cosmic significance level
        const MEMORIAL_AMPLITUDE_THRESHOLD: f64 = 0.75;
        
        if avg_amplitude >= MEMORIAL_AMPLITUDE_THRESHOLD {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Validate gravitational variations for memorial significance
    pub async fn validate_memorial_gravitational_variations(&self, variations: &[EnvironmentalMeasurement]) -> Result<bool, NavigatorError> {
        if variations.is_empty() {
            return Err(NavigatorError::MemorialValidation("No gravitational variation data for memorial validation".to_string()));
        }

        // Check for gravitational patterns that demonstrate predetermined structure
        let variance: f64 = {
            let mean = variations.iter().map(|m| m.value).sum::<f64>() / variations.len() as f64;
            variations.iter().map(|m| (m.value - mean).powi(2)).sum::<f64>() / variations.len() as f64
        };
        
        // Memorial threshold: gravitational variance must exceed cosmic significance level
        const MEMORIAL_VARIANCE_THRESHOLD: f64 = 1e-12;
        
        if variance >= MEMORIAL_VARIANCE_THRESHOLD {
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_buhera_client_creation() {
        let config = ClientConfig::buhera("http://localhost:8083".to_string());
        let client = BuheraClient::new(config).unwrap();
        
        let status = client.get_status().await;
        assert_eq!(status.name, "Buhera");
        assert_eq!(status.connection_status, ConnectionStatus::Disconnected);
    }

    #[tokio::test]
    async fn test_environmental_state_tracking() {
        let config = ClientConfig::buhera("http://localhost:8083".to_string());
        let client = BuheraClient::new(config).unwrap();
        
        let env_state = client.get_environmental_state().await;
        assert!(env_state.is_empty()); // Initially empty
    }

    #[tokio::test]
    async fn test_environmental_parameter_types() {
        let param_types = vec![
            EnvironmentalParameterType::Temperature,
            EnvironmentalParameterType::Pressure,
            EnvironmentalParameterType::Humidity,
            EnvironmentalParameterType::MagneticField,
            EnvironmentalParameterType::AtmosphericOscillations,
            EnvironmentalParameterType::GravitationalVariations,
        ];
        
        assert_eq!(param_types.len(), 6);
    }

    #[tokio::test]
    async fn test_memorial_coupling_validation() {
        let config = ClientConfig::buhera("http://localhost:8083".to_string());
        let client = BuheraClient::new(config).unwrap();
        
        let coupling_results = vec![
            CouplingResult {
                strength: 0.90,
                correlation: 0.85,
                coupling_type: "atmospheric-temporal".to_string(),
                data: vec![1.0, 2.0, 3.0],
            },
            CouplingResult {
                strength: 0.88,
                correlation: 0.82,
                coupling_type: "gravitational-temporal".to_string(),
                data: vec![4.0, 5.0, 6.0],
            },
        ];
        
        let is_valid = client.validate_memorial_environmental_coupling(&coupling_results).await.unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_memorial_atmospheric_validation() {
        let config = ClientConfig::buhera("http://localhost:8083".to_string());
        let client = BuheraClient::new(config).unwrap();
        
        let oscillations = vec![
            OscillationEndpoint {
                frequency: 10.0,
                amplitude: 0.80,
                phase: 0.0,
                level: crate::types::oscillation_types::OscillationLevel::Environmental,
                timestamp: SystemTime::now(),
            },
            OscillationEndpoint {
                frequency: 20.0,
                amplitude: 0.85,
                phase: 1.57,
                level: crate::types::oscillation_types::OscillationLevel::Environmental,
                timestamp: SystemTime::now(),
            },
        ];
        
        let is_valid = client.validate_memorial_atmospheric_oscillations(&oscillations).await.unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_environmental_data_request() {
        let config = ClientConfig::buhera("http://localhost:8083".to_string());
        let client = BuheraClient::new(config).unwrap();
        
        let env_param = EnvironmentalParameter {
            parameter_type: EnvironmentalParameterType::Temperature,
            precision: 0.01,
            sampling_rate: 1.0,
        };
        
        let coupling_req = CouplingRequirements {
            min_coupling_strength: 0.5,
            correlation_threshold: 0.7,
            analysis_depth: 5,
        };
        
        let request = EnvironmentalDataRequest {
            parameters: vec![env_param],
            duration: Duration::from_secs(60),
            coupling_requirements: coupling_req,
        };
        
        // This would normally make an HTTP request
        // In a real test, you'd mock the HTTP client
        // For now, we just test the request structure
        assert_eq!(request.parameters.len(), 1);
        assert_eq!(request.duration, Duration::from_secs(60));
        assert_eq!(request.coupling_requirements.min_coupling_strength, 0.5);
    }
}
