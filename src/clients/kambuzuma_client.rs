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
use crate::types::client_types::kambuzuma::{
    OscillationDataRequest, OscillationDataResponse, QuantumStateFilter, CoherenceMeasurement,
    QuantumStateData,
};
use crate::types::oscillation_types::{OscillationEndpoint, OscillationLevel};
use crate::types::error_types::NavigatorError;

/// Kambuzuma Biological Quantum Computing System Client
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This client interfaces with the Kambuzuma biological quantum computing system
/// to collect quantum coherence data and oscillation endpoints essential for
/// ultra-precise temporal coordinate navigation.
#[derive(Debug, Clone)]
pub struct KambuzumaClient {
    /// Client configuration
    config: ClientConfig,
    /// HTTP client
    http_client: Client,
    /// Client status
    status: Arc<RwLock<ClientStatus>>,
    /// Request statistics
    request_stats: Arc<RwLock<RequestStats>>,
}

impl KambuzumaClient {
    /// Create a new Kambuzuma client
    pub fn new(config: ClientConfig) -> Result<Self, NavigatorError> {
        let http_client = Client::builder()
            .timeout(config.timeout)
            .user_agent("MasundaNavigator/1.0")
            .build()
            .map_err(|e| NavigatorError::ClientConnection(format!("Failed to create HTTP client: {}", e)))?;

        let status = Arc::new(RwLock::new(ClientStatus {
            name: "Kambuzuma".to_string(),
            connection_status: ConnectionStatus::Disconnected,
            last_success: None,
            last_error: None,
            request_stats: RequestStats::new(),
            health_metrics: HashMap::new(),
        }));

        let request_stats = Arc::new(RwLock::new(RequestStats::new()));

        Ok(Self {
            config,
            http_client,
            status,
            request_stats,
        })
    }

    /// Connect to the Kambuzuma system
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

    /// Disconnect from the Kambuzuma system
    pub async fn disconnect(&self) -> Result<(), NavigatorError> {
        let mut status = self.status.write().await;
        status.connection_status = ConnectionStatus::Disconnected;
        Ok(())
    }

    /// Get client status
    pub async fn get_status(&self) -> ClientStatus {
        self.status.read().await.clone()
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

    /// Get quantum oscillation data from Kambuzuma
    /// 
    /// This method requests quantum coherence measurements and oscillation endpoints
    /// from the biological quantum computing system for temporal coordinate analysis.
    pub async fn get_oscillation_data(&self, request: OscillationDataRequest) -> Result<OscillationDataResponse, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("coherence_level".to_string(), serde_json::json!(request.coherence_level));
        params.insert("duration_secs".to_string(), serde_json::json!(request.duration.as_secs()));
        params.insert("precision_requirement".to_string(), serde_json::json!(request.precision_requirement));
        params.insert("state_filters".to_string(), serde_json::to_value(&request.state_filters).unwrap());

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
                    NavigatorError::DataProcessing("No data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse oscillation data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            ResponseStatus::Timeout => Err(NavigatorError::ClientConnection("Request timeout".to_string())),
            ResponseStatus::RateLimited => Err(NavigatorError::ClientConnection("Rate limited".to_string())),
            ResponseStatus::PartialSuccess(msg) => {
                // Still try to parse partial data
                if let Some(data) = response.data {
                    serde_json::from_value(data)
                        .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse partial data: {}", e)))
                } else {
                    Err(NavigatorError::DataProcessing(format!("Partial success but no data: {}", msg)))
                }
            }
        }
    }

    /// Get quantum coherence measurements
    pub async fn get_coherence_measurements(&self, duration: Duration) -> Result<Vec<CoherenceMeasurement>, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("duration_secs".to_string(), serde_json::json!(duration.as_secs()));

        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::Custom("GetCoherence".to_string()),
            parameters: params,
            timestamp: SystemTime::now(),
            priority: RequestPriority::Normal,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No coherence data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse coherence data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            _ => Err(NavigatorError::ClientConnection("Failed to get coherence measurements".to_string())),
        }
    }

    /// Get quantum state data
    pub async fn get_quantum_state(&self, filters: Vec<QuantumStateFilter>) -> Result<QuantumStateData, NavigatorError> {
        let mut params = HashMap::new();
        params.insert("filters".to_string(), serde_json::to_value(filters).unwrap());

        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::Custom("GetQuantumState".to_string()),
            parameters: params,
            timestamp: SystemTime::now(),
            priority: RequestPriority::Normal,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No quantum state data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse quantum state data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            _ => Err(NavigatorError::ClientConnection("Failed to get quantum state".to_string())),
        }
    }

    /// Get oscillation endpoints from the quantum level
    pub async fn get_quantum_oscillation_endpoints(&self) -> Result<Vec<OscillationEndpoint>, NavigatorError> {
        let client_request = ClientRequest {
            request_id: Uuid::new_v4().to_string(),
            request_type: RequestType::Custom("GetQuantumEndpoints".to_string()),
            parameters: HashMap::new(),
            timestamp: SystemTime::now(),
            priority: RequestPriority::High,
        };

        let response = self.send_request(client_request).await?;
        
        match response.status {
            ResponseStatus::Success => {
                let data = response.data.ok_or_else(|| {
                    NavigatorError::DataProcessing("No endpoint data in response".to_string())
                })?;
                
                serde_json::from_value(data)
                    .map_err(|e| NavigatorError::DataProcessing(format!("Failed to parse endpoint data: {}", e)))
            }
            ResponseStatus::Error(msg) => Err(NavigatorError::ClientConnection(msg)),
            _ => Err(NavigatorError::ClientConnection("Failed to get oscillation endpoints".to_string())),
        }
    }

    /// Send a request to the Kambuzuma system
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

/// Memorial dedication for the Kambuzuma client
/// 
/// This client honors Mrs. Stella-Lorraine Masunda's memory by facilitating
/// the collection of quantum coherence data essential for proving the
/// predetermined nature of temporal coordinates through biological quantum
/// computing systems.
impl KambuzumaClient {
    /// Memorial validation for quantum coherence data
    pub async fn validate_memorial_significance(&self, coherence_data: &Vec<CoherenceMeasurement>) -> Result<bool, NavigatorError> {
        // Validate that the coherence data contains sufficient information
        // for memorial validation within the eternal oscillatory manifold
        if coherence_data.is_empty() {
            return Err(NavigatorError::MemorialValidation("No coherence data for memorial validation".to_string()));
        }

        // Check for coherence patterns that demonstrate predetermined temporal structure
        let avg_coherence: f64 = coherence_data.iter().map(|m| m.coherence).sum::<f64>() / coherence_data.len() as f64;
        
        // Memorial threshold: coherence must exceed cosmic significance level
        const MEMORIAL_COHERENCE_THRESHOLD: f64 = 0.95;
        
        if avg_coherence >= MEMORIAL_COHERENCE_THRESHOLD {
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
    async fn test_kambuzuma_client_creation() {
        let config = ClientConfig::kambuzuma("http://localhost:8080".to_string());
        let client = KambuzumaClient::new(config).unwrap();
        
        let status = client.get_status().await;
        assert_eq!(status.name, "Kambuzuma");
        assert_eq!(status.connection_status, ConnectionStatus::Disconnected);
    }

    #[tokio::test]
    async fn test_memorial_validation() {
        let config = ClientConfig::kambuzuma("http://localhost:8080".to_string());
        let client = KambuzumaClient::new(config).unwrap();
        
        let coherence_data = vec![
            CoherenceMeasurement {
                time: 0.0,
                coherence: 0.96,
                decoherence_rate: 0.01,
            },
            CoherenceMeasurement {
                time: 1.0,
                coherence: 0.97,
                decoherence_rate: 0.008,
            },
        ];
        
        let is_valid = client.validate_memorial_significance(&coherence_data).await.unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_oscillation_data_request() {
        let config = ClientConfig::kambuzuma("http://localhost:8080".to_string());
        let client = KambuzumaClient::new(config).unwrap();
        
        let request = OscillationDataRequest {
            coherence_level: 0.95,
            duration: Duration::from_secs(10),
            precision_requirement: 1e-30,
            state_filters: vec![],
        };
        
        // This would normally make an HTTP request
        // In a real test, you'd mock the HTTP client
        // For now, we just test the request creation
        assert_eq!(request.coherence_level, 0.95);
        assert_eq!(request.duration, Duration::from_secs(10));
        assert_eq!(request.precision_requirement, 1e-30);
    }
}
