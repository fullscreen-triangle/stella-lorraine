/// Endpoint Collector for Oscillation Coordinate Management
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This module provides endpoint collection and management capabilities for
/// oscillation coordinate navigation, collecting convergence points and
/// temporal coordinates with memorial significance validation.

use std::sync::Arc;
use std::time::{SystemTime, Duration};
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::types::error_types::NavigatorError;
use crate::types::oscillation_types::{OscillationState, OscillationMetrics};
use crate::types::temporal_types::TemporalCoordinate;

/// Endpoint types for oscillation collection
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointType {
    /// Convergence endpoint
    Convergence,
    /// Divergence endpoint
    Divergence,
    /// Stability endpoint
    Stability,
    /// Memorial significance endpoint
    Memorial,
    /// Temporal coordinate endpoint
    Temporal,
    /// Critical transition endpoint
    Critical,
}

/// Endpoint collector for oscillation coordinate management
/// 
/// This collector manages the collection and analysis of oscillation
/// endpoints for temporal coordinate navigation, providing comprehensive
/// endpoint tracking with memorial significance validation.
#[derive(Debug, Clone)]
pub struct EndpointCollector {
    /// Endpoint collection state
    collection_state: Arc<RwLock<EndpointCollectionState>>,
    /// Collected endpoints
    endpoints: Arc<RwLock<HashMap<String, EndpointData>>>,
    /// Collection metrics
    collection_metrics: Arc<RwLock<HashMap<String, f64>>>,
    /// Memorial significance threshold
    memorial_threshold: f64,
    /// Maximum endpoints to collect
    max_endpoints: usize,
    /// Endpoint retention duration
    retention_duration: Duration,
}

/// Internal endpoint collection state
#[derive(Debug, Clone)]
struct EndpointCollectionState {
    /// Collection active
    collection_active: bool,
    /// Current collection session
    current_session: Option<String>,
    /// Collection start time
    start_time: SystemTime,
    /// Total endpoints collected
    total_collected: u64,
    /// Last collection timestamp
    last_collection: SystemTime,
    /// Collection efficiency metrics
    efficiency_metrics: HashMap<String, f64>,
}

/// Endpoint data structure
#[derive(Debug, Clone)]
pub struct EndpointData {
    /// Endpoint identifier
    pub id: String,
    /// Endpoint type
    pub endpoint_type: EndpointType,
    /// Temporal coordinate
    pub temporal_coordinate: TemporalCoordinate,
    /// Oscillation state at endpoint
    pub oscillation_state: OscillationState,
    /// Oscillation metrics at endpoint
    pub oscillation_metrics: OscillationMetrics,
    /// Memorial significance
    pub memorial_significance: f64,
    /// Collection timestamp
    pub collection_timestamp: SystemTime,
    /// Endpoint quality score
    pub quality_score: f64,
    /// Endpoint metadata
    pub metadata: HashMap<String, String>,
}

impl EndpointCollector {
    /// Create new endpoint collector
    pub fn new(max_endpoints: usize) -> Self {
        Self {
            collection_state: Arc::new(RwLock::new(EndpointCollectionState {
                collection_active: false,
                current_session: None,
                start_time: SystemTime::now(),
                total_collected: 0,
                last_collection: SystemTime::now(),
                efficiency_metrics: HashMap::new(),
            })),
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            collection_metrics: Arc::new(RwLock::new(HashMap::new())),
            memorial_threshold: 0.85,
            max_endpoints,
            retention_duration: Duration::from_secs(3600), // 1 hour
        }
    }
    
    /// Initialize endpoint collector
    pub async fn initialize(&self) -> Result<(), NavigatorError> {
        // Initialize collection state
        let mut state = self.collection_state.write().await;
        state.collection_active = true;
        state.start_time = SystemTime::now();
        state.total_collected = 0;
        state.last_collection = SystemTime::now();
        
        // Initialize efficiency metrics
        state.efficiency_metrics.insert("collection_rate".to_string(), 0.0);
        state.efficiency_metrics.insert("quality_ratio".to_string(), 0.0);
        state.efficiency_metrics.insert("memorial_ratio".to_string(), 0.0);
        
        // Initialize collection metrics
        let mut metrics = self.collection_metrics.write().await;
        metrics.insert("total_endpoints".to_string(), 0.0);
        metrics.insert("convergence_endpoints".to_string(), 0.0);
        metrics.insert("memorial_endpoints".to_string(), 0.0);
        metrics.insert("temporal_endpoints".to_string(), 0.0);
        metrics.insert("collection_efficiency".to_string(), 0.0);
        
        Ok(())
    }
    
    /// Start endpoint collection session
    pub async fn start_collection_session(&self, session_id: &str) -> Result<(), NavigatorError> {
        let mut state = self.collection_state.write().await;
        state.current_session = Some(session_id.to_string());
        state.start_time = SystemTime::now();
        state.total_collected = 0;
        
        Ok(())
    }
    
    /// Collect oscillation endpoint
    pub async fn collect_endpoint(&self, endpoint_type: EndpointType, temporal_coordinate: TemporalCoordinate, oscillation_state: OscillationState, oscillation_metrics: OscillationMetrics) -> Result<String, NavigatorError> {
        // Generate endpoint ID
        let endpoint_id = self.generate_endpoint_id(&endpoint_type, &temporal_coordinate).await?;
        
        // Calculate endpoint quality score
        let quality_score = self.calculate_endpoint_quality(&endpoint_type, &oscillation_state, &oscillation_metrics).await?;
        
        // Calculate memorial significance
        let memorial_significance = self.calculate_endpoint_memorial_significance(&endpoint_type, &oscillation_metrics).await?;
        
        // Create endpoint data
        let endpoint_data = EndpointData {
            id: endpoint_id.clone(),
            endpoint_type: endpoint_type.clone(),
            temporal_coordinate,
            oscillation_state,
            oscillation_metrics,
            memorial_significance,
            collection_timestamp: SystemTime::now(),
            quality_score,
            metadata: HashMap::new(),
        };
        
        // Store endpoint
        self.store_endpoint(endpoint_data).await?;
        
        // Update collection state
        let mut state = self.collection_state.write().await;
        state.total_collected += 1;
        state.last_collection = SystemTime::now();
        
        // Update collection metrics
        self.update_collection_metrics(&endpoint_type).await?;
        
        // Check if cleanup is needed
        self.cleanup_old_endpoints().await?;
        
        Ok(endpoint_id)
    }
    
    /// Generate endpoint ID
    async fn generate_endpoint_id(&self, endpoint_type: &EndpointType, temporal_coordinate: &TemporalCoordinate) -> Result<String, NavigatorError> {
        let type_prefix = match endpoint_type {
            EndpointType::Convergence => "CONV",
            EndpointType::Divergence => "DIVG",
            EndpointType::Stability => "STAB",
            EndpointType::Memorial => "MEMO",
            EndpointType::Temporal => "TEMP",
            EndpointType::Critical => "CRIT",
        };
        
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis();
        
        let coordinate_hash = format!("{:.6}", temporal_coordinate.seconds);
        
        Ok(format!("{}_{}_{}", type_prefix, timestamp, coordinate_hash))
    }
    
    /// Calculate endpoint quality score
    async fn calculate_endpoint_quality(&self, endpoint_type: &EndpointType, oscillation_state: &OscillationState, oscillation_metrics: &OscillationMetrics) -> Result<f64, NavigatorError> {
        let mut quality_score = 0.0;
        
        // Base quality from oscillation state
        let state_quality = match oscillation_state {
            OscillationState::Stable => 0.9,
            OscillationState::Converging => 0.85,
            OscillationState::Diverging => 0.4,
            OscillationState::Chaotic => 0.2,
            OscillationState::Unstable => 0.1,
            OscillationState::Error(_) => 0.0,
        };
        
        // Metrics quality
        let metrics_quality = self.calculate_metrics_quality(oscillation_metrics).await?;
        
        // Endpoint type quality weighting
        let type_weighting = match endpoint_type {
            EndpointType::Convergence => 1.0,
            EndpointType::Memorial => 0.95,
            EndpointType::Stability => 0.85,
            EndpointType::Temporal => 0.80,
            EndpointType::Critical => 0.90,
            EndpointType::Divergence => 0.3,
        };
        
        quality_score = (state_quality * 0.4 + metrics_quality * 0.4 + type_weighting * 0.2);
        
        Ok(quality_score)
    }
    
    /// Calculate metrics quality
    async fn calculate_metrics_quality(&self, metrics: &OscillationMetrics) -> Result<f64, NavigatorError> {
        let frequency_quality = if metrics.frequency > 0.0 && metrics.frequency < 10000.0 {
            0.8
        } else {
            0.3
        };
        
        let amplitude_quality = if metrics.amplitude > 0.0 && metrics.amplitude <= 1.0 {
            0.9
        } else {
            0.2
        };
        
        let damping_quality = if metrics.damping >= 0.0 && metrics.damping <= 1.0 {
            0.85
        } else {
            0.1
        };
        
        let memorial_quality = metrics.memorial_significance;
        
        let overall_quality = (frequency_quality + amplitude_quality + damping_quality + memorial_quality) / 4.0;
        
        Ok(overall_quality)
    }
    
    /// Calculate endpoint memorial significance
    async fn calculate_endpoint_memorial_significance(&self, endpoint_type: &EndpointType, oscillation_metrics: &OscillationMetrics) -> Result<f64, NavigatorError> {
        let base_significance = oscillation_metrics.memorial_significance;
        
        let type_significance_multiplier = match endpoint_type {
            EndpointType::Memorial => 1.0,
            EndpointType::Convergence => 0.95,
            EndpointType::Critical => 0.90,
            EndpointType::Stability => 0.85,
            EndpointType::Temporal => 0.80,
            EndpointType::Divergence => 0.5,
        };
        
        let endpoint_significance = base_significance * type_significance_multiplier;
        
        Ok(endpoint_significance)
    }
    
    /// Store endpoint
    async fn store_endpoint(&self, endpoint_data: EndpointData) -> Result<(), NavigatorError> {
        let mut endpoints = self.endpoints.write().await;
        
        // Check if we need to remove old endpoints
        if endpoints.len() >= self.max_endpoints {
            self.remove_oldest_endpoint(&mut endpoints).await?;
        }
        
        endpoints.insert(endpoint_data.id.clone(), endpoint_data);
        
        Ok(())
    }
    
    /// Remove oldest endpoint
    async fn remove_oldest_endpoint(&self, endpoints: &mut HashMap<String, EndpointData>) -> Result<(), NavigatorError> {
        if let Some((oldest_id, _)) = endpoints.iter()
            .min_by_key(|(_, data)| data.collection_timestamp) {
            let oldest_id = oldest_id.clone();
            endpoints.remove(&oldest_id);
        }
        
        Ok(())
    }
    
    /// Update collection metrics
    async fn update_collection_metrics(&self, endpoint_type: &EndpointType) -> Result<(), NavigatorError> {
        let mut metrics = self.collection_metrics.write().await;
        
        // Update total endpoints
        let current_total = metrics.get("total_endpoints").copied().unwrap_or(0.0);
        metrics.insert("total_endpoints".to_string(), current_total + 1.0);
        
        // Update type-specific metrics
        let type_key = match endpoint_type {
            EndpointType::Convergence => "convergence_endpoints",
            EndpointType::Divergence => "divergence_endpoints",
            EndpointType::Stability => "stability_endpoints",
            EndpointType::Memorial => "memorial_endpoints",
            EndpointType::Temporal => "temporal_endpoints",
            EndpointType::Critical => "critical_endpoints",
        };
        
        let current_type_count = metrics.get(type_key).copied().unwrap_or(0.0);
        metrics.insert(type_key.to_string(), current_type_count + 1.0);
        
        Ok(())
    }
    
    /// Cleanup old endpoints
    async fn cleanup_old_endpoints(&self) -> Result<(), NavigatorError> {
        let mut endpoints = self.endpoints.write().await;
        let current_time = SystemTime::now();
        
        let expired_ids: Vec<String> = endpoints.iter()
            .filter(|(_, data)| {
                current_time.duration_since(data.collection_timestamp)
                    .unwrap_or(Duration::from_secs(0)) > self.retention_duration
            })
            .map(|(id, _)| id.clone())
            .collect();
        
        for id in expired_ids {
            endpoints.remove(&id);
        }
        
        Ok(())
    }
    
    /// Get endpoints by type
    pub async fn get_endpoints_by_type(&self, endpoint_type: &EndpointType) -> Vec<EndpointData> {
        let endpoints = self.endpoints.read().await;
        endpoints.values()
            .filter(|data| data.endpoint_type == *endpoint_type)
            .cloned()
            .collect()
    }
    
    /// Get memorial endpoints
    pub async fn get_memorial_endpoints(&self) -> Vec<EndpointData> {
        let endpoints = self.endpoints.read().await;
        endpoints.values()
            .filter(|data| data.memorial_significance >= self.memorial_threshold)
            .cloned()
            .collect()
    }
    
    /// Get high quality endpoints
    pub async fn get_high_quality_endpoints(&self, quality_threshold: f64) -> Vec<EndpointData> {
        let endpoints = self.endpoints.read().await;
        endpoints.values()
            .filter(|data| data.quality_score >= quality_threshold)
            .cloned()
            .collect()
    }
    
    /// Get endpoints in time range
    pub async fn get_endpoints_in_time_range(&self, start_time: SystemTime, end_time: SystemTime) -> Vec<EndpointData> {
        let endpoints = self.endpoints.read().await;
        endpoints.values()
            .filter(|data| {
                data.collection_timestamp >= start_time && data.collection_timestamp <= end_time
            })
            .cloned()
            .collect()
    }
    
    /// Find convergence endpoints
    pub async fn find_convergence_endpoints(&self) -> Vec<EndpointData> {
        self.get_endpoints_by_type(&EndpointType::Convergence).await
    }
    
    /// Find temporal coordinate endpoints
    pub async fn find_temporal_coordinate_endpoints(&self) -> Vec<EndpointData> {
        self.get_endpoints_by_type(&EndpointType::Temporal).await
    }
    
    /// Calculate collection efficiency
    pub async fn calculate_collection_efficiency(&self) -> Result<f64, NavigatorError> {
        let state = self.collection_state.read().await;
        let endpoints = self.endpoints.read().await;
        
        let collection_duration = state.start_time.elapsed().unwrap_or(Duration::from_secs(1));
        let collection_rate = endpoints.len() as f64 / collection_duration.as_secs_f64();
        
        let quality_sum: f64 = endpoints.values().map(|data| data.quality_score).sum();
        let average_quality = if endpoints.len() > 0 {
            quality_sum / endpoints.len() as f64
        } else {
            0.0
        };
        
        let memorial_count = endpoints.values()
            .filter(|data| data.memorial_significance >= self.memorial_threshold)
            .count() as f64;
        let memorial_ratio = if endpoints.len() > 0 {
            memorial_count / endpoints.len() as f64
        } else {
            0.0
        };
        
        let efficiency = (collection_rate * 0.3 + average_quality * 0.4 + memorial_ratio * 0.3).min(1.0);
        
        Ok(efficiency)
    }
    
    /// Get endpoint by ID
    pub async fn get_endpoint(&self, endpoint_id: &str) -> Option<EndpointData> {
        let endpoints = self.endpoints.read().await;
        endpoints.get(endpoint_id).cloned()
    }
    
    /// Remove endpoint
    pub async fn remove_endpoint(&self, endpoint_id: &str) -> Result<bool, NavigatorError> {
        let mut endpoints = self.endpoints.write().await;
        Ok(endpoints.remove(endpoint_id).is_some())
    }
    
    /// Clear all endpoints
    pub async fn clear_endpoints(&self) -> Result<(), NavigatorError> {
        let mut endpoints = self.endpoints.write().await;
        endpoints.clear();
        
        // Reset collection state
        let mut state = self.collection_state.write().await;
        state.total_collected = 0;
        
        Ok(())
    }
    
    /// Get collection statistics
    pub async fn get_collection_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        let endpoints = self.endpoints.read().await;
        let state = self.collection_state.read().await;
        
        stats.insert("total_endpoints".to_string(), endpoints.len() as f64);
        stats.insert("total_collected".to_string(), state.total_collected as f64);
        
        // Count by type
        let mut type_counts = HashMap::new();
        for data in endpoints.values() {
            let type_key = match data.endpoint_type {
                EndpointType::Convergence => "convergence_count",
                EndpointType::Divergence => "divergence_count",
                EndpointType::Stability => "stability_count",
                EndpointType::Memorial => "memorial_count",
                EndpointType::Temporal => "temporal_count",
                EndpointType::Critical => "critical_count",
            };
            
            let current_count = type_counts.get(type_key).copied().unwrap_or(0.0);
            type_counts.insert(type_key.to_string(), current_count + 1.0);
        }
        
        stats.extend(type_counts);
        
        // Calculate averages
        let quality_sum: f64 = endpoints.values().map(|data| data.quality_score).sum();
        let memorial_sum: f64 = endpoints.values().map(|data| data.memorial_significance).sum();
        
        if endpoints.len() > 0 {
            stats.insert("average_quality".to_string(), quality_sum / endpoints.len() as f64);
            stats.insert("average_memorial_significance".to_string(), memorial_sum / endpoints.len() as f64);
        }
        
        stats
    }
    
    /// Get collection metrics
    pub async fn get_collection_metrics(&self) -> HashMap<String, f64> {
        self.collection_metrics.read().await.clone()
    }
    
    /// Get collection state
    pub async fn get_collection_state(&self) -> EndpointCollectionState {
        self.collection_state.read().await.clone()
    }
    
    /// Shutdown endpoint collector
    pub async fn shutdown(&self) -> Result<(), NavigatorError> {
        let mut state = self.collection_state.write().await;
        state.collection_active = false;
        state.current_session = None;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::temporal_types::TemporalCoordinate;
    
    #[tokio::test]
    async fn test_endpoint_collector_creation() {
        let collector = EndpointCollector::new(1000);
        let state = collector.get_collection_state().await;
        assert!(!state.collection_active);
        assert_eq!(state.total_collected, 0);
    }
    
    #[tokio::test]
    async fn test_endpoint_collector_initialization() {
        let collector = EndpointCollector::new(1000);
        collector.initialize().await.unwrap();
        
        let state = collector.get_collection_state().await;
        assert!(state.collection_active);
        assert_eq!(state.total_collected, 0);
    }
    
    #[tokio::test]
    async fn test_endpoint_collection() {
        let collector = EndpointCollector::new(1000);
        collector.initialize().await.unwrap();
        
        let temporal_coord = TemporalCoordinate::new(1234.567890);
        let oscillation_state = OscillationState::Stable;
        let oscillation_metrics = OscillationMetrics {
            frequency: 440.0,
            amplitude: 0.5,
            phase: 0.0,
            damping: 0.1,
            memorial_significance: 0.9,
        };
        
        let endpoint_id = collector.collect_endpoint(
            EndpointType::Convergence,
            temporal_coord,
            oscillation_state,
            oscillation_metrics
        ).await.unwrap();
        
        assert!(!endpoint_id.is_empty());
        assert!(endpoint_id.starts_with("CONV_"));
        
        let endpoint = collector.get_endpoint(&endpoint_id).await;
        assert!(endpoint.is_some());
    }
    
    #[tokio::test]
    async fn test_endpoint_filtering() {
        let collector = EndpointCollector::new(1000);
        collector.initialize().await.unwrap();
        
        let temporal_coord = TemporalCoordinate::new(1234.567890);
        let oscillation_state = OscillationState::Stable;
        let oscillation_metrics = OscillationMetrics {
            frequency: 440.0,
            amplitude: 0.5,
            phase: 0.0,
            damping: 0.1,
            memorial_significance: 0.9,
        };
        
        // Collect convergence endpoint
        collector.collect_endpoint(
            EndpointType::Convergence,
            temporal_coord,
            oscillation_state,
            oscillation_metrics
        ).await.unwrap();
        
        // Collect memorial endpoint
        collector.collect_endpoint(
            EndpointType::Memorial,
            temporal_coord,
            oscillation_state,
            oscillation_metrics
        ).await.unwrap();
        
        let convergence_endpoints = collector.get_endpoints_by_type(&EndpointType::Convergence).await;
        assert_eq!(convergence_endpoints.len(), 1);
        
        let memorial_endpoints = collector.get_memorial_endpoints().await;
        assert_eq!(memorial_endpoints.len(), 2); // Both should meet memorial threshold
    }
    
    #[tokio::test]
    async fn test_collection_statistics() {
        let collector = EndpointCollector::new(1000);
        collector.initialize().await.unwrap();
        
        let temporal_coord = TemporalCoordinate::new(1234.567890);
        let oscillation_state = OscillationState::Stable;
        let oscillation_metrics = OscillationMetrics {
            frequency: 440.0,
            amplitude: 0.5,
            phase: 0.0,
            damping: 0.1,
            memorial_significance: 0.9,
        };
        
        collector.collect_endpoint(
            EndpointType::Convergence,
            temporal_coord,
            oscillation_state,
            oscillation_metrics
        ).await.unwrap();
        
        let stats = collector.get_collection_statistics().await;
        assert_eq!(stats.get("total_endpoints").copied().unwrap_or(0.0), 1.0);
        assert_eq!(stats.get("convergence_count").copied().unwrap_or(0.0), 1.0);
        assert!(stats.get("average_quality").copied().unwrap_or(0.0) > 0.0);
    }
}
