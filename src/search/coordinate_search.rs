use std::sync::Arc;
use std::time::{Duration, SystemTime};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{info, debug, error, warn};

use crate::types::*;
use crate::config::system_config::SystemConfig;

/// Temporal coordinate search engine
/// 
/// This engine orchestrates the search for precise temporal coordinates by
/// coordinating data collection from external systems and applying quantum
/// search algorithms to navigate to predetermined temporal coordinates.
pub struct CoordinateSearchEngine {
    /// System configuration
    config: Arc<SystemConfig>,
    
    /// Current search state
    state: Arc<RwLock<SearchState>>,
    
    /// Search statistics
    statistics: Arc<RwLock<SearchStatistics>>,
    
    /// Quantum search space
    quantum_space: Arc<RwLock<QuantumSearchSpace>>,
    
    /// Search optimization parameters
    optimization_params: Arc<RwLock<SearchOptimizationParams>>,
}

/// Current state of the search engine
#[derive(Debug, Clone, PartialEq)]
pub struct SearchState {
    /// Current search status
    pub status: SearchStatus,
    
    /// Active searches
    pub active_searches: HashMap<String, ActiveSearch>,
    
    /// Search cache
    pub search_cache: HashMap<String, CachedSearchResult>,
    
    /// Last search timestamp
    pub last_search: Option<SystemTime>,
}

/// Search status
#[derive(Debug, Clone, PartialEq)]
pub enum SearchStatus {
    /// Search engine is idle
    Idle,
    
    /// Search engine is initializing
    Initializing,
    
    /// Search engine is ready
    Ready,
    
    /// Search engine is actively searching
    Searching {
        /// Number of active searches
        active_count: usize,
        /// Search start time
        start_time: SystemTime,
    },
    
    /// Search engine has an error
    Error {
        /// Error details
        error: String,
        /// Error timestamp
        timestamp: SystemTime,
    },
}

/// Active search information
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveSearch {
    /// Search ID
    pub search_id: String,
    
    /// Search parameters
    pub search_params: TemporalSearchParams,
    
    /// Search start time
    pub start_time: SystemTime,
    
    /// Search progress (0.0 to 1.0)
    pub progress: f64,
    
    /// Intermediate results
    pub intermediate_results: Vec<TemporalSearchResult>,
}

/// Temporal search parameters
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalSearchParams {
    /// Search window
    pub window: TemporalWindow,
    
    /// Search precision target
    pub precision_target: f64,
    
    /// Maximum search time
    pub max_search_time: Duration,
    
    /// Search algorithm
    pub algorithm: SearchAlgorithm,
    
    /// Search priority
    pub priority: SearchPriority,
}

/// Search algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum SearchAlgorithm {
    /// Quantum search algorithm
    Quantum,
    
    /// Oscillation convergence search
    OscillationConvergence,
    
    /// Hierarchical search
    Hierarchical,
    
    /// Adaptive search
    Adaptive,
    
    /// Hybrid search (combines multiple algorithms)
    Hybrid(Vec<SearchAlgorithm>),
}

/// Search priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum SearchPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Cached search result
#[derive(Debug, Clone, PartialEq)]
pub struct CachedSearchResult {
    /// Search parameters
    pub search_params: TemporalSearchParams,
    
    /// Search result
    pub result: TemporalSearchResult,
    
    /// Cache timestamp
    pub timestamp: SystemTime,
    
    /// Cache expiry
    pub expiry: SystemTime,
    
    /// Cache hit count
    pub hit_count: usize,
}

/// Search statistics
#[derive(Debug, Clone, PartialEq)]
pub struct SearchStatistics {
    /// Total searches performed
    pub total_searches: usize,
    
    /// Successful searches
    pub successful_searches: usize,
    
    /// Failed searches
    pub failed_searches: usize,
    
    /// Average search time
    pub avg_search_time: Duration,
    
    /// Best precision achieved
    pub best_precision: Option<f64>,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Search algorithm performance
    pub algorithm_performance: HashMap<SearchAlgorithm, AlgorithmPerformance>,
}

/// Algorithm performance metrics
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmPerformance {
    /// Number of uses
    pub uses: usize,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Average search time
    pub avg_time: Duration,
    
    /// Average precision achieved
    pub avg_precision: f64,
}

/// Quantum search space
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumSearchSpace {
    /// Search space dimensions
    pub dimensions: usize,
    
    /// Search space bounds
    pub bounds: Vec<(f64, f64)>,
    
    /// Quantum state preparation
    pub quantum_state: QuantumState,
    
    /// Search operators
    pub operators: Vec<SearchOperator>,
}

/// Quantum state for search
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumState {
    /// State vector
    pub state_vector: Vec<f64>,
    
    /// Probability amplitudes
    pub amplitudes: Vec<f64>,
    
    /// Entanglement measures
    pub entanglement: Vec<f64>,
}

/// Search operator
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOperator {
    /// Operator name
    pub name: String,
    
    /// Operator matrix
    pub matrix: Vec<Vec<f64>>,
    
    /// Operator parameters
    pub parameters: HashMap<String, f64>,
}

/// Search optimization parameters
#[derive(Debug, Clone, PartialEq)]
pub struct SearchOptimizationParams {
    /// Optimization target
    pub target: OptimizationTarget,
    
    /// Search convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
    
    /// Search constraints
    pub constraints: Vec<SearchConstraint>,
    
    /// Adaptive parameters
    pub adaptive_params: AdaptiveParams,
}

/// Search constraint
#[derive(Debug, Clone, PartialEq)]
pub struct SearchConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    
    /// Constraint value
    pub value: f64,
    
    /// Constraint priority
    pub priority: f64,
}

/// Adaptive search parameters
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveParams {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Exploration rate
    pub exploration_rate: f64,
    
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    
    /// Memory size
    pub memory_size: usize,
}

impl CoordinateSearchEngine {
    /// Creates a new coordinate search engine
    pub async fn new(config: Arc<SystemConfig>) -> NavigatorResult<Self> {
        info!("Initializing Coordinate Search Engine");
        
        // Initialize search state
        let state = Arc::new(RwLock::new(SearchState {
            status: SearchStatus::Initializing,
            active_searches: HashMap::new(),
            search_cache: HashMap::new(),
            last_search: None,
        }));
        
        // Initialize search statistics
        let statistics = Arc::new(RwLock::new(SearchStatistics {
            total_searches: 0,
            successful_searches: 0,
            failed_searches: 0,
            avg_search_time: Duration::from_millis(0),
            best_precision: None,
            cache_hit_rate: 0.0,
            algorithm_performance: HashMap::new(),
        }));
        
        // Initialize quantum search space
        let quantum_space = Arc::new(RwLock::new(QuantumSearchSpace {
            dimensions: 4, // 4D spacetime
            bounds: vec![
                (-1e10, 1e10), // x bounds
                (-1e10, 1e10), // y bounds
                (-1e10, 1e10), // z bounds
                (0.0, 1e20),   // t bounds
            ],
            quantum_state: QuantumState {
                state_vector: vec![1.0; 16], // 2^4 dimensions
                amplitudes: vec![0.25; 16],  // Uniform superposition
                entanglement: vec![0.0; 16],
            },
            operators: vec![
                SearchOperator {
                    name: "Grover".to_string(),
                    matrix: vec![vec![1.0; 16]; 16], // Identity for now
                    parameters: HashMap::new(),
                },
            ],
        }));
        
        // Initialize optimization parameters
        let optimization_params = Arc::new(RwLock::new(SearchOptimizationParams {
            target: OptimizationTarget::MinimizeUncertainty,
            convergence_criteria: ConvergenceCriteria {
                relative_tolerance: 1e-15,
                absolute_tolerance: 1e-30,
                max_function_evaluations: 10000,
                stall_generations: 100,
            },
            constraints: vec![
                SearchConstraint {
                    constraint_type: ConstraintType::MaxMeasurementTime,
                    value: 300.0, // 5 minutes
                    priority: 1.0,
                },
            ],
            adaptive_params: AdaptiveParams {
                learning_rate: 0.01,
                exploration_rate: 0.1,
                adaptation_threshold: 0.05,
                memory_size: 1000,
            },
        }));
        
        // Update state to ready
        {
            let mut state = state.write().await;
            state.status = SearchStatus::Ready;
        }
        
        info!("Coordinate Search Engine initialized successfully");
        
        Ok(Self {
            config,
            state,
            statistics,
            quantum_space,
            optimization_params,
        })
    }
    
    /// Performs a temporal coordinate search
    pub async fn search_coordinate(&self, params: TemporalSearchParams) -> NavigatorResult<TemporalSearchResult> {
        let search_id = format!("search_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos());
        let start_time = SystemTime::now();
        
        debug!("Starting coordinate search: {}", search_id);
        debug!("Search precision target: {}", params.precision_target);
        debug!("Search algorithm: {:?}", params.algorithm);
        
        // Check cache first
        if let Some(cached_result) = self.check_cache(&params).await {
            debug!("Cache hit for search: {}", search_id);
            return Ok(cached_result.result);
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.status = SearchStatus::Searching {
                active_count: state.active_searches.len() + 1,
                start_time,
            };
            
            state.active_searches.insert(search_id.clone(), ActiveSearch {
                search_id: search_id.clone(),
                search_params: params.clone(),
                start_time,
                progress: 0.0,
                intermediate_results: Vec::new(),
            });
        }
        
        // Perform search based on algorithm
        let result = match params.algorithm {
            SearchAlgorithm::Quantum => self.quantum_search(&search_id, &params).await,
            SearchAlgorithm::OscillationConvergence => self.oscillation_convergence_search(&search_id, &params).await,
            SearchAlgorithm::Hierarchical => self.hierarchical_search(&search_id, &params).await,
            SearchAlgorithm::Adaptive => self.adaptive_search(&search_id, &params).await,
            SearchAlgorithm::Hybrid(algorithms) => self.hybrid_search(&search_id, &params, algorithms).await,
        };
        
        let end_time = SystemTime::now();
        let search_time = end_time.duration_since(start_time).unwrap_or(Duration::from_secs(0));
        
        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_searches += 1;
            
            match &result {
                Ok(search_result) => {
                    stats.successful_searches += 1;
                    
                    // Update best precision
                    if stats.best_precision.is_none() || search_result.coordinate.precision_seconds() < stats.best_precision.unwrap() {
                        stats.best_precision = Some(search_result.coordinate.precision_seconds());
                    }
                    
                    // Update algorithm performance
                    let performance = stats.algorithm_performance.entry(params.algorithm.clone()).or_insert(AlgorithmPerformance {
                        uses: 0,
                        success_rate: 0.0,
                        avg_time: Duration::from_millis(0),
                        avg_precision: 0.0,
                    });
                    
                    performance.uses += 1;
                    performance.success_rate = (performance.success_rate * (performance.uses - 1) as f64 + 1.0) / performance.uses as f64;
                    performance.avg_time = Duration::from_nanos(
                        (performance.avg_time.as_nanos() as f64 * 0.9 + search_time.as_nanos() as f64 * 0.1) as u64
                    );
                    performance.avg_precision = performance.avg_precision * 0.9 + search_result.coordinate.precision_seconds() * 0.1;
                }
                Err(_) => {
                    stats.failed_searches += 1;
                }
            }
            
            stats.avg_search_time = Duration::from_nanos(
                (stats.avg_search_time.as_nanos() as f64 * 0.9 + search_time.as_nanos() as f64 * 0.1) as u64
            );
        }
        
        // Clean up active search
        {
            let mut state = self.state.write().await;
            state.active_searches.remove(&search_id);
            state.last_search = Some(end_time);
            
            if state.active_searches.is_empty() {
                state.status = SearchStatus::Ready;
            }
        }
        
        // Cache successful result
        if let Ok(search_result) = &result {
            self.cache_result(&params, search_result, end_time).await;
        }
        
        debug!("Search completed: {} in {:?}", search_id, search_time);
        
        result
    }
    
    /// Extracts a temporal coordinate from convergence and precision data
    pub async fn extract_coordinate(&self, convergence: &OscillationConvergence, precision: &PrecisionMeasurementResult) -> NavigatorResult<TemporalCoordinate> {
        debug!("Extracting temporal coordinate from convergence data");
        
        // Extract spatial coordinates (for now, use convergence point)
        let spatial = SpatialCoordinate::new(0.0, 0.0, 0.0, precision.uncertainty);
        
        // Extract temporal position from convergence
        let temporal = convergence.convergence_point.clone();
        
        // Create oscillatory signature from converged endpoints
        let oscillatory_signature = OscillatorySignature::new(
            convergence.converged_endpoints.iter().filter(|e| matches!(e.level, OscillationLevel::Quantum)).cloned().map(|e| OscillationComponent {
                frequency: e.frequency,
                amplitude: e.amplitude,
                phase: e.phase,
                termination_time: e.termination_time.total_seconds(),
            }).collect(),
            convergence.converged_endpoints.iter().filter(|e| matches!(e.level, OscillationLevel::Molecular)).cloned().map(|e| OscillationComponent {
                frequency: e.frequency,
                amplitude: e.amplitude,
                phase: e.phase,
                termination_time: e.termination_time.total_seconds(),
            }).collect(),
            convergence.converged_endpoints.iter().filter(|e| matches!(e.level, OscillationLevel::Biological)).cloned().map(|e| OscillationComponent {
                frequency: e.frequency,
                amplitude: e.amplitude,
                phase: e.phase,
                termination_time: e.termination_time.total_seconds(),
            }).collect(),
            convergence.converged_endpoints.iter().filter(|e| matches!(e.level, OscillationLevel::Consciousness)).cloned().map(|e| OscillationComponent {
                frequency: e.frequency,
                amplitude: e.amplitude,
                phase: e.phase,
                termination_time: e.termination_time.total_seconds(),
            }).collect(),
            convergence.converged_endpoints.iter().filter(|e| matches!(e.level, OscillationLevel::Environmental)).cloned().map(|e| OscillationComponent {
                frequency: e.frequency,
                amplitude: e.amplitude,
                phase: e.phase,
                termination_time: e.termination_time.total_seconds(),
            }).collect(),
        );
        
        // Create temporal coordinate
        let coordinate = TemporalCoordinate::new(
            spatial,
            temporal,
            oscillatory_signature,
            convergence.convergence_confidence,
        );
        
        debug!("Extracted temporal coordinate with precision: {:.2e} seconds", coordinate.precision_seconds());
        
        Ok(coordinate)
    }
    
    /// Checks if the search engine is healthy
    pub async fn is_healthy(&self) -> bool {
        let state = self.state.read().await;
        matches!(state.status, SearchStatus::Ready | SearchStatus::Searching { .. })
    }
    
    /// Performs quantum search
    async fn quantum_search(&self, search_id: &str, params: &TemporalSearchParams) -> NavigatorResult<TemporalSearchResult> {
        debug!("Performing quantum search: {}", search_id);
        
        // Update progress
        self.update_search_progress(search_id, 0.1).await;
        
        // Initialize quantum search space
        let quantum_space = self.quantum_space.read().await;
        
        // Perform quantum search iterations
        let mut best_coordinate = None;
        let mut best_confidence = 0.0;
        
        for iteration in 0..100 {
            self.update_search_progress(search_id, 0.1 + (iteration as f64 / 100.0) * 0.8).await;
            
            // Quantum amplitude amplification
            let coordinate = self.quantum_amplitude_amplification(&quantum_space, params).await?;
            
            if coordinate.confidence > best_confidence {
                best_confidence = coordinate.confidence;
                best_coordinate = Some(coordinate);
            }
            
            // Check convergence
            if best_confidence > 0.99 {
                break;
            }
        }
        
        self.update_search_progress(search_id, 1.0).await;
        
        Ok(TemporalSearchResult {
            coordinate: best_coordinate.unwrap_or_else(|| TemporalCoordinate::new(
                SpatialCoordinate::new(0.0, 0.0, 0.0, 1e-15),
                params.window.center.clone(),
                OscillatorySignature::new(vec![], vec![], vec![], vec![], vec![]),
                0.5,
            )),
            confidence: best_confidence,
            search_time: Duration::from_millis(500), // Mock search time
            validation_results: vec![],
        })
    }
    
    /// Performs oscillation convergence search
    async fn oscillation_convergence_search(&self, search_id: &str, params: &TemporalSearchParams) -> NavigatorResult<TemporalSearchResult> {
        debug!("Performing oscillation convergence search: {}", search_id);
        
        // Mock implementation - would integrate with actual oscillation analysis
        self.update_search_progress(search_id, 0.5).await;
        
        let coordinate = TemporalCoordinate::new(
            SpatialCoordinate::new(0.0, 0.0, 0.0, 1e-15),
            params.window.center.clone(),
            OscillatorySignature::new(vec![], vec![], vec![], vec![], vec![]),
            0.85,
        );
        
        self.update_search_progress(search_id, 1.0).await;
        
        Ok(TemporalSearchResult {
            coordinate,
            confidence: 0.85,
            search_time: Duration::from_millis(300),
            validation_results: vec![],
        })
    }
    
    /// Performs hierarchical search
    async fn hierarchical_search(&self, search_id: &str, params: &TemporalSearchParams) -> NavigatorResult<TemporalSearchResult> {
        debug!("Performing hierarchical search: {}", search_id);
        
        // Mock implementation
        self.update_search_progress(search_id, 1.0).await;
        
        let coordinate = TemporalCoordinate::new(
            SpatialCoordinate::new(0.0, 0.0, 0.0, 1e-15),
            params.window.center.clone(),
            OscillatorySignature::new(vec![], vec![], vec![], vec![], vec![]),
            0.80,
        );
        
        Ok(TemporalSearchResult {
            coordinate,
            confidence: 0.80,
            search_time: Duration::from_millis(400),
            validation_results: vec![],
        })
    }
    
    /// Performs adaptive search
    async fn adaptive_search(&self, search_id: &str, params: &TemporalSearchParams) -> NavigatorResult<TemporalSearchResult> {
        debug!("Performing adaptive search: {}", search_id);
        
        // Mock implementation
        self.update_search_progress(search_id, 1.0).await;
        
        let coordinate = TemporalCoordinate::new(
            SpatialCoordinate::new(0.0, 0.0, 0.0, 1e-15),
            params.window.center.clone(),
            OscillatorySignature::new(vec![], vec![], vec![], vec![], vec![]),
            0.90,
        );
        
        Ok(TemporalSearchResult {
            coordinate,
            confidence: 0.90,
            search_time: Duration::from_millis(600),
            validation_results: vec![],
        })
    }
    
    /// Performs hybrid search
    async fn hybrid_search(&self, search_id: &str, params: &TemporalSearchParams, algorithms: Vec<SearchAlgorithm>) -> NavigatorResult<TemporalSearchResult> {
        debug!("Performing hybrid search: {} with {} algorithms", search_id, algorithms.len());
        
        let mut best_result = None;
        let mut best_confidence = 0.0;
        
        for (i, algorithm) in algorithms.iter().enumerate() {
            let sub_params = TemporalSearchParams {
                algorithm: algorithm.clone(),
                ..params.clone()
            };
            
            let result = match algorithm {
                SearchAlgorithm::Quantum => self.quantum_search(search_id, &sub_params).await,
                SearchAlgorithm::OscillationConvergence => self.oscillation_convergence_search(search_id, &sub_params).await,
                SearchAlgorithm::Hierarchical => self.hierarchical_search(search_id, &sub_params).await,
                SearchAlgorithm::Adaptive => self.adaptive_search(search_id, &sub_params).await,
                SearchAlgorithm::Hybrid(_) => continue, // Avoid infinite recursion
            };
            
            if let Ok(search_result) = result {
                if search_result.confidence > best_confidence {
                    best_confidence = search_result.confidence;
                    best_result = Some(search_result);
                }
            }
            
            self.update_search_progress(search_id, (i + 1) as f64 / algorithms.len() as f64).await;
        }
        
        best_result.ok_or_else(|| NavigatorError::TemporalNavigation(
            TemporalNavigationError::CoordinateSearchFailed {
                reason: "All hybrid search algorithms failed".to_string(),
                search_time: Duration::from_millis(1000),
                target_precision: params.precision_target,
            }
        ))
    }
    
    /// Performs quantum amplitude amplification
    async fn quantum_amplitude_amplification(&self, quantum_space: &QuantumSearchSpace, params: &TemporalSearchParams) -> NavigatorResult<TemporalCoordinate> {
        // Mock quantum amplitude amplification
        let coordinate = TemporalCoordinate::new(
            SpatialCoordinate::new(0.0, 0.0, 0.0, 1e-15),
            params.window.center.clone(),
            OscillatorySignature::new(vec![], vec![], vec![], vec![], vec![]),
            0.95,
        );
        
        Ok(coordinate)
    }
    
    /// Checks cache for existing search results
    async fn check_cache(&self, params: &TemporalSearchParams) -> Option<CachedSearchResult> {
        let state = self.state.read().await;
        
        // Simple cache key based on parameters
        let cache_key = format!("{:?}", params);
        
        if let Some(cached) = state.search_cache.get(&cache_key) {
            if cached.expiry > SystemTime::now() {
                return Some(cached.clone());
            }
        }
        
        None
    }
    
    /// Caches a search result
    async fn cache_result(&self, params: &TemporalSearchParams, result: &TemporalSearchResult, timestamp: SystemTime) {
        let mut state = self.state.write().await;
        
        let cache_key = format!("{:?}", params);
        let expiry = timestamp + Duration::from_secs(3600); // Cache for 1 hour
        
        state.search_cache.insert(cache_key, CachedSearchResult {
            search_params: params.clone(),
            result: result.clone(),
            timestamp,
            expiry,
            hit_count: 0,
        });
    }
    
    /// Updates search progress
    async fn update_search_progress(&self, search_id: &str, progress: f64) {
        let mut state = self.state.write().await;
        
        if let Some(active_search) = state.active_searches.get_mut(search_id) {
            active_search.progress = progress;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::system_config::SystemConfig;
    
    #[tokio::test]
    async fn test_search_engine_creation() {
        let config = Arc::new(SystemConfig::default());
        let engine = CoordinateSearchEngine::new(config).await;
        
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        assert!(engine.is_healthy().await);
    }
    
    #[tokio::test]
    async fn test_quantum_search() {
        let config = Arc::new(SystemConfig::default());
        let engine = CoordinateSearchEngine::new(config).await.unwrap();
        
        let params = TemporalSearchParams {
            window: TemporalWindow {
                center: TemporalPosition::new(1000.0, 0.0, 1e-30, PrecisionLevel::Target),
                radius: 1.0,
                precision_target: 1e-30,
                max_search_time: Duration::from_secs(60),
            },
            precision_target: 1e-30,
            max_search_time: Duration::from_secs(60),
            algorithm: SearchAlgorithm::Quantum,
            priority: SearchPriority::Normal,
        };
        
        let result = engine.search_coordinate(params).await;
        assert!(result.is_ok());
        
        let search_result = result.unwrap();
        assert!(search_result.confidence > 0.0);
    }
    
    #[tokio::test]
    async fn test_search_cache() {
        let config = Arc::new(SystemConfig::default());
        let engine = CoordinateSearchEngine::new(config).await.unwrap();
        
        let params = TemporalSearchParams {
            window: TemporalWindow {
                center: TemporalPosition::new(1000.0, 0.0, 1e-30, PrecisionLevel::Target),
                radius: 1.0,
                precision_target: 1e-30,
                max_search_time: Duration::from_secs(60),
            },
            precision_target: 1e-30,
            max_search_time: Duration::from_secs(60),
            algorithm: SearchAlgorithm::Quantum,
            priority: SearchPriority::Normal,
        };
        
        // First search
        let result1 = engine.search_coordinate(params.clone()).await;
        assert!(result1.is_ok());
        
        // Second search should use cache
        let result2 = engine.search_coordinate(params).await;
        assert!(result2.is_ok());
    }
}
