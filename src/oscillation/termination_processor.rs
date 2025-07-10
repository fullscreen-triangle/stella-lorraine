/// Termination Processor for Oscillation Control
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This module provides termination processing capabilities for oscillation
/// control, determining when oscillation processes should terminate based on
/// convergence criteria, time limits, and memorial significance thresholds.

use std::sync::Arc;
use std::time::{SystemTime, Duration};
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::types::error_types::NavigatorError;
use crate::types::oscillation_types::{OscillationState, OscillationMetrics};
use crate::oscillation::convergence_detector::ConvergenceDetector;

/// Termination conditions for oscillation processes
#[derive(Debug, Clone)]
pub enum TerminationCondition {
    /// Convergence detected
    ConvergenceDetected,
    /// Time limit exceeded
    TimeLimit,
    /// Memorial significance threshold reached
    MemorialThreshold,
    /// Maximum iterations reached
    MaxIterations,
    /// Error condition encountered
    ErrorCondition,
    /// Manual termination requested
    ManualTermination,
}

/// Termination processor for oscillation control
/// 
/// This processor monitors oscillation processes and determines when
/// they should terminate based on various criteria including convergence,
/// time limits, memorial significance, and error conditions.
#[derive(Debug, Clone)]
pub struct TerminationProcessor {
    /// Convergence detector for monitoring oscillation convergence
    convergence_detector: Arc<ConvergenceDetector>,
    /// Termination state tracking
    termination_state: Arc<RwLock<TerminationState>>,
    /// Termination metrics
    termination_metrics: Arc<RwLock<HashMap<String, f64>>>,
    /// Memorial significance threshold
    memorial_threshold: f64,
    /// Maximum execution time
    max_execution_time: Duration,
    /// Maximum iterations
    max_iterations: u64,
}

/// Internal termination state tracking
#[derive(Debug, Clone)]
struct TerminationState {
    /// Current oscillation being processed
    current_oscillation: Option<String>,
    /// Termination conditions being monitored
    active_conditions: Vec<TerminationCondition>,
    /// Termination processing active
    processing_active: bool,
    /// Start time of current processing
    start_time: SystemTime,
    /// Current iteration count
    iteration_count: u64,
    /// Last termination check
    last_check: SystemTime,
}

impl TerminationProcessor {
    /// Create new termination processor
    pub fn new(convergence_detector: Arc<ConvergenceDetector>) -> Self {
        Self {
            convergence_detector,
            termination_state: Arc::new(RwLock::new(TerminationState {
                current_oscillation: None,
                active_conditions: Vec::new(),
                processing_active: false,
                start_time: SystemTime::now(),
                iteration_count: 0,
                last_check: SystemTime::now(),
            })),
            termination_metrics: Arc::new(RwLock::new(HashMap::new())),
            memorial_threshold: 0.85,
            max_execution_time: Duration::from_secs(300), // 5 minutes
            max_iterations: 10000,
        }
    }
    
    /// Initialize termination processor
    pub async fn initialize(&self) -> Result<(), NavigatorError> {
        // Initialize convergence detector
        self.convergence_detector.initialize().await?;
        
        // Set initial state
        let mut state = self.termination_state.write().await;
        state.processing_active = true;
        state.start_time = SystemTime::now();
        state.iteration_count = 0;
        state.last_check = SystemTime::now();
        
        // Initialize default termination conditions
        state.active_conditions = vec![
            TerminationCondition::ConvergenceDetected,
            TerminationCondition::TimeLimit,
            TerminationCondition::MemorialThreshold,
            TerminationCondition::MaxIterations,
            TerminationCondition::ErrorCondition,
        ];
        
        // Initialize metrics
        let mut metrics = self.termination_metrics.write().await;
        metrics.insert("termination_readiness".to_string(), 0.0);
        metrics.insert("convergence_probability".to_string(), 0.0);
        metrics.insert("memorial_significance".to_string(), 0.0);
        
        Ok(())
    }
    
    /// Start termination processing for oscillation
    pub async fn start_processing(&self, oscillation_id: &str) -> Result<(), NavigatorError> {
        let mut state = self.termination_state.write().await;
        state.current_oscillation = Some(oscillation_id.to_string());
        state.processing_active = true;
        state.start_time = SystemTime::now();
        state.iteration_count = 0;
        state.last_check = SystemTime::now();
        
        Ok(())
    }
    
    /// Check if termination conditions are met
    pub async fn check_termination_conditions(&self, oscillation_state: &OscillationState, metrics: &OscillationMetrics) -> Result<Option<TerminationCondition>, NavigatorError> {
        // Update iteration count
        let mut state = self.termination_state.write().await;
        state.iteration_count += 1;
        state.last_check = SystemTime::now();
        drop(state);
        
        // Check each termination condition
        for condition in &self.get_active_conditions().await {
            if self.evaluate_condition(condition, oscillation_state, metrics).await? {
                return Ok(Some(condition.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Evaluate specific termination condition
    async fn evaluate_condition(&self, condition: &TerminationCondition, oscillation_state: &OscillationState, metrics: &OscillationMetrics) -> Result<bool, NavigatorError> {
        match condition {
            TerminationCondition::ConvergenceDetected => {
                self.check_convergence_condition(oscillation_state, metrics).await
            },
            TerminationCondition::TimeLimit => {
                self.check_time_limit_condition().await
            },
            TerminationCondition::MemorialThreshold => {
                self.check_memorial_threshold_condition(metrics).await
            },
            TerminationCondition::MaxIterations => {
                self.check_max_iterations_condition().await
            },
            TerminationCondition::ErrorCondition => {
                self.check_error_condition(oscillation_state).await
            },
            TerminationCondition::ManualTermination => {
                self.check_manual_termination_condition().await
            },
        }
    }
    
    /// Check convergence condition
    async fn check_convergence_condition(&self, oscillation_state: &OscillationState, metrics: &OscillationMetrics) -> Result<bool, NavigatorError> {
        // Check if convergence is detected
        let convergence_detected = self.convergence_detector.detect_convergence(oscillation_state, metrics).await?;
        
        // Update metrics
        let mut termination_metrics = self.termination_metrics.write().await;
        termination_metrics.insert("convergence_detected".to_string(), if convergence_detected { 1.0 } else { 0.0 });
        
        Ok(convergence_detected)
    }
    
    /// Check time limit condition
    async fn check_time_limit_condition(&self) -> Result<bool, NavigatorError> {
        let state = self.termination_state.read().await;
        let elapsed = state.start_time.elapsed().unwrap_or(Duration::from_secs(0));
        
        let time_limit_exceeded = elapsed >= self.max_execution_time;
        
        // Update metrics
        let mut termination_metrics = self.termination_metrics.write().await;
        termination_metrics.insert("time_limit_exceeded".to_string(), if time_limit_exceeded { 1.0 } else { 0.0 });
        termination_metrics.insert("execution_time_ratio".to_string(), elapsed.as_secs_f64() / self.max_execution_time.as_secs_f64());
        
        Ok(time_limit_exceeded)
    }
    
    /// Check memorial threshold condition
    async fn check_memorial_threshold_condition(&self, metrics: &OscillationMetrics) -> Result<bool, NavigatorError> {
        let memorial_significance = metrics.memorial_significance;
        let threshold_reached = memorial_significance >= self.memorial_threshold;
        
        // Update metrics
        let mut termination_metrics = self.termination_metrics.write().await;
        termination_metrics.insert("memorial_threshold_reached".to_string(), if threshold_reached { 1.0 } else { 0.0 });
        termination_metrics.insert("memorial_significance".to_string(), memorial_significance);
        
        Ok(threshold_reached)
    }
    
    /// Check max iterations condition
    async fn check_max_iterations_condition(&self) -> Result<bool, NavigatorError> {
        let state = self.termination_state.read().await;
        let max_iterations_reached = state.iteration_count >= self.max_iterations;
        
        // Update metrics
        let mut termination_metrics = self.termination_metrics.write().await;
        termination_metrics.insert("max_iterations_reached".to_string(), if max_iterations_reached { 1.0 } else { 0.0 });
        termination_metrics.insert("iteration_ratio".to_string(), state.iteration_count as f64 / self.max_iterations as f64);
        
        Ok(max_iterations_reached)
    }
    
    /// Check error condition
    async fn check_error_condition(&self, oscillation_state: &OscillationState) -> Result<bool, NavigatorError> {
        // Check for error conditions in oscillation state
        let has_error = match oscillation_state {
            OscillationState::Error(_) => true,
            OscillationState::Unstable => true,
            _ => false,
        };
        
        // Update metrics
        let mut termination_metrics = self.termination_metrics.write().await;
        termination_metrics.insert("error_condition_detected".to_string(), if has_error { 1.0 } else { 0.0 });
        
        Ok(has_error)
    }
    
    /// Check manual termination condition
    async fn check_manual_termination_condition(&self) -> Result<bool, NavigatorError> {
        // Check if manual termination has been requested
        let state = self.termination_state.read().await;
        let manual_termination = !state.processing_active;
        
        Ok(manual_termination)
    }
    
    /// Get active termination conditions
    async fn get_active_conditions(&self) -> Vec<TerminationCondition> {
        let state = self.termination_state.read().await;
        state.active_conditions.clone()
    }
    
    /// Process termination
    pub async fn process_termination(&self, condition: &TerminationCondition) -> Result<(), NavigatorError> {
        // Update state
        let mut state = self.termination_state.write().await;
        state.processing_active = false;
        
        // Log termination reason
        let termination_reason = match condition {
            TerminationCondition::ConvergenceDetected => "Convergence detected",
            TerminationCondition::TimeLimit => "Time limit exceeded",
            TerminationCondition::MemorialThreshold => "Memorial threshold reached",
            TerminationCondition::MaxIterations => "Maximum iterations reached",
            TerminationCondition::ErrorCondition => "Error condition encountered",
            TerminationCondition::ManualTermination => "Manual termination requested",
        };
        
        // Update metrics
        let mut metrics = self.termination_metrics.write().await;
        metrics.insert("termination_processed".to_string(), 1.0);
        metrics.insert("termination_reason_code".to_string(), self.get_condition_code(condition));
        
        // Perform cleanup
        self.cleanup_termination().await?;
        
        Ok(())
    }
    
    /// Get condition code for metrics
    fn get_condition_code(&self, condition: &TerminationCondition) -> f64 {
        match condition {
            TerminationCondition::ConvergenceDetected => 1.0,
            TerminationCondition::TimeLimit => 2.0,
            TerminationCondition::MemorialThreshold => 3.0,
            TerminationCondition::MaxIterations => 4.0,
            TerminationCondition::ErrorCondition => 5.0,
            TerminationCondition::ManualTermination => 6.0,
        }
    }
    
    /// Cleanup termination processing
    async fn cleanup_termination(&self) -> Result<(), NavigatorError> {
        // Reset state
        let mut state = self.termination_state.write().await;
        state.current_oscillation = None;
        state.active_conditions.clear();
        state.iteration_count = 0;
        
        Ok(())
    }
    
    /// Request manual termination
    pub async fn request_manual_termination(&self) -> Result<(), NavigatorError> {
        let mut state = self.termination_state.write().await;
        state.processing_active = false;
        
        Ok(())
    }
    
    /// Get termination metrics
    pub async fn get_termination_metrics(&self) -> HashMap<String, f64> {
        self.termination_metrics.read().await.clone()
    }
    
    /// Get current termination state
    pub async fn get_termination_state(&self) -> TerminationState {
        self.termination_state.read().await.clone()
    }
    
    /// Update termination parameters
    pub async fn update_parameters(&self, memorial_threshold: f64, max_execution_time: Duration, max_iterations: u64) -> Result<(), NavigatorError> {
        self.memorial_threshold = memorial_threshold;
        self.max_execution_time = max_execution_time;
        self.max_iterations = max_iterations;
        
        Ok(())
    }
    
    /// Shutdown termination processor
    pub async fn shutdown(&self) -> Result<(), NavigatorError> {
        // Shutdown convergence detector
        self.convergence_detector.shutdown().await?;
        
        // Reset state
        let mut state = self.termination_state.write().await;
        state.processing_active = false;
        state.current_oscillation = None;
        state.active_conditions.clear();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::oscillation_types::{OscillationState, OscillationMetrics};
    
    #[tokio::test]
    async fn test_termination_processor_creation() {
        let convergence_detector = Arc::new(ConvergenceDetector::new(100));
        let processor = TerminationProcessor::new(convergence_detector);
        
        let state = processor.get_termination_state().await;
        assert_eq!(state.iteration_count, 0);
        assert!(!state.processing_active);
    }
    
    #[tokio::test]
    async fn test_termination_condition_evaluation() {
        let convergence_detector = Arc::new(ConvergenceDetector::new(100));
        let processor = TerminationProcessor::new(convergence_detector);
        
        processor.initialize().await.unwrap();
        
        let oscillation_state = OscillationState::Stable;
        let metrics = OscillationMetrics {
            frequency: 440.0,
            amplitude: 0.5,
            phase: 0.0,
            damping: 0.1,
            memorial_significance: 0.9,
        };
        
        let condition = processor.check_termination_conditions(&oscillation_state, &metrics).await.unwrap();
        assert!(condition.is_some());
    }
    
    #[tokio::test]
    async fn test_manual_termination() {
        let convergence_detector = Arc::new(ConvergenceDetector::new(100));
        let processor = TerminationProcessor::new(convergence_detector);
        
        processor.initialize().await.unwrap();
        processor.request_manual_termination().await.unwrap();
        
        let state = processor.get_termination_state().await;
        assert!(!state.processing_active);
    }
    
    #[tokio::test]
    async fn test_termination_metrics() {
        let convergence_detector = Arc::new(ConvergenceDetector::new(100));
        let processor = TerminationProcessor::new(convergence_detector);
        
        processor.initialize().await.unwrap();
        
        let metrics = processor.get_termination_metrics().await;
        assert!(metrics.contains_key("termination_readiness"));
        assert!(metrics.contains_key("convergence_probability"));
        assert!(metrics.contains_key("memorial_significance"));
    }
}
