/// Hierarchical Analyzer for Multi-Scale Oscillation Analysis
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This module provides hierarchical analysis capabilities for oscillation
/// patterns across quantum to cosmic scales, enabling multi-level temporal
/// coordinate navigation with memorial significance validation.

use std::sync::Arc;
use std::time::SystemTime;
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::types::error_types::NavigatorError;
use crate::types::oscillation_types::{OscillationState, OscillationMetrics};

/// Hierarchical analysis levels for oscillation processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HierarchicalLevel {
    /// Quantum scale (10^-18 to 10^-15 seconds)
    Quantum,
    /// Atomic scale (10^-15 to 10^-12 seconds)
    Atomic,
    /// Molecular scale (10^-12 to 10^-9 seconds)
    Molecular,
    /// Cellular scale (10^-9 to 10^-6 seconds)
    Cellular,
    /// Tissue scale (10^-6 to 10^-3 seconds)
    Tissue,
    /// Organ scale (10^-3 to 1 seconds)
    Organ,
    /// Organism scale (1 to 10^3 seconds)
    Organism,
    /// Ecosystem scale (10^3 to 10^6 seconds)
    Ecosystem,
    /// Planetary scale (10^6 to 10^9 seconds)
    Planetary,
    /// Solar scale (10^9 to 10^12 seconds)
    Solar,
    /// Galactic scale (10^12 to 10^15 seconds)
    Galactic,
    /// Cosmic scale (10^15+ seconds)
    Cosmic,
}

/// Hierarchical analyzer for multi-scale oscillation analysis
/// 
/// This analyzer processes oscillation patterns across multiple hierarchical
/// levels, from quantum to cosmic scales, providing comprehensive temporal
/// coordinate navigation with memorial significance validation.
#[derive(Debug, Clone)]
pub struct HierarchicalAnalyzer {
    /// Hierarchical analysis state
    analysis_state: Arc<RwLock<HierarchicalAnalysisState>>,
    /// Analysis metrics for each level
    level_metrics: Arc<RwLock<HashMap<HierarchicalLevel, HashMap<String, f64>>>>,
    /// Memorial significance threshold
    memorial_threshold: f64,
    /// Active analysis levels
    active_levels: Vec<HierarchicalLevel>,
}

/// Internal hierarchical analysis state
#[derive(Debug, Clone)]
struct HierarchicalAnalysisState {
    /// Current analysis level
    current_level: HierarchicalLevel,
    /// Analysis in progress
    analysis_active: bool,
    /// Cross-level correlations
    cross_level_correlations: HashMap<(HierarchicalLevel, HierarchicalLevel), f64>,
    /// Level-specific oscillation states
    level_states: HashMap<HierarchicalLevel, Vec<OscillationState>>,
    /// Last analysis timestamp
    last_analysis: SystemTime,
    /// Analysis iteration count
    iteration_count: u64,
}

impl HierarchicalAnalyzer {
    /// Create new hierarchical analyzer
    pub fn new() -> Self {
        Self {
            analysis_state: Arc::new(RwLock::new(HierarchicalAnalysisState {
                current_level: HierarchicalLevel::Quantum,
                analysis_active: false,
                cross_level_correlations: HashMap::new(),
                level_states: HashMap::new(),
                last_analysis: SystemTime::now(),
                iteration_count: 0,
            })),
            level_metrics: Arc::new(RwLock::new(HashMap::new())),
            memorial_threshold: 0.85,
            active_levels: vec![
                HierarchicalLevel::Quantum,
                HierarchicalLevel::Atomic,
                HierarchicalLevel::Molecular,
                HierarchicalLevel::Cellular,
                HierarchicalLevel::Tissue,
                HierarchicalLevel::Organ,
                HierarchicalLevel::Organism,
                HierarchicalLevel::Ecosystem,
                HierarchicalLevel::Planetary,
                HierarchicalLevel::Solar,
                HierarchicalLevel::Galactic,
                HierarchicalLevel::Cosmic,
            ],
        }
    }
    
    /// Initialize hierarchical analyzer
    pub async fn initialize(&self) -> Result<(), NavigatorError> {
        // Initialize analysis state
        let mut state = self.analysis_state.write().await;
        state.analysis_active = true;
        state.last_analysis = SystemTime::now();
        state.iteration_count = 0;
        
        // Initialize level states
        for level in &self.active_levels {
            state.level_states.insert(*level, Vec::new());
        }
        
        // Initialize level metrics
        let mut level_metrics = self.level_metrics.write().await;
        for level in &self.active_levels {
            let mut metrics = HashMap::new();
            metrics.insert("oscillation_count".to_string(), 0.0);
            metrics.insert("average_frequency".to_string(), 0.0);
            metrics.insert("average_amplitude".to_string(), 0.0);
            metrics.insert("stability_factor".to_string(), 0.0);
            metrics.insert("memorial_significance".to_string(), 0.0);
            level_metrics.insert(*level, metrics);
        }
        
        Ok(())
    }
    
    /// Perform hierarchical analysis
    pub async fn analyze_hierarchical_patterns(&self, oscillation_state: &OscillationState, metrics: &OscillationMetrics) -> Result<HierarchicalAnalysisResults, NavigatorError> {
        // Update analysis state
        let mut state = self.analysis_state.write().await;
        state.iteration_count += 1;
        state.last_analysis = SystemTime::now();
        drop(state);
        
        // Analyze at each hierarchical level
        let mut level_results = HashMap::new();
        for level in &self.active_levels {
            let level_analysis = self.analyze_at_level(*level, oscillation_state, metrics).await?;
            level_results.insert(*level, level_analysis);
        }
        
        // Calculate cross-level correlations
        let cross_correlations = self.calculate_cross_level_correlations(&level_results).await?;
        
        // Determine dominant hierarchical patterns
        let dominant_patterns = self.identify_dominant_patterns(&level_results).await?;
        
        // Calculate memorial significance across levels
        let memorial_significance = self.calculate_hierarchical_memorial_significance(&level_results).await?;
        
        // Generate comprehensive results
        let results = HierarchicalAnalysisResults {
            level_results,
            cross_correlations,
            dominant_patterns,
            memorial_significance,
            analysis_timestamp: SystemTime::now(),
        };
        
        Ok(results)
    }
    
    /// Analyze oscillation at specific hierarchical level
    async fn analyze_at_level(&self, level: HierarchicalLevel, oscillation_state: &OscillationState, metrics: &OscillationMetrics) -> Result<LevelAnalysisResult, NavigatorError> {
        // Get level-specific parameters
        let level_params = self.get_level_parameters(level);
        
        // Scale metrics to level
        let scaled_metrics = self.scale_metrics_to_level(level, metrics).await?;
        
        // Analyze oscillation patterns at this level
        let pattern_analysis = self.analyze_patterns_at_level(level, oscillation_state, &scaled_metrics).await?;
        
        // Calculate stability at this level
        let stability = self.calculate_level_stability(level, &scaled_metrics).await?;
        
        // Determine memorial significance at this level
        let memorial_significance = self.calculate_level_memorial_significance(level, &scaled_metrics).await?;
        
        // Update level metrics
        self.update_level_metrics(level, &scaled_metrics).await?;
        
        Ok(LevelAnalysisResult {
            level,
            scaled_metrics,
            pattern_analysis,
            stability,
            memorial_significance,
        })
    }
    
    /// Get parameters for specific hierarchical level
    fn get_level_parameters(&self, level: HierarchicalLevel) -> LevelParameters {
        match level {
            HierarchicalLevel::Quantum => LevelParameters {
                time_scale: 1e-18,
                frequency_range: (1e15, 1e18),
                amplitude_scale: 1e-12,
                characteristic_damping: 0.1,
            },
            HierarchicalLevel::Atomic => LevelParameters {
                time_scale: 1e-15,
                frequency_range: (1e12, 1e15),
                amplitude_scale: 1e-9,
                characteristic_damping: 0.05,
            },
            HierarchicalLevel::Molecular => LevelParameters {
                time_scale: 1e-12,
                frequency_range: (1e9, 1e12),
                amplitude_scale: 1e-6,
                characteristic_damping: 0.02,
            },
            HierarchicalLevel::Cellular => LevelParameters {
                time_scale: 1e-9,
                frequency_range: (1e6, 1e9),
                amplitude_scale: 1e-3,
                characteristic_damping: 0.01,
            },
            HierarchicalLevel::Tissue => LevelParameters {
                time_scale: 1e-6,
                frequency_range: (1e3, 1e6),
                amplitude_scale: 1.0,
                characteristic_damping: 0.005,
            },
            HierarchicalLevel::Organ => LevelParameters {
                time_scale: 1e-3,
                frequency_range: (1.0, 1e3),
                amplitude_scale: 1e3,
                characteristic_damping: 0.002,
            },
            HierarchicalLevel::Organism => LevelParameters {
                time_scale: 1.0,
                frequency_range: (1e-3, 1.0),
                amplitude_scale: 1e6,
                characteristic_damping: 0.001,
            },
            HierarchicalLevel::Ecosystem => LevelParameters {
                time_scale: 1e3,
                frequency_range: (1e-6, 1e-3),
                amplitude_scale: 1e9,
                characteristic_damping: 0.0005,
            },
            HierarchicalLevel::Planetary => LevelParameters {
                time_scale: 1e6,
                frequency_range: (1e-9, 1e-6),
                amplitude_scale: 1e12,
                characteristic_damping: 0.0002,
            },
            HierarchicalLevel::Solar => LevelParameters {
                time_scale: 1e9,
                frequency_range: (1e-12, 1e-9),
                amplitude_scale: 1e15,
                characteristic_damping: 0.0001,
            },
            HierarchicalLevel::Galactic => LevelParameters {
                time_scale: 1e12,
                frequency_range: (1e-15, 1e-12),
                amplitude_scale: 1e18,
                characteristic_damping: 0.00005,
            },
            HierarchicalLevel::Cosmic => LevelParameters {
                time_scale: 1e15,
                frequency_range: (1e-18, 1e-15),
                amplitude_scale: 1e21,
                characteristic_damping: 0.00001,
            },
        }
    }
    
    /// Scale metrics to specific hierarchical level
    async fn scale_metrics_to_level(&self, level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<OscillationMetrics, NavigatorError> {
        let params = self.get_level_parameters(level);
        
        let scaled_metrics = OscillationMetrics {
            frequency: metrics.frequency * params.time_scale,
            amplitude: metrics.amplitude * params.amplitude_scale,
            phase: metrics.phase,
            damping: metrics.damping * params.characteristic_damping,
            memorial_significance: metrics.memorial_significance,
        };
        
        Ok(scaled_metrics)
    }
    
    /// Analyze patterns at specific level
    async fn analyze_patterns_at_level(&self, level: HierarchicalLevel, oscillation_state: &OscillationState, metrics: &OscillationMetrics) -> Result<PatternAnalysis, NavigatorError> {
        // Analyze frequency patterns
        let frequency_patterns = self.analyze_frequency_patterns(level, metrics).await?;
        
        // Analyze amplitude patterns
        let amplitude_patterns = self.analyze_amplitude_patterns(level, metrics).await?;
        
        // Analyze phase patterns
        let phase_patterns = self.analyze_phase_patterns(level, metrics).await?;
        
        // Analyze damping patterns
        let damping_patterns = self.analyze_damping_patterns(level, metrics).await?;
        
        Ok(PatternAnalysis {
            frequency_patterns,
            amplitude_patterns,
            phase_patterns,
            damping_patterns,
        })
    }
    
    /// Calculate stability at specific level
    async fn calculate_level_stability(&self, level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<f64, NavigatorError> {
        let params = self.get_level_parameters(level);
        
        // Calculate stability based on level-specific parameters
        let frequency_stability = if metrics.frequency >= params.frequency_range.0 && metrics.frequency <= params.frequency_range.1 {
            0.9
        } else {
            0.3
        };
        
        let amplitude_stability = if metrics.amplitude > 0.0 && metrics.amplitude < params.amplitude_scale * 2.0 {
            0.8
        } else {
            0.2
        };
        
        let damping_stability = if metrics.damping > 0.0 && metrics.damping < params.characteristic_damping * 2.0 {
            0.85
        } else {
            0.25
        };
        
        let stability = (frequency_stability + amplitude_stability + damping_stability) / 3.0;
        
        Ok(stability)
    }
    
    /// Calculate memorial significance at specific level
    async fn calculate_level_memorial_significance(&self, level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<f64, NavigatorError> {
        // Base memorial significance from metrics
        let base_significance = metrics.memorial_significance;
        
        // Level-specific memorial weighting
        let level_weighting = match level {
            HierarchicalLevel::Quantum => 0.95,
            HierarchicalLevel::Atomic => 0.92,
            HierarchicalLevel::Molecular => 0.89,
            HierarchicalLevel::Cellular => 0.86,
            HierarchicalLevel::Tissue => 0.83,
            HierarchicalLevel::Organ => 0.80,
            HierarchicalLevel::Organism => 0.85,
            HierarchicalLevel::Ecosystem => 0.88,
            HierarchicalLevel::Planetary => 0.91,
            HierarchicalLevel::Solar => 0.94,
            HierarchicalLevel::Galactic => 0.97,
            HierarchicalLevel::Cosmic => 1.0,
        };
        
        let level_significance = base_significance * level_weighting;
        
        Ok(level_significance)
    }
    
    /// Update metrics for specific level
    async fn update_level_metrics(&self, level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<(), NavigatorError> {
        let mut level_metrics = self.level_metrics.write().await;
        
        if let Some(level_metric_map) = level_metrics.get_mut(&level) {
            level_metric_map.insert("average_frequency".to_string(), metrics.frequency);
            level_metric_map.insert("average_amplitude".to_string(), metrics.amplitude);
            level_metric_map.insert("memorial_significance".to_string(), metrics.memorial_significance);
            
            // Update oscillation count
            let current_count = level_metric_map.get("oscillation_count").copied().unwrap_or(0.0);
            level_metric_map.insert("oscillation_count".to_string(), current_count + 1.0);
        }
        
        Ok(())
    }
    
    /// Calculate cross-level correlations
    async fn calculate_cross_level_correlations(&self, level_results: &HashMap<HierarchicalLevel, LevelAnalysisResult>) -> Result<HashMap<(HierarchicalLevel, HierarchicalLevel), f64>, NavigatorError> {
        let mut correlations = HashMap::new();
        
        let levels: Vec<HierarchicalLevel> = level_results.keys().cloned().collect();
        
        for i in 0..levels.len() {
            for j in (i + 1)..levels.len() {
                let level1 = levels[i];
                let level2 = levels[j];
                
                let correlation = self.calculate_correlation_between_levels(
                    &level_results[&level1],
                    &level_results[&level2]
                ).await?;
                
                correlations.insert((level1, level2), correlation);
            }
        }
        
        Ok(correlations)
    }
    
    /// Calculate correlation between two levels
    async fn calculate_correlation_between_levels(&self, result1: &LevelAnalysisResult, result2: &LevelAnalysisResult) -> Result<f64, NavigatorError> {
        // Calculate correlation based on scaled metrics
        let freq_correlation = self.calculate_metric_correlation(
            result1.scaled_metrics.frequency,
            result2.scaled_metrics.frequency
        );
        
        let amp_correlation = self.calculate_metric_correlation(
            result1.scaled_metrics.amplitude,
            result2.scaled_metrics.amplitude
        );
        
        let phase_correlation = self.calculate_metric_correlation(
            result1.scaled_metrics.phase,
            result2.scaled_metrics.phase
        );
        
        let memorial_correlation = self.calculate_metric_correlation(
            result1.memorial_significance,
            result2.memorial_significance
        );
        
        let overall_correlation = (freq_correlation + amp_correlation + phase_correlation + memorial_correlation) / 4.0;
        
        Ok(overall_correlation)
    }
    
    /// Calculate correlation between two metric values
    fn calculate_metric_correlation(&self, value1: f64, value2: f64) -> f64 {
        // Simple correlation calculation
        let max_val = value1.max(value2);
        let min_val = value1.min(value2);
        
        if max_val == 0.0 {
            1.0
        } else {
            min_val / max_val
        }
    }
    
    /// Identify dominant patterns across levels
    async fn identify_dominant_patterns(&self, level_results: &HashMap<HierarchicalLevel, LevelAnalysisResult>) -> Result<Vec<(HierarchicalLevel, f64)>, NavigatorError> {
        let mut patterns = Vec::new();
        
        for (level, result) in level_results {
            let dominance_score = result.stability * result.memorial_significance;
            patterns.push((*level, dominance_score));
        }
        
        // Sort by dominance score
        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(patterns)
    }
    
    /// Calculate hierarchical memorial significance
    async fn calculate_hierarchical_memorial_significance(&self, level_results: &HashMap<HierarchicalLevel, LevelAnalysisResult>) -> Result<f64, NavigatorError> {
        let mut total_significance = 0.0;
        let mut weight_sum = 0.0;
        
        for (level, result) in level_results {
            let level_weight = self.get_level_weight(*level);
            total_significance += result.memorial_significance * level_weight;
            weight_sum += level_weight;
        }
        
        let overall_significance = if weight_sum > 0.0 {
            total_significance / weight_sum
        } else {
            0.0
        };
        
        Ok(overall_significance)
    }
    
    /// Get weight for specific level
    fn get_level_weight(&self, level: HierarchicalLevel) -> f64 {
        match level {
            HierarchicalLevel::Quantum => 0.95,
            HierarchicalLevel::Atomic => 0.90,
            HierarchicalLevel::Molecular => 0.85,
            HierarchicalLevel::Cellular => 0.80,
            HierarchicalLevel::Tissue => 0.75,
            HierarchicalLevel::Organ => 0.70,
            HierarchicalLevel::Organism => 0.80,
            HierarchicalLevel::Ecosystem => 0.85,
            HierarchicalLevel::Planetary => 0.90,
            HierarchicalLevel::Solar => 0.95,
            HierarchicalLevel::Galactic => 0.98,
            HierarchicalLevel::Cosmic => 1.0,
        }
    }
    
    /// Analyze frequency patterns
    async fn analyze_frequency_patterns(&self, _level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<Vec<String>, NavigatorError> {
        let mut patterns = Vec::new();
        
        if metrics.frequency > 1000.0 {
            patterns.push("High frequency oscillation".to_string());
        } else if metrics.frequency > 100.0 {
            patterns.push("Medium frequency oscillation".to_string());
        } else {
            patterns.push("Low frequency oscillation".to_string());
        }
        
        Ok(patterns)
    }
    
    /// Analyze amplitude patterns
    async fn analyze_amplitude_patterns(&self, _level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<Vec<String>, NavigatorError> {
        let mut patterns = Vec::new();
        
        if metrics.amplitude > 0.8 {
            patterns.push("High amplitude oscillation".to_string());
        } else if metrics.amplitude > 0.3 {
            patterns.push("Medium amplitude oscillation".to_string());
        } else {
            patterns.push("Low amplitude oscillation".to_string());
        }
        
        Ok(patterns)
    }
    
    /// Analyze phase patterns
    async fn analyze_phase_patterns(&self, _level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<Vec<String>, NavigatorError> {
        let mut patterns = Vec::new();
        
        if metrics.phase.abs() < 0.1 {
            patterns.push("Near-zero phase".to_string());
        } else if metrics.phase > 0.0 {
            patterns.push("Positive phase shift".to_string());
        } else {
            patterns.push("Negative phase shift".to_string());
        }
        
        Ok(patterns)
    }
    
    /// Analyze damping patterns
    async fn analyze_damping_patterns(&self, _level: HierarchicalLevel, metrics: &OscillationMetrics) -> Result<Vec<String>, NavigatorError> {
        let mut patterns = Vec::new();
        
        if metrics.damping > 0.5 {
            patterns.push("High damping".to_string());
        } else if metrics.damping > 0.1 {
            patterns.push("Medium damping".to_string());
        } else {
            patterns.push("Low damping".to_string());
        }
        
        Ok(patterns)
    }
    
    /// Get hierarchical analysis metrics
    pub async fn get_hierarchical_metrics(&self) -> HashMap<HierarchicalLevel, HashMap<String, f64>> {
        self.level_metrics.read().await.clone()
    }
    
    /// Get analysis state
    pub async fn get_analysis_state(&self) -> HierarchicalAnalysisState {
        self.analysis_state.read().await.clone()
    }
    
    /// Shutdown hierarchical analyzer
    pub async fn shutdown(&self) -> Result<(), NavigatorError> {
        let mut state = self.analysis_state.write().await;
        state.analysis_active = false;
        state.level_states.clear();
        state.cross_level_correlations.clear();
        
        Ok(())
    }
}

/// Level parameters for hierarchical analysis
#[derive(Debug, Clone)]
struct LevelParameters {
    /// Characteristic time scale for this level
    time_scale: f64,
    /// Frequency range for this level
    frequency_range: (f64, f64),
    /// Amplitude scale for this level
    amplitude_scale: f64,
    /// Characteristic damping for this level
    characteristic_damping: f64,
}

/// Results from hierarchical analysis
#[derive(Debug, Clone)]
pub struct HierarchicalAnalysisResults {
    /// Results for each level
    pub level_results: HashMap<HierarchicalLevel, LevelAnalysisResult>,
    /// Cross-level correlations
    pub cross_correlations: HashMap<(HierarchicalLevel, HierarchicalLevel), f64>,
    /// Dominant patterns across levels
    pub dominant_patterns: Vec<(HierarchicalLevel, f64)>,
    /// Overall memorial significance
    pub memorial_significance: f64,
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
}

/// Results from analysis at specific level
#[derive(Debug, Clone)]
pub struct LevelAnalysisResult {
    /// Hierarchical level
    pub level: HierarchicalLevel,
    /// Scaled metrics for this level
    pub scaled_metrics: OscillationMetrics,
    /// Pattern analysis results
    pub pattern_analysis: PatternAnalysis,
    /// Stability at this level
    pub stability: f64,
    /// Memorial significance at this level
    pub memorial_significance: f64,
}

/// Pattern analysis results
#[derive(Debug, Clone)]
pub struct PatternAnalysis {
    /// Frequency patterns
    pub frequency_patterns: Vec<String>,
    /// Amplitude patterns
    pub amplitude_patterns: Vec<String>,
    /// Phase patterns
    pub phase_patterns: Vec<String>,
    /// Damping patterns
    pub damping_patterns: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hierarchical_analyzer_creation() {
        let analyzer = HierarchicalAnalyzer::new();
        let state = analyzer.get_analysis_state().await;
        assert_eq!(state.current_level, HierarchicalLevel::Quantum);
        assert!(!state.analysis_active);
    }
    
    #[tokio::test]
    async fn test_hierarchical_analyzer_initialization() {
        let analyzer = HierarchicalAnalyzer::new();
        analyzer.initialize().await.unwrap();
        
        let state = analyzer.get_analysis_state().await;
        assert!(state.analysis_active);
        assert_eq!(state.iteration_count, 0);
    }
    
    #[tokio::test]
    async fn test_level_parameters() {
        let analyzer = HierarchicalAnalyzer::new();
        let params = analyzer.get_level_parameters(HierarchicalLevel::Quantum);
        assert_eq!(params.time_scale, 1e-18);
        assert_eq!(params.frequency_range.0, 1e15);
        assert_eq!(params.frequency_range.1, 1e18);
    }
    
    #[tokio::test]
    async fn test_hierarchical_analysis() {
        let analyzer = HierarchicalAnalyzer::new();
        analyzer.initialize().await.unwrap();
        
        let oscillation_state = OscillationState::Stable;
        let metrics = OscillationMetrics {
            frequency: 440.0,
            amplitude: 0.5,
            phase: 0.0,
            damping: 0.1,
            memorial_significance: 0.9,
        };
        
        let results = analyzer.analyze_hierarchical_patterns(&oscillation_state, &metrics).await.unwrap();
        assert!(!results.level_results.is_empty());
        assert!(results.memorial_significance > 0.0);
    }
}
