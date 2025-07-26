//! Core types for the Masunda Temporal Coordinate Navigator
//!
//! This module defines all fundamental data structures used throughout the system
//! for temporal navigation, S-entropy integration, and window combination advisory.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// S-constant representing observer-process separation distance
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SConstant {
    /// Knowledge distance component
    pub s_knowledge: f64,
    /// Time distance component
    pub s_time: f64,
    /// Entropy distance component
    pub s_entropy: f64,
}

impl SConstant {
    /// Create a new S-constant with tri-dimensional values
    pub fn new(s_knowledge: f64, s_time: f64, s_entropy: f64) -> Self {
        Self {
            s_knowledge,
            s_time,
            s_entropy,
        }
    }

    /// Perfect integration (Mrs. Masunda memorial value)
    pub fn perfect() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Calculate total S-distance
    pub fn total_distance(&self) -> f64 {
        (self.s_knowledge.powi(2) + self.s_time.powi(2) + self.s_entropy.powi(2)).sqrt()
    }

    /// Check if alignment is achieved (within tolerance)
    pub fn is_aligned(&self, tolerance: f64) -> bool {
        self.total_distance() <= tolerance
    }

    /// Calculate improvement from another S-constant
    pub fn improvement_from(&self, other: &SConstant) -> f64 {
        other.total_distance() - self.total_distance()
    }
}

impl Default for SConstant {
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0) // Maximum separation by default
    }
}

impl std::ops::Add for SConstant {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(
            self.s_knowledge + other.s_knowledge,
            self.s_time + other.s_time,
            self.s_entropy + other.s_entropy,
        )
    }
}

impl std::ops::Sub for SConstant {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(
            self.s_knowledge - other.s_knowledge,
            self.s_time - other.s_time,
            self.s_entropy - other.s_entropy,
        )
    }
}

/// Temporal coordinate in 4D spacetime
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalCoordinate {
    /// Spatial X coordinate
    pub x: f64,
    /// Spatial Y coordinate
    pub y: f64,
    /// Spatial Z coordinate
    pub z: f64,
    /// Temporal coordinate (seconds since epoch)
    pub t: f64,
    /// Precision achieved (seconds)
    pub precision: f64,
}

impl TemporalCoordinate {
    /// Create a new temporal coordinate
    pub fn new(x: f64, y: f64, z: f64, t: f64, precision: f64) -> Self {
        Self { x, y, z, t, precision }
    }

    /// Calculate distance to another temporal coordinate
    pub fn distance_to(&self, other: &TemporalCoordinate) -> f64 {
        let spatial_distance = ((self.x - other.x).powi(2)
            + (self.y - other.y).powi(2)
            + (self.z - other.z).powi(2)).sqrt();
        let temporal_distance = (self.t - other.t).abs();

        // Combine spatial and temporal distances with time weighting
        (spatial_distance.powi(2) + temporal_distance.powi(2)).sqrt()
    }

    /// Check if coordinate is within precision bounds of target
    pub fn is_within_precision_of(&self, target: &TemporalCoordinate) -> bool {
        self.distance_to(target) <= self.precision.min(target.precision)
    }
}

/// Problem description for Time Domain Service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemDescription {
    /// Unique problem identifier
    pub id: Uuid,
    /// Human-readable problem description
    pub description: String,
    /// Problem domain/category
    pub domain: String,
    /// Problem complexity estimate
    pub complexity: f64,
    /// Required solution precision
    pub precision_requirement: f64,
    /// Timestamp when problem was created
    pub created_at: DateTime<Utc>,
}

impl ProblemDescription {
    /// Create a new problem description
    pub fn new(description: String, domain: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            description,
            domain,
            complexity: 1.0,
            precision_requirement: 1e-9, // Default precision
            created_at: Utc::now(),
        }
    }

    /// Check if problem requires impossible solutions
    pub fn requires_impossible_solutions(&self) -> bool {
        self.complexity > 10.0 || self.precision_requirement < 1e-15
    }
}

impl From<&str> for ProblemDescription {
    fn from(description: &str) -> Self {
        Self::new(description.to_string(), "general".to_string())
    }
}

impl From<String> for ProblemDescription {
    fn from(description: String) -> Self {
        Self::new(description, "general".to_string())
    }
}

/// Time Domain Service requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDomainRequirement {
    /// Target S-distance for alignment
    pub s_target: f64,
    /// Maximum time budget for solution
    pub time_budget: Duration,
    /// Minimum solution truthfulness required
    pub truthfulness_minimum: f64,
    /// Precision target
    pub precision_target: f64,
}

impl Default for TimeDomainRequirement {
    fn default() -> Self {
        Self {
            s_target: 0.1,
            time_budget: Duration::from_secs(30),
            truthfulness_minimum: 0.8,
            precision_target: 1e-9,
        }
    }
}

/// S-time formatted solution unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STimeUnit {
    /// Unit description
    pub description: String,
    /// S-knowledge distance for this unit
    pub s_knowledge_distance: f64,
    /// S-time distance for this unit
    pub s_time_distance: f64,
    /// Truthfulness level of this unit
    pub truthfulness_level: f64,
    /// Pre-existing solution for this unit
    pub pre_existing_solution: String,
    /// Known processing time for this unit
    pub processing_time: Duration,
    /// Selection criteria metadata
    pub selection_criteria: SolutionSelectionCriteria,
}

/// Criteria for solution selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionSelectionCriteria {
    /// Confidence in solution accuracy
    pub confidence: f64,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Dependencies on other solutions
    pub dependencies: Vec<String>,
    /// Risk assessment
    pub risk_level: f64,
}

/// Resource requirements for solution execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU time required
    pub cpu_time: Duration,
    /// Memory required (bytes)
    pub memory: u64,
    /// Network bandwidth required (bytes/sec)
    pub bandwidth: u64,
    /// Storage required (bytes)
    pub storage: u64,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_time: Duration::from_millis(100),
            memory: 1024 * 1024, // 1 MB
            bandwidth: 1024 * 1024, // 1 MB/s
            storage: 1024 * 1024, // 1 MB
        }
    }
}

/// S-time formatted problem containing atomic solution units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STimeFormattedProblem {
    /// Original problem description
    pub original_problem: ProblemDescription,
    /// Atomic S-time units
    pub s_time_units: Vec<STimeUnit>,
    /// Total S-distance for complete problem
    pub total_s_distance: f64,
    /// Solution selection domain
    pub solution_selection_domain: SolutionSelectionDomain,
}

/// Domain of available solutions for selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionSelectionDomain {
    /// Available solutions with known properties
    pub available_solutions: Vec<STimeSolution>,
    /// Time cost mapping for each solution
    pub time_costs: std::collections::HashMap<String, Duration>,
    /// Reliability mapping for each solution
    pub reliability_map: std::collections::HashMap<String, f64>,
    /// Optimization routes available
    pub optimization_routes: Vec<SOptimizationRoute>,
}

/// Complete S-time solution with known properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STimeSolution {
    /// Solution identifier
    pub id: Uuid,
    /// Solution description
    pub description: String,
    /// S-distance values for this solution
    pub s_distance: SConstant,
    /// Time to execute this solution
    pub time_to_solution: Duration,
    /// Truthfulness level of solution
    pub truthfulness_level: f64,
    /// Total S-cost for this solution
    pub total_s_cost: f64,
    /// Implementation details
    pub implementation: SolutionImplementation,
}

impl STimeSolution {
    /// Calculate total S-cost
    pub fn total_s_cost(&self) -> f64 {
        self.s_distance.total_distance() +
        (self.time_to_solution.as_secs_f64() / 1000.0) +
        (1.0 - self.truthfulness_level)
    }
}

/// Solution implementation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionImplementation {
    /// Implementation steps
    pub steps: Vec<String>,
    /// Resource requirements
    pub resources: ResourceRequirements,
    /// Risk factors
    pub risks: Vec<String>,
    /// Success probability
    pub success_probability: f64,
}

/// S-optimization route through solution space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOptimizationRoute {
    /// Route identifier
    pub id: Uuid,
    /// Route description
    pub description: String,
    /// Steps in the optimization route
    pub steps: Vec<SOptimizationStep>,
    /// Total route cost
    pub total_cost: f64,
    /// Estimated completion time
    pub completion_time: Duration,
}

/// Individual step in S-optimization route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOptimizationStep {
    /// Step description
    pub description: String,
    /// S-distance change for this step
    pub s_distance_delta: SConstant,
    /// Time cost for this step
    pub time_cost: Duration,
    /// Step risk level
    pub risk_level: f64,
}

/// Window combination types for S-entropy navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowCombination {
    /// Temporal windows
    pub temporal_windows: Vec<TemporalWindow>,
    /// Entropy windows
    pub entropy_windows: Vec<EntropyWindow>,
    /// Knowledge windows
    pub knowledge_windows: Vec<KnowledgeWindow>,
    /// Combination effectiveness score
    pub effectiveness_score: f64,
    /// Impossibility factor (1.0 = normal, >1.0 = impossible)
    pub impossibility_factor: f64,
}

/// Temporal window for time-based navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWindow {
    /// Window start time
    pub start_time: f64,
    /// Window duration
    pub duration: Duration,
    /// Temporal precision in this window
    pub precision: f64,
    /// Window type
    pub window_type: TemporalWindowType,
}

/// Types of temporal windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalWindowType {
    /// Normal temporal processing
    Normal,
    /// High precision temporal navigation
    HighPrecision,
    /// Impossible temporal violations
    Impossible { violations: Vec<String> },
}

/// Entropy window for entropy-based navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyWindow {
    /// Starting entropy level
    pub start_entropy: f64,
    /// Target entropy level
    pub target_entropy: f64,
    /// Entropy change rate
    pub change_rate: f64,
    /// Window type
    pub window_type: EntropyWindowType,
}

/// Types of entropy windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropyWindowType {
    /// Normal entropy evolution
    Normal,
    /// Controlled entropy manipulation
    Controlled,
    /// Impossible entropy violations
    Impossible { violations: Vec<String> },
}

/// Knowledge window for information-based navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeWindow {
    /// Current knowledge level
    pub current_knowledge: f64,
    /// Target knowledge level
    pub target_knowledge: f64,
    /// Knowledge acquisition rate
    pub acquisition_rate: f64,
    /// Window type
    pub window_type: KnowledgeWindowType,
}

/// Types of knowledge windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeWindowType {
    /// Normal learning/acquisition
    Normal,
    /// Accelerated knowledge access
    Accelerated,
    /// Impossible knowledge violations
    Impossible { violations: Vec<String> },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_constant_operations() {
        let s1 = SConstant::new(0.5, 0.3, 0.2);
        let s2 = SConstant::new(0.2, 0.1, 0.1);

        let sum = s1 + s2;
        assert_eq!(sum.s_knowledge, 0.7);
        assert_eq!(sum.s_time, 0.4);
        assert_eq!(sum.s_entropy, 0.3);

        let improvement = s2.improvement_from(&s1);
        assert!(improvement > 0.0); // s2 is better than s1
    }

    #[test]
    fn test_temporal_coordinate_distance() {
        let coord1 = TemporalCoordinate::new(0.0, 0.0, 0.0, 0.0, 1e-9);
        let coord2 = TemporalCoordinate::new(1.0, 1.0, 1.0, 1.0, 1e-9);

        let distance = coord1.distance_to(&coord2);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_problem_description_creation() {
        let problem = ProblemDescription::from("Test problem");
        assert_eq!(problem.description, "Test problem");
        assert_eq!(problem.domain, "general");
        assert!(!problem.requires_impossible_solutions());
    }

    #[test]
    fn test_s_time_solution_cost_calculation() {
        let solution = STimeSolution {
            id: Uuid::new_v4(),
            description: "Test solution".to_string(),
            s_distance: SConstant::new(0.1, 0.2, 0.3),
            time_to_solution: Duration::from_secs(5),
            truthfulness_level: 0.9,
            total_s_cost: 0.0,
            implementation: SolutionImplementation {
                steps: vec!["Step 1".to_string()],
                resources: Default::default(),
                risks: vec![],
                success_probability: 0.95,
            },
        };

        let cost = solution.total_s_cost();
        assert!(cost > 0.0);
    }
}
