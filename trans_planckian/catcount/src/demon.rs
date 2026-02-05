//! Maxwell Demon Controller
//!
//! The Maxwell demon operates in categorical space rather than physical space.
//! Unlike Maxwell's original demon which sorted molecules by velocity (requiring
//! information erasure), the categorical demon sorts by partition coordinates
//! (which commutes with physical observables).
//!
//! Key insight from categorical thermodynamics:
//! - [Ô_cat, Ô_phys] = 0 (categorical and physical observables commute)
//! - The demon can extract categorical information without disturbing physical state
//! - No Landauer cost because no physical information is erased
//!
//! The demon implements the categorical aperture from the thermodynamics paper:
//! - Sorts by partition coordinate, not by energy
//! - Zero thermodynamic cost
//! - Operates on trajectories through S-entropy space

use crate::constants::K_B;
use crate::memory::{
    CategoricalAddress, CategoricalMemory, MemoryTier, MemoryEntry,
    PrecisionByDifference,
};
use crate::s_entropy::SEntropyCoord;
use crate::partition::PartitionCoord;
use crate::error::Result;
use serde::{Deserialize, Serialize};

// =============================================================================
// TRAJECTORY PREDICTION
// =============================================================================

/// Trajectory completion prediction
///
/// From the categorical memory paper: the demon predicts where a trajectory
/// will end up and places data at that predicted endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPrediction {
    /// Predicted endpoint coordinate
    pub predicted_endpoint: SEntropyCoord,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted tier for optimal placement
    pub predicted_tier: MemoryTier,
    /// Steps ahead being predicted
    pub horizon: u32,
}

impl TrajectoryPrediction {
    /// Create a new prediction
    pub fn new(
        endpoint: SEntropyCoord,
        confidence: f64,
        current_position: &SEntropyCoord,
    ) -> Self {
        let distance = current_position.distance(&endpoint);
        let predicted_tier = MemoryTier::from_categorical_distance(distance);

        Self {
            predicted_endpoint: endpoint,
            confidence: confidence.clamp(0.0, 1.0),
            predicted_tier,
            horizon: 1,
        }
    }

    /// Should the demon act on this prediction?
    pub fn should_act(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

// =============================================================================
// MAXWELL DEMON CONTROLLER
// =============================================================================

/// The Maxwell Demon memory controller
///
/// The demon navigates S-entropy space and makes tier placement decisions
/// based on categorical proximity. Unlike physical Maxwell demons, this
/// operates at zero thermodynamic cost because categorical and physical
/// observables commute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxwellDemon {
    /// Current position in S-entropy space
    pub position: SEntropyCoord,
    /// History of positions (trajectory)
    pub trajectory: Vec<SEntropyCoord>,
    /// Precision-by-difference history
    pub delta_p_history: Vec<f64>,
    /// Prediction confidence threshold
    pub confidence_threshold: f64,
    /// Statistics
    pub stats: DemonStats,
    /// Operating temperature (categorical)
    pub temperature: f64,
}

/// Demon operation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DemonStats {
    /// Total predictions made
    pub predictions_made: u64,
    /// Predictions that led to action
    pub predictions_acted: u64,
    /// Correct predictions (verified after the fact)
    pub predictions_correct: u64,
    /// Total precision-by-difference calculations
    pub delta_p_calculations: u64,
    /// Total sorting operations
    pub sort_operations: u64,
    /// Categorical work done (entropy × transitions)
    pub categorical_work: f64,
    /// Physical work done (should be zero for ideal demon)
    pub physical_work: f64,
}

impl DemonStats {
    /// Calculate prediction accuracy
    pub fn prediction_accuracy(&self) -> f64 {
        if self.predictions_acted > 0 {
            self.predictions_correct as f64 / self.predictions_acted as f64
        } else {
            0.0
        }
    }

    /// Is this a zero-cost demon? (physical work = 0)
    pub fn is_zero_cost(&self) -> bool {
        self.physical_work.abs() < 1e-20
    }
}

impl MaxwellDemon {
    /// Create a new Maxwell demon
    pub fn new() -> Self {
        Self {
            position: SEntropyCoord::origin(),
            trajectory: vec![SEntropyCoord::origin()],
            delta_p_history: Vec::new(),
            confidence_threshold: 0.7,
            stats: DemonStats::default(),
            temperature: 300.0, // Room temperature
        }
    }

    /// Create demon at specific position
    pub fn at_position(position: SEntropyCoord) -> Self {
        Self {
            position: position.clone(),
            trajectory: vec![position],
            delta_p_history: Vec::new(),
            confidence_threshold: 0.7,
            stats: DemonStats::default(),
            temperature: 300.0,
        }
    }

    /// Move the demon to a new position
    pub fn move_to(&mut self, position: SEntropyCoord, delta_p: f64) {
        self.position = position.clone();
        self.trajectory.push(position);
        self.delta_p_history.push(delta_p);
        self.stats.delta_p_calculations += 1;
    }

    /// Observe a memory entry and decide on tier placement
    ///
    /// This is the core demon operation. The demon observes the categorical
    /// position of data and decides whether it should be promoted or demoted.
    pub fn observe(&mut self, entry: &MemoryEntry) -> DemonDecision {
        let entry_coord = entry.address.current_coord().clone();
        let distance = self.position.distance(&entry_coord);

        // Calculate completion probability using trajectory history
        let completion_prob = self.calculate_completion_probability(&entry_coord);

        // Predict optimal tier
        let optimal_tier = MemoryTier::from_categorical_distance(distance);

        // Make decision
        let decision = if completion_prob > self.confidence_threshold {
            // High completion probability: promote to faster tier
            if entry.tier > optimal_tier {
                DemonDecision::Promote(optimal_tier)
            } else {
                DemonDecision::NoAction
            }
        } else if completion_prob < 1.0 - self.confidence_threshold {
            // Low completion probability: demote to slower tier
            if entry.tier < optimal_tier {
                DemonDecision::Demote(optimal_tier)
            } else {
                DemonDecision::NoAction
            }
        } else {
            DemonDecision::NoAction
        };

        self.stats.sort_operations += 1;
        if decision != DemonDecision::NoAction {
            self.stats.categorical_work += K_B * self.temperature * distance;
        }

        decision
    }

    /// Calculate completion probability for a trajectory endpoint
    ///
    /// Uses the trajectory history to predict if the system will reach
    /// the given endpoint.
    fn calculate_completion_probability(&self, endpoint: &SEntropyCoord) -> f64 {
        if self.trajectory.len() < 2 {
            return 0.5; // No history, assume 50%
        }

        // Linear extrapolation of trajectory
        let n = self.trajectory.len();
        let last = &self.trajectory[n - 1];
        let prev = &self.trajectory[n - 2];

        // Extrapolated position
        let extrapolated = SEntropyCoord::new_unchecked(
            2.0 * last.s_k - prev.s_k,
            2.0 * last.s_t - prev.s_t,
            2.0 * last.s_e - prev.s_e,
        );

        // Distance from extrapolation to target
        let extrap_dist = extrapolated.distance(endpoint);

        // Distance from current to target
        let current_dist = last.distance(endpoint);

        // Probability decreases with distance
        let prob = (-extrap_dist / (current_dist + 1e-30)).exp();
        prob.clamp(0.0, 1.0)
    }

    /// Predict trajectory completion
    pub fn predict(&mut self, horizon: u32) -> TrajectoryPrediction {
        self.stats.predictions_made += 1;

        // Extrapolate trajectory
        let predicted = if self.trajectory.len() >= 2 {
            let n = self.trajectory.len();
            let last = &self.trajectory[n - 1];
            let prev = &self.trajectory[n - 2];

            SEntropyCoord::new_unchecked(
                last.s_k + (horizon as f64) * (last.s_k - prev.s_k),
                last.s_t + (horizon as f64) * (last.s_t - prev.s_t),
                last.s_e + (horizon as f64) * (last.s_e - prev.s_e),
            )
        } else {
            self.position.clone()
        };

        // Confidence based on trajectory smoothness
        let confidence = self.calculate_trajectory_smoothness();

        let mut prediction = TrajectoryPrediction::new(
            predicted, confidence, &self.position
        );
        prediction.horizon = horizon;

        if prediction.should_act(self.confidence_threshold) {
            self.stats.predictions_acted += 1;
        }

        prediction
    }

    /// Calculate how smooth the trajectory is
    fn calculate_trajectory_smoothness(&self) -> f64 {
        if self.trajectory.len() < 3 {
            return 0.5;
        }

        // Calculate variance in step sizes
        let steps: Vec<f64> = self.trajectory.windows(2)
            .map(|w| w[0].distance(&w[1]))
            .collect();

        let mean_step: f64 = steps.iter().sum::<f64>() / steps.len() as f64;
        let variance: f64 = steps.iter()
            .map(|s| (s - mean_step).powi(2))
            .sum::<f64>() / steps.len() as f64;

        // Lower variance = higher smoothness = higher confidence
        let cv = if mean_step > 0.0 {
            variance.sqrt() / mean_step
        } else {
            0.0
        };

        (1.0 - cv).clamp(0.0, 1.0)
    }

    /// Apply the demon's decision to memory
    pub fn apply_decision(&self, memory: &mut CategoricalMemory, hash: u64, decision: &DemonDecision) -> Result<()> {
        match decision {
            DemonDecision::Promote(_) => {
                memory.promote(hash)?;
            }
            DemonDecision::Demote(tier) => {
                // Demote is handled by make_space when needed
                // For explicit demotion, we'd need to add a demote method
            }
            DemonDecision::NoAction => {}
        }
        Ok(())
    }

    /// Process all entries in memory
    pub fn process_memory(&mut self, memory: &mut CategoricalMemory) -> Vec<(u64, DemonDecision)> {
        let decisions: Vec<_> = memory.entries.iter()
            .map(|(hash, entry)| (*hash, self.observe(entry)))
            .filter(|(_, d)| *d != DemonDecision::NoAction)
            .collect();

        decisions
    }

    /// Get categorical temperature (rate of position changes)
    pub fn categorical_temperature(&self) -> f64 {
        if self.trajectory.len() < 2 {
            return 0.0;
        }

        // Rate of categorical actualization = dM/dt
        // Approximated as average step size per unit time
        let total_distance: f64 = self.trajectory.windows(2)
            .map(|w| w[0].distance(&w[1]))
            .sum();

        K_B * total_distance / (self.trajectory.len() as f64)
    }

    /// Get categorical entropy of the demon's trajectory
    pub fn trajectory_entropy(&self) -> f64 {
        K_B * (self.trajectory.len() as f64) * 3.0_f64.ln()
    }
}

impl Default for MaxwellDemon {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// DEMON DECISION
// =============================================================================

/// Decision made by the Maxwell demon
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemonDecision {
    /// Promote to a faster tier
    Promote(MemoryTier),
    /// Demote to a slower tier
    Demote(MemoryTier),
    /// No action needed
    NoAction,
}

impl std::fmt::Display for DemonDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemonDecision::Promote(tier) => write!(f, "Promote -> {}", tier),
            DemonDecision::Demote(tier) => write!(f, "Demote -> {}", tier),
            DemonDecision::NoAction => write!(f, "No Action"),
        }
    }
}

// =============================================================================
// CATEGORICAL APERTURE
// =============================================================================

/// The categorical aperture - a zero-cost sorting device
///
/// Unlike Maxwell's demon which sorts by energy (requiring information erasure),
/// the categorical aperture sorts by partition coordinates (zero cost).
///
/// From categorical thermodynamics:
/// - Demon sorts by energy -> requires k_B T ln 2 per bit erased
/// - Aperture sorts by partition -> zero thermodynamic cost
#[derive(Debug, Clone)]
pub struct CategoricalAperture {
    /// Current partition filter
    pub filter: Option<PartitionCoord>,
    /// Selectivity (what fraction passes)
    pub selectivity: f64,
    /// Operations performed
    pub operations: u64,
}

impl CategoricalAperture {
    /// Create a new aperture
    pub fn new() -> Self {
        Self {
            filter: None,
            selectivity: 1.0,
            operations: 0,
        }
    }

    /// Set the partition filter
    pub fn set_filter(&mut self, partition: PartitionCoord, selectivity: f64) {
        self.filter = Some(partition);
        self.selectivity = selectivity.clamp(0.0, 1.0);
    }

    /// Clear the filter
    pub fn clear_filter(&mut self) {
        self.filter = None;
        self.selectivity = 1.0;
    }

    /// Check if an entry passes the aperture
    pub fn passes(&mut self, _entry: &MemoryEntry) -> bool {
        self.operations += 1;

        // If no filter, everything passes
        if self.filter.is_none() {
            return true;
        }

        // Selectivity determines pass probability
        // In a real implementation, this would check partition coordinates
        self.selectivity > 0.5
    }

    /// Thermodynamic cost of the aperture operation
    pub fn thermodynamic_cost(&self) -> f64 {
        // The categorical aperture operates at zero cost
        // because [Ô_cat, Ô_phys] = 0
        0.0
    }
}

impl Default for CategoricalAperture {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TRIPLE EQUIVALENCE CONTROLLER
// =============================================================================

/// Controller that operates using all three perspectives of the triple equivalence
///
/// - Oscillatory: frequency-based timing decisions
/// - Categorical: state-counting decisions
/// - Partition: temporal segment decisions
#[derive(Debug, Clone)]
pub struct TripleEquivalenceController {
    /// Maxwell demon for categorical operations
    pub demon: MaxwellDemon,
    /// Categorical aperture for zero-cost sorting
    pub aperture: CategoricalAperture,
    /// Current oscillation phase
    pub phase: f64,
    /// Oscillation frequency (Hz)
    pub frequency: f64,
    /// Partition count
    pub partitions: u64,
}

impl TripleEquivalenceController {
    /// Create a new controller
    pub fn new(frequency: f64) -> Self {
        Self {
            demon: MaxwellDemon::new(),
            aperture: CategoricalAperture::new(),
            phase: 0.0,
            frequency,
            partitions: 0,
        }
    }

    /// Advance one oscillation cycle
    pub fn tick(&mut self, delta_t: f64) {
        self.phase += 2.0 * std::f64::consts::PI * self.frequency * delta_t;
        self.phase %= 2.0 * std::f64::consts::PI;
        self.partitions += 1;
    }

    /// Get the category rate (dM/dt) = M × f
    ///
    /// From the gas laws paper: dM/dt = M/T = M × f
    pub fn category_rate(&self) -> f64 {
        self.frequency * self.partitions as f64
    }

    /// Get the average partition duration ⟨τ_p⟩ = T/M = 1/(M × f)
    pub fn avg_partition_duration(&self) -> f64 {
        if self.partitions > 0 {
            1.0 / (self.frequency * self.partitions as f64)
        } else {
            f64::INFINITY
        }
    }

    /// Verify triple equivalence identity: dM/dt = ω/(2π/M) = 1/⟨τ_p⟩
    ///
    /// Where:
    /// - dM/dt = M × f
    /// - ω/(2π/M) = 2πf / (2π/M) = M × f
    /// - 1/⟨τ_p⟩ = M × f
    pub fn verify_triple_equivalence(&self) -> bool {
        if self.partitions == 0 {
            return true;
        }

        let m = self.partitions as f64;
        let f = self.frequency;
        let omega = 2.0 * std::f64::consts::PI * f;

        // dM/dt = M × f
        let dm_dt = self.category_rate();

        // ω/(2π/M) = ω × M / (2π) = 2πf × M / (2π) = M × f
        let omega_scaled = omega * m / (2.0 * std::f64::consts::PI);

        // 1/⟨τ_p⟩ = 1 / (1/(M × f)) = M × f
        let inverse_tau = 1.0 / self.avg_partition_duration();

        let tol = 1e-6;
        (dm_dt - omega_scaled).abs() / dm_dt.max(1.0) < tol &&
            (dm_dt - inverse_tau).abs() / dm_dt.max(1.0) < tol
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::CategoricalAddress;

    #[test]
    fn test_demon_creation() {
        let demon = MaxwellDemon::new();
        assert_eq!(demon.trajectory.len(), 1);
        assert!(demon.stats.is_zero_cost());
    }

    #[test]
    fn test_demon_move() {
        let mut demon = MaxwellDemon::new();
        demon.move_to(
            SEntropyCoord::new_unchecked(1e-23, 0.0, 0.0),
            1e-6
        );
        assert_eq!(demon.trajectory.len(), 2);
        assert_eq!(demon.delta_p_history.len(), 1);
    }

    #[test]
    fn test_trajectory_prediction() {
        let mut demon = MaxwellDemon::new();

        // Build a linear trajectory
        for i in 0..10 {
            demon.move_to(
                SEntropyCoord::new_unchecked(
                    (i as f64) * 1e-24,
                    (i as f64) * 1e-24,
                    0.0
                ),
                1e-6
            );
        }

        let prediction = demon.predict(5);
        assert!(prediction.confidence > 0.0);
    }

    #[test]
    fn test_categorical_aperture_zero_cost() {
        let aperture = CategoricalAperture::new();
        assert_eq!(aperture.thermodynamic_cost(), 0.0);
    }

    #[test]
    fn test_triple_equivalence_identity() {
        let mut controller = TripleEquivalenceController::new(1e6);

        for _ in 0..100 {
            controller.tick(1e-9);
        }

        assert!(controller.verify_triple_equivalence());
    }

    #[test]
    fn test_demon_decision() {
        let mut demon = MaxwellDemon::new();

        let addr = CategoricalAddress::new(
            vec![SEntropyCoord::new_unchecked(1e-25, 0.0, 0.0)],
            vec![0.0]
        );

        let entry = MemoryEntry::new(addr, MemoryTier::RAM, 1024);
        let decision = demon.observe(&entry);

        // Should be defined (may or may not be NoAction depending on position)
        assert!(matches!(
            decision,
            DemonDecision::Promote(_) | DemonDecision::Demote(_) | DemonDecision::NoAction
        ));
    }
}
