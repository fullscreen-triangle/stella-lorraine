//! Categorical Memory Module
//!
//! This module implements categorical memory addressing based on S-entropy
//! coordinates. Memory locations are addressed by trajectories through the
//! 3^k hierarchical structure, where the precision-by-difference values
//! accumulated during access form the address.
//!
//! The key insight from the molecular dynamics paper is that:
//! - Memory addresses ARE trajectories through S-entropy space
//! - The access history constitutes the address
//! - Data placement follows thermodynamic principles
//!
//! From the categorical thermodynamics paper:
//! - Temperature = rate of categorical actualization
//! - Pressure = categorical density
//! - Entropy = partition traversal count

use crate::constants::K_B;
use crate::s_entropy::{SEntropyCoord, SEntropyReferences};
use crate::partition::PartitionCoord;
use crate::error::{CatCountError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// CATEGORICAL ADDRESS
// =============================================================================

/// A categorical address is a trajectory through S-entropy space
///
/// Unlike physical addresses which are numeric positions, categorical
/// addresses encode the history of access patterns. Two accesses to the
/// same physical data at different times have different categorical addresses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalAddress {
    /// Trajectory of S-entropy coordinates
    pub trajectory: Vec<SEntropyCoord>,
    /// Precision-by-difference values at each step
    pub delta_p: Vec<f64>,
    /// Trajectory hash (the actual address identifier)
    pub hash: u64,
    /// Depth in the 3^k hierarchy
    pub depth: u32,
    /// Branch decisions (0, 1, or 2 at each level)
    pub branches: Vec<u8>,
}

impl CategoricalAddress {
    /// Create a new categorical address from a trajectory
    pub fn new(trajectory: Vec<SEntropyCoord>, delta_p: Vec<f64>) -> Self {
        let depth = trajectory.len() as u32;
        let branches = Self::compute_branches(&delta_p);
        let hash = Self::compute_hash(&branches);

        Self {
            trajectory,
            delta_p,
            hash,
            depth,
            branches,
        }
    }

    /// Create root address (empty trajectory)
    pub fn root() -> Self {
        Self {
            trajectory: vec![SEntropyCoord::origin()],
            delta_p: vec![0.0],
            hash: 0,
            depth: 0,
            branches: vec![],
        }
    }

    /// Extend the address by one step
    pub fn extend(&self, coord: SEntropyCoord, delta_p: f64) -> Self {
        let mut new_trajectory = self.trajectory.clone();
        new_trajectory.push(coord);

        let mut new_delta_p = self.delta_p.clone();
        new_delta_p.push(delta_p);

        Self::new(new_trajectory, new_delta_p)
    }

    /// Compute branch decisions from precision-by-difference values
    ///
    /// The delta_p value determines which of the three S-entropy dimensions
    /// to branch into at each level of the hierarchy.
    fn compute_branches(delta_p: &[f64]) -> Vec<u8> {
        delta_p.iter().map(|&dp| {
            // Map delta_p to branch 0, 1, or 2
            // Branch 0: S_k (knowledge entropy) - negative delta_p
            // Branch 1: S_t (temporal entropy) - near-zero delta_p
            // Branch 2: S_e (evolution entropy) - positive delta_p
            if dp < -1e-6 {
                0
            } else if dp > 1e-6 {
                2
            } else {
                1
            }
        }).collect()
    }

    /// Compute hash from branch sequence
    ///
    /// The hash uniquely identifies a position in the 3^k hierarchy.
    /// It is computed as a base-3 number from the branch sequence.
    fn compute_hash(branches: &[u8]) -> u64 {
        let mut hash: u64 = 0;
        let mut multiplier: u64 = 1;

        for &branch in branches {
            hash += (branch as u64) * multiplier;
            multiplier *= 3;
        }

        hash
    }

    /// Get the current S-entropy coordinate
    pub fn current_coord(&self) -> SEntropyCoord {
        self.trajectory.last().cloned().unwrap_or_else(SEntropyCoord::origin)
    }

    /// Calculate categorical distance to another address
    pub fn distance(&self, other: &Self) -> f64 {
        self.current_coord().distance(&other.current_coord())
    }

    /// Number of nodes at this depth: 3^k
    pub fn nodes_at_depth(&self) -> u64 {
        3_u64.pow(self.depth)
    }
}

impl std::fmt::Display for CategoricalAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CatAddr[depth={}, hash={}, branches={:?}]",
            self.depth, self.hash, self.branches)
    }
}

// =============================================================================
// MEMORY TIER
// =============================================================================

/// Memory tier levels (analogous to L1, L2, L3 cache, RAM, SSD)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    /// L1 Cache - categorically proximate (hot)
    L1 = 0,
    /// L2 Cache - moderate proximity
    L2 = 1,
    /// L3 Cache - lower proximity
    L3 = 2,
    /// Main Memory - categorical midpoint
    RAM = 3,
    /// Storage - categorically distant (cold)
    Storage = 4,
}

impl MemoryTier {
    /// Get the categorical temperature of this tier
    ///
    /// From categorical thermodynamics: T = U/(k_B × M)
    /// Higher temperature = faster categorical actualization = L1
    /// Lower temperature = slower categorical actualization = Storage
    pub fn temperature(&self) -> f64 {
        match self {
            MemoryTier::L1 => 1000.0,     // Hot
            MemoryTier::L2 => 500.0,      // Warm
            MemoryTier::L3 => 300.0,      // Room temperature
            MemoryTier::RAM => 100.0,     // Cool
            MemoryTier::Storage => 10.0,  // Cold
        }
    }

    /// Get access latency in nanoseconds
    pub fn latency_ns(&self) -> f64 {
        match self {
            MemoryTier::L1 => 1.0,
            MemoryTier::L2 => 4.0,
            MemoryTier::L3 => 12.0,
            MemoryTier::RAM => 100.0,
            MemoryTier::Storage => 100_000.0,
        }
    }

    /// Get capacity multiplier relative to L1
    pub fn capacity_multiplier(&self) -> f64 {
        match self {
            MemoryTier::L1 => 1.0,
            MemoryTier::L2 => 8.0,
            MemoryTier::L3 => 64.0,
            MemoryTier::RAM => 1000.0,
            MemoryTier::Storage => 100_000.0,
        }
    }

    /// Determine tier from categorical distance
    ///
    /// Data that is categorically close to current position goes to fast tiers;
    /// data that is categorically distant goes to slow tiers.
    pub fn from_categorical_distance(distance: f64) -> Self {
        if distance < 1e-24 {
            MemoryTier::L1
        } else if distance < 1e-23 {
            MemoryTier::L2
        } else if distance < 1e-22 {
            MemoryTier::L3
        } else if distance < 1e-21 {
            MemoryTier::RAM
        } else {
            MemoryTier::Storage
        }
    }
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryTier::L1 => write!(f, "L1"),
            MemoryTier::L2 => write!(f, "L2"),
            MemoryTier::L3 => write!(f, "L3"),
            MemoryTier::RAM => write!(f, "RAM"),
            MemoryTier::Storage => write!(f, "Storage"),
        }
    }
}

// =============================================================================
// MEMORY ENTRY
// =============================================================================

/// A memory entry with categorical addressing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Categorical address
    pub address: CategoricalAddress,
    /// Current memory tier
    pub tier: MemoryTier,
    /// Access count
    pub access_count: u64,
    /// Last access timestamp (in categorical time units)
    pub last_access: f64,
    /// Data size in bytes
    pub size_bytes: u64,
    /// Completion probability (predicted likelihood of future access)
    pub completion_probability: f64,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(address: CategoricalAddress, tier: MemoryTier, size_bytes: u64) -> Self {
        Self {
            address,
            tier,
            access_count: 0,
            last_access: 0.0,
            size_bytes,
            completion_probability: 0.5,
        }
    }

    /// Record an access
    pub fn access(&mut self, time: f64) {
        self.access_count += 1;
        self.last_access = time;
    }

    /// Calculate categorical entropy of this entry
    ///
    /// From categorical thermodynamics: S = k_B × M × ln(n)
    /// where M is the trajectory depth and n is the branching factor (3)
    pub fn categorical_entropy(&self) -> f64 {
        K_B * (self.address.depth as f64) * 3.0_f64.ln()
    }

    /// Calculate categorical temperature
    ///
    /// T = (access_rate × ℏ) / k_B
    /// Approximated as proportional to access frequency
    pub fn categorical_temperature(&self, observation_time: f64) -> f64 {
        if observation_time > 0.0 {
            (self.access_count as f64 / observation_time) * 1e-10 // Scale factor
        } else {
            0.0
        }
    }
}

// =============================================================================
// CATEGORICAL MEMORY
// =============================================================================

/// Categorical memory system
///
/// The memory system is organized as a 3^k hierarchical structure where
/// addresses are trajectories through S-entropy space. The Maxwell demon
/// controller manages data placement across tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalMemory {
    /// Memory entries indexed by hash
    pub entries: HashMap<u64, MemoryEntry>,
    /// Current position in S-entropy space
    pub current_position: CategoricalAddress,
    /// Total capacity per tier (bytes)
    pub tier_capacity: HashMap<MemoryTier, u64>,
    /// Used capacity per tier (bytes)
    pub tier_used: HashMap<MemoryTier, u64>,
    /// Statistics
    pub stats: MemoryStats,
    /// Current categorical time
    pub time: f64,
}

/// Memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total accesses
    pub total_accesses: u64,
    /// Cache hits (L1, L2, L3)
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Promotions (moved to faster tier)
    pub promotions: u64,
    /// Demotions (moved to slower tier)
    pub demotions: u64,
    /// Total latency (nanoseconds)
    pub total_latency_ns: f64,
}

impl MemoryStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_accesses > 0 {
            self.cache_hits as f64 / self.total_accesses as f64
        } else {
            0.0
        }
    }

    /// Calculate average latency
    pub fn avg_latency_ns(&self) -> f64 {
        if self.total_accesses > 0 {
            self.total_latency_ns / self.total_accesses as f64
        } else {
            0.0
        }
    }
}

impl CategoricalMemory {
    /// Create a new categorical memory system
    pub fn new() -> Self {
        let mut tier_capacity = HashMap::new();
        let mut tier_used = HashMap::new();

        // Default capacities (in bytes)
        tier_capacity.insert(MemoryTier::L1, 32 * 1024);           // 32 KB
        tier_capacity.insert(MemoryTier::L2, 256 * 1024);          // 256 KB
        tier_capacity.insert(MemoryTier::L3, 8 * 1024 * 1024);     // 8 MB
        tier_capacity.insert(MemoryTier::RAM, 16 * 1024 * 1024 * 1024); // 16 GB
        tier_capacity.insert(MemoryTier::Storage, 1024 * 1024 * 1024 * 1024); // 1 TB

        for tier in [MemoryTier::L1, MemoryTier::L2, MemoryTier::L3,
                     MemoryTier::RAM, MemoryTier::Storage] {
            tier_used.insert(tier, 0);
        }

        Self {
            entries: HashMap::new(),
            current_position: CategoricalAddress::root(),
            tier_capacity,
            tier_used,
            stats: MemoryStats::default(),
            time: 0.0,
        }
    }

    /// Write data at a categorical address
    pub fn write(&mut self, address: CategoricalAddress, size_bytes: u64) -> Result<()> {
        let distance = self.current_position.distance(&address);
        let tier = MemoryTier::from_categorical_distance(distance);

        // Check capacity
        let used = self.tier_used.get(&tier).copied().unwrap_or(0);
        let capacity = self.tier_capacity.get(&tier).copied().unwrap_or(0);

        if used + size_bytes > capacity {
            // Need to demote something
            self.make_space(tier, size_bytes)?;
        }

        let entry = MemoryEntry::new(address.clone(), tier, size_bytes);
        *self.tier_used.get_mut(&tier).unwrap() += size_bytes;
        self.entries.insert(address.hash, entry);

        Ok(())
    }

    /// Read data from a categorical address
    pub fn read(&mut self, address: &CategoricalAddress) -> Result<&MemoryEntry> {
        self.time += 1.0;
        self.stats.total_accesses += 1;

        if let Some(entry) = self.entries.get_mut(&address.hash) {
            entry.access(self.time);

            // Record hit/miss based on tier
            match entry.tier {
                MemoryTier::L1 | MemoryTier::L2 | MemoryTier::L3 => {
                    self.stats.cache_hits += 1;
                }
                _ => {
                    self.stats.cache_misses += 1;
                }
            }

            self.stats.total_latency_ns += entry.tier.latency_ns();

            // Update current position
            self.current_position = address.clone();

            Ok(self.entries.get(&address.hash).unwrap())
        } else {
            Err(CatCountError::ValidationFailed(
                format!("Address not found: {}", address.hash)
            ))
        }
    }

    /// Make space in a tier by demoting entries
    fn make_space(&mut self, tier: MemoryTier, needed: u64) -> Result<()> {
        // Find entries in this tier with lowest completion probability
        let mut candidates: Vec<_> = self.entries.iter()
            .filter(|(_, e)| e.tier == tier)
            .map(|(h, e)| (*h, e.completion_probability, e.size_bytes))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut freed = 0_u64;
        for (hash, _, size) in candidates {
            if freed >= needed {
                break;
            }

            // Demote to next tier
            if let Some(entry) = self.entries.get_mut(&hash) {
                let old_tier = entry.tier;
                let new_tier = match old_tier {
                    MemoryTier::L1 => MemoryTier::L2,
                    MemoryTier::L2 => MemoryTier::L3,
                    MemoryTier::L3 => MemoryTier::RAM,
                    MemoryTier::RAM => MemoryTier::Storage,
                    MemoryTier::Storage => MemoryTier::Storage,
                };

                *self.tier_used.get_mut(&old_tier).unwrap() -= size;
                *self.tier_used.get_mut(&new_tier).unwrap() += size;
                entry.tier = new_tier;
                freed += size;
                self.stats.demotions += 1;
            }
        }

        Ok(())
    }

    /// Promote an entry to a faster tier
    pub fn promote(&mut self, hash: u64) -> Result<()> {
        if let Some(entry) = self.entries.get_mut(&hash) {
            let old_tier = entry.tier;
            let new_tier = match old_tier {
                MemoryTier::Storage => MemoryTier::RAM,
                MemoryTier::RAM => MemoryTier::L3,
                MemoryTier::L3 => MemoryTier::L2,
                MemoryTier::L2 => MemoryTier::L1,
                MemoryTier::L1 => MemoryTier::L1,
            };

            if new_tier != old_tier {
                let size = entry.size_bytes;
                *self.tier_used.get_mut(&old_tier).unwrap() -= size;
                *self.tier_used.get_mut(&new_tier).unwrap() += size;
                entry.tier = new_tier;
                self.stats.promotions += 1;
            }
        }

        Ok(())
    }

    /// Calculate total categorical entropy of the memory system
    ///
    /// S_total = Σ S_i = Σ k_B × M_i × ln(3)
    pub fn total_entropy(&self) -> f64 {
        self.entries.values().map(|e| e.categorical_entropy()).sum()
    }

    /// Calculate categorical pressure
    ///
    /// From ideal gas law: P = k_B × T × (M/V)
    /// where M is the total categories and V is the total capacity
    pub fn categorical_pressure(&self, tier: MemoryTier) -> f64 {
        let used = self.tier_used.get(&tier).copied().unwrap_or(0) as f64;
        let capacity = self.tier_capacity.get(&tier).copied().unwrap_or(1) as f64;
        let temperature = tier.temperature();

        K_B * temperature * (used / capacity)
    }

    /// Get memory statistics summary
    pub fn summary(&self) -> String {
        format!(
            "Categorical Memory Summary\n\
             ==========================\n\
             Total entries: {}\n\
             Total accesses: {}\n\
             Hit rate: {:.2}%\n\
             Avg latency: {:.2} ns\n\
             Promotions: {}\n\
             Demotions: {}\n\
             Total entropy: {:.6e} J/K",
            self.entries.len(),
            self.stats.total_accesses,
            self.stats.hit_rate() * 100.0,
            self.stats.avg_latency_ns(),
            self.stats.promotions,
            self.stats.demotions,
            self.total_entropy()
        )
    }
}

impl Default for CategoricalMemory {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PRECISION-BY-DIFFERENCE
// =============================================================================

/// Precision-by-difference calculator
///
/// ΔP = T_ref - t_local
///
/// This is the core mechanism for generating S-entropy coordinates from
/// hardware timing variations.
#[derive(Debug, Clone)]
pub struct PrecisionByDifference {
    /// Reference clock value
    pub t_ref: f64,
    /// Local timing measurements
    pub measurements: Vec<f64>,
    /// Computed delta_p values
    pub delta_p: Vec<f64>,
}

impl PrecisionByDifference {
    /// Create a new calculator with reference time
    pub fn new(t_ref: f64) -> Self {
        Self {
            t_ref,
            measurements: Vec::new(),
            delta_p: Vec::new(),
        }
    }

    /// Add a timing measurement and compute delta_p
    pub fn measure(&mut self, t_local: f64) -> f64 {
        let dp = self.t_ref - t_local;
        self.measurements.push(t_local);
        self.delta_p.push(dp);
        dp
    }

    /// Convert delta_p sequence to S-entropy coordinate
    pub fn to_s_entropy(&self) -> Result<SEntropyCoord> {
        if self.delta_p.is_empty() {
            return Ok(SEntropyCoord::origin());
        }

        // Map delta_p statistics to S-entropy components
        let mean_dp: f64 = self.delta_p.iter().sum::<f64>() / self.delta_p.len() as f64;
        let var_dp: f64 = self.delta_p.iter()
            .map(|dp| (dp - mean_dp).powi(2))
            .sum::<f64>() / self.delta_p.len() as f64;

        let refs = SEntropyReferences::default();

        // S_k = knowledge entropy (from variance)
        let s_k = K_B * (1.0 + var_dp.abs() / refs.phi_0).ln();

        // S_t = temporal entropy (from mean)
        let s_t = K_B * (1.0 + mean_dp.abs() / refs.tau_0).ln();

        // S_e = evolution entropy (from trend)
        let trend = if self.delta_p.len() >= 2 {
            (self.delta_p.last().unwrap() - self.delta_p.first().unwrap()).abs()
        } else {
            0.0
        };
        let s_e = K_B * (1.0 + trend / refs.e_0).ln();

        SEntropyCoord::new(s_k, s_t, s_e)
    }

    /// Generate categorical address from measurements
    pub fn to_address(&self) -> CategoricalAddress {
        let coord = self.to_s_entropy().unwrap_or(SEntropyCoord::origin());
        CategoricalAddress::new(vec![coord], self.delta_p.clone())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorical_address() {
        let trajectory = vec![
            SEntropyCoord::origin(),
            SEntropyCoord::new_unchecked(1e-23, 0.0, 0.0),
            SEntropyCoord::new_unchecked(1e-23, 1e-23, 0.0),
        ];
        let delta_p = vec![0.0, -1e-3, 1e-3];

        let addr = CategoricalAddress::new(trajectory, delta_p);
        assert_eq!(addr.depth, 3);
        assert_eq!(addr.branches.len(), 3);
    }

    #[test]
    fn test_memory_tier_from_distance() {
        assert_eq!(MemoryTier::from_categorical_distance(1e-25), MemoryTier::L1);
        assert_eq!(MemoryTier::from_categorical_distance(5e-24), MemoryTier::L2);
        assert_eq!(MemoryTier::from_categorical_distance(5e-23), MemoryTier::L3);
        assert_eq!(MemoryTier::from_categorical_distance(5e-22), MemoryTier::RAM);
        assert_eq!(MemoryTier::from_categorical_distance(5e-20), MemoryTier::Storage);
    }

    #[test]
    fn test_categorical_memory_write_read() {
        let mut memory = CategoricalMemory::new();

        let addr = CategoricalAddress::new(
            vec![SEntropyCoord::origin()],
            vec![0.0]
        );

        memory.write(addr.clone(), 1024).unwrap();

        let entry = memory.read(&addr).unwrap();
        assert_eq!(entry.size_bytes, 1024);
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_precision_by_difference() {
        let mut pbd = PrecisionByDifference::new(1.0);

        pbd.measure(1.001);
        pbd.measure(0.999);
        pbd.measure(1.002);

        let coord = pbd.to_s_entropy().unwrap();
        assert!(coord.s_k > 0.0);
        assert!(coord.s_t > 0.0);
    }

    #[test]
    fn test_memory_entropy() {
        let mut memory = CategoricalMemory::new();

        for i in 0..10 {
            let addr = CategoricalAddress::new(
                vec![SEntropyCoord::new_unchecked(
                    (i as f64) * 1e-24, 0.0, 0.0
                )],
                vec![(i as f64) * 1e-6]
            );
            memory.write(addr, 100).unwrap();
        }

        let entropy = memory.total_entropy();
        assert!(entropy > 0.0);
    }
}
