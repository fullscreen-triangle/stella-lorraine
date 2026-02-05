//! # CatCount - Categorical State Counting Library
//!
//! A Rust implementation of the trans-Planckian temporal resolution framework
//! through categorical state counting in bounded phase space.
//!
//! ## Overview
//!
//! This library implements the theoretical framework for achieving temporal
//! resolution beyond the Planck scale (~10^-44 s) through categorical enumeration
//! rather than energy-mediated measurement.
//!
//! ## Key Components
//!
//! - [`constants`]: Physical constants (Planck time, Boltzmann constant, etc.)
//! - [`s_entropy`]: S-entropy coordinate system (S_k, S_t, S_e)
//! - [`partition`]: Partition coordinate algebra (n, l, m, s)
//! - [`enhancement`]: Five-mechanism enhancement chain
//! - [`resolution`]: Temporal resolution calculations
//! - [`triple_equivalence`]: Triple equivalence theorem validation
//! - [`spectroscopy`]: Spectroscopic mode predictions and validation
//! - [`memory`]: Categorical memory addressing and tier management
//! - [`demon`]: Maxwell demon controller for memory optimization
//!
//! ## Categorical Memory
//!
//! The library implements the categorical memory architecture from the molecular
//! dynamics paper. Memory addresses are trajectories through S-entropy space,
//! and the Maxwell demon controller manages tier placement using thermodynamic
//! principles.
//!
//! ## Example
//!
//! ```rust,no_run
//! use catcount::{EnhancementChain, TemporalResolution, constants::PLANCK_TIME};
//!
//! // Create enhancement chain with all mechanisms
//! let chain = EnhancementChain::full();
//!
//! // Calculate resolution at molecular vibration frequency
//! let freq_hz = 5.13e13; // CO stretch
//! let resolution = TemporalResolution::calculate(freq_hz, &chain);
//!
//! println!("Categorical resolution: {:.3e} s", resolution.delta_t);
//! println!("Orders below Planck: {:.2}", resolution.orders_below_planck());
//! ```

pub mod constants;
pub mod s_entropy;
pub mod partition;
pub mod enhancement;
pub mod resolution;
pub mod triple_equivalence;
pub mod spectroscopy;
pub mod validation;
pub mod memory;
pub mod demon;
pub mod error;

// Re-export commonly used types
pub use constants::*;
pub use s_entropy::SEntropyCoord;
pub use partition::{PartitionCoord, PartitionState};
pub use enhancement::{EnhancementChain, EnhancementMechanism};
pub use resolution::TemporalResolution;
pub use triple_equivalence::TripleEquivalence;
pub use memory::{CategoricalAddress, CategoricalMemory, MemoryTier};
pub use demon::{MaxwellDemon, DemonDecision, CategoricalAperture};
pub use error::{CatCountError, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Calculate categorical temporal resolution at given frequency
///
/// # Arguments
/// * `frequency_hz` - Process frequency in Hz
///
/// # Returns
/// * `TemporalResolution` containing resolution and metadata
pub fn resolve_at_frequency(frequency_hz: f64) -> TemporalResolution {
    let chain = EnhancementChain::full();
    TemporalResolution::calculate(frequency_hz, &chain)
}

/// Calculate entropy for M oscillators with n states
///
/// # Arguments
/// * `m` - Number of oscillators
/// * `n` - Number of states per oscillator
///
/// # Returns
/// * Entropy in J/K
pub fn entropy(m: u64, n: u64) -> f64 {
    constants::K_B * (m as f64) * (n as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_at_frequency() {
        let resolution = resolve_at_frequency(5.13e13);
        assert!(resolution.delta_t < PLANCK_TIME);
        assert!(resolution.orders_below_planck() > 90.0);
    }

    #[test]
    fn test_entropy() {
        let s = entropy(5, 4);
        let expected = K_B * 5.0 * 4.0_f64.ln();
        assert!((s - expected).abs() < 1e-30);
    }
}
